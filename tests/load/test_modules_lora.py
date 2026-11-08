#!/usr/bin/env python3
"""Live GGUF-LoRA apply (P2): the ``LoRAKQuantLinear`` wrap + ``install_lora_adapter``
leaf-swap. Pins the a/b orientation + scale against a hand-computed reference, the
qk_permute-on-B (q vs k head counts), dtype preservation, and the loud failures on
unwrappable / missing targets. CPU-only (tiny float Linears, no GGUF, no kernels)."""
from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

import gmlx.load.adapter as adapter  # noqa: E402
import gmlx.load.modules as modules  # noqa: E402
from gmlx.load.transforms import qk_permute_wire  # noqa: E402


def _lm(module_path, a, b, scale, transform="passthrough"):
    rank = min(set(a.shape) & set(b.shape))
    return adapter.LoraModule(module_path=module_path, a=a, b=b, rank=rank,
                              scale=scale, transform=transform)


def _plan(modules_map, *, alpha=16.0, arch="qwen3"):
    return adapter.LoraAdapter(alpha=alpha, arch=arch, modules=modules_map)


class _Tiny(nn.Module):
    def __init__(self, in_dims=4, out_dims=4):
        super().__init__()
        self.q_proj = nn.Linear(in_dims, out_dims, bias=False)
        self.o_proj = nn.Linear(in_dims, out_dims, bias=False)


class _TinyQK(nn.Module):
    def __init__(self, in_dims, out_q, out_k):
        super().__init__()
        self.q_proj = nn.Linear(in_dims, out_q, bias=False)
        self.k_proj = nn.Linear(in_dims, out_k, bias=False)


class _WithEmbed(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(10, 4)


def test_lora_forward_is_base_plus_scaled_ba():
    # A is (rank, in), B is (out, rank): delta = scale * (x @ A.T) @ B.T.
    mx.random.seed(0)
    in_dims, out_dims, rank = 4, 3, 2
    base = nn.Linear(in_dims, out_dims, bias=False)
    base.weight = mx.random.normal((out_dims, in_dims))
    a = mx.random.normal((rank, in_dims))
    b = mx.random.normal((out_dims, rank))
    scale = 0.5
    wrap = modules.LoRAKQuantLinear(base, a, b, scale)
    x = mx.random.normal((5, in_dims))
    ref = base(x) + scale * ((x @ a.T) @ b.T)
    assert mx.allclose(wrap(x), ref, atol=1e-5)


def test_lora_preserves_base_output_dtype():
    base = nn.Linear(4, 3, bias=False)
    base.weight = base.weight.astype(mx.bfloat16)
    a = mx.random.normal((2, 4))        # adapter stays f32
    b = mx.random.normal((3, 2))
    wrap = modules.LoRAKQuantLinear(base, a, b, 0.5)
    y = wrap(mx.zeros((1, 4), dtype=mx.bfloat16))
    assert y.dtype == mx.bfloat16       # f32 delta cast back to the residual dtype


def test_install_wraps_only_targeted_leaf():
    m = _Tiny()
    plan = _plan({"q_proj": _lm("q_proj", mx.zeros((2, 4)), mx.zeros((4, 2)), 1.0)})
    n = modules.install_lora_adapter(m, plan)
    assert n == 1
    assert isinstance(m.q_proj, modules.LoRAKQuantLinear)
    assert isinstance(m.o_proj, nn.Linear)
    assert not isinstance(m.o_proj, modules.LoRAKQuantLinear)


def test_install_missing_target_raises_not_silently_dropped():
    m = _Tiny()
    plan = _plan({"k_proj": _lm("k_proj", mx.zeros((2, 4)), mx.zeros((4, 2)), 1.0)})
    with pytest.raises(ValueError, match="no matching module"):
        modules.install_lora_adapter(m, plan)


def test_install_unwrappable_target_raises():
    m = _WithEmbed()
    plan = _plan({"embed": _lm("embed", mx.zeros((2, 4)), mx.zeros((4, 2)), 1.0)})
    with pytest.raises(NotImplementedError, match="wrappable Linear"):
        modules.install_lora_adapter(m, plan)


def test_install_qk_permute_applies_permute_to_b_only():
    mx.random.seed(1)
    in_dims, out_dims, rank, n_head = 4, 8, 2, 2
    m = _Tiny(in_dims, out_dims)
    a = mx.random.normal((rank, in_dims))
    b = mx.random.normal((out_dims, rank))
    plan = _plan({"q_proj": _lm("q_proj", a, b, 1.0, transform="qk_permute")},
                 arch="llama")
    modules.install_lora_adapter(m, plan, n_head=n_head)
    assert mx.array_equal(m.q_proj.lora_b, qk_permute_wire(b, n_head))
    assert mx.array_equal(m.q_proj.lora_a, a)   # permute is output-rows only


def test_install_qk_permute_k_proj_uses_n_head_kv():
    mx.random.seed(2)
    in_dims, out_q, out_k, n_head, n_kv = 4, 8, 4, 2, 1
    m = _TinyQK(in_dims, out_q, out_k)
    aq, bq = mx.random.normal((2, in_dims)), mx.random.normal((out_q, 2))
    ak, bk = mx.random.normal((2, in_dims)), mx.random.normal((out_k, 2))
    plan = _plan({
        "q_proj": _lm("q_proj", aq, bq, 1.0, transform="qk_permute"),
        "k_proj": _lm("k_proj", ak, bk, 1.0, transform="qk_permute"),
    }, arch="llama")
    modules.install_lora_adapter(m, plan, n_head=n_head, n_head_kv=n_kv)
    assert mx.array_equal(m.q_proj.lora_b, qk_permute_wire(bq, n_head))
    assert mx.array_equal(m.k_proj.lora_b, qk_permute_wire(bk, n_kv))


def test_install_qk_permute_without_head_count_raises():
    m = _Tiny(4, 8)
    plan = _plan({"q_proj": _lm("q_proj", mx.zeros((2, 4)), mx.zeros((8, 2)), 1.0,
                                transform="qk_permute")}, arch="llama")
    with pytest.raises(ValueError, match="qk_permute"):
        modules.install_lora_adapter(m, plan)


@pytest.mark.parametrize("config,expected", [
    ({"num_attention_heads": 16, "num_key_value_heads": 8}, (16, 8)),
    ({"num_attention_heads": 12}, (12, 12)),                       # kv defaults to n_head
    ({"text_config": {"num_attention_heads": 4,                    # multimodal-shaped
                      "num_key_value_heads": 2}}, (4, 2)),
])
def test_apply_reads_head_counts_from_config(monkeypatch, config, expected):
    """apply_gguf_adapter pulls n_head / n_head_kv from a dict, an object, or a
    nested text_config, and hands them to the installer."""
    seen = {}
    monkeypatch.setattr(adapter, "load_lora_adapter",
                        lambda path, **kw: _plan({}))
    monkeypatch.setattr("gmlx.load.modules.install_lora_adapter",
                        lambda model, plan, *, n_head=None, n_head_kv=None, slot=0, config=None:
                        seen.update(n_head=n_head, n_head_kv=n_head_kv) or 0)
    adapter.apply_gguf_adapter(object(), config, "ignored.gguf")
    assert (seen["n_head"], seen["n_head_kv"]) == expected


def test_apply_reads_head_counts_from_object_config(monkeypatch):
    class _Cfg:
        def to_dict(self):
            return {"num_attention_heads": 10, "num_key_value_heads": 5}

    seen = {}
    monkeypatch.setattr(adapter, "load_lora_adapter", lambda path, **kw: _plan({}))
    monkeypatch.setattr("gmlx.load.modules.install_lora_adapter",
                        lambda model, plan, *, n_head=None, n_head_kv=None, slot=0, config=None:
                        seen.update(n_head=n_head, n_head_kv=n_head_kv) or 0)
    adapter.apply_gguf_adapter(object(), _Cfg(), "ignored.gguf")
    assert (seen["n_head"], seen["n_head_kv"]) == (10, 5)


class _Multimodal(nn.Module):
    """A base shaped like the vlm runtime's models: the text stack under
    ``language_model``, whose leaf paths the adapter remap targets."""
    def __init__(self):
        super().__init__()
        self.language_model = _Tiny()


def test_apply_enters_the_language_model_of_a_multimodal_base(monkeypatch):
    a = mx.zeros((2, 4))
    b = mx.zeros((4, 2))
    plan = _plan({"q_proj": _lm("q_proj", a, b, 1.0)})
    monkeypatch.setattr(adapter, "load_lora_adapter", lambda path, **kw: plan)
    outer = _Multimodal()
    assert adapter.apply_gguf_adapter(outer, {}, "ignored.gguf") == 1
    assert isinstance(outer.language_model.q_proj, modules.LoRAKQuantLinear)


def test_apply_keeps_the_outer_model_when_the_plan_names_the_prefix(monkeypatch):
    a = mx.zeros((2, 4))
    b = mx.zeros((4, 2))
    path = "language_model.q_proj"
    plan = _plan({path: _lm(path, a, b, 1.0)})
    monkeypatch.setattr(adapter, "load_lora_adapter", lambda path, **kw: plan)
    outer = _Multimodal()
    assert adapter.apply_gguf_adapter(outer, {}, "ignored.gguf") == 1
    assert isinstance(outer.language_model.q_proj, modules.LoRAKQuantLinear)


def test_install_drops_fused_wires_on_the_wrapped_owner():
    """A fused decode wire the upstream patches cached on the attention
    module before the adapter arrived would keep serving the base alone;
    the install clears it so the next forward rebuilds with the adapter."""
    class _Attn(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(4, 4, bias=False)

    class _Outer(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = _Attn()
            self.other = _Attn()

    m = _Outer()
    object.__setattr__(m.attn, "_kq_wqkv", mx.zeros((1,)))
    object.__setattr__(m.attn, "_kq_bqkv", mx.zeros((1,)))
    object.__setattr__(m.other, "_kq_wqkv", mx.zeros((1,)))
    a = mx.zeros((2, 4))
    b = mx.zeros((4, 2))
    plan = _plan({"attn.q_proj": _lm("attn.q_proj", a, b, 1.0)})
    assert modules.install_lora_adapter(m, plan) == 1
    assert vars(m.attn)["_kq_wqkv"] is None and vars(m.attn)["_kq_bqkv"] is None
    assert vars(m.other)["_kq_wqkv"] is not None  # untouched owners keep their wires


# loaded identity: the bytes are copied out of the file and hashed


def _save_adapter(path, a, b, *, alpha=2.0):
    """A one-pair llama adapter on ``model.layers.0.mlp.down_proj``."""
    return adapter.save_lora_adapter(
        str(path), [("model.layers.0.mlp.down_proj", a, b)],
        alpha=alpha, base_arch="llama", n_head=4, n_head_kv=2, n_layers=1)


_DOWN = "model.layers.0.mlp.down_proj"


def test_loaded_adapter_carries_the_sha256_of_its_file(tmp_path):
    import hashlib

    p = tmp_path / "a.gguf"
    _save_adapter(p, mx.ones((1, 8)), mx.ones((4, 1)))
    plan = adapter.load_lora_adapter(str(p), base_arch="llama")
    data = p.read_bytes()
    assert plan.sha256 == hashlib.sha256(data).hexdigest()
    assert plan.nbytes == len(data)


def test_a_rewrite_after_load_leaves_the_loaded_delta(tmp_path):
    """The factors are copies: a same-size rewrite of the file before the
    first forward does not reach the served delta."""
    import shutil

    mx.random.seed(3)
    a, b = mx.random.normal((1, 8)), mx.random.normal((4, 1))
    p, other = tmp_path / "a.gguf", tmp_path / "b.gguf"
    _save_adapter(p, a, b)
    _save_adapter(other, -a, b)                  # the delta negated
    assert p.stat().st_size == other.stat().st_size
    plan = adapter.load_lora_adapter(str(p), base_arch="llama")
    lm = plan.modules[_DOWN]
    base = nn.Linear(8, 4, bias=False)
    base.weight = mx.zeros((4, 8))
    wrap = modules.LoRAKQuantLinear(base, lm.a, lm.b, lm.scale)
    with open(p, "r+b") as dst, open(other, "rb") as src:
        shutil.copyfileobj(src, dst)
    x = mx.ones((1, 8))
    ref = lm.scale * ((x @ a.T) @ b.T)
    assert mx.allclose(wrap(x), ref, atol=1e-5)


def test_a_file_that_changes_while_it_loads_is_refused(tmp_path, monkeypatch):
    p = tmp_path / "a.gguf"
    _save_adapter(p, mx.ones((1, 8)), mx.ones((4, 1)))
    keys = iter(range(100))
    monkeypatch.setattr(adapter, "_file_key", lambda path: next(keys))
    with pytest.raises(ValueError, match="changed while it loaded"):
        adapter.load_lora_adapter(str(p), base_arch="llama")
    # Two calls per try: three tries.
    assert next(keys) == 6


def test_a_file_that_changes_once_loads_on_the_retry(tmp_path, monkeypatch):
    import hashlib

    p = tmp_path / "a.gguf"
    _save_adapter(p, mx.ones((1, 8)), mx.ones((4, 1)))
    real = adapter._file_key
    calls = []

    def key(path):
        calls.append(path)
        if len(calls) == 2:              # the first read saw the old bytes
            _save_adapter(p, 2 * mx.ones((1, 8)), mx.ones((4, 1)))
            return (-1,) + real(path)[1:]
        return real(path)

    monkeypatch.setattr(adapter, "_file_key", key)
    plan = adapter.load_lora_adapter(str(p), base_arch="llama")
    assert len(calls) == 4
    assert plan.sha256 == hashlib.sha256(p.read_bytes()).hexdigest()
    assert mx.array_equal(plan.modules[_DOWN].a, 2 * mx.ones((1, 8)))


def test_apply_returns_the_module_count(tmp_path):
    class _Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = nn.Module()
            self.mlp.down_proj = nn.Linear(8, 4, bias=False)

    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Module()
            self.model.layers = [_Block()]

    p = tmp_path / "a.gguf"
    _save_adapter(p, mx.ones((1, 8)), mx.ones((4, 1)))
    m = _Model()
    assert adapter.apply_gguf_adapter(m, {}, str(p), base_arch="llama") == 1
    assert isinstance(m.model.layers[0].mlp.down_proj, modules.LoRAKQuantLinear)


def test_a_rank_one_lora_a_stored_without_its_unit_dim_loads(tmp_path):
    """ggml writes a rank-1 lora_a of ne [n_in, 1] with one dimension, and
    llama.cpp reads it back as [n_in, 1]."""
    import gguf

    mx.random.seed(5)
    a, b = mx.random.normal((1, 8)), mx.random.normal((4, 1))
    two_d = tmp_path / "two_d.gguf"
    _save_adapter(two_d, a, b)
    one_d = tmp_path / "one_d.gguf"
    w = gguf.GGUFWriter(str(one_d), "llama")
    w.add_type(gguf.GGUFType.ADAPTER)
    w.add_string(gguf.Keys.Adapter.TYPE, "lora")
    w.add_float32(gguf.Keys.Adapter.LORA_ALPHA, 1.0)
    w.add_tensor("blk.0.ffn_down.weight.lora_a", np.array(a.reshape(8)))
    w.add_tensor("blk.0.ffn_down.weight.lora_b", np.array(b))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    want = adapter.load_lora_adapter(str(two_d), base_arch="llama").modules[_DOWN]
    got = adapter.load_lora_adapter(str(one_d), base_arch="llama").modules[_DOWN]
    assert got.rank == 1
    assert mx.array_equal(got.b @ got.a, want.b @ want.a)
