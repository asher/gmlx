#!/usr/bin/env python3
"""Live GGUF-LoRA apply (P2): the ``LoRAKQuantLinear`` wrap + ``install_lora_adapter``
leaf-swap. Pins the a/b orientation + scale against a hand-computed reference, the
qk_permute-on-B (q vs k head counts), dtype preservation, and the loud failures on
unwrappable / missing targets. CPU-only (tiny float Linears, no GGUF, no kernels)."""
from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import pytest

from gmlx import adapter, modules  # noqa: E402
from gmlx.transforms import qk_permute_wire  # noqa: E402


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
    monkeypatch.setattr("gmlx.modules.install_lora_adapter",
                        lambda model, plan, *, n_head=None, n_head_kv=None:
                        seen.update(n_head=n_head, n_head_kv=n_head_kv) or 0)
    adapter.apply_gguf_adapter(object(), config, "ignored.gguf")
    assert (seen["n_head"], seen["n_head_kv"]) == expected


def test_apply_reads_head_counts_from_object_config(monkeypatch):
    class _Cfg:
        def to_dict(self):
            return {"num_attention_heads": 10, "num_key_value_heads": 5}

    seen = {}
    monkeypatch.setattr(adapter, "load_lora_adapter", lambda path, **kw: _plan({}))
    monkeypatch.setattr("gmlx.modules.install_lora_adapter",
                        lambda model, plan, *, n_head=None, n_head_kv=None:
                        seen.update(n_head=n_head, n_head_kv=n_head_kv) or 0)
    adapter.apply_gguf_adapter(object(), _Cfg(), "ignored.gguf")
    assert (seen["n_head"], seen["n_head_kv"]) == (10, 5)
