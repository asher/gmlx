#!/usr/bin/env python3
"""Token embedding LoRA: the ``LoRAKQuantEmbedding`` wrap and its install.
Pins the ``token_embd`` layout (A ``(n_vocab, r)``, B ``(n_embd, r)``)
against a numpy reference, the base's output dtype, the owned copy of A,
the resident bytes, attribute forwarding, and the install's refusals.
CPU-only (tiny tables, minted adapter files)."""
from __future__ import annotations

import gc

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

import gmlx.load.adapter as adapter
import gmlx.load.modules as modules

gguf = pytest.importorskip("gguf")

V, D, R = 12, 6, 2


def _ar(shape, k=1.0):
    n = int(np.prod(shape))
    return (np.arange(n, dtype=np.float32).reshape(shape) % 7 - 3) * k


def _embedding(dtype=mx.float32):
    e = nn.Embedding(V, D)
    e.weight = mx.array(_ar((V, D), 0.25)).astype(dtype)
    return e


def _pair():
    return mx.array(_ar((V, R), 0.5)), mx.array(_ar((D, R), 0.125))


def _ref(e, a, b, ids, scale):
    w = np.array(e.weight.astype(mx.float32))
    return w[ids] + scale * (np.array(a)[ids] @ np.array(b).T)


@pytest.mark.parametrize("ids", [np.array([[1, 5, 11], [0, 2, 2]]), np.array([3, 7, 0])],
                         ids=["batch", "flat"])
def test_the_forward_is_the_lookup_plus_the_scaled_gathered_product(ids):
    a, b = _pair()
    with mx.stream(mx.cpu):
        e = _embedding()
        w = modules.LoRAKQuantEmbedding(e, a, b, 0.75)
        got = np.array(w(mx.array(ids)))
    assert got.dtype == np.float32
    assert np.array_equal(got, _ref(e, a, b, ids, 0.75))


def test_the_bfloat16_forward_is_within_one_rounding():
    a, b = _pair()
    e = _embedding(mx.bfloat16)
    w = modules.LoRAKQuantEmbedding(e, a, b, 0.75)
    ids = np.array([[1, 5, 11], [0, 2, 2]])
    got = w(mx.array(ids))
    assert got.dtype == mx.bfloat16
    ref = _ref(e, a, b, ids, 0.75)
    assert np.allclose(np.array(got.astype(mx.float32)), ref, rtol=2 ** -7, atol=2 ** -7)


def _q8_table():
    from gguf import quants
    from gguf.constants import GGMLQuantizationType
    from mlx_kquant.nn import KQuantEmbedding

    dims = 32
    e = KQuantEmbedding(V, dims, "q8_0", out_dtype=mx.bfloat16)
    wires = quants.quantize(_ar((V, dims), 0.05), GGMLQuantizationType.Q8_0).astype(np.uint8)
    e.weight = mx.array(wires.reshape(V, -1))
    mx.eval(e.parameters())
    return e, dims


def test_the_output_dtype_is_the_bases():
    a, b = _pair()
    w = modules.LoRAKQuantEmbedding(_embedding(mx.float16), a, b, 1.0)
    assert w(mx.array([[1, 2]])).dtype == mx.float16
    e, dims = _q8_table()
    wq = modules.LoRAKQuantEmbedding(e, a, mx.array(_ar((dims, R), 0.1)), 1.0)
    y = wq(mx.array([[1, 2]]))
    assert y.dtype == mx.bfloat16 and y.shape == (1, 2, dims)


def test_attributes_read_through_to_the_base():
    a, b = _pair()
    e = _embedding()
    w = modules.LoRAKQuantEmbedding(e, a, b, 1.0)
    assert w.weight is e.weight and not hasattr(w, "biases")
    assert not hasattr(w, "lora_b") and "lora_b" not in w
    x = mx.ones((1, D))
    assert mx.array_equal(w.as_linear(x), e.as_linear(x))
    q = nn.QuantizedEmbedding(V, 64, group_size=32, bits=4)
    wq = modules.LoRAKQuantEmbedding(q, a, mx.array(_ar((64, R))), 1.0)
    assert hasattr(wq, "biases") and wq.biases is q.biases and wq.scales is q.scales
    assert wq.bits == 4 and wq.group_size == 32


def test_the_distill_off_switch_runs_the_base_without_a_table_rebuild():
    from gmlx.distill.student import adapter_disabled

    a, b = _pair()
    with mx.stream(mx.cpu):
        e = _embedding()
        w = modules.LoRAKQuantEmbedding(e, a, b, 1.0)
        m = nn.Module()
        m.embed_tokens = w
        ids = mx.array([[1, 4]])
        on = w(ids)
        with adapter_disabled(m):
            assert mx.array_equal(w(ids), e(ids))
        adapter_disabled._drop_tables(w)
        assert mx.array_equal(w(ids), on)


def test_a_second_adapter_takes_a_slot_and_adds_its_delta():
    a, b = _pair()
    a2, b2 = mx.array(_ar((V, 3), -0.25)), mx.array(_ar((D, 3), 0.5))
    with mx.stream(mx.cpu):
        e = _embedding()
        w = modules.LoRAKQuantEmbedding(e, a, b, 0.5)
        w.add_slot(a2, b2, 2.0, 1)
        assert w.slots == (0, 1)
        ids = np.array([[2, 9]])
        got = np.array(w(mx.array(ids)))
    want = _ref(e, a, b, ids, 0.5) + 2.0 * (np.array(a2)[ids] @ np.array(b2).T)
    assert np.allclose(got, want, rtol=1e-6, atol=1e-6)
    with pytest.raises(ValueError, match="already installed"):
        w.add_slot(a2, b2, 1.0, 1)


# the install


class _Inner(nn.Module):
    def __init__(self, dtype=mx.float32):
        super().__init__()
        self.embed_tokens = _embedding(dtype)
        self.q_proj = nn.Linear(D, D, bias=False)


class _Tiny(nn.Module):
    def __init__(self, dtype=mx.float32):
        super().__init__()
        self.model = _Inner(dtype)


def _plan(a, b, scale=1.0):
    lm = adapter.LoraModule(module_path="model.embed_tokens", a=a, b=b,
                            rank=int(a.shape[1]), scale=scale, embedding=True)
    return adapter.LoraAdapter(alpha=0.0, arch="llama", modules={lm.module_path: lm})


def test_the_install_wraps_the_embedding_of_an_untied_model():
    m = _Tiny()
    a, b = _pair()
    assert modules.install_lora_adapter(m, _plan(a, b), config={"tie_word_embeddings": False}) == 1
    assert isinstance(m.model.embed_tokens, modules.LoRAKQuantEmbedding)
    assert modules.lora_base(m.model.embed_tokens).weight is m.model.embed_tokens.weight
    a2, b2 = _pair()
    modules.install_lora_adapter(m, _plan(a2, b2), slot=1)
    assert m.model.embed_tokens.slots == (0, 1)


@pytest.mark.parametrize("config", [{"tie_word_embeddings": True},
                                    {"text_config": {"tie_word_embeddings": True}}],
                         ids=["top", "text_config"])
def test_a_tied_model_is_refused(config):
    a, b = _pair()
    with pytest.raises(ValueError, match="output head"):
        modules.install_lora_adapter(_Tiny(), _plan(a, b), config=config)


def test_a_pair_that_does_not_fit_is_refused():
    with pytest.raises(ValueError, match="does not fit the embedding"):
        modules.install_lora_adapter(_Tiny(), _plan(mx.zeros((V + 1, R)), mx.zeros((D, R))))


def test_a_granite_switch_model_is_refused():
    a, b = _pair()
    with pytest.raises(ValueError, match="Granite Switch"):
        modules.install_lora_adapter(_Tiny(), _plan(a, b), config={"model_type": "granite_switch"})


def test_a_folded_table_is_refused():
    m = _Tiny()
    object.__setattr__(m.model.embed_tokens, "_hadamard", object())
    a, b = _pair()
    with pytest.raises(ValueError, match="Hadamard-rotated"):
        modules.install_lora_adapter(m, _plan(a, b))


def test_an_embedding_pair_on_a_linear_is_refused():
    lm = adapter.LoraModule(module_path="model.q_proj", a=mx.zeros((D, R)), b=mx.zeros((D, R)),
                            rank=R, scale=1.0, embedding=True)
    plan = adapter.LoraAdapter(alpha=0.0, arch="llama", modules={lm.module_path: lm})
    with pytest.raises(NotImplementedError, match="only the token embedding"):
        modules.install_lora_adapter(_Tiny(), plan)


# the adapter file


def _mint_adapter(path, a, b, dtype=np.float32):
    w = gguf.GGUFWriter(str(path), "llama")
    w.add_string("adapter.type", "lora")
    w.add_float32("adapter.lora.alpha", float(a.shape[1]))
    w.add_tensor("token_embd.weight.lora_a", np.asarray(a, dtype))
    w.add_tensor("token_embd.weight.lora_b", np.asarray(b, dtype))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return str(path)


def _rewrite(path):
    reader = gguf.GGUFReader(path)
    spans = [(int(t.data_offset), int(t.n_bytes)) for t in reader.tensors]
    del reader
    with open(path, "r+b") as fh:
        for off, n in spans:
            fh.seek(off)
            fh.write(b"\x7f" * n)


@pytest.mark.parametrize("dtype", [np.float32, np.float16], ids=["f32", "f16"])
def test_the_file_round_trip_serves_the_numpy_layout(tmp_path, dtype):
    """The ggml ne order of a token_embd pair, through a minted file, lands
    as A (n_vocab, r) and B (n_embd, r)."""
    a, b = _pair()
    path = _mint_adapter(tmp_path / "e.gguf", np.array(a), np.array(b), dtype)
    ids = np.array([[1, 3, 8], [11, 0, 5]])
    with mx.stream(mx.cpu):
        m = _Tiny()
        adapter.apply_gguf_adapter(m, {"tie_word_embeddings": False}, path, base_arch="llama")
        got = np.array(m.model.embed_tokens(mx.array(ids)))
    assert np.array_equal(got, _ref(m.model.embed_tokens.base, a, b, ids, 1.0))


def test_a_rewrite_of_the_adapter_file_does_not_reach_the_served_rows(tmp_path):
    """The loader copies the pair out of the file."""
    a, b = _pair()
    path = _mint_adapter(tmp_path / "e.gguf", np.array(a), np.array(b))
    m = _Tiny()
    adapter.apply_gguf_adapter(m, {"tie_word_embeddings": False}, path, base_arch="llama")
    ids = mx.array([[1, 3, 8]])
    before = np.array(m.model.embed_tokens(ids))
    _rewrite(path)
    assert np.array_equal(np.array(m.model.embed_tokens(ids)), before)


def test_the_wrapper_owns_its_a_over_a_zero_copy_plan(tmp_path):
    """A plan whose A is a view of the mapped file: the wrapper's own f32
    copy keeps a rewrite of the file from its rows."""
    from gmlx.load.wire import load_gguf_wire_bytes

    a, b = _pair()
    path = _mint_adapter(tmp_path / "e.gguf", np.array(a), np.array(b))
    arrays, _q, _arch, meta, _s = load_gguf_wire_bytes(path, zero_copy=True, expect_quant=False)
    plan = adapter.build_adapter_plan(meta, arrays, base_arch="llama")
    m = _Tiny()
    adapter.install_gguf_adapter(m, {"tie_word_embeddings": False}, plan)
    w = m.model.embed_tokens
    assert w.lora_a is not plan.modules["model.embed_tokens"].a
    assert w.lora_a.dtype == mx.float32
    ids = mx.array([[1, 3, 8]])
    before = np.array(w(ids))
    del arrays, plan
    _rewrite(path)
    assert np.array_equal(np.array(w(ids)), before)


def test_the_install_keeps_only_the_f32_a_and_one_table(tmp_path):
    v, d, r = 4096, 1024, 16
    rng = np.random.default_rng(0)
    a = rng.standard_normal((v, r)).astype(np.float16)
    b = rng.standard_normal((d, r)).astype(np.float16)
    path = _mint_adapter(tmp_path / "big.gguf", a, b, np.float16)

    inner = nn.Module()
    inner.embed_tokens = nn.Embedding(v, d)
    inner.embed_tokens.weight = mx.zeros((v, d), mx.bfloat16)
    m = nn.Module()
    m.model = inner
    mx.eval(m.parameters())
    gc.collect()
    mx.clear_cache()
    before = mx.get_active_memory()
    adapter.apply_gguf_adapter(m, {"tie_word_embeddings": False}, path, base_arch="llama")
    gc.collect()
    mx.clear_cache()
    rise = mx.get_active_memory() - before
    resident = 4 * v * r + 2 * d * r
    pair_bytes = 2 * r * (v + d)
    assert resident <= rise <= resident + 16384
    assert rise <= 2 * pair_bytes
