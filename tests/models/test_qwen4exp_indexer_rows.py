"""The QSA indexer at verify widths: the fused block finish and query
norm and rope equal the op chains bit for bit, and one scorer call over the
whole verify block and both block key segments selects what calls of 4
queries over one segment select."""

from __future__ import annotations

import mlx.core as mx
import pytest

from helpers import _real_apple_gpu

import gmlx.models.qwen4_exp.model as q4
from gmlx.models.qwen4_exp.model import Attention, ModelArgs, QSAKVCache


gpu_only = pytest.mark.skipif(
    not _real_apple_gpu(), reason="needs a real Apple GPU")


def _layer(dtype, budget=2048, head_dim=256):
    args = ModelArgs(hidden_size=128, num_hidden_layers=1,
                     num_attention_heads=12, num_key_value_heads=2,
                     head_dim=head_dim, indexer_budget=budget,
                     compress_ratios=[4], layer_types=["full_attention"])
    mx.random.seed(13)
    layer = Attention(args, 0)
    layer.set_dtype(dtype)
    for lin in (layer.q_proj, layer.k_proj, layer.v_proj, layer.o_proj,
                layer.indexer.q_proj, layer.indexer.k_proj):
        lin.weight = (mx.random.normal(lin.weight.shape) * 0.05).astype(dtype)
    for norm in (layer.indexer.k_norm, layer.indexer.q_norm):
        norm.weight = (mx.random.uniform(0.3, 2.0, norm.weight.shape)
                       ).astype(dtype)
    return layer


def _bits(a):
    return a.view(mx.uint32 if a.dtype == mx.float32 else mx.uint16)


@gpu_only
@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16, mx.float32])
@pytest.mark.parametrize("weight_f32", [False, True])
def test_fused_block_finish_equals_the_op_chain(monkeypatch, dtype, weight_f32):
    ix = _layer(dtype).indexer
    if weight_f32:
        ix.k_norm.weight = ix.k_norm.weight.astype(mx.float32)
    mx.random.seed(3)
    for B, m, start in ((1, 1, 0), (1, 2, 517), (2, 5, 1029), (1, 3, 40000)):
        raw = (mx.random.normal((B, m * 4, ix.head_dim)) * 2).astype(dtype)
        fused = ix.finish_blocks(raw, start)
        assert ix._fused_finish(raw, start) is not None
        monkeypatch.setattr(q4, "_norm_rope", lambda: None)
        ref = ix.finish_blocks(raw, start)
        monkeypatch.undo()
        assert fused.dtype == ref.dtype
        assert mx.array_equal(_bits(fused), _bits(ref)).item(), (B, m, start)


@gpu_only
@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16, mx.float32])
@pytest.mark.parametrize("weight_f32", [False, True])
@pytest.mark.parametrize("out", [mx.bfloat16, mx.float16])
def test_fused_queries_equal_the_op_chain(dtype, weight_f32, out):
    ix = _layer(dtype).indexer
    if weight_f32:
        ix.q_norm.weight = ix.q_norm.weight.astype(mx.float32)
    mx.random.seed(4)
    for B, L, offset in ((1, 1, 0), (1, 3, 2305), (2, 8, 4151), (1, 5, 40000)):
        x = mx.random.normal((B, L, 128)).astype(dtype)
        fused = ix._fused_queries(x, offset, out)
        assert fused is not None
        ref = ix._queries(x, offset, None, None).astype(out)
        assert fused.shape == ref.shape and fused.dtype == out
        assert mx.array_equal(_bits(fused), _bits(ref)).item(), (B, L, offset)


@gpu_only
def test_fused_norm_rope_declines_a_scaled_rope():
    ix = _layer(mx.bfloat16).indexer
    raw = mx.zeros((1, 8, ix.head_dim), mx.bfloat16)
    ix._rope_scaled = object()
    assert ix._fused_finish(raw, 0) is None
    assert ix._fused_queries(mx.zeros((1, 2, 128), mx.bfloat16), 9,
                             mx.bfloat16) is None


@gpu_only
@pytest.mark.parametrize("L", [2, 5, 8])
def test_one_scorer_call_selects_as_four_query_calls(monkeypatch, L):
    if (q4._kq_score() is None or q4._kq_topk() is None
            or not q4._kq_score_wide()):
        pytest.skip("kq scorer without 8-row and key-tail calls")
    monkeypatch.setenv("GMLX_Q4_KV_TAIL", "64")
    layer = _layer(mx.bfloat16)
    mx.random.seed(5)
    cache = QSAKVCache(ratio=4)
    layer(mx.random.normal((1, 2301, 128)).astype(mx.bfloat16), mask=None,
          cache=cache)
    for _ in range(9):  # decode steps put block keys in the tail
        layer(mx.random.normal((1, 1, 128)).astype(mx.bfloat16), mask=None,
              cache=cache)
    x = mx.random.normal((1, L, 128)).astype(mx.bfloat16)
    _, _, k, v, ik = layer._project(x)
    offset = cache.offset
    cache.append_qsa(k, v, ik)
    calls = []
    real = q4._kq_score()

    def spy(*args, **kw):
        calls.append(kw.get("keys_tail") is not None)
        return real(*args, **kw)

    monkeypatch.setattr(q4, "_kq_score", lambda: spy)
    one, _ = layer.indexer.select(x, None, cache, offset, key_len=cache.offset)
    assert calls == [True]
    monkeypatch.setattr(q4, "_kq_score_wide_ok", False)
    calls.clear()
    split, _ = layer.indexer.select(x, None, cache, offset,
                                    key_len=cache.offset)
    assert len(calls) == 2 * ((L + 3) // 4)
    assert one.dtype == mx.int32
    assert mx.array_equal(mx.sort(one, axis=-1),
                          mx.sort(split.astype(mx.int32), axis=-1)).item()
