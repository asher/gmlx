"""Verify-width blocks past the sparse boundary on the kq paged-rows
kernel: each query walks its own selected blocks and its own tail block,
with K and V read in place. The outputs match the gathered path with the
rows in the cache tail and in its base, and the blocks the kernel does not
serve keep the gathered path."""

from __future__ import annotations

import functools

import mlx.core as mx
import pytest

from helpers import _real_apple_gpu

import gmlx.models.qwen4_exp.model as q4
from gmlx.models.qwen4_exp.model import Attention, ModelArgs, QSAKVCache


gpu_only = pytest.mark.skipif(
    not _real_apple_gpu(), reason="needs a real Apple GPU")

# relative norm of the difference, per 16-bit float dtype
REL_BOUND = {mx.bfloat16: 5e-3, mx.float16: 2e-3}


def _layer(head_dim, dtype, ratio=4, heads=12):
    args = ModelArgs(hidden_size=128, num_hidden_layers=1,
                     num_attention_heads=heads, num_key_value_heads=2,
                     head_dim=head_dim, indexer_budget=2048,
                     compress_ratios=[ratio], layer_types=["full_attention"])
    mx.random.seed(13)
    layer = Attention(args, 0)
    layer.set_dtype(dtype)
    for lin in (layer.q_proj, layer.k_proj, layer.v_proj, layer.o_proj,
                layer.indexer.q_proj, layer.indexer.k_proj):
        lin.weight = (mx.random.normal(lin.weight.shape) * 0.05).astype(dtype)
    return layer


def _steps(layer, widths, cap, monkeypatch):
    """A prompt past the sparse boundary, then verify blocks of ``widths``
    rows, each rolled back to its first row."""
    monkeypatch.setenv("GMLX_Q4_KV_TAIL", str(cap))
    mx.random.seed(31)
    dtype = layer.q_proj.weight.dtype
    cache = QSAKVCache(ratio=4)
    layer(mx.random.normal((1, 2301, 128)).astype(dtype), mask=None,
          cache=cache)
    outs = []
    for w in widths:
        outs.append(layer(mx.random.normal((1, w, 128)).astype(dtype),
                          mask=None, cache=cache))
        cache.trim(w - 1)
    mx.eval(*outs)
    return outs


@gpu_only
@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
@pytest.mark.parametrize("cap", [8, 0])
def test_rows_kernel_matches_gathered(monkeypatch, dtype, cap):
    """Blocks of 2 to 8 rows land in the tail (``cap`` 8, with folds) or in
    the base (``cap`` 0); blocks of 9 to 16 rows always land in the base."""
    rows = q4._kq_paged_rows()
    if rows is None:
        pytest.skip("kq.sdpa_paged_rows unavailable")
    layer = _layer(256, dtype)
    widths = [2, 3, 8, 5, 4, 12, 7, 16, 6, 2, 9]
    calls = []

    @functools.wraps(rows)
    def spy(*args, **kw):
        calls.append(len(args))
        return rows(*args, **kw)

    monkeypatch.setattr(q4, "_kq_rows_fn", spy)
    got = _steps(layer, widths, cap, monkeypatch)
    assert len(calls) == len(widths)
    # nine operands: the base and the tail; six: one buffer
    assert sum(n == 9 for n in calls) == (8 if cap else 0)
    monkeypatch.setattr(q4, "_kq_rows_fn", None)
    ref = _steps(layer, widths, cap, monkeypatch)
    for w, a, b in zip(widths, got, ref):
        a, b = a.astype(mx.float32), b.astype(mx.float32)
        rel = float(mx.sqrt(((a - b) ** 2).sum() / (b ** 2).sum()))
        assert rel < REL_BOUND[dtype], (w, rel)


@gpu_only
def test_rows_kernel_declines_what_it_does_not_serve(monkeypatch):
    if q4._kq_paged_rows() is None:
        pytest.skip("kq.sdpa_paged_rows unavailable")
    layer = _layer(256, mx.bfloat16)
    q = mx.zeros((1, 12, 4, 256), mx.bfloat16)
    k = mx.zeros((1, 2, 64, 256), mx.bfloat16)
    assert layer._rows_kernel(q, k, k, None, False, 4) is not None
    # one row is the paged decode kernel's; past 16 rows is prefill, served
    # only on tensor-op GPUs and with the lever on
    assert layer._rows_kernel(q, k, k, None, False, 1) is None
    assert (layer._rows_kernel(q, k, k, None, False, 17) is not None) == (
        q4._kq_rows_prefill())
    monkeypatch.setenv("GMLX_Q4_QSA_ROWS_PREFILL", "0")
    assert layer._rows_kernel(q, k, k, None, False, 17) is None
    monkeypatch.delenv("GMLX_Q4_QSA_ROWS_PREFILL")
    # a float32 step after a restore, or a cache in another dtype
    assert layer._rows_kernel(q.astype(mx.float32), k, k, None, False, 4) is None
    assert layer._rows_kernel(q, k.astype(mx.float16), k, None, False, 4) is None
    # a block ratio that is not a page size at this head dim
    assert _layer(64, mx.bfloat16)._rows_kernel(
        mx.zeros((1, 12, 4, 64), mx.bfloat16),
        mx.zeros((1, 2, 64, 64), mx.bfloat16),
        mx.zeros((1, 2, 64, 64), mx.bfloat16), None, False, 4) is None


@gpu_only
@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
def test_rows_kernel_prefill_matches_block_sparse(monkeypatch, dtype):
    """Prefill chunks on tensor-op GPUs: one that crosses the sparse
    boundary (its sparse rows), then whole sparse chunks, 4-aligned and
    ragged. The outputs match the block-sparse prefill kernel."""
    rows = q4._kq_paged_rows()
    if rows is None or q4._kq_bs_prefill() is None or not q4._kq_rows_prefill():
        pytest.skip("needs kq paged rows on a tensor-op GPU")
    # Flash-Next's 24 query heads over 2: the block-sparse kernel's grouping
    layer = _layer(256, dtype, heads=24)
    chunks = [2301, 213, 64, 37]
    calls = []

    @functools.wraps(rows)
    def spy(*args, **kw):
        calls.append(args[0].shape[2])
        return rows(*args, **kw)

    def run():
        mx.random.seed(37)
        cache = QSAKVCache(ratio=4)
        outs = [layer(mx.random.normal((1, n, 128)).astype(dtype), mask=None,
                      cache=cache) for n in chunks]
        mx.eval(*outs)
        return outs

    monkeypatch.setattr(q4, "_kq_rows_fn", spy)
    got = run()
    # the first chunk's 4-aligned sparse rows (queries 2051 to 2299; its
    # ragged last query keeps the gathered path), then every chunk whole
    assert calls == [249, 213, 64, 37]
    monkeypatch.setenv("GMLX_Q4_QSA_ROWS_PREFILL", "0")
    calls.clear()
    ref = run()
    assert calls == []
    for n, a, b in zip(chunks, got, ref):
        a, b = a.astype(mx.float32), b.astype(mx.float32)
        rel = float(mx.sqrt(((a - b) ** 2).sum() / (b ** 2).sum()))
        assert rel < REL_BOUND[dtype], (n, rel)
