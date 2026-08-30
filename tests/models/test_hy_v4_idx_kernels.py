#!/usr/bin/env python3
"""The kq indexer selection route against the ops argpartition path.

The two agree on the keys that matter and part only at the score
threshold, so the gate is an overlap floor over visible keys, plus the
shape guards that must decline the route.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from gmlx.models.hy_v4 import idx_kernels

H = 32
HD = 128
TOPK = 2048


def _ops(q, k, w, mask, topk):
    scores = mx.maximum(q @ k.swapaxes(-1, -2), 0)
    scores = (scores * w.swapaxes(-1, -2)[..., None]).sum(axis=1,
                                                          keepdims=True)
    if mask is not None:
        scores = mx.where(mask, scores, -float("inf"))
    return mx.argpartition(scores, kth=-topk, axis=-1)[..., -topk:]


def _inputs(length, keys, offset):
    mx.random.seed(11)
    q = (mx.random.normal((1, H, length, HD)) * 0.5).astype(mx.bfloat16)
    k = (mx.random.normal((1, 1, keys, HD)) * 0.5).astype(mx.bfloat16)
    w = (mx.random.normal((1, length, H)) * 0.5).astype(mx.float32)
    mask = None
    if length > 1:
        rows = mx.arange(offset, offset + length)[:, None]
        mask = mx.arange(keys)[None, :] <= rows
    mx.eval(q, k, w, *([mask] if mask is not None else []))
    return q, k, w, mask


def _visible_overlap(kernel, ops, length, keys, offset):
    """Agreement over the keys each row may actually attend to.

    Rows whose causal prefix is shorter than ``topk`` are padded out with
    arbitrary masked indices by both paths, and attention drops those, so
    counting them would compare noise.
    """
    a = np.array(kernel).astype(np.int64)[0, 0]
    b = np.array(ops).astype(np.int64)[0, 0]
    out = []
    for m in range(0, length, max(1, length // 8)):
        last = offset + m
        pa = {int(i) for i in a[m] if i <= last}
        pb = {int(i) for i in b[m] if i <= last}
        out.append(len(pa & pb) / min(last + 1, TOPK))
    return out


@pytest.mark.parametrize("length,keys,offset", [
    (64, 4160, 4096),   # a chunked prefill step past the selection boundary
    (1, 4160, 4159),    # decode
])
def test_the_kernel_selects_what_argpartition_selects(length, keys, offset):
    q, k, w, mask = _inputs(length, keys, offset)
    sel = idx_kernels.select(q, k, w, TOPK, offset, mask)
    if sel is None:
        pytest.skip("kq indexer route unavailable on this device")
    ops = _ops(q, k, w, mask, TOPK)
    mx.eval(sel, ops)
    assert sel.shape == ops.shape
    overlap = _visible_overlap(sel, ops, length, keys, offset)
    assert min(overlap) > 0.99, f"selection diverged: {min(overlap):.4f}"


def test_rows_inside_the_causal_prefix_select_every_visible_key():
    length, keys, offset = 64, 4096, 0
    q, k, w, mask = _inputs(length, keys, offset)
    sel = idx_kernels.select(q, k, w, TOPK, offset, mask)
    if sel is None:
        pytest.skip("kq indexer route unavailable on this device")
    mx.eval(sel)
    picked = np.array(sel).astype(np.int64)[0, 0]
    for m in range(length):
        assert {int(i) for i in picked[m] if i <= m} == set(range(m + 1))


@pytest.mark.parametrize("topk", [0, 1024, 4096])
def test_unsupported_topk_widths_decline(topk):
    q, k, w, mask = _inputs(64, 4160, 4096)
    assert idx_kernels.select(q, k, w, topk, 4096, mask) is None


def test_a_key_count_off_the_tile_declines():
    q, k, w, mask = _inputs(64, 4160, 4096)
    assert idx_kernels.select(q, k[:, :, :4130], w, TOPK, 4096, None) is None


def test_fewer_keys_than_topk_declines():
    # The caller returns None before this point, but the route must not
    # depend on that: a radix select over too few rows has no answer.
    q, k, w, _ = _inputs(64, 1024, 0)
    assert idx_kernels.select(q, k, w, TOPK, 0, None) is None


def test_a_mask_the_kernel_cannot_rebuild_declines():
    q, k, w, mask = _inputs(64, 4160, 4096)
    assert idx_kernels.select(q, k, w, TOPK, 4096, mask[:, :16]) is None


def test_a_chunk_off_the_query_tile_declines():
    q, k, w, mask = _inputs(70, 4160, 4096)
    assert idx_kernels.select(q, k, w, TOPK, 4096, mask) is None


def test_the_env_flag_returns_the_ops_path(monkeypatch):
    q, k, w, mask = _inputs(64, 4160, 4096)
    monkeypatch.setenv("GMLX_HY4_IDX_KERNEL", "0")
    assert idx_kernels.select(q, k, w, TOPK, 4096, mask) is None
