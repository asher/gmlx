"""Vendored kernels that read their row count from an input shape at run
time. One compiled kernel serves every count, so each count must still
index its own rows."""

from __future__ import annotations

import types

import mlx.core as mx
import numpy as np
import pytest

from gmlx._vendor.mlx_vlm.models.inkling import language as inkling
from gmlx._vendor.mlx_vlm.models.minimax_m3_vl import language as minimax

pytestmark = pytest.mark.skipif(
    mx.default_device() != mx.gpu or not mx.metal.is_available(),
    reason="the kernels run on the GPU",
)


def _route(logits, bias, device):
    moe = types.SimpleNamespace(
        n_routed=64,
        n_shared=1,
        top_k=4,
        intermediate_size=8,
        route_scale=2.5,
        global_scale=mx.array([1.0]),
        e_score_correction_bias=bias,
        _wscale=None,
    )
    old = mx.default_device()
    mx.set_default_device(device)
    try:
        out = inkling.InklingSparseMoE._route(moe, logits)
        mx.eval(out)
    finally:
        mx.set_default_device(old)
    return [np.array(a) for a in out]


@pytest.mark.parametrize("rows", [[1, 3, 130], [130, 7, 1]])
def test_the_inkling_router_matches_its_ops_at_every_row_count(rows):
    rng = np.random.default_rng(0)
    bias = mx.array(rng.normal(size=64).astype(np.float32) * 0.1)
    for n in rows:
        logits = mx.array(rng.normal(size=(n, 65)).astype(np.float32))
        k_idx, k_w, k_gamma = _route(logits, bias, mx.gpu)
        o_idx, o_w, o_gamma = _route(logits, bias, mx.cpu)
        # The kernel lists experts by rank; argpartition in no set order.
        k_order = np.argsort(k_idx, axis=-1)
        o_order = np.argsort(o_idx, axis=-1)
        np.testing.assert_array_equal(
            np.take_along_axis(k_idx, k_order, -1),
            np.take_along_axis(o_idx, o_order, -1),
        )
        np.testing.assert_allclose(
            np.take_along_axis(k_w, k_order, -1),
            np.take_along_axis(o_w, o_order, -1),
            rtol=1e-5,
            atol=1e-6,
        )
        np.testing.assert_allclose(k_gamma, o_gamma, rtol=1e-5, atol=1e-6)


def _sparse_reference(q, k, v, blocks, positions, scale, block):
    B, H, L, D = q.shape
    group = H // k.shape[1]
    out = np.zeros(q.shape, np.float32)
    for b in range(B):
        for h in range(H):
            for i in range(L):
                keys = [
                    blk * block + o
                    for blk in blocks[b, i]
                    for o in range(block)
                    if blk * block + o <= positions[b, i]
                ]
                kk = k[b, h // group, keys]
                s = (q[b, h, i] * scale) @ kk.T
                p = np.exp(s - s.max())
                out[b, h, i] = (p / p.sum()) @ v[b, h // group, keys]
    return out


@pytest.mark.parametrize("lengths", [[2, 5, 1], [5, 1, 2]])
def test_the_minimax_sparse_prefill_matches_a_reference_at_every_length(lengths):
    rng = np.random.default_rng(1)
    B, H, KVH, D, block, topk = 1, 4, 2, 64, 16, 2
    K = block * topk * 8
    scale = D**-0.5
    k = rng.normal(size=(B, KVH, K, D)).astype(np.float16)
    v = rng.normal(size=(B, KVH, K, D)).astype(np.float16)
    for L in lengths:
        q = rng.normal(size=(B, H, L, D)).astype(np.float16)
        positions = np.array([[K - L + i for i in range(L)]], np.int32)
        blocks = np.stack(
            [rng.choice(p // block + 1, size=topk, replace=False) for p in positions[0]]
        )[None].astype(np.int32)
        out = minimax._minimax_m3_sparse_prefill_attention(
            mx.array(q),
            mx.array(k),
            mx.array(v),
            mx.array(blocks),
            mx.array(positions),
            scale,
            block,
        )
        assert out is not None
        want = _sparse_reference(
            q.astype(np.float32),
            k.astype(np.float32),
            v.astype(np.float32),
            blocks,
            positions,
            scale,
            block,
        )
        np.testing.assert_allclose(
            np.array(out.astype(mx.float32)), want, atol=3e-3, rtol=3e-3
        )
