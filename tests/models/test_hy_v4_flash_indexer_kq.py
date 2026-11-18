"""Hy4-flash pooled indexer: the kq decode route against a fp32 reference.

The route stores its scores in 16 bits, so it may part from the fp32 top
set on pools that tie at the threshold in 16 bits. The gate is that every
pick is within that rounding of the threshold and nothing clearly above
it is missed.
"""
from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten

from gmlx.models.hy_v4_flash import model as flash
from gmlx.models.hy_v4_flash.model import HyV4FlashIndexer, ModelArgs

KPOOL, TOPK = 4, 2048
SELECT = TOPK // KPOOL - 1
HIDDEN, RANK = 64, 32


def _args() -> ModelArgs:
    return ModelArgs(
        model_type="hy_v4_flash", vocab_size=32, hidden_size=HIDDEN,
        intermediate_size=8, moe_intermediate_size=8, num_hidden_layers=1,
        num_attention_heads=[2], q_lora_rank=RANK, kv_lora_rank=8,
        qk_nope_head_dim=4, qk_rope_head_dim=64, v_head_dim=4,
        n_routed_experts=4, num_experts_per_tok=2, n_shared_experts=1,
        first_k_dense_replace=1, rms_norm_eps=1e-5, hc_mult=4, hc_eps=1e-6,
        hc_magnitude=2.0, sliding_window=8, sliding_window_pattern=[False],
        rope_theta=1e7, rope_theta_swa=1e4, moe_latent_size=8,
        index_n_heads=16, index_head_dim=128, index_topk=TOPK,
        index_kpool=KPOOL, index_select_tail=True, index_types=[True],
        engram_layer_ids=[], engram_max_ngram_size=3, engram_n_heads=2,
        engram_head_dim=8, engram_multipliers=[], engram_primes=[],
        engram_offsets=[], engram_pad_id=0, engram_token_map=[],
        engram_table_rows=0)


def _indexer(dtype) -> HyV4FlashIndexer:
    mx.random.seed(3)
    idx = HyV4FlashIndexer(_args(), 1e7)
    upd = []
    for k, v in tree_flatten(idx.parameters()):
        keep32 = k in ("weights_proj.weight", "ape")
        upd.append((k, mx.random.normal(v.shape, scale=0.3).astype(
            mx.float32 if keep32 else dtype)))
    idx.load_weights(upd)
    mx.eval(idx.parameters())
    return idx


def _ref_scores(idx, x, qr, pooled, offset):
    """fp64 scores [L, P] of the queries the indexer builds from ``qr``."""
    B, L, _ = x.shape
    q = idx.wq_b(qr).reshape(B, L, idx.n_heads, idx.head_dim)
    q = q.transpose(0, 2, 1, 3)
    nope, pe = mx.split(q, [idx.nope_head_dim], axis=-1)
    q = mx.concatenate([nope, idx.rope(pe, offset)], axis=-1)
    w = x.astype(mx.float32) @ idx.weights_proj.weight.T * idx._w_scale
    qf = np.array(q.astype(mx.float32))[0].astype(np.float64)
    kf = np.array(pooled.astype(mx.float32))[0].astype(np.float64)
    wf = np.array(w)[0].astype(np.float64)
    dots = np.maximum(np.einsum("hjd,pd->hjp", qf, kf), 0.0)
    return np.einsum("jh,hjp->jp", wf, dots)


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
@pytest.mark.parametrize("length", [1, 3, 4])
def test_the_kernel_route_picks_the_fp32_top_pools(dtype, length):
    if (mx.default_device().type != mx.DeviceType.gpu
            or flash._kq_indexer_kernels() is None):
        pytest.skip("kq indexer kernels unavailable")
    idx = _indexer(dtype)
    pool = flash.DS4PoolingCache(KPOOL, lookback=False)
    mx.random.seed(5)
    # A prefill that leaves a partial group, then steps that complete it.
    n0 = 3 * TOPK + 2
    x0 = mx.random.normal((1, n0, HIDDEN)).astype(dtype)
    qr0 = mx.random.normal((1, n0, RANK)).astype(dtype)
    idx(x0, qr0, pool, 0)
    offset = n0
    for step in range(6):
        x = mx.random.normal((1, length, HIDDEN)).astype(dtype)
        qr = mx.random.normal((1, length, RANK)).astype(dtype)
        sel = idx(x, qr, pool, offset)
        mx.eval(sel)
        assert sel.shape == (1, length, SELECT) and sel.dtype == mx.int32
        pooled = pool.pooled
        assert pooled.dtype == mx.float16
        ref = _ref_scores(idx, x, qr, pooled, offset)
        got = np.array(sel)[0]
        for j in range(length):
            visible = (offset + j) // KPOOL
            picks = got[j]
            assert len(set(picks.tolist())) == SELECT
            assert picks.min() >= 0 and picks.max() < visible
            row = ref[j, :visible]
            thr = np.partition(row, -SELECT)[-SELECT]
            # A 16-bit score keeps 11 bits; the head sum adds fp32 noise.
            tol = abs(thr) * 2.0 ** -9 + 1e-4
            assert row[picks].min() >= thr - tol, (step, j)
            clear = np.flatnonzero(row > thr + tol)
            assert np.isin(clear, picks).all(), (step, j)
        offset += length


def test_the_switch_restores_the_fp32_selection(monkeypatch):
    if (mx.default_device().type != mx.DeviceType.gpu
            or flash._kq_indexer_kernels() is None):
        pytest.skip("kq indexer kernels unavailable")
    monkeypatch.setenv("GMLX_HY4_IDX_KQ", "0")
    idx = _indexer(mx.bfloat16)
    pool = flash.DS4PoolingCache(KPOOL, lookback=False)
    mx.random.seed(6)
    n0 = TOPK + 400
    idx(mx.random.normal((1, n0, HIDDEN)).astype(mx.bfloat16),
        mx.random.normal((1, n0, RANK)).astype(mx.bfloat16), pool, 0)
    x = mx.random.normal((1, 1, HIDDEN)).astype(mx.bfloat16)
    qr = mx.random.normal((1, 1, RANK)).astype(mx.bfloat16)
    sel = idx(x, qr, pool, n0)
    ref = _ref_scores(idx, x, qr, pool.pooled, n0)[0, :n0 // KPOOL]
    want = np.argpartition(-ref, SELECT - 1)[:SELECT]
    # fp32 against fp64 can part on a tie: allow one pool.
    assert len(set(np.array(sel)[0, 0].tolist()) ^ set(want.tolist())) <= 2
