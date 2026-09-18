"""The V4.1 indexer's kq kernel path (dsa_indexer_scores + dsa_topk_indices)
selects the same positions as the inline fp32 path, up to fp16 rounding at
the top-k threshold, with and without the pooled-visibility mask."""
from __future__ import annotations

import os
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from gmlx.models.deepseek_v4 import model as _v4
from gmlx.models.deepseek_v41.model import Indexer, SharedStreams, _indexer_qat

pytestmark = pytest.mark.skipif(
    not mx.metal.is_available() or os.environ.get("KQUANT_FORCE_CPU") == "1"
    or not _v4._dsa_probe("indexer"),
    reason="kq indexer kernels need Metal",
)

H, D, K = 32, 128, 512


@pytest.fixture
def indexer():
    args = SimpleNamespace(
        hidden_size=48, q_lora_rank=24, head_dim=64, rms_norm_eps=1e-6,
        index_n_heads=H, index_head_dim=D, index_topk=K,
        candidate_source_layer=-1, candidate_topk_blocks=0,
        candidate_block_size=0,
    )
    mx.random.seed(7)
    idx = Indexer(args, layer_idx=2, owns_k=True)
    return args, idx


class _Rope:
    def __call__(self, x, offset=0):
        return x


def _np(a):
    return np.array(a.astype(mx.float32))


def _keys(P):
    """QAT-rounded fp16 keys, as make_keys stores them. The int8 arm packs
    both operands to the FP4 grid, where a QAT row is a fixed point."""
    return _indexer_qat(mx.random.normal((1, P, D))).astype(mx.float16)


def _reference(idx, x, q_residual, index_k, pmask):
    from gmlx.models.deepseek_v41.model import _indexer_qat

    B, L, _ = x.shape
    q = idx.wq_b(q_residual).reshape(B, L, idx.n_heads, idx.head_dim)
    q = _np(_indexer_qat(q.transpose(0, 2, 1, 3)))
    scores = np.maximum(np.einsum("bhld,bpd->bhlp", q, _np(index_k)), 0.0)
    scores = scores * idx.scale
    weights = _np(x) @ _np(idx.weights_proj.weight).T * idx.n_heads ** -0.5
    scores = (scores * weights.transpose(0, 2, 1)[..., None]).sum(axis=1)
    if pmask is not None:
        scores = np.where(np.array(pmask)[None], scores, -np.inf)
    return scores


def _check(got, scores):
    """Every pick scores at least the k-th best minus the fp16 rounding
    slack, and every position clearly above the k-th best is picked."""
    B, L, _ = scores.shape
    for b in range(B):
        for t in range(L):
            row = scores[b, t]
            picks = got[b, t].tolist()
            assert len(set(picks)) == K
            kth = np.sort(row)[::-1][K - 1]
            tol = 4e-3 * np.abs(row[np.isfinite(row)]).max()
            assert row[picks].min() >= kth - tol
            must = set(np.nonzero(row > kth + tol)[0].tolist())
            assert must <= set(picks)


@pytest.mark.parametrize("L,P", [(70, 1100), (64, 1024)])
def test_kernel_picks_match_the_inline_path(indexer, monkeypatch, L, P):
    args, idx = indexer
    x = mx.random.normal((1, L, args.hidden_size))
    q_residual = mx.random.normal((1, L, args.q_lora_rank))
    index_k = _keys(P)

    monkeypatch.setitem(_v4._dsa_state, "indexer", True)
    got = np.array(idx(x, q_residual, _Rope(), index_k, None, 0, SharedStreams()))
    _check(got, _reference(idx, x, q_residual, index_k, None))


@pytest.mark.parametrize("ratio,offset,block", [
    (2, 1400, 0), (1, 600, 0), (1, 600, 32), (2, 1400, 32),
])
def test_kernel_honors_the_pool_mask(indexer, monkeypatch, ratio, offset, block):
    """The causal tile skip arms under the pool mask: for ratio 1 and 2,
    with the query block loop on (block 32 cuts the 70 queries into
    blocks whose causal offsets differ) and off."""
    args, idx = indexer
    L, P = 70, 1100
    x = mx.random.normal((1, L, args.hidden_size))
    q_residual = mx.random.normal((1, L, args.q_lora_rank))
    index_k = _keys(P)
    pmask = _v4._cacheless_pool_mask(P, L, offset, ratio)
    assert pmask is not None and int(pmask.sum(axis=-1).min()) >= K
    assert int(pmask.sum(axis=-1).max()) < P
    streams = SharedStreams()
    streams.ratio = ratio

    monkeypatch.setitem(_v4._dsa_state, "indexer", True)
    monkeypatch.setattr(_v4, "_PREFILL_BLOCK", block)
    got = np.array(idx(x, q_residual, _Rope(), index_k, pmask, offset, streams))
    scores = _reference(idx, x, q_residual, index_k, pmask)
    _check(got, scores)
    hidden = ~np.array(pmask)
    for t in range(L):
        assert not hidden[t, got[0, t]].any()


def test_packed_rows_are_a_fixed_point_of_the_int8_pack():
    """The int8 arm's precondition: V4.1's QAT rows (FP4 block 32, no
    Hadamard) pack losslessly, so dsa_indexer_scores_q sees the same
    operands as the fp16 GEMM."""
    from gmlx.models.deepseek_v41.model import _indexer_qat

    kq = pytest.importorskip("mlx_kquant")
    if not hasattr(kq, "dsa_indexer_qat_pack"):
        pytest.skip("kq without the indexer pack")
    rows = _indexer_qat(mx.random.normal((1, 1, 96, D))).astype(mx.float16)
    codes, scales = kq.dsa_indexer_qat_pack(rows)
    back = (codes.astype(mx.float32).reshape(1, 1, 96, 4, 32)
            * scales[..., None]).reshape(1, 1, 96, D)
    assert mx.array_equal(back, rows.astype(mx.float32))


def _count_decode_calls(monkeypatch):
    kq = pytest.importorskip("mlx_kquant")
    if not hasattr(kq, "dsa_indexer_score_decode"):
        pytest.skip("kq without the decode scorer")
    real = kq.dsa_indexer_score_decode
    calls = []

    def counted(q, keys, w, q_offset, ratio):
        calls.append((q.shape[2], q_offset, ratio, keys.dtype))
        return real(q, keys, w, q_offset, ratio)

    monkeypatch.setattr(kq, "dsa_indexer_score_decode", counted)
    return calls


@pytest.mark.parametrize("L,offset,ratio", [(1, None, 2), (3, 1400, 2), (2, 600, 1)])
def test_decode_widths_take_the_fused_scorer(indexer, monkeypatch, L, offset, ratio):
    """One to four query rows score through dsa_indexer_score_decode: a
    lone row with no pool mask (every row visible), several rows under
    the pool mask, whose visibility the kernel derives from the offset."""
    args, idx = indexer
    P = 1100
    x = mx.random.normal((1, L, args.hidden_size))
    q_residual = mx.random.normal((1, L, args.q_lora_rank))
    index_k = mx.random.normal((1, P, D)).astype(mx.float16)
    pmask = None if offset is None else _v4._cacheless_pool_mask(P, L, offset, ratio)
    streams = SharedStreams()
    streams.ratio = ratio
    calls = _count_decode_calls(monkeypatch)
    monkeypatch.setitem(_v4._dsa_state, "indexer", True)
    got = np.array(idx(x, q_residual, _Rope(), index_k, pmask, offset, streams))
    assert calls == [(L, 0 if offset is None else offset, ratio, mx.float16)]
    scores = _reference(idx, x, q_residual, index_k, pmask)
    _check(got, scores)
    if pmask is not None:
        hidden = ~np.array(pmask)
        for t in range(L):
            assert not hidden[t, got[0, t]].any()


def test_decode_scorer_stays_inline_when_off(indexer, monkeypatch):
    args, idx = indexer
    x = mx.random.normal((1, 1, args.hidden_size))
    q_residual = mx.random.normal((1, 1, args.q_lora_rank))
    index_k = mx.random.normal((1, 1100, D))
    calls = _count_decode_calls(monkeypatch)
    monkeypatch.setitem(_v4._dsa_state, "indexer", True)
    monkeypatch.setenv("GMLX_DS41_INDEXER_DECODE", "0")
    got = np.array(idx(x, q_residual, _Rope(), index_k, None, None, SharedStreams()))
    assert calls == []
    _check(got, _reference(idx, x, q_residual, index_k, None))
    # several rows with no offset to derive the pool mask from: inline too
    monkeypatch.delenv("GMLX_DS41_INDEXER_DECODE")
    x3 = mx.random.normal((1, 3, args.hidden_size))
    r3 = mx.random.normal((1, 3, args.q_lora_rank))
    got = np.array(idx(x3, r3, _Rope(), index_k, None, None, SharedStreams()))
    assert calls == []
    _check(got, _reference(idx, x3, r3, index_k, None))


def test_index_keys_are_stored_fp16(indexer):
    args, idx = indexer
    latent = mx.random.normal((1, 5, args.head_dim))
    keys = idx.make_keys(latent, _Rope(), 0)
    assert keys.dtype == mx.float16 and keys.shape == (1, 5, D)


def test_inline_path_stays_when_the_kernel_is_off(indexer, monkeypatch):
    args, idx = indexer
    x = mx.random.normal((1, 70, args.hidden_size))
    q_residual = mx.random.normal((1, 70, args.q_lora_rank))
    index_k = mx.random.normal((1, 1100, D))
    monkeypatch.setitem(_v4._dsa_state, "indexer", False)
    got = np.array(idx(x, q_residual, _Rope(), index_k, None, 0, SharedStreams()))
    _check(got, _reference(idx, x, q_residual, index_k, None))
