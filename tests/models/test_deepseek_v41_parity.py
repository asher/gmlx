"""DeepSeek-V4.1-Flash block parity against numpy float64 ports.

The oracle is ``ds41-ref/inference``: ``kernel.hc_split_sinkhorn_kernel``,
``model.Block.forward``, ``model.Compressor.forward``, ``model.Indexer.forward``
and ``kernel.sparse_attn_kernel``, each ported here in float64. The ports carry
no fused kernel and no ``mx.fast`` call, so a fused route that drifts shows up
as a red test. CPU-only.
"""
from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from gmlx.models.deepseek_v4.hyper_connection import hc_expand
from gmlx.models.deepseek_v41.model import (
    Model,
    ModelArgs,
    SharedStreams,
    _hc_collapse,
    _hc_mixes,
    _kv_qat,
)

# Tolerances here allow for f32 GEMM noise on tensor-core hardware, where
# a batched matmul and a per-row reference of the same data disagree
# around 1e-3 relative. The tests run production numerics and do not set
# MLX_ENABLE_TF32=0.

_TABLE_ROWS = 3 * 2 * 7


def _args(**over) -> ModelArgs:
    base = dict(
        vocab_size=64, hidden_size=32, moe_intermediate_size=16,
        num_hidden_layers=6, num_attention_heads=4, head_dim=16,
        q_lora_rank=8, qk_rope_head_dim=4, o_groups=2, o_lora_rank=4,
        n_routed_experts=4, num_experts_per_tok=2, n_shared_experts=1,
        index_n_heads=2, index_head_dim=8, index_topk=4,
        sliding_window=4, hc_mult=4, rms_norm_eps=1e-6,
        compress_ratios=[0, 0, 2, 2, 1, 1],
        kv_source_layers=[2, 4], index_key_layers=[2, 4],
        index_source_layers=[2, 4, 5],
        engram_layer_ids=[1, 3],
        engram_table_rows=[_TABLE_ROWS, _TABLE_ROWS],
        engram_max_ngram_size=4, engram_n_heads=2, engram_head_dim=8,
        engram_pad_id=2,
        engram_multipliers=[3, 5, 7, 11, 13, 17, 19, 23],
        engram_primes=[7] * 12,
        engram_offsets=[0, 7, 14, 21, 28, 35] * 2,
        engram_token_map=[(i * 7) % 11 for i in range(64)],
        rope_scaling={"type": "yarn", "factor": 4.0,
                      "original_max_position_embeddings": 64,
                      "beta_fast": 32, "beta_slow": 1},
    )
    base.update(over)
    return ModelArgs(**base)


@pytest.fixture(scope="module")
def model():
    mx.random.seed(11)
    m = Model(_args())

    def walk(node):
        if isinstance(node, dict):
            return {k: walk(v) for k, v in node.items()}
        if isinstance(node, list):
            return [walk(v) for v in node]
        if node.dtype in (mx.float32, mx.float16, mx.bfloat16):
            return mx.random.normal(node.shape) * 0.2
        return node

    m.update(walk(m.parameters()))
    m.eval()
    mx.eval(m.parameters())
    return m


def _np(x) -> np.ndarray:
    return np.array(x, dtype=np.float64)


# --- reference ports --------------------------------------------------------


def _softmax(x, axis):
    z = x - x.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def _rms_norm(x, weight, eps):
    y = x / np.sqrt((x * x).mean(-1, keepdims=True) + eps)
    return y if weight is None else y * weight


def _ref_sinkhorn(mixes, scale, base, hc, iters, eps):
    """Port of ``hc_split_sinkhorn_kernel_``."""
    pre = 1.0 / (1.0 + np.exp(-(mixes[..., :hc] * scale[0] + base[:hc]))) + eps
    post = 2.0 / (
        1.0 + np.exp(-(mixes[..., hc:2 * hc] * scale[1] + base[hc:2 * hc]))
    )
    comb = mixes[..., 2 * hc:].reshape(*mixes.shape[:-1], hc, hc) * scale[2]
    comb = comb + base[2 * hc:].reshape(hc, hc)
    comb = _softmax(comb, -1) + eps
    comb = comb / (comb.sum(-2, keepdims=True) + eps)
    for _ in range(iters - 1):
        comb = comb / (comb.sum(-1, keepdims=True) + eps)
        comb = comb / (comb.sum(-2, keepdims=True) + eps)
    return pre, post, comb


def _ref_hc_mixes(hc, x):
    """Port of ``Block.hc_mixes``."""
    y = _np(x).reshape(*x.shape[:2], -1)
    rsqrt = 1.0 / np.sqrt((y * y).mean(-1, keepdims=True) + hc.norm_eps)
    mixes = (y @ _np(hc.fn).T) * rsqrt
    return _ref_sinkhorn(mixes, _np(hc.scale), _np(hc.base), hc.hc_mult,
                         hc.sinkhorn_iters, hc.hc_eps)


def _ref_hc_pre(x, pre):
    """Port of ``Block.hc_pre``."""
    return (pre[..., None] * _np(x)).sum(axis=2)


def _ref_hc_post(x, residual, post, comb):
    """Port of ``Block.hc_post``."""
    return post[..., None] * _np(x)[:, :, None] + (
        comb[..., None] * _np(residual)[:, :, :, None, :]
    ).sum(axis=2)


def _ref_sparse_attn(q, kv, sink, allowed, scale):
    """Port of ``sparse_attn_kernel_``: softmax over the allowed positions
    only, with the per-head sink as one extra logit."""
    scores = np.einsum("bhld,bnd->bhln", q, kv) * scale
    scores = np.where(allowed, scores, -np.inf)
    peak = np.maximum(scores.max(-1), sink[None, :, None])
    exp = np.exp(scores - peak[..., None])
    denom = exp.sum(-1) + np.exp(sink[None, :, None] - peak)
    return np.einsum("bhln,bnd->bhld", exp, kv) / denom[..., None]


# --- hyper connections ------------------------------------------------------


def test_hc_mixes_match_the_reference_sinkhorn(model):
    hc = model.model.layers[0].attn_hc
    x = mx.random.normal((2, 5, 4, 32))
    pre, post, comb = (_np(v) for v in _hc_mixes(hc, x))
    r_pre, r_post, r_comb = _ref_hc_mixes(hc, x)
    assert np.allclose(pre, r_pre, atol=1e-3)
    assert np.allclose(post, r_post, atol=1e-3)
    assert np.allclose(comb, r_comb, atol=1e-3)
    # Sinkhorn leaves a doubly near-stochastic matrix, so a dropped
    # iteration or a transposed normalization shows here.
    assert np.allclose(r_comb.sum(-1), 1.0, atol=1e-3)


def test_collapse_and_expand_match_the_reference(model):
    hc = model.model.layers[0].attn_hc
    x = mx.random.normal((2, 5, 4, 32))
    y = mx.random.normal((2, 5, 32))
    pre, post, comb = _hc_mixes(hc, x)
    r_pre, r_post, r_comb = _ref_hc_mixes(hc, x)
    assert np.allclose(_np(_hc_collapse(x, pre)), _ref_hc_pre(x, r_pre),
                       atol=2e-3)
    assert np.allclose(_np(hc_expand(y, x, post, comb)),
                       _ref_hc_post(y, x, r_post, r_comb), atol=2e-3)


@pytest.mark.skipif(mx.default_device() != mx.gpu,
                    reason="the MoE router top-k kernel is Metal-only")
def test_the_block_collapses_with_the_previous_sublayers_pre(model):
    """V4.1 lags by one sublayer. Attention collapses with the pre the
    caller hands in, and the FFN with the pre that attention produced.
    """
    layer = model.model.layers[0]          # ratio 0, owns no pool
    B, L = 1, 6
    h = mx.random.normal((B, L, 4, 32))
    pre_mix = mx.zeros((B, L, 4))
    pre_mix[:, :, 0] = 1.0
    ids = mx.array([[3, 9, 1, 4, 5, 2]])
    mask = mx.tril(mx.ones((L, L), dtype=mx.bool_))

    got, got_pre = layer(h, pre_mix, mask, (None, None, None), 0,
                         SharedStreams(), ids)

    a_pre, a_post, a_comb = _ref_hc_mixes(layer.attn_hc, h)
    x = layer.attn_norm(mx.array(_ref_hc_pre(h, _np(pre_mix)), dtype=h.dtype))
    x = layer.attn(x, mask, (None, None, None), 0, SharedStreams())
    mid = _ref_hc_post(x, h, a_post, a_comb)

    mid_mx = mx.array(mid, dtype=h.dtype)
    f_pre, f_post, f_comb = _ref_hc_mixes(layer.ffn_hc, mid_mx)
    y = layer.ffn_norm(mx.array(_ref_hc_pre(mid_mx, a_pre), dtype=h.dtype))
    y = layer.ffn(y, ids, None)
    want = _ref_hc_post(y, mid_mx, f_post, f_comb)

    assert np.allclose(_np(got), want, atol=1e-2)
    assert np.allclose(_np(got_pre), f_pre, atol=2e-3)


# --- compressor -------------------------------------------------------------


def test_compressor_pools_a_ratio_two_group_like_the_reference(model):
    comp = model.model.layers[2].attn.compressor
    assert comp.compress_ratio == 2
    L = 7                                   # odd: the tail group waits
    x = mx.random.normal((1, L, 32))
    got, base = comp(x, None, 0)
    assert base == 0
    assert got.shape[1] == L // 2

    kv = _np(x) @ _np(comp.wkv.weight).T
    gate = _np(x) @ _np(comp.wgate.weight).T
    usable = (L // 2) * 2
    kv = kv[:, :usable].reshape(1, L // 2, 2, -1)
    gate = gate[:, :usable].reshape(1, L // 2, 2, -1)
    pooled = (kv * _softmax(gate, 2)).sum(axis=2)
    want = _rms_norm(pooled, _np(comp.norm.weight), comp.norm.eps)
    assert np.allclose(_np(got), want, atol=3e-3)


def test_ratio_one_compressor_has_no_gate(model):
    comp = model.model.layers[4].attn.compressor
    assert comp.compress_ratio == 1
    assert not hasattr(comp, "wgate")
    x = mx.random.normal((1, 5, 32))
    got, _ = comp(x, None, 0)
    want = _rms_norm(_np(x) @ _np(comp.wkv.weight).T,
                     _np(comp.norm.weight), comp.norm.eps)
    assert np.allclose(_np(got), want, atol=2e-3)


# --- indexer ----------------------------------------------------------------


def test_indexer_picks_the_same_positions_as_float64(model):
    attn = model.model.layers[2].attn
    idx = attn.indexer
    B, L, P = 1, 6, 9
    x = mx.random.normal((B, L, 32))
    q_residual = mx.random.normal((B, L, 8))
    index_k = mx.random.normal((B, P, idx.head_dim))

    got = _np(idx(x, q_residual, attn.rope, index_k, None, 0, SharedStreams()))

    q = idx.wq_b(q_residual).reshape(B, L, idx.n_heads, idx.head_dim)
    q = attn.rope(q.transpose(0, 2, 1, 3), 0)
    from gmlx.models.deepseek_v41.model import _indexer_qat
    q = _np(_indexer_qat(q))
    scores = np.maximum(np.einsum("bhld,bpd->bhlp", q, _np(index_k)), 0.0)
    scores = scores * idx.scale
    weights = _np(x) @ _np(idx.weights_proj.weight).T * idx.n_heads ** -0.5
    scores = (scores * weights.transpose(0, 2, 1)[..., None]).sum(axis=1)
    want = np.argsort(-scores, axis=-1)[..., :idx.index_topk]

    for b in range(B):
        for t in range(L):
            assert set(got[b, t].tolist()) == set(want[b, t].tolist())


# --- attention --------------------------------------------------------------


def _attn_qkv(attn, x, offset=0):
    """The prologue of ``Attention.forward``: q and the window KV."""
    B, L, _ = x.shape
    q = attn.wq_b(attn.q_norm(attn.wq_a(x)))
    q = q.reshape(B, L, attn.n_heads, attn.head_dim).transpose(0, 2, 1, 3)
    q = attn.rope(q, offset)
    kv = attn.kv_norm(attn.wkv(x)).reshape(B, 1, L, attn.head_dim)
    kv = _kv_qat(attn.rope(kv, offset))
    return q, kv


def test_window_attention_matches_float64_with_sinks(model):
    attn = model.model.layers[0].attn     # ratio 0: window only
    assert attn.compress_ratio == 0
    B, L = 1, 7
    x = mx.random.normal((B, L, 32))
    win = model.args.sliding_window
    pos = np.arange(L)
    allowed = (pos[:, None] >= pos[None, :]) & (pos[:, None] - pos[None, :] < win)
    mask = mx.array(allowed)

    got = _np(attn(x, mask, (None, None, None), 0, SharedStreams()))

    q, kv = _attn_qkv(attn, x)
    out = _ref_sparse_attn(_np(q), _np(kv)[:, 0], _np(attn.attn_sink),
                           allowed[None, None], attn.scale)
    want = _np(attn._project(mx.array(out, dtype=x.dtype), 0, B, L))
    assert np.allclose(got, want, atol=2e-3)


def test_window_and_latent_attention_match_float64(model):
    """A kv-source layer attends over its window and the compressed pool
    in one softmax, so the reference scores both against the same sink.
    """
    attn = model.model.layers[2].attn
    B, L = 1, 12
    x = mx.random.normal((B, L, 32))
    win = model.args.sliding_window
    pos = np.arange(L)
    allowed = (pos[:, None] >= pos[None, :]) & (pos[:, None] - pos[None, :] < win)
    mask = mx.array(allowed)

    streams = SharedStreams()
    got = _np(attn(x, mask, (None, None, None), 0, streams))

    pooled = _np(streams.pooled)
    topk = np.array(streams.topk)
    ratio = attn.compress_ratio
    # A latent stands for its group's first token, so group j is readable
    # once the query has passed the group's last token.
    reach = np.arange(pooled.shape[1])[None, :] < ((pos[:, None] + 1) // ratio)
    keep = np.zeros((L, pooled.shape[1]), dtype=bool)
    for t in range(L):
        for j in topk[0, t]:
            keep[t, j] = reach[t, j]

    q, kv = _attn_qkv(attn, x)
    both = np.concatenate([_np(kv)[:, 0], pooled], axis=1)
    allow = np.concatenate([allowed, keep], axis=-1)[None, None]
    out = _ref_sparse_attn(_np(q), both, _np(attn.attn_sink), allow, attn.scale)
    want = _np(attn._project(mx.array(out, dtype=x.dtype), 0, B, L))
    assert np.allclose(got, want, atol=2e-3)
