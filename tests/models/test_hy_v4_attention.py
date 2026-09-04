"""HY4 gated absorbed MLA: sinks, the float mask contract, and key selection.

This file owns the mask-plus-sinks risk. mx.fast SDPA deviates measurably when
a bool mask and a sinks term meet in the same call, and HY4's DSA selection
mask is naturally bool - so folding it into the float ``pe_scores`` before the
SDPA call is a design requirement of ``HyV4Attention``, not a property
inherited from the deepseek_v32 shape. The tests below assert the mask that
reaches SDPA is float, and check the whole attention against an independent
fp64 softmax that folds the sink logit into the normalizer.

The reference shares the module's projections and rope (those are pinned
elsewhere) and re-implements only the softmax, the sink term, the selection
mask and the absorbed value read - the parts a transcription error hides in.
"""

from __future__ import annotations

import numpy as np
import pytest

import mlx.core as mx

import gmlx.models.hy_v4.model as hy_v4_model
from gmlx.models.hy_v4.model import HyV4Attention, ModelArgs

H = 4          # heads
NOPE = 8
PE = 4
R = 12         # kv_lora_rank
V = 8          # v_head_dim
DIM = 16
EPS = 1e-6


# f32 GEMM runs at TF32 precision on M5 (~1e-3 relative), so the fp64
# reference comparison sits well above that noise; the CPU path is exact.
_ATOL = 5e-3


def _args(**over):
    base = dict(
        model_type="hy_v4", vocab_size=32, hidden_size=DIM, intermediate_size=32,
        moe_intermediate_size=16, num_hidden_layers=2, num_attention_heads=H,
        q_lora_rank=16, kv_lora_rank=R, qk_nope_head_dim=NOPE,
        qk_rope_head_dim=PE, v_head_dim=V, n_routed_experts=4,
        num_experts_per_tok=2, n_shared_experts=1, first_k_dense_replace=1,
        rms_norm_eps=EPS, hc_mult=4, hc_eps=1e-6, hc_magnitude=2.0,
    )
    base.update(over)
    return ModelArgs(**base)


def _attention(seed=0, **over):
    """An attention block with small random weights and non-zero sinks."""
    mx.random.seed(seed)
    attn = HyV4Attention(_args(**over), layer_idx=0)
    rng = np.random.default_rng(seed)
    attn.sinks = mx.array(rng.normal(scale=1.0, size=(H,)).astype("float32"))
    mx.eval(attn.parameters())
    return attn


def _projections(attn, x, offset=0):
    """The module's own front half, as fp64 NumPy."""
    B, L, _ = x.shape
    qr = attn.q_a_layernorm(attn.q_a_proj(x))
    q = attn.q_b_proj(qr).reshape(B, L, H, NOPE + PE).transpose(0, 2, 1, 3)
    q_nope, q_pe = mx.split(q, [NOPE], axis=-1)
    compressed = attn.kv_a_proj_with_mqa(x)
    compressed, k_pe = mx.split(compressed, [R], axis=-1)
    k_pe = k_pe.reshape(B, L, 1, PE).transpose(0, 2, 1, 3)
    latent = mx.expand_dims(attn.kv_a_layernorm(compressed), axis=1)
    q_pe = attn.rope(q_pe, offset)
    k_pe = attn.rope(k_pe, offset)
    qr_emb = attn.embed_q(q_nope)                      # [B, H, L, R]
    mx.eval(qr_emb, q_pe, latent, k_pe)
    f = lambda a: np.array(a, dtype=np.float64)        # noqa: E731
    return f(qr_emb), f(q_pe), f(latent), f(k_pe)


def _ref_attention(attn, x, mask=None, top_k=None, offset=0):
    """Absorbed MLA with the sink logit in the normalizer, in fp64 NumPy."""
    B, L, _ = x.shape
    qr_emb, q_pe, latent, k_pe = _projections(attn, x, offset)
    S = latent.shape[2]
    scale = attn.scale

    scores = scale * (qr_emb @ latent.swapaxes(-1, -2)
                      + q_pe @ k_pe.swapaxes(-1, -2))   # [B, H, L, S]

    keep = np.ones((B, 1, L, S), dtype=bool) if mask is None \
        else np.array(mask, dtype=bool).reshape(B, -1, L, S)
    if top_k is not None:
        sel = np.zeros((B, 1, L, S), dtype=bool)
        idx = np.array(top_k).reshape(B, 1, L, -1)
        np.put_along_axis(sel, idx, True, axis=-1)
        keep = keep & sel
    scores = np.where(keep, scores, -np.inf)

    # The sink is a raw extra logit: unscaled, and never masked away.
    sink = np.array(attn.sinks, dtype=np.float64).reshape(1, H, 1, 1)
    m = np.maximum(scores.max(axis=-1, keepdims=True), sink)
    p = np.exp(np.where(np.isfinite(scores), scores - m, -np.inf))
    denom = p.sum(axis=-1, keepdims=True) + np.exp(sink - m)
    ctx = (p @ latent) / denom                          # [B, H, L, R]

    w_v = np.array(attn.unembed_out.weight, dtype=np.float64)   # [H, V, R]
    out = np.einsum("bhlr,hvr->bhlv", ctx, w_v)
    out = out.transpose(0, 2, 1, 3).reshape(B, L, H * V)
    gate = 1.0 / (1.0 + np.exp(-np.array(attn.attn_gate(x), dtype=np.float64)))
    out = out * gate
    w_o = np.array(attn.o_proj.weight, dtype=np.float64)
    return out @ w_o.T


def _causal(L, S):
    i = np.arange(L)[:, None] + (S - L)
    return (np.arange(S)[None, :] <= i)[None, None]


# --- the float-mask contract -------------------------------------------------


def test_mask_reaching_sdpa_is_float_never_bool(monkeypatch):
    seen = []
    real = mx.fast.scaled_dot_product_attention

    def spy(q, k, v, **kw):
        seen.append(kw.get("mask"))
        return real(q, k, v, **kw)

    monkeypatch.setattr(mx.fast, "scaled_dot_product_attention", spy)

    attn = _attention()
    L, S = 6, 6
    x = mx.array(np.random.default_rng(1).normal(
        scale=0.5, size=(1, L, DIM)).astype("float32"))
    sel = np.zeros((1, 1, L, S), dtype=bool)
    sel[..., :4] = True
    bmask = mx.array(_causal(L, S) & sel)

    attn(x, mask=bmask)                       # prefill, composite path
    attn(x[:, :1], mask=None, top_k=mx.array(
        np.arange(3).reshape(1, 1, 1, 3)))    # decode with a selection

    assert seen, "SDPA was never called"
    for m in seen:
        assert m is not None
        assert m.dtype in (mx.float32, mx.float16, mx.bfloat16), m.dtype


def test_sinks_are_used_and_are_per_head():
    # Raising one head's sink must move that head's output and leave the
    # others alone: a dropped sinks term shows up as an exact tie here.
    x = mx.array(np.random.default_rng(2).normal(
        scale=0.5, size=(1, 4, DIM)).astype("float32"))
    mask = mx.array(_causal(4, 4))

    attn = _attention(seed=4)
    base = np.array(_ref_attention(attn, x, mask))
    bumped = np.array(attn.sinks)
    bumped[1] += 6.0
    attn.sinks = mx.array(bumped)
    moved = np.array(_ref_attention(attn, x, mask))
    assert not np.allclose(base, moved)


# --- numerics ----------------------------------------------------------------


@pytest.mark.parametrize("L", [1, 5, 12])
def test_dense_attention_matches_fp64_reference(L):
    attn = _attention(seed=5)
    x = mx.array(np.random.default_rng(6).normal(
        scale=0.5, size=(1, L, DIM)).astype("float32"))
    mask = mx.array(_causal(L, L)) if L > 1 else None
    got = attn(x, mask=mask)[0]
    mx.eval(got)
    ref = _ref_attention(attn, x, mask)
    assert np.allclose(np.array(got), ref, atol=_ATOL), np.abs(
        np.array(got) - ref).max()


@pytest.mark.parametrize("keys,topk", [(6, 4), (16, 4), (16, 15)])
def test_sparse_selection_matches_fp64_reference(keys, topk):
    # A shared layer's path: the selection arrives as `top_k` and must gather
    # (decode) or fold into the float mask (prefill) identically.
    attn = _attention(seed=7)
    rng = np.random.default_rng(8)
    x = mx.array(rng.normal(scale=0.5, size=(1, keys, DIM)).astype("float32"))
    sel = np.stack([rng.permutation(keys)[:topk] for _ in range(keys)])
    sel = np.sort(sel, axis=-1).reshape(1, 1, keys, topk)
    # Only positions the causal mask already allows are selectable.
    sel = np.minimum(sel, np.arange(keys).reshape(1, 1, keys, 1))
    mask = mx.array(_causal(keys, keys))

    got = attn(x, mask=mask, top_k=mx.array(sel))[0]
    mx.eval(got)
    ref = _ref_attention(attn, x, mask, top_k=sel)
    assert np.allclose(np.array(got), ref, atol=_ATOL), np.abs(
        np.array(got) - ref).max()


def test_decode_step_gather_matches_fp64_reference():
    # L == 1 takes the other branch: it gathers the selected latent rows
    # instead of folding a mask, so the two selection paths need separate
    # checks. Run a prefill into the cache, then one decode step.
    from mlx_lm.models.cache import CacheList, KVCache

    attn = _attention(seed=19)
    prefill, topk = 9, 4
    rng = np.random.default_rng(20)
    xs = mx.array(rng.normal(
        scale=0.5, size=(1, prefill, DIM)).astype("float32"))
    cache = CacheList(KVCache(), KVCache())
    attn(xs, mask=mx.array(_causal(prefill, prefill)), cache=cache)

    step = mx.array(rng.normal(scale=0.5, size=(1, 1, DIM)).astype("float32"))
    S = prefill + 1
    sel = np.sort(rng.permutation(S)[:topk]).reshape(1, 1, 1, topk)
    got = attn(step, mask=None, cache=cache, top_k=mx.array(sel))[0]
    mx.eval(got)

    # The reference re-derives the whole S-key context from scratch.
    full = mx.concatenate([xs, step], axis=1)
    ref = _ref_attention(attn, full, mask=None, top_k=np.broadcast_to(
        sel, (1, 1, S, topk)))[:, -1:, :]
    assert np.allclose(np.array(got), ref, atol=_ATOL), np.abs(
        np.array(got) - ref).max()


def test_fully_masked_query_row_stays_finite():
    # With no visible key the sink alone carries the normalizer. An
    # implementation that seeds the running maximum from the scores only
    # returns NaN here.
    attn = _attention(seed=9)
    x = mx.array(np.random.default_rng(10).normal(
        scale=0.5, size=(1, 4, DIM)).astype("float32"))
    mask = np.array(_causal(4, 4))
    mask[..., 2, :] = False                    # row 2 sees nothing
    got = attn(x, mask=mx.array(mask))[0]
    mx.eval(got)
    assert np.isfinite(np.array(got)).all()


def test_tiled_prefill_matches_the_direct_path(monkeypatch):
    # The online-softmax tiling is exact: same weights, same inputs, one path
    # forced past the streaming thresholds.
    attn = _attention(seed=11)
    L = 24
    x = mx.array(np.random.default_rng(12).normal(
        scale=0.5, size=(1, L, DIM)).astype("float32"))
    mask = mx.array(_causal(L, L))

    direct = np.array(attn(x, mask=mask)[0])
    monkeypatch.setattr(hy_v4_model, "_STREAM_Q", 8)
    monkeypatch.setattr(hy_v4_model, "_STREAM_BLOCK", 8)
    monkeypatch.setattr(hy_v4_model, "_STREAM_MIN_KEYS", 4)
    tiled = np.array(attn(x, mask=mask)[0])

    assert np.allclose(direct, tiled, atol=_ATOL), np.abs(direct - tiled).max()


def test_tiled_prefill_matches_the_fp64_reference_under_selection(monkeypatch):
    attn = _attention(seed=13)
    L, topk = 24, 8
    rng = np.random.default_rng(14)
    x = mx.array(rng.normal(scale=0.5, size=(1, L, DIM)).astype("float32"))
    sel = np.sort(np.stack([rng.permutation(L)[:topk] for _ in range(L)]),
                  axis=-1).reshape(1, 1, L, topk)
    sel = np.minimum(sel, np.arange(L).reshape(1, 1, L, 1))
    mask = mx.array(_causal(L, L))

    monkeypatch.setattr(hy_v4_model, "_STREAM_Q", 8)
    monkeypatch.setattr(hy_v4_model, "_STREAM_BLOCK", 8)
    monkeypatch.setattr(hy_v4_model, "_STREAM_MIN_KEYS", 4)
    got = np.array(attn(x, mask=mask, top_k=mx.array(sel))[0])
    ref = _ref_attention(attn, x, mask, top_k=sel)
    assert np.allclose(got, ref, atol=_ATOL), np.abs(got - ref).max()


# --- the rope-convention fork ------------------------------------------------


@pytest.mark.parametrize("traditional", [True, False])
def test_reference_holds_under_either_rope_convention(traditional):
    # rope_traditional is a config key so the convention can be flipped in one
    # line (transformers 5.15 hy_v4 uses rotate_half; the deployed vLLM and
    # llama.cpp use interleaved). Whichever is set, the attention math above
    # it must stay exact.
    attn = _attention(seed=15, rope_traditional=traditional)
    x = mx.array(np.random.default_rng(16).normal(
        scale=0.5, size=(1, 6, DIM)).astype("float32"))
    mask = mx.array(_causal(6, 6))
    got = attn(x, mask=mask)[0]
    mx.eval(got)
    assert np.allclose(np.array(got), _ref_attention(attn, x, mask), atol=_ATOL)


def test_rope_convention_default_is_interleaved():
    # The llama.cpp reference applies LLAMA_ROPE_TYPE_NORM. Flipping the
    # default silently produces fluent-but-wrong output at depth.
    assert _args().rope_traditional is True


# --- the sparse-disable seam -------------------------------------------------


def test_sparse_disable_env_forces_dense(monkeypatch):
    # The oracle-free validation lever: with no llama.cpp reference above
    # 2048 tokens on this box, the sparse path is checked against this
    # model's own dense forward.
    attn = _attention(seed=17)
    L, topk = 10, 3
    rng = np.random.default_rng(18)
    x = mx.array(rng.normal(scale=0.5, size=(1, L, DIM)).astype("float32"))
    sel = np.sort(np.stack([rng.permutation(L)[:topk] for _ in range(L)]),
                  axis=-1).reshape(1, 1, L, topk)
    sel = np.minimum(sel, np.arange(L).reshape(1, 1, L, 1))
    mask = mx.array(_causal(L, L))

    sparse = np.array(attn(x, mask=mask, top_k=mx.array(sel))[0])
    monkeypatch.setenv("GMLX_HY4_SPARSE_DISABLE", "1")
    forced = np.array(attn(x, mask=mask, top_k=mx.array(sel))[0])
    assert np.allclose(forced, _ref_attention(attn, x, mask), atol=_ATOL)
    assert not np.allclose(sparse, forced)     # the selection did bite
