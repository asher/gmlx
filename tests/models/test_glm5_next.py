#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Asher Feldman
"""GLM-5.3-Flash vendored model: forward smoke, KDA layer parity vs a naive
reference, MLA absorbed/naive equivalence, pooled-indexer selection and mask
invariants, MoE selection, and the synth -> ModelArgs round-trip.

All tests run on tiny dims with random weights - no GGUF, no download. The
numeric references are self-contained (naive python loops implementing the
llama.cpp PR-27754 glm5next semantics), so a behavior drift in the vendored
module fails here before any conversion-level parity run.
"""

from __future__ import annotations

import numpy as np
import pytest

import mlx.core as mx

import gmlx.models.glm5_next.model as glm5_model
from gmlx.models.deepseek_v4.cache import PoolingCache
from gmlx.models.glm5_next.model import (
    Glm5NextIndexer,
    Glm5NextMLAAttention,
    Model,
    ModelArgs,
    _limited_swiglu,
)


def _tiny_args(**over) -> ModelArgs:
    kw = dict(
        model_type="glm5_next",
        vocab_size=64,
        hidden_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=1,
        intermediate_size=96,
        rms_norm_eps=1e-5,
        layer_types=["linear_attention", "full_attention",
                     "linear_attention", "full_attention"],
        kda_head_dim=16,
        ssm_conv_kernel=4,
        kda_gate_lower_bound=-5.0,
        q_lora_rank=32,
        kv_lora_rank=32,
        qk_nope_head_dim=16,
        qk_rope_head_dim=0,
        v_head_dim=16,
        index_n_heads=2,
        index_head_dim=8,
        index_topk=8,
        index_kpool=4,
        index_knorm_eps=1e-6,
        n_routed_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=24,
        n_shared_experts=1,
        first_k_dense_replace=1,
        routed_scaling_factor=2.5,
        norm_topk_prob=True,
        scoring_func="sigmoid",
        swiglu_limit=10.0,
        hc_mult=4,
        hc_sinkhorn_iters=20,
        hc_eps=1e-6,
        max_position_embeddings=4096,
        tie_word_embeddings=False,
    )
    kw.update(over)
    return ModelArgs(**kw)


def _random_model(args: ModelArgs, seed: int = 0) -> Model:
    mx.random.seed(seed)
    model = Model(args)
    from mlx.utils import tree_flatten, tree_unflatten
    params = [(k, mx.random.normal(v.shape) * 0.05)
              for k, v in tree_flatten(model.parameters())]
    model.update(tree_unflatten(params))
    # a_folded must stay negative (-exp(A_log)); rebuild it explicitly.
    for layer in model.layers:
        if layer.is_linear:
            layer.self_attn.a_folded = -mx.random.uniform(
                low=1.0, high=4.0, shape=(args.num_attention_heads,))
    mx.eval(model.parameters())
    return model


# n_select = index_topk + kpool - 1 = 11 for the tiny args: prompts of
# length <= 11 are exactly dense; longer ones engage sparse selection.
_N_SELECT = 11


def test_forward_smoke_and_cache_shapes():
    args = _tiny_args()
    model = _random_model(args)
    out = model(mx.array([[1, 2, 3, 4, 5]]))
    mx.eval(out)
    assert out.shape == (1, 5, args.vocab_size)
    assert bool(mx.isfinite(out).all())

    caches = model.make_cache()
    out = model(mx.array([[1, 2, 3, 4, 5]]), cache=caches)
    mx.eval(out)
    d_inner = args.num_attention_heads * args.kda_head_dim
    for layer, c in zip(model.layers, caches):
        if layer.is_linear:
            # slots 0..2 = per-branch conv tails, 3 = fp32 recurrent state.
            for slot in range(3):
                assert c[slot].shape == (1, args.ssm_conv_kernel - 1, d_inner)
            assert c[3].shape == (1, args.num_attention_heads,
                                  args.kda_head_dim, args.kda_head_dim)
            assert c[3].dtype == mx.float32
        else:
            assert c[0].offset == 5
            pool = c[1]
            assert isinstance(pool, PoolingCache)
            assert pool.lookback is False
            assert pool.quantizable is False
            assert pool._plen == 1 and pool.remainder == 1  # 5 = 4 + 1


def test_pool_bookkeeping_and_make_mask_convention():
    # make_mask pins the +1 pool-visibility form: query at absolute position
    # q sees pool p iff p < (q + 1) // ratio - the query's own just-completed
    # pool included at q % 4 == 3 (llama.cpp PR-27754 pool_visible).
    pool = PoolingCache(4, lookback=False)
    k = mx.random.normal((1, 10, 8))
    g = mx.random.normal((1, 10, 8))
    rk, rg, _ = pool.accumulate_windows(k, g, 0)
    assert rk.shape[1] == 8 and pool.remainder == 2  # disjoint: exact windows
    pool.update_and_fetch(mx.random.normal((1, 2, 8)))
    assert pool._plen == 2

    m = np.array(pool.make_mask(L=8, offset=2))
    for j in range(8):
        q = 2 + j
        for p in range(2):
            assert m[j, p] == (p < (q + 1) // 4), (j, p)


def test_prefill_matches_stepwise_decode():
    # T tokens in one forward == the same T tokens fed one at a time, kept
    # below the sparse threshold so both runs are exactly dense (sparse
    # selection tie-breaks are covered by the dedicated tests below).
    args = _tiny_args()
    model = _random_model(args)
    toks = [3, 9, 27, 40, 11, 5, 33, 60, 2, 17]

    full = model(mx.array([toks]), cache=model.make_cache())
    step_cache = model.make_cache()
    steps = [model(mx.array([[t]]), cache=step_cache) for t in toks]
    mx.eval(full, steps)

    last_full = np.array(full[0, -1], dtype=np.float32)
    last_step = np.array(steps[-1][0, 0], dtype=np.float32)
    np.testing.assert_allclose(last_step, last_full, rtol=2e-2, atol=2e-2)
    for t, s in enumerate(steps):
        a = np.array(full[0, t], dtype=np.float32).argmax()
        b = np.array(s[0, 0], dtype=np.float32).argmax()
        assert a == b, f"argmax diverged at position {t}"


def test_batched_matches_single():
    args = _tiny_args()
    model = _random_model(args)
    toks = [3, 9, 27, 40]
    single = model(mx.array([toks]), cache=model.make_cache())
    batched = model(mx.array([toks, toks]), cache=model.make_cache())
    mx.eval(single, batched)
    for b in range(2):
        np.testing.assert_allclose(
            np.array(batched[b, -1], dtype=np.float32),
            np.array(single[0, -1], dtype=np.float32),
            rtol=2e-2, atol=2e-2)


def _np_rms(x, eps):
    x = np.asarray(x, dtype=np.float64)
    return x / np.sqrt((x ** 2).mean(-1, keepdims=True) + eps)


def test_kda_layer_matches_naive_reference():
    # End-to-end KDA layer vs a step-loop reference implementing the
    # llama.cpp semantics: causal depthwise conv then silu, l2-normalized
    # q/k with the scale folded (q carries 1/sqrt(d) twice), decay
    # exp(lb * sigmoid(exp(A_log) * (f_b(f_a(x)) + dt_bias))) per key
    # channel, delta-rule state update, then rmsnorm * sigmoid(low-rank
    # gate) and o_proj.
    args = _tiny_args()
    model = _random_model(args, seed=7)
    attn = model.layers[0].self_attn
    H, D = args.num_attention_heads, args.kda_head_dim
    T = 6
    mx.random.seed(11)
    x = mx.random.normal((1, T, args.hidden_size)) * 0.5
    got = np.array(attn(x), dtype=np.float32)[0]

    xn = np.array(x, dtype=np.float64)[0]

    def lin(w):
        return np.array(w, dtype=np.float64)

    def conv_silu(y, conv_w):
        w = np.array(conv_w, dtype=np.float64)[:, :, 0]  # (C, K)
        K = w.shape[1]
        pad = np.concatenate([np.zeros((K - 1, y.shape[1])), y], axis=0)
        out = np.stack(
            [sum(w[:, j] * pad[t + j] for j in range(K)) for t in range(len(y))])
        return out / (1.0 + np.exp(-out))

    q = conv_silu(xn @ lin(attn.q_proj.weight).T, attn.q_conv.conv.weight)
    k = conv_silu(xn @ lin(attn.k_proj.weight).T, attn.k_conv.conv.weight)
    v = conv_silu(xn @ lin(attn.v_proj.weight).T, attn.v_conv.conv.weight)
    q = q.reshape(T, H, D)
    k = k.reshape(T, H, D)
    v = v.reshape(T, H, D)
    scale = D ** -0.5
    q = (scale ** 2) * _np_rms(q, 1e-6)
    k = scale * _np_rms(k, 1e-6)

    a_raw = (xn @ lin(attn.f_a_proj.weight).T @ lin(attn.f_b_proj.weight).T
             ).reshape(T, H, D)
    dt = np.array(attn.dt_bias, dtype=np.float64).reshape(H, D)
    a_exp = -np.array(attn.a_folded, dtype=np.float64)  # exp(A_log)
    g = np.exp(-5.0 / (1.0 + np.exp(-a_exp[:, None] * (a_raw + dt))))
    beta = 1.0 / (1.0 + np.exp(-(xn @ lin(attn.b_proj.weight).T)))

    y = np.zeros((T, H, D))
    S = np.zeros((H, D, D))  # [head, key channel, value channel]
    for t in range(T):
        for h in range(H):
            S[h] *= g[t, h][:, None]
            kv = (k[t, h][:, None] * S[h]).sum(0)
            delta = (v[t, h] - kv) * beta[t, h]
            S[h] += k[t, h][:, None] * delta[None, :]
            y[t, h] = (q[t, h][:, None] * S[h]).sum(0)

    onw = np.array(attn.o_norm.weight, dtype=np.float64)
    gate = (xn @ lin(attn.g_a_proj.weight).T @ lin(attn.g_b_proj.weight).T
            ).reshape(T, H, D)
    out = _np_rms(y, args.rms_norm_eps) * onw * (1.0 / (1.0 + np.exp(-gate)))
    ref = out.reshape(T, -1) @ lin(attn.o_proj.weight).T

    np.testing.assert_allclose(got, ref, rtol=5e-3, atol=5e-3)


def test_mla_absorbed_matches_naive_prefill(monkeypatch):
    # The absorbed form (embed_q -> MQA over the latent -> unembed_out) and
    # the naive expansion are the same function up to rounding order.
    args = _tiny_args()
    model = _random_model(args, seed=3)
    attn = model.layers[1].self_attn
    mx.random.seed(4)
    x = mx.random.normal((1, 7, args.hidden_size)) * 0.5
    mask = mx.tril(mx.ones((7, 7), dtype=mx.bool_))

    naive = np.array(attn(x, mask=mask), dtype=np.float32)
    monkeypatch.setattr(glm5_model, "_ABSORBED_PREFILL", True)
    absorbed = np.array(attn(x, mask=mask), dtype=np.float32)
    np.testing.assert_allclose(absorbed, naive, rtol=2e-3, atol=2e-3)


def test_indexer_dense_bypass_thresholds():
    # Selection must bind only above n_select total keys (and once enough
    # pools exist); below, it returns None and the dense path runs.
    args = _tiny_args()
    mx.random.seed(9)
    idx = Glm5NextIndexer(args)
    from mlx.utils import tree_flatten, tree_unflatten
    params = [(k, mx.random.normal(v.shape) * 0.1)
              for k, v in tree_flatten(idx.parameters())]
    idx.update(tree_unflatten(params))

    x = mx.random.normal((1, 20, args.hidden_size))
    qr = mx.random.normal((1, 20, args.q_lora_rank))
    pool = PoolingCache(4, lookback=False)
    assert idx(x[:, :11], qr[:, :11], pool, 0, 11) is None  # at threshold
    pool2 = PoolingCache(4, lookback=False)
    sel = idx(x, qr, pool2, 0, 20)
    assert sel is not None and sel.shape == (1, 20, idx.select_k)


def test_indexer_selection_matches_naive_reference():
    # Pooled scoring reference: per-channel softmax over slots with the ape
    # table, relu between the per-head dot and the (sign-free, 1/sqrt(h*d)-
    # scaled) head weighting, validity p < (q+1)//4, top-k over POOLS.
    args = _tiny_args()
    mx.random.seed(13)
    idx = Glm5NextIndexer(args)
    from mlx.utils import tree_flatten, tree_unflatten
    params = [(k, mx.random.normal(v.shape) * 0.2)
              for k, v in tree_flatten(idx.parameters())]
    idx.update(tree_unflatten(params))
    mx.eval(idx.parameters())

    T = 20
    x = mx.random.normal((1, T, args.hidden_size)) * 0.5
    qr = mx.random.normal((1, T, args.q_lora_rank)) * 0.5
    pool = PoolingCache(4, lookback=False)
    sel = np.array(idx(x, qr, pool, 0, T))[0]  # [T, select_k]

    xn = np.array(x, dtype=np.float64)[0]
    qn = np.array(qr, dtype=np.float64)[0]
    wk = np.array(idx.wk.weight, dtype=np.float64)
    kw = np.array(idx.k_norm.weight, dtype=np.float64)
    kb = np.array(idx.k_norm.bias, dtype=np.float64)
    keys = xn @ wk.T
    keys = ((keys - keys.mean(-1, keepdims=True))
            / np.sqrt(keys.var(-1, keepdims=True) + 1e-6)) * kw + kb
    gates = xn @ np.array(idx.compressor.wgate.weight, dtype=np.float64).T
    ape = np.array(idx.compressor.ape, dtype=np.float64)

    n_pools = T // 4
    pooled = np.zeros((n_pools, args.index_head_dim))
    for p in range(n_pools):
        gw = gates[4 * p:4 * p + 4] + ape          # [slot, channel]
        pw = np.exp(gw - gw.max(0, keepdims=True))
        pw = pw / pw.sum(0, keepdims=True)          # softmax over slots
        pooled[p] = (keys[4 * p:4 * p + 4] * pw).sum(0)

    Hh, Dd = args.index_n_heads, args.index_head_dim
    q = (qn @ np.array(idx.wq_b.weight, dtype=np.float64).T).reshape(T, Hh, Dd)
    w = (xn @ np.array(idx.weights_proj.weight, dtype=np.float64).T
         ) * (Hh * Dd) ** -0.5
    scores = np.maximum(np.einsum("thd,pd->thp", q, pooled), 0.0)
    scores = (scores * w[..., None]).sum(1)          # [T, n_pools]
    for j in range(T):
        scores[j, (j + 1) // 4:] = -np.inf           # validity mask

    for j in range(T):
        order = np.argsort(-scores[j])
        n_valid = (j + 1) // 4
        if n_valid < idx.select_k:
            continue  # spill rows neutralize downstream; order unconstrained
        # Tie robustness: only rows with a clear margin at the cut count.
        cut, nxt = scores[j, order[idx.select_k - 1]], scores[j, order[idx.select_k]]
        if not np.isfinite(nxt) or cut - nxt > 1e-6:
            assert set(sel[j]) == set(order[:idx.select_k]), f"row {j}"
        got_valid = [p for p in sel[j] if p < n_valid]
        assert len(set(got_valid)) == len(got_valid)


def test_sparse_mask_full_selection_equals_causal():
    # With every pool selected the sparse mask must reduce to plain causal:
    # pools + tail partition [0..q] for every query (coverage invariant),
    # swept over all four q % 4 residues via L=8 at offset 4.
    args = _tiny_args()
    attn = Glm5NextMLAAttention(args)
    L, offset = 8, 4
    S = offset + L
    n_pools = S // 4
    sel = mx.broadcast_to(mx.arange(n_pools)[None, None], (1, L, n_pools))
    m = np.array(attn._sparse_mask(sel, offset, L, S))[0]
    for j in range(L):
        q = offset + j
        causal = np.arange(S) <= q
        np.testing.assert_array_equal(m[j], causal, err_msg=f"row {j} (q={q})")


def test_sparse_mask_tail_and_selection_semantics():
    # A single selected pool: visible keys are exactly that pool's members
    # (causally visible ones) plus the incomplete trailing pool.
    args = _tiny_args()
    attn = Glm5NextMLAAttention(args)
    L, offset, S = 4, 8, 12
    sel = mx.ones((1, L, 1), dtype=mx.int32)  # pool 1 = keys 4..7
    m = np.array(attn._sparse_mask(sel, offset, L, S))[0]
    for j in range(L):
        q = offset + j
        tail_start = (q + 1) // 4 * 4
        expect = np.zeros(S, dtype=bool)
        expect[4:8] = True                      # selected pool members
        expect[tail_start:q + 1] = True          # trailing tail
        expect[q + 1:] = False                   # causal
        np.testing.assert_array_equal(m[j], expect, err_msg=f"q={q}")


def test_sparse_decode_gather_matches_masked_path():
    # The decode gather (selected latents + tail rows, no mask) and the
    # masked-SDPA application must agree at every q % 4 residue.
    args = _tiny_args()
    a = _random_model(args, seed=21)
    b = _random_model(args, seed=21)
    for layer in b.model.layers:
        if not layer.is_linear:
            layer.self_attn._decode_gather = False

    toks = [3, 9, 27, 40, 11, 5, 33, 60, 2, 17, 44, 8, 19, 52]  # T=14 > 11
    ca, cb = a.make_cache(), b.make_cache()
    oa = a(mx.array([toks]), cache=ca)
    ob = b(mx.array([toks]), cache=cb)
    mx.eval(oa, ob)
    np.testing.assert_allclose(
        np.array(oa[0, -1], dtype=np.float32),
        np.array(ob[0, -1], dtype=np.float32), rtol=2e-3, atol=2e-3)
    for step, t in enumerate([7, 21, 42, 13, 30, 6]):  # offsets 14..19
        oa = a(mx.array([[t]]), cache=ca)
        ob = b(mx.array([[t]]), cache=cb)
        mx.eval(oa, ob)
        np.testing.assert_allclose(
            np.array(oa[0, 0], dtype=np.float32),
            np.array(ob[0, 0], dtype=np.float32),
            rtol=2e-3, atol=2e-3, err_msg=f"decode step {step}")


def test_streamed_absorbed_attention_matches_sdpa(monkeypatch):
    # The online-softmax key-block accumulation must reproduce plain
    # masked SDPA, including rows whose first blocks are fully masked.
    import mlx.core as _mx

    monkeypatch.setattr(glm5_model, "_STREAM_BLOCK", 8)
    mx.random.seed(17)
    B, H, L, S, D = 1, 4, 6, 29, 16
    q = mx.random.normal((B, H, L, D)) * 0.5
    latent = mx.random.normal((B, 1, S, D)) * 0.5
    mask = _mx.random.uniform(shape=(B, 1, L, S)) > 0.4
    # Guarantee every row attends to something.
    mask = mask | (mx.arange(S) == 0)[None, None, None]

    got = np.array(glm5_model._streamed_absorbed_attention(
        q, latent, mask, 0.25), dtype=np.float32)

    qn = np.array(q, dtype=np.float64) * 0.25
    kn = np.array(latent, dtype=np.float64)[:, 0]
    s = np.einsum("bhld,bsd->bhls", qn, kn)
    s = np.where(np.array(mask), s, -np.inf)
    p = np.exp(s - s.max(-1, keepdims=True))
    p = p / p.sum(-1, keepdims=True)
    ref = np.einsum("bhls,bsd->bhld", p, kn)
    np.testing.assert_allclose(got, ref, rtol=2e-3, atol=2e-3)


def test_streamed_prefill_matches_short_path(monkeypatch):
    # Force the streaming thresholds tiny: a full-model prefill through the
    # streamed branches must match the plain (unstreamed) forward.
    args = _tiny_args()
    model = _random_model(args, seed=31)
    toks = [3, 9, 27, 40, 11, 5, 33, 60, 2, 17, 44, 8, 19, 52, 7, 21]

    base = np.array(model(mx.array([toks]), cache=model.make_cache()),
                    dtype=np.float32)
    monkeypatch.setattr(glm5_model, "_STREAM_MIN_KEYS", 4)
    monkeypatch.setattr(glm5_model, "_STREAM_BLOCK", 8)
    streamed = np.array(model(mx.array([toks]), cache=model.make_cache()),
                        dtype=np.float32)
    np.testing.assert_allclose(streamed, base, rtol=5e-3, atol=5e-3)


@pytest.mark.parametrize("total", [14, 15, 16, 17, 21])
def test_gathered_sparse_prefill_matches_masked_path(monkeypatch, total):
    # The union-gather sparse application must match both the full-width
    # masked (streamed) application and the plain unstreamed forward, at
    # every prompt-length residue mod kpool, with multi-block and partial
    # query blocks, and at nonzero chunk offsets.
    args = _tiny_args()
    model = _random_model(args, seed=31)
    toks = [int(t) for t in
            np.random.RandomState(3).randint(2, 60, size=total)]
    base = np.array(model(mx.array([toks]), cache=model.make_cache()),
                    dtype=np.float32)

    monkeypatch.setattr(glm5_model, "_STREAM_MIN_KEYS", 4)
    monkeypatch.setattr(glm5_model, "_STREAM_Q", 4)

    def run(split):
        cache = model.make_cache()
        if split:
            model(mx.array([toks[:split]]), cache=cache)
            out = model(mx.array([toks[split:]]), cache=cache)
        else:
            out = model(mx.array([toks]), cache=cache)
        return np.array(out, dtype=np.float32)[0, -1]

    for split in (0, 6, 9):
        monkeypatch.setattr(glm5_model, "_SPARSE_GATHER", True)
        gathered = run(split)
        monkeypatch.setattr(glm5_model, "_SPARSE_GATHER", False)
        masked = run(split)
        np.testing.assert_allclose(
            gathered, masked, rtol=2e-3, atol=2e-3,
            err_msg=f"gather vs masked, split={split}")
        np.testing.assert_allclose(
            gathered, base[0, -1], rtol=5e-3, atol=5e-3,
            err_msg=f"gather vs base, split={split}")


def test_sparse_disable_is_identity_below_threshold(monkeypatch):
    args = _tiny_args()
    model = _random_model(args)
    toks = [3, 9, 27, 40, 11]
    on = np.array(model(mx.array([toks]), cache=model.make_cache()),
                  dtype=np.float32)
    monkeypatch.setattr(glm5_model, "_SPARSE_DISABLE", True)
    off = np.array(model(mx.array([toks]), cache=model.make_cache()),
                   dtype=np.float32)
    np.testing.assert_array_equal(on, off)


def test_limited_swiglu_matches_reference():
    # gate is clamped one-sided (min with limit), up two-sided; silu after.
    g = mx.array([[-20.0, -1.0, 0.0, 5.0, 15.0]])
    u = mx.array([[-20.0, -1.0, 0.5, 5.0, 15.0]])
    got = np.array(_limited_swiglu(g, u, 10.0), dtype=np.float64)
    gn = np.minimum(np.array(g, dtype=np.float64), 10.0)
    un = np.clip(np.array(u, dtype=np.float64), -10.0, 10.0)
    ref = gn / (1.0 + np.exp(-gn)) * un
    np.testing.assert_allclose(got, ref, rtol=1e-5, atol=1e-6)


def test_moe_gate_selection_matches_reference():
    # Correction bias steers SELECTION only; weights are the unbiased
    # sigmoid scores of the selected experts, renormalized, then x2.5.
    args = _tiny_args()
    mx.random.seed(5)
    moe = glm5_model.Glm5NextMoE(args)
    from mlx.utils import tree_flatten, tree_unflatten
    params = [(k, mx.random.normal(v.shape) * 0.1)
              for k, v in tree_flatten(moe.parameters())]
    moe.update(tree_unflatten(params))
    bias = mx.array([10.0, 0.0, 0.0, 0.0])   # force expert 0 into the top-k
    moe.gate.e_score_correction_bias = bias
    mx.eval(moe.parameters())

    x = mx.random.normal((1, 3, args.hidden_size)) * 0.5
    inds, weights = moe.gate(x)
    inds_n = np.array(inds)
    w_n = np.array(weights, dtype=np.float64)

    logits = (np.array(x, dtype=np.float64)
              @ np.array(moe.gate.weight, dtype=np.float64).T)
    scores = 1.0 / (1.0 + np.exp(-logits))
    biased = scores + np.array(bias, dtype=np.float64)
    ref_inds = np.argsort(-biased, axis=-1)[..., :2]
    assert (inds_n == 0).any(), "bias must pull expert 0 into selection"
    assert (np.sort(inds_n, -1) == np.sort(ref_inds, -1)).all()
    picked = np.take_along_axis(scores, ref_inds, axis=-1)
    ref_w = picked / picked.sum(-1, keepdims=True) * 2.5
    # M5 f32 GEMM runs at TF32 precision by default (~1e-4 here); a real
    # renorm or bias-in-weights bug is O(1).
    np.testing.assert_allclose(np.sort(w_n, -1), np.sort(ref_w, -1),
                               rtol=2e-3, atol=1e-3)

    out = moe(x)
    mx.eval(out)
    assert out.shape == x.shape and bool(mx.isfinite(out).all())


def test_synth_config_instantiates_model():
    # The _synth_glm5next output must build this module 1:1 via ModelArgs
    # (the same path loader.build_model takes after ensure_registered).
    from gmlx.load.config_synth import synthesize_config
    from test_config_synth import _GLM5NEXT_SHAPES, _glm5next_meta

    glm5_model.ensure_registered()
    c = synthesize_config(_glm5next_meta(), tensor_shapes=_GLM5NEXT_SHAPES)
    args = ModelArgs.from_dict(c)
    assert args.layer_types == ["linear_attention", "full_attention",
                                "linear_attention", "full_attention"]
    model = Model(args)
    mx.eval(model.parameters())
    out = model(mx.array([[1, 2, 3]]))
    mx.eval(out)
    assert out.shape[-1] == c["vocab_size"]
    # KDA layer 0 dense (first_k_dense_replace=1); MLA layer 1 MoE + indexer.
    assert hasattr(model.layers[0].mlp, "gate_proj")
    assert hasattr(model.layers[1].mlp, "switch_mlp")
    assert hasattr(model.layers[1].self_attn, "indexer")
    assert hasattr(model.layers[1].self_attn, "embed_q")
    assert model.layers[1].self_attn.scale == pytest.approx(16 ** -0.5)


def test_nope_and_softmax_configs_rejected():
    with pytest.raises(ValueError, match="nope-only"):
        _tiny_args(qk_rope_head_dim=8)
    with pytest.raises(ValueError, match="sigmoid"):
        _tiny_args(scoring_func="softmax")


def test_cast_predicate_pins_fp32_params():
    args = _tiny_args()
    model = Model(args)
    pred = model.cast_predicate
    for path in ("model.layers.1.mlp.gate.e_score_correction_bias",
                 "model.layers.0.self_attn.a_folded",
                 "model.layers.0.self_attn.dt_bias",
                 "model.layers.1.attn_hc.fn",
                 "model.layers.1.ffn_hc.scale",
                 "model.layers.1.mlp.gate.weight",
                 "model.layers.1.self_attn.indexer.weights_proj.weight",
                 "model.layers.1.self_attn.indexer.compressor.ape"):
        assert pred(path) is False, path
    assert pred("model.layers.0.self_attn.q_proj.weight") is True

    qpred = model.quant_predicate
    assert qpred("model.layers.1.self_attn.indexer.weights_proj", None) is False
    assert qpred("model.layers.1.self_attn.indexer.wq_b", None) == {
        "group_size": 64, "bits": 8}
    assert qpred("model.layers.1.self_attn.indexer.compressor.wgate",
                 None) == {"group_size": 64, "bits": 8}
    assert qpred("model.layers.0.self_attn.q_proj", None) is True


def test_remap_covers_every_wire_tensor_onto_real_params():
    # Enumerate every GGUF name the PR-27754 converter emits for the tiny
    # config and assert each one remaps onto an actual parameter path of the
    # built model. Catches typos in both directions.
    from mlx.utils import tree_flatten

    from gmlx.load.remap import RemapDecision, parse_gguf_name

    args = _tiny_args()
    model = Model(args)
    params = {name for name, _ in tree_flatten(model.parameters())}

    names = ["token_embd.weight", "output_norm.weight", "output.weight"]
    for i, lt in enumerate(args.layer_types):
        names += [f"blk.{i}.attn_norm.weight", f"blk.{i}.ffn_norm.weight",
                  f"blk.{i}.attn_output.weight"]
        names += [f"blk.{i}.{t}.weight" for t in (
            "hc_attn_fn", "hc_attn_base", "hc_attn_scale",
            "hc_ffn_fn", "hc_ffn_base", "hc_ffn_scale")]
        if lt == "linear_attention":
            names += [f"blk.{i}.{t}" for t in (
                "attn_q.weight", "attn_k.weight", "attn_v.weight",
                "ssm_conv1d_q.weight", "ssm_conv1d_k.weight",
                "ssm_conv1d_v.weight", "ssm_f_a.weight", "ssm_f_b.weight",
                "ssm_beta.weight", "ssm_a", "ssm_dt.bias",
                "ssm_g_a.weight", "ssm_g_b.weight", "ssm_norm.weight")]
        else:
            names += [f"blk.{i}.{t}" for t in (
                "attn_q_a.weight", "attn_q_a_norm.weight", "attn_q_b.weight",
                "attn_kv_a_mqa.weight", "attn_kv_a_norm.weight",
                "attn_k_b.weight", "attn_v_b.weight",
                "indexer.attn_q_b.weight", "indexer.attn_k.weight",
                "indexer.k_norm.weight", "indexer.k_norm.bias",
                "indexer.proj.weight", "indexer_compressor_ape.weight",
                "indexer_compressor_gate.weight")]
        if i < args.first_k_dense_replace:
            names += [f"blk.{i}.ffn_{t}.weight"
                      for t in ("gate", "up", "down")]
        else:
            names += [f"blk.{i}.{t}" for t in (
                "ffn_gate_inp.weight", "exp_probs_b.bias",
                "ffn_gate_exps.weight", "ffn_up_exps.weight",
                "ffn_down_exps.weight", "ffn_gate_shexp.weight",
                "ffn_up_shexp.weight", "ffn_down_shexp.weight")]

    for name in names:
        dec = parse_gguf_name("glm5next", name)
        assert dec.kind == RemapDecision.KIND_MAP, (name, dec.reason)
        assert dec.hf_name in params, (name, dec.hf_name)

    # The MTP tail block: standard tensors map one past the trunk (sanitize
    # drops them; the drafter loads them separately); nextn extras skip.
    n = args.num_hidden_layers
    dec = parse_gguf_name("glm5next", f"blk.{n}.attn_q_a.weight")
    assert dec.kind == RemapDecision.KIND_MAP
    assert dec.hf_name.startswith(f"model.layers.{n}.")
    for t in ("nextn.eh_proj.weight", "nextn.enorm.weight",
              "nextn.hnorm.weight", "nextn.shared_head_norm.weight"):
        dec = parse_gguf_name("glm5next", f"blk.{n}.{t}")
        assert dec.kind == RemapDecision.KIND_SKIP, (t, dec.kind)


def test_dequantized_fetch_strided_cache_slices():
    # _dequantized must hold the round trip on strided cache slices.
    class _QCache:
        group_size, bits = 64, 8

    mx.random.seed(11)
    k = mx.random.normal((1, 4, 63, 64)).astype(mx.float16)
    w, s, b = mx.quantize(k, group_size=64, bits=8)
    bufs = tuple(
        mx.zeros((1, 4, 256, t.shape[-1]), dtype=t.dtype)
        for t in (w, s, b))
    for buf, t in zip(bufs, (w, s, b)):
        buf[..., :63, :] = t
    fetch = tuple(buf[..., :63, :] for buf in bufs)
    got = glm5_model._dequantized(fetch, _QCache())
    err = mx.abs(got - k).max().item()
    assert err < 0.05, f"strided dequantized fetch err={err}"
    # dense fetches pass through untouched
    assert glm5_model._dequantized(k, _QCache()) is k


def test_moe_gate_kq_router_matches_compiled_select(monkeypatch):
    import gmlx.models.glm5_next.model as glm
    if not mx.metal.is_available() or not glm._kq_router_available():
        pytest.skip("kq router sigmoid arm unavailable")
    args = _tiny_args()
    model = _random_model(args, seed=7)
    gate = next(ly.mlp.gate for ly in model.layers
                if hasattr(ly.mlp, "gate"))
    gate.e_score_correction_bias = mx.random.normal(gate.e_score_correction_bias.shape)
    x = mx.random.normal((1, 3, args.hidden_size)).astype(mx.bfloat16)
    monkeypatch.setattr(glm, "_KQ_ROUTER_ENABLED", True)
    inds_kq, w_kq = gate(x)
    monkeypatch.setattr(glm, "_KQ_ROUTER_ENABLED", False)
    inds_ref, w_ref = gate(x)
    mx.eval(inds_kq, w_kq, inds_ref, w_ref)
    assert inds_kq.shape == inds_ref.shape and w_kq.shape == w_ref.shape
    ik, ir = np.array(inds_kq).reshape(-1, gate.top_k), np.array(inds_ref).reshape(-1, gate.top_k)
    wk, wr = np.array(w_kq).reshape(-1, gate.top_k), np.array(w_ref).reshape(-1, gate.top_k)
    for t in range(ik.shape[0]):
        assert set(ik[t]) == set(ir[t]), t
        ok = np.argsort(ik[t])
        orr = np.argsort(ir[t])
        np.testing.assert_allclose(wk[t][ok], wr[t][orr], rtol=1e-5, atol=1e-6)


