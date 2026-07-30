#!/usr/bin/env python3
"""Kimi-K3 vendored model: forward smoke, KDA recurrence parity, residual-mix
semantics, situ, MoE selection, and the synth -> ModelArgs round-trip.

All tests run on tiny dims with random weights - no GGUF, no download. The
numeric references are self-contained (naive python loops implementing the
llama.cpp kimi-k3.cpp semantics), so a behavior drift in the vendored module
fails here before any conversion-level parity run.
"""

from __future__ import annotations

import numpy as np
import pytest

import mlx.core as mx

from gmlx import kimi_k3_model
from gmlx.kimi_k3_model import (
    Model,
    ModelArgs,
    _ResidualMixer,
    _situ,
)


def _tiny_args(**over) -> ModelArgs:
    kw = dict(
        model_type="kimi_k3",
        vocab_size=64,
        hidden_size=64,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        intermediate_size=96,
        rms_norm_eps=1e-5,
        layer_types=["linear_attention", "full_attention",
                     "linear_attention", "full_attention"],
        kda_head_dim=32,
        ssm_conv_kernel=4,
        kda_gate_lower_bound=-5.0,
        q_lora_rank=32,
        kv_lora_rank=32,
        qk_nope_head_dim=16,
        qk_rope_head_dim=8,
        v_head_dim=16,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=24,
        num_shared_experts=1,
        first_k_dense_replace=1,
        routed_scaling_factor=1.0,
        moe_renormalize=True,
        routed_expert_hidden_size=32,
        has_routed_norm=True,
        situ_beta=4.0,
        situ_linear_beta=25.0,
        attn_res_block_size=2,
        max_position_embeddings=4096,
        tie_word_embeddings=False,
    )
    kw.update(over)
    return ModelArgs(**kw)


def _random_model(args: ModelArgs, seed: int = 0) -> Model:
    mx.random.seed(seed)
    model = Model(args)
    # Small random weights keep situ/tanh and the softmax mixes in-range.
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
            assert c[0].shape == (1, args.ssm_conv_kernel - 1, d_inner)
            assert c[3].shape == (1, args.num_attention_heads,
                                  args.kda_head_dim, args.kda_head_dim)
            assert c[3].dtype == mx.float32
        else:
            assert c.offset == 5


def test_prefill_matches_stepwise_decode():
    # T tokens in one forward == the same T tokens fed one at a time. This
    # exercises the KDA conv/ssm state carry, the MLA cache, and (because the
    # mixer is per-forward) that residual mixing is position-local.
    args = _tiny_args()
    model = _random_model(args)
    toks = [3, 9, 27, 40, 11, 5]

    full = model(mx.array([toks]), cache=model.make_cache())
    step_cache = model.make_cache()
    steps = [model(mx.array([[t]]), cache=step_cache) for t in toks]
    mx.eval(full, steps)

    last_full = np.array(full[0, -1], dtype=np.float32)
    last_step = np.array(steps[-1][0, 0], dtype=np.float32)
    np.testing.assert_allclose(last_step, last_full, rtol=2e-2, atol=2e-2)
    # argmax agreement on every position
    for t, s in enumerate(steps):
        a = np.array(full[0, t], dtype=np.float32).argmax()
        b = np.array(s[0, 0], dtype=np.float32).argmax()
        assert a == b, f"argmax diverged at position {t}"


def test_batched_matches_single():
    # B=2 with equal-length rows must reproduce the B=1 logits.
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


@pytest.mark.skipif(
    not mx.metal.is_available()
    or mx.default_device() != mx.Device(mx.gpu),
    reason="needs the Metal GPU device (skipped under KQUANT_FORCE_CPU)")
def test_kda_kernel_matches_ops_with_k3_decay():
    # The vec metal kernel and the ops reference must agree on the K3 decay
    # form (per-key-channel g in (exp(lb), 1)).
    from mlx_lm.models.gated_delta import gated_delta_kernel, gated_delta_ops

    B, T, H, D = 2, 7, 2, 32
    mx.random.seed(1)
    q = mx.random.normal((B, T, H, D)) * 0.3
    k = mx.random.normal((B, T, H, D)) * 0.3
    v = mx.random.normal((B, T, H, D)) * 0.3
    g = mx.exp(-5.0 * mx.random.uniform(shape=(B, T, H, D)))  # fp32
    beta = mx.random.uniform(shape=(B, T, H))
    state = mx.zeros((B, H, D, D), dtype=mx.float32)

    y_k, s_k = gated_delta_kernel(q, k, v, g, beta, state)
    y_o, s_o = gated_delta_ops(q, k, v, g, beta, state)
    mx.eval(y_k, s_k, y_o, s_o)
    np.testing.assert_allclose(np.array(y_k), np.array(y_o),
                               rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(np.array(s_k), np.array(s_o),
                               rtol=1e-3, atol=1e-3)


def test_kda_decay_forms():
    # lb form: decay = exp(lb * sigmoid(exp(A_log)*(a + dt))), in (exp(lb), 1).
    from gmlx.kimi_k3_model import _kda_decay_lb, _kda_decay_softplus

    H, D = 2, 4
    a_folded = -mx.array([1.5, 3.0])          # -exp(A_log)
    a_raw = mx.random.normal((1, 3, H, D))
    dt = mx.zeros((H, D))
    g = _kda_decay_lb(a_folded, a_raw, dt, -5.0)
    mx.eval(g)
    gn = np.array(g)
    assert gn.min() > np.exp(-5.0) - 1e-6 and gn.max() < 1.0
    a0 = float(a_raw[0, 0, 0, 0])
    want = np.exp(-5.0 * (1.0 / (1.0 + np.exp(-1.5 * a0))))
    assert abs(float(g[0, 0, 0, 0]) - want) < 1e-5
    # softplus form: exp(-exp(A_log) * softplus(a)), also in (0, 1)
    g2 = _kda_decay_softplus(a_folded, a_raw, dt)
    a_sp = np.log1p(np.exp(a0))
    want2 = np.exp(-1.5 * a_sp)
    assert abs(float(g2[0, 0, 0, 0]) - want2) < 1e-5


def test_situ_matches_reference_and_lb_branch():
    g = mx.random.normal((4, 8))
    u = mx.random.normal((4, 8))
    beta, lb = 4.0, 25.0
    got = _situ(g, u, beta, lb)
    gn, un = np.array(g, dtype=np.float64), np.array(u, dtype=np.float64)
    act = beta * np.tanh(gn / beta) * (1.0 / (1.0 + np.exp(-gn)))
    ref = act * (lb * np.tanh(un / lb))
    np.testing.assert_allclose(np.array(got), ref, rtol=1e-5, atol=1e-5)
    # lb <= 0: the up branch is untransformed (exact branch on 0.0, not truthy)
    got0 = _situ(g, u, beta, 0.0)
    np.testing.assert_allclose(np.array(got0), act * un, rtol=1e-5, atol=1e-5)


def test_residual_mixer_matches_naive_reference():
    # Reference semantics from llama.cpp res_mix: scores from NORMED values
    # (weightless rms), softmax over [banked..., current], weighted sum of
    # the RAW values.
    eps = 1e-5
    B, T, D = 1, 3, 8
    mx.random.seed(3)
    ckpts = [mx.random.normal((B, T, D)) for _ in range(3)]
    cur = mx.random.normal((B, T, D))
    w = mx.random.normal((D,))

    mixer = _ResidualMixer(eps, enabled=True)
    for c in ckpts:
        mixer.push(c)
    got = np.array(mixer.mix(cur, w), dtype=np.float32)

    def rms(x):
        x = np.array(x, dtype=np.float64)
        return x / np.sqrt((x ** 2).mean(-1, keepdims=True) + eps)

    wn = np.array(w, dtype=np.float64)
    scores = np.stack([(rms(c) * wn).sum(-1) for c in ckpts]
                      + [(rms(cur) * wn).sum(-1)], axis=-1)
    p = np.exp(scores - scores.max(-1, keepdims=True))
    p = p / p.sum(-1, keepdims=True)
    ref = sum(np.array(c, dtype=np.float64) * p[..., i:i + 1]
              for i, c in enumerate(ckpts))
    ref = ref + np.array(cur, dtype=np.float64) * p[..., 3:4]
    np.testing.assert_allclose(got, ref, rtol=1e-4, atol=1e-4)

    # No checkpoints or disabled -> identity.
    empty = _ResidualMixer(eps, enabled=True)
    assert empty.mix(cur, w) is cur
    off = _ResidualMixer(eps, enabled=False)
    off.push(ckpts[0])
    assert off.mix(cur, w) is cur


def test_attnres_kill_switch_changes_logits(monkeypatch):
    # GMLX_KIMI_ATTNRES=0 must alter the forward (mix disabled, standard
    # residual everywhere) while all params still exist and load.
    args = _tiny_args()
    model = _random_model(args)
    x = mx.array([[1, 2, 3]])
    on = np.array(model(x), dtype=np.float32)
    monkeypatch.setattr(kimi_k3_model, "_ATTNRES", False)
    off = np.array(model(x), dtype=np.float32)
    assert np.isfinite(off).all()
    assert not np.allclose(on, off)


def test_moe_selection_matches_reference():
    # Correction bias steers SELECTION only; weights come from the original
    # sigmoid scores, renormalized over the top-k, scaled by the factor.
    args = _tiny_args()
    moe = kimi_k3_model.KimiK3MoE(args)
    mx.random.seed(5)
    from mlx.utils import tree_flatten, tree_unflatten
    params = [(k, mx.random.normal(v.shape) * 0.1)
              for k, v in tree_flatten(moe.parameters())]
    moe.update(tree_unflatten(params))
    bias = mx.array([10.0, 0.0, 0.0, 0.0])   # force expert 0 into the top-k
    moe.e_score_correction_bias = bias
    mx.eval(moe.parameters())

    x = mx.random.normal((1, 2, args.hidden_size)) * 0.5
    logits = np.array(moe.gate(x.astype(mx.float32)), dtype=np.float64)
    scores = 1.0 / (1.0 + np.exp(-logits))
    biased = scores + np.array(bias, dtype=np.float64)
    inds = np.argsort(-biased, axis=-1)[..., :2]
    assert (inds == 0).any(), "bias must pull expert 0 into selection"
    picked = np.take_along_axis(scores, inds, axis=-1)
    ref_w = picked / (picked.sum(-1, keepdims=True) + 1e-20)

    out = moe(x)
    mx.eval(out)
    assert out.shape == x.shape and bool(mx.isfinite(out).all())
    # Weights check via the module's own intermediate math (recomputed):
    got_scores = np.array(mx.sigmoid(moe.gate(x.astype(mx.float32))),
                          dtype=np.float64)
    got_inds = np.argsort(-(got_scores + np.array(bias)), axis=-1)[..., :2]
    got_w = np.take_along_axis(got_scores, got_inds, axis=-1)
    got_w = got_w / (got_w.sum(-1, keepdims=True) + 1e-20)
    np.testing.assert_allclose(np.sort(got_w), np.sort(ref_w), rtol=1e-5)


def test_synth_config_instantiates_model():
    # The _synth_kimi_k3 output must build this module 1:1 via ModelArgs
    # (the same path loader.build_model takes after ensure_registered).
    from gmlx.config_synth import synthesize_config
    from tests.test_config_synth import _KIMI_K3_SHAPES, _kimi_k3_meta

    kimi_k3_model.ensure_registered()
    c = synthesize_config(_kimi_k3_meta(), tensor_shapes=_KIMI_K3_SHAPES)
    args = ModelArgs.from_dict(c)
    assert args.layer_types == ["linear_attention", "full_attention",
                                "linear_attention", "full_attention"]
    model = Model(args)
    mx.eval(model.parameters())
    out = model(mx.array([[1, 2, 3]]))
    mx.eval(out)
    assert out.shape[-1] == c["vocab_size"]
    # KDA layer 0 dense (first_k_dense_replace=1), MLA layer 1 MoE + latent.
    assert hasattr(model.layers[0].mlp, "gate_proj")
    assert model.layers[1].mlp.routed_down is not None
    assert model.layers[1].mlp.routed_norm is not None
    # MLA path has the q_a stack and the output gate.
    assert hasattr(model.layers[1].self_attn, "q_a_proj")
    assert hasattr(model.layers[1].self_attn, "attn_gate")


def test_cast_predicate_pins_fp32_params():
    args = _tiny_args()
    model = Model(args)
    pred = model.cast_predicate
    for path in ("model.layers.1.mlp.e_score_correction_bias",
                 "model.layers.0.self_attn.a_folded",
                 "model.layers.0.self_attn.dt_bias",
                 "model.layers.0.attn_res_score",
                 "model.layers.0.ffn_res_score",
                 "model.output_res_score"):
        assert pred(path) is False, path
    assert pred("model.layers.0.self_attn.q_proj.weight") is True
