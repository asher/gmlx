#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Asher Feldman
"""Muse Spark model against a naive float64 reference of llama.cpp's
muse-spark graph, on tiny random weights. No GGUF and no download.

The reference covers both norm epsilons, a non-ones k_norm, sliding and
global layers, the per-expert norm before the router weights, and the
selection-only router bias. The loader-shaped tests check that the residual
stream stays bf16 and that the fused MoE gate accepts the block's input.
"""

from __future__ import annotations

import numpy as np
import pytest

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten

from gmlx.models.muse_spark.model import (
    DecoderLayer,
    Model,
    ModelArgs,
    MuseSparkMoE,
)

_LAYER_TYPES = [
    "sliding_attention",
    "full_attention",
    "sliding_attention",
    "sliding_attention",
]


def _tiny_args(**over) -> ModelArgs:
    kw = dict(
        model_type="muse_spark",
        hidden_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        vocab_size=48,
        layer_types=list(_LAYER_TYPES),
        num_experts=8,
        num_experts_per_tok=3,
        moe_intermediate_size=24,
        moe_latent_size=32,
        sliding_window=4,
        rms_norm_eps=1e-5,
        post_norm_eps=1e-8,
        rope_theta=500000.0,
        rope_parameters={"rope_theta": 500000.0, "rope_type": "default"},
        max_position_embeddings=4096,
        output_multiplier=0.1767766922712326,
        final_logit_softcapping=20.0,
        routed_scaling_factor=1.5,
        norm_topk_prob=True,
    )
    kw.update(over)
    return ModelArgs(**kw)


def _randomize(module, seed: int, branch_scale: float = 1.0):
    """Random weights with norms around one and alpha/beta in (0, 1).
    ``branch_scale`` shrinks the o_proj and latent_up outputs, so the two
    post-norms see inputs small enough for their epsilon to matter."""
    rng = np.random.default_rng(seed)
    upd = []
    for k, v in tree_flatten(module.parameters()):
        shape = v.shape
        if k.endswith(("_resid_alpha", "_resid_beta")):
            a = rng.uniform(0.2, 0.9, shape)
        elif "norm" in k or k.endswith("layernorm.weight"):
            a = 1.0 + 0.3 * rng.standard_normal(shape)
        elif k.endswith("e_score_correction_bias"):
            a = 0.1 * rng.standard_normal(shape)
        else:
            a = rng.standard_normal(shape) / np.sqrt(shape[-1])
            if k.endswith(("o_proj.weight", "latent_up.weight")):
                a = a * branch_scale
        upd.append((k, mx.array(a.astype(np.float32))))
    module.update(tree_unflatten(upd))
    mx.eval(module.parameters())
    return module


def _np(x) -> np.ndarray:
    return np.array(x.astype(mx.float32), dtype=np.float64)


def _rms(x, w=None, eps=1e-5):
    y = x / np.sqrt((x * x).mean(-1, keepdims=True) + eps)
    return y if w is None else y * w


def _rope_traditional(x, theta):
    # x: [T, H, D], interleaved pairs (NORM rope).
    T, _, D = x.shape
    freqs = theta ** (-np.arange(0, D, 2) / D)
    ang = np.arange(T)[:, None] * freqs[None, :]
    cos, sin = np.cos(ang)[:, None, :], np.sin(ang)[:, None, :]
    x0, x1 = x[..., 0::2], x[..., 1::2]
    out = np.empty_like(x)
    out[..., 0::2] = x0 * cos - x1 * sin
    out[..., 1::2] = x0 * sin + x1 * cos
    return out


def _ref_moe(mlp, c, args, eps, post_eps, *, norm_after_mix=False):
    """llama.cpp build_moe_ffn with probs_in, exp_probs_b and post_exp_norm,
    then latent_up and the post-FFN norm. ``c`` is [T, d]."""
    logits = c @ _np(mlp.gate.weight).T
    scores = 1.0 / (1.0 + np.exp(-logits))
    choice = scores + _np(mlp.e_score_correction_bias)
    k = mlp.num_experts_per_tok
    inds = np.argsort(-choice, axis=-1, kind="stable")[:, :k]
    w = np.take_along_axis(scores, inds, axis=-1)
    if args.norm_topk_prob:
        w = w / np.maximum(w.sum(-1, keepdims=True), 6.103515625e-5)
    w = w * args.routed_scaling_factor

    z = _rms(c @ _np(mlp.latent_down.weight).T, _np(mlp.pre_exp_norm.weight), eps)
    sw = mlp.switch_mlp
    gw, uw, dw = (
        _np(sw.gate_proj.weight),
        _np(sw.up_proj.weight),
        _np(sw.down_proj.weight),
    )
    pe = _np(mlp.post_exp_norm.weight)
    y = np.zeros_like(z)
    for t in range(c.shape[0]):
        for j, e in enumerate(inds[t]):
            g, u = z[t] @ gw[e].T, z[t] @ uw[e].T
            ye = (g / (1.0 + np.exp(-g)) * u) @ dw[e].T
            if not norm_after_mix:
                ye = _rms(ye * pe, None, post_eps)
            y[t] += w[t, j] * ye
    if norm_after_mix:
        y = _rms(y * pe, None, post_eps)
    y = y @ _np(mlp.latent_up.weight).T
    return _rms(y * _np(mlp.post_ffw_norm.weight), None, post_eps), inds


def _ref_layer(layer, h, args, eps=None, post_eps=None):
    """One decoder layer of llama.cpp's muse-spark graph. ``h`` is [T, d]."""
    eps = args.rms_norm_eps if eps is None else eps
    post_eps = args.post_norm_eps if post_eps is None else post_eps
    at = layer.self_attn
    T = h.shape[0]
    H, KV, D = args.num_attention_heads, args.num_key_value_heads, args.head_dim

    xn = _rms(h, _np(layer.input_layernorm.weight), eps)
    q = (xn @ _np(at.q_proj.weight).T).reshape(T, H, D)
    k = (xn @ _np(at.k_proj.weight).T).reshape(T, KV, D)
    v = (xn @ _np(at.v_proj.weight).T).reshape(T, KV, D)
    gate = xn @ _np(at.gate_proj.weight).T
    q = _rms(q, _np(at.q_norm.weight), eps)
    k = _rms(k, _np(at.k_norm.weight), eps)
    if layer.use_sliding:
        q = _rope_traditional(q, args.rope_theta)
        k = _rope_traditional(k, args.rope_theta)

    i, j = np.arange(T)[:, None], np.arange(T)[None, :]
    allowed = j <= i
    if layer.use_sliding:
        allowed &= (i - j) < args.sliding_window
    o = np.zeros((T, H, D))
    for hh in range(H):
        kv = hh // (H // KV)
        s = (q[:, hh] @ k[:, kv].T) * D**-0.5
        s = np.where(allowed, s, -np.inf)
        p = np.exp(s - s.max(-1, keepdims=True))
        p /= p.sum(-1, keepdims=True)
        o[:, hh] = p @ v[:, kv]
    o = _rms(o, None, eps).reshape(T, H * D)
    o = o * (1.0 / (1.0 + np.exp(-gate)))
    r = _rms(o @ _np(at.o_proj.weight).T, None, post_eps)
    h = h * _np(layer.attn_resid_alpha) + r * _np(layer.attn_resid_beta)

    c = _rms(h, _np(layer.post_attention_layernorm.weight), eps)
    y, _ = _ref_moe(layer.mlp, c, args, eps, post_eps)
    return h * _np(layer.ffn_resid_alpha) + y * _np(layer.ffn_resid_beta)


def _ref_model(model, tokens, args):
    h = _np(model.model.embed_tokens.weight)[tokens]
    h = _rms(h, None, args.rms_norm_eps)
    for layer in model.layers:
        h = _ref_layer(layer, h, args)
    h = _rms(h, _np(model.model.norm.weight), args.rms_norm_eps)
    out = (h @ _np(model.lm_head.weight).T) * args.output_multiplier
    cap = args.final_logit_softcapping
    return np.tanh(out / cap) * cap


def _mask(T, sliding, args):
    i, j = np.arange(T)[:, None], np.arange(T)[None, :]
    allowed = j <= i
    if sliding:
        allowed &= (i - j) < args.sliding_window
    return mx.array(allowed)


def _rel(a, b):
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-12))


def test_layer_matches_reference_on_both_epsilons():
    """A sliding and a global layer against the reference. The inputs and
    branch weights are small enough that a norm run at the other epsilon
    moves the output far past the tolerance."""
    args = _tiny_args()
    for idx in (0, 1):
        layer = _randomize(
            DecoderLayer(args, _LAYER_TYPES[idx] == "sliding_attention"),
            seed=10 + idx,
            branch_scale=1e-4,
        )
        assert not np.allclose(_np(layer.self_attn.k_norm.weight), 1.0)
        rng = np.random.default_rng(idx)
        x = 3e-3 * rng.standard_normal((6, args.hidden_size))
        got = _np(
            layer(
                mx.array(x.astype(np.float32))[None], _mask(6, layer.use_sliding, args)
            )[0]
        )
        want = _ref_layer(layer, x, args)
        assert _rel(got, want) < 2e-4, idx
        swapped = _ref_layer(
            layer, x, args, eps=args.post_norm_eps, post_eps=args.rms_norm_eps
        )
        assert _rel(swapped, want) > 1e-1, idx


def test_full_model_matches_reference_across_the_window():
    """Seven tokens over a window of four: the sliding layers mask the far
    keys and rotate, the global layer sees every key with no rope."""
    args = _tiny_args()
    model = _randomize(Model(args), seed=3)
    tokens = np.array([5, 17, 2, 40, 33, 8, 21])
    got = _np(model(mx.array(tokens)[None])[0])
    want = _ref_model(model, tokens, args)
    assert _rel(got, want) < 1e-4
    # The mask matters: a global-only reference differs.
    for layer in model.layers:
        layer.use_sliding = False
    wide = _ref_model(model, tokens, args)
    assert _rel(wide, want) > 1e-2


def test_cached_decode_matches_full_forward():
    args = _tiny_args()
    model = _randomize(Model(args), seed=4)
    tokens = [5, 17, 2, 40, 33, 8, 21]
    full = model(mx.array([tokens]))
    cache = model.make_cache()
    model(mx.array([tokens[:3]]), cache=cache)
    steps = [model(mx.array([[t]]), cache=cache) for t in tokens[3:]]
    mx.eval(full, steps)
    for n, s in enumerate(steps):
        np.testing.assert_allclose(
            _np(s[0, 0]), _np(full[0, 3 + n]), rtol=1e-4, atol=1e-4
        )


def test_moe_bias_moves_selection_but_not_weights():
    """The bias picks the experts. The weights are the unbiased sigmoid
    scores of the picked experts, renormalized and scaled, and each expert
    output is normed before its weight applies."""
    args = _tiny_args()
    moe = _randomize(MuseSparkMoE(args), seed=5)
    rng = np.random.default_rng(6)
    c = rng.standard_normal((4, args.hidden_size))
    cx = mx.array(c.astype(np.float32))[None]

    scores = 1.0 / (1.0 + np.exp(-(c @ _np(moe.gate.weight).T)))
    low = int(np.argmin(scores[0]))
    bias = np.zeros(args.num_experts)
    bias[low] = 10.0
    moe.e_score_correction_bias = mx.array(bias.astype(np.float32))

    got = _np(moe(cx)[0])
    want, inds = _ref_moe(moe, c, args, args.rms_norm_eps, args.post_norm_eps)
    assert (inds == low).any(axis=-1).all()
    assert _rel(got, want) < 1e-4

    # A shift shared by every expert keeps the selection, so the output holds.
    moe.e_score_correction_bias = mx.array((bias + 3.0).astype(np.float32))
    np.testing.assert_allclose(_np(moe(cx)[0]), got, rtol=1e-5, atol=1e-6)

    # The reference is sensitive to the norm's place.
    after, _ = _ref_moe(
        moe, c, args, args.rms_norm_eps, args.post_norm_eps, norm_after_mix=True
    )
    assert _rel(after, want) > 1e-2


def test_top_k_is_read_from_the_block():
    """The streaming top-k override writes num_experts_per_tok on the block,
    and the forward reads it there."""
    args = _tiny_args()
    moe = _randomize(MuseSparkMoE(args), seed=7)
    seen = []
    orig = moe.switch_mlp

    class _Spy(nn.Module):
        def __call__(self, x, inds):
            seen.append(inds.shape[-1])
            return orig(x, inds)

    moe.switch_mlp = _Spy()
    moe.num_experts_per_tok = 1
    mx.eval(moe(mx.zeros((1, 2, args.hidden_size))))
    assert isinstance(MuseSparkMoE(args).num_experts_per_tok, int)
    assert seen == [1]


def test_sliding_layers_get_a_rotating_cache():
    from gmlx.cache.kv_cache import KVCache, RotatingKVCache

    args = _tiny_args()
    caches = Model(args).make_cache()
    for kind, c in zip(_LAYER_TYPES, caches):
        if kind == "sliding_attention":
            assert isinstance(c, RotatingKVCache)
            assert c.max_size == args.sliding_window
        else:
            assert type(c) is KVCache


def test_lookahead_input_is_kept_only_for_one_token_steps():
    """Lookahead reads the full-width input on one-token steps only. A
    prefill chunk must not stay referenced after the call."""
    args = _tiny_args()
    moe = _randomize(MuseSparkMoE(args), seed=8)
    object.__setattr__(moe.switch_mlp, "_kq_lookahead", object())
    rng = np.random.default_rng(9)

    one = mx.array(rng.standard_normal((1, 1, args.hidden_size)).astype(np.float32))
    mx.eval(moe(one))
    assert moe.switch_mlp._kq_la_input is one

    chunk = mx.array(rng.standard_normal((1, 5, args.hidden_size)).astype(np.float32))
    mx.eval(moe(chunk))
    assert moe.switch_mlp._kq_la_input is None


def test_zero_weight_scale_is_skipped_like_llama_cpp():
    """llama.cpp applies expert_weights_scale only when it is not 0 or 1."""
    rng = np.random.default_rng(10)
    c = mx.array(rng.standard_normal((1, 3, 64)).astype(np.float32))
    outs = []
    for scale in (0.0, 1.0):
        moe = _randomize(MuseSparkMoE(_tiny_args(routed_scaling_factor=scale)), seed=11)
        outs.append(_np(moe(c)))
    np.testing.assert_array_equal(outs[0], outs[1])
    assert np.abs(outs[0]).max() > 0.1


def test_cast_predicate_matches_the_loader_keep_list():
    from gmlx.load.loader import _FP32_KEEP_BY_MODEL_TYPE

    keep = _FP32_KEEP_BY_MODEL_TYPE["muse_spark"]
    model = Model(_tiny_args())
    pred = model.cast_predicate
    pinned = set()
    for path, _ in tree_flatten(model.parameters()):
        assert pred(path) == (not any(s in path for s in keep)), path
        if not pred(path):
            pinned.add(path.split(".", 3)[-1])
    assert pinned == {
        "mlp.gate.weight",
        "mlp.e_score_correction_bias",
        "attn_resid_alpha",
        "attn_resid_beta",
        "ffn_resid_alpha",
        "ffn_resid_beta",
    }


def test_remap_covers_every_wire_tensor_onto_real_params():
    """Every tensor llama.cpp's muse-spark loader creates maps onto a
    parameter of the model, and every parameter is loaded."""
    from gmlx.load.remap import RemapDecision, parse_gguf_name

    args = _tiny_args()
    params = {k for k, _ in tree_flatten(Model(args).parameters())}
    per_layer = [
        "attn_norm.weight",
        "attn_q.weight",
        "attn_k.weight",
        "attn_v.weight",
        "attn_output.weight",
        "attn_q_norm.weight",
        "attn_k_norm.weight",
        "attn_gate.weight",
        "attn_resid_alpha.weight",
        "attn_resid_beta.weight",
        "ffn_norm.weight",
        "post_ffw_norm.weight",
        "ffn_resid_alpha.weight",
        "ffn_resid_beta.weight",
        "ffn_gate_inp.weight",
        "exp_probs_b.bias",
        "ffn_latent_down.weight",
        "ffn_latent_up.weight",
        "ffn_pre_exp_norm.weight",
        "ffn_post_exp_norm.weight",
        "ffn_gate_exps.weight",
        "ffn_up_exps.weight",
        "ffn_down_exps.weight",
    ]
    names = ["token_embd.weight", "output_norm.weight", "output.weight"]
    names += [f"blk.{i}.{t}" for i in range(args.num_hidden_layers) for t in per_layer]
    mapped = set()
    for name in names:
        dec = parse_gguf_name("muse-spark", name)
        assert dec.kind == RemapDecision.KIND_MAP, (name, dec.reason)
        assert dec.transform == "passthrough", name
        assert dec.hf_name in params, (name, dec.hf_name)
        mapped.add(dec.hf_name)
    assert mapped == params


def test_synth_config_builds_this_model():
    from gmlx.load.config_synth import synthesize_config
    from test_config_synth import _MUSE_SPARK_SHAPES, _muse_spark_meta

    c = synthesize_config(_muse_spark_meta(), tensor_shapes=_MUSE_SPARK_SHAPES)
    args = ModelArgs.from_dict(c)
    model = Model(args)
    assert [ly.use_sliding for ly in model.layers] == [True, False, True, True]
    assert model.layers[0].mlp.switch_mlp.gate_proj.weight.shape == (8, 24, 32)
    assert model.layers[0].mlp.latent_down.weight.shape == (32, 64)


def _loader_shaped():
    from tune.tiny_train_archs import build

    model, config, _ = build("muse_spark")
    return model, config


def test_residual_stream_stays_bf16_and_feeds_the_fused_moe(monkeypatch):
    """With the loader's dtypes (fp32 router, bias and alpha/beta, bf16
    elsewhere) every layer returns bf16, and the expert container receives a
    bf16 input that the fused decode gate accepts."""
    import gmlx.load.modules as modules
    from gmlx.load.dtypes import activation_dtype

    if activation_dtype() != mx.bfloat16:
        pytest.skip("the loader picks float16 activations on this device")
    model, _ = _loader_shaped()
    layer0 = model.layers[0]
    assert layer0.attn_resid_alpha.dtype == mx.float32
    assert layer0.mlp.gate.weight.dtype == mx.float32
    assert layer0.mlp.e_score_correction_bias.dtype == mx.float32
    assert layer0.post_attention_layernorm.weight.dtype == mx.bfloat16

    glu_cls = type(layer0.mlp.switch_mlp)
    if not hasattr(glu_cls, "_fused_ok"):
        pytest.skip("fused MoE glu not installable in this build")
    monkeypatch.setattr(modules, "_kq_fused_device_ok", lambda *m: True)
    orig = glu_cls._fused_ok
    verdicts = []

    def record(self, x, indices):
        verdicts.append((x.dtype, orig(self, x, indices)))
        return False  # the stock path runs on the CPU

    monkeypatch.setattr(glu_cls, "_fused_ok", record)

    outs = []
    call = DecoderLayer.__call__

    def spy(self, *a, **k):
        out = call(self, *a, **k)
        outs.append(out.dtype)
        return out

    monkeypatch.setattr(DecoderLayer, "__call__", spy)
    cache = model.make_cache()
    model(mx.array([[1, 2, 3]]), cache=cache)
    verdicts.clear()
    mx.eval(model(mx.array([[4]]), cache=cache))
    assert outs and set(outs) == {mx.bfloat16}
    assert len(verdicts) == len(model.layers)
    assert all(v == (mx.bfloat16, True) for v in verdicts), verdicts
