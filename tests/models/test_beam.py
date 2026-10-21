#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Asher Feldman
"""Beam model against a naive float64 reference of llama.cpp's beam graph,
on tiny random weights. No GGUF and no download.

The reference covers the per-head gate bias, NeoX rope on the sliding
layers, the NoPE query scale on the full layers, the sandwich norms, the
depth-scaled residuals, dense and MoE layers, and grouped routing with
-inf group masking. The loader-shaped tests check that the residual stream
stays bf16 and that the fused MoE gate accepts the block's input.
"""

from __future__ import annotations

import re

import numpy as np
import pytest

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_map, tree_unflatten

from gmlx.models.beam.model import (
    BeamGate,
    BeamMoE,
    DecoderLayer,
    Model,
    ModelArgs,
    nope_query_scale,
)

_LAYER_TYPES = [
    "sliding_attention",
    "sliding_attention",
    "full_attention",
    "sliding_attention",
]


def _tiny_args(**over) -> ModelArgs:
    kw = dict(
        model_type="beam",
        hidden_size=64,
        num_hidden_layers=4,
        intermediate_size=96,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        vocab_size=48,
        layer_types=list(_LAYER_TYPES),
        num_experts=16,
        num_experts_per_tok=3,
        moe_intermediate_size=24,
        first_k_dense_replace=1,
        n_group=4,
        topk_group=2,
        sliding_window=4,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        rope_parameters={"rope_theta": 10000.0, "rope_type": "default"},
        max_position_embeddings=4096,
        # A short length makes the query scale move within a few tokens.
        attn_temperature_scale=0.5,
        attn_temperature_length=2,
        routed_scaling_factor=2.5,
        norm_topk_prob=True,
        final_logit_softcapping=30.0,
    )
    kw.update(over)
    return ModelArgs(**kw)


def _randomize(module, seed: int):
    rng = np.random.default_rng(seed)
    upd = []
    for k, v in tree_flatten(module.parameters()):
        shape = v.shape
        if "norm" in k or k.endswith("layernorm.weight"):
            a = 1.0 + 0.3 * rng.standard_normal(shape)
        elif k.endswith(("e_score_correction_bias", "gate_bias")):
            a = 0.5 * rng.standard_normal(shape)
        else:
            a = rng.standard_normal(shape) / np.sqrt(shape[-1])
        upd.append((k, mx.array(a.astype(np.float32))))
    module.update(tree_unflatten(upd))
    mx.eval(module.parameters())
    return module


def _np(x) -> np.ndarray:
    return np.array(x.astype(mx.float32), dtype=np.float64)


def _rms(x, w=None, eps=1e-6):
    y = x / np.sqrt((x * x).mean(-1, keepdims=True) + eps)
    return y if w is None else y * w


def _rope_neox(x, pos, theta):
    # x: [T, H, D], split halves (NEOX rope).
    D = x.shape[-1]
    freqs = theta ** (-np.arange(0, D, 2) / D)
    ang = pos[:, None] * freqs[None, :]
    cos, sin = np.cos(ang)[:, None, :], np.sin(ang)[:, None, :]
    x0, x1 = x[..., : D // 2], x[..., D // 2:]
    return np.concatenate([x0 * cos - x1 * sin, x0 * sin + x1 * cos], axis=-1)


def _ref_route(gate, c, args, *, mask_value=-np.inf):
    """llama.cpp build_moe_ffn routing: sigmoid probs, biased selection
    within the best groups, weights from the unbiased probs."""
    scores = 1.0 / (1.0 + np.exp(-(c @ _np(gate.weight).T)))
    choice = scores + _np(gate.e_score_correction_bias)
    T, E = choice.shape
    G = args.n_group
    if G > 1:
        grouped = choice.reshape(T, G, E // G)
        top2 = np.sort(grouped, axis=-1)[..., -2:].sum(-1)
        drop = np.argsort(-top2, axis=-1, kind="stable")[:, args.topk_group:]
        for t in range(T):
            grouped[t, drop[t]] = mask_value
        choice = grouped.reshape(T, E)
    inds = np.argsort(-choice, axis=-1, kind="stable")[:, : args.num_experts_per_tok]
    w = np.take_along_axis(scores, inds, axis=-1)
    if args.norm_topk_prob:
        w = w / np.maximum(w.sum(-1, keepdims=True), 6.103515625e-5)
    if args.routed_scaling_factor not in (0.0, 1.0):
        w = w * args.routed_scaling_factor
    return inds, w


def _ref_moe(moe, c, args):
    inds, w = _ref_route(moe.gate, c, args)
    sw = moe.switch_mlp
    gw, uw, dw = (_np(sw.gate_proj.weight), _np(sw.up_proj.weight),
                  _np(sw.down_proj.weight))
    y = np.zeros_like(c)
    for t in range(c.shape[0]):
        for j, e in enumerate(inds[t]):
            g, u = c[t] @ gw[e].T, c[t] @ uw[e].T
            y[t] += w[t, j] * ((g / (1.0 + np.exp(-g)) * u) @ dw[e].T)
    return y, inds


def _ref_dense(mlp, c):
    g, u = c @ _np(mlp.gate_proj.weight).T, c @ _np(mlp.up_proj.weight).T
    return (g / (1.0 + np.exp(-g)) * u) @ _np(mlp.down_proj.weight).T


def _ref_layer(layer, h, args, il, pos0=0):
    """One decoder layer of llama.cpp's beam graph. ``h`` is [T, d]."""
    at = layer.self_attn
    T = h.shape[0]
    H, KV, D = args.num_attention_heads, args.num_key_value_heads, args.head_dim
    pos = np.arange(pos0, pos0 + T, dtype=np.float64)

    xn = _rms(h, _np(layer.input_layernorm.weight))
    q = (xn @ _np(at.q_proj.weight).T).reshape(T, H, D)
    k = (xn @ _np(at.k_proj.weight).T).reshape(T, KV, D)
    v = (xn @ _np(at.v_proj.weight).T).reshape(T, KV, D)
    q = _rms(q, _np(at.q_norm.weight))
    k = _rms(k, _np(at.k_norm.weight))
    if layer.use_sliding:
        q = _rope_neox(q, pos, args.rope_theta)
        k = _rope_neox(k, pos, args.rope_theta)
    else:
        scale = (1.0 + (pos + 1.0) / args.attn_temperature_length) ** \
            args.attn_temperature_scale
        q = q * scale[:, None, None]

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
    gate = (xn @ _np(at.gate_proj.weight).T).reshape(T, H, D)
    gate = 1.0 / (1.0 + np.exp(-(gate + _np(at.gate_bias)[None, :, None])))
    o = (o * gate).reshape(T, H * D) @ _np(at.o_proj.weight).T
    r = _rms(o, _np(layer.post_attn_norm.weight))

    ia = 2 * (il + 1)
    h = r / np.sqrt(ia) + h * np.sqrt((ia - 1) / ia)
    c = _rms(h, _np(layer.post_attention_layernorm.weight))
    if isinstance(layer.mlp, BeamMoE):
        y, _ = _ref_moe(layer.mlp, c, args)
    else:
        y = _ref_dense(layer.mlp, c)
    y = _rms(y, _np(layer.post_ffw_norm.weight))
    ib = ia + 1
    return y / np.sqrt(ib) + h * np.sqrt((ib - 1) / ib)


def _ref_model(model, tokens, args):
    h = _np(model.model.embed_tokens.weight)[tokens]
    for il, layer in enumerate(model.layers):
        h = _ref_layer(layer, h, args, il)
    h = _rms(h, _np(model.model.norm.weight))
    out = h @ _np(model.lm_head.weight).T
    cap = args.final_logit_softcapping
    return np.tanh(out / cap) * cap


# M5 f32 GEMM runs at TF32 precision by default (up to ~3e-3 of scale here);
# a wrong window, query scale or layer index moves the output 5e-2 or more.
def _rel(a, b):
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-12))


def test_full_model_matches_reference_across_the_window():
    """Nine tokens over a window of four: the sliding layers mask the far
    keys and rotate, the full layer sees every key and scales q."""
    args = _tiny_args()
    model = _randomize(Model(args), seed=3)
    tokens = np.array([5, 17, 2, 40, 33, 8, 21, 11, 30])
    got = _np(model(mx.array(tokens)[None])[0])
    want = _ref_model(model, tokens, args)
    assert _rel(got, want) < 1e-2
    # The reference is sensitive to the window and to the query scale.
    for layer in model.layers:
        layer.use_sliding = False
    assert _rel(_ref_model(model, tokens, args), want) > 5e-2
    for layer, kind in zip(model.layers, _LAYER_TYPES):
        layer.use_sliding = kind == "sliding_attention"
    flat = _ref_model(model, tokens, _tiny_args(attn_temperature_scale=1e-9))
    assert _rel(flat, want) > 5e-2


def test_layers_match_reference_at_depth():
    """The residual scale depends on the layer index. A layer run with
    another index's scales misses the reference."""
    args = _tiny_args()
    rng = np.random.default_rng(1)
    x = rng.standard_normal((6, args.hidden_size))
    for il in range(args.num_hidden_layers):
        layer = _randomize(DecoderLayer(args, il), seed=20 + il)
        mask = "causal"
        if layer.use_sliding:
            mask = mx.array(np.tril(np.ones((6, 6), bool))
                            & ~np.tril(np.ones((6, 6), bool), -args.sliding_window))
        q_scale = None if layer.use_sliding else nope_query_scale(
            0, 6, args.attn_temperature_scale, args.attn_temperature_length)
        got = _np(layer(mx.array(x.astype(np.float32))[None], mask, None,
                        q_scale)[0])
        want = _ref_layer(layer, x, args, il)
        assert _rel(got, want) < 5e-3, il
        other = _ref_layer(layer, x, args, il + 1)
        assert _rel(other, want) > 2e-2, il


def test_cached_decode_matches_full_forward():
    args = _tiny_args()
    model = _randomize(Model(args), seed=4)
    tokens = [5, 17, 2, 40, 33, 8, 21, 11]
    full = model(mx.array([tokens]))
    cache = model.make_cache()
    model(mx.array([tokens[:3]]), cache=cache)
    steps = [model(mx.array([[t]]), cache=cache) for t in tokens[3:]]
    mx.eval(full, steps)
    for n, s in enumerate(steps):
        np.testing.assert_allclose(
            _np(s[0, 0]), _np(full[0, 3 + n]), rtol=0, atol=1.5e-2
        )


def test_moe_bias_moves_selection_but_not_weights():
    """The bias picks the experts. The weights are the unbiased sigmoid
    scores of the picked experts, renormalized and scaled."""
    args = _tiny_args(n_group=1, topk_group=1)
    moe = _randomize(BeamMoE(args), seed=5)
    rng = np.random.default_rng(6)
    c = rng.standard_normal((4, args.hidden_size))
    cx = mx.array(c.astype(np.float32))[None]

    scores = 1.0 / (1.0 + np.exp(-(c @ _np(moe.gate.weight).T)))
    low = int(np.argmin(scores[0]))
    bias = np.zeros(args.num_experts)
    bias[low] = 10.0
    moe.gate.e_score_correction_bias = mx.array(bias.astype(np.float32))

    got = _np(moe(cx)[0])
    want, inds = _ref_moe(moe, c, args)
    assert (inds == low).any(axis=-1).all()
    assert _rel(got, want) < 1e-4

    # A shift shared by every expert keeps the selection, so the output holds.
    moe.gate.e_score_correction_bias = mx.array((bias + 3.0).astype(np.float32))
    np.testing.assert_allclose(_np(moe(cx)[0]), got, rtol=1e-5, atol=1e-6)


def test_grouped_selection_masks_dropped_groups_to_minus_inf():
    """With every biased score below zero, a 0.0 group mask would pick
    experts from the dropped groups. llama.cpp masks with -inf, so every
    pick comes from a kept group."""
    args = _tiny_args()
    gate = _randomize(BeamGate(args), seed=7)
    gate.e_score_correction_bias = mx.full((args.num_experts,), -5.0)
    rng = np.random.default_rng(8)
    c = rng.standard_normal((6, args.hidden_size))
    inds, w = gate(mx.array(c.astype(np.float32))[None])
    got = np.sort(np.array(inds[0]), axis=-1)

    want, want_w = _ref_route(gate, c, args)
    np.testing.assert_array_equal(got, np.sort(want, axis=-1))
    zero_mask, _ = _ref_route(gate, c, args, mask_value=0.0)
    assert (np.sort(zero_mask, axis=-1) != got).any()

    order = np.argsort(np.array(inds[0]), axis=-1)
    np.testing.assert_allclose(
        np.take_along_axis(_np(w[0]), order, axis=-1),
        np.take_along_axis(want_w, np.argsort(want, axis=-1), axis=-1),
        rtol=1e-3)


def test_top_k_is_read_from_the_gate_at_call_time():
    """The streaming top-k override writes ``top_k`` on the gate, and the
    forward reads it there."""
    args = _tiny_args()
    moe = _randomize(BeamMoE(args), seed=9)
    seen = []
    orig = moe.switch_mlp

    class _Spy(nn.Module):
        def __call__(self, x, inds):
            seen.append(inds.shape[-1])
            return orig(x, inds)

    moe.switch_mlp = _Spy()
    moe.gate.top_k = 1
    mx.eval(moe(mx.zeros((1, 2, args.hidden_size))))
    assert isinstance(BeamGate(args).top_k, int)
    assert seen == [1]


def test_route_replay_weights_match_the_gate():
    args = _tiny_args()
    gate = _randomize(BeamGate(args), seed=10)
    x = mx.array(np.random.default_rng(11).standard_normal(
        (1, 3, args.hidden_size)).astype(np.float32))
    inds, w = gate(x)
    np.testing.assert_allclose(_np(gate._kq_route_weights(x, inds)), _np(w),
                               rtol=1e-6)


def test_nope_query_scale_clamps_left_padding_rows():
    """A left-padded batch row starts below zero. Past -temp_length - 1 the
    power form has no real value, so positions clamp at 0."""
    per_row = nope_query_scale(mx.array([-2100, 7]), 3, 0.1, 2048)
    single = nope_query_scale(7, 3, 0.1, 2048)
    mx.eval(per_row, single)
    assert per_row.shape == (2, 1, 3, 1)
    assert np.isfinite(np.array(per_row)).all()
    np.testing.assert_allclose(np.array(per_row[1, 0]), np.array(single),
                               rtol=1e-6)
    np.testing.assert_allclose(np.array(per_row[0, 0, :, 0]),
                               (1 + 1 / 2048) ** 0.1, rtol=1e-6)


def test_left_padded_batch_matches_unpadded_rows():
    """A row left-padded by more than temp_length + 1 tokens matches the
    same prompt run alone, on prefill and on decode, with no NaN."""
    from gmlx.cache.kv_cache import BatchKVCache, BatchRotatingKVCache

    args = _tiny_args(attn_temperature_length=2048, attn_temperature_scale=0.1)
    model = _randomize(Model(args), seed=12)
    rng = np.random.default_rng(13)
    pad = 2100
    short = rng.integers(1, args.vocab_size, 5).tolist()
    long = rng.integers(1, args.vocab_size, 5 + pad).tolist()
    batch = mx.array([[0] * pad + short, long])
    cache = [
        BatchRotatingKVCache(args.sliding_window, [pad, 0])
        if layer.use_sliding else BatchKVCache([pad, 0])
        for layer in model.layers
    ]
    out = model(batch, cache=cache)
    nxt = mx.array([[3], [4]])
    step = model(nxt, cache=cache)
    mx.eval(out, step)
    assert np.isfinite(_np(out[0, pad:])).all()
    assert np.isfinite(_np(step)).all()

    alone = model.make_cache()
    ref = model(mx.array([short]), cache=alone)
    ref_step = model(mx.array([[3]]), cache=alone)
    np.testing.assert_allclose(_np(out[0, pad:]), _np(ref[0]),
                               rtol=0, atol=2e-2)
    np.testing.assert_allclose(_np(step[0]), _np(ref_step[0]),
                               rtol=0, atol=2e-2)


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


def test_prefill_decay_prices_no_score_tensor_at_head_dim_128(monkeypatch):
    """Depth decay sizes Beam's prefill chunks as fused attention: no score
    tensor at head dim 128, so a deep streamed prefill keeps wide chunks."""
    import gmlx.gen.prefill_decay as pd

    assert pd._SCORE_PROFILES["beam"] is pd.fused_attention_profile
    monkeypatch.setattr(pd, "cpu_default_device", lambda: False)
    model = Model(_tiny_args(head_dim=128))
    profile = pd.resolve_score_profile(model, model.make_cache())
    assert profile is not None and profile.fused_min_step == 9


@pytest.mark.parametrize("mode, norms", [("auto", True), ("fp32", True),
                                         ("activation", False)])
def test_cast_predicate_matches_the_loader_keep_list(monkeypatch, mode, norms):
    from gmlx.load.loader import fp32_keep_for

    monkeypatch.setenv("GMLX_NORM_DTYPE", mode)
    keep = fp32_keep_for("beam")
    model = Model(_tiny_args())
    pred = model.cast_predicate
    pinned = set()
    for path, _ in tree_flatten(model.parameters()):
        assert pred(path) == (not any(s in path for s in keep)), path
        if not pred(path):
            pinned.add(re.sub(r"^model\.layers\.\d+\.", "", path))
    want = {
        "mlp.gate.weight",
        "mlp.gate.e_score_correction_bias",
        "self_attn.gate_bias",
    }
    if norms:
        want |= {"input_layernorm.weight", "post_attn_norm.weight",
                 "post_attention_layernorm.weight", "post_ffw_norm.weight",
                 "self_attn.q_norm.weight", "self_attn.k_norm.weight",
                 "model.norm.weight"}
    assert pinned == want


def test_fp32_norm_weights_keep_a_bf16_residual():
    model = Model(_tiny_args())
    model.update(tree_map(lambda p: p.astype(mx.bfloat16), model.parameters()))
    norm = model.model.layers[0].post_attn_norm
    norm.weight = norm.weight.astype(mx.float32)
    x = mx.random.normal((1, 3, norm.weight.shape[0])).astype(mx.bfloat16)
    y = norm(x)
    assert y.dtype == mx.bfloat16
    want = mx.fast.rms_norm(x.astype(mx.float32), norm.weight, norm.eps)
    assert mx.allclose(y.astype(mx.float32), want, atol=2e-2).item()


def test_remap_covers_every_wire_tensor_onto_real_params():
    """Every tensor llama.cpp's beam loader creates maps onto a parameter
    of the model, and every parameter is loaded."""
    from gmlx.load.remap import RemapDecision, parse_gguf_name

    args = _tiny_args()
    params = {k for k, _ in tree_flatten(Model(args).parameters())}
    common = [
        "attn_norm.weight",
        "post_attention_norm.weight",
        "attn_q.weight",
        "attn_k.weight",
        "attn_v.weight",
        "attn_output.weight",
        "attn_q_norm.weight",
        "attn_k_norm.weight",
        "attn_gate.weight",
        "attn_gate.bias",
        "ffn_norm.weight",
        "post_ffw_norm.weight",
    ]
    dense = ["ffn_gate.weight", "ffn_up.weight", "ffn_down.weight"]
    moe = [
        "ffn_gate_inp.weight",
        "exp_probs_b.bias",
        "ffn_gate_exps.weight",
        "ffn_up_exps.weight",
        "ffn_down_exps.weight",
    ]
    names = ["token_embd.weight", "output_norm.weight", "output.weight"]
    for i in range(args.num_hidden_layers):
        ffn = dense if i < args.first_k_dense_replace else moe
        names += [f"blk.{i}.{t}" for t in common + ffn]
    mapped = set()
    for name in names:
        dec = parse_gguf_name("beam", name)
        assert dec.kind == RemapDecision.KIND_MAP, (name, dec.reason)
        assert dec.transform == "passthrough", name
        assert dec.hf_name in params, (name, dec.hf_name)
        mapped.add(dec.hf_name)
    assert mapped == params


def test_synth_config_builds_this_model():
    from gmlx.load.config_synth import synthesize_config
    from test_config_synth import _BEAM_SHAPES, _beam_meta

    c = synthesize_config(_beam_meta(), tensor_shapes=_BEAM_SHAPES)
    model = Model(ModelArgs.from_dict(c))
    assert [ly.use_sliding for ly in model.layers] == [True] * 5 + [False]
    assert model.layers[2].mlp.switch_mlp.gate_proj.weight.shape == (16, 24, 64)
    assert model.layers[1].mlp.gate_proj.weight.shape == (96, 64)
    assert model.layers[2].mlp.gate.n_group == 4


def _loader_shaped():
    from tune.tiny_train_archs import build

    model, config, _ = build("beam")
    return model, config


def test_residual_stream_stays_bf16_and_feeds_the_fused_moe(monkeypatch):
    """With the loader's dtypes (fp32 router, router bias and gate bias,
    bf16 elsewhere) every layer returns bf16, and the expert container
    receives a bf16 input that the fused decode gate accepts."""
    import gmlx.load.modules as modules
    from gmlx.load.dtypes import activation_dtype

    if activation_dtype() != mx.bfloat16:
        pytest.skip("the loader picks float16 activations on this device")
    model, _ = _loader_shaped()
    layer = model.layers[1]
    assert layer.mlp.gate.weight.dtype == mx.float32
    assert layer.mlp.gate.e_score_correction_bias.dtype == mx.float32
    assert layer.self_attn.gate_bias.dtype == mx.float32
    assert layer.post_attention_layernorm.weight.dtype == mx.bfloat16

    glu_cls = type(layer.mlp.switch_mlp)
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
    n_moe = sum(isinstance(ly.mlp, BeamMoE) for ly in model.layers)
    assert outs and set(outs) == {mx.bfloat16}
    assert len(verdicts) == n_moe
    assert all(v == (mx.bfloat16, True) for v in verdicts), verdicts
