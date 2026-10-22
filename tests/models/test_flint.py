#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Asher Feldman
"""Flint model against a naive float64 reference of llama.cpp's flint
graph, on tiny random weights. No GGUF and no download.

The reference covers the per-head attention sinks, NeoX rope on the
sliding layers and NoPE on the full ones, the window, dense and MoE
layers, sigmoid routing with an optional selection-only bias and the
ungated shared expert. The loader-shaped tests check the fp32 router pins
and the residual stream's dtype.
"""

from __future__ import annotations

import re

import numpy as np
import pytest

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten

from gmlx.models.flint.model import (
    DecoderLayer,
    FlintGate,
    FlintModel,
    FlintMoE,
    Model,
    ModelArgs,
)

_LAYER_TYPES = [
    "full_attention",
    "sliding_attention",
    "sliding_attention",
    "full_attention",
]


def _tiny_args(**over) -> ModelArgs:
    kw = dict(
        model_type="flint",
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
        shared_expert_intermediate_size=48,
        first_k_dense_replace=1,
        router_bias=False,
        sliding_window=4,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        rope_parameters={"rope_theta": 10000.0, "rope_type": "default"},
        max_position_embeddings=4096,
        routed_scaling_factor=0.0,
        norm_topk_prob=True,
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
        elif k.endswith("e_score_correction_bias"):
            a = 0.5 * rng.standard_normal(shape)
        elif k.endswith("sinks"):
            a = 1.5 * rng.standard_normal(shape)
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


def _ref_route(gate, c, args):
    """llama.cpp build_moe_ffn routing: sigmoid probs, selection on the
    probs plus the optional bias, weights from the unbiased probs."""
    scores = 1.0 / (1.0 + np.exp(-(c @ _np(gate.weight).T)))
    choice = scores
    if "e_score_correction_bias" in gate:
        choice = scores + _np(gate.e_score_correction_bias)
    inds = np.argsort(-choice, axis=-1, kind="stable")[:, : args.num_experts_per_tok]
    w = np.take_along_axis(scores, inds, axis=-1)
    if args.norm_topk_prob:
        w = w / np.maximum(w.sum(-1, keepdims=True), 6.103515625e-5)
    if args.routed_scaling_factor not in (0.0, 1.0):
        w = w * args.routed_scaling_factor
    return inds, w


def _ref_dense(mlp, c):
    g, u = c @ _np(mlp.gate_proj.weight).T, c @ _np(mlp.up_proj.weight).T
    return (g / (1.0 + np.exp(-g)) * u) @ _np(mlp.down_proj.weight).T


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
    return y + _ref_dense(moe.shared_experts, c), inds


def _ref_layer(layer, h, args, pos0=0):
    """One decoder layer of llama.cpp's flint graph. ``h`` is [T, d]."""
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

    i, j = np.arange(T)[:, None], np.arange(T)[None, :]
    allowed = j <= i
    if layer.use_sliding:
        # llama.cpp masks p1 - p0 >= n_swa: the query sees n_swa positions,
        # itself included.
        allowed &= (i - j) < args.sliding_window
    sinks = _np(at.sinks)
    o = np.zeros((T, H, D))
    for hh in range(H):
        kv = hh // (H // KV)
        s = (q[:, hh] @ k[:, kv].T) * D**-0.5
        s = np.where(allowed, s, -np.inf)
        # The sink is one more logit in the normalizer with no value row.
        m = np.maximum(s.max(-1, keepdims=True), sinks[hh])
        p = np.exp(s - m)
        p /= p.sum(-1, keepdims=True) + np.exp(sinks[hh] - m)
        o[:, hh] = p @ v[:, kv]
    o = o.reshape(T, H * D) @ _np(at.o_proj.weight).T
    h = h + o
    c = _rms(h, _np(layer.post_attention_layernorm.weight))
    if isinstance(layer.mlp, FlintMoE):
        y, _ = _ref_moe(layer.mlp, c, args)
    else:
        y = _ref_dense(layer.mlp, c)
    return h + y


def _ref_model(model, tokens, args):
    h = _np(model.model.embed_tokens.weight)[tokens]
    for layer in model.layers:
        h = _ref_layer(layer, h, args)
    h = _rms(h, _np(model.model.norm.weight))
    return h @ _np(model.lm_head.weight).T


# M5 f32 GEMM runs at TF32 precision by default (up to ~3e-3 of scale here);
# a wrong window, rope layer or sink moves the output 5e-2 or more.
def _rel(a, b):
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-12))


def test_full_model_matches_reference_across_the_window():
    """Nine tokens over a window of four: the sliding layers mask the far
    keys and rotate, the full layers see every key unrotated."""
    args = _tiny_args()
    model = _randomize(Model(args), seed=3)
    tokens = np.array([5, 17, 2, 40, 33, 8, 21, 11, 30])
    got = _np(model(mx.array(tokens)[None])[0])
    want = _ref_model(model, tokens, args)
    assert _rel(got, want) < 1e-2
    # The reference is sensitive to the window and to which layers rotate.
    wide = _ref_model(model, tokens, _tiny_args(sliding_window=64))
    assert _rel(wide, want) > 5e-2
    for layer in model.layers:
        layer.use_sliding = not layer.use_sliding
    assert _rel(_ref_model(model, tokens, args), want) > 5e-2


def test_sinks_change_the_output():
    """Each head's sink takes softmax mass from every key. Zeroed sinks
    still add one exp(0) to the normalizer, so the reference differs from
    a plain softmax too."""
    args = _tiny_args()
    model = _randomize(Model(args), seed=14)
    tokens = np.array([3, 9, 27, 1, 44, 12])
    want = _ref_model(model, tokens, args)
    for layer in model.layers:
        layer.self_attn.sinks = mx.full(layer.self_attn.sinks.shape, -30.0)
    no_sink = _np(model(mx.array(tokens)[None])[0])
    assert _rel(no_sink, _ref_model(model, tokens, args)) < 1e-2
    assert _rel(no_sink, want) > 5e-2


def test_layers_match_reference():
    args = _tiny_args()
    rng = np.random.default_rng(1)
    x = rng.standard_normal((6, args.hidden_size))
    for il in range(args.num_hidden_layers):
        layer = _randomize(DecoderLayer(args, il), seed=20 + il)
        mask = "causal"
        if layer.use_sliding:
            mask = mx.array(np.tril(np.ones((6, 6), bool))
                            & ~np.tril(np.ones((6, 6), bool), -args.sliding_window))
        got = _np(layer(mx.array(x.astype(np.float32))[None], mask, None)[0])
        assert _rel(got, _ref_layer(layer, x, args)) < 5e-3, il


def test_cached_decode_matches_full_forward_past_the_window():
    args = _tiny_args()
    model = _randomize(Model(args), seed=4)
    tokens = [5, 17, 2, 40, 33, 8, 21, 11, 30, 7]
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
    scores of the picked experts, renormalized."""
    args = _tiny_args(router_bias=True)
    moe = _randomize(FlintMoE(args), seed=5)
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


def test_shared_expert_is_added_ungated():
    args = _tiny_args()
    moe = _randomize(FlintMoE(args), seed=15)
    c = np.random.default_rng(16).standard_normal((3, args.hidden_size))
    cx = mx.array(c.astype(np.float32))[None]
    got = _np(moe(cx)[0])
    want, _ = _ref_moe(moe, c, args)
    assert _rel(got, want) < 1e-4
    routed = want - _ref_dense(moe.shared_experts, c)
    assert _rel(routed, want) > 5e-2


def test_routing_scale_applies_unless_zero_or_one():
    args = _tiny_args(routed_scaling_factor=2.5)
    gate = _randomize(FlintGate(args), seed=17)
    c = np.random.default_rng(18).standard_normal((2, args.hidden_size))
    _, w = gate(mx.array(c.astype(np.float32))[None])
    _, want = _ref_route(gate, c, args)
    np.testing.assert_allclose(np.sort(_np(w[0]), -1), np.sort(want, -1),
                               rtol=1e-4)
    assert np.allclose(_np(w[0]).sum(-1), 2.5, rtol=1e-4)


def test_top_k_is_read_from_the_gate_at_call_time():
    """The streaming top-k override writes ``top_k`` on the gate, and the
    forward reads it there."""
    args = _tiny_args()
    moe = _randomize(FlintMoE(args), seed=9)
    seen = []
    orig = moe.switch_mlp

    class _Spy(nn.Module):
        def __call__(self, x, inds):
            seen.append(inds.shape[-1])
            return orig(x, inds)

    moe.switch_mlp = _Spy()
    moe.gate.top_k = 1
    mx.eval(moe(mx.zeros((1, 2, args.hidden_size))))
    assert isinstance(FlintGate(args).top_k, int)
    assert seen == [1]


def test_route_replay_weights_match_the_gate():
    args = _tiny_args(router_bias=True)
    gate = _randomize(FlintGate(args), seed=10)
    x = mx.array(np.random.default_rng(11).standard_normal(
        (1, 3, args.hidden_size)).astype(np.float32))
    inds, w = gate(x)
    np.testing.assert_allclose(_np(gate._kq_route_weights(x, inds)), _np(w),
                               rtol=1e-6)


def test_left_padded_batch_matches_unpadded_rows():
    """A left-padded row past the window matches the same prompt run alone,
    on prefill and on decode."""
    from gmlx.cache.kv_cache import BatchKVCache, BatchRotatingKVCache

    args = _tiny_args()
    model = _randomize(Model(args), seed=12)
    rng = np.random.default_rng(13)
    pad = 7
    short = rng.integers(1, args.vocab_size, 5).tolist()
    long = rng.integers(1, args.vocab_size, 5 + pad).tolist()
    batch = mx.array([[0] * pad + short, long])
    cache = [
        BatchRotatingKVCache(args.sliding_window, [pad, 0])
        if layer.use_sliding else BatchKVCache([pad, 0])
        for layer in model.layers
    ]
    out = model(batch, cache=cache)
    step = model(mx.array([[3], [4]]), cache=cache)
    mx.eval(out, step)

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


def test_sinks_decline_quantized_kv():
    from gmlx.cache.kv_policy import attention_kv_decline, attention_sinks
    from gmlx.serve.capacity import _header_attention_decline

    model = Model(_tiny_args())
    assert attention_sinks(model)
    assert "sinks" in attention_kv_decline(model)
    assert _header_attention_decline({"model_type": "flint"})


def test_prefill_decay_prices_no_score_tensor_at_head_dim_128(monkeypatch):
    """Depth decay sizes Flint's prefill chunks as fused attention: no
    score tensor at head dim 128."""
    import gmlx.gen.prefill_decay as pd

    assert pd._SCORE_PROFILES["flint"] is pd.fused_attention_profile
    monkeypatch.setattr(pd, "cpu_default_device", lambda: False)
    model = Model(_tiny_args(head_dim=128))
    profile = pd.resolve_score_profile(model, model.make_cache())
    assert profile is not None and profile.fused_min_step == 9


@pytest.mark.parametrize("mode, norms", [("auto", False), ("fp32", True),
                                         ("activation", False)])
def test_cast_predicate_matches_the_loader_keep_list(monkeypatch, mode, norms):
    from gmlx.load.loader import fp32_keep_for

    monkeypatch.setenv("GMLX_NORM_DTYPE", mode)
    keep = fp32_keep_for("flint")
    model = Model(_tiny_args(router_bias=True))
    pred = model.cast_predicate
    pinned = set()
    for path, _ in tree_flatten(model.parameters()):
        assert pred(path) == (not any(s in path for s in keep)), path
        if not pred(path):
            pinned.add(re.sub(r"^model\.layers\.\d+\.", "", path))
    want = {"mlp.gate.weight", "mlp.gate.e_score_correction_bias"}
    if norms:
        want |= {"input_layernorm.weight", "post_attention_layernorm.weight",
                 "self_attn.q_norm.weight", "self_attn.k_norm.weight",
                 "model.norm.weight"}
    assert pinned == want


def test_remap_covers_every_wire_tensor_onto_real_params():
    """Every tensor llama.cpp's flint loader creates maps onto a parameter
    of the model, and every parameter is loaded."""
    from gmlx.load.remap import RemapDecision, parse_gguf_name

    args = _tiny_args(router_bias=True)
    params = {k for k, _ in tree_flatten(Model(args).parameters())}
    common = [
        "attn_norm.weight",
        "attn_q.weight",
        "attn_k.weight",
        "attn_v.weight",
        "attn_output.weight",
        "attn_q_norm.weight",
        "attn_k_norm.weight",
        "attn_sinks.weight",
        "ffn_norm.weight",
    ]
    dense = ["ffn_gate.weight", "ffn_up.weight", "ffn_down.weight"]
    moe = [
        "ffn_gate_inp.weight",
        "exp_probs_b.bias",
        "ffn_gate_exps.weight",
        "ffn_up_exps.weight",
        "ffn_down_exps.weight",
        "ffn_gate_shexp.weight",
        "ffn_up_shexp.weight",
        "ffn_down_shexp.weight",
    ]
    names = ["token_embd.weight", "output_norm.weight", "output.weight"]
    for i in range(args.num_hidden_layers):
        ffn = dense if i < args.first_k_dense_replace else moe
        names += [f"blk.{i}.{t}" for t in common + ffn]
    mapped = set()
    for name in names:
        dec = parse_gguf_name("flint", name)
        assert dec.kind == RemapDecision.KIND_MAP, (name, dec.reason)
        assert dec.transform == "passthrough", name
        assert dec.hf_name in params, (name, dec.hf_name)
        mapped.add(dec.hf_name)
    assert mapped == params


def test_synth_config_builds_this_model():
    from gmlx.load.config_synth import synthesize_config
    from test_config_synth import _FLINT_SHAPES, _flint_meta

    c = synthesize_config(_flint_meta(), tensor_shapes=_FLINT_SHAPES)
    model = Model(ModelArgs.from_dict(c))
    assert [ly.use_sliding for ly in model.layers] == [
        False, False, True, True, True, False]
    assert model.layers[2].mlp.switch_mlp.gate_proj.weight.shape == (16, 24, 64)
    assert model.layers[1].mlp.gate_proj.weight.shape == (96, 64)
    assert model.layers[2].mlp.shared_experts.down_proj.weight.shape == (64, 48)


def _loader_shaped():
    from tune.tiny_train_archs import build

    model, config, _ = build("flint")
    return model, config


@pytest.mark.parametrize("act", ["bfloat16", "float16"])
def test_residual_stream_dtype(monkeypatch, act):
    """With the loader's dtypes (fp32 router, everything else in the
    activation dtype) a bfloat16 graph keeps every layer in bfloat16. A
    float16 graph holds the residual stream in float32, runs the MoE in
    bfloat16 and returns float16 logits."""
    monkeypatch.setenv("GMLX_ACTIVATION_DTYPE", act)
    model, _ = _loader_shaped()
    dt = getattr(mx, act)
    layer = model.layers[1]
    assert layer.mlp.gate.weight.dtype == mx.float32
    assert layer.self_attn.sinks.dtype == dt
    outs, moe_ins = [], []
    call, moe_call = DecoderLayer.__call__, FlintMoE.__call__

    def spy(self, *a, **k):
        out = call(self, *a, **k)
        outs.append(out.dtype)
        return out

    def moe_spy(self, x):
        moe_ins.append(x.dtype)
        return moe_call(self, x)

    DecoderLayer.__call__, FlintMoE.__call__ = spy, moe_spy
    try:
        cache = model.make_cache()
        logits = model(mx.array([[1, 2, 3]]), cache=cache)
        mx.eval(logits, model(mx.array([[4]]), cache=cache))
    finally:
        DecoderLayer.__call__, FlintMoE.__call__ = call, moe_call
    assert logits.dtype == dt
    wide = dt == mx.float16
    assert outs and set(outs) == {mx.float32 if wide else dt}
    assert moe_ins and set(moe_ins) == {mx.bfloat16}


def test_float16_graph_survives_swiglu_past_float16_range(monkeypatch):
    """Routed SwiGLU outputs past 65504, as the release's last layers give
    on number tokens, stay finite in a float16 graph and match the
    reference. A float16 residual stream turns them into inf."""
    monkeypatch.setenv("GMLX_ACTIVATION_DTYPE", "float16")
    args = _tiny_args()
    model = _randomize(Model(args), seed=11)
    upd = []
    for k, v in tree_flatten(model.parameters()):
        if ".switch_mlp." in k:
            v = v * (1e-4 if "down_proj" in k else 150.0)
        upd.append((k, v))
    model.update(tree_unflatten(upd))
    tokens = np.array([3, 17, 5, 29, 8, 41])
    want = _ref_model(model, tokens, args)

    keep = model.cast_predicate
    model.update(tree_unflatten([
        (k, v.astype(mx.float16) if keep(k) else v)
        for k, v in tree_flatten(model.parameters())]))
    got = model(mx.array(tokens)[None])
    assert got.dtype == mx.float16
    # The CPU gather_mm takes float32 only, so the float16 arm runs on the GPU.
    if mx.default_device().type == mx.DeviceType.gpu:
        with monkeypatch.context() as m:
            m.setattr(FlintModel, "_WIDE_FROM", mx.bfloat16)
            unwide = model(mx.array(tokens)[None])
            mx.eval(unwide)
            assert not np.isfinite(_np(unwide)).all()
    got = _np(got[0])
    assert np.isfinite(got).all()
    assert _rel(got, want) < 3e-2
