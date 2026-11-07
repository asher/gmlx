#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Asher Feldman
"""Ripple model against a naive float64 reference of llama.cpp's ricursive
graph, on tiny random weights. No GGUF and no download.

The reference covers NeoX rope on the sliding layers and NoPE on the full
ones, the window, the embedding scale, the value residual from the scaled
embeddings, the sigmoid output gate, the four norms and the clamped
SwiGLU. The loader-shaped tests check the fp32 pins and the remap.
"""

from __future__ import annotations

import numpy as np
import pytest

import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten

from gmlx.models.ricursive.model import DecoderLayer, MLP, Model, ModelArgs

_LAYER_TYPES = [
    "sliding_attention",
    "sliding_attention",
    "full_attention",
    "sliding_attention",
]


def _tiny_args(**over) -> ModelArgs:
    kw = dict(
        model_type="ricursive",
        hidden_size=64,
        num_hidden_layers=4,
        intermediate_size=96,
        num_attention_heads=6,
        num_key_value_heads=2,
        head_dim=16,
        vocab_size=48,
        layer_types=list(_LAYER_TYPES),
        sliding_window=4,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        rope_parameters={"rope_theta": 10000.0, "rope_type": "default"},
        max_position_embeddings=4096,
        embedding_scale=2.5,
        swiglu_limits=[0.0, 1.5, 0.0, 2.0],
        use_gated_attention=True,
        use_valres=True,
    )
    kw.update(over)
    return ModelArgs(**kw)


def _randomize(module, seed: int):
    rng = np.random.default_rng(seed)
    upd = []
    for k, v in tree_flatten(module.parameters()):
        shape = v.shape
        if "norm" in k:
            a = 1.0 + 0.3 * rng.standard_normal(shape)
        elif k.endswith("vres_lambda"):
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


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _rope_neox(x, pos, theta):
    # x: [T, H, D], split halves (NEOX rope).
    D = x.shape[-1]
    freqs = theta ** (-np.arange(0, D, 2) / D)
    ang = pos[:, None] * freqs[None, :]
    cos, sin = np.cos(ang)[:, None, :], np.sin(ang)[:, None, :]
    x0, x1 = x[..., : D // 2], x[..., D // 2:]
    return np.concatenate([x0 * cos - x1 * sin, x0 * sin + x1 * cos], axis=-1)


def _ref_mlp(mlp, c):
    """ggml_swiglu_clamp: silu(min(gate, L)) * clamp(up, -L, L); L = 0 is
    ggml_swiglu_split."""
    g, u = c @ _np(mlp.gate_proj.weight).T, c @ _np(mlp.up_proj.weight).T
    lim = mlp.swiglu_limit
    if lim > 0:
        g, u = np.minimum(g, lim), np.clip(u, -lim, lim)
    return (g * _sigmoid(g) * u) @ _np(mlp.down_proj.weight).T


def _ref_layer(layer, h, x0, args, pos0=0):
    """One decoder layer of llama.cpp's ricursive graph. ``h`` is [T, d]."""
    at = layer.self_attn
    T = h.shape[0]
    H, KV, D = args.num_attention_heads, args.num_key_value_heads, args.head_dim
    pos = np.arange(pos0, pos0 + T, dtype=np.float64)

    xn = _rms(h, _np(layer.input_layernorm.weight))
    v_in = xn
    if args.use_valres:
        v_in = x0 + (xn - x0) * _sigmoid(_np(at.vres_lambda))
    q = (xn @ _np(at.q_proj.weight).T).reshape(T, H, D)
    k = (xn @ _np(at.k_proj.weight).T).reshape(T, KV, D)
    v = (v_in @ _np(at.v_proj.weight).T).reshape(T, KV, D)
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
    o = np.zeros((T, H, D))
    for hh in range(H):
        kv = hh // (H // KV)
        s = (q[:, hh] @ k[:, kv].T) * D**-0.5
        s = np.where(allowed, s, -np.inf)
        p = np.exp(s - s.max(-1, keepdims=True))
        p /= p.sum(-1, keepdims=True)
        o[:, hh] = p @ v[:, kv]
    o = o.reshape(T, H * D)
    if args.use_gated_attention:
        o = o * _sigmoid(xn @ _np(at.gate_proj.weight).T)
    o = o @ _np(at.o_proj.weight).T
    h = h + _rms(o, _np(layer.post_attention_layernorm.weight))
    c = _rms(h, _np(layer.pre_feedforward_layernorm.weight))
    y = _ref_mlp(layer.mlp, c)
    return h + _rms(y, _np(layer.post_feedforward_layernorm.weight))


def _ref_model(model, tokens, args):
    h = _np(model.model.embed_tokens.weight)[tokens] * args.embedding_scale
    x0 = h
    for layer in model.layers:
        h = _ref_layer(layer, h, x0, args)
    h = _rms(h, _np(model.model.norm.weight))
    return h @ _np(model.lm_head.weight).T


# M5 f32 GEMM runs at TF32 precision by default (up to ~3e-3 of scale here);
# a wrong window, rope layer, gate or residual moves the output 5e-2 or more.
def _rel(a, b):
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-12))


def test_full_model_matches_reference_across_the_window():
    """Nine tokens over a window of four: the sliding layers mask the far
    keys and rotate, the full layer sees every key unrotated."""
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


@pytest.mark.parametrize("over", [
    {"use_valres": False},
    {"use_gated_attention": False},
    {"embedding_scale": 1.0},
    {"swiglu_limits": None},
])
def test_each_mechanism_changes_the_output(over):
    """With one mechanism off, the model still matches the reference, and
    the output differs from the model with it on."""
    tokens = np.array([3, 9, 27, 1, 44, 12, 6])
    on_args = _tiny_args()
    on = _randomize(Model(on_args), seed=14)
    off_args = _tiny_args(**over)
    off = _randomize(Model(off_args), seed=14)
    got = _np(off(mx.array(tokens)[None])[0])
    want_off = _ref_model(off, tokens, off_args)
    assert _rel(got, want_off) < 1e-2
    assert _rel(want_off, _ref_model(on, tokens, on_args)) > 5e-2


def test_layers_match_reference():
    args = _tiny_args()
    rng = np.random.default_rng(1)
    x = rng.standard_normal((6, args.hidden_size))
    x0 = rng.standard_normal((6, args.hidden_size))
    for il in range(args.num_hidden_layers):
        layer = _randomize(DecoderLayer(args, il), seed=20 + il)
        mask = "causal"
        if layer.use_sliding:
            mask = mx.array(np.tril(np.ones((6, 6), bool))
                            & ~np.tril(np.ones((6, 6), bool), -args.sliding_window))
        got = _np(layer(mx.array(x.astype(np.float32))[None],
                        mx.array(x0.astype(np.float32))[None], mask, None)[0])
        assert _rel(got, _ref_layer(layer, x, x0, args)) < 5e-3, il


def test_clamp_engages_on_large_activations():
    """Activations past the limit are clamped: gate from above only, up on
    both sides. Unclamped, the output moves well away."""
    args = _tiny_args()
    mlp = _randomize(MLP(args.hidden_size, args.intermediate_size, 1.0), seed=7)
    c = 4.0 * np.random.default_rng(8).standard_normal((5, args.hidden_size))
    got = _np(mlp(mx.array(c.astype(np.float32))[None])[0])
    assert _rel(got, _ref_mlp(mlp, c)) < 5e-3
    mlp.swiglu_limit = 0.0
    assert _rel(_ref_mlp(mlp, c), got) > 5e-2


def test_given_embeddings_are_scaled_too():
    """Embeddings handed in are scaled and feed the value residual, as
    token ids do."""
    args = _tiny_args()
    model = _randomize(Model(args), seed=9)
    tokens = mx.array([[4, 8, 15, 16, 23]])
    emb = model.model.embed_tokens(tokens)
    np.testing.assert_allclose(
        _np(model(tokens, input_embeddings=emb)), _np(model(tokens)),
        rtol=0, atol=1e-5)


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


def test_prefill_decay_prices_no_score_tensor_at_head_dim_128(monkeypatch):
    """Depth decay sizes Ripple's prefill chunks as fused attention: no
    score tensor at head dim 128."""
    import gmlx.gen.prefill_decay as pd

    assert pd._SCORE_PROFILES["ricursive"] is pd.fused_attention_profile
    monkeypatch.setattr(pd, "cpu_default_device", lambda: False)
    model = Model(_tiny_args(head_dim=128))
    profile = pd.resolve_score_profile(model, model.make_cache())
    assert profile is not None and profile.fused_min_step == 9


def test_cast_predicate_matches_the_loader_keep_list():
    from gmlx.load.loader import fp32_keep_for

    keep = fp32_keep_for("ricursive")
    model = Model(_tiny_args())
    pred = model.cast_predicate
    pinned = set()
    for path, _ in tree_flatten(model.parameters()):
        assert pred(path) == (not any(s in path for s in keep)), path
        if not pred(path):
            pinned.add(path.rsplit(".", 1)[-1])
    assert pinned == {"vres_lambda"}


_WIRE = [
    "attn_norm.weight",
    "attn_q.weight",
    "attn_k.weight",
    "attn_v.weight",
    "attn_output.weight",
    "attn_q_norm.weight",
    "attn_k_norm.weight",
    "attn_gate.weight",
    "attn_vres_lambda.weight",
    "post_attention_norm.weight",
    "ffn_norm.weight",
    "ffn_gate.weight",
    "ffn_up.weight",
    "ffn_down.weight",
    "post_ffw_norm.weight",
]


def test_remap_covers_every_wire_tensor_onto_real_params():
    """Every tensor llama.cpp's ricursive loader creates maps onto a
    parameter of the model, and every parameter is loaded."""
    from gmlx.load.remap import RemapDecision, parse_gguf_name

    args = _tiny_args()
    params = {k for k, _ in tree_flatten(Model(args).parameters())}
    names = ["token_embd.weight", "output_norm.weight", "output.weight"]
    for i in range(args.num_hidden_layers):
        names += [f"blk.{i}.{t}" for t in _WIRE]
    mapped = set()
    for name in names:
        dec = parse_gguf_name("ricursive", name)
        assert dec.kind == RemapDecision.KIND_MAP, (name, dec.reason)
        assert dec.transform == "passthrough", name
        assert dec.hf_name in params, (name, dec.hf_name)
        mapped.add(dec.hf_name)
    assert mapped == params


def test_synth_config_builds_this_model():
    from gmlx.load.config_synth import synthesize_config
    from test_config_synth import _RICURSIVE_SHAPES, _ricursive_meta

    c = synthesize_config(_ricursive_meta(), tensor_shapes=_RICURSIVE_SHAPES)
    model = Model(ModelArgs.from_dict(c))
    assert [ly.use_sliding for ly in model.layers] == [
        True, True, True, True, False]
    attn = model.layers[0].self_attn
    assert attn.gate_proj.weight.shape == (8 * 16, 64)
    assert attn.vres_lambda.shape == (1,)
    assert [ly.mlp.swiglu_limit for ly in model.layers] == [20.0] * 5
