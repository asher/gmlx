#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Asher Feldman
"""LFM3 model against a naive float64 reference of llama.cpp's lfm3 graph,
on tiny random weights. No GGUF and no download.

The reference covers the gated short convolution, the sliding and full
attention layers with their two rope bases and the full layers' partial
rotation, the per-head output gate and the sandwich norms. The other
tests check the caches, a left-padded batch, the rope_freqs factors and
the loader's float32 pins.
"""

from __future__ import annotations

import re

import numpy as np
import pytest

import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten

from gmlx.models.lfm3.model import (
    CONV,
    FULL,
    SLIDING,
    DecoderLayer,
    Model,
    ModelArgs,
    ShortConv,
    rotated_dims_from_factors,
)

# The shipped order: conv, conv, full, conv, sliding, then a second round.
_LAYER_TYPES = [CONV, CONV, FULL, CONV, SLIDING, CONV, FULL]
_ROTATED = 8


def _tiny_args(**over) -> ModelArgs:
    kw = dict(
        model_type="lfm3",
        hidden_size=64,
        num_hidden_layers=len(_LAYER_TYPES),
        intermediate_size=96,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        vocab_size=48,
        layer_types=list(_LAYER_TYPES),
        sliding_window=4,
        conv_L_cache=7,
        rms_norm_eps=1e-5,
        # A small base, so the pairs past the rotated width would turn
        # visibly over a few positions if the model rotated them.
        rope_theta=100.0,
        rope_theta_swa=10000.0,
        max_position_embeddings=4096,
    )
    kw.update(over)
    return ModelArgs(**kw)


def _factors(rotated: int = _ROTATED, head_dim: int = 16) -> mx.array:
    n = rotated // 2
    return mx.array([1.0] * n + [1e30] * (head_dim // 2 - n))


def _randomize(module, seed: int):
    rng = np.random.default_rng(seed)
    upd = []
    for k, v in tree_flatten(module.parameters()):
        shape = v.shape
        if "norm" in k:
            a = 1.0 + 0.3 * rng.standard_normal(shape)
        else:
            a = rng.standard_normal(shape) / np.sqrt(shape[-1])
        upd.append((k, mx.array(a.astype(np.float32))))
    module.update(tree_unflatten(upd))
    mx.eval(module.parameters())
    return module


def _model(seed: int, **over) -> Model:
    model = _randomize(Model(_tiny_args(**over)), seed)
    model.set_full_rope_factors(_factors())
    return model


def _np(x) -> np.ndarray:
    return np.array(x.astype(mx.float32), dtype=np.float64)


def _rms(x, w, eps=1e-5):
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * w


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _rope_neox(x, pos, theta, rotated):
    """NeoX rope over [T, H, D]: pair i is dims i and i + D/2. Only the
    first ``rotated // 2`` pairs turn, at the full width's frequencies."""
    D = x.shape[-1]
    freqs = theta ** (-np.arange(0, D, 2) / D)
    freqs[rotated // 2:] = 0.0
    ang = pos[:, None] * freqs[None, :]
    cos, sin = np.cos(ang)[:, None, :], np.sin(ang)[:, None, :]
    x0, x1 = x[..., : D // 2], x[..., D // 2:]
    return np.concatenate([x0 * cos - x1 * sin, x0 * sin + x1 * cos], axis=-1)


def _ref_conv(conv, xn):
    """llama.cpp's build_lfm_shortconv from a zero state. ``xn`` is [T, d]."""
    T, d = xn.shape
    bcx = xn @ _np(conv.in_proj.weight).T
    b, c, x = bcx[:, :d], bcx[:, d:2 * d], bcx[:, 2 * d:]
    w = _np(conv.kernel)
    width = w.shape[1]
    win = np.concatenate([np.zeros((width - 1, d)), b * x])
    out = np.stack([(win[t:t + width] * w.T).sum(0) for t in range(T)])
    return (c * out) @ _np(conv.out_proj.weight).T


def _ref_attn(layer, xn, args, rotated):
    at = layer.self_attn
    T = xn.shape[0]
    H, KV, D = args.num_attention_heads, args.num_key_value_heads, args.head_dim
    pos = np.arange(T, dtype=np.float64)
    q = (xn @ _np(at.q_proj.weight).T).reshape(T, H, D)
    k = (xn @ _np(at.k_proj.weight).T).reshape(T, KV, D)
    v = (xn @ _np(at.v_proj.weight).T).reshape(T, KV, D)
    q = _rms(q, _np(at.q_norm.weight))
    k = _rms(k, _np(at.k_norm.weight))
    if layer.use_sliding:
        theta, turn = args.rope_theta_swa, D
    else:
        theta, turn = args.rope_theta, rotated
    q = _rope_neox(q, pos, theta, turn)
    k = _rope_neox(k, pos, theta, turn)

    i, j = np.arange(T)[:, None], np.arange(T)[None, :]
    allowed = j <= i
    if layer.use_sliding:
        # llama.cpp masks p1 - p0 >= n_swa: the query sees n_swa positions,
        # itself included.
        allowed &= (i - j) < args.sliding_window
    gate = _sigmoid(xn @ _np(at.gate_proj.weight).T)
    o = np.zeros((T, H, D))
    for hh in range(H):
        kv = hh // (H // KV)
        s = (q[:, hh] @ k[:, kv].T) * D**-0.5
        s = np.where(allowed, s, -np.inf)
        p = np.exp(s - s.max(-1, keepdims=True))
        p /= p.sum(-1, keepdims=True)
        o[:, hh] = (p @ v[:, kv]) * gate[:, hh:hh + 1]
    return o.reshape(T, H * D) @ _np(at.o_proj.weight).T


def _ref_layer(layer, h, args, rotated=_ROTATED):
    """One layer of llama.cpp's lfm3 graph. ``h`` is [T, d]."""
    xn = _rms(h, _np(layer.input_layernorm.weight))
    if layer.is_attention_layer:
        r = _ref_attn(layer, xn, args, rotated)
    else:
        r = _ref_conv(layer.conv, xn)
    if layer.use_sandwich_norm:
        r = _rms(r, _np(layer.post_attn_norm.weight))
    h = h + r
    c = _rms(h, _np(layer.post_attention_layernorm.weight))
    mlp = layer.mlp
    g = c @ _np(mlp.gate_proj.weight).T
    y = (g * _sigmoid(g) * (c @ _np(mlp.up_proj.weight).T)) \
        @ _np(mlp.down_proj.weight).T
    if layer.use_sandwich_norm:
        y = _rms(y, _np(layer.post_ffw_norm.weight))
    return h + y


def _ref_model(model, tokens, args, rotated=_ROTATED):
    embed = _np(model.model.embed_tokens.weight)
    h = embed[tokens]
    for layer in model.layers:
        h = _ref_layer(layer, h, args, rotated)
    return _rms(h, _np(model.model.norm.weight)) @ embed.T


# M5 f32 GEMM runs at TF32 precision by default (up to ~3e-3 of scale here);
# a wrong window, rope base, rotated width or conv tap order moves the
# output 5e-2 or more.
def _rel(a, b):
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-12))


_TOKENS = np.array([5, 17, 2, 40, 33, 8, 21, 11, 30, 7, 19])


def test_full_model_matches_reference():
    """Eleven tokens over a window of four and a conv width of seven."""
    model = _model(seed=3)
    args = model.args
    got = _np(model(mx.array(_TOKENS)[None])[0])
    want = _ref_model(model, _TOKENS, args)
    assert _rel(got, want) < 1e-2
    # The reference is sensitive to the window, the rotated width and the
    # two rope bases.
    assert _rel(_ref_model(model, _TOKENS, _tiny_args(sliding_window=64)),
                want) > 5e-2
    assert _rel(_ref_model(model, _TOKENS, args, rotated=16), want) > 5e-2
    assert _rel(_ref_model(model, _TOKENS, _tiny_args(rope_theta_swa=100.0)),
                want) > 5e-2


def test_layers_match_reference():
    args = _tiny_args()
    rng = np.random.default_rng(1)
    T = 9
    x = rng.standard_normal((T, args.hidden_size))
    for il, kind in enumerate(_LAYER_TYPES):
        layer = _randomize(DecoderLayer(args, il), seed=20 + il)
        mask = None
        if kind == FULL:
            mask = "causal"
            from gmlx._vendor.mlx_lm.models.rope_utils import ProportionalRoPE

            layer.self_attn.rope = ProportionalRoPE(
                args.head_dim, _ROTATED, base=args.rope_theta)
        elif kind == SLIDING:
            tril = np.tril(np.ones((T, T), bool))
            mask = mx.array(tril & ~np.tril(np.ones((T, T), bool),
                                            -args.sliding_window))
        got = _np(layer(mx.array(x.astype(np.float32))[None], mask, None)[0])
        assert _rel(got, _ref_layer(layer, x, args)) < 5e-3, (il, kind)


def test_sandwich_norms_are_optional():
    model = _model(seed=6, use_sandwich_norm=False)
    assert "post_attn_norm" not in model.layers[0]
    got = _np(model(mx.array(_TOKENS)[None])[0])
    assert _rel(got, _ref_model(model, _TOKENS, model.args)) < 1e-2


def test_conv_tap_order_and_state():
    """The last tap reads the current position, and a decode step from the
    cached state equals the same position of a whole pass."""
    args = _tiny_args()
    conv = _randomize(ShortConv(args), seed=8)
    rng = np.random.default_rng(9)
    x = rng.standard_normal((10, args.hidden_size))
    xa = mx.array(x.astype(np.float32))[None]
    want = _ref_conv(conv, x)
    assert _rel(_np(conv(xa)[0]), want) < 5e-3

    from gmlx.cache.kv_cache import ArraysCache

    cache = ArraysCache(size=1)
    head = conv(xa[:, :4], cache=cache)
    steps = [conv(xa[:, t:t + 1], cache=cache) for t in range(4, 10)]
    got = np.concatenate([_np(head[0])] + [_np(s[0]) for s in steps])
    assert _rel(got, want) < 5e-3
    assert cache[0].shape == (1, args.conv_L_cache - 1, args.hidden_size)

    # Only the current position: every tap but the last is zeroed.
    conv.kernel = mx.concatenate(
        [mx.zeros((args.hidden_size, 6)), mx.ones((args.hidden_size, 1))],
        axis=1)
    bcx = x @ _np(conv.in_proj.weight).T
    d = args.hidden_size
    local = (bcx[:, d:2 * d] * bcx[:, :d] * bcx[:, 2 * d:]) \
        @ _np(conv.out_proj.weight).T
    assert _rel(_np(conv(xa)[0]), local) < 5e-3


def test_cached_decode_matches_full_forward_past_the_window():
    model = _model(seed=4)
    tokens = _TOKENS.tolist()
    full = model(mx.array([tokens]))
    cache = model.make_cache()
    model(mx.array([tokens[:3]]), cache=cache)
    steps = [model(mx.array([[t]]), cache=cache) for t in tokens[3:]]
    mx.eval(full, steps)
    for n, s in enumerate(steps):
        np.testing.assert_allclose(
            _np(s[0, 0]), _np(full[0, 3 + n]), rtol=0, atol=1.5e-2)


def test_chunked_prefill_matches_one_pass():
    """A prompt read in chunks, one longer than the conv state, matches
    the prompt read at once."""
    model = _model(seed=5)
    tokens = (_TOKENS.tolist() * 2)[:20]
    full = model(mx.array([tokens]))
    cache = model.make_cache()
    parts = [model(mx.array([tokens[a:b]]), cache=cache)
             for a, b in ((0, 9), (9, 11), (11, 20))]
    got = mx.concatenate(parts, axis=1)
    mx.eval(full, got)
    np.testing.assert_allclose(_np(got), _np(full), rtol=0, atol=1.5e-2)


def test_wide_prefill_chunk_evaluates_per_layer_and_matches():
    """A chunk of PREFILL_EVAL_ROWS or more evaluates each layer as it
    finishes. The output and the decode after it match the lazy pass."""
    from gmlx.models.lfm3.model import PREFILL_EVAL_ROWS

    model = _model(seed=14)
    rng = np.random.default_rng(15)
    n = PREFILL_EVAL_ROWS + 6
    tokens = rng.integers(1, model.args.vocab_size, n + 1).tolist()
    full = model(mx.array([tokens]))
    cache = model.make_cache()
    got = model(mx.array([tokens[:n]]), cache=cache)
    step = model(mx.array([tokens[n:]]), cache=cache)
    mx.eval(full, got, step)
    np.testing.assert_allclose(_np(got), _np(full[:, :n]), rtol=0, atol=2e-2)
    np.testing.assert_allclose(_np(step[0, 0]), _np(full[0, n]),
                               rtol=0, atol=2e-2)


def test_left_padded_batch_matches_unpadded_rows():
    """A left-padded row matches the same prompt run alone, on prefill and
    on decode."""
    from gmlx.cache.kv_cache import (
        ArraysCache,
        BatchKVCache,
        BatchRotatingKVCache,
    )

    model = _model(seed=12)
    args = model.args
    rng = np.random.default_rng(13)
    pad = 7
    short = rng.integers(1, args.vocab_size, 5).tolist()
    long = rng.integers(1, args.vocab_size, 5 + pad).tolist()
    batch = mx.array([[0] * pad + short, long])
    cache = []
    for layer in model.layers:
        if layer.layer_type == CONV:
            cache.append(ArraysCache(size=1, left_padding=[pad, 0]))
        elif layer.use_sliding:
            cache.append(BatchRotatingKVCache(args.sliding_window, [pad, 0]))
        else:
            cache.append(BatchKVCache([pad, 0]))
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


def test_each_layer_type_gets_its_cache():
    from gmlx.cache.kv_cache import ArraysCache, KVCache, RotatingKVCache

    args = _tiny_args()
    for kind, c in zip(_LAYER_TYPES, Model(args).make_cache()):
        if kind == CONV:
            assert type(c) is ArraysCache
        elif kind == SLIDING:
            assert isinstance(c, RotatingKVCache)
            assert c.max_size == args.sliding_window
        else:
            assert type(c) is KVCache


def test_rope_factors_set_the_full_layers_only():
    from gmlx._vendor.mlx_lm.models.rope_utils import ProportionalRoPE
    import mlx.nn as nn

    model = Model(_tiny_args())
    assert model.set_full_rope_factors(_factors()) == _ROTATED
    for layer in model.layers:
        if layer.layer_type == FULL:
            assert type(layer.self_attn.rope) is ProportionalRoPE
        elif layer.layer_type == SLIDING:
            rope = layer.self_attn.rope
            assert type(rope) is nn.RoPE and rope.base == 10000.0


@pytest.mark.parametrize("values, match", [
    ([1.0] * 4, "8 factors|has 4 factors"),
    ([1e30] * 8, "not a run of 1.0"),
    ([1.0, 1.0, 2.0, 1e30, 1e30, 1e30, 1e30, 1e30], "not a run of 1.0"),
    ([1.0, 1e30, 1.0, 1e30, 1e30, 1e30, 1e30, 1e30], "not a run of 1.0"),
])
def test_rope_factors_refuse_other_patterns(values, match):
    with pytest.raises(ValueError, match=match):
        rotated_dims_from_factors(mx.array(values), 16)


def test_rope_factors_accept_the_shipped_pattern():
    shipped = mx.array([1.0] * 32 + [1e30] * 32)
    assert rotated_dims_from_factors(shipped, 128) == 64
    # Every dimension rotated is the same pattern with no tail.
    assert rotated_dims_from_factors(mx.ones((8,)), 16) == 16


def test_captures_layer_inputs():
    """A drafter reads each captured layer's input."""
    model = _model(seed=7)
    tokens = mx.array(_TOKENS)[None]
    sink: list = []
    out = model(tokens, capture_layer_ids=[0, 3], hidden_sink=sink)
    assert len(sink) == 2
    embed = model.model.embed_tokens(tokens)
    np.testing.assert_allclose(_np(sink[0]), _np(embed), rtol=0, atol=1e-6)
    h = embed
    for layer in model.layers[:3]:
        mask = "causal" if layer.is_attention_layer else None
        h = layer(h, mask, None)
    np.testing.assert_allclose(_np(sink[1]), _np(h), rtol=0, atol=1e-4)
    np.testing.assert_allclose(_np(out), _np(model(tokens)), rtol=0, atol=1e-5)


def test_prefill_decay_prices_no_score_tensor_at_head_dim_128(monkeypatch):
    """Depth decay sizes LFM3's prefill chunks as fused attention: no
    score tensor at head dim 128. The conv layers' caches do not disarm
    the profile."""
    import gmlx.gen.prefill_decay as pd
    import gmlx.models.lfm3  # noqa: F401  registers the profile

    assert pd._SCORE_PROFILES["lfm3"] is pd.fused_attention_profile
    monkeypatch.setattr(pd, "cpu_default_device", lambda: False)
    model = Model(_tiny_args(head_dim=128))
    profile = pd.resolve_score_profile(model, model.make_cache())
    assert profile is not None and profile.fused_min_step == 9


@pytest.mark.parametrize("mode, norms", [("auto", False), ("fp32", True),
                                         ("activation", False)])
def test_cast_predicate_matches_the_loader_keep_list(monkeypatch, mode, norms):
    from gmlx.load.loader import fp32_keep_for

    monkeypatch.setenv("GMLX_NORM_DTYPE", mode)
    keep = fp32_keep_for("lfm3")
    model = Model(_tiny_args())
    pred = model.cast_predicate
    pinned = set()
    for path, _ in tree_flatten(model.parameters()):
        assert pred(path) == (not any(s in path for s in keep)), path
        if not pred(path):
            pinned.add(re.sub(r"^model\.layers\.\d+\.", "", path))
    want = {"conv.kernel"}
    if norms:
        want |= {"input_layernorm.weight", "post_attention_layernorm.weight",
                 "post_attn_norm.weight", "post_ffw_norm.weight",
                 "self_attn.q_norm.weight", "self_attn.k_norm.weight",
                 "model.norm.weight"}
    assert pinned == want


def test_remap_covers_every_wire_tensor_onto_real_params():
    """Every tensor llama.cpp's lfm3 loader creates maps onto a parameter
    of the model, and every parameter is loaded."""
    from gmlx.load.remap import RemapDecision, parse_gguf_name

    args = _tiny_args()
    params = {k for k, _ in tree_flatten(Model(args).parameters())}
    common = ["attn_norm.weight", "post_attention_norm.weight",
              "ffn_norm.weight", "post_ffw_norm.weight", "ffn_gate.weight",
              "ffn_up.weight", "ffn_down.weight"]
    attn = ["attn_q.weight", "attn_k.weight", "attn_v.weight",
            "attn_output.weight", "attn_q_norm.weight", "attn_k_norm.weight",
            "attn_gate.weight"]
    conv = ["shortconv.conv.weight", "shortconv.in_proj.weight",
            "shortconv.out_proj.weight"]
    names = ["token_embd.weight", "output_norm.weight"]
    for i, kind in enumerate(_LAYER_TYPES):
        names += [f"blk.{i}.{t}"
                  for t in common + (conv if kind == CONV else attn)]
    mapped = set()
    for name in names:
        dec = parse_gguf_name("lfm3", name)
        assert dec.kind == RemapDecision.KIND_MAP, (name, dec.reason)
        assert dec.transform == "passthrough", name
        assert dec.hf_name in params, (name, dec.hf_name)
        mapped.add(dec.hf_name)
    assert mapped == params
