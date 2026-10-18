#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Asher Feldman
"""Nemotron-H MoE block against a float64 reference of llama.cpp's
``build_moe_ffn`` as the nemotron_h graph calls it, on tiny random weights.
No GGUF and no download.

The reference covers per-layer expert counts, widths and top-k (Bolt), the
fp32 router, the selection-only bias, the weight-sum clamp at
6.103515625e-5, renormalization at top-1, and the scale rule (0 and 1 leave
the weights alone). The synth tests drive the per-layer derivation from
router shapes.
"""

from __future__ import annotations

import numpy as np
import pytest

import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten

from gmlx.load.config_synth import synthesize_config
from gmlx.models.nemotron_h.model import Model, ModelArgs, per_layer

ARCH = "nemotron_h_moe"
# M E * E E: experts 8/6/4 wide 24/16/32 top 3/2/1 on the MoE layers.
_PATTERN = ["M", "E", "*", "E", "E"]
_COUNTS = [0, 8, 0, 6, 4]
_WIDTHS = [0, 24, 0, 16, 32]
_TOPK = [0, 3, 0, 2, 1]


def _tiny_args(**over) -> ModelArgs:
    kw = dict(
        model_type="nemotron_h",
        vocab_size=40,
        hidden_size=64,
        intermediate_size=32,
        num_hidden_layers=len(_PATTERN),
        max_position_embeddings=1024,
        num_attention_heads=4,
        num_key_value_heads=2,
        attention_bias=False,
        mamba_num_heads=4,
        mamba_head_dim=16,
        mamba_proj_bias=False,
        ssm_state_size=8,
        conv_kernel=4,
        n_groups=2,
        mlp_bias=False,
        layer_norm_epsilon=1e-5,
        use_bias=False,
        use_conv_bias=True,
        hybrid_override_pattern=list(_PATTERN),
        head_dim=16,
        moe_intermediate_size=list(_WIDTHS),
        moe_shared_expert_intermediate_size=48,
        moe_latent_size=16,
        n_routed_experts=list(_COUNTS),
        n_shared_experts=1,
        num_experts_per_tok=list(_TOPK),
        norm_topk_prob=True,
        routed_scaling_factor=5.0,
        n_group=1,
        topk_group=1,
    )
    kw.update(over)
    return ModelArgs(**kw)


# Random values here put the Mamba scan outside its working range.
_KEEP_INIT = (".A_log", ".dt_bias", ".D", ".conv1d.")


def _randomized(args: ModelArgs, seed: int = 0) -> Model:
    model = Model(args)
    rng = np.random.default_rng(seed)
    params = [(k, mx.array(rng.normal(0, 0.3, v.shape).astype(np.float32)))
              for k, v in tree_flatten(model.parameters())
              if not any(s in k for s in _KEEP_INIT)]
    model.update(tree_unflatten(params))
    mx.eval(model.parameters())
    return model


def _np(a) -> np.ndarray:
    return np.array(a.astype(mx.float32), dtype=np.float64)


def _relu2(v):
    return np.maximum(v, 0.0) ** 2


def _ref_moe(moe, x: np.ndarray, top_k: int, norm: bool, scale: float):
    """llama.cpp build_moe_ffn (sigmoid, selection bias, no gate, relu^2)
    plus nemotron_h's latent projections and shared expert."""
    w_r = _np(moe.gate.weight)
    bias = _np(moe.gate.e_score_correction_bias)
    probs = 1.0 / (1.0 + np.exp(-(x @ w_r.T)))
    sel = probs + bias
    idx = np.argsort(-sel, axis=-1, kind="stable")[..., :top_k]
    w = np.take_along_axis(probs, idx, axis=-1)
    if norm:
        w = w / np.maximum(w.sum(-1, keepdims=True), 6.103515625e-5)
    if scale not in (0.0, 1.0):
        w = w * scale
    xl = x @ _np(moe.fc1_latent_proj.weight).T
    up = _np(moe.switch_mlp.fc1.weight)
    down = _np(moe.switch_mlp.fc2.weight)
    y = np.zeros_like(xl)
    for t in range(x.shape[0]):
        for j in range(top_k):
            e = idx[t, j]
            y[t] += w[t, j] * (down[e] @ _relu2(up[e] @ xl[t]))
    out = y @ _np(moe.fc2_latent_proj.weight).T
    sh = moe.shared_experts
    return out + _relu2(x @ _np(sh.up_proj.weight).T) @ _np(sh.down_proj.weight).T


@pytest.mark.parametrize("layer", [1, 3, 4])
def test_per_layer_moe_sizes(layer):
    model = _randomized(_tiny_args())
    moe = model.layers[layer].mixer
    assert moe.gate.weight.shape == (_COUNTS[layer], 64)
    assert moe.gate.weight.dtype == mx.float32
    assert moe.switch_mlp.fc1.weight.shape == (_COUNTS[layer], _WIDTHS[layer], 16)
    assert moe.gate.top_k == _TOPK[layer]


@pytest.mark.parametrize("layer", [1, 3, 4])
@pytest.mark.parametrize("scale", [5.0, 1.0, 0.0])
def test_moe_matches_reference(layer, scale):
    model = _randomized(_tiny_args(routed_scaling_factor=scale), seed=layer)
    moe = model.layers[layer].mixer
    x = np.random.default_rng(7).normal(0, 1, (5, 64)).astype(np.float32)
    got = _np(moe(mx.array(x)[None]))[0]
    ref = _ref_moe(moe, x.astype(np.float64), _TOPK[layer], True, scale)
    np.testing.assert_allclose(got, ref, rtol=2e-3, atol=2e-3)


def test_weight_sum_clamp():
    # Router logits far below zero: the top-k probabilities sum under the
    # floor, so llama.cpp divides by the floor, not the sum.
    model = _randomized(_tiny_args(), seed=3)
    moe = model.layers[1].mixer
    moe.gate.weight = mx.full(moe.gate.weight.shape, -0.5)  # logits -32
    moe.gate.e_score_correction_bias = mx.array(
        np.linspace(0, 1, 8).astype(np.float32))
    inds, scores = moe.gate(mx.ones((1, 2, 64)))
    probs = 1.0 / (1.0 + np.exp(32.0))
    np.testing.assert_allclose(
        np.array(scores), probs / 6.103515625e-5 * 5.0, rtol=1e-3)
    assert set(np.array(inds)[0, 0].tolist()) == {5, 6, 7}


def test_top1_still_renormalizes():
    model = _randomized(_tiny_args(), seed=4)
    _, scores = model.layers[4].mixer.gate(mx.ones((1, 3, 64)))
    np.testing.assert_allclose(np.array(scores), 5.0, rtol=1e-6)


def test_forward_and_cached_decode_agree():
    model = _randomized(_tiny_args(), seed=5)
    tokens = mx.array([[1, 5, 9, 2, 7, 3]])
    full = model(tokens)
    cache = model.make_cache()
    model(tokens[:, :-1], cache=cache)
    step = model(tokens[:, -1:], cache=cache)
    # The chunked scan and the one-step update round differently.
    np.testing.assert_allclose(
        _np(step[0, -1]), _np(full[0, -1]), rtol=5e-3, atol=5e-3)


def test_per_layer_index_past_list_takes_largest():
    assert per_layer([0, 8, 0, 6], 3) == 6
    assert per_layer([0, 8, 0, 6], 4) == 8
    assert per_layer([0, 8, 0, 6], None) == 8
    assert per_layer(5, 2) == 5


def test_cast_predicate_matches_loader_keep_row():
    from gmlx.load.loader import _FP32_KEEP_BY_MODEL_TYPE

    keep = _FP32_KEEP_BY_MODEL_TYPE["nemotron_h"]
    model = Model(_tiny_args())
    for name, _ in tree_flatten(model.parameters()):
        assert model.cast_predicate(name) == (
            not any(s in name for s in keep)), name
    assert not model.cast_predicate("backbone.layers.1.mixer.gate.weight")
    assert model.cast_predicate("backbone.layers.1.mixer.switch_mlp.fc1.weight")


def _bolt_meta():
    """Tiny Bolt-shaped KV, per-layer arrays over the 5 blocks above."""
    kv = [0, 0, 2, 0, 0]
    return {
        "general.architecture": ARCH,
        f"{ARCH}.embedding_length": 64,
        f"{ARCH}.block_count": 5,
        f"{ARCH}.attention.head_count": 4,
        f"{ARCH}.attention.head_count_kv": kv,
        f"{ARCH}.feed_forward_length": list(_WIDTHS),
        f"{ARCH}.context_length": 1024,
        f"{ARCH}.attention.layer_norm_rms_epsilon": 1e-5,
        f"{ARCH}.attention.key_length": 16,
        f"{ARCH}.ssm.inner_size": 64,
        f"{ARCH}.ssm.time_step_rank": 4,
        f"{ARCH}.ssm.state_size": 8,
        f"{ARCH}.ssm.conv_kernel": 4,
        f"{ARCH}.ssm.group_count": 2,
        f"{ARCH}.expert_count": 8,
        f"{ARCH}.expert_used_count": list(_TOPK),
        f"{ARCH}.expert_feed_forward_length": list(_WIDTHS),
        f"{ARCH}.expert_shared_count": 1,
        f"{ARCH}.expert_shared_feed_forward_length": 48,
        f"{ARCH}.expert_weights_norm": True,
        f"{ARCH}.expert_weights_scale": 5.0,
        f"{ARCH}.moe_latent_size": 16,
        "tokenizer.ggml.tokens": ["t"] * 40,
    }


def _bolt_shapes():
    return {f"blk.{i}.ffn_gate_inp.weight": [64, n]
            for i, n in enumerate(_COUNTS) if n}


def test_synth_reads_per_layer_sizes():
    config = synthesize_config(_bolt_meta(), tensor_shapes=_bolt_shapes())
    assert config["hybrid_override_pattern"] == _PATTERN
    assert config["n_routed_experts"] == _COUNTS
    assert config["moe_intermediate_size"] == _WIDTHS
    assert config["num_experts_per_tok"] == _TOPK
    model = Model(ModelArgs.from_dict(config))
    assert model.layers[4].mixer.gate.weight.shape == (4, 64)


def test_synth_uniform_sizes_stay_scalar():
    meta = _bolt_meta()
    meta[f"{ARCH}.expert_used_count"] = 2
    meta[f"{ARCH}.expert_feed_forward_length"] = 24
    shapes = {k: [64, 8] for k in _bolt_shapes()}
    config = synthesize_config(meta, tensor_shapes=shapes)
    assert config["n_routed_experts"] == 8
    assert config["moe_intermediate_size"] == 24
    assert config["num_experts_per_tok"] == 2


def test_synth_width_falls_back_to_ff_over_topk():
    # No expert_feed_forward_length: llama.cpp takes n_ff / n_expert_used.
    meta = _bolt_meta()
    del meta[f"{ARCH}.expert_feed_forward_length"]
    config = synthesize_config(meta, tensor_shapes=_bolt_shapes())
    assert config["moe_intermediate_size"] == [0, 8, 0, 8, 32]


def test_synth_norm_defaults_off_like_llama_cpp():
    meta = _bolt_meta()
    del meta[f"{ARCH}.expert_weights_norm"]
    assert synthesize_config(meta, tensor_shapes=_bolt_shapes())[
        "norm_topk_prob"] is False


def test_synth_rejects_router_past_expert_count():
    shapes = _bolt_shapes()
    shapes["blk.3.ffn_gate_inp.weight"] = [64, 9]
    with pytest.raises(ValueError, match="blk.3.ffn_gate_inp"):
        synthesize_config(_bolt_meta(), tensor_shapes=shapes)
