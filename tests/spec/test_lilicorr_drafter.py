#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Asher Feldman
"""LiLiCorr drafter units: the lattice head against a float64 reference of
llama.cpp's ``build_lilicorr_scores`` and its greedy walk, the closed
remap of the GGUF tensor set, container and loader routing, and the
nemotron_h target's packed-hidden capture with its batched verify
rollback. Tiny random weights; no GGUF and no download."""

from __future__ import annotations

import numpy as np
import pytest

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten

from gmlx.spec.dflash_drafter import DFlashConfig, greedy_path
from gmlx.spec.lilicorr_drafter import LiLiCorrDrafter

H, V, BLOCK, K = 64, 48, 5, 4          # 4 slots x 4 candidates = 16 nodes


def _config(**over) -> DFlashConfig:
    kw = dict(
        hidden_size=H, intermediate_size=96, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        rms_norm_eps=1e-6, vocab_size=V, max_position_embeddings=4096,
        rope_theta=10000.0, block_size=BLOCK, native_block_size=BLOCK,
        mask_token_id=V - 1, target_layer_ids=[0, 2], num_target_layers=4,
        layer_types=["sliding_attention"] * 2, sliding_window=16,
        conv_kernel_size=2, conv_group_size=16, attention_sinks=True,
        lilicorr_layers=2, lilicorr_hidden_size=32, lilicorr_top_k=K,
        lilicorr_heads=4, lilicorr_factor_dim=24, lilicorr_mlp_size=40,
        lilicorr_vector_eps=1e-4, lilicorr_logit_scale=8.0)
    kw.update(over)
    return DFlashConfig(**kw)


def _randomize(module: nn.Module, seed: int, scale: float = 0.3):
    rng = np.random.default_rng(seed)
    module.update(tree_unflatten([
        (k, mx.array(rng.normal(0, scale, v.shape).astype(np.float32)))
        for k, v in tree_flatten(module.parameters())]))
    mx.eval(module.parameters())


def _np(a) -> np.ndarray:
    return np.array(a.astype(mx.float32), dtype=np.float64)


def _lin(m, x):
    y = x @ _np(m.weight).T
    return y + _np(m.bias) if "bias" in m else y


def _rms(x, w, eps):
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * w


def _silu(x):
    return x / (1.0 + np.exp(-x))


def _l2n(x, eps):
    return x / np.maximum(np.sqrt((x * x).sum(-1, keepdims=True)), eps)


def _ref_lattice(head, passes, logits, embed_w, anchor):
    """build_lilicorr_scores in float64, node order slot-major."""
    slots, k = logits.shape[0], head.top_k
    probs = np.exp(logits - logits.max(-1, keepdims=True))
    probs /= probs.sum(-1, keepdims=True)
    cands = np.argsort(-logits, axis=-1, kind="stable")[:, :k]
    p = np.take_along_axis(probs, cands, -1)
    lp = np.log(p)
    rank = np.arange(k, dtype=np.float64)
    feat = np.stack([lp, p, lp - lp[:, :1],
                     np.broadcast_to(rank / (k - 1), lp.shape),
                     np.broadcast_to((rank == 0) * 1.0, lp.shape)], -1)
    mu = feat.mean(-1, keepdims=True)
    var = ((feat - mu) ** 2).mean(-1, keepdims=True)
    fn = (feat - mu) / np.sqrt(var + 1e-5) * _np(head.feature_norm.weight) \
        + _np(head.feature_norm.bias)
    fs = _lin(head.feature_down, _silu(_lin(head.feature_up, fn)))
    tok = _lin(head.token_proj, embed_w[cands])
    ps = _lin(head.pass_hidden_proj, passes)[:, None, :]
    pos = _np(head.slot_embedding)[:, None, :] + _np(head.rank_embedding)[None]
    x = (tok + ps + fs + pos).reshape(slots * k, -1)
    n, hidden = x.shape
    bias = _np(head.attention_bias)
    for layer in head.layers:
        heads = layer.heads
        hd = hidden // heads
        a = _rms(x, _np(layer.attn_norm), head.eps)
        q, kk, v = np.split(_lin(layer.in_proj, a), 3, -1)
        q, kk, v = (t.reshape(n, heads, hd).transpose(1, 0, 2) for t in (q, kk, v))
        s = q @ kk.transpose(0, 2, 1) / np.sqrt(hd) + bias
        s = np.exp(s - s.max(-1, keepdims=True))
        s /= s.sum(-1, keepdims=True)
        o = (s @ v).transpose(1, 0, 2).reshape(n, hidden)
        x = x + _lin(layer.out_proj, o)
        m = _rms(x, _np(layer.mlp_norm), head.eps)
        x = x + _lin(layer.mlp_down, _silu(_lin(layer.mlp_up, m)))
    x = _rms(x, _np(head.output_norm), head.eps)
    a = _rms(_lin(head.context_proj, anchor[None]), _np(head.anchor_norm), head.eps)
    ab = np.broadcast_to(a, x.shape)
    f = _silu(_lin(head.factor_input_proj, np.concatenate([x, ab, x * ab], -1)))
    eps = head.vector_eps
    out = _l2n(_lin(head.out_head, f), eps).reshape(slots, k, -1)
    inn = _l2n(_lin(head.in_head, f), eps).reshape(slots, k, -1)
    a_out = _l2n(_lin(head.anchor_out_head, a), eps)[0]
    first = head.logit_scale * inn[0] @ a_out
    edges = head.logit_scale * np.einsum("spf,snf->spn", out[:-1], inn[1:])
    return cands, first, edges


def _ref_walk(cands, first, edges, n_max):
    """common/speculative.cpp's lattice loop over the packed rows."""
    rows = [first] + [e.reshape(-1) for e in edges]
    pred, out = 0, []
    for i in range(n_max):
        if i == 0:
            scores = rows[0]
        else:
            scores = rows[i][pred * len(first):(pred + 1) * len(first)]
        pred = int(np.argmax(scores))
        out.append(int(cands[i][pred]))
    return out


def _head_inputs(seed=1):
    rng = np.random.default_rng(seed)
    slots = BLOCK - 1
    passes = rng.normal(0, 1, (slots, H)).astype(np.float32)
    logits = rng.normal(0, 3, (slots, V)).astype(np.float32)
    anchor = rng.normal(0, 1, (H,)).astype(np.float32)
    embed = nn.Embedding(V, H)
    _randomize(embed, seed + 7, 1.0)
    return passes, logits, anchor, embed


def test_lattice_matches_reference():
    drafter = LiLiCorrDrafter(_config())
    head = drafter.lilicorr
    _randomize(head, 3)
    passes, logits, anchor, embed = _head_inputs()
    cands, first, edges = head.lattice(
        mx.array(passes), mx.array(logits), embed, mx.array(anchor))
    rc, rf, re_ = _ref_lattice(head, passes.astype(np.float64),
                               logits.astype(np.float64),
                               _np(embed.weight), anchor.astype(np.float64))
    np.testing.assert_array_equal(np.array(cands), rc)
    np.testing.assert_allclose(_np(first), rf, rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(_np(edges), re_, rtol=1e-3, atol=1e-3)
    assert edges.shape == (BLOCK - 2, K, K)


@pytest.mark.parametrize("n_max", [1, 2, BLOCK - 1])
def test_greedy_walk_matches_llama_cpp_loop(n_max):
    drafter = LiLiCorrDrafter(_config())
    _randomize(drafter.lilicorr, 5)
    passes, logits, anchor, embed = _head_inputs(2)
    cands, first, edges = drafter.lilicorr.lattice(
        mx.array(passes), mx.array(logits), embed, mx.array(anchor))
    sel = greedy_path(first, edges[: n_max - 1])
    got = np.take_along_axis(np.array(cands)[:n_max], np.array(sel)[:, None], -1)
    want = _ref_walk(np.array(cands), _np(first), _np(edges), n_max)
    assert got[:, 0].tolist() == want


# Random values here put the Mamba scan outside its working range.
_KEEP_INIT = (".A_log", ".dt_bias", ".D", ".conv1d.")


def _tiny_target(seed=0):
    from gmlx.models.nemotron_h.model import ModelArgs
    from gmlx.models.nemotron_h.mtp import NemotronHSpecLM

    args = ModelArgs(
        model_type="nemotron_h", vocab_size=40, hidden_size=64,
        intermediate_size=32, num_hidden_layers=4,
        max_position_embeddings=1024, num_attention_heads=4,
        num_key_value_heads=2, attention_bias=False, mamba_num_heads=4,
        mamba_head_dim=16, mamba_proj_bias=False, ssm_state_size=8,
        conv_kernel=4, n_groups=2, mlp_bias=False, layer_norm_epsilon=1e-5,
        use_bias=False, use_conv_bias=True,
        hybrid_override_pattern=["M", "*", "E", "M"], head_dim=16,
        moe_intermediate_size=24, moe_shared_expert_intermediate_size=48,
        moe_latent_size=16, n_routed_experts=8, n_shared_experts=1,
        num_experts_per_tok=2, norm_topk_prob=True, routed_scaling_factor=5.0,
        n_group=1, topk_group=1)
    model = NemotronHSpecLM(args)
    rng = np.random.default_rng(seed)
    model.update(tree_unflatten([
        (k, mx.array(rng.normal(0, 0.3, v.shape).astype(np.float32)))
        for k, v in tree_flatten(model.parameters())
        if not any(s in k for s in _KEEP_INIT)]))
    mx.eval(model.parameters())
    return model


def test_draft_block_runs_full_block_and_walks_requested_slots():
    model = _tiny_target()
    cfg = _config(hidden_size=64, vocab_size=40, mask_token_id=39,
                  target_layer_ids=[0, 2], num_target_layers=4)
    drafter = LiLiCorrDrafter(cfg)
    _randomize(drafter, 9, 0.1)
    drafter.reset(model)
    model.set_dflash_capture([0, 2])
    out = model(mx.array([[1, 5, 9, 2, 7]]), cache=model.make_cache(),
                return_hidden=True)
    drafter.prefill_from_target_hidden(None, out.hidden_states[-1], 3, None)
    assert drafter._cache.anchor is not None
    for bs in (2, 3, BLOCK):
        toks = drafter.draft_block(4, None, None, bs, None, greedy=True)
        assert toks.shape == (1, bs - 1)


def test_capture_packs_trunk_and_layers():
    model = _tiny_target(1)
    ids = mx.array([[1, 5, 9, 2]])
    plain = model(ids, cache=model.make_cache(), return_hidden=True)
    model.set_dflash_capture([0, 2])
    packed = model(ids, cache=model.make_cache(), return_hidden=True)
    h = packed.hidden_states[-1]
    assert h.shape[-1] == 64 * 3
    np.testing.assert_allclose(_np(h[..., :64]), _np(plain.hidden_states[-1]),
                               rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(_np(packed.logits), _np(plain.logits),
                               rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(
        _np(model.speculative_logits_from_hidden(h)), _np(plain.logits),
        rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("accepted", [0, 2, 4])
def test_batched_verify_rollback_matches_a_fresh_prefix(accepted):
    # The DFlash verify runs batched; rolling back to the accepted prefix
    # must leave the caches as running that prefix would.
    model = _tiny_target(2)
    model.set_dflash_capture([0, 2])
    prompt = mx.array([[1, 5, 9, 2]])
    block = mx.array([[7, 3, 8, 6, 4]])
    cache = model.make_cache()
    model(prompt, cache=cache)
    _, _, sink = model.speculative_verify_hidden(block, cache)
    model.rollback_speculative_cache(cache, sink, accepted, block.shape[1])
    ref = model.make_cache()
    model(prompt, cache=ref)
    model(block[:, : accepted + 1], cache=ref)
    nxt = mx.array([[2]])
    np.testing.assert_allclose(
        _np(model(nxt, cache=cache).logits), _np(model(nxt, cache=ref).logits),
        rtol=5e-3, atol=5e-3)


# The real header's names and shapes (GGUF order), at tiny width.
def _lilicorr_arrays(cfg: DFlashConfig) -> dict:
    n, hd, hl = cfg.num_hidden_layers, cfg.lilicorr_hidden_size, cfg.lilicorr_layers
    ff, k, slots = cfg.intermediate_size, cfg.lilicorr_top_k, BLOCK - 1
    kv = cfg.num_key_value_heads * cfg.head_dim
    groups = H // cfg.conv_group_size
    shapes = {"fc.weight": [2 * H, H], "fc.scale": [1],
              "enc.output_norm.weight": [H], "output_norm.weight": [H]}
    for i in range(n):
        b = f"blk.{i}."
        shapes.update({
            b + "attn_norm.weight": [H], b + "ffn_norm.weight": [H],
            b + "attn_q.weight": [H, H], b + "attn_k.weight": [H, kv],
            b + "attn_v.weight": [H, kv], b + "attn_output.weight": [H, H],
            b + "attn_q_norm.weight": [cfg.head_dim],
            b + "attn_k_norm.weight": [cfg.head_dim],
            b + "attn_sinks.weight": [cfg.num_attention_heads],
            b + "ffn_gate.weight": [H, ff], b + "ffn_up.weight": [H, ff],
            b + "ffn_down.weight": [ff, H], b + "ffn_down.scale": [1],
            b + "attn_conv_base": [H, 2, 2], b + "ffn_conv_base": [H, 2, 2],
            b + "attn_conv_proj.weight": [H, 2 * 2 * groups],
            b + "ffn_conv_proj.weight": [H, 2 * 2 * groups],
        })
    for i in range(hl):
        b = f"blk.{n + i}."
        shapes.update({
            b + "attn_norm.weight": [hd], b + "ffn_norm.weight": [hd],
            b + "attn_output.weight": [hd, hd], b + "attn_output.bias": [hd],
            b + "attn_output.scale": [1],
            b + "ffn_up.weight": [hd, cfg.lilicorr_mlp_size],
            b + "ffn_up.bias": [cfg.lilicorr_mlp_size],
            b + "ffn_down.weight": [cfg.lilicorr_mlp_size, hd],
            b + "ffn_down.bias": [hd],
            f"lilicorr.layers.{i}.attn.in_proj.weight": [hd, 3 * hd],
            f"lilicorr.layers.{i}.attn.in_proj.bias": [3 * hd],
        })
    f = f"blk.{n + hl}."
    shapes.update({f + "ffn_norm.weight": [5], f + "ffn_norm.bias": [5],
                   f + "ffn_up.weight": [5, hd], f + "ffn_up.bias": [hd],
                   f + "ffn_down.weight": [hd, hd], f + "ffn_down.bias": [hd]})
    for name in ("token_proj", "pass_hidden_proj", "context_proj"):
        shapes.update({f"lilicorr.{name}.weight": [H, hd],
                       f"lilicorr.{name}.bias": [hd],
                       f"lilicorr.{name}.scale": [1]})
    for name in ("out_head", "in_head", "anchor_out_head"):
        shapes.update({f"lilicorr.{name}.weight": [hd, cfg.lilicorr_factor_dim],
                       f"lilicorr.{name}.bias": [cfg.lilicorr_factor_dim]})
    shapes.update({
        "lilicorr.factor_input_proj.weight": [3 * hd, hd],
        "lilicorr.factor_input_proj.bias": [hd],
        "lilicorr.output_norm.weight": [hd], "lilicorr.anchor_norm.weight": [hd],
        "lilicorr.slot_embedding.weight": [hd, slots],
        "lilicorr.rank_embedding.weight": [hd, k],
        "lilicorr.attention_bias.bias": [slots * k, slots * k, cfg.lilicorr_heads],
    })
    return {name: mx.zeros(tuple(reversed(ne))) for name, ne in shapes.items()}


def test_remap_fills_every_param_from_the_closed_set():
    from gmlx.load.out_scale import install_out_scales
    from gmlx.spec.mtp_load import dflash_container, remap_dflash_arrays

    cfg = _config(target_layer_ids=[0, 1])
    arrays = _lilicorr_arrays(cfg)
    assert dflash_container(arrays) == "lilicorr"
    weights, _, stats = remap_dflash_arrays(
        arrays, {}, "lilicorr", n_layers=cfg.num_hidden_layers,
        head_layers=cfg.lilicorr_layers)
    assert stats["mapped"] == len(arrays)
    drafter = LiLiCorrDrafter(cfg)
    install_out_scales(drafter, weights)
    params = dict(tree_flatten(drafter.parameters()))
    assert set(weights) == set(params)
    for name, arr in weights.items():
        assert tuple(arr.shape) == tuple(params[name].shape), name


def test_remap_refuses_an_unknown_tensor():
    from gmlx.spec.mtp_load import remap_dflash_arrays

    cfg = _config()
    arrays = _lilicorr_arrays(cfg)
    arrays["lilicorr.mystery.weight"] = mx.zeros((4,))
    with pytest.raises(RuntimeError, match="lilicorr.mystery"):
        remap_dflash_arrays(arrays, {}, "lilicorr", n_layers=2, head_layers=2)


def test_nemotron_dflash_header_routes_to_the_dflash_loader(monkeypatch):
    import gmlx.spec.mtp_load as ml

    monkeypatch.setattr(ml, "_drafter_header_arch", lambda p: "dflash")
    assert ml._assistant_kind("nemotron_h", "x.gguf") == "dflash"
    monkeypatch.setattr(ml, "_drafter_header_arch", lambda p: "nemotron_h_moe")
    assert ml._assistant_kind("nemotron_h", "x.gguf") == "nemotron"
