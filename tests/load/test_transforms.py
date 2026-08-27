#!/usr/bin/env python3
"""Wire-byte layout transforms: shape correctness and round-trip identities.

These run on tiny arrays (no GGUF, no GPU kernels) but do need ``mlx.core``
since the transforms operate on ``mx.array``.
"""

from __future__ import annotations

import numpy as np
import pytest

import mlx.core as mx  # noqa: E402

from gmlx.transforms import (  # noqa: E402
    coalesce_split_experts,
    fuse_shexp_gate_up,
    qk_permute_wire,
    retarget,
    split_fused_gate_up_kquant,
)


# coalesce_split_experts
def test_coalesce_stacks_per_expert_into_exps():
    # Two layers, 3 experts each: legacy per-expert split tensors (gate/up/down).
    n_exp = 3
    arrays, kquant = {}, {}
    for L in (0, 1):
        for proj, out, bpr in (("gate", 8, 5), ("up", 8, 5), ("down", 4, 7)):
            for e in range(n_exp):
                nm = f"blk.{L}.ffn_{proj}.{e}.weight"
                # row-distinct contents so stack order is verifiable.
                arrays[nm] = mx.full((out, bpr), L * 100 + e, dtype=mx.uint8)
                arrays[nm[:-len(".weight")] + ".scales"] = mx.zeros((1,), mx.uint8)
                kquant[nm] = "q3_k"
    # a non-expert tensor must pass through untouched.
    arrays["blk.0.attn_q.weight"] = mx.zeros((4, 5), mx.uint8)

    out_arrays, out_kquant, n_groups = coalesce_split_experts(arrays, kquant)

    assert n_groups == 2 * 3                       # 2 layers x {gate,up,down}
    # per-expert tensors are gone; stacked _exps tensors are present + codec'd.
    assert "blk.0.ffn_gate.0.weight" not in out_arrays
    g = out_arrays["blk.0.ffn_gate_exps.weight"]
    assert g.shape == (n_exp, 8, 5)
    assert out_kquant["blk.0.ffn_gate_exps.weight"] == "q3_k"
    assert out_arrays["blk.0.ffn_down_exps.weight"].shape == (n_exp, 4, 7)
    # expert axis is in index order 0,1,2 (leading-axis slices match inputs).
    assert bool(mx.all(g[0] == 0)) and bool(mx.all(g[1] == 1)) and bool(mx.all(g[2] == 2))
    # vestigial scales placeholder emitted; non-expert tensor untouched.
    assert out_arrays["blk.0.ffn_gate_exps.scales"].shape == (1,)
    assert "blk.0.attn_q.weight" in out_arrays


def test_coalesce_is_noop_on_stacked_form():
    arrays = {"blk.0.ffn_gate_exps.weight": mx.zeros((8, 4, 5), mx.uint8),
              "blk.0.attn_q.weight": mx.zeros((4, 5), mx.uint8)}
    kquant = {"blk.0.ffn_gate_exps.weight": "q4_k"}
    out_arrays, out_kquant, n_groups = coalesce_split_experts(arrays, kquant)
    assert n_groups == 0
    assert out_arrays is arrays and out_kquant is kquant


# fuse_shexp_gate_up (granitemoehybrid shared MLP input_linear)
def test_fuse_shexp_concats_gate_first():
    arrays = {
        "blk.0.ffn_gate_shexp.weight": mx.full((4, 6), 1, dtype=mx.uint8),
        "blk.0.ffn_up_shexp.weight": mx.full((4, 6), 2, dtype=mx.uint8),
        "blk.0.ffn_gate_shexp.scales": mx.zeros((1,), mx.uint8),
        "blk.0.ffn_up_shexp.scales": mx.zeros((1,), mx.uint8),
        "blk.0.ffn_down_shexp.weight": mx.zeros((3, 6), mx.uint8),  # untouched
    }
    kquant = {"blk.0.ffn_gate_shexp.weight": "q4_k",
              "blk.0.ffn_up_shexp.weight": "q4_k"}
    out, out_kq, n = fuse_shexp_gate_up(arrays, kquant)
    assert n == 1
    fused = out["blk.0.ffn_gate_up_shexp.weight"]
    # gate rows first, then up - mlx-lm splits input_linear's output in half
    # in exactly that order.
    assert fused.shape == (8, 6)
    assert bool(mx.all(fused[:4] == 1)) and bool(mx.all(fused[4:] == 2))
    assert out_kq["blk.0.ffn_gate_up_shexp.weight"] == "q4_k"
    assert "blk.0.ffn_gate_shexp.weight" not in out
    assert "blk.0.ffn_up_shexp.weight" not in out
    assert "blk.0.ffn_gate_shexp.scales" not in out
    assert out["blk.0.ffn_gate_up_shexp.scales"].shape == (1,)
    assert "blk.0.ffn_down_shexp.weight" in out      # down untouched


def test_fuse_shexp_noop_without_shexp_tensors():
    arrays = {"blk.0.attn_q.weight": mx.zeros((4, 5), mx.uint8)}
    out, out_kq, n = fuse_shexp_gate_up(arrays, {})
    assert n == 0 and out is arrays


def test_fuse_shexp_rejects_codec_mismatch():
    arrays = {
        "blk.0.ffn_gate_shexp.weight": mx.zeros((4, 6), mx.uint8),
        "blk.0.ffn_up_shexp.weight": mx.zeros((4, 6), mx.uint8),
    }
    kquant = {"blk.0.ffn_gate_shexp.weight": "q4_k",
              "blk.0.ffn_up_shexp.weight": "q6_k"}
    with pytest.raises(ValueError, match="cannot fuse losslessly"):
        fuse_shexp_gate_up(arrays, kquant)


def test_fuse_shexp_rejects_missing_half():
    arrays = {"blk.0.ffn_gate_shexp.weight": mx.zeros((4, 6), mx.uint8)}
    with pytest.raises(ValueError, match="needs both gate and up"):
        fuse_shexp_gate_up(arrays, {})


# split_fused_gate_up_kquant
def test_split_halves_along_penultimate_axis():
    # (n_experts, 2*intermediate, bytes_per_row)
    w = mx.arange(2 * 6 * 5).reshape(2, 6, 5)
    gate, up = split_fused_gate_up_kquant(w)
    assert gate.shape == (2, 3, 5)
    assert up.shape == (2, 3, 5)
    # halves are exactly the leading/trailing block of the fused row-group...
    assert bool(mx.all(gate == w[:, :3, :]))
    assert bool(mx.all(up == w[:, 3:, :]))
    # ...and recombine to the original.
    assert bool(mx.all(mx.concatenate([gate, up], axis=-2) == w))


def test_split_rejects_non_3d():
    with pytest.raises(ValueError):
        split_fused_gate_up_kquant(mx.zeros((6, 5)))


# qk_permute_wire
def _llamacpp_forward_permute(w_np, n_head):
    """convert_hf_to_gguf's LlamaModel.permute - the forward op we must undo."""
    n_out = w_np.shape[0]
    head_dim = n_out // n_head
    return (w_np.reshape(n_head, 2, head_dim // 2, *w_np.shape[1:])
                .swapaxes(1, 2)
                .reshape(w_np.shape))


def test_qk_permute_is_exact_inverse_of_llamacpp_permute():
    n_head, head_dim, hidden = 4, 8, 3
    rng = np.random.default_rng(0)
    w = rng.standard_normal((n_head * head_dim, hidden)).astype(np.float32)
    permuted = _llamacpp_forward_permute(w, n_head)
    undone = np.array(qk_permute_wire(mx.array(permuted), n_head))
    assert np.allclose(undone, w)


def test_qk_permute_preserves_shape_and_actually_permutes():
    n_head, head_dim = 4, 8
    w = mx.arange(n_head * head_dim * 2).reshape(n_head * head_dim, 2)
    out = qk_permute_wire(w, n_head)
    assert out.shape == w.shape
    # head_dim > 2 => the permute is non-trivial (not the identity).
    assert not bool(mx.all(out == w))


# retarget
def test_retarget_prepends_prefix():
    assert retarget("model.embed_tokens.weight", "language_model") == \
        "language_model.model.embed_tokens.weight"


def test_retarget_empty_prefix_is_noop():
    assert retarget("a.b.weight", "") == "a.b.weight"


def test_retarget_idempotent_when_already_prefixed():
    assert retarget("language_model.x", "language_model") == "language_model.x"
