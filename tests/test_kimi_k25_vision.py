"""Kimi-K2.5/K2.7 vision (mmproj projector ``kimik25`` -> mlx-vlm kimi_k25).

CPU-only: synthetic mmproj metadata + tensors, no GGUF and no real weights.
The load-bearing property is the Q/K RoPE de-permutation - MoonViT rotates
interleaved (x, y) dim pairs, llama.cpp's converter rewrites Q/K into the
split halves ``build_rope_2d`` wants, and mlx-vlm implements the interleaved
form. Getting that wrong still produces finite, plausible-looking features, so
it is pinned here against a literal port of the converter's own permute.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from gmlx.vlm import (
    _kimi_k25_qkv_split_to_interleaved,
    _kimi_k25_vision_name,
    _synthesize_kimi_k25_vlm_config,
    remap_vision_arrays,
)

N_HEAD = 4
HEAD_DIM = 8          # divisible by 4, as MoonViT's 2-D rope requires
DIM = N_HEAD * HEAD_DIM
IN = 3

MM_META = {
    "clip.projector_type": "kimik25",
    "clip.vision.block_count": 2,
    "clip.vision.embedding_length": DIM,
    "clip.vision.attention.head_count": N_HEAD,
    "clip.vision.feed_forward_length": 5 * DIM,
    "clip.vision.image_size": 896,
    "clip.vision.patch_size": 14,
    "clip.vision.projection_dim": 16,
    "clip.vision.projector.scale_factor": 2,
    "clip.vision.attention.layer_norm_epsilon": 1e-5,
    "clip.vision.image_mean": [0.5, 0.5, 0.5],
    "clip.vision.image_std": [0.5, 0.5, 0.5],
    "clip.vision.image_max_pixels": 3211264,
    "vision.pos_emb_height": 64,
    "vision.pos_emb_width": 64,
}


def _converter_permute(w: np.ndarray, n_head: int) -> np.ndarray:
    """llama.cpp ``conversion/kimivl.py`` KimiK25Model.permute, in numpy.

    Interleaved ``(freq, axis, pair)`` -> split ``(axis, freq, pair)``. Also
    covers the bias, which the converter permutes with the same axis move on a
    1-D view."""
    out_dim = w.shape[0]
    head_dim = out_dim // n_head
    tail = w.shape[1:]
    w = w.reshape(n_head, head_dim // 4, 2, 2, *tail)
    w = np.swapaxes(w, 1, 2)
    return w.reshape(out_dim, *tail)


def _gguf_qkv(native_q, native_k, native_v):
    """The fused qkv stack as the converter writes it: Q/K permuted, V raw."""
    return np.concatenate(
        [_converter_permute(native_q, N_HEAD),
         _converter_permute(native_k, N_HEAD),
         native_v], axis=0)


@pytest.mark.parametrize("tail", [(IN,), ()], ids=["weight", "bias"])
def test_qkv_unpermute_inverts_the_converter(tail):
    rng = np.random.default_rng(0)
    q, k, v = (rng.standard_normal((DIM, *tail)).astype(np.float32)
               for _ in range(3))
    native = np.concatenate([q, k, v], axis=0)

    got = np.array(_kimi_k25_qkv_split_to_interleaved(
        mx.array(_gguf_qkv(q, k, v)), N_HEAD))

    assert np.array_equal(got, native)
    # The permutation is not a no-op, so a missing un-permute is a real defect
    # and not a silently-equivalent layout.
    assert not np.array_equal(_gguf_qkv(q, k, v), native)


def test_qkv_unpermute_leaves_v_alone():
    """V carries no rope, and the converter never touches it."""
    rng = np.random.default_rng(1)
    q, k, v = (rng.standard_normal((DIM, IN)).astype(np.float32)
               for _ in range(3))
    got = np.array(_kimi_k25_qkv_split_to_interleaved(
        mx.array(_gguf_qkv(q, k, v)), N_HEAD))
    assert np.array_equal(got[2 * DIM:], v)


def test_qkv_unpermute_moves_x_and_y_dims_back_together():
    """Layout check independent of the converter port: in the split form each
    head's first half is all x-axis dims, in the native form the x pair leads
    each group of four."""
    # Tag every dim with its head-local index so the move is readable.
    tags = np.tile(np.arange(HEAD_DIM, dtype=np.float32), N_HEAD)[:, None]
    split = np.concatenate([tags, tags, tags], axis=0)
    native = np.array(_kimi_k25_qkv_split_to_interleaved(
        mx.array(split), N_HEAD))[:HEAD_DIM, 0]
    # split half 0 = x pairs (0,1),(2,3); half 1 = y pairs (4,5),(6,7)
    # native = x0,x1, y0,y1, x2,x3, y2,y3
    assert list(native) == [0, 1, 4, 5, 2, 3, 6, 7]


def _fake_mmproj_arrays():
    """One block plus the top-level tensors, in GGUF (numpy) axis order."""
    rng = np.random.default_rng(2)

    def a(*shape):
        return mx.array(rng.standard_normal(shape).astype(np.float32))

    merged = DIM * 2 * 2
    arrays = {
        "v.patch_embd.weight": a(DIM, 3, 14, 14),
        "v.patch_embd.bias": a(DIM),
        "v.position_embd.weight": a(64, 64, DIM),
        "v.post_ln.weight": a(DIM),
        "v.post_ln.bias": a(DIM),
        "mm.input_norm.weight": a(DIM),
        "mm.input_norm.bias": a(DIM),
        "mm.1.weight": a(merged, merged),
        "mm.1.bias": a(merged),
        "mm.2.weight": a(16, merged),
        "mm.2.bias": a(16),
    }
    for b in range(2):
        arrays.update({
            f"v.blk.{b}.attn_qkv.weight": a(3 * DIM, DIM),
            f"v.blk.{b}.attn_qkv.bias": a(3 * DIM),
            f"v.blk.{b}.attn_out.weight": a(DIM, DIM),
            f"v.blk.{b}.attn_out.bias": a(DIM),
            f"v.blk.{b}.ln1.weight": a(DIM),
            f"v.blk.{b}.ln1.bias": a(DIM),
            f"v.blk.{b}.ln2.weight": a(DIM),
            f"v.blk.{b}.ln2.bias": a(DIM),
            f"v.blk.{b}.ffn_up.weight": a(5 * DIM, DIM),
            f"v.blk.{b}.ffn_up.bias": a(5 * DIM),
            f"v.blk.{b}.ffn_down.weight": a(DIM, 5 * DIM),
            f"v.blk.{b}.ffn_down.bias": a(DIM),
        })
    return arrays


def test_remap_covers_every_tensor_and_transposes_the_patch_conv():
    arrays = _fake_mmproj_arrays()
    out, skipped, kq = remap_vision_arrays(
        arrays, "kimi_k25", mm_meta=MM_META)

    assert skipped == [] and kq == {}
    assert len(out) == len(arrays)
    # GGUF conv [out, in, kH, kW] -> mlx Conv2d [out, kH, kW, in].
    assert out["vision_tower.patch_embed.proj.weight"].shape == (DIM, 14, 14, 3)
    # The learned position grid is already in [H, W, dim].
    assert out["vision_tower.patch_embed.pos_emb.weight"].shape == (64, 64, DIM)
    # ln1 is pre-attention (norm0), ln2 pre-MLP (norm1) - swapping them is
    # shape-compatible and therefore silent.
    assert np.array_equal(
        np.array(out["vision_tower.blocks.0.norm0.weight"]),
        np.array(arrays["v.blk.0.ln1.weight"]))
    assert np.array_equal(
        np.array(out["vision_tower.blocks.0.norm1.weight"]),
        np.array(arrays["v.blk.0.ln2.weight"]))
    # mm.2 lands on proj.2: proj.1 is the GELU, which holds no weights.
    assert "mm_projector.proj.2.weight" in out
    assert "mm_projector.proj.1.weight" not in out


def test_remap_unpermutes_only_the_qkv_stacks():
    arrays = _fake_mmproj_arrays()
    out, _, _ = remap_vision_arrays(arrays, "kimi_k25", mm_meta=MM_META)

    qkv = np.array(out["vision_tower.blocks.0.attn.wqkv.weight"])
    assert not np.array_equal(qkv, np.array(arrays["v.blk.0.attn_qkv.weight"]))
    # attn_out shares the qkv shape family but carries no rope.
    assert np.array_equal(
        np.array(out["vision_tower.blocks.0.attn.wo.weight"]),
        np.array(arrays["v.blk.0.attn_out.weight"]))


def test_remap_needs_the_head_count():
    """The de-permutation is a hyperparameter, not a shape: without mm_meta the
    remap must fail rather than emit a wrongly-ordered qkv stack."""
    with pytest.raises(ValueError, match="head_count"):
        remap_vision_arrays(_fake_mmproj_arrays(), "kimi_k25")


def test_vision_name_skips_unknown_tensors():
    assert _kimi_k25_vision_name("v.blk.0.attn_norm.weight") is None
    assert _kimi_k25_vision_name("a.blk.0.ffn_up.weight") is None


def test_config_synth_reads_the_mmproj_and_the_vocab():
    text_config = {"vocab_size": 163840, "hidden_size": 7168,
                   "model_type": "deepseek_v3"}
    llm_meta = {"tokenizer.ggml.tokens": ["a"] * 100 + ["<|media_pad|>"]}
    cfg = _synthesize_kimi_k25_vlm_config(text_config, MM_META, llm_meta)

    assert cfg["model_type"] == "kimi_k25"
    vc = cfg["vision_config"]
    assert vc["model_type"] == "moonvit"
    assert (vc["depth"], vc["num_heads"], vc["embed_dim"]) == (2, N_HEAD, DIM)
    assert vc["merge_kernel_size"] == [2, 2]
    assert (vc["init_pos_emb_height"], vc["init_pos_emb_width"]) == (64, 64)
    # The placeholder id comes from this GGUF's vocab, not mlx-vlm's default
    # (which belongs to a different Kimi release).
    assert cfg["media_placeholder_token_id"] == 100
    assert cfg["image_token_index"] == 100


def test_config_synth_without_the_placeholder_token():
    """A vocab with no <|media_pad|> leaves the id unset rather than guessing."""
    cfg = _synthesize_kimi_k25_vlm_config(
        {"vocab_size": 10}, MM_META, {"tokenizer.ggml.tokens": ["a"]})
    assert "media_placeholder_token_id" not in cfg
