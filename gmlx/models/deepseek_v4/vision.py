# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Asher Feldman
"""DeepSeek-V4-Flash-Vision-Exp vision tower + aligner (mmproj ``deepseek4v``).

Reference: the checkpoint's ``inference/vision.py`` (deepseek-ai/
DeepSeek-V4-Flash-Vision-Exp, HF commit 6821d6ad). The ViT is a plain
pre-norm transformer over 14x14 patches: a linear patch embed over the
(C, ph, pw)-flattened patch, RMSNorm(1e-6) -> attention (16 heads of 64,
biased q/k/v/out, 2-D rope: 16 angles from the row index and 16 from the
column index, rotate-half over the full 64) -> RMSNorm -> SwiGLU (no
biases), then a final RMSNorm. Attention is full bidirectional over one
image; there are no learned position embeddings and no CLS token.

The aligner downsamples 3x3 patch neighbourhoods: the (n_h, n_w, 1024)
grid is zero-padded to multiples of 3 and unfolded channel-major
(c, ki, kj) into 9216-wide rows, then w1 -> GELU(erf) -> w2 onto the
4096-dim text embedding. Its rows come out in row-major (block row,
block column) order; the token block assembly (``vlm_model``) permutes
them into the interleaved N-layout the text model expects.

v1 runs one image per call (B=1 serve-aligned, like the other vendored
towers). The tower takes the processor's flattened patches [N, 588] plus
the grid (n_h, n_w).
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from mlx_vlm.models.base import BaseModelConfig


@dataclass
class VisionConfig(BaseModelConfig):
    model_type: str = "deepseek_v4_vl"
    depth: int = 32
    hidden_size: int = 1024
    num_heads: int = 16
    intermediate_size: int = 2816
    patch_size: int = 14
    in_channels: int = 3
    rms_norm_eps: float = 1e-6
    out_hidden_size: int = 4096
    downsample_ratio: int = 3
    rope_theta: float = 10000.0
    # Arch constants (llama.cpp clip.cpp PROJECTOR_TYPE_DEEPSEEK4V; not GGUF
    # metadata): token budget per image, width cap, pre-resize minimum.
    max_n_token: int = 384
    max_wh_ratio: int = 8
    min_pixels: int = 147456


def vision_rope_tables(n_h: int, n_w: int, dim: int, theta: float):
    """(cos, sin) [N, dim] fp32 for a row-major n_h x n_w patch grid.
    ``dim`` is half the head dim: angle j < dim/2 rotates on the row index
    with frequency j, angle dim/2 + j on the column index with the same
    frequency (reference ``get_vision_cos_sin``)."""
    inv_freq = 1.0 / (theta ** (mx.arange(0, dim, 2, dtype=mx.float32) / dim))
    hpos = mx.repeat(mx.arange(n_h, dtype=mx.float32), n_w)
    wpos = mx.tile(mx.arange(n_w, dtype=mx.float32), n_h)
    freqs = mx.concatenate(
        [hpos[:, None] * inv_freq[None], wpos[:, None] * inv_freq[None]], axis=-1
    )
    return mx.cos(freqs), mx.sin(freqs)


def _apply_rotary(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    """``x`` [N, heads, D]; cos/sin [N, D/2]. Pairs x[..., j] with
    x[..., D/2 + j] (reference ``apply_rotary``: fp32 math)."""
    dtype = x.dtype
    xf = x.astype(mx.float32)
    half = xf.shape[-1] // 2
    x1, x2 = xf[..., :half], xf[..., half:]
    c, s = cos[:, None, :], sin[:, None, :]
    return mx.concatenate([x1 * c - x2 * s, x2 * c + x1 * s], axis=-1).astype(dtype)


class DeepseekV4VisionAttention(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        dim = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = dim // config.num_heads
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(dim, dim, bias=True)
        self.k_proj = nn.Linear(dim, dim, bias=True)
        self.v_proj = nn.Linear(dim, dim, bias=True)
        self.out_proj = nn.Linear(dim, dim, bias=True)

    def __call__(self, x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
        N = x.shape[0]
        q = self.q_proj(x).reshape(N, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(N, self.num_heads, self.head_dim)
        v = self.v_proj(x).reshape(N, self.num_heads, self.head_dim)
        q = _apply_rotary(q, cos, sin)
        k = _apply_rotary(k, cos, sin)
        # One image per call: full bidirectional attention over its patches.
        q, k, v = (t.transpose(1, 0, 2)[None] for t in (q, k, v))
        out = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=self.scale, mask=None)
        return self.out_proj(out[0].transpose(1, 0, 2).reshape(N, -1))


class DeepseekV4VisionMLP(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden, bias=False)
        self.up_proj = nn.Linear(dim, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class DeepseekV4VisionBlock(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.norm1 = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.norm2 = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attn = DeepseekV4VisionAttention(config)
        self.mlp = DeepseekV4VisionMLP(config.hidden_size, config.intermediate_size)

    def __call__(self, x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
        x = x + self.attn(self.norm1(x), cos, sin)
        return x + self.mlp(self.norm2(x))


class DeepseekV4PatchEmbed(nn.Module):
    """Linear over the (C, ph, pw)-flattened patch (the GGUF conv weight
    [out, C, ph, pw] reshapes to [out, C*ph*pw] in the same order)."""

    def __init__(self, config: VisionConfig):
        super().__init__()
        self.proj = nn.Linear(
            config.in_channels * config.patch_size * config.patch_size,
            config.hidden_size, bias=True)

    def __call__(self, patches: mx.array) -> mx.array:
        return self.proj(patches)


class DeepseekV4Aligner(nn.Module):
    """3x3 stride-3 unfold of the patch grid -> w1 -> GELU(erf) -> w2."""

    def __init__(self, config: VisionConfig):
        super().__init__()
        self.ratio = int(config.downsample_ratio)
        in_dim = config.hidden_size * self.ratio * self.ratio
        self.w1 = nn.Linear(in_dim, config.out_hidden_size, bias=True)
        self.w2 = nn.Linear(config.out_hidden_size, config.out_hidden_size, bias=True)

    def unfold(self, x: mx.array, n_h: int, n_w: int) -> mx.array:
        """[n_h * n_w, C] row-major grid -> [ceil(n_h/r) * ceil(n_w/r), C*r*r]
        rows, each the (c, ki, kj)-ordered 3x3 neighbourhood, zero padded
        at the bottom/right edge (torch ``F.pad`` + ``F.unfold``)."""
        r = self.ratio
        C = x.shape[-1]
        pad_h, pad_w = (-n_h) % r, (-n_w) % r
        g = x.reshape(n_h, n_w, C)
        if pad_h or pad_w:
            g = mx.pad(g, [(0, pad_h), (0, pad_w), (0, 0)])
        H, W = n_h + pad_h, n_w + pad_w
        g = g.reshape(H // r, r, W // r, r, C)          # (bh, ki, bw, kj, c)
        g = g.transpose(0, 2, 4, 1, 3)                   # (bh, bw, c, ki, kj)
        return g.reshape((H // r) * (W // r), C * r * r)

    def __call__(self, x: mx.array, n_h: int, n_w: int) -> mx.array:
        x = self.unfold(x, n_h, n_w)
        return self.w2(nn.gelu(self.w1(x)))


class DeepseekV4VisionModel(nn.Module):
    """ViT + aligner. ``__call__(patches, n_h, n_w)`` -> [n_llm_h * n_llm_w,
    out_hidden_size] aligner rows in row-major grid order."""

    def __init__(self, config: VisionConfig):
        super().__init__()
        self.config = config
        self.rope_dim = config.hidden_size // config.num_heads // 2
        self.patch_embed = DeepseekV4PatchEmbed(config)
        self.blocks = [DeepseekV4VisionBlock(config) for _ in range(config.depth)]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.aligner = DeepseekV4Aligner(config)

    def encode(self, patches: mx.array, n_h: int, n_w: int) -> mx.array:
        """Tower only: [N, hidden] post-norm patch features."""
        x = self.patch_embed(patches)
        cos, sin = vision_rope_tables(n_h, n_w, self.rope_dim, self.config.rope_theta)
        for block in self.blocks:
            x = block(x, cos, sin)
        return self.norm(x)

    def __call__(self, patches: mx.array, n_h: int, n_w: int) -> mx.array:
        return self.aligner(self.encode(patches, n_h, n_w), n_h, n_w)
