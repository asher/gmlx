# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Asher Feldman
"""Hy4-flash vision tower and projector (mmproj ``hy_v4_flash``).

Reference: the Hy4-flash model code (transformers ``modeling_hy_v4_flash``)
with the checkpoint's config. The tower is a pre-norm transformer over
16x16 patches: a linear patch embed over the (C, ph, pw)-flattened patch
plus a learned 128x128 position grid resampled to the image's patch grid,
then LayerNorm -> attention (biased q/k/v/out, no rope) -> LayerNorm ->
GELU MLP per block. No class token and no final norm. Attention is full
and bidirectional over one image.

The projector merges 2x2 patch neighbourhoods: RMSNorm, a stride-2 2x2
convolution applied as a Linear over the (c, kh, kw)-flattened
neighbourhood, GELU, and a 1x1 convolution onto the text width. One learned
newline row closes every row of the merged grid, so an image of H x W
merged cells gives H * (W + 1) rows in reading order.

One image per call. The tower takes the processor's flattened patches
[N, C * 16 * 16] and the patch grid (n_h, n_w).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from gmlx._vendor.mlx_vlm.models.base import BaseModelConfig


@dataclass
class VisionConfig(BaseModelConfig):
    model_type: str = "hy_v4_flash_vl"
    depth: int = 27
    hidden_size: int = 1152
    num_heads: int = 16
    intermediate_size: int = 4304
    patch_size: int = 16
    in_channels: int = 3
    spatial_merge_size: int = 2
    merge_hidden_size: int = 4608
    out_hidden_size: int = 3072
    layer_norm_eps: float = 1e-5
    # Edge of the learned position grid (image_size / patch_size).
    position_grid: int = 128
    # The reference model code uses the erf GELU in the tower's MLP.
    # llama.cpp runs this projector's tower on the tanh form.
    gelu_tanh: bool = False


# Queries an attention call scores at once in a large image. The largest
# grid, 128 x 128 patches, holds 8.6 GB of 16-bit scores against itself;
# a tile of it holds 1.1 GB.
_Q_TILE = 2048


def _gelu_tanh(x: mx.array) -> mx.array:
    return 0.5 * x * (1 + mx.tanh(
        math.sqrt(2 / math.pi) * (x + 0.044715 * (x * x * x))))


def resampled_positions(table: mx.array, edge: int, n_h: int,
                        n_w: int) -> mx.array:
    """The learned ``edge`` x ``edge`` position grid ``table`` [edge * edge,
    D] (row-major) at an ``n_h`` x ``n_w`` patch grid, [n_h * n_w, D] fp32.

    Bilinear, corners not aligned, at the reference's scale factor of
    ``(n + 0.1) / edge`` an axis: output index d reads the source
    coordinate ``(d + 0.5) * edge / (n + 0.1) - 0.5``, held inside the
    table. That is not a resize to n at the default scale. torch's CPU
    kernel copies an axis whose size is the table's; its GPU kernels and
    llama.cpp apply the formula there too, as this does."""
    grid = table.reshape(edge, edge, -1).astype(mx.float32)

    def axis(n: int):
        src = (mx.arange(n, dtype=mx.float32) + 0.5) * (edge / (n + 0.1)) - 0.5
        src = mx.maximum(src, 0.0)
        lo = mx.minimum(mx.floor(src).astype(mx.int32), edge - 1)
        hi = mx.minimum(lo + 1, edge - 1)
        return lo, hi, mx.clip(src - lo.astype(mx.float32), 0.0, 1.0)

    r0, r1, fr = axis(n_h)
    c0, c1, fc = axis(n_w)
    fc = fc[None, :, None]
    fr = fr[:, None, None]

    def along_columns(rows: mx.array) -> mx.array:
        return rows[:, c0] * (1 - fc) + rows[:, c1] * fc

    out = along_columns(grid[r0]) * (1 - fr) + along_columns(grid[r1]) * fr
    return out.reshape(n_h * n_w, -1)


class VisionAttention(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        dim = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = dim // config.num_heads
        self.q_proj = nn.Linear(dim, dim, bias=True)
        self.k_proj = nn.Linear(dim, dim, bias=True)
        self.v_proj = nn.Linear(dim, dim, bias=True)
        self.out_proj = nn.Linear(dim, dim, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        n = x.shape[0]

        def heads(t: mx.array) -> mx.array:
            return t.reshape(1, n, self.num_heads, self.head_dim).transpose(
                0, 2, 1, 3)

        q, k, v = (heads(self.q_proj(x)), heads(self.k_proj(x)),
                   heads(self.v_proj(x)))
        scale = self.head_dim ** -0.5
        if n <= 2 * _Q_TILE:
            out = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)
        else:
            parts = []
            for i in range(0, n, _Q_TILE):
                parts.append(mx.fast.scaled_dot_product_attention(
                    q[:, :, i:i + _Q_TILE], k, v, scale=scale))
                mx.eval(parts[-1])
            out = mx.concatenate(parts, axis=2)
        return self.out_proj(out.transpose(0, 2, 1, 3).reshape(n, -1))


class VisionMLP(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=True)
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=True)
        self._act = _gelu_tanh if config.gelu_tanh else nn.gelu

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(self._act(self.up_proj(x)))


class VisionBlock(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attn = VisionAttention(config)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = VisionMLP(config)

    def __call__(self, x: mx.array) -> mx.array:
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class PatchEmbed(nn.Module):
    """The patch convolution as a Linear over the flattened patch, plus
    the position grid at the image's patch grid."""

    def __init__(self, config: VisionConfig):
        super().__init__()
        self.edge = int(config.position_grid)
        patch = config.in_channels * config.patch_size * config.patch_size
        self.proj = nn.Linear(patch, config.hidden_size, bias=True)
        self.position_embedding = mx.zeros(
            (self.edge * self.edge, config.hidden_size))

    def __call__(self, patches: mx.array, n_h: int, n_w: int) -> mx.array:
        x = self.proj(patches)
        pos = resampled_positions(self.position_embedding, self.edge, n_h, n_w)
        return x + pos.astype(x.dtype)


class Projector(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.merge_size = int(config.spatial_merge_size)
        dim = config.hidden_size
        self.pre_norm = nn.RMSNorm(dim, eps=config.layer_norm_eps)
        self.merge = nn.Linear(
            dim * self.merge_size * self.merge_size, config.merge_hidden_size,
            bias=True)
        self.out = nn.Linear(
            config.merge_hidden_size, config.out_hidden_size, bias=True)

    def __call__(self, x: mx.array, n_h: int, n_w: int) -> mx.array:
        """Patch rows [n_h * n_w, D] -> the merged grid [H, W, out]."""
        m = self.merge_size
        rows, cols = n_h // m, n_w // m
        x = self.pre_norm(x)
        # (H, kh, W, kw, c) -> (H, W, c, kh, kw): the order a convolution
        # weight [out, c, kh, kw] flattens to.
        x = x.reshape(rows, m, cols, m, -1).transpose(0, 2, 4, 1, 3)
        x = x.reshape(rows * cols, -1)
        x = self.out(nn.gelu(self.merge(x)))
        return x.reshape(rows, cols, -1)


class VisionModel(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.config = config
        self.patch_embed = PatchEmbed(config)
        self.blocks = [VisionBlock(config) for _ in range(config.depth)]
        self.projector = Projector(config)
        self.image_newline = mx.zeros((config.out_hidden_size,))

    def __call__(self, patches: mx.array, n_h: int, n_w: int) -> mx.array:
        """Flattened patches [n_h * n_w, C * p * p] of one image -> the rows
        its slots take in the prompt, [H * (W + 1), out]: each row of the
        merged grid, then the newline row."""
        m = self.projector.merge_size
        if (patches.shape[0] != n_h * n_w or n_h % m or n_w % m):
            raise ValueError(
                f"{patches.shape[0]} patches for a {n_h} x {n_w} grid merged "
                f"{m} x {m}")
        x = self.patch_embed(patches, n_h, n_w)
        for block in self.blocks:
            x = block(x)
        grid = self.projector(x, n_h, n_w)
        rows, cols = grid.shape[0], grid.shape[1]
        newline = mx.broadcast_to(
            self.image_newline.astype(grid.dtype)[None, None],
            (rows, 1, grid.shape[-1]))
        return mx.concatenate([grid, newline], axis=1).reshape(
            rows * (cols + 1), -1)
