# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Asher Feldman
"""Inkling 2 vision tower and projector (mmproj ``inkling2v``).

Reference: llama.cpp ``tools/mtmd/models/inkling2v.cpp`` with
``mtmd-graph.cpp`` (``build_vit``, ``build_rope_2d``,
``build_patch_merge_permute``) and the position input in
``mtmd-subcontext.cpp``.

A pre-norm ViT over 20 px patches: a patch convolution with no bias, the
learned 32 x 32 position grid resampled bilinearly to the patch grid, a
class token in front, then LayerNorm. Each block is LayerNorm, attention
(biased q/k/v/out, 2D rope), a layer scale, the residual, LayerNorm, a
GELU MLP, a layer scale and the residual. There is no final norm.

The 2D rope turns adjacent pairs: the first half of each head by the
column and the second half by the row. The class token sits at (0, 0) and
the patch at column x, row y at (x + 1, y + 1).

The class token is dropped, each 2 x 2 patch neighbourhood is concatenated
in the order (x, y), (x + 1, y), (x, y + 1), (x + 1, y + 1), and a two
layer GELU MLP with no biases projects onto the text width.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from gmlx._vendor.mlx_vlm.models.base import BaseModelConfig


@dataclass
class VisionConfig(BaseModelConfig):
    model_type: str = "inkling2_vl"
    depth: int = 23
    hidden_size: int = 1024
    num_heads: int = 16
    intermediate_size: int = 4096
    patch_size: int = 20
    in_channels: int = 3
    spatial_merge_size: int = 2
    merge_hidden_size: int = 4096
    out_hidden_size: int = 2304
    layer_norm_eps: float = 1e-5
    # Edge of the learned position grid.
    position_grid: int = 32
    rope_theta: float = 10000.0


# Queries one attention call scores at once in a large image.
_Q_TILE = 2048


def resampled_positions(table: mx.array, edge: int, n_h: int,
                        n_w: int) -> mx.array:
    """The ``edge`` x ``edge`` position grid ``table`` [edge * edge, D]
    (row-major) at an ``n_h`` x ``n_w`` patch grid, [n_h * n_w, D] float32.

    Bilinear with half-pixel centres and edge clamping, as ggml's
    ``ggml_interpolate`` in bilinear mode: output index d reads source
    coordinate ``(d + 0.5) * edge / n - 0.5``."""
    grid = table.reshape(edge, edge, -1).astype(mx.float32)
    if n_h == edge and n_w == edge:
        return grid.reshape(edge * edge, -1)

    def axis(n: int):
        src = (mx.arange(n, dtype=mx.float32) + 0.5) * (edge / n) - 0.5
        lo = mx.floor(src).astype(mx.int32)
        hi = mx.clip(lo + 1, 0, edge - 1)
        lo = mx.clip(lo, 0, edge - 1)
        frac = mx.clip(src - lo.astype(mx.float32), 0.0, 1.0)
        return lo, hi, frac

    r0, r1, fr = axis(n_h)
    c0, c1, fc = axis(n_w)
    fc = fc[None, :, None]
    fr = fr[:, None, None]

    def along_columns(rows: mx.array) -> mx.array:
        return rows[:, c0] * (1 - fc) + rows[:, c1] * fc

    out = along_columns(grid[r0]) * (1 - fr) + along_columns(grid[r1]) * fr
    return out.reshape(n_h * n_w, -1)


def rope_positions(n_h: int, n_w: int) -> tuple[mx.array, mx.array]:
    """(column, row) rope positions of the class token and the patches."""
    xs = mx.tile(mx.arange(n_w, dtype=mx.int32), n_h) + 1
    ys = mx.repeat(mx.arange(n_h, dtype=mx.int32), n_w) + 1
    zero = mx.zeros((1,), dtype=mx.int32)
    return mx.concatenate([zero, xs]), mx.concatenate([zero, ys])


def _rope_tables(pos: mx.array, dims: int, base: float):
    """cos and sin [L, dims / 2] for a rope over ``dims`` dims: pair j
    turns at ``base ** (-2j / dims)``."""
    inv = mx.exp(-mx.arange(dims // 2, dtype=mx.float32)
                 * (math.log(base) * 2.0 / dims))
    theta = pos.astype(mx.float32)[:, None] * inv[None, :]
    return mx.cos(theta), mx.sin(theta)


def _rotate_pairs(v: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    """Rotate adjacent pairs of [B, H, L, D] by tables [L, D / 2]."""
    B, H, L, D = v.shape
    v = v.reshape(B, H, L, D // 2, 2)
    x0, x1 = v[..., 0], v[..., 1]
    c, s = cos.astype(v.dtype), sin.astype(v.dtype)
    return mx.stack([x0 * c - x1 * s, x0 * s + x1 * c],
                    axis=-1).reshape(B, H, L, D)


def rope_2d(x: mx.array, tables_w, tables_h) -> mx.array:
    half = x.shape[-1] // 2
    return mx.concatenate([_rotate_pairs(x[..., :half], *tables_w),
                           _rotate_pairs(x[..., half:], *tables_h)], axis=-1)


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

    def __call__(self, x: mx.array, tables_w, tables_h) -> mx.array:
        n = x.shape[0]

        def heads(t: mx.array) -> mx.array:
            return t.reshape(1, n, self.num_heads, self.head_dim).transpose(
                0, 2, 1, 3)

        q = rope_2d(heads(self.q_proj(x)), tables_w, tables_h)
        k = rope_2d(heads(self.k_proj(x)), tables_w, tables_h)
        v = heads(self.v_proj(x))
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

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.gelu(self.up_proj(x)))


class VisionBlock(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        dim = config.hidden_size
        self.norm1 = nn.LayerNorm(dim, eps=config.layer_norm_eps)
        self.attn = VisionAttention(config)
        self.ls1 = mx.ones((dim,))
        self.norm2 = nn.LayerNorm(dim, eps=config.layer_norm_eps)
        self.mlp = VisionMLP(config)
        self.ls2 = mx.ones((dim,))

    def __call__(self, x: mx.array, tables_w, tables_h) -> mx.array:
        x = x + self.attn(self.norm1(x), tables_w, tables_h) * self.ls1
        return x + self.mlp(self.norm2(x)) * self.ls2


def merge_patches(x: mx.array, n_h: int, n_w: int, m: int) -> mx.array:
    """Patch rows [n_h * n_w, D] in row-major order -> one row per m x m
    neighbourhood, [(n_h / m) * (n_w / m), m * m * D], row-major. A row
    holds (x, y), (x + 1, y), (x, y + 1), (x + 1, y + 1) for m = 2."""
    rows, cols = n_h // m, n_w // m
    x = x.reshape(rows, m, cols, m, -1).transpose(0, 2, 1, 3, 4)
    return x.reshape(rows * cols, -1)


class Projector(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        m = config.spatial_merge_size
        self.merge_size = m
        self.fc1 = nn.Linear(config.hidden_size * m * m,
                             config.merge_hidden_size, bias=False)
        self.fc2 = nn.Linear(config.merge_hidden_size,
                             config.out_hidden_size, bias=False)

    def __call__(self, x: mx.array, n_h: int, n_w: int) -> mx.array:
        """Patch rows [n_h * n_w, D] in row-major order -> merged rows
        [(n_h / m) * (n_w / m), out] in row-major order."""
        x = merge_patches(x, n_h, n_w, self.merge_size)
        return self.fc2(nn.gelu(self.fc1(x)))


class VisionModel(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.config = config
        dim = config.hidden_size
        self.edge = int(config.position_grid)
        patch = config.in_channels * config.patch_size * config.patch_size
        self.patch_embed = nn.Linear(patch, dim, bias=False)
        self.position_embedding = mx.zeros((self.edge * self.edge, dim))
        self.class_embedding = mx.zeros((dim,))
        self.class_position_embedding = mx.zeros((dim,))
        self.pre_norm = nn.LayerNorm(dim, eps=config.layer_norm_eps)
        self.blocks = [VisionBlock(config) for _ in range(config.depth)]
        self.projector = Projector(config)

    def patches(self, image: mx.array) -> tuple[mx.array, int, int]:
        """A preprocessed image [C, H, W] -> its flattened patches
        [n_h * n_w, C * p * p] in row-major order, and (n_h, n_w)."""
        p = self.config.patch_size
        c, h, w = image.shape
        n_h, n_w = h // p, w // p
        x = image.reshape(c, n_h, p, n_w, p).transpose(1, 3, 0, 2, 4)
        return x.reshape(n_h * n_w, c * p * p), n_h, n_w

    def __call__(self, image: mx.array) -> mx.array:
        """One preprocessed image [C, H, W], both edges a multiple of
        patch_size * merge_size -> its rows in the prompt, [N, out]."""
        m = self.projector.merge_size
        patches, n_h, n_w = self.patches(image)
        if n_h % m or n_w % m or n_h * n_w == 0:
            raise ValueError(
                f"inkling2 vision: a {n_h} x {n_w} patch grid does not merge "
                f"{m} x {m}")
        x = self.patch_embed(patches)
        x = x + resampled_positions(
            self.position_embedding, self.edge, n_h, n_w).astype(x.dtype)
        cls = (self.class_embedding + self.class_position_embedding).astype(
            x.dtype)
        x = mx.concatenate([cls[None], x], axis=0)
        x = self.pre_norm(x)
        pos_w, pos_h = rope_positions(n_h, n_w)
        dims = (self.config.hidden_size // self.config.num_heads) // 2
        tables_w = _rope_tables(pos_w, dims, self.config.rope_theta)
        tables_h = _rope_tables(pos_h, dims, self.config.rope_theta)
        for block in self.blocks:
            x = block(x, tables_w, tables_h)
        return self.projector(x[1:], n_h, n_w)
