# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Asher Feldman
"""Vision tower for Nemotron-3.5 Bolt and Super VL (mmproj projector
``nemotron_omni``): a C-RADIO ViT and the RMSNorm / relu^2 projector.

The reference is llama.cpp ``tools/mtmd/models/nemotron-omni.cpp`` with
``clip_graph::build_vit``, ``resize_position_embeddings`` and
``build_patch_merge_permute`` from ``clip.cpp``:

- a 16x16 patch embed with no bias, plus the 128x128 position table resized
  to the patch grid (bilinear, half-pixel centers, edge clamp, no
  antialias, no crop);
- the register tokens prepended after the position add, and dropped after
  the last block;
- pre-norm blocks with biased LayerNorms, biased q/k/v/out projections
  (one fused ``[q | k | v]`` projection in the Super VL file), full
  attention over the image, and an exact (erf) GELU MLP;
- a final biased LayerNorm over every token when the file has ``post_ln``;
- a 2x2 pixel shuffle (channel ``c + C*dx + 2*C*dy``), then RMSNorm and a
  bias-free relu^2 MLP into the text width.

The tower runs in float32 against the mmproj's BF16 weights, as llama.cpp
does. Q8_0 weights run on the kquant modules. Each image runs on its own.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from gmlx._vendor.mlx_vlm.models.base import BaseModelConfig


@dataclass
class VisionConfig(BaseModelConfig):
    model_type: str = "nemotron_h_omni"
    hidden_size: int = 1280
    intermediate_size: int = 5120
    num_hidden_layers: int = 32
    num_attention_heads: int = 16
    patch_size: int = 16
    num_channels: int = 3
    layer_norm_eps: float = 1e-6
    num_registers: int = 10
    position_grid: int = 128
    merge: int = 2
    projector_hidden_size: int = 20480
    projection_dim: int = 4096
    projector_norm_eps: float = 1e-5
    fused_qkv: bool = False
    post_norm: bool = False


def _axis_weights(n_out: int, n_in: int) -> np.ndarray:
    """ggml's bilinear upscale weights along one axis, ``[n_out, n_in]``:
    half-pixel sample points, edge indices clamped, the fraction clamped to
    [0, 1], in float32 as ggml computes them."""
    w = np.zeros((n_out, n_in), dtype=np.float32)
    sf = np.float32(n_out) / np.float32(n_in)
    for i in range(n_out):
        x = (np.float32(i) + np.float32(0.5)) / sf - np.float32(0.5)
        x0 = int(np.floor(x))
        x1 = x0 + 1
        x0 = min(max(x0, 0), n_in - 1)
        x1 = min(max(x1, 0), n_in - 1)
        d = float(np.clip(x - np.float32(x0), 0.0, 1.0))
        w[i, x0] += 1.0 - d
        w[i, x1] += d
    return w


@lru_cache(maxsize=32)
def _resize_weights(grid_h: int, grid_w: int, side: int):
    return (mx.array(_axis_weights(grid_h, side)),
            mx.array(_axis_weights(grid_w, side)))


def resize_positions(table: mx.array, grid_h: int, grid_w: int,
                     side: int) -> mx.array:
    """The ``[side*side, C]`` table (row ``y*side + x``) resized to
    ``[grid_h*grid_w, C]`` in patch order."""
    c = table.shape[-1]
    t = table.astype(mx.float32)
    if grid_h == side and grid_w == side:
        return t
    ry, rx = _resize_weights(grid_h, grid_w, side)
    t = (ry @ t.reshape(side, side * c)).reshape(grid_h, side, c)
    t = mx.swapaxes(t, 1, 2) @ rx.T                    # [gh, C, gw]
    return mx.swapaxes(t, 1, 2).reshape(grid_h * grid_w, c)


class Attention(nn.Module):
    def __init__(self, dims: int, num_heads: int, fused_qkv: bool = False):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dims // num_heads) ** -0.5
        if fused_qkv:
            self.qkv_proj = nn.Linear(dims, 3 * dims, bias=True)
        else:
            self.q_proj = nn.Linear(dims, dims, bias=True)
            self.k_proj = nn.Linear(dims, dims, bias=True)
            self.v_proj = nn.Linear(dims, dims, bias=True)
        self.out_proj = nn.Linear(dims, dims, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        n, d = x.shape
        h = self.num_heads

        def heads(t):
            return t.reshape(1, n, h, d // h).transpose(0, 2, 1, 3)

        if "qkv_proj" in self:
            q, k, v = mx.split(self.qkv_proj(x), 3, axis=-1)
        else:
            q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        o = mx.fast.scaled_dot_product_attention(
            heads(q), heads(k), heads(v), scale=self.scale)
        return self.out_proj(o.transpose(0, 2, 1, 3).reshape(n, d))


class MLP(nn.Module):
    def __init__(self, dims: int, hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(dims, hidden, bias=True)
        self.fc2 = nn.Linear(hidden, dims, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        return self.fc2(nn.gelu(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        d = config.hidden_size
        self.norm1 = nn.LayerNorm(d, eps=config.layer_norm_eps)
        self.attn = Attention(d, config.num_attention_heads, config.fused_qkv)
        self.norm2 = nn.LayerNorm(d, eps=config.layer_norm_eps)
        self.mlp = MLP(d, config.intermediate_size)

    def __call__(self, x: mx.array) -> mx.array:
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class Projector(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        merged = config.hidden_size * config.merge * config.merge
        self.norm = nn.RMSNorm(merged, eps=config.projector_norm_eps)
        self.fc1 = nn.Linear(merged, config.projector_hidden_size, bias=False)
        self.fc2 = nn.Linear(config.projector_hidden_size,
                             config.projection_dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.fc2(nn.relu2(self.fc1(self.norm(x))))


def pixel_shuffle(x: mx.array, grid_h: int, grid_w: int,
                  merge: int) -> mx.array:
    """``[gh*gw, C]`` patch rows to ``[(gh/m)*(gw/m), m*m*C]``, each row
    holding a m x m block with channel ``c + C*dx + m*C*dy``. An odd edge
    pads with zeros, as ``build_patch_merge_permute`` does."""
    c = x.shape[-1]
    x = x.reshape(grid_h, grid_w, c)
    ph, pw = (-grid_h) % merge, (-grid_w) % merge
    if ph or pw:
        x = mx.pad(x, [(0, ph), (0, pw), (0, 0)])
    h, w = (grid_h + ph) // merge, (grid_w + pw) // merge
    x = x.reshape(h, merge, w, merge, c).transpose(0, 2, 1, 3, 4)
    return x.reshape(h * w, merge * merge * c)


class VisionModel(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.config = config
        d = config.hidden_size
        p = config.patch_size
        self.patch_embed = nn.Linear(config.num_channels * p * p, d, bias=False)
        self.position_embedding = mx.zeros(
            (config.position_grid * config.position_grid, d))
        self.registers = mx.zeros((config.num_registers, d))
        self.blocks = [Block(config) for _ in range(config.num_hidden_layers)]
        if config.post_norm:
            self.post_norm = nn.LayerNorm(d, eps=config.layer_norm_eps)
        self.projector = Projector(config)

    def __call__(self, patches: mx.array,
                 grid: Tuple[int, int]) -> mx.array:
        """``patches``: ``[gh*gw, C*p*p]`` in row-major patch order, each
        flattened channel-major (``c, y, x``). Returns the projected
        ``[(gh/2)*(gw/2), projection_dim]`` features."""
        gh, gw = (int(v) for v in grid)
        cfg = self.config
        x = self.patch_embed(patches.astype(mx.float32))
        x = x + resize_positions(self.position_embedding, gh, gw,
                                 cfg.position_grid)
        x = mx.concatenate(
            [self.registers.astype(x.dtype), x], axis=0)
        for block in self.blocks:
            x = block(x)
        if "post_norm" in self:
            x = self.post_norm(x)
        x = x[cfg.num_registers:]
        return self.projector(pixel_shuffle(x, gh, gw, cfg.merge))
