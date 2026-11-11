# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Asher Feldman
"""Vision tower and projector of LFM3-VL and LFM3-Omni (mmproj projector
``lfm3vl``).

Ported from Liquid AI's ``modeling_lfm3_vl.py`` (Apache-2.0) and checked
against llama.cpp ``tools/mtmd/models/lfm3vl.cpp`` (MIT, contributed by
Liquid AI):

- a bias-free linear patchifier over ``channels * 2 * patch * patch``
  values, then RMSNorm. The two temporal slices are the two patch kernels
  of the GGUF. A still image fills both slices;
- gated depthwise 3x3 convolution layers and bidirectional grouped-query
  attention layers, each with a SwiGLU feed-forward and pre-norms;
- attention over the patches of one frame only, with RMSNorm on the query
  and key heads and an axial 2-D RoPE: the first quarter of the rotation
  pairs turn with the row and the next quarter with the column;
- no final norm. The projector applies LayerNorm, joins each 2x2 block of
  patches and runs a GELU MLP into the text width.

Patches arrive grouped by merge block, as the processor writes them, so
the projector's join is a reshape. The convolution layers move a frame to
raster order and back.

The tower runs in the activation dtype on 16-bit weights. Norms, RoPE and
the convolution sum run in float32.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import List, Sequence, Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from gmlx._vendor.mlx_vlm.models.base import BaseModelConfig

from .model import _Norm

CONV = "conv"
FULL = "full_attention"

# Query rows per attention call. A 4,096-token image has 16,384 patches,
# and one call over all of them holds the GPU for too long.
Q_BLOCK = 2048
# Patches in a pass from which each layer is evaluated as it finishes.
# Unevaluated, the layers of a large image overlap in memory.
EVAL_PATCHES = 4096
# Patches of a clip that run in one pass. The slices of a clip do not see
# each other, so a clip runs a few slices at a time and its memory does
# not grow with its length.
CLIP_PATCHES = 4096


def _default_layer_types() -> List[str]:
    return [CONV, CONV, FULL, CONV, FULL, CONV, FULL, CONV, FULL, CONV,
            FULL, CONV, FULL, CONV]


@dataclass
class VisionConfig(BaseModelConfig):
    model_type: str = "lfm3_vl_vision"
    hidden_size: int = 1024
    intermediate_size: int = 2560
    num_hidden_layers: int = 14
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    in_channels: int = 3
    patch_size: int = 16
    temporal_patch_size: int = 2
    spatial_merge_size: int = 2
    conv_kernel_size: int = 3
    norm_eps: float = 1e-5
    rope_theta: float = 1000000.0
    layer_types: List[str] = field(default_factory=_default_layer_types)
    projector_hidden_size: int = 4096
    projection_dim: int = 2048
    projector_layer_norm_eps: float = 1e-6
    projector_bias: bool = True


@lru_cache(maxsize=64)
def _rope_tables(grid_h: int, grid_w: int, merge: int, head_dim: int,
                 theta: float) -> Tuple[mx.array, mx.array]:
    """``cos`` and ``sin``, each ``[grid_h * grid_w, head_dim]`` float32,
    for the patches of one frame in merge-block order."""
    spatial = head_dim // 2
    inv_freq = 1.0 / (theta ** (np.arange(0, spatial, 2, dtype=np.float32)
                                / np.float32(spatial)))
    rows, cols = np.meshgrid(np.arange(grid_h), np.arange(grid_w),
                             indexing="ij")
    block = (grid_h // merge, merge, grid_w // merge, merge)
    rows = rows.reshape(block).transpose(0, 2, 1, 3).reshape(-1)
    cols = cols.reshape(block).transpose(0, 2, 1, 3).reshape(-1)
    freq_h = rows[:, None].astype(np.float32) * inv_freq.astype(np.float32)
    freq_w = cols[:, None].astype(np.float32) * inv_freq.astype(np.float32)
    freq = np.concatenate([freq_h, freq_w], axis=-1)
    freq = np.concatenate([freq, freq], axis=-1)
    return mx.array(np.cos(freq)), mx.array(np.sin(freq))


def _rotate(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    """``x`` is ``[frames, heads, patches, head_dim]``. The rotation runs
    in float32, as the reference does."""
    dtype = x.dtype
    x = x.astype(mx.float32)
    half = x.shape[-1] // 2
    turned = mx.concatenate([-x[..., half:], x[..., :half]], axis=-1)
    return (x * cos + turned * sin).astype(dtype)


class VisionAttention(nn.Module):
    """Bidirectional grouped-query attention over the patches of one
    frame."""

    def __init__(self, config: VisionConfig):
        super().__init__()
        dim = config.hidden_size
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        self.head_dim = dim // self.n_heads
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(dim, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim,
                                bias=False)
        self.v_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim,
                                bias=False)
        self.out_proj = nn.Linear(self.n_heads * self.head_dim, dim,
                                  bias=False)
        self.q_layernorm = _Norm(self.head_dim, eps=config.norm_eps)
        self.k_layernorm = _Norm(self.head_dim, eps=config.norm_eps)

    def __call__(self, x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
        frames, n, _ = x.shape

        def heads(t: mx.array, count: int) -> mx.array:
            return t.reshape(frames, n, count, self.head_dim).transpose(
                0, 2, 1, 3)

        q = _rotate(self.q_layernorm(heads(self.q_proj(x), self.n_heads)),
                    cos, sin)
        k = _rotate(self.k_layernorm(heads(self.k_proj(x), self.n_kv_heads)),
                    cos, sin)
        v = heads(self.v_proj(x), self.n_kv_heads)
        outs = []
        for q0 in range(0, n, Q_BLOCK):
            out = mx.fast.scaled_dot_product_attention(
                q[:, :, q0:q0 + Q_BLOCK], k, v, scale=self.scale)
            if n > Q_BLOCK:
                mx.eval(out)
            outs.append(out)
        out = outs[0] if len(outs) == 1 else mx.concatenate(outs, axis=2)
        return self.out_proj(out.transpose(0, 2, 1, 3).reshape(frames, n, -1))


class DepthwiseConv2d(nn.Module):
    """A depthwise convolution with zero padding that keeps the grid.
    ``weight`` is ``[channels, k, k, 1]``, the layout of an MLX
    convolution with one group per channel. It runs in float32."""

    def __init__(self, channels: int, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.weight = mx.zeros((channels, kernel_size, kernel_size, 1))

    def __call__(self, x: mx.array) -> mx.array:
        """``x`` is ``[frames, height, width, channels]``."""
        out = mx.conv2d(
            x.astype(mx.float32), self.weight.astype(mx.float32),
            padding=self.kernel_size // 2, groups=x.shape[-1])
        return out.astype(x.dtype)


class VisionConv2d(nn.Module):
    """Gated depthwise convolution over the patch grid of each frame."""

    def __init__(self, config: VisionConfig):
        super().__init__()
        dim = config.hidden_size
        self.merge = config.spatial_merge_size
        self.in_proj = nn.Linear(dim, 3 * dim, bias=False)
        self.spatial_conv = DepthwiseConv2d(dim, config.conv_kernel_size)
        self.out_proj = nn.Linear(dim, dim, bias=False)

    def __call__(self, x: mx.array, grid: Tuple[int, int]) -> mx.array:
        frames, n, dim = x.shape
        height, width = grid
        m = self.merge
        b, c, z = mx.split(self.in_proj(x), 3, axis=-1)
        gated = b * z
        # Merge-block order to the raster grid, and back.
        grid_x = gated.reshape(frames, height // m, width // m, m, m, dim)
        grid_x = grid_x.transpose(0, 1, 3, 2, 4, 5).reshape(
            frames, height, width, dim)
        conv = self.spatial_conv(grid_x)
        conv = conv.reshape(frames, height // m, m, width // m, m, dim)
        conv = conv.transpose(0, 1, 3, 2, 4, 5).reshape(frames, n, dim)
        return self.out_proj(c * conv)


class VisionMLP(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        dim, hidden = config.hidden_size, config.intermediate_size
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w3 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(hidden, dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.w2(nn.silu(self.w1(x)) * self.w3(x))


class VisionLayer(nn.Module):
    def __init__(self, config: VisionConfig, layer_idx: int):
        super().__init__()
        self.is_attention_layer = config.layer_types[layer_idx] == FULL
        if self.is_attention_layer:
            self.self_attn = VisionAttention(config)
        else:
            self.conv = VisionConv2d(config)
        self.feed_forward = VisionMLP(config)
        self.operator_norm = _Norm(config.hidden_size, eps=config.norm_eps)
        self.ffn_norm = _Norm(config.hidden_size, eps=config.norm_eps)

    def __call__(self, x: mx.array, grid: Tuple[int, int], cos: mx.array,
                 sin: mx.array) -> mx.array:
        if self.is_attention_layer:
            x = x + self.self_attn(self.operator_norm(x), cos, sin)
        else:
            x = x + self.conv(self.operator_norm(x), grid)
        return x + self.feed_forward(self.ffn_norm(x))


class VisionModel(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.config = config
        patch_dim = (config.in_channels * config.temporal_patch_size
                     * config.patch_size ** 2)
        self.patchifier = nn.Linear(patch_dim, config.hidden_size, bias=False)
        self.embedding_norm = _Norm(config.hidden_size, eps=config.norm_eps)
        self.layers = [VisionLayer(config, i)
                       for i in range(config.num_hidden_layers)]
        # The stream follows the patch weights' dtype unless this is set.
        # Float32 here runs the 16-bit weights as llama.cpp does.
        self.stream_dtype = None

    def __call__(self, patches: mx.array, grid_thw: Sequence[int]) -> mx.array:
        """``patches`` is ``[t * h * w, patch_dim]``: the flattened patches
        of one image or video, frame by frame, each frame grouped by merge
        block. Returns the last hidden state, ``[t * h * w, hidden]``."""
        cfg = self.config
        frames, height, width = (int(v) for v in grid_thw)
        n = height * width
        cos, sin = _rope_tables(
            height, width, cfg.spatial_merge_size,
            cfg.hidden_size // cfg.num_attention_heads, float(cfg.rope_theta))
        x = self.patchifier(patches.astype(
            self.stream_dtype or self.patchifier.weight.dtype))
        x = self.embedding_norm(x).reshape(frames, n, cfg.hidden_size)
        settle = frames * n >= EVAL_PATCHES
        for layer in self.layers:
            x = layer(x, (height, width), cos, sin)
            if settle:
                mx.eval(x)
        return x.reshape(frames * n, cfg.hidden_size)


class MultiModalProjector(nn.Module):
    """Joins each block of ``merge * merge`` patches into one text token."""

    def __init__(self, config: VisionConfig):
        super().__init__()
        dim = config.hidden_size
        self.block = config.spatial_merge_size ** 2
        self.layer_norm = nn.LayerNorm(
            dim, eps=config.projector_layer_norm_eps)
        self.linear_1 = nn.Linear(dim * self.block,
                                  config.projector_hidden_size,
                                  bias=config.projector_bias)
        self.linear_2 = nn.Linear(config.projector_hidden_size,
                                  config.projection_dim,
                                  bias=config.projector_bias)

    def __call__(self, x: mx.array) -> mx.array:
        dtype = x.dtype
        x = mx.fast.layer_norm(
            x.astype(mx.float32), self.layer_norm.weight.astype(mx.float32),
            self.layer_norm.bias.astype(mx.float32),
            self.layer_norm.eps).astype(dtype)
        x = x.reshape(-1, x.shape[-1] * self.block)
        return self.linear_2(nn.gelu(self.linear_1(x)))
