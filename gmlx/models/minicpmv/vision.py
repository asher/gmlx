# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 OpenBMB and the HuggingFace Inc. team.
# Ported from transformers src/transformers/models/minicpmv4_6/
# modeling_minicpmv4_6.py and src/transformers/vision_utils.py at commit
# 674c0e3 (the SigLIP tower, the window-attention vit_merger and the
# DownsampleMLP merger), by way of the mlx-vlm 0.7.3 minicpmv4_6 port (MIT),
# and modified for gmlx (Apache-2.0; see licenses/transformers-LICENSE).
"""MiniCPM-V 4.6 / 4.7 vision tower.

Each crop (the overview or one slice) is encoded on its own: patches carry
a learned position looked up on a 70 x 70 grid, the SigLIP blocks attend
within the crop, and in 16x mode a window-attention merger after block
``insert_layer_id`` folds every 2x2 patch window into one token. The final
merger folds another 2x2 and projects to the text width.

Crops of the same size run as one batch.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from gmlx._vendor.mlx_vlm.models.base import BaseModelConfig


@dataclass
class VisionConfig(BaseModelConfig):
    model_type: str = "minicpmv4_7_vision"
    hidden_size: int = 1152
    intermediate_size: int = 4304
    num_hidden_layers: int = 27
    num_attention_heads: int = 16
    num_channels: int = 3
    image_size: int = 448
    patch_size: int = 14
    position_grid: int = 70
    layer_norm_eps: float = 1e-6
    insert_layer_id: int = 6
    downsample_mode: str = "16x"
    window_intermediate_size: int = 17216
    window_kernel_size: list = field(default_factory=lambda: [2, 2])
    merge_kernel_size: list = field(default_factory=lambda: [2, 2])
    out_hidden_size: int = 2048

    @property
    def use_vit_merger(self) -> bool:
        return self.downsample_mode != "4x"

    @property
    def merge_factor(self) -> int:
        """Patches per side folded into one text token."""
        return 4 if self.use_vit_merger else 2


def position_bucket_ids(h: int, w: int, side: int) -> np.ndarray:
    """Learned-position rows of an ``h x w`` crop, row-major.

    The reference buckets ``i / h`` with ``torch.bucketize`` on
    ``arange(1/side, 1, 1/side)``; the exact integer form is
    ``(side * i) // h``.
    """
    bh = (side * np.arange(h)) // h
    bw = (side * np.arange(w)) // w
    return (bh[:, None] * side + bw[None, :]).reshape(-1)


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(dim, dim, bias=True)
        self.k_proj = nn.Linear(dim, dim, bias=True)
        self.v_proj = nn.Linear(dim, dim, bias=True)
        self.out_proj = nn.Linear(dim, dim, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        b, n, d = x.shape

        def heads(t):
            return t.reshape(b, n, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        out = mx.fast.scaled_dot_product_attention(
            heads(self.q_proj(x)), heads(self.k_proj(x)), heads(self.v_proj(x)),
            scale=self.scale)
        return self.out_proj(out.transpose(0, 2, 1, 3).reshape(b, n, d))


class MLP(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden, bias=True)
        self.fc2 = nn.Linear(hidden, dim, bias=True)
        self.act = nn.GELU(approx="tanh")

    def __call__(self, x: mx.array) -> mx.array:
        return self.fc2(self.act(self.fc1(x)))


class EncoderLayer(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        d = config.hidden_size
        self.layer_norm1 = nn.LayerNorm(d, eps=config.layer_norm_eps)
        self.self_attn = Attention(d, config.num_attention_heads)
        self.layer_norm2 = nn.LayerNorm(d, eps=config.layer_norm_eps)
        self.mlp = MLP(d, config.intermediate_size)

    def __call__(self, x: mx.array) -> mx.array:
        x = x + self.self_attn(self.layer_norm1(x))
        return x + self.mlp(self.layer_norm2(x))


class Encoder(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.layers = [EncoderLayer(config) for _ in range(config.num_hidden_layers)]


class Embeddings(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.patch_size = config.patch_size
        self.side = config.position_grid
        self.patch_embedding = nn.Conv2d(
            config.num_channels, config.hidden_size,
            kernel_size=config.patch_size, stride=config.patch_size, bias=True)
        self.position_embedding = nn.Embedding(
            config.position_grid ** 2, config.hidden_size)

    def __call__(self, patches: mx.array, h: int, w: int) -> mx.array:
        """``patches [B, h * w, p * p * C]`` in ``(kh, kw, C)`` order."""
        weight = self.patch_embedding.weight            # [D, kh, kw, C]
        x = patches @ weight.reshape(weight.shape[0], -1).T
        x = x + self.patch_embedding.bias
        pos = mx.array(position_bucket_ids(h, w, self.side), dtype=mx.int32)
        return x + self.position_embedding(pos)[None]


class VitMerger(nn.Module):
    """Window self-attention over each 2x2 patch window, then a 4-to-1
    MLP fold plus the window mean. Windows come out row-major, which is the
    downsampled grid's row-major order."""

    def __init__(self, config: VisionConfig):
        super().__init__()
        d = config.hidden_size
        kh, kw = config.window_kernel_size
        self.kernel = (int(kh), int(kw))
        group = d * self.kernel[0] * self.kernel[1]
        self.layer_norm1 = nn.LayerNorm(d, eps=config.layer_norm_eps)
        self.self_attn = Attention(d, config.num_attention_heads)
        self.pre_norm = nn.LayerNorm(group, eps=config.layer_norm_eps)
        self.linear_1 = nn.Linear(group, config.window_intermediate_size, bias=True)
        self.linear_2 = nn.Linear(config.window_intermediate_size, d, bias=True)
        self.act = nn.GELU(approx="tanh")

    def __call__(self, x: mx.array, h: int, w: int) -> mx.array:
        b, _, d = x.shape
        kh, kw = self.kernel
        if h % kh or w % kw:
            raise ValueError(f"MiniCPM-V vit_merger: crop grid {(h, w)} is not "
                             f"divisible by {self.kernel}")

        def windows(t):
            t = t.reshape(b, h // kh, kh, w // kw, kw, d).transpose(0, 1, 3, 2, 4, 5)
            return t.reshape(b * (h // kh) * (w // kw), kh * kw, d)

        win = windows(x)
        win = win + self.self_attn(windows(self.layer_norm1(x)))
        mean = win.mean(axis=1)
        folded = self.linear_2(self.act(self.linear_1(
            self.pre_norm(win.reshape(win.shape[0], -1)))))
        return (folded + mean).reshape(b, -1, d)


class MergerBlock(nn.Module):
    def __init__(self, dim: int, out_dim: int):
        super().__init__()
        self.pre_norm = nn.LayerNorm(dim, eps=1e-6)
        self.linear_1 = nn.Linear(dim, dim, bias=True)
        self.act = nn.GELU()
        self.linear_2 = nn.Linear(dim, out_dim, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        return self.linear_2(self.act(self.linear_1(self.pre_norm(x))))


class Merger(nn.Module):
    """2x2 fold onto the text width (the reference's single merge round)."""

    def __init__(self, config: VisionConfig):
        super().__init__()
        kh, kw = config.merge_kernel_size
        self.kernel = (int(kh), int(kw))
        self.mlp = [MergerBlock(config.hidden_size * self.kernel[0] * self.kernel[1],
                                config.out_hidden_size)]

    def __call__(self, x: mx.array, h: int, w: int) -> mx.array:
        b, _, d = x.shape
        kh, kw = self.kernel
        if h % kh or w % kw:
            raise ValueError(f"MiniCPM-V merger: grid {(h, w)} is not "
                             f"divisible by {self.kernel}")
        x = x.reshape(b, h // kh, kh, w // kw, kw, d).transpose(0, 1, 3, 2, 4, 5)
        x = x.reshape(b, (h // kh) * (w // kw), kh * kw * d)
        return self.mlp[0](x)


class VisionModel(nn.Module):
    """The tower plus both mergers. ``__call__`` takes packed patches and
    the per-crop patch grids and returns the text-width tokens of every
    crop, concatenated in crop order."""

    def __init__(self, config: VisionConfig):
        super().__init__()
        self.config = config
        self.embeddings = Embeddings(config)
        self.encoder = Encoder(config)
        self.post_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        if config.use_vit_merger:
            self.vit_merger = VitMerger(config)
        self.merger = Merger(config)

    def _encode(self, patches: mx.array, h: int, w: int) -> mx.array:
        x = self.embeddings(patches, h, w)
        insert = self.config.insert_layer_id if self.config.use_vit_merger else -1
        for i, layer in enumerate(self.encoder.layers):
            x = layer(x)
            if i == insert:
                x = self.vit_merger(x, h, w)
                h, w = h // 2, w // 2
        x = self.post_layernorm(x)
        return self.merger(x, h, w)

    def __call__(self, pixel_values: mx.array, target_sizes) -> mx.array:
        """``pixel_values [N, p * p * C]``: every crop's patches row-major in
        ``(kh, kw, C)`` order, crops in prompt order. ``target_sizes [n, 2]``:
        each crop's patch grid."""
        sizes = [tuple(int(v) for v in hw)
                 for hw in np.asarray(target_sizes).reshape(-1, 2)]
        dtype = self.embeddings.patch_embedding.weight.dtype
        pixel_values = pixel_values.astype(dtype)
        starts = np.concatenate([[0], np.cumsum([h * w for h, w in sizes])])
        if int(starts[-1]) != int(pixel_values.shape[0]):
            raise ValueError(
                f"MiniCPM-V vision: {pixel_values.shape[0]} patch rows but the "
                f"target sizes cover {int(starts[-1])}")
        groups: dict[tuple[int, int], list[int]] = {}
        for i, hw in enumerate(sizes):
            groups.setdefault(hw, []).append(i)
        outs: list = [None] * len(sizes)
        for (h, w), members in groups.items():
            batch = mx.stack([pixel_values[int(starts[i]):int(starts[i + 1])]
                              for i in members])
            tokens = self._encode(batch, h, w)
            for j, i in enumerate(members):
                outs[i] = tokens[j]
        return mx.concatenate(outs, axis=0)
