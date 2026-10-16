# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Asher Feldman
"""Audio encoder for Meta Muse Spark (mmproj projector ``muse-spark-audio``).

Ported from llama.cpp's ``clip_graph_muse_spark_audio::build`` and its loader
in ``clip.cpp``. The encoder reads the raw mono waveform at 24 kHz. It has no
spectrogram and no attention:

  1. A pyramid of stages. Each stage folds ``patch`` consecutive time steps
     into the channel axis, projects to the stage width, and applies an
     RMSNorm. Then ``depth`` residual blocks follow.
  2. Each block runs a causal depthwise conv over time (left padded by
     ``kernel - 1``), adds a bias, applies an RMSNorm, then a GELU (erf) MLP.
     The block input is the residual.
  3. After the last stage: RMSNorm, a biased projection straight to the text
     width, and a final RMSNorm.

The patch sizes multiply to the samples per output embedding (1920, or 80 ms
at 24 kHz). Every op is causal or local in time, so embedding ``i`` depends
only on samples before ``(i + 1) * 1920``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List

import mlx.core as mx
import mlx.nn as nn

from gmlx._vendor.mlx_vlm.models.base import BaseModelConfig


@dataclass
class AudioConfig(BaseModelConfig):
    model_type: str = "muse_spark_audio"
    hidden_sizes: List[int] = field(default_factory=lambda: [256, 512, 1024, 2048])
    patch_sizes: List[int] = field(default_factory=lambda: [16, 6, 5, 4])
    depths: List[int] = field(default_factory=lambda: [2, 2, 2, 2])
    # Per-stage MLP widths. Empty means four times the stage width.
    intermediate_sizes: List[int] = field(default_factory=list)
    conv_kernel_size: int = 7
    rms_norm_eps: float = 1e-6
    projection_dim: int = 8192
    sample_rate: int = 24000

    @property
    def samples_per_token(self) -> int:
        return math.prod(self.patch_sizes)


class _Norm(nn.Module):
    def __init__(self, dims: int, eps: float):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        return mx.fast.rms_norm(x, self.weight.astype(x.dtype), self.eps)


class PatchEmbed(nn.Module):
    def __init__(self, in_dims: int, out_dims: int, eps: float):
        super().__init__()
        self.proj = nn.Linear(in_dims, out_dims, bias=True)
        self.norm = _Norm(out_dims, eps)

    def __call__(self, x: mx.array) -> mx.array:
        return self.norm(self.proj(x))


class CausalDepthwiseConv(nn.Module):
    """``out[t] = sum_k weight[:, k] * x[t - (K - 1) + k] + bias``."""

    def __init__(self, dims: int, kernel: int):
        super().__init__()
        self.weight = mx.zeros((dims, kernel))
        self.bias = mx.zeros((dims,))

    def __call__(self, x: mx.array) -> mx.array:
        T, C = x.shape
        K = self.weight.shape[1]
        w = self.weight.astype(x.dtype)
        xp = mx.concatenate([mx.zeros((K - 1, C), dtype=x.dtype), x], axis=0)
        out = xp[0:T] * w[:, 0]
        for k in range(1, K):
            out = out + xp[k : k + T] * w[:, k]
        return out + self.bias.astype(x.dtype)


class AudioMLP(nn.Module):
    def __init__(self, dims: int, hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(dims, hidden, bias=True)
        self.fc2 = nn.Linear(hidden, dims, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        return self.fc2(nn.gelu(self.fc1(x)))


class ConvBlock(nn.Module):
    def __init__(self, dims: int, hidden: int, kernel: int, eps: float):
        super().__init__()
        self.conv = CausalDepthwiseConv(dims, kernel)
        self.conv_norm = _Norm(dims, eps)
        self.mlp = AudioMLP(dims, hidden)

    def __call__(self, x: mx.array) -> mx.array:
        return x + self.mlp(self.conv_norm(self.conv(x)))


class AudioModel(nn.Module):
    """Waveform in, ``[n_samples // samples_per_token, projection_dim]`` out.

    The caller pads the waveform to a whole number of embeddings."""

    def __init__(self, config: AudioConfig):
        super().__init__()
        self.config = config
        eps = config.rms_norm_eps
        widths = list(config.hidden_sizes)
        ffn = list(config.intermediate_sizes) or [4 * w for w in widths]
        n = len(config.patch_sizes)
        if not (len(widths) == len(config.depths) == len(ffn) == n):
            raise ValueError(
                "muse_spark audio: hidden_sizes, patch_sizes, depths and "
                "intermediate_sizes need one entry per stage")
        self.patches = []
        self.blocks = []
        prev = 1
        for i in range(n):
            self.patches.append(
                PatchEmbed(prev * config.patch_sizes[i], widths[i], eps))
            for _ in range(config.depths[i]):
                self.blocks.append(
                    ConvBlock(widths[i], ffn[i], config.conv_kernel_size, eps))
            prev = widths[i]
        self.norm = _Norm(prev, eps)
        self.proj = nn.Linear(prev, config.projection_dim, bias=True)
        self.post_norm = _Norm(config.projection_dim, eps)

    def __call__(self, waveform: mx.array) -> mx.array:
        x = waveform.reshape(-1, 1)
        bid = 0
        for stage, patch in enumerate(self.config.patch_sizes):
            T, C = x.shape
            if T % patch:
                raise ValueError(
                    f"muse_spark audio: {T} steps at stage {stage} do not fold "
                    f"by {patch}; pad the waveform to a multiple of "
                    f"{self.config.samples_per_token} samples")
            x = self.patches[stage](x.reshape(T // patch, patch * C))
            for _ in range(self.config.depths[stage]):
                x = self.blocks[bid](x)
                bid += 1
        return self.post_norm(self.proj(self.norm(x)))
