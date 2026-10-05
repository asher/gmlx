# SPDX-License-Identifier: Apache-2.0 AND MIT
# SPDX-FileCopyrightText: Copyright 2025 Alibaba Z-Image Team and The HuggingFace Team
# SPDX-FileCopyrightText: Copyright (c) 2026 Filip Strand
# Ported from TimestepEmbedder.timestep_embedding and
# RopeEmbedder.precompute_freqs_cis in diffusers'
# src/diffusers/models/transformers/transformer_z_image.py at e0abab83 and
# modified for gmlx (Apache-2.0; see licenses/diffusers-LICENSE). mflux's
# timestep and RoPE embedders at e066d98 follow the same code (MIT; see
# licenses/mflux-LICENSE).
"""Timestep embeddings and N-axis RoPE tables, computed on the host."""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np


def timestep_embedding(t, dim: int = 256,
                       max_period: float = 10000.0) -> mx.array:
    """Sinusoidal embedding of timesteps ``t`` (B,), cosines first, fp32."""
    half = dim // 2
    freqs = np.exp(-math.log(max_period)
                   * np.arange(half, dtype=np.float32) / half)
    args = np.asarray(t, np.float32).reshape(-1, 1) * freqs[None]
    emb = np.concatenate([np.cos(args), np.sin(args)], axis=-1)
    if dim % 2:
        emb = np.concatenate([emb, np.zeros_like(emb[:, :1])], axis=-1)
    return mx.array(emb.astype(np.float32))


def rope_tables(ids: np.ndarray, axes_dims, theta: float
                ) -> tuple[mx.array, mx.array]:
    """cos and sin tables (L, sum(axes_dims) // 2) for integer position ids
    (L, n_axes). Angles are formed in float64 and rounded to fp32 before
    cos and sin, as the reference tables are."""
    cos, sin = [], []
    for axis, d in enumerate(axes_dims):
        freqs = 1.0 / theta ** (np.arange(0, d, 2, dtype=np.float64) / d)
        ang = np.outer(ids[:, axis].astype(np.float64), freqs)
        ang = ang.astype(np.float32)
        cos.append(np.cos(ang))
        sin.append(np.sin(ang))
    return (mx.array(np.concatenate(cos, axis=1)),
            mx.array(np.concatenate(sin, axis=1)))
