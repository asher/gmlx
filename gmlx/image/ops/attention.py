# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Asher Feldman
# Portions copyright (c) 2026 Filip Strand (mflux Z-Image attention, MIT)
# RoPE semantics follow diffusers' transformer_z_image.py at e0abab83
# (Apache-2.0, see licenses/diffusers-LICENSE).
"""Attention seams: q and k RMSNorm with RoPE, and SDPA."""

from __future__ import annotations

import mlx.core as mx

from .norms import rms_norm


def _rope(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    """Rotate interleaved pairs of x (B, L, H, D) by per-row angles in fp32."""
    b, n, h, d = x.shape
    x = x.astype(mx.float32).reshape(b, n, h, d // 2, 2)
    x0, x1 = x[..., 0], x[..., 1]
    c = cos[None, :, None, :]
    s = sin[None, :, None, :]
    out = mx.stack([x0 * c - x1 * s, x0 * s + x1 * c], axis=-1)
    return out.reshape(b, n, h, d)


def qk_norm_rope(q: mx.array, k: mx.array, q_w: mx.array, k_w: mx.array,
                 cos: mx.array, sin: mx.array, eps: float
                 ) -> tuple[mx.array, mx.array]:
    """RMSNorm over the head dim, then RoPE in fp32.

    q, k are (B, L, H, D). cos and sin are fp32 (L, D // 2). Returns q and k
    as (B, H, L, D) in the input dtype.
    """
    dt = q.dtype
    q = _rope(rms_norm(q, q_w, eps), cos, sin).astype(dt)
    k = _rope(rms_norm(k, k_w, eps), cos, sin).astype(dt)
    return q.transpose(0, 2, 1, 3), k.transpose(0, 2, 1, 3)


def sdpa(q: mx.array, k: mx.array, v: mx.array,
         mask: mx.array | None = None) -> mx.array:
    """Scaled dot-product attention over (B, H, L, D) at scale D ** -0.5."""
    return mx.fast.scaled_dot_product_attention(
        q, k, v, scale=q.shape[-1] ** -0.5, mask=mask)
