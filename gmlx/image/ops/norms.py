# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Asher Feldman
# RMSNorm rounding follows diffusers' normalization.py at e0abab83
# (Apache-2.0, see licenses/diffusers-LICENSE).
"""Norms with fp32 accumulation.

``mx.fast.rms_norm`` sums in fp32 and rounds the normalized value to the
input dtype before the weight product, as diffusers' RMSNorm does for a
half-precision weight. A weight in another dtype than the input, such as an
fp32 weight kept for precision, runs the whole norm in fp32 and casts back.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


def rms_norm(x: mx.array, w: mx.array, eps: float) -> mx.array:
    if w.dtype == x.dtype:
        return mx.fast.rms_norm(x, w, eps)
    y = mx.fast.rms_norm(x.astype(mx.float32), w.astype(mx.float32), eps)
    return y.astype(x.dtype)


def layer_norm(x: mx.array, eps: float) -> mx.array:
    """LayerNorm without affine parameters."""
    return mx.fast.layer_norm(x, None, None, eps)


class RMSNorm(nn.Module):
    def __init__(self, dims: int, eps: float):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        return rms_norm(x, self.weight, self.eps)
