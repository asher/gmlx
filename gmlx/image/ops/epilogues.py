# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Asher Feldman
"""Elementwise seams that a matmul epilogue can absorb."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


def swiglu(a: mx.array, b: mx.array) -> mx.array:
    """silu(a) * b."""
    return nn.silu(a) * b


def modulate(x: mx.array, scale: mx.array) -> mx.array:
    """x * (1 + scale), the adaLN scale."""
    return x * (1 + scale)


def gated_residual(x: mx.array, gate: mx.array, y: mx.array) -> mx.array:
    """x + gate * y. The gate is already squashed."""
    return x + gate * y
