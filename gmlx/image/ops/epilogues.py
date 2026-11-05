# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Asher Feldman
"""Elementwise seams that a matmul epilogue can absorb."""

from __future__ import annotations

from functools import partial

import mlx.core as mx
import mlx.nn as nn

# Rows of axis -2 per evaluation of gated_residual_once in a process where
# mx.compile does not compile.
ONCE_ROWS = 8192


def swiglu(a: mx.array, b: mx.array) -> mx.array:
    """silu(a) * b."""
    return nn.silu(a) * b


def modulate(x: mx.array, scale: mx.array) -> mx.array:
    """x * (1 + scale), the adaLN scale."""
    return x * (1 + scale)


def gated_residual(x: mx.array, gate: mx.array, y: mx.array) -> mx.array:
    """x + gate * y. The gate is already squashed."""
    return x + gate * y


@partial(mx.compile, shapeless=True)
def _added_once(x: mx.array, gate: mx.array, y: mx.array) -> mx.array:
    f = mx.float32
    return (x.astype(f) + gate.astype(f) * y.astype(f)).astype(x.dtype)


_traces = [0]


def _traced(x: mx.array) -> mx.array:
    _traces[0] += 1
    return x


_probe = mx.compile(_traced)


def _compiles() -> bool:
    """Whether mx.compile compiles in this process now. Under
    MLX_DISABLE_COMPILE or after mx.disable_compile, a compiled function
    runs its Python body at each call."""
    n, one = _traces[0], mx.array(1)
    _probe(one)
    _probe(one)
    return _traces[0] - n < 2


def gated_residual_once(x: mx.array, gate: mx.array,
                        y: mx.array) -> mx.array:
    """x + gate * y in float32, rounded once to the dtype of x, where
    ``gated_residual`` also rounds the product. The gate is already
    squashed. Compiled, it is one kernel with no float32 copy. Where
    mx.compile does not compile, the float32 copies of a half precision
    call hold 8 bytes an element beside its output, so a call over more
    than ``ONCE_ROWS`` rows of axis -2 evaluates ``ONCE_ROWS`` rows at a
    time into its output."""
    n = x.shape[-2] if x.ndim > 1 else 0
    if n <= ONCE_ROWS or _compiles():
        return _added_once(x, gate, y)
    by_row = gate.ndim > 1 and gate.shape[-2] == n
    out = mx.zeros(x.shape, x.dtype)
    for a in range(0, n, ONCE_ROWS):
        b = a + ONCE_ROWS
        g = gate[..., a:b, :] if by_row else gate
        out[..., a:b, :] = _added_once(x[..., a:b, :], g, y[..., a:b, :])
        mx.eval(out)
    return out
