"""Sigma schedules, computed on the host in numpy with float32 results.

A schedule is part of a family's numerics, so it must equal the reference
bit for bit, and the order of casts is part of that contract. Every
function here returns float32 and states where it rounds.

Z-Image's reference forms its base sigmas with ``torch.linspace(1.0, 1/n,
n)`` in float32, passes them through Python floats back to float32, and
applies a static shift of 3.0. ``linspace_f32`` replicates the torch CPU
kernel: it fills the first half from the start and the second half from
the end, and each element is one fused multiply-add rounded once to
float32.
"""

from __future__ import annotations

import numpy as np

_F32 = np.float32


def fma_f32(a, b, c) -> np.ndarray:
    """float32 ``a * b + c`` rounded once, for float32 inputs.

    The product of two float32 values is exact in float64. The sum is
    rounded to float64 by round-to-odd, using its exact error term, so the
    final cast to float32 rounds once to nearest even.
    """
    a = np.asarray(a, _F32).astype(np.float64)
    b = np.asarray(b, _F32).astype(np.float64)
    c = np.asarray(c, _F32).astype(np.float64)
    p = a * b
    s = p + c
    t = s - p
    err = (p - (s - t)) + (c - t)
    odd = (s.view(np.int64) & 1) == 1
    up = np.nextafter(s, np.where(err > 0, np.inf, -np.inf))
    s = np.where((err != 0) & ~odd, up, s)
    return s.astype(_F32)


def linspace_f32(start: float, end: float, steps: int) -> np.ndarray:
    """``torch.linspace(start, end, steps)`` in float32, bit for bit.

    The step is ``(end - start) / (steps - 1)`` in float32. Index i below
    ``steps // 2`` is ``start + step * i`` and the rest are
    ``end - step * (steps - 1 - i)``, each as one fused multiply-add. One
    step gives ``[start]``.
    """
    if steps < 0:
        raise ValueError(f"steps must be non-negative, not {steps}")
    s, e = _F32(start), _F32(end)
    if steps <= 1:
        return np.full(steps, s, _F32)
    step = _F32(_F32(e - s) / _F32(steps - 1))
    i = np.arange(steps)
    half = steps // 2
    lo = fma_f32(step, i[:half].astype(_F32), s)
    hi = fma_f32(-step, (steps - 1 - i[half:]).astype(_F32), e)
    return np.concatenate([lo, hi]).astype(_F32)


def shift_static(sigmas, shift: float) -> np.ndarray:
    """``shift * s / (1 + (shift - 1) * s)``, each operation in float32."""
    s = np.asarray(sigmas, _F32)
    num = _F32(shift) * s
    den = _F32(1) + _F32(shift - 1) * s
    return (num / den).astype(_F32)


def zimage_sigmas(steps: int, shift: float = 3.0) -> np.ndarray:
    """Z-Image sigmas, length ``steps + 1`` with a final zero.

    The base sigmas are ``linspace_f32(1.0, 1 / steps, steps)``, and
    ``1 / steps`` is a Python float that rounds to float32 at the call, as
    torch rounds a scalar argument to the output dtype.
    """
    if steps < 1:
        raise ValueError(f"steps must be at least 1, not {steps}")
    base = linspace_f32(1.0, 1.0 / steps, steps)
    return np.append(shift_static(base, shift), _F32(0)).astype(_F32)


def check_sigmas(sigmas) -> np.ndarray:
    """An external schedule as float32, refused unless it is one dimension
    with at least two values, finite and not increasing."""
    s = np.asarray(sigmas, dtype=np.float64)
    if s.ndim != 1 or s.size < 2:
        raise ValueError("a sigma schedule needs at least two values")
    if not np.all(np.isfinite(s)):
        raise ValueError("a sigma schedule must be finite")
    if np.any(np.diff(s) > 0):
        raise ValueError("a sigma schedule must not increase")
    return s.astype(_F32)
