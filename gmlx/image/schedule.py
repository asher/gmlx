# SPDX-License-Identifier: Apache-2.0 AND MIT
# SPDX-FileCopyrightText: Copyright 2025 Stability AI, Katherine Crowson and The HuggingFace Team
# SPDX-FileCopyrightText: Copyright 2026 Qwen-Image Team and The HuggingFace Team
# calculate_shift is ported from diffusers'
# src/diffusers/pipelines/qwenimage21/pipeline_qwenimage21.py, and
# shift_exponential and stretch_to_terminal from
# src/diffusers/schedulers/scheduling_flow_match_euler_discrete.py, at
# e0abab83 and modified for gmlx (Apache-2.0; see licenses/diffusers-LICENSE).
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

Qwen-Image 2.1's reference forms its base sigmas with ``np.linspace(1.0,
1/n, n)`` in float64 and rounds them to float32. It shifts them
exponentially by a mu that grows with the image token count, and stretches
them so that the last one is the terminal value. Every operation after the
first rounding is in float32, with each Python scalar rounded to float32
where it meets an array, as NumPy 2 does.
"""

from __future__ import annotations

import math

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


def calculate_shift(tokens: int, base_len: int = 256, max_len: int = 8192,
                    base_shift: float = 0.5, max_shift: float = 0.9) -> float:
    """mu of the dynamic shift for an image of ``tokens`` tokens: the line
    through (base_len, base_shift) and (max_len, max_shift), in Python
    floats and in the reference's order of operations."""
    m = (max_shift - base_shift) / (max_len - base_len)
    b = base_shift - m * base_len
    return tokens * m + b


def shift_exponential(sigmas, mu: float) -> np.ndarray:
    """``e / (e + (1 / s - 1))`` with ``e = exp(mu)`` rounded to float32,
    each operation in float32."""
    s = np.asarray(sigmas, _F32)
    e = _F32(math.exp(mu))
    return (e / (e + np.power(_F32(1) / s - _F32(1), _F32(1)))).astype(_F32)


def stretch_to_terminal(sigmas, terminal: float) -> np.ndarray:
    """``1 - (1 - s) / ((1 - s[-1]) / (1 - terminal))`` in float32, which
    moves the last sigma to ``terminal``. ``1 - terminal`` is a Python
    float rounded to float32."""
    s = np.asarray(sigmas, _F32)
    z = _F32(1) - s
    scale = _F32(z[-1] / _F32(1.0 - terminal))
    return (_F32(1) - z / scale).astype(_F32)


def qwen21_sigmas(steps: int, image_tokens: int, *, base_len: int = 256,
                  max_len: int = 8192, base_shift: float = 0.5,
                  max_shift: float = 0.9,
                  terminal: float = 0.02) -> np.ndarray:
    """Qwen-Image 2.1 sigmas for an image of ``image_tokens`` latent
    tokens, length ``steps + 1`` with a final zero. The defaults are the
    official scheduler config's.

    One step has no schedule: its only base sigma is 1, the stretch then
    divides zero by zero, and the reference returns NaN."""
    if steps < 2:
        raise ValueError(f"the Qwen-Image 2.1 schedule needs at least 2 "
                         f"steps, not {steps}; pass sigmas for fewer")
    base = np.linspace(1.0, 1.0 / steps, steps).astype(_F32)
    mu = calculate_shift(image_tokens, base_len, max_len, base_shift,
                         max_shift)
    s = stretch_to_terminal(shift_exponential(base, mu), terminal)
    return np.append(s, _F32(0)).astype(_F32)


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
