# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2025 Black Forest Labs, Inc.
# _flux2_mu and _flux2 follow compute_empirical_mu and get_schedule of
# src/flux2/sampling.py of BFL's FLUX.2 reference at commit 50fe516
# (Apache-2.0; see licenses/flux2-LICENSE).
# SPDX-FileCopyrightText: Copyright 2026 Qwen-Image Team and The HuggingFace Team
# _qwen_image21 follows calculate_shift of diffusers'
# src/diffusers/pipelines/qwenimage21/pipeline_qwenimage21.py and the
# exponential shift and terminal stretch of
# src/diffusers/schedulers/scheduling_flow_match_euler_discrete.py at
# e0abab83 (Apache-2.0; see licenses/diffusers-LICENSE).
"""The schedules of the variants whose sigmas follow the image size, as
the gmlx server's status names them under ``sampling.schedule``, in
float32 numpy as gmlx's own sampler forms them. It imports neither torch
nor ComfyUI."""

from __future__ import annotations

import math

import numpy as np

_F32 = np.float32


class SizeNeeded(ValueError):
    """The variant's schedule follows the image size, and none was given."""


def _fma(a, b, c) -> np.ndarray:
    """float32 ``a * b + c`` rounded once."""
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


def _linspace(start: float, end: float, steps: int) -> np.ndarray:
    """``torch.linspace(start, end, steps)`` in float32, bit for bit."""
    s, e = _F32(start), _F32(end)
    if steps <= 1:
        return np.full(steps, s, _F32)
    step = _F32(_F32(e - s) / _F32(steps - 1))
    i = np.arange(steps)
    half = steps // 2
    lo = _fma(step, i[:half].astype(_F32), s)
    hi = _fma(-step, (steps - 1 - i[half:]).astype(_F32), e)
    return np.concatenate([lo, hi]).astype(_F32)


def _flux2_mu(tokens: int, steps: int) -> float:
    a1, b1 = 8.73809524e-05, 1.89833333
    a2, b2 = 0.00016927, 0.45666666
    if tokens > 4300:
        return float(a2 * tokens + b2)
    m200 = a2 * tokens + b2
    m10 = a1 * tokens + b1
    a = (m200 - m10) / 190.0
    b = m200 - 200.0 * a
    return float(a * steps + b)


def _flux2(steps: int, tokens: int, rule: dict) -> np.ndarray:
    t = _linspace(1.0, 0.0, steps + 1)
    e = _F32(math.exp(_flux2_mu(tokens, steps)))
    with np.errstate(divide="ignore"):
        d = _F32(1) / t - _F32(1)
        return ((_F32(1) / (e + d)) * e).astype(_F32)


def _qwen_image21(steps: int, tokens: int, rule: dict) -> np.ndarray:
    if steps < 2:
        raise ValueError(f"the schedule needs at least 2 steps, not {steps}")
    base = np.linspace(1.0, 1.0 / steps, steps).astype(_F32)
    m = (rule["max_shift"] - rule["base_shift"]) / (rule["max_len"]
                                                    - rule["base_len"])
    mu = tokens * m + (rule["base_shift"] - m * rule["base_len"])
    e = _F32(math.exp(mu))
    s = (e / (e + np.power(_F32(1) / base - _F32(1), _F32(1)))).astype(_F32)
    z = _F32(1) - s
    scale = _F32(z[-1] / _F32(1.0 - rule["terminal"]))
    s = (_F32(1) - z / scale).astype(_F32)
    return np.append(s, _F32(0)).astype(_F32)


RULES = {"flux2": _flux2, "qwen_image21": _qwen_image21}
# The stock node that gives a rule's schedule in ComfyUI.
STOCK = {"flux2": "Flux2Scheduler"}


def size_sigmas(status: dict, model_id: str, steps: int, denoise: float,
                width: int, height: int) -> np.ndarray | None:
    """The last ``steps + 1`` sigmas of a schedule that follows the image
    size, for an image of ``width`` by ``height`` pixels, or None when the
    variant's schedule does not follow it. Below a denoise of 1 the
    schedule runs over ``steps / denoise`` steps, as ComfyUI's are."""
    rule = ((status.get("sampling") or {}).get("schedule")) or None
    if rule is None:
        return None
    form = RULES.get(rule.get("rule"))
    if form is None:
        raise ValueError(f"gmlx: {model_id!r} samples on the schedule "
                         f"{rule.get('rule')!r}, which these nodes do not "
                         "know; update gmlx")
    if width <= 0 or height <= 0:
        stock = STOCK.get(rule["rule"])
        raise SizeNeeded(
            f"gmlx: {model_id!r} samples on a schedule that follows the "
            "image size; give gmlx Sigmas the width and height of the image"
            + (f", or take the sigmas from {stock}" if stock else ""))
    latent = status.get("latent") or {}
    px = int(latent.get("stride") or 16) * int(latent.get("patch") or 1)
    tokens = (int(height) // px) * (int(width) // px)
    total = steps if denoise >= 1.0 else int(steps / max(denoise, 1e-4))
    return form(total, tokens, rule)[-(steps + 1):]
