# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Asher Feldman
"""Two-stream mHC routing for SenseNova Loop.

Follows llama.cpp's ``llama_model_sensenova_loop::graph`` (build_route,
build_merge): the router reads both residual streams through an RMS norm
without gain and one float32 linear with bias, and returns

- the branch input ``(sigmoid(l0) a + sigmoid(l1) b) * pre_scale``,
- the write gains ``2 sigmoid(l2)`` and ``2 sigmoid(l3)``,
- a 2x2 doubly stochastic mix from ``l4..l7`` after ``iterations`` Sinkhorn
  rounds in log space (rows, then columns).

The merge is ``a' = m00 a + m01 b + post0 y`` and ``b' = m10 a + m11 b +
post1 y`` for the branch output ``y``. The routing runs in float32 and the
streams keep the activation dtype.
"""

from __future__ import annotations

from functools import lru_cache

import mlx.core as mx
import mlx.nn as nn

_EPS = 1e-6


def _normalize(x, y):
    # Two-element log-softmax as llama.cpp writes it (softplus of the gap).
    d = x - y
    return -mx.logaddexp(-d, 0.0), -mx.logaddexp(d, 0.0)


@lru_cache(maxsize=None)
def _coef_fn(iterations: int, fixed_identity: bool):
    """logits [..., n] float32 -> coefficients [..., 8] float32:
    pre0, pre1, post0, post1, m00, m01, m10, m11."""

    def fn(logits):
        pre0 = mx.sigmoid(logits[..., 0:1])
        pre1 = mx.sigmoid(logits[..., 1:2])
        post0 = 2.0 * mx.sigmoid(logits[..., 2:3])
        post1 = 2.0 * mx.sigmoid(logits[..., 3:4])
        if fixed_identity:
            one = mx.ones_like(pre0)
            zero = mx.zeros_like(pre0)
            mix = [one, zero, zero, one]
        else:
            m = [logits[..., 4 + i:5 + i] for i in range(4)]
            for _ in range(iterations):
                m[0], m[1] = _normalize(m[0], m[1])
                m[2], m[3] = _normalize(m[2], m[3])
                m[0], m[2] = _normalize(m[0], m[2])
                m[1], m[3] = _normalize(m[1], m[3])
            mix = [mx.exp(v) for v in m]
        return mx.concatenate([pre0, pre1, post0, post1, *mix], axis=-1)

    return mx.compile(fn)


@lru_cache(maxsize=None)
def _branch_fn(pre_scale: float):
    def fn(a, b, coef):
        x = a.astype(mx.float32) * coef[..., 0:1] + b.astype(mx.float32) * coef[..., 1:2]
        if pre_scale != 1.0:
            x = x * pre_scale
        return x.astype(a.dtype)

    return mx.compile(fn)


def _merge(a, b, y, coef):
    af = a.astype(mx.float32)
    bf = b.astype(mx.float32)
    yf = y.astype(mx.float32)
    na = af * coef[..., 4:5] + bf * coef[..., 5:6] + yf * coef[..., 2:3]
    nb = af * coef[..., 6:7] + bf * coef[..., 7:8] + yf * coef[..., 3:4]
    return na.astype(a.dtype), nb.astype(b.dtype)


merge = mx.compile(_merge)


def _mean(a, b):
    return ((a.astype(mx.float32) + b.astype(mx.float32)) * 0.5).astype(a.dtype)


leave = mx.compile(_mean)


class MHCRouter(nn.Module):
    """One router: ``weight`` [n_route, 2 * hidden] and ``bias`` [n_route],
    both float32."""

    def __init__(self, hidden_size: int, settings: dict):
        super().__init__()
        self.iterations = int(settings.get("iterations", 20))
        self.fixed_identity = bool(settings.get("fixed_identity", False))
        self.pre_scale = float(settings.get("pre_scale", 1.0))
        n_route = 4 if self.fixed_identity else 8
        self.weight = mx.zeros((n_route, 2 * hidden_size), dtype=mx.float32)
        self.bias = mx.zeros((n_route,), dtype=mx.float32)

    def __call__(self, a: mx.array, b: mx.array):
        """Returns (branch input, coefficients)."""
        flat = mx.concatenate([a, b], axis=-1).astype(mx.float32)
        flat = mx.fast.rms_norm(flat, None, _EPS)
        logits = mx.addmm(self.bias, flat, self.weight.T)
        coef = _coef_fn(self.iterations, self.fixed_identity)(logits)
        return _branch_fn(self.pre_scale)(a, b, coef), coef
