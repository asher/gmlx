# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Asher Feldman
"""GroupNorm over channels-last activations, with contiguous groups.

torch and diffusers put channels ``[i * C / G, (i + 1) * C / G)`` in group
i. Stock ``nn.GroupNorm`` defaults to ``pytorch_compatible=False``, which
interleaves channels across groups, so it gives wrong numbers and no error
on torch weights. Statistics here are computed in fp32 by
``mx.fast.layer_norm``. The normalized value is returned in the input
dtype, and the affine weights are cast to that dtype first, since fp32
weights would promote the output.

A tiled pass normalizes each tile with statistics of the whole image.
``group_moments`` gives one region's count, mean and squared deviations
per group, ``combine_moments`` merges regions, and ``GroupNorm.steps``
yields its input to the caller and normalizes with the statistics sent
back.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import numpy as np


def group_norm(x: mx.array, weight: mx.array | None, bias: mx.array | None,
               groups: int, eps: float) -> mx.array:
    """Normalize ``x`` (B, ..., C) over its spatial axes and each group of
    ``C // groups`` contiguous channels."""
    shape = x.shape
    b, c = shape[0], shape[-1]
    if c % groups:
        raise ValueError(f"{c} channels do not split into {groups} groups")
    g = c // groups
    y = x.reshape(b, -1, groups, g).transpose(0, 2, 1, 3)
    y = mx.fast.layer_norm(y.reshape(b, groups, -1), None, None, eps)
    y = y.reshape(b, groups, -1, g).transpose(0, 2, 1, 3).reshape(shape)
    dt = x.dtype
    if weight is not None:
        y = y * weight.astype(dt)
    if bias is not None:
        y = y + bias.astype(dt)
    return y


def group_moments(x: mx.array, groups: int) -> tuple[int, mx.array, mx.array]:
    """Element count per group, and the fp32 mean and sum of squared
    deviations (B, groups) of ``x`` (B, ..., C)."""
    b, c = x.shape[0], x.shape[-1]
    y = x.astype(mx.float32).reshape(b, -1, groups, c // groups)
    mean = y.mean(axis=(1, 3), keepdims=True)
    m2 = mx.square(y - mean).sum(axis=(1, 3))
    return y.shape[1] * y.shape[3], mean.reshape(b, groups), m2


def combine_moments(parts) -> tuple[mx.array, mx.array]:
    """The fp32 mean and variance (B, 1, groups, 1) of the union of
    disjoint regions, from their ``group_moments``. Merged in float64."""
    n = 0
    mean = m2 = None
    for k, mk, m2k in parts:
        mk, m2k = np.array(mk, np.float64), np.array(m2k, np.float64)
        if mean is None:
            n, mean, m2 = k, mk, m2k
            continue
        d = mk - mean
        mean = mean + d * (k / (n + k))
        m2 = m2 + m2k + d * d * (n * k / (n + k))
        n += k
    b, g = mean.shape
    shape = (b, 1, g, 1)
    return (mx.array(mean.reshape(shape).astype(np.float32)),
            mx.array((m2 / n).reshape(shape).astype(np.float32)))


def group_norm_with(x: mx.array, mean: mx.array, var: mx.array,
                    weight: mx.array | None, bias: mx.array | None,
                    groups: int, eps: float) -> mx.array:
    """``group_norm`` with given fp32 ``mean`` and ``var`` (B, 1, groups, 1)
    in place of the statistics of ``x``."""
    shape, dt = x.shape, x.dtype
    b, c = shape[0], shape[-1]
    y = x.astype(mx.float32).reshape(b, -1, groups, c // groups)
    y = ((y - mean) * mx.rsqrt(var + eps)).reshape(shape).astype(dt)
    if weight is not None:
        y = y * weight.astype(dt)
    if bias is not None:
        y = y + bias.astype(dt)
    return y


class GroupNorm(nn.Module):
    """Affine GroupNorm with torch's channel grouping."""

    def __init__(self, groups: int, channels: int, eps: float):
        super().__init__()
        self.groups = groups
        self.eps = eps
        self.weight = mx.ones((channels,))
        self.bias = mx.zeros((channels,))

    def __call__(self, x: mx.array) -> mx.array:
        return group_norm(x, self.weight, self.bias, self.groups, self.eps)

    def steps(self, x: mx.array):
        """Yield ``(self, x)``, receive ``(mean, var)`` for the whole image
        and return ``x`` normalized with them."""
        mean, var = yield self, x
        return group_norm_with(x, mean, var, self.weight, self.bias,
                               self.groups, self.eps)
