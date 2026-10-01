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
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


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
