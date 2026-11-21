# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Asher Feldman
"""Row fusion: projections that read the same input, run as one matmul.

The members' weight rows are concatenated into one buffer after load, and
the members are rebound as row views of it, so the fusion adds no memory
and the members still serve every caller that uses them alone. A decode
row is exact: each output row is its own dot product. At other row counts
the kernel choice follows the fused row count, so a codec can differ by
one rounding step.

A model keeps the fused module in a slot listed in
``gmlx.load.modules._FUSED_CHILD_SLOTS``, so an adapter on any member
clears it and the forward falls back to the members.
"""

from __future__ import annotations

from typing import Optional, Sequence

import mlx.core as mx
import mlx.nn as nn

from gmlx.load.out_scale import cat_out_scale, has_out_scale, set_out_scale


def _rebind(mods, weight: mx.array) -> None:
    start = 0
    for m in mods:
        n = m.weight.shape[0]
        m.weight = weight[start:start + n]
        start += n


def fuse_kquant_rows(mods: Sequence, in_dims: int) -> Optional[nn.Module]:
    """One KQuantLinear serving the rows of ``mods`` in order; None when
    they are not all plain KQuantLinears of one codec, without a bias,
    over ``in_dims`` inputs."""
    try:
        from mlx_kquant.nn import KQuantLinear
    except Exception:
        return None
    # Exact type: a subclass (HadamardKQuantLinear) transforms its input,
    # which a plain fused KQuantLinear would drop.
    if len(mods) < 2 or any(type(m) is not KQuantLinear for m in mods):
        return None
    codec = mods[0].kquant_type
    if any(m.kquant_type != codec or "bias" in m or m.weight.ndim != 2
           or m.weight.shape[1] != mods[0].weight.shape[1] for m in mods):
        return None
    rows = sum(m.weight.shape[0] for m in mods)
    fused = KQuantLinear(in_dims, rows, bias=False, codec=codec)
    fused.weight = mx.concatenate([m.weight for m in mods], axis=0)
    fused.scales = mods[0].scales
    set_out_scale(fused, cat_out_scale(mods))
    mx.eval(fused.parameters())
    _rebind(mods, fused.weight)
    return fused


def fuse_float_rows(mods: Sequence) -> Optional[nn.Module]:
    """One nn.Linear serving the rows of ``mods`` in order; None when they
    are not all plain bias-free nn.Linears of one dtype and input size."""
    if len(mods) < 2 or any(type(m) is not nn.Linear for m in mods):
        return None
    w0 = mods[0].weight
    if any("bias" in m or has_out_scale(m) or m.weight.dtype != w0.dtype
           or m.weight.ndim != 2 or m.weight.shape[1] != w0.shape[1]
           for m in mods):
        return None
    fused = nn.Linear(w0.shape[1], sum(m.weight.shape[0] for m in mods),
                      bias=False)
    fused.weight = mx.concatenate([m.weight for m in mods], axis=0)
    mx.eval(fused.weight)
    _rebind(mods, fused.weight)
    return fused


def split_rows(fused: nn.Module, sizes: Sequence[int], x: mx.array) -> list:
    """The fused output, cut into its members' outputs."""
    y = fused(x)
    out, start = [], 0
    for n in sizes:
        out.append(y[..., start:start + n])
        start += n
    return out
