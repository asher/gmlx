# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Asher Feldman
"""Products of the per-step rows of a DiT: the time vectors and the
modulations derived from them, one row per sigma."""

from __future__ import annotations

import mlx.core as mx

_PRECISE_ROWS = 4096


def precise(lin, x: mx.array) -> mx.array:
    """``lin(x)`` in fp32 for the one row per sigma of the time vectors.
    A K-quant product of a row rounds its output to bf16 whatever the
    input dtype, which puts the time vectors past the reference's own
    rounding. A K-quant leaf with no live adapter delta therefore runs as
    an fp32 product on its weight dequantized a band of rows at a time;
    any other leaf runs its own call on the fp32 row."""
    from .kq_routes import kquant_leaf

    leaf = kquant_leaf(lin, mx.bfloat16)
    x = x.astype(mx.float32)
    if leaf is None:
        return lin(x).astype(mx.float32)
    import mlx_kquant as kq

    w = leaf["weight"]
    parts = []
    for r in range(0, w.shape[0], _PRECISE_ROWS):
        band = kq.dequantize(w[r:r + _PRECISE_ROWS], leaf["scales"],
                             leaf.kquant_type, mx.float32)
        parts.append(x @ band.T)
        mx.eval(parts[-1])
    y = mx.concatenate(parts, axis=-1)
    if "bias" in leaf:
        y = y + leaf["bias"].astype(mx.float32)
    return y


def own(lin, x: mx.array) -> mx.array:
    """``lin(x)`` by the leaf's own call, for fp32 activations, where the
    exact gate's dense products take every K-quant leaf."""
    return lin(x)
