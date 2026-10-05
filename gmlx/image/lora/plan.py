"""Band placement, rank concatenation and strength for one module.

A module's adapter state is a list of parts: a pair, its band and its
strength. The delta of a part is ``strength * (alpha / rank) * (up @
down)``, where rank is the first dimension of ``down`` and a missing alpha
gives the factor 1.0, placed on the band's rows of the weight. Parts sum.

``block_diagonal`` writes the sum as one pair over the whole weight. A is
``(sum of ranks, in)`` and B is ``(out, sum of ranks)``, each part owning
its rank columns, with zeros outside its band. Placing a band and stacking
adapters are then the same concatenation along rank.

Every factor is float32 with the scale folded into B. ``LoraModule`` from
``gmlx.load.adapter`` carries the result with ``scale`` 1.0.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx

from gmlx.load.adapter import LoraModule


@dataclass(frozen=True)
class Part:
    """One pair on a module: ``down`` (rank, in), ``up`` (out, rank), its
    alpha or None, the band (dim, start, length) or None, and strength."""
    down: mx.array
    up: mx.array
    alpha: float | None
    band: tuple | None
    strength: float


def pair_scale(alpha: float | None, rank: int) -> float:
    """``alpha / rank``, or 1.0 without an alpha."""
    return 1.0 if alpha is None else float(alpha) / float(rank)


def fit(up: tuple, down: tuple, band: tuple | None,
        dims: tuple) -> str | None:
    """Why a pair with these up and down shapes does not fit a weight of
    ``dims`` (out, in) at ``band``, or None when it fits."""
    out, inp = dims
    want_out, want_in = out, inp
    if band is not None:
        if band[0] == 0:
            want_out = band[2]
        else:
            want_in = band[2]
    if up[1] != down[0]:
        return f"up {tuple(up)} and down {tuple(down)} differ in rank"
    if up[0] != want_out or down[1] != want_in:
        return (f"the pair is {up[0]}x{down[1]} (out x in) and the target "
                f"is {want_out}x{want_in}")
    return None


def _scaled(part: Part) -> tuple[mx.array, mx.array]:
    """The part's float32 A (rank, in) and B (out, rank), scale in B."""
    a = part.down.astype(mx.float32)
    f = part.strength * pair_scale(part.alpha, int(a.shape[0]))
    b = part.up.astype(mx.float32) * mx.array(f, mx.float32)
    return a, b


def _rows(x: mx.array, start: int, total: int) -> mx.array:
    """``x`` at rows ``start`` of a ``total`` row matrix of zeros."""
    return mx.pad(x, ((start, total - start - x.shape[0]), (0, 0)))


def _cols(x: mx.array, start: int, total: int) -> mx.array:
    return mx.pad(x, ((0, 0), (start, total - start - x.shape[1])))


def block_diagonal(path: str, parts, dims: tuple) -> LoraModule:
    """All parts of a module as one pair over the whole weight."""
    out, inp = dims
    a_rows, b_cols = [], []
    for p in parts:
        a, b = _scaled(p)
        if p.band is not None and p.band[0] == 0:
            b = _rows(b, p.band[1], out)
        elif p.band is not None:
            a = _cols(a, p.band[1], inp)
        a_rows.append(a)
        b_cols.append(b)
    a = a_rows[0] if len(a_rows) == 1 else mx.concatenate(a_rows, axis=0)
    b = b_cols[0] if len(b_cols) == 1 else mx.concatenate(b_cols, axis=1)
    return LoraModule(module_path=path, a=a, b=b, rank=int(a.shape[0]),
                      scale=1.0)
