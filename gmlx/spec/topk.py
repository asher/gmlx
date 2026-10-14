"""Exact top-k ids over a large vocabulary without a full-width sort."""

from __future__ import annotations

HIER_ROW = 128      # vocab ids reduced to one max per row of this width
HIER_MIN_V = 8192   # below this width a flat argpartition is already cheap


def topk_ids(lp, k: int):
    """Top-k vocab ids of each row of ``lp`` [B, V] f32, hierarchically.

    ``mx.argpartition`` over a 200k vocab is a full-width sort on the GPU.
    Reshape each row to rows of ``HIER_ROW``, reduce those to their max (one
    bandwidth-bound pass), argpartition the ~V/128 maxes, then argpartition
    the k*128 surviving candidates. Exact: a value in the global top-k
    bounds its row's max from below, so fewer than k rows can outrank that
    row, and every top-k value's row survives the row cut. Only the order of
    ties at the k-th value can differ from the flat argpartition. The ids
    come back in no particular order.
    """
    import mlx.core as mx

    b, v = lp.shape
    n_rows = (v + HIER_ROW - 1) // HIER_ROW
    if n_rows * HIER_ROW != v:
        pad = mx.full((b, n_rows * HIER_ROW - v), -float("inf"), dtype=lp.dtype)
        lp = mx.concatenate([lp, pad], axis=-1)
    rows = lp.reshape(b, n_rows, HIER_ROW)
    top_rows = mx.argpartition(
        -rows.max(axis=2), kth=k - 1, axis=-1)[:, :k]             # [B, k]
    cand = mx.take_along_axis(
        rows, top_rows[:, :, None], axis=1).reshape(b, -1)       # [B, k*128]
    loc = mx.argpartition(-cand, kth=k - 1, axis=-1)[:, :k]      # [B, k]
    return (mx.take_along_axis(top_rows, loc // HIER_ROW, axis=-1)
            * HIER_ROW + (loc % HIER_ROW))
