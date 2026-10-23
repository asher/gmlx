# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Asher Feldman
"""KV cache append without a whole-cache copy, and the capacity grid.

A slice write puts the new rows in place only when MLX can donate the
cache buffer. Under pipelined decode the pending step still holds it, so
the write copies the whole cache once per token. ``kq.kv_append`` writes
the rows into the buffer itself. That is safe when nothing else reads or
writes those rows, which each cache tracks in two fields:

- ``_kv_own``: the cache's id and the keys and values arrays that the last
  append, regrow or allocation left.
- ``_kv_hw``: the row the next append must start at.

Every handout of a cache covers only rows below its offset at the time
(a ``state`` view, a prefix store entry, the MTP drafter's shared views).
The one exception is ``prefix_cache_reserve``, which returns the full
grown buffers to callers that only evaluate them. So rows from ``_kv_hw``
up were never seen. A path that swaps the arrays (restore, filter,
extend, extract) or moves the offset (trim, rollback) fails the check, and
its next append takes a slice write instead. A cache built with
``__new__`` lacks both fields and takes the slice write too.

The gated MTP round restores both fields with the rest of the cache after
dropping a verify graph that appended in place. Its rows are rewritten
safely because the gate clears the verify handle before the restore, so
nothing evaluates the dropped graph.
"""

import os
from bisect import bisect_left
from functools import lru_cache

import mlx.core as mx

try:
    import mlx_kquant as _kq
except ImportError:  # pragma: no cover - mlx_kquant is a dependency
    _kq = None

# Read once, like the other cache switches: a running process does not see
# a later export.
INPLACE = (_kq is not None and hasattr(_kq, "kv_append")
           and os.environ.get("GMLX_KV_APPEND_INPLACE", "1") != "0")


def mark_fresh(cache, prev) -> None:
    """Record freshly allocated cache buffers, next append at row ``prev``."""
    cache._kv_own = (id(cache), cache.keys, cache.values)
    cache._kv_hw = prev


def kv_write(cache, keys, values, prev, end, routed=True) -> None:
    """Write rows ``[prev, end)`` of ``cache.keys`` and ``cache.values`` and
    rebind both. ``keys`` and ``values`` are arrays, or sequences of arrays
    like the cache's for the quantized caches. ``routed`` False keeps the
    slice write on the arrays the cache holds, for the model-owned
    subclasses that run the base append.

    Both rebinds happen before anything can evaluate. Holding the old
    arrays in ``_kv_own`` until the end is harmless to donation only for
    that reason, so nothing here may evaluate. The decode path runs this
    once per attention layer and token, so it calls no other Python
    function on plain arrays.
    """
    ks, vs = cache.keys, cache.values
    if isinstance(ks, (tuple, list)):
        _kv_write_parts(cache, keys, values, prev, end, routed)
        return
    if not routed:
        ks[..., prev:end, :] = keys
        vs[..., prev:end, :] = values
        return
    # A slice write casts; kv_append takes one dtype.
    if keys.dtype != ks.dtype:
        keys = keys.astype(ks.dtype)
    if values.dtype != vs.dtype:
        values = values.astype(vs.dtype)
    own = getattr(cache, "_kv_own", None)
    if (INPLACE and own is not None and own[1] is ks and own[2] is vs
            and own[0] == id(cache) and getattr(cache, "_kv_hw", None) == prev):
        ks = _kq.kv_append(ks, keys, prev)
        vs = _kq.kv_append(vs, values, prev)
    else:
        # New handles, so a holder of the old arrays keeps its values. The
        # write donates a buffer nothing else holds and copies otherwise.
        ks, vs = ks.__copy__(), vs.__copy__()
        ks[..., prev:end, :] = keys
        vs[..., prev:end, :] = values
    cache.keys, cache.values = ks, vs
    cache._kv_own = (id(cache), ks, vs)
    cache._kv_hw = end


def _same(side, held) -> bool:
    # The MTP gate's restore rebuilds the sequences of a quantized cache,
    # so compare their arrays, not the sequences.
    return (isinstance(held, (tuple, list)) and len(side) == len(held)
            and all(a is b for a, b in zip(side, held)))


def _kv_write_parts(cache, keys, values, prev, end, routed) -> None:
    ks, vs = cache.keys, cache.values
    if not routed:
        for b, x in zip(ks, keys):
            b[..., prev:end, :] = x
        for b, x in zip(vs, values):
            b[..., prev:end, :] = x
        return
    own = getattr(cache, "_kv_own", None)
    inplace = (INPLACE and own is not None and own[0] == id(cache)
               and getattr(cache, "_kv_hw", None) == prev
               and _same(ks, own[1]) and _same(vs, own[2]))
    out = []
    for side, new in ((ks, keys), (vs, values)):
        parts = []
        for b, x in zip(side, new):
            if x.dtype != b.dtype:
                x = x.astype(b.dtype)
            if inplace:
                b = _kq.kv_append(b, x, prev)
            else:
                b = b.__copy__()
                b[..., prev:end, :] = x
            parts.append(b)
        out.append(type(side)(parts))
    cache.keys, cache.values = out
    cache._kv_own = (id(cache), cache.keys, cache.values)
    cache._kv_hw = end


def grow_levers() -> tuple:
    """(divisor, cap): the regrow increment is one ``divisor``-th of the
    capacity, at most ``cap`` rows. A divisor of 0 grows by one step; a
    cap of 0 means no cap."""
    div = int(os.environ.get("GMLX_KV_GROW_DIV", "8"))
    cap = int(os.environ.get("GMLX_KV_GROW_CAP", "8192"))
    return max(0, div), max(0, cap)


def _round_up(rows, step):
    return (rows + step - 1) // step * step


def _increment(capacity, step, div, cap):
    if div == 0:
        return step
    inc = _round_up(max(step, capacity // div), step)
    if cap > 0:
        inc = min(inc, _round_up(cap, step))
    return inc


@lru_cache(maxsize=16)
def _grid(step, div, cap):
    points = [step]
    while points[-1] < 1 << 22:
        points.append(points[-1] + _increment(points[-1], step, div, cap))
    return tuple(points)


def grid_capacity(rows, step, div=None, cap=None) -> int:
    """The smallest point of the capacity grid at or above ``rows``."""
    if div is None:
        div, cap = grow_levers()
    rows = max(int(rows), 1)
    if div == 0:
        return _round_up(rows, step)
    points = _grid(step, div, cap)
    i = bisect_left(points, rows)
    if i < len(points):
        return points[i]
    c = points[-1]
    while c < rows:
        c += _increment(c, step, div, cap)
    return c


def _parts(side):
    # The packed parts of a quantized side are a tuple, or a list after
    # ``to_quantized``.
    return tuple(side) if isinstance(side, (tuple, list)) else (side,)


def _like(side, parts):
    return type(side)(parts) if isinstance(side, (tuple, list)) else parts[0]


def grow(cache, keys, values, prev, need, make, grid=True, routed=True) -> None:
    """Allocate or regrow ``cache.keys`` and ``cache.values`` to hold
    ``need`` rows: the grid point at or above it, or the next multiple of
    ``cache.step`` when ``grid`` is False. ``make(x, rows)`` returns zeros
    shaped like one side ``x`` of the update with ``rows`` rows (an array,
    or a tuple for the quantized caches). A buffer whose offset is not a
    multiple of the step is cut to the offset first, as upstream does."""
    step = cache.step
    if cache.keys is None:
        rows = grid_capacity(need, step) if grid else _round_up(need, step)
        cache.keys, cache.values = make(keys, rows), make(values, rows)
    else:
        def extend(side, new):
            parts = _parts(side)
            if prev % step != 0:
                parts = tuple(a[..., :prev, :] for a in parts)
            base = parts[0].shape[2]
            if grid:
                rows = grid_capacity(need, step) - base
            else:
                rows = _round_up(need - prev, step)
            pads = _parts(make(new, rows))
            return _like(side, tuple(
                mx.concatenate([a, p], axis=2) for a, p in zip(parts, pads)))

        cache.keys = extend(cache.keys, keys)
        cache.values = extend(cache.values, values)
    if routed:
        mark_fresh(cache, prev)
