# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Asher Feldman
"""Rope positions of Hy4-flash where they are not the token slots.

Every key and query is rotated at three positions (t, h, w), interleaved
over the rope pairs (``pair_axes``). A text token has t = h = w, which is
the plain rope at that position. An image is H rows of W + 1 slots (W
merged cells and a newline). Its slot in row r, column c sits at
(p, p + r, p + c), where p is the position the image starts at, and the
image moves the position on by max(H, W + 1), fewer than its slots. So
the text after an image sits before its own slot index.

Windows, causal masks and the indexer's pools stay on slots. Only the
rope reads these positions.

The processor writes each image slot's kind into the ids as
``vocab_size + kind``. The walk's state between calls rides the engram
history cache as a small int array, with a host copy beside it so a decode
step reads no array.
"""

from __future__ import annotations

import functools
from typing import List, Optional, Sequence

import mlx.core as mx

TEXT = -1
PATCH, NEWLINE, LAST = 0, 1, 2
N_SLOT_KINDS = 3

# The reference config's mrope_section (t, h, w). The GGUF does not carry
# it.
ROPE_SECTIONS = (12, 10, 10)

_HOST = "_hy4_positions"


def image_slot_kinds(rows: int, cols: int) -> List[int]:
    """The slot kinds of an image of ``rows`` x ``cols`` merged cells, in
    prompt order: each row's cells, then its newline. The last newline is
    LAST, which closes the image."""
    kinds = ([PATCH] * cols + [NEWLINE]) * rows
    kinds[-1] = LAST
    return kinds


def image_advance(rows: int, cols: int) -> int:
    """How far an image of ``rows`` x ``cols`` merged cells moves the
    position on."""
    return max(rows, cols + 1)


@functools.cache
def pair_axes(n_pairs: int, sections: tuple = ROPE_SECTIONS) -> mx.array:
    """The axis (0 t, 1 h, 2 w) each rope pair reads: h on pairs 1, 4, ...
    and w on pairs 2, 5, ..., as many as their sections hold, t on the
    rest. Pair i is the i-th frequency in either rope layout."""
    _, sec_h, sec_w = (tuple(sections) + (0, 0))[:3]
    return mx.array(
        [1 if i % 3 == 1 and i < 3 * sec_h
         else 2 if i % 3 == 2 and i < 3 * sec_w else 0
         for i in range(n_pairs)], dtype=mx.int32)


class RopePositions:
    """One call's rope positions. ``start`` is set where its rows are text
    at consecutive positions, which is the plain rope from there; else
    ``rows`` holds (t, h, w) per row, [L, 3]."""

    def __init__(self, offset: int, tail: tuple, own: Optional[list],
                 start: Optional[int]):
        self.start = start
        self.rows = None if own is None else mx.array(own, dtype=mx.int32)
        self._offset = offset
        self._tail = tail
        self._own = own
        self._slots: dict = {}

    def _at(self, slot: int) -> tuple:
        i = slot - self._offset
        if i < 0:
            return self._tail[len(self._tail) + i]
        if self._own is not None:
            return self._own[i]
        p = self.start + i
        return (p, p, p)

    def slots(self, first: int, step: int, count: int) -> mx.array:
        """The positions of ``count`` slots ``step`` apart from ``first``,
        which may lie a few slots before the call: [count], or [count, 3]
        where one of them is an image slot."""
        key = (first, step, count)
        got = self._slots.get(key)
        if got is None:
            rows = [self._at(first + step * i) for i in range(count)]
            if all(t == h == w for t, h, w in rows):
                got = mx.array([t for t, _, _ in rows], dtype=mx.int32)
            else:
                got = mx.array(rows, dtype=mx.int32)
            self._slots[key] = got
        return got


def _load(slot, keep: int):
    """The walk's state held on ``slot``, or None before the first image.
    The host copy is good while the array beside it is the one stored."""
    arr = slot[1]
    if arr is None:
        return None
    held = getattr(slot, _HOST, None)
    if held is not None and held[0] is arr:
        return held[1]
    # A restored cache: read the array once.
    v = [int(x) for x in arr.reshape(-1).tolist()]
    n = v[4]
    tail = tuple(tuple(v[5 + 3 * i:8 + 3 * i]) for i in range(n))
    state = (v[0], v[1], v[2], v[3], tail[-keep:] if keep else ())
    setattr(slot, _HOST, (arr, state))
    return state


def _store(slot, state, keep: int) -> None:
    delta, base, row, col, tail = state
    flat = [delta, base, row, col, len(tail)]
    for entry in tail:
        flat.extend(entry)
    flat.extend([0] * (3 * (keep - len(tail))))
    # One row, as the history beside it: the cache's batch operations
    # read a leading batch axis.
    arr = mx.array([flat], dtype=mx.int64)
    slot[1] = arr
    setattr(slot, _HOST, (arr, state))


def active(slot) -> bool:
    """Whether ``slot`` carries a walk: an image is in the context."""
    return slot is not None and len(slot.cache) > 1 and slot[1] is not None


def snapshot(slot):
    """The walk's state on ``slot`` as it stands, for ``restore``. None
    where no walk is active."""
    if not active(slot):
        return None
    return slot[1], getattr(slot, _HOST, None)


def restore(slot, held) -> None:
    """Put back a ``snapshot``: the calls since it are forgotten."""
    slot[1] = held[0]
    setattr(slot, _HOST, held[1])


def plan(slot, kinds: Optional[Sequence[int]], offset: int, n: int,
         keep: int) -> RopePositions:
    """The rope positions of a call of ``n`` rows from slot ``offset``.

    ``kinds`` is each row's slot kind (TEXT or an image kind), or None for
    text rows. ``slot`` is the cache member the walk's state is kept on
    (None for a call without a cache). ``keep`` is how many slots back a
    later call may ask for: a pooled key sits at its group's first slot."""
    state = None
    if slot is not None:
        if len(slot.cache) < 2:
            raise ValueError(
                "hy_v4_flash: this cache has no place for image positions; "
                "build it from the vision model's make_cache")
        state = _load(slot, keep)
    if state is None:
        first = max(0, offset - keep)
        state = (0, -1, 0, 0, tuple((s, s, s) for s in range(first, offset)))
    delta, base, row, col, tail = state
    pos = offset + delta

    own = None
    start = None
    if kinds is None or all(k == TEXT for k in kinds):
        if base >= 0:
            raise ValueError(
                "hy_v4_flash: text rows inside an image's slots")
        start = pos
        pos += n
        fresh = tuple((p, p, p) for p in range(pos - min(n, keep), pos))
    else:
        if len(kinds) != n:
            raise ValueError(
                f"hy_v4_flash: {len(kinds)} slot kinds for {n} rows")
        own = []
        for kind in kinds:
            if kind == TEXT:
                if base >= 0:
                    raise ValueError(
                        "hy_v4_flash: text rows inside an image's slots")
                own.append((pos, pos, pos))
                pos += 1
                continue
            if base < 0:
                base, row, col = pos, 0, 0
            own.append((base, base + row, base + col))
            if kind == PATCH:
                col += 1
            elif kind == NEWLINE:
                row, col = row + 1, 0
            else:
                pos, base = base + max(row + 1, col + 1), -1
        fresh = tuple(own[-keep:]) if keep else ()

    if slot is not None:
        new_tail = (tail + fresh)[-keep:] if keep else ()
        new_delta = pos - (offset + n) if base < 0 else 0
        _store(slot, (new_delta, base, row, col, new_tail), keep)
    return RopePositions(offset, tail, own, start)
