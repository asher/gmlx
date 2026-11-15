"""Spatial tiles of the H3 VAEs, apart from any model's weights.

A decoder trained on tiles of one size decodes a larger frame as
overlapping tiles of that size, blended linearly across each overlap, and
the encoder tiles a frame the same way. MiniMax-H3 video and MiniMax-H3-Image
tile at ``TILE`` pixels with overlaps of at least ``MIN_OVERLAP``. Importing
this module loads no mlx.
"""

from __future__ import annotations

import math

TILE = 256
MIN_OVERLAP = 64


def split_tiles(length: int, tile: int, min_overlap: int,
                unit: int) -> tuple[list[int], list[int], list[int]]:
    """Starts, lengths and overlaps of the tiles that cover ``length``
    pixels: the fewest tiles of ``tile`` pixels whose overlaps are all at
    least ``min_overlap``, with the room left over spread across the
    overlaps in steps of ``unit``, round robin from the first. A length
    one tile covers is one tile of that length."""
    if tile >= length:
        return [0], [length], []
    n = math.ceil(length / tile)
    while tile * n - min_overlap * (n - 1) < length:
        n += 1
    overlaps = [min_overlap] * (n - 1)
    for i in range((tile * n - sum(overlaps) - length) // unit):
        overlaps[i % (n - 1)] += unit
    starts = [0]
    for o in overlaps:
        starts.append(starts[-1] + tile - o)
    return starts, [tile] * n, overlaps
