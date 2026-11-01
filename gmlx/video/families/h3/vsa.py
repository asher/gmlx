# SPDX-License-Identifier: Apache-2.0 AND MIT
# The tile map is ported from build_h3_segment_geometry and the index maps
# of MiniMaxH3VSAGeometry in FastVideo's
# fastvideo/mlx_runtime/minimax_h3_vsa.py at commit 4d0a10d, modified for
# gmlx (Apache-2.0; see licenses/FastVideo-LICENSE).
"""FastH3's video sparse attention (VSA) over one packed H3 sequence.

The packed rows are cut into tiles of 64: each dense segment (text,
audio, audio anchors) into chunks of its own rows, then the target video
into 4 x 4 x 4 tiles of its (latent frames, H / 2, W / 2) token grid.
Each video query tile attends every dense tile and its top-scoring video
tiles by pooled q and k means; the dense rows attend every row; the gate
of each row adds a pooled-compression term.

``tile_map`` is FastVideo's tile map of a segment list, in numpy.
``geometry`` converts it, once per segment list, to mlx-kquant's
``VSAGeometry``, outside any compiled function.
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import mlx_kquant as kq
import numpy as np

from gmlx.video.layout import Layout

TILE = 64
_TILE_SHAPE = (4, 4, 4)


def _kq(name: str):
    """mlx-kquant's op ``name``. The loader refuses a VSA variant on a
    build without ``vsa_attention`` before any call."""
    return getattr(kq, name)


@dataclass(frozen=True)
class TileMap:
    """FastVideo's tile map of a packed sequence of S rows: the dense
    tiles first, then the video tiles, each holding its live rows and
    then pad slots.

    ``variable_block_sizes`` [tiles] counts the live rows of each tile;
    ``tile_gather_index`` [tiles * 64] holds the packed row of each slot,
    S for a pad slot; ``prefix_gather_index`` [dense tiles * 64] the dense
    row of each dense slot, ``prefix_length`` for a pad slot;
    ``untile_index`` [S] the slot of each packed row. ``video_tile_spans``
    is the [start, end) tiles of each video region."""
    variable_block_sizes: np.ndarray
    num_prefix_tiles: int
    prefix_length: int
    tile_gather_index: np.ndarray
    prefix_gather_index: np.ndarray
    untile_index: np.ndarray
    video_tile_spans: tuple


def segments(layout: Layout, patch: tuple) -> tuple:
    """FastVideo's segments of an H3 layout, in packed order: the target
    video as its token grid, every other segment as its row count. FastH3
    is text to video, so a video anchor, a second video region, is
    refused."""
    anchors = sum(s.kind == "cond" for s in layout.segments)
    if anchors:
        raise ValueError(
            f"FastH3's video sparse attention runs over one video region, "
            f"the target video, and this layout also has {anchors} video "
            f"anchor segment{'s' if anchors > 1 else ''}")
    _, t, h, w = layout.signature[:4]
    grid = (t // patch[0], h // patch[1], w // patch[2])
    return tuple(grid if s.kind == "video" else s.rows
                 for s in layout.segments if s.rows)


def _video_tile_sizes(grid: tuple) -> np.ndarray:
    def sizes(n: int, tile: int) -> np.ndarray:
        count = math.ceil(n / tile)
        out = np.full((count,), tile, dtype=np.int64)
        rem = n - (count - 1) * tile
        out[-1] = rem if rem > 0 else tile
        return out

    t, h, w = (sizes(n, s) for n, s in zip(grid, _TILE_SHAPE))
    return (t[:, None, None] * h[None, :, None] * w[None, None, :]).reshape(-1)


def _video_tile_rows(grid: tuple) -> np.ndarray:
    """The rows of a video grid, tile by tile in (t, h, w) tile order."""
    t, h, w = grid
    st, sh, sw = _TILE_SHAPE
    rows = np.arange(t * h * w, dtype=np.int64).reshape(t, h, w)
    return np.concatenate([
        rows[a:a + st, b:b + sh, c:c + sw].reshape(-1)
        for a in range(0, t, st) for b in range(0, h, sh)
        for c in range(0, w, sw)])


def tile_map(segs: tuple) -> TileMap:
    """The tile map of ``segs``: dense row counts and video grids
    (t, h, w) in packed order."""
    dense_sizes: list[int] = []
    dense_rows: list[np.ndarray] = []
    regions: list[tuple[int, tuple]] = []
    cursor = 0
    for seg in segs:
        if isinstance(seg, tuple):
            regions.append((cursor, seg))
            cursor += math.prod(seg)
            continue
        if seg <= 0:
            continue
        full, rem = divmod(seg, TILE)
        dense_sizes += [TILE] * full + ([rem] if rem else [])
        dense_rows.append(np.arange(cursor, cursor + seg, dtype=np.int64))
        cursor += seg
    if not regions:
        raise ValueError("video sparse attention needs a video region")
    n_prefix = len(dense_sizes)
    rows, sizes, spans = list(dense_rows), [np.asarray(dense_sizes,
                                                       np.int64)], []
    at = n_prefix
    for start, grid in regions:
        s = _video_tile_sizes(grid)
        rows.append(_video_tile_rows(grid) + start)
        sizes.append(s)
        spans.append((at, at + int(s.size)))
        at += int(s.size)
    order = np.concatenate(rows)
    vbs = np.concatenate(sizes)
    slots = (np.arange(vbs.size, dtype=np.int64)[:, None] * TILE
             + np.arange(TILE, dtype=np.int64)[None, :])
    live = slots[np.arange(TILE)[None, :] < vbs[:, None]]
    untile = live[np.argsort(order, kind="stable")]
    prefix = int(sum(dense_sizes))
    gather = np.full(vbs.size * TILE, cursor, dtype=np.int32)
    gather[untile] = np.arange(cursor, dtype=np.int32)
    pgather = np.full(n_prefix * TILE, prefix, dtype=np.int32)
    pgather[untile[order[:prefix]]] = np.arange(prefix, dtype=np.int32)
    return TileMap(variable_block_sizes=vbs.astype(np.int32),
                   num_prefix_tiles=n_prefix, prefix_length=prefix,
                   tile_gather_index=gather, prefix_gather_index=pgather,
                   untile_index=untile.astype(np.int32),
                   video_tile_spans=tuple(spans))


@functools.lru_cache(maxsize=4)
def geometry(segs: tuple) -> Any:
    """mlx-kquant's ``VSAGeometry`` of ``segs``, built once per segment
    list."""
    m = tile_map(segs)
    return _kq("VSAGeometry")(
        variable_block_sizes=mx.array(m.variable_block_sizes),
        num_prefix_tiles=m.num_prefix_tiles, prefix_length=m.prefix_length,
        tile_gather_index=mx.array(m.tile_gather_index),
        prefix_gather_index=mx.array(m.prefix_gather_index),
        untile_index=mx.array(m.untile_index),
        video_tile_spans=m.video_tile_spans, tile_size=TILE)

