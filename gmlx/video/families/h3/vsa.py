# SPDX-License-Identifier: Apache-2.0 AND MIT
# The tile map is ported from build_h3_segment_geometry and the index maps
# of MiniMaxH3VSAGeometry in FastVideo's
# fastvideo/mlx_runtime/minimax_h3_vsa.py at commit 4d0a10d, and the traced
# call follows h3_vsa_attention of that file, by way of mlx-kquant's
# vsa_attention, modified for gmlx (Apache-2.0; see
# licenses/FastVideo-LICENSE).
"""FastH3's video sparse attention (VSA) over one packed H3 sequence.

The packed rows are cut into tiles of 64: each dense segment (text,
audio) into chunks of its own rows, then the target video into 4 x 4 x 4
tiles of its (latent frames, H / 2, W / 2) token grid. FastH3 runs text
to video only, so a layout with an anchor is refused (``anchor_refusal``).
Each video query tile attends every dense tile and its top-scoring video
tiles by pooled q and k means; the dense rows attend every row; the gate
of each row adds a pooled-compression term. mlx-kquant's
``vsa_attention`` runs the call.

``tile_map`` is FastVideo's tile map of a segment list, in numpy.
``geometry`` converts it, once per segment list, to mlx-kquant's
``VSAGeometry``, outside any compiled function.

``traced`` is a hook for tests: in its block, each call records its
block indices and its float32 pooled q and k means, and a call takes the
block indices of ``pins`` in place of its own routing. Outside it a call
is ``vsa_attention`` alone.
"""

from __future__ import annotations

import functools
import math
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import mlx.core as mx
import mlx_kquant as kq
import numpy as np

from gmlx.video.layout import Layout

from .family import FASTH3, TITLE

TILE = 64
_TILE_SHAPE = (4, 4, 4)

_local = threading.local()


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


def anchor_refusal(anchors: tuple) -> str | None:
    """Why FastH3 does not run a request with ``anchors`` (``Anchor`` or
    ``layout.AnchorRows``), or None for none. It runs text to video only,
    so it takes no keyframe or guide, video or audio."""
    n = len(anchors)
    if not n:
        return None
    return (f"{TITLE} {FASTH3} runs text to video only, and this request "
            f"has {n} anchor{'s' if n > 1 else ''}; run it without "
            f"keyframes, guides or audio anchors")


def segments(layout: Layout, patch: tuple) -> tuple:
    """FastVideo's segments of an H3 layout, in packed order: the target
    video as its token grid, every other segment as its row count. A
    layout with an anchor is refused, as ``api.sample`` refuses it before
    its first step."""
    why = anchor_refusal(layout.signature[5])
    if why:
        raise ValueError(why)
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


def attend(q: mx.array, k: mx.array, v: mx.array, gate: mx.array,
           geom: Any, sparsity: float) -> mx.array:
    """Video sparse attention of packed bfloat16 [S, heads, D] rows with
    the gate of each row, exempt mode: every dense key tile is kept."""
    trace = getattr(_local, "trace", None)
    if trace is not None:
        return trace.attend(q, k, v, gate, geom, sparsity)
    return _kq("vsa_attention")(q, k, v, geom, sparsity, gate=gate)


# The test hook. Each piece follows mlx-kquant's vsa_attention op for op,
# so a call with its own routing gives the bits of the shipped call.

def _tile(x: mx.array, geom) -> mx.array:
    zero = mx.zeros((1,) + x.shape[1:], dtype=x.dtype)
    return mx.concatenate([x, zero])[geom.tile_gather_index]


def _pool(x: mx.array, vbs: mx.array) -> mx.array:
    n = x.shape[0] // TILE
    pooled = x.astype(mx.float32).reshape(n, TILE, *x.shape[1:]).sum(axis=1)
    pooled = pooled / vbs.astype(mx.float32)[:, None, None]
    return pooled.transpose(1, 0, 2)


def _dense(q, k, v, scale):
    return mx.fast.scaled_dot_product_attention(
        q.transpose(1, 0, 2)[None], k.transpose(1, 0, 2)[None],
        v.transpose(1, 0, 2)[None], scale=scale)[0].transpose(1, 0, 2)


def _gate_term(coarse, gate_tiled, tiles):
    heads, dim = gate_tiled.shape[1:]
    out_c = coarse[:, :tiles].transpose(1, 0, 2)
    g = gate_tiled[:tiles * TILE].reshape(tiles, TILE, heads, dim)
    term = (out_c[:, None, :, :] * g).reshape(tiles * TILE, heads, dim)
    return term.astype(gate_tiled.dtype)


def _route(scores: mx.array, n_prefix: int, sparsity: float) -> mx.array:
    """FastVideo's exempt routing of one video region: every dense tile,
    then the top k video tiles of each video query tile."""
    heads, n = scores.shape[0], scores.shape[-1]
    n_video = n - n_prefix
    keep = max(1, min(math.ceil((1.0 - sparsity) * n_video), n_video))
    if keep == n_video:
        return mx.broadcast_to(mx.arange(n, dtype=mx.int32)[None, None, :],
                               (heads, n_video, n))
    cols = scores[:, n_prefix:, n_prefix:n]
    top = mx.argpartition(-cols, kth=keep - 1, axis=-1)[:, :, :keep]
    top = top.astype(mx.int32) + n_prefix
    if not n_prefix:
        return top
    dense = mx.broadcast_to(mx.arange(n_prefix, dtype=mx.int32)[None, None],
                            (heads, n_video, n_prefix))
    return mx.concatenate([dense, top], axis=-1)


@dataclass(frozen=True)
class Call:
    """One traced call: its block indices [heads, video tiles, kept], the
    float32 pooled q and k means [heads, tiles, D] and their scores
    [heads, tiles, tiles]. A pinned call ran the pins' indices; a call on
    its own routing records whether its output has the bits of the
    shipped call (``shipped``)."""
    block_idx: np.ndarray
    q_pooled: np.ndarray
    k_pooled: np.ndarray
    scores: np.ndarray
    pinned: bool
    shipped: bool | None


@dataclass
class Trace:
    """The calls of a ``traced`` block, in order. ``pins(n)`` gives call
    n's block indices, or None for its own routing."""
    pins: Callable | None = None
    calls: list = field(default_factory=list)

    def attend(self, q, k, v, gate, geom, sparsity):
        pin = None if self.pins is None else self.pins(len(self.calls))
        heads, dim = q.shape[1:]
        scale = dim ** -0.5
        vbs, n_prefix = geom.variable_block_sizes, geom.num_prefix_tiles
        qt, kt, vt, gt = (_tile(x, geom) for x in (q, k, v, gate))
        qp, kp = _pool(qt, vbs), _pool(kt, vbs)
        scores = (qp @ kp.transpose(0, 2, 1)) / (dim ** 0.5)
        coarse = mx.softmax(scores, axis=-1) @ _pool(vt, vbs)
        idx = (_route(scores, n_prefix, sparsity) if pin is None
               else mx.array(np.asarray(pin), dtype=mx.int32))
        num = mx.full(idx.shape[:2], idx.shape[-1], dtype=mx.int32)
        n = geom.prefix_length
        head = _dense(q[:n], k, v, scale) if n else q[:0]
        video = _kq("vsa_block_sparse")(
            qt, kt, vt, idx, num, vbs, n_prefix, scale, tile_size=TILE,
            coarse=coarse, gate=gt)[n_prefix * TILE:]
        zero = mx.zeros((1, heads, dim), dtype=q.dtype)
        prefix = mx.concatenate([head, zero])[geom.prefix_gather_index]
        prefix = prefix + _gate_term(coarse, gt, n_prefix)
        out = mx.concatenate([prefix, video], axis=0)[geom.untile_index]
        shipped = None
        if pin is None:
            ref = _kq("vsa_attention")(q, k, v, geom, sparsity, gate=gate)
            shipped = bool(mx.array_equal(out, ref).item())
            out = ref
        self.calls.append(Call(
            block_idx=np.array(idx), q_pooled=np.array(qp),
            k_pooled=np.array(kp), scores=np.array(scores),
            pinned=pin is not None, shipped=shipped))
        return out


@contextmanager
def traced(pins: Callable | None = None) -> Iterator[Trace]:
    """Trace every call on this thread in the block (a test hook)."""
    prev = getattr(_local, "trace", None)
    _local.trace = trace = Trace(pins)
    try:
        yield trace
    finally:
        _local.trace = prev
