# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 OpenBMB and the HuggingFace Inc. team.
# Ported from transformers src/transformers/models/minicpmv4_7/
# modular_minicpmv4_7.py at commit 674c0e3 (MiniCPMV4_7Model._group_visual_frames,
# get_vision_position_ids and get_rope_index) and modified for gmlx
# (Apache-2.0; see licenses/transformers-LICENSE).
"""Canvas M-RoPE positions for MiniCPM-V 4.7.

An image is one frame: an overview crop, then optional slice crops that
tile it. Every token of the frame shares one temporal position, the
canvas origin. The overview is stretched over the whole slice canvas,
each slice fills its own cell, and the ``<image>`` ``</image>``
``<slice>`` ``</slice>`` and newline markers are pinned to canvas corners
and edges. Text after the frame resumes at ``origin + max(CH, CW) + 1``.

The reference reads crops from ``mm_token_type_ids``. Here a crop is a
maximal run of the image placeholder id, which is the same set of runs.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CanvasTokens:
    """The token ids canvas positions read."""

    image_token_id: int
    image_start_id: int
    image_end_id: int
    slice_start_id: int
    slice_end_id: int
    newline_id: int

    @property
    def markers(self) -> tuple[int, ...]:
        # <slice> first: it alone tells a slice from a new frame's overview.
        return (self.slice_start_id, self.slice_end_id, self.image_start_id,
                self.image_end_id, self.newline_id)


def torch_linspace_round(end: int, steps: int) -> np.ndarray:
    """``torch.linspace(0, end, steps).round()`` on the CPU, as int64.

    torch computes linspace in float32 from both ends: the first half as
    ``start + step * i``, the second half as ``end - step * (steps - 1 - i)``.
    ``round`` breaks ties to even. Exact rational rounding differs from it
    at reachable canvases, for example ``(end, steps) = (47, 23)``.
    """
    if steps <= 0:
        return np.zeros(0, dtype=np.int64)
    if steps == 1:
        return np.zeros(1, dtype=np.int64)
    f32 = np.float32
    end32 = f32(end)
    step = f32(end32 / f32(steps - 1))
    i = np.arange(steps)
    half = steps // 2
    lo = (step * i[:half].astype(np.float32)).astype(np.float32)
    hi = (end32 - (step * (steps - 1 - i[half:]).astype(np.float32))
          .astype(np.float32)).astype(np.float32)
    return np.rint(np.concatenate([lo, hi])).astype(np.int64)


def _vision_positions(origin: int, grid_hw, merge: int, canvas_h: int = 0,
                      canvas_w: int = 0, h_off: int = 0,
                      w_off: int = 0) -> np.ndarray:
    """``(3, h * w)`` positions ``[T, H, W]`` of one crop on a canvas."""
    gh = int(grid_hw[0]) // merge
    gw = int(grid_hw[1]) // merge
    canvas_h = canvas_h or gh
    canvas_w = canvas_w or gw
    h = torch_linspace_round(canvas_h - 1, gh) + h_off
    w = torch_linspace_round(canvas_w - 1, gw) + w_off
    hh, ww = np.meshgrid(h, w, indexing="ij")
    pos = np.stack([np.zeros_like(hh), hh, ww]).reshape(3, -1)
    return pos + origin


def _frame_end(last_crop_end: int, ids: list[int], markers, limit: int) -> int:
    end = last_crop_end
    while end < limit and ids[end] in markers:
        end += 1
    return end


def _visual_groups(ids: list[int], toks: CanvasTokens):
    """``(group_start, group_end, frames)`` per run of frames that only
    markers separate. A frame is ``(thumb_start, thumb_end, frame_end,
    slices)`` and a slice is ``(start, end)``."""
    markers = toks.markers
    is_image = [int(t == toks.image_token_id) for t in ids]
    groups: list = []
    open_slices = None
    previous_end = 0
    for kind, run in itertools.groupby(enumerate(is_image), lambda x: x[1]):
        if kind == 0:
            continue
        run = list(run)
        crop = (run[0][0], run[-1][0] + 1)
        if open_slices is not None and ids[crop[0] - 1] == toks.slice_start_id:
            open_slices.append(crop)
            previous_end = crop[1]
            continue
        open_slices = []
        frame = (crop[0], crop[1], open_slices)
        gap = ids[previous_end:crop[0] - 1]
        if groups and all(t in markers for t in gap):
            groups[-1][1].append(frame)
        else:
            groups.append((crop[0] - 1, [frame]))
        previous_end = crop[1]

    out = []
    for start, frames in groups:
        built = [
            (ts, te, _frame_end(sl[-1][1] if sl else te, ids, markers, len(ids)), sl)
            for ts, te, sl in frames
        ]
        out.append((start, built[-1][2], built))
    return out


def _row_positions(ids: list[int], grids, toks: CanvasTokens, merge: int
                   ) -> np.ndarray:
    """``(3, L)`` canvas positions of one unpadded row. ``grids`` yields the
    ``(h, w)`` patch grid of each crop in order."""
    n = len(ids)
    pos = np.broadcast_to(np.arange(n, dtype=np.int64), (3, n)).copy()
    cur = 0
    idx = 0
    for g_start, g_end, frames in _visual_groups(ids, toks):
        if g_start > idx:
            pos[:, idx:g_start] = np.arange(g_start - idx) + cur
            cur += g_start - idx
        cursor = g_start
        for t_start, t_end, f_end, slices in frames:
            thumb = next(grids)
            f_start = t_start - 1
            if f_start > cursor:
                gap = f_start - cursor
                pos[:, cursor:f_start] = np.arange(gap) + cur
                cur += gap + 1
            f_end = min(f_end, g_end)
            origin = cur
            halo = max(origin - 1, 0)

            sh = sw = 0
            rows = cols = 0
            canvas_h = int(thumb[0]) // merge
            canvas_w = int(thumb[1]) // merge
            first = None
            if slices:
                # A gap wider than "</slice><slice>" holds the row's newline.
                cols = len(slices)
                for k in range(len(slices) - 1):
                    if slices[k + 1][0] - slices[k][1] > 2:
                        cols = k + 1
                        break
                rows = len(slices) // cols if cols > 0 else 1
                if rows * cols != len(slices):
                    rows, cols = 1, len(slices)
                first = next(grids)
                sh = int(first[0]) // merge
                sw = int(first[1]) // merge
                canvas_h = rows * sh
                canvas_w = cols * sw

            pos[:, f_start:f_end] = origin
            pos[1:, f_start] = halo
            if t_end < f_end:
                pos[1, t_end] = origin + canvas_h
                pos[2, t_end] = origin + canvas_w
            pos[:, t_start:t_end] = _vision_positions(
                origin, thumb, merge, canvas_h, canvas_w)

            for k, (s_start, s_end) in enumerate(slices):
                h_off = (k // cols) * sh
                w_off = (k % cols) * sw
                grid = first if k == 0 else next(grids)
                g_h = int(grid[0]) // merge
                g_w = int(grid[1]) // merge
                if s_start - 1 >= f_start:
                    pos[1, s_start - 1] = origin + h_off
                    pos[2, s_start - 1] = origin + w_off
                if s_end < f_end:
                    pos[1, s_end] = origin + h_off + g_h - 1
                    pos[2, s_end] = origin + w_off + g_w - 1
                pos[:, s_start:s_end] = _vision_positions(
                    origin, grid, merge, h_off=h_off, w_off=w_off)

            for k in range(len(slices) - 1):
                a, b = slices[k][1], slices[k + 1][0]
                if b - a <= 2:
                    continue
                edge_h = (k // cols + 1) * sh - 1
                edge_w = cols * sw
                for j in range(a + 1, b - 1):
                    pos[1, j] = origin + edge_h
                    pos[2, j] = origin + edge_w

            cur = origin + max(canvas_h, canvas_w) + 1
            cursor = f_end

        if cursor < g_end:
            pos[:, cursor:g_end] = np.arange(g_end - cursor) + cur
            cur += g_end - cursor
        idx = g_end

    if idx < n:
        pos[:, idx:n] = np.arange(n - idx) + cur
    return pos


def canvas_rope_index(input_ids, target_sizes, toks: CanvasTokens, *,
                      merge: int, attention_mask=None
                      ) -> tuple[np.ndarray, np.ndarray]:
    """``(position_ids [3, B, L], rope_deltas [B, 1])`` for a batch.

    ``target_sizes`` holds the ``(h, w)`` patch grid of every crop in
    prompt order across the batch. ``merge`` is the patch-to-token factor:
    4 in 16x mode, 2 in 4x mode. Padding (``attention_mask == 0``) keeps
    position 0 and does not shift the canvas. The deltas count against the
    padded row length, as the qwen3.5 text model's decode expects.
    """
    ids = np.asarray(input_ids, dtype=np.int64)
    if ids.ndim == 1:
        ids = ids[None]
    b, length = ids.shape
    mask = None if attention_mask is None else np.asarray(attention_mask).astype(bool)
    grids = iter(np.asarray(target_sizes, dtype=np.int64).reshape(-1, 2).tolist()
                 if target_sizes is not None else [])
    out = np.zeros((3, b, length), dtype=np.int64)
    for r in range(b):
        keep = mask[r] if mask is not None else np.ones(length, dtype=bool)
        row = ids[r][keep].tolist()
        try:
            out[:, r, keep] = _row_positions(row, grids, toks, merge)
        except StopIteration:
            raise ValueError(
                "MiniCPM-V canvas positions: the prompt has more image crops "
                "than target_sizes rows") from None
    deltas = out.max(axis=(0, 2)) + 1 - length
    return out, deltas.reshape(b, 1)


def sequential_rope_index(input_ids, attention_mask=None
                          ) -> tuple[np.ndarray, np.ndarray]:
    """Plain 1-D positions on all three rows, and zero deltas. Padding
    takes position 0 and the kept tokens count from 0."""
    ids = np.asarray(input_ids)
    if ids.ndim == 1:
        ids = ids[None]
    b, length = ids.shape
    if attention_mask is None:
        pos = np.broadcast_to(np.arange(length, dtype=np.int64), (b, length))
        deltas = np.zeros((b, 1), dtype=np.int64)
    else:
        m = np.asarray(attention_mask).astype(np.int64)
        pos = np.where(m > 0, np.cumsum(m, axis=-1) - 1, 0)
        deltas = pos.max(axis=-1, keepdims=True) + 1 - length
    return np.broadcast_to(pos, (3, b, length)).copy(), deltas.astype(np.int64)
