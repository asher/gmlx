# SPDX-License-Identifier: Apache-2.0 AND MIT
# SPDX-FileCopyrightText: Copyright 2026 Qwen-Image Team and The HuggingFace Team
# Ported from QwenImage21Rope, build_token_metadata and
# _qwenimage21_prefix_segments in diffusers'
# src/diffusers/models/transformers/transformer_qwenimage21.py at e0abab83
# and modified for gmlx (Apache-2.0; see licenses/diffusers-LICENSE).
"""Token layout and rotary tables of the Qwen-Image 2.1 joint sequence,
computed on the host.

The joint sequence is the encoder sequence with each image slot expanded to
four tokens, which the latents of the reference images overwrite in order,
followed by the target latents. Everything before the target is the prefix.

Each token has a position on three axes: frame, height and width. A text
token sits at one shared position on all three, and each text token
advances it by one. An image block freezes the frame axis at the position
reached, lays its tokens on a height and width grid centered on zero, and
then advances the position by the larger of its height and width in
tokens. The target is the last image block.

A prefix layout depends on the encoder sequence length, the image slots
and the reference shapes, and a target's positions on the position the
prefix reached and the target shape. The tables are memos of those.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

import mlx.core as mx
import numpy as np

SLOT_TOKENS = 4             # latent tokens per encoder image slot, 2 x 2

# The reference holds tables for positions 0 to 8191 and -1024 to -1.
_MIN_POS, _MAX_POS = -1024, 8191


@dataclass(frozen=True)
class PrefixLayout:
    """Where each prefix token comes from and where it sits.

    ``text_rows`` are the encoder rows that are text, ascending. ``source``
    maps a prefix position to a row of the text rows followed by all
    reference tokens. ``positions`` is (P, 3). ``segments`` are the runs
    ``(start, end, is_text)`` that attend as one block each. ``position``
    is where the target block starts on the frame axis.
    """
    text_rows: np.ndarray
    source: np.ndarray
    positions: np.ndarray
    segments: tuple
    position: int

    @property
    def tokens(self) -> int:
        return int(self.source.shape[0])


def grid(height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    """Row and column index of each token of an image block, row major,
    centered on zero."""
    rows = np.arange(-(height - height // 2), height // 2)
    cols = np.arange(-(width - width // 2), width // 2)
    return np.repeat(rows, width), np.tile(cols, height)


def _check_range(positions: np.ndarray) -> None:
    if positions.size and (positions.min() < _MIN_POS
                           or positions.max() > _MAX_POS):
        raise ValueError(
            f"a token position of {int(positions.min())} to "
            f"{int(positions.max())} is outside {_MIN_POS} to {_MAX_POS}, "
            "the range Qwen-Image 2.1 has rotary tables for")


@functools.lru_cache(maxsize=64)
def prefix_layout(n_rows: int, image_slots: tuple = (),
                  ref_shapes: tuple = ()) -> PrefixLayout:
    """The layout of a prefix of ``n_rows`` encoder rows. ``image_slots``
    are the rows the encoder gave to image tokens, and ``ref_shapes`` the
    ``(height, width)`` of each reference image in latent tokens. A
    reference fills the next ``height * width / 4`` slots, which must be
    consecutive rows."""
    slots = [int(s) for s in image_slots]
    if sorted(set(slots)) != slots or (slots and not
                                       0 <= slots[0] <= slots[-1] < n_rows):
        raise ValueError("image slots must be ascending rows of the "
                         "conditioning features, each named once")
    want = sum(int(h) * int(w) for h, w in ref_shapes)
    if SLOT_TOKENS * len(slots) != want:
        raise ValueError(
            f"{len(slots)} image slots hold {SLOT_TOKENS * len(slots)} "
            f"latent tokens, and the reference images have {want}")
    is_slot = np.zeros(n_rows, bool)
    is_slot[slots] = True
    text_rows = np.flatnonzero(~is_slot)
    n_text = int(text_rows.size)

    source: list[int] = []
    positions: list[tuple] = []
    segments: list[tuple] = []
    row = position = text = img = 0

    def text_until(stop: int) -> None:
        nonlocal row, position, text
        start = len(source)
        while row < stop:
            source.append(text)
            positions.append((position, position, position))
            position += 1
            text += 1
            row += 1
        if len(source) > start:
            segments.append((start, len(source), True))

    at = 0
    for h, w in ref_shapes:
        h, w = int(h), int(w)
        n = h * w // SLOT_TOKENS
        block = slots[at:at + n]
        if (h * w % SLOT_TOKENS or not block or len(block) != n
                or block != list(range(block[0], block[0] + n))):
            raise ValueError(
                f"a reference image of {h}x{w} latent tokens needs "
                f"{h * w / SLOT_TOKENS:g} consecutive image slots")
        at += n
        text_until(block[0])
        start = len(source)
        rows, cols = grid(h, w)
        source.extend(range(n_text + img, n_text + img + h * w))
        positions.extend(zip([position] * (h * w), rows.tolist(),
                             cols.tolist()))
        segments.append((start, len(source), False))
        img += h * w
        position += max(h, w)
        row = block[-1] + 1
    text_until(n_rows)
    pos = np.asarray(positions, np.int64).reshape(-1, 3)
    _check_range(pos)
    src = np.asarray(source, np.int64)
    for a in (text_rows, src, pos):
        a.setflags(write=False)
    return PrefixLayout(text_rows=text_rows, source=src, positions=pos,
                        segments=tuple(segments), position=position)


def target_positions(position: int, height: int, width: int) -> np.ndarray:
    """Positions (height * width, 3) of the target block, whose frame axis
    sits at ``position``."""
    rows, cols = grid(height, width)
    pos = np.stack([np.full(rows.shape, position), rows, cols], axis=1)
    pos = pos.astype(np.int64)
    _check_range(pos)
    return pos


def rope_freqs(dim: int, theta: float) -> np.ndarray:
    """float32 ``1 / theta ** (k / dim)`` for even k below ``dim``. The
    exponent and the reciprocal are float32, and the power is the float64
    power rounded once to float32. torch's own float32 power is one ulp
    off that at some elements."""
    k = np.arange(0, dim, 2).astype(np.float32) / np.float32(dim)
    power = np.power(float(theta), k.astype(np.float64)).astype(np.float32)
    return (np.float32(1) / power).astype(np.float32)


def rope_angles(positions: np.ndarray, axes_dims, theta: float) -> np.ndarray:
    """float32 angles (L, sum(axes_dims) // 2): each position times each
    frequency of its axis, as one float32 product."""
    out = []
    for axis, d in enumerate(axes_dims):
        p = positions[:, axis].astype(np.float32)
        out.append((p[:, None] * rope_freqs(d, theta)[None])
                   .astype(np.float32))
    return np.concatenate(out, axis=1)


def rope_tables(positions: np.ndarray, axes_dims,
                theta: float) -> tuple[mx.array, mx.array]:
    """cos and sin tables for positions (L, 3), float32."""
    ang = rope_angles(positions, axes_dims, theta)
    return mx.array(np.cos(ang)), mx.array(np.sin(ang))


@functools.lru_cache(maxsize=8)
def prefix_tables(axes_dims: tuple, theta: float, n_rows: int,
                  image_slots: tuple, ref_shapes: tuple):
    lay = prefix_layout(n_rows, image_slots, ref_shapes)
    return rope_tables(lay.positions, axes_dims, theta)


@functools.lru_cache(maxsize=8)
def target_tables(axes_dims: tuple, theta: float, position: int,
                  height: int, width: int):
    return rope_tables(target_positions(position, height, width), axes_dims,
                       theta)


def clear_caches() -> None:
    for memo in (prefix_layout, prefix_tables, target_tables):
        memo.cache_clear()
