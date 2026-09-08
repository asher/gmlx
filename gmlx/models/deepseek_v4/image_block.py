# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Asher Feldman
"""DeepSeek-V4-Flash-Vision-Exp image token blocks (weight-free geometry).

Ports the reference ``inference/image_processor.py`` (deepseek-ai/
DeepSeek-V4-Flash-Vision-Exp, HF commit 6821d6ad) minus the pixel work,
which lives with the other GGUF image processors in ``gmlx.load.vlm``.
Shared by the processor (which expands each ``<|deepseek_image|>``
placeholder into a block of sentinel ids) and the VLM container (which
needs the same permutation to lay the aligner rows onto the IMAGE slots).

Block layout (``build_image_block``), all ids = ``vocab_size + type``:

    [PAD * lead] START (rows of IMAGE * n_llm_w + NEWLINE, an odd row count
    padded by a full PAD row, adjacent row PAIRS interleaved column-wise)
    [PAD * pad_last] END

``lead = 3 - start_pos % 4`` aligns START to a multiple of 4 (the text
model's 4-token compressor pools); the pair interleave and ``pad_last`` keep
the body an even count. The whole block, lead pads included, is at most
``max_n_token`` (384) tokens: ``safe_resize`` shrinks the grid until the
body fits ``max_n_token - 3``.
"""

from __future__ import annotations

import math

import numpy as np

IMAGE_START, IMAGE_PAD, IMAGE, IMAGE_NEW_LINE, IMAGE_END = range(5)
N_BLOCK_TYPES = 5
COMPRESS_PAD_TO = 4


def lead_pads(start_pos: int) -> int:
    return COMPRESS_PAD_TO - 1 - int(start_pos) % COMPRESS_PAD_TO


def grid_tokens(best_height: int, best_width: int, patch_size: int,
                downsample_ratio: int) -> tuple[int, int, int]:
    """(n_llm_h, n_llm_w, body tokens incl. START/END, excl. lead pads)."""
    n_llm_h = math.ceil((best_height // patch_size) / downsample_ratio)
    n_llm_w = math.ceil((best_width // patch_size) / downsample_ratio)
    num_tokens = n_llm_h * (n_llm_w + 1) + 2
    if n_llm_h % 2 == 1:
        num_tokens += n_llm_w + 1
    num_tokens += (n_llm_h + 1) // 2 * (n_llm_w + 1) % 2 * 2
    return n_llm_h, n_llm_w, num_tokens


def solve_resize_ratio(height: int, width: int, patch_size: int,
                       downsample_ratio: int, max_n_token: int):
    r = height / width
    max_w_float = math.sqrt((max_n_token - 2) / r + 0.25) - 0.5
    max_h_float = max_w_float * r
    if max_w_float < 1.0:
        max_w = 1
        max_h = (max_n_token - 2) // (max_w + 1)
        if max_h % 2 == 1:
            max_h -= 1
        best_width = max_w * patch_size * downsample_ratio
        best_height = max_h * patch_size * downsample_ratio
    elif max_h_float < 2.0:
        max_h = 2
        max_w = ((max_n_token - 2) // max_h) - 1
        assert max_w > 1
        best_width = max_w * patch_size * downsample_ratio
        best_height = max_h * patch_size * downsample_ratio
    else:
        max_w = math.floor(max_w_float)
        max_h = math.floor(max_h_float)
        if max_h % 2 == 1:
            max_h -= 1
        beta = min(max_w * patch_size * downsample_ratio / width,
                   max_h * patch_size * downsample_ratio / height)
        best_width = math.floor(width * beta / patch_size) * patch_size
        best_height = math.floor(height * beta / patch_size) * patch_size
    n_llm_h, n_llm_w, num_tokens = grid_tokens(
        best_height, best_width, patch_size, downsample_ratio)
    return n_llm_h, n_llm_w, best_height, best_width, num_tokens


def safe_resize(height: int, width: int, best_height: int, best_width: int,
                patch_size: int, downsample_ratio: int, max_n_token: int):
    """Shrink (best_height, best_width) until the block body fits the
    budget less the 3 lead pads it may need."""
    max_n_token -= COMPRESS_PAD_TO - 1
    n_llm_h, n_llm_w, num_tokens = grid_tokens(
        best_height, best_width, patch_size, downsample_ratio)
    budget = max_n_token
    while num_tokens > max_n_token:
        n_llm_h, n_llm_w, best_height, best_width, num_tokens = solve_resize_ratio(
            height, width, patch_size, downsample_ratio, budget)
        budget -= 1
    return n_llm_h, n_llm_w, best_height, best_width


def resize_geometry(width: int, height: int, *, patch_size: int,
                    downsample_ratio: int, max_n_token: int,
                    max_wh_ratio: int | None, min_pixels: int):
    """(best_width, best_height, n_vit_h, n_vit_w, n_llm_h, n_llm_w,
    stretch) for a source image of (width, height); ``stretch`` is True
    when the reference resizes without aspect padding (w >= ratio * h)."""
    src_w, src_h = width, height
    if max_wh_ratio is not None and width > height * max_wh_ratio:
        width = height * max_wh_ratio
    if 0 < width * height < min_pixels:
        ratio = (min_pixels / (width * height)) ** 0.5
        width = int(width * ratio)
        height = int(height * ratio)
    p = patch_size
    best_width = math.ceil(width / p) * p
    best_height = math.ceil(height / p) * p
    n_llm_h, n_llm_w, best_height, best_width = safe_resize(
        height, width, best_height, best_width, p, downsample_ratio, max_n_token)
    n_vit_h, n_vit_w = best_height // p, best_width // p
    stretch = max_wh_ratio is not None and src_w >= max_wh_ratio * src_h
    return best_width, best_height, n_vit_h, n_vit_w, n_llm_h, n_llm_w, stretch


def build_image_block(n_llm_h: int, n_llm_w: int, start_pos: int):
    """(types, perm): ``types`` [n] int64 block token types in stream order
    (lead pads through END); ``perm`` the aligner-row index for each IMAGE
    slot, in stream order."""
    compress_pad = lead_pads(start_pos)
    pad_h = n_llm_h % 2
    rows = n_llm_h + pad_h
    row_len = n_llm_w + 1
    pad_last = rows // 2 * row_len % 2 * 2
    types = np.array(
        ([IMAGE] * n_llm_w + [IMAGE_NEW_LINE]) * n_llm_h
        + [IMAGE_PAD] * (row_len * pad_h), dtype=np.int64)
    order = np.arange(rows * row_len).reshape(rows // 2, 2, row_len)
    order = order.transpose(0, 2, 1).reshape(-1)
    image_idx = np.full((rows * row_len,), -1, dtype=np.int64)
    image_idx.reshape(rows, row_len)[:n_llm_h, :n_llm_w] = np.arange(
        n_llm_h * n_llm_w).reshape(n_llm_h, n_llm_w)
    perm = image_idx[order]
    perm = perm[perm >= 0]
    types = np.concatenate([
        np.full((compress_pad,), IMAGE_PAD, dtype=np.int64),
        np.array([IMAGE_START], dtype=np.int64),
        types[order],
        np.full((pad_last,), IMAGE_PAD, dtype=np.int64),
        np.array([IMAGE_END], dtype=np.int64),
    ])
    return types, perm


def block_length(n_llm_h: int, n_llm_w: int, start_pos: int) -> int:
    """Tokens the block at ``start_pos`` occupies, lead pads included."""
    _, _, body = grid_tokens(n_llm_h * 3, n_llm_w * 3, 1, 3)
    return lead_pads(start_pos) + body
