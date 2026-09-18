# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Asher Feldman
"""DeepSeek-V4.1-Flash-Vision image token blocks (weight-free geometry).

Ports the reference ``inference/image_processor.py`` minus the pixel work,
which lives with the other GGUF image processors in ``gmlx.load.vlm``.
Shared by the processor, which expands each image placeholder into a block
of sentinel ids, and the VLM container, which lays the aligner rows onto
the IMAGE slots.

Block layout (reference ``image_token_types``), all ids = ``vocab_size +
type``::

    START (IMAGE * n_llm_w + NEWLINE) * n_llm_h END

Plain reading order. V4.1 drops V4's lead pads, pad rows and column-wise
pair interleave, because it neither pools 4 tokens nor attends inside the
block.
"""

from __future__ import annotations

import math

import numpy as np

IMAGE_START, IMAGE, IMAGE_NEW_LINE, IMAGE_END = range(4)
N_BLOCK_TYPES = 4


def num_image_tokens(n_llm_h: int, n_llm_w: int) -> int:
    """Tokens one block occupies, START and END included."""
    return n_llm_h * (n_llm_w + 1) + 2


def llm_grid(best_height: int, best_width: int, patch_size: int,
             downsample_ratio: int) -> tuple[int, int]:
    """Token grid the aligner produces from a patch grid of this pixel size."""
    return (math.ceil((best_height // patch_size) / downsample_ratio),
            math.ceil((best_width // patch_size) / downsample_ratio))


def solve_resize_ratio(height: int, width: int, patch_size: int,
                       downsample_ratio: int, max_n_token: int):
    """Largest aspect-preserving pixel size whose grid still fits the budget."""
    r = height / width
    max_w_float = math.sqrt((max_n_token - 2) / r + 0.25) - 0.5
    max_h_float = max_w_float * r
    cell = patch_size * downsample_ratio
    if max_w_float < 1.0:                      # very tall: one column
        return (max_n_token - 2) // 2 * cell, cell
    if max_h_float < 1.0:                      # very wide: one row
        return cell, (max_n_token - 3) * cell
    beta = min(math.floor(max_w_float) * cell / width,
               math.floor(max_h_float) * cell / height)
    return (math.floor(height * beta / patch_size) * patch_size,
            math.floor(width * beta / patch_size) * patch_size)


def safe_resize(height: int, width: int, best_height: int, best_width: int,
                patch_size: int, downsample_ratio: int, max_n_token: int):
    """Shrink until the image costs at most ``max_n_token`` LLM tokens."""
    n_llm_h, n_llm_w = llm_grid(best_height, best_width, patch_size,
                                downsample_ratio)
    if num_image_tokens(n_llm_h, n_llm_w) > max_n_token:
        best_height, best_width = solve_resize_ratio(
            height, width, patch_size, downsample_ratio, max_n_token)
        n_llm_h, n_llm_w = llm_grid(best_height, best_width, patch_size,
                                    downsample_ratio)
        if num_image_tokens(n_llm_h, n_llm_w) > max_n_token:
            raise ValueError(
                f"{n_llm_h}x{n_llm_w} still costs "
                f"{num_image_tokens(n_llm_h, n_llm_w)} tokens over the "
                f"{max_n_token} budget")
    return n_llm_h, n_llm_w, best_height, best_width


def plan_image_grid(width: int, height: int, *, patch_size: int,
                    downsample_ratio: int, max_n_token: int, min_pixels: int,
                    max_wh_ratio: int | None):
    """Resize plan for one image; a pure function of its arguments.

    Returns ``(n_llm_h, n_llm_w, best_height, best_width)``.
    """
    p = patch_size
    if max_wh_ratio is not None and width > height * max_wh_ratio:
        width = height * max_wh_ratio
    if 0 < width * height < min_pixels:
        ratio = (min_pixels / (width * height)) ** 0.5
        width = int(width * ratio)
        height = int(height * ratio)
    best_width = math.ceil(width / p) * p
    best_height = math.ceil(height / p) * p
    return safe_resize(height, width, best_height, best_width, p,
                       downsample_ratio, max_n_token)


def image_token_types(n_llm_h: int, n_llm_w: int) -> np.ndarray:
    """The aligner grid in reading order, one NEWLINE per row."""
    types = [IMAGE_START]
    types += ([IMAGE] * n_llm_w + [IMAGE_NEW_LINE]) * n_llm_h
    types.append(IMAGE_END)
    return np.array(types, dtype=np.int64)


def build_image_block(n_llm_h: int, n_llm_w: int):
    """``(types, perm)``: block token types in stream order, and the
    aligner-row index for each IMAGE slot. Reading order, so ``perm`` is
    the identity; it exists to match the V4 container's call shape."""
    return (image_token_types(n_llm_h, n_llm_w),
            np.arange(n_llm_h * n_llm_w, dtype=np.int64))


def block_length(n_llm_h: int, n_llm_w: int) -> int:
    return num_image_tokens(n_llm_h, n_llm_w)
