"""Gemma-4 attention masks under bidirectional image attention.

transformers builds them with ``create_masks_for_vision_model``:
full-attention layers stay causal, and sliding layers take
AND(window, OR(causal, block)), so a token of an image or video block
sees the rest of its block within the sliding window. llama.cpp's gemma4
graph does the same (non-causal on sliding layers only). mlx-vlm 0.7.3
ORs the block into both layer types and past the window.

``image_masks`` wraps a stock-shaped ``_make_masks`` body. The body runs
without token types, so it builds the causal masks, one object per layer
type (the batched-sdpa pad relay keys on that identity). For a chunk
with image or video tokens the sliding mask is then rebuilt as an array
with the block overlay on its trailing columns, the chunk's own keys.
The prefill loops keep every block inside one chunk
(``gmlx.gen.media_spans``), so a block never reaches back into cached
keys.
"""
from __future__ import annotations

import mlx.core as mx
from gmlx._vendor.mlx_vlm.models.gemma4 import language as _G


def visual_tokens(token_types) -> bool:
    """True when the token types hold an image or video token."""
    return (token_types is not None
            and int(mx.sum((token_types == 1) | (token_types == 2)).item()) > 0)


def _block_ids(token_types: mx.array) -> mx.array:
    """Per position, the index of its image or video block, else -1."""
    is_vision = (token_types == 1) | (token_types == 2)
    prev = mx.concatenate(
        [mx.zeros_like(is_vision[:, :1]), is_vision[:, :-1]], axis=1)
    starts = is_vision & ~prev
    ids = mx.cumsum(starts.astype(mx.int32), axis=1) - 1
    return mx.where(is_vision, ids, -1)


def sliding_block_overlay(base: mx.array, token_types: mx.array,
                          window: int) -> mx.array:
    """``base`` of shape (..., n, cached + n) with each block token seeing
    its block's keys among the trailing n columns inside the window: key k
    is visible from query q when k > q - window."""
    n = token_types.shape[1]
    if base.shape[-2] != n or base.shape[-1] < n:
        return base
    ids = _block_ids(token_types)
    same = (ids[:, :, None] != -1) & (ids[:, :, None] == ids[:, None, :])
    pos = mx.arange(n)
    same = same & (pos[None, :] > pos[:, None] - window)
    cached = base.shape[-1] - n
    if cached:
        same = mx.concatenate(
            [mx.zeros((same.shape[0], n, cached), dtype=mx.bool_), same],
            axis=-1)
    return base | mx.expand_dims(same, 1)


def chunk_types(token_types, n: int, cache):
    """The token types of an ``n``-column chunk. A caller that passes the
    whole prompt's types with every chunk (mlx-vlm's ``generate_step``)
    gets them sliced at the cache offset; None when they cannot be
    aligned."""
    if token_types is None or token_types.shape[1] == n:
        return token_types
    offset = next((c.offset for c in cache
                   if c is not None and hasattr(c, "offset")), None)
    if isinstance(offset, int) and offset + n <= token_types.shape[1]:
        return token_types[:, offset:offset + n]
    return None


def image_masks(self, h, cache, token_types, build):
    """The masks ``build`` makes, with the sliding ones rebuilt for image
    blocks when the setting is "vision" and the chunk holds image or video
    tokens. Full-attention masks stay causal."""
    masks = build(self, h, cache, None)
    if not (getattr(self.config, "use_bidirectional_attention", None) == "vision"
            and h.shape[1] > 1):
        return masks
    token_types = chunk_types(token_types, h.shape[1], cache)
    if token_types is None or not visual_tokens(token_types):
        return masks
    sliding = [i for i, lyr in enumerate(self.layers)
               if lyr.layer_type == "sliding_attention"]
    if not sliding:
        return masks
    window = self.window_size
    mask = _G.create_attention_mask(
        h, cache[sliding[0]], window_size=window, return_array=True)
    if isinstance(mask, str) and mask == "causal":
        mask = _G.create_causal_mask(h.shape[1], window_size=window)
    if isinstance(mask, mx.array):
        mask = sliding_block_overlay(mask, token_types, window)
    for i in sliding:
        masks[i] = mask
    return masks
