"""Gemma-4 attention masks under bidirectional image attention.

transformers builds them with ``create_masks_for_vision_model``:
full-attention layers stay causal, and sliding layers take
AND(window, OR(causal, block)), so a token of an image block sees the
rest of its block within the sliding window. llama.cpp's gemma4 graph
does the same (non-causal on sliding layers only). mlx-vlm 0.7.3 ORs the
block into both layer types and past the window.

``image_masks`` wraps a stock-shaped ``_make_masks`` body. The body runs
without token types, so it builds the causal masks, one object per layer
type (the batched-sdpa pad relay keys on that identity). For a request
with image or video tokens the sliding mask is then rebuilt as an array
with the block overlay. As in stock, the overlay applies only to a
one-pass square mask from column 0.
"""
from __future__ import annotations

import mlx.core as mx
from mlx_vlm.models.gemma4 import language as _G


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
    """``base`` with each block token seeing its block's keys inside the
    window: key k is visible from query q when k > q - window."""
    n = base.shape[-1]
    if token_types.shape[1] != n or base.shape[-2] != n:
        return base
    ids = _block_ids(token_types)
    same = (ids[:, :, None] != -1) & (ids[:, :, None] == ids[:, None, :])
    pos = mx.arange(n)
    in_window = pos[None, :] > pos[:, None] - window
    return base | mx.expand_dims(same & in_window, 1)


def image_masks(self, h, cache, token_types, build):
    """The masks ``build`` makes, with the sliding ones rebuilt for image
    blocks when the setting is "vision" and the prompt holds image or video
    tokens. Full-attention masks stay causal."""
    masks = build(self, h, cache, None)
    if not (getattr(self.config, "use_bidirectional_attention", None) == "vision"
            and h.shape[1] > 1 and visual_tokens(token_types)):
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
