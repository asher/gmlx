# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Asher Feldman
# Portions copyright (c) 2025 Prince Canuma (mlx-vlm, MIT)
# Portions copyright (c) 2023-2024 Apple Inc. (mlx-lm, MIT)
# Vendored from mlx-vlm 0.7.3, mlx_vlm/models/base.py lines 263 to 430.
# Source sha256: a7cfa38413bea057946c5d56c9edd34dbcceb33f6613529f5201ca2ad5b41120
# Extract sha256: bb2b1a234597dedb7c65809adf34e6e3b1a13375caecb814bc03035ee3df2651
"""mlx-vlm 0.7.3 mask and SDPA helpers, owned by gmlx.

``create_attention_mask``, ``create_ssm_mask``,
``align_attention_mask_to_scores``, ``quantized_scaled_dot_product_attention``,
``_turboquant_attention_applies`` and ``scaled_dot_product_attention`` from
``mlx_vlm/models/base.py``. Edits to the upstream lines: the imports are
written here and ``create_causal_mask`` comes from ``kv_cache``; the
TurboQuant cache classes stay in mlx-vlm and are looked up in
``sys.modules`` at call time; comments are ASCII.
"""

import sys
from typing import Optional

import mlx.core as mx
from mlx.utils import tree_map

from .kv_cache import create_causal_mask


def create_attention_mask(
    h, cache=None, window_size: Optional[int] = None, return_array: bool = False
):
    N = h.shape[1]
    if (
        cache is not None
        and not isinstance(cache, mx.array)
        and hasattr(cache, "make_mask")
    ):
        return cache.make_mask(N, return_array=return_array, window_size=window_size)
    if N == 1:
        return None
    if return_array or (window_size and N > window_size):
        return create_causal_mask(N, window_size=window_size)
    return "causal"


def create_ssm_mask(h, cache=None):
    if (
        cache is not None
        and not isinstance(cache, mx.array)
        and hasattr(cache, "make_mask")
    ):
        return cache.make_mask(h.shape[1])
    return None


def align_attention_mask_to_scores(mask, scores: mx.array):
    """Broadcast an attention mask onto *scores* without mis-aligning batch vs heads.

    Quantized GQA expands scores to 5D ``(B, n_kv_heads, n_repeats, L, K)`` while
    batch left-pad masks are typically 4D ``(B, 1, L, K)``. Right-aligning those
    shapes aliases ``B`` with ``n_kv_heads`` and crashes for multi-row batches
    (see #1567). Insert singleton dims *before* the last two axes until ranks
    match so the layout is ``(B, 1, ..., 1, L, K)``.
    """
    if mask is None or isinstance(mask, str):
        return mask

    # Grow rank by inserting size-1 axes immediately before (L, K).
    while mask.ndim < scores.ndim:
        insert_at = max(mask.ndim - 2, 0)
        mask = mx.expand_dims(mask, axis=insert_at)
    return mask


def quantized_scaled_dot_product_attention(
    queries: mx.array,
    q_keys: tuple[mx.array, mx.array, mx.array],
    q_values: tuple[mx.array, mx.array, mx.array],
    scale: float,
    mask: Optional[mx.array],
    group_size: int = 64,
    bits: int = 8,
) -> mx.array:
    B, n_q_heads, L, D = queries.shape
    n_kv_heads = q_keys[0].shape[-3]
    n_repeats = n_q_heads // n_kv_heads

    queries *= scale

    if n_repeats > 1:
        queries = mx.reshape(queries, (B, n_kv_heads, n_repeats, L, D))
        q_keys = tree_map(lambda x: mx.expand_dims(x, axis=-3), q_keys)
        q_values = tree_map(lambda x: mx.expand_dims(x, axis=-3), q_values)

    scores = mx.quantized_matmul(
        queries, *q_keys, transpose=True, group_size=group_size, bits=bits
    )
    if mask is not None:
        if isinstance(mask, str):
            qL, kL = scores.shape[-2:]
            q_indices = mx.arange(kL - qL, kL)
            k_indices = mx.arange(kL)
            mask = q_indices[:, None] >= k_indices[None]
        mask = align_attention_mask_to_scores(mask, scores)
        if mask.dtype == mx.bool_:
            scores = mx.where(mask, scores, mx.finfo(scores.dtype).min)
        else:
            scores += mask
    scores = mx.softmax(scores, axis=-1, precise=True)
    out = mx.quantized_matmul(
        scores, *q_values, transpose=False, group_size=group_size, bits=bits
    )

    if n_repeats > 1:
        out = mx.reshape(out, (B, n_q_heads, L, D))

    return out


def _turboquant_attention_applies(cache) -> bool:
    """Whether the fused TurboQuant kernels can serve this cache.

    They read the quantized state directly, so they need one contiguous
    sequence starting at index 0. That always holds for the single-sequence
    cache; the batch cache qualifies only while it carries a single row with
    no left padding (padded rows would be attended to as real tokens).
    """
    tq = sys.modules.get("mlx_vlm.turboquant")
    if tq is None or not isinstance(cache, tq.BatchTurboQuantKVCache):
        return True
    if not cache.is_single_row():
        return False
    return cache.fused_attention_eligible


def scaled_dot_product_attention(
    queries,
    keys,
    values,
    cache,
    scale: float,
    mask: Optional[mx.array],
    sinks: Optional[mx.array] = None,
) -> mx.array:
    tq = sys.modules.get("mlx_vlm.turboquant")
    if tq is not None and isinstance(
        cache, (tq.TurboQuantKVCache, tq.BatchTurboQuantKVCache)
    ):
        # The fused kernels have no sink term, and the batch cache only shares
        # them when it holds a single unpadded row. Anything else dequantizes,
        # which is also the only path that can apply sinks.
        if sinks is None and _turboquant_attention_applies(cache):
            if queries.shape[-2] == 1:
                return cache.decode_attention(
                    queries,
                    keys_state=keys,
                    values_state=values,
                    scale=scale,
                    mask=mask,
                )
            result = cache.prefill_attention(
                queries,
                keys_state=keys,
                values_state=values,
                scale=scale,
                mask=mask,
            )
            if result is not None:
                return result
        dequantized_keys, dequantized_values = cache.dequantize(keys, values)
        return mx.fast.scaled_dot_product_attention(
            queries,
            dequantized_keys.astype(queries.dtype),
            dequantized_values.astype(queries.dtype),
            scale=scale,
            mask=mask,
            sinks=sinks,
        )

    if hasattr(cache, "bits"):
        if sinks is not None:
            raise ValueError("Quantized SDPA does not support attention sinks.")
        return quantized_scaled_dot_product_attention(
            queries,
            keys,
            values,
            scale=scale,
            mask=mask,
            group_size=cache.group_size,
            bits=cache.bits,
        )

    return mx.fast.scaled_dot_product_attention(
        queries,
        keys,
        values,
        scale=scale,
        mask=mask,
        sinks=sinks,
    )
