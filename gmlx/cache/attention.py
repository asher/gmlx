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
``sys.modules`` at call time; comments are ASCII. gmlx adds the decode
mask starts registry and two routes that
``quantized_scaled_dot_product_attention`` takes ahead of its upstream
body.
"""

import collections
import functools
import logging
import sys
from typing import Optional

import mlx.core as mx
from mlx.utils import tree_map

from gmlx.envflags import env_bool, env_int

from .kv_cache import create_causal_mask

_log = logging.getLogger(__name__)


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


# id(mask) -> (mask, starts, ends). A batch cache records the row starts
# of each decode mask it builds, from its left padding, so the decode
# route never reads a mask's contents: that read syncs the GPU, and under
# paced serving the sync waits behind whole prefill chunks. The entry pins
# the mask, so its id cannot be reused while the entry lives. All layers
# of a step share one mask object.
_STARTS_MEMO: collections.OrderedDict = collections.OrderedDict()
_STARTS_MEMO_CAP = 8


def _register_starts(mask, left_padding, ends=None):
    """Record a mask's row starts, and its row ends when the creator knows
    them. Both are the creator's own state, taken on trust: no sync."""
    ends = None if ends is None else ends.astype(mx.int32)
    _STARTS_MEMO[id(mask)] = (mask, left_padding.astype(mx.int32), ends)
    if len(_STARTS_MEMO) > _STARTS_MEMO_CAP:
        _STARTS_MEMO.popitem(last=False)


def _registered_starts(mask):
    """The starts recorded for this mask object, else None."""
    ent = _STARTS_MEMO.get(id(mask))
    if ent is not None and ent[0] is mask:
        return ent[1]
    return None


def _registered_geometry(mask):
    """(starts, ends) recorded for this mask object, else None. A record
    without ends gives None too."""
    ent = _STARTS_MEMO.get(id(mask))
    if ent is not None and ent[0] is mask and ent[2] is not None:
        return ent[1], ent[2]
    return None


# Query widths from GMLX_KV8_PREFILL_FLASH_MIN_L up take the prefill
# route. The upstream body runs two unfused quantized matmuls and a
# separate softmax, 3.4 to 4.2 times slower per call than fused SDPA at
# prefill chunk widths. The route dequantizes the K/V wire once, a copy
# the chunk amortizes, and runs the fused kernel. GMLX_KV8_PREFILL_FLASH=0
# turns it off.
_PREFILL_MIN_L = 8

# The decode route: mlx_kquant's sdpa_decode_gqa takes 8-bit K/V in
# 64-wide groups with the dequantize fused into its K/V walk, about 1.4
# times faster at depth than the upstream body. GMLX_QSDPA_KQ=0 turns it
# off.
_HD_OK = (64, 128, 256, 512)
_route_failed = False


@functools.cache
def _kq_decode_kernel():
    """sdpa_decode_gqa when the mlx_kquant build takes quantized K/V and
    row starts, else None."""
    try:
        import mlx_kquant as kq
    except ImportError:
        return None
    fn = getattr(kq, "sdpa_decode_gqa", None)
    doc = getattr(fn, "__doc__", None) or ""
    if "k_scales" not in doc or "starts" not in doc:
        return None
    return fn


def _prefill_route(queries, q_keys, q_values, scale, mask, group_size, bits):
    # Cache fetches are lazy strided slices, and mlx before 0.32.2
    # dequantized strided biases wrong (mlx #4381). A contiguous slice
    # costs nothing here.
    keys = mx.dequantize(*(mx.contiguous(t) for t in q_keys),
                         group_size=group_size, bits=bits)
    values = mx.dequantize(*(mx.contiguous(t) for t in q_values),
                           group_size=group_size, bits=bits)
    return mx.fast.scaled_dot_product_attention(
        queries, keys, values, scale=scale, mask=mask)


def _decode_route(queries, q_keys, q_values, scale, mask, group_size, bits):
    """The decode kernel's output, or None where it does not apply. An
    array mask goes to the kernel only as the starts its creator
    recorded; any other array mask takes the upstream body."""
    global _route_failed
    if not (
        group_size == 64
        and bits == 8
        and queries.ndim == 4
        and queries.shape[2] == 1
        and queries.shape[3] in _HD_OK
        and queries.dtype in (mx.float16, mx.bfloat16)
        and queries.shape[1] % q_keys[0].shape[1] == 0
        and 1 <= queries.shape[1] // q_keys[0].shape[1] <= 16
        and env_bool("GMLX_QSDPA_KQ", True)
    ):
        return None
    # The kernel is Metal only; on the CPU device the upstream body runs.
    if mx.default_device().type != mx.DeviceType.gpu:
        return None
    kernel = _kq_decode_kernel()
    if kernel is None:
        return None
    starts = None
    if isinstance(mask, mx.array):
        if not (
            mask.dtype == mx.bool_
            and mask.ndim == 4
            and mask.shape[0] == queries.shape[0]
            and mask.shape[1] == 1
            and mask.shape[2] == 1
            and mask.shape[3] == q_keys[0].shape[2]
        ):
            return None
        starts = _registered_starts(mask)
        if starts is None:
            return None
    kw, ks, kb = q_keys
    vw, vs, vb = q_values
    try:
        return kernel(queries, kw, vw, float(scale), starts=starts,
                      k_scales=ks, k_biases=kb, v_scales=vs, v_biases=vb)
    except Exception as exc:  # a layout the kernel refuses
        if not _route_failed:
            _route_failed = True
            _log.warning("quantized sdpa decode route failed (%s: %s); "
                         "such calls take the upstream body",
                         type(exc).__name__, exc)
        return None


def quantized_scaled_dot_product_attention(
    queries: mx.array,
    q_keys: tuple[mx.array, mx.array, mx.array],
    q_values: tuple[mx.array, mx.array, mx.array],
    scale: float,
    mask: Optional[mx.array],
    group_size: int = 64,
    bits: int = 8,
) -> mx.array:
    if queries.shape[-2] >= env_int(
        "GMLX_KV8_PREFILL_FLASH_MIN_L", _PREFILL_MIN_L
    ) and env_bool("GMLX_KV8_PREFILL_FLASH", True):
        return _prefill_route(
            queries, q_keys, q_values, scale, mask, group_size, bits
        )
    out = _decode_route(queries, q_keys, q_values, scale, mask, group_size, bits)
    if out is not None:
        return out

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
