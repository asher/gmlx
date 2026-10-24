"""Sliding-window SDPA route: prefill chunks on kq's window kernel.

A sliding-window layer prefills a chunk against a band mask. Stock
mx.fast attention walks every key block of the chunk for each query block
and masks most of them, so a 2048-row chunk over a 513-key window does
about five times the useful work. kq.sdpa_prefill_window reads only the
key blocks inside each query block's window.

claim() takes a chunk of at least 64 rows over a cache that hands back its
keys in position order with the queries at the end: a RotatingKVCache with
no keep tokens, or a left-padded BatchRotatingKVCache, whose rows start at
their left padding. It needs a GPU with tensor-op units. It returns None
for every other call, which the model then runs through its own
scaled_dot_product_attention global, so the routes patched onto that
global still see it. GMLX_SWA_KQ=0 turns the route off.
"""

from __future__ import annotations

import functools
from typing import Optional

import mlx.core as mx

from gmlx.envflags import env_bool

from .kv_cache import BatchRotatingKVCache, RotatingKVCache

_MIN_ROWS = 64
_HD_OK = (64, 128, 256)


@functools.cache
def _kq_window_kernel():
    """sdpa_prefill_window when the mlx_kquant build has it and the GPU has
    tensor-op units, else None."""
    try:
        import mlx_kquant as kq
    except ImportError:
        return None
    fn = getattr(kq, "sdpa_prefill_window", None)
    if fn is None or not kq.nax_available():
        return None
    return fn


def _on_gpu() -> bool:
    return mx.default_device().type == mx.DeviceType.gpu


def _cache_in_order(cache) -> bool:
    if type(cache) is RotatingKVCache:
        return cache.keep == 0
    # A right-padded batch (APC mixed prefill) holds pad keys between rows'
    # valid keys until finalize() rolls them out.
    return type(cache) is BatchRotatingKVCache and cache._lengths is None


def _window_route_applies(queries, keys, cache) -> bool:
    return (
        _cache_in_order(cache)
        and queries.ndim == 4
        and queries.shape[2] >= _MIN_ROWS
        and queries.shape[3] in _HD_OK
        and queries.dtype in (mx.float16, mx.bfloat16)
        and keys.dtype == queries.dtype
        and keys.shape[2] >= queries.shape[2]
        and env_bool("GMLX_SWA_KQ", True)
        and _on_gpu()
    )


def claim(
    queries,
    keys,
    values,
    cache,
    scale: float,
    window: Optional[int],
    sinks: Optional[mx.array] = None,
) -> Optional[mx.array]:
    """The window kernel's attention for a layer that sees the last
    ``window`` positions, its own included, under the band mask its cache
    makes. None where the route does not apply, or for ``window`` None."""
    if window is None or not _window_route_applies(queries, keys, cache):
        return None
    kernel = _kq_window_kernel()
    if kernel is None:
        return None
    if type(cache) is not BatchRotatingKVCache:
        return kernel(queries, keys, values, scale, window, sinks=sinks)
    # After the update, left_padding is each row's first valid key in the
    # returned buffer, negative once a trim passes the padding. The kernel
    # clamps it.
    starts = cache.left_padding.astype(mx.int32)
    return kernel(queries, keys, values, scale, window, sinks=sinks, starts=starts)
