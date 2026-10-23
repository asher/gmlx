"""Sliding-window SDPA route: prefill chunks on kq's window kernel.

A sliding-window layer prefills a chunk against a band mask. Stock
mx.fast attention walks every key block of the chunk for each query block
and masks most of them, so a 2048-row chunk over a 513-key window does
about five times the useful work. kq.sdpa_prefill_window reads only the
key blocks inside each query block's window.

The route takes a chunk of at least 64 rows over a plain RotatingKVCache
with no keep tokens, which hands back its keys in position order with the
queries at the end, on a GPU with tensor-op units. Every other call goes
to the owned scaled_dot_product_attention. GMLX_SWA_KQ=0 turns it off.
"""

from __future__ import annotations

import functools
from typing import Optional

import mlx.core as mx

from gmlx.envflags import env_bool

from .attention import scaled_dot_product_attention
from .kv_cache import RotatingKVCache

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


def _window_route_applies(queries, keys, cache) -> bool:
    return (
        type(cache) is RotatingKVCache
        and cache.keep == 0
        and queries.ndim == 4
        and queries.shape[2] >= _MIN_ROWS
        and queries.shape[3] in _HD_OK
        and queries.dtype in (mx.float16, mx.bfloat16)
        and keys.dtype == queries.dtype
        and keys.shape[2] >= queries.shape[2]
        and env_bool("GMLX_SWA_KQ", True)
        and _on_gpu()
    )


def sliding_window_attention(
    queries,
    keys,
    values,
    cache,
    scale: float,
    mask: Optional[mx.array],
    window: int,
    sinks: Optional[mx.array] = None,
) -> mx.array:
    """Attention for a layer that sees the last ``window`` positions, its
    own included. ``mask`` is the band mask the cache made for the same
    window, used where the route does not apply."""
    if _window_route_applies(queries, keys, cache):
        kernel = _kq_window_kernel()
        if kernel is not None:
            return kernel(queries, keys, values, scale, window, sinks=sinks)
    return scaled_dot_product_attention(
        queries, keys, values, cache=cache, scale=scale, mask=mask, sinks=sinks
    )
