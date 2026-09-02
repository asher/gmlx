"""Quantized-KV first-allocation width fix (upstream bug).

``QuantizedKVCache`` and ``BatchQuantizedKVCache`` size their first code
buffer as ``head_dim // (32 // bits)`` uint32 words, but ``mx.quantize``
bit-packs ``head_dim * bits // 32``. The two agree only when bits divides
32, so widths 3 and 6 allocate too wide and the very first append raises
``broadcast_shapes ... cannot be broadcast``.

It only fires when the cache is quantized while empty. The CLI converts
mid-stream, off a populated cache whose buffers came from the conversion,
and growth reuses that width -- which is why ``--kv-bits 6`` runs there.
Serve builds ``BatchQuantizedKVCache`` empty at construction, so on the
server every head_dim whose two formulas disagree fails the first
request at ``kv_bits`` 3 or 6.

The fix seeds the buffers at the packed width on the first call and lets
the stock body run from there: its realloc branch is then skipped
(the seeded rows already cover the append) and later growth copies the
seeded width forward.

Install is idempotent; GMLX_QKV_PACK_FIX=0 disables it.
"""

from __future__ import annotations

import os

_INSTALLED = "_gmlx_pack_fix"
_WIDTHS: dict = {}


def _packed_words(dim: int, group_size: int, bits: int, dtype) -> int:
    """Words mx.quantize emits for one head vector. Probed once per shape
    rather than assumed: the packing is upstream's to change."""
    key = (dim, group_size, bits, dtype)
    got = _WIDTHS.get(key)
    if got is None:
        import mlx.core as mx

        probe = mx.zeros((1, 1, 1, dim), dtype)
        got = mx.quantize(probe, group_size=group_size, bits=bits)[0].shape[-1]
        _WIDTHS[key] = got
    return got


def _seed(cache, keys, values) -> None:
    import mlx.core as mx

    B, n_kv_heads, num_steps, k_dim = keys.shape
    v_dim = values.shape[-1]
    step = cache.step
    rows = (step + num_steps - 1) // step * step
    shape = (B, n_kv_heads, rows)

    def alloc(dim):
        words = _packed_words(dim, cache.group_size, cache.bits, keys.dtype)
        return (
            mx.zeros((*shape, words), dtype=mx.uint32),
            mx.zeros((*shape, dim // cache.group_size), dtype=keys.dtype),
            mx.zeros((*shape, dim // cache.group_size), dtype=keys.dtype),
        )

    cache.keys, cache.values = alloc(k_dim), alloc(v_dim)


def _wrap(cls):
    stock = cls.update_and_fetch
    if getattr(stock, _INSTALLED, False):
        return False

    def update_and_fetch(self, keys, values):
        if self.keys is None:
            _seed(self, keys, values)
        return stock(self, keys, values)

    setattr(update_and_fetch, _INSTALLED, True)
    cls.update_and_fetch = update_and_fetch
    return True


def install_quantized_cache_pack_fix() -> bool:
    """Patch both quantized KV cache classes. Returns True when newly
    installed."""
    if os.environ.get("GMLX_QKV_PACK_FIX") == "0":
        return False
    from mlx_vlm.models.cache import BatchQuantizedKVCache, QuantizedKVCache

    done = _wrap(QuantizedKVCache)
    done |= _wrap(BatchQuantizedKVCache)
    try:
        from mlx_lm.models.cache import QuantizedKVCache as LmQuantizedKVCache

        if LmQuantizedKVCache is not QuantizedKVCache:
            done |= _wrap(LmQuantizedKVCache)
    except ImportError:
        pass
    return done
