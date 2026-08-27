"""Optional no-trim mode for RotatingKVCache during chunked prefill.

Stock mlx-lm's RotatingKVCache._update_concat trims the key buffer to
max_size + S - 1 at each chunk boundary. Trimming is mathematically lossless
for sliding-window attention (a query can never attend past its window), but
it changes the total K tensor size seen by SDPA versus an unchunked run: the
same 1024 valid keys appear in a 1535-entry tensor (trimmed) vs a 3000-entry
tensor (unchunked). MLX's flash attention tiles differently over those two
tensor sizes, so floating-point accumulation order differs - a tiny per-layer
divergence that can flip near-tie output tokens in bitwise chunked-vs-unchunked
comparisons.

Set GMLX_SLIDING_NOTRIM=1 to install the historical no-trim patch, which
defers all trimming to the first decode step so the chunked path bit-matches
the unchunked lifecycle. The cost is O(prompt_length) prefill memory per
sliding layer: on window-heavy models this is tens of GB at 32k context and
grows past physical memory at 200k, so windowed trim (stock behavior) is the
default. Cross-engine token-parity certification uses tie-robust prompts and
holds under either mode.
"""
from __future__ import annotations

import os

import mlx.core as mx

_installed = False


def _effective_cache_size(cache) -> int:
    """Pre-update cache occupancy: the K tensor size _temporal_order will produce."""
    if cache.keys is None:
        return 0
    if cache._idx == cache.keys.shape[2]:
        return cache.keys.shape[2]
    if cache._idx < cache.offset:
        return cache.keys.shape[2]
    return cache._idx


def install_rotating_cache_fix():
    """Patch RotatingKVCache to skip trim during prefill (opt-in). Idempotent.

    No-op unless GMLX_SLIDING_NOTRIM=1: stock windowed trim keeps sliding
    KV at O(window) through chunked prefill, which deep contexts require.
    """
    global _installed
    if os.environ.get("GMLX_SLIDING_NOTRIM", "0") != "1":
        return
    if _installed:
        return
    from mlx_lm.models.base import create_causal_mask

    from .cache_compat import cache_types

    for RotatingKVCache in cache_types("RotatingKVCache"):
        _install_notrim(RotatingKVCache, create_causal_mask)
    _installed = True


def _install_notrim(RotatingKVCache, create_causal_mask):
    _orig_make_mask = RotatingKVCache.make_mask

    def _notrim_update_concat(self, keys, values):
        if self.keys is None:
            self.keys = keys
            self.values = values
        else:
            self.keys = self._temporal_order(self.keys)
            self.values = self._temporal_order(self.values)
            self.keys = mx.concatenate([self.keys, keys], axis=2)
            self.values = mx.concatenate([self.values, values], axis=2)
        self.offset += keys.shape[2]
        self._idx = self.keys.shape[2]
        return self.keys, self.values

    def _notrim_make_mask(self, N, window_size=None, return_array=False):
        if N > 1:
            window_size = window_size or self.max_size
            offset = _effective_cache_size(self)
            if offset + N > window_size or return_array:
                return create_causal_mask(N, offset, window_size=window_size)
            return "causal"
        return _orig_make_mask(
            self, N, window_size=window_size, return_array=return_array
        )

    RotatingKVCache._update_concat = _notrim_update_concat
    RotatingKVCache.make_mask = _notrim_make_mask
