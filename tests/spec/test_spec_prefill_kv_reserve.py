#!/usr/bin/env python3
"""Owned-engine prefill sizes the KV for the whole prompt after the first
chunk. A KVCache otherwise regrows by concatenation on every chunk: each
chunk copies the whole KV and frees the old buffer, and at depth the freed
buffers outrun the driver's reclaim (3.3 to 3.9 GiB of footprint over MLX's
own count at 80K tokens on Qwen3.5-4B, flat at 0.8 with the KV sized once).
The cache contents must not change."""

from __future__ import annotations

import types

import mlx.core as mx
import pytest

from gmlx.cache.kv_append import grid_capacity
from gmlx.cache.kv_cache import KVCache
from gmlx.spec.speculative import stream_speculative

STEP = 4
CHUNK = 4


class _KVWriterLM:
    """Writes each position's index into every KV row; records each cache's
    capacity when a chunk arrives."""

    def __init__(self):
        self.capacity = []

    def rollback_speculative_cache(self, *a, **k):
        pass

    def __call__(self, ids, cache=None, return_hidden=False,
                 return_shared_kv=False, **kw):
        T = int(ids.shape[1])
        self.capacity.append([None if c.keys is None else int(c.keys.shape[2])
                              for c in cache])
        for c in cache:
            pos = mx.arange(c.offset, c.offset + T, dtype=mx.float32)
            k = mx.broadcast_to(pos[None, None, :, None], (1, 2, T, 8))
            c.update_and_fetch(k, k + 0.5)
        return types.SimpleNamespace(
            logits=mx.zeros((1, T, 8)),
            hidden_states=[mx.zeros((1, T, 4))],
            shared_kv_states={},
        )


def _cache(prefill=0):
    c = KVCache()
    c.step = STEP
    if prefill:
        pos = mx.arange(prefill, dtype=mx.float32)
        k = mx.broadcast_to(pos[None, None, :, None], (1, 2, prefill, 8))
        c.update_and_fetch(k, k + 0.5)
    return c


@pytest.mark.parametrize("prefill", [0, 3])
def test_prefill_sizes_the_kv_once(prefill):
    n = 19
    lm = _KVWriterLM()
    cache = [_cache(prefill), _cache(prefill)]
    drafter = types.SimpleNamespace(config=types.SimpleNamespace(
        block_size=2))
    gen = stream_speculative(types.SimpleNamespace(language_model=lm),
                             drafter, mx.zeros((1, n), dtype=mx.int32),
                             prompt_cache=cache, max_tokens=1,
                             prefill_chunk=CHUNK)
    list(gen)
    end = prefill + n
    want = -(-end // STEP) * STEP
    # From the second chunk on, every chunk finds the whole prompt's room.
    assert [cap for caps in lm.capacity[1:] for cap in caps] == (
        [want] * (2 * (len(lm.capacity) - 1)))
    for c in cache:
        assert c.offset == end
        k, v = c.state
        pos = mx.arange(end, dtype=mx.float32)
        assert mx.array_equal(k[0, 0, :, 0], pos)
        assert mx.array_equal(v[0, 1, :, 7], pos + 0.5)


class _SideArrayKV(KVCache):
    """A subclass with its own growth for side arrays: left as it was."""


def test_prefill_leaves_kv_subclasses_alone():
    lm = _KVWriterLM()
    c = _SideArrayKV()
    c.step = STEP
    drafter = types.SimpleNamespace(config=types.SimpleNamespace(
        block_size=2))
    list(stream_speculative(types.SimpleNamespace(language_model=lm),
                            drafter, mx.zeros((1, 12), dtype=mx.int32),
                            prompt_cache=[c], max_tokens=1,
                            prefill_chunk=CHUNK))
    assert [caps[0] for caps in lm.capacity] == [None, 4, 8]


def test_reserve_allocates_one_full_buffer():
    c = KVCache()
    k = mx.ones((1, 4, 256, 64), dtype=mx.bfloat16)
    c.update_and_fetch(k, k)
    mx.eval(c.keys, c.values)
    del k
    # The reserve rounds up to the capacity grid.
    rows = grid_capacity(4096, KVCache.step)
    full = 1 * 4 * rows * 64 * 2
    mx.reset_peak_memory()
    start = mx.get_active_memory()
    mx.eval(*c.prefix_cache_reserve(4096))
    assert c.keys.shape[2] == rows and c.offset == 256
    assert mx.array_equal(c.state[0], mx.ones((1, 4, 256, 64), mx.bfloat16))
    # Two new full buffers (keys and values), no pad buffers beside them.
    assert mx.get_peak_memory() - start <= 2 * full + full // 4
