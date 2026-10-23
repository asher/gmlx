"""The chunked KV cache (llama4's chunked attention layers) through the
APC batch merge and the speculative prefix cache. Its buffer starts at
the trimmed front, start_position, and both paths must carry that."""

from __future__ import annotations

import mlx.core as mx
import numpy as np

from gmlx.cache import kv_cache as owned
from gmlx.cache.prefix_cache import _restore_entry, _snapshot_entry


def _cols(k, row=0):
    return [int(x) for x in np.asarray(k[row, 0, :, 0])]


def _kv(cols):
    k = mx.array(cols, dtype=mx.float32)[None, None, :, None]
    k = mx.broadcast_to(k, (1, 1, len(cols), 2))
    return k, k


def _chunked(n, chunk=8):
    c = owned.ChunkedKVCache(chunk)
    c.update_and_fetch(*_kv(list(range(1, n + 1))))
    return c


def test_batch_merge_keeps_each_rows_keys():
    rows = [_chunked(3), _chunked(5)]
    merged = rows[0].prefix_cache_merge(rows, [3, 5])
    assert isinstance(merged, owned.BatchKVCache)
    assert merged.left_padding.tolist() == [2, 0]
    assert _cols(merged.keys, 0) == [0, 0, 1, 2, 3]
    assert _cols(merged.keys, 1) == [1, 2, 3, 4, 5]


def test_batch_merge_refuses_a_trimmed_front():
    trimmed = _chunked(12, chunk=8)
    trimmed.maybe_trim_front()
    assert trimmed.start_position == 4
    assert trimmed.prefix_cache_merge([_chunked(3), trimmed], [3, 12]) is None


def test_spec_prefix_restore_keeps_the_trimmed_front():
    live = _chunked(12, chunk=8)
    live.maybe_trim_front()
    snap = _snapshot_entry(live)
    fresh = owned.ChunkedKVCache(8)
    _restore_entry(fresh, snap)
    assert (fresh.start_position, fresh.offset) == (4, 12)
    k, _ = fresh.update_and_fetch(*_kv([13]))
    assert _cols(k) == list(range(5, 14))


def test_state_round_trips_in_either_setter_order():
    live = _chunked(12, chunk=8)
    live.maybe_trim_front()
    a = owned.ChunkedKVCache.from_state(live.state, live.meta_state)
    b = owned.ChunkedKVCache(8)
    b.meta_state = live.meta_state
    b.state = live.state
    for c in (a, b):
        assert (c.start_position, c.offset) == (4, 12)
        assert _cols(c.state[0]) == list(range(5, 13))
