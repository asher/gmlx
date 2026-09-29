"""The cache members gmlx relies on, held on the owned classes.

gmlx binds its own copy of the mlx-vlm cache classes under the stock
names, so these members are gmlx text and no seam row pins them. Each
test holds one member to what its gmlx caller needs: the snapshot, rollback
and retire paths, the governor's byte count, chat MTP cache construction,
and the three private RotatingKVCache methods the systemone prompt cache
and the spec helpers reach.
"""

from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import pytest

import gmlx.cache.kv_cache as owned

mlx_vlm_cache = pytest.importorskip("mlx_vlm.models.cache")


def _kv(rows=1, n=5, d=4, start=0):
    k = (mx.arange(start, start + n, dtype=mx.float32)[None, None, :, None]
         + mx.zeros((rows, 2, n, d)))
    return k, -k


def _cols(keys, row=0):
    return mx.array(keys[row, 0, :, 0]).astype(mx.int32).tolist()


def test_kvcache_state_round_trips_the_live_prefix():
    assert mlx_vlm_cache.KVCache is owned.KVCache
    c = owned.KVCache()
    c.update_and_fetch(*_kv(n=5))
    keys, values = c.state
    assert keys.shape[2] == c.offset == 5
    back = owned.KVCache.from_state(c.state, c.meta_state)
    assert type(back) is owned.KVCache and back.offset == 5
    assert mx.array_equal(back.state[0], keys).item()


def test_rotating_kvcache_exposes_its_window_fields_and_trims():
    assert mlx_vlm_cache.RotatingKVCache is owned.RotatingKVCache
    c = owned.RotatingKVCache(max_size=8)
    c.update_and_fetch(*_kv(n=6))
    assert (c.max_size, c.keep, c.offset, c._idx) == (8, 0, 6, 6)
    assert c.is_trimmable() and c.trim(2) == 2
    assert c.offset == 4


def test_cache_list_holds_its_children_and_round_trips():
    assert mlx_vlm_cache.CacheList is owned.CacheList
    c = owned.CacheList(owned.KVCache(), owned.RotatingKVCache(max_size=8))
    for child in c.caches:
        child.update_and_fetch(*_kv(n=3))
    back = owned.CacheList.from_state(c.state, c.meta_state)
    assert [type(x) for x in back.caches] == [owned.KVCache,
                                              owned.RotatingKVCache]
    assert back.caches[1].offset == 3


def test_arrays_cache_keeps_the_advance_counters_qwen35_settles():
    assert mlx_vlm_cache.ArraysCache is owned.ArraysCache
    c = owned.ArraysCache(size=2, left_padding=[2, 0])
    c.lengths = mx.array([5, 3])
    for name in ("_left_padding", "_left_padding_advance", "_lengths",
                 "_lengths_advance"):
        assert name in vars(c), name
    c.advance(1)
    assert c._left_padding_advance == 1 and c._lengths_advance == 1
    assert c.left_padding.tolist() == [1, -1]
    assert c.lengths.tolist() == [4, 2]
    assert callable(getattr(c, "_restore_speculative_metadata", None))


def test_batch_kvcache_merges_rows_and_keeps_left_padding():
    assert mlx_vlm_cache.BatchKVCache is owned.BatchKVCache
    a, b = owned.KVCache(), owned.KVCache()
    a.update_and_fetch(*_kv(n=5))
    b.update_and_fetch(*_kv(n=3, start=10))
    batch = owned.BatchKVCache.merge([a, b])
    assert batch.left_padding.tolist() == [0, 2]
    assert batch.offset.tolist() == [5, 3]


def test_batch_kvcache_filter_drops_rows_in_place():
    a, b = owned.KVCache(), owned.KVCache()
    a.update_and_fetch(*_kv(n=5))
    b.update_and_fetch(*_kv(n=3, start=10))
    batch = owned.BatchKVCache.merge([a, b])
    batch.filter(mx.array([1]))
    assert batch.keys.shape[0] == 1
    assert batch.offset.tolist() == [3]
    assert _cols(batch.keys)[-3:] == [10, 11, 12]


def test_batch_kvcache_extract_returns_one_contiguous_row():
    a, b = owned.KVCache(), owned.KVCache()
    a.update_and_fetch(*_kv(n=5))
    b.update_and_fetch(*_kv(n=3, start=10))
    batch = owned.BatchKVCache.merge([a, b])
    row = batch.extract(1)
    assert type(row) is owned.KVCache and row.offset == 3
    assert _cols(row.state[0]) == [10, 11, 12]


def test_arrays_cache_extract_returns_one_row():
    c = owned.ArraysCache(size=2)
    c[0] = mx.arange(6, dtype=mx.float32).reshape(2, 3)
    c[1] = mx.arange(8, dtype=mx.float32).reshape(2, 2, 2)
    row = c.extract(1)
    assert isinstance(row, owned.ArraysCache)
    assert row[0].tolist() == [[3.0, 4.0, 5.0]]
    assert row[1].shape == (1, 2, 2)


def test_base_cache_nbytes_counts_the_state():
    c = owned.KVCache()
    assert c.nbytes == 0
    c.update_and_fetch(*_kv(n=5))
    assert c.nbytes == c.keys.nbytes + c.values.nbytes
    assert "nbytes" in vars(owned._BaseCache)


def test_batch_rotating_kvcache_is_rotating_for_the_safe_quantizer():
    assert mlx_vlm_cache.BatchRotatingKVCache is owned.BatchRotatingKVCache
    c = owned.BatchRotatingKVCache(8, [0, 1])
    assert not isinstance(c, owned.RotatingKVCache)
    assert c.max_size == 8 and hasattr(c, "_idx") and hasattr(c, "rotated")
    assert callable(getattr(c, "to_quantized", None))


def test_buffered_rotating_kvcache_takes_a_rotating_cache_with_slack():
    assert (mlx_vlm_cache.BufferedRotatingKVCache
            is owned.BufferedRotatingKVCache)
    src = owned.RotatingKVCache(max_size=8)
    src.update_and_fetch(*_kv(n=12))
    buf = owned.BufferedRotatingKVCache.from_cache(src, buffer_size=32)
    assert isinstance(buf, owned.RotatingKVCache)
    assert buf.buffer_size == 32 and buf.max_size == 8
    # The window as a linear buffer [0, _idx), past the rollback slack.
    assert (buf.offset, buf.start_position, buf._idx) == (12, 4, 8)
    assert _cols(buf.keys[..., :buf._idx, :]) == list(range(4, 12))
    assert buf.keys.shape[2] >= 8 + 32


def test_make_prompt_cache_builds_the_stack_chat_mtp_uses():
    assert mlx_vlm_cache.make_prompt_cache is owned.make_prompt_cache
    plain = SimpleNamespace(layers=[None, None])
    assert [type(c) for c in owned.make_prompt_cache(plain)] == [
        owned.KVCache, owned.KVCache]
    windowed = owned.make_prompt_cache(plain, max_kv_size=16)
    assert [type(c) for c in windowed] == [owned.RotatingKVCache] * 2
    marker = [owned.CacheList(owned.KVCache())]
    own = SimpleNamespace(layers=[None], make_cache=lambda: marker)
    assert owned.make_prompt_cache(own) is marker


def test_rotating_update_concat_appends_a_block_in_temporal_order():
    c = owned.RotatingKVCache(max_size=8)
    c.update_and_fetch(*_kv(n=6))
    k, _ = c._update_concat(*_kv(n=4, start=6))
    assert _cols(k) == list(range(10))
    assert c.offset == 10


def test_rotating_update_in_place_appends_one_token_into_the_ring():
    c = owned.RotatingKVCache(max_size=4)
    for t in range(6):
        k, _ = c._update_in_place(*_kv(n=1, start=t))
    assert c.offset == 6 and k.shape[2] == 4
    assert sorted(_cols(k)) == [2, 3, 4, 5]


def test_rotating_temporal_order_puts_the_oldest_token_first():
    c = owned.RotatingKVCache(max_size=4)
    for t in range(6):
        c.update_and_fetch(*_kv(n=1, start=t))
    assert _cols(c.keys) != [2, 3, 4, 5]
    assert _cols(c._temporal_order(c.keys)) == [2, 3, 4, 5]
