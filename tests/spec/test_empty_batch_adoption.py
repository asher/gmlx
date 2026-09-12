"""The all-rows-finished injection adoption empties the batch caches and
extends them with the injected rows. An emptied cache must return to its
empty state (no buffers, watermark 0): with the finished batch's
watermark kept, extend() left-padded the adopted row by that watermark,
and the qwen3_5 language model resolves a one-row batch's decode
position from the watermark rather than the row's own offset, so the
adopted row decoded at the wrong position (Qwen3.8-27B with DFlash2:
token 2 of the injected row)."""

import mlx.core as mx
import pytest
from mlx_vlm.models.cache import (
    BatchKVCache,
    BatchRotatingKVCache,
    KVCache,
    RotatingKVCache,
)

from gmlx.spec.speculative import _filter_batch_rows_empty, _lift_injected_cache

H, D = 2, 4


def _fill(cache, n):
    for t in range(n):
        k = mx.full((1, H, 1, D), float(t + 1))
        cache.update_and_fetch(k, k)
    return cache


def _adopt(live, injected):
    live.extend(_lift_injected_cache(live, injected))
    return live


def test_emptied_batch_returns_to_the_empty_state():
    live = BatchKVCache.merge([_fill(KVCache(), 900), _fill(KVCache(), 700)])
    assert live._idx == 900
    _filter_batch_rows_empty([live])
    assert live._idx == 0
    assert live.keys is None and live.values is None
    assert live.offset.shape == (0,) and live.left_padding.shape == (0,)


def test_emptied_batch_adopts_a_row_without_padding():
    live = BatchKVCache.merge([_fill(KVCache(), 900), _fill(KVCache(), 700)])
    _filter_batch_rows_empty([live])
    _adopt(live, _fill(KVCache(), 300))
    assert live.left_padding.tolist() == [0]
    assert live.offset.tolist() == [300]
    # The language model reads a one-row batch's position off _idx.
    assert live._idx == 300
    assert live.keys[0, 0, :, 0].tolist() == [float(t + 1) for t in range(300)]


def test_emptied_batch_adopts_two_rows_right_aligned():
    live = BatchKVCache.merge([_fill(KVCache(), 900)])
    _filter_batch_rows_empty([live])
    _adopt(live, _fill(KVCache(), 300))
    _adopt(live, _fill(KVCache(), 200))
    assert live._idx == 300
    assert live.left_padding.tolist() == [0, 100]
    assert live.offset.tolist() == [300, 200]


def test_emptied_rotating_batch_adopts_a_row_without_padding():
    live = BatchRotatingKVCache.merge([_fill(RotatingKVCache(max_size=8), 5)])
    _fill(live, 10)  # past max_size: the ring has rotated
    assert live.rotated
    _filter_batch_rows_empty([live])
    assert live._idx == 0 and live.keys is None and not live.rotated
    _adopt(live, _fill(RotatingKVCache(max_size=8), 6))
    assert live.left_padding.tolist() == [0]
    assert live.offset.tolist() == [6]


@pytest.mark.parametrize("n", [1, 3])
def test_empty_state_survives_a_cache_list(n):
    class CacheList:
        def __init__(self, caches):
            self.caches = tuple(caches)

    members = [BatchKVCache.merge([_fill(KVCache(), 50)]) for _ in range(n)]
    _filter_batch_rows_empty([CacheList(members)])
    assert all(m._idx == 0 and m.keys is None for m in members)
