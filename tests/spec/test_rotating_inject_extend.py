"""Regression for the gemma SWA spec-injection crash (2026-07-23 bench:
gemma-4-31b mtp@3 lost nearly every c>1 cell on "'RotatingKVCache' object
has no attribute 'rotated'", x101). Admission prefill hands _drain_injections
single-sequence RotatingKVCache (or BufferedRotatingKVCache) layers; the
live batch holds BatchRotatingKVCache, whose extend() needs the batch
class. The old lift detector keyed on a missing `_idx`, which the single
rotating classes HAVE (they lack `rotated`), so they slipped through
unconverted. _lift_injected_cache adds the `rotated` discriminator and
lifts via the class's own merge (which temporal-orders rotated content)."""

import mlx.core as mx
from mlx_vlm.models.cache import (
    BatchKVCache,
    BatchRotatingKVCache,
    BufferedRotatingKVCache,
    KVCache,
    RotatingKVCache,
)

from gmlx.spec.speculative import _lift_injected_cache, _lift_live_cache

H, D = 2, 4


def _fill(cache, n, start=0):
    for t in range(start, start + n):
        k = mx.full((1, H, 1, D), float(t))
        cache.update_and_fetch(k, k)
    return cache


def _batch_of(*singles):
    return BatchRotatingKVCache.merge(list(singles))


def test_rotating_single_is_lifted_and_extends():
    live = _batch_of(_fill(RotatingKVCache(max_size=8), 5))
    injected = _fill(RotatingKVCache(max_size=8), 5)
    lifted = _lift_injected_cache(live, injected)
    assert lifted is not injected
    assert isinstance(lifted, BatchRotatingKVCache)
    live.extend(lifted)
    assert live.keys.shape[0] == 2
    assert live.offset.tolist() == [5, 5]


def test_mid_rotation_injected_stream_temporal_order():
    live = _batch_of(_fill(RotatingKVCache(max_size=8), 3))
    injected = _fill(RotatingKVCache(max_size=8), 13)  # rotated: 13 > 8
    lifted = _lift_injected_cache(live, injected)
    live.extend(lifted)
    assert live.keys.shape[0] == 2
    assert int(live.offset[1]) == 13
    # the injected row's retained window must be the LAST tokens in
    # temporal order (values 5..12 for max_size 8), newest at the end
    row = live.keys[1, 0, :, 0].tolist()
    kept = [v for v in row if v != 0.0] or row
    assert kept[-1] == 12.0
    assert kept == sorted(kept)


def test_buffered_subclass_lifts_through_same_path():
    live = _batch_of(_fill(RotatingKVCache(max_size=8), 4))
    injected = _fill(BufferedRotatingKVCache(max_size=8, buffer_size=4), 4)
    lifted = _lift_injected_cache(live, injected)
    assert isinstance(lifted, BatchRotatingKVCache)
    live.extend(lifted)
    assert live.keys.shape[0] == 2


def test_standard_kv_pair_lifted_as_before():
    # BatchKVCache HAS _idx and plain KVCache lacks it, so the original
    # detector fired here all along -- the (working) qwen injection path.
    # The rotated discriminator must not change that behavior.
    single = _fill(KVCache(), 4)
    live = BatchKVCache.merge([_fill(KVCache(), 4)])
    lifted = _lift_injected_cache(live, single)
    assert isinstance(lifted, BatchKVCache)
    live.extend(lifted)
    assert live.keys.shape[0] == 2


def test_already_batch_class_untouched():
    live = _batch_of(_fill(RotatingKVCache(max_size=8), 4))
    other = _batch_of(_fill(RotatingKVCache(max_size=8), 4))
    assert _lift_injected_cache(live, other) is other


# --- the live side: a batch formed from one row (DeepSeek V4 draft/MTP at
# width > 1, 2026-08-25 soak) keeps single-sequence layer caches, and the
# spec swap-in BufferedRotatingKVCache has no extend at all.


def test_live_cache_list_members_are_lifted_before_extend():
    from mlx_vlm.models.cache import CacheList

    live = CacheList(_fill(KVCache(), 4),
                     _fill(BufferedRotatingKVCache(max_size=8, buffer_size=4), 4))
    assert not hasattr(live.caches[1], "extend")
    out = _lift_live_cache(live)
    assert out is live                                   # the list object survives
    assert isinstance(live.caches[0], BatchKVCache)
    assert isinstance(live.caches[1], BatchRotatingKVCache)
    injected = CacheList(_fill(KVCache(), 3),
                         _fill(BufferedRotatingKVCache(max_size=8, buffer_size=4), 3))
    live.extend(_lift_injected_cache(live, injected))    # members lift on their own
    assert live.caches[0].keys.shape[0] == 2
    assert live.caches[1].keys.shape[0] == 2
    assert [int(v) for v in live.caches[1].offset.tolist()] == [4, 3]


def test_live_buffered_rotating_lifts_to_batch_rotating():
    live = _fill(BufferedRotatingKVCache(max_size=8, buffer_size=4), 5)
    out = _lift_live_cache(live)
    assert isinstance(out, BatchRotatingKVCache)
    assert out.keys.shape[0] == 1 and int(out.offset[0]) == 5


def test_live_batch_classes_untouched():
    live = _batch_of(_fill(RotatingKVCache(max_size=8), 4))
    assert _lift_live_cache(live) is live
    kv = BatchKVCache.merge([_fill(KVCache(), 4)])
    assert _lift_live_cache(kv) is kv


def test_injected_single_without_batch_api_is_lifted_by_merge():
    """DeepSeek V4's PoolingCache has neither _idx nor rotated; the batch
    class's extend() then read len() of the single row's scalar state."""
    class Batch:
        n = 1

        def extend(self, other):
            self.n += other.n

        def filter(self, idx):
            pass

    class Single:
        n = 1

        @classmethod
        def merge(cls, caches):
            out = Batch()
            out.n = len(caches)
            return out

    live = Batch()
    other = _lift_injected_cache(live, Single())
    assert isinstance(other, Batch)
    live.extend(other)
    assert live.n == 2
    already = Batch()
    assert _lift_injected_cache(live, already) is already
