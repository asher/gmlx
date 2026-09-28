"""cache.compat origin rebind: an mlx-lm ArraysCache swapped to the mlx-vlm
class keeps its metadata. The mlx-vlm 0.7 ArraysCache reads left_padding and
lengths through properties over fields its __new__ seeds, which a swapped
mlx-lm instance never had. No model."""

import mlx.core as mx
import mlx_lm.models.cache as lm_cache

from gmlx.cache.compat import rebind_to_runtime_origin, runtime_cache_module


def _populated(lp=(2, 0), lengths=(5, 3)):
    c = lm_cache.ArraysCache(size=2, left_padding=list(lp))
    c.lengths = mx.array(lengths)
    c[0] = mx.zeros((2, 3, 8))
    c[1] = mx.ones((2, 2, 4, 4))
    return c


def test_populated_arrays_cache_keeps_its_metadata_after_the_swap():
    vlm = runtime_cache_module()
    c, twin = _populated(), _populated()
    rebind_to_runtime_origin([c])

    assert type(c) is vlm.ArraysCache
    assert c.left_padding.tolist() == [2, 0]
    assert c.lengths.tolist() == [5, 3]
    assert c.batch_size == 2
    assert mx.array_equal(c[1], twin[1])
    assert not c.is_speculating
    assert c.make_mask(3).tolist() == twin.make_mask(3).tolist()

    c.advance(1)
    twin.advance(1)
    assert c.left_padding.tolist() == twin.left_padding.tolist() == [1, -1]
    assert c.lengths.tolist() == twin.lengths.tolist() == [4, 2]
    assert c.make_mask(3).tolist() == twin.make_mask(3).tolist()


def test_empty_arrays_cache_swaps_without_metadata():
    vlm = runtime_cache_module()
    c = lm_cache.ArraysCache(size=2)
    rebind_to_runtime_origin([c])

    assert type(c) is vlm.ArraysCache
    assert c.left_padding is None and c.lengths is None
    assert c.make_mask(3) is None
    assert c.batch_size == 1
    gen = c.start_speculation(2)
    c.abort_speculation(gen)
    assert c.left_padding is None


def test_arrays_cache_inside_a_cache_list_is_swapped_the_same_way():
    vlm = runtime_cache_module()
    inner = _populated()
    stack = [lm_cache.CacheList(inner, lm_cache.KVCache())]
    rebind_to_runtime_origin(stack)

    assert type(stack[0]) is vlm.CacheList
    assert type(stack[0].caches[0]) is vlm.ArraysCache
    assert stack[0].caches[0].left_padding.tolist() == [2, 0]
