"""The shape of the owned cache classes that borrowed callers rely on.

Stock code tests caches by truthiness and ``len`` nowhere, deep-copies
them for snapshots, reads ``bits`` to pick the quantized SDPA, and
subclasses them in model modules, so the owned classes keep the stock
layout: no ``__slots__``, ``__len__`` or ``__bool__``, one flat level
under ``_BaseCache``, and ``bits`` only on the quantized classes.

ArraysCache reads ``left_padding`` and ``lengths`` through an advance
offset, as mlx-vlm 0.7.3 does, so after ``advance`` every read is a new
array. The mlx-lm class subtracted in place and kept one array. From the
binding every mlx_lm state-cache architecture reads the owned semantics.
Qwen35ArraysCache settles both arrays at ``advance`` and returns the same
array until the next one.
"""

from __future__ import annotations

import copy

import mlx.core as mx
import pytest

import gmlx.cache.kv_cache as owned
from gmlx.cache import bind

_CLASSES = [getattr(owned, n) for n in bind.VLM_CACHE_NAMES
            if isinstance(getattr(owned, n), type)]
_CACHES = [c for c in _CLASSES if c not in (owned._BaseCache,
                                            owned.CacheMemory)]


def _kv(n=5, rows=1):
    k = mx.arange(n, dtype=mx.float32)[None, None, :, None] + mx.zeros(
        (rows, 2, n, 64))
    return k, -k


def _filled():
    """One filled instance of each owned cache class."""
    out = {}
    for cls, make, rows in (
            (owned.ConcatenateKVCache, owned.ConcatenateKVCache, 1),
            (owned.QuantizedKVCache, lambda: owned.QuantizedKVCache(64, 8), 1),
            (owned.KVCache, owned.KVCache, 1),
            (owned.RotatingKVCache, lambda: owned.RotatingKVCache(8), 1),
            (owned.ChunkedKVCache, lambda: owned.ChunkedKVCache(8), 1),
            (owned.BatchKVCache, lambda: owned.BatchKVCache([0, 1]), 2),
            (owned.BatchRotatingKVCache,
             lambda: owned.BatchRotatingKVCache(8, [0, 1]), 2),
            (owned.BufferedRotatingKVCache,
             lambda: owned.BufferedRotatingKVCache(8), 1),
            (owned.BatchQuantizedKVCache,
             lambda: owned.BatchQuantizedKVCache([0, 1]), 2),
            (owned.SimpleKVCache, owned.SimpleKVCache, 1),
            (owned.StaticPrefixKVCache,
             lambda: owned.StaticPrefixKVCache(16), 1)):
        c = make()
        c.update_and_fetch(*_kv(rows=rows))
        out[cls] = c
    arrays = owned.ArraysCache(2, left_padding=[1, 0])
    arrays[0] = mx.ones((2, 3, 8))
    out[owned.ArraysCache] = arrays
    out[owned.CacheList] = owned.CacheList(out[owned.KVCache],
                                           owned.ArraysCache(1))
    out[owned.PoolingCache] = owned.PoolingCache(4)
    out[owned.BatchPoolingCache] = owned.BatchPoolingCache(4, [0, 1])
    return out


def test_no_slots_len_or_bool():
    for cls in _CLASSES:
        for base in cls.__mro__[:-1]:
            assert "__slots__" not in vars(base), cls
        if cls is owned.CacheMemory:
            continue
        assert not hasattr(cls, "__len__"), cls
        assert not hasattr(cls, "__bool__"), cls


def test_the_hierarchy_is_flat_under_base_cache():
    for cls in _CACHES:
        want = (owned.RotatingKVCache if cls is owned.BufferedRotatingKVCache
                else owned._BaseCache)
        assert cls.__bases__ == (want,), cls
    assert owned._BaseCache.__bases__ == (object,)


def test_every_owned_cache_is_covered():
    assert set(_filled()) == set(_CACHES)


def test_only_the_quantized_classes_carry_bits():
    carry = {cls for cls, c in _filled().items() if hasattr(c, "bits")}
    assert carry == {owned.QuantizedKVCache, owned.BatchQuantizedKVCache}


@pytest.mark.parametrize("cls", _CACHES, ids=lambda c: c.__name__)
def test_deepcopy_is_independent(cls):
    c = _filled()[cls]
    twin = copy.deepcopy(c)
    assert type(twin) is cls
    before = [getattr(c, "offset", None)]
    for k, v in vars(c).items():
        if isinstance(v, mx.array):
            assert vars(twin)[k] is not v, k
            assert mx.array_equal(vars(twin)[k], v).item(), k
    if hasattr(twin, "update_and_fetch") and cls not in (
            owned.PoolingCache, owned.BatchPoolingCache, owned.ArraysCache,
            owned.CacheList, owned.StaticPrefixKVCache):
        rows = 2 if cls.__name__.startswith("Batch") else 1
        twin.update_and_fetch(*_kv(n=1, rows=rows))
        assert [getattr(c, "offset", None)] == before


def test_arrays_cache_reads_are_new_arrays_after_advance():
    c = owned.ArraysCache(1, left_padding=[2, 0])
    c.lengths = mx.array([5, 3])
    c.advance(1)
    assert c.left_padding is not c.left_padding
    assert c.lengths is not c.lengths
    assert c.left_padding.tolist() == [1, -1]
    assert c.lengths.tolist() == [4, 2]


def test_qwen35_arrays_cache_returns_one_array_until_the_next_advance():
    from gmlx.models.qwen35.cache import Qwen35ArraysCache

    c = Qwen35ArraysCache(1, left_padding=[2, 0])
    c.lengths = mx.array([5, 3])
    c.advance(1)
    lp, ln = c.left_padding, c.lengths
    assert c.left_padding is lp and c.lengths is ln
    assert lp.tolist() == [1, -1] and ln.tolist() == [4, 2]
    c.advance(1)
    assert c.left_padding is not lp
    assert c.left_padding.tolist() == [0, -2]
