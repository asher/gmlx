"""The owned batch caches carry the cascade stamp through filter and extend.

The shared-prefix decode route stamps each batch cache with the shared
prefix length and each row's packed prompt tokens. BatchKVCache and
BatchQuantizedKVCache keep the stamp entries of the rows a filter keeps,
named by index as the batch loop and the spec engine pass them, and
derive the stamp again when extend takes in rows: a same-prefix admission
narrows the shared prefix, and an unstamped or diverging one clears it.
The stock bodies leave the stamp as it was and must fail. CPU.
"""

from __future__ import annotations

import mlx.core as mx
import pytest

import gmlx.cache.kv_cache as owned
from gmlx.upstream import cascade_sdpa as cs
from stock_twin import stock_method

_CLASSES = (owned.BatchKVCache, owned.BatchQuantizedKVCache)
_PRE = list(range(64))


@pytest.fixture(params=["owned", "stock"])
def body(request, monkeypatch):
    if request.param == "stock":
        for cls in _CLASSES:
            for name in ("extend", "filter"):
                monkeypatch.setattr(cls, name, stock_method(cls, name))
    return request.param


def _rb(ids):
    return cs._row_bytes(ids)


def _cache(cls, rows, length=4):
    """A batch cache of ``rows`` rows holding ``length`` tokens, stamped
    with ``rows`` token rows when they share a prefix."""
    c = (cls([0] * len(rows)) if cls is owned.BatchKVCache
         else cls([0] * len(rows), group_size=64, bits=8))
    k = mx.ones((len(rows), 2, length, 64), dtype=mx.float16)
    c.update_and_fetch(k, k)
    cs._stamp_caches([c], rows)
    return c


@pytest.mark.parametrize("cls", _CLASSES, ids=lambda c: c.__name__)
@pytest.mark.parametrize("keep", [[0], [1], [0, 2], [1, 2]])
def test_filter_keeps_the_indexed_rows(body, cls, keep):
    rows = [_rb(_PRE[:32] + [i]) for i in range(3)]
    cache = _cache(cls, rows)
    cache.filter(mx.array(keep, mx.int32))
    want = {"P": 32, "tok": tuple(rows[k] for k in keep)}
    assert (cache.__dict__.get("_gmlx_cascade") == want) is (body == "owned")


@pytest.mark.parametrize("cls", _CLASSES, ids=lambda c: c.__name__)
def test_filter_drops_a_stamp_that_does_not_cover_the_rows(cls):
    cache = _cache(cls, [_rb(_PRE[:32] + [i]) for i in range(3)])
    cache._gmlx_cascade = {"P": 32, "tok": cache._gmlx_cascade["tok"][:2]}
    cache.filter(mx.array([2], mx.int32))
    assert "_gmlx_cascade" not in cache.__dict__


@pytest.mark.parametrize("cls", _CLASSES, ids=lambda c: c.__name__)
def test_filter_to_row_zero_keeps_cascading_after_admission(body, cls):
    rows = [_rb(_PRE[:32] + [1]), _rb(_PRE[:32] + [2])]
    cache = _cache(cls, rows)
    cache.filter(mx.array([0], mx.int32))
    joining = _rb(_PRE[:32] + [3])
    cache.extend(_cache(cls, [joining], length=6))
    want = {"P": 32, "tok": (rows[0], joining)}
    assert (cache.__dict__.get("_gmlx_cascade") == want) is (body == "owned")


@pytest.mark.parametrize("cls", _CLASSES, ids=lambda c: c.__name__)
def test_a_same_prefix_admission_narrows_the_prefix(body, cls):
    rows = [_rb(_PRE + [7, 8]), _rb(_PRE + [9])]
    cache = _cache(cls, rows)
    assert cache._gmlx_cascade["P"] == 64
    joining = _rb(_PRE[:40] + [999])
    cache.extend(_cache(cls, [joining]))
    want = {"P": 40, "tok": (*rows, joining)}
    assert (cache.__dict__.get("_gmlx_cascade") == want) is (body == "owned")


@pytest.mark.parametrize("cls", _CLASSES, ids=lambda c: c.__name__)
@pytest.mark.parametrize("other", ["unstamped", "diverging"])
def test_an_unstamped_or_diverging_admission_clears(body, cls, other):
    cache = _cache(cls, [_rb(_PRE + [1]), _rb(_PRE + [2])])
    joining = [] if other == "unstamped" else [_rb([9, 9, 9])]
    cache.extend(_cache(cls, joining or [b""]))
    assert ("_gmlx_cascade" not in cache.__dict__) is (body == "owned")


@pytest.mark.parametrize("cls", _CLASSES, ids=lambda c: c.__name__)
def test_an_unstamped_cache_stays_unstamped(cls):
    cache = _cache(cls, [b"", b""])
    assert "_gmlx_cascade" not in cache.__dict__
    cache.extend(_cache(cls, [_rb(_PRE)]))
    cache.filter(mx.array([0, 2], mx.int32))
    assert "_gmlx_cascade" not in cache.__dict__


def test_cascade_lcp():
    a = list(range(100))
    lcp = owned._cascade_lcp
    assert lcp([_rb(a), _rb(a)]) == 100
    assert lcp([_rb(a), _rb(a[:50] + [999] + a[51:])]) == 50
    assert lcp([_rb(a), _rb(a[:30])]) == 30
    assert lcp([_rb(a)]) == 100
    assert lcp([]) == 0
    assert lcp([_rb(a), b""]) == 0
