"""A BatchRotatingKVCache rebuilt with from_state decodes as the original.

The stock meta_state getter writes ``rotated`` as the string "True" or
"False", and the stock setter reads it back with ``bool``, so every
restored cache reads as rotated. ``from_state`` also skips ``__init__``,
which leaves ``_lengths`` unset, so the first update raises. The owned
setter parses the flag and seeds ``_lengths``. Each case also runs the
stock meta_state property, which must fail it. CPU.
"""

from __future__ import annotations

import mlx.core as mx
import pytest

import gmlx.cache.kv_cache as owned
from stock_twin import stock_method

_W = 8


def _keys(start, n, rows=2):
    cols = mx.arange(start + 1, start + n + 1, dtype=mx.float32)
    return mx.broadcast_to(cols[None, None, :, None], (rows, 1, n, 4))


def _filled(n):
    c = owned.BatchRotatingKVCache(_W, [0, 2])
    c.update_and_fetch(_keys(0, n), _keys(0, n))
    if n > _W:
        c.update_and_fetch(_keys(n, 1), _keys(n, 1))
    return c


def _step(c, col):
    mask = c.make_mask(1)
    k, _ = c.update_and_fetch(_keys(col, 1), _keys(col, 1))
    return k.tolist(), mx.broadcast_to(mask, (2, 1, 1, k.shape[2])).tolist()


def _round_trip_decodes_the_same(n) -> bool:
    ref = _filled(n)
    back = owned.BatchRotatingKVCache.from_state(ref.state, ref.meta_state)
    col = n + 1 if n > _W else n
    try:
        if back.rotated != ref.rotated:
            return False
        return _step(back, col) == _step(ref, col)
    except AttributeError:
        return False


@pytest.mark.parametrize("n", [3, 12], ids=["unrotated", "rotated"])
def test_a_restored_cache_decodes_as_the_original(n):
    assert _round_trip_decodes_the_same(n)


@pytest.mark.parametrize("n", [3, 12], ids=["unrotated", "rotated"])
def test_the_stock_restore_fails(n, monkeypatch):
    monkeypatch.setattr(owned.BatchRotatingKVCache, "meta_state",
                        stock_method(owned.BatchRotatingKVCache, "meta_state"))
    assert not _round_trip_decodes_the_same(n)


def test_the_setter_takes_a_bool_flag():
    c = _filled(3)
    meta = c.meta_state[:3] + (False,)
    back = owned.BatchRotatingKVCache.from_state(c.state, meta)
    assert back.rotated is False and back._lengths is None
