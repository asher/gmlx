"""The KV cache's regrow and batch extends call gmlx.mlx_guards.

Past 2^31 output elements the stock pad and concatenate misplace rows on
a GPU stream (docs/upstream-defects.md), so these sites go through the
guards. Each test runs a site at a few elements on the CPU and checks the
guard calls it makes, so a site that calls the stock op fails on every
run. tests/cache/test_kv_past_int32.py runs the sites past the real limit.
"""
from __future__ import annotations

import mlx.core as mx
import pytest

from gmlx.cache.kv_cache import BatchKVCache, BatchQuantizedKVCache, BatchRotatingKVCache
from helpers import spy_guards

H, D = 2, 4


@pytest.fixture(autouse=True)
def cpu():
    with mx.stream(mx.cpu):
        yield


def test_a_decode_regrow_concatenates_through_the_guard(monkeypatch):
    """A token past a full buffer regrows the keys and the values on the
    token axis."""
    cache = BatchKVCache([0, 0])
    cache.keys, cache.values = mx.ones((2, H, 256, D)), mx.ones((2, H, 256, D))
    cache._idx, cache.offset = 256, mx.array([256, 256])
    calls = spy_guards(monkeypatch)
    cache.update_and_fetch(mx.ones((2, H, 1, D)), mx.ones((2, H, 1, D)))
    assert calls == [("concatenate", 2)] * 2


def _join(a, b, n: int, idx: int, join: int) -> None:
    """``n`` rows at ``idx`` tokens and one row at ``join`` tokens join."""
    a.offset, b.offset = mx.array([idx] * n), mx.array([join])
    a._idx, b._idx = idx, join
    a.extend(b)


def test_a_batch_extend_pads_through_the_guard(monkeypatch):
    """A longer row joins 2 rows, which extend pads on the token axis."""
    a, b = BatchKVCache([0, 0]), BatchKVCache([0])
    a.keys, a.values = mx.ones((2, H, 5, D)), mx.ones((2, H, 5, D))
    b.keys, b.values = mx.ones((1, H, 8, D)), mx.ones((1, H, 8, D))
    calls = spy_guards(monkeypatch)
    _join(a, b, 2, 5, 6)
    assert calls == [("pad", (2,))] * 2


def test_a_rotating_batch_extend_pads_through_the_guard(monkeypatch):
    """A row whose window buffer is 6 tokens wide joins 2 rows that decode
    at a full window of 4, which extend pads on the token axis."""
    a, b = BatchRotatingKVCache(4, [0, 0]), BatchRotatingKVCache(4, [0])
    a.keys, a.values = mx.ones((2, H, 4, D)), mx.ones((2, H, 4, D))
    a._idx, a._offset, a.rotated = 1, 17, True
    a.offset = mx.array([17, 17])
    b.keys, b.values = mx.ones((1, H, 6, D)), mx.ones((1, H, 6, D))
    b._idx, b._offset, b.offset = 6, 9, mx.array([9])
    calls = spy_guards(monkeypatch)
    a.extend(b)
    assert calls == [("pad", (2,))] * 2


def _quant(n: int, tokens: int) -> tuple:
    return (mx.ones((n, H, tokens, 4), mx.uint32), mx.ones((n, H, tokens, 1)),
            mx.ones((n, H, tokens, 1)))


def test_a_quantized_batch_extend_pads_every_part_through_the_guard(monkeypatch):
    """A longer row joins 2 rows, and extend pads the packed part, the
    scales and the biases of their keys and values on the token axis."""
    a, b = BatchQuantizedKVCache([0, 0]), BatchQuantizedKVCache([0])
    a.keys, a.values = _quant(2, 3), _quant(2, 3)
    b.keys, b.values = _quant(1, 8), _quant(1, 8)
    calls = spy_guards(monkeypatch)
    _join(a, b, 2, 3, 5)
    assert calls == [("pad", (2,))] * 6
