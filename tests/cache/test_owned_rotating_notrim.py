"""GMLX_SLIDING_NOTRIM=1 makes RotatingKVCache keep every key through a
chunked prefill.

In the mode each chunk's update returns the keys of every earlier chunk
in order, and the mask built before it equals the rows of the unchunked
prefill's mask for that chunk over the same key columns, so a chunked
prefill hands SDPA a prefix of the tensor an unchunked one does. The
first decode step trims the buffer to the window and leaves the buffer
the default mode leaves. Without the switch the cache trims as stock.
Each key carries its column plus one. The stock bodies ignore the switch
and must fail the no-trim cases. CPU.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

import gmlx.cache.kv_cache as owned
from stock_twin import stock_method

_W = 8
_CHUNKS = [
    [4, 4, 4, 4],        # chunks inside the window, then past it
    [5, 11, 3],          # one chunk wider than the window
    [16, 16],            # every chunk past the window
    [3, 2],              # the window never fills
]


@pytest.fixture(params=["owned", "stock"])
def body(request, monkeypatch):
    if request.param == "stock":
        for name in ("_update_concat", "make_mask"):
            monkeypatch.setattr(owned.RotatingKVCache, name,
                                stock_method(owned.RotatingKVCache, name))
    return request.param


@pytest.fixture
def notrim(monkeypatch):
    monkeypatch.setenv("GMLX_SLIDING_NOTRIM", "1")


def _keys(start: int, n: int) -> mx.array:
    cols = mx.arange(start + 1, start + n + 1, dtype=mx.float32)
    return mx.broadcast_to(cols[None, None, :, None], (1, 2, n, 4))


def _dense(mask, n: int, width: int) -> np.ndarray:
    """(n, width) visibility; "causal" aligns the queries to the right."""
    if isinstance(mask, str):
        assert mask == "causal"
        return np.tril(np.ones((n, width), dtype=bool), k=width - n)
    return np.array(mx.broadcast_to(mask, (n, width)))


def _cols(keys: mx.array) -> list:
    return np.array(keys[0, 0, :, 0]).astype(np.int64).tolist()


def _unchunked_mask(total: int, keep: int) -> np.ndarray:
    c = owned.RotatingKVCache(_W, keep=keep)
    return _dense(c.make_mask(total), total, total)


def _prefill(chunks, keep=0):
    """Per chunk: (columns the update returns, its mask, the full-run mask
    rows over the same columns)."""
    total = sum(chunks)
    full = _unchunked_mask(total, keep)
    c = owned.RotatingKVCache(_W, keep=keep)
    done, out = 0, []
    for n in chunks:
        mask = c.make_mask(n)
        keys, _ = c.update_and_fetch(_keys(done, n), _keys(done, n))
        width = keys.shape[2]
        out.append((_cols(keys), _dense(mask, n, width),
                    full[done:done + n, :width]))
        done += n
    return c, out


@pytest.mark.parametrize("keep", [0, 2])
@pytest.mark.parametrize("chunks", _CHUNKS)
def test_each_chunk_sees_every_earlier_key(body, notrim, chunks, keep):
    _, out = _prefill(chunks, keep)
    done, ok = 0, True
    for n, (cols, _mask, _full) in zip(chunks, out):
        ok &= cols == list(range(1, done + n + 1))
        done += n
    assert ok is (body == "owned" or sum(chunks[:-1]) < _W), body


@pytest.mark.parametrize("chunks", _CHUNKS)
def test_each_chunk_mask_is_the_unchunked_mask_rows(body, notrim, chunks):
    _, out = _prefill(chunks)
    ok = all(np.array_equal(mask, full) for _cols_, mask, full in out)
    assert ok is (body == "owned" or sum(chunks[:-1]) < _W), body


@pytest.mark.parametrize("keep", [0, 2])
@pytest.mark.parametrize("chunks", _CHUNKS[:3])
def test_the_first_decode_step_trims_to_the_window(notrim, monkeypatch,
                                                   chunks, keep):
    total = sum(chunks)
    c, _ = _prefill(chunks, keep)
    steps = []
    for i in range(3):
        keys, _ = c.update_and_fetch(_keys(total + i, 1), _keys(total + i, 1))
        steps.append(_cols(keys))
    monkeypatch.delenv("GMLX_SLIDING_NOTRIM")
    d, _ = _prefill(chunks, keep)
    want = []
    for i in range(3):
        keys, _ = d.update_and_fetch(_keys(total + i, 1), _keys(total + i, 1))
        want.append(_cols(keys))
    assert all(len(s) == _W for s in steps)
    assert steps == want


@pytest.mark.parametrize("setting", [None, "0"])
@pytest.mark.parametrize("chunks", _CHUNKS)
def test_without_the_switch_the_cache_trims_as_stock(monkeypatch, chunks,
                                                     setting):
    if setting is None:
        monkeypatch.delenv("GMLX_SLIDING_NOTRIM", raising=False)
    else:
        monkeypatch.setenv("GMLX_SLIDING_NOTRIM", setting)
    _, got = _prefill(chunks)
    for name in ("_update_concat", "make_mask"):
        monkeypatch.setattr(owned.RotatingKVCache, name,
                            stock_method(owned.RotatingKVCache, name))
    _, want = _prefill(chunks)
    for (gc, gm, _), (wc, wm, _) in zip(got, want):
        assert gc == wc and np.array_equal(gm, wm)
    if sum(chunks[:-1]) >= _W:
        assert len(got[-1][0]) == _W - 1 + chunks[-1]
