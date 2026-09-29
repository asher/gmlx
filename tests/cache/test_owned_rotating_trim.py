"""A trim of a rotating cache in temporal order drops the trimmed columns,
and an armed verify update past a full window is undone from its stash.

Each key carries its position and a tag: 0 for the prompt, 99 for a
rejected draft, 1 for the token written after the rollback, 5 for a row
that joins later. After a verify block past a full window and a trim of
its drafts, a one-token step must attend what a single-row cache fed only
the kept history attends, both after a plain trim and after the
DeepSeek-V4 undo of an armed verify. The stock trim bodies keep the
trimmed columns and must fail the plain cases. CPU.
"""

from __future__ import annotations

import importlib

import mlx.core as mx
import numpy as np
import pytest

import gmlx.cache.kv_cache as owned
from stock_twin import stock_method

W = 8
_CLASSES = (owned.RotatingKVCache, owned.BatchRotatingKVCache)


@pytest.fixture(params=["owned", "stock"])
def body(request, monkeypatch):
    if request.param == "stock":
        for cls in _CLASSES:
            monkeypatch.setattr(cls, "trim", stock_method(cls, "trim"))
    return request.param


def _new(cls, pads=(0,)):
    if cls is owned.RotatingKVCache:
        return cls(max_size=W)
    return cls(W, list(pads))


def _keys(cols, tags, rows=1):
    k = mx.stack([mx.array(cols, dtype=mx.float32) + 1,
                  mx.array(tags, dtype=mx.float32)], axis=-1)
    return mx.broadcast_to(k[None, None], (rows, 1, len(cols), 2))


def _seen(cache, k):
    """The (position, tag) pairs each row's query attends in a one-token
    update with keys ``k``."""
    batch = hasattr(cache, "rotated")
    mask = cache.make_mask(1) if batch else None
    kk, _ = cache.update_and_fetch(k, k)
    out = []
    for b in range(k.shape[0]):
        if mask is None:
            vis = np.ones(kk.shape[2], dtype=bool)
        else:
            full = mx.broadcast_to(mask, (k.shape[0], 1, 1, kk.shape[2]))
            vis = np.array(full)[b, 0, 0]
        cols = np.array(kk[b, 0, :, 0]).astype(int) - 1
        tags = np.array(kk[b, 0, :, 1]).astype(int)
        out.append(sorted((int(c), int(t))
                          for c, t, v in zip(cols, tags, vis) if v))
    return out


def _truth(cols, tags, col, tag):
    """What a single-row cache fed ``cols`` attends at ``col``."""
    c = owned.RotatingKVCache(max_size=W)
    k = _keys(cols, tags)
    c.update_and_fetch(k, k)
    return _seen(c, _keys([col], [tag]))[0]


def _rolled_back(cache, rows, rejected=2):
    """Prompt 0..11, a one-token step at 12, a verify block 13..15 whose
    last ``rejected`` columns are drafts, then the rollback trim."""
    for cols, tags in ((list(range(12)), [0] * 12), ([12], [0]),
                       ([13, 14, 15], [0] * (3 - rejected) + [99] * rejected)):
        k = _keys(cols, tags, rows)
        cache.update_and_fetch(k, k)
    assert cache.trim(rejected) == rejected


def _one_token_after_trim(kind):
    """(seen, truth) for every row of a one-token step after the trim."""
    pads = {"single": None, "batch-1": [0], "batch-2": [0, 9]}[kind]
    cache = (owned.RotatingKVCache(max_size=W) if pads is None
             else owned.BatchRotatingKVCache(W, pads))
    rows = 1 if pads is None else len(pads)
    _rolled_back(cache, rows)
    seen = _seen(cache, _keys([14], [1], rows))
    truth = [_truth(list(range(p, 14)), [0] * (14 - p), 14, 1)
             for p in (pads or [0])]
    return seen, truth


def _extend_then_one_token():
    cache = owned.BatchRotatingKVCache(W, [0, 9])
    _rolled_back(cache, 2)
    joining = owned.RotatingKVCache(max_size=W)
    k = _keys(list(range(5)), [5] * 5)
    joining.update_and_fetch(k, k)
    cache.extend(owned.BatchRotatingKVCache.merge([joining]))
    k = mx.concatenate([_keys([14], [1], 2), _keys([5], [1])], axis=0)
    seen = _seen(cache, k)
    truth = [_truth(list(range(p, 14)), [0] * (14 - p), 14, 1)
             for p in (0, 9)]
    truth.append(_truth(list(range(5)), [5] * 5, 5, 1))
    return seen, truth


def _armed_undo(kind, rejected):
    """The DeepSeek-V4 path: an armed verify, then its trim."""
    pads = None if kind == "single" else [0, 9]
    cache = (owned.RotatingKVCache(max_size=W) if pads is None
             else owned.BatchRotatingKVCache(W, pads))
    rows = 1 if pads is None else 2
    for cols, tags in ((list(range(12)), [0] * 12), ([12], [0])):
        k = _keys(cols, tags, rows)
        cache.update_and_fetch(k, k)
    owned.set_undo_armed(True)
    try:
        k = _keys([13, 14, 15], [0] * (3 - rejected) + [99] * rejected, rows)
        cache.update_and_fetch(k, k)
    finally:
        owned.set_undo_armed(False)
    assert cache.is_trimmable()
    assert cache.trim(rejected) == rejected
    assert cache._mtp_undo is None
    col = 16 - rejected
    seen = _seen(cache, _keys([col], [1], rows))
    truth = [_truth(list(range(p, col)), [0] * (col - p), col, 1)
             for p in (pads or [0])]
    return seen, truth


@pytest.mark.parametrize("kind", ["single", "batch-1", "batch-2"])
def test_a_one_token_step_after_a_trim_matches_a_single_row(body, kind):
    seen, truth = _one_token_after_trim(kind)
    assert (seen == truth) is (body == "owned"), (body, kind)


def test_an_admission_then_a_one_token_step_matches_a_single_row(body):
    seen, truth = _extend_then_one_token()
    assert (seen == truth) is (body == "owned"), body


@pytest.mark.parametrize("rejected", [1, 2])
@pytest.mark.parametrize("kind", ["single", "batch"])
def test_an_armed_verify_rollback_matches_a_single_row(kind, rejected):
    seen, truth = _armed_undo(kind, rejected)
    assert seen == truth, (kind, rejected)


def test_a_trim_below_the_window_keeps_the_growing_buffer():
    cache = owned.BatchRotatingKVCache(W, [0])
    k = _keys(list(range(4)), [0] * 4)
    cache.update_and_fetch(k, k)
    k = _keys([4, 5, 6], [0, 99, 99])
    cache.update_and_fetch(k, k)
    assert cache.trim(2) == 2
    for col in (5, 6, 7, 8):
        seen = _seen(cache, _keys([col], [1]))[0]
        want = [(c, 0) for c in range(max(0, col - W + 1), 5)]
        want += [(c, 1) for c in range(5, col + 1)]
        assert seen == want, col


@pytest.mark.parametrize("cls", _CLASSES, ids=lambda c: c.__name__)
def test_a_trim_below_the_window_past_its_start_keeps_the_buffer(cls):
    """A trim of a whole verify block past a full window, trim(3) at _idx
    10, leaves _idx 7 below the window on a buffer that no longer starts at
    the first token. Cut to _idx, the buffer would make _update_in_place
    grow it from an offset past the window, which raises, so the trim
    keeps the buffer. No rollback trims a whole block."""
    cache = _new(cls)
    for cols in (list(range(12)), [12], [13, 14, 15]):
        k = _keys(cols, [0] * len(cols))
        cache.update_and_fetch(k, k)
    assert cache._idx == W + 2
    assert cache.trim(3) == 3
    assert cache._idx == W - 1 and cache.keys.shape[2] == W + 2, (
        cache._idx, cache.keys.shape)
    assert len(_seen(cache, _keys([13], [1]))[0]) == W


def _state(cache):
    offset = getattr(cache, "offset")
    if isinstance(offset, mx.array):
        offset = offset.tolist()
    return (cache.keys, cache._idx, offset, getattr(cache, "_offset", None),
            getattr(cache, "rotated", None))


@pytest.mark.parametrize("cls", _CLASSES, ids=lambda c: c.__name__)
@pytest.mark.parametrize("n", [2, W + 3])
def test_a_trim_past_the_write_position_raises(body, cls, n):
    """The stock trim leaves _idx negative: after a one-token step wraps
    the ring to _idx 1, and after a verify block puts the buffer back in
    order at _idx W + 2 with the older tokens off the window. The owned
    trim raises and leaves the cache as it was."""
    cache = _new(cls)
    steps = [list(range(12)), [12]]
    if n > W:
        steps.append([13, 14, 15])
    for cols in steps:
        k = _keys(cols, [0] * len(cols))
        cache.update_and_fetch(k, k)
    before = _state(cache)
    assert before[1] < n, before[1]
    if body == "stock":
        cache.trim(n)
        assert cache._idx < 0
        return
    with pytest.raises(ValueError, match="write position"):
        cache.trim(n)
    after = _state(cache)
    assert after[0] is before[0] and after[1:] == before[1:]
    # A trim within the write position still runs.
    assert cache.trim(1) == 1


def test_each_rotating_class_is_the_owned_class():
    """No rollback reaches the depth raise because is_trimmable reports a
    buffer past its window as not trimmable. Each origin's two classes are
    the owned ones, so is_trimmable is gmlx text."""
    for m in ("mlx_lm.models.cache", "mlx_vlm.models.cache"):
        mod = importlib.import_module(m)
        for c in ("RotatingKVCache", "BatchRotatingKVCache"):
            assert getattr(mod, c) is getattr(owned, c), (m, c)
