"""BatchQSAKVCache: the decode batch's cache on a qwen4exp QSA layer, one
single-row QSAKVCache per row. Joins, leaves and filters move rows and
copy nothing; the batch's readers see a batched cache with per-row
offsets; a row's extract is a plain KV view that no QSA lookup signs."""

from __future__ import annotations

import mlx.core as mx
import pytest

from gmlx.cache.apc_pooling import _is_batched_cache
from gmlx.cache.kv_append import eval_cache
from gmlx.cache.kv_cache import KVCache
from gmlx.cache.snapshot import _qsa_tag, row_kv_len
from gmlx.models.qwen4_exp.model import BatchQSAKVCache, QSAKVCache

H, D, DI = 2, 16, 8


def _row(n, seed, tail=0):
    """A QSA row of ``n`` prefill rows plus ``tail`` decode appends."""
    mx.random.seed(seed)
    c = QSAKVCache(ratio=4)
    c.append_qsa(mx.random.normal((1, H, n, D)), mx.random.normal((1, H, n, D)),
                 mx.random.normal((1, n, DI)))
    for _ in range(tail):
        c.append_qsa(mx.random.normal((1, H, 1, D)),
                     mx.random.normal((1, H, 1, D)),
                     mx.random.normal((1, 1, DI)))
    return c


def _offsets(c):
    return c.offset.tolist()


def test_merge_extend_and_filter_move_rows_without_copies():
    a, b, d = _row(13, 1), _row(9, 2, tail=3), _row(21, 3)
    held = [(r._kb, r._vb, r._ib) for r in (a, b, d)]
    c = BatchQSAKVCache.merge([a])
    assert c.rows == [a] and _offsets(c) == [13]
    c.extend(BatchQSAKVCache.merge([b]))
    c.extend(d)                      # a bare row joins as itself
    assert c.rows == [a, b, d]
    assert _offsets(c) == [13, 12, 21]
    both = BatchQSAKVCache.merge([c, _row(5, 4)])
    assert both.rows[:3] == [a, b, d] and _offsets(both) == [13, 12, 21, 5]
    c.filter(mx.array([2, 0], mx.int32))
    assert c.rows == [d, a]
    c.filter([1])
    assert c.rows == [a]
    c.filter([])
    assert c.rows == [] and c.empty()
    for r, (kb, vb, ib) in zip((a, b, d), held):
        assert (r._kb, r._vb, r._ib) == (kb, vb, ib)


def test_the_batch_readers_see_a_batched_cache_with_row_offsets():
    c = BatchQSAKVCache.merge([_row(13, 1), _row(9, 2, tail=3)])
    assert _is_batched_cache(c)
    assert c.left_padding.tolist() == [0, 0]
    assert row_kv_len([c], 0) == 13 and row_kv_len([c], 1) == 12
    assert c.size() == 13 and not c.empty()
    assert c.make_mask(1) is None and c.make_mask(4) == "causal"
    assert c.nbytes == sum(r.nbytes for r in c.rows)
    # no QSA tag: the live batch is not a record layout
    assert _qsa_tag(c) is None
    with pytest.raises(NotImplementedError):
        c.to_quantized()


def test_trim_rolls_every_row_back_by_one_count():
    a, b = _row(13, 1, tail=4), _row(9, 2, tail=2)
    c = BatchQSAKVCache.merge([a, b])
    assert c.is_trimmable()
    assert c.trim(3) == 3
    assert _offsets(c) == [14, 8]
    assert (a._tl, b._tl) == (1, 0)


def test_state_is_every_row_buffer_as_held():
    a, b = _row(13, 1, tail=2), _row(9, 2)
    c = BatchQSAKVCache.merge([a, b])
    tail_before = a._tl
    got = c.state
    assert a._tl == tail_before, "reading the state folds nothing"
    assert [len(s) for s in got] == [6, 3]
    assert got[0][0] is a._kb and got[0][3] is a._kt
    assert got[1] == (b._kb, b._vb, b._ib)


def test_eval_cache_evaluates_each_row(monkeypatch):
    a, b = _row(13, 1, tail=2), _row(9, 2)
    c = BatchQSAKVCache.merge([a, b])
    seen = []
    real = mx.eval

    def spy(*args):
        seen.extend(args[0] if len(args) == 1 and isinstance(args[0], list)
                    else args)
        return real(*args)

    monkeypatch.setattr(mx, "eval", spy)
    eval_cache(c)
    ids = {id(x) for x in seen}
    for r in (a, b):
        for arr in (r._kb, r._vb, r._ib):
            assert id(arr) in ids
    assert id(a._kt) in ids


def test_the_memory_walk_counts_the_rows():
    """Admission's KV rates, the governor's reserve and the memory trace
    walk the leaf caches. A batched QSA layer's arrays are on its rows."""
    from gmlx.serve.memtrace import _cache_report, _leaf_caches

    a, b = _row(13, 1, tail=2), _row(9, 2)
    c = BatchQSAKVCache.merge([a, b])
    assert list(_leaf_caches([c])) == [a, b]
    total, kinds, _, _ = _cache_report([c])
    assert total == a.nbytes + b.nbytes > 0
    assert kinds["QSAKVCache"]["n"] == 2


def test_extract_is_a_plain_kv_view_of_the_row():
    a, b = _row(13, 1), _row(9, 2, tail=3)
    c = BatchQSAKVCache.merge([a, b])
    got = c.extract(1)
    assert type(got) is KVCache and _qsa_tag(got) is None
    assert got.offset == 12
    assert mx.array_equal(got.keys, b.keys[..., :12, :]).item()
    assert mx.array_equal(got.values, b.values[..., :12, :]).item()
    assert c.rows == [a, b], "the row stays in the batch"

