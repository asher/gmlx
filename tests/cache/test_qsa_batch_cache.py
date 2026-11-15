"""BatchQSAKVCache: the decode batch's cache on a qwen4exp QSA layer, one
single-row QSAKVCache per row. Joins, leaves and filters move rows and
copy nothing; the batch's readers see a batched cache with per-row
offsets; a row's extract is the row as a single-row QSA cache, the layout
a lookup signs."""

from __future__ import annotations

import mlx.core as mx
import pytest

from gmlx.cache.apc_pooling import _is_batched_cache
from gmlx.cache.kv_append import eval_cache
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


def _chunk(seed, n=9):
    mx.random.seed(seed)
    return (mx.random.normal((1, H, n, D)), mx.random.normal((1, H, n, D)),
            mx.random.normal((1, n, DI)))


def test_extract_is_the_row_as_a_single_row_qsa_cache():
    """The extract tags "qsa:4", as a lookup signs, and holds the row's
    whole state. The live row's tail folds in place first. A prefill
    chunk on each side afterwards writes into its own rows only."""
    a, b = _row(13, 1), _row(9, 2, tail=3)
    c = BatchQSAKVCache.merge([a, b])
    k, v = b.kv_full()
    want = (k, v, b._ik_rows(0, 12))
    mx.eval(want)
    got = c.extract(1)
    assert type(got) is QSAKVCache and _qsa_tag(got) == "qsa:4"
    assert got.offset == 12 and b._tl == 0 and got._tl == 0
    for x, y in zip(got.state, want):
        assert mx.array_equal(x, y).item()
    assert c.rows == [a, b], "the row stays in the batch"
    mine, theirs = _chunk(5), _chunk(6)
    got.append_qsa(*mine)
    b.append_qsa(*theirs)
    for cache, rows in ((got, mine), (b, theirs)):
        k, v, ik = cache.state
        for x, y in zip((k[..., :12, :], v[..., :12, :], ik[:, :12]), want):
            assert mx.array_equal(x, y).item()
        for x, y in zip((k[..., 12:, :], v[..., 12:, :], ik[:, 12:]), rows):
            assert mx.array_equal(x, y).item()


def test_the_decode_lift_joins_qsa_rows_into_the_batch_form():
    """The decode batch lifts a layer by its class's merge: QSA rows join a
    BatchQSAKVCache as themselves, indexer stream included. The APC merge,
    which passes prefix lengths, still rebuilds one QSA row."""
    a, b = _row(13, 1), _row(9, 2, tail=3)
    lifted = type(a).merge([a, b])
    assert type(lifted) is BatchQSAKVCache and lifted.rows == [a, b]
    one = QSAKVCache.merge([a], prefix_lens=[8])
    assert type(one) is QSAKVCache and one.offset == 8
