"""CPU tests for row_snapshot / retirement_store (owned MTP APC retirement).

row_snapshot must be the round-trip inverse of the batch cache classes' merge():
merge([snap_0, .., snap_{B-1}]) reconstructs the batched cache (modulo
left-padding), so a per-row snapshot restores a single-row decode faithfully.
retirement_store must round-trip through a real APCManager's exact tier.

Model-free; run with KQUANT_FORCE_CPU=1 to stay off the GPU.
"""

import mlx.core as mx
import pytest

from gmlx.cache_compat import runtime_cache_module
from gmlx.cache_snapshot import (
    retirement_store,
    row_snapshot,
)

# Runtime-origin cache classes: these fixtures stand in for caches the
# mlx-vlm generate stack builds, which since 0.6.4 are vlm-origin (on 0.6.3
# the vlm module re-exports mlx-lm's, so the classes are the same there).
_cache = runtime_cache_module()
ArraysCache = _cache.ArraysCache
BatchKVCache = _cache.BatchKVCache
BatchRotatingKVCache = _cache.BatchRotatingKVCache
CacheList = _cache.CacheList
KVCache = _cache.KVCache
RotatingKVCache = _cache.RotatingKVCache

H, D = 2, 8  # kv heads, head dim -- small, deterministic


def _kv_row(length, salt):
    """A single-row KVCache holding `length` distinct known tokens."""
    c = KVCache()
    keys = (mx.arange(length * H * D).reshape(1, H, length, D) + salt).astype(
        mx.float32)
    values = keys + 1000.0
    c.update_and_fetch(keys, values)
    mx.eval(c.keys, c.values)
    return c


def _rot_row(max_size, length, salt, single_step=True):
    """A single-row RotatingKVCache fed `length` tokens (wraps if > max_size)."""
    c = RotatingKVCache(max_size)
    if single_step:
        for t in range(length):
            k = (mx.arange(H * D).reshape(1, H, 1, D) + salt + t * 7).astype(
                mx.float32)
            c.update_and_fetch(k, k + 1000.0)
    else:
        k = (mx.arange(length * H * D).reshape(1, H, length, D) + salt).astype(
            mx.float32)
        c.update_and_fetch(k, k + 1000.0)
    mx.eval(c.keys, c.values)
    return c


def _arrays_row(salt):
    """A single-row ArraysCache with one recurrent state tensor."""
    c = ArraysCache(1)
    c[0] = (mx.arange(1 * H * D).reshape(1, H, D) + salt).astype(mx.float32)
    mx.eval(c[0])
    return c


# --------------------------------------------------------------------------
# row_snapshot: inverse of merge
# --------------------------------------------------------------------------

def test_row_snapshot_batch_kv_merge_inverse():
    rows = [_kv_row(5, 0), _kv_row(2, 100), _kv_row(7, 200)]
    merged = BatchKVCache.merge(rows)
    mx.eval(merged.keys, merged.values)
    for i, orig in enumerate(rows):
        snap = row_snapshot([merged], i)
        assert snap is not None
        s = snap[0]
        assert s.offset == orig.offset, i
        assert s.keys.shape[2] == orig.offset, i
        assert mx.array_equal(s.keys, orig.keys[..., : orig.offset, :]), i
        assert mx.array_equal(s.values, orig.values[..., : orig.offset, :]), i


def test_row_snapshot_batch_rotating_unwrapped_inverse():
    # length < max_size -> no rotation; content recovered exactly.
    rows = [_rot_row(64, 5, 0), _rot_row(64, 3, 100)]
    merged = BatchRotatingKVCache.merge(rows)
    mx.eval(merged.keys, merged.values)
    for i, orig in enumerate(rows):
        snap = row_snapshot([merged], i)
        assert snap is not None
        s = snap[0]
        assert s.offset == orig.offset, i
        assert s.keys.shape[2] == orig.size(), i
        assert mx.array_equal(
            s.keys, orig._temporal_order(orig.keys)[..., -orig.size():, :]), i


def test_row_snapshot_batch_rotating_wrapped_window():
    # length > max_size: the retained window is what sliding attention re-reads.
    max_size = 8
    rows = [_rot_row(max_size, 20, 0), _rot_row(max_size, 12, 100)]
    merged = BatchRotatingKVCache.merge(rows)
    mx.eval(merged.keys, merged.values)
    for i, orig in enumerate(rows):
        snap = row_snapshot([merged], i)
        assert snap is not None
        s = snap[0]
        # true position preserved; window bounded by max_size
        assert s.offset == orig.offset, i
        assert s.keys.shape[2] == orig.size() == max_size, i


def test_row_snapshot_arrays_merge_inverse():
    rows = [_arrays_row(0), _arrays_row(100), _arrays_row(200)]
    merged = ArraysCache.merge(rows)
    mx.eval(*[c for c in merged.cache if c is not None])
    for i, orig in enumerate(rows):
        snap = row_snapshot([merged], i)
        assert snap is not None
        assert mx.array_equal(snap[0][0], orig[0]), i


def test_row_snapshot_cachelist_hybrid():
    # A hybrid layer stack: one attention row + one recurrent row per request.
    kv_rows = [_kv_row(4, 0), _kv_row(6, 100)]
    ar_rows = [_arrays_row(10), _arrays_row(110)]
    lists = [CacheList(kv_rows[i], ar_rows[i]) for i in range(2)]
    merged = CacheList.merge(lists)
    for i in range(2):
        snap = row_snapshot([merged], i)
        assert snap is not None
        cl = snap[0]
        assert mx.array_equal(cl.caches[0].keys, kv_rows[i].keys[
            ..., : kv_rows[i].offset, :]), i
        assert mx.array_equal(cl.caches[1][0], ar_rows[i][0]), i


def test_row_snapshot_empty_row_returns_none():
    # A row with no content (offset 0) must not be stored.
    empty = _kv_row(0, 0) if False else KVCache()  # keys stays None
    full = _kv_row(3, 0)
    merged = BatchKVCache.merge([empty, full])
    mx.eval(merged.keys, merged.values)
    assert row_snapshot([merged], 0) is None
    assert row_snapshot([merged], 1) is not None


def test_row_snapshot_single_row_clone_is_decoupled():
    # B=1 caches have no extract(); row_snapshot must clone, not alias.
    c = _kv_row(4, 0)
    snap = row_snapshot([c], 0)
    assert snap is not None
    before = snap[0].keys
    # mutate the live cache; the snapshot must not change
    c.update_and_fetch(
        mx.ones((1, H, 1, D), dtype=mx.float32),
        mx.ones((1, H, 1, D), dtype=mx.float32))
    mx.eval(c.keys, before)
    assert snap[0].keys.shape[2] == 4
    assert mx.array_equal(snap[0].keys, before)


def test_row_snapshot_empty_cache_list_input():
    assert row_snapshot([], 0) is None


# --------------------------------------------------------------------------
# retirement_store: round-trip through a real APCManager exact tier
# --------------------------------------------------------------------------

def _manager():
    from mlx_vlm.apc import APCManager
    return APCManager(num_blocks=256, block_size=16)


def test_retirement_store_exact_round_trip():
    mgr = _manager()
    try:
        seq = list(range(1, 40))  # >= 2 tokens
        cache = [_kv_row(len(seq), 0), _kv_row(len(seq), 50)]
        ok = retirement_store(mgr, "exact", seq, cache)
        assert ok is True
        snap, prefix_len = mgr.lookup_exact_cache(seq + [999])
        assert prefix_len == len(seq)
        assert snap is not None and len(snap) == 2
        # restored buffer is padded to capacity; compare the valid window
        assert snap[0].offset == cache[0].offset
        assert mx.array_equal(
            snap[0].keys[..., : snap[0].offset, :],
            cache[0].keys[..., : cache[0].offset, :])
    finally:
        mgr.close()


def test_retirement_store_batched_row_exact():
    mgr = _manager()
    try:
        seq = list(range(1, 20))
        rows = [_kv_row(len(seq), 0), _kv_row(len(seq), 500)]
        merged = BatchKVCache.merge(rows)
        mx.eval(merged.keys, merged.values)
        assert retirement_store(mgr, "exact", seq, [merged], row=1) is True
        snap, prefix_len = mgr.lookup_exact_cache(seq + [7])
        assert prefix_len == len(seq)
        assert snap[0].offset == rows[1].offset
        assert mx.array_equal(
            snap[0].keys[..., : snap[0].offset, :],
            rows[1].keys[..., : rows[1].offset, :])
    finally:
        mgr.close()


def test_retirement_store_guards():
    mgr = _manager()
    try:
        cache = [_kv_row(4, 0)]
        assert retirement_store(None, "exact", [1, 2, 3], cache) is False
        assert retirement_store(mgr, "exact", [1], cache) is False
        assert retirement_store(mgr, "exact", None, cache) is False
    finally:
        mgr.close()


def test_retirement_store_incomplete_snapshot_skipped():
    mgr = _manager()
    try:
        seq = list(range(1, 10))
        merged = BatchKVCache.merge([KVCache(), _kv_row(len(seq), 0)])
        mx.eval(merged.keys, merged.values)
        # row 0 has no content -> store must decline, not raise
        assert retirement_store(mgr, "exact", seq, [merged], row=0) is False
        snap, prefix_len = mgr.lookup_exact_cache(seq + [1])
        assert prefix_len == 0 and snap is None
    finally:
        mgr.close()


# --------------------------------------------------------------------------
# _retire_b1: the owned B=1 finish-seam wiring (offset guard, stash consume)
# --------------------------------------------------------------------------

# The retire context is stashed on the request's FIRST CACHE ENTRY (not the
# model: lazy generator close races with the next request's prefill), then
# popped into a generator local BEFORE target-cache buffering can swap
# rotating entries out from under the attr.

def _fake_model(manager):
    import types
    m = types.SimpleNamespace()
    m._kq_apc_manager = manager
    return m


def _ctx(full_ids):
    return {"full_ids": full_ids, "extra_hash": 0, "mode": "exact"}


def test_pop_retire_ctx_detaches_from_first_entry():
    from gmlx.speculative import _pop_retire_ctx
    cache = [_kv_row(4, 0)]
    cache[0]._kq_apc_retire = _ctx([1, 2])
    ctx = _pop_retire_ctx(cache)
    assert ctx == _ctx([1, 2])
    assert cache[0]._kq_apc_retire is None  # consumed
    # the popped ctx survives the entry being swapped out (the
    # BufferedRotatingKVCache replacement scenario on rotating archs)
    cache[0] = _kv_row(4, 99)
    assert ctx["full_ids"] == [1, 2]
    # no stash / empty list -> None
    assert _pop_retire_ctx(cache) is None
    assert _pop_retire_ctx([]) is None


def test_retire_b1_stores_on_offset_match():
    from gmlx.speculative import _retire_b1
    mgr = _manager()
    try:
        full_ids = list(range(1, 10))
        generated = [50, 51, 52]  # includes the first bonus as generated[0]
        seq = full_ids + generated
        cache = [_kv_row(len(seq), 0)]  # offset == len(seq): clean
        _retire_b1(_fake_model(mgr), cache, generated, _ctx(full_ids))
        snap, prefix_len = mgr.lookup_exact_cache(seq + [7])
        assert prefix_len == len(seq)
        assert snap is not None
    finally:
        mgr.close()


def test_retire_b1_pending_token_stores_all_but_last():
    # Round-boundary invariant: the newest sampled token's KV pends the next
    # verify, so offset == len(seq) - 1 is a clean finish -> store seq[:-1].
    from gmlx.speculative import _retire_b1
    mgr = _manager()
    try:
        full_ids = list(range(1, 10))
        generated = [50, 51, 52]
        seq = full_ids + generated
        cache = [_kv_row(len(seq) - 1, 0)]
        _retire_b1(_fake_model(mgr), cache, generated, _ctx(full_ids))
        snap, prefix_len = mgr.lookup_exact_cache(seq)
        assert prefix_len == len(seq) - 1
        assert snap is not None
    finally:
        mgr.close()


def test_retire_b1_skips_on_offset_mismatch():
    from gmlx.speculative import _retire_b1
    mgr = _manager()
    try:
        full_ids = list(range(1, 10))
        generated = [50, 51, 52]
        seq = full_ids + generated
        # MORE than len(seq) (stale rejected-draft KV tail) and more than one
        # short (hidden trim) must both skip; only len(seq)/len(seq)-1 store.
        for bad_len in (len(seq) + 4, len(seq) - 2):
            cache = [_kv_row(bad_len, 0)]
            _retire_b1(_fake_model(mgr), cache, generated, _ctx(full_ids))
            snap, prefix_len = mgr.lookup_exact_cache(seq + [7])
            assert prefix_len == 0 and snap is None  # nothing stored
    finally:
        mgr.close()


def test_retire_b1_noop_without_manager_or_context():
    from gmlx.speculative import _retire_b1
    cache = [_kv_row(6, 0)]
    # no retire context
    mgr = _manager()
    try:
        _retire_b1(_fake_model(mgr), cache, [1, 2, 3], None)  # must not raise
    finally:
        mgr.close()
    # retire context present but manager None
    _retire_b1(_fake_model(None), cache, [3], _ctx([1, 2]))
    # empty cache list with a context: offset 0 mismatch, no-op
    _retire_b1(_fake_model(None), [], [3], _ctx([1, 2]))


# --------------------------------------------------------------------------
# _retire_batch_row: the B>1 finish-seam wiring (block mode only)
# --------------------------------------------------------------------------


def test_retire_batch_row_block_mode_harvests():
    from gmlx.speculative import _retire_batch_row
    mgr = _manager()
    try:
        block = 16
        full_ids = list(range(1, 40))
        gen_row = [100 + i for i in range(30)]
        seq = full_ids + gen_row
        position = len(seq) - 1  # round-boundary invariant
        rows = [_kv_row(position, 0), _kv_row(position, 500)]
        merged = BatchKVCache.merge(rows)
        mx.eval(merged.keys, merged.values)
        ctx = {"full_ids": full_ids, "extra_hash": 0, "mode": "block"}
        _retire_batch_row(_fake_model(mgr), [merged], 1, ctx, gen_row,
                          position)
        # full blocks of the stored prefix are now in the pool
        n_blocks = (len(seq) - 1) // block
        blocks, matched = mgr.lookup_prefix(seq, extra_hash=0)
        try:
            assert matched == n_blocks * block
        finally:
            if blocks:
                mgr.release(blocks)
    finally:
        mgr.close()


def test_retire_batch_row_exact_mode_skips():
    # Exact-mode retirement is a full row clone: NEVER at B>1 (lane stall);
    # the drafter-state sidecar is the planned fix.
    from gmlx.speculative import _retire_batch_row
    mgr = _manager()
    try:
        full_ids = list(range(1, 20))
        gen_row = [50, 51, 52]
        seq = full_ids + gen_row
        cache = [_kv_row(len(seq) - 1, 0)]
        ctx = {"full_ids": full_ids, "extra_hash": 0, "mode": "exact"}
        _retire_batch_row(_fake_model(mgr), cache, 0, ctx, gen_row,
                          len(seq) - 1)
        snap, prefix_len = mgr.lookup_exact_cache(seq)
        assert prefix_len == 0 and snap is None
        blocks, matched = mgr.lookup_prefix(seq, extra_hash=0)
        try:
            assert matched == 0
        finally:
            if blocks:
                mgr.release(blocks)
    finally:
        mgr.close()


def test_retire_batch_row_short_position_skips():
    from gmlx.speculative import _retire_batch_row
    mgr = _manager()
    try:
        full_ids = list(range(1, 40))
        gen_row = [50, 51, 52]
        seq = full_ids + gen_row
        cache = [_kv_row(20, 0)]
        ctx = {"full_ids": full_ids, "extra_hash": 0, "mode": "block"}
        # position lags the token count (hidden trim): must skip, not store
        _retire_batch_row(_fake_model(mgr), cache, 0, ctx, gen_row, 20)
        blocks, matched = mgr.lookup_prefix(seq, extra_hash=0)
        try:
            assert matched == 0
        finally:
            if blocks:
                mgr.release(blocks)
    finally:
        mgr.close()


# --------------------------------------------------------------------------
# drafter-KV sidecar: store/lookup round-trip, exact-length gate, salt
# isolation, coverage guards
# --------------------------------------------------------------------------

from gmlx.cache_snapshot import (  # noqa: E402
    drafter_sidecar_lookup,
    drafter_sidecar_store,
)


class _FakeDrafter:
    supports_kv_sidecar = True

    def __init__(self, caches):
        self._caches = list(caches)

    def export_kv(self):
        return self._caches

    def restore_kv(self, caches):
        self._caches = list(caches)


def test_sidecar_store_lookup_round_trip():
    mgr = _manager()
    try:
        ids = list(range(1, 13))  # 12 prompt tokens
        drafter = _FakeDrafter([_kv_row(12, 0)])  # head offset == prompt len
        assert drafter_sidecar_store(mgr, drafter, ids, 10, extra_hash=0)
        # continuation prompt hits at exactly the stored prefix
        cont = ids + [200, 201, 202]
        side = drafter_sidecar_lookup(mgr, cont, 10, extra_hash=0)
        assert side is not None and len(side) == 1
        assert side[0].offset == 10  # trimmed to the stored prefix
        assert mx.array_equal(
            side[0].keys[..., :10, :],
            drafter._caches[0].keys[..., :10, :])
        # live drafter cache untouched by the store (clone-then-trim)
        assert drafter._caches[0].offset == 12
    finally:
        mgr.close()


def test_sidecar_lookup_exact_length_gate():
    # A near-miss sidecar is worse than a cold start (positional hole), so
    # only an exact-length match may be returned.
    mgr = _manager()
    try:
        ids = list(range(1, 13))
        drafter = _FakeDrafter([_kv_row(12, 0)])
        assert drafter_sidecar_store(mgr, drafter, ids, 10, extra_hash=0)
        cont = ids + [200, 201]
        assert drafter_sidecar_lookup(mgr, cont, 9, extra_hash=0) is None
        assert drafter_sidecar_lookup(mgr, cont, 11, extra_hash=0) is None
        assert drafter_sidecar_lookup(mgr, cont, 10, extra_hash=0) is not None
    finally:
        mgr.close()


def test_sidecar_salt_isolation():
    # Sidecar entries must never surface as real full-cache exact entries,
    # and a real entry must never satisfy a sidecar probe.
    mgr = _manager()
    try:
        ids = list(range(1, 13))
        drafter = _FakeDrafter([_kv_row(12, 0)])
        assert drafter_sidecar_store(mgr, drafter, ids, 10, extra_hash=0)
        snap, plen = mgr.lookup_exact_cache(ids + [7], extra_hash=0)
        assert plen == 0 and snap is None  # unsalted probe misses the sidecar
        assert mgr.store_exact_cache(ids[:11], [_kv_row(11, 500)],
                                     extra_hash=0)
        # real entry at 11 does not satisfy a sidecar probe at 11
        assert drafter_sidecar_lookup(mgr, ids + [7], 11, extra_hash=0) is None
    finally:
        mgr.close()


def test_sidecar_store_does_not_evict_real_exact_entries():
    # THE eviction regression: the manager's exact LRU defaults to 2 slots
    # (APC_EXACT_CACHE_ENTRIES); a request stores 2-3 sidecars, which used to
    # push the multi-GB real entries they accompany out of the LRU and turn
    # every follow-up L1 lookup into a miss. Sidecars live in their own index.
    mgr = _manager()
    try:
        ids = list(range(1, 30))
        assert mgr.store_exact_cache(ids, [_kv_row(len(ids), 0)],
                                     extra_hash=0)
        drafter = _FakeDrafter([_kv_row(len(ids), 100)])
        for store_len in (20, 24, 28):
            assert drafter_sidecar_store(mgr, drafter, ids, store_len)
        snap, plen = mgr.lookup_exact_cache(ids + [99], extra_hash=0)
        assert plen == len(ids) and snap is not None, (
            "sidecar stores evicted the real exact entry")
    finally:
        mgr.close()


def test_sidecar_index_capacity_lru():
    from gmlx import cache_snapshot as cs
    mgr = _manager()
    old = cs._SIDECAR_ENTRIES
    cs._SIDECAR_ENTRIES = 2
    try:
        ids = list(range(1, 31))
        drafter = _FakeDrafter([_kv_row(30, 0)])
        for store_len in (10, 20, 28):
            assert drafter_sidecar_store(mgr, drafter, ids, store_len)
        probe = ids + [99]
        assert drafter_sidecar_lookup(mgr, probe, 10) is None  # oldest out
        assert drafter_sidecar_lookup(mgr, probe, 20) is not None
        assert drafter_sidecar_lookup(mgr, probe, 28) is not None
    finally:
        cs._SIDECAR_ENTRIES = old
        mgr.close()


def test_sidecar_disk_restart(tmp_path):
    # Sidecars persist on the salted disk tier: a fresh manager over the same
    # directory (empty memory index) must serve the sidecar, and the loaded
    # entry must not satisfy an unsalted real-entry probe.
    from mlx_vlm.apc import APCManager, DiskBlockStore

    ids = list(range(1, 25))
    drafter = _FakeDrafter([_kv_row(24, 0)])
    mgr1 = APCManager(num_blocks=64, block_size=16,
                      disk=DiskBlockStore(tmp_path / "apc", namespace="t"))
    try:
        assert drafter_sidecar_store(mgr1, drafter, ids, 20)
    finally:
        mgr1.close()  # joins the disk writer; flushes pending shards

    mgr2 = APCManager(num_blocks=64, block_size=16,
                      disk=DiskBlockStore(tmp_path / "apc", namespace="t"))
    try:
        probe = ids + [99]
        side = drafter_sidecar_lookup(mgr2, probe, 20)
        assert side is not None and side[0].offset == 20
        assert mx.array_equal(side[0].keys[..., :20, :],
                              drafter._caches[0].keys[..., :20, :])
        # exact-length gate holds on the disk path too
        assert drafter_sidecar_lookup(mgr2, probe, 19) is None
        # real-entry probe never sees the salted sidecar shard
        snap, plen = mgr2.lookup_exact_cache(probe, extra_hash=0)
        assert plen == 0 and snap is None
    finally:
        mgr2.close()


def test_sidecar_store_guards():
    mgr = _manager()
    try:
        ids = list(range(1, 13))
        good = _FakeDrafter([_kv_row(12, 0)])
        assert drafter_sidecar_store(None, good, ids, 10) is False
        assert drafter_sidecar_store(mgr, None, ids, 10) is False
        assert drafter_sidecar_store(mgr, good, ids, 0) is False
        # missing capability flag
        plain = _FakeDrafter([_kv_row(12, 0)])
        plain.supports_kv_sidecar = False
        assert drafter_sidecar_store(mgr, plain, ids, 10) is False
        # head KV covers fewer rows than the store length: unfaithful, skip
        short = _FakeDrafter([_kv_row(8, 0)])
        assert drafter_sidecar_store(mgr, short, ids, 10) is False
        assert drafter_sidecar_lookup(mgr, ids + [7], 10) is None
    finally:
        mgr.close()


def test_pop_drafter_warm_detaches_from_first_entry():
    from gmlx.speculative import _pop_drafter_warm
    cache = [_kv_row(4, 0)]
    side = [_kv_row(3, 50)]
    cache[0]._kq_apc_drafter_warm = side
    warm = _pop_drafter_warm(cache)
    assert warm is side
    assert cache[0]._kq_apc_drafter_warm is None  # consumed
    assert _pop_drafter_warm(cache) is None
    assert _pop_drafter_warm([]) is None


def test_sidecar_post_prefill_covered_stores_both_keys():
    from gmlx.speculative import _sidecar_post_prefill
    mgr = _manager()
    try:
        full_ids = list(range(1, 13))  # 12 prompt tokens
        drafter = _FakeDrafter([_kv_row(12, 0)])
        ctx = {"full_ids": full_ids, "extra_hash": 0,
               "checkpoint_len": 9, "manager": mgr}
        _sidecar_post_prefill(drafter, ctx)
        assert drafter._kq_head_covered is True
        cont = full_ids + [200, 201]
        assert drafter_sidecar_lookup(mgr, cont, 9) is not None
        assert drafter_sidecar_lookup(mgr, cont, 12) is not None
    finally:
        mgr.close()


def test_sidecar_post_prefill_uncovered_head_skipped():
    # Suffix-only seeded head (offset < prompt len): rows sit at the wrong
    # positions; storing it would poison future turns. Flag stays False so
    # the retirement-time sidecar is skipped too.
    from gmlx.speculative import _sidecar_post_prefill
    mgr = _manager()
    try:
        full_ids = list(range(1, 13))
        drafter = _FakeDrafter([_kv_row(7, 0)])  # 12-token prompt, 7 rows
        ctx = {"full_ids": full_ids, "extra_hash": 0,
               "checkpoint_len": 9, "manager": mgr}
        _sidecar_post_prefill(drafter, ctx)
        assert drafter._kq_head_covered is False
        assert drafter_sidecar_lookup(mgr, full_ids + [7], 9) is None
        # no ctx / no capability: silently a no-op
        _sidecar_post_prefill(drafter, None)
        plain = _FakeDrafter([_kv_row(12, 0)])
        plain.supports_kv_sidecar = False
        _sidecar_post_prefill(plain, ctx)
        assert not hasattr(plain, "_kq_head_covered") or (
            plain._kq_head_covered is False)
    finally:
        mgr.close()


def test_retire_b1_stores_sidecar_when_covered():
    from gmlx.speculative import _retire_b1
    mgr = _manager()
    try:
        full_ids = list(range(1, 10))
        generated = [50, 51, 52]
        seq = full_ids + generated
        cache = [_kv_row(len(seq) - 1, 0)]  # pending-token state -> seq[:-1]
        # head faithfully covers the whole sequence
        drafter = _FakeDrafter([_kv_row(len(seq), 300)])
        drafter._kq_head_covered = True
        _retire_b1(_fake_model(mgr), cache, generated, _ctx(full_ids),
                   drafter=drafter)
        stored_len = len(seq) - 1
        snap, plen = mgr.lookup_exact_cache(seq)
        assert plen == stored_len  # target entry present
        side = drafter_sidecar_lookup(mgr, seq, stored_len)
        assert side is not None
        assert side[0].offset == stored_len
    finally:
        mgr.close()


def test_retire_b1_skips_sidecar_when_uncovered():
    from gmlx.speculative import _retire_b1
    mgr = _manager()
    try:
        full_ids = list(range(1, 10))
        generated = [50, 51, 52]
        seq = full_ids + generated
        cache = [_kv_row(len(seq) - 1, 0)]
        drafter = _FakeDrafter([_kv_row(len(seq), 300)])
        drafter._kq_head_covered = False  # suffix-only seeded head
        _retire_b1(_fake_model(mgr), cache, generated, _ctx(full_ids),
                   drafter=drafter)
        stored_len = len(seq) - 1
        snap, plen = mgr.lookup_exact_cache(seq)
        assert plen == stored_len  # target retirement unaffected
        assert drafter_sidecar_lookup(mgr, seq, stored_len) is None
    finally:
        mgr.close()


def test_retire_b1_sidecar_requires_request_nonce():
    # A lazy (GC-time) retirement can observe a drafter reseeded by a later
    # request; the sidecar store must be skipped unless the drafter's nonce
    # is this request's own sidecar_ctx object. Target retirement unaffected.
    from gmlx.speculative import _retire_b1
    mgr = _manager()
    try:
        full_ids = list(range(1, 10))
        generated = [50, 51, 52]
        seq = full_ids + generated
        stored_len = len(seq) - 1
        ctx = {"full_ids": full_ids}
        drafter = _FakeDrafter([_kv_row(len(seq), 300)])
        drafter._kq_head_covered = True
        drafter._kq_head_request = {"full_ids": full_ids}  # reseeded since
        _retire_b1(_fake_model(mgr), [_kv_row(stored_len, 0)], generated,
                   _ctx(full_ids), drafter=drafter, sidecar_ctx=ctx)
        snap, plen = mgr.lookup_exact_cache(seq)
        assert plen == stored_len  # target retirement unaffected
        assert drafter_sidecar_lookup(mgr, seq, stored_len) is None
        drafter._kq_head_request = ctx  # this request's own context object
        _retire_b1(_fake_model(mgr), [_kv_row(stored_len, 0)], generated,
                   _ctx(full_ids), drafter=drafter, sidecar_ctx=ctx)
        assert drafter_sidecar_lookup(mgr, seq, stored_len) is not None
    finally:
        mgr.close()


def test_owned_server_rounds_releases_request_state_on_finish(monkeypatch):
    # The server abandons finished generators (close fires only at GC), so
    # the wrapper stays suspended at its final yield. Its locals must not
    # pin the request state: at 32k depth the KV + captured hidden held
    # there is ~1.6 GB per request until process exit. On the terminal
    # token the wrapper nulls its heavy locals, so dropping the caller's
    # refs frees them immediately -- no gc pass, no close() needed.
    import weakref
    from types import SimpleNamespace
    from gmlx import speculative as spec

    def fake_rounds(model, drafter, lm, prompt_cache, **kwargs):
        for t in (11, 12, 13):
            yield t

    monkeypatch.setattr(spec, "_owned_decode_rounds", fake_rounds)

    class _Shared:
        pass

    drafter = _FakeDrafter([_kv_row(4, 300)])
    drafter.config = SimpleNamespace(block_size=4)
    entry = _kv_row(4, 0)
    shared = _Shared()
    ref_entry = weakref.ref(entry)
    ref_shared = weakref.ref(shared)

    gen = spec.owned_server_rounds(
        object(), drafter, [entry], mx.zeros((1, 4, D)),
        first_bonus=7, max_tokens=4, sampler=None,
        shared_kv_states=shared, prompt_tokens=[1, 2, 3],
        greedy_sampling=True)
    got = [next(gen)[0][0] for _ in range(3)]
    assert got == [11, 12, 13]

    del entry, shared, drafter  # generator deliberately NOT closed
    assert ref_entry() is None, (
        "abandoned finished generator still pins the request cache")
    assert ref_shared() is None, (
        "abandoned finished generator still pins the prefill shared-KV")
    gen.close()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
