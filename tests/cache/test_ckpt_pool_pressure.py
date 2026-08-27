"""Pool-pressure eviction: pinned ckpt records must yield blocks to new
stores.

Window chains are position-salted (no dedup), so records on a rotating
model consume fresh pool blocks per store and can exhaust the pool long
before the record-count or byte bounds trip. Without eviction at store
time the tier deadlocks: every store declines short, so insert-time
eviction never runs again - including for fresh trees after a history
rewrite. The depth harness's session phase caught exactly that on
gemma-4 12B (W=1024: 64 pinned blocks per record).
"""

from test_ckpt_tier import make_swa_cache

from gmlx.cache.apc_manager import GmlxAPCManager
from gmlx.cache.snapshot import _ckpt_records, ckpt_lookup, ckpt_store


def _ids(p):
    return list(range(300, 300 + p))


def test_store_evicts_deepest_own_chain_record_under_pool_pressure():
    # 10-block pool: rec48 pins 3 main + 2 window, rec64 shares the main
    # prefix and adds 1 main + 2 window (8 pinned). The p=80 store needs
    # 1 main + 2 window fresh blocks with only 2 free. Fat-chain
    # eviction (the 122B run-4 starvation fix) recycles the storing
    # chain's own deepest stale record: rec64 is superseded by the
    # incoming p=80, while the shallow rec48 keeps serving shorter
    # prefixes and branches (plain LRU evicted the oldest record and
    # starved cross-session adoption round-robin).
    man = GmlxAPCManager(num_blocks=10, block_size=16)
    assert ckpt_store(man, _ids(48), make_swa_cache(48, seed=1), extra_hash=0)
    assert ckpt_store(man, _ids(64), make_swa_cache(64, seed=2), extra_hash=0)
    assert ckpt_store(man, _ids(80), make_swa_cache(80, seed=3), extra_hash=0)
    snap = man.stats_snapshot()
    assert snap["ckpt_pool_evictions"] >= 1
    assert snap["ckpt_declines"].get("short_chain") is None
    assert snap["ckpt_evict_own_chain"] >= 1
    # deepest superseded record evicted; shallow and newest both live
    ps = {rec.p for rec in _ckpt_records(man).values()}
    assert 64 not in ps and 48 in ps and 80 in ps
    warm, got = ckpt_lookup(man, _ids(80) + [1, 2], extra_hash=0)
    assert got == 80


def test_fresh_tree_stores_after_exhaustion():
    # The session-phase compaction shape: the pool is pinned by one
    # conversation's records, then a rewritten history (disjoint ids)
    # must still be storable.
    man = GmlxAPCManager(num_blocks=12, block_size=16)
    assert ckpt_store(man, _ids(48), make_swa_cache(48, seed=1), extra_hash=0)
    assert ckpt_store(man, _ids(64), make_swa_cache(64, seed=2), extra_hash=0)
    other = list(range(900, 964))
    assert ckpt_store(man, other, make_swa_cache(64, seed=4), extra_hash=0)
    assert man.stats_snapshot()["ckpt_pool_evictions"] >= 1
    warm, got = ckpt_lookup(man, other + [1, 2], extra_hash=0)
    assert got == 64


def test_externally_pinned_pool_declines_without_drain():
    # Blocks pinned outside the record index (an in-flight lookup's
    # assembly refs, another request's live chain) do not free when a
    # record is released: an unreachable deficit must decline upfront
    # instead of draining the whole index for a store that still comes
    # up short.
    man = GmlxAPCManager(num_blocks=10, block_size=16)
    assert ckpt_store(man, _ids(48), make_swa_cache(48, seed=1), extra_hash=0)
    assert ckpt_store(man, _ids(64), make_swa_cache(64, seed=2), extra_hash=0)
    idx = _ckpt_records(man)
    pinned = []
    for rec in idx.values():
        for blocks in (rec.main_blocks, rec.bounded_blocks):
            for b in blocks:
                man._acquire_existing(b)
                pinned.append(b)
    assert not ckpt_store(man, _ids(80), make_swa_cache(80, seed=3),
                          extra_hash=0)
    snap = man.stats_snapshot()
    assert snap["ckpt_pool_evictions"] == 0
    assert snap["ckpt_declines"].get("short_chain", 0) >= 1
    assert {rec.p for rec in idx.values()} == {48, 64}
    # The external pins drop and the same store evicts normally.
    man.release(pinned)
    assert ckpt_store(man, _ids(80), make_swa_cache(80, seed=3), extra_hash=0)
    assert man.stats_snapshot()["ckpt_pool_evictions"] >= 1
    warm, got = ckpt_lookup(man, _ids(80) + [1, 2], extra_hash=0)
    assert got == 80


def test_oversized_store_declines_and_spares_records():
    # A span the pool can never hold declines upfront without draining
    # live records; a fitting store on a new tree then evicts normally.
    man = GmlxAPCManager(num_blocks=5, block_size=16)
    assert ckpt_store(man, _ids(48), make_swa_cache(48, seed=1), extra_hash=0)
    assert not ckpt_store(man, _ids(160), make_swa_cache(160, seed=2), extra_hash=0)
    snap = man.stats_snapshot()
    assert snap["ckpt_declines"].get("short_chain", 0) >= 1
    assert snap["ckpt_pool_evictions"] == 0
    assert [rec.p for rec in _ckpt_records(man).values()] == [48]
    other = list(range(900, 948))
    assert ckpt_store(man, other, make_swa_cache(48, seed=3), extra_hash=0)
    assert man.stats_snapshot()["ckpt_pool_evictions"] >= 1
    warm, got = ckpt_lookup(man, other + [1, 2], extra_hash=0)
    assert got == 48
