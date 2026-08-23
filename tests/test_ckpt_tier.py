"""Checkpoint tier: attn-KV blocks + recurrent-state sidecar for hybrids.

CPU-only unit tests against the real upstream APCManager. A checkpoint at
position p stores the attention layers' KV through the (salted) block pool
plus a small exact-tier sidecar entry carrying the recurrent states and a
1..block_size-token attention-KV tail; lookup reassembles a warm prompt
cache with KV offsets exactly p.
"""

import os
import subprocess
import sys
import time as _time
from types import SimpleNamespace

import mlx.core as mx
import pytest
from mlx_vlm.apc import APCManager, DiskBlockStore

from gmlx.cache_compat import runtime_cache_module
from gmlx.cache_snapshot import (
    _CKPT_SALT,
    _SIDECAR_SALT,
    ckpt_extra_hash,
    ckpt_lookup,
    ckpt_store,
    ckpt_supported,
    retirement_store,
    sidecar_extra_hash,
)

# Runtime-origin cache classes: the hybrid stacks these fixtures mimic are
# built by the mlx-vlm generate stack (vlm-origin classes since 0.6.4).
_cache = runtime_cache_module()
ArraysCache = _cache.ArraysCache
KVCache = _cache.KVCache
RotatingKVCache = _cache.RotatingKVCache

# 2 attention + 3 recurrent layers, interleaved like a real hybrid.
LAYOUT = ("kv", "arr", "arr", "kv", "arr")
H, D = 2, 8


def make_hybrid_cache(p, seed=0):
    caches = []
    for i, kind in enumerate(LAYOUT):
        mx.random.seed(seed * 1000 + i)
        if kind == "kv":
            c = KVCache()
            c.state = (
                mx.random.normal((1, H, p, D)),
                mx.random.normal((1, H, p, D)),
            )
        else:
            c = ArraysCache(size=2)
            c.cache = [
                mx.random.normal((1, 3, D)),
                mx.random.normal((1, H, D, D)),
            ]
        caches.append(c)
    return caches


def assert_warm_matches(warm, orig, p):
    assert len(warm) == len(orig)
    for w, o in zip(warm, orig):
        if isinstance(o, KVCache):
            assert isinstance(w, KVCache)
            assert int(w.offset) == p
            assert mx.array_equal(
                w.keys[..., :p, :], o.keys[..., :p, :]).item()
            assert mx.array_equal(
                w.values[..., :p, :], o.values[..., :p, :]).item()
        else:
            assert isinstance(w, ArraysCache)
            for a, b in zip(o.cache, w.cache):
                assert mx.array_equal(a, b).item()


def drain_disk(disk, timeout=30.0):
    """Block until the APC disk writer has published every queued shard.

    ``save_exact_cache`` only enqueues; a background writer thread indexes
    the shard later, and the lookup side (``find_exact_prefix``) reads the
    index with no in-flight wait. Any test that stores and then expects to
    read the result back must drain first, or it races the writer -- fast
    locally, lost on a loaded CI runner.
    """
    disk._q.join()
    deadline = _time.monotonic() + timeout
    while _time.monotonic() < deadline:
        with disk._in_flight_lock:
            if not disk._in_flight:
                return
        _time.sleep(0.005)
    raise AssertionError("APC disk writer did not drain within %.1fs" % timeout)


def test_ckpt_supported_shapes():
    assert ckpt_supported(make_hybrid_cache(4))
    assert not ckpt_supported([KVCache(), KVCache()])
    assert not ckpt_supported([ArraysCache(size=2), ArraysCache(size=2)])
    # off the block grid (8 % 16): geometry gate rejects
    assert not ckpt_supported([KVCache(), RotatingKVCache(max_size=8)])
    # grid-aligned sliding window: the three-way tier serves it now
    assert ckpt_supported([KVCache(), RotatingKVCache(max_size=32)])
    assert not ckpt_supported([])


def test_salts_are_distinct_keyspaces():
    assert _CKPT_SALT != _SIDECAR_SALT
    assert ckpt_extra_hash(0) != sidecar_extra_hash(0)
    assert ckpt_extra_hash(ckpt_extra_hash(7)) == 7


# p exercises the block/tail split: b_full = ((p-1)//16)*16, tail 1..16.
@pytest.mark.parametrize("p", [16, 17, 31, 32, 33, 48])
def test_store_lookup_roundtrip(p):
    man = APCManager(num_blocks=64, block_size=16)
    cache = make_hybrid_cache(p, seed=p)
    ids = list(range(100, 100 + p))
    assert ckpt_store(man, ids, cache, extra_hash=7)
    warm, got = ckpt_lookup(man, ids + [999, 998], extra_hash=7)
    assert got == p
    assert_warm_matches(warm, cache, p)
    # The checkpoint record pins its chain: main blocks stay ref-held
    # until the record is released (pin rather than repair).
    from gmlx.cache_snapshot import _ckpt_records, _release_record
    held = [b for b in man.pool if b.block_hash is not None]
    if p >= 16:
        assert held and all(b.ref_cnt == 1 for b in held)
    idx = _ckpt_records(man)
    for rec in list(idx.values()):
        _release_record(man, rec)
    idx.clear()
    assert all(b.ref_cnt == 0 for b in man.pool)


def _trim_kv(cache, p2):
    out = []
    for c in cache:
        if isinstance(c, KVCache):
            t = KVCache()
            t.state = (c.keys[..., :p2, :], c.values[..., :p2, :])
            out.append(t)
        else:
            out.append(c)
    return out


def test_identical_resend_needs_shorter_checkpoint():
    man = APCManager(num_blocks=64, block_size=16)
    p = 48
    cache = make_hybrid_cache(p)
    ids = list(range(100, 100 + p))
    assert ckpt_store(man, ids, cache, extra_hash=0)
    # The exact machinery never serves the final token: an identical
    # re-send cannot hit its own full-length checkpoint...
    warm, got = ckpt_lookup(man, ids, extra_hash=0)
    assert warm is None and got == 0
    # ...but hits a shorter one (the mid-prefill checkpoint_len store).
    p2 = 30
    trimmed = _trim_kv(cache, p2)
    assert ckpt_store(man, ids[:p2], trimmed, extra_hash=0)
    warm, got = ckpt_lookup(man, ids, extra_hash=0)
    assert got == p2
    assert_warm_matches(warm, trimmed, p2)


def test_replay_record_survives_retirement_insert():
    """The N-1 replay record exists for the identical resend, which
    arrives after retirement grows the chain -- exactly the moment
    strip-on-extend used to release it."""
    from gmlx.cache_snapshot import _ckpt_records

    man = APCManager(num_blocks=64, block_size=16)
    n = 48
    ids = list(range(100, 100 + n + 4))
    cache = make_hybrid_cache(n)
    replay = _trim_kv(cache, n - 1)
    assert ckpt_store(man, ids[:n - 1], replay, extra_hash=0,
                      kind="replay")
    # Post-prefill terminal store, then retirement at prompt+gen.
    assert ckpt_store(man, ids[:n], cache, extra_hash=0)
    assert retirement_store(man, "ckpt", ids, make_hybrid_cache(n + 4),
                            row=0, extra_hash=0)
    idx = _ckpt_records(man)
    assert (n - 1) in [r.p for r in idx.values()]
    # The terminal store is the chain's first restorable boundary (the
    # replay below it is gated), so it promotes to anchor.
    assert {r.kind for r in idx.values()} == {"replay", "anchor",
                                              "retire"}
    # The identical resend adopts at N-1.
    warm, got = ckpt_lookup(man, ids[:n], extra_hash=0)
    assert got == n - 1
    assert_warm_matches(warm, _trim_kv(cache, n - 1), n - 1)


def test_newer_replay_supersedes_older_on_chain():
    from gmlx.cache_snapshot import _ckpt_records

    man = APCManager(num_blocks=64, block_size=16)
    ids = list(range(100, 196))
    assert ckpt_store(man, ids[:47], make_hybrid_cache(47, seed=1),
                      extra_hash=0, kind="replay")
    assert ckpt_store(man, ids[:79], make_hybrid_cache(79, seed=2),
                      extra_hash=0, kind="replay")
    idx = _ckpt_records(man)
    assert [r.p for r in idx.values() if r.kind == "replay"] == [79]


def test_replay_record_not_exempt_from_entry_cap(monkeypatch):
    import gmlx.cache_snapshot as cs

    monkeypatch.setattr(cs, "_CKPT_RECORD_ENTRIES", 2)
    man = APCManager(num_blocks=64, block_size=16)
    assert ckpt_store(man, list(range(100, 147)),
                      make_hybrid_cache(47, seed=1), extra_hash=0,
                      kind="replay")
    # Unrelated chains (distinct extra_hash) push it out oldest-first.
    assert ckpt_store(man, list(range(300, 332)),
                      make_hybrid_cache(32, seed=2), extra_hash=1)
    assert ckpt_store(man, list(range(500, 532)),
                      make_hybrid_cache(32, seed=3), extra_hash=2)
    idx = cs._ckpt_records(man)
    assert len(idx) == 2
    assert all(r.kind != "replay" for r in idx.values())
    warm, got = ckpt_lookup(man, list(range(100, 148)), extra_hash=0)
    assert warm is None and got == 0


def test_replay_record_not_exempt_from_byte_budget(monkeypatch):
    import gmlx.cache_snapshot as cs

    man = APCManager(num_blocks=64, block_size=16)
    assert ckpt_store(man, list(range(100, 147)),
                      make_hybrid_cache(47, seed=1), extra_hash=0,
                      kind="replay")
    rec = next(iter(cs._ckpt_records(man).values()))
    monkeypatch.setattr(cs, "_CKPT_BUDGET_BYTES", rec.nbytes + 1)
    assert ckpt_store(man, list(range(300, 347)),
                      make_hybrid_cache(47, seed=2), extra_hash=1)
    idx = cs._ckpt_records(man)
    # The surviving boundary is its chain's first, hence anchor.
    assert [r.kind for r in idx.values()] == ["anchor"]


def test_salt_isolation_from_real_tiers():
    man = APCManager(num_blocks=64, block_size=16)
    p = 33
    cache = make_hybrid_cache(p)
    ids = list(range(100, 100 + p))
    assert ckpt_store(man, ids, cache, extra_hash=7)
    # Unsalted exact keyspace cannot see ckpt sidecars.
    entry, plen = man.lookup_exact_cache(ids + [1], extra_hash=7)
    assert entry is None and plen == 0
    # Unsalted block keyspace cannot see ckpt blocks.
    blocks, matched = man.lookup_prefix(ids[:32], extra_hash=7)
    man.release(blocks)
    assert matched == 0
    # A real full-cache exact entry does not satisfy a ckpt probe.
    ids2 = list(range(500, 500 + p))
    assert man.store_exact_cache(ids2, make_hybrid_cache(p, seed=2),
                                 extra_hash=7)
    warm, got = ckpt_lookup(man, ids2 + [1], extra_hash=7)
    assert warm is None and got == 0


ROT_LAYOUT = ("kv", "rot", "rot", "kv", "rot")
ROT_W = 32


def make_swa_cache(p, seed=0):
    """gemma-like sliding-window hybrid, fed as one concat chunk to p."""
    caches = []
    for i, kind in enumerate(ROT_LAYOUT):
        mx.random.seed(seed * 1000 + i)
        k = mx.random.normal((1, H, p, D))
        v = mx.random.normal((1, H, p, D))
        if kind == "kv":
            c = KVCache()
            c.state = (k, v)
        else:
            c = RotatingKVCache(max_size=ROT_W)
            c.update_and_fetch(k, v)
        caches.append(c)
    return caches


def assert_swa_warm_matches(warm, orig, p):
    from gmlx.cache_snapshot import (rotating_canonical_window,
                                     rotating_invariant)
    assert len(warm) == len(orig)
    for w, o in zip(warm, orig):
        if isinstance(o, RotatingKVCache):
            assert isinstance(w, RotatingKVCache)
            assert rotating_invariant(w) == (True, True)
            ko, vo, mo = rotating_canonical_window(o)
            kw, vw, mw = rotating_canonical_window(w)
            assert mo == mw
            assert mx.array_equal(ko, kw).item()
            assert mx.array_equal(vo, vw).item()
        else:
            assert int(w.offset) == p
            assert mx.array_equal(
                w.keys[..., :p, :], o.keys[..., :p, :]).item()


# Aligned below the wrap, then off-grid and aligned positions at and
# beyond W=32: a wrapped window is whole blocks at any p, so the store
# no longer waits for the grid.
@pytest.mark.parametrize("p", [16, 33, 40, 47, 48, 64, 65])
def test_swa_store_lookup_roundtrip(p):
    man = APCManager(num_blocks=64, block_size=16)
    cache = make_swa_cache(p, seed=p)
    ids = list(range(300, 300 + p))
    assert ckpt_store(man, ids, cache, extra_hash=9)
    warm, got = ckpt_lookup(man, ids + [1, 2], extra_hash=9)
    assert got == p
    assert_swa_warm_matches(warm, cache, p)


def test_swa_store_declines_off_grid_below_window():
    """Below the wrap there is no rot tail mechanism: an off-grid store
    would need a partial window block. Beyond it (see the roundtrip
    params) off-grid p stores. Without grid_truncate the store declines
    whole."""
    from gmlx.cache_snapshot import _ckpt_stats
    man = APCManager(num_blocks=64, block_size=16)
    p = 20                                    # < W=32 and 20 % 16 != 0
    cache = make_swa_cache(p)
    assert not ckpt_store(man, list(range(300, 300 + p)), cache)
    assert all(b.ref_cnt == 0 for b in man.pool)
    assert _ckpt_stats(man)["ckpt_declines"] == {"grid": 1}


def assert_grid_warm_matches(warm, orig, p):
    """warm at truncated p vs the deeper original: rot canonical is the
    temporal prefix [0..p), plain KV the same slice."""
    from gmlx.cache_snapshot import rotating_canonical_window
    assert len(warm) == len(orig)
    for w, o in zip(warm, orig):
        if isinstance(o, RotatingKVCache):
            kw, vw, mw = rotating_canonical_window(w)
            assert mw[2] == p and mw[3] == p
            assert mx.array_equal(kw, o.keys[..., :p, :]).item()
            assert mx.array_equal(vw, o.values[..., :p, :]).item()
        else:
            assert int(w.offset) == p
            assert mx.array_equal(
                w.keys[..., :p, :], o.keys[..., :p, :]).item()
            assert mx.array_equal(
                w.values[..., :p, :], o.values[..., :p, :]).item()


def test_grid_truncate_store_lookup_roundtrip():
    """grid_truncate turns the below-window decline into a terminal
    store at b_full; the record is a faithful shorter run."""
    import gmlx.cache_snapshot as cs
    man = APCManager(num_blocks=64, block_size=16)
    p = 20
    cache = make_swa_cache(p, seed=5)
    ids = list(range(300, 300 + p))
    assert ckpt_store(man, ids, cache, extra_hash=2,
                      grid_truncate=True) == 16
    st = cs._ckpt_stats(man)
    assert st["ckpt_grid_truncate"] == 1
    assert st["ckpt_declines"] == {}
    rec = next(iter(cs._ckpt_records(man).values()))
    assert rec.p == 16 and rec.b_full == 16
    assert rec.ids == tuple(ids[:16])
    warm, got = ckpt_lookup(man, ids + [1], extra_hash=2)
    assert got == 16
    assert_grid_warm_matches(warm, cache, 16)


def test_grid_truncate_sub_block_declines():
    """b_full < 2: nothing block-aligned to keep, decline as before."""
    from gmlx.cache_snapshot import _ckpt_stats
    man = APCManager(num_blocks=64, block_size=16)
    p = 10                                    # b_full = 0
    cache = make_swa_cache(p, seed=5)
    assert ckpt_store(man, list(range(300, 300 + p)), cache,
                      extra_hash=2, grid_truncate=True) == 0
    assert _ckpt_stats(man)["ckpt_declines"] == {"grid": 1}


def test_grid_truncate_recurrent_layout_declines():
    """State cannot rewind: an arr layer in the layout keeps the
    decline even with grid_truncate."""
    from gmlx.cache_snapshot import _ckpt_stats
    man = APCManager(num_blocks=64, block_size=16)
    p = 20
    cache = make_swa_cache(p, seed=5)
    arr = ArraysCache(size=2)
    arr.cache = [mx.random.normal((1, 3, D)),
                 mx.random.normal((1, H, D, D))]
    cache.append(arr)
    assert ckpt_store(man, list(range(300, 300 + p)), cache,
                      extra_hash=2, grid_truncate=True) == 0
    assert _ckpt_stats(man)["ckpt_declines"] == {"grid": 1}
    assert all(b.ref_cnt == 0 for b in man.pool)


def test_grid_truncate_beyond_window_stores_full():
    """At or beyond the wrap the off-grid store already works whole;
    grid_truncate must not truncate it."""
    from gmlx.cache_snapshot import _ckpt_stats
    man = APCManager(num_blocks=64, block_size=16)
    p = 40                                    # >= W=32, unaligned
    cache = make_swa_cache(p, seed=5)
    ids = list(range(300, 300 + p))
    assert ckpt_store(man, ids, cache, extra_hash=2,
                      grid_truncate=True) == p
    assert _ckpt_stats(man)["ckpt_grid_truncate"] == 0
    warm, got = ckpt_lookup(man, ids + [1], extra_hash=2)
    assert got == p
    assert_swa_warm_matches(warm, cache, p)


def test_retirement_rotating_short_prompt_stores_grid_prefix():
    """A below-window off-grid rotating retirement with no decode
    snapshot stores the block-grid prefix (grid_truncate), not nothing.
    No exact-tier fallback: the exact tier stays empty so the stock
    warm path never bypasses ckpt arming."""
    man = APCManager(num_blocks=64, block_size=16)
    p = 20                                    # < W and unaligned
    cache = make_swa_cache(p, seed=3)
    ids = list(range(300, 300 + p))
    assert retirement_store(man, "ckpt", ids, cache, row=0,
                            extra_hash=1) == 16
    warm, got = ckpt_lookup(man, ids + [1], extra_hash=1)
    assert got == 16
    assert_grid_warm_matches(warm, cache, 16)
    entry, plen = man.lookup_exact_cache(ids + [1], extra_hash=1)
    assert entry is None and plen == 0
    assert man.stats_snapshot()["exact_stores"] == 0


def test_retirement_rotating_off_grid_beyond_window_stores():
    man = APCManager(num_blocks=64, block_size=16)
    p = 40                                    # unaligned but >= W=32
    cache = make_swa_cache(p, seed=3)
    ids = list(range(300, 300 + p))
    assert retirement_store(man, "ckpt", ids, cache, row=0, extra_hash=1)
    warm, got = ckpt_lookup(man, ids + [1], extra_hash=1)
    assert got == p
    assert_swa_warm_matches(warm, cache, p)


def test_buffered_rotating_declines_loudly():
    """The spec path swaps rot layers to BufferedRotatingKVCache after
    prefill; its start_position/slack-buffer geometry has no canonical
    window yet, so stores and clones must decline (counted), never
    snapshot it silently wrong."""
    from mlx_vlm.models.cache import BufferedRotatingKVCache
    from gmlx.cache_snapshot import _ckpt_stats, _clone_single_row
    man = APCManager(num_blocks=64, block_size=16)
    p = 48
    cache = make_swa_cache(p, seed=6)
    buffered = [BufferedRotatingKVCache.from_cache(c, buffer_size=16)
                if isinstance(c, RotatingKVCache) else c for c in cache]
    ids = list(range(300, 300 + p))
    assert not ckpt_store(man, ids, buffered, extra_hash=0)
    assert _ckpt_stats(man)["ckpt_declines"] == {"buffered": 1}
    assert all(b.ref_cnt == 0 for b in man.pool)
    assert _clone_single_row(buffered[1]) is None


def test_lookup_pins_blocks_against_concurrent_release(monkeypatch):
    """A _record_insert on another thread can release a candidate's
    chains between selection and assembly; the lookup must pin them (+1
    ref under the lock) so assembly reads live tensors, and drop the pin
    on every exit path."""
    import gmlx.cache_snapshot as cs

    man = APCManager(num_blocks=64, block_size=16)
    p = 32
    cache = make_hybrid_cache(p, seed=21)
    ids = list(range(100, 100 + p))
    assert ckpt_store(man, ids, cache, extra_hash=0)
    idx = cs._ckpt_records(man)
    (key,) = idx.keys()

    real = cs._assemble_from_record

    def _release_then_assemble(manager, rec):
        # Simulate the concurrent insert's strip firing mid-lookup.
        with manager.lock:
            victim = idx.pop(key, None)
            if victim is not None:
                cs._release_record(manager, victim)
        return real(manager, rec)

    monkeypatch.setattr(cs, "_assemble_from_record", _release_then_assemble)
    warm, got = ckpt_lookup(man, ids + [1], extra_hash=0)
    assert got == p
    assert_warm_matches(warm, cache, p)      # tensors were still live
    # Record gone, pin dropped: every block back to refcount zero.
    assert not idx
    assert all(b.ref_cnt == 0 for b in man.pool)


def test_lookup_skips_record_released_after_selection(monkeypatch):
    """A candidate released after selection but before its turn in the
    walk is refused via the index membership check, never assembled from
    recycled blocks; surviving records keep their pins."""
    import gmlx.cache_snapshot as cs

    man = APCManager(num_blocks=64, block_size=16)
    ids = list(range(100, 196))
    short = make_hybrid_cache(32, seed=22)
    assert ckpt_store(man, ids[:32], short, extra_hash=0)
    assert ckpt_store(man, ids[:48], make_hybrid_cache(48, seed=23),
                      extra_hash=0)
    idx = cs._ckpt_records(man)
    short_key = (tuple(ids[:32]), 0)
    assert short_key in idx

    real = cs._assemble_from_record

    def _fail_deep_release_short(manager, rec):
        if rec.p == 48:
            with manager.lock:
                victim = idx.pop(short_key, None)
                if victim is not None:
                    cs._release_record(manager, victim)
            return None                      # deep candidate: assembly fails
        return real(manager, rec)

    monkeypatch.setattr(cs, "_assemble_from_record",
                        _fail_deep_release_short)
    warm, got = ckpt_lookup(man, ids[:60], extra_hash=0)
    assert warm is None and got == 0
    # The surviving p=48 record still pins exactly its own chain.
    assert [r.p for r in idx.values()] == [48]
    held = [b for b in man.pool if b.block_hash is not None]
    assert held and all(b.ref_cnt == 1 for b in held)
    monkeypatch.setattr(cs, "_assemble_from_record", real)
    warm, got = ckpt_lookup(man, ids[:60], extra_hash=0)
    assert got == 48


def test_lookup_survives_concurrent_clear(monkeypatch):
    """clear() while a lookup holds assembly pins: the pool free list is
    rebuilt under the held refs, so releasing them afterwards would push
    already-free blocks twice and corrupt the list. The generation check
    drops the refs instead and discards the pre-clear warm result."""
    import gmlx.cache_snapshot as cs
    from gmlx.apc_manager import GmlxAPCManager

    man = GmlxAPCManager(num_blocks=64, block_size=16)
    p = 32
    ids = list(range(100, 100 + p))
    assert ckpt_store(man, ids, make_hybrid_cache(p, seed=31), extra_hash=0)

    real = cs._assemble_from_record

    def _clear_then_assemble(manager, rec):
        manager.clear()
        return real(manager, rec)

    monkeypatch.setattr(cs, "_assemble_from_record", _clear_then_assemble)
    warm, got = ckpt_lookup(man, ids + [1], extra_hash=0)
    assert warm is None and got == 0
    # Free list intact: every pool block exactly once, all refs zero.
    seen = set()
    b = man._free_head
    while b is not None and len(seen) <= len(man.pool):
        assert id(b) not in seen
        seen.add(id(b))
        b = b.next
    assert len(seen) == len(man.pool)
    assert all(blk.ref_cnt == 0 for blk in man.pool)


def test_store_exception_after_insert_keeps_record_pinned(monkeypatch):
    """An exception after _record_insert transferred chain ownership must
    not release the blocks on the except path: the record is live in the
    index, so a second release would strip its pins while it serves."""
    import gmlx.cache_snapshot as cs

    man = APCManager(num_blocks=64, block_size=16)
    p = 32
    ids = list(range(100, 100 + p))

    real = cs._ckpt_bump

    def _boom(manager, key, n=1):
        if key == "ckpt_stores":
            raise RuntimeError("post-insert failure")
        return real(manager, key, n)

    monkeypatch.setattr(cs, "_ckpt_bump", _boom)
    assert not ckpt_store(man, ids, make_hybrid_cache(p, seed=41),
                          extra_hash=0)
    monkeypatch.setattr(cs, "_ckpt_bump", real)
    held = [b for b in man.pool if b.block_hash is not None]
    assert held and all(b.ref_cnt >= 1 for b in held)
    warm, got = ckpt_lookup(man, ids + [1], extra_hash=0)
    assert got == p


def test_pinning_survives_pool_pressure():
    man = APCManager(num_blocks=6, block_size=16)
    p = 32
    cache = make_hybrid_cache(p, seed=5)
    ids = list(range(300, 300 + p))
    assert ckpt_store(man, ids, cache, extra_hash=0)   # pins 2 blocks
    # Hammer the pool with unrelated stores until exhaustion.
    for s in range(4):
        other = make_hybrid_cache(64, seed=50 + s)
        lk = [c.keys for c in other if isinstance(c, KVCache)]
        lv = [c.values for c in other if isinstance(c, KVCache)]
        got = man.store_kv_blocks(list(range(1000 * s, 1000 * s + 64)),
                                  lk, lv)
        man.release(got)
    warm, got = ckpt_lookup(man, ids + [1], extra_hash=0)
    assert got == p                           # pinned chain survived
    assert_warm_matches(warm, cache, p)


def test_strip_on_extend_keeps_newest_two_plus_anchor():
    """A growing chain keeps the newest two records plus its anchor:
    the first restorable boundary is promoted and survives the strip
    (sibling fan-out adopts exactly that early prefix), while interior
    boundaries strip as before."""
    from gmlx.cache_snapshot import _ckpt_records
    man = APCManager(num_blocks=96, block_size=16)
    ids = list(range(400, 400 + 96))
    for p in (32, 48, 64, 80):
        cache = make_hybrid_cache(p, seed=p)
        assert ckpt_store(man, ids[:p], cache, extra_hash=0)
    idx = _ckpt_records(man)
    assert sorted(r.p for r in idx.values()) == [32, 64, 80]
    assert [r.p for r in idx.values() if r.kind == "anchor"] == [32]
    warm, got = ckpt_lookup(man, ids[:40], extra_hash=0)
    assert got == 32                          # the anchor serves siblings
    warm, got = ckpt_lookup(man, ids[:50], extra_hash=0)
    assert got == 32                          # p=48 stripped
    warm, got = ckpt_lookup(man, ids[:66], extra_hash=0)
    assert got == 64


def test_strip_on_extend_exempts_replay():
    """A growing chain strips boundary records past the cap but never a
    replay record; the replay stays adoptable through terminal and
    retirement inserts."""
    from gmlx.cache_snapshot import _ckpt_records
    man = APCManager(num_blocks=64, block_size=16)
    ids = list(range(400, 400 + 96))
    assert ckpt_store(man, ids[:32], make_hybrid_cache(32, seed=32),
                      extra_hash=0)
    assert ckpt_store(man, ids[:47], make_hybrid_cache(47, seed=47),
                      extra_hash=0, kind="replay")
    assert ckpt_store(man, ids[:64], make_hybrid_cache(64, seed=64),
                      extra_hash=0)
    assert ckpt_store(man, ids[:80], make_hybrid_cache(80, seed=80),
                      extra_hash=0, kind="retire")
    idx = _ckpt_records(man)
    # p=32 promoted to anchor (first restorable boundary), also exempt.
    assert sorted(r.p for r in idx.values()) == [32, 47, 64, 80]
    warm, got = ckpt_lookup(man, ids[:48], extra_hash=0)
    assert got == 47


def test_replay_gate_arr_adopts_exact_resend_only():
    """A recurrent-state replay record serves the identical resend
    (query = record + 1 token) and refuses longer suffixes -- the
    refusal is reason-counted and feeds missed-adoption accounting."""
    import gmlx.cache_snapshot as cs
    man = APCManager(num_blocks=64, block_size=16)
    ids = list(range(400, 432))
    assert ckpt_store(man, ids, make_hybrid_cache(32, seed=5),
                      extra_hash=0, kind="replay")
    warm, got = ckpt_lookup(man, ids + [1], extra_hash=0)
    assert got == 32 and warm is not None
    warm, got = ckpt_lookup(man, ids + [1, 2, 3], extra_hash=0)
    assert warm is None and got == 0
    st = cs.ckpt_stats_snapshot(man)
    assert st["ckpt_declines"] == {"replay_gate": 1}
    assert st["ckpt_missed_adoptions"] == 1


def test_replay_gate_rot_only_adopts_freely():
    """Attention-only replay records split exactly: any longer query
    adopts, same as a boundary record."""
    man = APCManager(num_blocks=64, block_size=16)
    ids = list(range(800, 848))                # p=48 >= W=32
    cache = make_swa_cache(48, seed=9)
    assert ckpt_store(man, ids, cache, extra_hash=0, kind="replay")
    warm, got = ckpt_lookup(man, ids + list(range(60, 70)), extra_hash=0)
    assert got == 48
    assert_swa_warm_matches(warm, cache, 48)


def test_replay_arr_skeleton_forced_off(tmp_path):
    """The disk path knows no kinds, so an arr replay skeleton would
    serve a restart past the adopt gate; ckpt_store forces it off even
    when the caller asks for one. Rot-only replay keeps its skeleton
    (free adoption makes the disk hit equivalent)."""
    import gmlx.cache_snapshot as cs
    disk = DiskBlockStore(root=tmp_path, namespace="m")
    man = APCManager(num_blocks=64, block_size=16, disk=disk)
    ids = list(range(100, 132))
    assert ckpt_store(man, ids, make_hybrid_cache(32, seed=6),
                      extra_hash=0, kind="replay", skeleton_disk=True)
    assert cs.ckpt_stats_snapshot(man)["ckpt_skeleton_writes"] == 0
    rot_ids = list(range(200, 248))
    assert ckpt_store(man, rot_ids, make_swa_cache(48, seed=7),
                      extra_hash=0, kind="replay", skeleton_disk=True)
    assert cs.ckpt_stats_snapshot(man)["ckpt_skeleton_writes"] == 1
    disk.close()


def test_post_prefill_store_records_boundary():
    """The spec-path p=N store appends to the settled variable the
    sidecar key set and the Stage 6 drop gate read; a declined store
    appends nothing."""
    from gmlx.speculative import _ckpt_post_prefill
    man = APCManager(num_blocks=64, block_size=16)
    ids = list(range(100, 132))
    model = SimpleNamespace(_kq_apc_manager=man)
    meta = {}
    _ckpt_post_prefill(model, make_hybrid_cache(32, seed=13),
                       {"full_ids": ids, "extra_hash": 0, "apc_meta": meta})
    assert meta["ckpt_stored_boundaries"] == [32]
    meta2 = {}
    _ckpt_post_prefill(model, make_hybrid_cache(40, seed=14),
                       {"full_ids": ids, "extra_hash": 0, "apc_meta": meta2})
    assert meta2 == {}


def test_full_store_drop_gate():
    """p=N drops only when an armed render-stable boundary LANDED --
    armed-then-declined keeps it (nothing else would cover turn 2's
    prefix class)."""
    from gmlx.cache_snapshot import ckpt_full_store_redundant
    assert not ckpt_full_store_redundant(None)
    assert not ckpt_full_store_redundant({})
    assert not ckpt_full_store_redundant(
        {"ckpt_p_stable_bounds": [2048], "ckpt_stored_boundaries": [4096]})
    assert ckpt_full_store_redundant(
        {"ckpt_p_stable_bounds": [2048, 2100],
         "ckpt_stored_boundaries": [2048, 4096]})


def test_post_prefill_store_dropped_when_p_stable_landed():
    from gmlx.cache_snapshot import _ckpt_records
    from gmlx.speculative import _ckpt_post_prefill
    man = APCManager(num_blocks=64, block_size=16)
    ids = list(range(100, 132))
    model = SimpleNamespace(_kq_apc_manager=man)
    meta = {"ckpt_p_stable_bounds": [16], "ckpt_stored_boundaries": [16]}
    _ckpt_post_prefill(model, make_hybrid_cache(32, seed=15),
                       {"full_ids": ids, "extra_hash": 0, "apc_meta": meta})
    assert meta["ckpt_stored_boundaries"] == [16]   # no p=N append
    assert len(_ckpt_records(man)) == 0             # no p=N record


def test_rot_clone_canonicalizes():
    """_clone_single_row trims a wrapped rotating buffer to the canonical
    window (min(offset, W) columns, ring pointer at the end) while
    preserving the concrete class; content matches the live canonical
    window bit-exactly. Below the wrap the partial fill clones whole."""
    from gmlx.cache_snapshot import (
        _clone_single_row,
        rotating_canonical_window,
        rotating_invariant,
    )
    src = next(c for c in make_swa_cache(96, seed=8)
               if isinstance(c, RotatingKVCache))
    assert src.keys.shape[2] > ROT_W          # raw ring is untrimmed
    out = _clone_single_row(src)
    assert type(out) is type(src)
    assert out.keys.shape[2] == ROT_W         # canonical: min(96, 32)
    assert rotating_invariant(out) == (True, True)
    ko, vo, mo = rotating_canonical_window(src)
    kw, vw, mw = rotating_canonical_window(out)
    assert mo == mw
    assert mx.array_equal(ko, kw).item() and mx.array_equal(vo, vw).item()
    src2 = next(c for c in make_swa_cache(20, seed=9)
                if isinstance(c, RotatingKVCache))
    out2 = _clone_single_row(src2)
    assert out2.keys.shape[2] == 20 and int(out2.offset) == 20


def test_layout_signature_rejects_mismatch():
    man = APCManager(num_blocks=64, block_size=16)
    p = 32
    cache = make_hybrid_cache(p, seed=2)
    ids = list(range(300, 300 + p))
    assert ckpt_store(man, ids, cache, extra_hash=0)
    warm, got = ckpt_lookup(man, ids + [1], extra_hash=0,
                            layout=("kv", "arr", "arr", "kv", "arr"))
    assert got == p
    warm, got = ckpt_lookup(man, ids + [1], extra_hash=0,
                            layout=("kv", "rot:32:0", "rot:32:0", "kv",
                                    "rot:32:0"))
    assert warm is None and got == 0


@pytest.mark.parametrize("p", [33, 47, 48, 65])
def test_swa_disk_restart_roundtrip(tmp_path, p):
    """Production-manager restart repair. GmlxAPCManager routes chain
    stores through store_ckpt_blocks, whose disk kwarg gates window-chain
    persistence by kind; the stock manager's fallback disk-writes every
    chain, and testing on it hid a live bug where rot window chains never
    reached disk at all."""
    from gmlx.apc_manager import GmlxAPCManager
    cache = make_swa_cache(p, seed=11)
    ids = list(range(700, 700 + p))
    disk = DiskBlockStore(root=tmp_path, namespace="m")
    man = GmlxAPCManager(num_blocks=64, block_size=16, disk=disk)
    assert ckpt_store(man, ids, cache, extra_hash=4, kind="replay")
    disk.close()
    disk2 = DiskBlockStore(root=tmp_path, namespace="m")
    man2 = GmlxAPCManager(num_blocks=64, block_size=16, disk=disk2)
    try:
        warm, got = ckpt_lookup(man2, ids + [77], extra_hash=4)
        assert got == p
        assert_swa_warm_matches(warm, cache, p)
        # Disk half of the geometry check: the stored entries carry the
        # writer's window, so a reader with a different one must miss.
        from gmlx.cache_snapshot import ckpt_layout
        live = tuple(ckpt_layout(cache, 16))
        other = tuple(t if not t.startswith("rot") else "rot:64:0"
                      for t in live)
        warm, got = ckpt_lookup(man2, ids + [77], extra_hash=4,
                                layout=other)
        assert warm is None and got == 0
    finally:
        disk2.close()


def test_mlx_lm_class_caches_roundtrip():
    """gmlx text models carry mlx_lm cache classes; the tier must clone
    them (upstream's clone isinstance-gates on the mlx_vlm twins)."""
    from mlx_lm.models.cache import ArraysCache as LmArrays
    from mlx_lm.models.cache import KVCache as LmKV
    man = APCManager(num_blocks=64, block_size=16)
    p = 33                                    # unaligned: tail clone too
    caches = []
    for i, kind in enumerate(LAYOUT):
        mx.random.seed(900 + i)
        if kind == "kv":
            c = LmKV()
            c.state = (mx.random.normal((1, H, p, D)),
                       mx.random.normal((1, H, p, D)))
        else:
            c = LmArrays(2)
            c.cache[0] = mx.random.normal((1, 3, 8))
            c.cache[1] = mx.random.normal((1, 4, D, D))
        caches.append(c)
    ids = list(range(300, 300 + p))
    assert ckpt_store(man, ids, caches, extra_hash=0)
    warm, got = ckpt_lookup(man, ids + [1], extra_hash=0)
    assert got == p
    for kind, w, o in zip(LAYOUT, warm, caches):
        if kind == "kv":
            assert int(w.offset) == p
            assert mx.array_equal(w.keys[..., :p, :], o.keys).item()
            assert mx.array_equal(w.values[..., :p, :], o.values).item()
        else:
            for ws, os_ in zip(w.cache, o.cache):
                assert mx.array_equal(ws, os_).item()


def test_layout_geometry_rejects_window_mismatch():
    """Same tag kinds, different window: the disk path must miss rather
    than restore the writer's geometry into the model."""
    man = APCManager(num_blocks=64, block_size=16)
    p = 48
    cache = make_swa_cache(p, seed=7)
    ids = list(range(300, 300 + p))
    assert ckpt_store(man, ids, cache, extra_hash=0)
    from gmlx.cache_snapshot import ckpt_layout
    live = tuple(ckpt_layout(cache, 16))
    other = tuple(t if not t.startswith("rot") else "rot:64:0"
                  for t in live)
    warm, got = ckpt_lookup(man, ids + [1], extra_hash=0, layout=live)
    assert got == p
    warm, got = ckpt_lookup(man, ids + [1], extra_hash=0, layout=other)
    assert warm is None and got == 0


def test_short_main_chain_declines_and_spares_chain():
    """A store that cannot pin its main chain is declined outright: an
    unpinnable record would displace restorable ones via strip-on-extend."""
    from gmlx.cache_snapshot import _ckpt_records
    man = APCManager(num_blocks=2, block_size=16)
    ids = list(range(400, 400 + 96))
    good = make_hybrid_cache(32, seed=1)
    assert ckpt_store(man, ids[:32], good, extra_hash=0)   # pins both blocks
    assert not ckpt_store(man, ids[:64], make_hybrid_cache(64, seed=2),
                          extra_hash=0)
    assert not ckpt_store(man, ids[:96], make_hybrid_cache(96, seed=3),
                          extra_hash=0)
    idx = _ckpt_records(man)
    assert [r.p for r in idx.values()] == [32]
    warm, got = ckpt_lookup(man, ids[:40], extra_hash=0)
    assert got == 32
    assert_warm_matches(warm, good, 32)


def test_ckpt_store_suppresses_layer_major():
    """Above the layer-major threshold the stock store returns no blocks
    and clones the prefix into the 2-slot exact LRU; the gmlx manager's
    ckpt path must stay per-block and leave the exact LRU alone."""
    from gmlx.apc_manager import GmlxAPCManager
    man = GmlxAPCManager(num_blocks=64, block_size=16)
    man._layer_major_memory_min_tokens = 32
    p = 64
    cache = make_hybrid_cache(p, seed=4)
    ids = list(range(500, 500 + p))
    assert ckpt_store(man, ids, cache, extra_hash=0)
    assert man.stats_snapshot()["exact_stores"] == 0
    warm, got = ckpt_lookup(man, ids + [1], extra_hash=0)
    assert got == p
    assert_warm_matches(warm, cache, p)
    # The stock path keeps its layer-major behavior.
    lk = [c.keys for c in cache if isinstance(c, KVCache)]
    lv = [c.values for c in cache if isinstance(c, KVCache)]
    got_blocks = man.store_kv_blocks(ids, lk, lv, extra_hash=9)
    assert got_blocks == []
    assert man.stats_snapshot()["exact_stores"] == 1


def test_stripped_boundary_recovers_via_skeleton(tmp_path):
    """Strip-on-extend releases a boundary record's pin, but its blocks
    stay indexed in the pool until reused and its skeleton stays on
    disk: a later shared-prefix lookup must re-assemble the record from
    skeleton + memory blocks. This is the divergent-suffix recovery
    path -- suppressing boundary skeletons broke it live (2d matrix,
    gemma-4: divergent/turn adoption fell to +0)."""
    from gmlx.apc_manager import GmlxAPCManager
    from gmlx.cache_snapshot import _ckpt_records, _release_record
    disk = DiskBlockStore(root=tmp_path, namespace="m")
    man = GmlxAPCManager(num_blocks=64, block_size=16, disk=disk)
    try:
        p = 48
        cache = make_swa_cache(p, seed=12)
        ids = list(range(300, 300 + p))
        assert ckpt_store(man, ids, cache, extra_hash=0)
        # Recovery reads the skeleton off disk, so the async writer has to
        # have published it before the lookup runs.
        drain_disk(disk)
        idx = _ckpt_records(man)
        (key, rec), = list(idx.items())
        _release_record(man, idx.pop(key))
        warm, got = ckpt_lookup(man, ids + [77], extra_hash=0)
        assert got == p
        assert_swa_warm_matches(warm, cache, p)
    finally:
        disk.close()


def test_window_chain_disk_follows_kind(tmp_path):
    """Position-salted window shards cannot dedup on disk, so they earn
    persistence only where restart repair reads them: replay and retire
    records. A boundary store keeps its window chain memory-only but
    still writes its skeleton -- within the process the skeleton
    re-indexes a record whose blocks survived strip-on-extend; the main
    chain writes through regardless (it dedups)."""
    from gmlx.apc_manager import GmlxAPCManager
    disk = DiskBlockStore(root=tmp_path, namespace="m")
    man = GmlxAPCManager(num_blocks=64, block_size=16, disk=disk)
    try:
        p = 48
        cache = make_swa_cache(p, seed=5)
        ids = list(range(600, 600 + p))
        assert ckpt_store(man, ids, cache, extra_hash=0)
        # boundary: 3 main blocks + the skeleton -- no window shards
        assert man.stats_snapshot()["disk_writes"] == 4
        assert man.stats_snapshot()["ckpt_skeleton_writes"] == 1
        cache2 = make_swa_cache(p, seed=6)
        ids2 = list(range(800, 800 + p))
        assert ckpt_store(man, ids2, cache2, extra_hash=0, kind="replay")
        # replay: 3 main + 2 window blocks + the skeleton entry
        assert man.stats_snapshot()["disk_writes"] == 10
        assert man.stats_snapshot()["ckpt_skeleton_writes"] == 2
    finally:
        disk.close()


def test_incomplete_block_chain_is_miss():
    man = APCManager(num_blocks=64, block_size=16)
    p = 48
    b_full = 32
    ids = list(range(100, 100 + p))
    # Sidecar present, blocks never stored (evicted-chain stand-in).
    sidecar = []
    for kind in LAYOUT:
        if kind == "kv":
            t = KVCache()
            t.state = (mx.random.normal((1, H, p - b_full, D)),
                       mx.random.normal((1, H, p - b_full, D)))
            sidecar.append(t)
        else:
            a = ArraysCache(size=2)
            a.cache = [mx.zeros((1, 3, D)), mx.zeros((1, H, D, D))]
            sidecar.append(a)
    assert man.store_exact_cache(ids, sidecar,
                                 extra_hash=ckpt_extra_hash(0))
    warm, got = ckpt_lookup(man, ids + [1], extra_hash=0)
    assert warm is None and got == 0
    assert all(b.ref_cnt == 0 for b in man.pool)


def test_offset_guard_skips_unfaithful_store():
    man = APCManager(num_blocks=64, block_size=16)
    p = 33
    cache = make_hybrid_cache(p)
    assert not ckpt_store(man, list(range(p + 1)), cache, extra_hash=0)
    assert not ckpt_store(man, list(range(p - 1)), cache, extra_hash=0)
    entry, plen = man.lookup_exact_cache(
        list(range(p + 2)), extra_hash=ckpt_extra_hash(0))
    assert entry is None and plen == 0


def test_lookup_returns_decoupled_clones():
    man = APCManager(num_blocks=64, block_size=16)
    p = 33
    cache = make_hybrid_cache(p)
    ids = list(range(100, 100 + p))
    assert ckpt_store(man, ids, cache, extra_hash=0)
    warm1, _ = ckpt_lookup(man, ids + [1], extra_hash=0)
    # Mutate the first warm copy in place.
    for c in warm1:
        if isinstance(c, KVCache):
            c.keys[..., 0:1, 0:1] = 12345.0
        else:
            c.cache[0][..., 0:1] = 12345.0
    warm2, got = ckpt_lookup(man, ids + [1], extra_hash=0)
    assert got == p
    assert_warm_matches(warm2, cache, p)


def test_retirement_store_ckpt_branch():
    man = APCManager(num_blocks=64, block_size=16)
    p = 40
    cache = make_hybrid_cache(p)
    ids = list(range(100, 100 + p))
    assert retirement_store(man, "ckpt", ids, cache, row=0, extra_hash=3)
    warm, got = ckpt_lookup(man, ids + [5], extra_hash=3)
    assert got == p
    assert_warm_matches(warm, cache, p)


def test_min_prefix_gate():
    man = APCManager(num_blocks=64, block_size=16)
    p = 33
    cache = make_hybrid_cache(p)
    ids = list(range(100, 100 + p))
    assert ckpt_store(man, ids, cache, extra_hash=0)
    warm, got = ckpt_lookup(man, ids + [1], extra_hash=0,
                            min_prefix_tokens=p)
    assert warm is None and got == 0
    warm, got = ckpt_lookup(man, ids + [1], extra_hash=0,
                            min_prefix_tokens=p - 1)
    assert got == p


def test_disk_restart_roundtrip(tmp_path):
    p = 33
    cache = make_hybrid_cache(p, seed=9)
    ids = list(range(7, 7 + p))
    disk = DiskBlockStore(root=tmp_path, namespace="m")
    man = APCManager(num_blocks=64, block_size=16, disk=disk)
    assert ckpt_store(man, ids, cache, extra_hash=3)
    disk.close()
    # Fresh manager + store over the same root: index rebuild, then the
    # sidecar promotes from the exact shard and the blocks restore from
    # the layer-major shard.
    disk2 = DiskBlockStore(root=tmp_path, namespace="m")
    man2 = APCManager(num_blocks=64, block_size=16, disk=disk2)
    try:
        warm, got = ckpt_lookup(man2, ids + [77], extra_hash=3)
        assert got == p
        assert_warm_matches(warm, cache, p)
    finally:
        disk2.close()


class _HybridModel:
    def make_cache(self):
        return [KVCache(), ArraysCache(size=2)]


def test_ckpt_active_gating(monkeypatch):
    import gmlx.spec_engine as se

    monkeypatch.setattr(se, "_SPEC_APC_CKPT_DISABLED", False)
    assert se._ckpt_active(_HybridModel(), "exact") is True
    assert se._ckpt_active(_HybridModel(), "block") is False
    assert se._ckpt_active(_HybridModel(), None) is False
    rot = _HybridModel()
    rot.make_cache = lambda: [KVCache(), RotatingKVCache(max_size=8)]
    assert se._ckpt_active(rot, "exact") is False
    # Kill switch wins over the cached shape verdict.
    warm_model = _HybridModel()
    assert se._ckpt_active(warm_model, "exact") is True
    monkeypatch.setattr(se, "_SPEC_APC_CKPT_DISABLED", True)
    assert se._ckpt_active(warm_model, "exact") is False


def test_mid_prefill_store_supersedes_stock(monkeypatch):
    import gmlx.spec_engine as se

    man = APCManager(num_blocks=64, block_size=16)
    ckpt_len = 32
    ids = list(range(200, 248))
    cache = make_hybrid_cache(ckpt_len)
    batch = SimpleNamespace(
        _kq_ckpt_armed=True,
        _apc_manager=man,
        _apc_meta=[{
            "full_input_ids": ids,
            "checkpoint_len": ckpt_len,
            "extra_hash": 5,
            "prefix_len": 0,
        }],
        prompt_cache=cache,
        _row_real_tokens_processed=lambda idx: ckpt_len,
    )
    se._ckpt_mid_prefill_store(batch)
    # checkpoint_done set: the stock exact-clone store is now a no-op.
    assert batch._apc_meta[0]["checkpoint_done"] is True
    warm, got = ckpt_lookup(man, ids, extra_hash=5)
    assert got == ckpt_len
    assert_warm_matches(warm, cache, ckpt_len)
    # Not armed -> untouched.
    batch2 = SimpleNamespace(
        _apc_meta=[{"checkpoint_len": ckpt_len, "extra_hash": 5,
                    "full_input_ids": ids, "prefix_len": 0}],
        _apc_manager=man,
        prompt_cache=cache,
        _row_real_tokens_processed=lambda idx: ckpt_len,
    )
    se._ckpt_mid_prefill_store(batch2)
    assert "checkpoint_done" not in batch2._apc_meta[0]


def test_master_kill_switch_cascades():
    """GMLX_SPEC_APC=0 must disable every layer at import time: L0/L1 and
    the derived flags in spec_engine, plus engine.speculative's independent
    env read. Fresh interpreter because the flags burn at import."""
    code = (
        "import gmlx.spec_engine as se;"
        "import gmlx.speculative as sp;"
        "assert se._SPEC_APC_DISABLED;"
        "assert se._SPEC_APC_RETIRE_DISABLED;"
        "assert se._SPEC_APC_SIDECAR_DISABLED;"
        "assert se._SPEC_APC_CKPT_DISABLED;"
        "assert sp._SIDECAR_DISABLED"
    )
    env = dict(os.environ, GMLX_SPEC_APC="0")
    for sub in ("RETIRE", "SIDECAR", "CKPT"):
        env.pop(f"GMLX_SPEC_APC_{sub}", None)
    proc = subprocess.run(
        [sys.executable, "-c", code], env=env,
        capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr[-2000:]


def test_spec_apc_master_disable_noops_store(monkeypatch):
    """docs/server-config.md: GMLX_SPEC_APC=0 turns every speculative cache
    layer off at once. Behavioral sibling of the flag-cascade test: drive the
    real owned-prefill APC entrypoint against a real APCManager with the master
    switch off and assert nothing is armed or stored; the switched-on control
    proves the same drive does store."""
    import gmlx.spec_engine as se

    p = 48
    ids = mx.array([list(range(100, 100 + p))])

    def drive(disabled):
        for flag in ("_SPEC_APC_DISABLED", "_SPEC_APC_RETIRE_DISABLED",
                     "_SPEC_APC_SIDECAR_DISABLED", "_SPEC_APC_CKPT_DISABLED"):
            monkeypatch.setattr(se, flag, disabled)
        se._bind_l1_view()
        man = APCManager(num_blocks=64, block_size=16)
        model = SimpleNamespace(
            _kq_apc_manager=man, _kq_apc_mode="exact",
            config=SimpleNamespace(),
            make_cache=lambda: [KVCache(), ArraysCache(size=2)])
        batch = SimpleNamespace(
            model=model, _input_ids=ids, _inputs_embeds=mx.zeros((1, p, 4)),
            prompt_cache=make_hybrid_cache(p), _prompt_kwargs={})
        se._mtp_prefill_init(batch)
        # The mid-prefill checkpoint moment fires either way; only an armed
        # batch stores.
        meta = (getattr(batch, "_apc_meta", None) or [{}])[0] or {}
        cl = int(meta.get("checkpoint_len") or 0)
        if cl:
            batch.prompt_cache = make_hybrid_cache(cl)
            batch._row_real_tokens_processed = lambda idx: cl
        se._ckpt_mid_prefill_store(batch)
        return man, model, batch

    man_on, _model, batch_on = drive(disabled=False)     # switched-on control
    assert getattr(batch_on, "_kq_ckpt_armed", False)
    assert man_on.stats_snapshot()["stores"] > 0

    man_off, model_off, batch_off = drive(disabled=True)  # master switch off
    snap = man_off.stats_snapshot()
    assert snap["stores"] == 0 and snap["exact_stores"] == 0
    assert not hasattr(batch_off, "_apc_manager")        # stock store never armed
    assert not hasattr(batch_off.prompt_cache[0], "_kq_apc_retire")
    assert se._get_spec_prefix_cache(model_off) is None  # L0 off too


# -- anchor records: the sibling fan-out exemption --

def test_anchor_kind_exempt_from_strip_and_superseded():
    """A tagged anchor survives strip-on-extend as the chain deepens; a
    newer tagged anchor on the same chain supersedes it (one anchor per
    chain)."""
    from gmlx.cache_snapshot import _ckpt_records

    man = APCManager(num_blocks=96, block_size=16)
    ids = list(range(400, 400 + 96))
    assert ckpt_store(man, ids[:32], make_hybrid_cache(32, seed=32),
                      extra_hash=0, kind="anchor")
    for p in (48, 64, 80):
        assert ckpt_store(man, ids[:p], make_hybrid_cache(p, seed=p),
                          extra_hash=0)
    idx = _ckpt_records(man)
    assert sorted(r.p for r in idx.values()) == [32, 64, 80]
    assert [r.p for r in idx.values() if r.kind == "anchor"] == [32]
    # Deeper records exist below the new anchor position: no promotion
    # happened at 48/64/80 (the chain was never fresh).
    assert ckpt_store(man, ids[:48], make_hybrid_cache(48, seed=1),
                      extra_hash=0, kind="anchor")
    idx = _ckpt_records(man)
    assert [r.p for r in idx.values() if r.kind == "anchor"] == [48]
    assert 32 not in [r.p for r in idx.values()]


def test_anchor_evicts_after_non_anchors_lru_by_hit(monkeypatch):
    """Entry-cap pressure: non-anchors go first even when an anchor is
    older; among anchors the least-recently-hit goes first."""
    import gmlx.cache_snapshot as cs

    monkeypatch.setattr(cs, "_CKPT_RECORD_ENTRIES", 3)
    man = APCManager(num_blocks=96, block_size=16)
    a = list(range(100, 148))
    b = list(range(300, 364))
    assert ckpt_store(man, a[:32], make_hybrid_cache(32, seed=1),
                      extra_hash=0)                       # anchor A
    assert ckpt_store(man, b[:32], make_hybrid_cache(32, seed=2),
                      extra_hash=1)                       # anchor B
    assert ckpt_store(man, b[:48], make_hybrid_cache(48, seed=3),
                      extra_hash=1)                       # plain boundary
    warm, got = ckpt_lookup(man, a[:40], extra_hash=0)    # hit refreshes A
    assert got == 32
    assert ckpt_store(man, list(range(500, 532)),
                      make_hybrid_cache(32, seed=4), extra_hash=2)
    idx = cs._ckpt_records(man)
    # The plain boundary (B:48) went first despite being newer than both
    # anchors.
    assert [(r.p, r.extra_hash) for r in idx.values() if r.kind != "anchor"] \
        == []
    assert {r.extra_hash for r in idx.values()} == {0, 1, 2}
    assert ckpt_store(man, list(range(700, 732)),
                      make_hybrid_cache(32, seed=5), extra_hash=3)
    idx = cs._ckpt_records(man)
    # All anchors now: the least-recently-hit one (B) went; the hit A
    # record stayed.
    assert {r.extra_hash for r in idx.values()} == {0, 2, 3}


def test_first_boundary_promotion_skips_retire_chains():
    """Promotion targets boundaries only: a chain whose first record is
    a retirement store gets no anchor from it, and a later boundary
    above it does not promote either (the chain is not fresh)."""
    from gmlx.cache_snapshot import _ckpt_records

    man = APCManager(num_blocks=96, block_size=16)
    ids = list(range(400, 400 + 96))
    assert ckpt_store(man, ids[:32], make_hybrid_cache(32, seed=1),
                      extra_hash=0, kind="retire")
    assert ckpt_store(man, ids[:64], make_hybrid_cache(64, seed=2),
                      extra_hash=0)
    idx = _ckpt_records(man)
    assert sorted((r.p, r.kind) for r in idx.values()) == \
        [(32, "retire"), (64, "boundary")]


def test_anchor_never_shadows_a_deeper_disk_skeleton(tmp_path):
    """Depth beats retention: with only the anchor pinned in memory and a
    deeper skeleton on disk, the lookup must return the disk depth. The
    pinned walk returns on first success, so an anchor left to win here
    caps every divergent query at its own p (the depth e2e's divergent
    and turns floors)."""
    from gmlx.apc_manager import GmlxAPCManager
    from gmlx.cache_snapshot import _ckpt_records, rotating_canonical_window

    ids = list(range(700, 700 + 96))
    disk = DiskBlockStore(root=tmp_path, namespace="m")
    man = GmlxAPCManager(num_blocks=96, block_size=16, disk=disk)
    try:
        deep = make_swa_cache(64, seed=11)
        # The shallow store must carry the same KV as the deep one's
        # prefix: the block pool dedups the shared chain by token hash,
        # so mismatched fixture content would be a fixture artifact.
        shallow = []
        for c in deep:
            k, v = rotating_canonical_window(c)[:2] \
                if isinstance(c, RotatingKVCache) else c.state
            if isinstance(c, RotatingKVCache):
                s = RotatingKVCache(max_size=ROT_W)
                s.update_and_fetch(k[..., :32, :], v[..., :32, :])
            else:
                s = KVCache()
                s.state = (k[..., :32, :], v[..., :32, :])
            shallow.append(s)
        assert ckpt_store(man, ids[:32], shallow,
                          extra_hash=4, kind="anchor")
        assert ckpt_store(man, ids[:64], deep, extra_hash=4)
        # The lookup below reads the deep skeleton off disk, so the async
        # writer has to have published it first.
        drain_disk(disk)
        # Drop the deep record from memory, keeping its disk skeleton:
        # exactly what strip-on-extend leaves behind as a chain deepens.
        idx = _ckpt_records(man)
        for k, r in list(idx.items()):
            if r.p == 64:
                idx.pop(k)
        assert [r.kind for r in idx.values()] == ["anchor"]
        warm, got = ckpt_lookup(man, ids + [77], extra_hash=4)
        assert got == 64
        assert_swa_warm_matches(warm, deep, 64)
        # Nothing deeper on disk: the anchor still serves.
        warm, got = ckpt_lookup(man, ids[:48] + [77], extra_hash=4)
        assert got == 32
    finally:
        disk.close()


def test_anchor_gets_no_pool_pressure_protection():
    """_evict_for_pool gives anchors no absolute protection: once a
    chain group has nothing else left its anchor reclaims like any
    record, so a pinned anchor can never starve the block pool."""
    import gmlx.cache_snapshot as cs

    man = APCManager(num_blocks=64, block_size=16)
    ids = list(range(400, 448))
    assert ckpt_store(man, ids[:32], make_hybrid_cache(32, seed=1),
                      extra_hash=0)
    idx = cs._ckpt_records(man)
    assert [r.kind for r in idx.values()] == ["anchor"]
    assert cs._evict_for_pool(man, 1) >= 1
    assert len(cs._ckpt_records(man)) == 0
