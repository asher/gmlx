"""Decode-time checkpoints at the predicted LCP + ckpt-tier byte budgets.

CPU-only unit tests. The tick clones recurrent states (and rotating
windows, at block-aligned positions) into a two-slot ring while the
predicted next-turn render still replays the sequence so far; retirement
assembles a checkpoint from the newest snapshot at or below the actual
LCP (plain KV truncates, everything else comes from the snapshot).
Boundaries anchor to the prompt end on the chunk grid.
"""

from types import SimpleNamespace

import mlx.core as mx
import pytest
from gmlx._vendor.mlx_vlm.apc import APCManager

import gmlx.cache.snapshot as cs
import gmlx.cache.retire_key as retire_key
from gmlx.cache.snapshot import (
    ckpt_lookup,
    ckpt_store,
    decode_ckpt_tick,
    retirement_store,
)
from test_ckpt_tier import LAYOUT, make_hybrid_cache


def _arr_states(cache):
    return [c for c in cache if not hasattr(c, "keys")]


def _stash(full_ids, **kw):
    d = {"full_ids": list(full_ids), "extra_hash": 0, "mode": "ckpt",
         "render_ctx": {"stub": True}, "snap_ok": True}
    d.update(kw)
    return d


def _tick_at(stash, p, gen_len, pred, monkeypatch):
    monkeypatch.setattr(retire_key, "next_turn_lcp",
                        lambda ctx, seq, gen, **kw: pred)
    cache = make_hybrid_cache(p, seed=p)
    gen = list(range(9000, 9000 + gen_len))
    decode_ckpt_tick(stash, cache, gen)
    return cache


def test_tick_ring_and_freeze(monkeypatch):
    monkeypatch.setenv("GMLX_APC_DECODE_CKPT", "96")
    full_ids = list(range(40))
    stash = _stash(full_ids)
    # First boundary anchors to the prompt end (40 + 96 = 136); a
    # pre-boundary call no-ops.
    _tick_at(stash, 50, 10, 50, monkeypatch)
    assert "snaps" not in stash and stash["snap_next"] == 136
    # Replaying render (pred == live length): snapshot lands.
    c136 = _tick_at(stash, 136, 96, 136, monkeypatch)
    assert [p for p, _ in stash["snaps"]] == [136]
    for snap, orig in zip(stash["snaps"][0][1], _arr_states(c136)):
        for a, b in zip(snap.cache, orig.cache):
            assert mx.array_equal(a, b).item()
    # Ring keeps the newest two.
    _tick_at(stash, 232, 192, 232, monkeypatch)
    _tick_at(stash, 328, 288, 328, monkeypatch)
    assert [p for p, _ in stash["snaps"]] == [232, 328]
    # Structural divergence at 240 (beyond the retokenization margin):
    # slots past it drop, ring freezes.
    _tick_at(stash, 424, 384, 240, monkeypatch)
    assert stash["snap_frozen"] and [p for p, _ in stash["snaps"]] == [232]
    before = list(stash["snaps"])
    _tick_at(stash, 456, 416, 456, monkeypatch)
    assert stash["snaps"] == before


def test_tick_retokenization_margin(monkeypatch):
    monkeypatch.setenv("GMLX_APC_DECODE_CKPT", "32")
    stash = _stash(list(range(40)))
    # pred a few tokens shy of live (tail retokenization): still advances.
    _tick_at(stash, 72, 32, 69, monkeypatch)
    assert [p for p, _ in stash["snaps"]] == [72] and not stash.get(
        "snap_frozen")


def test_tick_grid_alignment_for_rotating(monkeypatch):
    from test_ckpt_tier import make_swa_cache

    monkeypatch.setenv("GMLX_APC_DECODE_CKPT", "32")
    stash = _stash(list(range(40)), snap_grid=16, snap_align=16)
    monkeypatch.setattr(retire_key, "next_turn_lcp",
                        lambda ctx, seq, gen, **kw: 10_000)
    # First boundary: 40 + 32 snapped up the 16-token grid = 80.
    decode_ckpt_tick(stash, make_swa_cache(48), list(range(9000, 9008)))
    assert "snaps" not in stash and stash["snap_next"] == 80
    # Past the boundary but off the block grid: a rotating clone would be
    # unusable, so the tick waits.
    decode_ckpt_tick(stash, make_swa_cache(85), list(range(9000, 9045)))
    assert "snaps" not in stash
    # First aligned position at or past the boundary snapshots, cloning
    # the rotating windows alongside (nothing else here is cloneable).
    cache = make_swa_cache(96)
    decode_ckpt_tick(stash, cache, list(range(9000, 9056)))
    (p, states), = stash["snaps"]
    assert p == 96 and len(states) == 3
    for s in states:
        assert int(s.offset) == 96 and hasattr(s, "max_size")
    assert stash["snap_next"] == 128


def test_tick_failure_disables_ring(monkeypatch):
    monkeypatch.setenv("GMLX_APC_DECODE_CKPT", "32")
    stash = _stash(list(range(40)))
    calls = []

    def boom(ctx, seq, gen, **kw):
        calls.append(1)
        raise RuntimeError("drifted private")

    monkeypatch.setattr(retire_key, "next_turn_lcp", boom)
    cache = make_hybrid_cache(72, seed=44)
    gen = list(range(9000, 9032))
    decode_ckpt_tick(stash, cache, gen)     # first strike: disables
    assert stash["snap_ok"] is False and len(calls) == 1
    decode_ckpt_tick(stash, cache, gen)     # no retry, no second call
    assert len(calls) == 1


def test_tick_takes_no_snapshot_of_a_batched_qsa_row(monkeypatch):
    """A row left alone after a join keeps its QSA layers in a cache of one
    cache per row. The tick clones the recurrent states and skips that
    layer, so the row's K and V are neither extracted nor copied, and no
    snapshot assembles with it."""
    from gmlx.models.qwen4_exp.model import BatchQSAKVCache, QSAKVCache

    monkeypatch.setenv("GMLX_APC_DECODE_CKPT", "32")
    monkeypatch.setattr(retire_key, "next_turn_lcp",
                        lambda ctx, seq, gen, **kw: 72)
    cache = make_hybrid_cache(72, seed=45)
    for i, c in enumerate(cache):
        if hasattr(c, "keys"):
            row = QSAKVCache(ratio=4)
            row.append_qsa(c.keys, c.values, mx.zeros((1, 72, 8)))
            cache[i] = BatchQSAKVCache([row])
    extracts, cloned = [], []
    monkeypatch.setattr(BatchQSAKVCache, "extract",
                        lambda self, i: extracts.append(i))
    clone = cs._clone_single_row

    def spy(c, pending=None):
        cloned.append(c)
        return clone(c, pending)

    monkeypatch.setattr(cs, "_clone_single_row", spy)
    stash = _stash(list(range(40)))
    decode_ckpt_tick(stash, cache, list(range(9000, 9032)))
    (p, states), = stash["snaps"]
    assert p == 72 and extracts == []
    assert len(states) == LAYOUT.count("arr")
    assert not any(isinstance(c, BatchQSAKVCache) for c in cloned)
    assert cs._snap_assemble(cache, states, 72) is None


def test_a_snapshot_takes_a_qsa_layer_to_its_p_with_its_positions():
    """A decode snapshot's QSA layer is the live row's K, V, indexer keys
    and mrope positions to the snapshot's p, as a QSA cache, so the record
    tags the layout the lookups sign."""
    from gmlx.models.qwen4_exp.model import QSAKVCache

    n, p = 40, 32
    mx.random.seed(46)
    c = QSAKVCache(ratio=4)
    pos = mx.broadcast_to((mx.arange(n, dtype=mx.int32) // 2)[None, None],
                          (3, 1, n))
    c.update_and_fetch_qsa(mx.random.normal((1, 2, n, 8)),
                           mx.random.normal((1, 2, n, 8)),
                           mx.random.normal((1, n, 8)), pos=pos)
    (got,) = cs._snap_assemble([c], [], p)
    assert type(got) is QSAKVCache and got.offset == p
    assert cs._qsa_tag(got) == "qsa:4"
    k, v, ik, ps = c.state
    want = (k[..., :p, :], v[..., :p, :], ik[:, :p], ps[:, :, :p])
    assert len(got.state) == 4
    for x, y in zip(got.state, want):
        assert mx.array_equal(x, y).item()


def test_a_snapshots_qsa_layer_shares_the_live_rows_buffers():
    """The QSA layer a retirement assembles from a decode snapshot reads
    the live row's streams in place. The store copies the rows it keeps,
    so a copy here, 1.1 GB at 40k tokens on Qwen3.8-Flash-Next, would be
    a second one, made on the engine thread and priced nowhere."""
    from gmlx.models.qwen4_exp.model import QSAKVCache

    n, p, heads, dim = 4096, 4000, 2, 64
    mx.random.seed(47)
    c = QSAKVCache(ratio=4)
    c.update_and_fetch_qsa(
        mx.random.normal((1, heads, n, dim)).astype(mx.float16),
        mx.random.normal((1, heads, n, dim)).astype(mx.float16),
        mx.random.normal((1, n, dim)).astype(mx.float16))
    mx.eval(c._kb, c._vb, c._ib)
    before = mx.get_active_memory()
    (got,) = cs._snap_assemble([c], [], p)
    grown = mx.get_active_memory() - before
    assert grown < p * heads * dim * 2 // 8, grown
    k, v, ik = c.state
    for x, y in zip(got.state, (k[..., :p, :], v[..., :p, :], ik[:, :p])):
        assert mx.array_equal(x, y).item()


def test_retirement_uses_newest_snap_at_or_below_lcp():
    man = APCManager(num_blocks=64, block_size=16)
    full = 96
    cache = make_hybrid_cache(full, seed=3)
    ids = list(range(500, 500 + full))
    snaps = [(32, _arr_states(make_hybrid_cache(32, seed=91))),
             (64, _arr_states(make_hybrid_cache(64, seed=92)))]
    assert retirement_store(man, "ckpt", ids, cache, max_len=80,
                            decode_snaps=snaps)
    warm, got = ckpt_lookup(man, ids[:64] + [1], extra_hash=0)
    assert got == 64
    for w, o in zip(warm, cache):
        if hasattr(o, "keys"):
            assert int(w.offset) == 64
            assert mx.array_equal(w.keys, o.keys[..., :64, :]).item()
    for w, s in zip([w for w in warm if not hasattr(w, "keys")], snaps[1][1]):
        for a, b in zip(w.cache, s.cache):
            assert mx.array_equal(a, b).item()


def test_retirement_rotating_uses_grid_snap():
    """When the whole-sequence store cannot run (replayable prefix
    below the full length), the aligned decode snapshot still serves a
    rotating retirement. With no cap the same shape now stores its
    grid prefix directly -- see
    test_retirement_rotating_short_prompt_stores_grid_prefix."""
    from test_ckpt_tier import assert_swa_warm_matches, make_swa_cache

    man = APCManager(num_blocks=64, block_size=16)
    n = 30                                    # < W=32 and off-grid
    ids = list(range(800, 800 + n))
    cache = make_swa_cache(n, seed=21)
    snap_src = make_swa_cache(16, seed=22)
    states = [c for c in snap_src if hasattr(c, "max_size")]
    assert retirement_store(man, "ckpt", ids, cache, max_len=20,
                            decode_snaps=[(16, states)])
    warm, got = ckpt_lookup(man, ids[:16] + [1], extra_hash=0)
    assert got == 16 and warm is not None
    # Expected: plain KV from the live row, rotating windows from the
    # snapshot clones.
    expected = [s if hasattr(s, "max_size") else c
                for c, s in zip(cache, snap_src)]
    assert_swa_warm_matches(warm, expected, 16)
    # No exact-tier spill: the verbatim-row fallback is gone.
    assert man.stats_snapshot()["exact_stores"] == 0


def test_retirement_predicted_diverged_suppresses_fallback():
    """cap < len only when the next-turn render prediction succeeded:
    the client provably re-renders, so a full-sequence verbatim store
    can never match its next turn -- it ages in the pool and burns an
    ids_diverged decline per later same-chain lookup (gemma-4/gpt-oss
    cert arms, ~410 each). With no snapshot at or below the boundary
    the retirement stores nothing, counted."""
    man = APCManager(num_blocks=64, block_size=16)
    cache = make_hybrid_cache(96, seed=4)
    ids = list(range(96))
    snaps = [(90, _arr_states(make_hybrid_cache(90, seed=93)))]
    assert not retirement_store(man, "ckpt", ids, cache, max_len=80,
                                decode_snaps=snaps)
    _, got = ckpt_lookup(man, ids[:90] + [1], extra_hash=0)
    assert got == 0                       # the past-cap snap stayed unused
    _, got = ckpt_lookup(man, ids + [1], extra_hash=0)
    assert got == 0                       # no verbatim fallback record
    snap = cs.ckpt_stats_snapshot(man)
    assert snap["retire_fallback_suppressed"] == 1


def test_retirement_snap_path_never_falls_back():
    man = APCManager(num_blocks=64, block_size=16)
    cache = make_hybrid_cache(96, seed=5)
    ids = list(range(96))
    snaps = [(64, _arr_states(make_hybrid_cache(64, seed=94)))]
    assert retirement_store(man, "ckpt", ids, cache, max_len=80,
                            decode_snaps=snaps)
    assert cs.ckpt_stats_snapshot(man)["retire_fallback_suppressed"] == 0


def test_governor_evict_releases_ckpt_records():
    """The governor's red-band reclaim must reach ckpt records: releasing
    them unpins their blocks so the pool eviction in the same pass can
    reclaim those too (122B run-3 shed: pinned share invisible to
    reclaim). Fraction 1.0 empties the index; freed bytes and the
    ckpt_governor_released counter report the work."""
    from gmlx.cache.apc_manager import GmlxAPCManager
    man = GmlxAPCManager(num_blocks=64, block_size=16)
    for i in range(3):
        ids = list(range(i * 1000, i * 1000 + 64))
        assert ckpt_store(man, ids, make_hybrid_cache(64, seed=40 + i))
    idx = man._kq_ckpt_records
    assert len(idx) == 3
    pinned = sum(1 for b in man.pool if b.ref_cnt > 0)
    assert pinned > 0
    before = man.governor_bytes()
    freed = man.governor_evict(1.0)
    assert freed > 0
    assert len(man._kq_ckpt_records) == 0
    assert all(b.ref_cnt == 0 for b in man.pool)
    assert man.governor_bytes() < before
    assert cs.ckpt_stats_snapshot(man)["ckpt_governor_released"] == 3


def test_governor_evict_fraction_releases_lru_first():
    from gmlx.cache.apc_manager import GmlxAPCManager
    man = GmlxAPCManager(num_blocks=64, block_size=16)
    for i in range(4):
        ids = list(range(i * 1000, i * 1000 + 32))
        assert ckpt_store(man, ids, make_hybrid_cache(32, seed=50 + i))
    man.governor_evict(0.5)
    keys = [k[0][0] for k in man._kq_ckpt_records.keys()]
    # Oldest two chains released, newest two kept.
    assert keys == [2000, 3000]


def test_evict_bytes_releases_the_coldest_record_with_its_blocks():
    """With every block pinned by a record, an eviction by bytes releases
    the least recently used record and evicts the blocks it pinned. The
    records after it keep theirs."""
    from gmlx.cache.apc_manager import GmlxAPCManager
    man = GmlxAPCManager(num_blocks=64, block_size=16)
    for i in range(3):
        ids = list(range(i * 1000, i * 1000 + 64))
        assert ckpt_store(man, ids, make_hybrid_cache(64, seed=60 + i))
    recs = list(man._kq_ckpt_records.values())
    held = [list(r.main_blocks) + list(r.bounded_blocks) for r in recs]
    assert all(held)
    want = recs[0].nbytes + sum(b.resident_bytes() for b in held[0])
    assert man.reclaimable_bytes() >= want

    assert man.evict_bytes(want) == want
    assert [k[0][0] for k in man._kq_ckpt_records.keys()] == [1000, 2000]
    assert all(b.ref_cnt == 0 and not b.resident_bytes() for b in held[0])
    assert all(b.ref_cnt > 0 and b.resident_bytes()
               for blocks in held[1:] for b in blocks)


def test_pool_eviction_prefers_fattest_chain():
    """Pool-pressure eviction charges the chain group holding the most
    blocks, deepest record first, so under round-robin sessions no
    store evicts the record the next session is about to reuse (the
    122B run-4 starvation was plain LRU doing exactly that). The lean
    chain survives; the fat chain degrades to its shallower record."""
    man = APCManager(num_blocks=16, block_size=16)
    ids_a = list(range(1000, 1160))
    ids_b = list(range(5000, 5064))
    # Chain A: anchor at 96 (6 blocks) extended to 160 (10 blocks,
    # prefix shared). Chain B: one record at 64 (4 blocks).
    assert ckpt_store(man, ids_a[:96], make_hybrid_cache(96, seed=60))
    assert ckpt_store(man, ids_a, make_hybrid_cache(160, seed=61))
    assert ckpt_store(man, ids_b, make_hybrid_cache(64, seed=62))
    # Pool full (10 + 4 unique blocks, 2 free): a 4-block store for a
    # third chain must evict, and the victim is A's deepest record.
    ids_c = list(range(9000, 9064))
    assert ckpt_store(man, ids_c, make_hybrid_cache(64, seed=63))
    ps = {r.p for r in cs._ckpt_records(man).values()
          if r.ids[0] == 1000}
    assert 160 not in ps and 96 in ps
    _, got = ckpt_lookup(man, ids_b + [1], extra_hash=0)
    assert got == 64                    # lean chain untouched
    _, got = ckpt_lookup(man, ids_a + [1], extra_hash=0)
    assert got == 96                    # fat chain degrades, not zeroed


def test_pool_eviction_counts_own_chain_recycle():
    """A store whose own chain group is the fattest recycles its own
    record and the ckpt_evict_own_chain counter says so. A clean extend
    shares its prefix blocks and rarely evicts, so the pressured case
    is a divergent branch: same first block, disjoint tail."""
    man = APCManager(num_blocks=12, block_size=16)
    ids_a = list(range(1000, 1096))
    assert ckpt_store(man, ids_a, make_hybrid_cache(96, seed=70))
    assert ckpt_store(man, list(range(5000, 5032)),
                      make_hybrid_cache(32, seed=71))
    # Branch of A: shares only the first block, needs 7 fresh of 4
    # free; A (6 held) is the fattest group, so A's own record is the
    # victim, not the bystander.
    branch = ids_a[:16] + list(range(7000, 7112))
    assert ckpt_store(man, branch, make_hybrid_cache(128, seed=72))
    assert cs.ckpt_stats_snapshot(man)["ckpt_evict_own_chain"] >= 1
    _, got = ckpt_lookup(man, list(range(5000, 5032)) + [1], extra_hash=0)
    assert got == 32                    # bystander chain survives


def test_record_byte_budget_evicts_lru(monkeypatch):
    man = APCManager(num_blocks=64, block_size=16)
    ids_a = list(range(100, 132))
    ids_b = list(range(700, 732))
    assert ckpt_store(man, ids_a, make_hybrid_cache(32, seed=1))
    assert ckpt_store(man, ids_b, make_hybrid_cache(32, seed=2),
                      extra_hash=5)
    assert len(cs._ckpt_records(man)) == 2
    # Any GDN record overflows a 1-byte budget; the newest must survive.
    monkeypatch.setattr(cs, "_CKPT_BUDGET_BYTES", 1)
    ids_c = list(range(300, 332))
    assert ckpt_store(man, ids_c, make_hybrid_cache(32, seed=3),
                      extra_hash=9)
    recs = list(cs._ckpt_records(man).values())
    assert len(recs) == 1 and recs[0].ids == tuple(ids_c)
    assert recs[0].nbytes > 1


@pytest.mark.parametrize("bound", ["bytes-full", "count-full"])
def test_a_record_payload_materializes_within_the_record_bounds(
        monkeypatch, bound):
    """A checkpoint's recurrent states and attention tail make room among
    the records before they materialize, so the records and the new
    payload fit both bounds together. The records kept are the ones
    insert-then-trim kept."""
    from gmlx.eval_guard import guard

    man = APCManager(num_blocks=64, block_size=16)
    ids = {"a": (list(range(100, 132)), 0), "b": (list(range(700, 732)), 5),
           "c": (list(range(300, 332)), 9)}
    for seed, name in enumerate(("a", "b"), 1):
        assert ckpt_store(man, ids[name][0], make_hybrid_cache(32, seed=seed),
                          extra_hash=ids[name][1])
    idx = cs._ckpt_records(man)
    each = next(iter(idx.values())).nbytes
    entries, budget = ((cs._CKPT_RECORD_ENTRIES, 2 * each + each // 2)
                       if bound == "bytes-full" else (2, 1 << 30))
    monkeypatch.setattr(cs, "_CKPT_RECORD_ENTRIES", entries)
    monkeypatch.setattr(cs, "_CKPT_BUDGET_BYTES", budget)
    seen = []
    real = guard.eval

    def spy(*arrays, **kw):
        if kw.get("site") == "ckpt-store-clones":
            seen.append((len(idx), sum(r.nbytes for r in idx.values())))
        return real(*arrays, **kw)

    monkeypatch.setattr(guard, "eval", spy)
    assert ckpt_store(man, ids["c"][0], make_hybrid_cache(32, seed=3),
                      extra_hash=ids["c"][1])
    assert len(seen) == 1
    held_n, held_bytes = seen[0]
    assert held_n + 1 <= entries
    assert held_bytes + each <= budget
    assert [r.ids[0] for r in idx.values()] == [700, 300]
    assert all(r.nbytes == each for r in idx.values())


@pytest.mark.parametrize("at", ["payload copy", "clone decline"])
def test_a_reset_during_a_checkpoint_store_leaves_the_pool_free(
        monkeypatch, at):
    """A cache reset that lands while a checkpoint store holds its chains:
    the reset leaves the chains to the store, which inserts no record and
    releases them, so the free list holds every block once and a later
    lookup restores nothing."""
    from gmlx.cache.apc_manager import GmlxAPCManager
    from gmlx.eval_guard import guard

    man = GmlxAPCManager(num_blocks=64, block_size=16)
    ids = list(range(100, 132))
    if at == "payload copy":
        real = guard.eval

        def spy(*arrays, **kw):
            if kw.get("site") == "ckpt-store-clones":
                man.clear()
            return real(*arrays, **kw)

        monkeypatch.setattr(guard, "eval", spy)
    else:
        def clone(cache, eval_targets=None):
            man.clear()
            return None

        monkeypatch.setattr(cs, "_clone_single_row", clone)
    try:
        assert ckpt_store(man, ids, make_hybrid_cache(32, seed=1)) == 0
        assert not cs._ckpt_records(man)
        free, b = [], man._free_head
        while b is not None and len(free) <= len(man.pool):
            free.append(id(b))
            b = b.next
        assert sorted(free) == sorted(id(b) for b in man.pool)
        assert all(b.ref_cnt == 0 for b in man.pool)
        assert ckpt_lookup(man, ids + [1], extra_hash=0)[1] == 0
    finally:
        man.close()


def test_a_decode_snapshot_materializes_within_the_ring(monkeypatch):
    """The ring holds two decode snapshots. The oldest makes room before
    the third materializes, so a row never holds three, and the ring keeps
    the newest two as before."""
    monkeypatch.setenv("GMLX_APC_DECODE_CKPT", "96")
    stash = _stash(list(range(40)))
    _tick_at(stash, 136, 96, 136, monkeypatch)
    _tick_at(stash, 232, 192, 232, monkeypatch)
    assert [p for p, _ in stash["snaps"]] == [136, 232]
    seen = []
    real = mx.eval

    def spy(*arrays):
        seen.append(len(stash["snaps"]))
        real(*arrays)

    monkeypatch.setattr(retire_key, "next_turn_lcp",
                        lambda ctx, seq, gen, **kw: 328)
    cache = make_hybrid_cache(328, seed=328)
    gen = list(range(9000, 9288))
    monkeypatch.setattr(mx, "eval", spy)
    decode_ckpt_tick(stash, cache, gen)
    monkeypatch.setattr(mx, "eval", real)
    assert seen and max(seen) + 1 <= 2
    assert [p for p, _ in stash["snaps"]] == [232, 328]


def test_sidecar_byte_budget(monkeypatch):
    from gmlx.cache.snapshot import _sidecar_index, drafter_sidecar_store
    from test_ckpt_tier import KVCache

    man = APCManager(num_blocks=8, block_size=16)

    def head(p):
        c = KVCache()
        c.state = (mx.random.normal((1, 2, p, 8)),
                   mx.random.normal((1, 2, p, 8)))
        return c

    drafter = SimpleNamespace(supports_kv_sidecar=True,
                              export_kv=lambda: [head(16)])
    assert drafter_sidecar_store(man, drafter, list(range(16)), 16)
    monkeypatch.setattr(cs, "_SIDECAR_BUDGET_BYTES", 1)
    assert drafter_sidecar_store(man, drafter, list(range(50, 66)), 16)
    idx = _sidecar_index(man)
    assert len(idx) == 1
    (key, _), = idx.items()
    assert key[0][0] == 50


def test_skeleton_disk_flag(monkeypatch):
    calls = []
    monkeypatch.setattr(cs, "_ckpt_disk_write",
                        lambda *a, **k: calls.append(1))
    man = APCManager(num_blocks=64, block_size=16)
    assert ckpt_store(man, list(range(32)), make_hybrid_cache(32, seed=6),
                      skeleton_disk=False)
    assert not calls
    assert ckpt_store(man, list(range(200, 232)),
                      make_hybrid_cache(32, seed=7))
    assert len(calls) == 1


def test_cursor_skeleton_policy(monkeypatch):
    import gmlx.spec.ckpt as ckpt

    seen = []

    def rec_store(manager, ids, cache, *, extra_hash=0, skeleton_disk=True,
                  kind="boundary", keep=None, eval_async=False):
        seen.append((len(ids), skeleton_disk))
        return True

    monkeypatch.setattr(cs, "ckpt_store", rec_store)
    monkeypatch.setenv("GMLX_APC_CKPT_INTERVAL", "32")
    man = APCManager(num_blocks=8, block_size=16)
    tags = tuple("arr" if k == "arr" else "kv" for k in LAYOUT)
    meta = {"full_input_ids": list(range(96)), "extra_hash": 0}
    batch = SimpleNamespace(
        _kq_ckpt_armed=True, _apc_manager=man, _apc_meta=[meta],
        prefill_step_size=32, prompt_cache=[],
        model=SimpleNamespace(_kq_apc_ckpt_layout=tags),
        _row_real_tokens_processed=lambda i: meta["checkpoint_len"])
    ckpt._ckpt_arm_schedule(batch, meta, 80, 0, 16)
    assert meta["ckpt_boundaries"] == [(32, "boundary"), (64, "boundary")]
    ckpt._ckpt_mid_prefill_store(batch)   # boundary 32: interval
    ckpt._ckpt_mid_prefill_store(batch)   # boundary 64: terminal
    assert seen == [(32, False), (64, True)]


def test_disk_lookup_failure_attempts_every_release(caplog):
    """A failing release in the disk-lookup failure handler must not
    skip the other block set: each release gets its own attempt."""
    from gmlx.cache.kv_cache import KVCache

    releases = []

    class _Man:
        block_size = 16

        def lookup_exact_cache(self, ids, *, extra_hash, min_prefix_tokens):
            return [KVCache()], 32

        def lookup_prefix(self, ids, *, extra_hash):
            raise RuntimeError("prefix lookup broke")

        def release(self, blocks):
            releases.append(blocks)
            if len(releases) == 1:
                raise RuntimeError("first release broke")

    with caplog.at_level("WARNING", logger="gmlx.cache.snapshot"):
        out = cs._ckpt_disk_lookup(
            _Man(), list(range(40)), extra_hash=0,
            min_prefix_tokens=1, layout=None)
    assert out == (None, 0)
    assert len(releases) == 2
    assert any("disk lookup failed" in r.message for r in caplog.records)
