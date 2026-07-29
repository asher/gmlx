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
from mlx_vlm.apc import APCManager

from gmlx import cache_snapshot as cs
from gmlx import retire_key
from gmlx.cache_snapshot import (
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
                        lambda ctx, seq, gen: pred)
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
                        lambda ctx, seq, gen: 10_000)
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

    def boom(ctx, seq, gen):
        calls.append(1)
        raise RuntimeError("drifted private")

    monkeypatch.setattr(retire_key, "next_turn_lcp", boom)
    cache = make_hybrid_cache(72, seed=44)
    gen = list(range(9000, 9032))
    decode_ckpt_tick(stash, cache, gen)     # first strike: disables
    assert stash["snap_ok"] is False and len(calls) == 1
    decode_ckpt_tick(stash, cache, gen)     # no retry, no second call
    assert len(calls) == 1


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
    from test_ckpt_tier import assert_swa_warm_matches, make_swa_cache

    man = APCManager(num_blocks=64, block_size=16)
    n = 70                                    # retirement p: off-grid
    ids = list(range(800, 800 + n))
    cache = make_swa_cache(n, seed=21)
    snap_src = make_swa_cache(64, seed=22)
    states = [c for c in snap_src if hasattr(c, "max_size")]
    assert retirement_store(man, "ckpt", ids, cache,
                            decode_snaps=[(64, states)])
    warm, got = ckpt_lookup(man, ids[:64] + [1], extra_hash=0)
    assert got == 64 and warm is not None
    # Expected: plain KV from the live row, rotating windows from the
    # snapshot clones.
    expected = [s if hasattr(s, "max_size") else c
                for c, s in zip(cache, snap_src)]
    assert_swa_warm_matches(warm, expected, 64)
    # No exact-tier spill: the verbatim-row fallback is gone.
    assert man.stats_snapshot()["exact_stores"] == 0


def test_retirement_snap_past_lcp_is_skipped():
    man = APCManager(num_blocks=64, block_size=16)
    cache = make_hybrid_cache(96, seed=4)
    ids = list(range(96))
    snaps = [(90, _arr_states(make_hybrid_cache(90, seed=93)))]
    assert not retirement_store(man, "ckpt", ids, cache, max_len=80,
                                decode_snaps=snaps)
    _, got = ckpt_lookup(man, ids[:90] + [1], extra_hash=0)
    assert got == 0


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


def test_sidecar_byte_budget(monkeypatch):
    from gmlx.cache_snapshot import _sidecar_index, drafter_sidecar_store
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
    from gmlx import spec_engine

    seen = []

    def rec_store(manager, ids, cache, *, extra_hash=0, skeleton_disk=True):
        seen.append((len(ids), skeleton_disk))
        return True

    monkeypatch.setattr(cs, "ckpt_store", rec_store)
    man = APCManager(num_blocks=8, block_size=16)
    tags = tuple("arr" if k == "arr" else "kv" for k in LAYOUT)
    meta = {"full_input_ids": list(range(96)), "extra_hash": 0,
            "checkpoint_len": 32, "ckpt_terminal": 64, "ckpt_interval": 32,
            "ckpt_last_stored": 0}
    batch = SimpleNamespace(
        _kq_ckpt_armed=True, _apc_manager=man, _apc_meta=[meta],
        prompt_cache=[], model=SimpleNamespace(_kq_apc_ckpt_layout=tags),
        _row_real_tokens_processed=lambda i: meta["checkpoint_len"])
    spec_engine._ckpt_mid_prefill_store(batch)   # boundary 32: interval
    spec_engine._ckpt_mid_prefill_store(batch)   # boundary 64: terminal
    assert seen == [(32, False), (64, True)]
