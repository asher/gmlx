"""Checkpoint records take their room from the pool's share.

A static records-only budget beside a pool sized from free room dropped a
conversation's deepest record at width 4 while the pool sat mostly idle:
the conversation looked up last was the coldest one, so the budget took
its record as the other three stored theirs. Records and the pool's
committed blocks now share one figure; a chain of blocks is useless
without its record, so blocks no record holds go first, and a record goes
with the blocks only it held. Beside it: a retirement no lookup can use
stores nothing.
"""

import logging

import mlx.core as mx
import pytest

import gmlx.cache.apc_manager as am
import gmlx.cache.snapshot as cs
from gmlx.cache.apc_manager import GmlxAPCManager
from gmlx.cache.snapshot import ckpt_lookup, ckpt_store
from test_ckpt_tier import H, LAYOUT, make_hybrid_cache

pytestmark = pytest.mark.host_gb(16)

# A pool block of make_hybrid_cache: two KV layers, K and V of
# (1, 2, 16, 8) float32.
_BLOCK_BYTES = 2 * 2 * 2 * 16 * 8 * 4


def _ids(chain, p):
    return list(range(1000 * (chain + 1), 1000 * (chain + 1) + p))


def _records_bytes(man):
    return sum(r.nbytes for r in cs._ckpt_records(man).values())


def _block_bytes(man):
    return sum(int(b.resident_bytes()) for b in man.pool)


def _idle_bytes(man):
    return sum(int(b.resident_bytes()) for b in man.pool if b.ref_cnt == 0)


def _big(p, seed):
    """make_hybrid_cache with each recurrent layer's matrix state at
    512 KiB, so a record holds about 1.5 MiB and the MiB counter moves."""
    caches = make_hybrid_cache(p, seed=seed)
    mx.random.seed(seed)
    for c, kind in zip(caches, LAYOUT):
        if kind == "arr":
            c.cache = [c.cache[0], mx.random.normal((1, H, 256, 256))]
    return caches


def _store(man, chain, p=64):
    return ckpt_store(man, _ids(chain, p), _big(p, chain), extra_hash=chain)


def _chains(man):
    return [r.extra_hash for r in cs._ckpt_records(man).values()]


def _budget_lines(caplog):
    return [r.getMessage() for r in caplog.records
            if r.getMessage().startswith("APC ckpt budget")]


def test_round_robin_keeps_the_coldest_conversations_deep_record(monkeypatch):
    """Four conversations, each an anchor and a deep record. Three look up
    and store a deeper record; the fourth, looked up last, still restores
    its deep record. The records-only budget stands at what the four held
    before the stores, the proportion of the width 4 run (4096 MB against
    a 5737 MB peak); the pool's share holds them all."""
    man = GmlxAPCManager(num_blocks=64, block_size=16)
    for c in range(4):
        assert ckpt_store(man, _ids(c, 32), make_hybrid_cache(32, seed=c),
                          extra_hash=c)
        assert ckpt_store(man, _ids(c, 112),
                          make_hybrid_cache(112, seed=10 + c), extra_hash=c)
    monkeypatch.setattr(cs, "_CKPT_BUDGET_BYTES", _records_bytes(man) + 1)
    # The share autosize stashes: the pool's cap on this manager.
    man._kq_ckpt_share_bytes = 64 * _BLOCK_BYTES
    for c in range(3):
        _, got = ckpt_lookup(man, _ids(c, 112) + [7], extra_hash=c)
        assert got == 112
    for c in range(3):
        assert ckpt_store(man, _ids(c, 128),
                          make_hybrid_cache(128, seed=20 + c), extra_hash=c)
    _, got = ckpt_lookup(man, _ids(3, 112) + [7], extra_hash=3)
    assert got == 112


def test_over_the_share_idle_blocks_go_first_then_records_with_theirs(
        caplog):
    """Past the share, blocks no record holds are freed first and no
    record goes while they cover the excess. Past that, the least
    recently used record goes with the blocks only it held: no block is
    left committed without a record, both counters move, and one line
    names the victim."""
    man = GmlxAPCManager(num_blocks=64, block_size=16)
    for c in range(3):
        assert _store(man, c)
    # A record another path dropped leaves its blocks idle: hashed,
    # unreferenced, at the tail of the free queue.
    assert _store(man, 9)
    idx = cs._ckpt_records(man)
    cs._release_record(man, idx.pop(next(
        k for k, r in idx.items() if r.extra_hash == 9)))
    idle = _idle_bytes(man)
    assert idle > 0
    # What one more conversation adds, its record and its blocks.
    twin = GmlxAPCManager(num_blocks=64, block_size=16)
    assert _store(twin, 3)
    add = _records_bytes(twin) + _block_bytes(twin)

    total = _records_bytes(man) + _block_bytes(man)
    share = total + add - idle // 2
    man._kq_ckpt_share_bytes = share
    with caplog.at_level(logging.INFO, logger="gmlx.cache.snapshot"):
        assert _store(man, 3)
    assert _chains(man) == [0, 1, 2, 3]
    assert _idle_bytes(man) < idle
    assert _records_bytes(man) + _block_bytes(man) <= share
    st = man.stats_snapshot()
    assert st["ckpt_budget_evictions"] == 0
    assert st["ckpt_budget_evicted_mb"] == 0
    assert _budget_lines(caplog) == []

    left = _idle_bytes(man)
    share = _records_bytes(man) + _block_bytes(man) + add - left - add // 2
    man._kq_ckpt_share_bytes = share
    with caplog.at_level(logging.INFO, logger="gmlx.cache.snapshot"):
        assert _store(man, 4)
    assert _chains(man) == [1, 2, 3, 4]
    assert _idle_bytes(man) == 0
    assert all(b.ref_cnt > 0 for b in man.pool if b.resident_bytes())
    assert _records_bytes(man) + _block_bytes(man) <= share
    _, got = ckpt_lookup(man, _ids(0, 64) + [7], extra_hash=0)
    assert got == 0
    st = man.stats_snapshot()
    assert st["ckpt_budget_evictions"] == 1
    assert st["ckpt_budget_evicted_mb"] >= 1
    lines = _budget_lines(caplog)
    assert len(lines) == 1
    assert "dropped 1 record(s)" in lines[0]
    assert "p=64 anchor" in lines[0]
    assert "records and pool blocks" in lines[0]


def test_the_override_is_a_records_only_budget_and_wins(monkeypatch):
    """GMLX_APC_CKPT_BUDGET_MB set: records alone count against it, and
    a share the blocks already fill drops nothing."""
    man = GmlxAPCManager(num_blocks=64, block_size=16)
    for c in range(2):
        assert _store(man, c)
    one = _records_bytes(man) // 2
    man._kq_ckpt_share_bytes = 1
    monkeypatch.setattr(cs, "_CKPT_BUDGET_SET", True)
    monkeypatch.setattr(cs, "_CKPT_BUDGET_BYTES", 3 * one + 1)
    assert cs._record_room(man) == (3 * one + 1, False)
    assert _store(man, 2)
    assert _chains(man) == [0, 1, 2]
    assert man.stats_snapshot()["ckpt_budget_evictions"] == 0
    assert _store(man, 3)
    assert _chains(man) == [1, 2, 3]
    assert man.stats_snapshot()["ckpt_budget_evictions"] == 1


def test_autosize_stashes_the_pools_share(monkeypatch):
    """autosize stashes the share it sizes the pool from, the pool's own
    bytes when APC_NUM_BLOCKS sets it, and never less than the pool's
    cap. A manager autosize never sized keeps the records-only default."""
    from test_apc_manager import GB, _autosize_rig, _fake_model

    for k in ("APC_NUM_BLOCKS", "APC_EXACT_CACHE_ENTRIES",
              "APC_MEMORY_MAX_GB"):
        monkeypatch.delenv(k, raising=False)
    m = GmlxAPCManager(num_blocks=64, block_size=16)
    assert cs._record_room(m) == (cs._CKPT_BUDGET_BYTES, False)

    _autosize_rig(monkeypatch, budget=100 * GB)
    m.autosize(_fake_model())
    pool_cap = m.num_blocks * m.block_size * m._pool_per_token_bytes
    share = 100 * GB * am._POOL_BUDGET_FRACTION
    assert m._kq_ckpt_share_bytes == int(max(share, pool_cap))
    assert share <= m._kq_ckpt_share_bytes < share + 16 * m._pool_per_token_bytes
    assert cs._record_room(m) == (m._kq_ckpt_share_bytes, True)

    # A pool already past the share: the share is the pool's cap.
    _autosize_rig(monkeypatch, budget=8 << 20)
    m = GmlxAPCManager(num_blocks=64, block_size=16)
    m.autosize(_fake_model())
    assert m._kq_ckpt_share_bytes == int(64 * 16 * m._pool_per_token_bytes)

    monkeypatch.setenv("APC_NUM_BLOCKS", "64")
    _autosize_rig(monkeypatch, budget=100 * GB)
    m = GmlxAPCManager(num_blocks=64, block_size=16)
    m.autosize(_fake_model())
    assert m.num_blocks == 64
    assert m._kq_ckpt_share_bytes == int(64 * 16 * m._pool_per_token_bytes)


def test_a_retirement_no_lookup_can_use_stores_nothing():
    """The arming recorded the layout the request's lookup signed with. A
    retirement whose cache tags another layout (QSA layers out of a merge
    without their indexer keys tag "kv") stores nothing and counts a
    retire_layout decline; one that matches stores."""
    man = GmlxAPCManager(num_blocks=64, block_size=16)
    ids = _ids(0, 52)
    cache = make_hybrid_cache(52, seed=4)
    live = cs.ckpt_layout(cache, 16)
    signed = tuple("qsa:4" if t == "kv" else t for t in live)
    assert cs.retirement_store(man, "ckpt", ids, cache, layout=signed) == 0
    assert len(cs._ckpt_records(man)) == 0
    assert man.stats_snapshot()["ckpt_declines"] == {"retire_layout": 1}
    assert cs.retirement_store(man, "ckpt", ids, cache, layout=live) == 52
    assert [r.p for r in cs._ckpt_records(man).values()] == [52]
    # No signature recorded (an unsupported live cache at arming): no check.
    assert cs.retirement_store(man, "ckpt", _ids(1, 52), cache,
                               layout=None) == 52


def test_the_counters_reach_the_stats_and_the_metrics_payload(monkeypatch):
    """Both counters sit in stats_snapshot(), which /v1/cache/stats
    returns, and in the server snapshot /v1/metrics carries as "server":
    server.apc.ckpt_budget_evictions and server.apc.ckpt_budget_evicted_mb.
    The stock snapshot is read under any wrap the session installed."""
    import importlib

    # The package exports the FastAPI app under the module's name.
    app = importlib.import_module("gmlx._vendor.mlx_vlm.server.app")

    man = GmlxAPCManager(num_blocks=64, block_size=16)
    snap = man.stats_snapshot()
    assert snap["ckpt_budget_evictions"] == 0
    assert snap["ckpt_budget_evicted_mb"] == 0
    assert "_budget_evicted_bytes" not in snap

    cs._ckpt_bump(man, "ckpt_budget_evictions", 2)
    cs._ckpt_bump(man, "ckpt_budget_evicted_mb", 3)
    stock = app._server_runtime_snapshot
    while getattr(stock, "_gmlx_orig", None) is not None:
        stock = stock._gmlx_orig
    for name, value in (("apc_manager", man), ("model_cache", {}),
                        ("response_generator", None), ("audio_queue", None)):
        monkeypatch.setattr(app.runtime, name, value)
    apc = stock()["apc"]
    assert apc["enabled"] is True
    assert apc["ckpt_budget_evictions"] == 2
    assert apc["ckpt_budget_evicted_mb"] == 3
