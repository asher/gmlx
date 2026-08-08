"""Ckpt-tier counters, routing observability, and production tripwires.

The 2026-08 depth e2e proved the tier can be dead for months while
/v1/cache/stats reads healthy: pinned-record stores/hits bumped nothing.
These tests pin the counter contract -- every decline lands in a named
bucket, hits feed both the ckpt_* side ledger and the upstream token
ledger, and the two tripwires fire exactly once on the failure classes
that hid (zero stores while armed; stores and lookups never
intersecting) while staying silent on healthy one-shot traffic.
"""

import logging

import mlx.core as mx
import pytest
from mlx_vlm.apc import APCManager, DiskBlockStore

from gmlx.apc_manager import GmlxAPCManager
from gmlx.cache_snapshot import (
    ckpt_lookup,
    ckpt_note_armed,
    ckpt_store,
    drafter_sidecar_store,
)
from test_ckpt_tier import KVCache, make_hybrid_cache, make_swa_cache


def _ids(p, base=100):
    return list(range(base, base + p))


class _Drafter:
    supports_kv_sidecar = True

    def __init__(self, p):
        c = KVCache()
        c.state = (mx.random.normal((1, 2, p, 8)),
                   mx.random.normal((1, 2, p, 8)))
        self._kv = [c]

    def export_kv(self):
        return self._kv


def test_store_and_hit_bump_both_ledgers():
    man = GmlxAPCManager(num_blocks=64, block_size=16)
    p = 33
    assert ckpt_store(man, _ids(p), make_hybrid_cache(p, seed=1),
                      extra_hash=0)
    snap = man.stats_snapshot()
    assert snap["ckpt_stores"] == 1
    assert snap["ckpt_hits"] == 0
    warm, got = ckpt_lookup(man, _ids(p) + [1], extra_hash=0)
    assert got == p
    snap = man.stats_snapshot()
    assert snap["ckpt_hits"] == 1
    assert snap["ckpt_matched_tokens"] == p
    # A pinned-record hit is cache-served tokens: the upstream ledger
    # must see it too, or token_hit_rate lies on ckpt models.
    assert snap["lookups_hit"] == 1
    assert snap["matched_tokens"] == p


@pytest.mark.parametrize("case,reason", [
    ("offset", "offset"),
    ("grid", "grid"),
    ("layout", "layout"),
    ("short_chain", "short_chain"),
])
def test_decline_reasons_land_in_distinct_buckets(case, reason):
    if case == "short_chain":
        man = GmlxAPCManager(num_blocks=2, block_size=16)
        assert ckpt_store(man, _ids(32), make_hybrid_cache(32, seed=1),
                          extra_hash=0)          # pins the whole pool
        assert not ckpt_store(man, _ids(64), make_hybrid_cache(64, seed=2),
                              extra_hash=0)
    else:
        man = GmlxAPCManager(num_blocks=64, block_size=16)
        if case == "offset":
            assert not ckpt_store(man, _ids(32), make_hybrid_cache(33),
                                  extra_hash=0)
        elif case == "grid":
            assert not ckpt_store(man, _ids(40), make_swa_cache(40),
                                  extra_hash=0)
        elif case == "layout":
            assert not ckpt_store(man, _ids(32), [KVCache(), KVCache()],
                                  extra_hash=0)
    declines = man.stats_snapshot()["ckpt_declines"]
    assert declines == {reason: 1}, declines


def test_skeleton_write_counts_on_both_ledgers(tmp_path):
    disk = DiskBlockStore(root=tmp_path, namespace="m")
    man = GmlxAPCManager(num_blocks=64, block_size=16, disk=disk)
    try:
        p = 33
        assert ckpt_store(man, _ids(p), make_hybrid_cache(p, seed=2),
                          extra_hash=0)
        snap = man.stats_snapshot()
        assert snap["ckpt_skeleton_writes"] == 1
        # 2 main-chain shard blocks (b_full=32) + the skeleton entry.
        assert snap["disk_writes"] == 3
    finally:
        disk.close()


def test_sidecar_write_counts_on_both_ledgers(tmp_path):
    disk = DiskBlockStore(root=tmp_path, namespace="m")
    man = GmlxAPCManager(num_blocks=64, block_size=16, disk=disk)
    try:
        assert drafter_sidecar_store(man, _Drafter(32), _ids(32), 32,
                                     extra_hash=0)
        snap = man.stats_snapshot()
        assert snap["sidecar_writes"] == 1
        assert snap["disk_writes"] == 1
    finally:
        disk.close()


def test_reset_stats_keeps_records():
    man = GmlxAPCManager(num_blocks=64, block_size=16)
    p = 32
    assert ckpt_store(man, _ids(p), make_hybrid_cache(p, seed=3),
                      extra_hash=0)
    man.reset_stats()
    snap = man.stats_snapshot()
    assert snap["ckpt_stores"] == 0
    warm, got = ckpt_lookup(man, _ids(p) + [1], extra_hash=0)
    assert got == p                      # records survive a stats reset
    assert man.stats_snapshot()["ckpt_hits"] == 1


def test_clear_drops_records_and_counters():
    man = GmlxAPCManager(num_blocks=64, block_size=16)
    p = 32
    assert ckpt_store(man, _ids(p), make_hybrid_cache(p, seed=4),
                      extra_hash=0)
    man.clear()
    snap = man.stats_snapshot()
    assert snap["ckpt_stores"] == 0 and snap["ckpt_declines"] == {}
    # The pool wipe invalidated the pinned chains; the records must not
    # survive to assemble from zeroed blocks.
    assert not getattr(man, "_kq_ckpt_records")
    assert all(b.ref_cnt == 0 for b in man.pool)
    warm, got = ckpt_lookup(man, _ids(p) + [1], extra_hash=0)
    assert warm is None and got == 0


def test_mode_none_warns_once_with_class_names(monkeypatch, caplog):
    import gmlx.spec_engine as se

    se._bind_l1_view()
    monkeypatch.setattr(se, "_SPEC_APC_DISABLED", False)

    class _MSAKVCacheStandIn:
        pass

    class _Model:
        def make_cache(self):
            return [_MSAKVCacheStandIn()]

    model = _Model()
    model._kq_apc_manager = APCManager(num_blocks=4, block_size=16)
    with caplog.at_level(logging.WARNING, logger="gmlx.spec_engine"):
        assert se._resolve_l1(model) == (None, None)
        assert se._resolve_l1(model) == (None, None)
    warns = [r for r in caplog.records if "APC OFF" in r.getMessage()]
    assert len(warns) == 1
    assert "_MSAKVCacheStandIn" in warns[0].getMessage()


def test_ckpt_tier_log_once(monkeypatch, caplog):
    import gmlx.spec_engine as se
    from test_ckpt_tier import ArraysCache

    monkeypatch.setattr(se, "_SPEC_APC_CKPT_DISABLED", False)

    class _Hybrid:
        def make_cache(self):
            return [KVCache(), ArraysCache(size=2)]

    m = _Hybrid()
    with caplog.at_level(logging.INFO, logger="gmlx.spec_engine"):
        assert se._ckpt_active(m, "exact") is True
        assert se._ckpt_active(m, "exact") is True
    logs = [r for r in caplog.records if "APC tier: ckpt" in r.getMessage()]
    assert len(logs) == 1


def _tripwire_records(caplog):
    return [r for r in caplog.records if "tripwire" in r.getMessage()]


def test_tripwire_fires_once_on_armed_without_stores(monkeypatch, caplog):
    monkeypatch.setenv("GMLX_APC_CKPT_TRIPWIRE", "3")
    man = APCManager(num_blocks=4, block_size=16)
    with caplog.at_level(logging.WARNING, logger="gmlx.cache_snapshot"):
        for _ in range(3):
            ckpt_note_armed(man)
        assert not _tripwire_records(caplog)
        ckpt_note_armed(man)
        ckpt_note_armed(man)
    fired = _tripwire_records(caplog)
    assert len(fired) == 1
    assert "zero checkpoint stores" in fired[0].getMessage()


def test_tripwire_silent_when_stores_land(monkeypatch, caplog):
    monkeypatch.setenv("GMLX_APC_CKPT_TRIPWIRE", "2")
    man = APCManager(num_blocks=64, block_size=16)
    assert ckpt_store(man, _ids(32), make_hybrid_cache(32, seed=5),
                      extra_hash=0)
    with caplog.at_level(logging.WARNING, logger="gmlx.cache_snapshot"):
        for _ in range(10):
            ckpt_note_armed(man)
    assert not _tripwire_records(caplog)


def test_tripwire_fires_once_on_missed_adoptions(monkeypatch, caplog):
    monkeypatch.setenv("GMLX_APC_CKPT_TRIPWIRE", "3")
    man = APCManager(num_blocks=64, block_size=16)
    p = 32
    assert ckpt_store(man, _ids(p), make_hybrid_cache(p, seed=6),
                      extra_hash=0)
    with caplog.at_level(logging.WARNING, logger="gmlx.cache_snapshot"):
        for _ in range(4):
            # Identical resend: the record prefixes the query but the
            # strict p-bound refuses it -- the bug-1 signature.
            warm, got = ckpt_lookup(man, _ids(p), extra_hash=0)
            assert warm is None and got == 0
    snap_missed = man._kq_ckpt_stats["ckpt_missed_adoptions"]
    assert snap_missed == 4
    fired = _tripwire_records(caplog)
    assert len(fired) == 1
    assert "adopted nothing" in fired[0].getMessage()


def test_tripwire_silent_on_unrelated_traffic(monkeypatch, caplog):
    """The cry-wolf case: stores exist, lookups are unrelated one-shot
    prompts. There is nothing to hit; the tripwire must not train
    operators to filter it."""
    monkeypatch.setenv("GMLX_APC_CKPT_TRIPWIRE", "2")
    man = APCManager(num_blocks=64, block_size=16)
    assert ckpt_store(man, _ids(32), make_hybrid_cache(32, seed=7),
                      extra_hash=0)
    with caplog.at_level(logging.WARNING, logger="gmlx.cache_snapshot"):
        for i in range(6):
            warm, got = ckpt_lookup(man, _ids(40, base=5000 * (i + 1)),
                                    extra_hash=0)
            assert warm is None and got == 0
    assert man._kq_ckpt_stats["ckpt_missed_adoptions"] == 0
    assert not _tripwire_records(caplog)
