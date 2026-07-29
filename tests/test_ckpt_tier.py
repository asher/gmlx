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
    trimmed = []
    for c in cache:
        if isinstance(c, KVCache):
            t = KVCache()
            t.state = (c.keys[..., :p2, :], c.values[..., :p2, :])
            trimmed.append(t)
        else:
            trimmed.append(c)
    assert ckpt_store(man, ids[:p2], trimmed, extra_hash=0)
    warm, got = ckpt_lookup(man, ids, extra_hash=0)
    assert got == p2
    assert_warm_matches(warm, trimmed, p2)


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


@pytest.mark.parametrize("p", [16, 48, 64])   # below and beyond the W=32 wrap
def test_swa_store_lookup_roundtrip(p):
    man = APCManager(num_blocks=64, block_size=16)
    cache = make_swa_cache(p, seed=p)
    ids = list(range(300, 300 + p))
    assert ckpt_store(man, ids, cache, extra_hash=9)
    warm, got = ckpt_lookup(man, ids + [1, 2], extra_hash=9)
    assert got == p
    assert_swa_warm_matches(warm, cache, p)


def test_swa_store_declines_off_grid():
    man = APCManager(num_blocks=64, block_size=16)
    p = 40                                    # not a block multiple
    cache = make_swa_cache(p)
    assert not ckpt_store(man, list(range(300, 300 + p)), cache)
    assert all(b.ref_cnt == 0 for b in man.pool)


def test_retirement_rotating_falls_back_to_exact():
    man = APCManager(num_blocks=64, block_size=16)
    p = 40                                    # unaligned: ckpt declines
    cache = make_swa_cache(p, seed=3)
    ids = list(range(300, 300 + p))
    assert retirement_store(man, "ckpt", ids, cache, row=0, extra_hash=1)
    # landed on the exact tier (verbatim row with rotation meta)
    entry, plen = man.lookup_exact_cache(ids + [1], extra_hash=1)
    assert plen == p and entry is not None
    from gmlx.cache_snapshot import ckpt_extra_hash as _ceh
    e2, p2 = man.lookup_exact_cache(ids + [1], extra_hash=_ceh(1))
    assert e2 is None and p2 == 0             # nothing under the ckpt salt


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


def test_strip_on_extend_keeps_newest_two():
    from gmlx.cache_snapshot import _ckpt_records
    man = APCManager(num_blocks=64, block_size=16)
    ids = list(range(400, 400 + 96))
    for p in (32, 48, 64):
        cache = make_hybrid_cache(p, seed=p)
        assert ckpt_store(man, ids[:p], cache, extra_hash=0)
    idx = _ckpt_records(man)
    assert sorted(r.p for r in idx.values()) == [48, 64]
    warm, got = ckpt_lookup(man, ids[:40], extra_hash=0)
    assert warm is None and got == 0          # p=32 stripped
    warm, got = ckpt_lookup(man, ids[:66], extra_hash=0)
    assert got == 64


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


def test_swa_disk_restart_roundtrip(tmp_path):
    p = 48
    cache = make_swa_cache(p, seed=11)
    ids = list(range(700, 700 + p))
    disk = DiskBlockStore(root=tmp_path, namespace="m")
    man = APCManager(num_blocks=64, block_size=16, disk=disk)
    assert ckpt_store(man, ids, cache, extra_hash=4)
    disk.close()
    disk2 = DiskBlockStore(root=tmp_path, namespace="m")
    man2 = APCManager(num_blocks=64, block_size=16, disk=disk2)
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


def test_window_chain_is_memory_only(tmp_path):
    """Position-salted window shards cannot dedup on disk; the gmlx
    manager schedules disk writes for the main chain only."""
    from gmlx.apc_manager import GmlxAPCManager
    disk = DiskBlockStore(root=tmp_path, namespace="m")
    man = GmlxAPCManager(num_blocks=64, block_size=16, disk=disk)
    try:
        p = 48
        cache = make_swa_cache(p, seed=5)
        ids = list(range(600, 600 + p))
        assert ckpt_store(man, ids, cache, extra_hash=0)
        # 3 main blocks; the 2 window blocks stay memory-only.
        assert man.stats_snapshot()["disk_writes"] == 3
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
