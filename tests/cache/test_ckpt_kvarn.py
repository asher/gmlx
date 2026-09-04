"""Checkpoint tier x kvarn: layout truth table, blockless records,
cross-boot refusal.

CPU tests use real empty ``KVarNKVCache`` instances (the constructor is
pure Python and the layout tag keys on the concrete class, so duck-typed
stubs would never fire it); a "hollow" cache with only ``offset`` set
exercises the full store/lookup machinery through the unallocated-state
placeholder round trip. Filling a kvarn cache needs the Metal kernels, so
content-equality round trips are GPU-gated.
"""

import mlx.core as mx
import numpy as np
import pytest

from gmlx.cache.apc_manager import GmlxAPCManager
from gmlx.cache.compat import runtime_cache_module
from gmlx.cache.snapshot import ckpt_layout, ckpt_lookup, ckpt_store
from gmlx.cache.kvarn_cache import (
    BatchKVarNKVCache,
    KVarNKVCache,
    KVarNRotatingKVCache,
)

from test_ckpt_tier import D, H, make_hybrid_cache
from kvarn_testlib import needs_kvarn_ops, tokens

_cache = runtime_cache_module()
ArraysCache = _cache.ArraysCache
KVCache = _cache.KVCache
RotatingKVCache = _cache.RotatingKVCache

KVARN_TAG = "kvarn:6:6:1024"


def _ids(p):
    return list(range(500, 500 + p))


def _arr(seed=0):
    c = ArraysCache(size=2)
    mx.random.seed(seed)
    c.cache = [mx.random.normal((1, 3, D)),
               mx.random.normal((1, H, D, D))]
    return c


def _hollow_kvarn(p, **kw):
    c = KVarNKVCache(**kw)
    c.offset = p
    return c


def _kv(p):
    c = KVCache()
    c.update_and_fetch(mx.zeros((1, H, p, D)), mx.zeros((1, H, p, D)))
    return c


def test_layout_truth_table():
    kvarn = KVarNKVCache()
    assert ckpt_layout([kvarn, ArraysCache(size=2)]) == [KVARN_TAG, "arr"]
    assert ckpt_layout([KVarNKVCache(), RotatingKVCache(max_size=32)]) == \
        [KVARN_TAG, "rot:32:0"]
    assert ckpt_layout(
        [KVarNKVCache(), RotatingKVCache(max_size=32), ArraysCache(size=2)]
    ) == [KVARN_TAG, "rot:32:0", "arr"]
    # Pure-attention stacks stay on the block/exact tiers.
    assert ckpt_layout([KVarNKVCache()]) is None
    assert ckpt_layout([KVarNKVCache(), KVCache()]) is None
    assert ckpt_layout([ArraysCache(size=2), ArraysCache(size=2)]) is None


def test_layout_tag_carries_wire_config():
    c = KVarNKVCache(k_bits=5, v_bits=4, tail_tokens=256)
    assert ckpt_layout([c, ArraysCache(size=2)]) == ["kvarn:5:4:256", "arr"]


def test_layout_declines_rotating_and_batch_kvarn():
    rot = KVarNRotatingKVCache(max_size=2048)
    assert ckpt_layout([rot, ArraysCache(size=2)]) is None
    bat = BatchKVarNKVCache(left_padding=[0])
    assert ckpt_layout([bat, ArraysCache(size=2)]) is None


@pytest.mark.parametrize("p", [17, 32, 48])
def test_blockless_store_lookup_roundtrip(p):
    man = GmlxAPCManager(num_blocks=8, block_size=16)
    cache = [_hollow_kvarn(p), _arr(seed=p)]
    assert ckpt_store(man, _ids(p), cache, extra_hash=7)
    warm, got = ckpt_lookup(man, _ids(p) + [999, 998], extra_hash=7,
                            layout=(KVARN_TAG, "arr"))
    assert got == p
    assert type(warm[0]) is KVarNKVCache
    assert warm[0].offset == p and warm[0].k_bits == 6
    for a, b in zip(cache[1].cache, warm[1].cache):
        assert mx.array_equal(a, b).item()
    # Blockless record: nothing pinned in the pool.
    assert all(b.block_hash is None for b in man.pool)


def test_blockless_records_exempt_from_strip_on_extend():
    # A blockless record is its boundary's only carrier: no main chain
    # for skeleton re-index to re-cut, so strip-on-extend must not touch
    # it -- otherwise a divergent suffix or next turn has nothing to
    # adopt below the terminal (fp16 recovers via chain dedup; kvarn
    # cannot).
    man = GmlxAPCManager(num_blocks=8, block_size=16)
    for p in (32, 48, 64, 80):
        assert ckpt_store(man, _ids(p), [_hollow_kvarn(p), _arr(seed=p)],
                          extra_hash=7)
    from gmlx.cache.snapshot import _ckpt_records
    recs = _ckpt_records(man)
    assert sorted(r.p for r in recs.values()) == [32, 48, 64, 80]
    # Divergent suffix (shared 40-token prefix): adopts the shallow
    # boundary the deeper stores would have stripped.
    warm, got = ckpt_lookup(man, _ids(40) + [777] * 8, extra_hash=7,
                            layout=(KVARN_TAG, "arr"))
    assert got == 32 and type(warm[0]) is KVarNKVCache
    warm, got = ckpt_lookup(man, _ids(80) + [999], extra_hash=7,
                            layout=(KVARN_TAG, "arr"))
    assert got == 80 and type(warm[0]) is KVarNKVCache


def test_carve_out_stack_records_exempt_from_strip_on_extend():
    # The production shape: the fp16-held last layer beside the kvarn
    # layers gives the record a main chain, but that chain cannot restore
    # the kvarn rows, so the record stays exempt.
    man = GmlxAPCManager(num_blocks=16, block_size=16)
    layout = (KVARN_TAG, "kv", "arr")
    for p in (32, 48, 64, 80):
        cache = [_hollow_kvarn(p), _kv(p), _arr(seed=p)]
        assert ckpt_layout(cache) == list(layout)
        assert ckpt_store(man, _ids(p), cache, extra_hash=7)
    from gmlx.cache.snapshot import _ckpt_records
    assert sorted(r.p for r in _ckpt_records(man).values()) == [32, 48, 64, 80]
    warm, got = ckpt_lookup(man, _ids(56) + [777] * 8, extra_hash=7,
                            layout=layout)
    assert got == 48 and type(warm[0]) is KVarNKVCache
    assert warm[1].offset == 48


def test_blockless_records_release_under_byte_budget(monkeypatch):
    import gmlx.cache.snapshot as cs

    man = GmlxAPCManager(num_blocks=8, block_size=16)
    for p in (32, 48):
        assert ckpt_store(man, _ids(p), [_hollow_kvarn(p), _arr(seed=p)],
                          extra_hash=7)
    recs = cs._ckpt_records(man)
    nb = max(r.nbytes for r in recs.values())
    monkeypatch.setattr(cs, "_CKPT_BUDGET_BYTES", int(nb * 1.5))
    assert ckpt_store(man, _ids(64), [_hollow_kvarn(64), _arr(seed=64)],
                      extra_hash=7)
    # The budget, not strip-on-extend, bounds blockless retention: the
    # newest record survives, the oldest are released.
    assert {r.p for r in cs._ckpt_records(man).values()} == {64}


@needs_kvarn_ops
def test_record_bytes_count_content():
    from gmlx.cache.snapshot import _caches_nbytes, _clone_row_faithful

    c = KVarNKVCache(tail_tokens=256)
    c.update_and_fetch(*tokens(200))
    clone = _clone_row_faithful(c)
    assert _caches_nbytes([clone]) == clone.nbytes < c.nbytes / 3


def test_offset_gate_declines_stale_kvarn():
    man = GmlxAPCManager(num_blocks=8, block_size=16)
    assert not ckpt_store(man, _ids(32), [_hollow_kvarn(31), _arr()],
                          extra_hash=0)
    assert man.stats_snapshot()["ckpt_declines"] == {"offset": 1}


def test_cross_boot_refusal_both_directions():
    # Stock record cannot be adopted by a kvarn-boot layout...
    man = GmlxAPCManager(num_blocks=64, block_size=16)
    assert ckpt_store(man, _ids(32), make_hybrid_cache(32), extra_hash=0)
    kvarn_layout = (KVARN_TAG, "arr", "arr", "kvarn:6:6:1024", "arr")
    warm, got = ckpt_lookup(man, _ids(32) + [999], extra_hash=0,
                            layout=kvarn_layout)
    assert warm is None and got == 0
    # ...and a kvarn record cannot be adopted by a stock-boot layout.
    man2 = GmlxAPCManager(num_blocks=8, block_size=16)
    assert ckpt_store(man2, _ids(32), [_hollow_kvarn(32), _arr()],
                      extra_hash=0)
    warm, got = ckpt_lookup(man2, _ids(32) + [999], extra_hash=0,
                            layout=("kv", "arr"))
    assert warm is None and got == 0


def test_wire_config_mismatch_refuses():
    man = GmlxAPCManager(num_blocks=8, block_size=16)
    assert ckpt_store(man, _ids(32), [_hollow_kvarn(32), _arr()],
                      extra_hash=0)
    warm, got = ckpt_lookup(man, _ids(32) + [999], extra_hash=0,
                            layout=("kvarn:6:5:1024", "arr"))
    assert warm is None and got == 0


def test_kvarn_disk_restart_roundtrip(tmp_path):
    # Skeleton write -> process restart -> repair from disk. Exercises
    # the kvarn tag re-derivation on the loaded entries (without it a
    # restored KVarNKVCache classifies as "arr" and the layout check
    # permanently misses).
    from mlx_vlm.apc import DiskBlockStore

    from gmlx.cache.kvarn_apc import install_kvarn_apc

    install_kvarn_apc()  # the kq_kvarn disk kind (serve installs at boot)
    p = 48
    disk = DiskBlockStore(root=tmp_path, namespace="m")
    man = GmlxAPCManager(num_blocks=8, block_size=16, disk=disk)
    assert ckpt_store(man, _ids(p), [_hollow_kvarn(p), _arr(seed=p)],
                      extra_hash=5, skeleton_disk=True)
    disk.close()
    disk2 = DiskBlockStore(root=tmp_path, namespace="m")
    man2 = GmlxAPCManager(num_blocks=8, block_size=16, disk=disk2)
    try:
        warm, got = ckpt_lookup(man2, _ids(p) + [999], extra_hash=5,
                                layout=(KVARN_TAG, "arr"))
        assert got == p
        assert type(warm[0]) is KVarNKVCache and warm[0].offset == p
        # Wrong wire config must cold-miss, not adopt.
        warm, got = ckpt_lookup(man2, _ids(p) + [999], extra_hash=5,
                                layout=("kvarn:6:5:1024", "arr"))
        assert warm is None and got == 0
    finally:
        disk2.close()


def test_kvarn_rot_disk_restart_roundtrip(tmp_path):
    # kvarn+rot: boundary/turn-kind skeletons are the only restart-
    # restorable records (replay skeletons are heavy-suppressed), so the
    # rot window chain must persist alongside them -- without it the
    # skeleton loads and then dies at the window-chain lookup.
    from mlx_vlm.apc import DiskBlockStore

    from gmlx.cache.kvarn_apc import install_kvarn_apc

    install_kvarn_apc()
    p = 48
    w = 32
    rot = RotatingKVCache(max_size=w)
    mx.random.seed(9)
    rot.update_and_fetch(mx.random.normal((1, H, p, D)),
                         mx.random.normal((1, H, p, D)))
    cache = [_hollow_kvarn(p), rot]
    ids = list(range(500, 500 + p))
    disk = DiskBlockStore(root=tmp_path, namespace="m")
    man = GmlxAPCManager(num_blocks=64, block_size=16, disk=disk)
    assert ckpt_store(man, ids, cache, extra_hash=5, kind="boundary",
                      skeleton_disk=True)
    disk.close()
    disk2 = DiskBlockStore(root=tmp_path, namespace="m")
    man2 = GmlxAPCManager(num_blocks=64, block_size=16, disk=disk2)
    try:
        warm, got = ckpt_lookup(man2, ids + [999], extra_hash=5,
                                layout=(KVARN_TAG, f"rot:{w}:0"))
        assert got == p
        assert type(warm[0]) is KVarNKVCache and warm[0].offset == p
        assert type(warm[1]) is RotatingKVCache and warm[1].offset == p
    finally:
        disk2.close()


def test_kvarn_disk_write_mirrors_wire_salt(tmp_path):
    # The skeleton writer bypasses the manager's exact-cache wrapper, so
    # it must fold _exact_extra_salt into the hash AND the persisted
    # extra itself; the salted lookup side then matches after a restart
    # on the same wire config and misses across configs.
    from mlx_vlm.apc import DiskBlockStore

    from gmlx.cache.kvarn_apc import install_kvarn_apc

    install_kvarn_apc()
    p = 32
    disk = DiskBlockStore(root=tmp_path, namespace="m")
    man = GmlxAPCManager(num_blocks=8, block_size=16, disk=disk)
    man._exact_extra_salt = 0xA5A5
    assert ckpt_store(man, _ids(p), [_hollow_kvarn(p), _arr(seed=p)],
                      extra_hash=5, skeleton_disk=True)
    disk.close()
    disk2 = DiskBlockStore(root=tmp_path, namespace="m")
    man2 = GmlxAPCManager(num_blocks=8, block_size=16, disk=disk2)
    man2._exact_extra_salt = 0xA5A5
    try:
        warm, got = ckpt_lookup(man2, _ids(p) + [999], extra_hash=5,
                                layout=(KVARN_TAG, "arr"))
        assert got == p and type(warm[0]) is KVarNKVCache
    finally:
        disk2.close()
    disk3 = DiskBlockStore(root=tmp_path, namespace="m")
    man3 = GmlxAPCManager(num_blocks=8, block_size=16, disk=disk3)
    man3._exact_extra_salt = 0x5A5A
    try:
        warm, got = ckpt_lookup(man3, _ids(p) + [999], extra_hash=5,
                                layout=(KVARN_TAG, "arr"))
        assert warm is None and got == 0
    finally:
        disk3.close()


@needs_kvarn_ops
@pytest.mark.parametrize("p", [200, 300])
def test_filled_kvarn_roundtrip_content_equal(p):
    man = GmlxAPCManager(num_blocks=8, block_size=16)
    kv = KVarNKVCache(tail_tokens=256)
    kv.update_and_fetch(*tokens(p, seed=p))
    tag = "kvarn:6:6:256"
    cache = [kv, _arr(seed=p)]
    assert ckpt_store(man, _ids(p), cache, extra_hash=3)
    ref = [np.array(m) for m in kv.materialize()]
    # The live cache keeps decoding after the store.
    kv.update_and_fetch(*tokens(40, seed=p + 1))
    warm, got = ckpt_lookup(man, _ids(p) + [999, 998], extra_hash=3,
                            layout=(tag, "arr"))
    assert got == p
    assert warm[0].offset == p
    assert warm[0].n_sealed == (p - warm[0].sink_cap) // 128
    for got_m, want in zip(warm[0].materialize(), ref, strict=True):
        assert np.array_equal(np.array(got_m), want)


# -- mixed rotating stacks (the carve-out makes them heterogeneous) -----------


def test_mixed_rotating_kvarn_stack_declines_ckpt():
    """--max-kv-size under kvarn now yields N-1 KVarNRotatingKVCache beside
    one bare RotatingKVCache (the shared carve-out). The ckpt tier tags the
    plain kvarn class only, so the mixed stack must decline outright rather
    than tag the rotating layers as if they carried no window."""
    stack = [KVarNRotatingKVCache(4096, tail_tokens=1024),
             RotatingKVCache(max_size=4096)]
    assert ckpt_layout(stack) is None
    # ... and a rotating kvarn layer beside a state layer declines too.
    assert ckpt_layout([KVarNRotatingKVCache(4096, tail_tokens=1024),
                        ArraysCache(size=2)]) is None


def test_rotating_kvarn_declines_the_disk_snapshot():
    """The disk arm rebuilds through KVarNKVCache.from_state, whose meta
    arity the rotating subclass does not match: it must never be written
    under the plain kvarn kind."""
    from gmlx.cache.kvarn_apc import install_kvarn_apc

    install_kvarn_apc()
    import mlx_vlm.apc as apc

    rot = KVarNRotatingKVCache(4096, tail_tokens=1024)
    arrays, metadata = {}, {}
    apc.DiskBlockStore._snapshot_exact_cache_entry(
        None, rot, "l0", arrays, metadata)
    assert metadata.get("l0_kind") != "kq_kvarn"
    plain = KVarNKVCache(tail_tokens=1024)
    arrays, metadata = {}, {}
    assert apc.DiskBlockStore._snapshot_exact_cache_entry(
        None, plain, "l0", arrays, metadata)
    assert metadata["l0_kind"] == "kq_kvarn"
