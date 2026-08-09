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

from gmlx.apc_manager import GmlxAPCManager
from gmlx.cache_compat import runtime_cache_module
from gmlx.cache_snapshot import ckpt_layout, ckpt_lookup, ckpt_store
from gmlx.kvarn_cache import (
    BatchKVarNKVCache,
    KVarNKVCache,
    KVarNRotatingKVCache,
)

from test_ckpt_tier import D, H, make_hybrid_cache

_cache = runtime_cache_module()
ArraysCache = _cache.ArraysCache
KVCache = _cache.KVCache
RotatingKVCache = _cache.RotatingKVCache

_NEEDS_GPU = pytest.mark.skipif(
    mx.default_device() != mx.gpu,
    reason="kvarn kernels are Metal-only; needs the GPU device",
)

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


def test_blockless_strip_on_extend_releases_cleanly():
    man = GmlxAPCManager(num_blocks=8, block_size=16)
    for p in (32, 48, 64, 80):
        assert ckpt_store(man, _ids(p), [_hollow_kvarn(p), _arr(seed=p)],
                          extra_hash=7)
    from gmlx.cache_snapshot import _ckpt_records
    recs = _ckpt_records(man)
    # Heavy-per-chain retention: deepest kept, chain bounded.
    assert max(r.p for r in recs.values()) == 80
    warm, got = ckpt_lookup(man, _ids(80) + [999], extra_hash=7,
                            layout=(KVARN_TAG, "arr"))
    assert got == 80 and type(warm[0]) is KVarNKVCache


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


def _tokens(n, seed=0):
    rng = np.random.default_rng(seed)
    k = mx.array(rng.standard_normal((1, H, n, 128)).astype(np.float16))
    v = mx.array(rng.standard_normal((1, H, n, 128)).astype(np.float16))
    return k, v


@_NEEDS_GPU
@pytest.mark.parametrize("p", [200, 300])
def test_filled_kvarn_roundtrip_content_equal(p):
    man = GmlxAPCManager(num_blocks=8, block_size=16)
    kv = KVarNKVCache(tail_tokens=256)
    kv.update_and_fetch(*_tokens(p, seed=p))
    tag = "kvarn:6:6:256"
    cache = [kv, _arr(seed=p)]
    assert ckpt_store(man, _ids(p), cache, extra_hash=3)
    ref = [np.array(m) for m in kv.materialize()]
    # The live cache keeps decoding after the store.
    kv.update_and_fetch(*_tokens(40, seed=p + 1))
    warm, got = ckpt_lookup(man, _ids(p) + [999, 998], extra_hash=3,
                            layout=(tag, "arr"))
    assert got == p
    assert warm[0].offset == p
    assert warm[0].n_sealed == (p - warm[0].sink_cap) // 128
    for got_m, want in zip(warm[0].materialize(), ref, strict=True):
        assert np.array_equal(np.array(got_m), want)
