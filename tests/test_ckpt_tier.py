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
    assert not ckpt_supported([KVCache(), RotatingKVCache(max_size=8)])
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
    # All acquired blocks were released after materialization.
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
