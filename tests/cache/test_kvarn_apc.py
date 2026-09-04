"""kvarn APC exact tier: extract/merge batch ops, the clone/merge/mode
arms, the kq_kvarn disk kind (round trip + version fail-closed), the
manager entry salt, and snapshot content probes."""

from __future__ import annotations

import time

import numpy as np
import pytest

import mlx.core as mx

pytest.importorskip("mlx_vlm.apc")

from mlx_vlm import apc  # noqa: E402
from mlx_vlm.models.cache import KVCache  # noqa: E402

from gmlx.cache import kvarn_apc  # noqa: E402
from gmlx.cache.kvarn_cache import BatchKVarNKVCache, KVarNKVCache  # noqa: E402
from kvarn_testlib import D, H, needs_kvarn_ops  # noqa: E402


_ARM_NAMES = (
    "_cache_entry_supports_exact_apc",
    "_merge_exact_cache_entries",
    "_clone_cache_entry_for_apc",
    "_safetensors_dtype_info",
    "model_apc_mode",
)


@pytest.fixture
def restorable(monkeypatch):
    for name in _ARM_NAMES:
        monkeypatch.setattr(apc, name, getattr(apc, name))
    monkeypatch.setattr(
        apc.DiskBlockStore,
        "_snapshot_exact_cache_entry",
        apc.DiskBlockStore._snapshot_exact_cache_entry,
    )
    monkeypatch.setattr(
        apc.DiskBlockStore,
        "_load_exact_cache_entry",
        apc.DiskBlockStore._load_exact_cache_entry,
    )
    monkeypatch.setattr(apc, kvarn_apc._FLAG, False, raising=False)
    monkeypatch.delenv("KV_QUANT_SCHEME", raising=False)
    monkeypatch.delenv("KV_BITS", raising=False)
    monkeypatch.delenv("KV_TAIL_TOKENS", raising=False)
    return monkeypatch


def _slab(n, b=1, seed=0):
    rng = np.random.default_rng(seed)
    k = mx.array(rng.standard_normal((b, H, n, D)).astype(np.float16))
    v = mx.array(rng.standard_normal((b, H, n, D)).astype(np.float16))
    return k, v


def _single(n, seed=0, tail=256):
    c = KVarNKVCache(tail_tokens=tail)
    c.update_and_fetch(*_slab(n, seed=seed))
    return c


def _equal_content(a, b):
    for x, y in zip(a.materialize(), b.materialize(), strict=True):
        if not np.array_equal(np.array(x), np.array(y)):
            return False
    return True


# -- extract / merge ---------------------------------------------------------


@needs_kvarn_ops
def test_extract_padfree_row_is_bit_exact():
    k, v = _slab(600, b=2)
    batch = BatchKVarNKVCache([0, 150], tail_tokens=256)
    batch.update_and_fetch(k, v)
    row = batch.extract(0)
    assert type(row) is KVarNKVCache and row.offset == 600
    assert not row.horizon_valid
    bk, bv = batch.materialize()
    rk, rv = row.materialize()
    assert np.array_equal(np.array(rk), np.array(bk[0:1]))
    assert np.array_equal(np.array(rv), np.array(bv[0:1]))


@needs_kvarn_ops
def test_extract_padded_row_rebuilds():
    k, v = _slab(600, b=2)
    batch = BatchKVarNKVCache([0, 150], tail_tokens=256)
    batch.update_and_fetch(k, v)
    row = batch.extract(1)
    assert row.offset == 450
    # One extra quantization pass at the new alignment.
    rk, _ = row._raw_single()
    want = np.array(k[1:2, :, 150:].astype(mx.float32))
    assert np.abs(np.array(rk.astype(mx.float32)) - want).max() < 0.35


@needs_kvarn_ops
def test_merge_single_row_round_trip():
    c = _single(600)
    merged = BatchKVarNKVCache.merge([c])
    assert type(merged) is BatchKVarNKVCache
    assert np.array_equal(np.array(merged.left_padding), [0])
    assert merged._idx == 600 and merged.n_sealed == c.n_sealed
    assert _equal_content(merged, c)
    back = merged.extract(0)
    assert _equal_content(back, c)


@needs_kvarn_ops
def test_merge_multi_row_realigns():
    a, b = _single(600, seed=0), _single(400, seed=5)
    merged = BatchKVarNKVCache.merge([a, b])
    assert np.array_equal(np.array(merged.left_padding), [0, 200])
    assert merged._idx == 600
    rk, _ = merged._raw_rows()
    want_a = np.array(a._raw_single()[0].astype(mx.float32))
    got_a = np.array(rk[0:1].astype(mx.float32))
    assert np.abs(got_a - want_a).max() < 0.35
    want_b = np.array(b._raw_single()[0].astype(mx.float32))
    got_b = np.array(rk[1:2, :, 200:].astype(mx.float32))
    assert np.abs(got_b - want_b).max() < 0.35


def test_merge_rejects_mismatch():
    with pytest.raises(ValueError, match="at least one"):
        BatchKVarNKVCache.merge([])
    with pytest.raises(ValueError, match="matching"):
        BatchKVarNKVCache.merge(
            [KVarNKVCache(tail_tokens=256), KVarNKVCache(tail_tokens=512)]
        )
    with pytest.raises(ValueError, match="matching"):
        BatchKVarNKVCache.merge([KVCache()])


def test_finalize_allows_zero_right_padding():
    c = BatchKVarNKVCache([0])
    c.prepare(right_padding=[0])
    c.finalize()
    c.prepare(right_padding=[0, 4])
    with pytest.raises(RuntimeError, match="right padding"):
        c.finalize()


# -- arms --------------------------------------------------------------------


@needs_kvarn_ops
def test_clone_arm(restorable):
    kvarn_apc.install_kvarn_apc()
    c = _single(300)
    clones = apc._clone_prompt_cache_for_apc([c])
    assert clones is not None and type(clones[0]) is KVarNKVCache
    assert clones[0] is not c and _equal_content(clones[0], c)
    # Stored clones carry content, not the live cache's growth slack.
    assert clones[0].nbytes < c.nbytes / 3
    # Stored clones must be decoupled from the live cache.
    c.update_and_fetch(*_slab(1, seed=9))
    assert clones[0].offset == 300


@needs_kvarn_ops
def test_clone_arm_normalizes_lone_batch(restorable):
    kvarn_apc.install_kvarn_apc()
    k, v = _slab(300)
    batch = BatchKVarNKVCache([0], tail_tokens=256)
    batch.update_and_fetch(k, v)
    clones = apc._clone_prompt_cache_for_apc([batch])
    assert clones is not None and type(clones[0]) is KVarNKVCache
    assert clones[0].offset == 300
    wide = BatchKVarNKVCache([0, 4], tail_tokens=256)
    wide.update_and_fetch(*_slab(300, b=2))
    assert apc._clone_prompt_cache_for_apc([wide]) is None


@needs_kvarn_ops
def test_merge_arm(restorable):
    kvarn_apc.install_kvarn_apc()
    c = _single(300)
    merged = apc._merge_exact_cache_entries([c], [300])
    assert type(merged) is BatchKVarNKVCache
    # Multi-row and mixed warm/cold fall back to a cold prefill.
    assert apc._merge_exact_cache_entries([c, _single(200, seed=3)], [300, 200]) is None
    assert apc._merge_exact_cache_entries([c, KVCache()], [300, 0]) is None


def test_supports_and_mode(restorable):
    kvarn_apc.install_kvarn_apc()
    assert apc._cache_entry_supports_exact_apc(KVarNKVCache())

    class _LM:
        def make_cache(self):
            return [KVCache()]

    lm = _LM()
    assert apc.model_apc_mode(lm) == "block"
    kvarn_apc.stamp_model(lm)
    assert apc.model_apc_mode(lm) == "exact"


# -- disk kind ---------------------------------------------------------------


def _wait_for_exact_file(disk, timeout=10.0):
    end = time.time() + timeout
    while time.time() < end:
        if list(disk.dir.glob("exact_*.safetensors")):
            return True
        time.sleep(0.05)
    return False


@needs_kvarn_ops
def test_disk_round_trip_and_version_fail_closed(restorable, tmp_path):
    kvarn_apc.install_kvarn_apc()
    tokens = list(range(40))
    caches = [_single(300, seed=i) for i in range(2)]

    disk = apc.DiskBlockStore(tmp_path, namespace="t")
    mgr = apc.APCManager(num_blocks=4, block_size=16, disk=disk)
    assert mgr.store_exact_cache(tokens, caches, extra_hash=7)
    assert _wait_for_exact_file(disk)
    mgr.close()

    disk2 = apc.DiskBlockStore(tmp_path, namespace="t")
    mgr2 = apc.APCManager(num_blocks=4, block_size=16, disk=disk2)
    warm, plen = mgr2.lookup_exact_cache(tokens + [99], extra_hash=7)
    assert plen == 40 and warm is not None
    assert all(type(c) is KVarNKVCache for c in warm)
    for got, want in zip(warm, caches, strict=True):
        assert _equal_content(got, want)
    assert mgr2.stats.disk_hits > 0

    # A different extra hash (the scheme salt) must miss cleanly.
    miss, _ = mgr2.lookup_exact_cache(tokens + [99], extra_hash=8)
    assert miss is None
    mgr2.close()

    # A version bump turns stale shards into clean cold misses.
    restorable.setattr(KVarNKVCache, "kvarn_layout_version", 2)
    disk3 = apc.DiskBlockStore(tmp_path, namespace="t")
    mgr3 = apc.APCManager(num_blocks=4, block_size=16, disk=disk3)
    stale, _ = mgr3.lookup_exact_cache(tokens + [99], extra_hash=7)
    assert stale is None
    mgr3.close()


def test_entry_salt(restorable):
    assert kvarn_apc.kvarn_entry_salt() == 0
    restorable.setenv("KV_QUANT_SCHEME", "kvarn")
    salt = kvarn_apc.kvarn_entry_salt()
    assert salt != 0
    restorable.setenv("KV_BITS", "4")
    assert kvarn_apc.kvarn_entry_salt() not in (0, salt)


def test_manager_salts_exact_hashes(restorable):
    from gmlx.cache.apc_manager import GmlxAPCManager

    seen = []
    restorable.setattr(
        apc.APCManager,
        "lookup_exact_cache",
        lambda self, token_ids, extra_hash=0, **kw: (
            seen.append(extra_hash) or (None, 0)
        ),
    )
    restorable.setattr(
        apc.APCManager,
        "store_exact_cache",
        lambda self, token_ids, prompt_cache, *, extra_hash=0: (
            seen.append(extra_hash) or True
        ),
    )
    mgr = GmlxAPCManager(num_blocks=2, block_size=16)
    mgr._exact_extra_salt = 0xA5A5
    mgr.lookup_exact_cache([1, 2, 3], extra_hash=3)
    mgr.store_exact_cache([1, 2, 3], [], extra_hash=3)
    assert seen == [3 ^ 0xA5A5, 3 ^ 0xA5A5]
    mgr.close()


def test_layer_has_content_kvarn_arm():
    from gmlx.cache.snapshot import _layer_has_content

    assert not _layer_has_content(KVarNKVCache())


@needs_kvarn_ops
def test_layer_has_content_and_row_snapshot(restorable):
    kvarn_apc.install_kvarn_apc()
    from gmlx.cache.snapshot import _layer_has_content, row_snapshot

    k, v = _slab(300, b=2)
    batch = BatchKVarNKVCache([0, 64], tail_tokens=256)
    batch.update_and_fetch(k, v)
    snaps = row_snapshot([batch], row=1)
    assert snaps is not None and type(snaps[0]) is KVarNKVCache
    assert _layer_has_content(snaps[0])
    assert snaps[0].offset == 300 - 64
