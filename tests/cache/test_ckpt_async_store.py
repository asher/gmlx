"""A checkpoint store that submits its copies and returns
(GMLX_SPEC_CKPT_ASYNC) stores what the store that waits for them does."""

from types import SimpleNamespace

import mlx.core as mx
import pytest

from gmlx._vendor.mlx_vlm.apc import DiskBlockStore
from gmlx.cache import kv_cache as _cache
from gmlx.cache.apc_manager import GmlxAPCManager
from gmlx.cache.snapshot import _ckpt_records, ckpt_lookup, ckpt_store

pytestmark = pytest.mark.host_gb(16)

LAYOUT = ("kv", "arr", "arr", "kv", "arr")
H, D, BS = 2, 8, 16


def _cache_at(p, seed):
    from gmlx.models.qwen4_exp.model import QSAKVCache

    caches = []
    for i, kind in enumerate(LAYOUT):
        mx.random.seed(seed * 1000 + i)
        if kind == "kv":
            c = QSAKVCache(ratio=4)
            c.update_and_fetch_qsa(mx.random.normal((1, H, p, D)),
                                   mx.random.normal((1, H, p, D)),
                                   mx.random.normal((1, p, D)))
        else:
            c = _cache.ArraysCache(size=2)
            c.cache = [mx.random.normal((1, 3, D)),
                       mx.random.normal((1, H, D, D))]
        caches.append(c)
    return caches


def _extend(caches, n, seed):
    """The next chunk: K/V and ik rows written in place, states replaced."""
    mx.random.seed(seed)
    for c in caches:
        if isinstance(c, _cache.KVCache):
            c.update_and_fetch_qsa(mx.random.normal((1, H, n, D)),
                                   mx.random.normal((1, H, n, D)),
                                   mx.random.normal((1, n, D)))
        else:
            for k in range(len(c.cache)):
                c[k] = c.cache[k] + 1
    mx.eval([c.state for c in caches])


def _fenced(caches):
    return [c._fence for c in caches if isinstance(c, _cache.KVCache)]


def _record(man, p):
    return next(r for r in _ckpt_records(man).values() if r.p == p)


def _assert_same_record(ra, rb):
    assert ra.b_full == rb.b_full and ra.layout == rb.layout
    assert [b.block_hash for b in ra.main_blocks] == [
        b.block_hash for b in rb.main_blocks]
    for a, b in zip(ra.main_blocks, rb.main_blocks):
        for x, y in zip(a.kv_handle().parts, b.kv_handle().parts):
            assert mx.array_equal(x, y).item()
    for a, b in zip(ra.states, rb.states):
        for x, y in zip(a.cache, b.cache):
            assert mx.array_equal(x, y).item()
    for a, b in zip(ra.tails or (), rb.tails or ()):
        assert mx.array_equal(a.keys, b.keys).item()
        assert mx.array_equal(a.values, b.values).item()
    for (ia, pa, qa), (ib, pb, qb) in zip(ra.qsa, rb.qsa):
        assert qa == qb and pa is None and pb is None
        assert mx.array_equal(ia, ib).item()


@pytest.mark.parametrize("p2", [96, 100])
def test_async_store_matches_the_store_that_waits(p2):
    p1 = 64
    ids = list(range(100, 100 + p2 + 40))
    sync_man = GmlxAPCManager(num_blocks=64, block_size=BS)
    async_man = GmlxAPCManager(num_blocks=64, block_size=BS)
    live_a, live_b = _cache_at(p1, 5), _cache_at(p1, 5)

    for p, n, seed in ((p1, p2 - p1, 9), (p2, 40, 11)):
        assert ckpt_store(sync_man, ids[:p], live_a, extra_hash=3) == p
        assert ckpt_store(async_man, ids[:p], live_b, extra_hash=3,
                          eval_async=True) == p
        # The next chunk's write waits for the copies, then drops them.
        assert all(f for f in _fenced(live_b))
        _extend(live_a, n, seed)
        _extend(live_b, n, seed)
        assert all(f is None for f in _fenced(live_b))
        _assert_same_record(_record(sync_man, p), _record(async_man, p))

    wa, ga = ckpt_lookup(sync_man, ids[:p2] + [7], extra_hash=3)
    wb, gb = ckpt_lookup(async_man, ids[:p2] + [7], extra_hash=3)
    assert ga == gb == p2
    for a, b in zip(wa, wb):
        for x, y in zip(a.state, b.state):
            assert mx.array_equal(x, y).item()


def test_async_store_submits_each_part_as_one_buffer(monkeypatch):
    """Under the prefill caps each block would be its own command buffer,
    each waiting for a slot behind the forward still in flight."""
    from contextlib import contextmanager

    import gmlx.serve.cb_phase as cb_phase

    inside, outside = [False], []
    real = mx.async_eval

    @contextmanager
    def spy():
        inside[0] = True
        try:
            yield
        finally:
            inside[0] = False

    def submit(*a, **k):
        outside.append(not inside[0])
        return real(*a, **k)

    monkeypatch.setattr(cb_phase, "one_buffer", spy)
    monkeypatch.setattr(mx, "async_eval", submit)
    man = GmlxAPCManager(num_blocks=64, block_size=BS)
    assert ckpt_store(man, list(range(100, 200)), _cache_at(100, 5),
                      extra_hash=3, eval_async=True) == 100
    assert len(outside) >= 2 and not any(outside)


def test_async_store_waits_with_a_disk_tier(tmp_path):
    # The disk writer thread reads the shards: they must be evaluated.
    disk = DiskBlockStore(root=tmp_path, namespace="m")
    man = GmlxAPCManager(num_blocks=64, block_size=BS, disk=disk)
    try:
        live = _cache_at(64, 2)
        assert ckpt_store(man, list(range(100, 164)), live, extra_hash=3,
                          eval_async=True) == 64
        assert all(f is None for f in _fenced(live))
    finally:
        disk.close()


@pytest.mark.parametrize("asks", [True, False])
def test_boundary_store_is_async_when_the_batch_asks(asks):
    from gmlx.spec.ckpt import _ckpt_mid_prefill_store

    man = GmlxAPCManager(num_blocks=64, block_size=BS)
    ids = list(range(100, 228))
    meta = {"checkpoint_len": 64, "ckpt_boundaries": [(64, "boundary"),
                                                      (128, "boundary")],
            "full_input_ids": ids, "extra_hash": 3, "ckpt_last_stored": 0,
            "ckpt_stored_boundaries": []}
    live = _cache_at(64, 4)
    batch = SimpleNamespace(
        _kq_ckpt_armed=True, _apc_manager=man, _apc_meta=[meta],
        prompt_cache=live, _kq_ckpt_async=asks,
        _row_real_tokens_processed=lambda i: 64)
    _ckpt_mid_prefill_store(batch)
    assert meta["checkpoint_len"] == 128
    assert meta["ckpt_last_stored"] == 64
    assert _record(man, 64)
    assert all(bool(f) is asks for f in _fenced(live))
