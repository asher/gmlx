"""Disk exact-tier lookups answer from the in-memory index.

``find_exact_prefix`` reads no shard header. Each test compares it with
the header walk it replaced, kept here as the reference."""

import mlx.core as mx
import pytest

from gmlx._vendor.mlx_vlm import apc
from gmlx._vendor.mlx_vlm.models.cache import KVCache, RotatingKVCache
from gmlx.cache.apc_manager import GmlxAPCManager
from gmlx.serve import estimate as est

BS = 4
T = list(range(100, 140))

# cache_hash -> (token ids, extra_hash, trimmable)
ENTRIES = {
    1: (T[:12], 0, True),
    2: (T[:8] + list(range(900, 908)), 0, False),
    3: (T[:30], 7, True),
    4: ([-200] + T[:11], 0, False),
    5: (T[:36], 0, False),
}

# (query, keyword arguments)
QUERIES = [
    ([1, 2, 3, 4, 5, 6], {}),                          # miss
    (T[:16], {}),                                      # whole entry 1
    (T[:10] + [999] * 6, {}),                          # entry 1 trimmed to 8
    (T[:20], {}),                                      # entry 5 longer, not trimmable
    (T[:35], {"extra_hash": 7}),                       # entry 3 by its salt
    (T[:35], {"extra_hash": 99}),                      # no entry at this salt
    (T[:16], {"min_prefix_tokens": 12}),
    (T[:16], {"max_prefix_tokens": 10}),
    ([-200] + T[:11] + [1], {}),                       # negative id stored
    ([-5] + T[:15], {}),                               # negative id, miss
]


def _cache(n, trimmable):
    k = mx.zeros((1, 1, n, 4))
    if trimmable:
        c = KVCache()
    else:
        c = RotatingKVCache(max_size=64)
    c.update_and_fetch(k, k)
    return c


def _save(store, cache_hash):
    ids, extra, trimmable = ENTRIES[cache_hash]
    assert store.save_exact_cache(
        cache_hash, ids, extra, [_cache(len(ids), trimmable)],
        synchronous=True)


def _header_walk(store, token_ids, *, extra_hash=0, max_prefix_tokens=None,
                 min_prefix_tokens=0, block_size=BS):
    """The lookup as it read every shard header before the index."""
    token_tuple = tuple(int(t) for t in token_ids)
    max_len = len(token_tuple) - 1
    if max_prefix_tokens is not None and max_prefix_tokens > 0:
        max_len = min(max_len, int(max_prefix_tokens))
    if max_len <= min_prefix_tokens:
        return None
    best = None
    for cache_hash, path in list(store._exact_index.items()):
        metadata = apc._read_safetensors_metadata(path)
        if metadata is None or metadata.get("layout") != "exact_cache_v1":
            continue
        if int(metadata.get("extra_hash", "0")) != extra_hash:
            continue
        stored = tuple(int(x) for x in metadata.get("token_ids", "").split(",") if x)
        n = apc._checkpoint_match_len(
            token_tuple, stored, max_len, block_size,
            metadata.get("prefix_trimmable") == "1")
        if n > min_prefix_tokens and (best is None or n > best[1]):
            best = (int(cache_hash), n)
    return best


def _results(store):
    out = []
    for q, kw in QUERIES:
        kw = {"block_size": BS, **kw}
        new = store.find_exact_prefix(q, **kw)
        assert new == _header_walk(store, q, **kw), (q, kw)
        out.append(new)
    return out


EXPECTED = [None, (1, 12), (1, 8), (1, 12), (3, 30), None, None, (1, 8),
            (4, 12), None]


@pytest.fixture
def stores(tmp_path):
    opened = []

    def make():
        s = apc.DiskBlockStore(tmp_path, namespace="t")
        opened.append(s)
        return s

    yield make
    for s in opened:
        s.close()


def _filled(make):
    store = make()
    for h in ENTRIES:
        _save(store, h)
    return store


def test_lookups_after_the_write_match_the_header_walk(stores):
    store = _filled(stores)
    assert set(store._exact_meta) == set(ENTRIES)
    assert _results(store) == EXPECTED


def test_the_startup_scan_fills_the_index(stores):
    first = _filled(stores)
    first.close()
    store = stores()
    assert set(store._exact_meta) == set(ENTRIES)
    assert _results(store) == EXPECTED


def test_a_dropped_entry_is_gone_and_a_restore_finds_it_again(stores):
    store = _filled(stores)
    path = store._exact_index[1]
    store._drop_index_for_path(path)
    assert 1 not in store._exact_meta and path.exists()
    assert _results(store)[1] is None
    _save(store, 1)                                    # dedup: shard exists
    assert store._exact_meta[1][0] == 0 and store._exact_meta[1][2] is True
    assert _results(store) == EXPECTED


def test_a_failed_header_read_drops_the_entry(stores):
    store = _filled(stores)
    path = store._exact_index[1]
    path.write_bytes(b"bad")
    assert store.load_exact_cache(1) is None
    assert 1 not in store._exact_meta and 1 not in store._exact_index
    res = _results(store)
    assert res[1] is None and res[2] is None


def test_eviction_drops_the_index(stores):
    store = _filled(stores)
    store.max_bytes = 1
    assert store._maybe_evict() == len(ENTRIES)
    assert store._exact_meta == {}
    assert _results(store) == [None] * len(QUERIES)


def test_the_clears_drop_the_index(stores):
    store = _filled(stores)
    store._clear_disk_state_after_external_delete()
    assert store._exact_meta == {}
    assert _results(store) == [None] * len(QUERIES)
    store._rebuild_index()
    assert _results(store) == EXPECTED


def test_a_lookup_reads_no_header(stores, monkeypatch):
    store = _filled(stores)
    calls = []
    real = apc._read_safetensors_header
    monkeypatch.setattr(apc, "_read_safetensors_header",
                        lambda p: calls.append(p) or real(p))
    monkeypatch.setattr(store, "_open_shard_header",
                        lambda p: calls.append(p))
    for q, kw in QUERIES:
        store.find_exact_prefix(q, **{"block_size": BS, **kw})
    assert calls == []


@pytest.mark.parametrize("bad", [2**31, -(2**31) - 1, 2**64])
def test_an_id_out_of_int32_range_is_a_miss(stores, bad):
    store = _filled(stores)
    assert store.find_exact_prefix(T[:12] + [bad], block_size=BS) is None
    assert store.find_exact_prefix([bad] + T[:12], block_size=BS) is None


def test_the_estimate_probe_finds_a_disk_exact_hit(stores):
    store = _filled(stores)
    man = GmlxAPCManager(num_blocks=8, block_size=BS, disk=store)
    salt = int(getattr(man, "_exact_extra_salt", 0) or 0)
    assert salt == 0
    assert est._warm_tokens(man, T[:10] + [999] * 6, 0) == (8, "exact")
    assert est._warm_tokens(man, T[:16], 0) == (12, "exact")
    assert est._warm_tokens(man, [1, 2, 3], 0) == (0, None)
