"""SpecPrefixCache memory-safety: sliding-window trim + byte budget.

Deep-context MTP requests on window-heavy models (gemma-4) OOM'd the GPU:
post-prefill rotating caches hold the FULL prompt (trim is deferred to the
first decode step), so an APC snapshot pinned O(prompt_len) per sliding
layer, and the count-only LRU let tens of GB of dead snapshots accumulate
across requests. These tests pin the two fixes: snapshots trim sliding
layers to their window, and the cache enforces a total byte budget.
"""

import mlx.core as mx

from mlx_lm.models.cache import KVCache, RotatingKVCache

from gmlx.prefix_cache import SpecPrefixCache, _snapshot_entry


def _fill_rotating(window: int, total: int, keep: int = 0):
    """Build the post-prefill state the GMLX_SLIDING_NOTRIM=1 mode leaves:
    the full prompt accumulated in temporal order, trim deferred to decode.
    Snapshots must stay window-sized regardless of which prefill mode ran."""
    c = RotatingKVCache(window, keep=keep)
    k = mx.arange(total, dtype=mx.float32).reshape(1, 1, total, 1)
    k = mx.broadcast_to(k, (1, 2, total, 4))
    c.keys = k
    c.values = k + 1000
    c.offset = total
    c._idx = total
    return c


def _fill_kv(total: int):
    c = KVCache()
    k = mx.zeros((1, 2, total, 4), dtype=mx.float32)
    c.update_and_fetch(k, k)
    return c


class TestRotatingSnapshotTrim:
    def test_post_prefill_snapshot_is_window_sized(self):
        c = _fill_rotating(window=8, total=30)
        assert c.keys.shape[2] == 30  # no-trim prefill holds the full prompt
        keys, values, offset, idx = _snapshot_entry(c)
        assert keys.shape[2] == 8
        assert values.shape[2] == 8
        assert offset == 30
        assert idx == 8

    def test_snapshot_keeps_last_window_positions(self):
        c = _fill_rotating(window=8, total=30)
        keys, _, _, _ = _snapshot_entry(c)
        got = keys[0, 0, :, 0]
        assert got.tolist() == list(range(22, 30))

    def test_keep_head_preserved(self):
        c = _fill_rotating(window=8, total=30, keep=2)
        keys, _, _, _ = _snapshot_entry(c)
        got = keys[0, 0, :, 0].tolist()
        assert got[:2] == [0.0, 1.0]
        assert got[2:] == list(range(24, 30))

    def test_short_prompt_not_trimmed(self):
        c = _fill_rotating(window=8, total=5)
        keys, _, offset, _ = _snapshot_entry(c)
        assert keys.shape[2] == 5
        assert offset == 5

    def test_restored_cache_decodes(self):
        src = _fill_rotating(window=8, total=30)
        cache = SpecPrefixCache()
        ids = mx.arange(64)[None]
        hidden = mx.zeros((1, 64, 4))
        cache.store(ids, [src], hidden)
        hit = cache.lookup(mx.arange(65)[None])
        assert hit is not None
        prefix_len, entry = hit
        assert prefix_len == 64

        dst = RotatingKVCache(8)
        cache.restore(entry, [dst])
        assert dst.offset == 30
        assert dst.keys.shape[2] == 8
        # A decode-step update must extend cleanly from the restored state.
        k = mx.full((1, 2, 1, 4), 99.0)
        keys, _ = dst.update_and_fetch(k, k)
        mx.eval(keys)
        assert dst.offset == 31
        dst.make_mask(1)  # must not raise


def _decode_fill(window: int, total: int, keep: int = 0):
    """Drive a RotatingKVCache with single-token updates so the ring wraps
    exactly as decode/_update_in_place leaves it (post-wrap ring order)."""
    c = RotatingKVCache(window, keep=keep)
    for i in range(total):
        k = mx.full((1, 2, 1, 4), float(i))
        c.update_and_fetch(k, k + 1000.0)
    return c


class TestWrappedSnapshotOrder:
    def test_wrapped_buffer_at_window_is_snapshotted_temporally(self):
        # 20 tokens through window 8: buffer length == window but ring-rotated
        # (_idx != n). The snapshot is tagged temporally ordered, so it must
        # BE temporally ordered or a warm suffix update drops the newest
        # token and keeps the stale oldest.
        c = _decode_fill(window=8, total=20)
        assert int(c._idx) != int(c.keys.shape[2])  # genuinely wrapped
        keys, values, offset, idx = _snapshot_entry(c)
        assert offset == 20
        assert idx == keys.shape[2] == 8
        assert keys[0, 0, :, 0].tolist() == [float(i) for i in range(12, 20)]
        assert values[0, 0, :, 0].tolist() == [
            float(i) + 1000.0 for i in range(12, 20)]


class TestSnapshotIsolation:
    """A stored entry must survive in-place mutation of the live cache.

    RotatingKVCache._update_in_place __setitem__-mutates the buffer object,
    so a snapshot that aliases the live arrays (store side) or a restore
    that hands the stored arrays back to a live cache (restore side) lets
    decode silently corrupt the entry for all future warm hits."""

    def test_restore_does_not_alias_stored_entry(self):
        src = _decode_fill(window=8, total=20)
        cache = SpecPrefixCache()
        cache.store(mx.arange(64)[None], [src], mx.zeros((1, 64, 4)))
        hit = cache.lookup(mx.arange(65)[None])
        assert hit is not None
        entry = hit[1]
        before = entry.kv_snaps[0][0][0, 0, :, 0].tolist()

        dst = RotatingKVCache(8)
        cache.restore(entry, [dst])
        assert dst.keys is not entry.kv_snaps[0][0]
        # First decode step wraps _idx to keep and writes in place.
        k = mx.full((1, 2, 1, 4), -777.0)
        dst.update_and_fetch(k, k)
        mx.eval(dst.keys)
        assert entry.kv_snaps[0][0][0, 0, :, 0].tolist() == before

    def test_stored_entry_survives_continued_live_decode(self):
        # Exactly-full temporally-ordered buffer: no trim/reorder applies,
        # so pre-fix the snapshot WAS the live ring buffer.
        src = _decode_fill(window=8, total=8)
        cache = SpecPrefixCache()
        cache.store(mx.arange(64)[None], [src], mx.zeros((1, 64, 4)))
        hit = cache.lookup(mx.arange(65)[None])
        assert hit is not None
        entry = hit[1]
        before = entry.kv_snaps[0][0][0, 0, :, 0].tolist()

        k = mx.full((1, 2, 1, 4), -777.0)
        src.update_and_fetch(k, k)  # live decode continues after store
        mx.eval(src.keys)
        assert entry.kv_snaps[0][0][0, 0, :, 0].tolist() == before

    def test_k_eq_v_identity_preserved_on_restore(self):
        # K-eq-V layers store one array as both K and V; the restored cache
        # must keep that sharing so entry byte accounting stays truthful.
        c = _fill_rotating(window=8, total=8)
        c.values = c.keys
        snap = _snapshot_entry(c)
        dst = RotatingKVCache(8)
        from gmlx.prefix_cache import _restore_entry
        _restore_entry(dst, snap)
        assert dst.keys is dst.values


class TestByteBudget:
    def test_entry_bytes_counted_and_evicted(self):
        # Each KVCache entry: 2 arrays x (1*2*64*4) floats = 4096 B; hidden 16 B.
        cache = SpecPrefixCache(max_entries=10, max_bytes=9000)
        hidden = mx.zeros((1, 1, 4))
        for i in range(3):
            ids = mx.arange(i * 100, i * 100 + 64)[None]
            cache.store(ids, [_fill_kv(64)], hidden)
        assert cache.total_bytes <= 9000
        assert len(cache) == 2  # third store evicted the first

    def test_oversized_entry_skipped(self):
        cache = SpecPrefixCache(max_entries=10, max_bytes=1000)
        ids = mx.arange(64)[None]
        cache.store(ids, [_fill_kv(64)], mx.zeros((1, 1, 4)))
        assert len(cache) == 0
        assert cache.total_bytes == 0

    def test_restore_roundtrip_after_budget_eviction(self):
        cache = SpecPrefixCache(max_entries=10, max_bytes=1 << 30)
        ids = mx.arange(64)[None]
        src = _fill_kv(64)
        cache.store(ids, [src], mx.zeros((1, 1, 4)))
        hit = cache.lookup(mx.arange(65)[None])
        assert hit is not None
        dst = KVCache()
        cache.restore(hit[1], [dst])
        assert dst.offset == 64

    def test_clear_resets_bytes(self):
        cache = SpecPrefixCache()
        ids = mx.arange(64)[None]
        cache.store(ids, [_fill_kv(64)], mx.zeros((1, 1, 4)))
        assert cache.total_bytes > 0
        cache.clear()
        assert cache.total_bytes == 0
        assert len(cache) == 0
