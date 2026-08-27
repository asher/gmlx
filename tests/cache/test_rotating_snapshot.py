"""Rotating snapshot/restore inverse: canonical window, invariant halves,
geometry gate, and bit-exact continuation against an uninterrupted run."""

import mlx.core as mx
import pytest
from mlx_vlm.models.cache import RotatingKVCache

from gmlx.cache.snapshot import (
    rotating_canonical_window,
    rotating_geometry,
    rotating_invariant,
    rotating_restore,
)

H, D = 2, 4


def _kv(lo, hi):
    """K/V for tokens [lo, hi): value == token position, bitwise traceable."""
    pos = mx.arange(lo, hi, dtype=mx.float32).reshape(1, 1, hi - lo, 1)
    return (mx.broadcast_to(pos, (1, H, hi - lo, D)),
            mx.broadcast_to(pos + 0.5, (1, H, hi - lo, D)))


def _feed_chunks(cache, cuts):
    out = None
    for a, b in cuts:
        out = cache.update_and_fetch(*_kv(a, b))
    return out


def _feed_decode(cache, lo, n):
    out = None
    for t in range(lo, lo + n):
        out = cache.update_and_fetch(*_kv(t, t + 1))
    return out


def _snapshot_restore(cache):
    snap = rotating_canonical_window(cache)
    assert snap is not None
    k, v, meta = snap
    mx.eval(k, v)
    restored = rotating_restore(k, v, meta)
    assert restored is not None
    return restored


# -- inverse: restored continuation == original continuation --

@pytest.mark.parametrize("p", [10, 40])  # below and beyond wrap (W=16)
def test_concat_snapshot_then_chunk_matches(p):
    orig = RotatingKVCache(max_size=16)
    cuts = [(a, min(a + 8, p)) for a in range(0, p, 8)]
    _feed_chunks(orig, cuts)
    restored = _snapshot_restore(orig)
    assert rotating_invariant(restored) == (True, True)
    ko, vo = _feed_chunks(orig, [(p, p + 6)])
    kr, vr = _feed_chunks(restored, [(p, p + 6)])
    assert mx.array_equal(ko, kr) and mx.array_equal(vo, vr)
    ko, vo = _feed_decode(orig, p + 6, 3)
    kr, vr = _feed_decode(restored, p + 6, 3)
    assert mx.array_equal(ko, kr) and mx.array_equal(vo, vr)
    assert orig.offset == restored.offset


def test_ring_snapshot_converges_on_next_chunk():
    """Post-decode ring phase: the canonical restore and the live ring
    converge bit-exactly at the next concat update (the production shape -
    a warm hit always tail-prefills the new turn's suffix)."""
    orig = RotatingKVCache(max_size=16)
    _feed_chunks(orig, [(0, 12)])
    _feed_decode(orig, 12, 20)              # rotate well past the wrap
    assert orig._idx < orig.offset          # genuinely rotated
    restored = _snapshot_restore(orig)
    assert rotating_invariant(restored) == (True, True)
    p = orig.offset
    ko, vo = _feed_chunks(orig, [(p, p + 5)])
    kr, vr = _feed_chunks(restored, [(p, p + 5)])
    assert mx.array_equal(ko, kr) and mx.array_equal(vo, vr)


def test_restored_equals_uninterrupted_run():
    """The cert comparator at unit scale: restore at a mid-run boundary,
    continue, and match an uninterrupted run bit for bit."""
    W, p, N = 16, 24, 37
    uninterrupted = RotatingKVCache(max_size=W)
    ku, vu = _feed_chunks(uninterrupted, [(0, 8), (8, p), (p, N)])

    first = RotatingKVCache(max_size=W)
    _feed_chunks(first, [(0, 8), (8, p)])
    restored = _snapshot_restore(first)
    # The invariant is a restored-state property; continuation re-enters the
    # concat regime where the live buffer legitimately exceeds max_size.
    assert rotating_invariant(restored) == (True, True)
    kr, vr = _feed_chunks(restored, [(p, N)])
    assert mx.array_equal(ku, kr) and mx.array_equal(vu, vr)
    assert restored.offset == uninterrupted.offset


def test_keep_region_preserved():
    keep, W = 4, 16
    orig = RotatingKVCache(max_size=W, keep=keep)
    _feed_chunks(orig, [(0, 12), (12, 30)])
    k, v, meta = rotating_canonical_window(orig)
    assert meta == (keep, W, 30, W)
    assert k.shape[2] == W
    # first `keep` positions are tokens 0..3, the tail is the trailing window
    assert mx.array_equal(k[0, 0, :keep, 0],
                          mx.arange(0, keep, dtype=mx.float32))
    assert float(k[0, 0, -1, 0]) == 29.0
    restored = rotating_restore(k, v, meta)
    ko, vo = _feed_chunks(orig, [(30, 36)])
    kr, vr = _feed_chunks(restored, [(30, 36)])
    assert mx.array_equal(ko, kr) and mx.array_equal(vo, vr)


# -- invariant halves + meta discipline --

def test_restore_rejects_missing_meta():
    k, v = _kv(0, 16)
    assert rotating_restore(k, v, None) is None


def test_restore_rejects_length_mismatch():
    orig = RotatingKVCache(max_size=16)
    _feed_chunks(orig, [(0, 24)])
    k, v, meta = rotating_canonical_window(orig)
    short_k, short_v = k[..., :-1, :], v[..., :-1, :]   # a lost block's worth
    assert rotating_restore(short_k, short_v, meta) is None


def test_restore_rejects_stale_idx():
    orig = RotatingKVCache(max_size=16)
    _feed_chunks(orig, [(0, 24)])
    k, v, (keep, w, off, idx) = rotating_canonical_window(orig)
    assert rotating_restore(k, v, (keep, w, off, idx - 1)) is None


def test_invariant_halves_name_the_break():
    c = RotatingKVCache(max_size=16)
    _feed_chunks(c, [(0, 24)])
    restored = _snapshot_restore(c)
    ok_l, ok_idx = rotating_invariant(restored)
    assert ok_l and ok_idx
    restored._idx -= 2                       # ring pointer drifts
    assert rotating_invariant(restored) == (True, False)
    restored._idx += 2
    restored.offset = 10                     # claims less than the buffer holds
    assert rotating_invariant(restored) == (False, True)


# -- geometry gate --

def test_geometry_ok():
    caches = [RotatingKVCache(max_size=1024), RotatingKVCache(max_size=1024)]
    assert rotating_geometry(caches, 16) == (1024, 0)
    assert rotating_geometry(caches, 256) == (1024, 0)


def test_geometry_rejects_mixed_windows():
    caches = [RotatingKVCache(max_size=1024), RotatingKVCache(max_size=512)]
    assert rotating_geometry(caches, 16) is None


def test_geometry_rejects_off_grid():
    assert rotating_geometry([RotatingKVCache(max_size=1000)], 16) is None
    keep_bad = RotatingKVCache(max_size=1024, keep=8)
    assert rotating_geometry([keep_bad], 16) is None


def test_geometry_no_rotating_layers():
    assert rotating_geometry([], 16) is None
