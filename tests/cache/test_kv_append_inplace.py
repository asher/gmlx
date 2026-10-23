"""The owned KV caches append decode rows into their own buffer
(gmlx/cache/kv_append.py). The append is in place only while nothing else
can read the rows it writes; every other path takes a slice write that
copies a held buffer. These tests pin both sides: each path that hands rows
out or moves the offset keeps its readers' rows, and plain decode, the
gated MTP restore and the fresh buffers stay in place.

The memory tests need the Metal device and skip under KQUANT_FORCE_CPU."""

from __future__ import annotations

import copy
import os
from collections import OrderedDict
from math import ceil
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from gmlx._vendor.mlx_vlm import apc, apc_adapters
from gmlx.cache import kv_append
from gmlx.cache.apc_manager import GmlxAPCManager
from gmlx.cache import kv_cache as owned
from gmlx.cache.kv_policy import hold_fp16
from gmlx.cache.snapshot import _truncate_kv_snapshot, row_snapshot
from gmlx.serve.memtrace import _cache_report
from gmlx.spec.helpers import _slice_shared_kv_after_reject
from gmlx.spec.speculative import _cache_restore, _cache_snapshot

pytestmark = pytest.mark.skipif(
    not kv_append.INPLACE, reason="mlx_kquant without kv_append")

metal = pytest.mark.skipif(
    bool(os.environ.get("KQUANT_FORCE_CPU")) or not mx.metal.is_available(),
    reason="allocation is measured on the Metal device")

H, D = 2, 64


def _rows(n, seed, b=1, h=H, d=D):
    mx.random.seed(seed)
    k = mx.random.normal((b, h, n, d)).astype(mx.float16)
    v = mx.random.normal((b, h, n, d)).astype(mx.float16)
    mx.eval(k, v)
    return k, v


@pytest.fixture
def inplace(monkeypatch):
    """Offsets of the in-place appends, one per array written."""
    calls = []
    real = kv_append._kq.kv_append

    def spy(buf, x, offset):
        calls.append(offset)
        return real(buf, x, offset)

    monkeypatch.setattr(kv_append, "_kq", SimpleNamespace(kv_append=spy))
    return calls


class _Ref:
    """NumPy copy of the rows a cache should hold."""

    def __init__(self):
        self.k = np.zeros((1, H, 0, D), np.float16)
        self.v = np.zeros((1, H, 0, D), np.float16)

    def push(self, k, v):
        self.k = np.concatenate([self.k, np.asarray(k)], axis=2)
        self.v = np.concatenate([self.v, np.asarray(v)], axis=2)

    def trim(self, n):
        self.k, self.v = self.k[:, :, :-n], self.v[:, :, :-n]

    def copy(self):
        out = _Ref()
        out.k, out.v = self.k.copy(), self.v.copy()
        return out


def _decode(c, ref, n, seed):
    for i in range(n):
        k, v = _rows(1, seed + i)
        c.update_and_fetch(k, v)
        ref.push(k, v)
    mx.eval(c.keys, c.values)


def _same(c, ref):
    k, v = c.state[:2]
    assert c.offset == ref.k.shape[2]
    np.testing.assert_array_equal(np.asarray(k), ref.k)
    np.testing.assert_array_equal(np.asarray(v), ref.v)


def _filled(n=40, seed=0):
    c, ref = owned.KVCache(), _Ref()
    k, v = _rows(n - 3, seed)
    c.update_and_fetch(k, v)
    ref.push(k, v)
    _decode(c, ref, 3, seed + 1000)
    return c, ref


# Plain decode.

def _quantized_ref(rows, group_size, bits):
    k = mx.concatenate([r[0] for r in rows], axis=2)
    v = mx.concatenate([r[1] for r in rows], axis=2)
    return (mx.quantize(k, group_size=group_size, bits=bits),
            mx.quantize(v, group_size=group_size, bits=bits))


@pytest.mark.parametrize("make, parts", [
    (owned.KVCache, 1),
    (lambda: owned.BatchKVCache([0]), 1),
    (lambda: owned.ChunkedKVCache(4096), 1),
    (lambda: owned.QuantizedKVCache(64, 8), 3),
    (lambda: owned.BatchQuantizedKVCache([0], 64, 8), 3),
])
def test_decode_appends_in_place(make, parts, inplace):
    c = make()
    rows = [_rows(5, 0)] + [_rows(1, 1 + i) for i in range(6)]
    for k, v in rows:
        c.update_and_fetch(k, v)
    mx.eval(c.keys, c.values)
    # The first allocation is fresh, so every append writes in place.
    assert len(inplace) == len(rows) * 2 * parts
    if parts == 1:
        k, v = c.state[:2]
        np.testing.assert_array_equal(
            np.asarray(k), np.concatenate([np.asarray(r[0]) for r in rows], 2))
        np.testing.assert_array_equal(
            np.asarray(v), np.concatenate([np.asarray(r[1]) for r in rows], 2))
    else:
        want = _quantized_ref(rows, 64, 8)
        n = sum(r[0].shape[2] for r in rows)
        for side, ref in zip((c.keys, c.values), want):
            for a, b in zip(side, ref):
                np.testing.assert_array_equal(np.asarray(a[..., :n, :]),
                                              np.asarray(b))


def test_kill_switch_takes_the_slice_write(monkeypatch, inplace):
    monkeypatch.setattr(kv_append, "INPLACE", False)
    c, ref = _filled()
    _decode(c, ref, 3, 50)
    assert inplace == []
    _same(c, ref)


# Hazards: each path keeps the rows its readers hold.

def test_two_caches_sharing_one_buffer_keep_their_rows(inplace):
    # A by-reference restore: two caches get the entry's arrays as they are.
    c, ref = _filled()
    a, b = copy.copy(c), copy.copy(c)
    ref_a, ref_b = ref.copy(), ref.copy()
    for i in range(3):
        _decode(a, ref_a, 1, 100 + i)
        _decode(b, ref_b, 1, 200 + i)
    _same(a, ref_a)
    _same(b, ref_b)
    _same(c, ref)


def test_from_state_twice_from_one_entry(inplace):
    c, ref = _filled()
    state = c.state
    a = owned.KVCache.from_state(state, c.meta_state)
    b = owned.KVCache.from_state(state, c.meta_state)
    ref_a, ref_b = ref.copy(), ref.copy()
    _decode(a, ref_a, 2, 100)
    _decode(b, ref_b, 2, 200)
    _same(a, ref_a)
    _same(b, ref_b)


@pytest.mark.parametrize("how", ["trim", "offset"])
def test_trim_then_append_under_a_pending_reader(how, inplace):
    c, ref = _filled()
    k, v = c.state
    want = float(mx.sum(k.astype(mx.float32)).item())
    reader = mx.sum(c.state[0].astype(mx.float32))  # pending
    inplace.clear()
    if how == "trim":
        c.trim(2)
    else:
        c.offset -= 2
    ref.trim(2)
    nk, nv = _rows(1, 77)
    out = c.update_and_fetch(nk, nv)
    ref.push(nk, nv)
    mx.eval(out)
    assert reader.item() == want
    assert inplace == []
    _same(c, ref)


def test_drafter_views_after_a_reject_trim(inplace):
    c, ref = _filled()
    views = {0: c.state}
    mx.eval(views[0])
    held = [np.asarray(a).copy() for a in views[0]]
    sliced = _slice_shared_kv_after_reject(views, 3)
    want = mx.sum(views[0][0][..., :-3, :].astype(mx.float32)).item()
    draft = mx.sum(sliced[0][0].astype(mx.float32))  # pending
    c.trim(3)
    ref.trim(3)
    _decode(c, ref, 2, 300)
    assert draft.item() == want
    for a, b in zip(views[0], held):
        np.testing.assert_array_equal(np.asarray(a), b)
    _same(c, ref)


def test_offset_bump_then_trim_keeps_the_bumped_rows(inplace):
    # Only an exact start at the high-water row is in place: rows above it
    # were handed out by the bump.
    c, ref = _filled()
    hw = c.offset
    c.offset = hw + 5
    view = c.state[0]
    mx.eval(view)
    held = np.asarray(view).copy()
    inplace.clear()
    c.offset = hw + 2
    c.update_and_fetch(*_rows(1, 400))
    mx.eval(c.keys)
    assert inplace == []
    np.testing.assert_array_equal(np.asarray(view), held)


def test_truncated_snapshot_then_append(inplace):
    c, ref = _filled()
    snap = _truncate_kv_snapshot([copy.copy(c)], c.offset - 4)
    assert snap is not None
    ref_s = ref.copy()
    ref_s.trim(4)
    _decode(snap[0], ref_s, 3, 500)
    _same(snap[0], ref_s)
    _same(c, ref)


def test_extract_then_append(inplace):
    c, ref = _filled()
    row = c.extract(0)
    ref_r = ref.copy()
    _decode(row, ref_r, 3, 600)
    _same(row, ref_r)
    _decode(c, ref, 1, 700)
    _same(c, ref)
    # The extracted copy is private after one slice write.
    inplace.clear()
    _decode(row, ref_r, 1, 800)
    assert len(inplace) == 2


def _batch(lens, seed):
    """A BatchKVCache over left-padded rows, and each row's rows."""
    width = max(lens)
    pads = [width - n for n in lens]
    c = owned.BatchKVCache(pads)
    k, v = _rows(width, seed, b=len(lens))
    c.update_and_fetch(k, v)
    mx.eval(c.keys, c.values)
    rows = [(np.asarray(k[i:i + 1, :, p:]), np.asarray(v[i:i + 1, :, p:]))
            for i, p in enumerate(pads)]
    return c, rows


def _batch_row(c, i):
    p = int(c.left_padding[i].item())
    return (np.asarray(c.keys[i:i + 1, :, p:c._idx]),
            np.asarray(c.values[i:i + 1, :, p:c._idx]))


@pytest.mark.parametrize("op", ["filter_shift", "filter", "extend"])
def test_batch_reshape_then_append(op, inplace):
    c, rows = _batch([20, 23], 0)
    if op == "filter_shift":
        c.filter([0])  # the shorter row: a left shift of 3
        rows = rows[:1]
    elif op == "filter":
        c.filter([1])
        rows = rows[1:]
    else:
        other, more = _batch([10], 1)
        c.extend(other)
        rows = rows + more
    held = [(c.keys, c.values)]
    mx.eval(held)
    kept = [(np.asarray(a).copy(), np.asarray(b).copy()) for a, b in held]
    inplace.clear()
    for step in range(3):
        k, v = _rows(1, 900 + step, b=len(rows))
        c.update_and_fetch(k, v)
        rows = [(np.concatenate([rk, np.asarray(k[i:i + 1])], 2),
                 np.concatenate([rv, np.asarray(v[i:i + 1])], 2))
                for i, (rk, rv) in enumerate(rows)]
        if step == 0:
            assert inplace == []
    mx.eval(c.keys, c.values)
    # One slice write, then in place.
    assert len(inplace) == 4
    for i, (rk, rv) in enumerate(rows):
        gk, gv = _batch_row(c, i)
        np.testing.assert_array_equal(gk, rk)
        np.testing.assert_array_equal(gv, rv)
    for (a, b), (ka, kb) in zip(held, kept):
        np.testing.assert_array_equal(np.asarray(a), ka)
        np.testing.assert_array_equal(np.asarray(b), kb)


def test_disk_store_from_the_writer_thread(tmp_path, inplace):
    c, ref = _filled()
    saved = ref.copy()
    clones = row_snapshot([c])
    # The producer hands the writer evaluated arrays only: the writer
    # thread has no GPU stream to run a lazy op on.
    mx.eval([cl.state for cl in clones])
    store = apc.DiskBlockStore(tmp_path, namespace="t")
    try:
        ids = tuple(range(c.offset))
        assert store.save_exact_cache(42, ids, 0, clones)
        # The live cache moves on while the writer serializes the clone.
        c.trim(5)
        ref.trim(5)
        _decode(c, ref, 6, 1100)
        store.flush()
        loaded = store.load_exact_cache(42)
        assert loaded is not None
        _same(loaded[2][0], saved)
        _same(c, ref)
    finally:
        store.close()


# Restores and clones.

@pytest.mark.parametrize("kind", ["fp16", "q8"])
def test_gated_round_restore_stays_in_place(kind, inplace):
    c = owned.KVCache() if kind == "fp16" else owned.QuantizedKVCache(64, 8)
    rows = [_rows(20, 0)] + [_rows(1, 1 + i) for i in range(3)]
    for k, v in rows:
        c.update_and_fetch(k, v)
    mx.eval(c.keys, c.values)
    snap = _cache_snapshot([c])
    assert snap is not None
    verify = c.update_and_fetch(*_rows(4, 50))  # built, never evaluated
    verify = None  # noqa: F841 - the gate drops the handle before the restore
    _cache_restore(snap)
    inplace.clear()
    more = [_rows(1, 60 + i) for i in range(3)]
    for k, v in more:
        c.update_and_fetch(k, v)
    mx.eval(c.keys, c.values)
    assert len(inplace) == len(more) * 2 * (1 if kind == "fp16" else 3)
    rows += more
    if kind == "fp16":
        k, v = c.state
        np.testing.assert_array_equal(
            np.asarray(k), np.concatenate([np.asarray(r[0]) for r in rows], 2))
        np.testing.assert_array_equal(
            np.asarray(v), np.concatenate([np.asarray(r[1]) for r in rows], 2))
    else:
        n = c.offset
        for side, want in zip((c.keys, c.values), _quantized_ref(rows, 64, 8)):
            for a, b in zip(side, want):
                np.testing.assert_array_equal(np.asarray(a[..., :n, :]),
                                              np.asarray(b))


@pytest.mark.parametrize("clone", ["apc", "fp16_hold"])
def test_new_clones_append(clone, inplace):
    c, ref = _filled()
    if clone == "apc":
        targets = []
        out = apc_adapters.clone_cache_entry(
            c, min_capacity_tokens=c.offset + 16, eval_targets=targets)
        mx.eval(targets)
    else:
        out = hold_fp16(owned.KVCache())
        out.__dict__.update(copy.copy(c).__dict__)
    ref_o = ref.copy()
    _decode(out, ref_o, 1, 1200)
    _decode(out, ref_o, 2, 1300)
    _same(out, ref_o)
    _same(c, ref)
    assert len(inplace) >= 4  # in place after the first append at most


def test_byte_counts_skip_the_second_handle():
    c, _ = _filled()
    assert c._kv_own[1] is c.keys
    want = c.keys.nbytes + c.values.nbytes
    assert _cache_report([c])[0] == want
    mgr = GmlxAPCManager(num_blocks=4, block_size=16)
    mgr._exact_budget_bytes = 1 << 40
    mgr._exact_cache = OrderedDict(
        [(1, SimpleNamespace(prompt_cache=[c]))])
    mgr._trim_exact_to_budget()
    assert mgr.stats.exact_bytes == want


# Memory: what an append allocates while a step still holds the cache.

def _pipelined(c, steps, seed, measure_from=0):
    """Append ``steps`` rows the way pipelined decode does: step t's reader
    is still pending when step t+1 appends. Returns the peak above the
    start for the steps from ``measure_from`` on."""
    pending = None
    for i in range(steps):
        k, v = _rows(1, seed + i, b=c.keys.shape[0], h=c.keys.shape[1],
                     d=c.keys.shape[3])
        if i == measure_from:
            mx.synchronize()
            base = mx.get_active_memory()
            mx.reset_peak_memory()
        ko, vo = c.update_and_fetch(k, v)
        mx.eval(c.keys, c.values)  # the append, before the last reader
        if pending is not None:
            mx.eval(pending)
        pending = mx.sum(ko[..., -1, :]) + mx.sum(vo[..., -1, :])
    mx.eval(pending)
    mx.synchronize()
    return mx.get_peak_memory() - base


def _reserved_kv(rows):
    c = owned.KVCache()
    c.update_and_fetch(*_rows(64, 0, h=8, d=128))
    c.prefix_cache_reserve(rows)
    mx.eval(c.keys, c.values)
    return c


@metal
@pytest.mark.parametrize("route", [True, False])
def test_pipelined_kvcache_appends_allocate_nothing(route, monkeypatch):
    monkeypatch.setattr(kv_append, "INPLACE", route)
    c = _reserved_kv(8192)
    steps = 16
    assert c.keys.shape[2] >= c.offset + steps  # no regrow in the window
    size = c.keys.nbytes
    grown = _pipelined(c, steps, 10)
    if route:
        assert grown < 2 << 20
    else:
        assert grown >= size  # the control: today's slice write copies


@metal
def test_pipelined_batch_appends_allocate_nothing():
    c = owned.BatchKVCache([0])
    c.update_and_fetch(*_rows(7000, 0, h=8, d=128))
    mx.eval(c.keys, c.values)
    assert c.keys.shape[2] >= c._idx + 16
    assert _pipelined(c, 16, 10) < 2 << 20


@metal
def test_pipelined_batch_after_a_left_shift_copies_at_most_once():
    # The shift leaves an offset view. Its first append is a slice write,
    # which donates the view when nothing holds it; every append after it
    # is in place.
    c = owned.BatchKVCache([3, 0])
    k, v = _rows(7000, 0, b=2, h=8, d=128)
    c.update_and_fetch(k, v)
    mx.eval(c.keys, c.values)
    c.filter([0])
    mx.eval(c.keys, c.values)
    assert c.keys.shape[2] >= c._idx + 17
    size = c.keys.nbytes + c.values.nbytes
    assert _pipelined(c, 1, 10) < size * 2
    assert _pipelined(c, 16, 20) < 2 << 20
    np.testing.assert_array_equal(np.asarray(c.keys[..., :6997, :]),
                                  np.asarray(k[:1, :, 3:, :]))


@metal
def test_synchronous_trim_then_append_allocates_nothing():
    # The MTP round shape: nothing holds the buffer, so the slice write
    # after a trim donates it.
    c = _reserved_kv(8192)
    c.update_and_fetch(*_rows(4, 1, h=8, d=128))
    mx.eval(c.keys, c.values)
    c.trim(3)
    k, v = _rows(1, 2, h=8, d=128)
    mx.synchronize()
    base = mx.get_active_memory()
    mx.reset_peak_memory()
    c.update_and_fetch(k, v)
    mx.eval(c.keys, c.values)
    mx.synchronize()
    assert mx.get_peak_memory() - base < 2 << 20


# The capacity grid and the footprint prediction.

LEVERS = [{}, {"GMLX_KV_GROW_CAP": "0"}, {"GMLX_KV_GROW_DIV": "0"}]


_DECODE = [("append", 1)] * 3000

# Each step is ("append", rows) or ("trim", rows).
HISTORIES = {
    "decode": _DECODE,
    "prefill": [("append", 4096)] + _DECODE,
    # The regrow after the trim starts mid-step.
    "trim_mid_step": [("append", 512), ("trim", 7), ("append", 20)] + _DECODE,
    "trim_below_peak": [("append", 1)] * 2600 + [("trim", 1500)]
    + [("append", 1)] * 1200,
}


def _run(steps):
    c = owned.KVCache()
    profile = None
    peak = 0
    caps = []
    for kind, n in steps:
        if kind == "trim":
            c.trim(n)
            continue
        before = None if c.keys is None else c.keys.shape[2]
        c.update_and_fetch(mx.zeros((1, 1, n, 8), mx.float16),
                           mx.zeros((1, 1, n, 8), mx.float16))
        peak = max(peak, c.offset)
        if c.keys.shape[2] != before:
            if profile is None:
                profile = c.memory_profile(c.offset)
            caps.append((peak, c.keys.shape[2]))
    caps.append((peak, c.keys.shape[2]))
    return profile, caps


@pytest.mark.parametrize("env", LEVERS, ids=["default", "no_cap", "no_div"])
@pytest.mark.parametrize("name", list(HISTORIES))
def test_footprint_bounds_the_grid(env, name, monkeypatch):
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    profile, caps = _run(HISTORIES[name])
    per = profile.bytes_per_token
    for peak, cap in caps:
        assert cap % owned.KVCache.step == 0
        predicted = profile.footprint(peak)
        if name == "decode":
            assert ceil(cap * per) == predicted
        else:
            assert ceil(cap * per) <= predicted


def test_grid_levers():
    step = 256
    assert kv_append.grid_capacity(300, step, 0, 0) == 512
    points = kv_append._grid(step, 8, 8192)
    assert points[:9] == (256, 512, 768, 1024, 1280, 1536, 1792, 2048, 2304)
    assert all(b - a <= 8192 for a, b in zip(points, points[1:]))
    uncapped = kv_append._grid(step, 8, 0)
    assert max(b - a for a, b in zip(uncapped, uncapped[1:])) > 8192
    assert kv_append.grid_capacity(200_000, step, 8, 8192) < 210_000


def test_rotating_profile_keeps_step_rounding():
    c = owned.RotatingKVCache(max_size=4096)
    c.update_and_fetch(mx.zeros((1, 1, 300, 8)), mx.zeros((1, 1, 300, 8)))
    p = c.memory_profile(300)
    assert p.grow_div == 0
    rows = (3000 + p.step - 1) // p.step * p.step
    assert p.footprint(3000) == ceil(rows * p.bytes_per_token)
