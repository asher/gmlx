#!/usr/bin/env python3
"""Prefill feeder (``gmlx.stream.prefill_feeder``): router-aware partial staging -
sparse slot fill at original expert indices, drain-on-new-pass interplay
with whole-layer staging, weight-swap restore - and the offload wrapper's
branch ordering between arena, partial and whole-layer paths. Pure CPU:
``arena_alloc`` is faked with numpy buffers and the "GGUF" is a temp file
of known bytes (fixture shared with test_decode_feeder)."""

from __future__ import annotations

from contextlib import contextmanager

import numpy as np
import mlx.core as mx
from mlx_lm.models.switch_layers import SwitchGLU

from gmlx.stream.expert_streaming import install_expert_streaming

from test_decode_feeder import (
    _KINDS,
    _STRIDE,
    _expert_bytes,
    _fake_arena_alloc,
    _holder_model,
    _make_fixture,
)


def _make_prefill_feeder(monkeypatch, tmp_path, n_layers=2):
    import mlx_kquant as kq
    from gmlx.stream.prefill_feeder import PrefillFeeder

    monkeypatch.setattr(kq, "arena_alloc", _fake_arena_alloc, raising=False)
    offsets, modules = _make_fixture(tmp_path, n_layers)
    # PrefillFeeder's offsets tuples are (path, off, nbytes, n_exp, kind)
    return PrefillFeeder(offsets, modules), modules


def _slot_expert(feeder, li, kind, e):
    view = feeder._views[(li, feeder._slot_of[li])][kind]
    return bytes(view[e].reshape(-1))


def test_gapped_coverage_alternates_slots(monkeypatch, tmp_path):
    """Covered layers with an interval (e.g. every other MoE layer) must
    alternate ring slots by covered position, not absolute layer parity:
    same-parity neighbors would otherwise share a slot and the pipelined
    staging of the next layer would overwrite the one in use."""
    import mlx_kquant as kq
    from gmlx.stream.prefill_feeder import PrefillFeeder

    monkeypatch.setattr(kq, "arena_alloc", _fake_arena_alloc, raising=False)
    offsets, modules = _make_fixture(tmp_path, 5)
    for li in (1, 3):
        del offsets[li]
        del modules[li]
    feeder = PrefillFeeder(offsets, modules)
    assert feeder._slot_of[0] != feeder._slot_of[2]
    assert feeder._slot_of[2] != feeder._slot_of[4]
    with feeder.prefill_call(modules[2][0], 2):
        # prefill_call pipelines staging of the next covered layer (4);
        # layer 2's staged bytes must survive it.
        assert feeder._ready[4].wait(5)
        for kind in _KINDS:
            for e in (0, 3):
                assert _slot_expert(feeder, 2, kind, e) == _expert_bytes(2, kind, e)
    assert feeder._error is None


def test_partial_call_stages_only_routed_slices(monkeypatch, tmp_path):
    feeder, modules = _make_prefill_feeder(monkeypatch, tmp_path)
    mod = modules[0][0]
    orig = {k: getattr(mod, f"{k}_proj").weight for k in _KINDS}
    with feeder.prefill_partial_call(mod, 0, [1, 3]):
        for kind in _KINDS:
            assert getattr(mod, f"{kind}_proj").weight is not orig[kind]
            for e in (1, 3):
                assert _slot_expert(feeder, 0, kind, e) == _expert_bytes(0, kind, e)
            for e in (0, 2):  # unrouted slices stay unstaged
                assert _slot_expert(feeder, 0, kind, e) == b"\x00" * _STRIDE[kind]
    for kind in _KINDS:
        assert getattr(mod, f"{kind}_proj").weight is orig[kind]


def test_partial_call_after_whole_pass_drains_and_restages(monkeypatch, tmp_path):
    feeder, modules = _make_prefill_feeder(monkeypatch, tmp_path)
    with feeder.prefill_call(modules[0][0], 0):
        for e in range(4):  # whole-layer staging filled every slice
            assert _slot_expert(feeder, 0, "gate", e) == _expert_bytes(0, "gate", e)
    # Same layer again = new pass (short final chunk after a big one):
    # partial staging reuses the slot; its routed slices are freshly read.
    with feeder.prefill_partial_call(modules[0][0], 0, [2]):
        assert _slot_expert(feeder, 0, "gate", 2) == _expert_bytes(0, "gate", 2)
    assert feeder._error is None


def test_wrapper_partial_branch_and_ordering(monkeypatch):
    """32..64-token calls with a prefill feeder take the partial path with
    the routed unique ids; >64-token calls take the whole-layer path; when a
    decode feeder is present and fits, it wins over partial staging."""
    mx.random.seed(13)
    glu = SwitchGLU(16, 32, 4)
    mx.eval(glu.parameters())
    model = _holder_model(glu)
    model.parameters = lambda: {"glu": glu.parameters()}
    monkeypatch.setattr(
        mx, "device_info", lambda: {"max_recommended_working_set_size": 1024}
    )
    real_set_wired = mx.set_wired_limit
    try:
        install_expert_streaming(model)

        class _FakeFdr:
            partial_calls = []
            whole_calls = []

            def covers(self, li):
                return True

            @contextmanager
            def prefill_partial_call(self, module, li, ids):
                self.partial_calls.append((li, list(ids)))
                yield

            @contextmanager
            def prefill_call(self, module, li):
                self.whole_calls.append(li)
                yield

        fdr = _FakeFdr()
        object.__setattr__(glu, "_kq_feeder", fdr)
        object.__setattr__(glu, "_kq_li", 3)

        x40 = mx.random.normal((1, 40, 16))
        i40 = mx.concatenate(
            [mx.zeros((1, 40, 1), dtype=mx.uint32),
             mx.full((1, 40, 1), 2, dtype=mx.uint32)], axis=-1)
        ref40 = mx.array(glu(x40, i40))
        mx.eval(ref40)
        assert fdr.partial_calls == [(3, [0, 2])]
        assert fdr.whole_calls == []

        x80 = mx.random.normal((1, 80, 16))
        i80 = mx.zeros((1, 80, 2), dtype=mx.uint32)
        mx.eval(glu(x80, i80))
        assert fdr.partial_calls == [(3, [0, 2])]
        assert fdr.whole_calls == [3]

        class _FakeDF:
            stage_calls = []
            fits = True

            def covers(self, li):
                return True

            def stage(self, li, ids):
                self.stage_calls.append(li)
                return ids.astype(np.uint32) if self.fits else None

            def wedged_at(self, li):
                return False

            def can_stage_smaller(self, li):
                return False  # overflow here means fall through, not split

            def has_dead(self, li):
                return False

            def redirect_dead(self, li, ids):
                return ids

            @contextmanager
            def swapped(self, li):
                yield

        df = _FakeDF()
        object.__setattr__(glu, "_kq_decode_feeder", df)
        out = mx.array(glu(x40, i40))
        mx.eval(out)
        assert mx.allclose(ref40, out, atol=1e-6, rtol=1e-6)
        assert df.stage_calls == [3]  # arena won
        assert fdr.partial_calls == [(3, [0, 2])]  # partial not re-entered

        df.fits = False  # arena overflow falls through to partial staging
        mx.eval(glu(x40, i40))
        assert df.stage_calls == [3, 3]
        assert fdr.partial_calls == [(3, [0, 2]), (3, [0, 2])]
    finally:
        mx.set_wired_limit = real_set_wired


def test_slot_itemsize_and_wide_slot_view():
    """>2 GiB stacks: the slot granule widens past uint8 and slot_view
    lands back on the layer geometry byte-for-byte."""
    from gmlx.stream.feeder_common import slot_itemsize, slot_view
    import mlx_kquant as kq

    # In-int32 slots stay plain uint8; past int32 the granule is the widest
    # divisor of every row length that fits the element count back in int32.
    assert slot_itemsize(1 << 20, [672, 784]) == 1
    assert slot_itemsize((1 << 31) - 64, [31]) == 1
    assert slot_itemsize(3 << 30, [672, 784]) == 8
    assert slot_itemsize(3 << 30, [924]) == 4          # 924 % 8 == 4
    assert slot_itemsize((1 << 31) + 64, [30]) == 2    # 30 % 4 == 2
    assert slot_itemsize(1 << 32, [31]) == 0           # nothing divides

    # Wide slot round trip: bytes written through the memoryview surface
    # in the uint8 view at the right geometry.
    arr, mv = kq.arena_alloc([64], itemsize=8)         # 512 B arena
    assert arr.dtype == mx.uint64 and len(mv) == 512
    pat = bytes(range(256)) + bytes(reversed(range(256)))
    mv[:512] = pat
    nbytes, shape = 384, (2, 3, 64)                    # 64 % 8 == 0
    v = slot_view(arr, nbytes, shape)
    mx.eval(v)
    assert v.dtype == mx.uint8 and v.shape == shape
    assert bytes(np.array(v).reshape(-1)) == pat[:384]

    # uint8 slots take the plain path unchanged.
    arr8, mv8 = kq.arena_alloc([96])
    mv8[:96] = pat[:96]
    v8 = slot_view(arr8, 96, (2, 48))
    mx.eval(v8)
    assert bytes(np.array(v8).reshape(-1)) == pat[:96]


def test_release_slots_and_lazy_realloc(monkeypatch, tmp_path):
    """Decode releases the ring; the next prefill pass rebuilds it and
    staging works as before."""
    feeder, modules = _make_prefill_feeder(monkeypatch, tmp_path)
    with feeder.prefill_call(modules[0][0], 0):
        pass
    feeder.release_slots()
    assert feeder._slots == [] and feeder._views == {}
    feeder.release_slots()  # idempotent
    with feeder.prefill_call(modules[0][0], 0):
        for kind in _KINDS:
            assert _slot_expert(feeder, 0, kind, 1) == _expert_bytes(0, kind, 1)
    assert feeder._error is None


def test_ring_slots_are_wired_for_the_pass(monkeypatch, tmp_path):
    """The ring wires at allocation, like the arena, and unwires before
    its buffers drop."""
    import gmlx.stream.prefill_feeder as pfm

    locked, unlocked = [], []
    monkeypatch.setattr(
        pfm, "lock_pages",
        lambda mv: locked.append(len(mv)) or (id(mv), len(mv)))
    monkeypatch.setattr(pfm, "unlock_pages", lambda e: unlocked.append(e))
    monkeypatch.delenv("GMLX_DECODE_ARENA_MLOCK", raising=False)
    feeder, modules = _make_prefill_feeder(monkeypatch, tmp_path)
    assert len(locked) == 2 * len(_KINDS)
    assert sum(locked) >= 2 * feeder.slot_bytes
    assert len(feeder._locked) == len(locked)
    feeder.release_slots()
    assert len(unlocked) == len(locked) and feeder._locked == []
    with feeder.prefill_call(modules[0][0], 0):  # the rebuild wires again
        pass
    assert len(locked) == 4 * len(_KINDS)


def test_unshare_on_fork_marks_only_the_whole_pages_inside():
    # A page shared with other data (a heap buffer) must stay inherited:
    # marking it makes every later fork child in the process die before
    # its exec.
    import mmap
    import subprocess

    from gmlx.stream.feeder_common import inner_pages, unshare_on_fork

    page = mmap.PAGESIZE
    assert inner_pages(page, 3 * page) == (page, 4 * page)
    assert inner_pages(page + 1, 3 * page) == (2 * page, 4 * page)
    assert inner_pages(page + 1, page) == (2 * page, 2 * page)     # none
    m = mmap.mmap(-1, 4 * page)
    assert unshare_on_fork(memoryview(m)) is True
    assert unshare_on_fork(memoryview(m)[100:5000]) is False     # no whole page
    del m
    assert unshare_on_fork(memoryview(bytearray(64))) is False
    assert unshare_on_fork(memoryview(bytearray(3 * page))) in (True, False)
    assert subprocess.run(["/usr/bin/true"]).returncode == 0


def test_ring_bytes_is_twice_the_largest_layer(tmp_path):
    from gmlx.stream.prefill_feeder import ring_bytes

    offsets, _ = _make_fixture(tmp_path, 3)
    layer = sum(r[2] for r in offsets[0])
    assert ring_bytes(offsets) == 2 * layer
    # A layer with a missing kind is not covered and does not size the ring.
    offsets[1] = [r for r in offsets[1] if r[4] != "down"]
    offsets[2] = [(p, o, n * 3, e, k) for p, o, n, e, k in offsets[2]]
    assert ring_bytes(offsets) == 2 * 3 * layer


def test_read_range_aligned_matches_plain_reads(tmp_path, monkeypatch):
    """Every pread lands page-aligned; the bytes match a plain read for
    ranges that start, end, or sit inside a page, and at the file tail."""
    import mmap
    import os

    from gmlx.stream.feeder_common import read_range, read_range_aligned

    page = mmap.PAGESIZE
    data = os.urandom(5 * page + 123)
    path = tmp_path / "f.bin"
    path.write_bytes(data)
    fd = os.open(path, os.O_RDONLY)
    seen = []
    real = os.preadv

    def spy(fd_, bufs, off):
        seen.append((off, sum(len(b) for b in bufs)))
        return real(fd_, bufs, off)

    monkeypatch.setattr(os, "preadv", spy)
    try:
        for off, n in [(0, page), (7, 100), (page - 5, 10), (page + 3, 2 * page + 9),
                       (0, len(data)), (len(data) - 50, 50), (page, page),
                       (3, 0), (4 * page + 100, page + 23)]:
            want = bytearray(n)
            read_range(fd, memoryview(want), off)
            seen.clear()
            got = bytearray(n)
            read_range_aligned(fd, memoryview(got), off, len(data))
            assert got == want == data[off:off + n]
            for o, ln in seen:
                assert o % page == 0
                assert ln % page == 0 or o + ln == len(data)
    finally:
        os.close(fd)
