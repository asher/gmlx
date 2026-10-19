#!/usr/bin/env python3
"""Prefill feeder (``gmlx.stream.prefill_feeder``): router-aware partial staging -
sparse slot fill at original expert indices, drain-on-new-pass interplay
with whole-layer staging, weight-swap restore - and the offload wrapper's
branch ordering between arena, partial and whole-layer paths. Pure CPU:
``arena_alloc`` is faked with numpy buffers and the "GGUF" is a temp file
of known bytes (fixture shared with test_decode_feeder)."""

from __future__ import annotations

import threading
from contextlib import contextmanager

import numpy as np
import pytest
import mlx.core as mx
from gmlx._vendor.mlx_lm.models.switch_layers import SwitchGLU

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
            def prefill_partial_call(self, module, li, ids, routing=None):
                self.partial_calls.append((li, list(ids)))
                yield

            @contextmanager
            def prefill_call(self, module, li, ids=None):
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
            def forward_start(self, li):
                pass

            def repay_at(self, li, ids, ring_ok=False):
                return False

            stage_calls = []
            fits = True

            def covers(self, li):
                return True

            def slot_count(self, li):
                return 1 << 16


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


def test_staged_ahead_and_ring_live(monkeypatch, tmp_path):
    """The arena split rule reads these. A whole-layer call starts the
    next layer's read, which counts as staged ahead in its own pass only.
    A partial call starts no read ahead."""
    feeder, modules = _make_prefill_feeder(monkeypatch, tmp_path, n_layers=3)
    assert feeder.ring_live()
    assert not feeder.staged_ahead(1)
    with feeder.prefill_partial_call(modules[0][0], 0, [1]):
        pass
    assert not feeder.staged_ahead(1)
    with feeder.prefill_call(modules[0][0], 0):
        pass
    assert feeder.staged_ahead(1)
    assert not feeder.staged_ahead(0) and not feeder.staged_ahead(2)
    feeder.release_slots()
    assert not feeder.ring_live() and not feeder.staged_ahead(1)
    with feeder.prefill_call(modules[0][0], 0):
        pass
    assert feeder.ring_live() and feeder.staged_ahead(1)
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


def _seeded_pair(monkeypatch, tmp_path, n_layers=3, slots=2):
    from test_decode_feeder import _make_feeder

    dfeeder, _ = _make_feeder(
        monkeypatch, tmp_path, slots_per_layer=slots, n_layers=n_layers)
    feeder, modules = _make_prefill_feeder(monkeypatch, tmp_path, n_layers)
    feeder._seed_hook = dfeeder.seed_from_ring
    return dfeeder, feeder, modules


def test_prefill_pass_seeds_decode_arena(monkeypatch, tmp_path):
    """A whole-layer pass copies each layer's most-routed experts from its
    ring slot into the empty decode-arena slots; decode then hits them
    without a read. The ring must not rewrite a slot mid-copy: seeding is
    slowed so the next-but-one layer's staging would race it."""
    import time

    from test_decode_feeder import _arena_slot

    dfeeder, feeder, modules = _seeded_pair(monkeypatch, tmp_path)
    real = dfeeder._seed_layer

    def slow(li, pairs, mvs):
        time.sleep(0.15)
        real(li, pairs, mvs)

    monkeypatch.setattr(dfeeder, "_seed_layer", slow)
    routing = {
        0: np.array([[3, 1], [3, 0], [3, 1]]),  # top-2: 3, 1
        1: np.array([[2, 0], [2, 1]]),  # top-2: 2, 0 (tie broken low)
        2: np.array([[1, 2]]),
    }
    for li in range(3):
        with feeder.prefill_call(modules[li][0], li, ids=mx.array(routing[li])):
            pass
    feeder.release_slots()  # seeds the last layer, joins every copy
    assert feeder._seed_prev is None and not feeder._seed_futs
    assert feeder._error is None
    dfeeder.ensure_wired()  # publishes the seeds
    assert dfeeder._seeded == 6
    for li, top in ((0, (3, 1)), (1, (2, 0)), (2, (1, 2))):
        assert set(dfeeder._owner[li].tolist()) == set(top)
        for e in top:
            s = int(dfeeder._slot_of[li][e])
            assert s >= 0
            for kind in _KINDS:
                assert _arena_slot(dfeeder, li, kind, s) == _expert_bytes(li, kind, e)
    assert dfeeder._counts[0][3] > dfeeder._counts[0][1] > 0
    # Three prompt tokens routed expert 3 every time: its count is worth
    # the seed weight of 32 tokens, not 32 per routing (batch dim ignored).
    assert dfeeder._counts[0][3] == pytest.approx(32.0)
    dfeeder.stage(0, np.array([[3, 1]], dtype=np.uint32))
    assert dfeeder._hits == 2 and dfeeder._lookups == 2


def test_partial_pass_seeds_only_staged_experts(monkeypatch, tmp_path):
    from test_decode_feeder import _arena_slot

    dfeeder, feeder, modules = _seeded_pair(monkeypatch, tmp_path, n_layers=1)
    # Routing names expert 2 most, but the partial pass staged only 1 and
    # 3: seeding never copies an unstaged slice.
    with feeder.prefill_partial_call(
            modules[0][0], 0, [1, 3],
            routing=mx.array([[2, 3], [2, 1], [2, 3]])):
        pass
    feeder.release_slots()
    dfeeder.stage(0, np.array([[3]], dtype=np.uint32))  # publishes first
    assert set(dfeeder._owner[0].tolist()) == {1, 3}
    assert dfeeder._hits == 1
    s = int(dfeeder._slot_of[0][1])
    assert _arena_slot(dfeeder, 0, "down", s) == _expert_bytes(0, "down", 1)


def test_seeding_off_leaves_arena_cold(monkeypatch, tmp_path):
    monkeypatch.setenv("GMLX_DECODE_SEED", "0")
    dfeeder, feeder, modules = _seeded_pair(monkeypatch, tmp_path, n_layers=1)
    with feeder.prefill_call(modules[0][0], 0, ids=mx.array([[0, 1]])):
        pass
    feeder.release_slots()
    dfeeder.ensure_wired()
    assert dfeeder._seeded == 0 and (dfeeder._owner[0] == -1).all()


def _sourced_pair(monkeypatch, tmp_path, n_layers=2):
    """A ring and a decode arena whose layers 0..n-1 hold experts 1 and 3
    (in slots 0 and 1), wired as the loader wires them."""
    dfeeder, feeder, modules = _seeded_pair(monkeypatch, tmp_path, n_layers)
    feeder._arena_hook = dfeeder.arena_sources
    for li in range(n_layers):
        dfeeder.stage(li, np.array([[3, 1]], dtype=np.uint32))
    return dfeeder, feeder, modules


def _record_reads(monkeypatch, feeder):
    reads = []
    real = feeder._submit_read

    def rec(path, fd, dest, off):
        reads.append((off, len(dest)))
        return real(path, fd, dest, off)

    monkeypatch.setattr(feeder, "_submit_read", rec)
    return reads


def _layer_reads(feeder, li, reads):
    """kind -> sorted (expert offset, length) reads inside layer li's stack."""
    out = {}
    for kind, (_, _, off, nbytes) in feeder._layers[li].items():
        out[kind] = sorted((o - off, n) for o, n in reads
                           if off <= o < off + nbytes)
    return out


@pytest.mark.parametrize("from_arena", [True, False])
def test_ring_copies_the_experts_the_arena_holds(
        monkeypatch, tmp_path, from_arena):
    """A ring stage copies the experts the decode arena holds and reads
    only the others from the drive. The slot is the layer's stacks byte for
    byte either way."""
    if not from_arena:
        monkeypatch.setenv("GMLX_RING_FROM_ARENA", "0")
    dfeeder, feeder, modules = _sourced_pair(monkeypatch, tmp_path)
    reads = _record_reads(monkeypatch, feeder)
    with feeder.prefill_call(modules[0][0], 0):
        for kind in _KINDS:
            for e in range(4):
                assert _slot_expert(feeder, 0, kind, e) == \
                    _expert_bytes(0, kind, e)
    for kind, got in _layer_reads(feeder, 0, reads).items():
        st = _STRIDE[kind]
        want = [(0, st), (2 * st, st)] if from_arena else [(0, 4 * st)]
        assert got == want, kind
    # Layers 0 and 1 (kicked ahead), two experts each.
    assert feeder._bytes_copied == (4 * sum(_STRIDE.values())
                                    if from_arena else 0)
    feeder.release_slots()
    assert dfeeder._source_reads == 0
    # The arena kept its residents.
    assert dfeeder._owner[0].tolist() == [1, 3]


def test_partial_call_copies_the_routed_experts_the_arena_holds(
        monkeypatch, tmp_path):
    dfeeder, feeder, modules = _sourced_pair(monkeypatch, tmp_path, 1)
    reads = _record_reads(monkeypatch, feeder)
    with feeder.prefill_partial_call(modules[0][0], 0, [1, 2]):
        for kind in _KINDS:
            for e in (1, 2):
                assert _slot_expert(feeder, 0, kind, e) == \
                    _expert_bytes(0, kind, e)
            for e in (0, 3):  # unrouted, held or not: never staged
                assert _slot_expert(feeder, 0, kind, e) == \
                    b"\x00" * _STRIDE[kind]
    for kind, got in _layer_reads(feeder, 0, reads).items():
        assert got == [(2 * _STRIDE[kind], _STRIDE[kind])], kind
    assert dfeeder._source_reads == 0


@pytest.mark.parametrize("writer", ["stage", "prestage", "resize", "zero"])
def test_arena_writes_wait_for_the_ring_copies(
        monkeypatch, tmp_path, writer):
    """A stage of the ring copies out of arena slots on its own thread.
    Until it gives them back, no arena write may evict or move them."""
    dfeeder, feeder, _ = _sourced_pair(monkeypatch, tmp_path, 1)
    src = feeder._take_sources(0)
    assert src is not None and sorted(src.experts.tolist()) == [1, 3]
    done = threading.Event()

    def write():
        if writer == "stage":  # two misses evict both residents
            dfeeder.stage(0, np.array([[0, 2]], dtype=np.uint32))
        elif writer == "prestage":
            dfeeder.prestage(0, np.array([[0, 2]]), demand=True)
        elif writer == "resize":
            dfeeder._resize_layer(0, 1)
        else:  # a full layer zeroes a resident for its zero slot
            dfeeder._zero_slot(0, np.zeros(2, dtype=bool))
        done.set()

    threading.Thread(target=write, daemon=True).start()
    assert not done.wait(0.3), f"{writer} ran under a ring copy"
    assert dfeeder._owner[0].tolist() == [1, 3]
    src.release()
    assert done.wait(5)


def test_decode_stage_waits_for_a_live_ring_copy(monkeypatch, tmp_path):
    """A kicked-ahead stage copies out of layer 1's arena slots while a
    decode stage() evicts both residents. The ring slot keeps the file's
    bytes."""
    import time

    import gmlx.stream.prefill_feeder as pfm

    real = pfm.copy_experts

    def slow(*args):
        time.sleep(0.3)
        real(*args)

    monkeypatch.setattr(pfm, "copy_experts", slow)
    dfeeder, feeder, modules = _sourced_pair(monkeypatch, tmp_path)
    with feeder.prefill_call(modules[0][0], 0):  # kicks layer 1
        pass
    dfeeder.stage(1, np.array([[0, 2]], dtype=np.uint32))
    assert sorted(dfeeder._owner[1].tolist()) == [0, 2]
    assert feeder._ready[1].wait(5) and feeder._error is None
    with feeder.prefill_call(modules[1][0], 1):
        for kind in _KINDS:
            for e in range(4):
                assert _slot_expert(feeder, 1, kind, e) == \
                    _expert_bytes(1, kind, e)


def test_a_later_chunk_copies_what_an_earlier_chunk_seeded(
        monkeypatch, tmp_path):
    """A long cold prompt: chunk 1 seeds the empty arena from the ring, and
    chunk 2 copies those experts instead of reading them again."""
    dfeeder, feeder, modules = _seeded_pair(monkeypatch, tmp_path, 2)
    feeder._arena_hook = dfeeder.arena_sources
    routing = {0: mx.array([[3, 1], [3, 1]]), 1: mx.array([[2, 0]])}
    for li in (0, 1):  # chunk 1
        with feeder.prefill_call(modules[li][0], li, ids=routing[li]):
            pass
    dfeeder._seeds[0][1].result()
    reads = _record_reads(monkeypatch, feeder)
    with feeder.prefill_call(modules[0][0], 0):  # chunk 2
        for kind in _KINDS:
            for e in range(4):
                assert _slot_expert(feeder, 0, kind, e) == \
                    _expert_bytes(0, kind, e)
    assert sorted(dfeeder._owner[0].tolist()) == [1, 3]
    for kind, got in _layer_reads(feeder, 0, reads).items():
        st = _STRIDE[kind]
        assert got == [(0, st), (2 * st, st)], kind


def test_a_failed_partial_copy_waits_for_the_reads(monkeypatch, tmp_path):
    """A partial call whose arena copy fails still waits for its reads,
    which would otherwise write the slot the next pass stages."""
    import gmlx.stream.prefill_feeder as pfm

    def fail(*args):
        raise RuntimeError("copy failed")

    gate = threading.Event()

    def gated(real):
        def read(*args):
            gate.wait(5)
            return real(*args)
        return read

    monkeypatch.setattr(pfm, "copy_experts", fail)
    for name in ("read_range", "read_range_aligned"):
        monkeypatch.setattr(pfm, name, gated(getattr(pfm, name)))
    dfeeder, feeder, modules = _sourced_pair(monkeypatch, tmp_path, 1)
    futs = []
    submit = feeder._submit_read

    def rec(*args):
        futs.append(submit(*args))
        return futs[-1]

    monkeypatch.setattr(feeder, "_submit_read", rec)
    threading.Timer(0.3, gate.set).start()
    with pytest.raises(RuntimeError, match="copy failed"):
        with feeder.prefill_partial_call(modules[0][0], 0, [1, 2]):
            pass
    assert futs and all(f.done() for f in futs)
    assert dfeeder._source_reads == 0


def test_a_partial_call_counts_the_experts_it_stages(monkeypatch, tmp_path):
    dfeeder, feeder, modules = _sourced_pair(monkeypatch, tmp_path, 1)
    with feeder.prefill_partial_call(modules[0][0], 0, [1, 2]):
        pass
    per_expert = sum(_STRIDE.values())
    assert feeder._bytes_copied == per_expert  # expert 1, from the arena
    assert feeder._bytes_staged == 2 * per_expert


def test_a_failed_arena_claim_leaves_no_stage_to_wait_for(
        monkeypatch, tmp_path):
    """A raise while claiming the arena's residents leaves no staging
    marked in flight, so the next pass does not wait out the stage
    timeout for it."""
    dfeeder, feeder, modules = _sourced_pair(monkeypatch, tmp_path, 1)

    def fail(li, geometry):
        raise RuntimeError("verify failed")

    feeder._arena_hook = fail
    with pytest.raises(RuntimeError, match="verify failed"):
        with feeder.prefill_call(modules[0][0], 0):
            pass
    assert 0 not in feeder._ready
    feeder._arena_hook = dfeeder.arena_sources
    with feeder.prefill_call(modules[0][0], 0):
        assert _slot_expert(feeder, 0, "down", 1) == _expert_bytes(0, "down", 1)


def test_a_wedged_ring_gives_the_arena_back(monkeypatch, tmp_path):
    """A stage queued behind a wedged read never copies. Taking the ring
    out of service gives its arena slots back, so decode does not wait."""
    dfeeder, feeder, modules = _sourced_pair(monkeypatch, tmp_path, 3)
    gate = _wedge_stage(monkeypatch, feeder, 1)
    try:
        with feeder.prefill_call(modules[0][0], 0):  # kicks layer 1 too
            pass
        assert dfeeder._source_reads == 1
        with pytest.raises(RuntimeError, match="out of service"):
            with feeder.prefill_call(modules[0][0], 0):
                pass
        assert dfeeder._source_reads == 0
        dfeeder.stage(1, np.array([[0, 2]], dtype=np.uint32))
    finally:
        _release_wedge(feeder, gate)


def test_a_claim_held_past_the_join_timeout_ends_arena_copies(
        monkeypatch, tmp_path):
    import gmlx.stream.decode_feeder as dfm

    monkeypatch.setattr(dfm, "_SOURCE_JOIN_S", 0.1)
    dfeeder, feeder, _ = _sourced_pair(monkeypatch, tmp_path, 1)
    src = feeder._take_sources(0)
    dfeeder.stage(0, np.array([[0, 2]], dtype=np.uint32))  # after 0.1 s
    assert dfeeder._source_reads == 0
    assert feeder._take_sources(0) is None
    src.release()  # a late release leaves the count at zero
    assert dfeeder._source_reads == 0


def test_ring_depth_from_env(monkeypatch, tmp_path):
    """GMLX_PREFILL_RING_SLOTS deepens the ring: slots rotate over that
    many, ring_bytes prices them all, a call stages n-1 layers ahead and
    a pass limit stops staging past the last layer the pass uses."""
    from gmlx.stream.prefill_feeder import ring_bytes

    monkeypatch.setenv("GMLX_PREFILL_RING_SLOTS", "3")
    feeder, modules = _make_prefill_feeder(monkeypatch, tmp_path, n_layers=5)
    assert feeder.n_slots == 3
    assert [feeder._slot_of[li] for li in range(5)] == [0, 1, 2, 0, 1]
    (tmp_path / "one").mkdir()
    offsets, _ = _make_fixture(tmp_path / "one", 1)
    assert ring_bytes(offsets) == 3 * sum(r[2] for r in offsets[0])
    with feeder.prefill_call(modules[0][0], 0):
        pass
    assert set(feeder._ready) == {0, 1, 2}
    feeder.limit_pass(3)
    with feeder.prefill_call(modules[1][0], 1):
        pass
    assert set(feeder._ready) == {0, 1, 2, 3}
    with feeder.prefill_call(modules[2][0], 2):
        for e in range(4):
            assert _slot_expert(feeder, 2, "gate", e) == _expert_bytes(2, "gate", e)
    assert 4 not in feeder._ready
    assert feeder._error is None
    feeder.limit_pass(None)
    monkeypatch.setenv("GMLX_PREFILL_RING_SLOTS", "x")
    from gmlx.stream.prefill_feeder import ring_slots

    assert ring_slots() == 2


def _wedge_stage(monkeypatch, feeder, layer):
    """Make ``layer``'s staging hang like a read wedged in the kernel until
    the returned event is set. Pair with ``_release_wedge``."""
    import gmlx.stream.prefill_feeder as pfm

    monkeypatch.setattr(pfm, "_STAGE_TIMEOUT_S", 0.2)
    monkeypatch.setattr(pfm, "_QUARANTINED", [])
    stage, gate = feeder._stage, threading.Event()

    def wedged(li):
        if li == layer:
            gate.wait()
        stage(li)

    monkeypatch.setattr(feeder, "_stage", wedged)
    return gate


def _close_or_fail(feeder):
    """Close a feeder whose worker is wedged. A close that joins the worker
    would hang, so it runs on a daemon thread."""
    closer = threading.Thread(target=feeder.close, daemon=True)
    closer.start()
    closer.join(5)
    assert not closer.is_alive(), "close joined the wedged worker"


def _release_wedge(feeder, gate):
    """End the wedged stage, close the feeder, then close the fds its
    quarantine keeps open for the process."""
    import os

    import gmlx.stream.prefill_feeder as pfm

    gate.set()
    for ev in list(feeder._ready.values()):
        assert ev.wait(30)
    feeder.close()
    for held in pfm._QUARANTINED:
        if isinstance(held, dict):
            for fd in held.values():
                os.close(fd)
    pfm._QUARANTINED.clear()


def test_a_wedged_stage_takes_the_ring_out_of_service(monkeypatch, tmp_path):
    """A stage that outlived the timeout may still complete into its slot.
    The next pass must not stage into the ring: every layer takes the
    page-cache path and the slots stay allocated, never reused."""
    feeder, modules = _make_prefill_feeder(monkeypatch, tmp_path, n_layers=3)
    gate = _wedge_stage(monkeypatch, feeder, 1)
    try:
        with feeder.prefill_call(modules[0][0], 0):   # kicks layer 1 too
            pass
        slots = feeder._slots
        # A new pass: the drain finds layer 1 still staging.
        with pytest.raises(RuntimeError, match="out of service"):
            with feeder.prefill_partial_call(modules[0][0], 0, [2]):
                pass
        assert feeder._wedged == [1]
        assert not any(feeder.covers(li) for li in range(3))
        with pytest.raises(RuntimeError, match="out of service"):
            with feeder.prefill_call(modules[0][0], 0):
                pass
        feeder.release_slots()
        assert feeder._slots is slots, "a quarantined slot was dropped"
        # The late stage ends. The ring stays out of service all the same.
        gate.set()
        assert feeder._ready[1].wait(5)
        feeder.release_slots()
        assert feeder._slots is slots, "a quarantined slot was dropped"
        with pytest.raises(RuntimeError, match="out of service"):
            with feeder.prefill_call(modules[0][0], 0):
                pass
        assert not any(feeder.covers(li) for li in range(3))
    finally:
        _release_wedge(feeder, gate)


def test_a_stage_timeout_in_the_call_takes_the_ring_out_of_service(
        monkeypatch, tmp_path):
    feeder, modules = _make_prefill_feeder(monkeypatch, tmp_path)
    gate = _wedge_stage(monkeypatch, feeder, 0)
    try:
        with pytest.raises(RuntimeError, match="timed out"):
            with feeder.prefill_call(modules[0][0], 0):
                pass
        assert feeder._wedged == [0] and not feeder.covers(0)
    finally:
        _release_wedge(feeder, gate)


def test_ring_release_keeps_the_slots_of_a_wedged_stage(monkeypatch, tmp_path):
    """Decode releases the ring. A slot a wedged read may still write is
    never freed: freed, it would return to MLX's buffer cache and another
    array would get the late bytes. It is unwired, so it stops counting
    against the wired budget."""
    import gmlx.stream.prefill_feeder as pfm

    locked, unlocked = [], []
    monkeypatch.setattr(
        pfm, "lock_pages",
        lambda mv: locked.append((id(mv), len(mv))) or locked[-1])
    monkeypatch.setattr(pfm, "unlock_pages", lambda e: unlocked.append(e))
    monkeypatch.delenv("GMLX_DECODE_ARENA_MLOCK", raising=False)
    feeder, modules = _make_prefill_feeder(monkeypatch, tmp_path, n_layers=3)
    gate = _wedge_stage(monkeypatch, feeder, 1)
    try:
        with feeder.prefill_call(modules[0][0], 0):
            pass
        slots = feeder._slots
        assert feeder.release_slots() is False  # the arena keeps its lend
        assert feeder.release_slots() is False  # and on every later try
        assert feeder._slots is slots
        assert unlocked == locked and feeder._locked == []
        assert any(q is slots for q in pfm._QUARANTINED)
        assert not feeder.covers(0)
    finally:
        _release_wedge(feeder, gate)


def test_the_drain_waits_one_timeout_for_all_the_staging(monkeypatch, tmp_path):
    """A new pass gives the stages in flight one timeout between them, not
    one each. Layer 2 is queued behind the wedged layer 1 on the one stage
    worker, so it never starts either."""
    from types import SimpleNamespace

    import gmlx.stream.prefill_feeder as pfm

    monkeypatch.setenv("GMLX_PREFILL_RING_SLOTS", "3")
    feeder, modules = _make_prefill_feeder(monkeypatch, tmp_path, n_layers=4)
    gate = _wedge_stage(monkeypatch, feeder, 1)
    monkeypatch.setattr(pfm, "_STAGE_TIMEOUT_S", 30.0)
    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr(pfm, "time", SimpleNamespace(monotonic=lambda: clock.now))
    waits, started = [], []
    wedged = feeder._stage

    def stage(li):
        started.append(li)
        wedged(li)

    monkeypatch.setattr(feeder, "_stage", stage)

    class Clocked:
        """A stage event whose wait runs out on the clock: no stage can end
        while the gate holds layer 1."""

        def __init__(self, ev):
            self.ev = ev

        def set(self):
            self.ev.set()

        def wait(self, timeout=None):
            waits.append(timeout)
            if self.ev.is_set():
                return True
            clock.now += timeout
            return False

    real = {}
    try:
        with feeder.prefill_call(modules[0][0], 0):   # kicks layers 1 and 2
            pass
        assert sorted(feeder._ready) == [0, 1, 2] and feeder._ready[0].is_set()
        real = {li: feeder._ready[li] for li in (1, 2)}
        feeder._ready.update({li: Clocked(ev) for li, ev in real.items()})
        with pytest.raises(RuntimeError, match="out of service"):
            with feeder.prefill_call(modules[0][0], 0):
                pass
        assert feeder._wedged == [1, 2] and 2 not in started
        assert waits == [30.0, 0.0]
    finally:
        feeder._ready.update(real)
        _release_wedge(feeder, gate)


def test_a_wedged_read_leaves_the_model_and_its_fds_to_the_process(
        monkeypatch, tmp_path):
    """A read wedged in the kernel blocks its stage worker for the life of
    the process, and that worker's frame holds the feeder. The feeder lets
    go of the MoE modules and of the decode feeder behind its hooks, so
    they are freed with the model, and its close returns. Its fds stay
    open: the wedged read still uses one, and a closed fd number goes to
    the next open."""
    import gc
    import os
    import weakref

    import gmlx.stream.prefill_feeder as pfm

    monkeypatch.setattr(pfm, "_STAGE_TIMEOUT_S", 0.2)
    monkeypatch.setattr(pfm, "_QUARANTINED", [])
    monkeypatch.setattr(pfm, "lock_pages", lambda mv: None)
    feeder, modules = _make_prefill_feeder(monkeypatch, tmp_path, n_layers=3)
    wedge_off = feeder._layers[1]["gate"][2]
    gate = threading.Event()
    name = "read_range_aligned" if feeder._nocache else "read_range"
    real = getattr(pfm, name)

    def maybe_wedge(fd, dest, off, *rest):
        if off == wedge_off:
            gate.wait()
        return real(fd, dest, off, *rest)

    monkeypatch.setattr(pfm, name, maybe_wedge)

    from gmlx.stream.decode_feeder import ArenaSources

    class DecodeStandIn:
        def __init__(self):
            self.arena = {k: np.zeros(4 * _STRIDE[k], dtype=np.uint8)
                          for k in _KINDS}

        def seed_from_ring(self, *args):
            return []

        def lend_for_ring(self, nbytes):
            pass

        def arena_sources(self, li, geometry):
            # Expert 2 only: layer 1 still reads expert 0, where it wedges.
            bufs = {k: ((a, memoryview(a)), _STRIDE[k])
                    for k, a in self.arena.items()}
            return ArenaSources(np.array([2]), np.array([2]), bufs,
                                self.end_read)

        def end_read(self):
            pass

    decode = DecodeStandIn()
    feeder._seed_hook = decode.seed_from_ring
    feeder._lend_hook = decode.lend_for_ring
    feeder._arena_hook = decode.arena_sources
    fds = dict(feeder._fds)
    try:
        with feeder.prefill_call(modules[0][0], 0, ids=mx.array([[0, 1]])):
            pass   # kicks layer 1, whose first read wedges
        with pytest.raises(RuntimeError, match="out of service"):
            with feeder.prefill_call(modules[0][0], 0):
                pass
        assert feeder._wedged == [1]
        assert feeder._seed_prev is None and not feeder._seed_ids
        _close_or_fail(feeder)
        assert fds in pfm._QUARANTINED
        for fd in fds.values():
            os.fstat(fd)   # raises on a closed fd
        mod_refs = [weakref.ref(m) for ms in modules.values() for m in ms]
        decode_ref = weakref.ref(decode)
        arena_ref = weakref.ref(decode.arena["gate"])
        del modules, decode
        gc.collect()
        assert all(r() is None for r in mod_refs), "a MoE module outlived the model"
        assert decode_ref() is None, "the decode feeder outlived the model"
        assert arena_ref() is None, "the decode arena outlived the model"
    finally:
        gate.set()
        assert feeder._ready[1].wait(5)
        for fd in fds.values():
            os.close(fd)



def test_a_failed_seed_copy_leaves_the_decode_feeder_to_the_model(
        monkeypatch, tmp_path):
    """A failed seed copy holds its exception, and the traceback holds the
    decode feeder. After a wedge, neither the feeder's seed state nor the
    wedged stage's frame keeps such a copy. Layer 2 shares slot 0 with
    layer 0, so its stage joins layer 0's failed copy before its read
    wedges, and layer 1's failed copy is still in the seed state when the
    call times out."""
    import gc
    import os
    import weakref
    from concurrent.futures import Future

    import gmlx.stream.prefill_feeder as pfm

    monkeypatch.setattr(pfm, "_STAGE_TIMEOUT_S", 0.2)
    monkeypatch.setattr(pfm, "_QUARANTINED", [])
    monkeypatch.setattr(pfm, "lock_pages", lambda mv: None)
    feeder, modules = _make_prefill_feeder(monkeypatch, tmp_path, n_layers=3)
    wedge_off = feeder._layers[2]["gate"][2]
    started, gate = threading.Event(), threading.Event()
    name = "read_range_aligned" if feeder._nocache else "read_range"
    real = getattr(pfm, name)

    def maybe_wedge(fd, dest, off, *rest):
        if off == wedge_off:
            started.set()
            gate.wait()
        return real(fd, dest, off, *rest)

    monkeypatch.setattr(pfm, name, maybe_wedge)

    class DecodeStandIn:
        def _seed_layer(self, li):
            raise MemoryError(f"seed copy of layer {li} failed")

        def seed_from_ring(self, li, counts, n_tokens, mvs, present):
            fut = Future()
            try:
                self._seed_layer(li)
            except MemoryError as e:
                fut.set_exception(e)
            return [fut]

    decode = DecodeStandIn()
    feeder._seed_hook = decode.seed_from_ring
    fds = dict(feeder._fds)
    ids = mx.array([[0, 1]])
    try:
        for li in (0, 1):
            with feeder.prefill_call(modules[li][0], li, ids=ids):
                pass
        assert started.wait(5)
        assert 0 not in feeder._seed_futs, "layer 2's stage kept no copy"
        with pytest.raises(RuntimeError, match="timed out"):
            with feeder.prefill_call(modules[2][0], 2, ids=ids):
                pass
        assert feeder._wedged == [2]
        assert not feeder._seed_futs
        decode_ref = weakref.ref(decode)
        del decode
        gc.collect()
        assert decode_ref() is None, "a failed seed copy kept the decode feeder"
    finally:
        gate.set()
        assert feeder._ready[2].wait(5)
        feeder.close()
        for fd in fds.values():
            os.close(fd)

def test_a_failed_read_waits_for_its_siblings(monkeypatch, tmp_path):
    """A stage whose read fails still waits for its other reads before the
    layer counts as staged: one still in flight would write the slot the
    next pass stages."""
    import gmlx.stream.prefill_feeder as pfm

    feeder, _modules = _make_prefill_feeder(monkeypatch, tmp_path)
    offs = {kind: feeder._layers[0][kind][2] for kind in _KINDS}
    started, gate = threading.Event(), threading.Event()
    name = "read_range_aligned" if feeder._nocache else "read_range"
    real = getattr(pfm, name)

    def reads(fd, dest, off, *rest):
        if off == offs["gate"]:
            raise OSError(5, "Input/output error")
        if off == offs["down"]:
            started.set()
            gate.wait()
        return real(fd, dest, off, *rest)

    monkeypatch.setattr(pfm, name, reads)
    feeder._kick(0)
    try:
        assert started.wait(5)
        assert not feeder._ready[0].wait(0.3), \
            "the layer counted as staged under a read in flight"
    finally:
        gate.set()
    assert feeder._ready[0].wait(5)
    assert isinstance(feeder._error, OSError)


def test_a_wedged_partial_read_takes_the_ring_out_of_service(
        monkeypatch, tmp_path):
    """Partial staging reads outside the ring's stage events, so its wait
    carries the same timeout."""
    import os

    import gmlx.stream.prefill_feeder as pfm

    monkeypatch.setattr(pfm, "_STAGE_TIMEOUT_S", 0.2)
    monkeypatch.setattr(pfm, "_QUARANTINED", [])
    feeder, modules = _make_prefill_feeder(monkeypatch, tmp_path)
    gate = threading.Event()
    name = "read_range_aligned" if feeder._nocache else "read_range"
    real = getattr(pfm, name)

    def wedged(fd, dest, off, *rest):
        gate.wait()
        return real(fd, dest, off, *rest)

    monkeypatch.setattr(pfm, name, wedged)
    fds = dict(feeder._fds)
    raised = []

    def call():
        try:
            with feeder.prefill_partial_call(modules[0][0], 0, [1]):
                pass
        except RuntimeError as e:
            raised.append(str(e))

    caller = threading.Thread(target=call, daemon=True)
    try:
        caller.start()
        caller.join(5)
        assert not caller.is_alive(), "the partial staging wait has no timeout"
        assert raised == ["[feeder] partial staging of layer 0 timed out"]
        assert feeder._wedged == [0] and not feeder.covers(0)
        _close_or_fail(feeder)
        assert fds in pfm._QUARANTINED
    finally:
        gate.set()
        feeder._read_pool.shutdown(wait=True)
        for fd in fds.values():
            os.close(fd)


def test_process_exit_survives_a_wedged_prefill_read(tmp_path):
    """The stage and read workers are daemon threads, so a read that never
    returns does not hang interpreter exit."""
    import os
    import subprocess
    import sys
    import textwrap

    here = os.path.dirname(os.path.abspath(__file__))
    script = tmp_path / "wedge.py"
    script.write_text(textwrap.dedent(f"""
        import sys, threading
        from pathlib import Path
        sys.path.insert(0, {here!r})
        import mlx.core as mx
        mx.set_default_device(mx.cpu)
        import mlx_kquant as kq
        import gmlx.stream.prefill_feeder as pfm
        from test_decode_feeder import _fake_arena_alloc, _make_fixture

        kq.arena_alloc = _fake_arena_alloc
        pfm._STAGE_TIMEOUT_S = 0.2
        pfm.lock_pages = lambda mv: None
        offsets, modules = _make_fixture(Path({str(tmp_path)!r}), 2)
        feeder = pfm.PrefillFeeder(offsets, modules)
        forever = threading.Event()
        pfm.read_range_aligned = pfm.read_range = (
            lambda *args: forever.wait())
        try:
            with feeder.prefill_call(modules[0][0], 0):
                pass
        except RuntimeError as e:
            print(e)
        feeder.close()
        print("closed", flush=True)
    """))
    env = dict(os.environ, KQUANT_FORCE_CPU="1")
    r = subprocess.run(
        [sys.executable, str(script)], env=env, capture_output=True,
        text=True, timeout=60)
    assert r.returncode == 0, r.stderr[-2000:]
    assert "timed out" in r.stdout and "closed" in r.stdout


def test_finalizer_never_joins_its_pools(monkeypatch, tmp_path):
    """Same contract as the decode feeder: ``__del__`` never joins."""
    from gmlx.stream.feeder_common import DaemonPool

    waits = []
    orig = DaemonPool.shutdown

    def spy(self, wait=True, **kw):
        waits.append(wait)
        return orig(self, wait=wait, **kw)

    monkeypatch.setattr(DaemonPool, "shutdown", spy)
    collected, _ = _make_prefill_feeder(monkeypatch, tmp_path)
    collected.__del__()
    assert waits and all(w is False for w in waits)
    waits.clear()
    closed, _ = _make_prefill_feeder(monkeypatch, tmp_path)  # both stay bound: no GC here
    closed.close()
    assert waits and all(w is True for w in waits)


@pytest.mark.parametrize("timeshare", ["0", "1"])
@pytest.mark.parametrize("env_gb", [None, "100"])
@pytest.mark.parametrize("built", [True, False])
def test_arena_sizes_against_ram_before_the_ring_is_wired(
        monkeypatch, capsys, timeshare, env_gb, built):
    """The ring's slots are wired when its feeder is built, so a free-RAM
    snapshot taken after that would charge the ring twice. 7 GB free, 1 GB
    of KV room, a 2 GB ring: the arena keeping the ring apart gets 4 GB,
    and a time-shared arena gets 6 GB and lends 2 GB of it at prefill.
    A GMLX_DECODE_ARENA_GB past free RAM clamps to it less the ring and
    keeps the whole ring apart, as the serve footprint prices it. A ring
    that is never built is charged nowhere."""
    from types import SimpleNamespace

    import gmlx.stream.budget as budget
    import gmlx.stream.decode_feeder as dfm
    import gmlx.stream.expert_streaming as es
    import gmlx.stream.pin_weights as pw
    import gmlx.stream.prefetch as pf
    import gmlx.stream.prefill_feeder as pff

    GB = 1 << 30
    monkeypatch.setenv("GMLX_RING_TIMESHARE", timeshare)
    if env_gb:
        monkeypatch.setenv("GMLX_DECODE_ARENA_GB", env_gb)
    else:
        monkeypatch.delenv("GMLX_DECODE_ARENA_GB", raising=False)
    monkeypatch.delenv("GMLX_DECODE_ARENA_FORCE", raising=False)
    monkeypatch.setattr(mx, "set_wired_limit", mx.set_wired_limit)
    # Eight 1 GB layers in the file; the ring holds two of them.
    third = GB // 3
    offsets = {li: [("m.gguf", li * GB + k * third, third, 4, kind)
                    for k, kind in enumerate(("gate", "up", "down"))]
               for li in range(8)}
    free = {"bytes": 7 * GB}
    monkeypatch.setattr(es, "stream_budget_bytes", lambda: 100 * GB)
    monkeypatch.setattr(budget, "ceiling_bytes", lambda: 100 * GB)
    monkeypatch.setattr(budget, "_ram_floor_bytes", lambda ram: 0)
    monkeypatch.setattr(budget, "_available_ram_bytes",
                        lambda *a, **k: free["bytes"])
    monkeypatch.setattr(budget, "kv_room_bytes",
                        lambda path: SimpleNamespace(bytes=GB))
    monkeypatch.setattr(es, "_prefill_ring_reason", lambda *a: None)
    monkeypatch.setattr(pf, "maybe_make_prefetcher",
                        lambda path: SimpleNamespace(offsets=offsets))
    monkeypatch.setattr(pw, "maybe_pin_weights", lambda *a, **k: None)
    monkeypatch.setattr(pw, "cast_copies", lambda *a: SimpleNamespace(
        dead_bytes=0, names=frozenset()))

    def ring(offs, modules):
        if not built:
            return None
        free["bytes"] -= pff.ring_bytes(offs)  # wired at construction
        return SimpleNamespace(covers=lambda li: True, n_slots=2,
                               slot_bytes=GB, release_slots=lambda: None)

    sized = {}

    def arena(offs, modules, arena_bytes, *a, ring_share=0, **k):
        sized.update(arena=arena_bytes, share=ring_share)

    monkeypatch.setattr(pff, "maybe_make_prefill_feeder", ring)
    monkeypatch.setattr(dfm, "maybe_make_decode_feeder", arena)
    monkeypatch.setattr(es, "_install_gpu_residency",
                        lambda *a, room=None, **k: sized.update(room=room))
    monkeypatch.setenv("GMLX_GPU_RESIDENT", "1")
    glu = SwitchGLU(16, 32, 4)
    install_expert_streaming(_holder_model(glu), gguf_path="m.gguf",
                             force_stream=True)
    # The residency room charges the ring once, either way.
    ring_b = 6 * third
    want = {"0": (7 * GB - GB - ring_b, 0), "1": (6 * GB, ring_b)}[timeshare]
    if env_gb:
        want = (7 * GB - ring_b, 0)
    if not built:
        want, ring_b = ((7 if env_gb else 6) * GB, 0), 0
    assert sized == {"arena": want[0], "share": want[1],
                     "room": 100 * GB - want[0] - (ring_b - want[1])}
    # One clamp line, for the arena the loader keeps.
    clamps = [ln for ln in capsys.readouterr().out.splitlines()
              if "GMLX_DECODE_ARENA_GB=" in ln]
    assert len(clamps) == (1 if env_gb else 0)
    if env_gb:
        assert f"{want[0] / GB:.1f}GB" in clamps[0]
