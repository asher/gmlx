#!/usr/bin/env python3
"""Decode feeder (``gmlx.decode_feeder``): arena residency bookkeeping -
adopt-on-miss, popularity-driven eviction that never touches a slot the
current call routes to, overflow fallback, weight-swap restore - plus the
offload wrapper's decode-feeder branch. Pure CPU: ``arena_alloc`` is faked
with numpy buffers and the "GGUF" is a temp file of known bytes, so the
tests exercise every byte movement without Metal or a real model."""

from __future__ import annotations

from contextlib import contextmanager

import numpy as np
import mlx.core as mx
from mlx_lm.models.switch_layers import SwitchGLU

import gmlx.loader
from gmlx.loader import (
    _decode_arena_bytes,
    _resolve_feeder_defaults,
    install_expert_streaming,
)

_KINDS = ("gate", "up", "down")
_E = 4  # experts per layer
_STRIDE = {"gate": 64, "up": 64, "down": 48}  # bytes per expert slice


class _Proj:
    def __init__(self, w):
        self.weight = w
        self.bias = None


class _Mod:
    pass


def _expert_bytes(li: int, kind: str, e: int) -> bytes:
    k = _KINDS.index(kind)
    start = li * 1000 + k * 300 + e * 17
    return bytes((start + i) % 251 for i in range(_STRIDE[kind]))


def _make_fixture(tmp_path, n_layers=2):
    """Temp 'GGUF' + offsets map + fake modules whose weights are numpy
    copies of the file bytes (so the zero-copy verification passes)."""
    blob = bytearray()
    offsets: dict[int, list] = {}
    modules: dict[int, list] = {}
    for li in range(n_layers):
        mod = _Mod()
        offsets[li] = []
        for kind in _KINDS:
            off = len(blob)
            stack = b"".join(_expert_bytes(li, kind, e) for e in range(_E))
            blob += stack
            offsets[li].append(("PATH", off, len(stack), _E, kind))
            w = np.frombuffer(stack, dtype=np.uint8).reshape(
                (_E, 2, _STRIDE[kind] // 2)).copy()
            setattr(mod, f"{kind}_proj", _Proj(w))
        modules[li] = [mod]
    path = tmp_path / "fake.gguf"
    path.write_bytes(bytes(blob))
    offsets = {
        li: [(str(path),) + r[1:] for r in ranges]
        for li, ranges in offsets.items()
    }
    return offsets, modules


def _fake_arena_alloc(shape):
    buf = np.zeros(shape[0], dtype=np.uint8)
    return buf, memoryview(buf)


def _make_feeder(monkeypatch, tmp_path, slots_per_layer=2, n_layers=2,
                 pressure=False):
    import mlx_kquant as kq
    from gmlx.decode_feeder import DecodeFeeder

    monkeypatch.setattr(kq, "arena_alloc", _fake_arena_alloc, raising=False)
    if not pressure:
        # Keep residency tests deterministic: never read the host's real
        # memory-pressure level.
        monkeypatch.setenv("GMLX_DECODE_PRESSURE", "0")
    offsets, modules = _make_fixture(tmp_path, n_layers)
    per_expert = sum(_STRIDE.values())
    return DecodeFeeder(
        offsets, modules, arena_bytes=slots_per_layer * per_expert * n_layers
    ), modules


def _arena_slot(feeder, li, kind, slot):
    view = feeder._views[(li, kind)]
    return bytes(view[slot].reshape(-1))


def test_stage_adopts_misses_and_hits_after(monkeypatch, tmp_path):
    feeder, _ = _make_feeder(monkeypatch, tmp_path)
    ids = np.array([[0, 2]], dtype=np.uint32)
    slots = feeder.stage(0, ids)
    assert slots.shape == ids.shape and slots.dtype == np.uint32
    for e, s in zip((0, 2), slots.reshape(-1)):
        for kind in _KINDS:
            assert _arena_slot(feeder, 0, kind, int(s)) == _expert_bytes(0, kind, e)
    assert feeder._hits == 0 and feeder._lookups == 2

    again = feeder.stage(0, ids)
    assert np.array_equal(again, slots)
    assert feeder._hits == 2  # second pass is all arena hits


def test_eviction_prefers_cold_experts_and_spares_routed(monkeypatch, tmp_path):
    feeder, _ = _make_feeder(monkeypatch, tmp_path)
    feeder.stage(0, np.array([0, 1]))
    feeder.stage(0, np.array([0]))  # expert 0 now more popular than 1
    slots = feeder.stage(0, np.array([0, 2]))  # miss on 2; 0 is routed
    assert feeder._slot_of[0][1] == -1  # cold expert 1 evicted
    assert feeder._slot_of[0][0] >= 0  # popular + routed expert survives
    s2 = int(slots.reshape(-1)[1])
    for kind in _KINDS:
        assert _arena_slot(feeder, 0, kind, s2) == _expert_bytes(0, kind, 2)


def test_stage_overflow_returns_none(monkeypatch, tmp_path):
    feeder, _ = _make_feeder(monkeypatch, tmp_path)  # 2 slots per layer
    assert feeder.stage(0, np.array([0, 1, 2])) is None  # 3 distinct > 2 slots
    feeder.stage(0, np.array([0, 1]))
    # Full arena, both residents routed by the same call that misses.
    assert feeder.stage(0, np.array([0, 1, 3])) is None
    # State stayed consistent: the residents still serve hits.
    assert feeder._slot_of[0][0] >= 0 and feeder._slot_of[0][1] >= 0


def test_layers_are_independent(monkeypatch, tmp_path):
    feeder, _ = _make_feeder(monkeypatch, tmp_path)
    feeder.stage(0, np.array([3]))
    slots = feeder.stage(1, np.array([3]))
    s = int(slots.reshape(-1)[0])
    for kind in _KINDS:
        assert _arena_slot(feeder, 1, kind, s) == _expert_bytes(1, kind, 3)
    assert feeder._slot_of[0][3] >= 0 and feeder._slot_of[1][3] >= 0


def test_swapped_installs_views_and_restores(monkeypatch, tmp_path):
    feeder, modules = _make_feeder(monkeypatch, tmp_path)
    mod = modules[0][0]
    orig = {k: getattr(mod, f"{k}_proj").weight for k in _KINDS}
    with feeder.swapped(0):
        for kind in _KINDS:
            assert getattr(mod, f"{kind}_proj").weight is feeder._views[(0, kind)]
    for kind in _KINDS:
        assert getattr(mod, f"{kind}_proj").weight is orig[kind]


def _pressure_setup(monkeypatch, level, *, cooldown=1000, regrow_polls=2):
    """Poll every stage call against a mutable fake level; ``cooldown``
    high by default so a sustained level takes exactly one step."""
    from gmlx import decode_feeder as dfm

    monkeypatch.setattr(dfm, "_PRESSURE_POLL_EVERY", 1)
    monkeypatch.setattr(dfm, "_PRESSURE_COOLDOWN_POLLS", cooldown)
    monkeypatch.setattr(dfm, "_REGROW_NORMAL_POLLS", regrow_polls)
    monkeypatch.setattr(dfm, "_pressure_level", lambda: level["v"])


def test_pressure_shrink_keeps_hot_experts(monkeypatch, tmp_path):
    """A warning-level poll steps the target down one _PRESSURE_STEP_FRAC;
    each layer reallocates at its own stage call, carrying its most popular
    residents byte-for-byte and dropping the coldest."""
    level = {"v": 1}
    _pressure_setup(monkeypatch, level)
    feeder, _ = _make_feeder(
        monkeypatch, tmp_path, slots_per_layer=4, pressure=True)
    feeder.stage(0, np.array([0, 1, 2, 3]))
    feeder.stage(0, np.array([0, 1]))  # 0 and 1 now hotter than 2 and 3
    feeder.stage(1, np.array([2]))
    assert feeder._slots[0] == 4
    before = feeder.arena_bytes

    level["v"] = 2
    slots = feeder.stage(0, np.array([0, 1]))
    assert feeder._pressure_steps == 1
    assert feeder._slots[0] == 3  # 4 * (1 - 0.25)
    assert feeder._slot_of[0][3] == -1  # coldest dropped
    assert feeder._slot_of[0][2] >= 0
    for e, s in zip((0, 1), slots.reshape(-1)):
        for kind in _KINDS:
            assert _arena_slot(feeder, 0, kind, int(s)) == \
                _expert_bytes(0, kind, e)
    # The survivors were copies, not re-reads: the stage call was all hits.
    assert feeder._hits == 4
    # Layer 1 converges to the same target at its own stage call; a
    # sustained warning takes no second step inside the cooldown.
    feeder.stage(1, np.array([2]))
    assert feeder._pressure_steps == 1 and feeder._slots[1] == 3
    assert feeder.arena_bytes < before
    if feeder.locked_bytes:
        assert feeder.locked_bytes == feeder.arena_bytes


def test_pressure_critical_steps_twice_and_floors(monkeypatch, tmp_path):
    level = {"v": 4}
    _pressure_setup(monkeypatch, level, cooldown=1)
    feeder, _ = _make_feeder(
        monkeypatch, tmp_path, slots_per_layer=4, pressure=True)
    feeder.stage(0, np.array([0]))  # critical: two steps at once
    assert feeder._pressure_steps == 2 and feeder._slots[0] == 2
    feeder.stage(0, np.array([0]))  # cooldown passed: third step
    assert feeder._pressure_steps == 3 and feeder._slots[0] == 1
    feeder.stage(0, np.array([0]))  # capped at _PRESSURE_MAX_STEPS
    assert feeder._pressure_steps == 3 and feeder._slots[0] == 1
    # Shrunk below the call's fan-out, stage degrades to the None fallback
    # without corrupting residency.
    assert feeder.stage(0, np.array([0, 1])) is None
    assert feeder._slot_of[0][0] >= 0


def test_pressure_regrow_after_sustained_normal(monkeypatch, tmp_path):
    level = {"v": 2}
    _pressure_setup(monkeypatch, level, regrow_polls=2)
    avail = {"v": 0}  # no reclaimable RAM: regrow must wait
    seen_kwargs = []

    def _fake_avail(include_inactive=True):
        seen_kwargs.append(include_inactive)
        return avail["v"]

    monkeypatch.setattr(gmlx.loader, "_available_ram_bytes", _fake_avail)
    feeder, _ = _make_feeder(
        monkeypatch, tmp_path, slots_per_layer=4, pressure=True)
    feeder.stage(0, np.array([0, 1]))
    assert feeder._slots[0] == 3

    level["v"] = 1
    feeder.stage(0, np.array([0, 1]))
    feeder.stage(0, np.array([0, 1]))  # sustained normal, but no headroom
    assert feeder._pressure_steps == 1 and feeder._slots[0] == 3
    avail["v"] = 1 << 40
    feeder.stage(0, np.array([0, 1]))  # headroom back: regrow a step
    assert feeder._pressure_steps == 0
    assert feeder._slots[0] == 4
    # Regrow asked for the strict no-victims set (inactive excluded).
    assert seen_kwargs and all(k is False for k in seen_kwargs)
    for e in (0, 1):  # residents survived shrink and regrow copies
        s = int(feeder._slot_of[0][e])
        for kind in _KINDS:
            assert _arena_slot(feeder, 0, kind, s) == _expert_bytes(0, kind, e)
    # Regrown empty slots adopt organically.
    slots = feeder.stage(0, np.array([3]))
    s = int(slots.reshape(-1)[0])
    for kind in _KINDS:
        assert _arena_slot(feeder, 0, kind, s) == _expert_bytes(0, kind, 3)
    if feeder.locked_bytes:
        assert feeder.locked_bytes == feeder.arena_bytes


def test_pressure_disabled_by_env(monkeypatch, tmp_path):
    from gmlx import decode_feeder as dfm

    def _boom():
        raise AssertionError("pressure polled while disabled")

    monkeypatch.setattr(dfm, "_PRESSURE_POLL_EVERY", 1)
    monkeypatch.setattr(dfm, "_pressure_level", _boom)
    feeder, _ = _make_feeder(monkeypatch, tmp_path, slots_per_layer=4)
    feeder.stage(0, np.array([0]))
    assert feeder._slots[0] == 4


def test_arena_budget_math(monkeypatch):
    offsets = {0: [("p", 0, 80 << 30, 256, "gate")]}
    monkeypatch.delenv("GMLX_DECODE_ARENA_GB", raising=False)
    monkeypatch.delenv("GMLX_DECODE_KV_RESERVE_GB", raising=False)
    monkeypatch.delenv("GMLX_DECODE_ARENA_RAM_FRAC", raising=False)
    monkeypatch.delenv("GMLX_DECODE_RAM_FLOOR_GB", raising=False)
    monkeypatch.delenv("GMLX_DECODE_PAGECACHE_GB", raising=False)
    monkeypatch.delenv("GMLX_DECODE_ARENA_FORCE", raising=False)
    monkeypatch.setattr(
        gmlx.loader, "_available_ram_bytes", lambda: None
    )  # available-RAM ceiling out of the way for the deterministic cases
    monkeypatch.setattr(
        mx, "device_info", lambda: {"memory_size": 1000 << 30}
    )  # RAM cap far above the working-set budget: budget binds
    # budget - non-expert bytes (total - experts) - 8 GB reserve, capped at
    # expert bytes
    got = _decode_arena_bytes(100 << 30, offsets, budget=80 << 30)
    assert got == (80 << 30) - (20 << 30) - (8 << 30)
    assert _decode_arena_bytes(85 << 30, offsets, budget=200 << 30) == 80 << 30
    assert _decode_arena_bytes(60 << 30, offsets, budget=None) == 0
    # RAM-fraction ceiling binds when physical RAM is the scarce resource.
    monkeypatch.setattr(
        mx, "device_info", lambda: {"memory_size": 100 << 30}
    )
    got = _decode_arena_bytes(100 << 30, offsets, budget=90 << 30)
    assert got == int(0.6 * (100 << 30)) - (20 << 30) - (8 << 30)
    monkeypatch.setenv("GMLX_DECODE_ARENA_RAM_FRAC", "0.8")
    got = _decode_arena_bytes(100 << 30, offsets, budget=90 << 30)
    assert got == int(0.8 * (100 << 30)) - (20 << 30) - (8 << 30)
    # Available-RAM ceiling binds when the machine is busy: 40 GB
    # reclaimable minus the floor beats the fraction of a 100 GB machine.
    monkeypatch.setattr(
        gmlx.loader, "_available_ram_bytes", lambda: 40 << 30
    )
    monkeypatch.setenv("GMLX_DECODE_RAM_FLOOR_GB", "5")
    # The floor is the base margin plus the page-cache reserve (2.5 GB
    # default): buffered read paths need cache room even when the user
    # pins the base floor.
    floor = int(7.5 * (1 << 30))
    got = _decode_arena_bytes(100 << 30, offsets, budget=90 << 30)
    assert got == (40 << 30) - floor - (20 << 30) - (8 << 30)
    monkeypatch.setenv("GMLX_DECODE_PAGECACHE_GB", "0")
    got = _decode_arena_bytes(100 << 30, offsets, budget=90 << 30)
    assert got == (40 << 30) - (5 << 30) - (20 << 30) - (8 << 30)
    monkeypatch.delenv("GMLX_DECODE_PAGECACHE_GB", raising=False)
    # A modest GMLX_DECODE_ARENA_GB passes through untouched.
    monkeypatch.setenv("GMLX_DECODE_ARENA_GB", "2.5")
    assert _decode_arena_bytes(60 << 30, offsets, budget=None) == int(2.5 * (1 << 30))
    # An oversized override is clamped to reclaimable minus the floor...
    monkeypatch.setenv("GMLX_DECODE_ARENA_GB", "200")
    assert _decode_arena_bytes(60 << 30, offsets, budget=None) == (40 << 30) - floor
    # ...unless forced, and passes through when reclaimable is unknown.
    monkeypatch.setenv("GMLX_DECODE_ARENA_FORCE", "1")
    assert _decode_arena_bytes(60 << 30, offsets, budget=None) == 200 << 30
    monkeypatch.delenv("GMLX_DECODE_ARENA_FORCE", raising=False)
    monkeypatch.setattr(gmlx.loader, "_available_ram_bytes", lambda: None)
    assert _decode_arena_bytes(60 << 30, offsets, budget=None) == 200 << 30


def test_feeder_default_resolution(monkeypatch):
    """Defaults: prefill feeder on; decode feeder only when the every-token
    layers are on GPU. Env vars override defaults; explicit caller intent
    beats both."""
    monkeypatch.delenv("GMLX_FEEDER_PREFILL", raising=False)
    monkeypatch.delenv("GMLX_FEEDER_DECODE", raising=False)
    monkeypatch.setattr(mx, "default_device", lambda: "Device(gpu, 0)")
    assert _resolve_feeder_defaults(None, None) == (True, True)
    monkeypatch.setattr(mx, "default_device", lambda: "Device(cpu, 0)")
    assert _resolve_feeder_defaults(None, None) == (True, False)
    monkeypatch.setenv("GMLX_FEEDER_PREFILL", "0")
    monkeypatch.setenv("GMLX_FEEDER_DECODE", "1")
    assert _resolve_feeder_defaults(None, None) == (False, True)
    # Explicit flags beat env.
    assert _resolve_feeder_defaults(True, False) == (True, False)


def test_config_feeder_keys():
    """Per-model prefill_feeder/decode_feeder parse tri-state and are
    load-affecting (distinct load signatures)."""
    from gmlx.config import _normalize_optional_bool

    assert _normalize_optional_bool(None, "k") is None
    assert _normalize_optional_bool(True, "k") is True
    assert _normalize_optional_bool("off", "k") is False
    assert _normalize_optional_bool("yes", "k") is True
    import warnings

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        assert _normalize_optional_bool("sideways", "k") is None
        assert len(w) == 1


def _holder_model(glu):
    class _Holder:
        pass

    layer = _Holder()
    layer.modules = lambda: [glu]
    model = _Holder()
    model.layers = [layer]
    return model


def test_wrapper_decode_feeder_branch(monkeypatch):
    """A decode-sized call with a decode feeder attached stages the routed
    experts and runs from the (identity-mapped) arena numerically
    transparently; overflow (stage -> None) falls back to the page-cache
    path (on_decode still fires)."""
    mx.random.seed(11)
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
        assert getattr(glu, "_kq_cpu_only", False)
        x1 = mx.random.normal((1, 1, 16))
        i1 = mx.array([[[1, 3]]], dtype=mx.uint32)
        ref = mx.array(glu(x1, i1))
        mx.eval(ref)

        class _FakeDF:
            stage_calls = []
            swap_calls = []
            overflow = False

            def covers(self, li):
                return True

            def stage(self, li, ids):
                self.stage_calls.append((li, ids.copy()))
                if self.overflow:
                    return None
                return ids.astype(np.uint32)  # identity: same weights

            def wedged_at(self, li):
                return False

            @contextmanager
            def swapped(self, li):
                self.swap_calls.append(li)
                yield

        class _FakePF:
            enabled = True
            decode_calls = []

            def on_layer(self, li):
                pass

            def on_decode(self, li, expert_ids):
                self.decode_calls.append((li, list(expert_ids)))

        df, pf = _FakeDF(), _FakePF()
        object.__setattr__(glu, "_kq_decode_feeder", df)
        object.__setattr__(glu, "_kq_prefetcher", pf)
        object.__setattr__(glu, "_kq_li", 5)

        out = mx.array(glu(x1, i1))
        mx.eval(out)
        assert mx.allclose(ref, out, atol=1e-6, rtol=1e-6)
        assert len(df.stage_calls) == 1 and df.stage_calls[0][0] == 5
        assert np.array_equal(df.stage_calls[0][1].reshape(-1), [1, 3])
        assert df.swap_calls == [5]
        assert pf.decode_calls == []  # arena path skips the page-cache pull

        df.overflow = True
        out2 = mx.array(glu(x1, i1))
        mx.eval(out2)
        assert mx.allclose(ref, out2, atol=1e-6, rtol=1e-6)
        assert df.swap_calls == [5]  # no swap on fallback
        assert pf.decode_calls == [(5, [1, 3])]  # page-cache pull took over
    finally:
        mx.set_wired_limit = real_set_wired


def test_stage_read_failure_leaves_no_poisoned_slot(monkeypatch, tmp_path):
    """A failed read must not commit the expert->slot mapping: the victim
    slot holds partial bytes, so its old owner is evicted and the miss stays
    non-resident until a later call re-reads it."""
    import pytest

    feeder, _ = _make_feeder(monkeypatch, tmp_path)
    feeder.stage(0, np.array([0, 1]))
    feeder.stage(0, np.array([0]))  # expert 1 is the eviction victim

    real_read = feeder._read_expert
    def failing_read(li, kind, e, slot):
        if e == 2:
            raise OSError("injected read failure")
        return real_read(li, kind, e, slot)
    monkeypatch.setattr(feeder, "_read_expert", failing_read)
    with pytest.raises(OSError, match="injected"):
        feeder.stage(0, np.array([0, 2]))
    assert feeder._slot_of[0][2] == -1  # miss not adopted
    assert feeder._slot_of[0][1] == -1  # victim's old owner evicted
    assert feeder._slot_of[0][0] >= 0  # untouched resident still serves

    monkeypatch.setattr(feeder, "_read_expert", real_read)
    slots = feeder.stage(0, np.array([0, 2]))  # re-read on the next call
    s2 = int(slots.reshape(-1)[1])
    for kind in _KINDS:
        assert _arena_slot(feeder, 0, kind, s2) == _expert_bytes(0, kind, 2)


def _wedge_setup(monkeypatch, tmp_path, wedge_offs, timeout="0.2", **kw):
    """Feeder whose reads at the given file offsets block forever (a
    simulated kernel wedge). Alignment is patched to 16 bytes - every
    fixture stride is a multiple - so each expert read keeps its own
    distinct offset."""
    import threading

    from gmlx import decode_feeder as dfm

    monkeypatch.setattr(dfm, "_PAGE", 16)
    monkeypatch.setenv("GMLX_DECODE_READ_TIMEOUT", timeout)
    release = threading.Event()
    real = dfm.read_range

    def hang_range(fd, mv, off):
        if off in wedge_offs:
            release.wait()
            return
        real(fd, mv, off)

    monkeypatch.setattr(dfm, "read_range", hang_range)
    feeder, modules = _make_feeder(monkeypatch, tmp_path, **kw)
    return feeder, release


def test_wedged_read_quarantines_and_zero_maps(monkeypatch, tmp_path):
    """A read that outlives the timeout quarantines its slot, drops the
    expert to the layer's zero slot (gathers contribute nothing for it)
    and replaces the stranded worker; decode continues."""
    from gmlx import decode_feeder as dfm

    # Layer 0 gate stack starts at offset 0; expert 1's gate read is at 64.
    feeder, release = _wedge_setup(
        monkeypatch, tmp_path, {64}, slots_per_layer=3)
    try:
        slots = feeder.stage(0, np.array([[0, 1]], dtype=np.uint32))
        assert slots is not None
        s0, s1 = (int(v) for v in slots.reshape(-1))
        for kind in _KINDS:
            assert _arena_slot(feeder, 0, kind, s0) == _expert_bytes(0, kind, 0)
            assert _arena_slot(feeder, 0, kind, s1) == bytes(_STRIDE[kind])
        assert (feeder._owner[0] == -2).sum() == 1  # quarantined slot
        assert feeder._owner[0][s1] == -3  # zero slot
        assert feeder.wedged_at(0) and feeder.has_dead(0)
        assert feeder._wedges == 1
        assert len(feeder._read_pool._threads) == dfm._READ_WORKERS + 1
        # The dead expert is a permanent zero-slot resident: no re-read.
        again = feeder.stage(0, np.array([[1, 2]], dtype=np.uint32))
        assert int(again.reshape(-1)[0]) == s1
        # Fallback ids never touch the dead expert's range.
        fixed = feeder.redirect_dead(0, np.array([[1, 0], [2, 1]]))
        assert fixed.tolist() == [[0, 0], [2, 2]]
        assert not feeder.has_dead(1) and not feeder.wedged_at(1)
        # redirect_dead is the identity elsewhere.
        ids = np.array([[0, 2]])
        assert feeder.redirect_dead(1, ids) is ids
    finally:
        release.set()


def test_staging_disabled_after_max_wedges(monkeypatch, tmp_path):
    from gmlx import decode_feeder as dfm

    monkeypatch.setattr(dfm, "_MAX_WEDGES", 1)
    feeder, release = _wedge_setup(
        monkeypatch, tmp_path, {64}, slots_per_layer=3)
    try:
        feeder.stage(0, np.array([[0, 1]], dtype=np.uint32))
        assert feeder._staging_disabled
        # Misses no longer stage; resident calls (zero slot included)
        # still serve from the arena.
        assert feeder.stage(1, np.array([[0]], dtype=np.uint32)) is None
        assert feeder.stage(0, np.array([[0, 1]], dtype=np.uint32)) is not None
    finally:
        release.set()


def test_wedge_without_spare_slot_drops_layer(monkeypatch, tmp_path):
    feeder, release = _wedge_setup(
        monkeypatch, tmp_path, {64}, slots_per_layer=1)
    try:
        assert feeder.stage(0, np.array([[1]], dtype=np.uint32)) is None
        assert not feeder.covers(0)  # layer out of service
        assert feeder.covers(1)
        assert feeder.has_dead(0)
        # All-dead row falls back to the first surviving expert globally.
        assert feeder.redirect_dead(0, np.array([[1]])).tolist() == [[0]]
    finally:
        release.set()


def test_wedged_layer_never_resizes(monkeypatch, tmp_path):
    """Pressure shrink must skip wedged layers: reallocating would free
    the buffer the zombie read may still write into."""
    level = {"v": 1}
    _pressure_setup(monkeypatch, level)
    from gmlx import decode_feeder as dfm

    monkeypatch.setattr(dfm, "_PAGE", 16)
    monkeypatch.setenv("GMLX_DECODE_READ_TIMEOUT", "0.2")
    import threading

    release = threading.Event()
    real = dfm.read_range

    def hang_range(fd, mv, off):
        if off == 64:
            release.wait()
            return
        real(fd, mv, off)

    monkeypatch.setattr(dfm, "read_range", hang_range)
    feeder, _ = _make_feeder(
        monkeypatch, tmp_path, slots_per_layer=4, pressure=True)
    try:
        feeder.stage(0, np.array([[0, 1]], dtype=np.uint32))  # wedge on 1
        assert feeder.wedged_at(0)
        level["v"] = 2
        feeder.stage(0, np.array([[0]], dtype=np.uint32))
        assert feeder._pressure_steps == 1
        assert feeder._slots[0] == 4  # wedged layer pinned
        feeder.stage(1, np.array([[0]], dtype=np.uint32))
        assert feeder._slots[1] == 3  # healthy layer shrinks
    finally:
        release.set()


def test_aligned_reads_and_raw_fallback(monkeypatch, tmp_path):
    """Byte movement is exact when the alignment forces a nonzero head
    offset inside the bounce (stride 48 vs 32-byte pages), and with the
    kill switch the raw pread path serves identically."""
    from gmlx import decode_feeder as dfm

    monkeypatch.setattr(dfm, "_PAGE", 32)
    feeder, _ = _make_feeder(monkeypatch, tmp_path, slots_per_layer=4)
    slots = feeder.stage(0, np.array([[1, 3]], dtype=np.uint32))
    for e, s in zip((1, 3), slots.reshape(-1)):
        for kind in _KINDS:
            assert _arena_slot(feeder, 0, kind, int(s)) == \
                _expert_bytes(0, kind, e)

    monkeypatch.setenv("GMLX_DECODE_ALIGNED_READS", "0")
    raw, _ = _make_feeder(monkeypatch, tmp_path, slots_per_layer=4)
    assert not raw._aligned
    slots = raw.stage(1, np.array([[2]], dtype=np.uint32))
    s = int(slots.reshape(-1)[0])
    for kind in _KINDS:
        assert _arena_slot(raw, 1, kind, s) == _expert_bytes(1, kind, 2)


def test_wrapper_redirects_dead_ids_on_fallback(monkeypatch):
    """Post-wedge, every non-arena path must see rewritten ids: the
    output equals computing with the dead id replaced, and the decode
    advisory only sees surviving experts."""
    mx.random.seed(12)
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
        x1 = mx.random.normal((1, 1, 16))
        ref = mx.array(glu(x1, mx.array([[[1, 1]]], dtype=mx.uint32)))
        mx.eval(ref)

        class _WedgedDF:
            def covers(self, li):
                return True

            def stage(self, li, ids):
                return None  # staging disabled after the wedge

            def wedged_at(self, li):
                return True

            def has_dead(self, li):
                return True

            def redirect_dead(self, li, ids):
                out = ids.copy()
                out[ids == 3] = 1  # expert 3 died; row survivor is 1
                return out

        class _FakePF:
            enabled = True
            decode_calls = []

            def on_decode(self, li, expert_ids):
                self.decode_calls.append((li, list(expert_ids)))

        object.__setattr__(glu, "_kq_decode_feeder", _WedgedDF())
        object.__setattr__(glu, "_kq_prefetcher", _FakePF())
        object.__setattr__(glu, "_kq_li", 7)

        out = mx.array(glu(x1, mx.array([[[1, 3]]], dtype=mx.uint32)))
        mx.eval(out)
        assert mx.allclose(ref, out, atol=1e-6, rtol=1e-6)
        assert _FakePF.decode_calls == [(7, [1])]  # dead id never advised
    finally:
        mx.set_wired_limit = real_set_wired


def test_per_layer_hit_stats(monkeypatch, tmp_path, capsys):
    """Per-layer hit counters accumulate independently and the exit line
    reports the cross-layer spread with the coldest layers named."""
    feeder, _ = _make_feeder(monkeypatch, tmp_path)
    feeder.stage(0, np.array([0, 1]))  # 2 misses
    feeder.stage(0, np.array([0, 1]))  # 2 hits
    feeder.stage(1, np.array([2]))     # 1 miss
    assert feeder._layer_lookups[0] == 4 and feeder._layer_hits[0] == 2
    assert feeder._layer_lookups[1] == 1 and feeder._layer_hits[1] == 0
    feeder.close()
    out = capsys.readouterr().out
    assert "per-layer hit rate" in out
    assert "median 50.0% / max 50.0%" in out
    assert "L1 0.0%" in out  # coldest layer named first (= the min)


# ---- lookahead prestage ----

def _wait_published(feeder, li, timeout=5.0):
    """Poll until layer li has no pending prestage reads."""
    import time

    deadline = time.time() + timeout
    while feeder._pending.get(li):
        feeder._flush_pending(li)
        if time.time() > deadline:
            raise AssertionError("prestage never completed")
        time.sleep(0.005)


def _counting_reads(monkeypatch):
    from gmlx import decode_feeder as dfm

    calls = []
    real = dfm.read_range

    def counting(fd, mv, off):
        calls.append(off)
        real(fd, mv, off)

    monkeypatch.setattr(dfm, "read_range", counting)
    return calls


def test_prestage_publishes_and_stage_hits(monkeypatch, tmp_path):
    """Prestage fills empty slots asynchronously with the exact file
    bytes, never touches popularity, and the next stage() serves the
    predicted experts as hits without re-reading."""
    calls = _counting_reads(monkeypatch)
    feeder, _ = _make_feeder(monkeypatch, tmp_path)
    feeder.prestage(0, np.array([[1, 3]]))
    assert feeder._la_submitted == 2
    assert (feeder._owner[0] == -4).sum() == 2  # reserved while in flight
    _wait_published(feeder, 0)
    for e in (1, 3):
        s = int(feeder._slot_of[0][e])
        assert s >= 0 and feeder._owner[0][s] == e
        for kind in _KINDS:
            assert _arena_slot(feeder, 0, kind, s) == _expert_bytes(0, kind, e)
    assert not feeder._counts[0].any()  # predictions never bump counts
    n_reads = len(calls)
    assert feeder.stage(0, np.array([[1, 3]])) is not None
    assert feeder._hits == 2 and len(calls) == n_reads  # no re-read


def test_prestage_settle_joins_inflight(monkeypatch, tmp_path):
    """A pending prestage the router actually routes to is joined and
    published by stage()'s settle barrier - one read total - and its
    reserved slot is invisible to scans while the read is in flight."""
    import threading
    import time

    from gmlx import decode_feeder as dfm

    monkeypatch.setattr(dfm, "_PAGE", 16)
    monkeypatch.setenv("GMLX_DECODE_READ_TIMEOUT", "5")
    release = threading.Event()
    calls = []
    real = dfm.read_range
    blocked = {64, 320, 560}  # expert 1: gate/up/down slices

    def gated(fd, mv, off):
        calls.append(off)
        if off in blocked:
            release.wait(5)
        real(fd, mv, off)

    monkeypatch.setattr(dfm, "read_range", gated)
    feeder, _ = _make_feeder(monkeypatch, tmp_path)
    feeder.prestage(0, np.array([[1]]))
    assert 1 in feeder._pending[0]
    pending_slot = feeder._pending[0][1][0]
    assert feeder._owner[0][pending_slot] == -4  # reserved while in flight
    deadline = time.time() + 5  # all 3 prestage reads inside the gate
    while sum(o in blocked for o in calls) < 3 and time.time() < deadline:
        time.sleep(0.005)
    n_before = len(calls)
    threading.Timer(0.05, release.set).start()
    slots = feeder.stage(0, np.array([[1]]))
    s = int(slots.reshape(-1)[0])
    assert s == pending_slot and feeder._owner[0][s] == 1
    assert feeder._la_waited == 1
    # Adoption did not issue new reads for expert 1.
    assert [o for o in calls[n_before:] if o in blocked] == []
    for kind in _KINDS:
        assert _arena_slot(feeder, 0, kind, s) == _expert_bytes(0, kind, 1)


def test_settle_waits_unrouted_pending(monkeypatch, tmp_path):
    """The settle barrier joins pending prestages even when the call does
    not route to them: once stage() returns, nothing is still writing
    into the layer's arena. (Cancellation off: this test pins the pure
    wait-and-publish path.)"""
    import threading

    from gmlx import decode_feeder as dfm

    monkeypatch.setattr(dfm, "_PAGE", 16)
    monkeypatch.setenv("GMLX_DECODE_READ_TIMEOUT", "5")
    monkeypatch.setenv("GMLX_DECODE_LOOKAHEAD_CANCEL", "0")
    release = threading.Event()
    real = dfm.read_range
    blocked = {64, 320, 560}  # expert 1: gate/up/down slices

    def gated(fd, mv, off):
        if off in blocked:
            release.wait(5)
        real(fd, mv, off)

    monkeypatch.setattr(dfm, "read_range", gated)
    feeder, _ = _make_feeder(monkeypatch, tmp_path)
    feeder.prestage(0, np.array([[1]]))
    threading.Timer(0.05, release.set).start()
    feeder.stage(0, np.array([[0]]))
    assert not feeder._pending[0]
    s1 = int(feeder._slot_of[0][1])
    assert s1 >= 0 and feeder._owner[0][s1] == 1  # published, not adopted
    assert feeder._la_adopted == 0 and feeder._la_waited == 0
    for kind in _KINDS:
        assert _arena_slot(feeder, 0, kind, s1) == _expert_bytes(0, kind, 1)


def test_prestage_never_evicts_hotter(monkeypatch, tmp_path):
    """A full arena only yields slots whose residents are no more popular
    than the prediction."""
    feeder, _ = _make_feeder(monkeypatch, tmp_path)
    feeder.stage(0, np.array([[0, 1]]))
    feeder.stage(0, np.array([[0, 1]]))  # counts: e0=2, e1=2
    feeder.prestage(0, np.array([[2, 3]]))  # counts[2]=counts[3]=0
    assert feeder._la_submitted == 0 and not feeder._pending.get(0)
    assert feeder._slot_of[0][0] >= 0 and feeder._slot_of[0][1] >= 0
    # A prediction at least as popular as the coldest resident evicts it.
    feeder.stage(0, np.array([[0]]))  # counts: e0=3, e1=2
    feeder._counts[0][2] = 2.0
    feeder.prestage(0, np.array([[2]]))
    assert feeder._la_submitted == 1
    assert feeder._slot_of[0][1] == -1  # colder resident unmapped
    _wait_published(feeder, 0)
    assert feeder._slot_of[0][2] >= 0
    assert feeder._slot_of[0][0] >= 0  # hotter resident untouched


def test_prestage_wedge_quarantines(monkeypatch, tmp_path):
    """A prestage read that outlives the timeout is contained exactly like
    a demand wedge: slot quarantined, expert dead, lookahead pool worker
    replaced; the layer then refuses further prestaging."""
    import time

    feeder, release = _wedge_setup(
        monkeypatch, tmp_path, {64, 320, 560}, slots_per_layer=3)
    try:
        feeder.prestage(0, np.array([[1]]))
        assert feeder._la_submitted == 1
        n_threads = len(feeder._la_pool._threads)
        time.sleep(0.25)  # outlive the 0.2s timeout
        feeder._flush_pending(0)
        assert feeder.wedged_at(0) and feeder.has_dead(0)
        assert (feeder._owner[0] == -2).sum() == 1
        assert feeder._slot_of[0][1] == feeder._zslot[0]
        assert len(feeder._la_pool._threads) == n_threads + 3
        feeder.prestage(0, np.array([[3]]))  # wedged layer: no-op
        assert feeder._la_submitted == 1
    finally:
        release.set()


def test_pending_prestage_settles_before_resize(monkeypatch, tmp_path):
    """stage() settles every pending prestage - routed or not - before the
    resize check, so no speculative read can write into a reallocated
    buffer and the resize applies in the same call."""
    from concurrent.futures import Future

    feeder, _ = _make_feeder(monkeypatch, tmp_path)
    feeder._pressure_steps = 1  # target shrinks 2 -> 1
    for kind in _KINDS:  # real bytes, so settle's publish verifies clean
        feeder._read_expert(0, kind, 3, 0)
    fut = Future()
    fut.set_result(None)
    feeder._owner[0][0] = -4  # reserved, as a real prestage would mark it
    feeder._pending[0] = {3: (0, [fut], __import__("time").monotonic())}
    slots = feeder.stage(0, np.array([[0]]))
    assert slots is not None
    assert not feeder._pending[0]  # settled at entry
    assert feeder._slots[0] == 1  # resize applied in the same call


def test_prestage_rank_major_cap(monkeypatch, tmp_path):
    """Multi-row predictions submit rank-major (each row's head first),
    considering only the top GMLX_DECODE_LOOKAHEAD_K ranks per row."""
    monkeypatch.setenv("GMLX_DECODE_LOOKAHEAD_K", "1")
    feeder, _ = _make_feeder(monkeypatch, tmp_path)
    feeder.prestage(0, np.array([[1, 2], [3, 2]]))
    assert sorted(feeder._pending[0]) == [1, 3]  # rank-0 column only
    _wait_published(feeder, 0)
    assert feeder._slot_of[0][2] == -1


def test_prestage_shields_predicted_residents(monkeypatch, tmp_path):
    """A lower-ranked prediction never evicts a resident this call's own
    higher ranks predicted - the best slots in the arena are the ones the
    prediction says are about to be routed."""
    feeder, _ = _make_feeder(monkeypatch, tmp_path)
    feeder.stage(0, np.array([[0, 1]]))  # both resident, counts 1 each
    feeder._counts[0][2] = 5.0  # prediction hotter than either resident
    feeder.prestage(0, np.array([[0, 2]]))  # rank 0 resident, rank 1 missing
    _wait_published(feeder, 0)
    assert feeder._slot_of[0][0] >= 0  # predicted resident survived
    assert feeder._slot_of[0][1] == -1  # the unpredicted one paid the slot
    assert feeder._slot_of[0][2] >= 0


def test_settle_cancels_unstarted_unrouted(monkeypatch, tmp_path):
    """A pending prediction the call does not route to is cancelled at
    settle if its reads have not started: the slot returns to empty, no
    disk bandwidth is spent, nothing is published."""
    import threading

    monkeypatch.setenv("GMLX_DECODE_LOOKAHEAD_WORKERS", "1")
    feeder, _ = _make_feeder(monkeypatch, tmp_path)
    block = threading.Event()
    feeder._la_state().submit(block.wait)  # the lone worker parks here
    try:
        feeder.prestage(0, np.array([[1]]))  # 3 reads queued, unstarted
        assert 1 in feeder._pending[0]
        slots = feeder.stage(0, np.array([[0]]))
        assert slots is not None
        assert feeder._la_cancelled == 1
        assert feeder._slot_of[0][1] == -1  # never published
        assert not feeder._pending[0]
    finally:
        block.set()


def test_read_pool_on_start_hook():
    """Worker threads run the pool's on_start hook (the lookahead pool
    drops its disk-I/O priority there) before serving reads."""
    import threading

    from gmlx.decode_feeder import _DaemonReadPool

    ran = threading.Event()
    pool = _DaemonReadPool(1, on_start=ran.set)
    assert ran.wait(2)
    pool.shutdown()


def test_exit_close_hook_holds_only_weakref(monkeypatch, tmp_path):
    """The atexit stats hook must not pin an unloaded feeder: a strong
    reference would keep every evicted model's arena alive for the life
    of a long-running server."""
    import atexit
    import gc
    import weakref

    from gmlx.decode_feeder import _register_exit_close

    hooks = []
    monkeypatch.setattr(atexit, "register", hooks.append)
    feeder, _ = _make_feeder(monkeypatch, tmp_path)
    _register_exit_close(feeder)
    feeder.close()
    wref = weakref.ref(feeder)
    del feeder
    gc.collect()
    assert wref() is None
    hooks[0]()  # hook on a collected feeder is a quiet no-op


def test_close_stats_gated_on_session_verbosity(capsys):
    """Aggregate arena hit rate always prints; the per-layer breakdown and
    stall accounting follow the load session's verbosity captured at
    construction."""
    import time as _time

    from gmlx.decode_feeder import DecodeFeeder

    def bare(verbose):
        f = object.__new__(DecodeFeeder)
        f._closed = False
        f._stats_verbose = verbose
        f._lookups, f._hits, f._calls = 10, 9, 5
        f._layer_hits = {0: 5, 1: 4}
        f._layer_lookups = {0: 5, 1: 5}
        f._t_start = _time.monotonic()
        f._t_demand = f._t_settle = 0.5
        f._locked = {}
        f._fds = {}
        f.locked_bytes = 0
        return f

    bare(False).close()
    out = capsys.readouterr().out
    assert "arena hit rate" in out
    assert "per-layer hit rate" not in out
    assert "stalls" not in out
    bare(True).close()
    out = capsys.readouterr().out
    assert "per-layer hit rate" in out
    assert "stalls" in out
