"""An idle server renews its GPU residency for the keep-alive window, stops
under memory pressure and when other apps are being squeezed, and a model
load waits out the lapse first."""

import queue

import pytest

import gmlx.serve.idle_release as ir
import gmlx.serve.residency_keepalive as rk
from gmlx._vendor.mlx_vlm.server import generation as _GEN


@pytest.fixture
def rig(monkeypatch):
    state = {"beats": [], "level": 1, "unwired": 60e9,
             "compressor": 10e9, "reclaimable": 40e9}
    monkeypatch.setattr(rk, "_beat", lambda: state["beats"].append(1))
    monkeypatch.setattr(rk, "_compressor", lambda: state["compressor"])
    monkeypatch.setattr(rk, "_reclaimable", lambda: state["reclaimable"])
    monkeypatch.setattr(rk, "_ram", lambda: 128e9)
    monkeypatch.setattr(rk, "_baseline", None)
    monkeypatch.setattr(ir, "_pressure_level", lambda: state["level"])
    monkeypatch.setattr(ir, "_unwired", lambda: state["unwired"])
    monkeypatch.setattr(ir, "_floor", lambda: 4e9)
    monkeypatch.setattr(rk, "_last_active", None)
    monkeypatch.setattr(rk, "_last_beat", None)
    monkeypatch.setattr(rk, "_yielded", False)
    monkeypatch.delenv("GMLX_RESIDENCY_KEEPALIVE_S", raising=False)
    return state


def _ticks(times):
    return [t for t in times if rk.idle_tick(now=t)]


def test_an_idle_server_beats_every_second_within_the_window(rig):
    # Nothing ran yet: nothing to keep.
    assert _ticks([1.0, 5.0]) == []
    rk.note_active(now=10.0)
    assert _ticks([10.5, 11.0, 11.5, 12.0, 12.4, 13.1]) == [11.0, 12.0, 13.1]
    assert len(rig["beats"]) == 3
    # Past the window the residency lapses as it did without the keep-alive.
    assert _ticks([609.5, 611.0]) == [609.5]


def test_the_window_comes_from_the_setting(rig, monkeypatch):
    monkeypatch.setenv("GMLX_RESIDENCY_KEEPALIVE_S", "5")
    rk.note_active(now=10.0)
    assert _ticks([11.0, 12.0, 15.0, 16.0]) == [11.0, 12.0, 15.0]
    monkeypatch.setenv("GMLX_RESIDENCY_KEEPALIVE_S", "0")
    rk.note_active(now=20.0)
    assert _ticks([21.0, 22.0]) == []


@pytest.mark.parametrize("change", [{"level": 2}, {"unwired": 7e9}])
def test_memory_pressure_stops_the_beats(rig, change):
    rk.note_active(now=10.0)
    rig.update(change)
    assert _ticks([11.0, 12.0]) == []
    rig.update(level=1, unwired=60e9)
    assert _ticks([13.0]) == [13.0]


def test_a_growing_compressor_stops_the_beats_until_the_next_step(rig):
    # 128 GB of RAM: growth past 1.28 GB since the first beat means other
    # apps are being compressed to make room beside the wired memory.
    rk.note_active(now=10.0)
    assert _ticks([11.0]) == [11.0]
    rig["compressor"] = 11.2e9
    assert _ticks([12.0]) == [12.0]
    rig["compressor"] = 11.4e9
    assert _ticks([13.0, 14.0]) == []
    # Back down, the window stays stopped: the memory has unwired.
    rig["compressor"] = 10e9
    assert _ticks([15.0]) == []
    # The next step starts a window with its own first reading.
    rig["compressor"] = 20e9
    rk.note_active(now=30.0)
    assert _ticks([31.0, 32.0]) == [31.0, 32.0]


def test_free_memory_under_the_floor_stops_the_beats(rig):
    rk.note_active(now=10.0)
    assert _ticks([11.0]) == [11.0]
    rig["reclaimable"] = 3e9
    assert _ticks([12.0]) == []
    rig["reclaimable"] = 40e9
    assert _ticks([13.0]) == []
    rk.note_active(now=20.0)
    assert _ticks([21.0]) == [21.0]


def test_a_load_waits_out_the_lapse_and_stops_the_beats(rig):
    slept = []
    rk.note_active(now=10.0)
    assert _ticks([11.0, 12.0]) == [11.0, 12.0]
    assert rk.yield_for_load(now=12.5, sleep=slept.append) == pytest.approx(3.0)
    assert slept == [pytest.approx(3.0)]
    assert _ticks([13.0, 14.0, 20.0]) == []
    # The next step starts the keep-alive again.
    rk.note_active(now=30.0)
    assert _ticks([31.0]) == [31.0]


@pytest.mark.parametrize("last_beat", [None, 9.0])
def test_a_load_waits_for_nothing_the_keepalive_did_not_hold(rig, last_beat):
    # No beat since the last step: the memory is held as without it.
    slept = []
    rk._last_beat = last_beat
    rk.note_active(now=10.0)
    assert rk.yield_for_load(now=10.5, sleep=slept.append) == 0.0
    assert slept == []


def test_the_idle_engine_loop_runs_the_keepalive(rig, monkeypatch):
    clock = {"t": 100.0}
    monkeypatch.setattr(rk.time, "monotonic", lambda: clock["t"])
    ir.install_idle_release()
    rg = object.__new__(_GEN.ResponseGenerator)
    rg._stop = False
    rg.requests = queue.Queue()
    rg.model = None
    collect = _GEN.ResponseGenerator._collect_pending_requests
    collect(rg, active=True, idle_timeout=0.01)
    assert rk._last_active == 100.0 and rig["beats"] == []
    clock["t"] = 101.5
    collect(rg, active=False, idle_timeout=0.01)
    assert rig["beats"] == [1]
