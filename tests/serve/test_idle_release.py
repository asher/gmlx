"""Buffers GMLX_SPEC_KEEP_BUFFERS holds are given back while the server is
idle under outside memory pressure, and before a model load is measured."""

import queue

import pytest

import gmlx.serve.capacity as capacity
import gmlx.serve.idle_release as ir
from gmlx._vendor.mlx_vlm.server import generation as _GEN

GB = 1 << 30


@pytest.fixture
def rig(monkeypatch):
    state = {"cache": 4 * GB, "level": 1, "unwired": 60e9, "clears": 0}

    def clear():
        state["clears"] += 1
        state["cache"] = 0

    monkeypatch.setenv("GMLX_SPEC_KEEP_BUFFERS", "1")
    monkeypatch.setattr(ir.mx, "get_cache_memory", lambda: state["cache"])
    monkeypatch.setattr(ir.mx, "clear_cache", clear)
    monkeypatch.setattr(ir, "_pressure_level", lambda: state["level"])
    monkeypatch.setattr(ir, "_unwired", lambda: state["unwired"])
    monkeypatch.setattr(ir, "_floor", lambda: 4e9)
    monkeypatch.setattr(ir, "_last_check", 0.0)
    return state


def test_an_idle_server_keeps_its_buffers_while_memory_is_free(rig):
    assert ir.idle_check(now=10.0) == 0
    assert rig["clears"] == 0


@pytest.mark.parametrize("change", [{"level": 2}, {"level": 4},
                                    {"unwired": 7e9}])
def test_outside_pressure_releases_the_kept_buffers(rig, change):
    rig.update(change)
    assert ir.idle_check(now=10.0) == 4 * GB
    assert rig["clears"] == 1
    # Nothing left to give back.
    assert ir.idle_check(now=20.0) == 0
    assert rig["clears"] == 1


def test_checks_are_a_second_apart(rig):
    assert ir.idle_check(now=10.0) == 0
    rig["level"] = 4
    assert ir.idle_check(now=10.5) == 0
    assert ir.idle_check(now=11.1) == 4 * GB


def test_the_lever_off_never_clears(rig, monkeypatch):
    monkeypatch.setenv("GMLX_SPEC_KEEP_BUFFERS", "0")
    rig["level"] = 4
    assert ir.idle_check(now=10.0) == 0
    assert ir.before_load_gate() == 0
    assert rig["clears"] == 0


def test_the_idle_engine_loop_runs_the_check(rig, monkeypatch):
    monkeypatch.setattr(ir, "_CHECK_S", 0.0)
    ir.install_idle_release()
    rg = object.__new__(_GEN.ResponseGenerator)
    rg._stop = False
    rg.requests = queue.Queue()
    rg.model = None
    collect = _GEN.ResponseGenerator._collect_pending_requests
    assert collect(rg, active=False, idle_timeout=0.01) == ([], False)
    assert rig["clears"] == 0
    rig["level"] = 2
    # A loop with a row decoding leaves the cache to the governor.
    collect(rg, active=True, idle_timeout=0.01)
    assert rig["clears"] == 0
    collect(rg, active=False, idle_timeout=0.01)
    assert rig["clears"] == 1


def test_a_model_load_is_measured_without_the_kept_buffers(rig, monkeypatch):
    """Held buffers read as used under the wired cap; a load that fits
    without them is not deferred."""
    import gmlx.gen.prefill_decay as pd

    weights = 10 * GB
    monkeypatch.setattr(capacity, "overcommit", lambda: False)
    monkeypatch.setattr(capacity, "cap_applies", lambda: True)
    monkeypatch.setattr(capacity, "working_budget_bytes", lambda: 100 * GB)
    monkeypatch.setattr(capacity, "working_set_bytes", lambda: 100 * GB)
    monkeypatch.setattr(capacity, "_kernel_gate", lambda *a, **kw: None)
    monkeypatch.setattr(pd, "headroom_bytes",
                        lambda: 12 * GB - rig["cache"])
    capacity.preload_gate(weights, "m")
    assert rig["clears"] == 1
    rig["cache"] = 4 * GB
    monkeypatch.setenv("GMLX_SPEC_KEEP_BUFFERS", "0")
    with pytest.raises(capacity.LoadDeferred):
        capacity.preload_gate(weights, "m")
