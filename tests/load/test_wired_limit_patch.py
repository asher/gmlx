#!/usr/bin/env python3
"""The warn-once wired_limit replacement: mlx-lm's stock context manager
prints its near-budget large-model warning on every entry (once per chat
turn on a resident model just over the 0.9x threshold). The loader's
replacement must warn exactly once per process, keep the raise/sync/restore
wiring, and never overwrite the stricter streaming / CPU variants."""

from __future__ import annotations

import contextlib
import importlib

import mlx.core as mx
import pytest

from gmlx.stream.wired_limit import _install_wired_limit_warn_once

# `import mlx_lm.generate` binds the function mlx_lm re-exports in
# __init__, not the submodule - same trap the loader patches around.
_gen = importlib.import_module("gmlx._vendor.mlx_lm.generate")


@pytest.fixture
def restore_wired_limit():
    """Reset wired_limit to an unmarked stand-in so the installer engages
    even when an earlier test left the (marked) streaming / CPU replacement
    installed process-wide, and restore the prior state after."""
    mods = []
    for name in ("gmlx._vendor.mlx_lm.generate", "gmlx._vendor.mlx_lm.utils"):
        try:
            mod = importlib.import_module(name)
        except ImportError:
            continue
        if hasattr(mod, "wired_limit"):
            mods.append((mod, mod.wired_limit))

    @contextlib.contextmanager
    def _stock_stand_in(model, streams=None):
        yield

    for mod, _ in mods:
        mod.wired_limit = _stock_stand_in
    try:
        yield
    finally:
        for mod, orig in mods:
            mod.wired_limit = orig


@pytest.fixture
def fake_mx(monkeypatch):
    """A tiny fake GPU device: 100-byte recommended working set, recorded
    set_wired_limit calls, no-op synchronize."""
    calls = []

    monkeypatch.setattr(mx.metal, "is_available", lambda: True)
    monkeypatch.setattr(mx, "default_device", lambda: mx.Device(mx.gpu))
    monkeypatch.setattr(
        mx, "device_info",
        lambda: {"max_recommended_working_set_size": 100})
    monkeypatch.setattr(
        mx, "set_wired_limit", lambda v: (calls.append(v), 7)[1])
    monkeypatch.setattr(mx, "synchronize", lambda *a, **k: None)
    return calls


def _over_budget_model():
    # 32 f32 elements = 128 bytes > 0.9 * 100.
    return {"w": mx.zeros((32,), dtype=mx.float32)}


def test_warns_once_across_entries(restore_wired_limit, fake_mx, capsys):
    _install_wired_limit_warn_once()
    model = _over_budget_model()
    for _ in range(3):
        with _gen.wired_limit(model):
            pass
    out = capsys.readouterr().out
    assert out.count("close to the maximum recommended size") == 1

    # The wiring behavior is preserved: raise to the recommended size on
    # every entry, restore the previous limit on every exit.
    assert fake_mx == [100, 7, 100, 7, 100, 7]


def test_under_budget_never_warns(restore_wired_limit, fake_mx, capsys):
    _install_wired_limit_warn_once()
    model = {"w": mx.zeros((4,), dtype=mx.float32)}   # 16 bytes
    with _gen.wired_limit(model):
        pass
    assert "close to the maximum" not in capsys.readouterr().out
    assert fake_mx == [100, 7]


def test_the_cpu_device_sets_no_limit(restore_wired_limit, fake_mx,
                                      monkeypatch):
    # The CPU device's info has no working set key.
    monkeypatch.setattr(mx, "default_device", lambda: mx.Device(mx.cpu))
    monkeypatch.setattr(mx, "device_info", lambda: {"device_name": "cpu"})
    _install_wired_limit_warn_once()
    with _gen.wired_limit(_over_budget_model()):
        pass
    assert fake_mx == []


def test_install_is_idempotent(restore_wired_limit):
    _install_wired_limit_warn_once()
    first = _gen.wired_limit
    _install_wired_limit_warn_once()
    assert _gen.wired_limit is first


def test_never_overwrites_marked_variants(restore_wired_limit):
    # The streaming (_kq_no_sweep) and CPU variants are stricter; a later
    # load_model must leave them in place.
    @contextlib.contextmanager
    def _no_sweep(model, streams=None):
        yield

    _no_sweep._kq_no_sweep = True
    _gen.wired_limit = _no_sweep
    _install_wired_limit_warn_once()
    assert _gen.wired_limit is _no_sweep


def test_neutralize_lowers_a_raised_wired_limit(monkeypatch, capsys):
    # A generator that started before the streaming install left the limit
    # raised. Its exit restore is a no-op from here on, so the neutralization
    # lowers the real limit itself, or the next walk wires the file's pages.
    import mlx.core as mx

    from gmlx.stream.wired_limit import _neutralize_wired_limit_sweep

    calls = []

    def fake_set(limit):
        calls.append(limit)
        return 100 * 10**9

    monkeypatch.setattr(mx, "set_wired_limit", fake_set)
    _neutralize_wired_limit_sweep()
    assert calls == [0]
    assert getattr(mx.set_wired_limit, "_kq_no_sweep", False)
    assert mx.set_wired_limit(10**12) == 0
    assert "wired limit lowered from 100.0 GB to 0" in capsys.readouterr().out
    _neutralize_wired_limit_sweep()          # idempotent: no second lowering
    assert calls == [0]


def test_the_vlm_context_does_not_warn_with_the_sweep_off(fake_mx, monkeypatch,
                                                          capsys):
    """A streamed VLM pins the wired limit, and the copied mlx-vlm context
    must not warn against it on every turn. A resident model warns on its
    first turn only, as on the text path."""
    common = importlib.import_module("gmlx._vendor.mlx_vlm.generate.common")
    monkeypatch.setattr(common, "_large_model_warned", False)
    model = _over_budget_model()
    stock = mx.set_wired_limit

    def _no_sweep(*_a, **_k):
        return 0

    _no_sweep._kq_no_sweep = True
    monkeypatch.setattr(mx, "set_wired_limit", _no_sweep)
    with common.wired_limit(model):
        pass
    assert capsys.readouterr().out == ""

    monkeypatch.setattr(mx, "set_wired_limit", stock)
    for _ in range(3):
        with common.wired_limit(model):
            pass
    out = capsys.readouterr().out
    assert out.count("close to the maximum recommended size") == 1


def _recorded_limit(monkeypatch, start):
    calls = []

    def set_limit(n):
        prev = calls[-1] if calls else start
        calls.append(n)
        return prev

    monkeypatch.setattr(mx, "set_wired_limit", set_limit)
    return calls


def test_a_held_load_gets_the_limit_last_asked_for_at_its_end(monkeypatch):
    # Another model's generation raised the limit; it ends and a new one
    # starts while the load builds its buffers.
    from gmlx.stream.wired_limit import wired_limit_held

    calls = _recorded_limit(monkeypatch, 90)
    real = mx.set_wired_limit
    with wired_limit_held():
        assert calls == [0]
        assert mx.set_wired_limit(0) == 90
        assert mx.set_wired_limit(90) == 0
    assert calls == [0, 90] and mx.set_wired_limit is real


def test_a_held_load_that_streams_keeps_the_limit_at_0(monkeypatch,
                                                      restore_wired_limit):
    from gmlx.stream.wired_limit import (_neutralize_wired_limit_sweep,
                                         wired_limit_held)

    calls = _recorded_limit(monkeypatch, 90)
    with wired_limit_held():
        _neutralize_wired_limit_sweep()
    assert calls == [0]
    assert getattr(mx.set_wired_limit, "_kq_no_sweep", False)



def _decide_the_cap(monkeypatch) -> None:
    """The start decides the wired cap, as for a model predicted to map
    its table untracked (``capacity.decide_cap``): it reads the memory
    wired outside the process and installs the cap."""
    import gmlx.stream.table_stream as ts
    from gmlx.serve import capacity

    monkeypatch.setattr(ts, "predicts_untracked", lambda *a, **k: True)
    monkeypatch.setattr(capacity, "_CAP", None)
    assert capacity.decide_cap([("/m/Flash-Next-Q4.gguf", None)])


@pytest.mark.host_gb(128)
def test_a_caller_asking_for_the_working_set_gets_the_wired_cap(
        restore_wired_limit, monkeypatch):
    """mlx-lm's generation context asks for Metal's whole recommended
    working set. Under the clamp it gets the wired cap: the working set
    less the margin, or RAM less the reserve and the memory wired outside
    the server (read once when the start decides the cap), whichever is
    lower. The cap is the MLX memory limit too. A process the streaming
    install pinned stays at 0 and gets MLX's own memory limit back."""
    import host_box

    from gmlx.serve import capacity
    from gmlx.stream.wired_limit import _neutralize_wired_limit_sweep

    monkeypatch.setattr(mx.metal, "is_available", lambda: True)
    monkeypatch.setattr(mx, "default_device", lambda: mx.Device(mx.gpu))
    monkeypatch.setattr(mx, "synchronize", lambda *a, **k: None)
    monkeypatch.setattr(capacity, "_OUTSIDE_WIRED", None)
    calls = _recorded_limit(monkeypatch, 0)
    _install_wired_limit_warn_once()
    mlx_limit = mx.get_memory_limit()
    _decide_the_cap(monkeypatch)
    box, vm = host_box.mac(128)
    mem = box["memory_size"]
    cap = min(box["max_recommended_working_set_size"]
              * (1 - capacity.margin()),
              mem - capacity.reserve_bytes(mem) - vm["wired"])
    assert mx.get_memory_limit() == pytest.approx(cap, abs=1)
    model = {"w": mx.zeros((4,), dtype=mx.float32)}
    with _gen.wired_limit(model):
        pass
    assert calls == [pytest.approx(cap, abs=1), 0]
    _neutralize_wired_limit_sweep()
    assert mx.get_memory_limit() == mlx_limit
    with _gen.wired_limit(model):
        pass
    assert calls[2:] == [0]


# The line a load prints when its table is untracked and the clamp and the
# MLX memory limit both stand at the wired cap: what a server log shows.
B2_LINE = (r"\[memory\] per_layer_token_embd\.weight untracked, [0-9.]+ GB: "
           r"mx\.set_wired_limit clamped to the wired cap ([0-9.]+) GB, "
           r"MLX memory limit \1 GB$")


@pytest.mark.host_gb(128)
def test_a_load_with_an_untracked_table_prints_the_limits_it_leaves(
        restore_wired_limit, monkeypatch, capsys):
    """After the install, a load whose table MLX never tracks prints the
    limits it leaves, read back: the clamp owns mx.set_wired_limit at the
    wired cap, and the MLX memory limit is the same figure. A pin shows
    instead, with MLX's own memory limit back."""
    import re

    from gmlx.serve import capacity
    from gmlx.stream.wired_limit import (_neutralize_wired_limit_sweep,
                                         note_untracked_tables)

    monkeypatch.setattr(mx.metal, "is_available", lambda: True)
    monkeypatch.setattr(capacity, "_OUTSIDE_WIRED", None)
    _recorded_limit(monkeypatch, 0)
    mlx_limit = mx.get_memory_limit()
    _decide_the_cap(monkeypatch)
    capsys.readouterr()
    note_untracked_tables(["per_layer_token_embd.weight"], 28.8e9)
    line = capsys.readouterr().out.strip()
    m = re.search(B2_LINE, line)
    assert m, line
    assert m.group(1) == f"{capacity.wired_cap_bytes() / 1e9:.1f}"
    _neutralize_wired_limit_sweep()
    capsys.readouterr()
    note_untracked_tables(["per_layer_token_embd.weight"], 28.8e9)
    line = capsys.readouterr().out.strip()
    assert not re.search(B2_LINE, line)
    assert line == ("[memory] per_layer_token_embd.weight untracked, 28.8 GB: "
                    "the wired limit is pinned at 0, MLX memory limit "
                    f"{mlx_limit / 1e9:.1f} GB")


@pytest.mark.host_gb(128)
def test_a_process_that_maps_no_table_untracked_keeps_mlx_limits(
        restore_wired_limit, monkeypatch):
    """A start whose models predict no untracked map decides no wired
    cap, and the cache-limit steps install nothing: the process plans
    under the serve ceiling, and keeps MLX's own mx.set_wired_limit and
    memory limit, as before the wired cap. A generation then gets the
    whole working set it asks for."""
    from types import SimpleNamespace

    from gmlx.serve import capacity, memory

    monkeypatch.setattr(mx.metal, "is_available", lambda: True)
    monkeypatch.setattr(mx, "default_device", lambda: mx.Device(mx.gpu))
    monkeypatch.setattr(mx, "synchronize", lambda *a, **k: None)
    monkeypatch.setattr(mx, "set_cache_limit", lambda n: 0)
    monkeypatch.setattr(capacity, "_OUTSIDE_WIRED", None)
    monkeypatch.delenv("GMLX_CACHE_LIMIT_GB", raising=False)
    calls = _recorded_limit(monkeypatch, 0)
    stock = mx.set_wired_limit
    mlx_limit = mx.get_memory_limit()
    monkeypatch.setattr(capacity, "_CAP", None)
    assert not capacity.decide_cap([("/m/dense-Q4.gguf", None)])
    memory.apply_cache_limit(SimpleNamespace(cache_limit_gb=None, models={}))
    memory.apply_local_cache_limit([])
    assert not capacity.cap_applies()
    assert mx.set_wired_limit is stock
    assert mx.get_memory_limit() == mlx_limit
    ws = capacity.working_set_bytes()
    assert capacity.wired_cap_bytes() == capacity.ceiling_bytes(ws)
    _install_wired_limit_warn_once()
    with _gen.wired_limit({"w": mx.zeros((4,), dtype=mx.float32)}):
        pass
    assert calls == [pytest.approx(ws, abs=1), 0]
