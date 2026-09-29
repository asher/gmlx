"""The wired limit context sets no limit on the CPU device.

With Metal present and the CPU as the default device, the stock
``wired_limit`` of mlx-lm and mlx-vlm reads a key only the GPU's device
info carries and raises KeyError on entry. Each binding the guard wraps
must yield there without setting a limit, and on the GPU still raise the
limit and restore it. Metal is faked present, so the checks run on a
machine without it too.
"""

from __future__ import annotations

import importlib

import mlx.core as mx
import pytest

from gmlx.upstream.wired_limit_cpu_fix import (
    _BINDINGS,
    install_wired_limit_cpu_fix,
)


@pytest.fixture
def guarded(monkeypatch):
    """The guard on every binding, each binding back as it was after the
    test, and the limit setter recorded."""
    mods = [importlib.import_module(m) for m in _BINDINGS]
    for mod in mods:
        monkeypatch.setattr(mod, "wired_limit", mod.wired_limit)
    assert install_wired_limit_cpu_fix()
    calls = []
    monkeypatch.setattr(mx.metal, "is_available", lambda: True)
    monkeypatch.setattr(
        mx, "set_wired_limit", lambda v: (calls.append(v), 7)[1])
    return mods, calls


def _model():
    return {"w": mx.zeros((4,), dtype=mx.float32)}


def test_each_binding_sets_no_limit_on_the_cpu_device(guarded):
    mods, calls = guarded
    mx.set_default_device(mx.cpu)
    stream = mx.default_stream(mx.cpu)
    for mod in mods:
        with mod.wired_limit(_model(), [stream]):
            pass
        with mod.wired_limit(_model()):
            pass
    assert calls == []


def test_each_binding_raises_the_limit_on_the_gpu(guarded, monkeypatch):
    mods, calls = guarded
    monkeypatch.setattr(mx, "default_device", lambda: mx.Device(mx.gpu))
    monkeypatch.setattr(
        mx, "device_info", lambda: {"max_recommended_working_set_size": 100})
    monkeypatch.setattr(mx, "synchronize", lambda *a, **k: None)
    for mod in mods:
        with mod.wired_limit(_model()):
            pass
    assert calls == [100, 7] * len(mods)


def test_install_is_idempotent_and_leaves_the_gmlx_contexts(guarded):
    from gmlx.stream.wired_limit import _install_wired_limit_warn_once

    mods, _ = guarded
    first = [mod.wired_limit for mod in mods]
    assert install_wired_limit_cpu_fix()
    assert [mod.wired_limit for mod in mods] == first
    gen = importlib.import_module("mlx_lm.generate")
    gen.wired_limit = gen.wired_limit._gmlx_cpu_guard
    _install_wired_limit_warn_once()
    warn_once = gen.wired_limit
    install_wired_limit_cpu_fix()
    assert gen.wired_limit is warn_once
