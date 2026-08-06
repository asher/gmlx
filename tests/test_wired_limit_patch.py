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

from gmlx.loader import _install_wired_limit_warn_once

# `import mlx_lm.generate` binds the function mlx_lm re-exports in
# __init__, not the submodule - same trap the loader patches around.
_gen = importlib.import_module("mlx_lm.generate")


@pytest.fixture
def restore_wired_limit():
    orig = _gen.wired_limit
    try:
        yield
    finally:
        _gen.wired_limit = orig


@pytest.fixture
def fake_mx(monkeypatch):
    """A tiny fake device: 100-byte recommended working set, recorded
    set_wired_limit calls, no-op synchronize."""
    calls = []

    monkeypatch.setattr(mx.metal, "is_available", lambda: True)
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
