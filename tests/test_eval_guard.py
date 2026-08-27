#!/usr/bin/env python3
"""Guarded eval contract (gmlx/eval_guard.py) and the bare-eval ban.

The ban half is the regression that matters: no mx.eval or
mx.async_eval call may sit under an except-carrying try anywhere in
gmlx. That is what keeps a sixth catch-and-continue-around-eval site
from shipping unrouted, and it subsumes checking that the five known
sites route through the guard.
"""

from __future__ import annotations

import ast
import os
import re
from pathlib import Path

import pytest

import mlx.core as mx

import gmlx.eval_guard as eg
from gmlx.eval_guard import (
    guard,
    install_stream_registry,
    known_streams,
    recovery_state,
)

GMLX_DIR = Path(eg.__file__).resolve().parent


def _real_apple_gpu() -> bool:
    if os.environ.get("KQUANT_FORCE_CPU"):
        return False
    try:
        name = str(mx.device_info().get("device_name", ""))
    except Exception:
        return False
    return bool(re.search(r"Apple M\d", name))


def _trip():
    """Oversized allocation that throws at one malloc with nothing
    committed (same construction as tests/invariant_probes/_common.py:
    mx.contiguous is the materializer because scalar binary ops keep
    the broadcast strides and never reach the oversized malloc)."""
    n = 2048
    limit = mx.device_info()["max_buffer_length"]
    rows = int(limit * 1.2) // (n * n * 4) + 1
    mid = mx.full((n,), 1.0) * 1.0001 + 1.0
    return mid, mx.contiguous(mx.broadcast_to(mid, (rows, n, n)))


# ---------------------------------------------------------------- ban

BARE_EVAL = ("eval", "async_eval")


def _bare_evals_under_except(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    hits: list[str] = []

    class V(ast.NodeVisitor):
        def __init__(self):
            self.try_depth = 0

        def visit_Try(self, node):
            guarded = bool(node.handlers)
            if guarded:
                self.try_depth += 1
            for child in node.body + node.orelse:
                self.visit(child)
            if guarded:
                self.try_depth -= 1
            # handlers and finalbody are outside the caught region
            for child in node.handlers + node.finalbody:
                self.visit(child)

        def visit_Call(self, node):
            f = node.func
            if (self.try_depth > 0 and isinstance(f, ast.Attribute)
                    and f.attr in BARE_EVAL
                    and isinstance(f.value, ast.Name)
                    and f.value.id == "mx"):
                hits.append(f"{path.name}:{node.lineno} mx.{f.attr}")
            self.generic_visit(node)

    V().visit(tree)
    return hits


def test_no_bare_eval_under_except_anywhere():
    offenders: list[str] = []
    for p in sorted(GMLX_DIR.glob("*.py")):
        if p.name == "eval_guard.py":
            continue  # the guard owns the only sanctioned bare evals
        offenders += _bare_evals_under_except(p)
    assert not offenders, (
        "bare mx.eval/mx.async_eval under an except-carrying try; route "
        "through gmlx.eval_guard.guard instead:\n  " + "\n  ".join(offenders))


def test_ban_scanner_sees_the_pattern(tmp_path):
    bad = tmp_path / "bad.py"
    bad.write_text(
        "import mlx.core as mx\n"
        "def f(x):\n"
        "    try:\n"
        "        mx.eval(x)\n"
        "    except Exception:\n"
        "        pass\n")
    assert _bare_evals_under_except(bad) == ["bad.py:4 mx.eval"]
    ok = tmp_path / "ok.py"
    ok.write_text(
        "import mlx.core as mx\n"
        "def f(x):\n"
        "    mx.eval(x)\n"
        "    try:\n"
        "        pass\n"
        "    finally:\n"
        "        mx.eval(x)\n")
    assert _bare_evals_under_except(ok) == []


# ----------------------------------------------------------- behavior

def test_success_passthrough_and_no_stamp():
    x = mx.full((64,), 3.0)
    guard.eval(x, site="test-ok", owner="scratch")
    assert float(x.sum().item()) == pytest.approx(3.0 * 64)
    assert recovery_state() is None


def test_owner_validated():
    with pytest.raises(ValueError):
        guard.eval(mx.zeros((1,)), site="t", owner="borrowed")


def test_registry_wrap_records_owner_thread():
    install_stream_registry()
    before = len(known_streams())
    s = mx.new_stream(mx.default_device())
    assert len(known_streams()) == before + 1
    assert s in known_streams()


def test_kill_switch_bypasses(monkeypatch):
    monkeypatch.setenv("GMLX_EVAL_GUARD", "0")
    x = mx.full((8,), 2.0)
    guard.eval(x, site="test-off", owner="scratch")
    assert recovery_state() is None


@pytest.mark.skipif(not _real_apple_gpu(),
                    reason="allocator trip needs the Metal device")
def test_failure_drains_stamps_and_reraises():
    seen = {}
    orig_drain = eg._drain

    def spy(site):
        seen["state"] = recovery_state()
        orig_drain(site)

    eg._drain, restore = spy, orig_drain
    try:
        mid, big = _trip()
        with pytest.raises(RuntimeError):
            guard.eval(big, site="test-trip", owner="scratch")
    finally:
        eg._drain = restore
    # stamp was live during the drain, with the site and a deadline
    st = seen["state"]
    assert st and st["site"] == "test-trip" and st["path"] == "sync"
    assert st["deadline"] > st["started"]
    # and cleared on the re-raise exit
    assert recovery_state() is None
    # the survivor is rehabilitated and the stream usable (sync path)
    assert float(mid.sum().item()) == pytest.approx((1.0 * 1.0001 + 1.0)
                                                    * 2048)
    y = mx.full((256,), 5.0)
    guard.eval(y, site="test-post", owner="scratch")
    assert float(y.sum().item()) == pytest.approx(5.0 * 256)


@pytest.mark.skipif(not _real_apple_gpu(),
                    reason="allocator trip needs the Metal device")
def test_async_path_failure_reraises_and_clears():
    _, big = _trip()
    with pytest.raises(RuntimeError):
        guard.async_eval(big, site="test-async-trip", owner="owned")
    assert recovery_state() is None
    z = mx.full((32,), 7.0)
    guard.eval(z, site="test-post-async", owner="scratch")
    assert float(z.sum().item()) == pytest.approx(7.0 * 32)
