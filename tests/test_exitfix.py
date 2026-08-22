"""The mlx 0.32.1 exit-segv guard: gating, coercion, e2e protection."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

import gmlx._exitfix as ef

ROOT = str(Path(ef.__file__).resolve().parent.parent)

NEEDS_BUG = pytest.mark.skipif(
    getattr(pytest.importorskip("mlx.core"), "__version__", "") != "0.32.1",
    reason="mlx exit-segv is 0.32.1-specific",
)

# Child that reproduces the upstream hazard: a compiled function whose
# wrapper is still referenced when the interpreter shuts down.
CHILD = """\
import ctypes, sys
sys.path.insert(0, {root!r})
{guard}
import mlx.core as mx
mx.set_default_device(mx.cpu)
f = mx.compile(lambda x: (x + 1, x * 2))
f(mx.array(1.0))
ctypes.pythonapi.Py_IncRef(ctypes.py_object(f))
{code}
"""


def _fresh(monkeypatch):
    monkeypatch.setattr(ef, "_code", None)
    monkeypatch.setattr(ef, "_armed", False)


def test_set_code_coercion(monkeypatch):
    _fresh(monkeypatch)
    for rc, want in ((7, 7), (0, 0), (None, 0), ("boom", 1), (True, 1)):
        ef.set_code(rc)
        assert ef._code == want, rc


def test_affected_gates(monkeypatch):
    class FakeMx:
        __version__ = "0.32.1"

    monkeypatch.setitem(sys.modules, "mlx.core", FakeMx())
    monkeypatch.delenv("GMLX_EXITFIX", raising=False)
    assert ef.affected()
    monkeypatch.setenv("GMLX_EXITFIX", "0")
    assert not ef.affected()
    monkeypatch.delenv("GMLX_EXITFIX", raising=False)
    FakeMx.__version__ = "0.33.0"
    assert not ef.affected()


def test_guard_passive_without_recorded_code(monkeypatch):
    # A guard that fired here would kill the test process.
    _fresh(monkeypatch)
    ef._guard()


@NEEDS_BUG
def test_upstream_bug_still_present_exit_segv():
    # Tripwire: when this run starts exiting 0, mlx fixed upstream #4248.
    # Retire the guard then (drop the version from _AFFECTED_MLX) after the
    # usual A/B on the serve path.
    r = subprocess.run(
        [sys.executable, "-c", CHILD.format(root=ROOT, guard="", code="")],
        capture_output=True,
    )
    assert r.returncode in (-11, 139)


@NEEDS_BUG
def test_guard_preserves_exit_code_under_leak():
    guard = "from gmlx._exitfix import arm, set_code\narm()"
    r = subprocess.run(
        [
            sys.executable,
            "-c",
            CHILD.format(root=ROOT, guard=guard, code="set_code(5)"),
        ],
        capture_output=True,
    )
    assert r.returncode == 5, (r.returncode, r.stderr.decode()[-500:])


def test_cli_entry_exits_clean():
    env = dict(os.environ)
    env["PYTHONPATH"] = ROOT + os.pathsep + env.get("PYTHONPATH", "")
    r = subprocess.run(
        [sys.executable, "-m", "gmlx", "--version"],
        capture_output=True,
        env=env,
    )
    assert r.returncode == 0, r.stderr.decode()[-500:]
    assert r.stdout.strip()
