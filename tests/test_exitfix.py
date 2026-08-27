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
    # No fixed mlx release exists: everything from 0.32.1 on stays gated.
    FakeMx.__version__ = "0.33.0"
    assert ef.affected()
    FakeMx.__version__ = "0.32.0"
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
    #
    # The GIL-less decref lands as SIGSEGV, or as a Py_FatalError SIGABRT
    # on CPythons that detect the NULL thread state first (seen on 3.13);
    # the abort path must show the fatal-error signature so an unrelated
    # abort cannot pass as the bug.
    r = subprocess.run(
        [sys.executable, "-c", CHILD.format(root=ROOT, guard="", code="")],
        capture_output=True,
    )
    if r.returncode in (-6, 134):
        err = r.stderr.decode(errors="replace")
        assert ("thread state is NULL" in err
                or "runtime state: finalizing" in err), (
            r.returncode, err[-500:])
    else:
        assert r.returncode in (-11, 139), (
            r.returncode, r.stderr.decode(errors="replace")[-500:])


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


# Thread-exit guard: the engine thread parks instead of exiting at
# stop_and_join, and the parked frame must not pin the generator.

class _FakeEngine:
    """Minimal ResponseGenerator shape: queue-fed loop on a daemon thread."""

    def __init__(self):
        import queue
        import threading
        self.requests = queue.Queue()
        self._stop = False
        class _Weights:
            pass
        self.model = _Weights()   # stands in for the loaded weights
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while not self._stop:
            if self.requests.get() is None:
                break

    def stop_and_join(self):
        self._stop = True
        self.requests.put(None)
        self._thread.join(timeout=5.0)


def _patched_engine_class(monkeypatch, affected):
    import types

    from gmlx import _exitfix

    cls = type("_FakeEngineCopy", (_FakeEngine,), {})
    mod = types.SimpleNamespace(ResponseGenerator=cls)
    monkeypatch.setattr(_exitfix, "thread_guard_affected", lambda: affected)
    _exitfix.install_engine_thread_guard(mod)
    return cls


def test_thread_guard_parks_thread_and_signals_drained(monkeypatch):
    import time
    cls = _patched_engine_class(monkeypatch, affected=True)
    eng = cls()
    t = eng._thread
    t0 = time.monotonic()
    eng.stop_and_join()
    # Returned on the drained event, not the 5s join timeout. The guard
    # clears the instance dict afterwards, so assert via timing.
    assert time.monotonic() - t0 < 4.0
    assert t.is_alive()          # parked, not exited: TSD dtor never runs


def test_thread_guard_parked_skeleton_releases_model(monkeypatch):
    # The instance itself stays pinned by Thread.run()'s stack, but the
    # guard empties it: the model (weights) must be collectable and the
    # parked skeleton's __dict__ empty.
    import gc
    import time
    import weakref
    cls = _patched_engine_class(monkeypatch, affected=True)
    eng = cls()
    t = eng._thread
    ref = weakref.ref(eng.model)
    eng.stop_and_join()
    for _ in range(50):
        gc.collect()
        if ref() is None:
            break
        time.sleep(0.02)
    assert ref() is None         # weights released despite the parked thread
    assert eng.__dict__ == {}
    assert t.is_alive()


def test_thread_guard_unaffected_keeps_stock_join(monkeypatch):
    cls = _patched_engine_class(monkeypatch, affected=False)
    eng = cls()
    t = eng._thread
    eng.stop_and_join()
    t.join(timeout=5.0)
    assert not t.is_alive()      # stock behavior: thread exits


def test_thread_guard_install_is_idempotent(monkeypatch):
    import types

    from gmlx import _exitfix
    cls = type("_FakeEngineCopy2", (_FakeEngine,), {})
    mod = types.SimpleNamespace(ResponseGenerator=cls)
    _exitfix.install_engine_thread_guard(mod)
    first = cls._run
    _exitfix.install_engine_thread_guard(mod)
    assert cls._run is first
