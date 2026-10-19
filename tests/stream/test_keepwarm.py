"""GPU keep-warm heartbeat (--gpu-keepwarm): start/stop lifecycle, the
loader's env gate, and decode gating. The heartbeat itself is a trivial
periodic kernel; what matters is that it is opt-in, idempotent, stoppable,
and parked whenever no decode is touching it."""

import threading
import time
from types import SimpleNamespace

import mlx.core as mx
import pytest

import gmlx.stream.keepwarm as keepwarm


def _wait_for(cond, timeout=30.0):
    """Poll for a state that holds once reached."""
    t0 = time.monotonic()
    while not cond():
        if time.monotonic() - t0 > timeout:
            return False
        time.sleep(0.005)
    return True


class _Beats:
    """keepwarm's ``mx``, counting the evaluations of the heartbeat."""

    def __init__(self):
        self.evals = 0
        self._cond = threading.Condition()

    def __getattr__(self, name):
        return getattr(mx, name)

    def eval(self, *arrays):
        mx.eval(*arrays)
        with self._cond:
            self.evals += 1
            self._cond.notify_all()

    def wait(self, n: int, timeout: float = 30.0) -> bool:
        """Wait until ``n`` more evaluations have run."""
        with self._cond:
            want = self.evals + n
            return self._cond.wait_for(lambda: self.evals >= want, timeout)


@pytest.fixture
def clock(monkeypatch):
    """keepwarm's clock, which moves only when a test moves it."""
    now = SimpleNamespace(t=1000.0)
    monkeypatch.setattr(keepwarm, "time", SimpleNamespace(monotonic=lambda: now.t))
    return now


@pytest.fixture
def beats(monkeypatch):
    b = _Beats()
    monkeypatch.setattr(keepwarm, "mx", b)
    return b


def test_start_stop_idempotent(clock, beats):
    assert not keepwarm.running()
    try:
        assert keepwarm.start()
        assert keepwarm.running()
        assert keepwarm.start()  # second start: still one thread
        assert beats.wait(2)
        assert keepwarm.running()
    finally:
        keepwarm.stop()
    assert not keepwarm.running()
    keepwarm.stop()  # idempotent


def test_env_gate_defaults_off(monkeypatch):
    from gmlx.envflags import env_bool

    monkeypatch.delenv("GMLX_GPU_KEEPWARM", raising=False)
    assert not env_bool("GMLX_GPU_KEEPWARM", False)
    monkeypatch.setenv("GMLX_GPU_KEEPWARM", "1")
    assert env_bool("GMLX_GPU_KEEPWARM", False)


def test_parks_when_idle_and_wakes_on_touch(monkeypatch, clock, beats):
    """The heartbeat parks once no touch has arrived for the idle window
    and resumes on the next decode touch - the request-active contract."""
    monkeypatch.setenv("GMLX_KEEPWARM_IDLE_S", "0.05")
    try:
        keepwarm.start()
        assert beats.wait(2)
        assert not keepwarm.parked()  # inside the window it beats
        clock.t += 0.1
        assert _wait_for(keepwarm.parked)
        keepwarm.touch()
        assert _wait_for(lambda: not keepwarm.parked())
        assert beats.wait(2)
        assert keepwarm.running() and not keepwarm.parked()
        clock.t += 0.1
        assert _wait_for(keepwarm.parked)  # parks again after the window
        keepwarm.stop()  # stop while parked must not hang
    finally:
        keepwarm.stop()
    assert not keepwarm.running() and not keepwarm.parked()


def test_idle_zero_beats_continuously(monkeypatch, clock, beats):
    monkeypatch.setenv("GMLX_KEEPWARM_IDLE_S", "0")
    try:
        keepwarm.start()
        clock.t += 3600.0
        assert beats.wait(3)
        assert keepwarm.running() and not keepwarm.parked()
    finally:
        keepwarm.stop()


def test_touch_without_thread_is_noop():
    assert not keepwarm.running()
    keepwarm.touch()  # no thread: cheap early-out, no crash
    assert not keepwarm.running()
