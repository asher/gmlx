"""GPU keep-warm heartbeat (--gpu-keepwarm): start/stop lifecycle, the
loader's env gate, and decode gating. The heartbeat itself is a trivial
periodic kernel; what matters is that it is opt-in, idempotent, stoppable,
and parked whenever no decode is touching it."""

import time

from gmlx import keepwarm


def _wait_for(cond, timeout=2.0):
    t0 = time.monotonic()
    while not cond():
        if time.monotonic() - t0 > timeout:
            return False
        time.sleep(0.005)
    return True


def test_start_stop_idempotent():
    assert not keepwarm.running()
    try:
        assert keepwarm.start()
        assert keepwarm.running()
        assert keepwarm.start()  # second start: still one thread
        time.sleep(0.01)
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


def test_parks_when_idle_and_wakes_on_touch(monkeypatch):
    """The heartbeat parks once no touch has arrived for the idle window
    and resumes on the next decode touch - the request-active contract."""
    monkeypatch.setenv("GMLX_KEEPWARM_IDLE_S", "0.05")
    try:
        keepwarm.start()
        assert _wait_for(keepwarm.parked)
        keepwarm.touch()
        assert _wait_for(lambda: not keepwarm.parked())
        assert keepwarm.running()
        assert _wait_for(keepwarm.parked)  # parks again after the window
        keepwarm.stop()  # stop while parked must not hang
    finally:
        keepwarm.stop()
    assert not keepwarm.running() and not keepwarm.parked()


def test_idle_zero_beats_continuously(monkeypatch):
    monkeypatch.setenv("GMLX_KEEPWARM_IDLE_S", "0")
    try:
        keepwarm.start()
        time.sleep(0.15)
        assert keepwarm.running() and not keepwarm.parked()
    finally:
        keepwarm.stop()


def test_touch_without_thread_is_noop():
    assert not keepwarm.running()
    keepwarm.touch()  # no thread: cheap early-out, no crash
    assert not keepwarm.running()
