"""GPU keep-warm heartbeat (--gpu-keepwarm): start/stop lifecycle and the
loader's env gate. The heartbeat itself is a trivial periodic kernel; what
matters is that it is opt-in, idempotent, and stoppable."""

import time

from gmlx import keepwarm


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
