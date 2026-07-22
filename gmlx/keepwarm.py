"""GPU keep-warm heartbeat for over-RAM streamed decode.

Streamed decode alternates sub-millisecond GPU bursts with host/disk
gaps every MoE layer; the GPU races to idle between bursts and every
burst pays the clock ramp (measured 3-5x inflation of identical work
on GLM-5.2: 4.3 ms/layer cold vs 0.3 warm, +38% end-to-end tok/s from
an external heartbeat). This module holds clocks up from a background
thread submitting a tiny periodic kernel on its own stream.

Lossless but not free: it burns a few watts while decode is active,
so it is opt-in (``--gpu-keepwarm`` / ``GMLX_GPU_KEEPWARM=1``). The
real fix is the GPU-autonomous token (gpu-dispatch tier 2); this is
the shippable stopgap.
"""

from __future__ import annotations

import threading

import mlx.core as mx

_PERIOD_S = 0.5e-3
_DIM = 256

_lock = threading.Lock()
_thread: threading.Thread | None = None
_stop: threading.Event | None = None


def _run(stop: threading.Event):
    with mx.stream(mx.gpu):
        a = mx.random.normal((_DIM, _DIM))
        mx.eval(a)
        while not stop.wait(_PERIOD_S):
            mx.eval(a @ a)


def start() -> bool:
    """Start the heartbeat thread (idempotent). Returns True if running."""
    global _thread, _stop
    with _lock:
        if _thread is not None and _thread.is_alive():
            return True
        _stop = threading.Event()
        _thread = threading.Thread(
            target=_run, args=(_stop,), name="gmlx-keepwarm", daemon=True)
        _thread.start()
        return True


def stop() -> None:
    """Stop the heartbeat thread (idempotent)."""
    global _thread, _stop
    with _lock:
        if _stop is not None:
            _stop.set()
        t, _thread, _stop = _thread, None, None
    if t is not None:
        t.join(timeout=1.0)


def running() -> bool:
    with _lock:
        return _thread is not None and _thread.is_alive()
