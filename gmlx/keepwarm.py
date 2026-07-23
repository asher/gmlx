"""GPU keep-warm heartbeat for over-RAM streamed decode.

Streamed decode alternates sub-millisecond GPU bursts with host/disk
gaps every MoE layer; the GPU races to idle between bursts and every
burst pays the clock ramp (measured 3-5x inflation of identical work
on GLM-5.2: 4.3 ms/layer cold vs 0.3 warm, +38% end-to-end tok/s from
an external heartbeat). This module holds clocks up from a background
thread submitting a tiny periodic kernel on its own stream.

The heartbeat is decode-gated: the streamed decode path calls
``touch()`` every MoE layer, and the thread parks (zero GPU work, zero
power) once no touch has arrived for ``GMLX_KEEPWARM_IDLE_S`` seconds
(default 1.0; <=0 beats continuously). A long-lived server therefore
only pays for it while a request is actually decoding; the first token
after an idle gap eats one clock ramp, every later token is held.

Lossless but not free: it burns a few watts while decode is active,
so it is opt-in (``--gpu-keepwarm`` / ``GMLX_GPU_KEEPWARM=1``). The
real fix is the GPU-autonomous token (gpu-dispatch tier 2); this is
the shippable stopgap.
"""

from __future__ import annotations

import threading
import time

import mlx.core as mx

from .envflags import env_float

_PERIOD_S = 0.5e-3
_DIM = 256
_IDLE_S_ENV = "GMLX_KEEPWARM_IDLE_S"

_lock = threading.Lock()
_thread: threading.Thread | None = None
_stop: threading.Event | None = None
_wake: threading.Event | None = None
_parked = threading.Event()
_last_touch = 0.0


def touch() -> None:
    """Decode-activity ping (called per streamed MoE layer). A float
    store when the heartbeat is awake; wakes it when parked."""
    global _last_touch
    if _thread is None:
        return
    _last_touch = time.monotonic()
    if _parked.is_set():
        w = _wake
        if w is not None:
            w.set()


def _run(stop: threading.Event, wake: threading.Event, idle_s: float):
    with mx.stream(mx.gpu):
        a = mx.random.normal((_DIM, _DIM))
        mx.eval(a)
        while not stop.is_set():
            if idle_s > 0 and time.monotonic() - _last_touch > idle_s:
                _parked.set()
                wake.wait()
                wake.clear()
                _parked.clear()
                continue
            mx.eval(a @ a)
            stop.wait(_PERIOD_S)


def start() -> bool:
    """Start the heartbeat thread (idempotent). Returns True if running."""
    global _thread, _stop, _wake, _last_touch
    with _lock:
        if _thread is not None and _thread.is_alive():
            return True
        _stop = threading.Event()
        _wake = threading.Event()
        _parked.clear()
        idle_s = env_float(_IDLE_S_ENV, 1.0)
        _thread = threading.Thread(
            target=_run, args=(_stop, _wake, idle_s),
            name="gmlx-keepwarm", daemon=True)
        _last_touch = time.monotonic()
        _thread.start()
        return True


def stop() -> None:
    """Stop the heartbeat thread (idempotent)."""
    global _thread, _stop, _wake
    with _lock:
        if _stop is not None:
            _stop.set()
        if _wake is not None:
            _wake.set()  # unpark so the loop sees the stop
        t, _thread, _stop, _wake = _thread, None, None, None
    if t is not None:
        t.join(timeout=1.0)
    _parked.clear()


def running() -> bool:
    with _lock:
        return _thread is not None and _thread.is_alive()


def parked() -> bool:
    """True when the thread is alive but idle-parked (no GPU work)."""
    return running() and _parked.is_set()
