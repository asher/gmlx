"""Keep the served models' GPU memory resident between requests.

MLX requests residency for its residency sets once, and Metal holds such a
request about two seconds past the last command buffer. After that the
memory is unwired, and the next request's first command buffer waits
while all of it is wired again, about 6 ms a GiB. While the engine loop is
idle within ``GMLX_RESIDENCY_KEEPALIVE_S`` of its last step, it evaluates
one tiny array on the GPU every second, which renews the residency.

The keep-alive never holds memory someone else needs. It does not beat
while the kernel reports memory pressure or little memory not wired (the
checks the idle release makes), and a model load first stops it and waits
out the lapse, so the load measures what it measured without it. The next
step of any engine starts it again.
"""

from __future__ import annotations

import threading
import time

import mlx.core as mx

from gmlx.envflags import env_float

# Seconds between beats. Metal keeps the sets resident for about two
# seconds past the last command buffer; a beat at 1.5 s still held them.
_BEAT_S = 1.0
# Seconds after the last beat by which the residency has lapsed: past 2.5 s
# of idle the memory was unwired.
_LAPSE_S = 3.5

_lock = threading.Lock()
_last_active: float | None = None
_last_beat: float | None = None
_yielded = False


def keepalive_s() -> float:
    return env_float("GMLX_RESIDENCY_KEEPALIVE_S", 600.0)


def note_active(now: float | None = None) -> None:
    """An engine step ran or a request arrived."""
    global _last_active, _yielded
    _last_active = time.monotonic() if now is None else now
    _yielded = False


def _pressure() -> bool:
    from .idle_release import _floor, _pressure_level, _unwired

    if _pressure_level() > 1:
        return True
    unwired, floor = _unwired(), _floor()
    return unwired is not None and floor > 0 and unwired < 2 * floor


def _beat() -> None:
    mx.eval(mx.zeros((1,), dtype=mx.float32) + 1)


def idle_tick(now: float | None = None) -> bool:
    """One idle pass of an engine loop: beat when one is due. True when it
    beat."""
    global _last_beat
    keep = keepalive_s()
    if keep <= 0:
        return False
    now = time.monotonic() if now is None else now
    with _lock:
        if _yielded or _last_active is None or now - _last_active > keep:
            return False
        last = max(_last_active, _last_beat or _last_active)
        if now - last < _BEAT_S:
            return False
        if _pressure():
            return False
        # Booked before the beat, so a load that yields now waits for it.
        _last_beat = now
    _beat()
    return True


def yield_for_load(now: float | None = None, sleep=time.sleep) -> float:
    """Stop the keep-alive until the next step and wait until the memory it
    held has lapsed. Seconds waited: none when no beat ran since the last
    step, since the memory is then held as it was without the keep-alive."""
    global _yielded
    with _lock:
        _yielded = True
        beat, active = _last_beat, _last_active
    if beat is None or (active is not None and active >= beat):
        return 0.0
    now = time.monotonic() if now is None else now
    wait = _LAPSE_S - (now - beat)
    if wait <= 0:
        return 0.0
    sleep(wait)
    return wait
