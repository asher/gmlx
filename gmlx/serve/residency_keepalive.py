"""Keep the served models' GPU memory resident between requests.

MLX requests residency for its residency sets once, and Metal holds such a
request about two seconds past the last command buffer. After that the
memory is unwired, and the next request's first command buffer waits
while all of it is wired again, about 6 ms a GiB. While the engine loop is
idle within ``GMLX_RESIDENCY_KEEPALIVE_S`` of its last step, it evaluates
one tiny array on the GPU every second, which renews the residency.

The keep-alive never holds memory someone else needs. It does not beat
while the kernel reports memory pressure or little memory not wired (the
checks the idle release makes). It stops until the next step, so the
memory unwires, once other apps are being squeezed: the compressor has
grown past where it stood at the first beat by more than about 1% of RAM,
or free and reclaimable memory is under the governor's kernel floor. A
model load first stops it and waits out the lapse, so the load measures
what it measured without it. The next step of any engine starts it again.
"""

from __future__ import annotations

import logging
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

# Compressor growth past the window's first beat that stops the beats, as
# a share of physical RAM.
_SQUEEZE_SHARE = 0.01

_log = logging.getLogger(__name__)
_lock = threading.Lock()
_last_active: float | None = None
_last_beat: float | None = None
_yielded = False
# Compressor bytes at the window's first beat.
_baseline: int | None = None


def keepalive_s() -> float:
    return env_float("GMLX_RESIDENCY_KEEPALIVE_S", 600.0)


def note_active(now: float | None = None) -> None:
    """An engine step ran or a request arrived."""
    global _last_active, _yielded, _baseline
    _last_active = time.monotonic() if now is None else now
    _yielded = False
    _baseline = None


def _pressure() -> bool:
    from .idle_release import _floor, _pressure_level, _unwired

    if _pressure_level() > 1:
        return True
    unwired, floor = _unwired(), _floor()
    return unwired is not None and floor > 0 and unwired < 2 * floor


def _compressor() -> int | None:
    from .kernel_vm import snapshot

    s = snapshot()
    return None if s is None else int(s["compressor"])


def _reclaimable() -> int | None:
    from gmlx.stream.budget import reclaimable_ram_bytes

    return reclaimable_ram_bytes()


def _ram() -> int | None:
    from gmlx.stream.budget import ram_bytes

    return ram_bytes()


def _squeeze() -> str:
    """Why the beats stop until the next step: the memory the keep-alive
    holds is being taken from other apps. Empty when it is not."""
    from .idle_release import _floor

    global _baseline
    comp, ram = _compressor(), _ram()
    if comp is not None and ram:
        if _baseline is None:
            _baseline = comp
        elif comp - _baseline > _SQUEEZE_SHARE * ram:
            return (f"the compressor grew {(comp - _baseline) / 1e9:.2f} GB "
                    "since the first beat")
    free, floor = _reclaimable(), _floor()
    if free is not None and floor > 0 and free < floor:
        return (f"free and reclaimable memory {free / 1e9:.2f} GB is under "
                f"{floor / 1e9:.2f} GB")
    return ""


def _beat() -> None:
    mx.eval(mx.zeros((1,), dtype=mx.float32) + 1)


def idle_tick(now: float | None = None) -> bool:
    """One idle pass of an engine loop: beat when one is due. True when it
    beat."""
    global _last_beat, _yielded
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
        why = _squeeze()
        if why:
            _yielded = True
        else:
            # Booked before the beat, so a load that yields now waits for it.
            _last_beat = now
    if why:
        _log.info("[residency] keep-alive stopped until the next request: %s",
                  why)
        return False
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
