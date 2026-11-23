"""Release of the buffers GMLX_SPEC_KEEP_BUFFERS holds between requests.

With the lever on, a speculative request leaves its freed buffers in MLX's
allocator cache so the next one allocates nothing. Nothing else clears that
cache while the server is idle: the governor ticks only inside a generator
step. This module gives the held buffers back in the two places where they
would otherwise count against someone else: while the engine loop waits for
a request and the kernel reports memory pressure or little memory not
wired, and before a model load is measured against the free working set.
With the lever off both are no-ops.
"""

from __future__ import annotations

import logging
import time

import mlx.core as mx

from gmlx.envflags import env_bool
from gmlx.upstream.patching import wrapped_with

_log = logging.getLogger(__name__)

_INSTALLED_FLAG = "_kq_idle_release"
# Seconds between idle checks; a check is four sysctl reads.
_CHECK_S = 1.0
# A cache under this is not worth a clear.
_MIN_BYTES = 64 << 20
_last_check = 0.0


def keep_buffers() -> bool:
    return env_bool("GMLX_SPEC_KEEP_BUFFERS", False)


def _pressure_level() -> int:
    from gmlx.stream.decode_feeder import _pressure_level as level

    return level()


def _unwired():
    from .kernel_vm import unwired_bytes

    return unwired_bytes()


def _floor() -> float:
    from .governor import _kernel_floor_bytes

    return _kernel_floor_bytes()


def release(why: str) -> int:
    """Clear MLX's cache; the bytes it held."""
    from .capacity import note_own_before_clear

    held = int(mx.get_cache_memory())
    note_own_before_clear()
    mx.clear_cache()
    _log.info("[idle-release] %s: released %.2f GB of kept buffers",
              why, held / 1e9)
    return held


def before_load_gate() -> int:
    """A model load is measured with the kept buffers given back, as it is
    with the lever off. The bytes released."""
    if not keep_buffers() or int(mx.get_cache_memory()) < _MIN_BYTES:
        return 0
    return release("model load")


def idle_check(now: float | None = None) -> int:
    """One check from the idle engine loop: release the kept buffers when
    the kernel reports memory pressure, or when the memory not wired is
    under twice the governor's floor. The bytes released."""
    global _last_check
    if not keep_buffers():
        return 0
    now = time.monotonic() if now is None else now
    if now - _last_check < _CHECK_S:
        return 0
    _last_check = now
    if int(mx.get_cache_memory()) < _MIN_BYTES:
        return 0
    level = _pressure_level()
    if level > 1:
        return release(f"memory pressure level {level}")
    unwired, floor = _unwired(), _floor()
    if unwired is not None and floor > 0 and unwired < 2 * floor:
        return release(f"memory not wired {unwired / 1e9:.2f} GB under "
                       f"{2 * floor / 1e9:.2f} GB")
    return 0


def install_idle_release() -> None:
    """Wrap the engine loop's request wait: an idle pass (no active row,
    nothing queued) runs ``idle_check`` and the residency keep-alive's
    tick, and any other pass marks the engine active for it. Idempotent."""
    from . import residency_keepalive

    from gmlx._vendor.mlx_vlm.server import generation as _generation

    cls = _generation.ResponseGenerator
    original = cls._collect_pending_requests
    if wrapped_with(original, _INSTALLED_FLAG):
        return

    def _collect_pending_requests(self, *, active: bool, **kwargs):
        pending, should_stop = original(self, active=active, **kwargs)
        if active or pending:
            residency_keepalive.note_active()
        elif not should_stop:
            try:
                idle_check()
            except Exception:
                _log.warning("[idle-release] check failed", exc_info=True)
            try:
                residency_keepalive.idle_tick()
            except Exception:
                _log.warning("[residency] keep-alive tick failed",
                             exc_info=True)
        return pending, should_stop

    _collect_pending_requests.__dict__[_INSTALLED_FLAG] = True
    _collect_pending_requests.__wrapped__ = original  # type: ignore[attr-defined]
    cls._collect_pending_requests = _collect_pending_requests
