"""Workaround for the mlx 0.32.1 exit-time segfault.

mlx 0.32.1 (upstream commit 8e00a2d9d, PR 4248) dropped the exit-time
compile-cache cleanup: any mx.compile wrapper still referenced when the
interpreter shuts down leaves a cache entry whose Python objects are
decref'd by the dyld thread-local terminator after Py_Finalize, killing
the process with SIGSEGV and destroying the real exit code.

The guard here skips interpreter finalization on otherwise-clean exits:
arm() registers an atexit hook at entrypoint start, so it runs after
every hook the app registers later (atexit is LIFO); the hook flushes
the std streams and calls os._exit with the recorded exit code. It
fires only when the entrypoint recorded a code via set_code, so
abnormal teardowns (uncaught exceptions) keep stock behavior and are
never masked. Armed only in gmlx-owned processes, never on library
import. GMLX_EXITFIX=0 disables it; the mlx version gate retires it
automatically once a fixed release is pinned.
"""

from __future__ import annotations

import atexit
import os
import sys

_AFFECTED_MLX = ("0.32.1",)

_code: int | None = None
_armed = False


def affected() -> bool:
    """True when the running process is exposed to the exit segfault."""
    if os.environ.get("GMLX_EXITFIX") == "0":
        return False
    mx = sys.modules.get("mlx.core")
    if mx is None:
        # mlx never imported: no compile cache exists, nothing to guard.
        return False
    return getattr(mx, "__version__", "") in _AFFECTED_MLX


def arm() -> None:
    """Register the exit guard. Call first thing in a process entrypoint."""
    global _armed
    if _armed:
        return
    _armed = True
    atexit.register(_guard)


def set_code(rc: object) -> None:
    """Record the process exit code the guard should preserve."""
    global _code
    if isinstance(rc, bool) or not isinstance(rc, int):
        _code = 0 if rc is None else 1
    else:
        _code = rc


def guarded(fn, *args) -> int:
    """Run a process entrypoint under the exit guard; returns the exit code."""
    arm()
    try:
        rc = fn(*args)
    except SystemExit as e:
        rc = e.code
        if rc is not None and not isinstance(rc, int):
            print(rc, file=sys.stderr)
            rc = 1
    except KeyboardInterrupt:
        rc = 130
    set_code(rc)
    if isinstance(rc, bool) or not isinstance(rc, int):
        return 0 if rc is None else 1
    return rc


def _guard() -> None:
    if _code is None or not affected():
        return
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:
        pass
    os._exit(_code)
