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


# Thread-exit variant of the same upstream bug. The dropped cleanup also
# fires at *worker thread* exit: a thread that ever ran an mx.compile
# wrapper owns thread-local CompileCache entries, and pthread TSD
# finalization destructs them without the GIL. On a live server this is
# the engine thread mlx-vlm's ResponseGenerator joins at model unload;
# the idle-TTL reap of a 104 GB model crashed exactly there (tupledealloc
# under dyld ThreadLocalVariables::finalizeList, 2026-08-23 .ips). The
# guard parks the engine thread on a never-set Event instead of letting
# it return: the cache entries stay alive with the thread, and the parked
# frame drops every object reference first so the unloaded model's
# weights still free. Parked threads are daemonic and cost one idle
# stack apiece; unloads are rare (TTL reaps, model swaps).

_PARKED_THREADS = 0


def thread_guard_affected() -> bool:
    """True when engine-thread exits are exposed to the TSD segfault."""
    if os.environ.get("GMLX_THREADFIX") == "0":
        return False
    mx = sys.modules.get("mlx.core")
    if mx is None:
        return False
    return getattr(mx, "__version__", "") in _AFFECTED_MLX


def install_engine_thread_guard(generation_module) -> None:
    """Patch ResponseGenerator so its engine thread parks instead of
    exiting. Idempotent; a no-op when the running mlx is not affected
    (the version gate is evaluated per stop, so import order is free)."""
    rg = getattr(generation_module, "ResponseGenerator", None)
    if rg is None or getattr(rg, "_gmlx_thread_guard", False):
        return
    rg._gmlx_thread_guard = True
    orig_run = rg._run
    orig_stop = rg.stop_and_join

    def _drained_event(self):
        import threading
        # setdefault is atomic under the GIL: whichever of _run /
        # stop_and_join gets here first creates the event both share.
        return self.__dict__.setdefault("_gmlx_drained", threading.Event())

    def _run(self):
        import threading

        from . import _exitfix as _ef
        try:
            orig_run(self)
        except Exception:
            # The stock loop treats a crash as thread death; keep that
            # visible but still park below, and drop the traceback's
            # frame refs by leaving the handler.
            import logging
            logging.getLogger(__name__).warning(
                "engine thread crashed; parking anyway", exc_info=True)
        evt = _drained_event(self)
        if not thread_guard_affected():
            evt.set()
            return
        # The generator instance cannot be released while parked:
        # Thread.run()'s evaluation stack pins the bound _run method for
        # the duration of the call. Empty the instance instead, so the
        # parked skeleton holds no model, cache, or queue references.
        # It is stop_and_join'd and about to be discarded either way.
        evt.set()
        del evt
        self.__dict__.clear()
        self = None  # noqa: F841
        _ef._PARKED_THREADS += 1
        import logging
        logging.getLogger(__name__).info(
            "engine thread parked (mlx compile-cache thread-exit guard, "
            "%d parked)", _ef._PARKED_THREADS)
        threading.Event().wait()

    def stop_and_join(self):
        if not thread_guard_affected():
            return orig_stop(self)
        self._stop = True
        self.requests.put(None)
        # join() would wait on a thread that never exits; the drained
        # event marks the same milestone (loop done, refs dropped).
        _drained_event(self).wait(timeout=5.0)

    rg._run = _run
    rg.stop_and_join = stop_and_join
