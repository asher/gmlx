"""Guarded eval calls for catch-and-continue sites (guard.eval).

A handler that catches an error from an eval and continues on the same
stream is running the exact schedule the stream-poisoning measurements
corrupt: the failed tape's stale kernels sit in the stream's open
command buffer, and the handler's continue (or its own eval) is the
commit that scribbles recycled memory, possibly under another stream's
feet. The fix is one contract, shipped as code rather than prose:
every such site routes its eval through this module, which owns the
eval call, drains in the measured-safe order before any handler logic
runs, and stamps a process-global recovery flag so a staleness clock
can tell a worst-case drain from the hang it exists to detect.

The eval path is bound to the call name (guard.eval vs
guard.async_eval, never a context manager around a bare call) because
the path decides what a failure leaves behind: a sync-path failure is
fully rehabilitated by the drain, while an async-path survivor's
never-signaled event is a landmine that only total discard removes.
Swapping mx.eval for mx.async_eval is a one-token perf edit this
program makes often; binding the path to the call makes that drift
impossible rather than detectable. The regression test in
tests/test_eval_guard.py bans bare mx.eval and mx.async_eval under an
except-carrying try anywhere in gmlx, which is what keeps a sixth
unrouted site from ever shipping.

On failure the guard stamps, drains, and re-raises the original
exception; the call site's own except clause remains the exit and its
existing degrade-gracefully behavior (decline the store, disable the
predictors, skip the probe) stays legitimate because it now runs
after the drain, not instead of it. Contract per path and owner:

  sync + scratch    catch and continue; drop the failed arrays.
  sync + owned      catch and continue; owned survivors (a live
                    prompt cache, a generation's KV) are rehabilitated
                    by the drain and may be retained.
  async + scratch   catch only with TOTAL discard of every array the
                    failed call touched; a retained async survivor
                    can never signal and hangs or aborts a later
                    reader.
  async + owned     do not swallow: escalate to the owning frame.
                    The guard still drains and stamps, but the site
                    must let the re-raise propagate.

The drain, in the measured order (docs/oom-protection-plan.md,
Verified ground truth): commit and synchronize the ambient default
stream first (auto-key and aux subgraphs live there; synchronizing
the thrower first commits an unsatisfiable cross-stream fence wait,
GPU timeout, abort, measured 21/21 on 0.31.2), then synchronize every
registered stream this thread can reach. On mlx 0.32.1 the upstream
unwind handler (#3675) has already signaled events and synchronized
open streams for mid-tape throws, so these syncs double as
verification, and they are also where a stored command-buffer error
from an execution failure resurfaces; secondary errors are recorded
and never mask the original. References to the failed arrays are held
by this frame through the drain, which is what makes the syncs fire
the stale kernels into still-referenced memory instead of the global
recycler.

The stream registry exists because MLX exposes no stream enumeration
to Python and streams are thread-scoped: install_stream_registry()
wraps mx.new_stream to record creations with their owning thread, and
the known upstream streams (device defaults, mlx_lm's
generation_stream) are harvested lazily. A stream owned by another
thread is skipped during the drain, which matches the measured rule
that only the owning thread can drain its streams.

Knobs:
    GMLX_EVAL_GUARD=0           kill switch: guard calls become bare
                                evals (drain, stamp, and registry off)
    GMLX_GUARD_DRAIN_S          recovery-stamp deadline in seconds
                                (default 180; the driver-timeout term
                                dominates the drain's worst case)
"""

from __future__ import annotations

import logging
import os
import threading
import time

import mlx.core as mx

_log = logging.getLogger(__name__)

_INSTALLED_FLAG = "_kq_gguf_stream_registry"

# stream -> owning thread ident (None for harvested); Stream hashes
# and compares by value in the python bindings
_registry: dict = {}
_registry_lock = threading.Lock()
_harvested = False

# The recovery stamp: while a guard is draining, the engine thread is
# inside the handler and the engine progress stamp goes stale for the
# drain's whole worst case; a staleness clock must read this flag or
# it will kill the worker mid-recovery. Read via recovery_state().
_recovery: dict = {}
_recovery_lock = threading.Lock()


def _enabled() -> bool:
    return os.environ.get("GMLX_EVAL_GUARD", "1") != "0"


def _drain_deadline_s() -> float:
    try:
        return float(os.environ.get("GMLX_GUARD_DRAIN_S", "180"))
    except ValueError:
        return 180.0


def register_stream(s, owner: int | None = None) -> None:
    with _registry_lock:
        if s not in _registry:
            _registry[s] = owner


def _harvest() -> None:
    """Record the streams that exist before the wrap could see them:
    the device default streams and mlx_lm's generation stream."""
    global _harvested
    if _harvested:
        return
    _harvested = True
    register_stream(mx.default_stream(mx.Device(mx.cpu)))
    try:
        register_stream(mx.default_stream(mx.Device(mx.gpu)))
    except Exception:
        pass  # no Metal device (CI)
    try:
        from mlx_lm.generate import generation_stream
        register_stream(generation_stream)
    except Exception:
        pass


def install_stream_registry() -> None:
    """Wrap mx.new_stream so every stream created after install is
    registered with its owning thread. Idempotent; harvesting of
    pre-existing streams happens lazily at first guard failure too."""
    if not _enabled():
        return
    if getattr(mx.new_stream, _INSTALLED_FLAG, False):
        return
    _orig = mx.new_stream

    def _recording_new_stream(*a, **kw):
        s = _orig(*a, **kw)
        register_stream(s, threading.get_ident())
        return s

    setattr(_recording_new_stream, _INSTALLED_FLAG, True)
    mx.new_stream = _recording_new_stream
    _harvest()
    _log.info("stream registry installed")


def known_streams() -> list:
    with _registry_lock:
        return list(_registry)


def recovery_state() -> dict | None:
    """The active recovery stamp, or None. Keys: site, path, started,
    deadline (absolute perf_counter seconds). For the staleness clock:
    a stale engine progress stamp is not a hang while this is set and
    now < deadline."""
    with _recovery_lock:
        return dict(_recovery) if _recovery else None


def _stamp(site: str, path: str) -> None:
    now = time.perf_counter()
    with _recovery_lock:
        _recovery.clear()
        _recovery.update(
            site=site, path=path, started=now,
            deadline=now + _drain_deadline_s())


def _clear_stamp() -> None:
    with _recovery_lock:
        _recovery.clear()


def _drain(site: str) -> None:
    """Peer-first ordered drain over the registry, secondary errors
    recorded but never raised (they must not mask the original)."""
    _harvest()
    me = threading.get_ident()
    try:
        mx.synchronize()  # ambient default stream first, measured order
    except Exception as e:
        _log.warning("[guard:%s] default-stream drain: %s: %s",
                     site, type(e).__name__, e)
    with _registry_lock:
        entries = list(_registry.items())
    for s, owner in entries:
        if owner is not None and owner != me:
            continue  # another thread's stream; only its owner can drain
        try:
            mx.synchronize(s)
        except Exception as e:
            _log.warning("[guard:%s] drain of %r: %s: %s",
                         site, s, type(e).__name__, e)


def _guarded(path: str, arrays, site: str, owner: str):
    if owner not in ("scratch", "owned"):
        raise ValueError(f"owner must be scratch|owned, got {owner!r}")
    do_eval = mx.eval if path == "sync" else mx.async_eval
    if not _enabled():
        return do_eval(*arrays)
    try:
        return do_eval(*arrays)
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        _stamp(site, path)
        try:
            _drain(site)
            if path == "async":
                _log.warning(
                    "[guard:%s] async-path eval failed (owner=%s): %s",
                    site, owner,
                    "site must TOTALLY discard the failed arrays"
                    if owner == "scratch"
                    else "escalating to the owning frame; do not swallow")
            else:
                _log.warning("[guard:%s] eval failed; drained, "
                             "re-raising to the site handler", site)
        finally:
            _clear_stamp()
        raise


def drain_for(site: str) -> None:
    """Stamp, drain in the measured order, clear: for a handler that
    caught a memory error around something other than a bare eval (the
    engine tick). The stamp keeps the staleness clock honest for the
    drain's duration exactly as the guarded evals do."""
    if not _enabled():
        return
    _stamp(site, "sync")
    try:
        _drain(site)
    finally:
        _clear_stamp()


def guard_eval(*arrays, site: str, owner: str) -> None:
    """mx.eval with the drain-stamp-reraise contract (sync path)."""
    return _guarded("sync", arrays, site, owner)


def guard_async_eval(*arrays, site: str, owner: str) -> None:
    """mx.async_eval with the drain-stamp-reraise contract (async
    path). owner="owned" sites must let the re-raise propagate."""
    return _guarded("async", arrays, site, owner)


class _Guard:
    eval = staticmethod(guard_eval)
    async_eval = staticmethod(guard_async_eval)


guard = _Guard()
