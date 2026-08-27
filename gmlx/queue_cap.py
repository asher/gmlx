"""Queue depth cap with Retry-After (GMLX_QUEUE_DEPTH_CAP).

The token queue is bounded only by its timeout. Under a fan-out burst
beyond capacity, excess requests hold sockets for up to half an hour
with no tokens and race client-side timeouts. Harness SDKs handle a 503
with Retry-After by backing off; they handle silent stalls badly.

The cap rejects before enqueue: a generation request that would push the
waiting queue past GMLX_QUEUE_DEPTH_CAP gets an immediate HTTP 503 with
a JSON body naming the cap and current depth, plus a Retry-After header.
The header value is the estimated drain time (queue depth x recent mean
tokens per request / current aggregate decode rate), clamped to [2, 60]
seconds, static 5 when no stats exist yet. Estimation reads existing
metrics counters only; nothing new lands in the token loop.

Depth is the engine's waiting census: requests sitting in the server's
request queue plus prompt candidates the batch generator has not yet
admitted. The metrics in-flight gauge is not a usable source: the
gmlx-owned chat routes never call begin_request, so it reads zero under
any load (the depth e2e queue-cap phase caught exactly this). A tiny
publisher wrapper on ``BatchGenerator._next`` keeps a weakref to the
live generator so the check can read its pending list; the live-request
publisher (which sees the engine and its generator together on every
tick) registers the pair, so with several resident models the check,
the metrics census and readiness all judge the same pool-wide figure:
every resident engine's queue summed, not the last-used one. (Handlers
see the residency proxy's per-request guard around the engine, so the
census unwraps it before the registry lookup; judged through the guard,
the pending list was invisible and the cap never fired with a pool.)

Knobs:
    GMLX_QUEUE_DEPTH_CAP   waiting-queue cap (default 2 x decode
                               concurrency; 0 = off)
"""

from __future__ import annotations

import importlib
import logging
import os
import weakref

_log = logging.getLogger(__name__)

_CAP_FLAG = "_kq_gguf_queue_cap"
_PUB_FLAG = "_kq_gguf_queue_census"
_RETRY_MIN_S = 2
_RETRY_MAX_S = 60
_RETRY_DEFAULT_S = 5

# Server-wide counters for /v1/metrics.
_REJECTIONS = 0
_LAST_REJECT = ""

# Weakref to the live BatchGenerator, published by the census wrapper
# (the most recently ticked one; the fallback when the engine has no
# registered generator).
_GEN_REF = None
# id(response_generator) -> (weakref rg, weakref BatchGenerator), from the
# live-request publisher's tick.
_ENGINES: dict = {}


def note_engine(rg, gen) -> None:
    """Remember which BatchGenerator serves ``rg`` (called per tick; a
    cheap identity check keeps the common case free)."""
    key = id(rg)
    cur = _ENGINES.get(key)
    if cur is not None and cur[1]() is gen and cur[0]() is rg:
        return
    try:
        _ENGINES[key] = (weakref.ref(rg), weakref.ref(gen))
    except TypeError:
        return
    for k, (r, g) in list(_ENGINES.items()):
        if r() is None or g() is None:
            _ENGINES.pop(k, None)


def _all_engines() -> list | None:
    """Every resident engine, or None without a residency pool."""
    try:
        pkg = importlib.import_module("mlx_vlm.server")
        pool = getattr(pkg, "_kq_residency_pool", None)
        if pool is None or not hasattr(pool, "response_generators"):
            return None
        return [rg for _, rg in pool.response_generators()]
    except Exception:
        return None


def _waiting_depth_all() -> int:
    """The waiting census across every resident engine; falls back to the
    runtime's current engine without a pool. 0 with no engine at all."""
    rgs = _all_engines()
    if rgs is None:
        runtime = importlib.import_module("mlx_vlm.server.runtime").runtime
        rg = getattr(runtime, "response_generator", None)
        return _waiting_depth(rg) if rg is not None else 0
    return sum(_waiting_depth(rg) for rg in rgs)


def queue_cap_stats() -> dict:
    """The ``queue`` section of /v1/metrics: the rejection counters plus
    the live waiting census, the cap it is judged against, and the
    drain estimate a client would get as Retry-After right now
    (``eta_s`` is 0 with nothing waiting). Read-only; a probe failure
    leaves the live fields None rather than failing the snapshot."""
    out = {"rejections": _REJECTIONS,
           "last_reject_reason": _LAST_REJECT or None,
           "waiting": None, "cap": _cap(), "eta_s": None}
    try:
        runtime = importlib.import_module("mlx_vlm.server.runtime").runtime
        # No engine yet (nothing loaded): nothing can be waiting.
        depth = _waiting_depth_all()
        out["waiting"] = depth
        out["eta_s"] = (_retry_after_s(getattr(runtime, "metrics", None),
                                       depth) if depth > 0 else 0)
    except Exception:
        pass
    return out


def concurrency_stats() -> dict:
    """The ``concurrency`` section of /v1/metrics: the effective decode
    width (``decode_batch``), the waiting-queue cap, streams generating
    now (``in_flight``, the residency pool's busy refcounts summed) and
    the waiting census. The four numbers a dispatcher needs to size a
    fan-out without guessing at ``GMLX_DECODE_BATCH``."""
    out = {"decode_batch": _decode_concurrency(), "queue_cap": _cap(),
           "in_flight": None, "waiting": None}
    try:
        pkg = importlib.import_module("mlx_vlm.server")
        pool = getattr(pkg, "_kq_residency_pool", None)
        if pool is not None:
            # in_flight excludes process-lifetime holds (the primary
            # preload); older pool stats without it fall back to busy.
            out["in_flight"] = sum(int(e.get("in_flight", e.get("busy")) or 0)
                                   for e in pool.stats().get("resident", []))
    except Exception:
        pass
    try:
        out["waiting"] = _waiting_depth_all()
    except Exception:
        pass
    return out


def _decode_concurrency() -> int:
    try:
        from .decode_batch import decode_batch
        return int(decode_batch())
    except Exception:
        return 8


def _cap() -> int:
    raw = os.environ.get("GMLX_QUEUE_DEPTH_CAP", "")
    if raw:
        try:
            return int(raw)
        except ValueError:
            pass
    return 2 * _decode_concurrency()


def _waiting_depth(rg) -> int:
    """Requests waiting for a decode slot: the server request queue plus
    the batch generator's unadmitted prompt candidates. Both reads are
    racy by design; the cap needs magnitude, not a barrier."""
    # Handlers see the residency proxy's per-request _GenerationGuard,
    # not the entry's ResponseGenerator; the engine registry is keyed by
    # the real object, so unwrap before the identity lookup.
    rg = getattr(rg, "_rg", rg)
    depth = 0
    qsize = getattr(getattr(rg, "requests", None), "qsize", None)
    if callable(qsize):
        try:
            depth += max(0, int(qsize()))
        except Exception:
            pass
    reg = _ENGINES.get(id(rg))
    if reg is not None and reg[0]() is rg:
        gen = reg[1]()
    elif _ENGINES:
        gen = None      # another engine's generator is not this queue
    else:
        gen = _GEN_REF() if _GEN_REF is not None else None
    pending = getattr(gen, "_unprocessed_sequences", None)
    if pending is not None:
        depth += len(pending)
    return depth


def _retry_after_s(metrics, depth: int) -> int:
    done = int(getattr(metrics, "_requests_completed", 0) or 0)
    toks = int(getattr(metrics, "_completion_tokens_total", 0) or 0)
    gen_toks = int(getattr(metrics, "_generated_tokens_total", 0) or 0)
    decode_s = float(getattr(metrics, "_decode_time_total_s", 0.0) or 0.0)
    if done <= 0 or toks <= 0 or gen_toks <= 0 or decode_s <= 0:
        return _RETRY_DEFAULT_S
    mean_tokens = toks / done
    rate = gen_toks / decode_s
    est = depth * mean_tokens / rate
    return int(min(max(est, _RETRY_MIN_S), _RETRY_MAX_S))


def check_queue_depth():
    """Return a 503 JSONResponse when the waiting queue is at the cap,
    else None. Read-only; a probe failure admits."""
    try:
        cap = _cap()
        if cap <= 0:
            return None
        runtime = importlib.import_module("mlx_vlm.server.runtime").runtime
        # Pool-wide census (every resident engine), the same figure
        # /v1/metrics and readiness report; without a pool, the runtime's
        # engine. No engine at all: nothing can be waiting.
        depth = _waiting_depth_all()
        if depth < cap:
            return None
        retry = _retry_after_s(getattr(runtime, "metrics", None), depth)
        global _REJECTIONS, _LAST_REJECT
        _REJECTIONS += 1
        _LAST_REJECT = f"depth {depth} at cap {cap}, retry {retry}s"
        _log.info("queue cap: rejected request (%s)", _LAST_REJECT)
        from fastapi.responses import JSONResponse
        return JSONResponse(
            status_code=503,
            content={"error": {
                "message": (f"server queue is full: {depth} requests "
                            f"waiting, cap {cap}; retry after {retry}s"),
                "type": "server_overloaded",
                "queue_depth": depth,
                "queue_cap": cap,
            }},
            headers={"Retry-After": str(retry)},
        )
    except Exception:
        _log.warning("queue cap probe failed; admitting", exc_info=True)
        return None


def _install_census() -> None:
    """Publish a weakref to the live BatchGenerator from its tick.

    Pure passthrough wrapper; the engine swaps generators across idle
    gaps, so the ref re-publishes whenever the instance changes."""
    from mlx_vlm.generate import ar as _ar

    if getattr(_ar.BatchGenerator._next, _PUB_FLAG, False):
        return
    _orig_next = _ar.BatchGenerator._next

    def _published_next(self, **kwargs):
        global _GEN_REF
        ref = _GEN_REF
        if ref is None or ref() is not self:
            _GEN_REF = weakref.ref(self)
        return _orig_next(self, **kwargs)

    setattr(_published_next, _PUB_FLAG, True)
    _ar.BatchGenerator._next = _published_next


def install_queue_depth_cap() -> None:
    """Wrap the generation POST routes with the queue-depth check.

    The check runs before the stock handler, so a rejected request never
    opens an SSE stream and never reaches the engine. Idempotent per
    route; GMLX_QUEUE_DEPTH_CAP=0 disables at install.
    """
    if _cap() <= 0:
        return
    _install_census()
    from .server_patches._common import _CHAT_PATHS, _wrap_post_routes

    app = importlib.import_module("mlx_vlm.server.app").app
    paths = _CHAT_PATHS + ("/responses", "/v1/responses",
                           "/messages", "/v1/messages",
                           "/completions", "/v1/completions")

    def _make(original):
        async def endpoint(*args, **kwargs):
            rejected = check_queue_depth()
            if rejected is not None:
                return rejected
            return await original(*args, **kwargs)
        return endpoint

    _wrap_post_routes(app, paths, _CAP_FLAG, _make)
    _log.info("queue depth cap installed (cap=%d)", _cap())
