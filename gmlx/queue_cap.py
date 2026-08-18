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
live generator so the check can read its pending list.

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

# Weakref to the live BatchGenerator, published by the census wrapper.
_GEN_REF = None


def queue_cap_stats() -> dict:
    return {"rejections": _REJECTIONS,
            "last_reject_reason": _LAST_REJECT or None}


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
    depth = 0
    qsize = getattr(getattr(rg, "requests", None), "qsize", None)
    if callable(qsize):
        try:
            depth += max(0, int(qsize()))
        except Exception:
            pass
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
        rg = getattr(runtime, "response_generator", None)
        if rg is None:
            return None
        depth = _waiting_depth(rg)
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
