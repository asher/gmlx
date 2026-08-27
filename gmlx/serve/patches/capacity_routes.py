"""Capacity-facing routes: a keyless readiness probe on ``/health``, a
Prometheus rendering of ``/metrics``, and a per-model ``/v1/cache/reset``.

The routes a load balancer or a harness dispatcher reaches for first:

* ``GET /health?ready=1`` answers 200 or 503 (+ ``Retry-After``) from the
  same numbers the admission path uses (governor band, decode width vs
  in-flight, waiting census). It stays on the auth-exempt route (a load
  balancer polls it keyless) and adds ``ready``, a one-word ``reason``
  (``pressure`` / ``queue`` / ``busy``) and a ``Retry-After`` derived from
  the recent decode rate. That is deliberately more than the liveness
  body says - a coarse busy/not-busy plus a rough throughput hint - and
  it is all an unauthenticated caller can learn here; the numbers behind
  it stay on the authed ``/metrics``.
* ``GET /metrics`` (and ``/v1/metrics``) render Prometheus text when the
  client asks for it (``?format=prometheus`` or an ``Accept`` header
  naming ``text/plain`` / OpenMetrics). The JSON body is unchanged for
  every other caller; the text is a flattening of that same JSON.
* ``POST /v1/cache/reset`` takes an optional ``{"model": "<id>"}`` and
  clears that resident model's prefix cache; with no body it clears every
  resident model's, not just the request context's (which is what the
  stock handler reaches through ``runtime.apc_manager``).
* ``GET /v1/capacity/plan?width=W&depth=D`` and ``POST /v1/estimate`` (or
  ``"dry_run": true`` on chat completions) answer the questions a
  dispatcher asks before sending: does this fan-out fit, may it start
  now, does this prompt fit, how warm is its prefix, how long to first
  token (``gmlx.serve.estimate``).
"""
from __future__ import annotations

import importlib
import inspect
import logging
import math
import os
import re

from fastapi import Request

from ._common import (_CHAT_PATHS, _find_route, _get_pool, _remove_routes,
                      _wrap_post_routes)
import gmlx.serve.bridge_vlm as serving

_log = logging.getLogger(__name__)

_READY_FLAG = "_kq_gguf_health_readiness"
_PROM_FLAG = "_kq_gguf_metrics_prometheus"
_RESET_FLAG = "_kq_gguf_cache_reset_scoped"
_PLAN_FLAG = "_kq_gguf_capacity_plan"
_ESTIMATE_FLAG = "_kq_gguf_estimate"
_DRY_RUN_FLAG = "_kq_gguf_chat_dry_run"

_BAND_LEVEL = {"green": 0, "yellow": 1, "orange": 2, "red": 3}


# readiness
def readiness() -> tuple:
    """``(ready, reason, retry_after_s)`` from the live admission numbers.

    Not ready when the governor is orange or red (``pressure``), when
    requests are already waiting for a slot (``queue``), or when every
    decode slot is generating (``busy``). Yellow still admits (the server
    throttles rather than refuses there). Any probe failure reads ready:
    a readiness check must never be the thing that takes a healthy
    server out of rotation."""
    try:
        from ..governor import governor_stats
        from ..queue_cap import _RETRY_MIN_S, _retry_after_s, concurrency_stats

        band = governor_stats().get("band")
        if band in ("orange", "red"):
            return False, "pressure", _RETRY_MIN_S
        conc = concurrency_stats()
        waiting = conc.get("waiting")
        if isinstance(waiting, int) and waiting > 0:
            runtime = importlib.import_module("mlx_vlm.server.runtime").runtime
            return False, "queue", _retry_after_s(
                getattr(runtime, "metrics", None), waiting)
        in_flight = conc.get("in_flight")
        width = conc.get("decode_batch")
        if (isinstance(in_flight, int) and isinstance(width, int)
                and width > 0 and in_flight >= width
                and not _any_model_has_headroom(width)):
            return False, "busy", _RETRY_MIN_S
    except Exception:
        _log.debug("readiness probe failed; reporting ready", exc_info=True)
    return True, "ok", 0


def _any_model_has_headroom(width: int) -> bool:
    """Each resident model decodes on its own engine with its own width,
    so the server is only ``busy`` when every one of them is at width.
    Without a pool the summed in_flight is the single engine's."""
    try:
        pkg = importlib.import_module("mlx_vlm.server")
        pool = getattr(pkg, "_kq_residency_pool", None)
        if pool is None:
            return False
        per = [int(e.get("in_flight", e.get("busy")) or 0)
               for e in pool.stats().get("resident", [])]
        return bool(per) and any(x < width for x in per)
    except Exception:
        return False


def install_health_readiness() -> None:
    """Add ``?ready=1`` to the liveness-only ``/health``. Without the
    query the body is exactly the liveness override's (``status``,
    ``pid``). Idempotent; installs after the liveness override so it
    replaces that handler rather than the stock one."""
    from fastapi.responses import JSONResponse

    app = importlib.import_module("mlx_vlm.server.app").app
    route = _find_route(app, "/health", "GET")
    if route is not None and getattr(route.endpoint, _READY_FLAG, False):
        return

    async def health_endpoint(request: Request):
        body = {"status": "healthy", "pid": os.getpid()}
        flag = (request.query_params.get("ready") or "").strip().lower()
        if flag not in ("1", "true", "yes"):
            return body
        ready, reason, retry = readiness()
        body["ready"] = ready
        body["reason"] = reason
        if ready:
            return body
        return JSONResponse(status_code=503, content=body,
                            headers={"Retry-After": str(int(retry))})

    health_endpoint.__dict__[_READY_FLAG] = True
    _remove_routes(app, "/health")
    app.add_api_route("/health", health_endpoint, methods=["GET"],
                      include_in_schema=False)


# prometheus
def _metric_name(parts) -> str:
    raw = "_".join(str(p) for p in parts)
    raw = re.sub(r"[^a-zA-Z0-9_]", "_", raw).strip("_").lower()
    return "gmlx_" + re.sub(r"_+", "_", raw)


def _label_str(labels: dict) -> str:
    if not labels:
        return ""
    inner = ",".join(
        f'{k}="{str(v).replace(chr(92), chr(92) * 2).replace(chr(34), chr(92) + chr(34))}"'
        for k, v in sorted(labels.items()))
    return "{" + inner + "}"


def _entry_label(entry: dict, index: int) -> dict:
    # A resident entry also carries its ``profile`` (adapter / load
    # signature, ``residency.profile_label``): two entries can back the same
    # GGUF and would otherwise share a label set - and Prometheus drops the
    # whole scrape on a duplicate sample, not just the row. The label is
    # fixed for the entry's lifetime (the pool's ``seq`` is not: it is the
    # LRU clock, and a label that moved per acquire would mint a new series
    # on every scrape).
    extra = {}
    profile = entry.get("profile")
    if isinstance(profile, str) and profile:
        extra["profile"] = profile
    # ``model`` is the configured id that built the entry when known (two
    # ids on one GGUF with different load params are two entries; naming
    # each by its own id reads better than ids[0] of the path), else the
    # path's first id.
    loaded_as = entry.get("loaded_as")
    if isinstance(loaded_as, str) and loaded_as:
        return {"model": loaded_as, **extra}
    ids = entry.get("ids")
    if isinstance(ids, list) and ids:
        return {"model": str(ids[0]), **extra}
    for key in ("id", "model"):
        if isinstance(entry.get(key), str):
            return {"model": entry[key], **extra}
    path = entry.get("model_path")
    if isinstance(path, str) and path:
        return {"model": os.path.basename(path), **extra}
    return {"index": str(index), **extra}


_SKIP_LISTS = ("recent", "requests", "preload_failures", "available_models")
_SKIP_KEYS = ("latest",)


def flatten_prometheus(payload: dict) -> str:
    """Render the metrics JSON as Prometheus text (v0.0.4).

    Numbers and booleans become gauges named by their JSON path under a
    ``gmlx_`` prefix (the ``server`` level is dropped from the name).
    ``*_total`` names are typed counter. Lists of objects (resident
    models) become one series per entry with a ``model`` label; tables
    keyed by width/depth (``capacity.max_ctx``) become a labelled series.
    ``governor.band`` is emitted both as a labelled indicator and as a
    numeric level (green 0 .. red 3). High-cardinality lists (``recent``,
    ``requests``) contribute only their length. Strings are otherwise
    skipped; the JSON stays the full-fidelity form."""
    series: dict = {}

    def emit(parts, value, labels=None):
        name = _metric_name(parts)
        series.setdefault(name, []).append((labels or {}, value))

    def walk(obj, parts, labels):
        if isinstance(obj, bool):
            emit(parts, 1 if obj else 0, labels)
        elif isinstance(obj, (int, float)):
            if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
                return
            emit(parts, obj, labels)
        elif isinstance(obj, str):
            if parts and parts[-1] == "band":
                emit(parts, 1, dict(labels, band=obj))
                if obj in _BAND_LEVEL:
                    emit(parts + ["level"], _BAND_LEVEL[obj], labels)
        elif isinstance(obj, dict):
            keys = list(obj.keys())
            if keys and all(str(k).lstrip("-").isdigit() for k in keys):
                axis = ("width" if parts and parts[-1] == "max_ctx"
                        else "depth" if parts and "depth" in str(parts[-1])
                        else "key")
                for k, v in obj.items():
                    if isinstance(v, (int, float)) and not isinstance(v, bool):
                        emit(parts, v, dict(labels, **{axis: str(k)}))
                return
            for k, v in obj.items():
                if k in _SKIP_KEYS:
                    continue
                walk(v, parts + [k], labels)
        elif isinstance(obj, list):
            if parts and parts[-1] in _SKIP_LISTS:
                emit(parts + ["count"], len(obj), labels)
                return
            if obj and all(isinstance(e, dict) for e in obj):
                for i, e in enumerate(obj):
                    walk(e, parts, dict(labels, **_entry_label(e, i)))
            # lists of scalars carry no stable identity; skipped

    for key, value in payload.items():
        if key == "server" and isinstance(value, dict):
            for k, v in value.items():
                if k in _SKIP_KEYS:
                    continue
                walk(v, [k], {})
        elif key not in _SKIP_KEYS:
            walk(value, [key], {})

    lines = []
    for name in sorted(series):
        kind = "counter" if name.endswith("_total") else "gauge"
        lines.append(f"# TYPE {name} {kind}")
        for labels, value in series[name]:
            if isinstance(value, float):
                text = repr(value) if value != int(value) else str(int(value))
            else:
                text = str(value)
            lines.append(f"{name}{_label_str(labels)} {text}")
    return "\n".join(lines) + ("\n" if lines else "")


def wants_prometheus(request) -> bool:
    fmt = (request.query_params.get("format") or "").strip().lower()
    if fmt:
        return fmt in ("prometheus", "text", "openmetrics")
    accept = (request.headers.get("accept") or "").lower()
    return ("text/plain" in accept or "openmetrics" in accept) \
        and "application/json" not in accept.split(",")[0]


def install_metrics_prometheus() -> None:
    """Content-negotiate ``/metrics`` and ``/v1/metrics``. The stock handler
    (auth included) still produces the payload; this only decides the
    rendering. Idempotent."""
    from fastapi.responses import PlainTextResponse

    app = importlib.import_module("mlx_vlm.server.app").app
    for path in ("/metrics", "/v1/metrics"):
        route = _find_route(app, path, "GET")
        if route is None or getattr(route.endpoint, _PROM_FLAG, False):
            continue
        original = route.endpoint

        def _make(original):
            async def metrics_endpoint(request: Request):
                payload = await original(request)
                if not wants_prometheus(request):
                    return payload
                return PlainTextResponse(
                    flatten_prometheus(payload),
                    media_type="text/plain; version=0.0.4; charset=utf-8")
            return metrics_endpoint

        endpoint = _make(original)
        endpoint.__signature__ = inspect.signature(original)
        endpoint.__dict__[_PROM_FLAG] = True
        _remove_routes(app, path)
        app.add_api_route(path, endpoint, methods=["GET"],
                          include_in_schema=(path == "/metrics"))


# cache reset
def install_scoped_cache_reset() -> None:
    """``POST /v1/cache/reset`` (and ``/cache/reset``) with an optional
    ``{"model": "<id>"}`` body. Without a residency pool the stock handler
    runs unchanged. Idempotent."""
    from fastapi.responses import JSONResponse

    app = importlib.import_module("mlx_vlm.server.app").app

    def make_endpoint(original):
        async def cache_reset_endpoint(request: Request):
            pool = _get_pool()
            if pool is None:
                return await original(request)
            app_mod = importlib.import_module("mlx_vlm.server.app")
            app_mod._require_management_api_key(request)
            try:
                body = await request.json()
            except Exception:
                body = None
            model_id = (body or {}).get("model") if isinstance(body, dict) else None
            path = None
            if model_id:
                try:
                    path, _spec = serving.resolve_request_model(model_id)
                except (KeyError, serving.ModelFileMissing):
                    return JSONResponse(status_code=404, content={
                        "status": "unknown_model", "model": model_id})
            managers = pool.apc_managers(path)
            if model_id and not managers:
                return JSONResponse(status_code=404, content={
                    "status": "not_resident", "model": model_id})
            path_to_ids = getattr(serving, "_PATH_TO_IDS", {})
            cleared = []
            for mpath, mgr in managers:
                try:
                    mgr.clear()
                except Exception:
                    _log.warning("cache reset failed for %s", mpath,
                                 exc_info=True)
                    continue
                ids = path_to_ids.get(mpath) or [os.path.basename(mpath)]
                cleared.append(ids[0])
            return {"enabled": bool(managers),
                    "status": "cleared" if managers else "no_cache",
                    "models": cleared}
        return cache_reset_endpoint

    _wrap_post_routes(app, ("/v1/cache/reset", "/cache/reset"), _RESET_FLAG,
                      make_endpoint)


# capacity plan
def install_capacity_plan() -> None:
    """``GET /v1/capacity/plan?width=W&depth=D`` (and ``/capacity/plan``):
    the fan-out policy answered from the capacity table, the governor
    band and the live concurrency (``gmlx.serve.estimate.capacity_plan``).
    Authed like the metrics routes. Idempotent."""
    from fastapi.responses import JSONResponse

    app = importlib.import_module("mlx_vlm.server.app").app

    async def capacity_plan_endpoint(request: Request):
        app_mod = importlib.import_module("mlx_vlm.server.app")
        app_mod._require_management_api_key(request)
        q = request.query_params
        try:
            width = int(q.get("width", 1))
            depth = int(q.get("depth", 0))
        except (TypeError, ValueError):
            return JSONResponse(status_code=400, content={"error": {
                "message": "width and depth must be integers",
                "type": "invalid_request_error"}})
        if width < 1 or depth < 0:
            return JSONResponse(status_code=400, content={"error": {
                "message": "width >= 1 and depth >= 0 required",
                "type": "invalid_request_error"}})
        from ..estimate import capacity_plan

        return capacity_plan(width, depth)

    capacity_plan_endpoint.__dict__[_PLAN_FLAG] = True
    for path in ("/v1/capacity/plan", "/capacity/plan"):
        route = _find_route(app, path, "GET")
        if route is not None and getattr(route.endpoint, _PLAN_FLAG, False):
            continue
        _remove_routes(app, path)
        app.add_api_route(path, capacity_plan_endpoint, methods=["GET"],
                          include_in_schema=(path == "/v1/capacity/plan"))


# dry-run admission
def _tenant_of(request) -> str | None:
    try:
        app_mod = importlib.import_module("mlx_vlm.server.app")
        fn = getattr(app_mod, "_read_tenant_id", None)
        return fn(request) if callable(fn) else None
    except Exception:
        return None


def install_estimate() -> None:
    """``POST /v1/estimate`` (and ``/estimate``): dry-run a chat-completions
    body (``gmlx.serve.estimate.estimate_request``), and ``"dry_run": true`` on
    the chat-completions routes, which answers the same estimate instead
    of generating. Idempotent."""
    from fastapi.responses import JSONResponse
    from starlette.concurrency import run_in_threadpool

    app = importlib.import_module("mlx_vlm.server.app").app

    async def estimate_endpoint(request: Request):
        app_mod = importlib.import_module("mlx_vlm.server.app")
        app_mod._require_management_api_key(request)
        try:
            body = await request.json()
        except Exception:
            body = None
        if not isinstance(body, dict):
            return JSONResponse(status_code=400, content={"error": {
                "message": "JSON body required", "type": "invalid_request_error"}})
        from ..estimate import estimate_request

        status, payload = await run_in_threadpool(
            estimate_request, body, tenant_id=_tenant_of(request))
        return JSONResponse(status_code=status, content=payload)

    estimate_endpoint.__dict__[_ESTIMATE_FLAG] = True
    for path in ("/v1/estimate", "/estimate"):
        route = _find_route(app, path, "POST")
        if route is not None and getattr(route.endpoint, _ESTIMATE_FLAG, False):
            continue
        _remove_routes(app, path)
        app.add_api_route(path, estimate_endpoint, methods=["POST"],
                          include_in_schema=(path == "/v1/estimate"))

    def make_endpoint(original):
        async def chat_endpoint(*args, **kwargs):
            req = kwargs.get("request")
            if req is None and args:
                req = args[0]
            flag = getattr(req, "dry_run", None)
            if flag is None and hasattr(req, "model_extra"):
                flag = (req.model_extra or {}).get("dry_run")
            if not flag:
                return await original(*args, **kwargs)
            http = kwargs.get("http_request")
            body = req.model_dump(exclude_none=True) if hasattr(req, "model_dump") else {}
            body.update(getattr(req, "model_extra", None) or {})
            body.pop("dry_run", None)
            from ..estimate import estimate_request

            status, payload = await run_in_threadpool(
                estimate_request, body,
                tenant_id=_tenant_of(http) if http is not None else None)
            return JSONResponse(status_code=status, content=payload)
        return chat_endpoint

    _wrap_post_routes(app, _CHAT_PATHS, _DRY_RUN_FLAG, make_endpoint)
