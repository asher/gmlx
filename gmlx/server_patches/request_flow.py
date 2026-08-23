"""Request-flow patches on the chat routes: off-loop model
pre-warm, SSE keepalive during silent prefill, and request-scoped model /
profile capture."""

from __future__ import annotations

import importlib


from .. import server_bridge_vlm as serving
from ._common import (
    _CHAT_PATHS,
    _wrap_post_routes,
)
from . import routes as _routes


# Off-loop model load - keep /health (and siblings) responsive during a load
_LOAD_OFFLOAD_FLAG = "_kq_gguf_load_offload"

_LOAD_OFFLOAD_PATHS = _CHAT_PATHS + ("/responses", "/v1/responses",
                                     "/messages", "/v1/messages",
                                     "/completions", "/v1/completions")


async def _extract_request_model(values, inherit):
    """The body ``model`` (+ the stock handler's adapter resolution: an explicit
    ``adapter_path`` wins, otherwise inherit the resident adapter) from a
    wrapped endpoint's arguments. A parsed pydantic request is read directly; a
    raw-``Request`` route (anthropic / responses) reads the JSON body - safe,
    starlette caches ``_body`` so the stock handler's own parse still works.
    Returns ``(model_id, adapter)``; ``model_id=None`` means skip the warm."""
    parsed = [o for o in values if hasattr(o, "model_fields_set")]
    for obj in parsed:
        model_id = getattr(obj, "model", None)
        if model_id is None:
            continue
        adapter = inherit
        if "adapter_path" in obj.model_fields_set:
            adapter = getattr(obj, "adapter_path", inherit)
        return model_id, adapter
    if parsed:
        return None, inherit              # parsed body seen; no model sent
    for obj in values:
        if hasattr(obj, "receive") and hasattr(obj, "json"):   # starlette Request
            try:
                body = await obj.json()
            except Exception:
                return None, inherit
            if isinstance(body, dict) and isinstance(body.get("model"), str):
                adapter = body["adapter_path"] if "adapter_path" in body else inherit
                return body["model"], adapter
    return None, inherit


def install_chat_load_offload() -> None:
    """Wrap the model-serving routes (chat / anthropic / responses) to resolve +
    load the request's model off the event loop before the stock handler runs.

    mlx-vlm calls ``get_cached_model`` synchronously inside the async handlers,
    so a cold first load - or a model swap - runs the multi-second build on the
    single uvicorn event loop and head-of-line-blocks every other request,
    ``/health`` included, until it finishes (the menu bar then flaps to "down"
    and back, though the process never restarts). Decode is already offloaded
    by mlx-vlm (``asyncio.to_thread``); this closes the one remaining on-loop
    blocking call: pre-warm the model on a worker thread, after which the stock
    handler's own ``get_cached_model`` is a fast in-memory cache hit. A warm
    failure is swallowed - the stock handler re-resolves and surfaces the real
    error (404/400/500) normally. Idempotent per route.

    Benign race: if the warm is evicted (memory pressure) before the stock handler
    re-acquires, that handler reloads on the loop - i.e. degrades to today's
    behaviour only under concurrent pressure, never worse."""
    import asyncio

    app_mod = importlib.import_module("mlx_vlm.server.app")
    app = app_mod.app
    inherit = app_mod._INHERIT_ADAPTER

    def _make(original):
        async def endpoint(*args, **kwargs):
            model_id, adapter = await _extract_request_model(
                list(args) + list(kwargs.values()), inherit)
            if model_id is not None:
                try:
                    await asyncio.to_thread(_routes._warm_and_release, model_id, adapter)
                except Exception:
                    pass        # stock handler re-resolves + surfaces errors
            return await original(*args, **kwargs)
        return endpoint

    _wrap_post_routes(app, _LOAD_OFFLOAD_PATHS, _LOAD_OFFLOAD_FLAG, _make)


# 8b2. SSE keepalive - a deep-context dense prefill can exceed 10 minutes with
# zero bytes on the wire (the stream generator blocks on the first token), so
# any client with a between-bytes read timeout tears the socket down
# mid-prefill. SSE comment lines are legal, invisible to event parsers, and
# reset such timeouts.
_KEEPALIVE_FLAG = "_kq_gguf_sse_keepalive"


def _upgrade_shed_chunk(item, json_mod):
    """The upstream stream handlers swallow a mid-stream RowShedError
    into a plain ``data: {"error": str}`` event; recognize ours by the
    self-owned marker in the message and upgrade it to the contract's
    terminal shape (typed error object plus finish_reason "shed").
    Returns the upgraded SSE string, or None to pass the chunk through."""
    from .row_failed import SHED_MARKER

    try:
        text = item.decode() if isinstance(item, bytes) else item
        if not isinstance(text, str) or not text.startswith("data: {"):
            return None
        if SHED_MARKER not in text:
            return None
        payload = json_mod.loads(text[len("data: "):].strip())
    except Exception:
        return None
    msg = payload.get("error")
    if not isinstance(msg, str) or SHED_MARKER not in msg:
        return None
    return "data: " + json_mod.dumps({
        "error": {"message": msg, "type": "server_overloaded_shed",
                  "code": "row_shed"},
        "finish_reason": "shed",
    }) + "\n\n"


async def _keepalive_sse(body, interval: float | None):
    """Yield ``body``'s chunks unchanged, inserting a ``: keepalive`` SSE
    comment whenever the upstream is silent for ``interval`` seconds
    (``None`` disables the keepalive but keeps the error translation). A
    pump task feeds a queue so the timeout never cancels the upstream
    generator; closing this generator cancels the pump and closes
    ``body``, preserving the disconnect path (stream finally ->
    token_iter.close -> batch cancel).

    A ``RowShedError`` surfacing mid-stream becomes a terminal SSE error
    event with its own finish_reason and a clean close: by then the
    headers are long gone, and the client must be able to tell a
    governor shed from both a normal stop and a transport fault (a
    dropped connection means worker death and has its own retry rule)."""
    import asyncio
    import json

    queue: "asyncio.Queue" = asyncio.Queue()

    async def _pump():
        try:
            async for chunk in body:
                await queue.put(("chunk", chunk))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await queue.put(("error", exc))
        else:
            await queue.put(("done", None))

    task = asyncio.create_task(_pump())
    try:
        while True:
            try:
                kind, item = await asyncio.wait_for(queue.get(), interval)
            except asyncio.TimeoutError:
                yield ": keepalive\n\n"
                continue
            if kind == "chunk":
                upgraded = _upgrade_shed_chunk(item, json)
                if upgraded is not None:
                    yield upgraded
                    yield "data: [DONE]\n\n"
                    return
                yield item
            elif kind == "error":
                from .row_failed import RowShedError

                if isinstance(item, RowShedError):
                    payload = {
                        "error": {
                            "message": str(item),
                            "type": "server_overloaded_shed",
                            "code": "row_shed",
                            **{k: item.info[k]
                               for k in ("prompt_len", "delivered")
                               if item.info.get(k) is not None},
                        },
                        "finish_reason": "shed",
                    }
                    yield f"data: {json.dumps(payload)}\n\n"
                    yield "data: [DONE]\n\n"
                    return
                raise item
            else:
                return
    finally:
        task.cancel()
        try:
            await task
        except BaseException:
            pass
        aclose = getattr(body, "aclose", None)
        if aclose is not None:
            try:
                await aclose()
            except Exception:
                pass


def install_sse_keepalive() -> None:
    """Wrap the streaming POST routes so SSE comment keepalives flow while the
    engine produces no bytes. Interval from ``GMLX_SSE_KEEPALIVE_S``
    (seconds, default 15, <=0 disables), read per request. Must install after
    every other route patch (stop filter, assistant re-registration) so the
    keepalive wrapper is outermost. Idempotent per route."""
    from starlette.responses import StreamingResponse

    from ..envflags import env_float

    app = importlib.import_module("mlx_vlm.server.app").app

    def _make(original):
        async def endpoint(*args, **kwargs):
            result = await original(*args, **kwargs)
            interval = env_float("GMLX_SSE_KEEPALIVE_S", 15.0)
            if isinstance(result, StreamingResponse):
                # interval <= 0 disables the keepalive but keeps the
                # shed-error translation: a terminal event beats a
                # dropped connection regardless of keepalive policy.
                result.body_iterator = _keepalive_sse(
                    result.body_iterator,
                    interval if interval > 0 else None)
            return result
        return endpoint

    _wrap_post_routes(app, _LOAD_OFFLOAD_PATHS, _KEEPALIVE_FLAG, _make)


# Request-body `profile` capture
# The residency seam (pooled_get_cached_model) receives only the model string,
# so a body `profile:` field never reached the spec resolution - it shaped stop
# sequences (the stop resolver re-resolves with it) but not sampling, load
# params, or the residency cache key; only inline `id@profile` worked fully.
# Capture the field into a request-scoped ContextVar at the outermost route
# wrapper instead. Installed after the load offload so it wraps outside it:
# `asyncio.to_thread` copies contextvars, so the pre-warm resolves with the
# same profile (a load-affecting profile would otherwise pre-warm the wrong
# resident entry). Preload / TTL / warm threads have no request context and
# see the default None - unchanged behaviour.
_PROFILE_CAPTURE_FLAG = "_kq_gguf_request_profile_capture"


_PROFILE_CAPTURE_PATHS = _CHAT_PATHS + ("/messages", "/v1/messages",
                                        "/responses", "/v1/responses",
                                        "/completions", "/v1/completions")


async def _extract_request_profile(values) -> str | None:
    """The body ``profile`` from a wrapped endpoint's arguments. A parsed
    pydantic request (schemas are ``extra="allow"``) is checked directly; a
    raw-``Request`` route (anthropic / responses) reads the JSON body - safe,
    starlette caches ``_body`` so the stock handler's own parse still works."""
    parsed = [o for o in values if hasattr(o, "model_fields_set")]
    for obj in parsed:
        p = getattr(obj, "profile", None)
        if isinstance(p, str) and p:
            return p
    if parsed:
        return None                       # parsed body seen; no profile sent
    for obj in values:
        if hasattr(obj, "receive") and hasattr(obj, "json"):   # starlette Request
            try:
                body = await obj.json()
            except Exception:
                return None
            if isinstance(body, dict):
                p = body.get("profile")
                if isinstance(p, str) and p:
                    return p
    return None


def install_optional_request_model() -> None:
    """Make ``model`` optional (default ``""``) on the request schemas.

    The config scaffold, ``init --help``, and ``list`` all promise
    ``server.defaults.model`` as "the fallback model when a request omits
    `model`", and the resolve seam already treats an empty field exactly that
    way - only the pydantic schema stood in the way, 422ing the omission
    before resolution could run. Idempotent."""
    schemas = importlib.import_module("mlx_vlm.server.schemas")
    for name in ("VLMRequest", "GenerationRequest", "ChatRequest",
                 "AnthropicRequest", "OpenAIRequest"):
        cls = getattr(schemas, name, None)
        field = getattr(cls, "model_fields", {}).get("model") if cls else None
        if field is None or not field.is_required():
            continue
        field.default = ""
        cls.model_rebuild(force=True)


def install_request_profile_capture() -> None:
    """Bind the request body's ``profile`` field into serving's request-scoped
    ContextVar around the chat / anthropic / responses handlers, so the
    residency seam resolves the model spec with it (sampling, load params,
    cache key) - not just the stop resolver. Also rewrites an empty/omitted
    request ``model`` to the resolved default id, so the response echoes the
    model that actually served it rather than ``""``. Idempotent per route."""
    app = importlib.import_module("mlx_vlm.server.app").app

    def _make(original):
        async def endpoint(*args, **kwargs):
            for arg in list(args) + list(kwargs.values()):
                m = getattr(arg, "model", None)
                if m is not None and not str(m).strip():
                    try:
                        arg.model = serving._default_model_id()
                    except Exception:
                        pass   # the resolver raises its typed error below
                    break
            profile = await _extract_request_profile(
                list(args) + list(kwargs.values()))
            token = serving.set_request_profile(profile)
            try:
                # The spec resolution (get_cached_model) happens inside this
                # await, before any StreamingResponse is returned - the
                # reset in `finally` never races the stream body.
                return await original(*args, **kwargs)
            finally:
                serving.reset_request_profile(token)
        return endpoint

    _wrap_post_routes(app, _PROFILE_CAPTURE_PATHS, _PROFILE_CAPTURE_FLAG, _make)
