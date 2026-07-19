"""Served assistant aliases: the built-in tool loop behind plain
chat-completions.

``server.assistants`` exposes pseudo-model ids that run requests through
:class:`~gmlx.assistant_brain.AssistantBrain` server-side, so thin
clients (curl, Open WebUI, phone apps) get MCP tools with no client loop.
Routing contract per request to ``/v1/chat/completions``:

- assistant id + no ``tools`` -> the server runs the tool loop and streams
  (or returns) only the prose answer under the alias id. ``tools: []``
  counts as no tools.
- assistant id + a non-empty ``tools`` list -> the client is running its own
  loop; the model id is rewritten to the underlying model and the request
  passes through untouched (the config's tools are not offered).
- any other model id -> untouched.
- ``/v1/responses`` and ``/v1/messages`` (+ token-count sub-routes) reject
  assistant ids with 400; ``/v1/models`` lists them.

Execution is a loopback self-call: each round re-enters this server's own
``/v1/chat/completions`` as a normal client (carrying the server's own
``api_key``), so profiles, stop sequences, spec decode and batching all
apply per round. The loop runs on a dedicated daemon thread per request
(never the event loop, never starlette's threadpool - the inner request
needs both free), handing events to the response through an asyncio.Queue.
Concurrent loops are capped with a try-acquire counter: over cap is an
immediate 429, never a wait (a waiting cap can deadlock when a tool calls
back into an alias).

Tools come only from the config allowlist: the shared ``assistant.mcp`` list
for aliases without their own ``mcp:`` key, a private registry per alias
that scopes one (``mcp: []`` = tool-less loop). Per-request MCP specs do not
exist. Memory (``memory: true``) is one shared store per alias across every
client, in its own db file - never the talk store.

Known limits (documented, accepted): a stop sequence forwards per round and
can in principle truncate an intermediate tool round; a non-streaming
assistant turn cannot be cancelled by client disconnect (it runs its rounds
to completion); stream-path cancellation lands at the next inner delta or
tool boundary.
"""

from __future__ import annotations

import asyncio
import importlib
import inspect
import json
import os
import sys
import threading
import time
import uuid

from .assistant_brain import AssistantBrain, ToolRegistry
from .talk_client import TalkClientError, stream_chat

_ASSISTANT_FLAG = "_kq_gguf_assistant_serve"
_MAX_CONCURRENT_TURNS = 4    # immediate 429 above this, per process
_DEFAULT_MAX_TOKENS = 4096

# Declared ChatRequest fields forwarded verbatim to every inner round when the
# outer request set them. `stop` rides on extra="allow" and is added apart;
# response_format is deliberately excluded (it would force JSON onto tool
# rounds and break tool-call emission).
_SAMPLING_FIELDS = ("temperature", "top_p", "top_k", "min_p", "seed",
                    "repetition_penalty", "repetition_context_size",
                    "presence_penalty", "frequency_penalty", "logit_bias")

_GUARDED_PATHS = ("/responses", "/v1/responses",
                  "/responses/input_tokens", "/v1/responses/input_tokens",
                  "/messages", "/v1/messages",
                  "/messages/count_tokens", "/v1/messages/count_tokens")
_ANTHROPIC_PREFIXES = ("/messages", "/v1/messages")


class _AssistantCancelled(Exception):
    """Raised inside the stream seam when the outer client went away."""


class _AssistantState:
    """Everything the wrappers close over. Built once at install; holds the
    MCP hosts and memory stores alive for the server's lifetime."""

    def __init__(self, cfg, base_url: str):
        self.cfg = cfg
        self.base_url = base_url
        self.api_key = getattr(cfg, "api_key", None)
        self.aliases: dict = {}      # id -> (AssistantAlias, ToolRegistry)
        self.memories: dict = {}     # id -> MemoryStore
        self.hosts: list = []        # McpToolHost keep-alives
        self._active = 0
        self._lock = threading.Lock()

    def close_memories(self) -> None:
        """Drain + close every alias memory store so facts queued behind the
        extraction worker survive a server stop (the CLI paths already close
        via their own atexit; the served path gets the same guarantee)."""
        stores, self.memories = dict(self.memories), {}
        for m in stores.values():
            try:
                m.close()
            except Exception:
                pass

    def try_acquire(self) -> bool:
        with self._lock:
            if self._active >= _MAX_CONCURRENT_TURNS:
                return False
            self._active += 1
            return True

    def make_release_guard(self):
        """A one-shot release for one acquired slot: calling the returned
        callable more than once is a no-op. Lets the stream body's finally and
        the response's __call__ backstop both call it without double-decrementing
        (an immediate client disconnect can skip the body's finally entirely)."""
        lock = self._lock
        done = [False]

        def _release() -> None:
            with lock:
                if done[0]:
                    return
                done[0] = True
                self._active -= 1

        return _release


def _self_base_url(cfg) -> str:
    """The URL the loop calls itself back on. Bind-all is not connectable and
    a loopback bind answers on 127.0.0.1; only an interface-specific bind
    must be dialed as-is (127.0.0.1 would be refused there)."""
    host = getattr(cfg, "host", None) or "127.0.0.1"
    if host in ("0.0.0.0", "::", "127.0.0.1", "localhost"):
        host = "127.0.0.1"
    elif host == "::1":
        host = "[::1]"
    elif ":" in host:                       # non-loopback IPv6 literal
        host = f"[{host}]"
    return f"http://{host}:{cfg.port}/v1"


def _lock_registry(registry: ToolRegistry) -> ToolRegistry:
    """Serialize tool calls within one registry: concurrent assistant turns
    share its live MCP sessions and the SDK's session concurrency is
    unproven. One lock per registry (coarser than per-server, always safe)."""
    lock = threading.Lock()
    for name in registry.names():
        tool = registry.get(name)
        inner = tool.call

        def call(args, _inner=inner):
            with lock:
                return _inner(args)

        tool.call = call
    return registry


def _openai_error(status: int, message: str, err_type: str,
                  code: str | None = None):
    from starlette.responses import JSONResponse
    err = {"message": message, "type": err_type}
    if code:
        err["code"] = code
    return JSONResponse(status_code=status, content={"error": err})


def _anthropic_error(status: int, message: str):
    from starlette.responses import JSONResponse
    return JSONResponse(status_code=status, content={
        "type": "error",
        "error": {"type": "invalid_request_error", "message": message}})


# -- request -> brain inputs -------------------------------------------------

def _seed_history(messages) -> list:
    """Prior-turn ChatMessages -> plain dicts for the loopback rounds.
    ``exclude_none`` keeps hand-built payload junk out; an assistant message
    whose content was null gets it back as '' (templates need the key)."""
    out = []
    for m in messages:
        d = m.model_dump(exclude_none=True)
        if "content" not in d:
            d["content"] = ""
        out.append(d)
    return out


def _last_user_text(message):
    """(text, error_message). The turn driving the loop must be user text -
    the brain owns the request assembly from there."""
    if message.role != "user":
        return None, "the last message must have role 'user'"
    content = message.content
    if isinstance(content, str):
        return content, None
    if isinstance(content, list):
        parts = []
        for p in content:
            d = p.model_dump(exclude_none=True) if hasattr(
                p, "model_dump") else dict(p)
            if d.get("type") in ("text", "input_text") and "text" in d:
                parts.append(str(d["text"]))
            else:
                return None, "assistant models are text-only"
        return "".join(parts), None
    return None, "the last message must carry text content"


def _request_extra(request) -> dict:
    """The per-round payload extras: whitelisted sampling the outer request
    actually set, its `stop` (the outer stop wrapper never sees loop rounds),
    and stream_options - the inner server gates usage chunks on it."""
    extra: dict = {}
    for f in _SAMPLING_FIELDS:
        if f in request.model_fields_set:
            extra[f] = getattr(request, f)
    stop = (request.model_extra or {}).get("stop")
    if stop is not None:
        extra["stop"] = stop
    extra["stream_options"] = {"include_usage": True}
    return extra


def _build_brain(state: _AssistantState, alias_id: str, alias, registry,
                 request, cancel: threading.Event, usage: dict):
    """A per-request AssistantBrain wired to the loopback seam. The seam
    checks the cancel event between deltas and aggregates usage across
    rounds (completion_tokens summed; prompt_tokens = the final round's -
    rounds re-send the growing history, summing them double-counts)."""

    extra = _request_extra(request)

    def seam(base_url, *, model, messages, max_tokens, api_key=None,
             tools=None, timeout=600.0):
        usage["rounds"] += 1
        for delta in stream_chat(base_url, model=model, messages=messages,
                                 max_tokens=max_tokens, api_key=api_key,
                                 tools=tools, timeout=timeout, extra=extra):
            if cancel.is_set():
                raise _AssistantCancelled()
            if "_usage" in delta:
                u = delta["_usage"] or {}
                usage["completion_tokens"] += int(
                    u.get("completion_tokens") or 0)
                usage["prompt_tokens"] = int(u.get("prompt_tokens") or 0)
            yield delta

    max_tokens = (request.max_tokens
                  if "max_tokens" in request.model_fields_set
                  else _DEFAULT_MAX_TOKENS)
    a = state.cfg.assistant
    brain = AssistantBrain(
        base_url=state.base_url, model=alias.model, api_key=state.api_key,
        system=None, max_tokens=max_tokens, tools=registry,
        max_tool_rounds=a.max_tool_rounds, tool_timeout_s=a.tool_timeout_s,
        memory=state.memories.get(alias_id), stream=seam)
    brain.messages = _seed_history(request.messages[:-1])
    return brain


def _spawn_turn(brain, user_text: str, loop, queue: asyncio.Queue):
    """Run one brain turn on a dedicated daemon thread, forwarding events
    into ``queue``. Dedicated (not a pool thread): the inner loopback request
    needs the event loop and starlette's pool free - this can never deadlock
    them. Ends with a ("_finished", ...) sentinel, error or not."""

    def emit(item):
        loop.call_soon_threadsafe(queue.put_nowait, item)

    def work():
        error = None
        try:
            for ev in brain.turn(user_text):
                emit(ev)
        except _AssistantCancelled:
            pass
        except TalkClientError as e:
            error = str(e)
        except Exception as e:                    # noqa: BLE001 - to client
            error = f"{type(e).__name__}: {e}"
        emit(("_finished", error))

    t = threading.Thread(target=work, daemon=True, name="assistant-turn")
    t.start()
    return t


def _log_turn(alias_id: str, usage: dict, *, error: str | None,
              cancelled: bool = False) -> None:
    state = "cancelled" if cancelled else ("error" if error else "ok")
    print(f"[req] chat.completions assistant={alias_id} "
          f"rounds={usage['rounds']} "
          f"completion_tokens={usage['completion_tokens']} {state}",
          file=sys.stderr, flush=True)


# -- the two response shapes -------------------------------------------------

def _chunk(cid: str, created: int, model: str, delta: dict,
           finish: str | None = None) -> str:
    choice: dict = {"index": 0, "delta": delta}
    if finish:
        choice["finish_reason"] = finish
    return "data: " + json.dumps({
        "id": cid, "object": "chat.completion.chunk", "created": created,
        "model": model, "choices": [choice]}) + "\n\n"


async def _stream_response(release, alias_id, brain, user_text, request):
    """Streaming: role chunk, content chunks as the loop speaks, stop, a
    usage chunk iff the outer request asked, [DONE]. Tool activity surfaces
    only as SSE comment lines (keepalive through buffering proxies). In-band
    error object on worker failure - the 200 is already committed.

    ``release`` is the one-shot concurrency-slot release (see
    ``make_release_guard``): the body's finally calls it on the normal path, and
    the guarded response's __call__ backstops it if an immediate client
    disconnect cancels the response before the body generator ever starts."""
    from starlette.responses import StreamingResponse

    class _GuardedStreamingResponse(StreamingResponse):
        async def __call__(self, scope, receive, send):
            try:
                await super().__call__(scope, receive, send)
            finally:
                # If body() ran, it already released (no-op here). If the client
                # disconnected before body()'s first __anext__, this is the only
                # place the slot gets released.
                release()

    usage = brain._kq_usage
    cancel = brain._kq_cancel
    cid = "chatcmpl-" + uuid.uuid4().hex[:24]
    created = int(time.time())
    want_usage = bool(request.stream_options
                      and request.stream_options.include_usage)

    async def body():
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()
        error = None
        cancelled = True                     # flipped when we reach the end
        try:
            _spawn_turn(brain, user_text, loop, queue)
            yield _chunk(cid, created, alias_id, {"role": "assistant"})
            while True:
                kind, payload = await queue.get()
                if kind == "say":
                    yield _chunk(cid, created, alias_id, {"content": payload})
                elif kind == "status":
                    yield f": assistant {payload}\n\n"
                elif kind == "_finished":
                    error = payload
                    break
            cancelled = False
            if error is not None:
                yield "data: " + json.dumps({"error": {
                    "message": f"assistant upstream round failed: {error}",
                    "type": "api_error",
                    "code": "assistant_upstream_error"}}) + "\n\n"
            else:
                yield _chunk(cid, created, alias_id, {}, finish="stop")
                if want_usage:
                    yield "data: " + json.dumps({
                        "id": cid, "object": "chat.completion.chunk",
                        "created": created, "model": alias_id,
                        "choices": [],
                        "usage": _usage_payload(usage)}) + "\n\n"
            yield "data: [DONE]\n\n"
        finally:
            # Disconnect cancels this generator: without this set() the brain
            # thread orphans and keeps firing loopback rounds.
            cancel.set()
            release()
            _log_turn(alias_id, usage, error=error, cancelled=cancelled)

    return _GuardedStreamingResponse(body(), media_type="text/event-stream")


def _usage_payload(usage: dict) -> dict:
    return {"prompt_tokens": usage["prompt_tokens"],
            "completion_tokens": usage["completion_tokens"],
            "total_tokens": usage["prompt_tokens"]
            + usage["completion_tokens"]}


async def _json_response(release, alias_id, brain, user_text):
    """Non-streaming: drain the whole turn, one chat.completion. A client
    disconnect cannot cancel this path (no finalization hook on a plain
    response) - the turn runs to completion. ``release`` is the one-shot slot
    release (see ``make_release_guard``)."""
    from starlette.responses import JSONResponse

    usage = brain._kq_usage
    cancel = brain._kq_cancel
    parts: list = []
    error = None
    try:
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()
        _spawn_turn(brain, user_text, loop, queue)
        while True:
            kind, payload = await queue.get()
            if kind == "say":
                parts.append(payload)
            elif kind == "_finished":
                error = payload
                break
    finally:
        cancel.set()
        release()
        _log_turn(alias_id, usage, error=error)
    if error is not None:
        return _openai_error(502, f"assistant upstream round failed: {error}",
                             "api_error", code="assistant_upstream_error")
    return JSONResponse(content={
        "id": "chatcmpl-" + uuid.uuid4().hex[:24],
        "object": "chat.completion",
        "created": int(time.time()),
        "model": alias_id,
        "choices": [{"index": 0,
                     "message": {"role": "assistant",
                                 "content": "".join(parts)},
                     "finish_reason": "stop"}],
        "usage": _usage_payload(usage)})


async def _assistant_completion(state, alias_id, alias, registry, request):
    """The assistant path of the chat wrapper: validate, try-acquire (the
    handler is the only place a true HTTP 429 can originate - a streaming
    response commits its status before the body generator runs), build the
    per-request brain, answer in the requested shape."""
    if not request.messages:
        return _openai_error(400, "messages must not be empty",
                             "invalid_request_error")
    user_text, err = _last_user_text(request.messages[-1])
    if err is not None:
        return _openai_error(400, err, "invalid_request_error")
    if not state.try_acquire():
        return _openai_error(
            429, f"assistant is at its concurrency cap "
            f"({_MAX_CONCURRENT_TURNS} turns); retry shortly",
            "rate_limit_error")
    release = state.make_release_guard()     # one-shot; safe to call twice
    try:
        usage = {"rounds": 0, "prompt_tokens": 0, "completion_tokens": 0}
        cancel = threading.Event()
        brain = _build_brain(state, alias_id, alias, registry, request,
                             cancel, usage)
        brain._kq_usage = usage
        brain._kq_cancel = cancel
    except Exception:
        release()
        raise
    if request.stream:
        try:
            return await _stream_response(release, alias_id, brain, user_text,
                                          request)
        except Exception:
            release()
            raise
    return await _json_response(release, alias_id, brain, user_text)


# -- route wrappers ----------------------------------------------------------

def _wrap_chat_routes(app, state) -> None:
    from .server_patches._common import _CHAT_PATHS, _wrap_post_routes

    def _make(original):
        async def endpoint(request, http_request):
            entry = state.aliases.get(getattr(request, "model", None))
            if entry is None:
                return await original(request, http_request)
            alias_id = request.model
            alias, registry = entry
            if getattr(request, "tools", None):
                # The client runs its own loop: unwrap to the underlying
                # model, pass through untouched.
                request.model = alias.model
                return await original(request, http_request)
            return await _assistant_completion(state, alias_id, alias,
                                               registry, request)
        return endpoint

    _wrap_post_routes(app, _CHAT_PATHS, _ASSISTANT_FLAG, _make)


def _wrap_models_routes(app, state) -> None:
    from .server_patches._common import _remove_routes
    for path in ("/models", "/v1/models"):
        route = next(
            (r for r in app.router.routes
             if getattr(r, "path", None) == path
             and "GET" in (getattr(r, "methods", None) or ())),
            None)
        if route is None or getattr(route.endpoint, _ASSISTANT_FLAG, False):
            continue
        original = route.endpoint

        def _make(original):
            async def endpoint():
                payload = await original()
                for aid, (alias, _reg) in sorted(state.aliases.items()):
                    payload["data"].append({
                        "id": aid, "object": "model", "created": 0,
                        "owned_by": "gmlx", "assistant": True,
                        "alias_of": alias.model})
                return payload

            endpoint.__dict__[_ASSISTANT_FLAG] = True
            return endpoint

        wrapped = _make(original)
        _remove_routes(app, path)
        app.add_api_route(path, wrapped, methods=["GET"],
                          include_in_schema=False)


def _wrap_foreign_routes(app, state) -> None:
    """Assistant ids are chat-completions only: 400 on the responses /
    anthropic surfaces (in each surface's own error shape). Reading the
    cached request body is safe - the original handler re-parses it."""
    from .server_patches._common import _remove_routes
    for path in _GUARDED_PATHS:
        route = next(
            (r for r in app.router.routes
             if getattr(r, "path", None) == path
             and "POST" in (getattr(r, "methods", None) or ())),
            None)
        if route is None or getattr(route.endpoint, _ASSISTANT_FLAG, False):
            continue
        original = route.endpoint
        anthropic = path.startswith(_ANTHROPIC_PREFIXES)

        def _make(original, anthropic):
            async def endpoint(*args, **kwargs):
                for obj in list(args) + list(kwargs.values()):
                    if hasattr(obj, "receive") and hasattr(obj, "json"):
                        try:
                            body = await obj.json()
                        except Exception:          # noqa: BLE001 - not ours
                            break
                        model = (body or {}).get("model") \
                            if isinstance(body, dict) else None
                        if model in state.aliases:
                            msg = ("assistant models are chat-completions "
                                   "only: use /v1/chat/completions with "
                                   f"model {model!r}")
                            if anthropic:
                                return _anthropic_error(400, msg)
                            return _openai_error(400, msg,
                                                 "invalid_request_error")
                        break
                return await original(*args, **kwargs)

            endpoint.__signature__ = inspect.signature(original)
            endpoint.__dict__[_ASSISTANT_FLAG] = True
            return endpoint

        wrapped = _make(original, anthropic)
        _remove_routes(app, path)
        app.add_api_route(path, wrapped, methods=["POST"],
                          include_in_schema=False)


# -- startup -----------------------------------------------------------------

def _build_state(cfg) -> _AssistantState:
    from .talk_mcp import connect_servers
    state = _AssistantState(cfg, _self_base_url(cfg))
    aliases = cfg.assistants

    shared_registry = None
    if any(alias.mcp is None for alias in aliases.values()):
        host, shared_registry, warns = connect_servers(
            cfg.assistant.mcp, call_timeout_s=cfg.assistant.tool_timeout_s)
        for w in warns:
            print(f"[server] assistant mcp: {w}", file=sys.stderr, flush=True)
        if host is not None:
            state.hosts.append(host)
        _lock_registry(shared_registry)

    for aid, alias in aliases.items():
        if alias.mcp is None:
            registry = shared_registry
        elif not alias.mcp:
            registry = ToolRegistry()        # explicit lockdown: no tools
        else:
            host, registry, warns = connect_servers(
                alias.mcp, call_timeout_s=cfg.assistant.tool_timeout_s)
            for w in warns:
                print(f"[server] assistant '{aid}' mcp: {w}",
                      file=sys.stderr, flush=True)
            if host is not None:
                state.hosts.append(host)
            _lock_registry(registry)
        state.aliases[aid] = (alias, registry)

        if alias.memory:
            state.memories[aid] = _open_memory(state, aid, alias)
        tools = ", ".join(registry.names()) or "(none)"
        print(f"[server] assistant '{aid}' -> {alias.model}  tools: {tools}"
              + ("  memory: on" if alias.memory else ""), flush=True)
    if state.memories:
        import atexit
        atexit.register(state.close_memories)
    return state


def _open_memory(state: _AssistantState, alias_id: str, alias):
    """One shared store per alias, in its own db (never the talk store).
    No hard dependency on cfg.embeddings: the store embeds via the server's
    own /v1/embeddings at runtime and self-disables if that 404s - the unset
    config is only an early liveness warning."""
    from .talk_memory import MemoryStore, default_memory_path, make_extractor
    if not getattr(state.cfg, "embeddings", None):
        print(f"[server] assistant '{alias_id}': memory will be inert - "
              "/v1/embeddings is not served (set server.embeddings)",
              file=sys.stderr, flush=True)
    mem_cfg = state.cfg.assistant.memory
    path = os.path.join(os.path.dirname(default_memory_path()),
                        f"assistant-{alias_id}.db")
    entry = state.aliases[alias_id]
    extractor = (make_extractor(state.base_url, entry[0].model,
                                api_key=state.api_key)
                 if mem_cfg.extract else None)
    return MemoryStore(base_url=state.base_url, api_key=state.api_key,
                       path=path, top_k=mem_cfg.top_k, extract=extractor,
                       ttl_days=mem_cfg.ttl_days, max_items=mem_cfg.max_items)


def install_assistant_serve(cfg) -> None:
    """Wire ``server.assistants`` into the app. Call last in
    install_server_patches: the chat wrapper must be outermost so alias ids
    never reach the resolver (they are not loadable models), and the models
    wrapper must wrap the configured-ids override. No-op without aliases."""
    if not getattr(cfg, "assistants", None):
        return
    app = importlib.import_module("mlx_vlm.server.app").app
    state = _build_state(cfg)
    _wrap_chat_routes(app, state)
    _wrap_models_routes(app, state)
    _wrap_foreign_routes(app, state)
