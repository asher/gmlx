"""Hardening patches: API-key auth, JSON content-type tolerance,
loopback Host guard, credential-less CORS, and the liveness-only /health body."""

from __future__ import annotations

import importlib
import os


from ._common import (
    _error_content,
    _remove_routes,
)


# API-key auth
_AUTH_FLAG = "_kq_gguf_api_key_auth"


def install_api_key_auth(api_key: str | None) -> None:
    """Require a static API key on every route except ``/health`` (kept open so
    liveness probes and ``launch``'s reachability check work unauthenticated).
    Accepts ``Authorization: Bearer <key>`` (OpenAI-style clients) or
    ``x-api-key: <key>`` (Anthropic-style clients); compares in constant time.
    No-op without a key; idempotent.

    Note: this is HTTP middleware - a future websocket route would bypass it
    (Starlette ``http``-type middleware never sees websocket scopes); none
    exists today."""
    if not api_key:
        return
    import hmac

    from fastapi.responses import JSONResponse

    app = importlib.import_module("mlx_vlm.server.app").app
    if getattr(app.state, _AUTH_FLAG, False):
        return
    key_bytes = api_key.encode()         # bytes: str compare_digest rejects non-ASCII

    async def _auth_middleware(request, call_next):
        # OPTIONS = CORS preflight, which browsers send credential-less by spec
        # (CORSMiddleware answers it); the actual request still authenticates.
        if request.method == "OPTIONS" or request.url.path == "/health":
            return await call_next(request)
        auth = request.headers.get("authorization", "")
        provided = (auth[7:] if auth[:7].lower() == "bearer "
                    else request.headers.get("x-api-key", ""))
        # latin-1 reverses Starlette's header decode byte-for-byte, so a
        # utf-8 key sent on the wire compares equal to key_bytes (utf-8).
        if not (provided
                and hmac.compare_digest(provided.encode("latin-1"), key_bytes)):
            return JSONResponse(status_code=401, content=_error_content(
                request.url.path, 401, "authentication_error",
                "invalid or missing API key (send `Authorization: "
                "Bearer <key>`, or `x-api-key: <key>`)"))
        return await call_next(request)

    app.middleware_stack = None          # allow install after a stack build
    app.middleware("http")(_auth_middleware)
    app.state._kq_gguf_api_key_auth = True


_JSON_CT_FLAG = "_kq_gguf_json_ct_tolerance"


def install_json_content_type_tolerance() -> None:
    """Treat body-bearing API requests without a JSON content-type as JSON.

    ``curl -d '{...}'`` - the shape of every copy-paste API example - sends
    ``application/x-www-form-urlencoded``, which FastAPI 422s with a raw
    pydantic error before the body is even parsed. The JSON endpoints here
    accept exactly one body shape, so a missing, form-encoded, or text/plain
    content-type is rewritten to ``application/json``; multipart uploads
    (audio transcription) pass through untouched. Idempotent."""
    app = importlib.import_module("mlx_vlm.server.app").app
    if getattr(app.state, _JSON_CT_FLAG, False):
        return

    rewritable = ("", "application/x-www-form-urlencoded", "text/plain")

    async def _json_ct_middleware(request, call_next):
        if request.method in ("POST", "PUT", "PATCH"):
            ct = request.headers.get("content-type", "")
            if ct.split(";", 1)[0].strip().lower() in rewritable:
                headers = [(k, v) for k, v in request.scope["headers"]
                           if k.lower() != b"content-type"]
                headers.append((b"content-type", b"application/json"))
                request.scope["headers"] = headers
        return await call_next(request)

    app.middleware_stack = None          # allow install after a stack build
    app.middleware("http")(_json_ct_middleware)
    setattr(app.state, _JSON_CT_FLAG, True)


# 10. Loopback host guard (DNS-rebinding) + CORS credential drop + /health trim
_HOST_GUARD_FLAG = "_kq_gguf_host_guard"


def _host_header_name(value: str) -> str:
    """The hostname part of a ``Host`` header value, lowercased (host names
    compare case-insensitively): port stripped, ``[::1]:8080`` bracket form
    handled, a bare unbracketed IPv6 address left intact."""
    value = value.strip().lower()
    if value.startswith("["):
        return value.partition("]")[0].lstrip("[")
    if value.count(":") > 1:  # unbracketed IPv6 - the colons aren't a port
        return value
    return value.rsplit(":", 1)[0] if ":" in value else value


def install_loopback_host_guard(bind_host: str) -> None:
    """Reject requests whose ``Host`` header isn't a loopback name (403).

    DNS rebinding: a page on evil.com re-points its hostname at 127.0.0.1 and
    reaches a loopback-bound server *same-origin*, bypassing CORS entirely -
    but the browser still sends ``Host: evil.com``, so checking it defeats the
    attack. Installed only for loopback binds; non-loopback binds are covered
    by the api-key policy instead. A missing Host header (non-browser HTTP/1.0
    clients) is allowed. Idempotent."""
    from fastapi.responses import JSONResponse

    from ..config import LOOPBACK_HOSTS

    app = importlib.import_module("mlx_vlm.server.app").app
    if getattr(app.state, _HOST_GUARD_FLAG, False):
        return
    allowed = {h.lower() for h in LOOPBACK_HOSTS} | {bind_host.lower()}

    async def _host_guard(request, call_next):
        host = request.headers.get("host")
        if host and _host_header_name(host) not in allowed:
            return JSONResponse(status_code=403, content=_error_content(
                request.url.path, 403, "invalid_host",
                f"Host {host!r} is not a loopback name - this "
                f"server is bound to {bind_host} and rejects "
                f"non-loopback Host headers (DNS-rebinding guard). "
                f"Connect via http://127.0.0.1:<port>."))
        return await call_next(request)

    app.middleware_stack = None
    app.middleware("http")(_host_guard)  # added last => outermost, runs first
    app.state._kq_gguf_host_guard = True


def disable_credentialed_cors() -> None:
    """Flip the stock app's CORS to ``allow_credentials=False``.

    Starlette implements ``allow_origins=["*"]`` + ``allow_credentials=True``
    by reflecting any request Origin with ``Access-Control-Allow-Credentials:
    true`` - credentialed cross-origin access from every website. Auth here is
    header-based (no cookies), so credentialed CORS is never needed; without
    it the response carries a literal ``*``."""
    from fastapi.middleware.cors import CORSMiddleware

    app = importlib.import_module("mlx_vlm.server.app").app
    for m in app.user_middleware:
        kwargs = getattr(m, "kwargs", None)
        if getattr(m, "cls", None) is CORSMiddleware \
                and kwargs and kwargs.get("allow_credentials"):
            kwargs["allow_credentials"] = False
            app.middleware_stack = None  # rebuilt on next startup


def install_health_liveness_override() -> None:
    """Trim ``/health`` - the one route the api-key auth exempts - to a pure
    liveness body. The stock handler returns absolute model/adapter paths,
    readable by any unauthenticated caller (or, on a loopback bind, any local
    webpage). The full detail (``resident_models[]``, context limits) stays on
    the authed ``/v1/metrics`` via the runtime snapshot."""
    app = importlib.import_module("mlx_vlm.server.app").app

    async def health_endpoint():
        # pid lets the CLI verify it is talking to the process it manages (a
        # foreign server on the same port answers with a different pid).
        return {"status": "healthy", "pid": os.getpid()}

    _remove_routes(app, "/health")
    app.add_api_route("/health", health_endpoint, methods=["GET"],
                      include_in_schema=False)
