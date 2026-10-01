"""Hardening patches: API-key auth, JSON content-type tolerance,
loopback Host guard, the browser origin guard, credential-less CORS, and the
liveness-only /health body."""

from __future__ import annotations

import importlib
import os
import re
import time


from ._common import (
    SESSION_SCOPE_KEY,
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

    This is HTTP middleware, and Starlette HTTP middleware never sees a
    WebSocket connection. The media gate therefore removes every WebSocket
    route and closes every WebSocket connection.

    A request on a launch session socket needs no key. The scope of its
    socket limits it instead (:mod:`.session_sockets`)."""
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
        if request.method == "OPTIONS" or request.url.path == "/health" \
                or request.scope.get(SESSION_SCOPE_KEY) is not None:
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


def install_json_content_type_tolerance(app=None) -> None:
    """Treat body-bearing API requests without a JSON content-type as JSON.

    ``curl -d '{...}'`` - the shape of every copy-paste API example - sends
    ``application/x-www-form-urlencoded``, which FastAPI 422s with a raw
    pydantic error before the body is even parsed. The JSON endpoints here
    accept exactly one body shape, so a missing, form-encoded, or text/plain
    content-type is rewritten to ``application/json``; multipart uploads
    (audio transcription) pass through untouched. ``app`` defaults to
    mlx-vlm's. Idempotent."""
    if app is None:
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

    from gmlx.config import LOOPBACK_HOSTS

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
    header-based (no cookies), so credentialed CORS is never needed.
    :func:`install_origin_guard` replaces the wildcard origin list."""
    from fastapi.middleware.cors import CORSMiddleware

    app = importlib.import_module("mlx_vlm.server.app").app
    for m in app.user_middleware:
        kwargs = getattr(m, "kwargs", None)
        if getattr(m, "cls", None) is CORSMiddleware \
                and kwargs and kwargs.get("allow_credentials"):
            kwargs["allow_credentials"] = False
            app.middleware_stack = None  # rebuilt on next startup


# Browser origin guard
_ORIGIN_FLAG = "_kq_gguf_origin_guard"
# The loopback and desktop-app origins, in the form Starlette's CORS
# middleware matches with ``fullmatch``. Keep in step with
# gmlx.config.origin_is_loopback and origin_is_app, which decide; this only
# lets the CORS headers name such an origin.
LOOPBACK_ORIGIN_REGEX = (r"https?://(localhost|127(\.[0-9]{1,3}){3}|\[::1\]"
                         r"|\[::ffff:7f[0-9a-f]{2}:[0-9a-f]{1,4}\])(:[0-9]{1,5})?")
APP_ORIGIN_REGEX = r"(?i:(?:app|file|tauri|vscode-file|vscode-webview)://[a-z0-9._-]*)"
# The normalized server.cors_origins, set by install_origin_guard.
_allowed_origins: frozenset[str] = frozenset()
# A refusal is logged at most once a minute for each origin, and at most
# _REFUSALS_LOGGED_MAX times a minute in all.
_REFUSAL_LOG_EVERY = 60.0
_REFUSALS_LOGGED_MAX = 20
_refusals_logged: dict[str, float] = {}
_refusal_window = [0.0, 0]


def origin_allowed(origin: str) -> bool:
    """Whether a page at ``origin`` may call the server: a loopback origin,
    which only a process on this Mac can serve, a desktop app's origin, which
    no web page can send, a listed one, or an extension of a browser whose
    wildcard entry, such as ``chrome-extension://*``, is listed."""
    from gmlx.config import normalize_origin, origin_is_app, origin_is_loopback

    try:
        norm = normalize_origin(origin)
    except ValueError:
        return False                  # "null" and anything malformed
    return (origin_is_loopback(norm) or origin_is_app(norm) or norm in _allowed_origins
            or f"{norm.partition('://')[0]}://*" in _allowed_origins)


# The origins browser extensions send. A page cannot send one.
EXTENSION_SCHEMES = ("chrome-extension", "moz-extension", "safari-web-extension",
                     "ms-browser-extension")
_RESTART = "then run gmlx restart"


def _origin_refusal(origin: str) -> str:
    from gmlx.config import normalize_origin

    shown = origin[:200]
    if shown.strip() == "null":
        return ("This server does not answer a page with no origin of its own "
                "(Origin: null), such as a file opened from disk or a sandboxed frame. "
                "Serve the page from a loopback address instead.")
    try:
        shown = normalize_origin(shown)
    except ValueError:
        return (f"The Origin header \"{shown}\" is not an origin, so the server "
                "cannot tell which page sent the request.")
    add = f"Add {shown} to server.cors_origins in the server's config file, {_RESTART}."
    scheme = shown.partition("://")[0]
    if scheme == "safari-web-extension":
        return (f"The browser extension at {shown} may not call this server. Safari "
                "gives an extension a new ID at each launch, so add "
                "safari-web-extension://* to server.cors_origins in the server's config "
                f"file, which lets every Safari extension call it, {_RESTART}.")
    if scheme in EXTENSION_SCHEMES:
        return f"The browser extension at {shown} may not call this server. {add}"
    if not shown.startswith(("http://", "https://")):
        return f"The app that sent Origin {shown} may not call this server. {add}"
    return (f"Pages from {shown} may not call this server. {add} Or serve the page "
            "from a loopback address.")


def _log_refusal(origin: str, message: str) -> None:
    """Print a refusal to the server's log, where the operator can read it.
    A browser shows the page only a CORS error, never the 403's body."""
    from gmlx.container.text import printable

    now = time.monotonic()
    if now - _refusal_window[0] >= _REFUSAL_LOG_EVERY:
        _refusal_window[:] = [now, 0]
    key = origin[:200]
    last = _refusals_logged.pop(key, None)
    if last is not None and now - last < _REFUSAL_LOG_EVERY:
        _refusals_logged[key] = last
        return
    if _refusal_window[1] >= _REFUSALS_LOGGED_MAX:
        return
    _refusal_window[1] += 1
    _refusals_logged[key] = now
    while len(_refusals_logged) > 256:
        del _refusals_logged[next(iter(_refusals_logged))]
    print(f"[server] refused a request with status 403: {printable(message)}", flush=True)


def _session_page(origin: str, scope) -> tuple[str, int, bool] | None:
    """The normalized ``origin``, its port, and whether the session is still
    open, when a request on the TCP listener comes from a loopback page on
    the web port of a launch session that is open or ended a short time
    ago, else None. The session's browser app reaches the server through
    the session, so its pages have no reason to call the TCP port, and a
    page can stay open in a tab after the session ends. The port decides,
    since the page can load itself under any loopback name."""
    import urllib.parse

    from gmlx.config import normalize_origin, origin_is_loopback

    from .session_sockets import ended_web_ports, session_web_ports

    if scope.get(SESSION_SCOPE_KEY) is not None:
        return None
    ports, ended = session_web_ports(), ended_web_ports()
    if not ports and not ended:
        return None
    try:
        norm = normalize_origin(origin)
    except ValueError:
        return None
    if not origin_is_loopback(norm):
        return None
    split = urllib.parse.urlsplit(norm)
    port = split.port or (443 if split.scheme == "https" else 80)
    if port in ports:
        return norm, port, True
    return (norm, port, False) if port in ended else None


def _session_page_refusal(origin: str, port: int, open_: bool) -> str:
    from .session_sockets import WEB_PORT_GRACE

    if open_:
        return (f"Pages from {origin} are on port {port}, where a launch container "
                "session serves its browser app. The app reaches this server through "
                "its session, so these pages may not call this port.")
    return (f"Pages from {origin} are on port {port}, where a launch container session "
            f"served its browser app until it ended. These pages may not call this port "
            f"for {WEB_PORT_GRACE / 60:.0f} minutes after the session ends. Close the "
            "app's browser tabs.")


def _restrict_cors(app) -> None:
    """Answer CORS for the loopback, desktop-app and listed origins only,
    never with ``*``. A listed origin matches in any case, as the guard
    matches it, and a wildcard entry matches every ID of its scheme."""
    from fastapi.middleware.cors import CORSMiddleware

    listed = sorted(o for o in _allowed_origins if not o.endswith("://*"))
    parts = [LOOPBACK_ORIGIN_REGEX, APP_ORIGIN_REGEX]
    if listed:
        parts.append(f"(?i:{'|'.join(re.escape(o) for o in listed)})")
    parts += [f"(?i:{re.escape(o[:-1])}[a-z0-9._-]+)"
              for o in sorted(_allowed_origins) if o.endswith("://*")]
    for m in app.user_middleware:
        kwargs = getattr(m, "kwargs", None)
        if getattr(m, "cls", None) is CORSMiddleware and kwargs is not None:
            kwargs["allow_origins"] = listed
            kwargs["allow_origin_regex"] = "|".join(parts)
            app.middleware_stack = None


def install_origin_guard(allowed_origins=()) -> None:
    """Refuse a request, with 403, whose ``Origin`` header names a page that
    may not call the server: anything but a loopback origin, a desktop app's
    origin or one in ``allowed_origins`` (``server.cors_origins``), ``null``
    included. Each refusal is also printed to the server's log.

    A server without a key answers any local process, and a browser sends a
    page's requests from the user's machine. Without this check any website
    the user visits could call a loopback server and read the answers, and a
    ``text/plain`` POST, which needs no preflight, would run as JSON. A
    client that is not a browser sends no ``Origin`` and passes. A loopback
    page on the web port of a launch session is refused too, except on the
    session's own socket, while the session is open and for a grace after
    it ends. The check reads no body. Install it after the
    host guard, so it is the outermost middleware and runs first. Each call
    replaces the allowed list."""
    global _allowed_origins
    from fastapi.responses import JSONResponse

    from gmlx.config import normalize_cors_entry

    _allowed_origins = frozenset(normalize_cors_entry(o) for o in allowed_origins)
    app = importlib.import_module("mlx_vlm.server.app").app
    _restrict_cors(app)
    # Found by the dispatch function rather than an app.state flag, so a
    # test that restores the middleware list also restores the guard.
    if any(getattr((getattr(m, "kwargs", None) or {}).get("dispatch"), _ORIGIN_FLAG, False)
           for m in app.user_middleware):
        return

    async def _origin_guard(request, call_next):
        origin = request.headers.get("origin")
        if origin is not None:
            if not origin_allowed(origin):
                message = _origin_refusal(origin)
            elif (page := _session_page(origin, request.scope)) is not None:
                message = _session_page_refusal(*page)
            else:
                return await call_next(request)
            _log_refusal(origin, message)
            return JSONResponse(status_code=403, content=_error_content(
                request.url.path, 403, "origin_not_allowed", message))
        return await call_next(request)

    _origin_guard.__dict__[_ORIGIN_FLAG] = True
    app.middleware_stack = None
    app.middleware("http")(_origin_guard)   # added last => outermost, runs first


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
