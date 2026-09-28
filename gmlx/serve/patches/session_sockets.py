"""Launch session sockets: one scoped Unix listener per container session.

A ``gmlx launch`` container client reaches the server through the launch
relay, and the relay connects to a session socket, not to the TCP port.
``POST /v1/launch/sessions`` on the TCP listener opens a session socket for
a list of assistant aliases, and ``DELETE /v1/launch/sessions/<id>`` closes
it. A request on a session socket needs no key, and the session scope
limits it instead:

- Only the inference routes in :data:`SESSION_PATHS` answer. Every other
  route answers 404, the session endpoints included.
- A request that names an assistant alias outside the session's list gets
  the unknown-model 404, and ``/v1/models`` leaves such aliases out.

Each socket has a uvicorn server of its own in the server's event loop,
over the same app. That server leaves the process signal handlers to the
main server. A socket is mode 0600, in a folder of mode 0700 that only the
server uses, and the server removes every session socket at start and at
stop.
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import importlib
import json
import logging
import os
import re
import secrets
import socket
import stat
from pathlib import Path

import uvicorn
from fastapi import Request  # module-level so stringized annotations resolve

from ._common import SESSION_SCOPE_KEY, _error_content, _remove_routes

_log = logging.getLogger(__name__)

ENDPOINT = "/v1/launch/sessions"

# The routes a session socket serves, each in the ``/v1/<p>`` and the
# ``/<p>`` form. These are what the launch clients send.
_SESSION_ROUTES = (
    "models", "chat/completions", "completions", "responses",
    "responses/input_tokens", "messages", "messages/count_tokens",
    "embeddings", "rerank", "audio/speech", "audio/voices",
    "audio/transcriptions", "audio/translations", "images/generations",
    "images/edits", "systemone")
SESSION_PATHS = frozenset(
    ["/health"] + [f"/{p}" for p in _SESSION_ROUTES]
    + [f"/v1/{p}" for p in _SESSION_ROUTES])
# The routes that treat an assistant alias as a model id: the chat routes
# run its tool loop, and the others refuse it with a 400.
_ALIAS_ROUTES = ("chat/completions", "responses", "responses/input_tokens",
                 "messages", "messages/count_tokens")
_ALIAS_PATHS = frozenset(
    [f"/{p}" for p in _ALIAS_ROUTES] + [f"/v1/{p}" for p in _ALIAS_ROUTES])
_MODELS_PATHS = frozenset({"/models", "/v1/models"})

# macOS holds 104 bytes for a socket path, the final NUL included.
SOCKET_PATH_MAX = 103
_ID_BYTES = 6
_SOCKET_NAME = re.compile(r"^[0-9a-f]{12}\.sock$")
_SOCKET_NAME_LEN = 2 * _ID_BYTES + len(".sock")
_BACKLOG = 2048
# Past this many open sessions, a new session closes the oldest one that
# has no open connection. A launch that dies before its DELETE leaves its
# session open, and each session server wakes ten times a second.
SESSIONS_MAX = 32

# Set in the task of each session server. uvicorn logs these lines when a
# server starts and stops, and from a session server they would read as
# the whole server starting or stopping.
_IN_SESSION_SERVER = contextvars.ContextVar("gmlx_session_server",
                                            default=False)
_LIFECYCLE_MESSAGES = frozenset({
    "Started server process [%d]", "Finished server process [%d]",
    "Shutting down",
    "Waiting for connections to close. (CTRL+C to force quit)",
    "Waiting for background tasks to complete. (CTRL+C to force quit)"})


class _LifecycleFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return not (_IN_SESSION_SERVER.get()
                    and record.msg in _LIFECYCLE_MESSAGES)


_LOG_FILTER = _LifecycleFilter()


class _SessionServer(uvicorn.Server):
    """A uvicorn server that leaves the process signals to the main server."""

    @contextlib.contextmanager
    def capture_signals(self):
        yield

    def install_signal_handlers(self) -> None:
        """uvicorn before 0.29 installs its handlers through this method."""


class _Session:
    def __init__(self, sid: str, client: str, path: str, allowed: frozenset):
        self.id = sid
        self.client = client
        self.path = path
        self.allowed = allowed
        self.server: _SessionServer | None = None
        self.task: asyncio.Task | None = None

    def idle(self) -> bool:
        return self.server is None or not self.server.server_state.connections


class _Sessions:
    """The open sessions of this server, and the aliases they can name."""

    def __init__(self, cfg):
        self.host = getattr(cfg, "host", None) or "127.0.0.1"
        self.port = getattr(cfg, "port", None) or 0
        assistant = getattr(cfg, "assistant", None)
        shared = list(getattr(assistant, "mcp", None) or [])
        # Alias id -> the names of the MCP servers its tool loop uses.
        self.tools: dict[str, list[str]] = {
            aid: [s.name for s in (shared if alias.mcp is None else alias.mcp)]
            for aid, alias in (getattr(cfg, "assistants", None) or {}).items()}
        self.open: dict[str, _Session] = {}

    def hidden(self, model: str, session: _Session) -> bool:
        return model in self.tools and model not in session.allowed

    async def start(self, client: str, allowed: frozenset) -> _Session:
        folder = socket_folder(self.host, self.port)
        while len(self.open) >= SESSIONS_MAX:
            oldest = next((s for s in self.open.values() if s.idle()), None)
            if oldest is None:
                raise RuntimeError(f"{SESSIONS_MAX} launch sessions are open "
                                   "and each one has an open connection")
            _log.warning("launch session %s (%s) closed to make room for a "
                         "new session", oldest.id, oldest.client)
            self.stop(oldest.id)
        sid = secrets.token_hex(_ID_BYTES)
        session = _Session(sid, client, str(folder / f"{sid}.sock"), allowed)
        sock = _listen(session.path)
        config = uvicorn.Config(
            _SessionApp(self, session), lifespan="off", ws="none",
            proxy_headers=False, server_header=False, log_config=None,
            backlog=_BACKLOG)
        session.server = _SessionServer(config)
        logging.getLogger("uvicorn.error").addFilter(_LOG_FILTER)
        session.task = asyncio.get_running_loop().create_task(
            _serve(session.server, sock))
        session.task.add_done_callback(
            lambda task, s=session: self._ended(task, s))
        self.open[sid] = session
        _log.info("launch session %s opened for %r, assistants: %s", sid,
                  client, ", ".join(sorted(allowed)) or "none")
        return session

    def stop(self, sid: str) -> _Session | None:
        """Close session ``sid``. Its socket goes at once, and the requests
        it holds open may finish."""
        session = self.open.pop(sid, None)
        if session is None:
            return None
        if session.server is not None:
            session.server.should_exit = True
        _unlink_socket(session.path)
        return session

    def stop_all(self) -> None:
        for sid in list(self.open):
            self.stop(sid)

    def _ended(self, task: asyncio.Task, session: _Session) -> None:
        if not task.cancelled() and task.exception() is not None:
            _log.error("launch session %s stopped: %s", session.id,
                       task.exception())
        if self.open.get(session.id) is session:
            self.stop(session.id)


_STATE: _Sessions | None = None


async def _serve(server: _SessionServer, sock: socket.socket) -> None:
    _IN_SESSION_SERVER.set(True)
    await server.serve(sockets=[sock])


def _listen(path: str) -> socket.socket:
    """A listening socket at ``path``, mode 0600 before it listens."""
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        sock.bind(path)
        os.chmod(path, 0o600)
        sock.listen(_BACKLOG)
    except OSError:
        sock.close()
        _unlink_socket(path)
        raise
    return sock


def _unlink_socket(path) -> None:
    """Remove ``path`` when it is a socket. Anything else stays."""
    try:
        if stat.S_ISSOCK(os.lstat(path).st_mode):
            os.unlink(path)
    except OSError:
        pass


# Folders

def _key(host: str, port) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", f"{host}-{port}").strip("-")


def socket_folders(host: str, port) -> list[Path]:
    """The folders that can hold the session sockets of the server at
    ``host:port``: one in the gmlx cache folder, and a shorter one under
    ``$TMPDIR`` for when the first would make a socket path too long."""
    cache = Path(os.environ.get("XDG_CACHE_HOME") or "~/.cache").expanduser()
    tmp = Path(os.environ.get("TMPDIR") or "/tmp")
    key = _key(host, port)
    return [cache / "gmlx" / f"sessions-{key}", tmp / f"gmlx-sessions-{key}"]


def _fits(folder: Path) -> bool:
    return len(os.fsencode(folder)) + 1 + _SOCKET_NAME_LEN <= SOCKET_PATH_MAX


def _owned_folder(folder: Path) -> bool:
    """Whether ``folder`` is a real folder of this user, not a link."""
    try:
        st = os.lstat(folder)
    except OSError:
        return False
    return stat.S_ISDIR(st.st_mode) and st.st_uid == os.getuid()


def socket_folder(host: str, port) -> Path:
    """The folder for new session sockets, created with mode 0700."""
    for folder in socket_folders(host, port):
        if not _fits(folder):
            continue
        folder.parent.mkdir(parents=True, exist_ok=True)
        try:
            folder.mkdir(mode=0o700)
        except FileExistsError:
            pass
        if not _owned_folder(folder):
            raise OSError(f"{folder} is not a folder of this user")
        os.chmod(folder, 0o700)
        return folder
    raise OSError("no session socket folder gives a path shorter than "
                  f"{SOCKET_PATH_MAX + 1} bytes; set TMPDIR to a shorter path")


def clear_session_sockets(host: str, port) -> None:
    """Remove every session socket of the server at ``host:port``."""
    for folder in socket_folders(host, port):
        if not _owned_folder(folder):
            continue
        try:
            names = os.listdir(folder)
        except OSError:
            continue
        for name in names:
            if _SOCKET_NAME.match(name):
                _unlink_socket(folder / name)


# The app of a session socket

async def _read_body(receive) -> bytes | None:
    """The whole request body, or None when the client went away."""
    chunks = []
    while True:
        message = await receive()
        if message["type"] == "http.disconnect":
            return None
        chunks.append(message.get("body", b""))
        if not message.get("more_body", False):
            return b"".join(chunks)


def _replay(body: bytes, receive):
    pending = [body]

    async def replay():
        if pending:
            return {"type": "http.request", "body": pending.pop(),
                    "more_body": False}
        return await receive()

    return replay


def _model_field(body: bytes) -> str | None:
    # The routes parse their body with json.loads too, so both read the
    # same model when a key repeats. A model that is not a string names no
    # alias, and the route refuses it.
    try:
        doc = json.loads(body)
    except ValueError:
        return None
    model = doc.get("model") if isinstance(doc, dict) else None
    return model if isinstance(model, str) else None


def _unknown_model(path: str, model: str) -> dict:
    """The body the server answers for a model id it does not have."""
    from .. import bridge_vlm as serving
    cfg = serving._SERVER_CFG
    exc = serving.ModelNotFound(model, cfg.models if cfg is not None else {})
    return _error_content(path, 404, "model_not_found", str(exc),
                          available_models=exc.available)


def _filter_models(body: bytes, hide) -> bytes:
    try:
        doc = json.loads(body)
    except ValueError:
        return body
    if not isinstance(doc, dict) or not isinstance(doc.get("data"), list):
        return body
    doc["data"] = [e for e in doc["data"]
                   if not (isinstance(e, dict) and hide(e.get("id")))]
    return json.dumps(doc).encode()


def _models_send(send, hide):
    """``send`` with the hidden aliases taken out of a model list."""
    start: dict = {}
    chunks: list = []

    async def filtered(message):
        if message["type"] == "http.response.start":
            start.update(message)
            return
        if message["type"] != "http.response.body":
            await send(message)
            return
        chunks.append(message.get("body", b""))
        if message.get("more_body", False):
            return
        body = b"".join(chunks)
        if start.get("status") == 200:
            body = _filter_models(body, hide)
        headers = [(k, v) for k, v in start.get("headers", [])
                   if k.lower() != b"content-length"]
        headers.append((b"content-length", str(len(body)).encode()))
        await send({**start, "headers": headers})
        await send({"type": "http.response.body", "body": body})

    return filtered


class _SessionApp:
    """The ASGI app of one session socket: the server's app under the
    session scope."""

    def __init__(self, sessions: _Sessions, session: _Session):
        self.sessions = sessions
        self.session = session
        self.app = importlib.import_module("mlx_vlm.server.app").app

    async def __call__(self, scope, receive, send):
        from fastapi.responses import JSONResponse

        if scope["type"] != "http":
            return
        path = scope["path"]
        if path not in SESSION_PATHS:
            await JSONResponse(status_code=404, content=_error_content(
                path, 404, "invalid_request_error",
                f"{path} is not available on a launch session socket"))(
                    scope, receive, send)
            return
        scope = {**scope, SESSION_SCOPE_KEY: self.session.id}
        method = scope.get("method")
        if method == "POST" and path in _ALIAS_PATHS:
            body = await _read_body(receive)
            if body is None:
                return
            model = _model_field(body)
            if model is not None and self.sessions.hidden(model, self.session):
                await JSONResponse(status_code=404, content=_unknown_model(
                    path, model))(scope, receive, send)
                return
            receive = _replay(body, receive)
        elif method == "GET" and path in _MODELS_PATHS:
            send = _models_send(
                send, lambda mid: self.sessions.hidden(mid, self.session))
        await self.app(scope, receive, send)


# The session endpoints

def _refused(path: str):
    from fastapi.responses import JSONResponse
    return JSONResponse(status_code=404, content=_error_content(
        path, 404, "invalid_request_error",
        f"{path} is not available on a launch session socket"))


def _bad_request(path: str, message: str):
    from fastapi.responses import JSONResponse
    return JSONResponse(status_code=400, content=_error_content(
        path, 400, "invalid_request_error", message))


async def _open_session(request: Request):
    from fastapi.responses import JSONResponse

    path = request.url.path
    if request.scope.get(SESSION_SCOPE_KEY) is not None:
        return _refused(path)
    try:
        body = await request.json()
    except ValueError:
        return _bad_request(path, "the body must be JSON")
    if not isinstance(body, dict) or set(body) != {"client", "assistants"}:
        return _bad_request(path, 'the body must hold exactly "client" and '
                                  '"assistants"')
    client, listed = body["client"], body["assistants"]
    if not isinstance(client, str) or not isinstance(listed, list) \
            or not all(isinstance(a, str) for a in listed):
        return _bad_request(path, '"client" must be a string and '
                                  '"assistants" a list of strings')
    sessions = _STATE
    if sessions is None:
        return _refused(path)
    listed = list(dict.fromkeys(listed))
    allowed = [a for a in listed if a in sessions.tools]
    try:
        session = await sessions.start(client, frozenset(allowed))
    except (OSError, RuntimeError) as e:
        return JSONResponse(status_code=503, content=_error_content(
            path, 503, "server_error",
            f"cannot open a launch session socket: {e}"))
    return {"id": session.id, "socket": session.path,
            "assistants": {a: {"tools": list(sessions.tools[a])}
                           for a in allowed},
            "unknown": [a for a in listed if a not in sessions.tools]}


async def _close_session(session_id: str, request: Request):
    from fastapi.responses import JSONResponse, Response

    path = request.url.path
    if request.scope.get(SESSION_SCOPE_KEY) is not None:
        return _refused(path)
    if _STATE is None or _STATE.stop(session_id) is None:
        return JSONResponse(status_code=404, content=_error_content(
            path, 404, "invalid_request_error",
            f"no launch session {session_id!r}"))
    _log.info("launch session %s closed", session_id)
    return Response(status_code=204)


def stop_sessions() -> None:
    """Close every open session."""
    if _STATE is not None:
        _STATE.stop_all()


def install_session_sockets(cfg) -> None:
    """Add the session endpoints for the server that ``cfg`` describes.
    ``cfg.host`` and ``cfg.port`` must hold the resolved bind."""
    global _STATE
    stop_sessions()
    _STATE = _Sessions(cfg)
    app = importlib.import_module("mlx_vlm.server.app").app
    _remove_routes(app, ENDPOINT, ENDPOINT + "/{session_id}")
    app.add_api_route(ENDPOINT, _open_session, methods=["POST"],
                      include_in_schema=False)
    app.add_api_route(ENDPOINT + "/{session_id}", _close_session,
                      methods=["DELETE"], include_in_schema=False)


def prepare_session_sockets(host: str, port) -> None:
    """Remove the session sockets an earlier run left, and remove them again
    when the server stops. The serve path calls this before uvicorn starts."""
    clear_session_sockets(host, port)
    app = importlib.import_module("mlx_vlm.server.app").app
    original = app.router.lifespan_context

    @contextlib.asynccontextmanager
    async def lifespan(app_):
        async with original(app_) as state:
            try:
                yield state
            finally:
                stop_sessions()
                clear_session_sockets(host, port)

    app.router.lifespan_context = lifespan
