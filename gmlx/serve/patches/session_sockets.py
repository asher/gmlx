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

A session can also name the loopback ports its browser app serves pages
on. The app's backend calls the server through the session, so while the
session is open the origin guard refuses any loopback page on those ports on
the TCP listener, whatever loopback name the page uses.

Each socket has a uvicorn server of its own in the server's event loop,
over the same app. That server leaves the process signal handlers to the
main server. A socket is mode 0600, in a folder of mode 0700 that only the
server uses. The server removes its own session sockets when it stops, and
at start and at stop it removes those that no server listens on.
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import errno
import functools
import importlib
import inspect
import json
import logging
import os
import secrets
import socket
import stat
from pathlib import Path

import uvicorn
from fastapi import Request  # module-level so stringized annotations resolve

from gmlx.serve.session_paths import (ID_BYTES, SESSION_CONNECTIONS_MAX, SOCKET_NAME,
                                      SOCKET_NAME_LEN, SOCKET_PATH_MAX, owned_folder,
                                      socket_folders)

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

_BACKLOG = 2048
# Past this many open sessions, a new session closes the oldest one that
# has no open connection. A launch that dies before its DELETE leaves its
# session open, and each session server wakes ten times a second.
SESSIONS_MAX = 32
# The most web ports one session can name.
WEB_PORTS_MAX = 8

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
    def __init__(self, sid: str, client: str, path: str, allowed: frozenset,
                 web_ports: frozenset = frozenset()):
        self.id = sid
        self.client = client
        self.path = path
        self.allowed = allowed
        self.web_ports = web_ports
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

    async def start(self, client: str, allowed: frozenset,
                    web_ports: frozenset = frozenset()) -> _Session:
        folder = socket_folder(self.host, self.port)
        while len(self.open) >= SESSIONS_MAX:
            oldest = next((s for s in self.open.values() if s.idle()), None)
            if oldest is None:
                raise RuntimeError(f"{SESSIONS_MAX} launch sessions are open "
                                   "and each one has an open connection")
            _log.warning("launch session %s (%s) closed to make room for a "
                         "new session", oldest.id, oldest.client)
            self.stop(oldest.id)
        sid = secrets.token_hex(ID_BYTES)
        session = _Session(sid, client, str(folder / f"{sid}.sock"), allowed,
                           web_ports)
        sock = _listen(session.path)
        config = uvicorn.Config(
            _SessionApp(self, session), lifespan="off", ws="none",
            proxy_headers=False, server_header=False, log_config=None,
            backlog=_BACKLOG, limit_concurrency=SESSION_CONNECTIONS_MAX)
        session.server = _SessionServer(config)
        logging.getLogger("uvicorn.error").addFilter(_LOG_FILTER)
        session.task = asyncio.get_running_loop().create_task(
            _serve(session.server, sock))
        session.task.add_done_callback(
            lambda task, s=session: self._ended(task, s))
        self.open[sid] = session
        _log.info("launch session %s opened for %r, assistants: %s", sid,
                  client, ", ".join(sorted(allowed)) or "none")
        if web_ports:
            _log.info("launch session %s refuses the loopback pages on port %s "
                      "on the TCP listener", sid,
                      ", ".join(str(p) for p in sorted(web_ports)))
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


def session_web_ports() -> frozenset[int]:
    """The web ports that the open sessions name."""
    if _STATE is None:
        return frozenset()
    return frozenset().union(*(s.web_ports for s in _STATE.open.values()))


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

def _fits(folder: Path) -> bool:
    return len(os.fsencode(folder)) + 1 + SOCKET_NAME_LEN <= SOCKET_PATH_MAX


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
        if not owned_folder(folder):
            raise OSError(f"{folder} is not a folder of this user")
        os.chmod(folder, 0o700)
        return folder
    raise OSError("no session socket folder gives a path shorter than "
                  f"{SOCKET_PATH_MAX + 1} bytes. Set TMPDIR to a shorter path.")


def _listened(path) -> bool:
    """Whether a server listens on the Unix socket ``path``. A connect that
    the server accepts, or that waits because its queue is full, means it
    does."""
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        sock.setblocking(False)
        rc = sock.connect_ex(str(path))
    except OSError as e:
        rc = e.errno
    finally:
        sock.close()
    return rc not in (errno.ECONNREFUSED, errno.ENOENT)


def clear_session_sockets(host: str, port) -> None:
    """Remove the session sockets of the server at ``host:port`` that no
    server listens on. A second server on the same bind runs this before
    its bind fails, and the sockets of the server that holds the bind must
    stay."""
    for folder in socket_folders(host, port):
        if not owned_folder(folder):
            continue
        try:
            names = os.listdir(folder)
        except OSError:
            continue
        for name in names:
            if SOCKET_NAME.match(name) and not _listened(folder / name):
                _unlink_socket(folder / name)


# The app of a session socket

class _TooLarge(Exception):
    pass


async def _read_body(receive, limit: int) -> bytes | None:
    """The whole request body, or None when the client went away. Raises
    :class:`_TooLarge` once the body passes ``limit`` bytes."""
    chunks, size = [], 0
    while True:
        message = await receive()
        if message["type"] == "http.disconnect":
            return None
        chunk = message.get("body", b"")
        size += len(chunk)
        if size > limit:
            raise _TooLarge
        chunks.append(chunk)
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


async def _unknown_model(app, path: str, model: str, hide) -> bytes:
    """The body a session answers for a model id it does not serve, in the
    dialect of ``path``. It lists the ids the session's own model list
    shows, less the aliases that ``hide`` hides and ``model`` itself, which
    the list can hold when the route does not serve that kind of model. A
    hidden alias and an id the server lacks get the same answer, so the
    answer never tells that the alias exists."""
    from fastapi.responses import JSONResponse

    from .. import bridge_vlm as serving
    from ._common import _find_route
    route = _find_route(app, "/v1/models", "GET")
    try:
        payload = route.endpoint()
        if inspect.isawaitable(payload):
            payload = await payload
        ids = [e.get("id") for e in payload["data"] if isinstance(e, dict)]
    except Exception:                  # noqa: BLE001 - the answer is a 404 either way
        _log.exception("the model list for an unknown-model answer failed")
        ids = list(serving.resolved_models())
    visible = [i for i in dict.fromkeys(ids)
               if isinstance(i, str) and i != model and not hide(i)]
    exc = serving.ModelNotFound(model, visible)
    return JSONResponse(status_code=404, content=_error_content(
        path, 404, "model_not_found", str(exc), available_models=exc.available)).body


def _is_unknown_model(body: bytes) -> bool:
    """Whether a 404 body is the server's unknown-model answer, in either
    dialect. The Anthropic one has the generic ``not_found_error`` type, and
    only the unknown-model answer carries ``available_models`` in it."""
    try:
        doc = json.loads(body)
    except ValueError:
        return False
    err = doc.get("error") if isinstance(doc, dict) else None
    if not isinstance(err, dict):
        return False
    if err.get("type") == "model_not_found":
        return True
    return (doc.get("type") == "error" and err.get("type") == "not_found_error"
            and "available_models" in err)


def _unknown_send(app, send, path: str, model: str, hide):
    """``send`` with the server's own unknown-model answer replaced by the
    session's, which lists only what the session can reach."""
    start: dict = {}
    chunks: list = []

    async def replaced(message):
        if message["type"] == "http.response.start" and message.get("status") != 404:
            start.clear()
            await send(message)
            return
        if message["type"] == "http.response.start":
            start.update(message)
            return
        if not start or message["type"] != "http.response.body":
            await send(message)
            return
        chunks.append(message.get("body", b""))
        if message.get("more_body", False):
            return
        body = b"".join(chunks)
        if _is_unknown_model(body):
            body = await _unknown_model(app, path, model, hide)
        headers = [(k, v) for k, v in start.get("headers", [])
                   if k.lower() != b"content-length"]
        headers.append((b"content-length", str(len(body)).encode()))
        await send({**start, "headers": headers})
        await send({"type": "http.response.body", "body": body})

    return replaced


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
        from fastapi.responses import JSONResponse, Response

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
            # The alias check reads the body before the media gate does, so
            # it applies the gate's session ceiling itself.
            from starlette.datastructures import Headers

            from . import media_gate as mg
            limit = mg.body_limit(False, True)
            length = Headers(scope=scope).get("content-length", "")
            try:
                if length.isdigit() and int(length) > limit:
                    raise _TooLarge
                body = await _read_body(receive, limit)
            except _TooLarge:
                await JSONResponse(status_code=413, content=_error_content(
                    path, 413, "invalid_request_error",
                    mg._body_refusal(False, session=True, path=path)))(scope, receive, send)
                return
            if body is None:
                return
            model = _model_field(body)
            hide = functools.partial(self.sessions.hidden, session=self.session)
            if model is not None and hide(model):
                body = await _unknown_model(self.app, path, model, hide)
                await Response(body, status_code=404,
                               media_type="application/json")(scope, receive, send)
                return
            if model is not None:
                send = _unknown_send(self.app, send, path, model, hide)
            receive = _replay(body, receive)
        elif method == "GET" and path in _MODELS_PATHS:
            send = _models_send(
                send, lambda mid: self.sessions.hidden(mid, self.session))
        await self.app(scope, receive, send)


# The session endpoints

def _refused(path: str, where: str = "on a launch session socket"):
    from fastapi.responses import JSONResponse
    return JSONResponse(status_code=404, content=_error_content(
        path, 404, "invalid_request_error", f"{path} is not available {where}"))


def _from_page(request: Request) -> bool:
    """Whether a browser page sent ``request``. Launch sends no Origin, and
    no page has a reason to open or end a launch session. A page that opens
    one with web ports would lock other local pages out of the TCP port."""
    return request.headers.get("origin") is not None


def _bad_request(path: str, message: str):
    from fastapi.responses import JSONResponse
    return JSONResponse(status_code=400, content=_error_content(
        path, 400, "invalid_request_error", message))


_OPEN_KEYS = frozenset({"client", "assistants", "web_ports"})


def _web_ports_field(value) -> frozenset[int] | str:
    """The web ports of a session request, or the message that refuses
    them."""
    if not isinstance(value, list) or not all(
            type(p) is int and 0 < p < 65536 for p in value):
        return '"web_ports" must be a list of port numbers from 1 to 65535'
    if len(value) > WEB_PORTS_MAX:
        return f'"web_ports" may name at most {WEB_PORTS_MAX} ports'
    return frozenset(value)


async def _open_session(request: Request):
    from fastapi.responses import JSONResponse

    path = request.url.path
    if request.scope.get(SESSION_SCOPE_KEY) is not None:
        return _refused(path)
    if _from_page(request):
        return _refused(path, "to a browser page")
    try:
        body = await request.json()
    except ValueError:
        return _bad_request(path, "the body must be JSON")
    if not isinstance(body, dict) \
            or not {"client", "assistants"} <= set(body) <= _OPEN_KEYS:
        return _bad_request(path, 'the body must hold "client" and '
                                  '"assistants", and may hold "web_ports"')
    client, listed = body["client"], body["assistants"]
    if not isinstance(client, str) or not isinstance(listed, list) \
            or not all(isinstance(a, str) for a in listed):
        return _bad_request(path, '"client" must be a string and '
                                  '"assistants" a list of strings')
    web_ports = _web_ports_field(body.get("web_ports", []))
    if isinstance(web_ports, str):
        return _bad_request(path, web_ports)
    sessions = _STATE
    if sessions is None:
        return _refused(path)
    listed = list(dict.fromkeys(listed))
    allowed = [a for a in listed if a in sessions.tools]
    try:
        session = await sessions.start(client, frozenset(allowed), web_ports)
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
    if _from_page(request):
        return _refused(path, "to a browser page")
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
    """Remove the session sockets an earlier run left, and this run's own
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
