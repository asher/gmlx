"""Launch session sockets: the session endpoints, the scope of a session
socket, and socket cleanup. Each test runs the app in a real uvicorn server
on a loopback port, and the session sockets are real Unix sockets. CPU only:
the chat route is a stub and no model loads."""
from __future__ import annotations

import asyncio
import contextlib
import errno
import http.client
import importlib
import inspect
import json
import os
import shutil
import socket
import stat
import tempfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("mlx_vlm")

import uvicorn  # noqa: E402
from fastapi import Request as _Request  # noqa: E402
from fastapi.responses import StreamingResponse  # noqa: E402

import gmlx.assistant.mcp as talk_mcp  # noqa: E402
import gmlx.assistant.serve as aserve  # noqa: E402
import gmlx.serve.bridge_vlm as serving  # noqa: E402
import gmlx.serve.patches as sp  # noqa: E402
from gmlx.assistant.brain import Tool, ToolRegistry  # noqa: E402
from gmlx.config import build_config  # noqa: E402
from gmlx.serve.patches import _common as sp_common  # noqa: E402
from gmlx.serve.patches import hardening as sp_hardening  # noqa: E402
from gmlx.serve.patches import session_sockets as ss  # noqa: E402

_APP = importlib.import_module("mlx_vlm.server.app")
_SCHEMAS = importlib.import_module("mlx_vlm.server.schemas")
_KEY = "sekrit-key"

# Routes a session socket refuses, as the test app serves them on TCP.
_REFUSED = [("POST", "/v1/reload"), ("POST", "/unload"), ("POST", "/v1/keep"),
            ("POST", "/v1/cache/reset"), ("POST", "/cache/reset"),
            ("POST", "/v1/prewarm"), ("GET", "/v1/capacity/plan"),
            ("POST", "/v1/estimate"), ("GET", "/v1/metrics"),
            ("GET", "/metrics"), ("GET", "/docs"), ("GET", "/redoc"),
            ("GET", "/openapi.json")]


@pytest.fixture(autouse=True)
def _restore_app():
    app = _APP.app
    routes = sp_common._snapshot_routes(app)
    middleware = list(app.user_middleware)
    mw_kwargs = [(m, dict(getattr(m, "kwargs", {}) or {}))
                 for m in app.user_middleware]
    allowed = sp_hardening._allowed_origins
    handlers = dict(app.exception_handlers)
    lifespan = app.router.lifespan_context
    yield
    ss.stop_sessions()
    ss._STATE = None
    sp_common._restore_routes(app, routes)
    app.user_middleware[:] = middleware
    for m, kw in mw_kwargs:
        if getattr(m, "kwargs", None) is not None:
            m.kwargs.clear()
            m.kwargs.update(kw)
    sp_hardening._allowed_origins = allowed
    app.exception_handlers.clear()
    app.exception_handlers.update(handlers)
    app.router.lifespan_context = lifespan
    app.middleware_stack = None
    if hasattr(app.state, sp_hardening._AUTH_FLAG):
        delattr(app.state, sp_hardening._AUTH_FLAG)


@pytest.fixture
def short_dirs(monkeypatch):
    """A cache folder and a TMPDIR of this test's own, short enough for a
    socket path."""
    root = Path(os.path.realpath(tempfile.mkdtemp(prefix="gss-", dir="/tmp")))
    monkeypatch.setenv("XDG_CACHE_HOME", str(root / "c"))
    monkeypatch.setenv("TMPDIR", str(root / "t"))
    (root / "t").mkdir()
    yield root
    shutil.rmtree(root, ignore_errors=True)


def _free_socket() -> socket.socket:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    sock.listen(64)
    return sock


class _Live:
    """The app in a uvicorn server on a loopback port, on its own thread.
    The socket listens before the thread starts, so no wait is needed."""

    def __init__(self, sock: socket.socket, *, lifespan: str = "off"):
        self.port = sock.getsockname()[1]
        config = uvicorn.Config(_APP.app, lifespan=lifespan, ws="none",
                                log_config=None)
        self.server = uvicorn.Server(config)
        self.thread = threading.Thread(
            target=self.server.run, kwargs={"sockets": [sock]}, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.server.should_exit = True
        self.thread.join(30)


class _UnixConnection(http.client.HTTPConnection):
    def __init__(self, path: str):
        super().__init__("127.0.0.1", timeout=30)
        self.unix_path = path

    def connect(self):
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(30)
        sock.connect(self.unix_path)
        self.sock = sock


class _Answer:
    def __init__(self, status: int, body: bytes):
        self.status = status
        self.body = body

    def json(self):
        return json.loads(self.body)


def _request(conn, method, path, body=None, key=None, origin=None) -> _Answer:
    headers = {}
    if origin is not None:
        headers["origin"] = origin
    data = None
    if body is not None:
        data = json.dumps(body).encode()
        headers["content-type"] = "application/json"
    if key:
        headers["authorization"] = f"Bearer {key}"
    try:
        conn.request(method, path, body=data, headers=headers)
        resp = conn.getresponse()
        return _Answer(resp.status, resp.read())
    finally:
        conn.close()


def _tcp(live, method, path, body=None, key=None, origin=None) -> _Answer:
    conn = http.client.HTTPConnection("127.0.0.1", live.port, timeout=30)
    return _request(conn, method, path, body, key, origin)


def _unix(path, method, url, body=None, origin=None) -> _Answer:
    return _request(_UnixConnection(path), method, url, body, origin=origin)


def _chat(model, tools=None) -> dict:
    body = {"model": model, "stream": False,
            "messages": [{"role": "user", "content": "what time is it?"}]}
    if tools is not None:
        body["tools"] = tools
    return body


def _cfg(port, *, api_key=None) -> object:
    doc = {
        "server": {"model_dirs": ["/models"], "host": "127.0.0.1",
                   "port": port,
                   "assistants": {
                       "home": {"model": "m-a",
                                "mcp": [{"name": "clockd",
                                         "command": ["clockd"]}]},
                       "hidden": {"model": "m-a"},
                       "bare": {"model": "m-a", "mcp": []}}},
        "assistant": {"mcp": [{"name": "shared", "command": ["shared"]}]},
        "models": {"m-a": {"path": "/abs/a.gguf"}},
    }
    if api_key:
        doc["server"]["api_key"] = api_key
    return build_config(doc)


def _stub_chat(record):
    """The original chat handler. It streams a tool call to ``clock`` while
    the last message is the user's, then prose, and answers the unknown-model
    error for an id the config does not hold."""

    async def stub(request, http_request):
        record.append({"model": request.model,
                       "auth": http_request.headers.get("authorization"),
                       "tenant": http_request.headers.get("x-apc-tenant"),
                       "tenant_id": http_request.headers.get("x-tenant-id")})
        if request.model not in serving._SERVER_CFG.models:
            raise serving.ModelNotFound(request.model,
                                        serving._SERVER_CFG.models)
        if not request.stream:
            return _SCHEMAS.ChatResponse(choices=[_SCHEMAS.ChatChoice(
                finish_reason="stop", message=_SCHEMAS.ChatMessage(
                    role="assistant", content="stub answer"))])
        last = request.messages[-1]
        role = last.get("role") if isinstance(last, dict) else last.role
        if role == "user":
            delta = {"tool_calls": [{"index": 0, "id": "c1", "type": "function",
                                     "function": {"name": "clock",
                                                  "arguments": "{}"}}]}
            finish = "tool_calls"
        else:
            delta, finish = {"content": "It is noon."}, "stop"

        def events():
            for choice in ({"index": 0, "delta": delta},
                           {"index": 0, "delta": {}, "finish_reason": finish}):
                yield f"data: {json.dumps({'choices': [choice]})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(events(), media_type="text/event-stream")

    stub.__signature__ = inspect.Signature([
        inspect.Parameter("request", inspect.Parameter.POSITIONAL_OR_KEYWORD,
                          annotation=_SCHEMAS.ChatRequest),
        inspect.Parameter("http_request",
                          inspect.Parameter.POSITIONAL_OR_KEYWORD,
                          annotation=_Request),
    ])
    return stub


def _stub_other(record):
    """The Messages and Responses routes as the server answers them: the
    unknown-model error for an id the config does not hold, which the
    server's handler shapes for the route's dialect, and a stub answer
    otherwise. The real routes would try to fetch an unknown id."""

    async def stub(http_request):
        model = (await http_request.json()).get("model")
        record.append({"model": model, "path": http_request.url.path})
        if model not in serving._SERVER_CFG.models:
            raise serving.ModelNotFound(model, serving._SERVER_CFG.models)
        return {"stub": True}

    stub.__signature__ = inspect.Signature([inspect.Parameter(
        "http_request", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=_Request)])
    return stub


def _admin_stub(hits, path):
    async def endpoint():
        hits.append(path)
        return {"ok": path}
    return endpoint


class _Server:
    """A live server over the test app, with what the tests read back."""

    def __init__(self, live, cfg, chat, tools, admin):
        self.live, self.cfg = live, cfg
        self.chat, self.tools, self.admin = chat, tools, admin

    def tcp(self, method, path, body=None, key=None, origin=None) -> _Answer:
        return _tcp(self.live, method, path, body, key, origin)

    def open_session(self, assistants, key=None, web_ports=None) -> dict:
        body = {"client": "opencode", "assistants": assistants}
        if web_ports is not None:
            body["web_ports"] = web_ports
        r = self.tcp("POST", ss.ENDPOINT, body, key=key or self.cfg.api_key)
        assert r.status == 200, r.body
        return r.json()


@pytest.fixture
def server(monkeypatch, short_dirs):
    """Build the test app and serve it. ``server(api_key=...)``."""
    started = []

    def build(*, api_key=None, lifespan="off", prepare=False):
        sock = _free_socket()
        cfg = _cfg(sock.getsockname()[1], api_key=api_key)
        monkeypatch.setattr(serving, "_SERVER_CFG", cfg)
        app = _APP.app
        chat: list = []
        sp_common._remove_routes(app, *sp_common._CHAT_PATHS)
        for path in sp_common._CHAT_PATHS:
            app.add_api_route(path, _stub_chat(chat), methods=["POST"])
        others = sorted(ss._ALIAS_PATHS - set(sp_common._CHAT_PATHS))
        sp_common._remove_routes(app, *others)
        for path in others:
            app.add_api_route(path, _stub_other(chat), methods=["POST"])
        admin: list = []
        sp_common._remove_routes(app, *[p for _, p in _REFUSED])
        for method, path in _REFUSED:
            app.add_api_route(path, _admin_stub(admin, path),
                              methods=[method])
        tools: list = []
        registry = ToolRegistry([Tool(
            name="clock", description="time",
            call=lambda args: tools.append(args) or "12:00")])
        monkeypatch.setattr(talk_mcp, "connect_servers",
                            lambda servers, **kw: (None, registry, []))
        sp.install_models_endpoint_override()
        sp.install_resolver_error_handlers()
        aserve.install_assistant_serve(cfg)
        ss.install_session_sockets(cfg)
        sp_hardening.install_api_key_auth(cfg.api_key)
        sp_hardening.install_origin_guard(cfg.cors_origins)
        if prepare:
            ss.prepare_session_sockets(cfg.host, cfg.port)
        live = _Live(sock, lifespan=lifespan)
        started.append(live)
        return _Server(live, cfg, chat, tools, admin)

    yield build
    for live in started:
        live.stop()


# The finding and its fix

def test_alias_tool_loop_runs_over_tcp_and_not_through_a_session_without_it(
        server):
    srv = server()
    # Today's relay path: loopback TCP to a keyless server runs the alias's
    # tool loop, and its MCP tool.
    r = srv.tcp("POST", "/v1/chat/completions", _chat("hidden"))
    assert r.status == 200, r.body
    assert r.json()["choices"][0]["message"]["content"] == "It is noon."
    assert srv.tools == [{}]
    # Through a session socket whose list lacks the alias, the alias is an
    # unknown model and the tool never runs.
    session = srv.open_session(["home"])
    srv.tools.clear()
    srv.chat.clear()
    r = _unix(session["socket"], "POST", "/v1/chat/completions",
              _chat("hidden"))
    assert r.status == 404, r.body
    assert r.json()["error"]["type"] == "model_not_found"
    assert srv.tools == [] and srv.chat == []


# The session endpoint

def test_open_session_answers_the_contract_body(server):
    srv = server()
    body = srv.open_session(["home", "bare", "hidden", "nope", "home"])
    assert set(body) == {"id", "socket", "assistants", "unknown"}
    assert body["assistants"] == {"home": {"tools": ["clockd"]},
                                  "bare": {"tools": []},
                                  "hidden": {"tools": ["shared"]}}
    assert body["unknown"] == ["nope"]
    path = body["socket"]
    assert os.path.isabs(path) and len(os.fsencode(path)) < 104
    assert stat.S_ISSOCK(os.lstat(path).st_mode)
    assert stat.S_IMODE(os.lstat(path).st_mode) == 0o600
    folder = os.lstat(os.path.dirname(path))
    assert stat.S_IMODE(folder.st_mode) == 0o700
    assert folder.st_uid == os.getuid()
    # The check launch runs on the path accepts it, for this port only.
    from gmlx.serve.session_paths import socket_refusal
    assert socket_refusal(path, srv.live.port) is None
    assert socket_refusal(path, srv.live.port + 1) is not None


@pytest.mark.parametrize("body", [
    {"client": "opencode"},
    {"assistants": []},
    {"client": "opencode", "assistants": [], "extra": 1},
    {"client": 3, "assistants": []},
    {"client": "opencode", "assistants": "home"},
    {"client": "opencode", "assistants": [1]},
    ["opencode"],
    # The body launch probes with: a 400 tells it the server offers sessions.
    {"probe": True},
    {"client": "opencode", "assistants": [], "web_ports": 3000},
    {"client": "opencode", "assistants": [], "web_ports": ["3000"]},
    {"client": "opencode", "assistants": [], "web_ports": [True]},
    {"client": "opencode", "assistants": [], "web_ports": [0]},
    {"client": "opencode", "assistants": [], "web_ports": [65536]},
    {"client": "opencode", "assistants": [], "web_ports": [3000.0]},
    {"client": "opencode", "assistants": [],
     "web_ports": list(range(1, ss.WEB_PORTS_MAX + 2))},
    {"web_ports": [3000]},
    {"client": "opencode", "assistants": [], "project": 3},
])
def test_open_session_refuses_a_bad_body(server, body):
    srv = server()
    r = srv.tcp("POST", ss.ENDPOINT, body)
    assert r.status == 400, r.body
    assert ss._STATE.open == {}


def test_a_session_sets_the_prompt_cache_tenant_of_its_client_and_project(server):
    """The prompt cache keys a request by its tenant headers. A guest that
    could choose them could read or fill another client's cache."""
    srv = server()

    def tenant_of(body):
        path = srv.tcp("POST", ss.ENDPOINT, body).json()["socket"]
        conn = _UnixConnection(path)
        conn.request("POST", "/v1/chat/completions", body=json.dumps(_chat("m-a")),
                     headers={"content-type": "application/json",
                              "x-apc-tenant": "victim", "X-Tenant-Id": "victim"})
        assert conn.getresponse().status == 200
        conn.close()
        assert srv.chat[-1]["tenant"] == srv.chat[-1]["tenant_id"]
        return srv.chat[-1]["tenant"]

    first = tenant_of({"client": "opencode", "assistants": [], "project": "p-1"})
    again = tenant_of({"client": "opencode", "assistants": [], "project": "p-1"})
    others = [tenant_of({"client": "opencode", "assistants": [], "project": "p-2"}),
              tenant_of({"client": "pi", "assistants": [], "project": "p-1"}),
              tenant_of({"client": "opencode", "assistants": []}),
              tenant_of({"client": "opencode", "assistants": []})]
    assert first == again == ss.session_tenant("opencode", "p-1", "any")
    assert len({first, *others, "victim"}) == 6
    # The TCP listener keeps the headers the client sends.
    srv.tcp("POST", "/v1/chat/completions", _chat("m-a"))
    assert srv.chat[-1]["tenant"] is None


def test_a_session_takes_up_to_the_most_web_ports(server):
    srv = server()
    srv.open_session([], web_ports=list(range(1, ss.WEB_PORTS_MAX + 1)))
    assert ss.session_web_ports() == frozenset(range(1, ss.WEB_PORTS_MAX + 1))


# The browser app of a session

_PAGE = "http://127.0.0.1:18123"


def _refusal(r: _Answer) -> str:
    assert r.status == 403, r.body
    assert r.json()["error"]["type"] == "origin_not_allowed"
    return r.json()["error"]["message"]


@pytest.mark.parametrize("api_key", [None, _KEY])
def test_a_session_page_is_refused_on_tcp_while_the_session_is_open(server, api_key):
    srv = server(api_key=api_key)
    assert srv.tcp("GET", "/v1/models", key=api_key, origin=_PAGE).status == 200
    session = srv.open_session([], web_ports=[18123])
    # Every loopback name the page can load itself under, and with the key
    # too, since the page is the app's and not the user's.
    for origin in (_PAGE, "http://localhost:18123", "HTTP://LOCALHOST:18123",
                   "http://[::1]:18123", "http://[::ffff:127.0.0.1]:18123",
                   "http://127.0.0.2:18123", "https://127.0.0.1:18123"):
        message = _refusal(srv.tcp("GET", "/v1/models", key=api_key, origin=origin))
        assert "launch container session" in message and "port 18123" in message
        _refusal(srv.tcp("POST", "/v1/chat/completions", _chat("m-a"), key=api_key,
                         origin=origin))
    # A preflight is refused too, so the browser sends nothing after it.
    conn = http.client.HTTPConnection("127.0.0.1", srv.live.port, timeout=30)
    conn.request("OPTIONS", "/v1/chat/completions", headers={
        "origin": _PAGE, "access-control-request-method": "POST"})
    assert conn.getresponse().status == 403
    conn.close()
    # Loopback pages on other ports, and clients that send no origin, pass.
    assert srv.tcp("GET", "/v1/models", key=api_key,
                   origin="http://127.0.0.1:18124").status == 200
    assert srv.tcp("GET", "/v1/models", key=api_key).status == 200
    # The session's own socket answers the page's origin.
    assert _unix(session["socket"], "GET", "/v1/models", origin=_PAGE).status == 200
    assert srv.tcp("DELETE", f"{ss.ENDPOINT}/{session['id']}",
                   key=api_key).status == 204
    assert "until it ended" in _refusal(srv.tcp("GET", "/v1/models", key=api_key,
                                                origin=_PAGE))


def _past_the_grace(monkeypatch):
    real = ss._now
    monkeypatch.setattr(ss, "_now", lambda: real() + ss.WEB_PORT_GRACE + 1)


def test_a_session_page_stays_refused_for_a_grace_after_the_session_ends(
        server, monkeypatch):
    """The app's page can stay open in a tab after its session ends, and on
    a keyless server it would then reach every route the session hid."""
    srv = server()
    session = srv.open_session([], web_ports=[18123])
    srv.tcp("DELETE", f"{ss.ENDPOINT}/{session['id']}")
    for origin in (_PAGE, "http://localhost:18123"):
        message = _refusal(srv.tcp("POST", "/v1/chat/completions", _chat("hidden"),
                                   origin=origin))
        assert "until it ended" in message and "port 18123" in message
        assert "15 minutes" in message and "Close the app's browser tabs." in message
    assert srv.tools == [] and srv.chat == []
    assert srv.tcp("GET", "/v1/models", origin="http://127.0.0.1:18124").status == 200
    _past_the_grace(monkeypatch)
    assert ss.ended_web_ports() == frozenset()
    assert srv.tcp("GET", "/v1/models", origin=_PAGE).status == 200


def test_an_evicted_session_page_stays_refused(server, monkeypatch):
    monkeypatch.setattr(ss, "SESSIONS_MAX", 1)
    srv = server()
    srv.open_session([], web_ports=[18123])
    srv.open_session([])                           # closes the first, which is idle
    assert ss.session_web_ports() == frozenset()
    assert "until it ended" in _refusal(srv.tcp("GET", "/v1/models", origin=_PAGE))


@pytest.mark.parametrize("stopped", [True, False], ids=["stopped", "killed"])
def test_a_restarted_server_keeps_refusing_session_pages(server, monkeypatch, stopped):
    """The next server on the same bind reads the record. A server that was
    killed never ended its sessions, so their grace starts at the restart."""
    srv = server()
    srv.open_session([], web_ports=[18123])
    srv.open_session([], web_ports=[18125])
    old = ss._STATE
    if stopped:
        ss.stop_sessions()
    ss._STATE = None
    try:
        ss.install_session_sockets(srv.cfg)
        assert ss.session_web_ports() == frozenset()
        assert ss.ended_web_ports() == {18123, 18125}
        assert "until it ended" in _refusal(srv.tcp("GET", "/v1/models", origin=_PAGE))
        _past_the_grace(monkeypatch)
        assert srv.tcp("GET", "/v1/models", origin=_PAGE).status == 200
    finally:
        assert old is not None
        old.stop_all()


def test_the_grace_after_a_crash_starts_once(server, monkeypatch):
    """A crashed server leaves its open ports in the record. The first start
    after it starts their grace, and the starts after the grace refuse
    nothing."""
    srv = server()
    folder = ss.socket_folder(srv.cfg.host, srv.cfg.port)
    record = folder / ss._WEB_PORTS_RECORD
    record.write_text(json.dumps({"open": [18123], "ended": {}}))
    ss.install_session_sockets(srv.cfg)
    assert ss.ended_web_ports() == {18123}
    doc = json.loads(record.read_text())
    assert doc["open"] == [] and list(doc["ended"]) == ["18123"]
    _past_the_grace(monkeypatch)
    ss.install_session_sockets(srv.cfg)
    assert ss.ended_web_ports() == frozenset()
    assert srv.tcp("GET", "/v1/models", origin=_PAGE).status == 200


def test_a_start_keeps_the_record_of_a_live_server_on_the_same_bind(server):
    """A second server on the same bind reads the record before its bind
    fails. The open ports of the live server stay open in the record."""
    srv = server()
    folder = ss.socket_folder(srv.cfg.host, srv.cfg.port)
    record = folder / ss._WEB_PORTS_RECORD
    doc = {"open": [18123], "ended": {}}
    record.write_text(json.dumps(doc))
    live = ss._listen(str(folder / "0123456789ab.sock"))
    try:
        ss.install_session_sockets(srv.cfg)
        assert ss.ended_web_ports() == {18123}
        assert json.loads(record.read_text()) == doc
    finally:
        live.close()
        ss._unlink_socket(folder / "0123456789ab.sock")


@pytest.mark.parametrize("record", [b"not json", b"[]", b'{"open": "18123"}',
                                    b'{"open": [0, 70000], "ended": {"x": 1}}'])
def test_a_bad_web_ports_record_refuses_nothing(server, record):
    srv = server()
    folder = ss.socket_folder(srv.cfg.host, srv.cfg.port)
    (folder / ss._WEB_PORTS_RECORD).write_bytes(record)
    ss.install_session_sockets(srv.cfg)
    assert ss.ended_web_ports() == frozenset()
    assert srv.tcp("GET", "/v1/models", origin=_PAGE).status == 200


def test_a_default_port_page_is_matched_by_its_port(server):
    srv = server()
    srv.open_session([], web_ports=[80])
    assert "port 80" in _refusal(srv.tcp("GET", "/v1/models", origin="http://localhost"))
    assert srv.tcp("GET", "/v1/models", origin="https://localhost").status == 200


def test_a_session_page_stays_refused_while_any_session_names_it(server):
    srv = server()
    first = srv.open_session([], web_ports=[18123])
    srv.open_session([], web_ports=[18123])
    srv.tcp("DELETE", f"{ss.ENDPOINT}/{first['id']}")
    _refusal(srv.tcp("GET", "/v1/models", origin=_PAGE))


def test_a_session_without_web_ports_refuses_no_page(server):
    srv = server()
    srv.open_session(["home"])
    assert ss.session_web_ports() == frozenset()
    assert srv.tcp("GET", "/v1/models", origin=_PAGE).status == 200


def test_keyed_server_endpoint_refuses_a_request_without_the_key(server):
    srv = server(api_key=_KEY)
    body = {"client": "opencode", "assistants": ["home"]}
    assert srv.tcp("POST", ss.ENDPOINT, body).status == 401
    assert srv.tcp("POST", ss.ENDPOINT, body, key="wrong").status == 401
    assert srv.tcp("DELETE", ss.ENDPOINT + "/x").status == 401
    assert ss._STATE.open == {}
    assert srv.tcp("POST", ss.ENDPOINT, body, key=_KEY).status == 200


def test_delete_removes_the_socket(server):
    srv = server()
    session = srv.open_session(["home"])
    path = session["socket"]
    assert _unix(path, "GET", "/health").status == 200
    r = srv.tcp("DELETE", f"{ss.ENDPOINT}/{session['id']}")
    assert r.status == 204, r.body
    assert not os.path.lexists(path)
    with pytest.raises(OSError) as err:
        _unix(path, "GET", "/health")
    assert err.value.errno in (errno.ENOENT, errno.ECONNREFUSED)
    assert srv.tcp("DELETE", f"{ss.ENDPOINT}/{session['id']}").status == 404


def test_sessions_are_separate(server):
    srv = server()
    home = srv.open_session(["home"])
    other = srv.open_session(["hidden"])
    assert home["socket"] != other["socket"]
    assert _unix(home["socket"], "POST", "/v1/chat/completions",
                 _chat("hidden")).status == 404
    assert _unix(other["socket"], "POST", "/v1/chat/completions",
                 _chat("hidden")).status == 200


# The scope of a session socket

@pytest.mark.parametrize("api_key", [None, _KEY])
def test_listed_alias_runs_its_tool_loop_on_the_socket(server, api_key):
    srv = server(api_key=api_key)
    session = srv.open_session(["home"])
    r = _unix(session["socket"], "POST", "/v1/chat/completions",
              _chat("home"))
    assert r.status == 200, r.body
    body = r.json()
    assert body["model"] == "home"
    assert body["choices"][0]["message"]["content"] == "It is noon."
    assert srv.tools == [{}]
    # The loop's rounds re-enter over TCP, with the server's own key.
    assert [c["model"] for c in srv.chat] == ["m-a", "m-a"]
    expect = f"Bearer {api_key}" if api_key else None
    assert [c["auth"] for c in srv.chat] == [expect, expect]


def test_listed_alias_with_client_tools_passes_through(server):
    srv = server()
    session = srv.open_session(["home"])
    tools = [{"type": "function", "function": {"name": "t"}}]
    r = _unix(session["socket"], "POST", "/v1/chat/completions",
              _chat("home", tools))
    assert r.status == 200, r.body
    assert r.json()["choices"][0]["message"]["content"] == "stub answer"
    assert srv.tools == [] and [c["model"] for c in srv.chat] == ["m-a"]


@pytest.mark.parametrize("model", [["hidden"], {"id": "hidden"}, 3, None])
def test_a_model_that_is_not_a_string_passes_to_the_route(server, model):
    srv = server()
    path = srv.open_session([])["socket"]
    body = {**_chat("x"), "model": model}
    r = _unix(path, "POST", "/v1/chat/completions", body)
    assert r.status != 500, r.body
    assert srv.tools == []


def test_launch_opens_renews_and_ends_a_session_through_the_real_route(server):
    """The launch side's request and its checks of the reply, against the
    server's own session route rather than the launch tests' fake server."""
    from gmlx.commands import launch_container as lc
    from gmlx.serve.session_paths import socket_refusal
    srv = server(api_key="k")
    base = f"http://127.0.0.1:{srv.live.port}/v1"
    assert lc.sessions_offered(base, "k") is True
    launch_side = lc.ServerSession(base, "k", "opencode", ["home", "nope"])
    first = launch_side.open()
    assert socket_refusal(first, srv.live.port) is None
    assert "home" in launch_side.allowed and launch_side.unknown == ["nope"]
    assert _unix(first, "GET", "/health").status == 200
    second = launch_side.renew()
    assert second != first and socket_refusal(second, srv.live.port) is None
    assert launch_side.refused is None
    deadline = time.monotonic() + 5
    while os.path.exists(first):                  # the old session ends in the background
        assert time.monotonic() < deadline, "the old session was not ended"
        time.sleep(0.02)
    launch_side.close()
    assert not os.path.exists(second)


def test_a_launch_that_replaces_its_evicted_session_closes_no_other(server, monkeypatch):
    """Past the most sessions, a new session closes the oldest idle one. The
    launch of that session asks for a new one within seconds. If that closed
    another session, the running launches would close each other's
    sessions in turn and never stop."""
    from gmlx.commands import launch_container as lc
    monkeypatch.setattr(ss, "SESSIONS_MAX", 1)
    srv = server()
    base = f"http://127.0.0.1:{srv.live.port}/v1"
    a = lc.ServerSession(base, None, "opencode", [], project="a-1")
    b = lc.ServerSession(base, None, "opencode", [], project="b-2")
    a.open()
    b.open()                                       # closes a's session
    assert list(ss._STATE.open) == [b.id]
    assert a.renew() is not None
    assert set(ss._STATE.open) == {a.id, b.id}
    try:
        # A session the server did not close to make room is no reason to
        # keep one more session: the oldest idle session closes.
        body = {"client": "opencode", "assistants": [], "replaces": "0123456789ab"}
        r = srv.tcp("POST", ss.ENDPOINT, body)
        assert r.status == 200 and list(ss._STATE.open) == [a.id, r.json()["id"]]
        r = srv.tcp("POST", ss.ENDPOINT, dict(body, replaces=1))
        assert r.status == 400
        assert r.json()["error"]["message"] == '"replaces" must be a string'
    finally:
        a.close()
        b.close()


def _sessions_past_the_most(monkeypatch, opens: int):
    """Sessions in an event loop of their own, with a limit of 2. Each
    launch whose session closes to make room opens a new session at once,
    as its relay does. Returns the launches, the sessions and the loop."""
    monkeypatch.setattr(ss, "SESSIONS_MAX", 2)
    loop = asyncio.new_event_loop()
    sessions = ss._Sessions(SimpleNamespace(host="127.0.0.1", port=18999))
    launches: dict[str, str] = {}
    closed_per_open = []
    for n in range(opens):
        before = set(sessions.open)
        launches[f"L{n}"] = loop.run_until_complete(
            sessions.start("pi", frozenset())).id
        closed = [name for name, sid in launches.items()
                  if sid in before and sid not in sessions.open]
        closed_per_open.append(len(closed))
        for name in closed:
            launches[name] = loop.run_until_complete(sessions.start(
                "pi", frozenset(), replaces=launches[name])).id
    return launches, sessions, loop, closed_per_open


def _close_sessions(sessions, loop) -> None:
    sessions.stop_all()
    tasks = asyncio.all_tasks(loop)
    for task in tasks:
        task.cancel()
    loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))
    loop.close()


def test_a_new_session_past_the_most_closes_one_idle_session(monkeypatch, short_dirs):
    """The renewals keep one session open for each running launch, so the
    count can pass the limit. A new launch then closes one session, and
    not every session over the limit."""
    launches, sessions, loop, closed_per_open = _sessions_past_the_most(monkeypatch, 6)
    try:
        assert closed_per_open == [0, 0, 1, 1, 1, 1]
        assert set(sessions.open) == set(launches.values())
        assert len(sessions.open) == 6
    finally:
        _close_sessions(sessions, loop)


def test_a_new_session_that_finds_no_idle_session_closes_none(monkeypatch, short_dirs):
    _launches, sessions, loop, _closed = _sessions_past_the_most(monkeypatch, 3)
    try:
        before = list(sessions.open)
        for session in sessions.open.values():
            monkeypatch.setattr(session, "idle", lambda: False)
        with pytest.raises(RuntimeError) as e:
            loop.run_until_complete(sessions.start("pi", frozenset()))
        assert str(e.value).startswith(
            "3 launch sessions are open, and each one has an open connection.")
        assert list(sessions.open) == before
    finally:
        _close_sessions(sessions, loop)


def test_a_launch_past_the_session_limit_exits_to_try_again(server, monkeypatch):
    """Each open session has a connection, so none can close to make room.
    The launch prints the server's step once and exits 75, as for every
    other refusal that clears by itself."""
    from gmlx.commands import launch as L
    from gmlx.commands import launch_container as lc
    srv = server()
    monkeypatch.setattr(ss, "SESSIONS_MAX", 1)
    monkeypatch.setattr(ss._Session, "idle", lambda self: False)
    base = f"http://127.0.0.1:{srv.live.port}/v1"
    first = lc.ServerSession(base, None, "pi", [])
    first.open()
    try:
        with pytest.raises(L.LaunchError) as e:
            lc.ServerSession(base, None, "pi", []).open()
        assert str(e.value) == (
            f"the server at {base} could not open a session (503): 1 launch sessions "
            "are open, and each one has an open connection. Wait for a request to "
            "end, or stop another launch, then launch again.")
        assert L.exit_code(e.value) == L.EXIT_TEMPFAIL
    finally:
        first.close()


def test_launch_opens_a_new_session_soon_after_its_session_ends(server, short_dirs):
    """The session ends without a failed client request, as after a server
    restart. The launch relay notices that its socket is gone and opens a
    new session, so the server refuses the app's pages as open again."""
    from gmlx.commands import launch_container as lc
    from gmlx.container import relay
    srv = server()
    base = f"http://127.0.0.1:{srv.live.port}/v1"
    launch_side = lc.ServerSession(base, None, "open-webui", [], web_ports=[18123])
    first = launch_side.open()
    loop = relay.RelayLoop()
    loop.start()
    try:
        relay.Relay(loop, str(short_dirs / "api.sock"), [first], name="gmlx api",
                    renew=launch_side.renew, check_every=0.05)
        srv.tcp("DELETE", f"{ss.ENDPOINT}/{launch_side.id}")
        assert ss.session_web_ports() == frozenset()
        deadline = time.monotonic() + 5
        while launch_side.socket == first:
            assert time.monotonic() < deadline, "no new session"
            time.sleep(0.02)
        assert ss.session_web_ports() == {18123}
        assert "serves its browser app" in _refusal(srv.tcp("GET", "/v1/models",
                                                            origin=_PAGE))
    finally:
        loop.stop()
        launch_side.close()


def test_the_alias_check_keeps_no_copy_of_the_body_while_the_route_runs():
    """The route reads the body again, so a reference held here doubles the
    memory a session body costs."""
    import asyncio
    import types

    seen = {}

    async def route(scope, receive, send):
        message = await receive()
        frame = inspect.currentframe()
        while frame is not None and frame.f_code is not ss._SessionApp.__call__.__code__:
            frame = frame.f_back
        seen.update(body=message["body"], held="body" in frame.f_locals)

    sessions = ss._Sessions(types.SimpleNamespace())
    app = ss._SessionApp(sessions, ss._Session("s1", "pi", "/x.sock", frozenset()))
    app.app = route
    raw = json.dumps(_chat("m-a")).encode()
    messages = [{"type": "http.request", "body": raw, "more_body": False}]

    async def receive():
        return messages.pop(0)

    async def send(message):
        pass

    scope = {"type": "http", "method": "POST", "path": "/v1/chat/completions",
             "headers": [(b"content-length", str(len(raw)).encode())]}
    asyncio.run(app(scope, receive, send))
    assert seen == {"body": raw, "held": False}


def test_a_chat_body_over_the_ceiling_is_refused_on_the_socket(server, monkeypatch):
    from gmlx.serve.patches import media_gate as mg
    monkeypatch.setattr(mg, "SESSION_BODY_MAX_BYTES", 1000)
    srv = server()
    path = srv.open_session([])["socket"]
    for route in ("/v1/chat/completions", "/messages"):
        r = _unix(path, "POST", route, {**_chat("x"), "pad": " " * 2000})
        assert r.status == 413, (route, r.body)
        message = r.json()["error"]["message"]
        assert "limit of a launch session" in message
        assert "new conversation" in message
    # A declared length over the ceiling is refused before the body is sent.
    conn = _UnixConnection(path)
    try:
        conn.putrequest("POST", "/v1/chat/completions")
        conn.putheader("content-type", "application/json")
        conn.putheader("content-length", str(mg.SESSION_BODY_MAX_BYTES + 1))
        conn.endheaders()
        assert conn.getresponse().status == 413
    finally:
        conn.close()
    assert srv.chat == []
    assert _unix(path, "POST", "/v1/chat/completions", _chat("m-a")).status == 200
    # The TCP listener keeps its own, larger ceiling.
    assert srv.tcp("POST", "/v1/chat/completions",
                   {**_chat("m-a"), "pad": " " * 2000}).status == 200


def test_a_session_socket_serves_a_capped_number_of_connections(server, monkeypatch):
    """The socket serves a request on each connection the relay can hold.
    Once more connections are open, it answers 503 without reading a body,
    so held connections cannot make the server hold more bodies."""
    monkeypatch.setattr(ss, "SESSION_CONNECTIONS_MAX", 2)
    srv = server()
    session = srv.open_session([])
    path = session["socket"]
    state = ss._STATE.open[session["id"]].server.server_state
    held = []

    def hold():
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.connect(path)
        held.append(s)
        deadline = time.monotonic() + 5
        while len(state.connections) < len(held):        # the server accepted it
            assert time.monotonic() < deadline
            time.sleep(0.02)

    def health_until(status):
        deadline = time.monotonic() + 5
        while (got := _unix(path, "GET", "/health").status) != status:
            assert time.monotonic() < deadline, got
            time.sleep(0.02)

    try:
        hold()
        # The second of the relay's two connections is served.
        assert _unix(path, "GET", "/health").status == 200
        hold()
        health_until(503)
    finally:
        for s in held:
            s.close()
    health_until(200)


def test_the_launch_relay_gets_no_503_from_a_session_socket(server, short_dirs):
    """Bursts of more clients than the relay's cap wait in the relay's
    listen queue, and none of them gets the session socket's 503."""
    from collections import Counter

    from gmlx.container import relay
    from gmlx.serve.session_paths import SESSION_CONNECTIONS_MAX
    srv = server()
    session = srv.open_session([])
    loop = relay.RelayLoop()
    loop.start()
    try:
        path = str(short_dirs / "api.sock")
        relay.Relay(loop, path, [session["socket"]], name="gmlx api",
                    idle_until_head=True, max_connections=SESSION_CONNECTIONS_MAX)
        clients = 3 * SESSION_CONNECTIONS_MAX
        for _ in range(3):
            start, got = threading.Barrier(clients), Counter()

            def ask():
                start.wait()
                got[_unix(path, "GET", "/health").status] += 1
            threads = [threading.Thread(target=ask) for _ in range(clients)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(60)
            assert got == {200: clients}
    finally:
        loop.stop()


@pytest.mark.parametrize("api_key", [None, _KEY])
def test_unlisted_alias_is_an_unknown_model(server, api_key):
    srv = server(api_key=api_key)
    path = srv.open_session(["home"])["socket"]
    hidden = _unix(path, "POST", "/v1/chat/completions", _chat("hidden"))
    unknown = _unix(path, "POST", "/v1/chat/completions", _chat("nope"))
    assert hidden.status == unknown.status == 404
    assert hidden.body.replace(b"hidden", b"nope") == unknown.body
    # The answer lists what the session's model list shows.
    listed = [e["id"] for e in _unix(path, "GET", "/v1/models").json()["data"]]
    available = unknown.json()["error"]["available_models"]
    assert "home" in available and "hidden" not in available
    assert set(available) <= set(listed)
    assert srv.tools == [] and [c["model"] for c in srv.chat] == ["nope"]
    # The other surfaces answer the same 404 in their own shape, where TCP
    # names the alias as chat-completions only. On each, a hidden alias and
    # an id the server lacks get the same answer.
    for route in ("/v1/messages", "/messages/count_tokens", "/v1/responses",
                  "/responses/input_tokens", "/chat/completions"):
        for stream in (False, True):
            r = _unix(path, "POST", route, {**_chat("hidden"), "stream": stream})
            assert r.status == 404, (route, r.body)
            assert "unknown model id 'hidden'" in r.body.decode(), route
            other = _unix(path, "POST", route, {**_chat("nope"), "stream": stream})
            assert r.body.replace(b"hidden", b"nope") == other.body, (route, other.body)
    messages = _unix(path, "POST", "/v1/messages", _chat("nope")).json()
    assert messages["type"] == "error"
    assert messages["error"]["type"] == "not_found_error"
    assert "home" in messages["error"]["available_models"]
    assert "hidden" not in messages["error"]["available_models"]
    tcp = srv.tcp("POST", "/v1/messages", _chat("hidden"), key=api_key)
    assert tcp.status == 400


def test_an_unknown_model_answer_never_lists_the_id_it_refuses(monkeypatch):
    """A route that does not serve a listed id, such as a speech model on the
    chat route, calls it unknown, so the list it gives leaves it out."""
    import asyncio

    from fastapi import FastAPI

    app = FastAPI()
    app.add_api_route("/v1/models", lambda: {"data": [{"id": "m-a"}, {"id": "kokoro"}]},
                      methods=["GET"])
    body = asyncio.run(ss._unknown_model(app, "/v1/chat/completions", "kokoro",
                                         lambda mid: False))
    err = json.loads(body)["error"]
    assert err["type"] == "model_not_found"
    assert err["available_models"] == ["m-a"]
    assert err["message"] == "unknown model id 'kokoro'; available: ['m-a']"


def test_models_lists_only_the_listed_aliases(server):
    srv = server(api_key=_KEY)
    path = srv.open_session(["home", "bare"])["socket"]
    for route in ("/v1/models", "/models"):
        r = _unix(path, "GET", route)
        assert r.status == 200, r.body
        ids = [e["id"] for e in r.json()["data"]]
        assert "home" in ids and "bare" in ids
        assert "hidden" not in ids
    tcp = srv.tcp("GET", "/v1/models", key=_KEY).json()["data"]
    assert "hidden" in [e["id"] for e in tcp]


@pytest.mark.parametrize("api_key", [None, _KEY])
def test_refused_routes_answer_404_on_the_socket_only(server, api_key):
    srv = server(api_key=api_key)
    path = srv.open_session(["home"])["socket"]
    for method, route in _REFUSED:
        r = _unix(path, method, route, {} if method == "POST" else None)
        assert r.status == 404, (route, r.body)
        assert "not available on a launch session socket" in \
            r.body.decode(), route
    assert srv.admin == []
    for method, route in _REFUSED:
        r = srv.tcp(method, route, {} if method == "POST" else None,
                    key=api_key)
        assert r.status == 200, (route, r.body)
    assert srv.admin == [route for _, route in _REFUSED]


def test_session_endpoints_are_refused_on_the_socket(server):
    srv = server()
    session = srv.open_session(["home"])
    path = session["socket"]
    r = _unix(path, "POST", ss.ENDPOINT,
              {"client": "opencode", "assistants": ["hidden"]})
    assert r.status == 404, r.body
    assert _unix(path, "DELETE", f"{ss.ENDPOINT}/{session['id']}").status \
        == 404
    assert list(ss._STATE.open) == [session["id"]]
    assert os.path.lexists(path)


@pytest.mark.parametrize("api_key", [None, _KEY])
def test_session_endpoints_are_refused_to_a_browser_page(server, api_key):
    """A loopback page passes the origin guard, but it may not open a session
    that names web ports, which would lock other pages out of the TCP port."""
    srv = server(api_key=api_key)
    body = {"client": "opencode", "assistants": [], "web_ports": [18124]}
    for origin in (_PAGE, "http://localhost:5173"):
        r = srv.tcp("POST", ss.ENDPOINT, body, key=api_key, origin=origin)
        assert r.status == 404, r.body
        assert "not available to a browser page" in r.json()["error"]["message"]
    assert ss._STATE.open == {}
    session = srv.open_session([], key=api_key)
    r = srv.tcp("DELETE", f"{ss.ENDPOINT}/{session['id']}", key=api_key, origin=_PAGE)
    assert r.status == 404, r.body
    assert list(ss._STATE.open) == [session["id"]]
    assert srv.tcp("GET", "/v1/models", key=api_key,
                   origin="http://127.0.0.1:18124").status == 200


def test_allowed_routes_reach_the_app_on_the_socket(server):
    srv = server(api_key=_KEY)
    path = srv.open_session([])["socket"]
    assert _unix(path, "GET", "/health").status == 200
    for route in sorted(ss.SESSION_PATHS):
        r = _unix(path, "POST", route, {})
        assert b"not available on a launch session socket" not in r.body, \
            route
        assert r.status != 401, route


def test_session_scope_is_not_settable_from_tcp(server):
    srv = server(api_key=_KEY)
    r = srv.tcp("GET", "/v1/metrics")
    assert r.status == 401
    conn = http.client.HTTPConnection("127.0.0.1", srv.live.port, timeout=30)
    conn.request("GET", "/v1/metrics", headers={"gmlx.session": "x"})
    assert conn.getresponse().status == 401
    conn.close()


# Folders and cleanup

def test_long_cache_path_moves_the_folder_to_tmpdir(server, monkeypatch,
                                                    short_dirs):
    long_cache = short_dirs / ("c" * 90)
    monkeypatch.setenv("XDG_CACHE_HOME", str(long_cache))
    srv = server()
    path = srv.open_session([])["socket"]
    assert path.startswith(str(short_dirs / "t") + "/")
    assert len(os.fsencode(path)) < 104
    from gmlx.serve.session_paths import socket_refusal
    assert socket_refusal(path, srv.live.port) is None


def test_folder_that_is_a_link_is_refused(server, short_dirs):
    srv = server()
    folder = ss.socket_folders("127.0.0.1", srv.live.port)[0]
    folder.parent.mkdir(parents=True)
    elsewhere = short_dirs / "elsewhere"
    elsewhere.mkdir()
    folder.symlink_to(elsewhere)
    r = srv.tcp("POST", ss.ENDPOINT, {"client": "x", "assistants": []})
    assert r.status == 503, r.body
    assert list(elsewhere.iterdir()) == []


def _leftover(folder: Path, name: str) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.bind(str(path))
    sock.close()
    return path


def test_start_removes_leftover_sockets(short_dirs):
    cache, tmp = ss.socket_folders("127.0.0.1", 18123)
    stale = [_leftover(cache, "0123456789ab.sock"),
             _leftover(tmp, "ba9876543210.sock")]
    other = _leftover(ss.socket_folders("127.0.0.1", 18124)[0],
                      "0123456789ab.sock")
    keep = cache / "notes.txt"
    keep.write_text("x")
    ss.prepare_session_sockets("127.0.0.1", 18123)
    assert not any(os.path.lexists(p) for p in stale)
    assert os.path.lexists(other) and keep.exists()


def _listening(folder: Path, name: str) -> socket.socket:
    """A socket that listens at ``folder``/``name``, as a live server's does."""
    folder.mkdir(parents=True, exist_ok=True)
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.bind(str(folder / name))
    sock.listen(1)
    return sock


def test_start_keeps_the_sockets_a_live_server_listens_on(short_dirs):
    """A second server on the same bind clears the folder before its bind
    fails. The first server's sessions must keep their sockets."""
    cache, tmp = ss.socket_folders("127.0.0.1", 18123)
    stale = _leftover(cache, "0123456789ab.sock")
    with _listening(cache, "aaaaaaaaaaaa.sock") as live, \
            _listening(tmp, "bbbbbbbbbbbb.sock") as live_tmp:
        ss.prepare_session_sockets("127.0.0.1", 18123)
        assert not os.path.lexists(stale)
        for sock in (live, live_tmp):
            assert stat.S_ISSOCK(os.lstat(sock.getsockname()).st_mode)


def test_stop_removes_the_session_sockets(server, short_dirs):
    app = _APP.app

    @contextlib.asynccontextmanager
    async def bare_lifespan(app_):
        yield

    app.router.lifespan_context = bare_lifespan
    srv = server(lifespan="on", prepare=True)
    path = srv.open_session(["home"])["socket"]
    folder = Path(path).parent
    stale = _leftover(folder, "0123456789ab.sock")
    with _listening(folder, "aaaaaaaaaaaa.sock") as live:
        srv.live.stop()
        assert not srv.live.thread.is_alive()
        assert not os.path.lexists(path) and not os.path.lexists(stale)
        assert os.path.lexists(live.getsockname())
    assert ss._STATE.open == {}


def test_the_media_gate_reads_the_body_document_the_alias_check_parsed(monkeypatch):
    """A session body is parsed once before the route: the alias check
    hands its document to the media gate, which takes it out of the scope."""
    import asyncio
    import types

    from fastapi import FastAPI

    from gmlx.serve.patches import media_gate as mg

    reached = []
    inner = FastAPI()

    @inner.post("/v1/chat/completions")
    async def route(request: _Request):
        reached.append((await request.json(), sp_common.SESSION_BODY_KEY in request.scope))
        return {"ok": True}
    mg.install_media_gate(False, app=inner)
    sessions = ss._Sessions(types.SimpleNamespace())
    app = ss._SessionApp(sessions, ss._Session("s1", "pi", "/x.sock", frozenset()))
    app.app = inner
    parses = []
    monkeypatch.setattr(mg, "json", types.SimpleNamespace(
        loads=lambda raw, *a, **k: parses.append(1) or json.loads(raw, *a, **k)))

    def post(body: dict) -> int:
        raw = json.dumps(body).encode()
        messages = [{"type": "http.request", "body": raw, "more_body": False}]
        status = []

        async def receive():
            return messages.pop(0) if messages else {"type": "http.disconnect"}

        async def send(message):
            if message["type"] == "http.response.start":
                status.append(message["status"])
        scope = {"type": "http", "method": "POST", "path": "/v1/chat/completions",
                 "headers": [(b"content-type", b"application/json"),
                             (b"content-length", str(len(raw)).encode())],
                 "query_string": b""}
        asyncio.run(app(scope, receive, send))
        return status[0]

    assert post(_chat("m-a")) == 200
    assert parses == [] and reached == [(_chat("m-a"), False)]
    # The gate checks the document it was handed.
    image = {"type": "image_url", "image_url": {"url": "/Users/me/a.png"}}
    refused = {**_chat("m-a"), "messages": [{"role": "user", "content": [image]}]}
    assert post(refused) == 400
    assert parses == [] and len(reached) == 1


def test_a_failed_web_ports_record_is_logged(tmp_path, caplog):
    import types

    sessions = ss._Sessions(types.SimpleNamespace())
    with caplog.at_level("WARNING", logger=ss._log.name):
        sessions._record(tmp_path / "gone")
    assert ("cannot record the web ports of the launch sessions, so a restarted "
            "server will not refuse their pages: ") in caplog.text
    assert str(tmp_path / "gone") in caplog.text


def test_a_crashed_session_task_is_logged_and_ends_its_session(tmp_path, caplog,
                                                               monkeypatch):
    import types

    sessions = ss._Sessions(types.SimpleNamespace())
    recorded = []
    monkeypatch.setattr(sessions, "_record", lambda folder=None: recorded.append(folder))
    sock = tmp_path / "s1.sock"
    session = ss._Session("s1", "pi", str(sock), frozenset(), frozenset({8080}))
    sessions.open["s1"] = session

    async def crash():
        raise RuntimeError("the listener failed")

    async def run():
        task = asyncio.get_running_loop().create_task(crash())
        await asyncio.wait([task])
        return task
    task = asyncio.run(run())
    with caplog.at_level("ERROR", logger=ss._log.name):
        sessions._ended(task, session)
    assert "launch session s1 stopped: the listener failed" in caplog.text
    assert "s1" not in sessions.open
    assert 8080 in sessions.ended and recorded == [None]
    # A task of a session that another one replaced ends nothing.
    caplog.clear()
    other = ss._Session("s1", "pi", str(sock), frozenset())
    sessions.open["s1"] = other
    sessions._ended(task, session)
    assert sessions.open["s1"] is other


def test_an_unknown_model_answer_lists_the_configured_models_when_the_list_fails(
        monkeypatch, caplog):
    import asyncio

    from fastapi import FastAPI

    def models():
        raise RuntimeError("the model list failed")
    app = FastAPI()
    app.add_api_route("/v1/models", models, methods=["GET"])
    monkeypatch.setattr(serving, "resolved_models",
                        lambda: {"m-a": object(), "kokoro": object(), "hidden": object()})
    with caplog.at_level("ERROR", logger=ss._log.name):
        body = asyncio.run(ss._unknown_model(app, "/v1/chat/completions", "kokoro",
                                             lambda mid: mid == "hidden"))
    assert "the model list for an unknown-model answer failed" in caplog.text
    err = json.loads(body)["error"]
    assert err["type"] == "model_not_found" and err["available_models"] == ["m-a"]
