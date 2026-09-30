"""Launch session sockets: the session endpoints, the scope of a session
socket, and socket cleanup. Each test runs the app in a real uvicorn server
on a loopback port, and the session sockets are real Unix sockets. CPU only:
the chat route is a stub and no model loads."""
from __future__ import annotations

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
from pathlib import Path

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
                       "auth": http_request.headers.get("authorization")})
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
])
def test_open_session_refuses_a_bad_body(server, body):
    srv = server()
    r = srv.tcp("POST", ss.ENDPOINT, body)
    assert r.status == 400, r.body
    assert ss._STATE.open == {}


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
    assert srv.tcp("GET", "/v1/models", key=api_key, origin=_PAGE).status == 200


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


def test_a_chat_body_over_the_ceiling_is_refused_on_the_socket(server, monkeypatch):
    from gmlx.serve.patches import media_gate as mg
    monkeypatch.setattr(mg, "BODY_MAX_BYTES", 1000)
    srv = server()
    path = srv.open_session([])["socket"]
    for route in ("/v1/chat/completions", "/messages"):
        r = _unix(path, "POST", route, {**_chat("x"), "pad": " " * 2000})
        assert r.status == 413, (route, r.body)
        assert "new conversation" in r.json()["error"]["message"]
    # A declared length over the ceiling is refused before the body is sent.
    conn = _UnixConnection(path)
    try:
        conn.putrequest("POST", "/v1/chat/completions")
        conn.putheader("content-type", "application/json")
        conn.putheader("content-length", str(mg.BODY_MAX_BYTES + 1))
        conn.endheaders()
        assert conn.getresponse().status == 413
    finally:
        conn.close()
    assert srv.chat == []
    assert _unix(path, "POST", "/v1/chat/completions", _chat("m-a")).status == 200


@pytest.mark.parametrize("api_key", [None, _KEY])
def test_unlisted_alias_is_an_unknown_model(server, api_key):
    srv = server(api_key=api_key)
    path = srv.open_session(["home"])["socket"]
    hidden = _unix(path, "POST", "/v1/chat/completions", _chat("hidden"))
    unknown = _unix(path, "POST", "/v1/chat/completions", _chat("nope"))
    assert hidden.status == unknown.status == 404
    assert hidden.body.replace(b"hidden", b"nope") == unknown.body
    assert srv.tools == [] and [c["model"] for c in srv.chat] == ["nope"]
    # The other surfaces answer the same 404 in their own shape, where TCP
    # names the alias as chat-completions only.
    for route in ("/v1/messages", "/messages/count_tokens", "/v1/responses",
                  "/responses/input_tokens", "/chat/completions"):
        r = _unix(path, "POST", route, _chat("hidden"))
        assert r.status == 404, (route, r.body)
        assert "unknown model id 'hidden'" in r.body.decode(), route
    tcp = srv.tcp("POST", "/v1/messages", _chat("hidden"), key=api_key)
    assert tcp.status == 400


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
    srv.live.stop()
    assert not srv.live.thread.is_alive()
    assert not os.path.lexists(path) and not os.path.lexists(stale)
    assert ss._STATE.open == {}
