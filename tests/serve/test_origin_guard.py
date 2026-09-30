"""The browser origin guard: a page may call the server only from a loopback
origin or one listed in server.cors_origins. The app is the stock mlx-vlm
app with the middleware stack install_server_patches builds, in its order.
CPU only: the routes these tests call are stubs."""
from __future__ import annotations

import importlib
import sys

import pytest

pytest.importorskip("mlx_vlm")

# Module level, so FastAPI resolves the stringized annotation.
from fastapi import Request  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import gmlx.serve.patches as sp  # noqa: E402
from gmlx.config import build_config  # noqa: E402
from gmlx.serve.patches import _common as sp_common  # noqa: E402
from gmlx.serve.patches import hardening as sp_hardening  # noqa: E402
from gmlx.serve.patches import media_gate  # noqa: E402

_APP = importlib.import_module("mlx_vlm.server.app")
_LISTED = "https://ui.example"


@pytest.fixture(autouse=True)
def _restore_app():
    app = _APP.app
    openai = sys.modules.get("mlx_vlm.server.openai") \
        or importlib.import_module("mlx_vlm.server.openai")
    saved = {
        "routes": sp_common._snapshot_routes(app),
        "middleware": list(app.user_middleware),
        "mw_kwargs": [(m, dict(getattr(m, "kwargs", {}) or {}))
                      for m in app.user_middleware],
        "gen_image": openai.generate_image,
        "edit_image": openai.edit_image,
        "guarded": getattr(openai, "_kq_media_guarded", False),
        "allowed": sp_hardening._allowed_origins,
    }
    yield
    sp_common._restore_routes(app, saved["routes"])
    app.user_middleware[:] = saved["middleware"]
    for m, kw in saved["mw_kwargs"]:
        if getattr(m, "kwargs", None) is not None:
            m.kwargs.clear()
            m.kwargs.update(kw)
    app.middleware_stack = None
    openai.generate_image = saved["gen_image"]
    openai.edit_image = saved["edit_image"]
    openai._kq_media_guarded = saved["guarded"]
    sp_hardening._allowed_origins = saved["allowed"]
    for flag in (sp_hardening._AUTH_FLAG, sp_hardening._HOST_GUARD_FLAG,
                 sp_hardening._JSON_CT_FLAG, media_gate._FLAG):
        if hasattr(app.state, flag):
            delattr(app.state, flag)


def _server(origins=(_LISTED,), api_key=None):
    """The hardening prefix of install_server_patches, in its order, over
    two recording stub routes."""
    app = _APP.app
    calls = []

    async def probe():
        calls.append("probe")
        return {"ok": True}

    async def chat_endpoint(request: Request):
        calls.append(("chat", request.headers.get("content-type"), await request.body()))
        return {"ok": True}

    sp_common._remove_routes(app, "/v1/chat/completions")
    app.add_api_route("/zz-origin-probe", probe, methods=["GET", "POST"])
    app.add_api_route("/v1/chat/completions", chat_endpoint, methods=["POST"])
    cfg = build_config({"server": {"cors_origins": list(origins),
                                   **({"api_key": api_key} if api_key else {})}})
    media_gate.install_media_gate(False)
    sp.install_api_key_auth(cfg.api_key)
    sp.install_json_content_type_tolerance()
    sp.install_loopback_host_guard("127.0.0.1")
    sp_hardening.install_origin_guard(cfg.cors_origins)
    sp.disable_credentialed_cors()
    return TestClient(app, base_url="http://127.0.0.1"), calls


def _dispatch_names():
    return [getattr((getattr(m, "kwargs", None) or {}).get("dispatch"), "__name__",
                    getattr(m.cls, "__name__", "?"))
            for m in _APP.app.user_middleware]


@pytest.mark.parametrize("origin", ["https://evil.example", "http://example.com:8080",
                                    "https://ui.example.evil.example"])
def test_a_foreign_origin_gets_403_on_get_post_and_preflight(origin):
    client, calls = _server()
    get = client.get("/zz-origin-probe", headers={"Origin": origin})
    post = client.post("/zz-origin-probe", headers={"Origin": origin}, json={})
    pre = client.options("/zz-origin-probe", headers={
        "Origin": origin, "Access-Control-Request-Method": "POST"})
    for r in (get, post, pre):
        assert r.status_code == 403, r.text
        err = r.json()["error"]
        assert err["type"] == "origin_not_allowed"
        assert err["message"] == (
            f"Pages from {origin} may not call this server. Add {origin} to "
            "server.cors_origins in the server's config file to allow them, or "
            "serve the page from a loopback address.")
        assert "access-control-allow-origin" not in r.headers
    assert calls == []


def test_a_request_without_an_origin_passes():
    client, calls = _server()
    r = client.get("/zz-origin-probe")
    assert r.status_code == 200 and calls == ["probe"]
    assert "access-control-allow-origin" not in r.headers


@pytest.mark.parametrize("origin", ["http://localhost:3000", "http://127.0.0.1:5173",
                                    "http://[::1]:8000", "http://127.0.0.2:1",
                                    "https://localhost", "http://localhost"])
def test_a_loopback_origin_on_any_port_passes_and_is_echoed(origin):
    client, calls = _server(origins=())
    r = client.get("/zz-origin-probe", headers={"Origin": origin})
    assert r.status_code == 200, r.text
    assert r.headers["access-control-allow-origin"] == origin
    assert "access-control-allow-credentials" not in r.headers
    pre = client.options("/zz-origin-probe", headers={
        "Origin": origin, "Access-Control-Request-Method": "POST"})
    assert pre.status_code == 200, pre.text
    assert pre.headers["access-control-allow-origin"] == origin


@pytest.mark.parametrize("listed", [_LISTED, "HTTPS://UI.EXAMPLE:443", "https://ui.example/"])
def test_a_listed_origin_passes_with_its_own_value_echoed(listed):
    client, calls = _server(origins=(listed,))
    r = client.post("/zz-origin-probe", headers={"Origin": _LISTED}, json={})
    assert r.status_code == 200, r.text
    assert r.headers["access-control-allow-origin"] == _LISTED
    assert "origin" in r.headers.get("vary", "").lower()
    pre = client.options("/zz-origin-probe", headers={
        "Origin": _LISTED, "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "authorization, content-type"})
    assert pre.status_code == 200, pre.text
    assert pre.headers["access-control-allow-origin"] == _LISTED
    assert "access-control-allow-credentials" not in pre.headers
    assert calls == ["probe"]


@pytest.mark.parametrize("origin", [
    "https://ui.example:8443",        # another port
    "http://ui.example",              # another scheme
    "https://ui.example.:443",        # a trailing dot is another host to a browser
    "https://www.ui.example",         # a subdomain
    "https://ui.examplex"])
def test_near_misses_of_a_listed_origin_are_refused(origin):
    client, calls = _server()
    r = client.get("/zz-origin-probe", headers={"Origin": origin})
    assert r.status_code == 403, (origin, r.text)
    assert calls == []


def test_listing_one_loopback_name_does_not_widen_a_foreign_one():
    # 127.0.0.1 and localhost are both loopback, so both pass whatever the
    # list says, while a LAN address passes only when listed.
    client, _ = _server(origins=("http://192.168.1.20:3000",))
    assert client.get("/zz-origin-probe",
                      headers={"Origin": "http://192.168.1.20:3000"}).status_code == 200
    assert client.get("/zz-origin-probe",
                      headers={"Origin": "http://192.168.1.21:3000"}).status_code == 403
    assert client.get("/zz-origin-probe",
                      headers={"Origin": "http://192.168.1.20:3001"}).status_code == 403


@pytest.mark.parametrize("origin", ["null", "", "not a url", "app://./index.html", "*"])
def test_null_and_malformed_origins_are_foreign(origin):
    client, calls = _server()
    r = client.get("/zz-origin-probe", headers={"Origin": origin})
    assert r.status_code == 403, r.text
    message = r.json()["error"]["message"]
    if origin == "null":
        assert "Origin: null" in message
    else:
        assert message == (f'The Origin header "{origin}" is not an origin, so the server '
                           "cannot tell which page sent the request.")
    assert calls == []


@pytest.mark.parametrize("origin", ["tauri://localhost", "app://.", "file://",
                                    "vscode-file://vscode-app", "vscode-webview://1a2b3c4d",
                                    "TAURI://localhost"])
def test_a_desktop_app_origin_passes_and_is_echoed(origin):
    """Electron, Tauri and VS Code webview apps send these, and a web page
    cannot, so they pass as a loopback page does."""
    client, calls = _server(origins=())
    r = client.post("/zz-origin-probe", headers={"Origin": origin}, json={})
    assert r.status_code == 200, r.text
    assert r.headers["access-control-allow-origin"] == origin
    pre = client.options("/zz-origin-probe", headers={
        "Origin": origin, "Access-Control-Request-Method": "POST"})
    assert pre.status_code == 200, pre.text
    assert pre.headers["access-control-allow-origin"] == origin
    assert calls == ["probe"]


def test_another_app_scheme_is_refused_until_listed():
    client, calls = _server(origins=())
    r = client.get("/zz-origin-probe", headers={"Origin": "capacitor://localhost"})
    assert r.status_code == 403
    assert r.json()["error"]["message"] == (
        "The app that sent Origin capacitor://localhost may not call this server. Add "
        "capacitor://localhost to server.cors_origins in the server's config file to "
        "allow it.")
    client, calls = _server(origins=("capacitor://localhost",))
    r = client.get("/zz-origin-probe", headers={"Origin": "capacitor://localhost"})
    assert r.status_code == 200, r.text
    assert r.headers["access-control-allow-origin"] == "capacitor://localhost"


def test_a_refusal_is_logged_once_a_minute_for_each_origin(monkeypatch, capsys):
    """The page sees only a CORS error, so the log is where the key is named."""
    monkeypatch.setattr(sp_hardening, "_refusals_logged", {})
    monkeypatch.setattr(sp_hardening, "_refusal_window", [0.0, 0])
    now = [1000.0]
    monkeypatch.setattr(sp_hardening.time, "monotonic", lambda: now[0])
    client, _ = _server()
    lan = "http://192.168.1.20:3000"
    for _ in range(3):
        client.options("/zz-origin-probe", headers={
            "Origin": lan, "Access-Control-Request-Method": "POST"})
    client.get("/zz-origin-probe", headers={"Origin": "https://evil.example"})
    lines = capsys.readouterr().out.splitlines()
    assert lines == [
        f"[server] refused a request with status 403: Pages from {lan} may not call this "
        f"server. Add {lan} to server.cors_origins in the server's config file to allow "
        "them, or serve the page from a loopback address.",
        "[server] refused a request with status 403: Pages from https://evil.example may "
        "not call this server. Add https://evil.example to server.cors_origins in the "
        "server's config file to allow them, or serve the page from a loopback address."]
    now[0] += 61
    client.get("/zz-origin-probe", headers={"Origin": lan})
    assert len(capsys.readouterr().out.splitlines()) == 1


def test_refusal_lines_stop_at_the_limit_for_a_minute(monkeypatch, capsys):
    monkeypatch.setattr(sp_hardening, "_refusals_logged", {})
    monkeypatch.setattr(sp_hardening, "_refusal_window", [0.0, 0])
    now = [1000.0]
    monkeypatch.setattr(sp_hardening.time, "monotonic", lambda: now[0])
    client, _ = _server()
    for i in range(sp_hardening._REFUSALS_LOGGED_MAX + 5):
        client.get("/zz-origin-probe", headers={"Origin": f"https://h{i}.example"})
    assert len(capsys.readouterr().out.splitlines()) == sp_hardening._REFUSALS_LOGGED_MAX
    now[0] += 61
    client.get("/zz-origin-probe", headers={"Origin": "https://late.example"})
    assert len(capsys.readouterr().out.splitlines()) == 1


def test_a_refusal_line_escapes_control_characters(monkeypatch, capsys):
    monkeypatch.setattr(sp_hardening, "_refusals_logged", {})
    monkeypatch.setattr(sp_hardening, "_refusal_window", [0.0, 0])
    client, _ = _server()
    client.get("/zz-origin-probe", headers={"Origin": "x\x1b[2Jy"})
    out = capsys.readouterr().out
    assert "\x1b" not in out and "\\x1b[2Jy" in out


def test_a_foreign_text_plain_post_never_reaches_the_chat_route():
    """A text/plain POST needs no preflight, and the content-type tolerance
    turns it into JSON, so without the guard it would run."""
    client, calls = _server()
    body = '{"model": "m", "messages": [{"role": "user", "content": "hi"}]}'
    r = client.post("/v1/chat/completions", content=body,
                    headers={"Origin": "https://evil.example", "Content-Type": "text/plain"})
    assert r.status_code == 403 and calls == []
    # The same request from a loopback page runs as JSON, which shows what
    # the guard stops.
    ok = client.post("/v1/chat/completions", content=body,
                     headers={"Origin": "http://localhost:3000", "Content-Type": "text/plain"})
    assert ok.status_code == 200, ok.text
    assert calls == [("chat", "application/json", body.encode())]


def test_the_guard_runs_before_the_key_check():
    client, calls = _server(api_key="sekrit")
    foreign = client.get("/zz-origin-probe", headers={"Origin": "https://evil.example"})
    assert foreign.status_code == 403
    assert client.get("/zz-origin-probe").status_code == 401
    ok = client.get("/zz-origin-probe", headers={
        "Origin": _LISTED, "Authorization": "Bearer sekrit"})
    assert ok.status_code == 200 and calls == ["probe"]


def test_the_guard_is_the_outermost_middleware():
    _server(api_key="sekrit")
    assert _dispatch_names()[:5] == ["_origin_guard", "_host_guard", "_json_ct_middleware",
                                     "_auth_middleware", "_media_gate"]


def test_install_is_idempotent_and_replaces_the_list():
    client, _ = _server()
    before = len(_APP.app.user_middleware)
    sp_hardening.install_origin_guard(["https://other.example"])
    assert len(_APP.app.user_middleware) == before
    assert client.get("/zz-origin-probe", headers={"Origin": _LISTED}).status_code == 403
    r = client.get("/zz-origin-probe", headers={"Origin": "https://other.example"})
    assert r.status_code == 200
    assert r.headers["access-control-allow-origin"] == "https://other.example"


def test_no_response_carries_a_wildcard_origin():
    client, _ = _server()
    for origin in ("http://localhost:3000", _LISTED):
        r = client.get("/zz-origin-probe", headers={"Origin": origin})
        assert r.headers.get("access-control-allow-origin") != "*"


def test_install_server_patches_passes_the_configured_list(monkeypatch):
    seen = []

    class _Stop(Exception):
        pass

    def fake(origins=()):
        seen.append(list(origins))
        raise _Stop

    monkeypatch.setattr(sp_hardening, "install_origin_guard", fake)
    cfg = build_config({"server": {"cors_origins": [_LISTED]}})
    with pytest.raises(_Stop):
        sp.install_server_patches(cfg, reload_fn=None)
    assert seen == [[_LISTED]]


def test_start_up_lines_name_pages_and_apps():
    from gmlx.serve.server import cors_origin_lines

    cfg = build_config({"server": {"cors_origins": [
        _LISTED, "capacitor://localhost", "http://localhost:3000"]}})
    assert cors_origin_lines(cfg.cors_origins) == [
        "[server] browser pages at https://ui.example may call this server "
        "(server.cors_origins)",
        "[server] the app that sends Origin capacitor://localhost may call this server "
        "(server.cors_origins)"]
