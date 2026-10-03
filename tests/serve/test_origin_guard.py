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
from gmlx.serve.patches import routes as sp_routes  # noqa: E402

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
                 sp_hardening._JSON_CT_FLAG, media_gate._FLAG,
                 sp_routes._UNCONFIGURED_FLAG):
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
    sp_routes.install_unconfigured_answers(cfg)
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
            "server.cors_origins in the server's config file, then run gmlx restart. "
            "Or serve the page from a loopback address.")
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
        "capacitor://localhost to server.cors_origins in the server's config file, then "
        "run gmlx restart.")
    client, calls = _server(origins=("capacitor://localhost",))
    r = client.get("/zz-origin-probe", headers={"Origin": "capacitor://localhost"})
    assert r.status_code == 200, r.text
    assert r.headers["access-control-allow-origin"] == "capacitor://localhost"


def test_a_refusal_is_logged_once_a_minute_for_each_origin(monkeypatch, capsys):
    """The page sees only a CORS error, so the log is where the key is named."""
    monkeypatch.setattr(sp_hardening, "_refusal_logs", {})
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
        f"server. Add {lan} to server.cors_origins in the server's config file, then run "
        "gmlx restart. Or serve the page from a loopback address.",
        "[server] refused a request with status 403: Pages from https://evil.example may "
        "not call this server. Add https://evil.example to server.cors_origins in the "
        "server's config file, then run gmlx restart. Or serve the page from a loopback "
        "address."]
    now[0] += 61
    client.get("/zz-origin-probe", headers={"Origin": lan})
    assert len(capsys.readouterr().out.splitlines()) == 1


def test_refusal_lines_stop_at_the_limit_for_a_minute(monkeypatch, capsys):
    monkeypatch.setattr(sp_hardening, "_refusal_logs", {})
    now = [1000.0]
    monkeypatch.setattr(sp_hardening.time, "monotonic", lambda: now[0])
    client, _ = _server()
    for i in range(sp_hardening._REFUSALS_LOGGED_MAX + 5):
        client.get("/zz-origin-probe", headers={"Origin": f"https://h{i}.example"})
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == sp_hardening._REFUSALS_LOGGED_MAX + 1
    assert lines[-1] == ("[server] refused more requests because of the page or app that "
                         "sent them in this minute. The log shows at most 20 such refusals "
                         "a minute.")
    now[0] += 61
    client.get("/zz-origin-probe", headers={"Origin": "https://late.example"})
    assert len(capsys.readouterr().out.splitlines()) == 1


def test_a_flood_of_one_kind_of_refusal_hides_no_other_kind(monkeypatch, capsys):
    monkeypatch.setattr(sp_hardening, "_refusal_logs", {})
    monkeypatch.setattr(sp_hardening.time, "monotonic", lambda: 1000.0)
    for i in range(100):
        sp_hardening._log_refusal(f"400 bad body {i}", f"bad body {i}", 400, kind="media")
    capsys.readouterr()
    client, _ = _server()
    client.get("/zz-origin-probe", headers={"Origin": "https://evil.example"})
    sp_hardening._log_refusal("peer 192.168.64.7", "from a guest", kind="peer")
    assert capsys.readouterr().out.splitlines() == [
        "[server] refused a request with status 403: Pages from https://evil.example may "
        "not call this server. Add https://evil.example to server.cors_origins in the "
        "server's config file, then run gmlx restart. Or serve the page from a loopback "
        "address.",
        "[server] refused a request with status 403: from a guest"]


def test_a_refusal_line_escapes_control_characters(monkeypatch, capsys):
    monkeypatch.setattr(sp_hardening, "_refusal_logs", {})
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
    keyed = client.get("/zz-origin-probe", headers={
        "Origin": "https://evil.example", "Authorization": "Bearer sekrit"})
    assert keyed.status_code == 403            # a valid key does not lift the guard
    assert client.get("/zz-origin-probe").status_code == 401
    ok = client.get("/zz-origin-probe", headers={
        "Origin": _LISTED, "Authorization": "Bearer sekrit"})
    assert ok.status_code == 200 and calls == ["probe"]


def test_the_guard_is_the_outermost_middleware():
    _server(api_key="sekrit")
    assert _dispatch_names()[:7] == ["_origin_guard", "CORSMiddleware", "_host_guard",
                                     "_json_ct_middleware", "_auth_middleware",
                                     "_unconfigured", "_media_gate"]


def _image_chat(url: str) -> dict:
    return {"model": "m", "messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": url}}]}]}


def test_a_page_that_may_call_the_server_reads_every_refusal(monkeypatch, capsys):
    """The gate, unconfigured-service and key refusals carry the CORS
    headers of an allowed origin, and the first two are logged."""
    monkeypatch.setattr(sp_hardening, "_refusal_logs", {})
    monkeypatch.setattr(media_gate, "BODY_MAX_BYTES", 1 << 10)
    client, calls = _server(api_key="sekrit")
    page = {"Origin": "http://localhost:3000"}
    keyed = {**page, "Authorization": "Bearer sekrit"}
    refused = {
        400: client.post("/v1/chat/completions", headers=keyed,
                         json=_image_chat("https://example.com/a.png")),
        413: client.post("/v1/chat/completions", headers=keyed, content=b"x" * 2048),
        404: client.post("/v1/embeddings", headers=keyed, json={"input": "hi"}),
        401: client.post("/v1/chat/completions", headers=page, json={}),
    }
    for status, r in refused.items():
        assert r.status_code == status, r.text
        assert r.headers["access-control-allow-origin"] == "http://localhost:3000"
    pre = client.options("/v1/embeddings", headers={
        **page, "Access-Control-Request-Method": "POST"})
    assert pre.status_code == 200
    assert pre.headers["access-control-allow-origin"] == "http://localhost:3000"
    assert calls == []
    lines = capsys.readouterr().out.splitlines()
    assert [line.split(":", 1)[0] for line in lines] == [
        "[server] refused a request with status 400",
        "[server] refused a request with status 413",
        "[server] refused a request with status 404"]
    assert "server.media_urls" in lines[0] and "server.embeddings" in lines[2]


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

    ext = "chrome-extension://jfgfabcdefghijklmnop"
    cfg = build_config({"server": {"cors_origins": [
        _LISTED, "capacitor://localhost", "http://localhost:3000", ext]}})
    assert cors_origin_lines(cfg.cors_origins) == [
        "[server] browser pages at https://ui.example may call this server "
        "(server.cors_origins)",
        "[server] the app that sends Origin capacitor://localhost may call this server "
        "(server.cors_origins)",
        f"[server] the browser extension at {ext} may call this server "
        "(server.cors_origins)"]


@pytest.mark.parametrize("origin", ["chrome-extension://jfgfabcdefghijklmnop",
                                    "moz-extension://0d2c1234-aaaa-bbbb-cccc-1234567890ab"])
def test_a_browser_extension_is_named_as_one(origin):
    client, calls = _server(origins=())
    r = client.get("/zz-origin-probe", headers={"Origin": origin})
    assert r.status_code == 403 and calls == []
    assert r.json()["error"]["message"] == (
        f"The browser extension at {origin} may not call this server. Add {origin} to "
        "server.cors_origins in the server's config file, then run gmlx restart.")


_SAFARI = "safari-web-extension://3f6c1d0e-9a8b-4c7d-8e2f-1a2b3c4d5e6f"


@pytest.mark.parametrize("wildcard, origin", [
    ("chrome-extension://*", "chrome-extension://jfgfabcdefghijklmnop"),
    ("moz-extension://*", "moz-extension://0d2c1234-aaaa-bbbb-cccc-1234567890ab"),
    ("Safari-Web-Extension://*", _SAFARI),
    ("safari-web-extension://*", _SAFARI.upper().replace("SAFARI-WEB-EXTENSION",
                                                          "safari-web-extension"))])
def test_a_wildcard_entry_lets_every_extension_of_that_browser_call(wildcard, origin):
    client, calls = _server(origins=(wildcard,))
    r = client.post("/zz-origin-probe", headers={"Origin": origin}, json={})
    assert r.status_code == 200, r.text
    assert r.headers["access-control-allow-origin"] == origin
    pre = client.options("/zz-origin-probe", headers={
        "Origin": origin, "Access-Control-Request-Method": "POST"})
    assert pre.status_code == 200, pre.text
    assert pre.headers["access-control-allow-origin"] == origin
    for other in ("https://evil.example", "chrome-extension://*",
                  "ms-browser-extension://abc", *(
                      o for o in ("chrome-extension://jfgf", "moz-extension://0d2c",
                                  "safari-web-extension://3f6c")
                      if not o.startswith(wildcard.lower()[:-1]))):
        assert client.get("/zz-origin-probe", headers={"Origin": other}).status_code \
            == 403, other


def test_a_listed_extension_matches_in_any_case():
    listed = "chrome-extension://jfgfabcdefghijklmnop"
    client, _ = _server(origins=(listed,))
    r = client.get("/zz-origin-probe", headers={"Origin": listed.upper()})
    assert r.status_code == 200, r.text
    assert r.headers["access-control-allow-origin"] == listed.upper()


def test_a_safari_extension_refusal_names_the_wildcard():
    client, _ = _server(origins=())
    r = client.get("/zz-origin-probe", headers={"Origin": _SAFARI})
    assert r.status_code == 403
    assert r.json()["error"]["message"] == (
        f"The browser extension at {_SAFARI} may not call this server. Safari gives an "
        "extension a new ID at each launch, so add safari-web-extension://* to "
        "server.cors_origins in the server's config file, which lets every Safari "
        "extension call it, then run gmlx restart.")


def test_start_up_lines_name_the_browser_of_a_wildcard_entry():
    from gmlx.serve.server import cors_origin_lines

    cfg = build_config({"server": {"cors_origins": [
        "chrome-extension://*", "moz-extension://*", "safari-web-extension://*"]}})
    assert cors_origin_lines(cfg.cors_origins) == [
        "[server] every browser extension in Chrome, Edge and other Chromium browsers "
        "may call this server (chrome-extension://* in server.cors_origins)",
        "[server] every browser extension in Firefox may call this server "
        "(moz-extension://* in server.cors_origins)",
        "[server] every browser extension in Safari may call this server "
        "(safari-web-extension://* in server.cors_origins)"]


def test_a_file_page_is_named_as_one():
    client, calls = _server(origins=())
    r = client.get("/zz-origin-probe", headers={"Origin": "null"})
    assert r.status_code == 403 and calls == []
    assert "such as a file opened from disk" in r.json()["error"]["message"]
