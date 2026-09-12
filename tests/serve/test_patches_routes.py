#!/usr/bin/env python3
"""/v1/models payload, HF gate, snapshot enrichment, unload and keep
routes - carved from test_server_patches.py. CPU-only."""
from __future__ import annotations

import importlib
import os

import pytest

pytest.importorskip("mlx_vlm")

import gmlx.serve.patches as sp  # noqa: E402
from gmlx.serve.patches import _common as sp_common  # noqa: E402
from gmlx.serve.patches import routes as sp_routes  # noqa: E402
import gmlx.serve.bridge_vlm as serving  # noqa: E402

_APP = importlib.import_module("mlx_vlm.server.app")
_UTILS = importlib.import_module("mlx_vlm.utils")
_PKG = importlib.import_module("mlx_vlm.server")

from test_server_patches import _FakeKeepPool, _register  # noqa: E402


def test_models_payload_lists_configured_ids_not_hf():
    _register({"models": {
        "qwen": {"path": "/abs/qwen.gguf"},
        "gemma-vlm": {"path": "/abs/g.gguf", "mmproj": "/abs/mm.gguf"},
    }})
    payload = sp_routes._models_payload()
    ids = {m["id"] for m in payload["data"]}
    assert ids == {"qwen", "gemma-vlm"}
    vlm = next(m for m in payload["data"] if m["id"] == "gemma-vlm")
    assert vlm["vlm"] is True
    assert all(m["resident"] is False for m in payload["data"])  # no pool


def test_models_payload_marks_resident_from_pool():
    _register({"models": {"qwen": {"path": "/abs/qwen.gguf", "pin": True}}})

    class _FakePool:
        def stats(self):
            return {"resident": [{"model_path": "/abs/qwen.gguf", "pinned": True,
                                  "footprint_bytes": 10, "idle_s": 3.0,
                                  "ttl_s": 900}]}

    _PKG._kq_residency_pool = _FakePool()
    m = sp_routes._models_payload()["data"][0]
    assert m["resident"] is True and m["pinned"] is True


def test_models_payload_lists_aliases_as_pickable_entries():
    _register({
        "profiles": {"coder": {"sampling": {"temperature": 0.2}}},
        "models": {"qwen": {"path": "/abs/qwen.gguf", "speculative": False}},
        "aliases": {"big": "qwen", "coder-preset": "qwen@coder"},
    })
    payload = sp_routes._models_payload()
    by_id = {m["id"]: m for m in payload["data"]}
    assert set(by_id) == {"qwen", "big", "coder-preset"}        # aliases listed
    assert by_id["big"]["alias_of"] == "qwen"
    assert by_id["coder-preset"]["alias_of"] == "qwen"
    assert by_id["coder-preset"]["profile"] == "coder"          # baked profile shown
    assert "alias_of" not in by_id["qwen"]                      # real model unmarked


def test_models_payload_marks_default():
    _register({
        "server": {"defaults": {"model": "qwen"}},
        "models": {"qwen": {"path": "/abs/qwen.gguf"},
                   "gemma": {"path": "/abs/g.gguf"}},
    })
    by_id = {m["id"]: m for m in sp_routes._models_payload()["data"]}
    assert by_id["qwen"]["default"] is True
    assert by_id["gemma"]["default"] is False


def test_models_override_registers_single_route():
    sp.install_models_endpoint_override()
    paths = [getattr(r, "path", None) for r in _APP.app.router.routes]
    assert paths.count("/v1/models") == 1
    sp.install_models_endpoint_override()                       # idempotent-ish
    paths = [getattr(r, "path", None) for r in _APP.app.router.routes]
    assert paths.count("/v1/models") == 1


# 4. HF gate
def test_gate_allows_local_and_gguf(tmp_path):
    calls = []
    orig = lambda p, *a, **k: calls.append(p) or "OK"
    local = tmp_path / "f"
    local.write_text("x")
    assert sp_routes._gate_model_path(str(local), False, orig) == "OK"
    assert sp_routes._gate_model_path("/x/model.gguf", False, orig) == "OK"
    assert len(calls) == 2


def test_gate_blocks_hf_id_when_disabled():
    orig = lambda p, *a, **k: "OK"
    with pytest.raises(sp.HFAccessDisabled):
        sp_routes._gate_model_path("org/model", False, orig)


def test_gate_allows_hf_id_when_cache_on():
    calls = []
    orig = lambda p, *a, **k: calls.append(p) or "OK"
    assert sp_routes._gate_model_path("org/model", True, orig) == "OK"
    assert calls == ["org/model"]


def test_install_hf_gate_sets_offline_env(monkeypatch):
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    sp.install_hf_download_gate(hf_cache=True)
    assert os.environ.get("HF_HUB_OFFLINE") == "1"


# 5. runtime-snapshot enrichment
def test_snapshot_enrichment_adds_resident_models():
    _APP._server_runtime_snapshot = lambda: {"loaded_model": "x"}

    class _FakePool:
        def stats(self):
            return {"resident": [{"model_path": "/abs/qwen.gguf", "pinned": False,
                                  "busy": 3, "footprint_bytes": 99,
                                  "idle_s": 1.234, "ttl_s": 900}]}

    _PKG._kq_residency_pool = _FakePool()
    serving._PATH_TO_IDS["/abs/qwen.gguf"] = ["qwen"]
    try:
        sp.install_runtime_snapshot_enrichment()
        snap = _APP._server_runtime_snapshot()
    finally:
        serving._PATH_TO_IDS.pop("/abs/qwen.gguf", None)
    assert snap["loaded_model"] == "x"                          # base preserved
    assert snap["resident_models"][0]["ids"] == ["qwen"]
    assert snap["resident_models"][0]["idle_s"] == 1.2          # rounded
    assert snap["resident_models"][0]["busy"] == 3              # in-flight count


# 6. error handlers + reload + unload route
def test_error_content_dialect_shapes():
    # One condition, two envelopes: OpenAI-style everywhere, Anthropic's
    # {"type": "error", ...} with its fixed taxonomy on /v1/messages.
    openai = sp_common._error_content(
        "/v1/chat/completions", 404, "model_not_found", "no such model",
        available_models=["a"])
    assert openai == {"error": {"type": "model_not_found",
                                "message": "no such model",
                                "available_models": ["a"]}}
    anthropic = sp_common._error_content(
        "/v1/messages", 404, "model_not_found", "no such model")
    assert anthropic["type"] == "error"
    assert anthropic["error"] == {"type": "not_found_error",
                                  "message": "no such model"}
    assert sp_common._error_content("/v1/messages", 500, "server_error",
                                    "x")["error"]["type"] == "api_error"


def test_http_exception_envelope_unwrapped():
    # The residency resolver path raises HTTPException carrying the unified
    # {"error": {...}} detail; the app-level handler must serve that body
    # directly (no {"detail": ...} wrapper) and wrap plain-string details.
    from fastapi import HTTPException
    from fastapi.testclient import TestClient

    app = _APP.app
    if not any(getattr(r, "path", None) == "/test/raise-envelope"
               for r in app.router.routes):
        @app.get("/test/raise-envelope")
        async def _raise_envelope():
            raise HTTPException(status_code=404, detail={"error": {
                "type": "model_not_found", "message": "no such model",
                "available_models": ["a"]}})

        @app.get("/test/raise-string")
        async def _raise_string():
            raise HTTPException(status_code=500, detail="it broke")

    sp.install_resolver_error_handlers()
    client = TestClient(app)
    r = client.get("/test/raise-envelope")
    assert r.status_code == 404
    assert r.json() == {"error": {"type": "model_not_found",
                                  "message": "no such model",
                                  "available_models": ["a"]}}
    r2 = client.get("/test/raise-string")
    assert r2.status_code == 500
    assert r2.json() == {"error": {"type": "server_error",
                                   "message": "it broke"}}


def test_resolver_error_handlers_registered():
    sp.install_resolver_error_handlers()
    handlers = _APP.app.exception_handlers
    assert serving.ModelNotFound in handlers
    assert serving.ModelFileMissing in handlers
    assert serving.UnknownProfile in handlers
    assert sp.HFAccessDisabled in handlers


def test_unload_and_reload_routes_register():
    sp.install_pool_aware_unload()
    sp.install_reload_route(lambda: {"reloaded": 1})
    paths = [getattr(r, "path", None) for r in _APP.app.router.routes]
    assert paths.count("/unload") == 1
    assert paths.count("/v1/reload") == 1


def test_unload_accepts_body_and_empty_post():
    """Regression: the ``request: Request`` annotation must resolve at module level.
    Under ``from __future__ import annotations`` a locally-imported ``Request`` left
    FastAPI treating ``request`` as a required query param, so every POST 422'd before
    the body was read (caught only at e2e). A route-count check can't see this - POST
    it for real and assert the body is actually consumed."""
    from fastapi.testclient import TestClient

    sp.install_pool_aware_unload()
    client = TestClient(_APP.app)
    # no pool registered -> handler runs (no 422) and reports the absence
    # cleanly, as a 503: the unload cannot be honored without a pool
    r_body = client.post("/unload", json={"model": "m"})
    assert r_body.status_code == 503, r_body.text
    assert r_body.json() == {"status": "error", "message": "no residency pool"}
    r_empty = client.post("/unload")
    assert r_empty.status_code == 200, r_empty.text
    assert r_empty.json()["status"] == "no_model_loaded"


def test_json_content_type_tolerance():
    # `curl -d '{...}'` (every doc example) sends form-encoded; the middleware
    # must rewrite it to application/json so pydantic parses the body instead
    # of 422ing. Multipart (audio uploads) must pass through untouched.
    from fastapi.testclient import TestClient

    app = _APP.app
    if not any(getattr(r, "path", None) == "/test/echo-ct"
               for r in app.router.routes):
        # `dict` (a builtin) survives this module's stringized annotations; a
        # test-local pydantic class would resolve as a query param instead.
        @app.post("/test/echo-ct")
        async def _echo_ct(body: dict):
            return {"model": body.get("model")}

    sp.install_json_content_type_tolerance()
    client = TestClient(app)
    r = client.post("/test/echo-ct", content=b'{"model": "m1"}',
                    headers={"Content-Type": "application/x-www-form-urlencoded"})
    assert r.status_code == 200 and r.json() == {"model": "m1"}
    r = client.post("/test/echo-ct", content=b'{"model": "m2"}',
                    headers={"Content-Type": "text/plain"})
    assert r.status_code == 200 and r.json() == {"model": "m2"}
    r = client.post("/test/echo-ct", json={"model": "m3"})   # normal path intact
    assert r.status_code == 200
    r = client.post("/test/echo-ct", files={"file": ("a.txt", b"x")})
    assert r.status_code == 422                              # multipart not rewritten


def test_keep_route_registers():
    sp.install_keep_route()
    paths = [getattr(r, "path", None) for r in _APP.app.router.routes]
    assert paths.count("/v1/keep") == 1


def test_keep_no_pool_reports_error():
    from fastapi.testclient import TestClient

    sp.install_keep_route()
    client = TestClient(_APP.app)
    r = client.post("/v1/keep", json={"model": "m"})
    assert r.status_code == 503, r.text
    assert r.json() == {"status": "error", "message": "no residency pool"}


def test_keep_marks_resolved_model(monkeypatch):
    from fastapi.testclient import TestClient

    monkeypatch.setattr(sp_routes, "_spawn_keep_warm", lambda model_id: None)
    _register({"models": {"qwen": {"path": "/abs/qwen.gguf"}}})
    pool = _FakeKeepPool()
    _PKG._kq_residency_pool = pool
    sp.install_keep_route()
    client = TestClient(_APP.app)
    r = client.post("/v1/keep", json={"model": "qwen", "warm": False})
    assert r.status_code == 200, r.text
    assert r.json() == {"status": "kept", "model": "qwen", "warming": False}
    assert pool.kept == [("/abs/qwen.gguf", True)]


def test_keep_warm_default_spawns_warm(monkeypatch):
    from fastapi.testclient import TestClient

    warmed = []
    monkeypatch.setattr(sp_routes, "_spawn_keep_warm", lambda model_id: warmed.append(model_id))
    _register({"models": {"qwen": {"path": "/abs/qwen.gguf"}}})
    _PKG._kq_residency_pool = _FakeKeepPool()
    sp.install_keep_route()
    client = TestClient(_APP.app)
    r = client.post("/v1/keep", json={"model": "qwen"})    # warm omitted -> default True
    assert r.json() == {"status": "kept", "model": "qwen", "warming": True}
    assert warmed == ["qwen"]


def test_keep_false_releases_without_evicting(monkeypatch):
    # A voice session ending releases its hold; the model stays resident
    # under normal LRU/TTL rather than being dumped.
    from fastapi.testclient import TestClient

    warmed = []
    monkeypatch.setattr(sp_routes, "_spawn_keep_warm", lambda model_id: warmed.append(model_id))
    _register({"models": {"qwen": {"path": "/abs/qwen.gguf"}}})
    pool = _FakeKeepPool()
    _PKG._kq_residency_pool = pool
    sp.install_keep_route()
    client = TestClient(_APP.app)
    r = client.post("/v1/keep", json={"model": "qwen", "keep": False})
    assert r.status_code == 200, r.text
    assert r.json() == {"status": "released", "model": "qwen"}
    assert pool.kept == [("/abs/qwen.gguf", False)]
    assert warmed == []                                    # release never warms


def test_keep_unknown_model_graceful():
    from fastapi.testclient import TestClient

    _register({"models": {"qwen": {"path": "/abs/qwen.gguf"}}})
    _PKG._kq_residency_pool = _FakeKeepPool()
    sp.install_keep_route()
    client = TestClient(_APP.app)
    r = client.post("/v1/keep", json={"model": "nope"})
    # 404, not 200: a typo'd keep must not read as success (launch checks the
    # status code); the body still names the id for older/other clients.
    assert r.status_code == 404, r.text
    assert r.json() == {"status": "unknown_model", "model": "nope"}


def test_keep_missing_model_field():
    from fastapi.testclient import TestClient

    _PKG._kq_residency_pool = _FakeKeepPool()
    sp.install_keep_route()
    client = TestClient(_APP.app)
    r = client.post("/v1/keep", json={})
    assert r.status_code == 400, r.text
    assert r.json() == {"status": "error", "message": "missing 'model'"}
