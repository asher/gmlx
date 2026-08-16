"""Queue depth cap: decision, Retry-After estimate, route rejection."""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest

pytest.importorskip("mlx_vlm")

import gmlx.queue_cap as qc  # noqa: E402

_APP = importlib.import_module("mlx_vlm.server.app")
_RUNTIME = importlib.import_module("mlx_vlm.server.runtime").runtime


def _metrics(done=0, toks=0, gen_toks=0, decode_s=0.0):
    return SimpleNamespace(
        _requests_completed=done,
        _completion_tokens_total=toks, _generated_tokens_total=gen_toks,
        _decode_time_total_s=decode_s)


def _rg(qsize=0):
    return SimpleNamespace(requests=SimpleNamespace(qsize=lambda: qsize))


def _census(monkeypatch, rg_qsize=0, pending=0):
    """Wire a fake waiting census: server queue size + generator pending."""
    monkeypatch.setattr(_RUNTIME, "response_generator", _rg(rg_qsize),
                        raising=False)
    gen = SimpleNamespace(_unprocessed_sequences=[object()] * pending)
    monkeypatch.setattr(qc, "_GEN_REF", lambda: gen)


def test_default_cap_formula():
    assert qc._cap() == max(4 * qc._decode_concurrency(), 32)


def test_cap_env_override(monkeypatch):
    monkeypatch.setenv("GMLX_QUEUE_DEPTH_CAP", "7")
    assert qc._cap() == 7


def test_depth_counts_queue_and_pending(monkeypatch):
    gen = SimpleNamespace(_unprocessed_sequences=[1, 2, 3])
    monkeypatch.setattr(qc, "_GEN_REF", lambda: gen)
    assert qc._waiting_depth(_rg(qsize=2)) == 5


def test_depth_survives_dead_generator(monkeypatch):
    monkeypatch.setattr(qc, "_GEN_REF", lambda: None)
    assert qc._waiting_depth(_rg(qsize=1)) == 1


def test_depth_survives_broken_qsize(monkeypatch):
    monkeypatch.setattr(qc, "_GEN_REF", None)
    rg = SimpleNamespace(
        requests=SimpleNamespace(qsize=lambda: 1 / 0))
    assert qc._waiting_depth(rg) == 0


def test_census_publisher_stashes_generator():
    from mlx_vlm.generate import ar as _ar

    qc._install_census()
    assert getattr(_ar.BatchGenerator._next, qc._PUB_FLAG, False)

    class _Stub:  # SimpleNamespace refuses weakrefs
        _unprocessed_sequences = [1]

    stub = _Stub()
    try:
        _ar.BatchGenerator._next(stub)
    except Exception:
        pass  # stock body needs real state; publish happens first
    assert qc._GEN_REF() is stub


def test_retry_after_no_stats_is_static():
    assert qc._retry_after_s(_metrics(), 100) == qc._RETRY_DEFAULT_S


def test_retry_after_estimate_and_clamp():
    # 10 waiting x 200 mean tokens / 100 tok/s = 20 s
    m = _metrics(done=5, toks=1000, gen_toks=5000, decode_s=50.0)
    assert qc._retry_after_s(m, 10) == 20
    assert qc._retry_after_s(m, 1) == qc._RETRY_MIN_S
    assert qc._retry_after_s(m, 1000) == qc._RETRY_MAX_S


def test_check_below_cap_admits(monkeypatch):
    _census(monkeypatch, rg_qsize=0, pending=1)
    monkeypatch.setattr(_RUNTIME, "metrics", _metrics(), raising=False)
    assert qc.check_queue_depth() is None


def test_check_at_cap_rejects(monkeypatch):
    monkeypatch.setenv("GMLX_QUEUE_DEPTH_CAP", "4")
    _census(monkeypatch, rg_qsize=1, pending=3)
    monkeypatch.setattr(_RUNTIME, "metrics", _metrics(), raising=False)
    resp = qc.check_queue_depth()
    assert resp is not None and resp.status_code == 503
    assert resp.headers["retry-after"] == str(qc._RETRY_DEFAULT_S)


def test_check_disabled_by_zero(monkeypatch):
    monkeypatch.setenv("GMLX_QUEUE_DEPTH_CAP", "0")
    _census(monkeypatch, rg_qsize=999, pending=0)
    assert qc.check_queue_depth() is None


def test_check_no_engine_admits(monkeypatch):
    monkeypatch.setenv("GMLX_QUEUE_DEPTH_CAP", "1")
    monkeypatch.setattr(_RUNTIME, "response_generator", None,
                        raising=False)
    assert qc.check_queue_depth() is None


@pytest.fixture
def app_routes():
    saved = list(_APP.app.router.routes)
    yield _APP.app
    _APP.app.router.routes[:] = saved


def test_route_rejects_with_503_body(app_routes, monkeypatch):
    from fastapi.testclient import TestClient

    monkeypatch.setenv("GMLX_QUEUE_DEPTH_CAP", "2")
    _census(monkeypatch, rg_qsize=2, pending=3)
    monkeypatch.setattr(_RUNTIME, "metrics",
                        _metrics(done=2, toks=100,
                                 gen_toks=100, decode_s=10.0),
                        raising=False)
    qc.install_queue_depth_cap()
    client = TestClient(_APP.app)
    r = client.post("/v1/chat/completions",
                    json={"model": "m", "messages": [
                        {"role": "user", "content": "hi"}]})
    assert r.status_code == 503
    body = r.json()["error"]
    assert body["queue_cap"] == 2 and body["queue_depth"] == 5
    assert body["type"] == "server_overloaded"
    assert "Retry-After" in r.headers
    assert 2 <= int(r.headers["Retry-After"]) <= 60
    assert qc.queue_cap_stats()["rejections"] >= 1


def test_route_at_cap_minus_one_serves(app_routes, monkeypatch):
    """A request admitted below the cap reaches the stock handler."""
    from fastapi.testclient import TestClient

    monkeypatch.setenv("GMLX_QUEUE_DEPTH_CAP", "6")
    _census(monkeypatch, rg_qsize=5, pending=0)
    monkeypatch.setattr(_RUNTIME, "metrics", _metrics(), raising=False)
    seen = []

    async def _stub(*a, **k):
        seen.append(1)
        return {"ok": True}

    for path in ("/v1/chat/completions",):
        route = next(r for r in _APP.app.router.routes
                     if getattr(r, "path", None) == path
                     and "POST" in (getattr(r, "methods", None) or ()))
        import inspect
        _stub.__signature__ = inspect.signature(route.endpoint)
        from gmlx.server_patches._common import _remove_routes
        _remove_routes(_APP.app, path)
        _APP.app.add_api_route(path, _stub, methods=["POST"],
                               include_in_schema=False)
    qc.install_queue_depth_cap()
    client = TestClient(_APP.app)
    r = client.post("/v1/chat/completions",
                    json={"model": "m", "messages": []})
    assert r.status_code == 200 and seen == [1]


def test_install_idempotent(app_routes):
    qc.install_queue_depth_cap()
    n = len(_APP.app.router.routes)
    qc.install_queue_depth_cap()
    assert len(_APP.app.router.routes) == n
