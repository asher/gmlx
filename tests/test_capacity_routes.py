"""Capacity-facing routes: /health?ready=1, Prometheus /metrics, scoped
/v1/cache/reset, and the queue / concurrency metrics sections."""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest

pytest.importorskip("mlx_vlm")

import gmlx.queue_cap as qc  # noqa: E402
from gmlx import server_bridge_vlm as serving  # noqa: E402
from gmlx.config import build_config  # noqa: E402
from gmlx.server_patches import _common as sp_common  # noqa: E402
from gmlx.server_patches import capacity_routes as cr  # noqa: E402
from gmlx.server_patches import hardening as sp_hardening  # noqa: E402

_APP = importlib.import_module("mlx_vlm.server.app")
_PKG = importlib.import_module("mlx_vlm.server")
_RUNTIME = importlib.import_module("mlx_vlm.server.runtime").runtime


@pytest.fixture(autouse=True)
def _restore():
    app = _APP.app
    saved_routes = sp_common._snapshot_routes(app)
    saved_pool = getattr(_PKG, "_kq_residency_pool", None)
    saved_mw = list(app.user_middleware)
    serving.clear_resolved_models()
    yield
    sp_common._restore_routes(app, saved_routes)
    _PKG._kq_residency_pool = saved_pool
    app.user_middleware[:] = saved_mw
    app.middleware_stack = None       # rebuild from the restored list
    if hasattr(app.state, sp_hardening._AUTH_FLAG):
        delattr(app.state, sp_hardening._AUTH_FLAG)
    serving.clear_resolved_models()


def _metrics(done=0, toks=0, gen_toks=0, decode_s=0.0):
    return SimpleNamespace(
        _requests_completed=done, _completion_tokens_total=toks,
        _generated_tokens_total=gen_toks, _decode_time_total_s=decode_s)


def _census(monkeypatch, rg_qsize=0, pending=0):
    rg = SimpleNamespace(requests=SimpleNamespace(qsize=lambda: rg_qsize))
    monkeypatch.setattr(_RUNTIME, "response_generator", rg, raising=False)
    gen = SimpleNamespace(_unprocessed_sequences=[object()] * pending)
    monkeypatch.setattr(qc, "_GEN_REF", lambda: gen)


class _Pool:
    def __init__(self, busy=0, managers=()):
        self._busy = busy
        self._managers = list(managers)

    def stats(self):
        return {"resident": [{"model_path": "/abs/a.gguf", "busy": self._busy,
                              "pinned": False, "footprint_bytes": 1,
                              "idle_s": 0.0, "ttl_s": None}]}

    def apc_managers(self, model_path=None):
        return [(p, m) for p, m in self._managers
                if model_path is None or p == model_path]


class _Mgr:
    def __init__(self, fail=False):
        self.cleared = 0
        self.fail = fail

    def clear(self):
        if self.fail:
            raise RuntimeError("boom")
        self.cleared += 1


# queue / concurrency sections
def test_queue_stats_carry_waiting_cap_and_eta(monkeypatch):
    monkeypatch.setenv("GMLX_QUEUE_DEPTH_CAP", "6")
    _census(monkeypatch, rg_qsize=2, pending=1)
    monkeypatch.setattr(_RUNTIME, "metrics",
                        _metrics(done=5, toks=1000, gen_toks=5000, decode_s=50.0),
                        raising=False)
    st = qc.queue_cap_stats()
    assert st["waiting"] == 3 and st["cap"] == 6
    assert st["eta_s"] == 6            # 3 x 200 tokens / 100 tok/s


def test_queue_stats_eta_zero_when_empty(monkeypatch):
    _census(monkeypatch, rg_qsize=0, pending=0)
    monkeypatch.setattr(_RUNTIME, "metrics", _metrics(), raising=False)
    assert qc.queue_cap_stats()["eta_s"] == 0


def test_queue_stats_no_engine_reads_empty(monkeypatch):
    # nothing loaded yet: nothing can be waiting, so 0 rather than unknown
    monkeypatch.setattr(_RUNTIME, "response_generator", None, raising=False)
    st = qc.queue_cap_stats()
    assert st["waiting"] == 0 and st["eta_s"] == 0
    assert "rejections" in st


def test_concurrency_stats(monkeypatch):
    monkeypatch.setenv("GMLX_DECODE_BATCH", "4")
    monkeypatch.delenv("GMLX_QUEUE_DEPTH_CAP", raising=False)
    _census(monkeypatch, rg_qsize=1, pending=0)
    _PKG._kq_residency_pool = _Pool(busy=3)
    st = qc.concurrency_stats()
    assert st == {"decode_batch": 4, "queue_cap": 8, "in_flight": 3,
                  "waiting": 1}


def test_concurrency_stats_without_pool(monkeypatch):
    monkeypatch.setenv("GMLX_DECODE_BATCH", "4")
    _PKG._kq_residency_pool = None
    monkeypatch.setattr(_RUNTIME, "response_generator", None, raising=False)
    st = qc.concurrency_stats()
    assert st["in_flight"] is None and st["waiting"] == 0


# readiness
def _ready_env(monkeypatch, band="green", in_flight=0, width=4, waiting=0):
    import gmlx.governor as gov
    monkeypatch.setattr(gov, "governor_stats", lambda: {"band": band})
    monkeypatch.setattr(qc, "concurrency_stats", lambda: {
        "decode_batch": width, "queue_cap": 2 * width,
        "in_flight": in_flight, "waiting": waiting})
    monkeypatch.setattr(_RUNTIME, "metrics", _metrics(), raising=False)


def test_readiness_ok(monkeypatch):
    _ready_env(monkeypatch)
    assert cr.readiness() == (True, "ok", 0)


def test_readiness_pressure_beats_everything(monkeypatch):
    _ready_env(monkeypatch, band="red", in_flight=0, waiting=0)
    assert cr.readiness() == (False, "pressure", qc._RETRY_MIN_S)
    _ready_env(monkeypatch, band="orange")
    assert cr.readiness()[1] == "pressure"


def test_readiness_yellow_still_admits(monkeypatch):
    _ready_env(monkeypatch, band="yellow")
    assert cr.readiness()[0] is True


def test_readiness_queue_uses_drain_estimate(monkeypatch):
    _ready_env(monkeypatch, waiting=2)
    ready, reason, retry = cr.readiness()
    assert (ready, reason) == (False, "queue")
    assert retry == qc._RETRY_DEFAULT_S      # no stats -> static estimate


def test_readiness_busy_at_width(monkeypatch):
    _ready_env(monkeypatch, in_flight=4, width=4)
    assert cr.readiness() == (False, "busy", qc._RETRY_MIN_S)
    _ready_env(monkeypatch, in_flight=3, width=4)
    assert cr.readiness()[0] is True


def test_readiness_probe_failure_reads_ready(monkeypatch):
    import gmlx.governor as gov

    def _boom():
        raise RuntimeError("no governor")
    monkeypatch.setattr(gov, "governor_stats", _boom)
    assert cr.readiness() == (True, "ok", 0)


def test_health_route_plain_and_ready(monkeypatch):
    from fastapi.testclient import TestClient

    sp_hardening.install_health_liveness_override()
    cr.install_health_readiness()
    cr.install_health_readiness()             # idempotent
    client = TestClient(_APP.app)

    r = client.get("/health")
    assert r.status_code == 200
    assert set(r.json()) == {"status", "pid"}   # liveness body unchanged

    _ready_env(monkeypatch)
    r = client.get("/health?ready=1")
    assert r.status_code == 200 and r.json()["ready"] is True

    _ready_env(monkeypatch, band="red")
    r = client.get("/health?ready=true")
    assert r.status_code == 503
    assert r.json()["ready"] is False and r.json()["reason"] == "pressure"
    assert r.headers["retry-after"] == str(qc._RETRY_MIN_S)
    assert "loaded_model" not in r.json()        # still leaks nothing


def test_health_ready_is_auth_exempt(monkeypatch):
    from fastapi.testclient import TestClient

    sp_hardening.install_api_key_auth("sekrit")
    sp_hardening.install_health_liveness_override()
    cr.install_health_readiness()
    _ready_env(monkeypatch, band="orange")
    client = TestClient(_APP.app)
    r = client.get("/health?ready=1")
    assert r.status_code == 503          # answered, not 401
    r = client.get("/v1/metrics")
    assert r.status_code == 401


# prometheus rendering
_PAYLOAD = {
    "requests_completed": 12, "requests_started": 13, "in_flight": 1,
    "uptime_s": 3.5, "latest": {"prompt_tokens": 9},
    "recent": [{"a": 1}, {"a": 2}],
    "server": {
        "loaded_model": "qwen", "continuous_batching_enabled": True,
        "request_queue_depth": 0,
        "resident_models": [
            {"ids": ["qwen", "q"], "busy": 2, "footprint_bytes": 1000,
             "pinned": False, "model_path": "/abs/qwen.gguf"},
            {"ids": [], "busy": 0, "footprint_bytes": 5,
             "model_path": "/abs/other.gguf"},
        ],
        "governor": {"band": "orange", "red_failures": 0,
                     "last_action": "evict", "kernel_floor": None},
        "capacity": {"max_ctx": {"1": 131072, "8": 16384},
                     "max_width_at_depth": {"4096": 32}, "overcommit": False},
        "queue": {"rejections": 3, "waiting": 0, "eta_s": 0,
                  "last_reject_reason": None},
        "requests": [{"id": "r1"}, {"id": "r2"}],
        "memory": {"active_bytes": 1.5e9, "headroom_bytes": float("nan")},
    },
}


def test_flatten_prometheus_shapes():
    text = cr.flatten_prometheus(_PAYLOAD)
    lines = text.splitlines()
    assert "# TYPE gmlx_requests_completed gauge" in lines
    assert "gmlx_requests_completed 12" in lines
    assert "gmlx_uptime_s 3.5" in lines
    assert "gmlx_continuous_batching_enabled 1" in lines     # bool -> 0/1
    assert "gmlx_request_queue_depth 0" in lines             # 'server' dropped
    assert 'gmlx_resident_models_busy{model="qwen"} 2' in lines
    assert 'gmlx_resident_models_busy{model="other.gguf"} 0' in lines
    assert 'gmlx_governor_band{band="orange"} 1' in lines
    assert "gmlx_governor_band_level 2" in lines
    assert 'gmlx_capacity_max_ctx{width="8"} 16384' in lines
    assert 'gmlx_capacity_max_width_at_depth{depth="4096"} 32' in lines
    assert "gmlx_capacity_overcommit 0" in lines
    assert "gmlx_recent_count 2" in lines and "gmlx_requests_count 2" in lines
    assert "gmlx_memory_active_bytes 1500000000" in lines
    assert not any(ln.startswith("gmlx_latest") for ln in lines)
    assert not any("headroom" in ln for ln in lines)           # nan skipped
    assert not any("last_action" in ln for ln in lines)        # strings skipped
    assert not any("kernel_floor" in ln for ln in lines)       # None skipped
    assert text.endswith("\n")
    # deterministic: names sorted, one TYPE per name
    names = [ln.split()[2] for ln in lines if ln.startswith("# TYPE")]
    assert names == sorted(names) and len(names) == len(set(names))


def test_flatten_counter_type_for_totals():
    text = cr.flatten_prometheus({"prompt_tokens_total": 5})
    assert "# TYPE gmlx_prompt_tokens_total counter" in text


def test_flatten_label_escaping():
    text = cr.flatten_prometheus({"server": {"resident_models": [
        {"ids": ['we"ird\\'], "busy": 1}]}})
    assert 'gmlx_resident_models_busy{model="we\\"ird\\\\"} 1' in text


def test_flatten_empty():
    assert cr.flatten_prometheus({}) == ""


class _Req:
    def __init__(self, query=None, accept=None):
        self.query_params = query or {}
        self.headers = {"accept": accept} if accept else {}


def test_wants_prometheus_negotiation():
    assert cr.wants_prometheus(_Req({"format": "prometheus"}))
    assert not cr.wants_prometheus(_Req({"format": "json"},
                                        accept="text/plain"))
    assert cr.wants_prometheus(_Req(accept="text/plain;version=0.0.4"))
    assert cr.wants_prometheus(_Req(
        accept="application/openmetrics-text;version=1.0.0,text/plain;q=0.5"))
    assert not cr.wants_prometheus(_Req(accept="application/json"))
    assert not cr.wants_prometheus(_Req(accept="*/*"))
    assert not cr.wants_prometheus(_Req())


def test_metrics_route_negotiates(monkeypatch):
    from fastapi.testclient import TestClient

    monkeypatch.setattr(_RUNTIME, "metrics",
                        SimpleNamespace(snapshot=lambda: {"requests_completed": 7}),
                        raising=False)
    monkeypatch.setattr(_APP, "_server_runtime_snapshot",
                        lambda: {"request_queue_depth": 1})
    cr.install_metrics_prometheus()
    cr.install_metrics_prometheus()            # idempotent
    client = TestClient(_APP.app)

    r = client.get("/v1/metrics")
    assert r.status_code == 200 and r.json()["requests_completed"] == 7

    r = client.get("/metrics?format=prometheus")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/plain; version=0.0.4")
    assert "gmlx_requests_completed 7" in r.text
    assert "gmlx_request_queue_depth 1" in r.text

    r = client.get("/v1/metrics", headers={"accept": "text/plain"})
    assert "gmlx_requests_completed 7" in r.text


def test_metrics_prometheus_keeps_auth(monkeypatch):
    from fastapi.testclient import TestClient

    sp_hardening.install_api_key_auth("sekrit")
    cr.install_metrics_prometheus()
    client = TestClient(_APP.app)
    assert client.get("/metrics?format=prometheus").status_code == 401


# scoped cache reset
def _register(doc):
    serving.register_resolved_models(build_config(doc))


def test_cache_reset_clears_every_resident_model():
    from fastapi.testclient import TestClient

    _register({"models": {"a": {"path": "/abs/a.gguf"},
                          "b": {"path": "/abs/b.gguf"}}})
    ma, mb = _Mgr(), _Mgr()
    _PKG._kq_residency_pool = _Pool(managers=[("/abs/a.gguf", ma),
                                              ("/abs/b.gguf", mb)])
    cr.install_scoped_cache_reset()
    cr.install_scoped_cache_reset()            # idempotent
    client = TestClient(_APP.app)
    r = client.post("/v1/cache/reset")
    assert r.status_code == 200, r.text
    assert r.json() == {"enabled": True, "status": "cleared",
                        "models": ["a", "b"]}
    assert (ma.cleared, mb.cleared) == (1, 1)


def test_cache_reset_scoped_to_one_model():
    from fastapi.testclient import TestClient

    _register({"models": {"a": {"path": "/abs/a.gguf"},
                          "b": {"path": "/abs/b.gguf"}}})
    ma, mb = _Mgr(), _Mgr()
    _PKG._kq_residency_pool = _Pool(managers=[("/abs/a.gguf", ma),
                                              ("/abs/b.gguf", mb)])
    cr.install_scoped_cache_reset()
    client = TestClient(_APP.app)
    r = client.post("/cache/reset", json={"model": "b"})
    assert r.status_code == 200, r.text
    assert r.json()["models"] == ["b"]
    assert (ma.cleared, mb.cleared) == (0, 1)


def test_cache_reset_unknown_and_not_resident():
    from fastapi.testclient import TestClient

    _register({"models": {"a": {"path": "/abs/a.gguf"},
                          "cold": {"path": "/abs/cold.gguf"}}})
    _PKG._kq_residency_pool = _Pool(managers=[("/abs/a.gguf", _Mgr())])
    cr.install_scoped_cache_reset()
    client = TestClient(_APP.app)
    r = client.post("/v1/cache/reset", json={"model": "nope"})
    assert r.status_code == 404 and r.json()["status"] == "unknown_model"
    r = client.post("/v1/cache/reset", json={"model": "cold"})
    assert r.status_code == 404 and r.json()["status"] == "not_resident"


def test_cache_reset_no_cache_and_failure_tolerated():
    from fastapi.testclient import TestClient

    _register({"models": {"a": {"path": "/abs/a.gguf"}}})
    _PKG._kq_residency_pool = _Pool(managers=[])
    cr.install_scoped_cache_reset()
    client = TestClient(_APP.app)
    r = client.post("/v1/cache/reset")
    assert r.json() == {"enabled": False, "status": "no_cache", "models": []}

    bad = _Mgr(fail=True)
    _PKG._kq_residency_pool = _Pool(managers=[("/abs/a.gguf", bad)])
    r = client.post("/v1/cache/reset")
    assert r.status_code == 200
    assert r.json()["models"] == []          # failed clear not reported cleared


def test_cache_reset_without_pool_falls_through(monkeypatch):
    from fastapi.testclient import TestClient

    _PKG._kq_residency_pool = None
    monkeypatch.setattr(_RUNTIME, "apc_manager", None, raising=False)
    cr.install_scoped_cache_reset()
    client = TestClient(_APP.app)
    r = client.post("/v1/cache/reset")
    assert r.json() == {"enabled": False}    # stock handler's answer


def test_concurrency_in_flight_prefers_per_entry_in_flight(monkeypatch):
    monkeypatch.setenv("GMLX_DECODE_BATCH", "4")
    _census(monkeypatch, rg_qsize=0, pending=0)

    class _P:
        def stats(self):
            return {"resident": [{"busy": 3, "in_flight": 2},   # 1 retained
                                 {"busy": 1}]}                  # older shape
    _PKG._kq_residency_pool = _P()
    assert qc.concurrency_stats()["in_flight"] == 3


def test_readiness_busy_needs_every_model_at_width(monkeypatch):
    class _MultiPool:
        def __init__(self, per):
            self.per = per

        def stats(self):
            return {"resident": [{"model_path": f"/abs/{i}.gguf", "in_flight": n,
                                  "busy": n} for i, n in enumerate(self.per)]}

    _ready_env(monkeypatch, in_flight=5, width=4)
    _PKG._kq_residency_pool = _MultiPool([4, 1])       # the second engine has room
    assert cr.readiness() == (True, "ok", 0)
    _PKG._kq_residency_pool = _MultiPool([4, 4])
    assert cr.readiness() == (False, "busy", qc._RETRY_MIN_S)


def test_flatten_prometheus_distinct_series_per_resident_entry():
    """Two resident entries backing the same GGUF (different load
    profiles) must not collapse into duplicate samples - Prometheus drops
    the whole scrape on one. The entry's ``profile`` tells them apart."""
    text = cr.flatten_prometheus({"resident_models": [
        {"ids": ["qwen"], "profile": "default", "busy": 1,
         "model_path": "/abs/q.gguf"},
        {"ids": ["qwen"], "profile": "lora.gguf", "busy": 0,
         "model_path": "/abs/q.gguf"},
    ]})
    lines = text.splitlines()
    assert 'gmlx_resident_models_busy{model="qwen",profile="default"} 1' in lines
    assert 'gmlx_resident_models_busy{model="qwen",profile="lora.gguf"} 0' in lines
    samples = [ln.split(" ")[0] for ln in lines if not ln.startswith("#")]
    assert len(samples) == len(set(samples))
