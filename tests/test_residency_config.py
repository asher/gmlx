#!/usr/bin/env python3
"""Residency config extensions: load_signature widens the cache_key (one GGUF,
two profiles -> two entries), the per-build env window is set then restored (no
leak to a sibling load), and the idle-TTL reaper unloads idle non-pinned entries
only when the server is drained. CPU-only - fake stock load/teardown, injected
footprint + clock + in-flight reader, so no GPU and no model files."""
from __future__ import annotations

import os
import sys
import threading
import types

import pytest

from gmlx import residency  # noqa: E402
from gmlx import server_bridge_vlm as serving  # noqa: E402
from gmlx.residency import (  # noqa: E402
    _active_entry,
    _ResidencyPool,
    _RuntimeProxy,
)

GB = 1024**3


class _Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t


class _FakeOriginal:
    def __init__(self):
        self.metrics = "METRICS"


def make_pool(*, budget_gb=200, footprint_gb=10, pinned=(), in_flight=0,
              clock=None):
    """A pool wired to fake stock loaders that record the env seen at build time
    and the teardown order. ``in_flight`` is the server-wide count the reaper gates
    on (int or None)."""
    proxy = _RuntimeProxy(_FakeOriginal())
    seen_env = []
    teardowns = []

    def fake_stock_get(model_path, adapter_path, *, model_kind="auto"):
        row = {k: os.environ.get(k)
               for k in ("KV_BITS", "APC_ENABLED", "APC_DISK_PATH")}
        # The build spec the load bridge would read - set by _build around this call.
        row["build_spec"] = serving.get_build_spec()
        seen_env.append(row)
        proxy.model_cache = {"model_path": model_path, "model": f"M:{model_path}"}
        proxy.response_generator = f"RG:{model_path}"
        proxy.apc_manager = f"APC:{model_path}"

    def fake_stock_unload():
        teardowns.append(proxy.model_cache.get("model_path"))
        return True

    pool = _ResidencyPool(
        proxy, fake_stock_get, fake_stock_unload,
        int(budget_gb * GB), pinned,
        footprint_fn=lambda p: int(footprint_gb * GB),
        time_fn=clock,
        in_flight_fn=(lambda: in_flight),
    )
    return proxy, pool, seen_env, teardowns


def _acq(pool, path, **kw):
    e = pool.acquire(path, None, "auto", **kw)
    _active_entry.set(e)
    return e


# load_signature widens the cache_key
def test_distinct_load_signature_makes_distinct_entries():
    _proxy, pool, _seen, _td = make_pool()
    _acq(pool, "/m/a.gguf", cache_key_extra=("/m/a.gguf", None, None, False,
                                             (("kv_bits", "8"),), ()))
    _acq(pool, "/m/a.gguf", cache_key_extra=("/m/a.gguf", None, None, False,
                                             (), ()))
    resident = pool.stats()["resident"]
    assert len(resident) == 2                      # same path, two load profiles
    assert {e["model_path"] for e in resident} == {"/m/a.gguf"}


def test_same_signature_shares_one_entry():
    _proxy, pool, _seen, _td = make_pool()
    e1 = _acq(pool, "/m/a.gguf", cache_key_extra=("sig",))
    e2 = pool.acquire("/m/a.gguf", None, "auto", cache_key_extra=("sig",))
    assert e2 is e1
    assert len(pool.stats()["resident"]) == 1


# per-build env window: set around the load, restored after
def test_env_window_set_during_build_then_restored(monkeypatch):
    monkeypatch.delenv("KV_BITS", raising=False)
    monkeypatch.delenv("APC_ENABLED", raising=False)
    _proxy, pool, seen, _td = make_pool()
    _acq(pool, "/m/a.gguf", env={"KV_BITS": "8", "APC_ENABLED": "1"})
    assert seen[-1]["KV_BITS"] == "8"              # set during the stock load
    assert seen[-1]["APC_ENABLED"] == "1"
    assert os.environ.get("KV_BITS") is None       # restored (was unset)
    assert os.environ.get("APC_ENABLED") is None


def test_env_window_does_not_leak_to_sibling_build(monkeypatch):
    monkeypatch.delenv("KV_BITS", raising=False)
    _proxy, pool, seen, _td = make_pool()
    _acq(pool, "/m/a.gguf", env={"KV_BITS": "8"})
    _acq(pool, "/m/b.gguf", env=None)              # no env -> must not see a's KV_BITS
    assert seen[-1]["KV_BITS"] is None


def test_env_window_restores_preexisting_value(monkeypatch):
    monkeypatch.setenv("KV_BITS", "orig")
    _proxy, pool, seen, _td = make_pool()
    _acq(pool, "/m/a.gguf", env={"KV_BITS": "8"})
    assert seen[-1]["KV_BITS"] == "8"
    assert os.environ["KV_BITS"] == "orig"         # prior value restored


# build-spec window: the resolved spec is published for the load bridge (which
# runs in the engine's WORKER thread, where the request-thread ContextVar is
# invisible) during the stock load, then cleared afterward.
def test_build_spec_published_during_build_then_cleared():
    serving.set_build_spec(None)
    try:
        _proxy, pool, seen, _td = make_pool()
        spec = object()
        _acq(pool, "/m/a.gguf", build_spec=spec)
        assert seen[-1]["build_spec"] is spec       # visible to the bridge at load
        assert serving.get_build_spec() is None      # cleared after the build
    finally:
        serving.set_build_spec(None)


def test_build_spec_does_not_leak_to_sibling_build():
    serving.set_build_spec(None)
    try:
        _proxy, pool, seen, _td = make_pool()
        _acq(pool, "/m/a.gguf", build_spec=object())
        _acq(pool, "/m/b.gguf", build_spec=None)     # single-model-style build
        assert seen[-1]["build_spec"] is None        # no carryover from a's build
    finally:
        serving.set_build_spec(None)


# idle-TTL reaper
def test_reaper_unloads_idle_nonpinned_when_drained():
    clock = _Clock(1000.0)
    _proxy, pool, _seen, teardowns = make_pool(in_flight=0, clock=clock)
    _acq(pool, "/m/a.gguf", ttl=900)
    clock.t = 1000 + 901                           # idle past ttl
    reaped = pool.reap_idle()
    assert reaped == ["/m/a.gguf"]
    assert teardowns == ["/m/a.gguf"]
    assert pool.stats()["resident"] == []


def test_reaper_skips_within_ttl():
    clock = _Clock(1000.0)
    _proxy, pool, _seen, _td = make_pool(in_flight=0, clock=clock)
    _acq(pool, "/m/a.gguf", ttl=900)
    clock.t = 1000 + 100                           # still fresh
    assert pool.reap_idle() == []
    assert len(pool.stats()["resident"]) == 1


def test_reaper_skips_pinned():
    clock = _Clock(1000.0)
    _proxy, pool, _seen, _td = make_pool(in_flight=0, clock=clock,
                                         pinned={"/m/a.gguf"})
    _acq(pool, "/m/a.gguf", ttl=900)
    clock.t = 1000 + 5000
    assert pool.reap_idle() == []                  # pinned: never idle-unloaded


def test_reaper_skips_when_server_busy():
    clock = _Clock(1000.0)
    _proxy, pool, _seen, _td = make_pool(in_flight=3, clock=clock)
    _acq(pool, "/m/a.gguf", ttl=900)
    clock.t = 1000 + 5000
    assert pool.reap_idle() == []                  # in-flight > 0 -> never reap


def test_reaper_skips_when_in_flight_unknown():
    clock = _Clock(1000.0)
    _proxy, pool, _seen, _td = make_pool(in_flight=None, clock=clock)
    _acq(pool, "/m/a.gguf", ttl=900)
    clock.t = 1000 + 5000
    assert pool.reap_idle() == []                  # can't confirm drained -> skip


def test_reaper_skips_ttl_none_or_zero():
    clock = _Clock(1000.0)
    _proxy, pool, _seen, _td = make_pool(in_flight=0, clock=clock)
    _acq(pool, "/m/none.gguf", ttl=None)           # never auto-unload
    _acq(pool, "/m/zero.gguf", ttl=0)              # 0 == never
    clock.t = 1000 + 99999
    assert pool.reap_idle() == []
    assert len(pool.stats()["resident"]) == 2


def test_reacquire_resets_idle_timer():
    clock = _Clock(1000.0)
    _proxy, pool, _seen, _td = make_pool(in_flight=0, clock=clock)
    _acq(pool, "/m/a.gguf", ttl=900)
    clock.t = 1500                                 # idle 500
    _acq(pool, "/m/a.gguf", ttl=900)               # touch -> last_access=1500
    clock.t = 2000                                 # idle 500 < 900
    assert pool.reap_idle() == []                  # timer was reset by re-acquire
    assert len(pool.stats()["resident"]) == 1


def test_evict_tears_down_all_entries_for_a_path():
    _proxy, pool, _seen, teardowns = make_pool()
    e1 = _acq(pool, "/m/a.gguf", cache_key_extra=("p1",))
    e2 = _acq(pool, "/m/a.gguf", cache_key_extra=("p2",))   # same path, 2 profiles
    _acq(pool, "/m/b.gguf")
    pool.release(e1)
    pool.release(e2)                                   # requests done -> evictable
    assert pool.evict("/m/a.gguf") is True
    assert teardowns.count("/m/a.gguf") == 2           # both profile entries gone
    assert {e["model_path"] for e in pool.stats()["resident"]} == {"/m/b.gguf"}
    assert pool.evict("/m/missing.gguf") is False


# install-time env resolution: run install_gguf_residency_pool against fake
# mlx_vlm.server modules (importlib returns the sys.modules entries), with the
# reaper thread and atexit hook stubbed - no real mlx_vlm, no daemon leak.
_RESIDENCY_ENV = (
    "MLX_VLM_RESIDENT_BUDGET_GB",
    "MLX_VLM_MAX_RESIDENT_MODELS",
    "MLX_VLM_PINNED_MODELS",
    "MLX_VLM_PRELOAD_MODEL",
    "MLX_VLM_RESIDENT_TTL_DISABLE",
    "MLX_VLM_RESIDENT_TTL_TICK",
)


def install_with_fakes(monkeypatch, env=None, **kwargs):
    """Install the pool over fake mlx_vlm.server modules with exactly ``env``
    of the residency vars set; returns (pool, app module, list of reaper-thread
    kwargs recorded instead of started)."""
    for k in _RESIDENCY_ENV:
        monkeypatch.delenv(k, raising=False)
    for k, v in (env or {}).items():
        monkeypatch.setenv(k, v)
    pkg = types.ModuleType("mlx_vlm.server")
    app = types.ModuleType("mlx_vlm.server.app")
    rt = types.ModuleType("mlx_vlm.server.runtime")
    rt.runtime = object()
    app._INHERIT_ADAPTER = object()
    app.get_cached_model = lambda *a, **k: None
    app.unload_model_sync = lambda: True
    monkeypatch.setitem(sys.modules, "mlx_vlm.server", pkg)
    monkeypatch.setitem(sys.modules, "mlx_vlm.server.app", app)
    monkeypatch.setitem(sys.modules, "mlx_vlm.server.runtime", rt)
    threads = []
    fake_threading = types.SimpleNamespace(
        Thread=lambda **kw: threads.append(kw) or types.SimpleNamespace(
            start=lambda: None),
        RLock=threading.RLock, Lock=threading.Lock)
    monkeypatch.setattr(residency, "threading", fake_threading)
    monkeypatch.setattr(residency.atexit, "register", lambda fn: None)
    residency.install_gguf_residency_pool(**kwargs)
    return pkg._kq_residency_pool, app, threads


def test_resident_env_budget_used_when_unset(monkeypatch, capsys):
    # docs: MLX_VLM_RESIDENT_BUDGET_GB is consulted when neither --budget-gb
    # nor server.budget_gb sets one.
    pool, _app, _threads = install_with_fakes(
        monkeypatch, env={"MLX_VLM_RESIDENT_BUDGET_GB": "8"})
    assert pool.stats()["budget_bytes"] == 8 * GB


def test_resident_flag_beats_env(monkeypatch, capsys):
    # explicit budget_bytes (the --budget-gb / server.budget_gb value) wins
    # over the env var.
    pool, _app, _threads = install_with_fakes(
        monkeypatch, env={"MLX_VLM_RESIDENT_BUDGET_GB": "8"},
        budget_bytes=4 * GB)
    assert pool.stats()["budget_bytes"] == 4 * GB


def test_resident_pinned_models_env_unioned(monkeypatch, capsys):
    pool, _app, _threads = install_with_fakes(
        monkeypatch, env={"MLX_VLM_PINNED_MODELS": "/m/x.gguf, /m/y.gguf"},
        budget_bytes=GB, pinned={"/m/z.gguf"})
    assert pool._pinned_paths == {"/m/x.gguf", "/m/y.gguf", "/m/z.gguf"}


def test_resident_max_models_env(monkeypatch, capsys):
    pool, _app, _threads = install_with_fakes(
        monkeypatch, env={"MLX_VLM_MAX_RESIDENT_MODELS": "3"},
        budget_bytes=GB)
    assert pool._max == 3


def test_resident_ttl_disable_env(monkeypatch, capsys):
    _pool, _app, threads = install_with_fakes(
        monkeypatch, env={"MLX_VLM_RESIDENT_TTL_DISABLE": "1"},
        budget_bytes=GB)
    assert threads == []                           # no reaper thread
    _pool, _app, threads = install_with_fakes(
        monkeypatch, env={"MLX_VLM_RESIDENT_TTL_DISABLE": "0"},
        budget_bytes=GB)
    assert [t["name"] for t in threads] == ["gmlx-residency-ttl"]
    assert threads[0]["daemon"] is True


def test_default_budget_fraction(monkeypatch):
    # docs/server-config.md: default budget = 0.8x the GPU recommended working set.
    import mlx.core as mx

    monkeypatch.setattr(
        mx, "device_info",
        lambda: {"max_recommended_working_set_size": 100 * GB})
    assert residency._default_budget_bytes() == int(0.8 * 100 * GB)
    assert residency._DEFAULT_BUDGET_FRACTION == 0.8


def test_resolver_error_becomes_http_404_through_dispatch(monkeypatch, capsys):
    # The real path: pooled_get_cached_model raises the HTTPException from
    # _http_from_resolver_error, and the endpoint's mlx-vlm-style wrap
    # (except HTTPException: raise; except Exception: -> 500) must pass the
    # 404 through with the documented body, never a 500.
    from fastapi import FastAPI, HTTPException
    from fastapi.testclient import TestClient

    _pool, app_mod, _threads = install_with_fakes(monkeypatch, budget_bytes=GB)
    getter = app_mod.get_cached_model              # the pooled replacement
    monkeypatch.setattr(serving, "server_config", lambda: object())

    def raise_not_found(model, *, profile_field=None):
        raise serving.ModelNotFound(model, ["alpha", "beta"])

    monkeypatch.setattr(serving, "resolve_request_model", raise_not_found)

    api = FastAPI()

    @api.get("/probe/{model}")
    def probe(model: str):
        try:
            getter(model)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        return {"ok": True}

    r = TestClient(api).get("/probe/nope")
    assert r.status_code == 404
    # A bare app (no gmlx handlers) renders FastAPI's {"detail": ...} wrapper;
    # the detail itself is the unified {"error": {...}} envelope the real app's
    # HTTPException handler serves directly.
    err = r.json()["detail"]["error"]
    assert err["type"] == "model_not_found"
    assert err["available_models"] == ["alpha", "beta"]
    assert "nope" in err["message"]


def test_http_from_resolver_error_maps_status():
    # mlx-vlm endpoints swallow non-HTTPException into a 500; the resolver errors
    # must become HTTPException with the right 4xx + the available ids/profiles.
    from gmlx import residency, server_bridge_vlm as serving
    nf = residency._http_from_resolver_error(
        serving.ModelNotFound("x", ["a", "b"]))
    assert nf.status_code == 404
    assert nf.detail["error"]["type"] == "model_not_found"
    assert nf.detail["error"]["available_models"] == ["a", "b"]
    up = residency._http_from_resolver_error(
        serving.UnknownProfile("p", ["coder", "creative"]))
    assert up.status_code == 400
    assert up.detail["error"]["available_profiles"] == ["coder", "creative"]
    ns = residency._http_from_resolver_error(serving.NoModelSpecified(["a"]))
    assert ns.status_code == 400
    assert ns.detail["error"]["available_models"] == ["a"]
    assert residency._http_from_resolver_error(ValueError("other")) is None


# keep tier: TTL-exempt but NOT LRU-exempt (a softer pin; the LRU-still-evicts
# half lives in test_residency.py, which has the byte-budget harness)
def test_kept_path_survives_the_reaper():
    clock = _Clock(1000.0)
    _proxy, pool, _seen, teardowns = make_pool(in_flight=0, clock=clock)
    _acq(pool, "/m/a.gguf", ttl=900)
    pool.set_keep("/m/a.gguf", True)
    clock.t = 1000 + 5000                          # long idle, well past the ttl
    assert pool.reap_idle() == []                  # kept: never idle-unloaded
    assert teardowns == []
    assert len(pool.stats()["resident"]) == 1


def test_unkeep_re_arms_the_reaper():
    clock = _Clock(1000.0)
    _proxy, pool, _seen, teardowns = make_pool(in_flight=0, clock=clock)
    _acq(pool, "/m/a.gguf", ttl=900)
    pool.set_keep("/m/a.gguf", True)
    pool.set_keep("/m/a.gguf", False)              # release the keep
    clock.t = 1000 + 901
    assert pool.reap_idle() == ["/m/a.gguf"]       # reaper re-armed
    assert teardowns == ["/m/a.gguf"]


def test_evict_drops_the_keep_mark():
    clock = _Clock(1000.0)
    _proxy, pool, _seen, _td = make_pool(in_flight=0, clock=clock)
    e = _acq(pool, "/m/a.gguf", ttl=900)
    pool.set_keep("/m/a.gguf", True)
    pool.release(e)                                # request done -> evictable
    assert pool.evict("/m/a.gguf") is True
    # a later reload of the same path is NOT silently re-kept - the ttl governs again
    _acq(pool, "/m/a.gguf", ttl=900)
    clock.t = 1000 + 901
    assert pool.reap_idle() == ["/m/a.gguf"]


def test_stats_surfaces_the_kept_flag():
    _proxy, pool, _seen, _td = make_pool(in_flight=0)
    _acq(pool, "/m/a.gguf", ttl=900)
    _acq(pool, "/m/b.gguf", ttl=900)
    pool.set_keep("/m/a.gguf", True)
    by_path = {e["model_path"]: e for e in pool.stats()["resident"]}
    assert by_path["/m/a.gguf"]["kept"] is True
    assert by_path["/m/b.gguf"]["kept"] is False


def test_stats_exposes_ttl_and_idle():
    clock = _Clock(1000.0)
    _proxy, pool, _seen, _td = make_pool(in_flight=0, clock=clock)
    _acq(pool, "/m/a.gguf", ttl=900)
    clock.t = 1300
    row = pool.stats()["resident"][0]
    assert row["ttl_s"] == 900
    assert row["idle_s"] == pytest.approx(300.0)


# explicit unload vs in-flight requests: an /unload must never stop another
# client's running generation
def test_evict_refuses_busy_model():
    from gmlx.residency import ModelBusyError
    _proxy, pool, _seen, teardowns = make_pool()
    entry = _acq(pool, "/m/a.gguf")            # acquire holds busy=1
    with pytest.raises(ModelBusyError):
        pool.evict("/m/a.gguf")
    assert teardowns == []                     # nothing was torn down
    pool.release(entry)
    assert pool.evict("/m/a.gguf") is True     # released -> evictable


def test_clear_skips_busy_entries():
    _proxy, pool, _seen, teardowns = make_pool()
    busy = _acq(pool, "/m/busy.gguf")
    idle = _acq(pool, "/m/idle.gguf")
    pool.release(idle)
    assert pool.clear() is True                # idle went, busy stayed
    assert teardowns == ["/m/idle.gguf"]
    assert pool.busy_paths() == ["/m/busy.gguf"]
    pool.release(busy)
    assert pool.clear() is True
    assert pool.busy_paths() == []
