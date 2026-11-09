#!/usr/bin/env python3
"""Residency config extensions: load_signature widens the cache_key (one GGUF,
two profiles -> two entries), the per-build env window is set then restored (no
leak to a sibling load), and the idle-TTL reaper unloads idle non-pinned entries
only when the server is drained. CPU-only - fake stock load/teardown, injected
footprint + clock + in-flight reader, so no GPU and no model files."""
from __future__ import annotations

import contextvars
import os
import sys
import threading
import types

import pytest

import gmlx.serve.residency as residency  # noqa: E402
import gmlx.serve.bridge_vlm as serving  # noqa: E402
from gmlx.serve.residency import (  # noqa: E402
    _active_entry,
    _ResidencyPool,
    _RuntimeProxy,
)

# A Mac of 64 GB: the pool juggles models of 10 to 22 GB.
pytestmark = pytest.mark.host_gb(64)

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
               for k in ("KV_BITS", "APC_ENABLED", "GMLX_APC_ENABLED",
                         "APC_DISK_PATH")}
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


# per-build env window: set around the load, restored after. APC_ENABLED is
# the exception: the window value is captured into GMLX_APC_ENABLED for the
# bridge's gmlx-owned manager build, and the stock var is pinned to 0 so
# apc.from_env stays unpatched and unused (builds nothing to discard).
def test_env_window_set_during_build_then_restored(monkeypatch):
    monkeypatch.delenv("KV_BITS", raising=False)
    monkeypatch.delenv("APC_ENABLED", raising=False)
    monkeypatch.delenv("GMLX_APC_ENABLED", raising=False)
    _proxy, pool, seen, _td = make_pool()
    _acq(pool, "/m/a.gguf", env={"KV_BITS": "8", "APC_ENABLED": "1"})
    assert seen[-1]["KV_BITS"] == "8"              # set during the stock load
    assert seen[-1]["APC_ENABLED"] == "0"          # pinned: from_env builds nothing
    assert seen[-1]["GMLX_APC_ENABLED"] == "1"     # effective value, gmlx gate
    assert os.environ.get("KV_BITS") is None       # restored (was unset)
    assert os.environ.get("APC_ENABLED") is None
    assert os.environ.get("GMLX_APC_ENABLED") is None


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
    pool.release(_acq(pool, "/m/a.gguf", ttl=900))  # request done
    clock.t = 1000 + 901                           # idle past ttl
    reaped = pool.reap_idle()
    assert reaped == ["/m/a.gguf"]
    assert teardowns == ["/m/a.gguf"]
    assert pool.stats()["resident"] == []


def test_reaper_keeps_a_chat_model_held_by_a_job_outside_the_requests():
    """A video job rewrites its prompt through a chat model on its driver
    thread, where no request is in flight, and the caption can outlast
    the model's TTL. The job's hold keeps the model until the caption
    ends. A preload's lifetime hold keeps none."""
    clock = _Clock(1000.0)
    _proxy, pool, _seen, teardowns = make_pool(in_flight=0, clock=clock)
    up = pool.acquire("/m/up.gguf", None, "auto", ttl=900)
    pre = pool.acquire("/m/pre.gguf", None, "auto", ttl=900)
    preload = residency._BusyHold(pool, pre)
    pool.mark_retained(preload)
    clock.t = 1000 + 901
    assert pool.reap_idle() == ["/m/pre.gguf"]
    assert teardowns == ["/m/pre.gguf"]
    pool.release(up)
    assert pool.reap_idle() == ["/m/up.gguf"]
    assert preload._entry is pre


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
    pkg = types.ModuleType("gmlx._vendor.mlx_vlm.server")
    app = types.ModuleType("gmlx._vendor.mlx_vlm.server.app")
    rt = types.ModuleType("gmlx._vendor.mlx_vlm.server.runtime")
    rt.runtime = object()
    app._INHERIT_ADAPTER = object()
    app.get_cached_model = lambda *a, **k: None
    app.unload_model_sync = lambda: True
    monkeypatch.setitem(sys.modules, "gmlx._vendor.mlx_vlm.server", pkg)
    monkeypatch.setitem(sys.modules, "gmlx._vendor.mlx_vlm.server.app", app)
    monkeypatch.setitem(sys.modules, "gmlx._vendor.mlx_vlm.server.runtime", rt)
    threads = []
    fake_threading = types.SimpleNamespace(
        Thread=lambda **kw: threads.append(kw) or types.SimpleNamespace(
            start=lambda: None),
        RLock=threading.RLock, Lock=threading.Lock,
        Condition=threading.Condition,
        current_thread=threading.current_thread)
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
    # docs/config.md: default budget = 0.8x the GPU recommended working set.
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
    import gmlx.serve.residency as residency
    import gmlx.serve.bridge_vlm as serving
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
    pool.release(_acq(pool, "/m/a.gguf", ttl=900))
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
    pool.release(_acq(pool, "/m/a.gguf", ttl=900))
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
    from gmlx.serve.residency import ModelBusyError
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


def test_deferred_load_inside_the_handler_is_the_typed_503(monkeypatch, request):
    # The chat pre-warm answers a gate-deferred load typed; a load that only
    # begins at the handler's own acquire (the room went to another request
    # in between) used to fall to mlx-vlm's generic 500. pooled_get_cached_model
    # now raises the same 503 body + Retry-After as an HTTPException, which
    # the endpoint wrappers re-raise untouched.
    from fastapi import FastAPI, HTTPException
    from fastapi.testclient import TestClient

    from gmlx.serve.capacity import LoadDeferred
    import gmlx.serve.patches.capacity_routes as cr

    from gmlx.config import build_config

    pool, app_mod, _threads = install_with_fakes(monkeypatch, budget_bytes=GB)
    serving.register_resolved_models(build_config({"models": {"m": {"path": "/abs/m.gguf"}}}))
    request.addfinalizer(serving.clear_resolved_models)
    getter = app_mod.get_cached_model

    def deferred(*a, **k):
        raise LoadDeferred("model load deferred: m weights 86.7 GB exceed the "
                           "measured free working set -0.7 GB")

    monkeypatch.setattr(pool, "acquire", deferred)
    monkeypatch.setattr(cr, "readiness", lambda: (False, "busy", 9))

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

    r = TestClient(api).get("/probe/m")
    assert r.status_code == 503
    assert r.headers["retry-after"] == "9"
    err = r.json()["detail"]["error"]
    assert err["type"] == "model_load_deferred" and err["model"] == "m"
    assert err["retry_after_s"] == 9 and "86.7 GB" in err["message"]


def test_evict_and_clear_can_discount_retained_holds():
    """An explicit unload judges busy-ness on real streams: the preload's
    process-lifetime hold (retained) is discounted, a second acquire is
    not, and a refused eviction leaves the entry (and so the hold) in
    place - ``entry_resident`` is how the route tells the two apart."""
    import gmlx.serve.residency as residency
    from gmlx.serve.residency import ModelBusyError

    _proxy, pool, _env, teardowns = make_pool()
    entry = _acq(pool, "/abs/a.gguf")
    hold = residency._BusyHold(pool, entry)
    pool.mark_retained(hold)
    assert entry.busy == 1 and entry.retained == 1
    with pytest.raises(ModelBusyError):
        pool.evict("/abs/a.gguf")                    # stock: the hold counts
    stream = _acq(pool, "/abs/a.gguf")               # a real in-flight stream
    assert stream is entry and entry.busy == 2
    with pytest.raises(ModelBusyError) as ei:
        pool.evict("/abs/a.gguf", ignore_retained=True)
    assert ei.value.in_flight == 1                   # the stream, not the hold
    assert pool.entry_resident(entry)
    assert pool.clear(ignore_retained=True) is False and pool.entry_resident(entry)
    pool.release(entry)                              # stream done
    assert pool.evict("/abs/a.gguf", ignore_retained=True) is True
    assert not pool.entry_resident(entry) and teardowns == ["/abs/a.gguf"]
    hold.release()                                   # late release: harmless


def test_profile_label_is_stable_across_acquires_and_unique_per_entry():
    """The metrics label for a resident entry must not move: ``seq`` is
    the LRU clock (re-stamped per acquire) and labelling on it minted a
    new Prometheus series on every scrape. ``profile`` is fixed for the
    entry's lifetime and still distinguishes two entries on one GGUF."""
    import importlib
    from gmlx.serve.patches import capacity_routes as cr
    from gmlx.serve.patches import routes as sp_routes

    _proxy, pool, _env, _td = make_pool()
    pkg = importlib.import_module("gmlx._vendor.mlx_vlm.server")
    saved = getattr(pkg, "_kq_residency_pool", None)
    pkg._kq_residency_pool = pool
    try:
        def labels():
            return [cr._entry_label(e, i)
                    for i, e in enumerate(sp_routes._resident_models_view())]

        e = _acq(pool, "/abs/a.gguf")
        first = labels()
        seq0 = pool.stats()["resident"][0]["seq"]
        for _ in range(3):
            pool.release(_acq(pool, "/abs/a.gguf"))
        assert pool.stats()["resident"][0]["seq"] > seq0        # LRU clock moved
        assert labels() == first                                 # label did not
        assert first == [{"model": "a.gguf", "profile": "default"}]
        # same GGUF, an adapter and a different load signature: two entries,
        # two labels, no duplicate series
        pool.acquire("/abs/a.gguf", "/abs/lora.gguf", "auto")
        pool.acquire("/abs/a.gguf", None, "auto", cache_key_extra=("kv4",))
        # ... and a service-kind route (--embeddings) on the same GGUF: the
        # kind is the only key component that differs
        pool.acquire("/abs/a.gguf", None, "embedding")
        # upstream's explicit text_generation kind is its own key: it must
        # not collapse into the same ``default`` label as gmlx's ``auto``
        pool.acquire("/abs/a.gguf", None, "text_generation")
        # a config-mode load names the entry by the id that built it
        pool.acquire("/abs/a.gguf", None, "auto", cache_key_extra=("kv8",),
                     loaded_as="a-q8@coder")
        labs = labels()
        assert len(labs) == 6 and len({tuple(sorted(d.items())) for d in labs}) == 6
        named = [d for d in labs if d["model"] == "a-q8@coder"]
        assert len(named) == 1 and named[0]["profile"] != "default"
        assert {"model": "a.gguf", "profile": "lora.gguf"} in labs
        assert {"model": "a.gguf", "profile": "embedding"} in labs
        assert {"model": "a.gguf", "profile": "text_generation"} in labs
        pool.release(e)
    finally:
        pkg._kq_residency_pool = saved


class _Acquired(Exception):
    pass


def _acquired(monkeypatch, pool):
    """Record each path the pool is asked to load, then stop there."""
    seen = []

    def acquire(path, *a, **k):
        seen.append(path)
        raise _Acquired(path)
    monkeypatch.setattr(pool, "acquire", acquire)
    return seen


@pytest.mark.parametrize("model", ["/Users/you/src/proj/planted.gguf",
                                   "mlx-community/some-model", "planted.gguf"])
def test_no_registered_config_never_loads_a_request_path(monkeypatch, model):
    """A request never names a load path or a hub id, even with no config
    registered, so a client holding the key cannot make the Mac load a file."""
    from fastapi import HTTPException

    serving.clear_resolved_models()
    pool, app_mod, _threads = install_with_fakes(monkeypatch, budget_bytes=GB)
    seen = _acquired(monkeypatch, pool)
    with pytest.raises(HTTPException) as e:
        app_mod.get_cached_model(model)
    assert e.value.status_code == 404 and seen == []


@pytest.mark.parametrize("model", ["/Users/you/src/proj/planted.gguf",
                                   "mlx-community/some-model"])
def test_a_single_model_server_serves_only_its_own_model(monkeypatch, request, tmp_path,
                                                         model):
    """`gmlx serve model.gguf` wraps the file in a one-model config, so another
    path or a hub id in a request gets the same 404 as in config mode."""
    import argparse

    from fastapi import HTTPException

    from gmlx.serve import server

    own = tmp_path / "own.gguf"
    own.write_bytes(b"GGUF")
    ap = argparse.ArgumentParser()
    server._add_serve_args(ap)
    a = ap.parse_args([str(own)])
    cfg = server._single_model_cfg(a)
    serving.register_resolved_models(cfg)
    request.addfinalizer(serving.clear_resolved_models)
    pool, app_mod, _threads = install_with_fakes(monkeypatch, budget_bytes=GB)
    seen = _acquired(monkeypatch, pool)
    with pytest.raises(HTTPException) as e:
        app_mod.get_cached_model(model)
    assert e.value.status_code == 404 and seen == []
    # In a copy of the context, so the request's active spec stays behind.
    ctx = contextvars.copy_context()
    with pytest.raises(_Acquired):
        ctx.run(app_mod.get_cached_model, next(iter(cfg.models)))
    assert seen == [str(own)]


# A chat route that names an image model, and the reverse, is a 400 that
# names the route the id belongs to.
_MIXED = {
    "models": {"qwen": {"path": "/abs/qwen.gguf"}},
    "aliases": {"q": "qwen"},
    "image_models": {"zit": {"path": "/abs/zit.safetensors",
                             "encoder": "/abs/enc.gguf",
                             "vae": "/abs/vae.safetensors",
                             "ttl_s": 60}},
}


@pytest.fixture
def mixed_cfg(request):
    from gmlx.config import build_config

    cfg = build_config(_MIXED)
    serving.register_resolved_models(cfg)
    request.addfinalizer(serving.clear_resolved_models)
    return cfg


@pytest.mark.parametrize("kind", ["image_generation", "image_edit"])
def test_pooled_get_refuses_the_stock_image_kinds(monkeypatch, kind):
    from fastapi import HTTPException

    pool, app_mod, _threads = install_with_fakes(monkeypatch, budget_bytes=GB)
    seen = _acquired(monkeypatch, pool)
    with pytest.raises(HTTPException) as e:
        app_mod.get_cached_model("anything", model_kind=kind)
    assert e.value.status_code == 400 and seen == []


def test_a_chat_route_naming_an_image_model_is_a_400(monkeypatch, mixed_cfg):
    from fastapi import HTTPException

    pool, app_mod, _threads = install_with_fakes(monkeypatch, budget_bytes=GB)
    seen = _acquired(monkeypatch, pool)
    with pytest.raises(HTTPException) as e:
        app_mod.get_cached_model("zit")
    err = e.value.detail["error"]
    assert e.value.status_code == 400 and seen == []
    assert err["type"] == "wrong_model_kind"
    assert "/v1/images/generations" in err["message"]
    assert err["available_models"] == ["qwen"]


@pytest.mark.parametrize("model", ["qwen", "q"])
def test_an_image_route_naming_a_chat_model_is_a_400(monkeypatch, mixed_cfg,
                                                     model):
    from fastapi import HTTPException

    proxy, pool, _env, _td = make_pool()
    seen = _acquired(monkeypatch, pool)
    with pytest.raises(HTTPException) as e:
        residency.acquire_image_model(model, pool=pool)
    err = e.value.detail["error"]
    assert e.value.status_code == 400 and seen == []
    assert err["type"] == "wrong_model_kind"
    assert err["available_models"] == ["zit"]


def test_an_image_route_naming_an_unknown_model_is_a_404(monkeypatch, mixed_cfg):
    from fastapi import HTTPException

    proxy, pool, _env, _td = make_pool()
    with pytest.raises(HTTPException) as e:
        residency.acquire_image_model("nope", pool=pool)
    assert e.value.status_code == 404
    assert e.value.detail["error"]["available_models"] == ["zit"]


def test_acquire_image_model_builds_the_image_kind(monkeypatch, mixed_cfg):
    from gmlx.serve import capacity as cap

    monkeypatch.setattr(cap, "preload_gate", lambda *a, **k: None)
    proxy, pool, seen_env, _td = make_pool()
    specs = []
    pool._image_fns = (lambda spec: 9 * GB,
                       lambda spec: specs.append(spec) or "PIPE",
                       lambda box: box.clear())
    for field in ("zit", "", None):           # the sole image model is the default
        hold = residency.acquire_image_model(field, pool=pool)
        entry = hold._entry
        hold.release()
    assert seen_env == []                     # the chat loader never ran
    assert [s.id for s in specs] == ["zit"]   # one build, then hits
    assert entry.kind == residency.IMAGE_KIND and entry.pipeline == "PIPE"
    assert entry.ttl == 60 and entry.loaded_as == "zit"
    assert entry.cache_key[:3] == ("/abs/zit.safetensors", None,
                                   residency.IMAGE_KIND)


_VIDEO = {**_MIXED, "video_models": {"h3": {
    "path": "/abs/h3.gguf", "encoder": "/abs/qwen3vl.gguf",
    "vae": "/abs/video_vae.safetensors",
    "audio_vae": "/abs/audio_vae.safetensors"}}}


def test_a_video_model_loads_and_releases_through_its_own_kind(monkeypatch,
                                                                request):
    """A video id builds on the video triple, never the image one, the
    runtime view names it by its id, and its eviction releases it through
    the video release."""
    from gmlx.config import build_config
    from gmlx.serve import capacity as cap

    serving.register_resolved_models(build_config(_VIDEO))
    request.addfinalizer(serving.clear_resolved_models)
    monkeypatch.setattr(cap, "preload_gate", lambda *a, **k: None)
    proxy, pool, seen_env, _td = make_pool()
    calls = []
    pool._image_fns = tuple(
        (lambda *a, n=n: calls.append(("image", n))) for n in range(3))
    pool._video_fns = (lambda spec: 30 * GB,
                       lambda spec: calls.append(("load", spec.id)) or "VID",
                       lambda box: calls.append(("release", box.pop())))
    hold = residency.acquire_video_model("h3", pool=pool)
    entry = hold._entry
    hold.release()
    assert seen_env == []
    assert entry.kind == residency.VIDEO_KIND and entry.pipeline == "VID"
    assert entry.footprint == 30 * GB and entry.loaded_as == "h3"
    import importlib

    from gmlx.serve.patches import routes as sp_routes

    pkg = importlib.import_module("gmlx._vendor.mlx_vlm.server")
    monkeypatch.setattr(pkg, "_kq_residency_pool", pool, raising=False)
    view = sp_routes._resident_models_view()
    assert [(v["kind"], v["ids"]) for v in view] == [("video", ["h3"])]
    assert pool.evict("/abs/h3.gguf")
    assert calls == [("load", "h3"), ("release", "VID")]


def test_two_video_entries_apart_only_in_int8_load_apart(monkeypatch,
                                                           request):
    """h3 and h3-int8 on the same files at int8 off and all, asked for in
    turn, each load at their own setting: the pool does not hand one the
    other's DiT."""
    from gmlx.config import build_config
    from gmlx.serve import capacity as cap

    h3 = _VIDEO["video_models"]["h3"]
    serving.register_resolved_models(build_config({**_VIDEO, "video_models": {
        "h3": {**h3, "defaults": {"int8": "off"}},
        "h3-int8": {**h3, "defaults": {"int8": "all"}}}}))
    request.addfinalizer(serving.clear_resolved_models)
    monkeypatch.setattr(cap, "preload_gate", lambda *a, **k: None)
    proxy, pool, seen_env, _td = make_pool()
    loads = []
    pool._video_fns = (lambda spec: 30 * GB,
                       lambda spec: loads.append(
                           (spec.id, spec.defaults["int8"])) or spec.id,
                       lambda box: box.clear())
    for name in ("h3", "h3-int8"):
        hold = residency.acquire_video_model(name, pool=pool)
        assert hold._entry.pipeline == name
        hold.release()
    assert loads == [("h3", "off"), ("h3-int8", "all")]


# The published files of the h3 recipes, in bytes: the DiT, the text
# encoder, the video and audio VAEs, the 8-step LoRA and FastH3's DiT.
_H3_FILES = {"dit": 11_420_663_904, "enc": 19_762_151_200,
             "vae": 5_207_808_496, "audio": 605_254_808,
             "lora": 1_956_193_000, "fdit": 13_613_532_704,
             "mmproj": 1_200_334_496}
# The bytes unsloth's mmproj-BF16.gguf holds once its tower is attached.
_TOWER = 2_383_479_488


def _h3_entry(monkeypatch, variant, size, seconds, budget=120 * GB,
              extra=None, mmproj=False):
    """An h3 recipe's entry at a default clip, from its published file
    sizes, on a working budget of ``budget`` bytes, with a bf16 LoRA file
    of ``extra`` bytes beside the recipe's and with the encoder's mmproj
    when ``mmproj``. FastH3 runs as on an M5 with an mlx-kquant that has
    its sparse attention."""
    from gmlx.serve import capacity
    from gmlx.video import loader, memfit, pipeline, registry
    from gmlx.video.registry import get_family
    from video_fixtures import H3_HEADER, fixture_view

    fast = variant == "fasth3"
    monkeypatch.setattr(loader, "identify_dit", lambda path, variant=None:
                        types.SimpleNamespace(
                            family=get_family("h3"),
                            variant="fasth3" if fast else "fl2va",
                            config={"text_dim": 5120,
                                    "hidden_size": 5376},
                            view=fixture_view(H3_HEADER)))
    if fast:
        monkeypatch.setattr(registry, "run_refusal",
                            lambda fam, v, int8="off": None)
    files = {**_H3_FILES, "extra": extra}
    monkeypatch.setattr(memfit, "file_bytes", lambda p: files[p])
    monkeypatch.setattr(memfit, "lora_dtype", lambda p: "BF16")
    monkeypatch.setattr(memfit, "tower_bytes", lambda p: {
        "mmproj": _TOWER}[p])
    monkeypatch.setattr(pipeline, "vet_adapter", lambda *a, **k: None)
    monkeypatch.setattr(capacity, "working_budget_bytes", lambda: budget)
    return types.SimpleNamespace(
        id=variant, path="fdit" if fast else "dit", variant=None,
        encoder="enc", vae="vae", audio_vae="audio",
        encoder_mmproj="mmproj" if mmproj else None,
        adapters=((("lora", 1.0),) if variant == "h3" else ())
        + ((("extra", 1.0),) if extra else ()),
        defaults={"size": size, "seconds": seconds})


# The refined text rows an h3 entry keeps: 8 prompts of the priced rows
# at its 5376-wide DiT in bf16, 84 MiB.
_CAPTIONS = 8 * 1024 * 5376 * 2


def _cache_term() -> int:
    """The caches an entry books at a cache share of 1: the tensor caches
    with the step routes' tensor cache. A request's LoRA bytes are booked
    at its job's turns, outside it."""
    from gmlx.serve import image_service

    tensor, _cap = image_service.cache_bytes()
    return 2 * tensor + image_service._CACHE_BYTES["steps"]


@pytest.mark.parametrize("variant, size, seconds", [
    # h3-base at 512x512 for a second: the encoder alone is the larger.
    ("h3-base", "512x512", 1),
    # h3 at 768x768 for five seconds, a 32 GB Mac's clip: the DiT with its
    # LoRA and the video decode.
    ("h3", "768x768", 5),
])
def test_a_video_entry_is_priced_at_its_largest_phase(monkeypatch, variant,
                                                      size, seconds):
    """The encode runs with the encoder alone, and every other phase with
    the DiT and its LoRAs: the entry counts the larger, each with the
    caches the server holds through it and its refined text rows."""
    from gmlx.serve import video_service as vs

    got = vs.entry_plan(_h3_entry(monkeypatch, variant, size, seconds))
    clip = vs.plan_clip(got, frames=got.frames, width=got.size[0],
                        height=got.size[1], text=vs.PRICED_TEXT_ROWS)
    dit = clip.terms["denoise"]["dit"] + clip.terms["denoise"]["lora"]
    caches = _cache_term()
    encode = clip.phases["encode"] + caches + _CAPTIONS
    resident = dit + clip.phases["decode"] + caches + _CAPTIONS
    assert got.cache_share == 1.0
    assert got.lora_term == clip.terms["denoise"]["lora"]
    if variant == "h3-base":
        assert got.footprint == encode and encode > resident
    else:
        assert clip.phases["decode"] > clip.terms["denoise"]["step"]
        assert got.footprint == resident and resident > encode


@pytest.mark.parametrize("extra", [None, 300_000_000])
def test_a_video_entrys_own_lora_term_is_its_file_once_alone(monkeypatch,
                                                              extra):
    """The footprint books an h3 entry's own LoRA at its file once when it
    is the one bf16 file, and at lora_k times the files beside a second:
    the term its books hold its own pairs and default table against."""
    from gmlx.serve import video_service as vs
    from gmlx.video import memfit

    got = vs.entry_plan(_h3_entry(monkeypatch, "h3", "768x768", 5,
                                  extra=extra))
    own = vs.ServedVideo(pipe=None, spec=None, plan=got).lora_own_term
    lora = _H3_FILES["lora"]
    want = lora if extra is None else int(
        memfit.PEAKS["h3"].lora_k * (lora + extra))
    assert own == got.lora_term == want


def test_an_int8_entry_checks_the_kernel_and_books_its_step(monkeypatch):
    """An entry whose defaults hold int8: attention is refused on an
    mlx-kquant without sdpa_qk8 before it is priced, and its step books
    the int8 copies of q and k by packed row."""
    import mlx_kquant as kq

    from gmlx.serve import video_service as vs
    from gmlx.video import memfit
    from gmlx.video.fingerprint import VideoModelError

    dense = vs.entry_plan(_h3_entry(monkeypatch, "h3-base", "768x768", 5))
    spec = _h3_entry(monkeypatch, "h3-base", "768x768", 5)
    spec.defaults["int8"] = "attention"
    monkeypatch.delattr(kq, "sdpa_qk8", raising=False)
    with pytest.raises(VideoModelError) as e:
        vs.entry_plan(spec)
    assert str(e.value).startswith(
        "video model 'h3-base': MiniMax-H3 fl2va runs with int8 attention, "
        "and the installed mlx-kquant")
    monkeypatch.setattr(kq, "sdpa_qk8", lambda *a: None, raising=False)
    monkeypatch.setattr(kq, "nax_available", lambda: True)
    got = vs.entry_plan(spec)
    assert (dense.attention, got.attention) == ("dense", "int8")
    plans = [vs.plan_clip(p, frames=p.frames, width=p.size[0],
                          height=p.size[1], text=vs.PRICED_TEXT_ROWS)
             for p in (dense, got)]
    step = [p.terms["denoise"]["step"] for p in plans]
    assert step[1] - step[0] == (memfit.PEAKS["h3"].int8_attention
                                 * plans[1].rows)


# The int8 DiT's bytes past its file, from the H3 header: each block
# linear at one int8 byte a weight and a float32 scale a row, less its
# Q4_K bytes (design 3.5).
_INT8_DELTA = 8_441_753_600


@pytest.mark.parametrize("int8", ["linears", "all"])
def test_an_int8_linears_entry_books_its_int8_dit_and_step(monkeypatch,
                                                           int8):
    """An entry at int8 linears or all is refused on an mlx-kquant without
    int8_matmul before it is priced. Its plan books the int8 DiT, the file
    and the header's delta, and its step books the int8 copy of fc2's
    input by packed row; its signature carries the setting."""
    import mlx_kquant as kq

    from gmlx.serve import video_service as vs
    from gmlx.video import memfit
    from gmlx.video.fingerprint import VideoModelError
    from video_fixtures import int8_kernel

    int8_kernel(monkeypatch)
    off = vs.entry_plan(_h3_entry(monkeypatch, "h3-base", "768x768", 5))
    spec = _h3_entry(monkeypatch, "h3-base", "768x768", 5)
    spec.defaults["int8"] = int8
    monkeypatch.delattr(kq, "int8_matmul")
    with pytest.raises(VideoModelError) as e:
        vs.entry_plan(spec)
    assert str(e.value) == (
        "video model 'h3-base': MiniMax-H3 fl2va runs with int8 linears, "
        f"and the installed mlx-kquant {kq.__version__} has no int8_matmul; "
        "update mlx-kquant to 0.5.11 or later")
    int8_kernel(monkeypatch)
    got = vs.entry_plan(spec)
    assert (off.linears, got.linears, got.int8) == ("file", "int8", int8)
    assert got.dit_bytes - off.dit_bytes == _INT8_DELTA
    plans = [vs.plan_clip(p, frames=p.frames, width=p.size[0],
                          height=p.size[1], text=vs.PRICED_TEXT_ROWS)
             for p in (off, got)]
    step = [p.terms["denoise"]["step"] for p in plans]
    qk8 = memfit.PEAKS["h3"].int8_attention if int8 == "all" else 0
    assert step[1] - step[0] == ((memfit.PEAKS["h3"].int8_linears + qk8)
                                 * plans[1].rows)
    assert plans[1].terms["denoise"]["dit"] - plans[0].terms["denoise"][
        "dit"] == _INT8_DELTA


def test_the_caches_of_a_tight_video_entry_take_the_room_its_clip_leaves(
        monkeypatch):
    """On a budget the clip fills but for half the room its caches are
    bounded at, the entry books each cache at about half its bound."""
    from gmlx.serve import video_service as vs

    caches = _cache_term()
    full = vs.entry_plan(_h3_entry(monkeypatch, "h3", "768x768", 5))
    bare = full.footprint - caches
    budget = bare + caches // 2
    got = vs.entry_plan(_h3_entry(monkeypatch, "h3", "768x768", 5, budget))
    assert 0.4 < got.cache_share <= 0.5
    assert bare < got.footprint <= budget


def _bare(monkeypatch, variant, size, seconds) -> tuple:
    """(entry at share 1, bare clip price) of an h3 recipe's entry: what
    it books with every cache empty."""
    from gmlx.serve import video_service as vs

    caches = _cache_term()
    full = vs.entry_plan(_h3_entry(monkeypatch, variant, size, seconds))
    return full, full.footprint - caches


# One cond of the priced rows of H3's 5120-wide encoder features in f32,
# 20 MiB, which the step routes' tensor cache keeps room for.
_COND = 1024 * 5120 * 4


def test_a_tight_video_entry_keeps_one_cond_in_the_tensor_cache(
        monkeypatch):
    """A video entry books the step routes' tensor cache at no less than
    one cond of its priced rows, and every other cache at the share its
    clip leaves. h3 at 1344x768 for 5 s needs its bare clip and one cond
    on a 32 GB Mac's budget, and is refused one byte under that, the
    need shown rounded up and the budget down."""
    from gmlx.serve import capacity
    from gmlx.serve import video_service as vs
    from gmlx.video import memfit

    monkeypatch.delenv("GMLX_OVERCOMMIT", raising=False)
    _full, bare = _bare(monkeypatch, "h3", "1344x768", 5)
    need = bare + _COND
    got = vs.entry_plan(_h3_entry(monkeypatch, "h3", "1344x768", 5, need))
    assert (got.cache_share, got.footprint, got.tensor_floor) == (
        0.0, need, _COND)
    with pytest.raises(memfit.VideoMemoryError) as e:
        vs.entry_plan(_h3_entry(monkeypatch, "h3", "1344x768", 5, need - 1))
    assert str(e.value) == (
        "video model 'h3' needs 19.76 GiB for one 1344x768 clip of 124 "
        "frames, over the working budget of 19.75 GiB. Set a smaller "
        "defaults.size or defaults.seconds. GMLX_OVERCOMMIT=1 overrides.")
    budget = capacity.ram_budget_bytes(32 << 30)
    got = vs.entry_plan(_h3_entry(monkeypatch, "h3", "1344x768", 5, budget))
    assert got.cache_share == 1.0 and got.footprint <= budget


@pytest.mark.parametrize("variant, int8, tail", [
    ("h3-base", "off", "Set a smaller defaults.size or defaults.seconds."),
    ("h3-base", "attention",
     "Set a smaller defaults.size or defaults.seconds."),
    ("h3-base", "linears", "Set a smaller defaults.size or "
     "defaults.seconds, or defaults.int8 off or attention."),
    ("h3-base", "all", "Set a smaller defaults.size or defaults.seconds, "
     "or defaults.int8 off or attention."),
    ("fasth3", "linears", "Set a smaller defaults.size or "
     "defaults.seconds, or defaults.int8 off.")])
def test_an_int8_linears_entry_over_its_budget_names_the_setting(
        monkeypatch, variant, int8, tail):
    """An entry refused for memory at int8 linears or all names the int8
    settings that keep the file's linears beside the clip's size and
    length, off alone on FastH3, whose sparse attention refuses int8
    attention. At off or attention it names the clip alone."""
    from gmlx.serve import video_service as vs
    from gmlx.video import memfit
    from video_fixtures import int8_kernel

    monkeypatch.delenv("GMLX_OVERCOMMIT", raising=False)
    int8_kernel(monkeypatch)
    spec = _h3_entry(monkeypatch, variant, "1344x768", 5, 16 * GB)
    spec.defaults["int8"] = int8
    with pytest.raises(memfit.VideoMemoryError) as e:
        vs.entry_plan(spec)
    assert str(e.value).endswith(f"{tail} GMLX_OVERCOMMIT=1 overrides.")


# A caption of the H3 upsampler's 6144 tokens and the one row of its
# empty template.
_CAPTION_ROWS = 6145


def test_an_entry_with_an_upsampler_books_a_caption_of_its_max_tokens(
        monkeypatch):
    """An entry that names an upsampler prices its encode, its tensor
    cache's cond and its refined rows at a caption of the upsampler's
    max_tokens. The cond, 120 MiB of f32 rows, stands past the 64 MiB
    tensor cache, booked in the footprint, so the encode's put takes it."""
    from gmlx.serve import image_service
    from gmlx.serve import video_service as vs

    spec = _h3_entry(monkeypatch, "h3-base", "512x512", 1)
    spec.upsampler = "chat"
    got = vs.entry_plan(spec)
    cond = _CAPTION_ROWS * 5120 * 4
    tensor, _cap = image_service.cache_bytes()
    assert (got.text_rows, got.tensor_floor) == (_CAPTION_ROWS, cond)
    assert cond > tensor
    clip = vs.plan_clip(got, frames=got.frames, width=got.size[0],
                        height=got.size[1], text=_CAPTION_ROWS)
    held = (_cache_term() + cond - tensor
            + vs.caption_bytes(5376, _CAPTION_ROWS))
    assert got.footprint == clip.phases["encode"] + held
    spec.upsampler = None
    bare = vs.entry_plan(spec)
    assert (bare.text_rows, bare.tensor_floor) == (vs.PRICED_TEXT_ROWS,
                                                   _COND)
    assert got.footprint > bare.footprint


def test_an_entry_with_an_mmproj_books_the_clip_it_books_without_one(
        monkeypatch):
    """The mmproj loads for an encode with keyframes alone, so an entry
    that names one keeps its price and holds the mmproj's bytes for the
    encode check."""
    from gmlx.serve import capacity
    from gmlx.serve import video_service as vs

    budget = capacity.ram_budget_bytes(32 << 30)
    bare = vs.entry_plan(_h3_entry(monkeypatch, "h3", "1344x768", 5, budget))
    got = vs.entry_plan(_h3_entry(monkeypatch, "h3", "1344x768", 5, budget,
                                  mmproj=True))
    assert (bare.mmproj_bytes, bare.tower_bytes) == (0, 0)
    assert (got.mmproj_bytes, got.tower_bytes) == (_H3_FILES["mmproj"],
                                                   _TOWER)
    assert (got.footprint, got.cache_share) == (bare.footprint,
                                                bare.cache_share)


def _booked(entry) -> int:
    """The cache bytes an entry books at its share: the caches, the
    tensor cache's cond floor and the refined text rows."""
    from gmlx.serve import image_service

    caches = _cache_term()
    tensor, _cap = image_service.cache_bytes()
    return (int(caches * entry.cache_share)
            + max(0, entry.tensor_floor - int(tensor * entry.cache_share))
            + _CAPTIONS)


@pytest.mark.parametrize("size, mode, empty, beside", [
    ("1344x768", "first frame", True, True),
    ("1344x768", "last frame", True, True),
    ("1344x768", "image guide", True, True),
    ("1344x768", "5-frame guide clip with sound", True, False),
    ("1344x768", "both keyframes", False, False),
    ("768x768", "both keyframes", True, True)])
def test_a_32_gb_h3_entry_runs_a_keyframe_step_beside_the_caches_it_books(
        monkeypatch, size, mode, empty, beside):
    """docs/comfyui.md: through the nodes, a step is priced beside the
    caches the server holds, at the share its model's entry books. On a
    32 GB Mac an entry of the default 1344x768 clip runs a step with a
    first frame, a last frame or an image guide beside the caches, a step
    with a 5-frame guide clip with sound only while they are empty, and no
    step with both keyframes; at 768x768 a step with both runs beside
    them."""
    from gmlx.serve import capacity
    from gmlx.serve import video_service as vs
    from gmlx.video.families.h3 import family as hf
    from gmlx.video.layout import AnchorRows

    budget = capacity.ram_budget_bytes(32 << 30)
    entry = vs.entry_plan(_h3_entry(monkeypatch, "h3", size, 5, budget))
    w, h = entry.size
    tok = hf.vision_tokens(w, h)
    first, last = AnchorRows(0, 1, 0), AnchorRows(entry.frames - 1, 1, 0)
    keyframes, anchors = {
        "first frame": (1, (first,)), "last frame": (1, (last,)),
        "image guide": (0, (AnchorRows(entry.frames // 2, 1, 0),)),
        "both keyframes": (2, (first, last)),
        "5-frame guide clip with sound": (
            0, (AnchorRows(0, 2, hf.audio_frames(5)),))}[mode]

    def clip(anchors=(), keyframes=0):
        return vs.plan_clip(entry, frames=entry.frames, width=w, height=h,
                            text=vs.PRICED_TEXT_ROWS + keyframes * (tok + 8),
                            anchors=anchors, budget=budget)

    d = clip(anchors, keyframes).terms["denoise"]
    step = int(d["dit"] + d["lora"] + d["step"])
    # The prompt's keyframe blocks and the anchors' latent rows.
    assert clip(anchors, keyframes).rows == clip().rows + keyframes * (
        tok + 8) + sum(a.video_t * tok + 2 * a.audio_t for a in anchors)
    d = clip().terms["denoise"]
    assert int(d["dit"] + d["lora"] + d["step"]) + _booked(entry) <= budget
    assert (step <= budget, step + _booked(entry) <= budget) == (empty,
                                                                 beside)


@pytest.mark.parametrize("mac, size, admitted", [
    (32, "1344x768", False), (32, "768x768", True), (36, "1344x768", True)])
def test_an_h3_entry_with_an_upsampler_on_a_32_or_36_gb_mac(
        monkeypatch, mac, size, admitted):
    """docs/config.md and docs/video.md: an entry with an upsampler books a
    caption of the upsampler's max_tokens, so on a 32 GB Mac an h3 entry
    of the default 1344x768 clip is refused at its load, and one of
    768x768 loads; on a 36 GB Mac the 1344x768 entry loads, and a step
    with both keyframes runs beside the caches it books."""
    from gmlx.serve import capacity, image_service
    from gmlx.serve import video_service as vs
    from gmlx.video import memfit
    from gmlx.video.families.h3 import family as hf
    from gmlx.video.layout import AnchorRows

    monkeypatch.delenv("GMLX_OVERCOMMIT", raising=False)
    budget = capacity.ram_budget_bytes(mac << 30)
    spec = _h3_entry(monkeypatch, "h3", size, 5, budget)
    spec.upsampler = "chat"
    if not admitted:
        with pytest.raises(memfit.VideoMemoryError, match="over the working "
                           "budget"):
            vs.entry_plan(spec)
        return
    entry = vs.entry_plan(spec)
    w, h = entry.size
    tok = hf.vision_tokens(w, h)
    anchors = (AnchorRows(0, 1, 0), AnchorRows(entry.frames - 1, 1, 0))
    d = vs.plan_clip(entry, frames=entry.frames, width=w, height=h,
                     text=vs.PRICED_TEXT_ROWS + 2 * (tok + 8),
                     anchors=anchors, budget=budget).terms["denoise"]
    tensor, _cap = image_service.cache_bytes()
    booked = (int(_cache_term() * entry.cache_share)
              + max(0, entry.tensor_floor - int(tensor * entry.cache_share))
              + vs.caption_bytes(5376, entry.text_rows))
    assert int(d["dit"] + d["lora"] + d["step"]) + booked <= budget


@pytest.mark.parametrize("steps_api, tensor_mb, floor", [
    (False, 64, 0), (True, 16, 16 << 20)])
def test_a_video_entry_keeps_cond_room_only_in_a_tensor_cache_it_has(
        monkeypatch, steps_api, tensor_mb, floor):
    """With the step routes off, no tensor cache takes a cond, and with a
    tensor cache under one cond the entry keeps it all: the bare clip and
    that room admit the entry, and a byte less refuses it."""
    import dataclasses

    from gmlx.config import ImageServerCfg
    from gmlx.serve import image_service
    from gmlx.serve import video_service as vs
    from gmlx.video import memfit

    monkeypatch.delenv("GMLX_OVERCOMMIT", raising=False)
    monkeypatch.setattr(image_service, "_CACHE_BYTES",
                        dict(image_service._CACHE_BYTES))
    image_service.configure_caches(dataclasses.replace(
        ImageServerCfg(), steps_api=steps_api, tensor_cache_mb=tensor_mb))
    _full, bare = _bare(monkeypatch, "h3", "1344x768", 5)
    got = vs.entry_plan(_h3_entry(monkeypatch, "h3", "1344x768", 5,
                                  bare + floor))
    assert (got.cache_share, got.footprint, got.tensor_floor) == (
        0.0, bare + floor, floor)
    with pytest.raises(memfit.VideoMemoryError):
        vs.entry_plan(_h3_entry(monkeypatch, "h3", "1344x768", 5,
                                bare + floor - 1))


def test_fasth3_at_1344x768_fits_a_36_gb_mac_and_floors_a_cond_when_tight(
        monkeypatch):
    """FastH3 at 1344x768 for 5 s fits a 36 GB Mac's budget with each cache
    at its whole bound. On a budget that leaves its caches 12/64 of their
    bounds, the tensor cache's share holds 12 MiB, so the entry books the
    8 MiB a cond needs beside it, there alone."""
    from gmlx.serve import capacity, image_service
    from gmlx.serve import video_service as vs

    full, bare = _bare(monkeypatch, "fasth3", "1344x768", 5)
    budget = capacity.ram_budget_bytes(36 << 30)
    got = vs.entry_plan(_h3_entry(monkeypatch, "fasth3", "1344x768", 5,
                                  budget))
    assert (got.cache_share, got.footprint) == (1.0, full.footprint)
    assert got.footprint <= budget
    caches = _cache_term()
    tensor, _cap = image_service.cache_bytes()
    share = 12 / 64
    tight = bare + int(caches * share) + _COND - int(tensor * share)
    got = vs.entry_plan(_h3_entry(monkeypatch, "fasth3", "1344x768", 5,
                                  tight))
    assert (got.cache_share, got.footprint) == (share, tight)


@pytest.mark.parametrize("under", ["cond", "clip"])
def test_overcommit_admits_a_video_entry_over_its_budget_at_full_caches(
        monkeypatch, under):
    """GMLX_OVERCOMMIT=1 admits a video entry its budget refuses, under
    its clip with one cond or under its clip alone, with each cache at
    its full bound."""
    from gmlx.serve import video_service as vs

    full, bare = _bare(monkeypatch, "h3", "768x768", 5)
    monkeypatch.setenv("GMLX_OVERCOMMIT", "1")
    budget = {"cond": bare + _COND, "clip": bare}[under] - 1
    got = vs.entry_plan(_h3_entry(monkeypatch, "h3", "768x768", 5, budget))
    assert (got.cache_share, got.footprint) == (1.0, full.footprint)


@pytest.mark.parametrize("variant, extra, fits", [
    ("h3-base", 3 * GB, True), ("h3", 10**6, False)])
def test_a_32_gb_server_entry_has_the_lora_room_the_video_page_gives(
        monkeypatch, variant, extra, fits):
    """docs/video.md: on a 32 GB Mac an entry whose default clip is
    768x768 for 5 s has room in its adapters for a bf16 LoRA file of up to
    3 GB beside h3-base, and none beside h3, whose own LoRA it stacks on."""
    from gmlx.serve import capacity
    from gmlx.serve import video_service as vs
    from gmlx.video import memfit

    monkeypatch.delenv("GMLX_OVERCOMMIT", raising=False)
    spec = _h3_entry(monkeypatch, variant, "768x768", 5,
                     capacity.ram_budget_bytes(32 << 30), extra)
    if fits:
        assert vs.entry_plan(spec).footprint
    else:
        with pytest.raises(memfit.VideoMemoryError):
            vs.entry_plan(spec)


@pytest.mark.parametrize("variant, free", [("h3-base", 1.4), ("h3", 1.2)])
def test_a_32_gb_server_entry_leaves_the_free_memory_the_video_page_gives(
        monkeypatch, variant, free):
    """docs/video.md: on a 32 GB Mac an entry whose default clip is
    768x768 for 5 s leaves about 1.4 GB free beside h3-base and 1.2 GB
    beside h3, the memory a request's LoRA is booked in when it arrives.
    The entry reserves none for request LoRAs at load."""
    from gmlx.serve import capacity
    from gmlx.serve import video_service as vs

    monkeypatch.delenv("GMLX_OVERCOMMIT", raising=False)
    budget = capacity.ram_budget_bytes(32 << 30)
    plan = vs.entry_plan(_h3_entry(monkeypatch, variant, "768x768", 5,
                                   budget))
    assert round((budget - plan.footprint) / 1e9, 1) == free


def test_an_image_route_names_a_video_id_as_no_image_model(request):
    from gmlx.config import build_config

    serving.register_resolved_models(build_config(_VIDEO))
    request.addfinalizer(serving.clear_resolved_models)
    with pytest.raises(serving.WrongModelKind) as e:
        serving.resolve_image_request("h3")
    assert e.value.message == ("'h3' is not an image model; the image models "
                               "are ['zit']")
    assert serving.resolve_pool_path("h3") == "/abs/h3.gguf"


def test_resolve_pool_path_takes_either_kind(mixed_cfg):
    assert serving.resolve_pool_path("zit") == "/abs/zit.safetensors"
    assert serving.resolve_pool_path("q") == "/abs/qwen.gguf"
    assert serving.image_ids_for_path("/abs/zit.safetensors") == ["zit"]
    with pytest.raises(serving.ModelNotFound):
        serving.resolve_pool_path("nope")


def test_the_chat_warm_never_loads_an_image_model(monkeypatch, mixed_cfg):
    """Every chat, Messages and Responses route warms its model through
    _load_resident; an image id there loads nothing and evicts nothing,
    and the route answers it 400. Keep and preload still load it."""
    import importlib

    from gmlx.serve.patches import routes as sp_routes

    loads = []
    monkeypatch.setattr(residency, "acquire_image_model",
                        lambda *a, **k: loads.append(("image",) + a))
    app = importlib.import_module("gmlx._vendor.mlx_vlm.server.app")
    monkeypatch.setattr(app, "get_cached_model",
                        lambda *a, **k: loads.append(("chat",) + a))
    assert sp_routes._load_resident("zit", app._INHERIT_ADAPTER) is None
    assert loads == []
    sp_routes._spawn_keep_warm("zit").join(timeout=5)
    assert loads == [("image", "zit")]


def _image_pool(monkeypatch, footprint):
    from gmlx.serve import capacity as cap

    monkeypatch.setattr(cap, "preload_gate", lambda *a, **k: None)
    proxy, pool, _seen, _td = make_pool()
    pool._image_fns = (footprint, lambda spec: "PIPE", lambda box: box.clear())
    return pool


def test_an_image_model_whose_file_is_gone_is_a_typed_404(monkeypatch, mixed_cfg):
    from fastapi import HTTPException

    def gone(spec):
        raise FileNotFoundError(2, "No such file", spec.path)

    pool = _image_pool(monkeypatch, gone)
    with pytest.raises(HTTPException) as e:
        residency.acquire_image_model("zit", pool=pool)
    assert e.value.status_code == 404
    assert e.value.detail["error"]["type"] == "model_file_missing"


def test_an_image_part_of_another_family_is_a_typed_400(monkeypatch, mixed_cfg):
    from fastapi import HTTPException

    from gmlx.image.vae import VAEError

    def wrong_vae(spec):
        raise VAEError("this VAE holds the qwen_image21 VAE")

    pool = _image_pool(monkeypatch, wrong_vae)
    with pytest.raises(HTTPException) as e:
        residency.acquire_image_model("zit", pool=pool)
    err = e.value.detail["error"]
    assert e.value.status_code == 400 and err["type"] == "image_model_mismatch"
    assert "qwen_image21" in err["message"] and err["model"] == "zit"


def test_a_deferred_image_load_is_the_typed_503(monkeypatch, mixed_cfg):
    from fastapi import HTTPException

    from gmlx.serve import capacity as cap

    pool = _image_pool(monkeypatch, lambda spec: 9 * GB)

    def deferred(*a, **k):
        raise cap.LoadDeferred("model load deferred: zit")

    monkeypatch.setattr(cap, "preload_gate", deferred)
    monkeypatch.setattr(
        residency, "_http_from_load_deferred",
        lambda mid, e: HTTPException(status_code=503, detail={"error": {
            "type": "model_load_deferred", "model": mid}}))
    with pytest.raises(HTTPException) as e:
        residency.acquire_image_model("zit", pool=pool)
    assert e.value.status_code == 503
    assert e.value.detail["error"]["model"] == "zit"
    assert not pool._entries
