#!/usr/bin/env python3
"""Residency pool: weight-byte budget eviction (+ pinning, count cap, per-context
resolution). CPU-only - fake stock load/teardown and injected footprints, so no
GPU, no model files, no mlx_vlm.server mutation."""
from __future__ import annotations

from contextvars import copy_context
from types import SimpleNamespace

from gmlx.serve.residency import (  # noqa: E402
    _BusyHold,
    _GenerationGuard,
    _ResidencyPool,
    _RuntimeProxy,
    _active_entry,
    _active_hold,
    _build_scratch,
    _Scratch,
)

GB = 1024**3


class _FakeOriginal:
    def __init__(self):
        self.metrics = "METRICS"


def make_pool(budget_gb, footprints_gb, pinned=(), max_models=None):
    """A pool wired to fake stock loaders; ``footprints_gb`` maps model_path ->
    size in GB (the injected on-disk footprint)."""
    proxy = _RuntimeProxy(_FakeOriginal())
    teardowns = []

    def fake_stock_get(model_path, adapter_path, *, model_kind="auto"):
        proxy.apc_manager = f"APC:{model_path}"
        proxy.response_generator = f"RG:{model_path}"
        proxy.model_cache = {
            "cache_key": (model_path, adapter_path, model_kind),
            "model_path": model_path,
            "model": f"M:{model_path}",
        }

    def fake_stock_unload():
        teardowns.append(proxy.model_cache.get("model_path"))
        return True

    pool = _ResidencyPool(
        proxy,
        fake_stock_get,
        fake_stock_unload,
        int(budget_gb * GB),
        pinned,
        max_models=max_models,
        footprint_fn=lambda p: int(footprints_gb[p] * GB),
    )
    return proxy, pool, teardowns


def acquire(pool, path):
    """Acquire + immediately release the in-flight hold - models a request that
    has already finished, so LRU tests see idle (evictable) entries. The busy
    tests call ``pool.acquire`` directly to keep the hold."""
    entry = pool.acquire(path, None, "auto")
    pool.release(entry)
    _active_entry.set(entry)
    return entry


def resident(pool):
    return {e["model_path"] for e in pool.stats()["resident"]}


def busy_count(pool, path):
    return next(e["busy"] for e in pool.stats()["resident"]
                if e["model_path"] == path)


def test_byte_budget_evicts_lru_unpinned():
    # budget 25 GB holds two 10 GB models; a third forces an eviction.
    proxy, pool, teardowns = make_pool(25, {"A": 10, "B": 10, "C": 10}, pinned={"A"})
    acquire(pool, "A")
    acquire(pool, "B")
    assert resident(pool) == {"A", "B"}
    assert pool.stats()["resident_bytes"] == 20 * GB
    acquire(pool, "C")  # 20 + 10 = 30 > 25 -> evict LRU unpinned (B), keep pinned A
    assert teardowns == ["B"]
    assert resident(pool) == {"A", "C"}
    assert pool.stats()["resident_bytes"] == 20 * GB


def test_failed_build_after_eviction_drops_proxy_last_ref():
    # A build that fails after evicting for room (a preload-gate deferral)
    # must not leave the proxy's _last_entry pinning the evicted model: with
    # the model's internal reference cycles that kept the evicted weights
    # measurable as live, so the gate deferred every later load.
    import gc
    import weakref

    proxy = _RuntimeProxy(_FakeOriginal())
    teardowns = []

    def fake_stock_get(model_path, adapter_path, *, model_kind="auto"):
        if model_path == "B":
            raise RuntimeError("build failed")
        proxy.apc_manager = f"APC:{model_path}"
        proxy.response_generator = f"RG:{model_path}"
        proxy.model_cache = {
            "cache_key": (model_path, adapter_path, model_kind),
            "model_path": model_path,
            "model": f"M:{model_path}",
        }

    def fake_stock_unload():
        teardowns.append(proxy.model_cache.get("model_path"))
        return True

    pool = _ResidencyPool(
        proxy, fake_stock_get, fake_stock_unload, int(15 * GB), (),
        max_models=None, footprint_fn=lambda p: int(10 * GB))
    a = acquire(pool, "A")
    assert proxy._last_entry is a
    ref = weakref.ref(a)
    try:
        pool.acquire("B", None, "auto")  # evicts A, then the build raises
    except RuntimeError:
        pass
    else:
        raise AssertionError("B build should have raised")
    assert teardowns == ["A"]
    assert proxy._last_entry is not a
    _active_entry.set(None)
    del a
    gc.collect()
    assert ref() is None


def test_pin_protected_even_when_over_budget():
    # Two 10 GB models can't both fit a 15 GB budget, but A is pinned: it is
    # never evicted, so B is admitted over budget rather than refused.
    proxy, pool, teardowns = make_pool(15, {"A": 10, "B": 10}, pinned={"A"})
    acquire(pool, "A")
    acquire(pool, "B")
    assert teardowns == []
    assert resident(pool) == {"A", "B"}
    assert pool.stats()["resident_bytes"] == 20 * GB  # exceeds the 15 GB budget


def test_kept_model_is_still_lru_evicted_under_pressure():
    # The keep tier exempts a model from the idle-TTL reaper, NOT from LRU: under
    # byte pressure a kept-but-unpinned model is still a valid eviction victim
    # (the difference from a full pin).
    proxy, pool, teardowns = make_pool(25, {"A": 10, "B": 10, "C": 10})
    acquire(pool, "A")
    pool.set_keep("A", True)        # keep A (TTL-exempt) but do NOT pin it
    acquire(pool, "B")
    acquire(pool, "C")              # 30 > 25 -> LRU victim is A; kept != pinned
    assert teardowns == ["A"]
    assert resident(pool) == {"B", "C"}


def test_oversized_single_model_is_admitted():
    # A model larger than the whole budget still loads (never refuse a request).
    proxy, pool, teardowns = make_pool(5, {"A": 10})
    acquire(pool, "A")
    assert resident(pool) == {"A"}
    assert teardowns == []


def test_count_cap_is_secondary_bound():
    # Huge byte budget, but max_models=2 caps the count.
    proxy, pool, teardowns = make_pool(
        1000, {"A": 1, "B": 1, "C": 1}, pinned={"A"}, max_models=2
    )
    acquire(pool, "A")
    acquire(pool, "B")
    acquire(pool, "C")  # count would hit 3 > 2 -> evict LRU unpinned (B)
    assert teardowns == ["B"]
    assert resident(pool) == {"A", "C"}


def test_resident_hit_does_not_rebuild_or_evict():
    proxy, pool, teardowns = make_pool(25, {"A": 10, "B": 10}, pinned={"A"})
    ea = acquire(pool, "A")
    acquire(pool, "B")
    ea2 = pool.acquire("A", None, "auto")  # already resident
    assert ea2 is ea
    assert teardowns == []


def test_per_context_resolution_and_passthrough():
    proxy, pool, _ = make_pool(25, {"A": 10, "B": 10}, pinned={"A"})
    ea = acquire(pool, "A")
    eb = acquire(pool, "B")

    def in_ctx(entry, expected):
        _active_entry.set(entry)
        assert proxy.response_generator == expected

    copy_context().run(in_ctx, ea, "RG:A")
    copy_context().run(in_ctx, eb, "RG:B")
    # build scratch isolates an in-flight load from resident state
    def scratch_isolation():
        tok = _build_scratch.set(_Scratch())
        try:
            assert proxy.model_cache == {}
            assert proxy.response_generator is None
        finally:
            _build_scratch.reset(tok)

    copy_context().run(scratch_isolation)
    assert proxy.metrics == "METRICS"  # unknown attrs pass through


def test_stats_reports_budget_and_footprints():
    proxy, pool, _ = make_pool(25, {"A": 10}, pinned={"A"})
    acquire(pool, "A")
    s = pool.stats()
    assert s["budget_bytes"] == 25 * GB
    assert s["resident_bytes"] == 10 * GB
    assert s["resident"][0]["footprint_bytes"] == 10 * GB
    assert s["resident"][0]["pinned"] is True


def test_busy_entry_skipped_idle_entry_evicted():
    # A is LRU but mid-generation (in-flight hold kept); eviction must pass it
    # over and take the idle B instead - never tear down a live generation.
    proxy, pool, teardowns = make_pool(25, {"A": 10, "B": 10, "C": 10})
    pool.acquire("A", None, "auto")        # hold kept: in-flight
    acquire(pool, "B")                     # released: idle
    acquire(pool, "C")
    assert teardowns == ["B"]
    assert resident(pool) == {"A", "C"}


def test_all_busy_exceeds_budget_with_warning(caplog):
    # Both residents can't fit, but A is busy: admit B over budget (same policy
    # as all-pinned) and say so, rather than killing A's generation.
    proxy, pool, teardowns = make_pool(15, {"A": 10, "B": 10})
    pool.acquire("A", None, "auto")        # hold kept: in-flight
    with caplog.at_level("WARNING", logger="gmlx.serve.residency"):
        acquire(pool, "B")
    assert teardowns == []
    assert resident(pool) == {"A", "B"}
    assert pool.stats()["resident_bytes"] == 20 * GB
    assert "pinned or busy" in caplog.text and "exceed" in caplog.text


def test_streaming_incoming_defers_over_budget_beside_a_busy_resident():
    # Over the budget with only pinned or busy residents, a streaming load
    # is refused with the typed 503, not admitted. Its arena fills the
    # ceiling, so the overcommit takes the room the other model's KV needs
    # and the governor sheds it. A dense incoming keeps warn-and-admit.
    import pytest

    from gmlx.serve.capacity import LoadDeferred

    proxy, pool, teardowns = make_pool(100, {"A": 16, "B": 96, "C": 10})
    pool.acquire("A", None, "auto")        # hold kept: in-flight
    spec = SimpleNamespace(stream="experts")
    with pytest.raises(LoadDeferred) as info:
        pool.acquire("B", None, "auto", build_spec=spec)
    msg = str(info.value)
    assert "model load deferred: B" in msg
    assert "GB resident" in msg and "GB budget beside" in msg
    assert "GB of pinned or busy" in msg and "GMLX_DECODE_ARENA_GB" in msg
    assert teardowns == [] and resident(pool) == {"A"}
    # Room again once A's hold is released.
    pool.release(next(iter(pool._entries.values())))
    entry = pool.acquire("B", None, "auto", build_spec=spec)
    pool.release(entry)
    assert teardowns == ["A"] and resident(pool) == {"B"}


def test_streaming_incoming_under_budget_loads_beside_a_busy_resident():
    proxy, pool, teardowns = make_pool(200, {"A": 16, "B": 60})
    pool.acquire("A", None, "auto")        # hold kept: in-flight
    entry = pool.acquire("B", None, "auto",
                         build_spec=SimpleNamespace(stream="experts"))
    pool.release(entry)
    assert teardowns == [] and resident(pool) == {"A", "B"}


def test_streaming_incoming_defers_over_budget_beside_a_pinned_resident():
    import pytest

    from gmlx.serve.capacity import LoadDeferred

    proxy, pool, teardowns = make_pool(100, {"A": 16, "B": 96}, pinned=("A",))
    acquire(pool, "A")
    with pytest.raises(LoadDeferred):
        pool.acquire("B", None, "auto", build_spec=SimpleNamespace(stream="experts"))
    assert teardowns == [] and resident(pool) == {"A"}


def test_streaming_incoming_over_the_count_cap_keeps_the_admit_policy(caplog):
    # Only the byte budget defers a streaming load. Over the count cap with
    # room in bytes, the warn-and-admit policy stands.
    proxy, pool, teardowns = make_pool(200, {"A": 16, "B": 96}, pinned=("A",),
                                       max_models=1)
    acquire(pool, "A")
    with caplog.at_level("WARNING", logger="gmlx.serve.residency"):
        entry = pool.acquire("B", None, "auto",
                             build_spec=SimpleNamespace(stream="experts"))
    pool.release(entry)
    assert resident(pool) == {"A", "B"}
    assert "pinned or busy" in caplog.text


def test_streaming_build_lowers_the_wired_limit_and_credits_the_walk(monkeypatch):
    # The walk wraps the routed experts as tracked views long before the
    # install credits them at the first request. The build credit covers
    # what the load adds to active memory, up to the streamed bytes, from
    # the build until the install's credit, and a failed build drops it.
    # The wired-limit neutralization runs before the walk.
    import pytest

    import gmlx.gen.prefill_decay as pd
    import gmlx.stream.wired_limit as wired_limit
    import gmlx.serve.capacity as cap

    monkeypatch.setattr(pd, "_STREAMED_TRACKED", {})
    monkeypatch.setattr(pd, "_STREAMED_CAP", {})
    monkeypatch.setattr(pd, "_UNTRACKED_OWNERS", {})
    monkeypatch.setattr(pd, "_UNTRACKED_WEIGHTS", {})
    monkeypatch.setattr(pd, "_BUILD_CREDIT", {})
    box = {"active": 10 * GB}
    monkeypatch.setattr(pd.mx, "get_active_memory", lambda: box["active"])
    monkeypatch.setattr(cap, "streamed_expert_bytes", lambda p: 330 * GB)
    monkeypatch.setattr(cap, "install_boot_table",
                        lambda path, b, label, env=None: None)
    monkeypatch.setattr(cap, "preload_gate", lambda *a, **k: None)
    monkeypatch.setenv("GMLX_DECODE_ARENA_GB", "60")
    seen = []
    monkeypatch.setattr(wired_limit, "_neutralize_wired_limit_sweep",
                        lambda: seen.append("neutralized"))
    proxy = _RuntimeProxy(_FakeOriginal())

    def fake_stock_get(model_path, adapter_path, *, model_kind="auto"):
        box["active"] = 349 * GB
        pd.note_untracked_weights(0.0, key=("/w",))   # the warm pass registers
        seen.append(pd.build_credit_bytes())
        proxy.response_generator = SimpleNamespace(model=None)
        proxy.model_cache = {
            "cache_key": (model_path, adapter_path, model_kind),
            "model_path": model_path,
            "model": "M",
        }

    pool = _ResidencyPool(
        proxy, fake_stock_get, lambda: True, int(100 * GB), (),
        footprint_fn=lambda p: int(340 * GB))
    spec = SimpleNamespace(stream="experts")
    entry = pool.acquire("a", None, "auto", build_spec=spec)
    pool.release(entry)
    assert seen == ["neutralized", 330 * GB]
    assert pd.build_credit_bytes() == 330 * GB      # still there after the build
    pd.note_streamed_tracked_bytes(300 * GB, key=("/w",), source="experts")
    assert pd.build_credit_bytes() == 0            # the install took over
    # A dense build neither neutralizes nor credits.
    seen.clear()
    dense = pool.acquire("b", None, "auto")
    pool.release(dense)
    assert seen == [0]

    def failing_get(model_path, adapter_path, *, model_kind="auto"):
        raise RuntimeError("boom")

    pool2 = _ResidencyPool(
        proxy, failing_get, lambda: True, int(100 * GB), (),
        footprint_fn=lambda p: int(340 * GB))
    with pytest.raises(RuntimeError):
        pool2.acquire("c", None, "auto", build_spec=spec)
    assert not pd._BUILD_CREDIT


def test_release_returns_refcount_to_zero_and_entry_evictable():
    proxy, pool, teardowns = make_pool(15, {"A": 10, "B": 10})
    ea = pool.acquire("A", None, "auto")
    e2 = pool.acquire("A", None, "auto")   # concurrent request, same entry
    assert e2 is ea
    assert busy_count(pool, "A") == 2
    pool.release(ea)
    assert busy_count(pool, "A") == 1      # still one generation in flight
    pool.release(ea)
    assert busy_count(pool, "A") == 0
    acquire(pool, "B")                     # now A is evictable again
    assert teardowns == ["A"]
    assert resident(pool) == {"B"}


class _FakeRG:
    """Stand-in ResponseGenerator: generate yields two tokens; other attrs prove
    the guard delegates."""

    requests = "QUEUE"

    def generate(self, prompt, **kwargs):
        return "CTX", iter(["t1", "t2"])


def test_generation_guard_holds_busy_until_stream_ends():
    proxy, pool, _ = make_pool(25, {"A": 10})
    entry = pool.acquire("A", None, "auto")          # the request's hold
    guard = _GenerationGuard(_FakeRG(), _BusyHold(pool, entry))
    assert guard.requests == "QUEUE"                 # delegation
    ctx, it = guard.generate("hi")
    assert ctx == "CTX"
    assert busy_count(pool, "A") == 1                # hold rides the iterator
    assert list(it) == ["t1", "t2"]                  # exhaustion releases
    assert busy_count(pool, "A") == 0


def test_generation_guard_releases_on_close():
    proxy, pool, _ = make_pool(25, {"A": 10})
    entry = pool.acquire("A", None, "auto")
    _ctx, it = _GenerationGuard(_FakeRG(), _BusyHold(pool, entry)).generate("hi")
    assert next(it) == "t1"
    it.close()                                       # client disconnect mid-stream
    assert busy_count(pool, "A") == 0
    it.close()                                       # hold is one-shot
    assert busy_count(pool, "A") == 0


def test_proxy_wraps_entry_generator_with_request_hold():
    # The proxy hands back a guard carrying the context's hold for a real (has
    # .generate) generator; exhausting the stream drops the acquire's refcount.
    proxy, pool, _ = make_pool(25, {"A": 10})
    entry = pool.acquire("A", None, "auto")
    entry.response_generator = _FakeRG()             # fake build left a string

    def request():
        _active_entry.set(entry)
        _active_hold.set(_BusyHold(pool, entry))
        rg = proxy.response_generator
        assert isinstance(rg, _GenerationGuard)
        _ctx, it = rg.generate("hi")
        list(it)

    copy_context().run(request)
    assert busy_count(pool, "A") == 0


def test_teardown_drops_mtp_stash():
    import gmlx.serve.bridge_vlm as serving

    proxy, pool, teardowns = make_pool(15, {"/m/a.gguf": 10, "/m/b.gguf": 10})
    serving._MTP_DRAFTER_STASH["/m/a.gguf"] = (object(), "mtp")
    try:
        acquire(pool, "/m/a.gguf")
        acquire(pool, "/m/b.gguf")                   # evicts /m/a.gguf
        assert teardowns == ["/m/a.gguf"]
        assert "/m/a.gguf" not in serving._MTP_DRAFTER_STASH
    finally:
        serving._MTP_DRAFTER_STASH.clear()


def test_teardown_closes_apc_disk_writer():
    """On reap/evict the pool calls ``apc_manager.close()`` so the SSD-KV disk
    writer thread is stopped - the stock unload only ``clear()``s the in-memory
    tier, which would leak a daemon writer per reaped model. (close() never
    deletes the on-disk shards; they persist for a later warm reload.)"""
    closed = []

    class _APC:
        def __init__(self, path):
            self.path = path

        def close(self):
            closed.append(self.path)

    proxy = _RuntimeProxy(_FakeOriginal())

    def fake_stock_get(model_path, adapter_path, *, model_kind="auto"):
        proxy.apc_manager = _APC(model_path)
        proxy.response_generator = f"RG:{model_path}"
        proxy.model_cache = {
            "cache_key": (model_path, adapter_path, model_kind),
            "model_path": model_path,
            "model": f"M:{model_path}",
        }

    pool = _ResidencyPool(
        proxy, fake_stock_get, lambda: True, int(15 * GB), (),
        footprint_fn=lambda p: int(10 * GB),
    )
    acquire(pool, "/m/a.gguf")
    acquire(pool, "/m/b.gguf")           # budget 15 < 20 => evicts /m/a.gguf
    assert closed == ["/m/a.gguf"]       # its APC disk writer was closed on teardown


def test_teardown_closes_prefetcher_and_feeders():
    """Eviction must close the expert prefetcher, prefill feeder, AND decode
    feeder: each holds shard fds and (for the decode feeder) a mlocked wired
    arena that a reference cycle keeps alive past the stock unload."""
    closed = []

    class _Closeable:
        def __init__(self, name):
            self.name = name

        def close(self):
            closed.append(self.name)

    class _Model:
        pass

    proxy = _RuntimeProxy(_FakeOriginal())

    def fake_stock_get(model_path, adapter_path, *, model_kind="auto"):
        m = _Model()
        m._kq_prefetcher = _Closeable(f"prefetch:{model_path}")
        m._kq_feeder = _Closeable(f"prefill:{model_path}")
        m._kq_decode_feeder = _Closeable(f"decode:{model_path}")
        proxy.apc_manager = None
        proxy.response_generator = f"RG:{model_path}"
        proxy.model_cache = {
            "cache_key": (model_path, adapter_path, model_kind),
            "model_path": model_path,
            "model": m,
        }

    pool = _ResidencyPool(
        proxy, fake_stock_get, lambda: True, int(15 * GB), (),
        footprint_fn=lambda p: int(10 * GB),
    )
    acquire(pool, "/m/a.gguf")
    acquire(pool, "/m/b.gguf")           # budget 15 < 20 => evicts /m/a.gguf
    assert sorted(closed) == [
        "decode:/m/a.gguf", "prefetch:/m/a.gguf", "prefill:/m/a.gguf"]


def test_flush_all_closes_resident_apc():
    """Graceful-shutdown flush closes every *still-resident* model's APC disk
    writer (eviction only covers the ones it tears down). A None apc_manager is
    skipped, not crashed."""
    closed = []

    class _APC:
        def __init__(self, path):
            self.path = path

        def close(self):
            closed.append(self.path)

    proxy = _RuntimeProxy(_FakeOriginal())

    def fake_stock_get(model_path, adapter_path, *, model_kind="auto"):
        proxy.apc_manager = None if model_path.endswith("none.gguf") else _APC(model_path)
        proxy.response_generator = f"RG:{model_path}"
        proxy.model_cache = {
            "cache_key": (model_path, adapter_path, model_kind),
            "model_path": model_path,
            "model": f"M:{model_path}",
        }

    pool = _ResidencyPool(
        proxy, fake_stock_get, lambda: True, int(100 * GB), (),
        footprint_fn=lambda p: int(10 * GB),     # 3 models fit; nothing evicts
    )
    acquire(pool, "/m/a.gguf")
    acquire(pool, "/m/none.gguf")        # apc_manager is None -> skipped
    acquire(pool, "/m/b.gguf")
    n = pool.flush_all()
    assert n == 2
    assert sorted(closed) == ["/m/a.gguf", "/m/b.gguf"]


def test_warn_if_batch_unsafe_fires_once_for_gemma2(monkeypatch, caplog, capsys):
    """gemma2 warns once on load (batched decode is unreliable); a clean arch is
    silent; a header-read failure is silent. Pure header read - no model, no GPU."""
    import gmlx.serve.residency as residency

    class _Field:
        def __init__(self, s):
            self.parts = [s.encode("utf-8")]
            self.data = [0]

    class _Reader:
        arch = ""

        def __init__(self, path, mode="r"):
            self.fields = (
                {"general.architecture": _Field(_Reader.arch)} if _Reader.arch else {}
            )

    monkeypatch.setattr("gguf.GGUFReader", _Reader)
    residency._batch_unsafe_warned.clear()

    with caplog.at_level("WARNING", logger="gmlx.serve.residency"):
        # a clean (sdpa) arch never warns
        _Reader.arch = "qwen3"
        residency._warn_if_batch_unsafe("/x/qwen3.gguf")
        assert caplog.text == ""

        # gemma2 warns once, naming the arch and the failure mode
        _Reader.arch = "gemma2"
        residency._warn_if_batch_unsafe("/x/gemma2.gguf")
        assert "gemma2" in caplog.text
        assert "batched decode is unreliable" in caplog.text
        caplog.clear()

        # the same path again is silent (warn-once) - no log record and no
        # stray print either (a demoted or print-based repeat would still nag)
        capsys.readouterr()
        residency._warn_if_batch_unsafe("/x/gemma2.gguf")
        assert caplog.text == ""
        out = capsys.readouterr()
        assert out.err == "" and out.out == ""


def test_module_annotations_resolve():
    """Module-level forward-ref unions must quote the whole union
    (ContextVar["_Entry | None"], not ContextVar["_Entry" | None]) or any
    annotation introspection raises TypeError."""
    import typing

    import gmlx.serve.residency as residency

    typing.get_type_hints(residency)


def test_teardown_closes_streaming_helpers_under_the_serving_wrapper():
    # The served object is a wrapper (the text-only vlm adapter) around the
    # stock model the loader hung the streaming helpers on. Teardown must
    # find and close them through the wrapper: unclosed feeders keep their
    # worker threads, whose frames pin every expert weight past unload.
    closed = []

    class _Helper:
        def __init__(self, name):
            self.name = name

        def close(self):
            closed.append(self.name)

    class _Inner:
        pass

    class _LanguageModel:
        def __init__(self, model):
            self._model = model

    class _Wrapper:
        def __init__(self, model):
            self.language_model = _LanguageModel(model)

    proxy = _RuntimeProxy(_FakeOriginal())

    def fake_stock_get(model_path, adapter_path, *, model_kind="auto"):
        inner = _Inner()
        for attr in ("_kq_prefetcher", "_kq_feeder", "_kq_decode_feeder"):
            setattr(inner, attr, _Helper(f"{model_path}:{attr}"))
        proxy.apc_manager = None
        proxy.response_generator = None
        proxy.model_cache = {
            "cache_key": (model_path, adapter_path, model_kind),
            "model_path": model_path,
            "model": _Wrapper(inner),
        }

    pool = _ResidencyPool(
        proxy, fake_stock_get, lambda: True, int(15 * GB), (),
        max_models=None, footprint_fn=lambda p: int(10 * GB))
    acquire(pool, "A")
    acquire(pool, "B")  # evicts A
    assert sorted(closed) == [
        "A:_kq_decode_feeder", "A:_kq_feeder", "A:_kq_prefetcher"]


def test_teardown_collects_the_model_cycle_before_returning():
    # The evicted model's modules sit in reference cycles; teardown must
    # collect them itself, after dropping its own references, so the next
    # load's headroom gate measures the freed bytes instead of deferring
    # against a model that is only dead once a later gen-2 pass runs.
    import weakref

    class _Feeder:
        def __init__(self, model):
            self.model = model  # feeder<->module cycle, as the loader builds

        def close(self):
            pass

    class _Model:
        def __init__(self):
            self._kq_decode_feeder = _Feeder(self)

    proxy = _RuntimeProxy(_FakeOriginal())
    models = {}

    def fake_stock_get(model_path, adapter_path, *, model_kind="auto"):
        models[model_path] = weakref.ref(m := _Model())
        proxy.apc_manager = None
        proxy.response_generator = None
        proxy.model_cache = {
            "cache_key": (model_path, adapter_path, model_kind),
            "model_path": model_path,
            "model": m,
        }

    pool = _ResidencyPool(
        proxy, fake_stock_get, lambda: True, int(15 * GB), (),
        max_models=None, footprint_fn=lambda p: int(10 * GB))
    acquire(pool, "A")
    acquire(pool, "B")  # evicts A; no gc.collect() here on purpose
    assert models["A"]() is None
    assert models["B"]() is not None


def test_stats_in_flight_excludes_retained_holds():
    # The primary preload keeps a busy hold for the process lifetime; it
    # must not read as a generating stream.
    from types import SimpleNamespace
    proxy, pool, _ = make_pool(25, {"A": 10})
    entry = pool.acquire("A", None, "auto")  # hold kept: busy 1 (the lifetime hold)
    pool.mark_retained(SimpleNamespace(_entry=entry))
    row = pool.stats()["resident"][0]
    assert (row["busy"], row["retained"], row["in_flight"]) == (1, 1, 0)
    pool.acquire("A", None, "auto")          # a real request, hold kept
    row = pool.stats()["resident"][0]
    assert (row["busy"], row["in_flight"]) == (2, 1)
    pool.mark_retained(SimpleNamespace(_entry=None))   # tolerated


def test_build_prices_the_decode_arena_into_the_table(monkeypatch):
    from types import SimpleNamespace

    import gmlx.serve.capacity as cap

    calls = []
    monkeypatch.setattr(
        cap, "install_boot_table",
        lambda path, b, label, env=None: calls.append(int(b)))
    proxy = _RuntimeProxy(_FakeOriginal())
    feeder = SimpleNamespace(nominal_bytes=int(5 * GB))

    def fake_stock_get(model_path, adapter_path, *, model_kind="auto"):
        proxy.response_generator = SimpleNamespace(
            model=SimpleNamespace(_kq_decode_feeder=feeder))
        proxy.model_cache = {
            "cache_key": (model_path, adapter_path, model_kind),
            "model_path": model_path,
            "model": "M",
        }

    pool = _ResidencyPool(
        proxy, fake_stock_get, lambda: True, int(100 * GB), (),
        footprint_fn=lambda p: int(20 * GB))
    acquire(pool, "a")
    assert calls == [int(20 * GB), int(25 * GB)]


def test_build_stamps_boot_kv_costs_where_the_generator_reads(monkeypatch):
    # The batch generator is built on rg.model.language_model (mlx-vlm
    # batch_generate), and a streaming model's helpers hang deeper still.
    # boot_kv_rates must find the stamp from any of those objects.
    from types import SimpleNamespace

    import gmlx.serve.capacity as cap

    monkeypatch.setattr(cap, "install_boot_table",
                        lambda path, b, label, env=None: None)
    monkeypatch.setitem(cap._BOOT_KV_COSTS, "a", [(None, 1000.0), (4096, 500.0)])
    proxy = _RuntimeProxy(_FakeOriginal())
    stock = SimpleNamespace(_kq_decode_feeder=None)
    lm = SimpleNamespace(_model=stock)
    wrapper = SimpleNamespace(language_model=lm)

    def fake_stock_get(model_path, adapter_path, *, model_kind="auto"):
        proxy.response_generator = SimpleNamespace(model=wrapper)
        proxy.model_cache = {
            "cache_key": (model_path, adapter_path, model_kind),
            "model_path": model_path,
            "model": "M",
        }

    pool = _ResidencyPool(
        proxy, fake_stock_get, lambda: True, int(100 * GB), (),
        footprint_fn=lambda p: int(20 * GB))
    acquire(pool, "a")
    want = {"_boot:None": {"rate": 1000.0, "window": None},
            "_boot:4096": {"rate": 500.0, "window": 4096}}
    assert cap.boot_kv_rates(lm) == want            # what the generator holds
    assert cap.boot_kv_rates(wrapper) == want
    assert cap.boot_kv_rates(stock) == want
    # A dense chain (no streaming helper anywhere) resolves the same way.
    dense_lm = SimpleNamespace()
    dense = SimpleNamespace(language_model=dense_lm)
    import gmlx.serve.residency as residency
    residency._stamp_boot_kv_costs(SimpleNamespace(model=dense), "a")
    assert cap.boot_kv_rates(dense_lm) == want


def test_load_gate_keeps_the_resident_streams_ring_and_kv_room(monkeypatch):
    # With a streamed model resident, the second load's headroom check
    # takes off the ring room (when the ring is released) and the KV room
    # the stream keeps. A dense model loaded into those rooms before
    # this, under the weight budget.
    from types import SimpleNamespace

    import gmlx.serve.capacity as cap

    seen = []
    monkeypatch.setattr(
        cap, "preload_gate",
        lambda w, m, streaming=False, reserved_bytes=0.0:
            seen.append(int(reserved_bytes)))
    monkeypatch.setattr(cap, "install_boot_table",
                        lambda path, b, label, env=None: None)
    proxy = _RuntimeProxy(_FakeOriginal())
    ring = SimpleNamespace(slot_bytes=2 * GB, _slots=[])
    stock = SimpleNamespace(_kq_decode_feeder=SimpleNamespace(
                                nominal_bytes=50 * GB, _room_bytes=10 * GB),
                            _kq_feeder=ring)
    wrapper = SimpleNamespace(language_model=SimpleNamespace(_model=stock))
    dense = SimpleNamespace(language_model=SimpleNamespace(
        _model=SimpleNamespace()))

    def fake_stock_get(model_path, adapter_path, *, model_kind="auto"):
        proxy.response_generator = SimpleNamespace(
            model=wrapper if model_path == "stream" else dense)
        proxy.model_cache = {
            "cache_key": (model_path, adapter_path, model_kind),
            "model_path": model_path,
            "model": "M",
        }

    pool = _ResidencyPool(
        proxy, fake_stock_get, lambda: True, int(200 * GB), (),
        footprint_fn=lambda p: int(20 * GB))
    held = acquire(pool, "stream")
    assert seen == [0]                       # nothing resident yet
    acquire(pool, "dense")
    assert seen[-1] == 14 * GB               # 10 room + 2 x 2 ring
    ring._slots = [object()]                 # ring filled: its room is live
    acquire(pool, "dense2")
    assert seen[-1] == 10 * GB
    assert held.busy == 0


def test_streaming_entry_is_priced_at_every_token_plus_arena(monkeypatch):
    # A stream: experts entry holds its every-token weights, its decode
    # arena and its prefill ring's room, not the file. Admission prices the
    # arena from the planner (or GMLX_DECODE_ARENA_GB) and the ring from
    # the planner; the built entry carries the feeder's nominal arena and
    # the ring it allocated. Pricing the file evicted every other model
    # for 330 GB of experts that never leave the disk; pricing without the
    # ring loaded a dense model into the ring's room beside a busy stream.
    from types import SimpleNamespace

    import gmlx.serve.capacity as cap
    import gmlx.serve.residency as residency
    import gmlx.stream.plan as plan

    monkeypatch.delenv("GMLX_DECODE_ARENA_GB", raising=False)
    monkeypatch.setattr(cap, "streamed_expert_bytes", lambda p: 330 * GB)
    monkeypatch.setattr(plan, "plan_path",
                        lambda p: (SimpleNamespace(ring_bytes=6 * GB),
                                   SimpleNamespace(arena_bytes=80 * GB,
                                                   ring_fits=True)))
    monkeypatch.setattr(cap, "install_boot_table",
                        lambda path, b, label, env=None: None)
    monkeypatch.setattr(cap, "preload_gate", lambda *a, **k: None)
    proxy = _RuntimeProxy(_FakeOriginal())
    feeder = SimpleNamespace(nominal_bytes=70 * GB)
    stock = SimpleNamespace(_kq_decode_feeder=feeder,
                            _kq_feeder=SimpleNamespace(slot_bytes=2 * GB))
    wrapper = SimpleNamespace(language_model=SimpleNamespace(_model=stock))

    def fake_stock_get(model_path, adapter_path, *, model_kind="auto"):
        proxy.response_generator = SimpleNamespace(model=wrapper)
        proxy.model_cache = {
            "cache_key": (model_path, adapter_path, model_kind),
            "model_path": model_path,
            "model": "M",
        }

    pool = _ResidencyPool(
        proxy, fake_stock_get, lambda: True, int(100 * GB), (),
        footprint_fn=lambda p: int(340 * GB))
    asked = []
    monkeypatch.setattr(pool, "_evict_for_room",
                        lambda incoming, **k: asked.append(incoming))
    entry = pool.acquire("a", None, "auto", build_spec=SimpleNamespace(stream="experts"))
    pool.release(entry)
    assert asked == [10 * GB + 80 * GB + 6 * GB]
    assert entry.footprint == 10 * GB + 70 * GB + 4 * GB
    assert pool.stats()["resident_bytes"] == 84 * GB
    # The env cap is the arena the loader will size; the ring stays priced.
    monkeypatch.setenv("GMLX_DECODE_ARENA_GB", "60")
    assert residency._streaming_footprint("a", 340 * GB) == 10 * GB + 66 * GB
    assert residency._streaming_footprint("a", 340 * GB, {"GMLX_DECODE_ARENA_GB": "50"}) \
        == 10 * GB + 56 * GB
    # A ring the box cannot hold is not priced.
    monkeypatch.setattr(plan, "plan_path",
                        lambda p: (SimpleNamespace(ring_bytes=6 * GB),
                                   SimpleNamespace(arena_bytes=0,
                                                   ring_fits=False)))
    assert residency._streaming_footprint("a", 340 * GB) == 10 * GB + 60 * GB
    # A dense entry (no feeder) is still its file.
    stock._kq_decode_feeder = None
    dense = pool.acquire("b", None, "auto")
    pool.release(dense)
    assert dense.footprint == 340 * GB


def _table_only_header(monkeypatch):
    """A header the install's ladder sends table-only: 150 GB of text
    tensors over a 130 GB budget, a 30 GB table that alone brings it
    under, 100 GB of routed experts."""
    import gmlx.device as dev
    import gmlx.load.discovery as disc
    import gmlx.serve.capacity as cap
    import gmlx.stream.table_stream as ts

    monkeypatch.setattr(cap, "streamed_expert_bytes", lambda p: 100 * GB)
    monkeypatch.setattr(cap, "streamed_table_bytes", lambda p: 30 * GB)
    monkeypatch.setattr(disc, "text_tensor_bytes", lambda p: 150 * GB)
    monkeypatch.setattr(dev, "stream_budget_bytes", lambda: 130 * GB)
    monkeypatch.setattr(ts, "table_streaming_active",
                        lambda m: getattr(m, "tables", False))
    monkeypatch.delenv("GMLX_STREAM_PLE", raising=False)
    monkeypatch.delenv("GMLX_STREAM_PLE_COMPOSE", raising=False)


def _table_only_pool(monkeypatch, streamed_model):
    import gmlx.gen.prefill_decay as pd
    import gmlx.serve.capacity as cap

    for name in ("_STREAMED_TRACKED", "_STREAMED_CAP", "_UNTRACKED_OWNERS",
                 "_UNTRACKED_WEIGHTS", "_BUILD_CREDIT"):
        monkeypatch.setattr(pd, name, {})
    monkeypatch.setattr(pd.mx, "get_active_memory", lambda: 0)
    tables = []
    monkeypatch.setattr(cap, "install_boot_table",
                        lambda path, b, label, env=None: tables.append(b))
    monkeypatch.setattr(cap, "preload_gate", lambda *a, **k: None)
    proxy = _RuntimeProxy(_FakeOriginal())
    credits = []

    def fake_stock_get(model_path, adapter_path, *, model_kind="auto"):
        credits.append(dict(pd._BUILD_CREDIT))
        proxy.response_generator = SimpleNamespace(model=streamed_model)
        proxy.model_cache = {
            "cache_key": (model_path, adapter_path, model_kind),
            "model_path": model_path,
            "model": "M",
        }

    pool = _ResidencyPool(
        proxy, fake_stock_get, lambda: True, int(200 * GB), (),
        footprint_fn=lambda p: int(150 * GB))
    asked = []
    monkeypatch.setattr(pool, "_evict_for_room",
                        lambda incoming, **k: asked.append(incoming))
    return pool, tables, credits, asked


def test_a_table_only_stream_build_prices_the_resident_experts(monkeypatch):
    """Over the budget, a table that alone brings the model under it
    streams and the experts stay resident: admission, the entry, the boot
    table and the build credit discount the table alone."""
    _table_only_header(monkeypatch)
    model = SimpleNamespace(_kq_streaming=False, tables=True)
    pool, tables, credits, asked = _table_only_pool(monkeypatch, model)
    entry = pool.acquire("a", None, "auto",
                         build_spec=SimpleNamespace(stream="experts"))
    pool.release(entry)
    assert asked == [120 * GB]                  # no arena or ring either
    assert entry.footprint == 120 * GB
    assert tables == [120 * GB]
    assert [c[1] for c in credits[0].values()] == [30 * GB]


def test_a_stream_build_reprices_what_the_install_streamed(monkeypatch, caplog):
    _table_only_header(monkeypatch)
    model = SimpleNamespace(_kq_streaming=True, tables=True)   # it composed
    pool, tables, credits, asked = _table_only_pool(monkeypatch, model)
    with caplog.at_level("WARNING", logger="gmlx.serve.residency"):
        entry = pool.acquire("a", None, "auto",
                             build_spec=SimpleNamespace(stream="experts"))
    pool.release(entry)
    assert entry.footprint == 20 * GB
    assert tables == [120 * GB, 20 * GB]
    assert "re-pricing" in caplog.text


def test_streaming_footprint_credits_streamable_tables(monkeypatch):
    """deepseek41's engram tables serve from disk like the experts; pricing
    them resident would evict every other model for room they never take."""
    from types import SimpleNamespace

    import gmlx.serve.capacity as cap
    import gmlx.serve.residency as residency
    import gmlx.stream.plan as plan

    monkeypatch.setattr(cap, "streamed_expert_bytes", lambda p: 190 * GB)
    monkeypatch.setattr(cap, "streamed_table_bytes", lambda p: 64 * GB)
    monkeypatch.setattr(plan, "plan_path",
                        lambda p: (SimpleNamespace(ring_bytes=10 * GB),
                                   SimpleNamespace(arena_bytes=75 * GB,
                                                   ring_fits=True)))
    monkeypatch.setenv("GMLX_DECODE_ARENA_GB", "0")
    every = 260 * GB - 190 * GB - 64 * GB
    assert residency._streaming_footprint("a", 260 * GB) == every + 10 * GB


def test_failed_build_is_torn_down_before_the_lock_drops(monkeypatch):
    """A build that fails after the stock load (a kv policy refusal)
    tears its model down and collects it before the next acquire gates on
    live memory, and the exception keeps its type and message."""
    import pytest

    from gmlx.serve import residency as R

    proxy, pool, teardowns = make_pool(100, {"/m.gguf": 10})
    stock_get = pool._stock_get

    def failing_stock_get(model_path, adapter_path, *, model_kind="auto"):
        stock_get(model_path, adapter_path, model_kind=model_kind)
        raise RuntimeError("kv refused")

    pool._stock_get = failing_stock_get
    collected = []
    monkeypatch.setattr(R, "_collect_failed_build", lambda: collected.append(1))
    with pytest.raises(RuntimeError, match="kv refused") as info:
        acquire(pool, "/m.gguf")
    assert teardowns == ["/m.gguf"]
    assert collected == [1]
    assert not pool._entries
    # The frames of the failed build are detached from the exception.
    tb = info.value.__traceback__
    names = []
    while tb is not None:
        names.append(tb.tb_frame.f_code.co_name)
        tb = tb.tb_next
    assert "failing_stock_get" not in names


def test_failed_build_of_a_registry_cache_is_torn_down(monkeypatch):
    """Stock load turns the model cache into mlx-vlm's ModelCacheRegistry.
    A build that fails after it still tears the model down."""
    import pytest
    from gmlx._vendor.mlx_vlm.server.runtime import ModelCacheRegistry

    from gmlx.serve import residency as R

    proxy, pool, teardowns = make_pool(100, {"/m.gguf": 10})
    stock_get = pool._stock_get

    def failing_stock_get(model_path, adapter_path, *, model_kind="auto"):
        stock_get(model_path, adapter_path, model_kind=model_kind)
        registry = ModelCacheRegistry()
        registry.set("text_generation", proxy.model_cache)
        proxy.model_cache = registry
        raise RuntimeError("kv refused")

    pool._stock_get = failing_stock_get
    monkeypatch.setattr(R, "_collect_failed_build", lambda: None)
    with pytest.raises(RuntimeError, match="kv refused"):
        acquire(pool, "/m.gguf")
    assert teardowns == ["/m.gguf"]
    assert not pool._entries


# Image models share the pool with chat models: their own build and
# teardown, the same budget and LRU.
def make_mixed_pool(budget_gb, footprints_gb, image_gb, monkeypatch=None):
    """A pool whose chat builds take the fake stock loaders and whose image
    builds take injected (footprint, load, release) functions sized from
    ``image_gb``: image path -> GB."""
    from gmlx.serve import capacity as cap

    if monkeypatch is not None:
        monkeypatch.setattr(cap, "preload_gate", lambda *a, **k: None)
    proxy, pool, teardowns = make_pool(budget_gb, footprints_gb)
    loads, releases = [], []

    def footprint(spec):
        return int(image_gb[spec.path] * GB)

    def load(spec):
        loads.append(spec.path)
        return SimpleNamespace(pipe=f"PIPE:{spec.path}", spec=spec)

    def release(box):
        releases.append(box[0].spec.path)
        box.clear()

    pool._image_fns = (footprint, load, release)
    return proxy, pool, teardowns, loads, releases


def acquire_image(pool, path):
    from gmlx.serve.residency import IMAGE_KIND

    spec = SimpleNamespace(path=path, id=f"img:{path}")
    entry = pool.acquire(path, None, IMAGE_KIND, build_spec=spec,
                         cache_key_extra=(path,), loaded_as=spec.id)
    pool.release(entry)
    return entry


def test_image_build_takes_its_own_load_never_the_stock_get(monkeypatch):
    proxy, pool, teardowns, loads, _ = make_mixed_pool(
        100, {}, {"/dit.safetensors": 9}, monkeypatch)

    def no_stock_get(*a, **k):
        raise AssertionError("an image build reached the stock loader")

    pool._stock_get = no_stock_get
    entry = acquire_image(pool, "/dit.safetensors")
    assert loads == ["/dit.safetensors"]
    assert entry.is_image and entry.pipeline.pipe == "PIPE:/dit.safetensors"
    assert entry.footprint == 9 * GB
    assert entry.response_generator is None and entry.apc_manager is None
    assert proxy._last_entry is None          # never the chat fallback
    row = pool.stats()["resident"][0]
    assert row["kind"] == "image" and row["loaded_as"] == "img:/dit.safetensors"


def test_image_teardown_releases_on_its_own_path(monkeypatch):
    _, pool, teardowns, _, releases = make_mixed_pool(
        100, {}, {"/dit.safetensors": 9}, monkeypatch)
    entry = acquire_image(pool, "/dit.safetensors")
    assert pool.evict("/dit.safetensors")
    assert releases == ["/dit.safetensors"]
    assert teardowns == []                    # the stock unload never ran
    assert entry.pipeline is None and not pool._entries


def test_image_load_evicts_a_chat_model(monkeypatch):
    _, pool, teardowns, _, releases = make_mixed_pool(
        25, {"A.gguf": 10, "B.gguf": 10}, {"/dit.safetensors": 9},
        monkeypatch)
    acquire(pool, "A.gguf")
    acquire(pool, "B.gguf")
    acquire_image(pool, "/dit.safetensors")   # 20 + 9 > 25: evict LRU A
    assert teardowns == ["A.gguf"] and releases == []
    assert resident(pool) == {"B.gguf", "/dit.safetensors"}
    assert pool.stats()["resident_bytes"] == 19 * GB


def test_chat_load_evicts_an_image_model(monkeypatch):
    proxy, pool, teardowns, _, releases = make_mixed_pool(
        25, {"A.gguf": 10, "B.gguf": 10}, {"/dit.safetensors": 9},
        monkeypatch)
    acquire_image(pool, "/dit.safetensors")
    acquire(pool, "A.gguf")
    acquire(pool, "B.gguf")                   # 19 + 10 > 25: evict LRU image
    assert releases == ["/dit.safetensors"] and teardowns == []
    assert resident(pool) == {"A.gguf", "B.gguf"}
    kinds = {e["model_path"]: e["kind"] for e in pool.stats()["resident"]}
    assert kinds == {"A.gguf": "chat", "B.gguf": "chat"}


def test_busy_image_entry_is_not_evicted_for_a_chat_load(monkeypatch):
    from gmlx.serve.residency import IMAGE_KIND

    _, pool, _, _, releases = make_mixed_pool(
        15, {"A.gguf": 10}, {"/dit.safetensors": 9}, monkeypatch)
    spec = SimpleNamespace(path="/dit.safetensors", id="img")
    pool.acquire("/dit.safetensors", None, IMAGE_KIND, build_spec=spec)
    acquire(pool, "A.gguf")                   # over budget beside a live job
    assert releases == []
    assert resident(pool) == {"A.gguf", "/dit.safetensors"}


def test_an_image_footprint_failure_evicts_nothing(monkeypatch):
    import pytest

    _, pool, teardowns, loads, _ = make_mixed_pool(
        25, {"A.gguf": 10, "B.gguf": 10}, {}, monkeypatch)
    acquire(pool, "A.gguf")
    acquire(pool, "B.gguf")

    def not_an_image_model(spec):
        raise ValueError("no image model")

    pool._image_fns = (not_an_image_model,) + pool._image_fns[1:]
    with pytest.raises(ValueError, match="no image model"):
        acquire_image(pool, "/x.safetensors")
    assert teardowns == [] and loads == []


def test_the_stock_image_kinds_are_refused():
    import pytest

    _, pool, _ = make_pool(100, {"/m": 1})
    for kind in ("image_generation", "image_edit"):
        with pytest.raises(ValueError, match="gmlx_image"):
            pool.acquire("/m", None, kind)
    assert not pool._entries


def test_the_image_worker_is_one_named_thread():
    from gmlx.serve import image_service

    names = {image_service.WORKER.submit(
        lambda: __import__("threading").current_thread().name).result()
        for _ in range(3)}
    assert len(names) == 1 and names.pop().startswith("image-worker")


def test_no_gmlx_route_passes_a_stock_image_kind():
    """mlx-vlm's ``image_generation`` and ``image_edit`` kinds load its own
    pipelines from a request; only the copied mlx-vlm code names them."""
    import pathlib
    import re

    import gmlx

    root = pathlib.Path(gmlx.__file__).parent
    pat = re.compile(r"model_kind\s*=\s*[\"'](image_generation|image_edit)")
    hits = [str(p.relative_to(root)) for p in root.rglob("*.py")
            if "_vendor" not in p.parts and pat.search(p.read_text())]
    assert hits == []


def _release_on_another_thread(releases):
    """An image release that runs on another thread and collects there, as
    the image worker does, and waits for it."""
    import gc
    import threading

    def release(box):
        done = threading.Event()

        def run():
            try:
                releases.append(box[0].spec.path)
                box.clear()
                gc.collect()
            finally:
                done.set()

        threading.Thread(target=run, name="image-worker-stub",
                         daemon=True).start()
        done.wait()
    return release


def test_image_eviction_with_a_cyclic_busy_hold_does_not_deadlock(monkeypatch):
    """The image release waits on the worker, whose collection can finalize
    a hold that takes the pool lock: the pool must not hold it then."""
    import gc
    import threading

    _, pool, _, _, releases = make_mixed_pool(
        100, {"A.gguf": 10}, {"/dit.safetensors": 9}, monkeypatch)
    acquire_image(pool, "/dit.safetensors")
    entry = pool.acquire("A.gguf", None, "auto")
    pool._image_fns = pool._image_fns[:2] + (_release_on_another_thread(releases),)
    gc.disable()
    try:
        cycle = [_BusyHold(pool, entry)]
        cycle.append(cycle)
        del cycle
        t = threading.Thread(target=pool.evict, args=("/dit.safetensors",),
                             daemon=True)
        t.start()
        t.join(5.0)
        assert not t.is_alive(), "evict deadlocked against the image worker"
    finally:
        gc.enable()
    assert releases == ["/dit.safetensors"] and entry.busy == 0


def test_an_image_release_runs_without_the_pool_lock(monkeypatch):
    """Every path that tears an image entry down releases it with the pool
    lock free, so a chat acquire on another thread never waits on it."""
    import threading

    for how in ("evict", "clear", "reap", "room"):
        _, pool, _, _, releases = make_mixed_pool(
            10, {"A.gguf": 5}, {"/dit.safetensors": 9}, monkeypatch)
        acquire_image(pool, "/dit.safetensors")
        locked = []

        def release(box, _pool=pool):
            got = []

            def probe():
                ok = _pool._lock.acquire(timeout=1.0)
                got.append(ok)
                if ok:
                    _pool._lock.release()

            t = threading.Thread(target=probe)
            t.start()
            t.join()
            locked.append(not got[0])
            box.clear()

        pool._image_fns = pool._image_fns[:2] + (release,)
        if how == "evict":
            pool.evict("/dit.safetensors")
        elif how == "clear":
            pool.clear()
        elif how == "reap":
            pool._in_flight_fn = lambda: 0
            entry = next(iter(pool._entries.values()))
            entry.ttl = 1
            entry.last_access -= 10
            pool.reap_idle()
        else:
            acquire(pool, "A.gguf")
        assert locked == [False], how


def test_reaper_skips_an_image_entry_with_a_job_running(monkeypatch):
    """The server's in-flight count does not see an image job; its hold
    does."""
    from gmlx.serve.residency import IMAGE_KIND

    clock = [1000.0]
    _, pool, _, _, releases = make_mixed_pool(
        100, {}, {"/dit.safetensors": 9}, monkeypatch)
    pool._time_fn = lambda: clock[0]
    pool._in_flight_fn = lambda: 0
    spec = SimpleNamespace(path="/dit.safetensors", id="img")
    entry = pool.acquire("/dit.safetensors", None, IMAGE_KIND,
                         build_spec=spec, ttl=30)
    assert entry.busy == 1
    clock[0] += 60
    assert pool.reap_idle() == [] and releases == []
    pool.release(entry)
    assert pool.reap_idle() == ["/dit.safetensors"]


def test_an_image_gguf_takes_no_chat_preflight(monkeypatch, tmp_path):
    """The chat header preflight refuses a DiT GGUF; an image entry is
    checked by its own footprint function."""
    from gmlx.load import preflight as pf

    dit = str(tmp_path / "dit.gguf")
    open(dit, "wb").write(b"GGUF")

    def chat_preflight(path):
        raise AssertionError("the chat preflight ran for an image entry")

    monkeypatch.setattr(pf, "preflight", chat_preflight)
    _, pool, _, loads, _ = make_mixed_pool(100, {}, {dit: 9}, monkeypatch)
    acquire_image(pool, dit)
    assert loads == [dit]


def test_an_image_build_takes_the_load_gate_as_an_image(monkeypatch):
    from gmlx.serve import capacity as cap

    _, pool, _, loads, _ = make_mixed_pool(
        100, {}, {"/dit.safetensors": 9}, monkeypatch)
    gated = []
    monkeypatch.setattr(cap, "preload_gate",
                        lambda b, m, **k: gated.append((b, m, k.get("image"))))
    acquire_image(pool, "/dit.safetensors")
    assert gated == [(9 * GB, "/dit.safetensors", True)]
    assert loads == ["/dit.safetensors"]


def test_the_load_gate_names_an_image_entry_by_what_it_books(monkeypatch):
    import pytest

    from gmlx.serve import capacity as cap

    monkeypatch.setattr(cap, "overcommit", lambda: False)
    monkeypatch.setattr(cap, "working_budget_bytes", lambda: 10 * GB)
    monkeypatch.setattr(cap, "working_set_bytes", lambda: 12 * GB)
    import gmlx.gen.prefill_decay as pd

    monkeypatch.setattr(pd, "headroom_bytes", lambda: 50 * GB)
    with pytest.raises(RuntimeError) as e:
        cap.preload_gate(20 * GB, "zit", image=True)
    msg = str(e.value)
    assert "one image at its default size" in msg
    assert "stream-experts" not in msg and " weights 21" not in msg


def test_a_refused_chat_load_installs_no_boot_table(monkeypatch):
    """The load gate runs before the capacity table of the incoming model
    replaces the resident one's."""
    import pytest

    from gmlx.serve import capacity as cap

    _, pool, _ = make_pool(100, {"A.gguf": 10})
    order = []

    def gate(*a, **k):
        order.append("gate")
        raise RuntimeError("model does not fit")

    monkeypatch.setattr(cap, "preload_gate", gate)
    monkeypatch.setattr(cap, "install_boot_table",
                        lambda *a, **k: order.append("table"))
    with pytest.raises(RuntimeError, match="does not fit"):
        pool.acquire("A.gguf", None, "auto")
    assert order == ["gate"]


def test_a_chat_cold_load_completes_while_an_image_job_runs(monkeypatch):
    """An image build waits for the image worker before it takes the build
    lock, so a chat load never queues behind a running image job."""
    import threading

    from gmlx.serve import image_service

    _, pool, _, loads, _ = make_mixed_pool(
        100, {"A.gguf": 10}, {"/dit.safetensors": 9}, monkeypatch)
    footprint, _load, release = pool._image_fns

    def load(spec):
        # As load_on_worker: the pipeline is built on the image worker.
        return image_service._on_worker(_load, spec)

    pool._image_fns = (footprint, load, release)
    gate, running = threading.Event(), threading.Event()

    def job():
        running.set()
        gate.wait(10)

    busy = image_service.WORKER.submit(job)
    assert running.wait(5)
    try:
        img = threading.Thread(target=acquire_image,
                               args=(pool, "/dit.safetensors"), daemon=True)
        img.start()
        chat = threading.Thread(target=acquire, args=(pool, "A.gguf"),
                                daemon=True)
        chat.start()
        chat.join(3.0)
        assert not chat.is_alive(), "the chat load waited on the image job"
        assert loads == []
    finally:
        gate.set()
    busy.result(5)
    img.join(5.0)
    assert not img.is_alive() and loads == ["/dit.safetensors"]
    assert resident(pool) == {"A.gguf", "/dit.safetensors"}
