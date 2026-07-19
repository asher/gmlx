#!/usr/bin/env python3
"""Residency pool: weight-byte budget eviction (+ pinning, count cap, per-context
resolution). CPU-only - fake stock load/teardown and injected footprints, so no
GPU, no model files, no mlx_vlm.server mutation."""
from __future__ import annotations

from contextvars import copy_context

from gmlx.residency import (  # noqa: E402
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
    # (the load-bearing difference from a full pin).
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
    with caplog.at_level("WARNING", logger="gmlx.residency"):
        acquire(pool, "B")
    assert teardowns == []
    assert resident(pool) == {"A", "B"}
    assert pool.stats()["resident_bytes"] == 20 * GB
    assert "pinned or busy" in caplog.text and "exceed" in caplog.text


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
    from gmlx import server_bridge_vlm as serving

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
    from gmlx import residency

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

    with caplog.at_level("WARNING", logger="gmlx.residency"):
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

    from gmlx import residency

    typing.get_type_hints(residency)
