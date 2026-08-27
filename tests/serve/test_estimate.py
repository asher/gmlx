"""Queryable admission: dry-run estimates, the capacity plan, rates, the
context window on /v1/models and in launch pi, and the explicit-unload
release of the preload hold."""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest

pytest.importorskip("mlx_vlm")

import gmlx.serve.capacity as cap  # noqa: E402
import gmlx.serve.estimate as est  # noqa: E402
import gmlx.serve.queue_cap as qc  # noqa: E402
import gmlx.commands.launch as launch  # noqa: E402
import gmlx.serve.bridge_vlm as serving  # noqa: E402
from gmlx.config import build_config  # noqa: E402
from gmlx.serve.patches import _common as sp_common  # noqa: E402
from gmlx.serve.patches import capacity_routes as cr  # noqa: E402
from gmlx.serve.patches import routes as sp_routes  # noqa: E402

_APP = importlib.import_module("mlx_vlm.server.app")
_PKG = importlib.import_module("mlx_vlm.server")
_RUNTIME = importlib.import_module("mlx_vlm.server.runtime").runtime


@pytest.fixture(autouse=True)
def _restore(monkeypatch):
    app = _APP.app
    saved_routes = sp_common._snapshot_routes(app)
    saved_pool = getattr(_PKG, "_kq_residency_pool", None)
    saved_table = cap._TABLE
    saved_holds = list(sp_routes._PRELOAD_HOLDS)
    monkeypatch.delenv("GMLX_OVERCOMMIT", raising=False)
    serving.clear_resolved_models()
    yield
    sp_common._restore_routes(app, saved_routes)
    _PKG._kq_residency_pool = saved_pool
    cap._TABLE = saved_table
    sp_routes._PRELOAD_HOLDS[:] = saved_holds
    serving.clear_resolved_models()


def _env(monkeypatch, *, band="green", width=4, in_flight=0, waiting=0, table=True):
    import gmlx.serve.governor as gov
    monkeypatch.setattr(gov, "governor_stats", lambda: {"band": band})
    monkeypatch.setattr(qc, "concurrency_stats", lambda: {
        "decode_batch": width, "queue_cap": 2 * width,
        "in_flight": in_flight, "waiting": waiting})
    cap._TABLE = ({"path": "/abs/q.gguf",
                   "max_ctx": {1: 100000, 2: 60000, 4: 30000, 8: 10000}}
                  if table else None)


# --- capacity plan
def test_plan_geometry_and_admission(monkeypatch):
    _env(monkeypatch)
    p = est.capacity_plan(2, 50000)
    assert p["ok"] is True and p["max_context_at_width"] == 60000
    assert p["max_width_at_depth"] == 2 and p["admit_now"] is True
    assert p["slots"] == 4 and p["reason"] == "ok"
    # width 3 reads the width-4 row (conservative between rows)
    assert est.capacity_plan(3, 1000)["max_context_at_width"] == 30000
    # past the widest row: no geometry
    p = est.capacity_plan(16, 100)
    assert p["ok"] is False and p["max_context_at_width"] is None


def test_plan_reasons_in_order(monkeypatch):
    _env(monkeypatch)
    p = est.capacity_plan(4, 50000)
    assert p["ok"] is False and p["admit_now"] is False
    assert p["reason"].startswith("depth 50000 exceeds")
    _env(monkeypatch, band="orange")
    assert est.capacity_plan(1, 100)["reason"] == "governor orange"
    _env(monkeypatch, waiting=2)
    assert est.capacity_plan(1, 100)["reason"] == "2 waiting for a slot"
    _env(monkeypatch, in_flight=3)
    p = est.capacity_plan(2, 100)
    assert p["admit_now"] is False and p["slots"] == 1
    _env(monkeypatch, band="yellow")
    p = est.capacity_plan(2, 100)
    assert p["slots"] == 1 and p["admit_now"] is False          # yellow: one at a time
    assert est.capacity_plan(1, 100)["admit_now"] is True


def test_plan_without_table_admits_on_timing_only(monkeypatch):
    _env(monkeypatch, table=False)
    p = est.capacity_plan(2, 100000)
    assert p["ok"] is None and p["admit_now"] is True
    assert "no capacity table" in p["reason"]


def test_plan_route(monkeypatch):
    from fastapi.testclient import TestClient

    _env(monkeypatch)
    cr.install_capacity_plan()
    cr.install_capacity_plan()                    # idempotent
    client = TestClient(_APP.app)
    r = client.get("/v1/capacity/plan?width=2&depth=1000")
    assert r.status_code == 200 and r.json()["ok"] is True
    assert client.get("/capacity/plan?width=x").status_code == 400
    assert client.get("/v1/capacity/plan?width=0").status_code == 400


# --- rates
def test_rates_view(monkeypatch):
    import gmlx.serve.live_requests as lr
    monkeypatch.setattr(lr, "live_requests_view", lambda: [
        {"state": "decode", "decode_tok_s": 30.5},
        {"state": "decode", "decode_tok_s": 40.0},
        {"state": "queued", "decode_tok_s": None}])
    metrics = SimpleNamespace(
        _recent=[{"prefill_tok_s": 400.0, "decode_tok_s": 30.0},
                 {"prefill_tok_s": 0, "decode_tok_s": 50.0}],
        _generated_tokens_total=1000, _decode_time_total_s=20.0)
    monkeypatch.setattr(_RUNTIME, "metrics", metrics, raising=False)
    r = est.rates_view()
    assert r == {"decode_tok_s": 70.5, "decode_streams": 2,
                 "prefill_tok_s_recent": 400.0, "decode_tok_s_recent": 40.0,
                 "decode_tok_s_lifetime": 50.0}


def test_rates_survive_missing_metrics(monkeypatch):
    monkeypatch.setattr(_RUNTIME, "metrics", None, raising=False)
    r = est.rates_view()
    assert r["decode_tok_s_lifetime"] is None and r["decode_streams"] == 0


# --- dry run
def test_normalize_messages_text_and_media():
    msgs, media = est._normalize_messages([
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": [{"type": "text", "text": "hi"},
                                     {"type": "image_url", "image_url": {"url": "x"}}]},
        {"role": "assistant", "content": None,
         "tool_calls": [{"id": "1", "function": {"name": "f", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "1", "content": "42", "name": "f"},
    ])
    assert media == 1
    assert msgs[0] == {"role": "system", "content": "be brief"}
    assert msgs[1]["content"] == "hi"
    assert msgs[2]["tool_calls"][0]["id"] == "1"
    assert msgs[3]["tool_call_id"] == "1" and msgs[3]["name"] == "f"


def test_warm_tokens_probes_and_releases():
    released = []

    class _Mgr:
        def lookup_prefix(self, ids, extra_hash=0):
            return ["b1", "b2"], 512

        def release(self, blocks):
            released.extend(blocks)

        def find_exact_prefix(self, ids, extra_hash=0):
            return (0xabc, 700)                    # (cache_hash, prefix_len)

    assert est._warm_tokens(_Mgr(), list(range(1000)), 0) == (700, "exact")
    assert released == ["b1", "b2"]
    assert est._warm_tokens(None, [1, 2, 3], 0) == (0, None)


def test_estimate_validation_and_non_resident():
    serving.register_resolved_models(build_config(
        {"models": {"q": {"path": "/abs/q.gguf"}}}))
    st, body = est.estimate_request({"model": "q"})
    assert st == 400
    st, body = est.estimate_request({"model": "nope", "messages": [{"role": "user", "content": "x"}]})
    assert st == 404 and body["error"]["model"] == "nope"
    _PKG._kq_residency_pool = SimpleNamespace(resident_entry=lambda p: None)
    st, body = est.estimate_request({"model": "q", "messages": [{"role": "user", "content": "x"}]})
    assert st == 200 and body["resident"] is False and body["fits_now"] is None
    assert body["model"] == "q" and "hint" in body


def test_estimate_resident_prices_like_the_preflight(monkeypatch):
    import mlx.core as mx

    import gmlx.serve.mem_preflight as mp
    import gmlx.gen.prefill_decay as pd

    serving.register_resolved_models(build_config(
        {"models": {"q": {"path": "/abs/q.gguf"}}}))
    cfg = {"num_hidden_layers": 2, "num_attention_heads": 4,
           "num_key_value_heads": 2, "head_dim": 8, "hidden_size": 32}
    model = SimpleNamespace(config=cfg)
    rg = SimpleNamespace(
        _preprocess_request=lambda prompt: {"input_ids": mx.array([[1, 2, 3, 4, 5, 6]])},
        kv_bits=None, apc_mode=None)
    released = []

    class _Mgr:
        def lookup_prefix(self, ids, extra_hash=0):
            return ["b"], 4

        def release(self, blocks):
            released.extend(blocks)

        def find_exact_prefix(self, ids, extra_hash=0):
            return None

    entry = SimpleNamespace(response_generator=rg, apc_manager=_Mgr(),
                            model_cache={"model": model, "processor": None, "config": cfg})
    _PKG._kq_residency_pool = SimpleNamespace(resident_entry=lambda p: entry)
    monkeypatch.setattr(_PKG, "apply_chat_template",
                        lambda *a, **k: "<rendered prompt>", raising=False)
    app_mod = importlib.import_module("mlx_vlm.server.app")
    monkeypatch.setattr(app_mod, "_build_gen_args", lambda req, processor=None, tenant_id=None:
                        SimpleNamespace(to_template_kwargs=lambda: {}, max_tokens=256))
    # deterministic memory: 1 KB per token, 10 KB drained, 4 KB now
    monkeypatch.setattr(mp, "_need_bytes", lambda m, c, t, g=0: 1024.0 * (t + g))
    monkeypatch.setattr(mp, "available_drained_bytes", lambda: 10 * 1024.0)
    monkeypatch.setattr(pd, "headroom_bytes", lambda: 4 * 1024.0)
    gen_mod = importlib.import_module("mlx_vlm.server.generation")
    monkeypatch.setattr(gen_mod, "get_configured_context_limit", lambda: 8)
    monkeypatch.setattr(_RUNTIME, "metrics", SimpleNamespace(
        _recent=[{"prefill_tok_s": 100.0}], _requests_completed=0), raising=False)
    import gmlx.serve.governor as gov
    monkeypatch.setattr(gov, "governor_stats", lambda: {"band": "green"})
    monkeypatch.setattr(qc, "concurrency_stats", lambda: {
        "decode_batch": 4, "queue_cap": 8, "in_flight": 1, "waiting": 0})

    body = {"model": "q", "messages": [{"role": "user", "content": "hello"}]}
    st, out = est.estimate_request(body)
    assert st == 200 and out["resident"] is True and out["media"] is False
    assert out["prompt_tokens"] == 6 and out["prompt_chars"] == len("<rendered prompt>")
    assert out["max_tokens"] is None                         # not pinned by the client
    assert out["warm_tokens"] == 4 and out["cache_tier"] == "block" and released == ["b"]
    assert out["need_bytes"] == 6 * 1024 and out["need_prompt_bytes"] == 6 * 1024
    assert out["fits_drained"] is True and out["fits_now"] is False
    assert out["context_ok"] is True and out["context_limit"] == 8
    assert out["context_limit_source"] == "configured"
    assert out["est_ttft_s"] == 0.02                          # 2 cold tokens at 100 tok/s
    assert out["in_flight"] == 1 and out["band"] == "green"

    # a pinned max_tokens is priced in, like the preflight does
    st, out = est.estimate_request(dict(body, max_tokens=256))
    assert out["max_tokens"] == 256 and out["need_bytes"] == (6 + 256) * 1024
    assert out["fits_drained"] is False and out["context_ok"] is False

    # nothing configured: judged against the GGUF's trained context instead
    monkeypatch.setattr(gen_mod, "get_configured_context_limit", lambda: None)
    import gmlx.serve.capacity as _cap
    monkeypatch.setattr(_cap, "trained_context_length", lambda p: 200)
    st, out = est.estimate_request(body)
    assert out["context_limit"] == 200 and out["context_limit_source"] == "trained"
    assert out["context_ok"] is True
    st, out = est.estimate_request(dict(body, max_tokens=256))
    assert out["context_ok"] is False                        # 6 + 256 > 200
    monkeypatch.setattr(_cap, "trained_context_length", lambda p: None)
    st, out = est.estimate_request(body)
    assert out["context_limit"] is None and out["context_ok"] is None

    # media: rendered but not priced
    st, out = est.estimate_request({"model": "q", "messages": [
        {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "u"}}]}]})
    assert st == 200 and out["media"] is True and out["prompt_tokens"] is None


def test_estimate_route_and_chat_dry_run(monkeypatch):
    from fastapi.testclient import TestClient

    seen = []

    def _fake(body, tenant_id=None):
        seen.append(body)
        return 200, {"ok": True, "model": body.get("model")}
    monkeypatch.setattr(est, "estimate_request", _fake)
    cr.install_estimate()
    cr.install_estimate()                                     # idempotent
    client = TestClient(_APP.app)
    r = client.post("/v1/estimate", json={"model": "q", "messages": [{"role": "user", "content": "x"}]})
    assert r.status_code == 200 and r.json() == {"ok": True, "model": "q"}
    assert client.post("/estimate", content=b"nope",
                       headers={"content-type": "application/json"}).status_code == 400
    r = client.post("/v1/chat/completions", json={
        "model": "q", "dry_run": True, "max_tokens": 5,
        "messages": [{"role": "user", "content": "x"}]})
    assert r.status_code == 200 and r.json()["model"] == "q"
    assert "dry_run" not in seen[-1] and seen[-1]["max_tokens"] == 5


# --- context window on /v1/models and in launch pi
def test_trained_context_length_cached_by_mtime(monkeypatch, tmp_path):
    import gmlx.load.headerscan as hs
    f = tmp_path / "m.gguf"
    f.write_bytes(b"GGUF")
    calls = []

    def _scan(path, include_tensors=True):
        calls.append(path)
        return SimpleNamespace(kv={"general.architecture": "qwen3",
                                   "qwen3.context_length": 40960})
    monkeypatch.setattr(hs, "scan_gguf", _scan)
    cap._CTX_CACHE.clear()
    assert cap.trained_context_length(str(f)) == 40960
    assert cap.trained_context_length(str(f)) == 40960 and len(calls) == 1
    assert cap.trained_context_length(str(tmp_path / "missing.gguf")) is None
    monkeypatch.setattr(hs, "scan_gguf", lambda *a, **k: (_ for _ in ()).throw(ValueError("bad")))
    cap._CTX_CACHE.clear()
    assert cap.trained_context_length(str(f)) is None


def test_max_context_at_width_1_only_for_table_model(monkeypatch):
    monkeypatch.delenv("GMLX_OVERCOMMIT", raising=False)
    cap._TABLE = {"path": "/abs/q.gguf", "max_ctx": {1: 65536, 2: 30000}}
    assert cap.max_context_at_width_1("/abs/q.gguf") == 65536
    assert cap.max_context_at_width_1("/abs/other.gguf") is None
    cap._TABLE = None
    assert cap.max_context_at_width_1("/abs/q.gguf") is None


def test_models_payload_carries_context_fields(monkeypatch):
    monkeypatch.setattr(cap, "trained_context_length", lambda p: 32768 if p.endswith("q.gguf") else None)
    cap._TABLE = {"path": "/abs/q.gguf", "max_ctx": {1: 20000}}
    serving.register_resolved_models(build_config(
        {"models": {"q": {"path": "/abs/q.gguf"}, "g": {"path": "/abs/g.gguf"}}}))
    by_id = {m["id"]: m for m in sp_routes._models_payload()["data"]}
    assert by_id["q"]["context_length"] == 32768
    assert by_id["q"]["max_context_at_width_1"] == 20000
    assert by_id["g"]["context_length"] is None and by_id["g"]["max_context_at_width_1"] is None


def test_pi_model_entry_window_and_max_tokens():
    assert launch.pi_model_entry({"id": "a"}) == {"id": "a"}
    e = launch.pi_model_entry({"id": "a", "context_length": 131072, "max_context_at_width_1": 65536})
    assert e == {"id": "a", "contextWindow": 65536, "maxTokens": 8192}
    e = launch.pi_model_entry({"id": "a", "context_length": 8192})
    assert e == {"id": "a", "contextWindow": 8192, "maxTokens": 2048}
    e = launch.pi_model_entry({"id": "a", "context_length": 2048, "max_context_at_width_1": None})
    assert e["maxTokens"] == launch._PI_MAX_TOKENS_FLOOR


def test_build_pi_configs_writes_context_window():
    models = [{"id": "q", "context_length": 40960, "max_context_at_width_1": 30000},
              {"id": "t", "context_length": None}]
    models_doc, _ = launch.build_pi_configs("http://x/v1", models)
    entries = models_doc["providers"]["gmlx"]["models"]
    assert entries[0] == {"id": "q", "contextWindow": 30000, "maxTokens": 7500}
    assert entries[1] == {"id": "t"}


# --- explicit unload releases the preload hold
class _Hold:
    def __init__(self, path):
        self._entry = SimpleNamespace(model_path=path)
        self.released = 0

    def release(self):
        self.released += 1
        self._entry = None


class _HoldPool:
    def __init__(self):
        self.unmarked = []
        self.evicted = []

    def unmark_retained(self, hold):
        self.unmarked.append(hold)

    def evict(self, path):
        self.evicted.append(path)
        return True

    def stats(self):
        return {"resident": []}

    def clear(self):
        return True

    def busy_paths(self):
        return []


def test_release_preload_holds_scoped_and_all():
    a, b = _Hold("/abs/a.gguf"), _Hold("/abs/b.gguf")
    sp_routes._PRELOAD_HOLDS[:] = [a, b]
    pool = _HoldPool()
    assert sp_routes._release_preload_holds(pool, "/abs/a.gguf") == 1
    assert a.released == 1 and b.released == 0 and pool.unmarked == [a]
    assert sp_routes._PRELOAD_HOLDS == [b]
    assert sp_routes._release_preload_holds(pool, None) == 1
    assert b.released == 1 and sp_routes._PRELOAD_HOLDS == []


def test_unload_route_drops_preload_hold_then_evicts():
    from fastapi.testclient import TestClient

    serving.register_resolved_models(build_config(
        {"models": {"q": {"path": "/abs/q.gguf"}}}))
    hold = _Hold("/abs/q.gguf")
    sp_routes._PRELOAD_HOLDS[:] = [hold]
    pool = _HoldPool()
    _PKG._kq_residency_pool = pool
    sp_routes.install_pool_aware_unload()
    client = TestClient(_APP.app)
    r = client.post("/unload", json={"model": "q"})
    assert r.status_code == 200 and r.json()["status"] == "success"
    assert hold.released == 1 and pool.evicted == ["/abs/q.gguf"]
    assert sp_routes._PRELOAD_HOLDS == []


def test_pool_unmark_retained_and_resident_entry():
    import gmlx.serve.residency as residency

    pool = residency._ResidencyPool.__new__(residency._ResidencyPool)
    import threading
    pool._lock = threading.Lock()
    e1 = residency._Entry(("k1",), "/abs/q.gguf", {}, object(), None, last_access=1.0, busy=1, retained=1)
    e2 = residency._Entry(("k2",), "/abs/q.gguf", {}, object(), None, last_access=2.0)
    pool._entries = {("k1",): e1, ("k2",): e2}
    hold = SimpleNamespace(_entry=e1)
    pool.unmark_retained(hold)
    assert e1.retained == 0
    pool.unmark_retained(hold)
    assert e1.retained == 0                          # never negative
    assert pool.resident_entry("/abs/q.gguf") is e2  # most recently used
    assert pool.resident_entry("/abs/none.gguf") is None


def test_stamp_apc_mode_sets_generator_mode_once(monkeypatch):
    """The post-load APC wiring stamps rg.apc_mode (what mlx-vlm's own init
    does when the manager exists at construction) so the server precomputes
    the semantic salt instead of the engine hashing the embedding matrix."""
    from mlx_vlm import apc as _apc

    from gmlx.serve.residency import _stamp_apc_mode

    class _LM:
        pass

    probed = []
    monkeypatch.setattr(_apc, "model_apc_mode", lambda lm: probed.append(lm) or "block")
    lm = _LM()
    rg = SimpleNamespace(model=SimpleNamespace(language_model=lm), apc_mode=None)
    _stamp_apc_mode(rg)
    assert rg.apc_mode == "block" and probed == [lm]    # probes the bare language model
    rg.apc_mode = "exact"
    _stamp_apc_mode(rg)
    assert rg.apc_mode == "exact"                       # never overrides a set mode
    rg2 = SimpleNamespace(model=SimpleNamespace(_kq_apc_mode="exact", language_model=_LM()),
                          apc_mode=None)
    _stamp_apc_mode(rg2)
    assert rg2.apc_mode == "exact"                      # the spec engine's stamp wins
    rg3 = SimpleNamespace(apc_mode=None)                # no model: a no-op
    _stamp_apc_mode(rg3)
    assert rg3.apc_mode is None


def test_ckpt_peek_reports_deepest_pinned_record_without_assembly():
    """The checkpoint tier (hybrid models) keeps its warm starts in pinned
    records, not the block chain: the dry-run's probe reads the deepest
    matching record's depth and neither assembles a cache nor moves a
    counter."""
    import threading
    from collections import OrderedDict

    from gmlx.cache.snapshot import _CkptRecord, ckpt_peek

    class _Man:
        block_size = 16
        lock = threading.RLock()
        _kq_ckpt_gen = 0

    man = _Man()
    ids = list(range(100, 200))
    man._kq_ckpt_records = OrderedDict()
    for p, kind in ((32, "anchor"), (64, "boundary")):
        rec = _CkptRecord(ids=tuple(ids[:p]), extra_hash=0, p=p, kind=kind, layout=("kv", "arr"))
        man._kq_ckpt_records[(rec.ids, 0)] = rec
    assert ckpt_peek(man, ids, extra_hash=0, layout=("kv", "arr")) == 64
    assert ckpt_peek(man, ids, extra_hash=7, layout=("kv", "arr")) == 0      # other salt
    assert ckpt_peek(man, ids, extra_hash=0, layout=("kv",)) == 0            # other layout
    assert ckpt_peek(man, ids[:40], extra_hash=0, layout=("kv", "arr")) == 32  # 64 > n
    assert ckpt_peek(man, [5] + ids, extra_hash=0) == 0                      # diverged
    assert ckpt_peek(None, ids) == 0

    # and the estimate's probe prefers it over a shallower block-chain match
    man.lookup_prefix = lambda ids, extra_hash=0: ([], 16)
    man.release = lambda blocks: None
    man.find_exact_prefix = lambda ids, extra_hash=0: None
    model = SimpleNamespace(_kq_apc_ckpt_layout=("kv", "arr"))
    assert est._warm_tokens(man, ids, 0, model) == (64, "ckpt")
    assert est._warm_tokens(man, ids, 0, None) == (16, "block")


def test_exact_peek_reads_the_in_memory_exact_tier():
    """find_exact_prefix scans disk shards only; the in-memory exact tier
    (retirement stores on exact-mode models) is peeked with the stock
    lookup's rule: strict prefix of the query, same salt, deepest wins."""
    import threading
    from collections import OrderedDict

    class _Man:
        lock = threading.RLock()

        def lookup_prefix(self, ids, extra_hash=0):
            return [], 0

        def release(self, blocks):
            pass

        def find_exact_prefix(self, ids, extra_hash=0):
            return None

    man = _Man()
    ids = list(range(100, 200))
    man._exact_cache = OrderedDict()
    man._exact_cache[1] = SimpleNamespace(token_ids=tuple(ids[:40]), extra_hash=0)
    man._exact_cache[2] = SimpleNamespace(token_ids=tuple(ids[:70]), extra_hash=0)
    man._exact_cache[3] = SimpleNamespace(token_ids=tuple(ids[:90]), extra_hash=5)   # other salt
    man._exact_cache[4] = SimpleNamespace(token_ids=tuple(ids), extra_hash=0)        # whole query: no
    assert est._exact_peek(man, ids, 0) == 70
    assert est._exact_peek(man, ids, 5) == 90
    assert est._exact_peek(man, ids[:30], 0) == 0
    assert est._warm_tokens(man, ids, 0) == (70, "exact")
    # a stored row longer than the query (verbatim resend after a retirement
    # store) never matches - the tier serves the next turn
    man._exact_cache = OrderedDict([(9, SimpleNamespace(token_ids=tuple(ids + [1, 2]), extra_hash=0))])
    assert est._exact_peek(man, ids, 0) == 0
    assert est._exact_peek(man, ids + [1, 2, 3], 0) == len(ids) + 2


def test_plan_reason_follows_reported_band_when_concurrency_fails(monkeypatch):
    """If the concurrency read throws after the band was recorded, the
    reason is still judged on the band the payload reports."""
    _env(monkeypatch, band="red")

    def _boom():
        raise RuntimeError("no engine")

    monkeypatch.setattr(qc, "concurrency_stats", _boom)
    p = est.capacity_plan(1, 100)
    assert p["band"] == "red" and p["reason"] == "governor red"
    assert p["admit_now"] is False
