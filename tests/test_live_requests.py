"""Live per-request view (server.requests[] on /v1/metrics)."""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest

pytest.importorskip("mlx_vlm")

import gmlx.live_requests as lr  # noqa: E402
from gmlx import server_bridge_vlm as serving  # noqa: E402

_PKG = importlib.import_module("mlx_vlm.server")


@pytest.fixture(autouse=True)
def _clean():
    lr._reset()
    saved = getattr(_PKG, "_kq_residency_pool", None)
    yield
    lr._reset()
    _PKG._kq_residency_pool = saved
    serving._PATH_TO_IDS.pop("/abs/q.gguf", None)


class _Pool:
    def __init__(self, rg, path="/abs/q.gguf"):
        self.rg, self.path = rg, path

    def model_path_for_generator(self, rg):
        return self.path if rg is self.rg else None


class _Row:
    def __init__(self, ids, m):
        self.prompt_ids, self.max_tokens = list(ids), m


def _queued(rid, ptoks, mt, at):
    return SimpleNamespace(request_id=rid, prompt_tokens=ptoks,
                           args=SimpleNamespace(max_tokens=mt), queued_at=at)


def _engine(*, unprocessed=(), prompt=None, decode=None, ledger=None):
    gen = SimpleNamespace(_unprocessed_sequences=list(unprocessed),
                          _prompt_batch=prompt, _generation_batch=decode)
    gen._kq_tick_guard = SimpleNamespace(ledger=dict(ledger or {}))
    return gen


def _rg(queue=(), draft=None):
    q = SimpleNamespace(queue=list(queue), qsize=lambda: len(queue))
    return SimpleNamespace(requests=q, draft_model=draft)


def test_rows_cover_queue_prefill_decode():
    rg = _rg(queue=[_queued("srv1", 500, 64, 90.0)])
    _PKG._kq_residency_pool = _Pool(rg)
    serving._PATH_TO_IDS["/abs/q.gguf"] = ["qwen", "q"]
    prompt = SimpleNamespace(
        uids=[7], _cached_tokens_per_row=[120],
        _apc_meta=[{"prefix_len": 120, "apc_blocks": [object()]}])
    decode = SimpleNamespace(uids=[5], _num_tokens=[40])
    gen = _engine(
        unprocessed=[(9, list(range(300)), 32, {}, None, None)],
        prompt=prompt, decode=decode,
        ledger={5: _Row(range(1000), 256), 7: _Row(range(2000), 128)})
    active = {
        5: {"request_id": "r5", "queued_at": 80.0,
            "decode_started_at": 96.0, "cached_tokens": 900},
        7: {"request_id": "r7", "queued_at": 95.0},
        9: {"request_id": "r9", "queued_at": 99.0},
    }
    rows = lr.build_rows(rg, gen, active, now=100.0)
    by_id = {r["id"]: r for r in rows}
    assert [r["state"] for r in rows] == ["queued", "queued", "prefill", "decode"]

    srv = by_id["srv1"]
    assert srv["position"] == 0 and srv["uid"] is None
    assert srv["prompt_tokens"] == 500 and srv["max_tokens"] == 64
    assert srv["elapsed_s"] == 10.0 and srv["model"] == "qwen"

    eng = by_id["r9"]
    assert eng["position"] == 1 and eng["uid"] == 9
    assert eng["prompt_tokens"] == 300 and eng["max_tokens"] == 32

    pre = by_id["r7"]
    assert pre["prompt_tokens"] == 2000 and pre["max_tokens"] == 128
    assert pre["cache"] == {"tier": "block", "warm_tokens": 120}
    assert pre["generated"] == 0 and pre["ttft_s"] is None

    dec = by_id["r5"]
    assert dec["generated"] == 40 and dec["prompt_tokens"] == 1000
    assert dec["elapsed_s"] == 20.0 and dec["ttft_s"] == 16.0
    assert dec["decode_tok_s"] == 10.0                # 40 tokens / 4 s
    assert dec["cache"]["warm_tokens"] == 900
    assert dec["speculative"] is None                  # no drafter


def test_tier_captured_at_prefill_survives_into_decode():
    rg = _rg()
    prompt = SimpleNamespace(uids=[1], _cached_tokens_per_row=[10],
                             _apc_meta=[{"prefix_len": 10, "apc_blocks": []}])
    gen = _engine(prompt=prompt, ledger={1: _Row(range(50), 8)})
    active = {1: {"request_id": "a", "queued_at": 0.0}}
    rows = lr.build_rows(rg, gen, active, now=1.0)
    assert rows[0]["cache"]["tier"] == "exact"

    gen = _engine(decode=SimpleNamespace(uids=[1], _num_tokens=[3]),
                  ledger={1: _Row(range(50), 8)})
    rows = lr.build_rows(rg, gen, active, now=2.0)
    assert rows[0]["state"] == "decode" and rows[0]["cache"]["tier"] == "exact"

    # uid gone from active -> memo dropped
    lr.build_rows(rg, _engine(), {}, now=3.0)
    assert (id(rg), 1) not in lr._TIER


def test_miss_tier_and_unknown_model():
    rg = _rg()
    _PKG._kq_residency_pool = None
    prompt = SimpleNamespace(uids=[1], _cached_tokens_per_row=[0],
                             _apc_meta=[{"prefix_len": 0, "apc_blocks": []}])
    gen = _engine(prompt=prompt)
    rows = lr.build_rows(rg, gen, {1: {"queued_at": 0.0}}, now=1.0)
    assert rows[0]["cache"]["tier"] == "miss"
    assert rows[0]["model"] is None and rows[0]["id"] == "uid1"
    assert rows[0]["prompt_tokens"] is None            # no ledger entry


def test_speculative_batch_and_acceptance(monkeypatch):
    common = importlib.import_module("mlx_vlm.speculative.common")
    monkeypatch.setattr(common, "speculative_stats_since",
                        lambda draft, snap: (4, 6.0, 12))
    rg = _rg(draft=object())
    # speculative batch shape: live uids subset of _all_uids
    decode = SimpleNamespace(uids=[2], _all_uids=[1, 2], _num_tokens=[5, 9])
    gen = _engine(decode=decode, ledger={2: _Row(range(10), 4)})
    active = {2: {"request_id": "s", "queued_at": 0.0, "spec_snapshot": (0, 0, 0)}}
    rows = lr.build_rows(rg, gen, active, now=1.0)
    assert len(rows) == 1 and rows[0]["generated"] == 9
    assert rows[0]["speculative"] == {"rounds": 4, "accepted": 6.0,
                                      "drafted": 12, "accept_rate": 0.5}


def test_finished_rows_not_in_active_are_skipped():
    rg = _rg()
    decode = SimpleNamespace(uids=[1, 2], _num_tokens=[5, 5])
    gen = _engine(decode=decode)
    rows = lr.build_rows(rg, gen, {2: {"queued_at": 0.0}}, now=1.0)
    assert [r["uid"] for r in rows] == [2]


def test_build_rows_tolerates_garbage():
    rows = lr.build_rows(object(), object(), None, now=1.0)
    assert rows == []


def test_publish_rate_limit_force_and_staleness(monkeypatch):
    rg = _rg()
    gen = _engine(decode=SimpleNamespace(uids=[1], _num_tokens=[1]))
    active = {1: {"queued_at": 0.0}}
    clock = [100.0]
    monkeypatch.setattr(lr.time, "perf_counter", lambda: clock[0])

    lr.publish(rg, gen, active)
    assert len(lr.live_requests_view()) == 1

    # within the interval: a changed engine is not re-read
    gen2 = _engine(decode=SimpleNamespace(uids=[1, 2], _num_tokens=[1, 1]))
    active2 = {1: {"queued_at": 0.0}, 2: {"queued_at": 0.0}}
    clock[0] += 0.1
    lr.publish(rg, gen2, active2)
    assert len(lr.live_requests_view()) == 1
    lr.publish(rg, gen2, active2, force=True)
    assert len(lr.live_requests_view()) == 2

    # empty active with nothing queued publishes [] (idle engine)
    clock[0] += 1.0
    lr.publish(rg, _engine(), {}, force=True)
    assert lr.live_requests_view() == []

    # stale snapshot reads empty
    lr.publish(rg, gen2, active2, force=True)
    assert len(lr.live_requests_view()) == 2
    clock[0] += lr._STALE_S + 1
    assert lr.live_requests_view() == []


def test_publish_keeps_queued_rows_with_empty_active():
    rg = _rg(queue=[_queued("q", 10, 5, 0.0)])
    lr.publish(rg, _engine(), {}, force=True)
    assert [r["id"] for r in lr.live_requests_view()] == ["q"]


def test_install_wraps_step_passthrough_and_idempotent(monkeypatch):
    gen_mod = importlib.import_module("mlx_vlm.server.generation")
    saved = gen_mod.ResponseGenerator._step
    calls = []

    def fake_step(self, batch_gen, active, gen_kwargs=None):
        calls.append(gen_kwargs)
        return "stepped"
    monkeypatch.setattr(gen_mod.ResponseGenerator, "_step", fake_step)
    try:
        lr.install_live_requests()
        first = gen_mod.ResponseGenerator._step
        lr.install_live_requests()
        assert gen_mod.ResponseGenerator._step is first      # idempotent
        rg = _rg()
        gen = _engine(decode=SimpleNamespace(uids=[1], _num_tokens=[2]))
        out = gen_mod.ResponseGenerator._step(rg, gen, {1: {"queued_at": 0.0}},
                                              {"k": 1})
        assert out == "stepped" and calls == [{"k": 1}]
        assert lr.live_requests_view()[0]["generated"] == 2
    finally:
        gen_mod.ResponseGenerator._step = saved


def test_snapshot_enrichment_carries_requests(monkeypatch):
    from gmlx.server_patches import routes as sp_routes
    _APP = importlib.import_module("mlx_vlm.server.app")
    saved = _APP._server_runtime_snapshot
    _APP._server_runtime_snapshot = lambda: {"loaded_model": "x"}
    try:
        monkeypatch.setattr(lr, "live_requests_view",
                            lambda: [{"id": "r1", "state": "decode"}])
        sp_routes.install_runtime_snapshot_enrichment()
        snap = _APP._server_runtime_snapshot()
        assert snap["requests"] == [{"id": "r1", "state": "decode"}]
        assert set(snap["concurrency"]) == {"decode_batch", "queue_cap",
                                            "in_flight", "waiting"}
        assert "eta_s" in snap["queue"] and "waiting" in snap["queue"]
    finally:
        _APP._server_runtime_snapshot = saved


def test_tier_memo_runs_ahead_of_rate_limit(monkeypatch):
    # A one-tick prefill: the rate limit skips the publish, but the tier
    # is still memoized for the decode rows that follow.
    rg = _rg()
    clock = [50.0]
    monkeypatch.setattr(lr.time, "perf_counter", lambda: clock[0])
    lr.publish(rg, _engine(), {}, force=True)          # sets the last-publish time
    prompt = SimpleNamespace(uids=[3], _cached_tokens_per_row=[64],
                             _apc_meta=[{"prefix_len": 64, "apc_blocks": [object()]}])
    clock[0] += 0.01
    lr.publish(rg, _engine(prompt=prompt), {3: {"queued_at": 0.0}})   # rate-limited
    assert lr._TIER[(id(rg), 3)] == "block"
    gen = _engine(decode=SimpleNamespace(uids=[3], _num_tokens=[2]))
    rows = lr.build_rows(rg, gen, {3: {"queued_at": 0.0, "cached_tokens": 64}}, now=51.0)
    assert rows[0]["cache"] == {"tier": "block", "warm_tokens": 64}


def test_tier_fallback_from_warm_tokens():
    rg = _rg()
    gen = _engine(decode=SimpleNamespace(uids=[1, 2], _num_tokens=[1, 1]))
    active = {1: {"queued_at": 0.0, "cached_tokens": 30},
              2: {"queued_at": 0.0, "cached_tokens": 0}}
    rows = {r["uid"]: r for r in lr.build_rows(rg, gen, active, now=1.0)}
    assert rows[1]["cache"]["tier"] == "hit" and rows[2]["cache"]["tier"] == "miss"


# multi-engine + speculative
class _Engine:            # weakref-able stand-in for a ResponseGenerator
    def __init__(self, draft=None):
        self.requests = SimpleNamespace(queue=[], qsize=lambda: 0)
        self.draft_model = draft


def test_per_engine_snapshots_merge_in_view(monkeypatch):
    clock = [10.0]
    monkeypatch.setattr(lr.time, "perf_counter", lambda: clock[0])
    a, b = _Engine(), _Engine()
    lr.publish(a, _engine(decode=SimpleNamespace(uids=[1], _num_tokens=[1])),
               {1: {"request_id": "a1", "queued_at": 0.0}}, force=True)
    lr.publish(b, _engine(decode=SimpleNamespace(uids=[1], _num_tokens=[2])),
               {1: {"request_id": "b1", "queued_at": 0.0}}, force=True)
    assert sorted(r["id"] for r in lr.live_requests_view()) == ["a1", "b1"]
    # b idles: its own snapshot empties, a's rows stay
    lr.publish(b, _engine(), {}, force=True)
    assert [r["id"] for r in lr.live_requests_view()] == ["a1"]
    # a's snapshot ages out on its own clock
    clock[0] += lr._STALE_S + 1
    assert lr.live_requests_view() == []


def test_dead_engine_snapshot_dropped():
    a = _Engine()
    lr.publish(a, _engine(decode=SimpleNamespace(uids=[1], _num_tokens=[1])),
               {1: {"queued_at": 0.0}}, force=True)
    assert len(lr.live_requests_view()) == 1
    del a
    assert lr.live_requests_view() == [] and not lr._SNAPS


def test_tier_memo_keyed_per_engine():
    a, b = _Engine(), _Engine()
    prompt = SimpleNamespace(uids=[7], _cached_tokens_per_row=[10],
                             _apc_meta=[{"prefix_len": 10, "apc_blocks": []}])
    lr._memo_tiers(a, _engine(prompt=prompt), {7: {}})
    assert lr._TIER == {(id(a), 7): "exact"}
    assert lr._tier_fallback(b, 7, 0) == "miss"          # b's uid 7 is another request
    lr.build_rows(b, _engine(), {}, now=1.0)              # b's cleanup leaves a's memo
    assert (id(a), 7) in lr._TIER


def test_speculative_hooks_track_a_batch(monkeypatch):
    common = importlib.import_module("mlx_vlm.speculative.common")
    monkeypatch.setattr(common, "speculative_stats_snapshot", lambda d: (1, 1.0, 1))
    monkeypatch.setattr(common, "speculative_stats_since", lambda d, s: (3, 5.0, 9))
    clock = [10.0]
    monkeypatch.setattr(lr.time, "perf_counter", lambda: clock[0])
    rg = _Engine(draft=object())
    _PKG._kq_residency_pool = _Pool(rg)
    serving._PATH_TO_IDS["/abs/q.gguf"] = ["q8"]
    q1, q2 = object(), object()
    r1 = SimpleNamespace(rqueue=q1, prompt_tokens=40, args=SimpleNamespace(max_tokens=64))
    r2 = SimpleNamespace(rqueue=q2, prompt_tokens=20, args=SimpleNamespace(max_tokens=32))
    s1 = {"request_id": "s1", "queued_at": 0.0, "decode_started_at": None, "generated_tokens": 0}
    s2 = {"request_id": "s2", "queued_at": 0.0, "decode_started_at": None, "generated_tokens": 0}
    lr.spec_prefill_started(rg, r1, s1)
    lr.spec_prefill_started(rg, r2, s2)
    rows = {r["id"]: r for r in lr.live_requests_view()}
    assert rows["s1"]["state"] == "prefill" and rows["s1"]["model"] == "q8"
    assert rows["s1"]["prompt_tokens"] == 40 and rows["s1"]["max_tokens"] == 64
    assert rows["s1"]["speculative"]["rounds"] == 3 and rows["s1"]["cache"]["tier"] is None

    s1["generated_tokens"], s1["decode_started_at"] = 5, 0.5
    clock[0] += lr._MIN_INTERVAL_S + 0.01           # a token past the rate limit
    lr.spec_decode_progress(id(q1), s1, None)
    rows = {r["id"]: r for r in lr.live_requests_view()}
    assert rows["s1"]["state"] == "decode" and rows["s1"]["generated"] == 5
    assert rows["s2"]["state"] == "prefill"

    lr.spec_decode_progress(id(q1), s1, "stop")
    lr.spec_decode_progress(id(q2), s2, "length")
    assert lr.live_requests_view() == []
    # untracked uid (an AR engine's token) is a no-op
    lr.spec_decode_progress(12345, {}, None)
    assert lr.live_requests_view() == []


def test_speculative_hook_install_passthrough_and_idempotent():
    gen_mod = importlib.import_module("mlx_vlm.server.generation")
    cls = gen_mod.ResponseGenerator
    saved_prefill = cls.__dict__["_log_prefill_started"]
    saved_progress = cls.__dict__["_log_decode_progress"]
    try:
        lr._install_speculative_hooks(gen_mod)
        wrapped = cls.__dict__["_log_decode_progress"]
        lr._install_speculative_hooks(gen_mod)
        assert cls.__dict__["_log_decode_progress"] is wrapped
        assert isinstance(wrapped, staticmethod)
        info = {"generated_tokens": 0}
        at = cls._log_decode_progress(1, info, token=0, text="", finish_reason=None)
        assert isinstance(at, float) and info["generated_tokens"] == 1
    finally:
        cls._log_prefill_started = saved_prefill
        cls._log_decode_progress = saved_progress


def test_spec_of_falls_back_to_drafter_round_lists(monkeypatch):
    common = importlib.import_module("mlx_vlm.speculative.common")
    monkeypatch.setattr(common, "speculative_stats_since",
                        lambda d, s: (None, None, None))     # stock totals not moved
    draft = SimpleNamespace(accept_lens=[3, 2.0, 4], draft_lens=[5, 5, 5])
    rg = _rg(draft=draft)
    info = {"spec_snapshot": (0, 0.0, 0)}
    assert lr._spec_of(rg, info) == {"rounds": 3, "accepted": 9.0, "drafted": 15,
                                     "accept_rate": 0.6}
    # nothing recorded yet -> None, and never an exception into the row
    assert lr._spec_of(_rg(draft=SimpleNamespace(accept_lens=[])), info) is None
    decode = SimpleNamespace(uids=[1], _all_uids=[1], _num_tokens=[4])
    rows = lr.build_rows(_rg(draft=SimpleNamespace()), _engine(decode=decode),
                         {1: {"queued_at": 0.0, "spec_snapshot": (0, 0, 0)}}, now=1.0)
    assert rows[0]["state"] == "decode" and rows[0]["speculative"] is None


def test_gmlx_restored_prefix_sets_tier_and_warm_tokens():
    rg = _rg()
    # a ckpt restore: stock meta says prefix 0, the batch records the restore
    prompt = SimpleNamespace(uids=[4], _cached_tokens_per_row=[0],
                             _apc_meta=[{"prefix_len": 0, "apc_blocks": []}],
                             _kq_apc_restored=(4236, "ckpt"))
    gen = _engine(prompt=prompt, ledger={4: _Row(range(4237), 96)})
    active = {4: {"request_id": "c", "queued_at": 0.0}}
    rows = lr.build_rows(rg, gen, active, now=1.0)
    assert rows[0]["cache"] == {"tier": "ckpt", "warm_tokens": 4236}
    # carried into decode, where stock's cached_tokens still reads 0
    gen = _engine(decode=SimpleNamespace(uids=[4], _num_tokens=[3]),
                  ledger={4: _Row(range(4237), 96)})
    active[4]["cached_tokens"] = 0
    rows = lr.build_rows(rg, gen, active, now=2.0)
    assert rows[0]["state"] == "decode"
    assert rows[0]["cache"] == {"tier": "ckpt", "warm_tokens": 4236}
    # the record is ignored on multi-row batches (single-row paths only)
    prompt = SimpleNamespace(uids=[5, 6], _cached_tokens_per_row=[0, 0],
                             _apc_meta=[{"prefix_len": 0}, {"prefix_len": 0}],
                             _kq_apc_restored=(100, "exact"))
    rows = lr.build_rows(rg, _engine(prompt=prompt), {5: {}, 6: {}}, now=3.0)
    assert {r["cache"]["tier"] for r in rows} == {"miss"}
