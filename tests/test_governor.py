"""Governor band ladder: sampling, rungs, shed paths, client bridge."""

import queue
import time
import types

import pytest

import mlx.core as mx

import gmlx.governor as gov
import gmlx.tick_guard as tg
from gmlx.server_patches.row_failed import RowShedError


class FakeGB:
    def __init__(self, uids, num_tokens):
        self.uids = list(uids)
        self._all_uids = list(uids)
        self._num_tokens = list(num_tokens)

    def __len__(self):
        return len(self.uids)


class FakeGen:
    def __init__(self, rows=2, rate=1e6, live=8e9):
        uids = list(range(rows))
        self._generation_batch = FakeGB(uids, [4] * rows)
        self._unprocessed_sequences = []
        self._prompt_batch = None
        self._kq_admit_kv_rates = {"layers.0.k": {"rate": rate,
                                                  "window": None}}
        self._kq_admit_live_bytes = live
        self._kq_admit_n_caches = 2
        self.prefill_step_size = 2048
        self.removed = []

    def remove(self, uid):
        self.removed.append(uid)
        if uid in self._generation_batch.uids:
            self._generation_batch.uids.remove(uid)
            return True
        return False


@pytest.fixture
def rig(monkeypatch):
    """Deterministic counters: headroom/ws/active/peak all scripted."""
    monkeypatch.delenv("GMLX_GOVERNOR", raising=False)
    box = {"head": 100e9, "ws": 120e9, "active": 20e9, "peak": 20e9,
           "mem_limits": [], "cache_limits": []}
    monkeypatch.setattr(gov, "_headroom_and_ws",
                        lambda margin: (box["head"] - margin * box["ws"],
                                        box["ws"]))
    monkeypatch.setattr(gov.mx, "get_active_memory",
                        lambda: box["active"])
    monkeypatch.setattr(gov.mx, "get_peak_memory", lambda: box["peak"])
    monkeypatch.setattr(gov.mx, "reset_peak_memory", lambda: None)
    monkeypatch.setattr(gov.mx, "clear_cache", lambda: None)
    monkeypatch.setattr(
        gov.mx, "set_memory_limit",
        lambda v: (box["mem_limits"].append(v), 0)[1])
    monkeypatch.setattr(
        gov.mx, "set_cache_limit",
        lambda v: (box["cache_limits"].append(v), 7)[1])
    monkeypatch.setattr(gov, "_maybe_register_apc", lambda gen: None)
    import gmlx.server_memory as sm
    monkeypatch.setattr(sm, "admit_reserve_bytes",
                        lambda ws, gen=None: 2e9)
    import gmlx.prefill_decay as pd
    monkeypatch.setattr(pd, "untracked_weight_bytes", lambda: 50e9)
    gov._REG.clear()
    gov._INJECT_FIRED.clear()
    tg._INJECT_FIRED.clear()
    for k in gov._STATS:
        if isinstance(gov._STATS[k], int):
            gov._STATS[k] = 0
    return box


def test_green_when_headroom_flat(rig):
    gen = FakeGen(rows=2, rate=1e6)  # demand ~2 MB/tick vs ~94 GB head
    st = gov._state(gen)
    for _ in range(4):
        gov._governor_tick(gen)
    assert st.band == gov.GREEN
    assert rig["mem_limits"] == []


def test_oneshot_projection_does_not_collapse_ttc(rig):
    # a pending deep prompt batch projects tens of GB (one-time join
    # cost) against wide headroom and a tiny rate: the band stays
    # green and admissions are not held
    import time as _t
    gen = FakeGen(rows=2, rate=1e6)
    gen._unprocessed_sequences = [("u", None)]   # join pending
    gen._kq_admit_last_projection = (_t.perf_counter(), 25e9)
    st = gov._state(gen)
    for _ in range(6):
        gov._governor_tick(gen)
    assert st.band == gov.GREEN
    assert gov.admission_hold_reason(gen) is None
    assert gov._STATS["red_failures"] == 0
    assert gov._STATS["orange_evictions"] == 0


def test_static_box_never_sheds_sweep(rig):
    # arch-independence invariant: sweep the decision inputs across
    # the realistic space. With a static box (no measured growth), a
    # claimed rate or one-shot of any size may throttle but must
    # never shed; when headroom also covers the level terms, the band
    # never passes yellow.
    import itertools
    import gmlx.server_memory as sm
    for head, rate, oneshot, live, rows in itertools.product(
            (8e9, 34e9, 94e9), (1e3, 1e9, 60e9), (0.0, 25e9, 60e9),
            (1e9, 30e9), (1, 8)):
        rig["head"] = head + 0.05 * 120e9
        gen = FakeGen(rows=rows, rate=rate, live=live)
        if oneshot:
            gen._unprocessed_sequences = [("u", None)]
            gen._kq_admit_last_projection = (time.perf_counter(), oneshot)
        st = gov._state(gen)
        tg_st = tg._state(gen)
        for u in range(rows):
            tg_st.ledger[u] = tg._Row([1] * 4, 64, {}, None, None)
        for _ in range(6):
            gov._governor_tick(gen)
        combo = f"head={head:g} rate={rate:g} oneshot={oneshot:g} "                 f"live={live:g} rows={rows}"
        assert gen.removed == [], combo
        assert gov._STATS["red_failures"] == 0, combo
        assert gov._STATS["orange_retires"] == 0, combo
        st_bytes = gov._shed_transient_bytes(gen, st)
        if head - oneshot > st_bytes + sm.admit_reserve_bytes(120e9, gen)                 + 4 * 64e6:
            assert st.band <= gov.YELLOW, combo
        gov._STATS["orange_evictions"] = 0


def test_yellow_arms_throttle_and_restores(rig, monkeypatch):
    monkeypatch.setenv("GMLX_GOV_MIN_DWELL_S", "0")
    gen = FakeGen(rows=2, rate=1e6)
    st = gov._state(gen)
    # demand so high that ttc <= KY but orange threshold not crossed:
    # head 94e9, KY=16 -> rate*rows*1 >= 5.9e9 per tick
    gen._kq_admit_kv_rates["layers.0.k"]["rate"] = 3.0e9
    st.obs_delta_ema = 1e9      # measured growth corroborates the claim
    gov._governor_tick(gen)
    assert st.band == gov.YELLOW
    # armed: tracked budget + untracked weights, cache pool to zero
    assert rig["mem_limits"] == [int(120e9 * 0.95 + 50e9)]
    assert rig["cache_limits"] == [0]
    assert gov._STATS["yellow_entries"] == 1

    gen._kq_admit_kv_rates["layers.0.k"]["rate"] = 1e6
    for _ in range(int(gov._env_f("GMLX_GOV_DY", 8)) + 1):
        gov._governor_tick(gen)
    assert st.band == gov.GREEN
    assert rig["mem_limits"][-1] == 0          # restored to saved value
    assert rig["cache_limits"][-1] == 7


def test_yellow_demand_rungs_on_measured_miss(rig, monkeypatch):
    monkeypatch.setenv("GMLX_GOV_RUNG_TICKS", "2")
    clamps = []
    import gmlx.speculative as spec
    monkeypatch.setattr(spec, "set_governor_width_clamp",
                        lambda n: clamps.append(n))
    gen = FakeGen(rows=4, rate=2.0e9)
    gen.draft_model = object()      # rung 2 only clamps armed speculation
    st = gov._state(gen)
    # peak and active keep climbing: every rung window is a miss and
    # measured growth corroborates the claimed rate
    for i in range(16):
        rig["peak"] = 20e9 + i * 1e9
        rig["active"] = 20e9 + i * 1.5e9
        gov._governor_tick(gen)
    assert st.band == gov.YELLOW
    assert gen.prefill_step_size == 1024        # rung 1: halved
    assert st.saved_prefill_step == 2048
    assert clamps == [2]                        # rung 2: width clamp rows/2

    # de-escalation restores both
    monkeypatch.setenv("GMLX_GOV_MIN_DWELL_S", "0")
    gen._kq_admit_kv_rates["layers.0.k"]["rate"] = 1e6
    for _ in range(9):
        gov._governor_tick(gen)
    assert st.band == gov.GREEN
    assert gen.prefill_step_size == 2048
    assert clamps[-1] == 0


def test_orange_evicts_registered_then_retires(rig, monkeypatch):
    monkeypatch.setenv("GMLX_GOV_MIN_DWELL_S", "0")
    evictions = []

    def evict(f):
        evictions.append(f)
        rig["active"] -= 1e9  # measured recovery: active actually drops
        return int(9e9)

    gov.register_cache("t", lambda: 12e9, evict)
    gen = FakeGen(rows=2, rate=1e9, live=21e9)
    st = gov._state(gen)
    st.obs_delta_ema = 0.3e9    # observed growth keeps the clock running
    tg_st = tg._state(gen)
    tg_st.ledger[0] = tg._Row([1] * 8, 64, {}, None, None)
    tg_st.ledger[1] = tg._Row([1] * 4, 64, {}, None, None)
    gen.apc_manager = types.SimpleNamespace(disk=object())

    # orange: ttc 28/2 = 14 <= 16 and head_eff 28e9 - ko*rate 8e9
    # < shed(21e9 + 5.25e9) + reserve 2e9
    rig["head"] = 34e9
    gov._governor_tick(gen)
    assert st.band == gov.ORANGE
    assert evictions == [0.5]
    assert gen.removed == []                    # first pass: evict only

    gov._governor_tick(gen)                     # fraction ramped to 1.0
    assert evictions == [0.5, 1.0]
    # full-fraction evict covers the shortfall (8.25e9 < freed 9e9),
    # so the condition is clearable and no retire fires
    assert gen.removed == []


def test_orange_retire_when_caches_dry(rig, monkeypatch):
    monkeypatch.setenv("GMLX_GOV_MIN_DWELL_S", "0")
    gov.register_cache("dry", lambda: 0, lambda f: 0)
    gen = FakeGen(rows=2, rate=2e9, live=30e9)
    st = gov._state(gen)
    st.obs_delta_ema = 1e9
    st.evict_fraction = 1.0                     # already ramped
    tg_st = tg._state(gen)
    tg_st.ledger[0] = tg._Row([1] * 8, 64, {}, None, None)
    tg_st.ledger[1] = tg._Row([1] * 4, 64, {}, None, None)
    gen.apc_manager = types.SimpleNamespace(disk=object())

    rig["head"] = 44e9
    gov._governor_tick(gen)
    assert st.band == gov.ORANGE
    assert gen.removed == [0]                   # largest row retired
    # requeued as itself at the queue head
    uid, replay, remaining, kw, lp, tc = gen._unprocessed_sequences[0]
    assert uid == 0 and len(replay) == 8
    assert gov._STATS["orange_retires"] == 1


def test_orange_without_apc_falls_to_red(rig, monkeypatch):
    monkeypatch.setenv("GMLX_GOV_MIN_DWELL_S", "0")
    gov.register_cache("dry", lambda: 0, lambda f: 0)
    failed = []
    monkeypatch.setattr(tg, "_row_failed_callbacks",
                        [lambda uid, info: failed.append(uid)])
    gen = FakeGen(rows=2, rate=2e9, live=30e9)
    st = gov._state(gen)
    st.obs_delta_ema = 1e9
    st.evict_fraction = 1.0
    tg_st = tg._state(gen)
    tg_st.ledger[0] = tg._Row([1] * 8, 64, {}, None, None)
    tg_st.ledger[1] = tg._Row([1] * 4, 64, {}, None, None)
    # no apc_manager: retire rung gated off

    rig["head"] = 44e9
    gov._governor_tick(gen)
    assert st.band == gov.ORANGE and st.orange_failed
    gov._governor_tick(gen)                     # orange_failed -> red
    assert st.band == gov.RED
    assert failed == [0]
    assert gov._STATS["red_failures"] == 1


def test_red_on_negative_next_tick_headroom(rig, monkeypatch):
    failed = []
    monkeypatch.setattr(tg, "_row_failed_callbacks",
                        [lambda uid, info: failed.append((uid, info))])
    gen = FakeGen(rows=2, rate=60e9, live=30e9)  # demand > headroom
    st = gov._state(gen)
    st.obs_delta_ema = 20e9     # observed growth backs the claim
    tg_st = tg._state(gen)
    tg_st.ledger[0] = tg._Row([1] * 8, 64, {}, None, None)
    tg_st.ledger[1] = tg._Row([1] * 4, 64, {}, None, None)
    gov._governor_tick(gen)
    assert st.band == gov.RED
    assert failed and failed[0][0] == 0
    assert "governor red" in failed[0][1]["error"]


def test_red_does_not_latch_on_healthy_headroom(rig, monkeypatch):
    # One forced red tick on a healthy box: survivors keep growing, so
    # the shed's measured recovery is <= 0. The failed-recovery latch
    # must not re-arm red from red (only orange escalates on it); the
    # band de-escalates and exactly one row is shed.
    monkeypatch.setenv("GMLX_OOM_INJECT", "red@1")
    monkeypatch.setenv("GMLX_GOV_MIN_DWELL_S", "0")
    failed = []
    monkeypatch.setattr(tg, "_row_failed_callbacks",
                        [lambda uid, info: failed.append(uid)])
    gen = FakeGen(rows=2, rate=1e6)
    st = gov._state(gen)
    tg_st = tg._state(gen)
    tg_st.ledger[0] = tg._Row([1] * 8, 64, {}, None, None)
    tg_st.ledger[1] = tg._Row([1] * 4, 64, {}, None, None)
    gov._governor_tick(gen)
    assert st.band == gov.RED and failed == [0]
    rig["active"] += 1e9                        # survivors grew
    for _ in range(int(gov._env_f("GMLX_GOV_DY", 8)) + 2):
        gov._governor_tick(gen)
    assert st.band == gov.GREEN
    assert failed == [0]                        # no repeat sheds
    assert not st.orange_failed


def test_shed_rate_cap(rig, monkeypatch):
    monkeypatch.setenv("GMLX_GOV_SHEDS_PER_MIN", "1")
    failed = []
    monkeypatch.setattr(tg, "_row_failed_callbacks",
                        [lambda uid, info: failed.append(uid)])
    gen = FakeGen(rows=3, rate=60e9, live=30e9)
    gov._state(gen).obs_delta_ema = 20e9
    tg_st = tg._state(gen)
    for u in range(3):
        tg_st.ledger[u] = tg._Row([1] * (8 - u), 64, {}, None, None)
    gov._governor_tick(gen)
    gov._governor_tick(gen)
    assert failed == [0]                        # second shed suppressed
    assert gov._STATS["sheds_suppressed"] == 1


def test_admission_hold_and_ceiling_handoff(rig):
    gen = FakeGen(rows=2, rate=3.0e9)
    assert gov.admission_hold_reason(gen) is None
    gov._state(gen).obs_delta_ema = 1e9
    gov._governor_tick(gen)
    assert gov.admission_hold_reason(gen) == "governor yellow"

    evicted = []
    gov.register_cache("t", lambda: 1e9,
                       lambda f: (evicted.append(f), int(1e9))[1])
    gov.make_room_for_admission(gen)
    assert evicted == [1.0]


def test_inject_forces_bands_once(rig, monkeypatch):
    monkeypatch.setenv("GMLX_OOM_INJECT", "yellow@2")
    gen = FakeGen(rows=2, rate=1e6)
    st = gov._state(gen)
    gov._governor_tick(gen)
    assert st.band == gov.GREEN
    gov._governor_tick(gen)                     # forced at tick 2
    assert st.band == gov.YELLOW
    assert ("yellow", 2) in st.injected


def test_kill_switch(monkeypatch):
    monkeypatch.setenv("GMLX_GOVERNOR", "0")
    assert gov.install_governor() is False
    gen = FakeGen()
    gov.make_room_for_admission(gen)            # no-op, no crash
    assert gov.admission_hold_reason(gen) is None


def test_row_failed_bridge_delivers_and_closes(monkeypatch):
    from mlx_vlm.server import generation as gen_mod

    from gmlx.server_patches.row_failed import install_row_failed_bridge

    install_row_failed_bridge()
    stepped = []
    orig = gen_mod.ResponseGenerator._step

    def fake_inner(self, batch_gen, active, gen_kwargs=None):
        stepped.append(True)
        # the engine tick fails uid 5 permanently during the step
        for fn in tg._row_failed_callbacks:
            fn(5, {"prompt_len": 8, "delivered": 3, "error": "pressure"})

    # rebuild the wrapper over the fake inner
    monkeypatch.setattr(gen_mod.ResponseGenerator, "_step", fake_inner)
    delattr_installed = getattr(orig, "_kq_gguf_row_failed_bridge", False)
    assert delattr_installed  # the real wrapper was installed above
    install_row_failed_bridge()  # re-wrap the fake (flag lives on wrapper)

    rq = queue.Queue()
    active = {5: {"rqueue": rq}, 6: {"rqueue": queue.Queue()}}
    gen_mod.ResponseGenerator._step(object(), None, active)
    assert stepped == [True]
    err = rq.get_nowait()
    assert isinstance(err, RowShedError) and err.uid == 5
    assert "3 tokens" in str(err) and "retryable" in str(err)
    assert rq.get_nowait() is None
    assert 5 not in active and 6 in active


def test_keepalive_sse_translates_shed_error():
    import asyncio
    import json

    from gmlx.server_patches.request_flow import _keepalive_sse

    async def body():
        yield "data: x\n\n"
        raise RowShedError(7, {"prompt_len": 4, "delivered": 2,
                               "error": "pressure"})

    async def run():
        return [c async for c in _keepalive_sse(body(), None)]

    chunks = asyncio.run(run())
    assert chunks[0] == "data: x\n\n"
    payload = json.loads(chunks[1][len("data: "):])
    assert payload["finish_reason"] == "shed"
    assert payload["error"]["type"] == "server_overloaded_shed"
    assert payload["error"]["delivered"] == 2
    assert chunks[2] == "data: [DONE]\n\n"


def test_keepalive_sse_upgrades_swallowed_shed_chunk():
    # The upstream chat/responses stream handlers catch the error and
    # yield a plain {"error": str} event; the wrapper must upgrade it.
    import asyncio
    import json

    from gmlx.server_patches.request_flow import _keepalive_sse

    err = RowShedError(7, {"prompt_len": 4, "delivered": 2,
                           "error": "pressure"})

    async def body():
        yield "data: x\n\n"
        yield f"data: {json.dumps({'error': str(err)})}\n\n"
        yield "data: [DONE]\n\n"  # never reached by the wrapper

    async def run():
        return [c async for c in _keepalive_sse(body(), None)]

    chunks = asyncio.run(run())
    payload = json.loads(chunks[1][len("data: "):])
    assert payload["finish_reason"] == "shed"
    assert payload["error"]["type"] == "server_overloaded_shed"
    assert chunks[2] == "data: [DONE]\n\n" and len(chunks) == 3


def test_keepalive_sse_other_errors_still_raise():
    import asyncio

    from gmlx.server_patches.request_flow import _keepalive_sse

    async def body():
        yield "a"
        raise RuntimeError("boom")

    async def run():
        out = []
        async for c in _keepalive_sse(body(), None):
            out.append(c)

    with pytest.raises(RuntimeError, match="boom"):
        asyncio.run(run())


def test_apc_governor_bytes_and_evict():
    from mlx_vlm import apc as _apc

    from gmlx.apc_manager import GmlxAPCManager

    mgr = GmlxAPCManager(num_blocks=4, block_size=4)
    # commit two blocks; block 0 stays referenced by a live row
    for i, ref in ((0, 1), (1, 0)):
        b = mgr.pool[i]
        b.keys = [mx.zeros((1, 1, 4, 8))]
        b.values = [mx.zeros((1, 1, 4, 8))]
        b.block_hash = 1000 + i
        mgr.hash_table[1000 + i] = b
        b.ref_cnt = ref
        if ref:
            mgr._free_remove(b)
    per_block = 2 * 4 * 8 * 4  # K+V float32
    entry_cache = types.SimpleNamespace(nbytes=64)
    mgr._exact_cache[1] = _apc.APCExactCacheEntry(
        token_ids=(1,), extra_hash=0, prompt_cache=[entry_cache],
        last_used=0.0)
    assert mgr.governor_bytes() == 2 * per_block + 64

    freed = mgr.governor_evict(1.0)
    assert freed == per_block + 64              # referenced block kept
    assert mgr.pool[1].keys is None
    assert 1001 not in mgr.hash_table
    assert mgr.pool[0].keys is not None         # ref_cnt 1 survives
    assert len(mgr._exact_cache) == 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))


def test_red_reclaims_before_shedding(rig, monkeypatch):
    # Static red with a fat reclaimable cache: evict + pool clear
    # restore headroom, so no row is shed.
    failed = []
    monkeypatch.setattr(tg, "_row_failed_callbacks",
                        [lambda uid, info: failed.append(uid)])
    gen = FakeGen(rows=2, rate=60e9, live=30e9)
    st = gov._state(gen)
    st.obs_delta_ema = 20e9
    tg_st = tg._state(gen)
    tg_st.ledger[0] = tg._Row([1] * 8, 64, {}, None, None)
    tg_st.ledger[1] = tg._Row([1] * 4, 64, {}, None, None)
    gov._REG["pool"] = (lambda: 50e9,
                        lambda fraction: (rig.__setitem__("head", 200e9),
                                          50e9)[1])
    gov._governor_tick(gen)
    assert failed == []
    assert gov._STATS["red_failures"] == 0
    assert "shed skipped" in gov._STATS["last_action"]
    assert st.band == gov.RED  # band holds; next healthy ticks demote
