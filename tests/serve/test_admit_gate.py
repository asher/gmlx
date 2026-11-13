"""Admission headroom gate: decision rules, stash nesting, projection."""

from types import SimpleNamespace

import mlx.core as mx
import pytest

from gmlx._vendor.mlx_vlm.generate import ar

import gmlx.serve.admit_gate as ag
import gmlx.serve.batch_sched as batch_sched
import gmlx.serve.memory as sm


class _Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += dt


class FakeKV:
    def __init__(self, rows=1, length=256, offset=100, window=None):
        self.keys = mx.zeros((rows, 4, length, 8), dtype=mx.float16)
        self.values = mx.zeros((rows, 4, length, 8), dtype=mx.float16)
        self.offset = offset
        if window is not None:
            self.max_size = window


class FakeBatch:
    def __init__(self, uids, prompt_cache):
        self.uids = list(uids)
        self.prompt_cache = prompt_cache

    def __len__(self):
        return len(self.uids)


class FakeModel:
    config = SimpleNamespace(num_attention_heads=4, model_type="faketype")


def _pending(uid, prompt_toks=300, max_toks=200):
    return (uid, [0] * prompt_toks, max_toks, {}, None, None)


class FakeGen:
    completion_batch_size = 32
    prefill_batch_size = 1
    model = FakeModel()

    def __init__(self, rows=1, pending=(1,)):
        cache = [FakeKV(rows=max(rows, 1))] if rows else []
        self._generation_batch = FakeBatch(range(100, 100 + rows), cache)
        self._prompt_batch = None
        self._unprocessed_sequences = [_pending(u) for u in pending]
        self._prompt_time_counter = 0.0
        self.admitted = []
        self.ticks = 0


def _fake_next(self, **kw):
    self.ticks += 1
    if self._unprocessed_sequences and self._prompt_batch is None:
        n = min(self.prefill_batch_size, len(self._unprocessed_sequences))
        for s in self._unprocessed_sequences[:n]:
            self.admitted.append(s[0])
        self._unprocessed_sequences = self._unprocessed_sequences[n:]
    return [], []


@pytest.fixture
def gated(monkeypatch):
    monkeypatch.setattr(ar.BatchGenerator, "_next", _fake_next)
    clock = _Clock()
    monkeypatch.setattr(ag, "time", SimpleNamespace(perf_counter=clock))
    ag.install_admit_headroom_gate()
    yield ar.BatchGenerator._next, clock


def _always_decline(monkeypatch, projected=100e9, headroom=10e9):
    monkeypatch.setattr(
        sm, "project_admission",
        lambda gen, cands: (projected, headroom, "kv 90.0"))


def test_kill_switch_skips_install(monkeypatch):
    monkeypatch.setenv("GMLX_ADMIT_HEADROOM", "0")
    monkeypatch.setattr(ar.BatchGenerator, "_next", _fake_next)
    ag.install_admit_headroom_gate()
    assert ar.BatchGenerator._next is _fake_next


def test_no_projection_admits(gated):
    wrapped, _ = gated
    g = FakeGen(rows=1)
    # fresh model: rates measurable but headroom probe may fail off-device;
    # decision errors must degrade to admission
    wrapped(g)
    assert g.admitted == [1]


def test_decline_hides_pending_and_merges_arrivals(gated, monkeypatch):
    wrapped, clock = gated
    _always_decline(monkeypatch)
    g = FakeGen(rows=1, pending=(1, 2))
    pending = g._unprocessed_sequences

    orig_fake = _fake_next

    def _next_with_arrival(self, **kw):
        self._unprocessed_sequences.append(_pending(3))
        return orig_fake(self, **kw)

    monkeypatch.setattr(ar.BatchGenerator, "_next", _next_with_arrival,
                        raising=False)
    # re-wrap: the fixture installed over _fake_next; call the wrapper we got
    for _ in range(3):
        clock.advance(0.05)
        wrapped(g)
    assert g.admitted == []  # never formed a batch
    assert g._unprocessed_sequences is pending
    assert [s[0] for s in pending[:2]] == [1, 2]  # order preserved
    assert 1 in g._kq_admit_deferred_s


def test_never_declines_idle_server(gated, monkeypatch):
    wrapped, _ = gated
    _always_decline(monkeypatch)
    g = FakeGen(rows=0)
    wrapped(g)
    assert g.admitted == [1]


def test_never_declines_with_prompt_batch_live(gated, monkeypatch):
    wrapped, _ = gated
    calls = []
    monkeypatch.setattr(sm, "project_admission",
                        lambda gen, cands: calls.append(1) or None)
    g = FakeGen(rows=1)
    g._prompt_batch = object()
    wrapped(g)
    # decision never consulted: formation impossible this tick
    assert calls == []


def test_defer_ceiling_admits_loudly(gated, monkeypatch, caplog):
    wrapped, clock = gated
    _always_decline(monkeypatch)
    monkeypatch.setenv("GMLX_ADMIT_DEFER_MAX_S", "1")
    g = FakeGen(rows=1)
    with caplog.at_level("WARNING"):
        for _ in range(5):
            clock.advance(0.6)
            wrapped(g)
    assert g.admitted == [1]
    assert any("defer ceiling" in r.message for r in caplog.records)


def test_decision_failure_degrades_to_admission(gated, monkeypatch):
    wrapped, _ = gated

    def _boom(gen, cands):
        raise RuntimeError("probe broke")

    monkeypatch.setattr(sm, "project_admission", _boom)
    g = FakeGen(rows=1)
    wrapped(g)
    assert g.admitted == [1]


def test_gate_outside_pacer_composes(monkeypatch):
    """8.6: both wrappers installed, mid-call arrival survives the nested
    stashes, and a declined tick starves the pacer of prefill work."""
    monkeypatch.setenv("GMLX_DECODE_PREFILL_RATIO", "1.0")

    def _next_with_arrival(self, **kw):
        self._unprocessed_sequences.append(_pending(9))
        return _fake_next(self, **kw)

    monkeypatch.setattr(ar.BatchGenerator, "_next", _next_with_arrival)
    clock = _Clock()
    monkeypatch.setattr(batch_sched, "time",
                        SimpleNamespace(perf_counter=clock))
    monkeypatch.setattr(ag, "time", SimpleNamespace(perf_counter=clock))
    batch_sched.install_decode_priority_sched()
    ag.install_admit_headroom_gate()
    _always_decline(monkeypatch)
    wrapped = ar.BatchGenerator._next
    g = FakeGen(rows=1, pending=(1,))
    pending = g._unprocessed_sequences
    for _ in range(3):
        clock.advance(0.05)
        wrapped(g)
    # The gated candidate is never admitted; a mid-call arrival lands in
    # the stash-tick's temp list and may be admitted by the stock body
    # that same tick (the same race exists un-gated) but is never lost.
    assert g.admitted == [9, 9, 9]
    assert g._unprocessed_sequences is pending
    assert [s[0] for s in pending] == [1]  # candidate kept its position


def _held_then_green(monkeypatch, bands):
    """Both wrappers over a stock body whose tick is 0.5 s of decode. The
    last chunk took 3 s and just ran, so the pacer owes 3 s of decode
    before the next. ``bands`` gives the governor band tick by tick: a
    hold while yellow, and a join that fits while green."""
    import gmlx.serve.governor as gov

    monkeypatch.setenv("GMLX_DECODE_PREFILL_RATIO", "1.0")
    clock = _Clock()

    def _decode_tick(self, **kw):
        clock.advance(0.5)
        return _fake_next(self, **kw)

    monkeypatch.setattr(ar.BatchGenerator, "_next", _decode_tick)
    monkeypatch.setattr(batch_sched, "time",
                        SimpleNamespace(perf_counter=clock))
    monkeypatch.setattr(ag, "time", SimpleNamespace(perf_counter=clock))
    batch_sched.install_decode_priority_sched()
    ag.install_admit_headroom_gate()
    band = iter(bands)
    monkeypatch.setattr(gov, "admission_hold_reason",
                        lambda gen: next(band) or None)
    monkeypatch.setattr(ag, "_evict_shortfall", lambda *a: False)
    monkeypatch.setattr(ag, "_evict_shortfall_held", lambda *a: False)
    _always_decline(monkeypatch, projected=1e9, headroom=10e9)
    g = FakeGen(rows=1)
    g._kq_last_chunk_time = 3.0
    g._kq_decode_since_chunk = 0.0
    return ar.BatchGenerator._next, g


def test_a_join_held_past_what_the_pacer_owes_is_taken_when_the_band_clears(
        monkeypatch, caplog):
    """The governed rung: a join held 4 s in yellow, then one green tick
    before the band turns back. The decode the hold ran pays the 3 s the
    pacer owed, so the green tick takes the join. Without it the pacer
    held the join that tick, the band turned back and held it again."""
    yellow = "governor yellow"
    wrapped, g = _held_then_green(monkeypatch, [yellow] * 8 + [""]
                                  + [yellow] * 20)
    with caplog.at_level("INFO", logger=ag._log.name):
        for _ in range(9):
            wrapped(g)
    assert g.admitted == [1]
    admits = [r.message for r in caplog.records if "admitting" in r.message]
    assert admits == ["[admit] admitting uid=[1] after 3.0s deferred"]


def test_the_admitting_line_is_logged_once_on_the_tick_that_takes_the_join(
        monkeypatch, caplog):
    """A join held 1 s, under the 3 s of decode the pacer owes: the pacer
    holds it for the rest. The admitting line comes once, on the tick the
    join is taken, not on every tick the gate lets it through."""
    yellow = "governor yellow"
    wrapped, g = _held_then_green(monkeypatch, [yellow] * 2 + [""] * 20)
    taken_at = None
    with caplog.at_level("INFO", logger=ag._log.name):
        for tick in range(1, 13):
            wrapped(g)
            if g.admitted and taken_at is None:
                taken_at = tick
                lines = sum("admitting" in r.message for r in caplog.records)
    assert g.admitted == [1]
    # 1 s of held decode and 4 paced ticks of 0.5 s pay the 3 s.
    assert taken_at == 7
    assert lines == 1
    assert sum("admitting" in r.message for r in caplog.records) == 1


def test_update_kv_rates_and_projection(monkeypatch):
    import gmlx.gen.prefill_decay as pd

    monkeypatch.setenv("GMLX_ADMIT_RESERVE_GB", "2")
    g = FakeGen(rows=1)
    kv = g._generation_batch.prompt_cache[0]
    per_tok = (kv.keys.nbytes + kv.values.nbytes) / kv.offset  # min(100,256)
    sm.update_kv_rates(g)
    rates = g._kq_admit_kv_rates
    assert rates["FakeKV"]["rate"] == pytest.approx(per_tok)
    assert g._kq_admit_live_depth == 100

    monkeypatch.setattr(pd, "headroom_bytes", lambda: 10e9)
    out = sm.project_admission(g, [_pending(2, 300, 200)])
    assert out is not None
    projected, head, parts = out
    assert head == 10e9
    # padded form: width 2, depth round_block(500) = 512
    kv_total = per_tok * 2 * 512
    kv_new = kv_total - (kv.keys.nbytes + kv.values.nbytes)
    assert projected == pytest.approx(
        kv_new + pd.score_transient_bytes(g.model, None, 500)
        + sm.admit_reserve_bytes(0), rel=0.05)
    assert "kv" in parts and "reserve" in parts


class FakeBatchKV(FakeKV):
    """A batch class: an int ``_idx`` watermark, ``offset`` a per-row vector."""

    def __init__(self, rows=2, length=256, idx=100):
        super().__init__(rows=rows, length=length, offset=idx)
        self._idx = idx
        self.offset = mx.full((rows,), idx, dtype=mx.int32)


class FakeAcctKV:
    """A cache that owns its accounting; its arrays' time axis (a ring of
    1280 rows) is not its token count."""

    def __init__(self, per_token=1000.0, depth=300, row_bytes=5000.0):
        self.acct = (per_token, depth, row_bytes)
        self.keys = mx.zeros((2, 4, 1280, 8), dtype=mx.float16)
        self.offset = mx.zeros((2,), dtype=mx.int32)

    def admit_bytes(self):
        return self.acct


def test_update_kv_rates_reads_the_batch_watermark():
    g = FakeGen(rows=2)
    kv = FakeBatchKV(rows=2, idx=100)
    g._generation_batch.prompt_cache = [kv]
    sm.update_kv_rates(g)
    per_tok = (kv.keys.nbytes + kv.values.nbytes) / 100 / 2
    # the walk also counts the offset vector's few bytes
    assert g._kq_admit_kv_rates["FakeBatchKV"]["rate"] == pytest.approx(
        per_tok, rel=1e-3)
    assert g._kq_admit_live_depth == 100
    assert g._kq_admit_live_bytes == pytest.approx(
        kv.keys.nbytes + kv.values.nbytes, rel=1e-3)
    assert g._kq_admit_spec_row_const == 0


def test_update_kv_rates_takes_the_cache_accounting(monkeypatch, caplog):
    import gmlx.gen.prefill_decay as pd

    g = FakeGen(rows=2)
    kv = FakeAcctKV()
    g._generation_batch.prompt_cache = [kv]
    sm.update_kv_rates(g)
    assert g._kq_admit_kv_rates["FakeAcctKV"]["rate"] == pytest.approx(500.0)
    assert g._kq_admit_live_depth == 300
    assert g._kq_admit_spec_row_const == pytest.approx(2500.0)
    assert g._kq_admit_live_bytes == pytest.approx(1000.0 * 300 + 5000.0)
    # the model reproduces the measurement at the live width: no rescale
    monkeypatch.setattr(pd, "headroom_bytes", lambda: 10e9)
    with caplog.at_level("WARNING", logger=sm._log.name):
        out = sm.project_admission(g, [_pending(2, 300, 200)])
    assert out is not None
    assert "projection rescaled" not in caplog.text


class FakeStateCache:
    """Offset-less constant-size state (GDN/conv), ArraysCache-shaped."""

    def __init__(self, rows=1):
        self.cache = [mx.zeros((rows, 4, 128, 32), dtype=mx.float16)]


def test_state_cache_is_row_const_not_rate(monkeypatch):
    import gmlx.gen.prefill_decay as pd

    monkeypatch.setenv("GMLX_ADMIT_RESERVE_GB", "2")
    g = FakeGen(rows=1)
    kv = g._generation_batch.prompt_cache[0]
    state = FakeStateCache(rows=1)
    g._generation_batch.prompt_cache.append(state)
    sm.update_kv_rates(g)
    assert "FakeStateCache" not in g._kq_admit_kv_rates
    state_bytes = state.cache[0].nbytes
    assert g._kq_admit_spec_row_const == pytest.approx(state_bytes)
    assert g._kq_admit_live_depth == 100

    monkeypatch.setattr(pd, "headroom_bytes", lambda: 10e9)
    per_tok = (kv.keys.nbytes + kv.values.nbytes) / kv.offset
    projected, head, parts = sm.project_admission(g, [_pending(2, 300, 200)])
    kv_total = per_tok * 2 * 512 + state_bytes * 2
    kv_new = kv_total - (kv.keys.nbytes + kv.values.nbytes + state_bytes)
    assert projected == pytest.approx(
        kv_new + pd.score_transient_bytes(g.model, None, 500)
        + sm.admit_reserve_bytes(0), rel=0.05)


class FakeRotKV(FakeKV):
    pass


class FakeQuantKV:
    """Quantized storage: (data, scales, biases) tuples, offset-bearing."""

    def __init__(self, rows=1, length=256, offset=100):
        self.keys = (mx.zeros((rows, 4, length, 8), dtype=mx.uint32),
                     mx.zeros((rows, 4, length, 1), dtype=mx.float16),
                     mx.zeros((rows, 4, length, 1), dtype=mx.float16))
        self.values = (mx.zeros((rows, 4, length, 8), dtype=mx.uint32),
                       mx.zeros((rows, 4, length, 1), dtype=mx.float16),
                       mx.zeros((rows, 4, length, 1), dtype=mx.float16))
        self.offset = offset


def _zoo_gen():
    g = FakeGen(rows=1)
    g._generation_batch.prompt_cache = [
        FakeKV(offset=100),
        FakeRotKV(offset=100, window=128),
        FakeQuantKV(offset=100),
        FakeStateCache(),
    ]
    return g


def test_cache_zoo_rates_and_consistency(monkeypatch, caplog):
    import gmlx.gen.prefill_decay as pd

    monkeypatch.setenv("GMLX_ADMIT_RESERVE_GB", "2")
    g = _zoo_gen()
    sm.update_kv_rates(g)
    rates = g._kq_admit_kv_rates
    assert set(rates) == {"FakeKV", "FakeRotKV", "FakeQuantKV"}
    assert rates["FakeRotKV"]["window"] == 128
    assert g._kq_admit_spec_row_const > 0  # state cache landed here

    monkeypatch.setattr(pd, "headroom_bytes", lambda: 10e9)
    with caplog.at_level("WARNING"):
        out = sm.project_admission(g, [_pending(2, 300, 200)])
    assert out is not None
    assert "projection rescaled" not in caplog.text


def test_poisoned_rate_is_rescaled(monkeypatch, caplog):
    import gmlx.gen.prefill_decay as pd

    monkeypatch.setenv("GMLX_ADMIT_RESERVE_GB", "2")
    g = FakeGen(rows=1)
    kv = g._generation_batch.prompt_cache[0]
    honest_per_tok = (kv.keys.nbytes + kv.values.nbytes) / kv.offset
    # Poison the live kind: the EWMA merge keeps only kinds present in
    # the fresh walk, so the bad rate must ride an existing kind.
    g._kq_admit_kv_rates = {"FakeKV": {"rate": 1e9, "window": None}}
    monkeypatch.setattr(pd, "headroom_bytes", lambda: 10e9)
    with caplog.at_level("WARNING"):
        projected, head, parts = sm.project_admission(
            g, [_pending(2, 300, 200)])
    assert "projection rescaled" in caplog.text
    honest_kv_new = honest_per_tok * 2 * 512 - (kv.keys.nbytes
                                                + kv.values.nbytes)
    assert projected < 10 * (honest_kv_new
                             + pd.score_transient_bytes(g.model, None, 500)
                             + sm.admit_reserve_bytes(0))


def test_understated_rate_is_rescaled_up(monkeypatch, caplog):
    import gmlx.gen.prefill_decay as pd

    monkeypatch.setenv("GMLX_ADMIT_RESERVE_GB", "2")
    g = FakeGen(rows=1)
    kv = g._generation_batch.prompt_cache[0]
    honest_per_tok = (kv.keys.nbytes + kv.values.nbytes) / kv.offset
    g._kq_admit_kv_rates = {"FakeKV": {"rate": honest_per_tok * 1e-3,
                                       "window": None}}
    monkeypatch.setattr(pd, "headroom_bytes", lambda: 10e9)
    with caplog.at_level("WARNING"):
        projected, head, parts = sm.project_admission(
            g, [_pending(2, 300, 200)])
    assert "projection rescaled" in caplog.text
    live = kv.keys.nbytes + kv.values.nbytes
    # rescale restores the honest scale: kv_total ~= live * depth growth
    assert projected >= 0.5 * (honest_per_tok * 2 * 512 - live)


def _spec_gen(rows=1):
    g = FakeGen(rows=rows)
    b = g._generation_batch
    b.hidden = mx.zeros((rows, 16, 8))            # row const
    b.first_tokens = mx.zeros((rows,), dtype=mx.int32)
    b.shared_kv_states = [mx.zeros((rows, 4, 64))]  # depth scaled
    g.draft_model = SimpleNamespace(_cache=[FakeKV(rows=rows, offset=100)])
    return g


def test_spec_state_bytes_split():
    g = _spec_gen()
    depth_scaled, row_const = sm.spec_state_bytes(g)
    dkv = g.draft_model._cache[0]
    assert depth_scaled == (g._generation_batch.shared_kv_states[0].nbytes
                            + dkv.keys.nbytes + dkv.values.nbytes)
    assert row_const == (g._generation_batch.hidden.nbytes
                         + g._generation_batch.first_tokens.nbytes)
    plain = FakeGen(rows=1)
    assert sm.spec_state_bytes(plain) == (0.0, 0.0)


def test_update_kv_rates_folds_spec_state():
    g = _spec_gen()
    sm.update_kv_rates(g)
    rates = g._kq_admit_kv_rates
    depth_scaled, row_const = sm.spec_state_bytes(g)
    assert rates["_spec_state"]["rate"] == pytest.approx(
        depth_scaled / 100)  # live depth 100, one row
    assert rates["_spec_state"]["window"] is None
    kv = g._generation_batch.prompt_cache[0]
    assert g._kq_admit_live_bytes == pytest.approx(
        kv.keys.nbytes + kv.values.nbytes + depth_scaled + row_const)
    assert g._kq_admit_spec_row_const == pytest.approx(row_const)
    plain = FakeGen(rows=1)
    sm.update_kv_rates(plain)
    assert "_spec_state" not in plain._kq_admit_kv_rates
    assert plain._kq_admit_spec_row_const == 0.0


def test_projection_prices_spec_growth(monkeypatch):
    import gmlx.gen.prefill_decay as pd

    monkeypatch.setenv("GMLX_ADMIT_RESERVE_GB", "2")
    monkeypatch.setattr(pd, "headroom_bytes", lambda: 10e9)
    plain, spec = FakeGen(rows=1), _spec_gen()
    p_plain = sm.project_admission(plain, [_pending(2, 300, 200)])[0]
    p_spec = sm.project_admission(spec, [_pending(2, 300, 200)])[0]
    depth_scaled, row_const = sm.spec_state_bytes(spec)
    # spec projects extra: rate x width x depth + const x width - live spec
    extra = (depth_scaled / 100) * 2 * 512 + row_const * 2 \
        - (depth_scaled + row_const)
    assert p_spec - p_plain == pytest.approx(extra, rel=0.01)


def test_reserve_geometry_derived_after_walk(monkeypatch):
    monkeypatch.delenv("GMLX_ADMIT_RESERVE_GB", raising=False)
    g = FakeGen(rows=1)
    sm.update_kv_rates(g)
    kv = g._generation_batch.prompt_cache[0]
    cache_bytes = kv.keys.nbytes + kv.values.nbytes
    want = max(1e9, cache_bytes + g._kq_admit_live_bytes / (1 * 1))
    assert sm.admit_reserve_bytes(120e9, g) == pytest.approx(want)
    # no walk yet: the old constant stands in
    assert sm.admit_reserve_bytes(120e9, FakeGen(rows=1)) == \
        pytest.approx(max(2e9, 0.05 * 120e9))
    # env still wins
    monkeypatch.setenv("GMLX_ADMIT_RESERVE_GB", "3")
    assert sm.admit_reserve_bytes(120e9, g) == 3e9


class _AcctKV:
    """A cache that reports its own bytes, as the kvarn classes do."""

    def __init__(self, per_token, depth):
        self.per_token, self.depth = per_token, depth

    def admit_bytes(self):
        return self.per_token, self.depth, 0.0


def test_reserve_priced_from_the_header_is_what_the_walk_measures(monkeypatch):
    """A streamed install prices the room's reserve before any batch
    exists: the header's per-layer KV costs, at a batch of 8 rows at 32k
    tokens, give the reserve the walk measures on that batch."""
    monkeypatch.delenv("GMLX_ADMIT_RESERVE_GB", raising=False)
    from gmlx.serve.mem_preflight import FixedRows

    rows, depth = 8, 32768
    layer_bpt = [2048.0, 2048.0, 8192.0]
    state = 4e6
    g = FakeGen(rows=rows)
    cache = [_AcctKV(b * rows, depth) for b in layer_bpt]
    state_cache = SimpleNamespace(admit_bytes=lambda: (0.0, 0, state * rows))
    g._generation_batch = FakeBatch(range(rows), cache + [state_cache])
    sm.update_kv_rates(g)
    walked = sm.admit_reserve_bytes(120e9, g)
    costs = [(None, b) for b in layer_bpt] + [(FixedRows(1), state)]
    assert walked > 2e9
    assert sm.priced_admit_reserve_bytes(costs, depth, rows) == \
        pytest.approx(walked)
    # Near-zero KV, as on DeepSeek V4.1, takes the walk's 1 GB floor, not
    # the 5% of the working set that stands in before a walk.
    assert sm.priced_admit_reserve_bytes([(None, 8.0)] * 40, depth, rows) == 1e9
    monkeypatch.setenv("GMLX_ADMIT_RESERVE_GB", "3")
    assert sm.priced_admit_reserve_bytes(costs, depth, rows) == 3e9


def test_cache_release_gate_arms_syncs_and_restores(monkeypatch):
    calls = []
    monkeypatch.setattr(mx, "get_cache_memory", lambda: 777)
    monkeypatch.setattr(mx, "set_cache_limit",
                        lambda n: calls.append(("limit", n)) or 555)
    monkeypatch.setattr(mx, "synchronize",
                        lambda *a: calls.append(("sync", a)))
    with sm.cache_release_gate():
        calls.append(("body",))
    assert calls == [("limit", 777), ("body",), ("sync", ()),
                     ("limit", 555)]


def test_projection_window_caps_rotating_kinds(monkeypatch):
    import gmlx.gen.prefill_decay as pd

    monkeypatch.setenv("GMLX_ADMIT_RESERVE_GB", "2")
    g = FakeGen(rows=1)
    g._generation_batch.prompt_cache = [
        FakeKV(offset=100, window=128)]
    monkeypatch.setattr(pd, "headroom_bytes", lambda: 10e9)
    out = sm.project_admission(g, [_pending(2, 5000, 1000)])
    projected, _, _ = out
    kv = g._generation_batch.prompt_cache[0]
    per_tok = (kv.keys.nbytes + kv.values.nbytes) / 100
    # capped at round_block(128) = 256, never the 6144-token depth
    kv_new = per_tok * 2 * 256 - (kv.keys.nbytes + kv.values.nbytes)
    transient = pd.score_transient_bytes(g.model, None, 6000)
    assert projected == pytest.approx(
        kv_new + transient + sm.admit_reserve_bytes(0), rel=0.05)


class FakeGridKV(FakeKV):
    """A cache that grows on the capacity grid, as the gmlx KV caches do."""

    _kv_grid = True


def test_a_join_on_the_grid_is_priced_at_the_grid_point(monkeypatch):
    """A row 80000 tokens deep holds the grid point 80384. A join that
    takes the batch to 83000 tokens grows both rows to the next grid
    point, 88576, not to 83200, the block past it."""
    import gmlx.gen.prefill_decay as pd
    from gmlx.cache.kv_append import grid_capacity

    monkeypatch.setenv("GMLX_ADMIT_RESERVE_GB", "2")
    monkeypatch.setattr(pd, "headroom_bytes", lambda: 10e9)
    monkeypatch.setattr(pd, "score_transient_bytes", lambda *a, **k: 0.0)
    g = FakeGen(rows=1)
    kv = FakeGridKV(length=80384, offset=80000)
    g._generation_batch.prompt_cache = [kv]
    live = kv.keys.nbytes + kv.values.nbytes
    assert grid_capacity(83000, 256) == 88576

    projected, _, _ = sm.project_admission(g, [_pending(2, 82000, 1000)])

    kv_new = live / 80384 * 2 * 88576 - live
    assert projected - 2e9 == pytest.approx(kv_new, rel=1e-3)


class _LayerKV:
    """One attention layer that reports its bytes: ``per_token`` for the
    whole batch at ``depth``."""

    def __init__(self, per_token, depth):
        self.per_token, self.depth = per_token, depth

    def admit_bytes(self):
        return self.per_token, self.depth, 0.0


@pytest.mark.parametrize("rows, depth, prompt, gen",
                         [(1, 80000, 40000, 1000), (3, 40000, 1000, 80000)],
                         ids=["the-join", "the-final-depth"])
def test_a_join_reserves_its_own_caches_or_one_layer_at_the_final_depth(
        monkeypatch, rows, depth, prompt, gen):
    """16 layers of 16 KiB a row token. Beside a row 80000 deep, a join
    with a 40000 token prompt holds its own 16 layers at that prompt
    beside one live layer until the join returns: about 12 GB. Beside
    three rows 40000 deep, a short join that may write 80000 tokens takes
    the batch to that depth, where one layer at the joined width is held
    twice by a regrow: about 5 GB. The reserve held one live layer and
    one row of it, under 3 GB in both."""
    import gmlx.gen.prefill_decay as pd

    monkeypatch.delenv("GMLX_ADMIT_RESERVE_GB", raising=False)
    monkeypatch.setattr(pd, "headroom_bytes", lambda: 200e9)
    monkeypatch.setattr(pd, "score_transient_bytes", lambda *a, **k: 0.0)
    bpt, layers = 16384.0, 16
    g = FakeGen(rows=rows)
    g._generation_batch = FakeBatch(
        range(rows), [_LayerKV(bpt * rows, depth) for _ in range(layers)])
    live = layers * bpt * rows * depth

    projected, _, parts = sm.project_admission(
        g, [_pending(2, prompt, gen)])

    width, final = rows + 1, sm._round_block(max(depth, prompt + gen))
    kv_new = layers * bpt * width * final - live
    held_join = layers * bpt * sm._round_block(prompt) + bpt * rows * depth
    held_later = width * bpt * final
    want = held_join if rows == 1 else held_later
    assert want == max(held_join, held_later)
    assert projected - kv_new == pytest.approx(want, rel=1e-6)
    assert f"reserve {want / 1e9:.1f}" in parts


def test_empty_batch_projection_none():
    g = FakeGen(rows=0)
    g._generation_batch.prompt_cache = []
    assert sm.project_admission(g, [_pending(1)]) is None


def test_defer_ceiling_admits_one_row_per_tick(gated, monkeypatch):
    # past the ceiling the gate admits blind by design, but one row per
    # tick: the stock body must never form a full blind prompt batch
    wrapped, clock = gated
    _always_decline(monkeypatch)
    monkeypatch.setenv("GMLX_ADMIT_DEFER_MAX_S", "1")
    g = FakeGen(rows=1, pending=(1, 2, 3, 4, 5))
    g.prefill_batch_size = 4
    per_tick = []
    for _ in range(12):
        clock.advance(0.6)
        before = len(g.admitted)
        wrapped(g)
        per_tick.append(len(g.admitted) - before)
        assert not getattr(g, "_kq_admit_ceiling_tick", False)  # cleared
    assert max(per_tick) == 1
    assert g.admitted == [1, 2, 3, 4, 5]  # FCFS survives the trims
    # the final row admits through the stock path (nothing left to trim),
    # which rebinds the list; the trimmed ticks preserved identity
    assert g._unprocessed_sequences == []


def test_one_row_next_restore_semantics():
    g = FakeGen(rows=1, pending=(1, 2, 3))
    pending = g._unprocessed_sequences

    def consume_with_arrival(self, **kw):
        # stock body consumed the head; a handler-thread insert landed in
        # the temp list mid-call
        self._unprocessed_sequences = [_pending(9)]
        return "r"

    assert ag._one_row_next(g, consume_with_arrival, {}) == "r"
    assert g._unprocessed_sequences is pending
    assert [s[0] for s in pending] == [2, 3, 9]  # arrival at the tail

    def untouched(self, **kw):
        return "r2"

    assert ag._one_row_next(g, untouched, {}) == "r2"
    assert [s[0] for s in pending] == [2, 3, 9]  # unconsumed head kept


def test_fresh_model_projects_from_boot_rates(monkeypatch):
    import gmlx.gen.prefill_decay as pd
    import gmlx.serve.capacity as cap

    monkeypatch.setenv("GMLX_ADMIT_RESERVE_GB", "2")
    g = FakeGen(rows=0)
    monkeypatch.setattr(pd, "headroom_bytes", lambda: 10e9)
    monkeypatch.setattr(cap, "_TABLE", None)
    assert sm.project_admission(g, [_pending(2, 300, 200)]) is None
    monkeypatch.setattr(g.model, "_kq_boot_kv_costs", [(None, 1000.0)],
                        raising=False)
    out = sm.project_admission(g, [_pending(2, 300, 200)])
    assert out is not None
    projected, head, parts = out
    assert head == 10e9
    assert projected == pytest.approx(
        1000.0 * 512 + pd.score_transient_bytes(g.model, None, 500)
        + sm.admit_reserve_bytes(0), rel=0.05)
    assert "kv" in parts


def _pooled_gen(monkeypatch, projected_blocks, band=None, fits_ws=False):
    """A decoding row, a waiting request, and a prompt cache pool of two
    finished prompts: 32 idle blocks released first, then 4. The budget
    has room for 2 blocks, and each evicted block gives its bytes back."""
    import time

    from gmlx._vendor.mlx_vlm import apc
    from gmlx.cache.apc_manager import GmlxAPCManager

    import gmlx.gen.prefill_decay as pd
    import gmlx.serve.governor as gov

    monkeypatch.delenv("GMLX_GOVERNOR", raising=False)
    monkeypatch.setattr(apc, "_free_ram_bytes", lambda: 100 << 30)
    monkeypatch.setattr(mx, "get_active_memory", lambda: 0)
    monkeypatch.setattr(mx, "get_cache_memory", lambda: 0)
    monkeypatch.setattr(pd, "streamed_tracked_bytes", lambda: 0.0)
    m = GmlxAPCManager(num_blocks=64, block_size=16)
    for start, n in ((0, 512), (1000, 64)):
        k = mx.zeros((1, 2, n, 16), mx.float16)
        m.release(m.store_kv_blocks(
            list(range(start, start + n)), [k, k], [k, k]))
    assert len(m.hash_table) == 36
    full = m.resident_bytes()
    block = full // 36

    def head():
        return 2 * block + full - m.resident_bytes()

    monkeypatch.setattr(gov, "_headroom_and_ws",
                        lambda margin: (head(), 100e9))
    def project(gen, cands):
        # Stamped for the governor's charge, as the real projection is.
        gen._kq_admit_last_projection = (
            time.perf_counter(), projected_blocks * block)
        return (projected_blocks * block,
                head() + (1e9 if fits_ws else 0), "kv")

    monkeypatch.setattr(sm, "project_admission", project)
    monkeypatch.setitem(gov._STATS, "shortfall_evictions", 0)
    g = FakeGen(rows=1)
    g.apc_manager = m
    g._kq_admit_kv_rates = {"k": {"rate": 1.0, "window": None}}
    if band is not None:
        gov._state(g).band = band
    return g, m, block


def _cold_first(m):
    out, b = [], m._free_head
    while b is not None:
        if b.block_hash is not None:
            out.append(b.block_hash)
        b = b.next
    return out


@pytest.mark.parametrize("band, fits_ws", [(0, False), (0, True), (1, False)],
                         ids=["would-defer", "fits-the-working-set",
                              "held-in-yellow"])
def test_a_join_over_budget_evicts_its_shortfall_and_admits_at_once(
        gated, monkeypatch, band, fits_ws):
    """A join 5 blocks over the budget, with a pool that holds 36. The
    gate takes the shortfall from the pool's coldest blocks and admits on
    the same tick: no deferral, and the rest of the pool stays. Without
    the eviction the join defers, or it is admitted over the governed
    budget and every join after it is held."""
    import gmlx.serve.governor as gov

    wrapped, _ = gated
    g, m, block = _pooled_gen(monkeypatch, 7, band=band, fits_ws=fits_ws)
    cold = _cold_first(m)
    full = m.resident_bytes()

    wrapped(g)

    assert g.admitted == [1]
    assert not getattr(g, "_kq_admit_deferred_s", None)
    gone = [h for h in cold if h not in m.hash_table]
    assert gone == cold[:6]
    # The shortfall is 5 blocks and 16 bytes, the slack 512 bytes: one
    # more block covers both.
    assert full - m.resident_bytes() == 6 * block
    assert gov._STATS["shortfall_evictions"] == 1


@pytest.mark.parametrize("projected_blocks, band", [(50, 0), (7, 2)],
                         ids=["pool-short", "orange"])
def test_a_shortfall_the_pool_cannot_give_leaves_it_whole_and_defers(
        gated, monkeypatch, projected_blocks, band):
    """A join the pool cannot cover takes nothing from it, and neither
    does a join under an orange hold, where the governor's own rung owns
    the pool. The join defers as before."""
    wrapped, clock = gated
    g, m, _ = _pooled_gen(monkeypatch, projected_blocks, band=band)
    for _ in range(3):
        clock.advance(0.05)
        wrapped(g)
    assert g.admitted == []
    assert 1 in g._kq_admit_deferred_s
    assert len(m.hash_table) == 36


def test_a_join_still_held_in_yellow_keeps_its_projection_stamp(
        gated, monkeypatch):
    """Under a yellow hold the gate projects only to price an eviction. A
    join the pool cannot cover stays held with the stamp of its last
    projection, so the governor's charge for it runs out as it did. A join
    the eviction admits takes the new stamp, as any admitted join does."""
    wrapped, _ = gated
    g, _, block = _pooled_gen(monkeypatch, 50, band=1)
    old = g._kq_admit_last_projection = (1.5, 3.0 * block)
    wrapped(g)
    assert g.admitted == []
    assert g._kq_admit_last_projection is old

    g, _, block = _pooled_gen(monkeypatch, 7, band=1)
    g._kq_admit_last_projection = (1.5, 3.0 * block)
    wrapped(g)
    assert g.admitted == [1]
    assert g._kq_admit_last_projection[1] == 7 * block
