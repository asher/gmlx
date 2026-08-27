"""Admission headroom gate: decision rules, stash nesting, projection."""

from types import SimpleNamespace

import mlx.core as mx
import pytest

from mlx_vlm.generate import ar

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
