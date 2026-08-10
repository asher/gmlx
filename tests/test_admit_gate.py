"""Admission headroom gate: decision rules, stash nesting, projection."""

from types import SimpleNamespace

import mlx.core as mx
import pytest

from mlx_vlm.generate import ar

import gmlx.admit_gate as ag
import gmlx.batch_sched as batch_sched
import gmlx.server_memory as sm


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
    import gmlx.prefill_decay as pd

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


def test_projection_window_caps_rotating_kinds(monkeypatch):
    import gmlx.prefill_decay as pd

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
