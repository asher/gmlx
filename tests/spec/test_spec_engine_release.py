#!/usr/bin/env python3
"""SpecBatch finish-release: a finished or aborted speculative batch must not
pin request state while parked in ``BatchGenerator._generation_batch``.

The engine only replaces the parked batch at the NEXT request's
prefill->decode seam, so anything the batch (or its abandoned rounds
generator) still references -- target KV, captured full-prompt hidden,
prefill shared-KV, the drafter's own head KV -- survives that request's
entire prefill. At deep context that stacks two requests' footprints
(~65 GB for ~18 minutes at d200k on gemma-4-31b) and runs the box into
the wired ceiling.
"""
from __future__ import annotations

import weakref

import pytest

import mlx.core as mx

from gmlx.spec.admission import (
    _RELEASED_FLAG,
    install_continuous_batch_admission,
)
from gmlx.spec.engine import _OWNED_MTP_ROUND_FLAG, install_owned_spec_engine


class _Entry:
    """Weakref-able stand-in for a prompt_cache entry."""


class _Shared:
    """Weakref-able stand-in for prefill shared-KV state."""


class _Drafter:
    def __init__(self):
        self.reset_calls = 0

    def reset(self, model, left_padding=None):
        self.reset_calls += 1
        return []


def _make_batch(ar, *, max_tokens=3, uids=(0,), drafter=None):
    batch = ar.SpeculativeGenerationBatch(
        model=object(),
        draft_model=drafter if drafter is not None else _Drafter(),
        draft_kind="mtp",
        uids=list(uids),
        first_tokens=mx.array([5] * len(uids)),
        prompt_cache=[_Entry()],
        sampler=None,
        stop_criteria=lambda tok: False,
        max_tokens=[max_tokens] * len(uids),
        hidden=mx.zeros((1, 4, 8)),
        shared_kv_states=_Shared(),
        prompt_tokens=mx.array([[1, 2, 3]]),
        greedy_sampling=True,
    )
    return batch


def _drain_heavy_attrs_assert_released(batch, closed):
    assert getattr(batch, _RELEASED_FLAG, False) is True
    assert batch.prompt_cache == []
    assert batch.hidden is None
    assert batch.shared_kv_states is None
    assert batch.prompt_tokens is None
    assert batch.first_tokens is None
    assert batch._rounds_iter is None
    assert closed == [True]


def test_finished_batch_releases_heavy_state(monkeypatch):
    from gmlx._vendor.mlx_vlm.generate import ar

    install_continuous_batch_admission()
    closed = []

    def fake_rounds(model, draft_model, prompt_cache, hidden, **kw):
        try:
            for t in (11, 12, 13):
                yield [t], None
        finally:
            closed.append(True)

    monkeypatch.setattr(ar, "run_speculative_server_rounds", fake_rounds)

    drafter = _Drafter()
    batch = _make_batch(ar, max_tokens=3, drafter=drafter)
    ref_entry = weakref.ref(batch.prompt_cache[0])
    ref_shared = weakref.ref(batch.shared_kv_states)

    toks = [r.token for r in batch.next()]  # first token
    assert toks == [5]
    assert [r.token for r in batch.next()] == [11]
    final = batch.next()  # token 12 -> length finish
    assert final[-1].finish_reason == "length"

    _drain_heavy_attrs_assert_released(batch, closed)
    assert drafter.reset_calls == 1
    # no other refs anywhere: the request state is actually freed
    assert ref_entry() is None, "finished spec batch still pins the request KV"
    assert ref_shared() is None, "finished spec batch still pins shared-KV"


def test_aborted_batch_releases_via_filter(monkeypatch):
    from gmlx._vendor.mlx_vlm.generate import ar

    install_continuous_batch_admission()
    closed = []

    def fake_rounds(model, draft_model, prompt_cache, hidden, **kw):
        try:
            for t in (11, 12, 13):
                yield [t], None
        finally:
            closed.append(True)

    monkeypatch.setattr(ar, "run_speculative_server_rounds", fake_rounds)

    batch = _make_batch(ar, max_tokens=100)
    ref_entry = weakref.ref(batch.prompt_cache[0])
    batch.next()  # first token
    batch.next()  # one round; generator now suspended
    batch.filter([])  # client abort: no rows kept

    _drain_heavy_attrs_assert_released(batch, closed)
    assert ref_entry() is None


def test_abort_inside_round_defers_then_releases(monkeypatch):
    # An abort that lands while the rounds generator is executing (filter
    # re-entered from within a round step) cannot close it there; the
    # release must defer, not crash, and complete once the round yields.
    from gmlx._vendor.mlx_vlm.generate import ar

    install_continuous_batch_admission()
    closed = []
    box = {}

    def fake_rounds(model, draft_model, prompt_cache, hidden, **kw):
        try:
            yield [11], None
            box["batch"].filter([])  # abort races the running round
            yield [12], None
        finally:
            closed.append(True)

    monkeypatch.setattr(ar, "run_speculative_server_rounds", fake_rounds)

    batch = _make_batch(ar, max_tokens=100)
    box["batch"] = batch
    batch.next()  # first token
    batch.next()  # round 1
    batch.next()  # round 2: filter fires mid-step, release defers to exit

    _drain_heavy_attrs_assert_released(batch, closed)


def test_promotion_clears_release_flag(monkeypatch):
    from gmlx._vendor.mlx_vlm.generate import ar

    install_continuous_batch_admission()

    def fake_rounds(model, draft_model, prompt_cache, hidden, **kw):
        yield [11], None

    monkeypatch.setattr(ar, "run_speculative_server_rounds", fake_rounds)

    batch = _make_batch(ar, max_tokens=2)
    batch.next()
    batch.next()  # finishes + releases
    assert getattr(batch, _RELEASED_FLAG) is True

    fresh = _make_batch(ar, max_tokens=5, uids=(1,))
    batch._pending_injections = [fresh]
    assert len(batch) == 1  # promotion adopted the pending batch
    assert getattr(batch, _RELEASED_FLAG) is False
    assert batch.prompt_cache == fresh.prompt_cache


def test_owned_round_wrapper_frame_pins_nothing(monkeypatch):
    # The delegation wrapper installed by install_owned_spec_engine stays
    # suspended at its yield-from for the life of the abandoned generator;
    # its argument bindings must not re-pin what the inner loop released.
    from gmlx._vendor.mlx_vlm.generate import ar
    import gmlx.spec.speculative as spec

    def fake_inner(model, drafter, prompt_cache, hidden, **kw):
        del model, drafter, prompt_cache, hidden, kw  # inner pins nothing
        for t in (11, 12):
            yield [t], None

    def stock(*a, **kw):  # unflagged stand-in so install re-wraps
        yield [0], None

    monkeypatch.setattr(spec, "owned_server_rounds", fake_inner)
    monkeypatch.setattr(ar, "run_speculative_server_rounds", stock)
    install_owned_spec_engine()
    wrapper = ar.run_speculative_server_rounds
    assert wrapper.__dict__.get(_OWNED_MTP_ROUND_FLAG)

    entry = _Entry()
    shared = _Shared()
    ref_entry = weakref.ref(entry)
    ref_shared = weakref.ref(shared)
    gen = wrapper(
        object(), _Drafter(), [entry], mx.zeros((1, 4, 8)),
        draft_kind="mtp", first_bonus=mx.array([5]), max_tokens=4,
        sampler=None, shared_kv_states=shared,
        prompt_tokens=mx.array([[1, 2, 3]]),
    )
    assert next(gen) == ([11], None)  # frame ran: args created + deleted

    del entry, shared  # generator deliberately left open (abandoned)
    assert ref_entry() is None, "wrapper frame still pins the request cache"
    assert ref_shared() is None, "wrapper frame still pins shared-KV"
    gen.close()


# --- per-row release (U3 prerequisite) ---------------------------------


class _FilterCache:
    """Batch-class stand-in: records filter keep lists."""

    def __init__(self):
        self.filtered = []

    def filter(self, idx):
        self.filtered.append([int(i) for i in idx.tolist()])

    def extend(self, other):
        pass


def _make_dict_batch(ar, *, uids, max_tokens, cache=None):
    B = len(uids)
    return ar.SpeculativeGenerationBatch(
        model=object(),
        draft_model=_Drafter(),
        draft_kind="mtp",
        uids=list(uids),
        first_tokens=mx.array([5] * B),
        prompt_cache=[cache if cache is not None else _Entry()],
        sampler=None,
        stop_criteria=lambda tok: False,
        max_tokens=list(max_tokens),
        hidden=mx.zeros((B, 1, 8)),
        shared_kv_states={"l0": (mx.zeros((B, 2, 4)), mx.ones((B, 2, 4)))},
        prompt_tokens=mx.zeros((B, 3), dtype=mx.int32),
        greedy_sampling=True,
    )


def test_live_filter_sheds_snapshot_rows(monkeypatch):
    # A row removed while the rounds generator runs keeps the upstream
    # mark-finished contract (the generator sheds its KV at the round
    # boundary) but the batch-held start-time snapshots must slice now.
    from gmlx._vendor.mlx_vlm.generate import ar

    install_continuous_batch_admission()

    def fake_rounds(model, draft_model, prompt_cache, hidden, **kw):
        while True:
            yield [11, 12, 13], None

    monkeypatch.setattr(ar, "run_speculative_server_rounds", fake_rounds)
    batch = _make_dict_batch(ar, uids=(0, 1, 2), max_tokens=(100,) * 3)
    cache = batch.prompt_cache[0]
    batch.next()  # first tokens
    batch.next()  # rounds started
    assert batch._rounds_iter is not None

    batch.filter([0, 2])  # drop uid 1 mid-flight

    assert batch.uids == [0, 2]
    assert batch._finished == [False, True, False]
    assert batch.first_tokens.shape == (2,)
    assert batch.hidden.shape == (2, 1, 8)
    assert batch.prompt_tokens.shape == (2, 3)
    K, V = batch.shared_kv_states["l0"]
    assert K.shape == (2, 2, 4) and V.shape == (2, 2, 4)
    assert batch._kq_attr_rows == [0, 2]
    assert batch.prompt_cache[0] is cache  # generator-owned; untouched


def test_natural_finish_sheds_snapshot_rows(monkeypatch):
    from gmlx._vendor.mlx_vlm.generate import ar

    install_continuous_batch_admission()

    def fake_rounds(model, draft_model, prompt_cache, hidden, **kw):
        while True:
            yield [11, 21], None

    monkeypatch.setattr(ar, "run_speculative_server_rounds", fake_rounds)
    batch = _make_dict_batch(ar, uids=(0, 1), max_tokens=(100, 2))
    batch.next()  # first tokens: row 1 at 1/2
    out = batch.next()  # row 1 hits length
    assert any(r.uid == 1 and r.finish_reason == "length" for r in out)

    assert batch.first_tokens.shape == (1,)
    assert batch.hidden.shape == (1, 1, 8)
    K, _ = batch.shared_kv_states["l0"]
    assert K.shape == (1, 2, 4)
    assert batch._kq_attr_rows == [0]


def test_first_token_finish_defers_shed_until_rounds_start(monkeypatch):
    # Pre-start, _start_rounds still needs snapshots row-aligned with
    # the caches; the shed must wait for the generator to own the state.
    from gmlx._vendor.mlx_vlm.generate import ar

    install_continuous_batch_admission()

    def fake_rounds(model, draft_model, prompt_cache, hidden, **kw):
        while True:
            yield [11, 21], None

    monkeypatch.setattr(ar, "run_speculative_server_rounds", fake_rounds)
    batch = _make_dict_batch(ar, uids=(0, 1), max_tokens=(100, 1))
    out = batch.next()  # row 1 finishes on its first token
    assert any(r.uid == 1 and r.finish_reason == "length" for r in out)
    assert batch.first_tokens.shape == (2,)  # deferred: rounds not started

    batch.next()  # rounds start; shed catches up
    assert batch.first_tokens.shape == (1,)
    assert batch._kq_attr_rows == [0]


def test_prestart_filter_compacts_rows(monkeypatch):
    from gmlx._vendor.mlx_vlm.generate import ar

    install_continuous_batch_admission()
    cache = _FilterCache()
    batch = _make_dict_batch(ar, uids=(0, 1, 2), max_tokens=(9, 9, 9),
                             cache=cache)
    batch._kq_row_procs = ["g0", None, "g2"]

    batch.filter([0, 2])  # cancel uid 1 before the first tick

    assert cache.filtered == [[0, 2]]
    assert batch._all_uids == [0, 2] and batch.uids == [0, 2]
    assert batch._finished == [False, False]
    assert batch._num_tokens == [0, 0]
    assert batch.max_tokens == [9, 9]
    assert batch._kq_row_procs == ["g0", "g2"]
    assert batch.first_tokens.shape == (2,)
    assert batch.hidden.shape == (2, 1, 8)
    K, V = batch.shared_kv_states["l0"]
    assert K.shape == (2, 2, 4) and V.shape == (2, 2, 4)
    assert len(batch) == 2


def test_prestart_filter_compacts_copy_prompts():
    """A cancelled pre-start row takes its context-copy prompt with it, so
    the surviving row copies from its own prompt."""
    from gmlx._vendor.mlx_vlm.generate import ar

    install_continuous_batch_admission()
    cache = _FilterCache()
    batch = _make_dict_batch(ar, uids=(0, 1), max_tokens=(9, 9),
                             cache=cache)
    batch._kq_copy_prompts = [[1, 2, 3], [7, 8, 9]]

    batch.filter([1])  # cancel uid 0 before the first tick

    assert batch._all_uids == [1]
    assert batch._kq_copy_prompts == [[7, 8, 9]]


def test_prestart_abort_releases_fully(monkeypatch):
    from gmlx._vendor.mlx_vlm.generate import ar

    install_continuous_batch_admission()
    batch = _make_dict_batch(ar, uids=(0, 1), max_tokens=(9, 9))
    batch.filter([])

    assert getattr(batch, _RELEASED_FLAG, False) is True
    assert batch.prompt_cache == []
    assert batch.hidden is None
    assert batch.shared_kv_states is None
    assert batch.first_tokens is None


def test_promotion_drops_stale_attr_rows(monkeypatch):
    from gmlx._vendor.mlx_vlm.generate import ar

    install_continuous_batch_admission()

    def fake_rounds(model, draft_model, prompt_cache, hidden, **kw):
        yield [11], None

    monkeypatch.setattr(ar, "run_speculative_server_rounds", fake_rounds)
    batch = _make_batch(ar, max_tokens=2)
    batch.next()
    batch.next()  # finishes + releases
    batch.__dict__["_kq_attr_rows"] = [7]  # stale survivor

    fresh = _make_batch(ar, max_tokens=5, uids=(1,))
    batch._pending_injections = [fresh]
    assert len(batch) == 1  # promotion adopts the pending batch
    assert "_kq_attr_rows" not in batch.__dict__


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))


_MEM = "[metal::malloc] Attempting to allocate 1 bytes (test)"


def _guarded(monkeypatch, batch, ledger, *, filtering=False):
    """The tick guard over a _next that steps ``batch``. With
    ``filtering`` a remove filters the row out of the batch, as
    BatchGenerator.remove does."""
    import types

    import gmlx.eval_guard as eg
    import gmlx.serve.tick_guard as tg
    from gmlx._vendor.mlx_vlm.generate import ar

    monkeypatch.delenv("GMLX_TICK_GUARD", raising=False)
    monkeypatch.delenv("GMLX_OOM_INJECT", raising=False)
    monkeypatch.setattr(eg, "drain_for", lambda site: None)
    monkeypatch.setattr(ar.BatchGenerator, "_next",
                        lambda self, **kw: ([], self._generation_batch.next()))
    assert tg.install_tick_guard()
    gen = types.SimpleNamespace(_unprocessed_sequences=[],
                                _generation_batch=batch, removed=[])
    def remove(uid):
        gen.removed.append(uid)
        gb = gen._generation_batch
        if filtering and uid in gb.uids:
            gb.filter([i for i, u in enumerate(gb.uids) if u != uid])

    gen.remove = remove
    st = gen._kq_tick_guard = tg._TickState()
    for uid, prompt in ledger.items():
        st.ledger[uid] = tg._Row(prompt, 64, {}, None, None)

    def tick():
        return ar.BatchGenerator._next(gen)[1]
    return gen, tick


def test_a_failed_first_token_eval_sends_the_first_tokens_next(monkeypatch):
    """M1: the first-token eval throws. The first tokens go out once, on
    the next call."""
    from gmlx._vendor.mlx_vlm.generate import ar

    install_continuous_batch_admission()

    def fake_rounds(model, draft_model, prompt_cache, hidden, **kw):
        for t in (11, 12, 13):
            yield [t, t + 100], None

    monkeypatch.setattr(ar, "run_speculative_server_rounds", fake_rounds)
    batch = _make_batch(ar, max_tokens=8, uids=(0, 1))
    calls = []
    orig = batch._first_entries

    def first_entries():
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError(_MEM)
        return orig()

    batch._first_entries = first_entries
    gen, tick = _guarded(monkeypatch, batch, {0: [1, 2], 1: [3, 4]})
    assert tick() == []
    sent = [(r.uid, r.token) for _ in range(3) for r in tick()]
    assert sent == [(0, 5), (1, 5), (0, 11), (1, 111), (0, 12), (1, 112)]
    st = gen._kq_tick_guard
    assert st.ledger[0].committed == [5, 11, 12]


def test_injected_first_tokens_go_out_on_a_failed_round(monkeypatch):
    """A2: the round after an injection throws. The injected rows' first
    tokens reach the caller in that tick, and the injection is not
    repeated."""
    from gmlx._vendor.mlx_vlm.generate import ar

    install_continuous_batch_admission()
    rounds = []

    def fake_rounds(model, draft_model, prompt_cache, hidden, **kw):
        n = 0
        while True:
            n += 1
            rounds.append(n)
            if n == 2:
                raise RuntimeError(_MEM)
            yield [20 + n, 30 + n], None

    monkeypatch.setattr(ar, "run_speculative_server_rounds", fake_rounds)
    host = _make_batch(ar, max_tokens=8, uids=(0, 1))
    other = _make_batch(ar, max_tokens=8, uids=(2,))
    other.first_tokens = mx.array([7])
    host.model = type("M", (), {})()
    gen, tick = _guarded(monkeypatch, host,
                         {0: [1], 1: [2], 2: [3]})
    assert [(r.uid, r.token) for r in tick()] == [(0, 5), (1, 5)]
    assert [(r.uid, r.token) for r in tick()] == [(0, 21), (1, 31)]
    host.extend(other)
    assert [(r.uid, r.token) for r in tick()] == [(2, 7)]
    assert host._all_uids == [0, 1, 2]
    assert host._pending_injections == []
    # The failed round took the generator and its queued injection with
    # it: every unfinished row is rebuilt from what it delivered.
    assert host.model._generator_injections == []
    assert gen.removed == [0, 1, 2]
    assert [(e[0], e[1]) for e in gen._unprocessed_sequences] == [
        (0, [1, 5, 21]), (1, [2, 5, 31]), (2, [3, 7])]


def test_a_joiners_own_caches_go_at_the_join(monkeypatch):
    """The live batch copies a joiner's rows, and the rounds drop the
    injection that carried the joiner's caches. Nothing else holds them
    through the step that follows the join."""
    from gmlx._vendor.mlx_vlm.generate import ar

    install_continuous_batch_admission()
    held = []

    def fake_rounds(model, draft_model, prompt_cache, hidden, **kw):
        rows, n = 2, 0
        while True:
            n += 1
            queue = getattr(model, "_generator_injections", None)
            if queue:
                rows += sum(len(inj["uids"]) for inj in queue)
                queue.clear()   # the drain, once the rows are joined
                held.append(entry() is not None)
            yield [20 + n] * rows, None

    monkeypatch.setattr(ar, "run_speculative_server_rounds", fake_rounds)
    host = _make_batch(ar, max_tokens=8, uids=(0, 1))
    other = _make_batch(ar, max_tokens=8, uids=(2,))
    entry = weakref.ref(other.prompt_cache[0])
    host.model = type("M", (), {})()
    host.next()
    host.next()
    host.extend(other)
    host.next()
    assert host._all_uids == [0, 1, 2]
    assert held == [False]


def _queue(gen):
    return [(e[0], e[1]) for e in gen._unprocessed_sequences]


def test_a_failed_round_requeues_every_unfinished_row(monkeypatch):
    """Q3: the round generator dies in its second round. No row ends with
    "length"; every unfinished row is requeued with what it delivered."""
    from gmlx._vendor.mlx_vlm.generate import ar

    install_continuous_batch_admission()

    def fake_rounds(model, draft_model, prompt_cache, hidden, **kw):
        yield [21, 31], None
        raise RuntimeError(_MEM)

    monkeypatch.setattr(ar, "run_speculative_server_rounds", fake_rounds)
    batch = _make_batch(ar, max_tokens=8, uids=(0, 1))
    batch.model = type("M", (), {})()
    batch.model._generator_injections = []
    gen, tick = _guarded(monkeypatch, batch, {0: [1], 1: [2]}, filtering=True)
    seen = tick() + tick()
    assert [(r.uid, r.token) for r in seen] == [(0, 5), (1, 5), (0, 21), (1, 31)]
    assert tick() == []
    assert batch._kq_rounds_dead
    assert gen.removed == [0, 1]
    assert _queue(gen) == [(0, [1, 5, 21]), (1, [2, 5, 31])]
    assert tick() == []
    assert not any(r.finish_reason for r in seen)


def test_a_failure_on_a_second_injected_batch_keeps_the_first_live(
        monkeypatch):
    """Each injected batch's rows are live once it joins, so a failure on
    the next one leaves them where the guard can see them."""
    from gmlx._vendor.mlx_vlm.generate import ar

    install_continuous_batch_admission()

    def fake_rounds(model, draft_model, prompt_cache, hidden, **kw):
        while True:
            yield [21, 31], None

    monkeypatch.setattr(ar, "run_speculative_server_rounds", fake_rounds)
    host = _make_batch(ar, max_tokens=8, uids=(0, 1))
    host.model = type("M", (), {})()
    first = _make_batch(ar, max_tokens=8, uids=(2,))
    second = _make_batch(ar, max_tokens=8, uids=(3,))
    gen, tick = _guarded(monkeypatch, host,
                         {0: [1], 1: [2], 2: [3], 3: [4]})
    tick()
    tick()
    host.extend(first)
    host.extend(second)

    def broken():
        raise RuntimeError(_MEM)

    second._first_entries = broken
    out = tick()
    assert [(r.uid, r.token) for r in out] == [(2, 5)]
    assert host.uids == [0, 1, 2]
    assert host._pending_injections == [second]


_CB_OOM = ("[METAL] Command buffer execution failed: Insufficient Memory "
           "(00000008:kIOGPUCommandBufferCallbackErrorOutOfMemory).")


def test_a_command_buffer_failure_drops_a_speculative_batch(monkeypatch):
    """A4: the failure reports at a wait after a round. The batch is
    released without another round or a filter, its rounds generator
    closes with retirement stores blocked, and every row requeues."""
    from gmlx._vendor.mlx_vlm.generate import ar
    from gmlx.cache import snapshot

    install_continuous_batch_admission()
    rounds, closed = [], []

    def fake_rounds(model, draft_model, prompt_cache, hidden, **kw):
        try:
            while True:
                rounds.append(1)
                yield [20 + len(rounds), 30 + len(rounds)], None
        finally:
            closed.append(snapshot._RETIRE_BLOCKED[0])

    monkeypatch.setattr(ar, "run_speculative_server_rounds", fake_rounds)
    batch = _make_batch(ar, max_tokens=8, uids=(0, 1))
    batch.model = type("M", (), {})()
    batch.model._generator_injections = []
    gen, tick = _guarded(monkeypatch, batch, {0: [1], 1: [2]})
    gen._prompt_batch = None
    gen._kq_empty_generation_batch = lambda: setattr(
        gen, "_generation_batch", _make_batch(ar, uids=()))
    assert [(r.uid, r.token) for r in tick() + tick()] == [
        (0, 5), (1, 5), (0, 21), (1, 31)]
    step = batch.next

    def reported_late():
        step()
        raise RuntimeError(_CB_OOM)

    def no_filter(keep):
        raise AssertionError("filtered a spoiled batch")

    batch.next = reported_late
    batch.filter = no_filter
    assert tick() == []
    assert len(rounds) == 2                     # no round ran after it
    assert closed == [1]                        # closed with stores blocked
    assert getattr(batch, _RELEASED_FLAG) is True
    assert gen._generation_batch is not batch
    assert gen.removed == []
    assert _queue(gen) == [(0, [1, 5, 21]), (1, [2, 5, 31])]
    assert snapshot._RETIRE_BLOCKED[0] == 0
