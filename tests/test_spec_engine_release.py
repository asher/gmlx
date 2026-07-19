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

from gmlx.spec_engine import (
    _OWNED_MTP_ROUND_FLAG,
    _RELEASED_FLAG,
    install_continuous_batch_admission,
    install_owned_spec_engine,
)


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
    from mlx_vlm.generate import ar

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
    from mlx_vlm.generate import ar

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
    from mlx_vlm.generate import ar

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
    from mlx_vlm.generate import ar

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
    from mlx_vlm.generate import ar
    from mlx_vlm.server import generation as gen_mod
    from gmlx import speculative as spec

    def fake_inner(model, drafter, prompt_cache, hidden, **kw):
        del model, drafter, prompt_cache, hidden, kw  # inner pins nothing
        for t in (11, 12):
            yield [t], None

    def stock(*a, **kw):  # unflagged stand-in so install re-wraps
        yield [0], None

    monkeypatch.setattr(spec, "owned_server_rounds", fake_inner)
    monkeypatch.setattr(ar, "run_speculative_server_rounds", stock)
    monkeypatch.setattr(gen_mod, "run_speculative_server_rounds", stock)
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


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
