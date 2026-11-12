# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Asher Feldman
"""A speaking row never requeues.

A requeue replays a row from its prompt ids and its delivered tokens. A
frame's token does not carry its codes, so the replay would feed the
wrong input, and a prompt without embeddings would replay with no error.
Five paths requeue a row. The orange retire passes over a speaking row,
and each of the other four fails it, closes its record, and requeues the
text rows beside it as before.

The scripted tests run the tick guard over a fake engine, as
tests/serve/test_tick_guard.py does. The last two inject a failure into
the real engine on a tiny LFM3."""

from __future__ import annotations

import types

import mlx.core as mx
import pytest

pytest.importorskip("gmlx._vendor.mlx_vlm")

import gmlx.eval_guard as eg  # noqa: E402
from gmlx._vendor.mlx_vlm.generate import ar  # noqa: E402
from gmlx.serve import governor  # noqa: E402
from gmlx.serve import speech_rows  # noqa: E402
from gmlx.serve import tick_guard as tg  # noqa: E402
from gmlx.serve.row_sampling import RowSettings  # noqa: E402
from tests.serve.test_tick_guard import (  # noqa: E402,F401
    MEM_ERR, OOM_ERR, TIMEOUT_ERR, FakeGB, _admitted, _broken, _droppable,
    _queue, _salvaged, _Spoiled, resp, tick, wrapped)

SPEAKS, WRITES = 7, 8
REFUSAL = "replay unsupported: a spoken reply cannot replay its frames"


def _fake_record() -> speech_rows.SpeechRow:
    state = types.SimpleNamespace(stream=object())
    record = speech_rows.SpeechRow(types.SimpleNamespace(), state, 1, 0)
    record.codes_next = record.embedding = object()
    record.slots.append(speech_rows.Slot("frame"))
    return record


def _closed(record) -> bool:
    return (record.closed and record.state.stream is None
            and record.codes_next is None and record.embedding is None
            and not record.slots)


@pytest.fixture
def rows(wrapped, monkeypatch):  # noqa: F811
    """The scripted engine with row 7 speaking and row 8 writing text.
    Yields ``(gen, script, record, failed)``."""
    gen, script, _drains = wrapped
    monkeypatch.setattr(tg.mx, "clear_cache", lambda: None)
    record = _fake_record()
    gen._row_settings = {SPEAKS: RowSettings(speech=record)}
    failed = []
    monkeypatch.setattr(tg, "_row_failed_callbacks",
                        [lambda uid, info: failed.append((uid, info))])
    return gen, script, record, failed


def _fails_alone(gen, record, failed) -> None:
    """Row 7 failed with the refusal and its record closed. Row 8 is
    back in the queue with what it delivered."""
    assert [uid for uid, _ in failed] == [SPEAKS]
    assert failed[0][1]["error"] == REFUSAL
    assert _closed(record)
    st = gen._kq_tick_guard
    assert SPEAKS not in st.ledger
    assert [entry[0] for entry in _queue(gen)] == [WRITES]


def test_a_failure_between_the_forward_and_the_sample_fails_a_speaking_row(
        rows):
    gen, script, record, failed = rows
    gen._generation_batch = FakeGB([SPEAKS, WRITES], [1, 1])
    _admitted(gen, script, resp(SPEAKS, 11), resp(WRITES, 21))
    script.append(_broken(MEM_ERR, decode=[SPEAKS, WRITES]))
    assert tick(gen) == ([], [])
    _fails_alone(gen, record, failed)
    assert _queue(gen) == [(WRITES, [4, 5, 6, 9, 9, 9, 21], 15)]


@pytest.mark.parametrize("err", [OOM_ERR, TIMEOUT_ERR],
                         ids=["command-buffer", "first-gpu-fault"])
def test_an_engine_drop_fails_a_speaking_row(rows, err):
    gen, script, record, failed = rows
    gen._generation_batch = FakeGB([SPEAKS, WRITES], [1, 1])
    _admitted(gen, script, resp(SPEAKS, 11), resp(WRITES, 21))
    gen._generation_batch = _Spoiled([SPEAKS, WRITES], [2, 2])
    _droppable(gen, [])
    script.append(_salvaged(err, resp(SPEAKS, 12)))
    assert tick(gen) == ([], [])
    _fails_alone(gen, record, failed)


def test_a_repeat_memory_error_fails_a_speaking_row_when_it_is_the_largest(
        rows):
    gen, script, record, failed = rows
    # Row 7 holds more tokens than row 8.
    gen._unprocessed_sequences[0] = (SPEAKS, list(range(40)), 64, {}, None,
                                     None)
    gen._generation_batch = FakeGB([SPEAKS, WRITES], [1, 1])
    _admitted(gen, script, resp(SPEAKS, 11), resp(WRITES, 21))
    script.append(MEM_ERR)
    assert tick(gen) == ([], [])
    assert not failed and not record.closed     # the first strike retries
    script.append(MEM_ERR)
    assert tick(gen) == ([], [])
    assert [uid for uid, _ in failed] == [SPEAKS]
    assert failed[0][1]["error"] == REFUSAL
    assert _closed(record)
    assert gen.removed == [SPEAKS]
    # The text row stays in the batch, and nothing is requeued.
    assert gen._generation_batch.uids == [WRITES] and _queue(gen) == []


def test_the_orange_retire_passes_over_a_speaking_row(rows, monkeypatch):
    gen, script, record, failed = rows
    # Row 7 holds more tokens than row 8, so it is the largest live row.
    gen._unprocessed_sequences[0] = (SPEAKS, list(range(40)), 64, {}, None,
                                     None)
    gen._generation_batch = FakeGB([SPEAKS, WRITES], [1, 1])
    _admitted(gen, script, resp(SPEAKS, 11), resp(WRITES, 21))
    assert tg._pick_victim(gen, gen._kq_tick_guard) == SPEAKS

    retired = []
    monkeypatch.setattr(tg, "_rebuild_row",
                        lambda g, st, uid: retired.append(uid) or True)
    monkeypatch.setattr(governor, "_shed_transient_bytes", lambda g, st: 0.0)
    assert governor._retire_largest(gen, governor._state(gen)) is governor.RetireOutcome.RETIRED
    assert retired == [WRITES]
    assert not failed and not record.closed

    # With only a speaking row live there is no row to retire.
    gen._generation_batch = FakeGB([SPEAKS], [1])
    assert governor._retire_largest(gen, governor._state(gen)) is governor.RetireOutcome.NONE
    assert retired == [WRITES]


def test_a_prompt_without_embeddings_would_replay_with_no_error(rows):
    """Why the refusal is explicit: the replay kwargs of a text-only
    prompt come back unchanged, with no error to stop the requeue."""
    gen, script, record, failed = rows
    row = tg._Row([1, 2, 3], 16, {}, None, None)
    row.committed = [11, 12]
    assert tg._replay_kwargs(gen, SPEAKS, row) == {}
    assert tg.speaks(gen, SPEAKS) and not tg.speaks(gen, WRITES)
    record.close()
    assert tg.speaks(gen, SPEAKS)       # a closed record still marks it


# -- the real engine, with an injected failure --------------------------------


@pytest.fixture
def engine(monkeypatch):
    """A guarded engine over the tiny LFM3 of tests/serve/test_speech_rows."""
    from gmlx._vendor.mlx_vlm.utils import StoppingCriteria
    from gmlx.upstream.patching import root
    from tests.models.test_lfm3_speak import (
        AUDIO_IN, AUDIO_OUT, AUDIO_START, END, IMAGE, MIXED_START, VIDEO,
        _model)

    monkeypatch.setenv("GMLX_PREFILL_DECAY", "0")
    monkeypatch.setenv("GMLX_CB_PHASE", "0")
    monkeypatch.delenv("GMLX_TICK_GUARD", raising=False)
    monkeypatch.delenv("GMLX_OOM_INJECT", raising=False)
    # A directive fires once in a process, so each test starts clean.
    monkeypatch.setattr(tg, "_INJECT_FIRED", set())
    monkeypatch.setattr(eg, "drain_for", lambda site: None)
    monkeypatch.setattr(ar.BatchGenerator, "_next",
                        root(ar.BatchGenerator._next))
    assert tg.install_tick_guard()
    failed = []
    monkeypatch.setattr(tg, "_row_failed_callbacks",
                        [lambda uid, info: failed.append((uid, info))])
    model = _model()
    model.language_model.set_suppressed_tokens(
        [IMAGE, VIDEO, AUDIO_IN, AUDIO_OUT, AUDIO_START, MIXED_START])
    depth = model.depth_decoder
    sample = depth.sample_frame
    monkeypatch.setattr(
        depth, "sample_frame",
        lambda hidden, key, seen=None, **kw: mx.where(
            sample(hidden, key, seen, **kw) == depth.config.eos_code, 0,
            sample(hidden, key, seen, **kw)))
    engines = []

    def make():
        from gmlx.serve.replay_prompt import embed_text

        processor = types.SimpleNamespace(tokenizer=types.SimpleNamespace(
            stopping_criteria=StoppingCriteria([END], None)))
        gen = ar.BatchGenerator(model.language_model, processor,
                                default_row=RowSettings(),
                                prefill_step_size=None)
        gen._kq_embed_text = lambda ids, rs: embed_text(model, ids, rs)
        engines.append(gen)
        return gen

    yield model, make, failed
    for gen in engines:
        gen.close()


def _embeds(model, prompt) -> dict:
    lm = model.language_model
    return {"inputs_embeds": lm.model.embed_tokens(mx.array([prompt]))}


def _run(model, make, inject, monkeypatch):
    """A speaking row and a text row to the end, with ``inject`` set
    from the fourth tick. Returns the text row's tokens, the speaking
    row's responses and its record."""
    text_prompt, speech_prompt = [7, 8, 9, 10, 11, 12], list(range(1, 15))
    gen = make()
    record = speech_rows.open_row(
        model, speech_prompt[-1], {"mode": "audio"}, 3, end_of_turn=())
    (speaks,) = gen.insert(
        [speech_prompt], max_tokens=41,
        prompt_kwargs=[_embeds(model, speech_prompt)],
        row_settings=[RowSettings(speech=record)])
    (writes,) = gen.insert([text_prompt], max_tokens=14,
                           prompt_kwargs=[_embeds(model, text_prompt)])
    out = {speaks: [], writes: []}
    ticks = 0
    while gen.has_work and ticks < 200:
        ticks += 1
        if ticks == 4 and inject:
            monkeypatch.setenv("GMLX_OOM_INJECT", inject)
        for r in gen.next()[1]:
            out[r.uid].append(r)
    assert not gen.has_work
    return [int(r.token) for r in out[writes]], out[speaks], record, speaks


@pytest.mark.parametrize("inject", ["cbfail@6", "throw@6,throw@7"],
                         ids=["command-buffer", "repeat-memory-error"])
def test_an_injected_failure_ends_the_spoken_reply_and_spares_the_text(
        engine, monkeypatch, inject):
    model, make, failed = engine
    alone, spoken, record, _ = _run(model, make, None, monkeypatch)
    assert not failed and spoken[-1].finish_reason == "length"
    assert len(alone) > 6       # the text row is live when the failure comes

    tokens, spoken, record, speaks = _run(model, make, inject, monkeypatch)
    assert [uid for uid, _ in failed] == [speaks]
    assert failed[0][1]["error"] == REFUSAL
    assert _closed(record)
    # The spoken reply stops where the failure came, with no finish.
    assert spoken and all(r.finish_reason is None for r in spoken)
    assert len(spoken) < 41
    # The text row writes what it writes with no failure.
    assert tokens == alone


# -- the prefix cache ---------------------------------------------------------


def test_a_speaking_row_leaves_no_entry_in_the_prefix_cache():
    """A row's retirement entry is keyed by its token ids. A frame's
    token does not describe what its step fed, so a spoken reply's
    entry would answer a later prompt with the wrong KV."""
    from gmlx.spec import ckpt

    spoken, text = {"mode": "exact", "gen": []}, {"mode": "exact", "gen": []}
    gb = types.SimpleNamespace(
        uids=[SPEAKS, WRITES], prompt_cache=[types.SimpleNamespace()],
        _kq_apc_retire_rows={SPEAKS: spoken, WRITES: text},
        _row_settings={SPEAKS: RowSettings(speech=_fake_record())})
    ckpt._plain_step_tick(gb, ([5, 9], None, None, None))
    assert gb._kq_apc_retire_rows == {WRITES: text}
    assert text["gen"] == [9] and spoken["gen"] == []
