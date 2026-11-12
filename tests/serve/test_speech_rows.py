# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Asher Feldman
"""Speaking rows in the batch engine, on a tiny LFM3 with a speech
generator and random weights.

Each test drives ``BatchGenerator`` and compares it with a loop written
out here: one row, one slot a step, the reply state deciding the kind of
each slot. A frame's key comes from the row's seed and its slot index, so
the loop derives the same keys."""

from __future__ import annotations

import types

import mlx.core as mx
import numpy as np
import pytest

pytest.importorskip("gmlx._vendor.mlx_vlm")

from gmlx._vendor.mlx_vlm.utils import StoppingCriteria  # noqa: E402
from gmlx._vendor.mlx_vlm.generate import ar  # noqa: E402
from gmlx._vendor.mlx_vlm.models import cache as vlm_cache  # noqa: E402

from gmlx.serve import speech_rows  # noqa: E402
from gmlx.serve.row_sampling import RowSettings  # noqa: E402
from tests.models.test_lfm3_speak import (  # noqa: E402
    AUDIO_IN, AUDIO_OUT, AUDIO_START, CODES, END, IMAGE, MIXED_START,
    TEXT_END, VIDEO, _model)

PROMPT = [1, 2, 3, 4, 5]
OTHER = [7, 8, 9, 10, 11, 12]
FRAME_SAMPLES = 96


@pytest.fixture(scope="module")
def model():
    model = _model()
    model.language_model.set_suppressed_tokens(
        [IMAGE, VIDEO, AUDIO_IN, AUDIO_OUT, AUDIO_START, MIXED_START])
    return model


_ENGINES: list = []


@pytest.fixture(autouse=True)
def _close_engines():
    yield
    while _ENGINES:
        _ENGINES.pop().close()


@pytest.fixture
def cpu():
    """The CPU stream, for a test that holds the engine to the loop
    exactly. M5 GPUs run float32 matmuls at TF32 precision, where the
    batch and the one-row loop round apart and a sampled code near a tie
    flips. The engine takes the stream too: its default stream, made at
    import, is the GPU's even when the tests run on the CPU device."""
    stream = mx.default_stream(mx.cpu)
    with mx.stream(stream):
        yield stream


def _engine(model, **kw):
    processor = types.SimpleNamespace(tokenizer=types.SimpleNamespace(
        stopping_criteria=StoppingCriteria([END], None)))
    kw.setdefault("prefill_step_size", None)
    gen = ar.BatchGenerator(
        model.language_model, processor, default_row=RowSettings(), **kw)
    _ENGINES.append(gen)
    return gen


def _kw(model, prompt) -> dict:
    lm = model.language_model
    return {"inputs_embeds": lm.model.embed_tokens(mx.array([prompt]))}


def _record(model, prompt, mode, seed=3, stop_at_end=False):
    return speech_rows.open_row(
        model, prompt[-1], {"mode": mode, "stop_at_end": stop_at_end}, seed,
        end_of_turn={END})


def _speak(gen, model, prompt, mode, budget, seed=3, stop_at_end=False,
           **insert):
    """Queue a speaking row with ``budget`` slots after its marker."""
    record = _record(model, prompt, mode, seed, stop_at_end)
    (uid,) = gen.insert([prompt], max_tokens=budget + 1,
                        prompt_kwargs=[_kw(model, prompt)],
                        row_settings=[RowSettings(speech=record)], **insert)
    return uid, record


def _drain(gen, uids, steps=400, on_step=None):
    """Run the engine until every uid finished: ``{uid: [responses]}``."""
    got = {uid: [] for uid in uids}
    open_ = set(uids)
    for step in range(steps):
        if not open_:
            break
        _prompts, responses = gen.next()
        for r in responses:
            if r.uid in got:
                got[r.uid].append(r)
                if r.finish_reason is not None:
                    open_.discard(r.uid)
        if on_step is not None:
            on_step(step, got)
    assert not open_, f"rows {open_} did not finish"
    return got


def _never_ends(model, monkeypatch, end_after=None):
    """Keep the end code out of the frames. With ``end_after``, a row's
    frame holds it once the row's penalty history holds that many
    frames, so each row ends on its own count."""
    depth = model.depth_decoder
    sample = depth.sample_frame

    def patched(hidden, key, seen=None, **kw):
        codes = sample(hidden, key, seen, **kw)
        return mx.where(codes == CODES - 1, 0, codes)

    monkeypatch.setattr(depth, "sample_frame", patched)
    if end_after is None:
        return
    step = model.speech_step

    def ending(states, hidden, keys):
        codes, _embedding = step(states, hidden, keys)
        done = mx.array([[s.frames >= end_after] for s in states])
        codes = mx.where(done, CODES - 1, codes)
        return codes, depth.feedback_embedding(codes)[:, None, :]

    monkeypatch.setattr(model, "speech_step", ending)


def _loop(model, prompt, mode, budget, seed=3, stop_at_end=False):
    """The reply of one row, written out: ``(tokens, kinds, codes,
    audio, finish)``. ``tokens`` starts with the marker."""
    from gmlx._vendor.mlx_vlm.generate.ar import _position_seed

    lm = model.language_model
    cfg = model.config.speech_config
    state, marker = model.speech_start(prompt[-1], mode, end_of_turn={END})
    cache = vlm_cache.make_prompt_cache(lm)
    hidden = lm.model(mx.array([prompt + [marker]]), cache)[:, -1]
    tokens, kinds, codes, audio = [marker], ["marker"], [], []
    finish = "length"
    for slot in range(1, budget + 1):
        if state.wants_frame:
            key = mx.random.key(_position_seed(
                (seed ^ speech_rows.FRAME_SALT) & 0xFFFFFFFF, 0, slot))
            frame, embedding = model.speech_step([state], hidden, key[None])
            frame = np.asarray(frame)[0]
            tokens.append(cfg.audio_out_token_id)
            if state.take_frame(frame):
                kinds.append("end")
                audio.append(model.speech_flush(state))
                if stop_at_end:
                    state.finished = True
            else:
                kinds.append("frame")
                codes.append(frame.tolist())
                audio.append(model.speech_audio(state, frame))
            ids = mx.zeros((1, 1), dtype=mx.int32)
            hidden = lm.model(ids, cache, embedding)[:, -1]
        else:
            logits = model.speech_logits(hidden, [state.text_only])
            token = int(mx.argmax(logits, axis=-1).item())
            tokens.append(token)
            kinds.append("text")
            state.take_token(token)
            hidden = lm.model(mx.array([[token]]), cache)[:, -1]
        if state.finished:
            finish = "stop"
            break
    audio.append(model.speech_flush(state))
    return tokens, kinds, codes, _join(audio), finish


def _join(chunks) -> np.ndarray:
    chunks = [c for c in chunks if c is not None and c.size]
    return np.concatenate(chunks) if chunks else np.zeros(0, np.float32)


def _reply(responses, record):
    """The engine's reply in the loop's shape."""
    return ([int(r.token) for r in responses],
            [r.speech.kind for r in responses],
            [c.tolist() for c in record.codes],
            _join([r.speech.audio for r in responses]),
            responses[-1].finish_reason)


def _same(got, want) -> None:
    assert got[0] == want[0]
    assert got[1] == want[1]
    assert got[2] == want[2]
    assert got[3].shape == want[3].shape
    np.testing.assert_allclose(got[3], want[3], atol=1e-5)
    assert got[4] == want[4]


# -- one row ------------------------------------------------------------------


def test_a_text_to_speech_row_is_the_marker_then_frames_to_the_end_frame(
        model, monkeypatch, cpu):
    _never_ends(model, monkeypatch, end_after=5)
    gen = _engine(model, stream=cpu)
    uid, record = _speak(gen, model, PROMPT, "audio", 40, stop_at_end=True)
    responses = _drain(gen, [uid])[uid]
    tokens, kinds, codes, audio, finish = _reply(responses, record)
    assert tokens == [AUDIO_START] + [AUDIO_OUT] * 6
    assert kinds == ["marker"] + ["frame"] * 5 + ["end"]
    assert finish == "stop"
    assert len(codes) == 5 and audio.size == 5 * FRAME_SAMPLES
    assert all(r.speech.quiet for r in responses)
    _same((tokens, kinds, codes, audio, finish),
          _loop(model, PROMPT, "audio", 40, stop_at_end=True))
    # The audio is the whole clip's.
    whole = model.codec_decoder.decode(np.asarray(codes))
    np.testing.assert_allclose(audio, whole, atol=1e-4)


def test_an_interleaved_row_is_the_references_slot_pattern(
        model, monkeypatch, cpu):
    _never_ends(model, monkeypatch)
    gen = _engine(model, stream=cpu)
    uid, record = _speak(gen, model, PROMPT, "interleaved", 40)
    got = _reply(_drain(gen, [uid])[uid], record)
    want = _loop(model, PROMPT, "interleaved", 40)
    _same(got, want)
    kinds = "".join(k[0] for k in got[1])
    # The marker is not one of the first six text slots.
    assert kinds.startswith("m" + "t" * 6 + "f" * 12 + "t" * 6) or \
        TEXT_END in got[0] or END in got[0]
    assert got[4] in ("length", "stop")


def test_the_budget_counts_every_slot_after_the_marker(model, monkeypatch,
                                                    cpu):
    _never_ends(model, monkeypatch)
    gen = _engine(model, stream=cpu)
    uid, record = _speak(gen, model, PROMPT, "audio", 7)
    responses = _drain(gen, [uid])[uid]
    assert len(responses) == 1 + 7
    assert responses[-1].finish_reason == "length"
    assert [r.finish_reason for r in responses[:-1]] == [None] * 7
    # The last response carries the samples the codec held back.
    audio = _join([r.speech.audio for r in responses])
    assert audio.size == 7 * FRAME_SAMPLES
    assert responses[-1].speech.audio.size > FRAME_SAMPLES
    _same(_reply(responses, record), _loop(model, PROMPT, "audio", 7))


def test_a_finished_row_leaves_no_record(model, monkeypatch):
    _never_ends(model, monkeypatch)
    gen = _engine(model)
    uid, record = _speak(gen, model, PROMPT, "audio", 4)
    _drain(gen, [uid])
    assert record.closed and record.state.stream is None
    assert not record.slots and record.embedding is None
    assert uid not in gen._row_settings
    assert len(record.codes) == 4


def test_a_cancelled_row_closes_its_record(model, monkeypatch):
    _never_ends(model, monkeypatch)
    gen = _engine(model)
    uid, record = _speak(gen, model, PROMPT, "audio", 50)
    got = {uid: []}
    for _ in range(6):
        for r in gen.next()[1]:
            got[uid].append(r)
    assert got[uid] and record.state.stream is not None
    assert gen.remove(uid)
    assert record.closed
    assert record.state.stream is None and record.embedding is None
    assert record.codes_next is None and not record.slots
    # The row is gone, and the engine runs on.
    assert gen.next()[1] == []


# -- rows side by side --------------------------------------------------------


def _text_alone(model, prompt, budget, **kw):
    gen = _engine(model, **kw)
    (uid,) = gen.insert([prompt], max_tokens=budget,
                        prompt_kwargs=[_kw(model, prompt)])
    return [int(r.token) for r in _drain(gen, [uid])[uid]]


def test_a_text_row_beside_a_speaking_row_writes_what_it_writes_alone(
        model, monkeypatch, cpu):
    _never_ends(model, monkeypatch)
    alone = _text_alone(model, OTHER, 12, stream=cpu)
    gen = _engine(model, stream=cpu)
    speaking, record = _speak(gen, model, PROMPT, "interleaved", 30)
    (text,) = gen.insert([OTHER], max_tokens=12,
                         prompt_kwargs=[_kw(model, OTHER)])
    got = _drain(gen, [speaking, text])
    assert [int(r.token) for r in got[text]] == alone
    assert all(r.speech is None for r in got[text])
    # A text row never samples a mode marker or a placeholder.
    assert not {IMAGE, VIDEO, AUDIO_IN, AUDIO_OUT, AUDIO_START,
                MIXED_START} & set(alone)
    _same(_reply(got[speaking], record),
          _loop(model, PROMPT, "interleaved", 30))


def test_two_speaking_rows_each_speak_what_they_speak_alone(
        model, monkeypatch, cpu):
    """The second row joins while the first is mid-reply, and the first
    leaves before the second ends."""
    _never_ends(model, monkeypatch)
    gen = _engine(model, stream=cpu)
    first, record_a = _speak(gen, model, PROMPT, "audio", 9, seed=11)
    joined = {}

    def join(step, got):
        if step == 3 and not joined:
            joined["uid"], joined["record"] = _speak(
                gen, model, OTHER, "interleaved", 26, seed=12)
            got[joined["uid"]] = []

    got = _drain(gen, [first], on_step=join)
    rest = _drain(gen, [joined["uid"]])
    got[joined["uid"]] += rest[joined["uid"]]
    _same(_reply(got[first], record_a),
          _loop(model, PROMPT, "audio", 9, seed=11))
    _same(_reply(got[joined["uid"]], joined["record"]),
          _loop(model, OTHER, "interleaved", 26, seed=12))


def test_a_seed_fixes_a_rows_frames(model, monkeypatch):
    _never_ends(model, monkeypatch)

    def codes(seed):
        gen = _engine(model)
        uid, record = _speak(gen, model, PROMPT, "audio", 6, seed=seed)
        _drain(gen, [uid])
        return [c.tolist() for c in record.codes]

    assert codes(5) == codes(5)
    assert codes(5) != codes(6)


# -- what a speaking row's processors read ------------------------------------


def test_a_speaking_rows_processors_read_its_text_slots_only(
        model, monkeypatch):
    _never_ends(model, monkeypatch)
    seen = []

    def processor(tokens, logits):
        seen.append(tokens.tolist())
        return logits

    gen = _engine(model)
    uid, record = _speak(gen, model, PROMPT, "interleaved", 30,
                         logits_processors=[[processor]])
    responses = _drain(gen, [uid])[uid]
    text = [int(r.token) for r in responses if r.speech.kind == "text"]
    assert seen
    for context in seen:
        assert AUDIO_OUT not in context and MIXED_START not in context
    # A context is the prompt and then the text the row wrote, in order.
    longest = max(seen, key=len)
    assert longest[:len(PROMPT)] == PROMPT
    written = longest[len(PROMPT):]
    assert written == text[:len(written)]
    assert len(written) >= len(text) - 1


# -- the record ----------------------------------------------------------------


def test_a_record_needs_a_model_that_speaks_and_a_prompt_without_a_marker(
        model):
    silent = _model(speech=False)
    with pytest.raises(ValueError, match="no speech output"):
        _record(silent, PROMPT, "audio")
    with pytest.raises(ValueError, match="not one of audio, interleaved"):
        _record(model, PROMPT, "text")
    with pytest.raises(ValueError, match="already ends in a speech marker"):
        _record(model, PROMPT + [AUDIO_START], "audio")
    record = _record(model, PROMPT, "interleaved", seed=None)
    from gmlx._vendor.mlx_vlm.server.generation import DEFAULT_SEED

    assert record.marker == MIXED_START and record.seed == DEFAULT_SEED
    assert record.steering == {AUDIO_START, MIXED_START, TEXT_END}
    assert not record.stop_at_end


def test_a_steering_token_is_quiet_and_plain_text_is_not(model):
    record = _record(model, PROMPT, "interleaved")
    record.take(MIXED_START)
    record.take(6)
    record.take(TEXT_END)
    kinds = [(s.kind, s.quiet) for s in record.slots]
    assert kinds == [("marker", True), ("text", False), ("text", True)]
    # The marker is not one of the six text slots.
    assert record.state.text_done and record.state.speaking


# -- the engine thread ---------------------------------------------------------


class _Streamer:
    def __init__(self):
        self.calls = []

    def advance(self, token, finish_reason):
        self.calls.append((token, finish_reason))
        return "" if finish_reason == "stop" else f"<{token}>"


def _engine_thread_step(responses):
    """Run the engine thread's response handling on ``responses``."""
    import queue

    from gmlx._vendor.mlx_vlm.server import generation

    rqueue, streamer = queue.Queue(), _Streamer()
    self = types.SimpleNamespace(
        _log_prefill_progress=lambda *a, **k: None,
        _log_decode_progress=lambda *a, **k: 0.0,
        _stream_text=lambda info, tok, reason: info["streamer"].advance(
            tok, reason),
        draft_model=None, draft_kind=None)
    gen = types.SimpleNamespace(next=lambda **k: ([], responses))
    active = {1: {"rqueue": rqueue, "streamer": streamer}}
    generation.ResponseGenerator._step(self, gen, active)
    out = []
    while not rqueue.empty():
        out.append(rqueue.get())
    return out, streamer.calls


def test_the_engine_thread_writes_no_text_for_a_quiet_slot():
    Response, Slot = ar.GenerationBatch.Response, speech_rows.Slot
    chunk = np.ones(4, np.float32)
    out, calls = _engine_thread_step([
        Response(1, MIXED_START, None, None, speech=Slot("marker", quiet=True)),
        Response(1, 6, None, None, speech=Slot("text")),
        Response(1, TEXT_END, None, None, speech=Slot("text", quiet=True)),
        Response(1, AUDIO_OUT, None, None,
                 speech=Slot("frame", chunk, quiet=True)),
        Response(1, AUDIO_OUT, None, "stop",
                 speech=Slot("end", chunk, quiet=True)),
    ])
    tokens = [t for t in out if t is not None]
    assert [t.text for t in tokens] == ["", "<6>", "", "", ""]
    assert [t.speech_kind for t in tokens] == [
        "marker", "text", "text", "frame", "end"]
    assert [t.audio is not None for t in tokens] == [
        False, False, False, True, True]
    # Only the text slot and the finish reach the detokenizer, and the
    # finish flushes it as a stop does.
    assert calls == [(6, None), (AUDIO_OUT, "stop")]
    assert out[-1] is None and tokens[-1].finish_reason == "stop"


def test_the_engine_thread_leaves_a_text_row_as_it_was():
    Response = ar.GenerationBatch.Response
    out, calls = _engine_thread_step([Response(1, 6, None, None)])
    assert out[0].text == "<6>" and out[0].audio is None
    assert out[0].speech_kind is None and calls == [(6, None)]


def _settings_for(model, draft_model, spec, prompt=PROMPT):
    from gmlx._vendor.mlx_vlm.server import generation

    self = types.SimpleNamespace(model=model, draft_model=draft_model,
                                 stop_tokens=())
    return generation.ResponseGenerator._speech_settings(
        self, RowSettings(top_k=3), spec, prompt,
        types.SimpleNamespace(seed=9))


def test_a_speech_request_gets_its_record_in_the_rows_settings(model):
    settings = _settings_for(model, None, {"mode": "audio",
                                           "stop_at_end": True})
    record = settings.speech
    assert settings.top_k == 3
    assert record.marker == AUDIO_START and record.seed == 9
    assert record.stop_at_end


def test_a_speech_request_is_refused_on_a_model_served_with_a_drafter(model):
    with pytest.raises(ValueError, match="served with a drafter"):
        _settings_for(model, object(), {"mode": "audio"})
