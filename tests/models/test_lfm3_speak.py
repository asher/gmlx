#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Asher Feldman
"""An LFM3 reply in text and speech on random weights: the reply's state
against the reference's loop, and the single-stream generator's slots,
budget and audio."""

from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten, tree_unflatten

from gmlx.models.lfm3 import speak
from gmlx.models.lfm3.model import CONV, FULL, ModelArgs
from gmlx.models.lfm3.speak import AUDIO, INTERLEAVED, TEXT, SpeechState
from gmlx.models.lfm3.speech import SpeechConfig
from gmlx.models.lfm3.vlm_model import Model, ModelConfig

BOOKS, CODES, VOCAB = 12, 37, 40
IMAGE, VIDEO, AUDIO_IN = 30, 31, 32
AUDIO_START, MIXED_START, TEXT_END, AUDIO_OUT = 34, 35, 36, 37
END = 39
FRAME = 96


def _config(**kw) -> SpeechConfig:
    base = dict(
        hidden_size=32, intermediate_size=48, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, projection_dim=24,
        codebook_size=CODES, codec_hidden_size=32,
        codec_intermediate_size=48, codec_num_attention_heads=4,
        codec_num_key_value_heads=2, codec_window=8, quantizer_dim=20,
        n_fft=64, hop_length=16, repetition_window=3,
        audio_start_token_id=AUDIO_START, mixed_start_token_id=MIXED_START,
        text_end_token_id=TEXT_END, audio_out_token_id=AUDIO_OUT)
    base.update(kw)
    return SpeechConfig(**base)


def _model(speech=True, **kw) -> Model:
    text = ModelArgs(
        model_type="lfm3", hidden_size=24, num_hidden_layers=2,
        intermediate_size=32, num_attention_heads=4, num_key_value_heads=2,
        head_dim=6, vocab_size=VOCAB, layer_types=[CONV, FULL])
    model = Model(ModelConfig(
        text_config=text, vision_config=None, model_type="lfm3_audio",
        speech_config=_config(**kw) if speech else None,
        image_token_id=IMAGE, video_token_id=VIDEO, audio_token_id=AUDIO_IN,
        vocab_size=VOCAB))
    mx.random.seed(0)
    model.update(tree_unflatten([
        (k, mx.random.normal(v.shape) * 0.3)
        for k, v in tree_flatten(model.parameters())]))
    mx.eval(model.parameters())
    return model


@pytest.fixture(scope="module")
def model():
    return _model()


def _never_ends(model, monkeypatch, end_at=None):
    """Keep the end code out of the sampled frames, or put it in frame
    ``end_at`` alone."""
    depth = model.depth_decoder
    sample, count = depth.sample_frame, [0]

    def patched(hidden, key, seen=None, **kw):
        codes = sample(hidden, key, seen, **kw)
        codes = mx.where(codes == CODES - 1, 0, codes)
        if count[0] == end_at:
            codes = mx.concatenate(
                [mx.full((codes.shape[0], 1), CODES - 1, dtype=codes.dtype),
                 codes[:, 1:]], axis=1)
        count[0] += 1
        return codes

    monkeypatch.setattr(depth, "sample_frame", patched)


def _scripted(tokens):
    """A sampler that gives ``tokens`` in turn."""
    left = list(tokens)
    return lambda logprobs: mx.array([left.pop(0)])


def _prompt(model, tokens=(1, 2, 3, 4, 5)):
    ids = mx.array([list(tokens)])
    return ids, model.language_model.model.embed_tokens(ids)


def _run(model, state, ids=None, **kw):
    ids, embeds = _prompt(model) if ids is None else ids
    events = list(speak.speech_slots(model, ids, embeds, state, **kw))
    return events[:-1], events[-1]


def _kinds(events) -> str:
    return "".join("t" if e.token is not None else "f" for e in events)


# -- the reply's state --------------------------------------------------------


def test_the_prompts_last_token_gives_the_mode():
    cfg = _config()
    assert speak.prompt_mode(cfg, AUDIO_START) == AUDIO
    assert speak.prompt_mode(cfg, AUDIO_OUT) == AUDIO
    assert speak.prompt_mode(cfg, MIXED_START) == INTERLEAVED
    assert speak.prompt_mode(cfg, 7) == TEXT


def test_a_named_mode_gives_the_marker_the_prompt_needs():
    cfg = _config()
    state, marker = speak.start(cfg, 7)
    assert (state.mode, state.text_only, marker) == (TEXT, False, None)
    state, marker = speak.start(cfg, 7, AUDIO)
    assert (state.mode, marker) == (AUDIO, AUDIO_START)
    state, marker = speak.start(cfg, 7, INTERLEAVED)
    assert (state.mode, marker) == (INTERLEAVED, MIXED_START)
    # A prompt that ends in its marker needs none.
    assert speak.start(cfg, AUDIO_START, AUDIO)[1] is None
    assert speak.start(cfg, MIXED_START, INTERLEAVED)[1] is None
    # A named text reply cannot start speech.
    state, marker = speak.start(cfg, MIXED_START, TEXT)
    assert (state.mode, state.text_only, marker) == (TEXT, True, None)
    with pytest.raises(ValueError, match="'sung' is not one of"):
        speak.start(cfg, 7, "sung")
    with pytest.raises(ValueError, match="no text_end_token_id"):
        speak.start(_config(text_end_token_id=None), 7)


def _reference(script, mode, cfg, end_of_turn):
    """One row of the reference's generate loop, on what each slot
    samples: its kinds of slot, whether it finished, and its last mode."""
    slots_left = cfg.interleaved_n_text
    speaking = text_done = finished = False
    kinds = []
    for token, ends in script:
        if finished:
            break
        interleaving = mode == INTERLEAVED
        if mode == AUDIO or (interleaving and speaking):
            kinds.append("f")
            if ends:
                mode = TEXT
        else:
            kinds.append("t")
            if token == cfg.text_end_token_id:
                if interleaving:
                    text_done, speaking = True, True
                else:
                    finished = True
            elif token in end_of_turn:
                finished = True
            elif interleaving:
                pass
            elif token == cfg.audio_start_token_id:
                mode = AUDIO
            elif token == cfg.mixed_start_token_id:
                mode = INTERLEAVED
                slots_left, speaking, text_done = (
                    cfg.interleaved_n_text, False, False)
        if interleaving and not text_done:
            slots_left -= 1
            if slots_left == 0:
                speaking = not speaking
                slots_left = (cfg.interleaved_n_audio if speaking
                              else cfg.interleaved_n_text)
    return "".join(kinds), finished, mode


def test_the_state_is_the_references_loop():
    cfg = _config(interleaved_n_text=3, interleaved_n_audio=4)
    rng = np.random.default_rng(0)
    tokens = [1, 2, 3, AUDIO_START, MIXED_START, TEXT_END, END]
    weights = np.array([10, 10, 10, 2, 3, 2, 1], dtype=np.float64)
    seen = set()
    for _ in range(300):
        script = [(int(rng.choice(tokens, p=weights / weights.sum())),
                   bool(rng.random() < 0.08)) for _ in range(60)]
        mode = (TEXT, AUDIO, INTERLEAVED)[int(rng.integers(3))]
        state = SpeechState(cfg, mode, end_of_turn=[END, TEXT_END])
        kinds = []
        for token, ends in script:
            if state.finished:
                break
            if state.wants_frame:
                kinds.append("f")
                frame = np.zeros(BOOKS, dtype=np.int32)
                if ends:
                    frame[5] = cfg.eos_code
                assert state.take_frame(frame) is ends
            else:
                kinds.append("t")
                state.take_token(token)
        got = ("".join(kinds), state.finished, state.mode)
        assert got == _reference(script, mode, cfg, {END})
        seen.add(got[0][:12])
    assert len(seen) > 50


def test_an_interleaved_reply_alternates_text_and_frames():
    state = SpeechState(_config(), INTERLEAVED, end_of_turn=[END])
    kinds = []
    for _ in range(40):
        kinds.append("f" if state.wants_frame else "t")
        if state.wants_frame:
            state.take_frame(np.zeros(BOOKS, dtype=np.int32))
        else:
            # A mode marker inside a block is text.
            state.take_token(AUDIO_START)
    assert "".join(kinds) == "t" * 6 + "f" * 12 + "t" * 6 + "f" * 12 + "tttt"
    assert state.mode == INTERLEAVED


def test_text_end_gives_the_rest_of_an_interleaved_reply_to_speech():
    state = SpeechState(_config(), INTERLEAVED, end_of_turn=[END, TEXT_END])
    state.take_token(1)
    state.take_token(TEXT_END)
    assert not state.finished
    for _ in range(30):
        assert state.wants_frame
        state.take_frame(np.zeros(BOOKS, dtype=np.int32))
    end = np.zeros(BOOKS, dtype=np.int32)
    end[3] = CODES - 1
    assert state.take_frame(end) is True
    assert state.mode == TEXT and not state.wants_frame
    assert state.frames == 30
    state.take_token(END)
    assert state.finished


def test_text_end_ends_a_text_reply():
    state = SpeechState(_config(), TEXT, end_of_turn=[END, TEXT_END])
    state.take_token(TEXT_END)
    assert state.finished


def test_the_history_keeps_the_penalty_window():
    state = SpeechState(_config(repetition_window=3), AUDIO)
    for i in range(5):
        state.take_frame(np.full(BOOKS, i, dtype=np.int32))
    assert [int(f[0]) for f in state.history] == [2, 3, 4]
    assert state.frames == 5
    none = SpeechState(_config(repetition_window=0), AUDIO)
    none.take_frame(np.zeros(BOOKS, dtype=np.int32))
    assert len(none.history) == 0


# -- the model's hook ---------------------------------------------------------


def test_a_model_says_what_its_speech_is(model):
    assert model.speech_capabilities() == {
        "sample_rate": 24000, "frame_samples": FRAME,
        "voices": ["female", "male"],
        "modes": ["text", "audio", "interleaved"]}
    silent = _model(speech=False)
    assert silent.speech_capabilities() is None
    with pytest.raises(ValueError, match="no speech generator"):
        silent.speech_start(7)
    with pytest.raises(ValueError, match="no speech output"):
        next(speak.stream_speech(silent, None, "hi"))


def test_a_text_slot_masks_the_placeholders_and_a_text_reply_the_markers(
        model):
    hidden = mx.zeros((1, 24))
    for text_only, masked in ((False, {IMAGE, VIDEO, AUDIO_IN, AUDIO_OUT}),
                              (True, {IMAGE, VIDEO, AUDIO_IN, AUDIO_OUT,
                                      AUDIO_START, MIXED_START})):
        state = SpeechState(model.config.speech_config, TEXT,
                            text_only=text_only)
        logits = np.asarray(model.speech_text_logits(state, hidden))
        assert logits.shape == (1, VOCAB)
        assert set(np.flatnonzero(np.isneginf(logits[0]))) == masked


def test_one_end_code_makes_the_whole_frame_the_end(model, monkeypatch):
    _never_ends(model, monkeypatch, end_at=0)
    state = SpeechState(model.config.speech_config, AUDIO)
    codes, embedding = model.speech_step(
        state, mx.ones((1, 24)), mx.random.key(0))
    assert np.asarray(codes).tolist() == [[CODES - 1] * BOOKS]
    assert embedding.shape == (1, 1, 24)
    want = model.depth_decoder.feedback_embedding(codes)
    assert mx.array_equal(embedding[:, 0], want)


# -- the generator ------------------------------------------------------------


def test_an_audio_reply_is_frames_up_to_the_budget(model, monkeypatch):
    _never_ends(model, monkeypatch)
    state = SpeechState(model.config.speech_config, AUDIO)
    events, summary = _run(model, state, max_tokens=5, key=mx.random.key(1))
    assert _kinds(events) == "fffff"
    assert all(e.codes.shape == (BOOKS,) for e in events)
    assert (summary.finish_reason, summary.frames, summary.text_tokens,
            summary.prompt_tokens) == ("length", 5, 0, 5)
    audio = np.concatenate([e.audio for e in events] + [summary.audio])
    assert audio.shape == (5 * FRAME,) and np.isfinite(audio).all()
    # The stream's audio is the codec's decode of the frames.
    whole = model.codec_decoder.decode(np.stack([e.codes for e in events]))
    np.testing.assert_allclose(audio, whole, atol=2e-6)


def test_a_key_fixes_the_frames(model, monkeypatch):
    _never_ends(model, monkeypatch)

    def frames(seed):
        state = SpeechState(model.config.speech_config, AUDIO)
        events, _ = _run(model, state, max_tokens=4, key=mx.random.key(seed))
        return np.stack([e.codes for e in events]).tolist()

    assert frames(3) == frames(3)
    assert any(frames(3) != frames(seed) for seed in (4, 5, 6))


def test_a_frames_embedding_is_the_text_models_next_input(monkeypatch):
    """The generator's frames against a loop written out here: each
    frame's embedding goes through the text model, and the hidden state
    it gives picks the next frame."""
    from gmlx._vendor.mlx_vlm.models import cache as vlm_cache

    model = _model(temperature=0.0)
    _never_ends(model, monkeypatch)
    ids, embeds = _prompt(model)
    lm = model.language_model
    cache = vlm_cache.make_prompt_cache(lm)
    hidden = lm.model(ids, cache, embeds)[:, -1]
    state = SpeechState(model.config.speech_config, AUDIO)
    want = []
    # The key also orders the reveal, so the loop splits it as the
    # generator does.
    key = mx.random.key(9)
    for _ in range(4):
        key, frame_key = mx.random.split(key)
        codes, embedding = model.speech_step(state, hidden, frame_key)
        state.take_frame(np.asarray(codes)[0])
        want.append(np.asarray(codes)[0].tolist())
        hidden = lm.model(mx.zeros((1, 1), dtype=mx.int32), cache,
                          embedding)[:, -1]
    assert len({tuple(f) for f in want}) > 1

    for step in (None, 2):
        state = SpeechState(model.config.speech_config, AUDIO)
        events, _ = _run(model, state, max_tokens=4, key=mx.random.key(9),
                         prefill_step_size=step)
        assert [e.codes.tolist() for e in events] == want, step


def test_text_and_frames_share_the_budget(model, monkeypatch):
    _never_ends(model, monkeypatch)
    state = SpeechState(model.config.speech_config, TEXT, end_of_turn=[END])
    events, summary = _run(
        model, state, max_tokens=6, key=mx.random.key(0),
        sampler=_scripted([5, 6, AUDIO_START]))
    assert _kinds(events) == "tttfff"
    assert [e.token for e in events[:3]] == [5, 6, AUDIO_START]
    assert (summary.finish_reason, summary.text_tokens, summary.frames) == (
        "length", 3, 3)
    audio = [e.audio for e in events[3:]] + [summary.audio]
    assert sum(a.size for a in audio) == 3 * FRAME


def test_the_end_frame_closes_the_audio_and_the_reply_goes_on_in_text(
        model, monkeypatch):
    _never_ends(model, monkeypatch, end_at=2)
    state = SpeechState(model.config.speech_config, AUDIO, end_of_turn=[END])
    events, summary = _run(
        model, state, max_tokens=20, key=mx.random.key(0),
        sampler=_scripted([7, END]))
    assert _kinds(events) == "ffft"
    # The end frame carries no codes. It gives the held-back samples.
    assert events[2].codes is None
    assert sum(e.audio.size for e in events[:3]) == 2 * FRAME
    assert events[3].token == 7
    assert (summary.finish_reason, summary.frames, summary.text_tokens) == (
        "stop", 2, 1)
    assert summary.audio is None


def test_an_interleaved_reply_runs_through_the_generator(model, monkeypatch):
    _never_ends(model, monkeypatch)
    cfg = model.config.speech_config
    state, marker = model.speech_start(5, INTERLEAVED, end_of_turn=[END])
    assert marker == MIXED_START
    ids = mx.array([[1, 2, 3, 4, 5, marker]])
    events, summary = _run(
        model, state, (ids, model.language_model.model.embed_tokens(ids)),
        max_tokens=26, key=mx.random.key(0), sampler=_scripted([1] * 30))
    assert _kinds(events) == "t" * 6 + "f" * 12 + "t" * 6 + "ff"
    assert summary.frames == 14 and summary.text_tokens == 12
    assert sum(e.audio.size for e in events if e.audio is not None) \
        + summary.audio.size == 14 * cfg.frame_samples


def test_logits_processors_see_the_text_tokens(model, monkeypatch):
    _never_ends(model, monkeypatch)
    seen = []

    def processor(tokens, logits):
        seen.append(np.asarray(tokens).tolist())
        return logits

    state = SpeechState(model.config.speech_config, TEXT, end_of_turn=[END])
    events, summary = _run(
        model, state, max_tokens=8, logits_processors=[processor],
        sampler=_scripted([5, AUDIO_START, 0, 0]), key=mx.random.key(0))
    assert _kinds(events) == "tt" + "f" * 6
    assert seen == [[], [5]]


def test_no_budget_gives_only_the_summary(model):
    state = SpeechState(model.config.speech_config, AUDIO)
    events, summary = _run(model, state, max_tokens=0)
    assert events == [] and summary.finish_reason == "length"
    assert summary.audio is None and summary.frames == 0


# -- the tts prompt -----------------------------------------------------------

# A stand-in with the template's variables. It is not the model's text.
_STAND_IN = ("{{ bos_token }}[{{ language | default('none') }}|"
             "{{ voice | default('none') }}] {{ messages[0]['content'] }}")


def test_the_tts_prompt_is_the_models_named_template():
    tok = SimpleNamespace(gguf_named_templates={"tts": _STAND_IN},
                          bos_token="<s>")
    assert speak.tts_prompt(tok, "Hello.") == "<s>[none|none] Hello."
    assert speak.tts_prompt(tok, "Hello.", "male", "fr") == \
        "<s>[fr|male] Hello."
    with pytest.raises(ValueError, match="'robot' is not one of female, male"):
        speak.tts_prompt(tok, "Hello.", "robot")
    with pytest.raises(ValueError, match="no tts chat template"):
        speak.tts_prompt(SimpleNamespace(gguf_named_templates={}), "Hello.")
