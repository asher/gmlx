# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Asher Feldman
"""Audio output on the chat route: which way a reply is made, what the
route sends for each token, the speech pass, the usage counts, and an
earlier spoken turn named by its audio id. The route tests run the chat
endpoint on a stub model whose generator plays scripted tokens."""

from __future__ import annotations

import base64
import importlib
import io
import json
import types
import wave
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("gmlx._vendor.mlx_vlm")

from fastapi import HTTPException  # noqa: E402
from transformers.utils.chat_template_utils import render_jinja_template  # noqa: E402

from gmlx.models.lfm3 import speak  # noqa: E402
from gmlx.serve import media_programs, native_tts  # noqa: E402
from gmlx.serve import speech_chat as sc  # noqa: E402
from gmlx.serve.patches import _common as sp_common  # noqa: E402

_OPENAI = importlib.import_module("gmlx._vendor.mlx_vlm.server.openai")
_APP = importlib.import_module("gmlx._vendor.mlx_vlm.server.app")
_GEN = importlib.import_module("gmlx._vendor.mlx_vlm.server.generation")
_SCHEMAS = importlib.import_module("gmlx._vendor.mlx_vlm.server.schemas")
_RUNTIME = importlib.import_module("gmlx._vendor.mlx_vlm.server.runtime").runtime
_TEMPLATE = (Path(__file__).parents[1] / "fixtures"
             / "qwen35_template.jinja").read_text()

MARKER_ID, MARKER = 7, "<|audio_start|>"
FRAME = np.full(1920, 0.25, np.float32)       # one frame, 80 ms
TAIL = np.full(960, 0.25, np.float32)         # what the codec held back


class _Model:
    """A model that speaks."""

    config = types.SimpleNamespace(speech_config=types.SimpleNamespace(
        audio_start_token_id=MARKER_ID))

    def speech_capabilities(self):
        return {"sample_rate": 24000, "voices": ["female", "male"]}


class _Processor:
    """Renders a chat template the way a tokenizer does."""

    chat_template = _TEMPLATE
    gguf_named_templates: dict = {}

    @property
    def tokenizer(self):
        return self

    def apply_chat_template(self, messages, tokenize=False,
                            add_generation_prompt=True, **kw):
        out = render_jinja_template(
            [messages], chat_template=self.chat_template,
            add_generation_prompt=add_generation_prompt, **kw)
        rendered = out[0] if isinstance(out, tuple) else out
        return rendered[0] if isinstance(rendered, list) else rendered

    def convert_ids_to_tokens(self, token):
        return MARKER if token == MARKER_ID else f"<{token}>"

    def encode(self, text):
        return list(range(len(text) // 4))


def _request(**kw):
    kw.setdefault("model", "omni")
    kw.setdefault("messages", [{"role": "user", "content": "Hi."}])
    return _SCHEMAS.ChatRequest(**kw)


def _plan(tools=None, generator=None, model=None, **kw):
    kw.setdefault("modalities", ["text", "audio"])
    return sc.plan(_request(**kw), model or _Model(),
                   generator or types.SimpleNamespace(draft_model=None),
                   tools)


def _token(text="", kind=None, audio=None, finish=None, token=1):
    return _GEN.StreamingToken(text=text, token=token, logprobs=None,
                               finish_reason=finish, speech_kind=kind,
                               audio=audio)


def _refused(match, **kw):
    with pytest.raises(HTTPException, match=match) as e:
        _plan(**kw)
    assert e.value.status_code == 400


# -- which way the reply is made ----------------------------------------------


def test_a_request_without_the_audio_modality_gets_text():
    assert _plan(modalities=None) is None
    assert _plan(modalities=["text"]) is None
    assert _plan(modalities=None, audio={"voice": "male"}) is None


def test_the_models_own_voice_is_a_spoken_reply():
    for audio in (None, {}, {"voice": "female"}, {"voice": "Female"}):
        reply = _plan(audio=audio)
        assert reply.kind == sc.SPOKEN and reply.voice == "female"
        assert reply.format == "wav" and not reply.text_is_content
    assert _plan(stream=True).format == "pcm16"


def test_the_other_voice_is_a_text_reply_then_a_speech_pass():
    reply = _plan(audio={"voice": "male", "language": "German"})
    assert (reply.kind, reply.voice, reply.language) == (
        sc.TEXT_FIRST, "male", "de")
    assert not reply.text_is_content


def test_tools_make_a_text_reply_whose_text_is_the_content():
    reply = _plan(tools=[{"type": "function"}])
    assert reply.kind == sc.TEXT_FIRST and reply.text_is_content


@pytest.mark.parametrize("kw,match", [
    ({"audio": {"voice": "alloy"}}, "which are female and male"),
    ({"audio": {"language": "hi"}}, "not one this model speaks"),
    ({"audio": {"format": "aac"}}, "unsupported audio format"),
    ({"stream": True, "audio": {"format": "wav"}}, "a stream sends audio"),
    ({"logprobs": True}, "logprobs are not available"),
    ({"response_format": {"type": "json_object"}},
     "response_format is not available"),
])
def test_a_request_the_model_cannot_speak_is_refused(kw, match):
    _refused(match, **kw)


def test_a_model_without_speech_and_a_drafter_are_refused():
    _refused("has no speech output", model=object())
    _refused("served with a drafter",
             generator=types.SimpleNamespace(draft_model=object()))


def test_a_format_that_needs_ffmpeg_is_refused_without_it(monkeypatch):
    monkeypatch.setattr(media_programs, "find", lambda name: None)
    _refused("needs ffmpeg", audio={"format": "mp3"})
    assert _plan(audio={"format": "wav"}).format == "wav"


def test_the_speech_route_is_the_pass_alone():
    token = sc.SPEECH_ONLY.set(sc.SpeechOnly("Read this.", "male", "de"))
    try:
        reply = _plan(modalities=None, stream=True)
    finally:
        sc.SPEECH_ONLY.reset(token)
    assert (reply.kind, reply.voice, reply.language) == (sc.ONLY, "male", "de")
    assert reply.marker == MARKER_ID


# -- before the generation ----------------------------------------------------


def test_a_spoken_reply_with_no_system_message_gets_the_cards():
    user = [{"role": "user", "content": "Hi."}]
    assert _plan().messages(user) == [
        {"role": "system", "content": speak.INTERLEAVED_SYSTEM}, *user]
    own = [{"role": "system", "content": "Be brief."}, *user]
    assert _plan().messages(own) == own
    assert _plan(audio={"voice": "male"}).messages(user) == user


def test_each_way_marks_its_row():
    args = types.SimpleNamespace()
    _plan().stamp(args)
    assert args._kq_speech == {"mode": "interleaved"}
    args = types.SimpleNamespace()
    _plan(audio={"voice": "male"}).stamp(args)
    assert not hasattr(args, "_kq_speech")


def test_the_speech_prompt_ends_before_the_marker():
    processor = _Processor()
    plain = sc.speech_prompt(processor, "Read this.", "male", None, MARKER_ID)
    assert "Perform TTS using a male voice." in plain
    assert "Read this." in plain and not plain.endswith(MARKER)
    # The model's own speech template ends in the marker.
    processor.gguf_named_templates = {
        "tts": "{{ language }}|{{ voice }}|{{ messages[0]['content'] }}"
               + MARKER}
    assert sc.speech_prompt(processor, "Read this.", "male", "de",
                            MARKER_ID) == "de|male|Read this."


# -- during a generation ------------------------------------------------------


def test_a_spoken_replys_text_is_its_transcript():
    reply = _plan(stream=True)
    assert reply.delta(_token(kind="marker"), None) == (None, None)
    content, audio = reply.delta(_token("Hi", kind="text"), "Hi")
    assert content is None and audio == {"transcript": "Hi", "id": reply.id}
    content, audio = reply.delta(_token(kind="frame", audio=FRAME), None)
    assert content is None and audio["id"] == reply.id
    assert base64.b64decode(audio["data"]) == sc.pcm16(FRAME)
    content, audio = reply.delta(_token(kind="end", audio=TAIL), None)
    assert base64.b64decode(audio["data"]) == sc.pcm16(TAIL)
    assert (reply.markers, reply.frames) == (1, 1)
    assert bytes(reply.pcm) == sc.pcm16(FRAME) + sc.pcm16(TAIL)
    assert reply.transcript() == "Hi"


def test_with_tools_the_text_stays_the_content():
    reply = _plan(tools=[{}], stream=True)
    assert reply.delta(_token("Hi"), "Hi") == ("Hi", None)
    assert reply.transcript() == "Hi"


def test_pcm_is_16_bit_and_clipped():
    assert sc.pcm16(np.array([0.0, 1.0, -1.0, 2.0], np.float32)) == \
        np.array([0, 32767, -32767, 32767], "<i2").tobytes()


# -- the speech pass ----------------------------------------------------------


class _Generator:
    """Plays a speech pass: the marker, ``frames`` frames and the end
    frame, or as many slots as the row's budget."""

    draft_model = None

    def __init__(self, frames=3, prompt_tokens=20):
        self.frames, self.prompt_tokens, self.calls = frames, prompt_tokens, []
        self.closed = 0

    def generate(self, prompt, images=None, audio=None, args=None,
                 videos=None):
        self.calls.append((prompt, args))
        slots = [_token(kind="marker")]
        slots += [_token(kind="frame", audio=FRAME)] * self.frames
        slots.append(_token(kind="end", audio=TAIL, finish="stop"))
        most = args.max_tokens + 1          # the engine adds the marker
        if len(slots) > most:
            slots = slots[:most]
            slots[-1] = _token(kind="frame", audio=FRAME, finish="length")
        return types.SimpleNamespace(prompt_tokens=self.prompt_tokens), \
            _Tokens(slots, self)


class _Tokens:
    def __init__(self, slots, owner):
        self.slots, self.owner = iter(slots), owner

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.slots)

    def close(self):
        self.owner.closed += 1


def _args(max_tokens=200, **kw):
    return _GEN.GenerationArguments(max_tokens=max_tokens, **kw)


def _spoken_pass(reply, generator, text, args=None, used=10):
    reply.text = [text]
    return list(reply.speak(generator, _Processor(), args or _args(), used))


def test_a_text_reply_with_no_tool_call_takes_a_pass():
    reply = _plan(audio={"voice": "male"})
    reply.text = ["Hello."]
    assert reply.wants_pass(False) and not reply.wants_pass(True)
    reply.text = ["  "]
    assert not reply.wants_pass(False)
    spoken = _plan()
    spoken.text = ["Hello."]
    assert not spoken.wants_pass(False)


def test_the_pass_speaks_the_sanitized_text_in_the_asked_voice():
    reply, generator = _plan(audio={"voice": "male"}), _Generator()
    chunks = _spoken_pass(reply, generator, "**Hello** there.",
                          _args(seed=4, temperature=0.9))
    assert chunks == [sc.pcm16(FRAME)] * 3 + [sc.pcm16(TAIL)]
    (prompt, args), = generator.calls
    assert "Perform TTS using a male voice." in prompt
    assert "Hello there." in prompt and "*" not in prompt
    assert args._kq_speech == {"mode": "audio", "stop_at_end": True}
    assert (args.seed, args.temperature) == (4, 0.0)
    assert generator.closed == 1
    assert (reply.frames, reply.markers, reply.pass_tokens) == (3, 1, 5)
    assert reply.pass_prompt_tokens == 20 and reply.pass_finish is None
    # The transcript is the reply's own text.
    assert reply.transcript() == "**Hello** there."


def test_the_pass_counts_against_the_requests_budget():
    text = "Hello there."
    allowance = native_tts.frame_budget(text)
    reply, generator = _plan(audio={"voice": "male"}), _Generator()
    _spoken_pass(reply, generator, text, _args(max_tokens=1000), used=10)
    assert generator.calls[0][1].max_tokens == allowance
    # Two slots are left: the audio ends there, and the reply's finish
    # reason is the budget.
    reply, generator = _plan(audio={"voice": "male"}), _Generator(frames=50)
    chunks = _spoken_pass(reply, generator, text, _args(max_tokens=12),
                          used=10)
    assert generator.calls[0][1].max_tokens == 2 and len(chunks) == 2
    assert reply.pass_finish == "length"
    assert reply.finish_reason("stop") == "length"
    # No slot is left: no pass runs.
    reply, generator = _plan(audio={"voice": "male"}), _Generator()
    assert _spoken_pass(reply, generator, text, _args(max_tokens=10),
                        used=10) == []
    assert not generator.calls and reply.pass_finish == "length"


def test_a_pass_that_runs_to_its_own_allowance_is_not_the_requests_limit(
        caplog):
    text = "Hi."
    allowance = native_tts.frame_budget(text)
    reply = _plan(audio={"voice": "male"})
    generator = _Generator(frames=allowance + 50)
    with caplog.at_level("WARNING"):
        chunks = _spoken_pass(reply, generator, text, _args(max_tokens=5000))
    assert len(chunks) == allowance and reply.pass_finish is None
    assert reply.finish_reason("stop") == "stop"
    assert "allowance" in caplog.text


def test_a_text_that_leaves_no_room_in_the_context_is_refused(monkeypatch):
    limit = [150]
    monkeypatch.setattr(_GEN, "get_configured_context_limit",
                        lambda: limit[0])
    reply, generator = _plan(audio={"voice": "male"}), _Generator()
    with pytest.raises(HTTPException, match="leaves no room") as e:
        _spoken_pass(reply, generator, "word " * 60)
    assert e.value.status_code == 400 and not generator.calls
    # With more context the pass gets the room the prompt leaves.
    limit[0] = 300
    reply = _plan(audio={"voice": "male"})
    _spoken_pass(reply, generator, "word " * 60, _args(max_tokens=5000))
    used = len(_Processor().encode(generator.calls[0][0]))
    assert generator.calls[0][1].max_tokens == 300 - used - 2


def test_a_long_text_is_spoken_piece_by_piece():
    text = "The ferry leaves the island at six each morning. " * 20
    pieces = native_tts.split_text(text.strip())
    assert len(pieces) > 2
    reply, generator = _plan(audio={"voice": "male"}), _Generator(frames=4)
    chunks = _spoken_pass(reply, generator, text,
                          _args(max_tokens=5000, seed=10))
    assert len(generator.calls) == len(pieces)
    for index, (piece, (prompt, args)) in enumerate(
            zip(pieces, generator.calls)):
        assert piece in prompt and args.seed == 10 + index
        assert args.max_tokens == native_tts.frame_budget(piece)
    assert len(chunks) == 5 * len(pieces)
    assert (reply.passes, reply.frames) == (len(pieces), 4 * len(pieces))
    # Each pass's prompt and marker are prompt tokens.
    counts = reply.usage(30, 8)
    assert counts["tts_tokens"] == 21 * len(pieces)
    assert counts["completion_tokens"] == 8 + 5 * len(pieces)


def test_the_requests_budget_ends_a_long_text_between_pieces():
    text = "The ferry leaves the island at six each morning. " * 20
    reply, generator = _plan(audio={"voice": "male"}), _Generator(frames=4)
    # Ten slots after the text: one piece of five, then five more.
    _spoken_pass(reply, generator, text, _args(max_tokens=20), used=10)
    assert [a.max_tokens for _, a in generator.calls] == [10, 5]
    assert reply.pass_finish == "length"


def test_the_speech_route_gets_the_speech_prompt_and_its_budget():
    token = sc.SPEECH_ONLY.set(sc.SpeechOnly("Read this.", "male", None))
    try:
        reply = _plan(modalities=None, stream=True)
    finally:
        sc.SPEECH_ONLY.reset(token)
    args = _args(max_tokens=10_000)
    prompt = reply.prompt(_Processor(), "the chat prompt", args, _Generator())
    assert "Perform TTS using a male voice." in prompt
    assert "Read this." in prompt and "the chat prompt" not in prompt
    assert args.max_tokens == native_tts.frame_budget("Read this.")
    reply.stamp(args)
    assert args._kq_speech == {"mode": "audio", "stop_at_end": True}
    # Another reply keeps the chat prompt.
    assert _plan().prompt(_Processor(), "the chat prompt", args,
                          _Generator()) == "the chat prompt"


# -- after the reply ----------------------------------------------------------


def _wav_seconds(data: str) -> float:
    with wave.open(io.BytesIO(base64.b64decode(data))) as w:
        return w.getnframes() / w.getframerate()


def test_a_message_carries_the_audio_and_its_transcript():
    reply, generator = _plan(audio={"voice": "male"}), _Generator(frames=4)
    audio = reply.message_audio(generator, _Processor(), _args(), 10,
                                "Hello there.", False)
    assert audio["id"] == reply.id and audio["transcript"] == "Hello there."
    assert _wav_seconds(audio["data"]) == pytest.approx(4 * 0.08 + 0.04)
    assert reply.message_content("Hello there.", audio) is None
    # With tools the text is the content too.
    reply = _plan(tools=[{}])
    audio = reply.message_audio(generator, _Processor(), _args(), 10,
                                "Hello there.", False)
    assert reply.message_content("Hello there.", audio) == "Hello there."
    # A tool call has no audio, and a reply with no audio keeps its text.
    reply = _plan(tools=[{}])
    assert reply.message_audio(generator, _Processor(), _args(), 10,
                               None, True) is None
    reply = _plan(audio={"voice": "male"})
    assert reply.message_content("Hello.", None) == "Hello."


def test_the_usage_counts_a_marker_as_a_prompt_token():
    # A spoken reply: the marker, 6 text slots, 3 frames, the end frame.
    reply = _plan()
    for token in ([_token(kind="marker")] + [_token("a", kind="text")] * 6
                  + [_token(kind="frame", audio=FRAME)] * 3
                  + [_token(kind="end", audio=TAIL)]):
        reply.take(token)
    assert reply.usage(30, 11) == {
        "prompt_tokens": 31, "completion_tokens": 10, "total_tokens": 41,
        "audio_tokens": 3, "tts_tokens": 0}
    # A text reply of 8 tokens, then a pass: 20 prompt tokens, the
    # marker, 3 frames and the end frame.
    reply, generator = _plan(audio={"voice": "male"}), _Generator()
    _spoken_pass(reply, generator, "Hello.", used=8)
    assert reply.usage(30, 8) == {
        "prompt_tokens": 51, "completion_tokens": 12, "total_tokens": 63,
        "audio_tokens": 3, "tts_tokens": 21}


def test_the_speech_routes_whole_prompt_is_speech_prompt():
    token = sc.SPEECH_ONLY.set(sc.SpeechOnly("Read this.", "female", None))
    try:
        reply = _plan(modalities=None, stream=True)
    finally:
        sc.SPEECH_ONLY.reset(token)
    for t in ([_token(kind="marker")]
              + [_token(kind="frame", audio=FRAME)] * 2
              + [_token(kind="end", audio=TAIL)]):
        reply.take(t)
    assert reply.usage(30, 4) == {
        "prompt_tokens": 31, "completion_tokens": 3, "total_tokens": 34,
        "audio_tokens": 2, "tts_tokens": 31}


def test_the_usage_details_appear_only_on_a_reply_with_audio():
    usage = _SCHEMAS.UsageStats(prompt_tokens=3, completion_tokens=4,
                                total_tokens=7)
    assert usage.model_dump() == {
        "prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7,
        "prompt_tokens_details": {"cached_tokens": 0}}
    sc.apply_usage(usage, {"prompt_tokens": 31, "completion_tokens": 10,
                           "total_tokens": 41, "audio_tokens": 3,
                           "tts_tokens": 21})
    assert json.loads(usage.model_dump_json()) == {
        "prompt_tokens": 31, "completion_tokens": 10, "total_tokens": 41,
        "prompt_tokens_details": {"cached_tokens": 0, "tts_tokens": 21},
        "completion_tokens_details": {"audio_tokens": 3}}


# -- an earlier spoken turn ---------------------------------------------------


def _assistant(**kw):
    return _SCHEMAS.ChatMessage(role="assistant", **kw)


def test_an_earlier_spoken_turn_goes_back_as_its_transcript(monkeypatch):
    monkeypatch.setattr(sc, "_TRANSCRIPTS", type(sc._TRANSCRIPTS)())
    expires = sc.remember("audio_a", "Red, blue and yellow.")
    assert expires > 0
    assert sc.history_text(_assistant(audio={"id": "audio_a"})) == \
        "Red, blue and yellow."
    # A turn that carries its transcript needs no lookup.
    assert sc.history_text(_assistant(
        audio={"id": "audio_x", "transcript": "Said."})) == "Said."
    assert sc.history_text(_assistant(content="Plain.")) is None
    assert sc.history_text(_SCHEMAS.ChatMessage(
        role="user", content="Hi.", audio={"id": "audio_a"})) is None


def test_an_audio_id_the_server_does_not_hold_is_refused(monkeypatch):
    monkeypatch.setattr(sc, "_TRANSCRIPTS", type(sc._TRANSCRIPTS)())
    with pytest.raises(HTTPException, match="is not held") as e:
        sc.history_text(_assistant(audio={"id": "audio_gone"}))
    assert e.value.status_code == 400
    sc.remember("audio_old", "Old.")
    monkeypatch.setattr(sc.time, "time", lambda: 1e12)
    with pytest.raises(HTTPException, match="is not held"):
        sc.history_text(_assistant(audio={"id": "audio_old"}))


def test_the_server_keeps_a_bounded_number_of_transcripts(monkeypatch):
    monkeypatch.setattr(sc, "_TRANSCRIPTS", type(sc._TRANSCRIPTS)())
    monkeypatch.setattr(sc, "TRANSCRIPTS_KEPT", 3)
    for i in range(5):
        sc.remember(f"audio_{i}", str(i))
    assert list(sc._TRANSCRIPTS) == ["audio_2", "audio_3", "audio_4"]


# -- the chat route -----------------------------------------------------------


class _Engine:
    """The response generator: a reply's row plays what its mark asks
    for, and a speech pass plays a marker, frames and the end frame."""

    draft_model = None

    def __init__(self):
        self.calls = []

    def validate_context_budget(self, prompt, **kw):
        return None

    def generate(self, prompt, images=None, audio=None, args=None,
                 videos=None):
        mark = getattr(args, "_kq_speech", None)
        self.calls.append((prompt, mark, args.max_tokens))
        if mark is None:
            slots = [_token("Red, "), _token("blue."), _token(finish="stop")]
        elif mark["mode"] == "interleaved":
            slots = ([_token(kind="marker")]
                     + [_token("Red, ", kind="text"),
                        _token("blue.", kind="text")]
                     + [_token(kind="frame", audio=FRAME)] * 3
                     + [_token(kind="end", audio=TAIL),
                        _token(kind="text", finish="stop")])
        else:
            slots = ([_token(kind="marker")]
                     + [_token(kind="frame", audio=FRAME)] * 2
                     + [_token(kind="end", audio=TAIL, finish="stop")])
        return types.SimpleNamespace(prompt_tokens=30, uid=1), _Tokens(
            slots, types.SimpleNamespace(closed=0))


@pytest.fixture
def chat(monkeypatch):
    """The chat route on a stub model that speaks. Returns
    ``(post, engine)``."""
    from fastapi.testclient import TestClient

    routes = sp_common._snapshot_routes(_APP.app)
    engine = _Engine()
    monkeypatch.setattr(_RUNTIME, "response_generator", engine, raising=False)
    monkeypatch.setattr(sc, "_TRANSCRIPTS", type(sc._TRANSCRIPTS)())
    monkeypatch.setattr(
        _OPENAI, "get_cached_model",
        lambda *a, **k: (_Model(), _Processor(), {"model_type": "qwen3_5"}))
    client = TestClient(_APP.app)

    def post(**body):
        body.setdefault("model", "omni")
        body.setdefault("messages", [{"role": "user", "content": "Colors?"}])
        body.setdefault("max_tokens", 100)
        return client.post("/v1/chat/completions", json=body)

    yield post, engine
    sp_common._restore_routes(_APP.app, routes)


def _events(response):
    assert response.status_code == 200, response.text
    return [json.loads(line[6:]) for line in response.text.splitlines()
            if line.startswith("data: ") and line != "data: [DONE]"]


def _deltas(events):
    return [c["delta"] for e in events for c in e["choices"]]


def test_a_text_request_is_as_it_was(chat):
    post, engine = chat
    body = post().json()
    message = body["choices"][0]["message"]
    assert message["content"] == "Red, blue." and "audio" not in message
    assert body["usage"] == {
        "prompt_tokens": 30, "completion_tokens": 3, "total_tokens": 33,
        "prompt_tokens_details": {"cached_tokens": 0}}
    assert engine.calls[0][1] is None


def test_a_spoken_reply_not_streamed(chat):
    post, engine = chat
    body = post(modalities=["text", "audio"]).json()
    prompt, mark, _ = engine.calls[0]
    assert mark == {"mode": "interleaved"} and len(engine.calls) == 1
    assert speak.INTERLEAVED_SYSTEM in prompt
    choice = body["choices"][0]
    message = choice["message"]
    assert message["content"] is None and choice["finish_reason"] == "stop"
    audio = message["audio"]
    assert audio["transcript"] == "Red, blue."
    assert _wav_seconds(audio["data"]) == pytest.approx(3 * 0.08 + 0.04)
    assert body["usage"]["completion_tokens_details"] == {"audio_tokens": 3}
    assert body["usage"]["prompt_tokens"] == 31
    assert body["usage"]["completion_tokens"] == 7
    # A later turn names the reply by its audio id.
    post(messages=[
        {"role": "user", "content": "Colors?"},
        {"role": "assistant", "audio": {"id": audio["id"]}},
        {"role": "user", "content": "Again."}])
    assert "Red, blue." in engine.calls[-1][0]


def test_a_spoken_reply_streamed(chat):
    post, engine = chat
    events = _events(post(modalities=["text", "audio"], stream=True,
                          stream_options={"include_usage": True}))
    deltas = _deltas(events)
    assert all(not d.get("content") for d in deltas)
    audio = [d["audio"] for d in deltas if d.get("audio")]
    assert "".join(a.get("transcript", "") for a in audio) == "Red, blue."
    pcm = b"".join(base64.b64decode(a["data"]) for a in audio if "data" in a)
    assert pcm == sc.pcm16(FRAME) * 3 + sc.pcm16(TAIL)
    assert len({a["id"] for a in audio}) == 1 and "expires_at" in audio[-1]
    assert events[-1]["usage"]["completion_tokens_details"] == {
        "audio_tokens": 3}
    finish = [c["finish_reason"] for e in events for c in e["choices"]
              if c.get("finish_reason")]
    assert finish == ["stop"]


def test_the_other_voice_speaks_after_the_text(chat):
    post, engine = chat
    body = post(modalities=["text", "audio"],
                audio={"voice": "male", "format": "wav"}).json()
    (_, first, _), (pass_prompt, second, _) = engine.calls
    assert first is None
    assert second == {"mode": "audio", "stop_at_end": True}
    assert "Perform TTS using a male voice." in pass_prompt
    assert "Red, blue." in pass_prompt
    message = body["choices"][0]["message"]
    assert message["content"] is None
    assert message["audio"]["transcript"] == "Red, blue."
    assert _wav_seconds(message["audio"]["data"]) == pytest.approx(
        2 * 0.08 + 0.04)
    usage = body["usage"]
    assert usage["prompt_tokens_details"]["tts_tokens"] == 31
    assert usage["completion_tokens_details"] == {"audio_tokens": 2}
    assert usage["prompt_tokens"] == 61


def test_the_other_voice_streams_the_transcript_then_the_audio(chat):
    post, engine = chat
    events = _events(post(modalities=["text", "audio"], stream=True,
                          audio={"voice": "male", "format": "pcm16"}))
    audio = [d["audio"] for d in _deltas(events) if d.get("audio")]
    kinds = ["t" if "transcript" in a else "a" if "data" in a else "x"
             for a in audio]
    assert kinds == ["t", "t", "a", "a", "a", "x"]
    assert len(engine.calls) == 2


def test_the_speech_route_reaches_the_model_as_the_pass_alone(chat):
    post, engine = chat
    token = sc.SPEECH_ONLY.set(sc.SpeechOnly("Read this.", "male", None))
    try:
        events = _events(post(stream=True, max_tokens=10_000,
                              stream_options={"include_usage": True}))
    finally:
        sc.SPEECH_ONLY.reset(token)
    (prompt, mark, budget), = engine.calls
    assert mark == {"mode": "audio", "stop_at_end": True}
    assert "Read this." in prompt and "Colors?" not in prompt
    assert budget == native_tts.frame_budget("Read this.")
    pcm = b"".join(base64.b64decode(d["audio"]["data"])
                   for d in _deltas(events)
                   if d.get("audio", {}).get("data"))
    assert pcm == sc.pcm16(FRAME) * 2 + sc.pcm16(TAIL)
    assert events[-1]["usage"]["prompt_tokens"] == 31


def test_a_refused_audio_request_is_a_400(chat):
    post, engine = chat
    r = post(modalities=["text", "audio"], audio={"voice": "alloy"})
    assert r.status_code == 400 and "female and male" in r.text
    assert not engine.calls
