# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Asher Feldman
"""The speech route on a served model that speaks: which models speak,
which model answers, the request fields, the formats, and the pass the
route sends through the chat route. The chat route is stubbed."""

from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import sys
import wave
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import gmlx.load.discovery as disc  # noqa: E402
from gmlx.serve import media_programs, speech_chat  # noqa: E402
from gmlx.serve import native_tts as nt  # noqa: E402
from gmlx.serve.tts import TTSRequestError  # noqa: E402
from lfm3_header import write_lfm3_header, write_projector_header  # noqa: E402


@pytest.fixture()
def files(tmp_path, monkeypatch):
    """Header-only models: LFM3 with a projector that speaks (`omni`),
    LFM3 with a vision projector (`vl`), LFM3 alone (`text`), and another
    family with a projector that speaks (`other`)."""
    import gguf

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setattr(disc, "_HEADER_DISK", None)
    lfm = write_lfm3_header(tmp_path / "lfm3.gguf", templates=("tts",))
    speaks = write_projector_header(tmp_path / "mmproj-omni.gguf",
                                    vision=True, audio=True, speech=True)
    sees = write_projector_header(tmp_path / "mmproj-vl.gguf", vision=True)
    w = gguf.GGUFWriter(str(tmp_path / "other.gguf"), "gemma4")
    w.add_tensor("token_embd.weight", np.zeros((2, 2), dtype=np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return {"omni": (lfm, speaks), "vl": (lfm, sees), "text": (lfm, None),
            "other": (str(tmp_path / "other.gguf"), speaks)}


@pytest.fixture()
def served(files, monkeypatch):
    """A server that serves every model of ``files`` under its key."""
    from gmlx.serve import bridge_vlm

    def spec(name):
        path, mmproj = files[name]
        return SimpleNamespace(path=path, mmproj=mmproj)

    def resolve(name, **kw):
        if name not in files:
            raise bridge_vlm.ModelNotFound(name, {})
        return files[name][0], spec(name)

    monkeypatch.setattr(bridge_vlm, "resolve_request_model", resolve)
    monkeypatch.setattr(bridge_vlm, "_resolve_served",
                        lambda mid, cfg, **kw: spec(mid))
    return files


def _cfg(*ids, default=None):
    return SimpleNamespace(models={i: object() for i in ids}, aliases={},
                           defaults=SimpleNamespace(model=default))


# -- which models speak -------------------------------------------------------


def test_an_lfm3_model_with_a_speech_generator_speaks(files):
    assert nt.tts_adapter(*files["omni"]) == "lfm3"
    for name in ("vl", "text", "other"):
        assert nt.tts_adapter(*files[name]) is None
    assert nt.tts_adapter(None, files["omni"][1]) is None


def test_an_unset_tts_picks_the_model_that_speaks(served):
    assert nt.native_models(_cfg("text", "omni", "vl", "other")) == ["omni"]
    assert nt.auto_native(_cfg("text", "omni", "vl")) == "model:omni"
    assert nt.auto_native(_cfg("text", "vl")) is None


def test_a_configured_model_id_resolves_to_the_served_model(served):
    from gmlx.config import ConfigError

    cfg = _cfg("omni", "vl")
    assert nt.native_for("omni", cfg) == "model:omni"
    assert nt.native_for("model:omni", cfg) == "model:omni"
    assert nt.native_for("kokoro", cfg) is None
    with pytest.raises(ConfigError, match="has no speech output"):
        nt.native_for("vl", cfg)


def test_the_server_start_sets_tts_only_when_it_is_unset(files):
    from gmlx.config import ModelCfg, ServerCfg
    from gmlx.serve import server as srv

    def start(**server_kw):
        path, mmproj = files["omni"]
        cfg = ServerCfg(models={"omni": ModelCfg(id="omni", path=path,
                                                 mmproj=mmproj)}, **server_kw)
        srv._resolve_service_models(cfg, SimpleNamespace(stt=None, tts=None))
        return cfg.tts

    assert start() == "model:omni"
    assert start(tts=False) is None
    assert start(tts="omni") == "model:omni"


def test_a_false_tts_in_the_config_file_stays_false(tmp_path):
    from gmlx.config import load_config

    path = write_lfm3_header(tmp_path / "m.gguf")
    for value, want in (("false", False), ("kokoro", "kokoro")):
        cfg_file = tmp_path / f"{value}.yaml"
        cfg_file.write_text(
            f"server:\n  tts: {value}\nmodels:\n  m:\n    path: {path}\n")
        assert load_config(str(cfg_file)).tts == want
    cfg_file = tmp_path / "unset.yaml"
    cfg_file.write_text(f"models:\n  m:\n    path: {path}\n")
    assert load_config(str(cfg_file)).tts is None


def test_a_served_model_needs_no_extra(served):
    cfg = _cfg("omni")
    assert not nt.needs_mlx_audio(None, cfg)
    assert not nt.needs_mlx_audio(False, cfg)
    assert not nt.needs_mlx_audio("omni", cfg)
    assert not nt.needs_mlx_audio("model:omni", cfg)
    assert nt.needs_mlx_audio("kokoro", cfg)


def test_the_models_list_names_what_each_model_takes_and_gives(files):
    from gmlx.serve.patches.routes import _modalities

    def entry(name):
        path, mmproj = files[name]
        return _modalities(SimpleNamespace(path=path, mmproj=mmproj))

    assert entry("omni") == ["text", "vision", "audio", "speech"]
    assert entry("vl") == ["text", "vision"]
    assert entry("text") == ["text"]
    # Another family's speech generator is not one the server runs.
    assert entry("other") == ["text", "vision", "audio"]


# -- which model answers ------------------------------------------------------


def test_the_configured_names_and_the_model_id_reach_the_model(served):
    for name in ("", "tts-1", "tts-1-hd", "gpt-4o-mini-tts"):
        assert nt.target(name, "model:omni") == "omni"
        assert nt.target(name, "mlx-community/Kokoro-82M-bf16") is None
    # A served model that speaks answers by its id beside another service.
    assert nt.target("omni", "mlx-community/Kokoro-82M-bf16") == "omni"
    assert nt.target("kokoro", "mlx-community/Kokoro-82M-bf16") is None


def test_a_model_that_does_not_speak_is_refused_by_name(served):
    with pytest.raises(TTSRequestError, match="has no speech output") as e:
        nt.target("vl", "model:omni")
    assert e.value.status_code == 400
    with pytest.raises(TTSRequestError, match="is not a served model"):
        nt.target("kokoro", "model:omni")


# -- request fields -----------------------------------------------------------


def test_the_defaults_are_the_first_voice_mp3_and_no_language(monkeypatch):
    monkeypatch.setattr(media_programs, "find", lambda name: "/bin/ffmpeg")
    f = nt.fields({"input": "Hello there."})
    assert (f.text, f.voice, f.language) == ("Hello there.", "female", None)
    assert (f.format, f.sse, f.seed) == ("mp3", False, None)


def test_a_voice_is_one_of_the_models_two():
    assert nt.voice_of(None) == "female" and nt.voice_of("  ") == "female"
    assert nt.voice_of("Male") == "male"
    with pytest.raises(ValueError, match="which are female and male"):
        nt.voice_of("alloy")
    with pytest.raises(TTSRequestError, match="'alloy'") as e:
        nt.fields({"input": "Hi.", "voice": "alloy",
                   "response_format": "wav"})
    assert e.value.status_code == 400


def test_a_language_is_a_code_or_a_name_the_speech_prompt_takes():
    assert nt.language_of(None) is None and nt.language_of("") is None
    assert nt.language_of("de") == "de" and nt.language_of("German") == "de"
    assert nt.language_of("EN_GB") == "en-gb"
    # The model transcribes Hindi and has no speech prompt for it.
    with pytest.raises(ValueError, match="not one this model speaks"):
        nt.language_of("hi")


def test_every_speech_language_is_a_code():
    from gmlx.models.lfm3 import speak
    from gmlx.serve.native_stt_lfm3 import LANGUAGES

    assert len(set(speak.TTS_LANGUAGES)) == len(speak.TTS_LANGUAGES)
    for code in speak.TTS_LANGUAGES:
        assert code.split("-")[0] in LANGUAGES


def test_the_text_is_sanitized_and_must_hold_words():
    f = nt.fields({"input": "# Title\n\n**Bold** words.",
                   "response_format": "wav"})
    assert "#" not in f.text and "*" not in f.text and "Bold" in f.text
    with pytest.raises(TTSRequestError, match="'input' is required"):
        nt.fields({"input": "   "})
    with pytest.raises(TTSRequestError, match="no speakable text"):
        nt.fields({"input": "***"})


@pytest.mark.parametrize("speed", [1.5, "0.5", "fast"])
def test_the_model_speaks_at_one_speed(speed):
    with pytest.raises(TTSRequestError) as e:
        nt.fields({"input": "Hi.", "speed": speed, "response_format": "wav"})
    assert e.value.status_code == 400
    for same in (None, 1, 1.0, "1.0", ""):
        nt.fields({"input": "Hi.", "speed": same, "response_format": "wav"})


def test_a_format_that_needs_ffmpeg_is_refused_without_it(monkeypatch):
    monkeypatch.setattr(media_programs, "find", lambda name: None)
    for fmt in ("pcm", "wav"):
        assert nt.fields({"input": "Hi.", "response_format": fmt}).format \
            == fmt
    for body in ({"input": "Hi."}, {"input": "Hi.", "response_format": "mp3"},
                 {"input": "Hi.", "response_format": "opus"}):
        with pytest.raises(TTSRequestError, match="needs ffmpeg") as e:
            nt.fields(body)
        assert "pcm, wav" in str(e.value)
    with pytest.raises(TTSRequestError, match="unsupported response_format"):
        nt.fields({"input": "Hi.", "response_format": "aac"})


def test_an_event_stream_sends_pcm(monkeypatch):
    monkeypatch.setattr(media_programs, "find", lambda name: None)
    f = nt.fields({"input": "Hi.", "stream_format": "sse"})
    assert f.sse and f.format == "pcm"
    with pytest.raises(TTSRequestError, match="sends 16-bit pcm"):
        nt.fields({"input": "Hi.", "stream_format": "sse",
                   "response_format": "wav"})
    with pytest.raises(TTSRequestError, match="unsupported stream_format"):
        nt.fields({"input": "Hi.", "stream_format": "chunks"})


def test_a_seed_is_an_integer():
    assert nt.fields({"input": "Hi.", "seed": 7,
                      "response_format": "wav"}).seed == 7
    for seed in ("7", 1.5, True):
        with pytest.raises(TTSRequestError, match="seed must be an integer"):
            nt.fields({"input": "Hi.", "seed": seed,
                       "response_format": "wav"})


def test_the_frame_budget_is_twice_what_the_text_should_take():
    assert nt.frame_budget("") == 125
    assert nt.frame_budget("a" * 100) == 125 + 160
    # A wide character takes longer to say than a letter.
    assert nt.frame_budget("\u4eca" * 100) == 125 + 600
    assert nt.frame_budget("\u05e9" * 100) == 125 + 240


# -- long texts ---------------------------------------------------------------

_PROSE = (
    "The ferry leaves the island at six each morning, when the water is "
    "still flat. Most of the passengers are teachers and nurses who work "
    "on the mainland! Do they know each other? \"Well enough,\" says the "
    "captain. ") * 6


def test_a_short_text_is_one_piece():
    assert nt.split_text("Hello there. How are you?") == [
        "Hello there. How are you?"]
    assert nt.split_text("") == []


def test_a_long_text_splits_at_sentence_ends_and_loses_nothing():
    pieces = nt.split_text(_PROSE)
    assert len(pieces) > 2
    assert " ".join(pieces) == _PROSE.strip()
    for piece in pieces:
        assert nt.estimated_frames(piece) <= nt.PIECE_FRAMES
        assert piece[-1] in ".!?\""


def test_wide_text_splits_with_no_spaces():
    text = "\u4eca\u5929\u5929\u6c14\u5f88\u597d\u3002" * 40
    pieces = nt.split_text(text)
    assert "".join(pieces) == text and len(pieces) > 2
    for piece in pieces:
        assert nt.estimated_frames(piece) <= nt.PIECE_FRAMES
        assert piece.endswith("\u3002")


def test_a_sentence_too_long_for_one_piece_splits_inside():
    clauses = ", ".join(["a clause of some words"] * 40) + "."
    pieces = nt.split_text(clauses)
    assert " ".join(pieces) == clauses
    assert all(p.endswith((",", ".")) for p in pieces)
    words = ("word " * 400).strip()
    assert " ".join(nt.split_text(words)) == words
    solid = "x" * 1000
    pieces = nt.split_text(solid)
    assert "".join(pieces) == solid
    assert all(nt.estimated_frames(p) <= nt.PIECE_FRAMES for p in pieces)


# -- formats ------------------------------------------------------------------


def _pcm(seconds=0.5):
    t = np.arange(int(24000 * seconds)) / 24000
    return speech_chat.pcm16(0.3 * np.sin(2 * np.pi * 220 * t))


def test_pcm_is_the_samples_and_wav_wraps_them():
    pcm = _pcm()
    assert nt.encode(pcm, 24000, "pcm") == pcm
    assert nt.encode(pcm, 24000, "pcm16") == pcm
    with wave.open(io.BytesIO(nt.encode(pcm, 24000, "wav"))) as w:
        assert (w.getframerate(), w.getnchannels(), w.getsampwidth()) == (
            24000, 1, 2)
        assert w.readframes(w.getnframes()) == pcm


@pytest.mark.parametrize("fmt,magic", [("mp3", (b"ID3", b"\xff")),
                                       ("flac", (b"fLaC",)),
                                       ("opus", (b"OggS",))])
def test_ffmpeg_writes_the_other_formats(fmt, magic):
    if media_programs.find("ffmpeg") is None:
        pytest.skip("needs ffmpeg")
    out = nt.encode(_pcm(), 24000, fmt)
    assert out.startswith(magic)


# -- the pass -----------------------------------------------------------------


def _event(**delta_audio):
    return "data: " + json.dumps({"choices": [{"delta": {
        "role": "assistant", "audio": delta_audio}}]}) + "\n\n"


class _Stream:
    """A streaming response of the chat route."""

    def __init__(self, parts):
        self.parts, self.closed = parts, False

    @property
    def body_iterator(self):
        return self

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self.parts:
            raise StopAsyncIteration
        return self.parts.pop(0)

    async def aclose(self):
        self.closed = True


class _Chat:
    """The chat route: records each request and what the speech route set
    for it, and answers ``parts`` as a stream."""

    def __init__(self, parts=(), error=None):
        self.parts, self.error = list(parts), error
        self.calls, self.only, self.stream = [], [], None

    async def __call__(self, request, http_request):
        self.calls.append(request)
        self.only.append(speech_chat.SPEECH_ONLY.get())
        if self.error is not None:
            raise self.error
        self.stream = _Stream(list(self.parts))
        return self.stream


def _two_chunks():
    a, b = _pcm(0.1), _pcm(0.2)
    usage = "data: " + json.dumps({"choices": [], "usage": {
        "prompt_tokens": 30, "completion_tokens": 12}}) + "\n\n"
    whole = (_event(id="audio_1", data=base64.b64encode(a).decode())
             + _event(id="audio_1", data=base64.b64encode(b).decode())
             + _event(id="audio_1", expires_at=9) + usage + "data: [DONE]\n\n")
    # The stream arrives in pieces that split an event.
    return a, b, [whole[:57], whole[57:300].encode(), whole[300:]]


def _answer(monkeypatch, chat, **body):
    monkeypatch.setattr(nt, "_chat_endpoint", lambda: chat)
    body.setdefault("input", "Hello there.")
    return asyncio.run(nt.answer(None, body, "omni"))


def _stream(monkeypatch, chat, **body):
    """The events of an event-stream answer, and its media type."""
    monkeypatch.setattr(nt, "_chat_endpoint", lambda: chat)
    body.setdefault("input", "Hello there.")

    async def read():
        response = await nt.answer(None, body, "omni")
        return [json.loads(part[6:])
                async for part in response.body_iterator], response.media_type

    return asyncio.run(read())


def test_the_pass_is_one_chat_request_with_the_text_alone(monkeypatch):
    a, b, parts = _two_chunks()
    chat = _Chat(parts)
    _answer(monkeypatch, chat, response_format="pcm", voice="male",
            language="de", seed=3)
    (request,) = chat.calls
    assert request.model == "omni" and request.stream
    assert [(m.role, m.content) for m in request.messages] == [
        ("user", "Hello there.")]
    assert request.max_tokens == nt.frame_budget("Hello there.")
    assert request.seed == 3 and request.temperature == 0.0
    # The chat route reads the voice and language from what the speech
    # route set, and nothing is left set after the request.
    assert chat.only == [speech_chat.SpeechOnly("Hello there.", "male", "de")]
    assert speech_chat.SPEECH_ONLY.get() is None


def test_a_long_text_is_one_chat_request_per_piece(monkeypatch):
    a, b, parts = _two_chunks()
    text = _PROSE.strip()
    pieces = nt.split_text(text)

    class Chat(_Chat):
        async def __call__(self, request, http_request):
            self.parts = list(_two_chunks()[2])
            return await super().__call__(request, http_request)

    chat = Chat()
    response = _answer(monkeypatch, chat, input=text, response_format="pcm",
                       seed=5)
    assert [r.messages[0].content for r in chat.calls] == pieces
    assert [r.seed for r in chat.calls] == [5 + i for i in range(len(pieces))]
    assert [o.text for o in chat.only] == pieces
    assert response.body == (a + b) * len(pieces)
    events, _ = _stream(monkeypatch, Chat(), input=text, stream_format="sse")
    assert events[-1]["usage"] == {
        "input_tokens": 30 * len(pieces), "output_tokens": 12 * len(pieces),
        "total_tokens": 42 * len(pieces)}


def test_the_answer_is_the_audio_in_the_asked_format(monkeypatch):
    a, b, parts = _two_chunks()
    chat = _Chat(parts)
    response = _answer(monkeypatch, chat, response_format="pcm")
    assert response.body == a + b and response.media_type == "audio/pcm"
    assert chat.stream.closed
    response = _answer(monkeypatch, _Chat(_two_chunks()[2]),
                       response_format="wav")
    assert response.media_type == "audio/wav"
    with wave.open(io.BytesIO(response.body)) as w:
        assert w.readframes(w.getnframes()) == a + b


def test_an_event_stream_sends_each_chunk_then_the_usage(monkeypatch):
    a, b, parts = _two_chunks()
    events, media_type = _stream(monkeypatch, _Chat(parts),
                                 stream_format="sse")
    assert [e["type"] for e in events] == [
        "speech.audio.delta", "speech.audio.delta", "speech.audio.done"]
    assert base64.b64decode(events[0]["audio"]) == a
    assert base64.b64decode(events[1]["audio"]) == b
    assert events[2]["usage"] == {"input_tokens": 30, "output_tokens": 12,
                                  "total_tokens": 42}
    assert media_type == "text/event-stream"


def test_a_refusal_of_the_chat_route_keeps_its_status(monkeypatch):
    from fastapi import HTTPException

    refused = _Chat(error=HTTPException(400, "the text is too long"))
    for body in ({"response_format": "wav"}, {"stream_format": "sse"}):
        with pytest.raises(TTSRequestError, match="too long") as e:
            _answer(monkeypatch, refused, **body)
        assert e.value.status_code == 400
    assert speech_chat.SPEECH_ONLY.get() is None
    with pytest.raises(HTTPException):
        _answer(monkeypatch, _Chat(error=HTTPException(500, "boom")),
                response_format="wav")


def test_an_error_event_fails_the_request(monkeypatch):
    parts = ['data: {"error": "replay unsupported: a spoken reply"}\n\n']
    with pytest.raises(RuntimeError, match="replay unsupported"):
        _answer(monkeypatch, _Chat(parts), response_format="wav")
    with pytest.raises(RuntimeError, match="no audio"):
        _answer(monkeypatch, _Chat(["data: [DONE]\n\n"]),
                response_format="wav")


def test_an_error_inside_an_event_stream_is_an_event(monkeypatch):
    a = _pcm(0.1)
    parts = [_event(id="x", data=base64.b64encode(a).decode()),
             'data: {"error": "the row failed"}\n\n']
    events, _ = _stream(monkeypatch, _Chat(parts), stream_format="sse")
    assert [e["type"] for e in events] == ["speech.audio.delta", "error"]
    assert events[1]["error"]["message"] == "the row failed"
