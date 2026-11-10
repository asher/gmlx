"""The transcription and translation routes on a served LFM3 model that
takes audio: which models have an adapter, which model answers, the form
fields, the chat turn the route sends, silence and formats. The chat route
and the upload decode are stubbed."""

from __future__ import annotations

import asyncio
import base64
import io
import os
import sys
import wave
from types import SimpleNamespace

import numpy as np
import pytest
from starlette.datastructures import FormData

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import gmlx.load.discovery as disc  # noqa: E402
from gmlx.serve import native_stt, stt  # noqa: E402
from gmlx.serve import native_stt_lfm3 as lfm3  # noqa: E402
from gmlx.serve.stt import STTRequestError  # noqa: E402
from lfm3_header import write_lfm3_header, write_projector_header  # noqa: E402


def _text_header(path, arch: str) -> str:
    """A text GGUF of another family: its arch and one tensor."""
    import gguf

    w = gguf.GGUFWriter(str(path), arch)
    w.add_tensor("token_embd.weight", np.zeros((2, 2), dtype=np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return str(path)


@pytest.fixture()
def files(tmp_path, monkeypatch):
    """Header-only models: LFM3 with an audio projector (`omni`), LFM3
    with a vision projector (`vl`), LFM3 alone (`text`), Gemma 4 with an
    audio projector (`gemma`) and a Nemotron pairing (`nemotron`)."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setattr(disc, "_HEADER_DISK", None)
    lfm = write_lfm3_header(tmp_path / "lfm3.gguf", templates=("tts",))
    hears = write_projector_header(tmp_path / "mmproj-omni.gguf", vision=True,
                                   audio=True, speech=True)
    sees = write_projector_header(tmp_path / "mmproj-vl.gguf", vision=True)
    return {
        "omni": (lfm, hears),
        "vl": (lfm, sees),
        "text": (lfm, None),
        "gemma": (_text_header(tmp_path / "gemma.gguf", "gemma4"), hears),
        "nemotron": (_text_header(tmp_path / "decoder.gguf", "nemotron_h"),
                     write_projector_header(tmp_path / "encoder.gguf",
                                            arch="pee")),
    }


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


# -- which models have an adapter --------------------------------------------


def test_an_lfm3_model_that_takes_audio_has_the_lfm3_adapter(files):
    assert stt.stt_adapter(*files["omni"]) == "lfm3"
    assert stt.stt_adapter(*files["nemotron"]) == "nemotron"


def test_other_models_with_an_audio_encoder_have_none(files):
    for name in ("gemma", "vl", "text"):
        assert stt.stt_adapter(*files[name]) is None
    assert stt.stt_adapter(None, files["omni"][1]) is None


def test_the_served_models_with_an_adapter_answer_stt(served):
    cfg = _cfg("text", "gemma", "omni", "vl", "nemotron")
    assert stt.native_models(cfg) == ["omni", "nemotron"]


def test_an_unset_stt_picks_the_lfm3_model(served):
    assert stt.auto_native(_cfg("text", "omni", "gemma")) == "model:omni"
    assert stt.auto_native(_cfg("gemma", "vl")) is None
    cfg = _cfg("omni", "nemotron", default="omni")
    assert stt.auto_native(cfg) == "model:omni"


def test_the_server_start_sets_stt_only_when_it_is_unset(files):
    from gmlx.config import ModelCfg, ServerCfg
    from gmlx.serve import server as srv

    def start(**server_kw):
        path, mmproj = files["omni"]
        cfg = ServerCfg(models={"omni": ModelCfg(id="omni", path=path,
                                                 mmproj=mmproj)}, **server_kw)
        srv._resolve_service_models(cfg, SimpleNamespace(stt=None, tts=None))
        return cfg.stt

    assert start() == "model:omni"
    assert start(stt=False) is None


# -- which model answers -----------------------------------------------------


def test_the_configured_names_and_the_model_id_reach_the_lfm3_model(served):
    for name in ("", "whisper-1", "default", "omni"):
        assert native_stt.target(name, "model:omni") == "omni"
    assert native_stt.adapter("omni") == "lfm3"
    assert native_stt.adapter("nemotron") == "nemotron"
    assert native_stt.adapter("nope") is None


def test_a_model_without_an_adapter_is_refused_by_name(served):
    for name in ("gemma", "vl", "text"):
        assert not native_stt.hears(name)
        with pytest.raises(STTRequestError, match="takes no audio input"):
            native_stt.target(name, "model:omni")


# -- form fields -------------------------------------------------------------


def _form(**fields) -> FormData:
    items = []
    for k, v in fields.items():
        for one in (v if isinstance(v, list) else [v]):
            items.append((k.replace("__", "[]"), one))
    return FormData(items)


def _fields(task="transcribe", **kw):
    return lfm3._fields(_form(**kw), task)


def test_the_defaults_are_json_greedy_and_no_language():
    assert _fields() == {"format": "json", "temperature": 0.0,
                         "language": None}
    assert _fields(response_format="TEXT", temperature="0.4",
                   language="de") == {"format": "text", "temperature": 0.4,
                                      "language": "German"}


@pytest.mark.parametrize("fmt", ["verbose_json", "srt", "vtt",
                                 "diarized_json"])
def test_a_timed_format_is_refused_with_the_reason(fmt):
    with pytest.raises(STTRequestError, match="no word times") as e:
        _fields(response_format=fmt)
    assert e.value.status_code == 400 and "json, text" in str(e.value)


def test_word_times_and_speakers_are_refused():
    with pytest.raises(STTRequestError, match="no word times"):
        _fields(timestamp_granularities__=["word"])
    with pytest.raises(STTRequestError, match="no word times"):
        _fields(timestamp_granularities="segment")
    with pytest.raises(STTRequestError, match="names no speakers"):
        _fields(diarization="true")
    assert _fields(diarization="false")["format"] == "json"


def test_a_bad_temperature_is_refused():
    with pytest.raises(STTRequestError, match="not a number"):
        _fields(temperature="warm")
    with pytest.raises(STTRequestError, match="between 0 and 2"):
        _fields(temperature="2.5")


def test_a_language_is_a_code_a_locale_or_a_name():
    assert lfm3.language_name("") is None
    assert lfm3.language_name("ja") == "Japanese"
    assert lfm3.language_name("pt-BR") == "Portuguese"
    assert lfm3.language_name("zh_CN") == "Chinese"
    assert lfm3.language_name(" French ") == "French"
    with pytest.raises(STTRequestError, match="leave it out") as e:
        lfm3.language_name("xx")
    assert e.value.status_code == 400


def test_every_listed_language_has_one_name():
    names = list(lfm3.LANGUAGES.values())
    assert len(names) == len(set(names)) == 23
    assert all(len(code) == 2 for code in lfm3.LANGUAGES)


# -- the chat turn -----------------------------------------------------------


def test_the_instruction_is_the_cards_prompt():
    assert lfm3.instruction("transcribe", "German") == "Perform ASR in German."
    assert lfm3.instruction("transcribe", None) == "Perform ASR."
    assert lfm3.instruction("translate", "Spanish") == (
        "Please translate the following spoken Spanish into English text.")
    assert lfm3.instruction("translate", None) == (
        "Please translate the following speech into English text.")


def test_the_output_budget_follows_the_clip_length():
    rate = lfm3.SAMPLE_RATE
    assert lfm3.output_budget("omni", rate * 2) == 64
    assert lfm3.output_budget("omni", rate * 60) == 720
    assert lfm3.output_budget("omni", rate * 600) == 7200


def test_a_long_clip_gets_the_room_the_context_leaves():
    from gmlx.load.config_synth import LFM3_SPEECH_CONTEXT
    from gmlx.models.lfm3.audio import num_audio_tokens

    samples = lfm3.SAMPLE_RATE * 60 * 25
    room = (LFM3_SPEECH_CONTEXT - num_audio_tokens(samples)
            - lfm3.TEMPLATE_TOKENS)
    assert lfm3.output_budget("omni", samples) == room < 25 * 60 * 12


def test_a_clip_that_leaves_no_room_is_refused_with_the_limit():
    with pytest.raises(STTRequestError, match="Send it in parts") as e:
        lfm3.output_budget("omni", lfm3.SAMPLE_RATE * 60 * 40)
    assert e.value.status_code == 400
    assert "40 minutes long" in str(e.value) and "omni takes" in str(e.value)


def _speech(seconds=2.0):
    rng = np.random.default_rng(0)
    return (0.1 * rng.standard_normal(int(16000 * seconds))).astype(np.float32)


def test_the_clip_travels_as_a_16_bit_wav():
    audio = np.array([0.0, 0.5, -0.5, 2.0, -2.0], np.float32)
    with wave.open(io.BytesIO(base64.b64decode(lfm3._wav(audio)))) as f:
        assert (f.getnchannels(), f.getsampwidth(), f.getframerate()) == (
            1, 2, 16000)
        pcm = np.frombuffer(f.readframes(f.getnframes()), "<i2")
    assert pcm.tolist() == [0, 16384, -16384, 32767, -32767]


def test_the_chat_turn_is_a_system_prompt_and_a_clip_alone():
    audio = _speech()
    request = lfm3.chat_request("omni", "transcribe",
                                _fields(language="fr"), audio)
    body = request.model_dump()
    assert body["model"] == "omni" and body["stream"] is False
    system, user = body["messages"]
    assert system["role"] == "system"
    assert system["content"] == "Perform ASR in French."
    assert user["role"] == "user" and len(user["content"]) == 1
    part = user["content"][0]
    assert part["type"] == "input_audio"
    assert part["input_audio"] == {"data": lfm3._wav(audio), "format": "wav"}
    assert body["max_tokens"] == 64 and body["temperature"] == 0.0
    assert (body["top_p"], body["top_k"], body["min_p"]) == (1.0, 0, 0.0)
    assert body["repetition_penalty"] == 1.0


class _Chat:
    """The chat route: records each request and answers ``text``."""

    def __init__(self, text="", error=None):
        self.text, self.error, self.calls = text, error, []

    async def __call__(self, request, http_request):
        self.calls.append(request)
        if self.error is not None:
            raise self.error
        return {"choices": [{"message": {"content": self.text}}],
                "usage": {"prompt_tokens": 9}}


def _run(monkeypatch, chat, audio, task="transcribe", **kw):
    monkeypatch.setattr(lfm3, "_chat_endpoint", lambda: chat)
    return asyncio.run(lfm3.run(None, "omni", audio, task,
                                _fields(task, **kw)))


def test_silence_never_reaches_the_model(monkeypatch):
    chat = _Chat("words")
    for audio in (np.zeros(32000, np.float32), np.zeros(0, np.float32),
                  np.full(16000, 5e-4, np.float32)):
        assert _run(monkeypatch, chat, audio) == {"text": ""}
    assert not chat.calls


def test_the_answer_is_the_models_text_trimmed(monkeypatch):
    chat = _Chat("  hello there.\n")
    assert _run(monkeypatch, chat, _speech()) == {"text": "hello there."}
    assert chat.calls[0].messages[0].content == "Perform ASR."
    _run(monkeypatch, chat, _speech(), task="translate", language="es")
    assert "spoken Spanish into English" in chat.calls[1].messages[0].content


def test_a_client_error_of_the_chat_route_keeps_its_status(monkeypatch):
    from fastapi import HTTPException

    refused = _Chat(error=HTTPException(413, "the prompt is too long"))
    with pytest.raises(STTRequestError, match="too long") as e:
        _run(monkeypatch, refused, _speech())
    assert e.value.status_code == 413
    with pytest.raises(HTTPException):
        _run(monkeypatch, _Chat(error=HTTPException(500, "boom")), _speech())


def test_the_two_formats():
    assert lfm3.format_result({"text": "hi"}, {"format": "json"}) == (
        {"text": "hi"}, "application/json")
    assert lfm3.format_result({"text": "hi"}, {"format": "text"}) == (
        "hi\n", "text/plain")


def test_an_empty_upload_is_refused():
    with pytest.raises(STTRequestError, match="'file' is required"):
        asyncio.run(lfm3.answer(None, _form(), "omni", b"", "a.wav",
                                "transcribe"))


# -- the routes --------------------------------------------------------------


@pytest.fixture()
def client(served, monkeypatch):
    """Both routes installed for ``model:omni``, with the chat route and
    the upload decode stubbed. Yields ``(post, chat)``."""
    pytest.importorskip("multipart")
    from fastapi.testclient import TestClient

    from gmlx.serve.patches import routes

    chat = _Chat("the transcript")
    monkeypatch.setattr(lfm3, "_chat_endpoint", lambda: chat)
    monkeypatch.setattr(lfm3, "decode", lambda data, name: _speech())
    monkeypatch.setattr(native_stt, "run", lambda *a, **k: pytest.fail(
        "the Nemotron adapter ran"))
    monkeypatch.setattr(stt, "run_transcription", lambda *a, **k: pytest.fail(
        "the Whisper path ran"))
    routes.install_audio_transcription_route("model:omni")
    routes.install_audio_translation_route("model:omni")
    app = __import__("gmlx._vendor.mlx_vlm.server.app", fromlist=["app"]).app
    http = TestClient(app)

    def post(path, **data):
        return http.post(path, data=data,
                         files={"file": ("a.wav", b"RIFF", "audio/wav")})

    yield post, chat
    routes.install_audio_transcription_route(None)
    routes.install_audio_translation_route(None)


@pytest.mark.parametrize("model", ["", "whisper-1", "omni"])
def test_the_transcription_route_reaches_the_lfm3_adapter(client, model):
    post, chat = client
    r = post("/v1/audio/transcriptions", model=model)
    assert r.status_code == 200 and r.json() == {"text": "the transcript"}
    assert chat.calls[0].model == "omni"
    assert chat.calls[0].messages[0].content == "Perform ASR."


def test_the_translation_route_reaches_the_lfm3_adapter(client):
    post, chat = client
    r = post("/v1/audio/translations", model="whisper-1",
             response_format="text")
    assert r.status_code == 200 and r.text == "the transcript\n"
    assert chat.calls[0].messages[0].content == (
        "Please translate the following speech into English text.")


def test_the_routes_refuse_what_the_model_cannot_give(client):
    post, chat = client
    r = post("/v1/audio/transcriptions", response_format="verbose_json")
    assert r.status_code == 400 and "no word times" in r.text
    r = post("/v1/audio/transcriptions", language="xx")
    assert r.status_code == 400 and "leave it out" in r.text
    r = post("/v1/audio/transcriptions", model="gemma")
    assert r.status_code == 400 and "takes no audio input" in r.text
    assert not chat.calls


def test_stt_off_leaves_no_route(served):
    pytest.importorskip("multipart")
    from fastapi.testclient import TestClient

    from gmlx.serve.patches import routes

    routes.install_audio_transcription_route(None)
    routes.install_audio_translation_route(None)
    app = __import__("gmlx._vendor.mlx_vlm.server.app", fromlist=["app"]).app
    for path in ("/v1/audio/transcriptions", "/v1/audio/translations"):
        r = TestClient(app).post(
            path, files={"file": ("a.wav", b"RIFF", "audio/wav")})
        assert r.status_code == 404
