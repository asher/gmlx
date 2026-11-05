"""The transcription and translation routes on a served model that hears
audio through its own speech encoder: form fields, which model answers,
the chat turn the route sends, silence, cleanup, formats and the
``server.stt`` setting. The chat route and the decode are stubbed."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import numpy as np
import pytest
from starlette.datastructures import FormData

from gmlx.config import ConfigError, build_config
from gmlx.serve import native_stt, stt
from gmlx.serve.stt import STTRequestError


def _form(**fields) -> FormData:
    items = []
    for k, v in fields.items():
        for one in (v if isinstance(v, list) else [v]):
            items.append((k.replace("__", "[]").replace("_dash_", "-"), one))
    return FormData(items)


def _fields(task="transcribe", **kw):
    return native_stt._fields(_form(**kw), task)


# -- form fields -------------------------------------------------------------


def test_the_formats_of_each_task():
    assert _fields()["format"] == "json"
    for fmt in native_stt.TRANSCRIBE_FORMATS:
        assert _fields(response_format=fmt)["format"] == fmt
    with pytest.raises(STTRequestError, match="unsupported"):
        _fields("translate", response_format="diarized_json")
    with pytest.raises(STTRequestError, match="unsupported"):
        _fields(response_format="xml")


def test_speakers_need_a_timed_format_and_a_transcription():
    assert _fields(response_format="diarized_json")["speakers"]
    assert _fields(response_format="vtt", diarization="true")["speakers"]
    assert not _fields(response_format="vtt", diarization="false")["speakers"]
    for fmt in ("json", "text"):
        with pytest.raises(STTRequestError, match="diarization needs"):
            _fields(response_format=fmt, diarization="true")
    with pytest.raises(STTRequestError, match="no speakers"):
        _fields("translate", response_format="srt", diarization="true")
    with pytest.raises(STTRequestError, match="true or false"):
        _fields(response_format="srt", diarization="maybe")


def test_speaker_enrollment_and_diarized_granularities_are_refused():
    with pytest.raises(STTRequestError, match="known_speaker_names"):
        _fields(response_format="diarized_json", known_speaker_names__="Ann")
    with pytest.raises(STTRequestError, match="known_speaker_references"):
        _fields(known_speaker_references__="data:audio/wav;base64,AA==")
    with pytest.raises(STTRequestError, match="diarized_json"):
        _fields(response_format="diarized_json", timestamp_granularities__="word")
    with pytest.raises(STTRequestError, match="word and segment"):
        _fields(response_format="verbose_json", timestamp_granularities__="char")


def test_the_other_fields():
    f = _fields(response_format="verbose_json",
                timestamp_granularities__=["segment", "word"],
                temperature="0.4", language="pt-BR", prompt="ignored",
                chunking_strategy="auto")
    assert f["words"] and f["temperature"] == 0.4 and f["language"] == "pt-BR"
    assert _fields()["temperature"] == 0.0
    assert _fields("translate", language="de")["language"] == ""
    with pytest.raises(STTRequestError, match="not a number"):
        _fields(temperature="warm")
    with pytest.raises(STTRequestError, match="between 0 and 2"):
        _fields(temperature="3")


# -- which model answers -----------------------------------------------------


@pytest.fixture()
def served(monkeypatch):
    """Two served models: `ears` hears audio, `text` does not."""
    from gmlx.serve import bridge_vlm

    def resolve(name, **kw):
        if name not in ("ears", "text"):
            raise bridge_vlm.ModelNotFound(name, {})
        return f"/m/{name}.gguf", SimpleNamespace()

    monkeypatch.setattr(bridge_vlm, "resolve_request_model", resolve)
    monkeypatch.setattr(native_stt, "hears", lambda name: name == "ears")


def test_the_configured_names_go_to_server_stt(served):
    for name in ("", "whisper-1", "default", "gpt-4o-transcribe-diarize"):
        assert native_stt.target(name, "model:ears") == "ears"
        assert native_stt.target(name, "mlx-community/whisper") is None


def test_a_request_may_name_a_served_model(served):
    assert native_stt.target("ears", "mlx-community/whisper") == "ears"
    with pytest.raises(STTRequestError, match="takes no audio input"):
        native_stt.target("text", "model:ears")
    with pytest.raises(STTRequestError, match="not a served model"):
        native_stt.target("nope", "model:ears")
    assert native_stt.target("whisper-large", "mlx-community/whisper") is None


# -- the chat turn -----------------------------------------------------------


def test_the_chat_turn_is_the_cards_instruction_greedy_and_budgeted():
    from gmlx.models.nemotron_h import transcription as tr

    req = native_stt.chat_request("ears", "transcribe", _fields(language="fr"), 60.0)
    text, audio = req.messages[0].content
    assert text["text"] == tr.transcription_instruction("fr")
    assert audio["type"] == "input_audio"
    assert req.max_tokens == 384 and req.temperature == 0.0
    assert req.enable_thinking is False and not req.stream
    # Every field a profile could fill is the client's.
    for f in ("temperature", "top_p", "top_k", "min_p", "repetition_penalty",
              "presence_penalty", "frequency_penalty", "enable_thinking",
              "thinking_budget", "max_tokens"):
        assert f in req.model_fields_set
    req = native_stt.chat_request("ears", "transcribe",
                                  _fields(response_format="srt", diarization="1"), 9.0)
    assert req.messages[0].content[0]["text"] == tr.SPEAKERS
    req = native_stt.chat_request("ears", "translate", _fields("translate"), 9.0)
    assert req.messages[0].content[0]["text"] == (
        "Please translate the given speech to English.")


def test_an_hour_gets_the_cards_cap():
    req = native_stt.chat_request("ears", "transcribe", _fields(), 3600.0)
    assert req.max_tokens == 21_624


class _Chat:
    """The chat route: records each call and answers ``text``."""

    def __init__(self, text="", encoded=None):
        self.text, self.encoded, self.calls = text, encoded, []

    async def __call__(self, request, http_request):
        from gmlx.serve.patches import native_audio

        self.calls.append((request, native_audio.DECODED.get()))
        hook = native_audio.ON_ENCODED.get()
        if hook is not None and self.encoded is not None:
            hook(self.encoded)
        return {"choices": [{"message": {"content": self.text}}],
                "usage": {"prompt_tokens": 9}}


def _run(monkeypatch, chat, audio, **kw):
    monkeypatch.setattr(native_stt, "_chat_endpoint", lambda: chat)
    return asyncio.run(native_stt.run(None, "ears", audio, "transcribe",
                                      _fields(**kw)))


def _speech(seconds=2.0):
    rng = np.random.default_rng(0)
    return (0.1 * rng.standard_normal(int(16000 * seconds))).astype(np.float32)


def test_digital_silence_never_reaches_the_model(monkeypatch):
    chat = _Chat("words")
    out = _run(monkeypatch, chat, np.zeros(32000, np.float32))
    assert out == {"text": "", "duration": 2.0, "usage": None}
    assert not chat.calls


def test_no_speaker_heard_skips_the_decoder(monkeypatch):
    clip = SimpleNamespace(speaker_probs=np.full((25, 8), 0.2, np.float32))
    out = _run(monkeypatch, _Chat("noise words", encoded=[clip]), _speech())
    assert out["text"] == ""
    clip.speaker_probs[3, 1] = 0.9
    out = _run(monkeypatch, _Chat("hello", encoded=[clip]), _speech())
    assert out["text"] == "hello"


def test_the_route_hands_its_samples_to_the_chat_turn(monkeypatch):
    chat = _Chat("hi")
    audio = _speech()
    out = _run(monkeypatch, chat, audio)
    assert chat.calls[0][1][0] is audio
    assert out["usage"] == {"prompt_tokens": 9} and out["duration"] == 2.0


def test_the_transcript_is_cleaned_and_loses_its_tags(monkeypatch):
    raw = "<spk:0> Hi <ev:laugh> there. " + "ha" * 30 + " <sty:fast>ok"
    out = _run(monkeypatch, _Chat(raw), _speech())
    assert out["text"] == "<spk:0> Hi there. ha ok"
    out = _run(monkeypatch, _Chat(raw), _speech(), response_format="vtt",
               diarization="true")
    assert out["text"] == "Hi there. ha ok"


def test_a_chat_refusal_keeps_its_status(monkeypatch):
    from fastapi.responses import JSONResponse

    async def refuse(request, http_request):
        return JSONResponse(status_code=400, content={"error": {
            "message": "this request needs an estimated 2.00 GB"}})

    monkeypatch.setattr(native_stt, "_chat_endpoint", lambda: refuse)
    with pytest.raises(STTRequestError, match="2.00 GB") as e:
        asyncio.run(native_stt.run(None, "ears", _speech(), "transcribe",
                                   _fields()))
    assert e.value.status_code == 400


# -- formats -----------------------------------------------------------------


def _format(fmt, text="Hello there.", **kw):
    fields = _fields(response_format=fmt, **kw)
    return native_stt.format_result({"text": text, "duration": 12.3456,
                                     "usage": None}, "transcribe", fields)


def test_json_and_text():
    assert _format("json") == ({"text": "Hello there."}, "application/json")
    assert _format("text") == ("Hello there.\n", "text/plain")


def _aligned(fmt, words, error=None, text="Hi there. How are you? Fine.", **kw):
    fields = _fields(response_format=fmt, **kw)
    return native_stt.format_result(
        {"text": text, "duration": 12.3456, "usage": None,
         "alignment": {"words": words, "error": error}}, "transcribe", fields)[0]


def _w(word, start, end, speaker=0):
    return {"word": word, "start": start, "end": end, "speaker": speaker}


WORDS = [_w("Hi", 0.1, 0.3), _w("there.", 0.35, 0.7), _w("How", 0.8, 0.9, 1),
         _w("are", 0.95, 1.0, 1), _w("you?", 1.05, 1.3, 1),
         _w("Fine.", 3.0, 3.40049, 0)]


def test_segments_break_at_sentences_pauses_and_speakers():
    segs = native_stt.segments([_w("a", 0, 1), _w("b", 1.1, 2), _w("c", 3.5, 4),
                                _w("d.", 4.1, 5), _w("e", 5.1, 6), _w("f", 6.1, 7, 1)])
    assert [g["text"] for g in segs] == ["a b", "c d.", "e", "f"]
    turns = native_stt.segments(WORDS, by_turn=True)
    assert [(g["text"], g["speaker"]) for g in turns] == [
        ("Hi there.", 0), ("How are you?", 1), ("Fine.", 0)]


def test_verbose_json_words_and_segments_name_speakers_only_when_asked():
    body = _aligned("verbose_json", WORDS, timestamp_granularities__="word")
    assert body["segments"][0] == {"id": 0, "start": 0.1, "end": 0.7,
                                   "text": "Hi there."}
    assert body["words"][-1] == {"word": "Fine.", "start": 3.0, "end": 3.4}
    assert "alignment_error" not in body
    body = _aligned("verbose_json", WORDS, diarization="true",
                    timestamp_granularities__="word")
    assert [g.get("speaker") for g in body["segments"]] == [1, 2, 1]
    assert body["words"][2]["speaker"] == 2
    assert "speaker" not in _aligned("verbose_json", [_w("x", 0, 1, None)],
                                     diarization="true")["segments"][0]


def test_subtitles_name_speakers_per_format():
    vtt = _aligned("vtt", WORDS, diarization="true")
    assert "<v Speaker 1>Hi there." in vtt and "<v Speaker 2>How are you?" in vtt
    srt = _aligned("srt", WORDS + [_w("Yes.", 5.0, 5.5)], diarization="true")
    assert "Speaker 1: Hi there." in srt and "Speaker 2: How are you?" in srt
    assert "\nYes." in srt                   # the same turn: no second label
    assert "Speaker" not in _aligned("srt", WORDS)


def test_diarized_json_has_one_segment_per_turn_lettered_by_first_appearance():
    words = [_w("One.", 0, 1, 3), _w("Two.", 1, 2, 3), _w("Three.", 2, 3, 0),
             _w("Four.", 5, 6, 3)]
    body = _aligned("diarized_json", words)
    assert [(g["speaker"], g["text"], g["id"], g["type"]) for g in body["segments"]] == [
        ("A", "One. Two.", "seg_0", "transcript.text.segment"),
        ("B", "Three.", "seg_1", "transcript.text.segment"),
        ("A", "Four.", "seg_2", "transcript.text.segment")]
    assert native_stt._letter(25) == "Z" and native_stt._letter(26) == "AA"


def test_a_failed_alignment_keeps_the_text_and_names_why():
    for fmt in ("verbose_json", "diarized_json"):
        body = _aligned(fmt, [], "no_legal_path_in_anchor_band",
                        **({"timestamp_granularities__": "word"} if fmt == "verbose_json" else {}))
        assert body["text"] and body["segments"] == []
        assert body["alignment_error"] == "no_legal_path_in_anchor_band"
        assert body.get("words", []) == []
    srt = _aligned("srt", [], "no_legal_path_in_anchor_band")
    assert srt.startswith("1\n00:00:00,000 --> 00:00:12,346\nHi there.")


def test_a_translation_is_one_segment_over_the_clip():
    fields = _fields("translate", response_format="verbose_json")
    body, _ = native_stt.format_result({"text": "Hello.", "duration": 2.0,
                                        "usage": None}, "translate", fields)
    assert body["segments"] == [{"id": 0, "start": 0.0, "end": 2.0, "text": "Hello."}]
    assert body["task"] == "translate" and "alignment_error" not in body


def test_a_timed_transcription_aligns_the_cleaned_transcript(monkeypatch):
    from gmlx.serve.patches import native_audio

    seen = {}

    class Chat(_Chat):
        async def __call__(self, request, http_request):
            native_audio.CAPTURE.get().append("capture")
            return await super().__call__(request, http_request)

    async def align_transcript(http_request, model_id, capture, transcript):
        seen.update(capture=capture, transcript=transcript)
        return {"words": [], "error": None}

    monkeypatch.setattr(native_stt, "align_transcript", align_transcript)
    raw = "<spk:0> Hi " + "ha" * 30 + " <ev:laughter> there"
    out = _run(monkeypatch, Chat(raw), _speech(), response_format="srt")
    assert seen == {"capture": "capture",
                    "transcript": "<spk:0> Hi ha <ev:laughter> there"}
    assert out["alignment"] == {"words": [], "error": None}
    out = _run(monkeypatch, _Chat(raw), _speech(), response_format="json")
    assert "alignment" not in out and native_audio.CAPTURE.get() is None


def test_silence_gives_empty_results():
    body, _ = _format("verbose_json", text="")
    assert body["text"] == "" and body["segments"] == [] and body["duration"]
    assert "alignment_error" not in body
    body, _ = _format("diarized_json", text="")
    assert body["segments"] == [] and "alignment_error" not in body
    assert _format("srt", text="")[0] == ""


# -- the route ---------------------------------------------------------------


def test_the_route_answers_from_the_served_model(monkeypatch, capsys):
    pytest.importorskip("multipart")
    from fastapi.testclient import TestClient

    from gmlx.serve.patches import routes

    seen = {}

    async def answer(request, form, model_id, audio_bytes, filename, task):
        seen.update(model=model_id, bytes=audio_bytes, task=task)
        return {"text": "hi"}, "application/json"

    monkeypatch.setattr(native_stt, "target", lambda m, c: "ears")
    monkeypatch.setattr(native_stt, "answer", answer)
    monkeypatch.setattr(stt, "run_transcription", lambda *a, **k: pytest.fail(
        "the Whisper path ran"))
    routes.install_audio_transcription_route("model:ears")
    app = __import__("gmlx._vendor.mlx_vlm.server.app", fromlist=["app"]).app
    r = TestClient(app).post("/v1/audio/transcriptions",
                             files={"file": ("a.wav", b"RIFF", "audio/wav")},
                             data={"model": "whisper-1"})
    assert r.status_code == 200 and r.json() == {"text": "hi"}
    assert seen == {"model": "ears", "bytes": b"RIFF", "task": "transcribe"}
    assert "/v1/audio/transcriptions" in capsys.readouterr().out


def test_the_route_answers_a_client_error_with_its_status(monkeypatch):
    pytest.importorskip("multipart")
    from fastapi.testclient import TestClient

    from gmlx.serve.patches import routes

    def target(m, c):
        raise STTRequestError(400, "model 'text' takes no audio input")

    monkeypatch.setattr(native_stt, "target", target)
    routes.install_audio_translation_route("model:ears")
    app = __import__("gmlx._vendor.mlx_vlm.server.app", fromlist=["app"]).app
    r = TestClient(app).post("/v1/audio/translations",
                             files={"file": ("a.wav", b"RIFF", "audio/wav")},
                             data={"model": "text"})
    assert r.status_code == 400 and "no audio input" in json.dumps(r.json())


# -- server.stt --------------------------------------------------------------


def test_server_stt_has_three_states():
    assert build_config({"server": {}}).stt is None
    assert build_config({"server": {"stt": False}}).stt is False
    assert build_config({"server": {"stt": "ears"}}).stt == "ears"


@pytest.fixture()
def models(monkeypatch):
    def native(cfg):
        return [m for m in cfg.models if m.startswith("ears")]

    monkeypatch.setattr(stt, "native_models", native)

    def cfg(*ids, default=None, aliases=None):
        return SimpleNamespace(models={i: object() for i in ids},
                               aliases=aliases or {},
                               defaults=SimpleNamespace(model=default))
    return cfg


def test_an_unset_stt_picks_the_model_that_hears(models):
    assert stt.auto_native(models("text", "ears")) == "model:ears"
    assert stt.auto_native(models("ears", "ears2", default="ears2")) == "model:ears2"
    assert stt.auto_native(models("ears", "ears2", default="text")) is None
    assert stt.auto_native(models("text")) is None


def test_stt_may_name_a_served_model(models):
    cfg = models("text", "ears", aliases={"scribe": "ears"})
    assert stt.native_for("ears", cfg) == "model:ears"
    assert stt.native_for("scribe", cfg) == "model:ears"
    assert stt.native_for("model:ears", cfg) == "model:ears"
    assert stt.native_for("whisper-turbo", cfg) is None
    with pytest.raises(ConfigError, match="takes no audio input"):
        stt.native_for("text", cfg)


def test_a_served_stt_model_needs_no_whisper(models):
    cfg = models("ears")
    assert not stt.needs_whisper("model:ears", cfg)
    assert not stt.needs_whisper("ears", cfg)
    assert not stt.needs_whisper("granite-speech", cfg)
    assert not stt.needs_whisper("/m/companion.gguf", cfg)
    assert not stt.needs_whisper(False, cfg)
    assert stt.needs_whisper("whisper-turbo", cfg)
    assert stt.needs_whisper(True, cfg)
    stt.check_backend("model:ears")        # no mlx-whisper import
    assert stt.prewarm("model:ears").result() is None


def test_clients_see_the_served_model_behind_whisper_1():
    from gmlx.serve.patches.routes import _service_entry

    entry = _service_entry("whisper-1", "stt", "model:ears")
    assert entry["stt"] is True and entry["alias_of"] == "ears"


# -- the output cap on audio turns ---------------------------------------------


def test_an_audio_turn_stays_inside_the_context():
    from gmlx.serve.patches.native_audio import output_cap

    assert output_cap(44_500, None) == 21_036
    assert output_cap(44_500, 21_624) == 21_036
    assert output_cap(1_000, 400) == 400
    assert output_cap(70_000, 400) == 1


def _audio_chat(**caps):
    schemas = __import__("gmlx._vendor.mlx_vlm.server.schemas",
                         fromlist=["ChatRequest"])
    return schemas.ChatRequest(model="ears", messages=[{"role": "user", "content": [
        {"type": "input_audio", "input_audio": {"data": "AA==", "format": "wav"}}]}],
        **caps)


def _stub_prepare(monkeypatch, cap=777):
    from gmlx.serve.patches import native_audio

    seen = {}

    async def prepare(model, parts, texts, http_request, max_tokens=None,
                      client_texts=None):
        seen["asked"] = max_tokens
        return ["clip"], min(cap, max_tokens or cap)

    monkeypatch.setattr(native_audio, "_prepare", prepare)
    monkeypatch.setattr(native_audio, "_is_native", lambda m: True)
    return native_audio, seen


def test_the_chat_wrap_caps_the_output(monkeypatch):
    native_audio, seen = _stub_prepare(monkeypatch)
    got = {}

    async def original(request, http_request):
        got["max_tokens"] = request.max_tokens
        got["mct"] = getattr(request, "max_completion_tokens", None)
        return "ok"

    wrap = native_audio._chat_wrap(original)
    http = SimpleNamespace(url=SimpleNamespace(path="/v1/chat/completions"))
    assert asyncio.run(wrap(_audio_chat(), http)) == "ok"
    assert seen["asked"] is None and got["max_tokens"] == 777
    asyncio.run(wrap(_audio_chat(max_tokens=100), http))
    assert seen["asked"] == 100 and got["max_tokens"] == 100
    asyncio.run(wrap(_audio_chat(max_tokens=50, max_completion_tokens=5000), http))
    assert seen["asked"] == 5000 and got["max_tokens"] == got["mct"] == 777


def test_the_responses_wrap_caps_the_output(monkeypatch):
    native_audio, seen = _stub_prepare(monkeypatch, cap=900)
    body = {"model": "ears", "max_output_tokens": 5000, "input": [
        {"type": "message", "role": "user", "content": [
            {"type": "input_audio", "input_audio": {"data": "AA==", "format": "wav"}}]}]}

    class Req:
        url = SimpleNamespace(path="/v1/responses")

        async def json(self):
            return body

    async def original(request):
        return (await request.json())["max_output_tokens"]

    wrap = native_audio._responses_wrap(original)
    assert asyncio.run(wrap(Req())) == 900 and seen["asked"] == 5000


def test_client_text_may_not_hold_a_control_token():
    from gmlx.serve.patches import native_audio

    class Tok:
        def get_added_vocab(self):
            return {"<SPECIAL_19>": 19, "<|im_end|>": 11, "<spk:0>": 100,
                    "<ev:laugh>": 120, "<sty:fast>": 130, "x": 5}

    tok = Tok()
    native_audio._check_control_tokens(tok, ["<spk:0> Hi <ev:laugh> x"])
    for bad in ("a <SPECIAL_19> b", "end<|im_end|>", "here <|audio|>"):
        with pytest.raises(native_audio._Refused, match="control token"):
            native_audio._check_control_tokens(tok, ["fine", bad])
