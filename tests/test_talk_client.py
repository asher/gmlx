"""talk_client: capability probe, audio endpoints, SSE brain, sentence
chunking - all against monkeypatched HTTP seams (no sockets, no audio)."""
from __future__ import annotations

import json

import numpy as np
import pytest

from gmlx import talk_client as tc
from gmlx.talk_client import (
    SentenceChunker,
    ServerChatBrain,
    TalkClientError,
    decode_wav,
    encode_wav,
)


def test_ensure_v1_base():
    # Bare host:port targets (menubar dynamic mode, talk --base-url) must
    # gain the /v1 the audio routes are registered under.
    assert tc.ensure_v1_base("http://127.0.0.1:8080") == "http://127.0.0.1:8080/v1"
    assert tc.ensure_v1_base("http://h:8080/") == "http://h:8080/v1"
    assert tc.ensure_v1_base("http://h:8080/v1") == "http://h:8080/v1"
    assert tc.ensure_v1_base("http://h:8080/v1/") == "http://h:8080/v1"


# WAV round-trip
def test_wav_round_trip():
    pcm = (np.sin(np.linspace(0, 40, 4800)) * 20000).astype(np.int16)
    data = encode_wav(pcm, 16000)
    back, rate = decode_wav(data)
    assert rate == 16000
    assert np.array_equal(back, pcm)


def test_decode_wav_downmixes_stereo():
    import io
    import wave
    stereo = np.stack([np.full(100, 1000, np.int16),
                       np.full(100, 3000, np.int16)], axis=1)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(24000)
        w.writeframes(stereo.tobytes())
    mono, rate = decode_wav(buf.getvalue())
    assert rate == 24000 and len(mono) == 100
    assert int(mono[0]) == 2000


# Capability probe
def _models_payload(*, stt=True, tts=True):
    data = [
        {"id": "qwen", "object": "model", "default": True},
        {"id": "gemma", "object": "model", "default": False},
    ]
    if stt:
        data.append({"id": "whisper-1", "stt": True, "alias_of": "x"})
    if tts:
        data.append({"id": "tts-1", "tts": True, "alias_of": "y"})
    data.append({"id": "text-embedding-3-small", "embeddings": True,
                 "alias_of": "z"})
    return {"object": "list", "data": data}


def test_probe_capabilities(monkeypatch):
    monkeypatch.setattr(tc, "_http_get_json",
                        lambda url, timeout=5.0, api_key=None:
                        _models_payload())
    caps = tc.probe_capabilities("http://h:1/v1")
    assert caps["stt"] and caps["tts"]
    assert caps["chat_ids"] == ["qwen", "gemma"]   # service entries excluded
    assert caps["default"] == "qwen"


def test_probe_capabilities_missing_services(monkeypatch):
    monkeypatch.setattr(tc, "_http_get_json",
                        lambda url, timeout=5.0, api_key=None:
                        _models_payload(stt=False, tts=False))
    caps = tc.probe_capabilities("http://h:1/v1")
    assert not caps["stt"] and not caps["tts"]


def test_probe_capabilities_server_down(monkeypatch):
    def boom(url, timeout=5.0, api_key=None):
        raise OSError("refused")
    monkeypatch.setattr(tc, "_http_get_json", boom)
    with pytest.raises(TalkClientError, match="refused"):
        tc.probe_capabilities("http://h:1/v1")


# Audio endpoints
def test_transcribe_wav_builds_multipart(monkeypatch):
    seen = {}

    def fake_post(url, data, headers, timeout):
        seen.update(url=url, data=data, headers=headers)
        return json.dumps({"text": "  hello there \n"}).encode()

    monkeypatch.setattr(tc, "_http_post", fake_post)
    text = tc.transcribe_wav("http://h:1/v1", b"WAVBYTES", language="en",
                             api_key="k")
    assert text == "hello there"
    assert seen["url"].endswith("/v1/audio/transcriptions")
    assert seen["headers"]["Authorization"] == "Bearer k"
    ctype = seen["headers"]["Content-Type"]
    assert ctype.startswith("multipart/form-data; boundary=")
    boundary = ctype.split("boundary=")[1].encode()
    assert seen["data"].count(b"--" + boundary) == 4   # 3 parts + terminator
    assert b"WAVBYTES" in seen["data"]
    assert b'name="language"' in seen["data"]
    assert seen["data"].endswith(b"--" + boundary + b"--\r\n")


def test_keep_model_posts_hold_and_release(monkeypatch):
    seen = []

    def fake_post(url, data, headers, timeout):
        seen.append((url, json.loads(data)))
        return json.dumps({"status": "kept" if json.loads(data)["keep"]
                           else "released"}).encode()

    monkeypatch.setattr(tc, "_http_post", fake_post)
    assert tc.keep_model("http://h:1/v1", "qwen", api_key="k") is True
    assert tc.keep_model("http://h:1/v1", "qwen", keep=False) is True
    assert seen[0] == ("http://h:1/v1/keep", {"model": "qwen", "keep": True})
    assert seen[1] == ("http://h:1/v1/keep", {"model": "qwen", "keep": False})


def test_keep_model_best_effort_on_foreign_server(monkeypatch):
    def gone(url, data, headers, timeout):
        raise OSError("no such route")
    monkeypatch.setattr(tc, "_http_post", gone)
    assert tc.keep_model("http://h:1/v1", "qwen") is False   # never raises


def test_synthesize_decodes_wav(monkeypatch):
    pcm = np.arange(200, dtype=np.int16)

    def fake_post(url, data, headers, timeout):
        payload = json.loads(data)
        assert payload["response_format"] == "wav"
        assert payload["voice"] == "am_adam" and payload["speed"] == 1.2
        return encode_wav(pcm, 24000)

    monkeypatch.setattr(tc, "_http_post", fake_post)
    out, rate = tc.synthesize("http://h:1/v1", "hi", voice="am_adam",
                              speed=1.2)
    assert rate == 24000 and np.array_equal(out, pcm)


def test_list_voices_degrades_to_empty(monkeypatch):
    def gone(url, timeout=5.0, api_key=None):
        raise OSError("404-ish")
    monkeypatch.setattr(tc, "_http_get_json", gone)
    assert tc.list_voices("http://h:1/v1") == []
    monkeypatch.setattr(tc, "_http_get_json",
                        lambda url, timeout=5.0, api_key=None:
                        {"voices": ["af_heart", "am_adam"]})
    assert tc.list_voices("http://h:1/v1") == ["af_heart", "am_adam"]


# SSE stream parsing
class _FakeResp:
    """Line-iterable fake of the urllib response stream_chat consumes."""

    def __init__(self, lines):
        self.lines = list(lines)
        self.closed = False

    def __iter__(self):
        return iter(self.lines)

    def close(self):
        self.closed = True


def _sse(*chunks) -> list[bytes]:
    out = [b": ping\n", b"\n"]                    # comment + blank line noise
    for c in chunks:
        out.append(b"data: " + json.dumps(c).encode() + b"\n")
    out.append(b"data: [DONE]\n")
    return out


def _chunk(delta, finish=None, usage=None):
    c = {"choices": [{"index": 0, "delta": delta,
                      **({"finish_reason": finish} if finish else {})}]}
    if usage:
        c["usage"] = usage
    return c


def test_stream_chat_yields_deltas_and_markers(monkeypatch):
    resp = _FakeResp(_sse(
        _chunk({"role": "assistant", "content": "Hel"}),
        _chunk({"content": "lo"}, finish="stop", usage={"total_tokens": 5}),
    ))
    monkeypatch.setattr(tc, "_open_stream",
                        lambda url, payload, api_key, timeout: resp)
    got = list(tc.stream_chat("http://h:1/v1", model="m", messages=[],
                              max_tokens=10))
    assert {"content": "lo"} in got or any(
        d.get("content") == "lo" for d in got)
    assert {"_finish": "stop"} in got
    assert {"_usage": {"total_tokens": 5}} in got
    assert resp.closed


def test_stream_chat_close_on_break(monkeypatch):
    resp = _FakeResp(_sse(_chunk({"content": "a"}), _chunk({"content": "b"})))
    monkeypatch.setattr(tc, "_open_stream",
                        lambda url, payload, api_key, timeout: resp)
    gen = tc.stream_chat("http://h:1/v1", model="m", messages=[],
                         max_tokens=10)
    next(gen)
    gen.close()
    assert resp.closed


# ServerChatBrain
def _wire_stream(monkeypatch, *chunks):
    monkeypatch.setattr(
        tc, "_open_stream",
        lambda url, payload, api_key, timeout: _FakeResp(_sse(*chunks)))


def test_brain_says_answers_only(monkeypatch):
    _wire_stream(
        monkeypatch,
        _chunk({"reasoning": "let me think"}),
        _chunk({"content": "<think>more inline</think>"}),
        _chunk({"content": "Paris is the answer."}),
        _chunk({"content": ""}, finish="stop", usage={"total_tokens": 9}),
    )
    brain = ServerChatBrain(base_url="http://h:1/v1", model="m",
                            system="Be brief.")
    events = list(brain.turn("capital of France?"))
    says = "".join(t for k, t in events if k == "say")
    assert says == "Paris is the answer."
    assert ("status", "thinking") in events
    assert events[-1] == ("done", {"total_tokens": 9})
    # History: system + user + clean assistant answer (no markers/thinking).
    assert brain.messages[0]["role"] == "system"
    assert brain.messages[-1] == {"role": "assistant",
                                  "content": "Paris is the answer."}


def test_brain_cancel_keeps_partial_history(monkeypatch):
    _wire_stream(
        monkeypatch,
        _chunk({"content": "One. "}),
        _chunk({"content": "Two. "}),
        _chunk({"content": "Three."}, finish="stop"),
    )
    brain = ServerChatBrain(base_url="http://h:1/v1", model="m")
    gen = brain.turn("count")
    first = next(e for e in gen if e[0] == "say")
    gen.close()                                    # user barged in
    assert brain.messages[-1]["role"] == "assistant"
    assert brain.messages[-1]["content"].startswith(first[1][:3])
    # done never fired on the canceled turn
    brain.reset()
    assert brain.messages == []


# SentenceChunker
def test_chunker_splits_sentences():
    ch = SentenceChunker(min_chars=5, fast_first_chars=200)
    out = ch.feed("The sky is blue today. The grass ")
    assert out == ["The sky is blue today."]
    out = ch.feed("is green. And")
    assert out == ["The grass is green."]
    assert ch.flush() == "And"


def test_chunker_holds_abbreviations_and_decimals():
    ch = SentenceChunker(min_chars=5, fast_first_chars=500)
    assert ch.feed("Dr. Smith weighs 3.14 stone according to Mr. Jones. ") \
        == ["Dr. Smith weighs 3.14 stone according to Mr. Jones."]


def test_chunker_first_chunk_fast_path():
    ch = SentenceChunker(min_chars=5, fast_first_chars=20)
    out = ch.feed("Well, considering everything you said about the plan")
    assert out == ["Well,"]                       # soft break, first chunk only
    out = ch.feed(", and more, and more, and more of the same clause here")
    assert out == []                              # fast path is first-chunk only
    assert ch.flush().startswith("considering everything")


def test_chunker_newline_is_a_boundary():
    ch = SentenceChunker(min_chars=5, fast_first_chars=500)
    assert ch.feed("First line of a list\nSecond") == ["First line of a list"]


def test_chunker_waits_for_stream_tail():
    ch = SentenceChunker(min_chars=5, fast_first_chars=500)
    assert ch.feed("It costs 3.") == []           # could be "3.14" mid-token
    assert ch.feed("14 total. Done") == ["It costs 3.14 total."]
