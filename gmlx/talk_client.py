"""HTTP client + turn logic for ``gmlx talk`` (see talk.py for the verb).

Everything network-facing lives here behind small module-level seams
(mirroring launch.py's ``_http_get_json`` pattern) so tests fake the server
with monkeypatched functions - no sockets, no audio. The pure pieces
(:class:`SentenceChunker`, WAV encode/decode) are unit-tested directly.

The turn logic is the :class:`Brain` protocol: the audio loop hands it the
user's words and consumes ``BrainEvent`` tuples - ``("say", text)`` speakable
answer deltas, ``("status", label)`` progress (thinking, tools), ``("done",
stats)`` end-of-turn. v1 ships :class:`ServerChatBrain` (rolling-history
``/v1/chat/completions``); the assistant brain (tool calls, MCP, memory)
is a drop-in behind the same protocol.
"""

from __future__ import annotations

import contextlib
import io
import json
import uuid
import urllib.error
import urllib.request
import wave
from typing import Protocol
from collections.abc import Iterator

import numpy as np


class TalkClientError(RuntimeError):
    """A server/transport problem the TUI reports as one friendly line."""


@contextlib.contextmanager
def _raise_talk_error(what: str):
    """Map transport errors inside the block to :class:`TalkClientError`,
    prefixed with the failing verb."""
    try:
        yield
    except urllib.error.HTTPError as e:
        raise TalkClientError(
            f"{what} failed: HTTP {e.code} "
            f"{e.read().decode(errors='replace')[:200]}") from e
    except (urllib.error.URLError, OSError) as e:
        raise TalkClientError(f"{what} failed: {e}") from e


def _headers(api_key: str | None, extra: dict | None = None) -> dict:
    h = dict(extra or {})
    if api_key:
        h["Authorization"] = f"Bearer {api_key}"
    return h


def _http_get_json(url: str, timeout: float = 5.0,
                   api_key: str | None = None):
    """GET + parse JSON (lifecycle.get_json). Seam: monkeypatched in tests."""
    from .lifecycle import get_json

    return get_json(url, api_key=api_key, timeout=timeout)


def _http_post(url: str, data: bytes, headers: dict, timeout: float) -> bytes:
    req = urllib.request.Request(url, data=data, headers=headers,
                                 method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310 (local)
        return r.read()


def _decode_body(what: str, fn):
    """Decode a 200 response body, mapping malformed payloads (a proxy's HTML
    maintenance page, some other service on the port) to TalkClientError - the
    talk workers catch only that, and anything else kills the TTS/STT thread
    for the rest of the session."""
    try:
        return fn()
    except Exception as e:
        raise TalkClientError(f"{what}: malformed server response ({e})") from e


def _open_stream(url: str, payload: dict, api_key: str | None,
                 timeout: float):
    """POST ``payload`` and return the live (line-iterable) HTTP response."""
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=data, method="POST",
        headers=_headers(api_key, {"Content-Type": "application/json"}))
    return urllib.request.urlopen(req, timeout=timeout)  # noqa: S310 (local)


# WAV <-> int16 PCM (stdlib `wave`; both sides of the loop speak 16-bit mono)

def encode_wav(pcm: np.ndarray, rate: int) -> bytes:
    """Mono int16 samples -> WAV bytes (the STT upload body)."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(np.asarray(pcm, dtype="<i2").tobytes())
    return buf.getvalue()


def decode_wav(data: bytes) -> tuple[np.ndarray, int]:
    """WAV bytes -> (mono int16 samples, rate). The header carries the sample
    rate - why talk requests ``response_format=wav`` (raw pcm is headerless and
    TTS models differ: Kokoro 24 kHz, qwen3-tts 12 kHz-family)."""
    with wave.open(io.BytesIO(data), "rb") as w:
        rate = w.getframerate()
        n = w.getnframes()
        raw = w.readframes(n)
        samples = np.frombuffer(raw, dtype="<i2")
        if w.getnchannels() > 1:
            samples = samples.reshape(-1, w.getnchannels()).mean(
                axis=1).astype(np.int16)
    return samples, rate


# Server capability probe + audio endpoints

def ensure_v1_base(base_url: str) -> str:
    """Every endpoint here expects a base ending in /v1 (the audio routes are
    registered only there); accept a bare http://host:port."""
    base = base_url.rstrip("/")
    return base if base.endswith("/v1") else base + "/v1"


def probe_capabilities(base_url: str, api_key: str | None = None,
                       timeout: float = 5.0) -> dict:
    """Read ``/v1/models`` -> what the talk loop needs from it: whether the
    server serves STT/TTS (the ``stt``/``tts`` marker entries the launch
    integrations also key on), the chat-model ids, and the default-marked id."""
    try:
        payload = _http_get_json(base_url.rstrip("/") + "/models",
                                 timeout=timeout, api_key=api_key)
    except urllib.error.HTTPError as e:
        if e.code == 401:
            raise TalkClientError(
                "server requires an API key (pass --api-key)") from e
        raise TalkClientError(f"could not read /v1/models: HTTP {e.code}") from e
    except (urllib.error.URLError, OSError, ValueError) as e:
        raise TalkClientError(f"could not read /v1/models: {e}") from e

    def _caps():
        entries = [e for e in (payload.get("data") or [])
                   if isinstance(e, dict)]
        chat_ids = [e["id"] for e in entries
                    if e.get("id") and not any(e.get(k) for k in
                                               ("stt", "tts", "embeddings",
                                                "rerank"))]
        default = next((e["id"] for e in entries if e.get("default")), None)
        return {
            "stt": any(e.get("stt") for e in entries),
            "tts": any(e.get("tts") for e in entries),
            "chat_ids": chat_ids,
            "default": default,
        }

    return _decode_body("capability probe", _caps)


def keep_model(base_url: str, model: str, *, api_key: str | None = None,
               keep: bool = True, timeout: float = 5.0) -> bool:
    """Hold ``model`` resident for a live voice session (``POST /v1/keep``:
    warms it and exempts it from the idle reaper); ``keep=False`` releases the
    hold at session end. Best-effort: only gmlx servers have the route, so
    any failure (foreign server, older version, server down) is just "no hold"
    and must never break the session."""
    url = f"{base_url.rstrip('/')}/keep"
    body = json.dumps({"model": model, "keep": keep}).encode()
    headers = _headers(api_key, {"Content-Type": "application/json"})
    try:
        raw = _http_post(url, body, headers, timeout)
        return json.loads(raw).get("status") in ("kept", "released")
    except Exception:  # noqa: BLE001 - best-effort by contract
        return False


def transcribe_wav(base_url: str, wav_bytes: bytes, *,
                   api_key: str | None = None,
                   language: str | None = None,
                   timeout: float = 120.0) -> str:
    """POST one utterance to ``/v1/audio/transcriptions`` -> its text."""
    boundary = uuid.uuid4().hex
    parts = [
        (b"--%s\r\nContent-Disposition: form-data; name=\"file\"; "
         b"filename=\"utterance.wav\"\r\nContent-Type: audio/wav\r\n\r\n"
         % boundary.encode()) + wav_bytes + b"\r\n",
        (b"--%s\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\n"
         b"whisper-1\r\n" % boundary.encode()),
    ]
    if language:
        parts.append(
            (b"--%s\r\nContent-Disposition: form-data; name=\"language\""
             b"\r\n\r\n" % boundary.encode()) + language.encode() + b"\r\n")
    body = b"".join(parts) + b"--%s--\r\n" % boundary.encode()
    headers = _headers(api_key, {
        "Content-Type": f"multipart/form-data; boundary={boundary}"})
    with _raise_talk_error("transcription"):
        raw = _http_post(base_url.rstrip("/") + "/audio/transcriptions",
                         body, headers, timeout)
    return _decode_body(
        "transcription",
        lambda: str(json.loads(raw).get("text") or "").strip())


def synthesize(base_url: str, text: str, *, voice: str | None = None,
               speed: float = 1.0, api_key: str | None = None,
               timeout: float = 120.0) -> tuple[np.ndarray, int]:
    """POST ``/v1/audio/speech`` (wav) -> (mono int16 samples, rate)."""
    payload = {"model": "tts-1", "input": text, "response_format": "wav",
               "speed": speed}
    if voice:
        payload["voice"] = voice
    headers = _headers(api_key, {"Content-Type": "application/json"})
    with _raise_talk_error("speech synthesis"):
        raw = _http_post(base_url.rstrip("/") + "/audio/speech",
                         json.dumps(payload).encode(), headers, timeout)
    return _decode_body("speech synthesis", lambda: decode_wav(raw))


def list_voices(base_url: str, api_key: str | None = None,
                timeout: float = 5.0) -> list[str]:
    """GET ``/v1/audio/voices`` -> voice names; [] when the server predates the
    route (404) or errors - voice switching then just takes free text."""
    try:
        payload = _http_get_json(base_url.rstrip("/") + "/audio/voices",
                                 timeout=timeout, api_key=api_key)
        return [str(v) for v in (payload.get("voices") or [])]
    except (urllib.error.HTTPError, urllib.error.URLError, OSError,
            ValueError):
        return []


def embed_texts(base_url: str, texts: list, *,
                api_key: str | None = None,
                timeout: float = 120.0) -> list:
    """POST ``/v1/embeddings`` -> one vector (list of floats) per input, in
    input order. No ``model`` field - the server maps that to its configured
    embedder (same convention as whisper-1/tts-1)."""
    payload = {"input": list(texts)}
    headers = _headers(api_key, {"Content-Type": "application/json"})
    with _raise_talk_error("embeddings"):
        raw = _http_post(base_url.rstrip("/") + "/embeddings",
                         json.dumps(payload).encode(), headers, timeout)

    def _vectors():
        data = sorted(json.loads(raw).get("data") or [],
                      key=lambda d: d.get("index", 0))
        return [d["embedding"] for d in data]

    return _decode_body("embeddings", _vectors)


def rerank_documents(base_url: str, query: str, documents: list, *,
                     top_n: int, api_key: str | None = None,
                     timeout: float = 120.0) -> list:
    """POST the Cohere/Jina-shaped ``/v1/rerank`` -> ``documents`` indices,
    best-first, cut to ``top_n``."""
    payload = {"query": query, "documents": list(documents), "top_n": top_n,
               "return_documents": False}
    headers = _headers(api_key, {"Content-Type": "application/json"})
    with _raise_talk_error("rerank"):
        raw = _http_post(base_url.rstrip("/") + "/rerank",
                         json.dumps(payload).encode(), headers, timeout)
    return _decode_body(
        "rerank",
        lambda: [int(r["index"])
                 for r in json.loads(raw).get("results") or []][:top_n])


# Streaming chat

def stream_chat(base_url: str, *, model: str, messages: list,
                max_tokens: int | None, api_key: str | None = None,
                tools: list | None = None,
                timeout: float = 600.0,
                extra: dict | None = None) -> Iterator[dict]:
    """Stream ``/v1/chat/completions`` -> the raw ``delta`` dict per SSE chunk
    (plus ``{"_finish": ...}``/``{"_usage": ...}`` markers). ``tools`` is an
    OpenAI function-spec list (the assistant brain's loop); ``tool_calls``
    deltas pass through verbatim. ``extra`` merges additional payload fields
    (sampling passthrough, stream_options). Closing the generator closes the
    HTTP response - that is the cancellation path."""
    payload = {"model": model, "messages": messages, "stream": True}
    if max_tokens is not None:      # None = server default (uncapped chat)
        payload["max_tokens"] = max_tokens
    if tools:
        payload["tools"] = tools
    if extra:
        payload.update(extra)
    with _raise_talk_error("chat"):
        resp = _open_stream(base_url.rstrip("/") + "/chat/completions",
                            payload, api_key, timeout)
    try:
        for raw in resp:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except ValueError:
                continue
            for choice in chunk.get("choices") or []:
                delta = choice.get("delta") or {}
                if delta:
                    yield delta
                if choice.get("finish_reason"):
                    yield {"_finish": choice["finish_reason"]}
            if chunk.get("usage"):
                yield {"_usage": chunk["usage"]}
    finally:
        resp.close()


# Sentence chunking for speech

# Don't end a chunk right after these (an abbreviation dot is not a sentence).
_ABBREVIATIONS = frozenset({
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc", "eg",
    "e.g", "ie", "i.e", "cf", "no", "vol", "approx", "dept", "est", "min",
    "max", "fig", "inc", "ltd", "co",
})
_ENDERS = ".!?"
_SOFT_BREAKS = ",;:"


class SentenceChunker:
    """Split streamed text into speakable chunks at sentence boundaries.

    ``feed`` returns complete chunks as they appear; ``flush`` returns the
    remainder. Until the first chunk is emitted, a *fast path* also splits at
    a soft break (``,;:``) once the buffer passes ``fast_first_chars`` - the
    first audio should not wait for a long opening sentence to finish."""

    def __init__(self, *, min_chars: int = 24, fast_first_chars: int = 60):
        self.min_chars = min_chars
        self.fast_first_chars = fast_first_chars
        self.buf = ""
        self.emitted = 0

    def feed(self, text: str) -> list[str]:
        self.buf += text
        out: list[str] = []
        while True:
            cut = self._boundary()
            if cut is None:
                break
            chunk, self.buf = self.buf[:cut].strip(), self.buf[cut:]
            if chunk:
                out.append(chunk)
                self.emitted += 1
        return out

    def flush(self) -> str | None:
        chunk, self.buf = self.buf.strip(), ""
        return chunk or None

    def _boundary(self) -> int | None:
        """Index just past the first speakable boundary, or None."""
        for i, ch in enumerate(self.buf):
            if ch in _ENDERS and self._is_sentence_end(i):
                end = i + 1
                # Pull trailing quotes/brackets into the chunk.
                while end < len(self.buf) and self.buf[end] in "\"')]}":
                    end += 1
                if end >= self.min_chars:
                    return end
            elif ch == "\n":
                if i >= self.min_chars:
                    return i + 1
        if self.emitted == 0 and len(self.buf) > self.fast_first_chars:
            for i in range(self.fast_first_chars, 0, -1):
                if self.buf[i] in _SOFT_BREAKS:
                    return i + 1
        return None

    def _is_sentence_end(self, i: int) -> bool:
        nxt = self.buf[i + 1: i + 2]
        if nxt and not nxt.isspace():
            return False              # decimal point, version number, URL, ...
        if not nxt:
            return False              # stream may still be mid-token
        if self.buf[i] == ".":
            word = ""
            j = i - 1
            while j >= 0 and (self.buf[j].isalnum() or self.buf[j] == "."):
                word = self.buf[j] + word
                j -= 1
            w = word.lower().rstrip(".")
            if w in _ABBREVIATIONS or (len(w) == 1 and w.isalpha()):
                return False          # "Dr." / an initial like "J."
        return True


# Brain protocol (the phase-2 seam)

BrainEvent = tuple  # ("say", text) | ("status", label) | ("done", stats: dict)


class Brain(Protocol):
    def turn(self, user_text: str) -> Iterator[BrainEvent]: ...
    def reset(self) -> None: ...


class ServerChatBrain:
    """v1 brain: rolling-history streamed chat against the server.

    Thinking never reaches the speaker: mlx-vlm streams reasoning models'
    chain-of-thought in a separate ``delta.reasoning`` field, and any inline
    control markers still in ``delta.content`` are split out by
    :class:`~gmlx.reasoning.ReasoningFilter` - both surface as
    ``("status", "thinking")`` events; only answer spans become ``("say", ...)``.
    History keeps the clean answer text (resending thinking is not OpenAI chat
    convention and would bloat every re-prefill)."""

    def __init__(self, *, base_url: str, model: str,
                 api_key: str | None = None,
                 system: str | None = None, max_tokens: int | None = None):
        self.base_url = base_url
        self.model = model
        self.api_key = api_key
        self.system = system
        self.max_tokens = max_tokens
        self.messages: list[dict] = []
        self.reset()

    def reset(self) -> None:
        self.messages = ([{"role": "system", "content": self.system}]
                         if self.system else [])

    def turn(self, user_text: str) -> Iterator[BrainEvent]:
        from .reasoning import ReasoningFilter

        self.messages.append({"role": "user", "content": user_text})
        rf = ReasoningFilter()
        answer: list[str] = []
        stats: dict = {}
        deltas = stream_chat(self.base_url, model=self.model,
                             messages=self.messages,
                             max_tokens=self.max_tokens,
                             api_key=self.api_key)
        completed = False
        try:
            for delta in deltas:
                if "_usage" in delta:
                    stats = delta["_usage"] or {}
                    continue
                if "_finish" in delta:
                    completed = True
                    continue
                if delta.get("reasoning"):
                    yield ("status", "thinking")
                    continue
                text = delta.get("content")
                if not text:
                    continue
                for span, mode in rf.feed(text):
                    if mode == "answer" and span:
                        answer.append(span)
                        yield ("say", span)
                    elif span:
                        yield ("status", "thinking")
            for span, mode in rf.flush():
                if mode == "answer" and span:
                    answer.append(span)
                    yield ("say", span)
        finally:
            # Runs on completion and on cancellation (generator .close()):
            # a canceled turn still keeps what was already said, so the next
            # turn's context matches what the user heard. No yields here - a
            # closing generator may not yield.
            deltas.close()
            self.messages.append({"role": "assistant",
                                  "content": "".join(answer)})
        if completed:
            yield ("done", stats)
