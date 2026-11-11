# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Asher Feldman
"""``/v1/audio/speech`` answered by a served chat model that speaks
(LFM3-Omni, LFM3-Audio).

``server.tts`` names such a model as ``model:<id>``. The route sends one
speech pass through the chat route in process: the model's speech prompt
with the text, answered in frames of audio. The memory check, the queue
and the prefix cache work as for a chat request.

The model has the voices its speech prompt names and speaks at one
speed. ``pcm`` and ``wav`` are written in process, and the other formats
through ffmpeg.
"""

from __future__ import annotations

import base64
import importlib
import io
import json
import os
import re
import subprocess
import tempfile
import wave
from dataclasses import dataclass

from .native_stt import _body, _chat_endpoint
from .tts import _CONFIGURED_NAMES, _MEDIA_TYPES, TTSRequestError

PREFIX = "model:"
# The formats of ``/v1/audio/speech``. ``pcm`` is raw 16-bit little-endian.
FORMATS = ("mp3", "wav", "flac", "opus", "pcm")
DEFAULT_FORMAT = "mp3"
# Written with no other program.
PLAIN_FORMATS = ("pcm", "wav")
STREAM_FORMATS = ("audio", "sse")
# ffmpeg's output arguments per format.
_FFMPEG_OUT = {
    "mp3": ("-f", "mp3"),
    "flac": ("-f", "flac"),
    "opus": ("-c:a", "libopus", "-f", "ogg"),
}
# The frames a character of text takes to say, about: a letter, a wide
# character (Chinese, Japanese, Korean), and any other.
_LETTER_FRAMES, _WIDE_FRAMES, _OTHER_FRAMES = 0.8, 3.0, 1.2
# One pass says about 30 s (375 frames) and then stops or runs on. A
# longer text is spoken in pieces of about this many frames.
PIECE_FRAMES = 300
# The frames a piece gets: a floor, and twice what its text should take.
MIN_FRAMES = 125
BUDGET_FACTOR = 2

# The text archs whose projector's speech generator the server runs.
_SPEECH_OUT_ADAPTERS = {"lfm3": "lfm3"}


def is_native(model) -> bool:
    """True when ``model`` is a served chat model's ``model:<id>``."""
    return str(model or "").startswith(PREFIX)


def native_id(value) -> str | None:
    """The served model id of a ``model:<id>`` TTS value, else None."""
    v = str(value or "")
    return v[len(PREFIX):] if v.startswith(PREFIX) else None


def tts_adapter(path, mmproj) -> str | None:
    """The name of the speech adapter of a chat model and its projector,
    or None: ``lfm3`` for an LFM3 model whose projector speaks."""
    from gmlx.load.discovery import header_meta, projector_caps

    if not path or not projector_caps(mmproj).speech_out:
        return None
    meta = header_meta(str(path))
    return _SPEECH_OUT_ADAPTERS.get((meta or {}).get("arch"))


def native_models(cfg) -> list[str]:
    """The ids of the configured chat models that answer the speech route
    themselves."""
    from .bridge_vlm import _resolve_served

    out = []
    for mid in (getattr(cfg, "models", None) or {}):
        try:
            rm = _resolve_served(mid, cfg)
        except Exception:  # noqa: BLE001  a skipped model says nothing
            continue
        if tts_adapter(rm.path, rm.mmproj):
            out.append(mid)
    return out


def auto_native(cfg) -> str | None:
    """``model:<id>`` for an unset ``server.tts``: the default model when
    it speaks, else the one configured model that does, else None."""
    native = native_models(cfg)
    default = getattr(getattr(cfg, "defaults", None), "model", None)
    if default in native:
        return PREFIX + default
    return PREFIX + native[0] if len(native) == 1 else None


def native_for(value, cfg) -> str | None:
    """``model:<id>`` when ``value`` names a configured chat model (by id
    or alias), else None. Raises ConfigError for a model that does not
    speak."""
    from gmlx.config import ConfigError

    if cfg is None or not isinstance(value, str):
        return None
    v = value.strip()
    v = v[len(PREFIX):] if v.startswith(PREFIX) else v
    mid = (getattr(cfg, "aliases", None) or {}).get(v, v)
    if mid not in (getattr(cfg, "models", None) or {}):
        return None
    if mid not in native_models(cfg):
        raise ConfigError(
            f"server.tts: model {v!r} has no speech output. Name a model "
            "that speaks, or a text-to-speech model")
    return PREFIX + mid


def needs_mlx_audio(value, cfg=None) -> bool:
    """Whether a raw ``server.tts`` value runs on mlx-audio (the ``tts``
    extra): not when it is off or names a served model."""
    if not value or value is False:
        return False
    v = str(value).strip()
    if is_native(v):
        return False
    models = getattr(cfg, "models", None) or {}
    aliases = getattr(cfg, "aliases", None) or {}
    return v not in models and v not in aliases


def speaks(model_field: str) -> bool:
    """Whether ``model_field`` names a served model that speaks."""
    from . import bridge_vlm as serving

    try:
        _, spec = serving.resolve_request_model(
            model_field, profile_field=serving.get_request_profile())
    except Exception:  # noqa: BLE001  the caller answers the error
        return False
    return tts_adapter(getattr(spec, "path", None),
                       getattr(spec, "mmproj", None)) is not None


def target(model_field: str, configured: str | None) -> str | None:
    """The served model that answers this request itself, or None when the
    configured text-to-speech model answers. A request that names a served
    chat model with no speech output is a 400."""
    from . import bridge_vlm as serving

    req = (model_field or "").strip()
    if req.lower() in _CONFIGURED_NAMES:
        return native_id(configured)
    try:
        serving.resolve_request_model(req)
    except Exception:  # noqa: BLE001  not a served chat model
        if native_id(configured) is not None:
            raise TTSRequestError(
                400, f"model {req!r} is not a served model; send "
                     "model='tts-1' (or omit it)") from None
        return None
    if speaks(req):
        return req
    raise TTSRequestError(
        400, f"model {req!r} has no speech output; send model='tts-1' (or "
             "omit it) for this server's text-to-speech model")


# -- request fields -----------------------------------------------------------


def voices() -> tuple:
    from gmlx.models.lfm3 import speak

    return speak.VOICES


def voice_of(value) -> str:
    """The model's voice a request's ``voice`` names. An absent value is
    the first voice. Raises ValueError for any other name."""
    names = voices()
    v = str(value or "").strip().lower()
    if not v:
        return names[0]
    if v in names:
        return v
    raise ValueError(
        f"voice {value!r} is not one of this model's voices, which are "
        f"{' and '.join(names)}")


def language_of(value) -> str | None:
    """The speech prompt's code for a request's ``language``: a code or
    an English name. None for an absent value, which leaves the language
    to the text. Raises ValueError."""
    from gmlx.models.lfm3 import speak

    from .native_stt_lfm3 import _CODES

    v = str(value or "").strip().lower().replace("_", "-")
    if not v:
        return None
    code = _CODES.get(v, v)
    if code in speak.TTS_LANGUAGES:
        return code
    raise ValueError(
        f"language {value!r} is not one this model speaks by name. Send "
        f"one of: {', '.join(speak.TTS_LANGUAGES)}, or leave it out")


def estimated_frames(text: str) -> float:
    """About how many frames ``text`` takes to say."""
    import unicodedata

    wide = sum(unicodedata.east_asian_width(ch) in ("W", "F") for ch in text)
    letters = sum(ch.isascii() for ch in text)
    other = len(text) - wide - letters
    return round(_LETTER_FRAMES * letters + _WIDE_FRAMES * wide
                 + _OTHER_FRAMES * other, 3)


def frame_budget(text: str) -> int:
    """The frames a pass over ``text`` gets."""
    return MIN_FRAMES + int(BUDGET_FACTOR * estimated_frames(text))


# The end of a sentence: its mark, any closing quote or bracket, and the
# space after. A wide mark needs no space.
_SENTENCE_END = re.compile(
    r"(?<=[.!?\u2026])[\"'\u201d\u2019)\]]*\s+|(?<=[\u3002\uff01\uff1f])")
_CLAUSE_END = re.compile(
    r"(?<=[,;:\u2014])\s+|(?<=[\u3001\uff0c\uff1b\uff1a])")


def _cut(text: str, pattern) -> list:
    out, start = [], 0
    for m in pattern.finditer(text):
        if m.end() > start:
            out.append(text[start:m.end()])
            start = m.end()
    if start < len(text):
        out.append(text[start:])
    return out


def _fit(text: str, most: float) -> list:
    """``text`` in parts of at most ``most`` frames each: clauses, then
    words, then characters."""
    if estimated_frames(text) <= most:
        return [text]
    for pattern in (_CLAUSE_END, re.compile(r"(?<=\s)")):
        parts = _cut(text, pattern)
        if len(parts) > 1:
            return [piece for part in parts for piece in _fit(part, most)]
    out, part = [], ""
    for ch in text:
        if part and estimated_frames(part + ch) > most:
            out.append(part)
            part = ""
        part += ch
    return out + [part] if part else out


def split_text(text: str, most: float = PIECE_FRAMES) -> list:
    """``text`` as the pieces one pass each says whole: sentences packed
    together up to ``most`` frames. A sentence over that is cut at its
    clauses, then at its words."""
    parts = [piece for sentence in _cut(text, _SENTENCE_END)
             for piece in _fit(sentence, most)]
    out, piece = [], ""
    for part in parts:
        if piece and estimated_frames(piece + part) > most:
            out.append(piece.strip())
            piece = ""
        piece += part
    if piece.strip():
        out.append(piece.strip())
    return out


def format_problem(fmt: str) -> str | None:
    """Why the server cannot write ``fmt`` now, or None."""
    if fmt in PLAIN_FORMATS or fmt == "pcm16":
        return None
    from . import media_programs

    if media_programs.find("ffmpeg") is None:
        return (f"response format {fmt!r} needs ffmpeg on the PATH of the "
                "server. Install it with `brew install ffmpeg`, or ask for "
                f"one of: {', '.join(PLAIN_FORMATS)}")
    return None


def encode(pcm: bytes, sample_rate: int, fmt: str) -> bytes:
    """16-bit mono samples as ``fmt`` bytes. Raises RuntimeError when
    ffmpeg fails."""
    if fmt in ("pcm", "pcm16"):
        return pcm
    if fmt == "wav":
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as out:
            out.setnchannels(1)
            out.setsampwidth(2)
            out.setframerate(sample_rate)
            out.writeframes(pcm)
        return buffer.getvalue()
    from . import media_programs

    # A file and not a pipe: ffmpeg goes back to the header of a flac or
    # mp3 clip to write its length.
    with tempfile.TemporaryDirectory(prefix="gmlx-speech-") as folder:
        path = os.path.join(folder, "clip")
        done = subprocess.run(
            [media_programs.program("ffmpeg"), "-hide_banner", "-loglevel",
             "error", "-f", "s16le", "-ar", str(sample_rate), "-ac", "1",
             "-i", "pipe:0", *_FFMPEG_OUT[fmt], "-y", path],
            input=pcm, capture_output=True, check=False)
        data = b""
        if done.returncode == 0 and os.path.exists(path):
            with open(path, "rb") as f:
                data = f.read()
    if not data:
        tail = done.stderr.decode("utf-8", "replace").strip()[-200:]
        raise RuntimeError(f"ffmpeg could not write {fmt}: {tail}")
    return data


@dataclass(frozen=True)
class Fields:
    text: str
    voice: str
    language: str | None
    format: str
    sse: bool
    seed: int | None = None


def fields(body: dict) -> Fields:
    """The request's fields, checked. Raises TTSRequestError."""
    from .tts import sanitize_speech_text

    raw = str(body.get("input") or "")
    if not raw.strip():
        raise TTSRequestError(400, "field 'input' is required")
    text = sanitize_speech_text(raw)
    if not text:
        raise TTSRequestError(400, "field 'input' has no speakable text")
    try:
        voice = voice_of(body.get("voice"))
        language = language_of(body.get("language"))
    except ValueError as e:
        raise TTSRequestError(400, str(e)) from None
    stream = str(body.get("stream_format") or "audio").strip().lower()
    if stream not in STREAM_FORMATS:
        raise TTSRequestError(
            400, f"unsupported stream_format {stream!r} "
                 f"(supported: {', '.join(STREAM_FORMATS)})")
    sse = stream == "sse"
    given = str(body.get("response_format") or "").strip().lower()
    fmt = given or ("pcm" if sse else DEFAULT_FORMAT)
    if fmt not in FORMATS:
        raise TTSRequestError(
            400, f"unsupported response_format {fmt!r} "
                 f"(supported: {', '.join(FORMATS)})")
    if sse and fmt != "pcm":
        raise TTSRequestError(
            400, "stream_format 'sse' sends 16-bit pcm; send "
                 "response_format 'pcm' or leave it out")
    problem = format_problem(fmt)
    if problem:
        raise TTSRequestError(400, problem)
    speed = body.get("speed")
    if speed is not None and str(speed).strip():
        try:
            same = float(speed) == 1.0
        except (TypeError, ValueError):
            raise TTSRequestError(
                400, f"speed {speed!r} is not a number") from None
        if not same:
            raise TTSRequestError(
                400, "speed is not supported: this model speaks at one "
                     "speed. Send 1.0 or leave it out")
    seed = body.get("seed")
    if seed is not None and (isinstance(seed, bool)
                             or not isinstance(seed, int)):
        raise TTSRequestError(400, "seed must be an integer")
    return Fields(text, voice, language, fmt, sse, seed)


# -- the pass -----------------------------------------------------------------


def chat_request(model_id: str, f: Fields, piece: str, index: int = 0):
    """The chat request that carries the speech pass over ``piece``, the
    request's piece ``index``. The route replaces its prompt with the
    model's speech prompt, so the message holds only the text. Every
    sampling field is set, so no profile adds to it."""
    schemas = importlib.import_module("gmlx._vendor.mlx_vlm.server.schemas")
    return schemas.ChatRequest(
        model=model_id, stream=True,
        stream_options={"include_usage": True},
        messages=[{"role": "user", "content": piece}],
        max_tokens=frame_budget(piece),
        temperature=0.0, top_p=1.0, top_k=0, min_p=0.0,
        repetition_penalty=1.0, presence_penalty=0.0, frequency_penalty=0.0,
        **({} if f.seed is None else {"seed": f.seed + index}))


async def chunks(http_request, model_id: str, f: Fields):
    """Yields the request's audio as 16-bit pcm chunks, then one dict of
    its ``usage``. A text too long for one pass is spoken piece by piece.
    Raises TTSRequestError for a refused request and RuntimeError for a
    pass that failed."""
    usage = {"prompt_tokens": 0, "completion_tokens": 0}
    for index, piece in enumerate(split_text(f.text)):
        async for chunk in _pass(http_request, model_id, f, piece, index):
            if isinstance(chunk, dict):
                for key in usage:
                    usage[key] += int(chunk.get(key) or 0)
            else:
                yield chunk
    yield usage


async def _pass(http_request, model_id: str, f: Fields, piece: str,
                index: int):
    """One pass: its pcm chunks, then its usage."""
    from fastapi import HTTPException

    from . import speech_chat

    token = speech_chat.SPEECH_ONLY.set(
        speech_chat.SpeechOnly(piece, f.voice, f.language))
    try:
        response = await _chat_endpoint()(
            chat_request(model_id, f, piece, index), http_request)
    except HTTPException as e:
        if e.status_code >= 500:
            raise
        raise TTSRequestError(e.status_code, str(e.detail)) from None
    finally:
        speech_chat.SPEECH_ONLY.reset(token)
    body = getattr(response, "body_iterator", None)
    if body is None:
        _body(response)     # raises with an error answer's status
        raise RuntimeError("the chat route answered with no stream")
    usage: dict = {}
    try:
        async for event in _events(body):
            if "error" in event:
                err = event["error"]
                raise RuntimeError(
                    err.get("message") if isinstance(err, dict) else str(err))
            if event.get("usage"):
                usage = event["usage"]
            for choice in event.get("choices") or ():
                audio = (choice.get("delta") or {}).get("audio") or {}
                if audio.get("data"):
                    yield base64.b64decode(audio["data"])
    finally:
        close = getattr(body, "aclose", None)
        if close is not None:
            await close()
    yield usage


async def _events(body):
    """The JSON events of a server-sent event stream."""
    pending = ""
    async for part in body:
        pending += part.decode() if isinstance(part, bytes) else part
        while "\n\n" in pending:
            block, pending = pending.split("\n\n", 1)
            for line in block.splitlines():
                if not line.startswith("data: "):
                    continue
                data = line[6:].strip()
                if data == "[DONE]":
                    return
                try:
                    event = json.loads(data)
                except ValueError:
                    continue
                if isinstance(event, dict):
                    yield event


def _usage(usage: dict) -> dict:
    """OpenAI's speech usage object from the chat usage."""
    prompt = int(usage.get("prompt_tokens") or 0)
    output = int(usage.get("completion_tokens") or 0)
    return {"input_tokens": prompt, "output_tokens": output,
            "total_tokens": prompt + output}


async def answer(http_request, body: dict, model_id: str):
    """The response for a speech request on the served model
    ``model_id``. Raises TTSRequestError for a client error."""
    from fastapi.responses import Response, StreamingResponse

    from . import speech_chat

    f = fields(body)
    stream = chunks(http_request, model_id, f)
    if not f.sse:
        pcm = bytearray()
        async for chunk in stream:
            if not isinstance(chunk, dict):
                pcm += chunk
        if not pcm:
            raise RuntimeError("the model produced no audio")
        import asyncio

        content = await asyncio.to_thread(
            encode, bytes(pcm), speech_chat.SAMPLE_RATE, f.format)
        return Response(content=content, media_type=_MEDIA_TYPES[f.format])
    # The first chunk comes before the response starts, so a refusal is
    # an error answer and not an event.
    first = await anext(stream)

    async def events():
        chunk = first
        try:
            while True:
                if isinstance(chunk, dict):
                    done = {"type": "speech.audio.done",
                            "usage": _usage(chunk)}
                    yield f"data: {json.dumps(done)}\n\n"
                    return
                delta = {"type": "speech.audio.delta",
                         "audio": base64.b64encode(chunk).decode()}
                yield f"data: {json.dumps(delta)}\n\n"
                chunk = await anext(stream)
        except Exception as e:  # noqa: BLE001  the stream is open
            error = {"type": "error", "error": {"message": str(e)}}
            yield f"data: {json.dumps(error)}\n\n"
        finally:
            await stream.aclose()

    return StreamingResponse(
        events(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive",
                 "X-Accel-Buffering": "no"})
