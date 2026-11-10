# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Asher Feldman
"""``/v1/audio/transcriptions`` and ``/v1/audio/translations`` answered by a
served LFM3 model that takes audio (LFM3-Omni, LFM3-Audio).

The route decodes the upload once. It then sends one chat turn through
the chat route in process, in the form the model card gives: the task as
the system prompt and a user turn that holds only the clip. The memory
check, the queue and the prefix cache work as for a chat request.

The model writes plain text, with no word times and no speakers, so the
formats are ``json`` and ``text``.
"""

from __future__ import annotations

import base64
import importlib
import io
import math
import wave

import numpy as np

from .native_stt import SAMPLE_RATE, _body, _chat_endpoint, decode
from .stt import STTRequestError

FORMATS = ("json", "text")
# The card's prompts.
ASR_IN = "Perform ASR in {language}."
# The form for a request that names no language. The card has none.
ASR_ANY = "Perform ASR."
TRANSLATE_FROM = "Please translate the following spoken {source} into {target} text."
# The form for a translation request that names no source language.
TRANSLATE_ANY = "Please translate the following speech into {target} text."
TARGET = "English"

# ISO 639-1 codes of the languages the family's cards list, to the names
# their prompts take. A model takes the ones its GGUF lists
# (``general.languages``).
LANGUAGES = {
    "ar": "Arabic", "da": "Danish", "de": "German", "el": "Greek",
    "en": "English", "es": "Spanish", "fi": "Finnish", "fr": "French",
    "he": "Hebrew", "hi": "Hindi", "id": "Indonesian", "it": "Italian",
    "ja": "Japanese", "ko": "Korean", "nl": "Dutch", "pl": "Polish",
    "pt": "Portuguese", "ru": "Russian", "sv": "Swedish", "th": "Thai",
    "tr": "Turkish", "vi": "Vietnamese", "zh": "Chinese",
}
_CODES = {name.lower(): code for code, name in LANGUAGES.items()}
_LISTED_MEMO: dict = {}

# Tokens the chat template and the system prompt add around the clip.
TEMPLATE_TOKENS = 48
# Output tokens allowed per second of audio, and the least allowed.
TOKENS_PER_SECOND = 12
MIN_OUTPUT_TOKENS = 64
# The least output room a clip must leave, per second of audio.
LEAST_TOKENS_PER_SECOND = 4
# Near-digital silence: a peak under -60 dBFS.
SILENCE_PEAK = 1e-3


def listed_languages(path) -> tuple | None:
    """The codes of :data:`LANGUAGES` that the GGUF at ``path`` lists, or
    None for a file that lists none."""
    import os

    try:
        st = os.stat(path)
    except (OSError, TypeError):
        return None
    key = (str(path), st.st_mtime_ns, st.st_size)
    if key not in _LISTED_MEMO:
        from gmlx.load.headerscan import scan_gguf

        try:
            listed = scan_gguf(str(path), include_tensors=False).kv.get(
                "general.languages")
        except Exception:  # noqa: BLE001  not a GGUF
            listed = None
        codes = tuple(sorted(
            {str(c).lower() for c in listed or ()} & set(LANGUAGES)))
        _LISTED_MEMO[key] = codes or None
    return _LISTED_MEMO[key]


def language_name(value: str, listed=None) -> str | None:
    """The English name of a request's ``language``: an ISO 639-1 code or
    a name. ``listed`` holds the codes the model takes, or None for all
    of :data:`LANGUAGES`. None for an empty value. Raises
    STTRequestError."""
    v = (value or "").strip().lower()
    if not v:
        return None
    takes = sorted(LANGUAGES if listed is None else listed)
    code = _CODES.get(v, v.split("-")[0].split("_")[0])
    if code in takes:
        return LANGUAGES[code]
    raise STTRequestError(
        400, f"language {value!r} is not one this model lists. Send one "
             f"of: {', '.join(takes)}, or leave it out")


def _fields(form, task: str, listed=None) -> dict:
    """The request's form fields, checked. ``listed`` holds the language
    codes the model takes. Raises STTRequestError."""
    fmt = str(form.get("response_format") or "json").strip().lower()
    if fmt not in FORMATS:
        raise STTRequestError(
            400, f"unsupported response_format {fmt!r}: this model writes "
                 "plain text, with no word times and no speakers "
                 f"(supported: {', '.join(FORMATS)})")
    for key in ("timestamp_granularities[]", "timestamp_granularities"):
        if form.getlist(key):
            raise STTRequestError(
                400, "timestamp_granularities is not supported: this model "
                     "gives no word times")
    if str(form.get("diarization") or "").strip().lower() in ("true", "1",
                                                             "yes"):
        raise STTRequestError(
            400, "diarization is not supported: this model names no "
                 "speakers")
    raw_t = str(form.get("temperature") or "").strip()
    try:
        temperature = float(raw_t) if raw_t else 0.0
    except ValueError:
        raise STTRequestError(
            400, f"temperature {raw_t!r} is not a number") from None
    if not 0.0 <= temperature <= 2.0:
        raise STTRequestError(400, "temperature must be between 0 and 2")
    return {"format": fmt, "temperature": temperature,
            "language": language_name(str(form.get("language") or ""),
                                      listed)}


def instruction(task: str, language: str | None) -> str:
    """The system prompt. ``language`` is the spoken language's name."""
    if task == "transcribe":
        return ASR_IN.format(language=language) if language else ASR_ANY
    if language:
        return TRANSLATE_FROM.format(source=language, target=TARGET)
    return TRANSLATE_ANY.format(target=TARGET)


def output_budget(model_id: str, samples: int) -> int:
    """The output tokens a clip of ``samples`` gets. Raises
    STTRequestError for a clip that leaves no room in the context."""
    from gmlx.load.config_synth import LFM3_SPEECH_CONTEXT
    from gmlx.models.lfm3.audio import num_audio_tokens

    seconds = samples / SAMPLE_RATE
    used = num_audio_tokens(samples) + TEMPLATE_TOKENS
    room = LFM3_SPEECH_CONTEXT - used
    if room < max(MIN_OUTPUT_TOKENS,
                  math.ceil(seconds * LEAST_TOKENS_PER_SECOND)):
        # 12.5 audio tokens and at least 4 output tokens per second.
        limit = (LFM3_SPEECH_CONTEXT - TEMPLATE_TOKENS) / (
            12.5 + LEAST_TOKENS_PER_SECOND) / 60
        raise STTRequestError(
            400, f"the clip is {seconds / 60:.0f} minutes long, and "
                 f"{model_id} takes about {limit:.0f} minutes in one "
                 "request. Send it in parts")
    return min(room, max(MIN_OUTPUT_TOKENS,
                         math.ceil(seconds * TOKENS_PER_SECOND)))


def _wav(audio: np.ndarray) -> str:
    """The samples as a base64 16-bit WAV, the chat turn's audio part."""
    pcm = np.round(np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(SAMPLE_RATE)
        out.writeframes(pcm.tobytes())
    return base64.b64encode(buffer.getvalue()).decode()


def chat_request(model_id: str, task: str, fields: dict, audio: np.ndarray):
    """The chat turn that transcribes or translates the clip: the card's
    system prompt, a user turn that holds only the clip, greedy unless
    the client sets a temperature. Every sampling field is set, so no
    profile adds to it."""
    schemas = importlib.import_module("gmlx._vendor.mlx_vlm.server.schemas")
    return schemas.ChatRequest(
        model=model_id, stream=False,
        messages=[
            {"role": "system",
             "content": instruction(task, fields["language"])},
            {"role": "user", "content": [
                {"type": "input_audio",
                 "input_audio": {"data": _wav(audio), "format": "wav"}}]}],
        max_tokens=output_budget(model_id, audio.size),
        temperature=fields["temperature"], top_p=1.0, top_k=0, min_p=0.0,
        repetition_penalty=1.0, presence_penalty=0.0, frequency_penalty=0.0)


async def run(http_request, model_id: str, audio: np.ndarray, task: str,
              fields: dict) -> dict:
    """``{"text"}`` for ``audio`` on ``model_id``."""
    from fastapi import HTTPException

    if audio.size == 0 or float(np.abs(audio).max()) < SILENCE_PEAK:
        return {"text": ""}
    request = chat_request(model_id, task, fields, audio)
    try:
        response = await _chat_endpoint()(request, http_request)
    except HTTPException as e:
        if e.status_code >= 500:
            raise
        raise STTRequestError(e.status_code, str(e.detail)) from None
    body = _body(response)
    choices = body.get("choices") or []
    text = ((choices[0].get("message") or {}).get("content")
            if choices else "")
    return {"text": str(text or "").strip()}


def format_result(result: dict, fields: dict):
    """``(content, media_type)`` in the requested format."""
    if fields["format"] == "text":
        return result["text"] + "\n", "text/plain"
    return {"text": result["text"]}, "application/json"


async def answer(http_request, form, model_id: str, audio_bytes: bytes,
                 filename: str, task: str):
    """``(content, media_type)`` for a transcription or translation on the
    served model ``model_id``. Raises STTRequestError for a client error."""
    import asyncio

    if not audio_bytes:
        raise STTRequestError(400, "multipart field 'file' is required")
    fields = _fields(form, task, _served_languages(model_id))
    audio = await asyncio.to_thread(decode, audio_bytes, filename)
    result = await run(http_request, model_id, audio, task, fields)
    return format_result(result, fields)


def _served_languages(model_id: str) -> tuple | None:
    from . import bridge_vlm as serving

    try:
        _, spec = serving.resolve_request_model(
            model_id, profile_field=serving.get_request_profile())
    except Exception:  # noqa: BLE001  the chat route answers the error
        return None
    return listed_languages(getattr(spec, "path", None))
