"""``/v1/audio/transcriptions`` and ``/v1/audio/translations`` answered by a
served chat model that hears audio through its own speech encoder
(Nemotron 3.5 Transcribe). An LFM3 model with audio input answers through
:mod:`gmlx.serve.native_stt_lfm3`.

The route decodes the upload once and checks it for silence. It then sends
one chat turn, the model card's instruction and the clip, through the chat
route in process, so the memory check, the encoder job, the queue and the
prefix cache work as for a chat request. A hook on the encoded clip skips
the decoder when no speaker is heard. The finished text gets the card's
loop cleanup and loses the model's event and style tags.

Timed formats align the transcript to the clip: the encoder keeps the
clip's ASR states and speaker activity, the word timestamp head reads them
as a step job once the decoder is done, and NeMo's aligner
(:mod:`gmlx.models.nemotron_h.align`) times each word and its speaker.

``server.stt`` names such a model as ``model:<id>``.
"""

from __future__ import annotations

import importlib
import math
import os
import re
import tempfile
from collections.abc import Callable

import numpy as np

from . import media_programs
from .stt import STTRequestError, _srt, _vtt

PREFIX = "model:"
# Request `model` values that mean the configured STT model.
CONFIGURED_NAMES = frozenset({"", "whisper-1", "default",
                              "gpt-4o-transcribe-diarize"})
TRANSCRIBE_FORMATS = ("json", "text", "verbose_json", "srt", "vtt",
                      "diarized_json")
TRANSLATE_FORMATS = ("json", "text", "verbose_json", "srt", "vtt")
_SPEAKER_FORMATS = ("verbose_json", "srt", "vtt", "diarized_json")
SAMPLE_RATE = 16000
# A speaker is heard when an 80 ms frame passes this probability.
SPEAKER_HEARD = 0.5
# Formats with timed segments: their transcription is aligned.
_ALIGNED_FORMATS = ("verbose_json", "srt", "vtt", "diarized_json")
# A segment ends after a sentence or at a pause this long; a speaker
# change always starts one.
SEGMENT_PAUSE = 1.0
_SENTENCE_END = (".", "?", "!", "\u3002", "\uff1f", "\uff01")
# Periods that end no sentence: the model writes "U. S." as two words.
_TITLES = frozenset({"mr.", "mrs.", "ms.", "dr.", "st."})


def _ends_sentence(word: str) -> bool:
    if not word.endswith(_SENTENCE_END):
        return False
    core = word.lstrip("\"'([")
    if core.endswith(".") and (len(core) == 2 and core[0].isalpha()
                               or core.lower() in _TITLES):
        return False
    return True


class NoSpeech(Exception):
    """The encoder heard no speaker: the transcript is empty."""


def native_id(value) -> str | None:
    """The served model id of a ``model:<id>`` STT value, else None."""
    v = str(value or "")
    return v[len(PREFIX):] if v.startswith(PREFIX) else None


def adapter(model_field: str) -> str | None:
    """The STT adapter of the served model ``model_field`` names
    (:func:`gmlx.serve.stt.stt_adapter`), or None."""
    from . import bridge_vlm as serving
    from .stt import stt_adapter

    try:
        _, spec = serving.resolve_request_model(
            model_field, profile_field=serving.get_request_profile())
    except Exception:  # noqa: BLE001  the caller answers the error
        return None
    return stt_adapter(getattr(spec, "path", None),
                       getattr(spec, "mmproj", None))


def hears(model_field: str) -> bool:
    """Whether ``model_field`` names a served model that answers the STT
    routes itself."""
    return adapter(model_field) is not None


def target(model_field: str, configured: str | None) -> str | None:
    """The served model that answers this request itself, or None when the
    configured backend (Whisper or a speech companion) answers. A request
    that names a served chat model without a speech encoder is a 400."""
    from . import bridge_vlm as serving

    req = (model_field or "").strip()
    if req.lower() in CONFIGURED_NAMES:
        return native_id(configured)
    try:
        serving.resolve_request_model(req)
    except Exception:  # noqa: BLE001  not a served chat model
        if native_id(configured) is not None:
            raise STTRequestError(
                400, f"model {req!r} is not a served model; send "
                     "model='whisper-1' (or omit it)") from None
        return None
    if hears(req):
        return req
    raise STTRequestError(
        400, f"model {req!r} takes no audio input; send model='whisper-1' "
             "(or omit it) for this server's speech-to-text model")


def _flag(value, name: str) -> bool:
    v = str(value or "").strip().lower()
    if v in ("", "false", "0", "no"):
        return False
    if v in ("true", "1", "yes"):
        return True
    raise STTRequestError(400, f"{name} must be true or false (got {value!r})")


def _fields(form, task: str) -> dict:
    """The request's form fields, checked. Raises STTRequestError."""
    fmt = str(form.get("response_format") or "json").strip().lower()
    formats = TRANSCRIBE_FORMATS if task == "transcribe" else TRANSLATE_FORMATS
    if fmt not in formats:
        raise STTRequestError(
            400, f"unsupported response_format {fmt!r} "
                 f"(supported: {', '.join(formats)})")
    speakers = _flag(form.get("diarization"), "diarization") or fmt == "diarized_json"
    if speakers and task != "transcribe":
        raise STTRequestError(400, "translations have no speakers; drop "
                                   "diarization")
    if speakers and fmt not in _SPEAKER_FORMATS:
        raise STTRequestError(
            400, f"diarization needs response_format verbose_json, srt, vtt "
                 f"or diarized_json, not {fmt}")
    for key in ("known_speaker_names[]", "known_speaker_references[]",
                "known_speaker_names", "known_speaker_references"):
        if form.getlist(key):
            raise STTRequestError(400, f"{key.rstrip('[]')} is not supported: "
                                       "this server does not enroll speakers")
    grains = [str(g) for k in ("timestamp_granularities[]",
                               "timestamp_granularities")
              for g in form.getlist(k)]
    if grains and fmt == "diarized_json":
        raise STTRequestError(400, "timestamp_granularities is not supported "
                                   "with response_format diarized_json")
    bad = [g for g in grains if g not in ("word", "segment")]
    if bad:
        raise STTRequestError(400, f"timestamp_granularities takes word and "
                                   f"segment (got {bad[0]!r})")
    raw_t = str(form.get("temperature") or "").strip()
    try:
        temperature = float(raw_t) if raw_t else 0.0
    except ValueError:
        raise STTRequestError(400, f"temperature {raw_t!r} is not a number") from None
    if not 0.0 <= temperature <= 2.0:
        raise STTRequestError(400, "temperature must be between 0 and 2")
    language = (str(form.get("language") or "").strip()
                if task == "transcribe" else "")
    return {"format": fmt, "speakers": speakers, "words": "word" in grains,
            "temperature": temperature, "language": language}


def decode(audio_bytes: bytes, filename: str) -> np.ndarray:
    """The upload as 16 kHz mono samples. Raises STTRequestError."""
    from .patches.media_gate import MediaRefused

    suffix = os.path.splitext(filename or "")[1] or ".wav"
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    try:
        tmp.write(audio_bytes)
        tmp.close()
        try:
            return media_programs.decode_mono(tmp.name, SAMPLE_RATE,
                                              field="the uploaded file")
        except MediaRefused as e:
            raise STTRequestError(400, str(e)) from None
        except RuntimeError as e:
            raise STTRequestError(400, str(e)) from None
    finally:
        tmp.close()
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


def no_speaker(clips) -> bool:
    """True when no 80 ms frame of any clip has a speaker above
    :data:`SPEAKER_HEARD`."""
    for clip in clips:
        probs = getattr(clip, "speaker_probs", None)
        if probs is None or np.asarray(probs).size == 0:
            continue
        if float(np.max(probs)) > SPEAKER_HEARD:
            return False
    return True


def _check_no_speaker(clips) -> None:
    if no_speaker(clips):
        raise NoSpeech()


# A WAV of 10 ms of silence: the chat turn's audio part, for the route
# checks the chat path runs on it. The model hears the route's samples.
_PLACEHOLDER_WAV = (
    b"RIFF" + (36 + 320).to_bytes(4, "little") + b"WAVEfmt "
    + (16).to_bytes(4, "little") + (1).to_bytes(2, "little")
    + (1).to_bytes(2, "little") + SAMPLE_RATE.to_bytes(4, "little")
    + (2 * SAMPLE_RATE).to_bytes(4, "little") + (2).to_bytes(2, "little")
    + (16).to_bytes(2, "little") + b"data" + (320).to_bytes(4, "little")
    + bytes(320))


def _instruction(task: str, fields: dict) -> str:
    from gmlx.models.nemotron_h import transcription as tr

    if task != "transcribe":
        return tr.translation_instruction("English")
    return tr.transcription_instruction(fields["language"] or None,
                                        speakers=fields["speakers"])


def chat_request(model_id: str, task: str, fields: dict, seconds: float):
    """The chat turn that transcribes or translates the clip: the card's
    instruction, greedy unless the client sets a temperature, thinking off,
    and the card's output budget. Every sampling field is set, so no
    profile adds to it."""
    import base64

    from gmlx.models.nemotron_h import transcription as tr

    schemas = importlib.import_module("gmlx._vendor.mlx_vlm.server.schemas")
    audio = base64.b64encode(_PLACEHOLDER_WAV).decode()
    # The context check counts the prompt; the budget leaves it room.
    budget = tr.output_budget(seconds, 0)
    return schemas.ChatRequest(
        model=model_id, stream=False,
        messages=[{"role": "user", "content": [
            {"type": "text", "text": _instruction(task, fields)},
            {"type": "input_audio",
             "input_audio": {"data": audio, "format": "wav"}}]}],
        max_tokens=budget, temperature=fields["temperature"], top_p=1.0,
        top_k=0, min_p=0.0, repetition_penalty=1.0, presence_penalty=0.0,
        frequency_penalty=0.0, enable_thinking=False, thinking_budget=None,
        thinking_budget_message=None)


def _chat_endpoint():
    from .patches._common import _find_route

    app = importlib.import_module("gmlx._vendor.mlx_vlm.server.app").app
    route = _find_route(app, "/v1/chat/completions", "POST")
    if route is None:
        raise RuntimeError("the chat completions route is not installed")
    return route.endpoint


def aligns(task: str, fields: dict) -> bool:
    return task == "transcribe" and fields["format"] in _ALIGNED_FORMATS


async def run(http_request, model_id: str, audio: np.ndarray, task: str,
              fields: dict) -> dict:
    """``{"text", "duration", "usage"}`` for ``audio`` on ``model_id``, and
    ``"alignment"`` (``words``, ``error``) for a timed transcription."""
    from gmlx.models.nemotron_h import transcription as tr

    from .patches import native_audio

    duration = audio.size / SAMPLE_RATE
    if tr.near_silent(audio):
        return {"text": "", "duration": duration, "usage": None}
    request = chat_request(model_id, task, fields, duration)
    captures: list | None = [] if aligns(task, fields) else None
    t1 = native_audio.DECODED.set([audio])
    t2 = native_audio.ON_ENCODED.set(_check_no_speaker)
    t3 = native_audio.CAPTURE.set(captures)
    from fastapi import HTTPException

    try:
        response = await _chat_endpoint()(request, http_request)
    except NoSpeech:
        return {"text": "", "duration": duration, "usage": None}
    except HTTPException as e:
        if e.status_code >= 500:
            raise
        raise STTRequestError(e.status_code, str(e.detail)) from None
    finally:
        native_audio.CAPTURE.reset(t3)
        native_audio.ON_ENCODED.reset(t2)
        native_audio.DECODED.reset(t1)
    body = _body(response)
    choices = body.get("choices") or []
    text = ((choices[0].get("message") or {}).get("content") if choices else "")
    looped = tr.cleanup_generation_loops(str(text or ""))
    out = {"text": tr.strip_tags(looped, speakers=fields["speakers"]),
           "duration": duration, "usage": body.get("usage")}
    if captures:
        out["alignment"] = await align_transcript(http_request, model_id,
                                                  captures[0], looped)
    return out


_TAG_WORD = re.compile(r"<[^<>\s]+>")


async def align_transcript(http_request, model_id: str, capture,
                           transcript: str) -> dict:
    """``{"words", "error"}``: the transcript's words with ``start``,
    ``end`` and ``speaker`` (the ``<spk:N>`` number, or None), event tags
    left out, or the reason there are none. Any stream without a legal
    path fails the whole alignment."""
    from gmlx.serve.engine_jobs import JobCancelled, JobTimeout

    from . import word_timestamps

    try:
        result, error = await word_timestamps.align(http_request, model_id,
                                                    capture, transcript)
    except JobCancelled as e:
        raise STTRequestError(499, str(e)) from None
    except JobTimeout as e:
        raise STTRequestError(504, str(e)) from None
    if result is None:
        return {"words": [], "error": error}
    if result.failed:
        return {"words": [], "error": ", ".join(sorted({f["reason"] for f in result.failed}))}
    return {"words": [w for w in result.words if not _TAG_WORD.fullmatch(w["word"])],
            "error": None}


def _body(response) -> dict:
    """The chat route's answer as a dict. An error answer raises with its
    status and message."""
    import json

    from fastapi.responses import Response

    if isinstance(response, Response):
        try:
            body = json.loads(bytes(response.body))
        except ValueError:
            body = {}
        if response.status_code >= 400:
            err = body.get("error") if isinstance(body, dict) else None
            msg = (err.get("message") if isinstance(err, dict)
                   else body.get("detail") if isinstance(body, dict) else None)
            raise STTRequestError(response.status_code,
                                  str(msg or "the transcription failed"))
        return body if isinstance(body, dict) else {}
    dump: Callable[[], dict] | None = getattr(response, "model_dump", None)
    return dump() if callable(dump) else dict(response or {})


def _ts_round(seconds: float) -> float:
    return math.floor(seconds * 1000 + 0.5) / 1000


def _join(words: list[str]) -> str:
    out = ""
    for w in words:
        cjk = len(w) == 1 and "\u3000" <= w <= "\u9fff"
        prev = out[-1:]
        out += w if not out or (cjk and "\u3000" <= prev <= "\u9fff") else " " + w
    return out


def segments(words: list[dict], *, by_turn: bool = False) -> list[dict]:
    """Aligned words as ``{"start", "end", "text", "speaker"}`` segments. A
    speaker change starts a segment; without ``by_turn``, so do a sentence
    end and a pause of :data:`SEGMENT_PAUSE` seconds."""
    out: list[dict] = []
    for w in words:
        last = out[-1] if out else None
        if (last is None or w["speaker"] != last["speaker"]
                or (not by_turn and (_ends_sentence(last["_words"][-1])
                                     or w["start"] - last["end"] >= SEGMENT_PAUSE))):
            out.append({"start": w["start"], "end": w["end"],
                        "speaker": w["speaker"], "_words": [w["word"]]})
        else:
            last["end"] = max(last["end"], w["end"])
            last["_words"].append(w["word"])
    return [{"start": _ts_round(g["start"]), "end": _ts_round(g["end"]),
             "text": _join(g.pop("_words")), "speaker": g["speaker"]} for g in out]


def _speaker_number(tag) -> int | None:
    """verbose_json's speaker: the ``<spk:N>`` number plus one, as
    NeMo-Speech.cpp writes it."""
    return None if tag is None else int(tag) + 1


def _cues(segs: list[dict], speakers: bool, vtt: bool) -> list[dict]:
    """SRT and VTT cues: VTT names each cue's speaker in a voice tag, SRT
    names a speaker on the first cue of each turn."""
    out, prev = [], object()
    for g in segs:
        n = _speaker_number(g["speaker"]) if speakers else None
        text = g["text"]
        if n is not None and vtt:
            text = f"<v Speaker {n}>{text}"
        elif n is not None and n != prev:
            text = f"Speaker {n}: {text}"
        prev = n
        out.append({"start": g["start"], "end": g["end"], "text": text})
    return out


def format_result(result: dict, task: str, fields: dict):
    """``(content, media_type)`` in the requested format. A transcription
    whose alignment failed has no words or segments and names why
    (``alignment_error``); its SRT or VTT is one cue over the clip."""
    text, fmt = result["text"], fields["format"]
    duration = _ts_round(result["duration"])
    speakers = fields["speakers"]
    alignment = result.get("alignment")
    error = alignment.get("error") if alignment else None
    words = [{**w, "start": _ts_round(w["start"]), "end": _ts_round(w["end"])}
             for w in (alignment or {}).get("words", [])]
    if alignment is None or not text:
        # A translation, or nothing heard: one segment over the clip.
        segs = ([{"start": 0.0, "end": duration, "text": text, "speaker": None}]
                if text else [])
    elif error:
        segs = []
    else:
        segs = segments(words)
    if fmt == "text":
        return text + "\n", "text/plain"
    if fmt in ("srt", "vtt"):
        cues = (_cues(segs, speakers, fmt == "vtt") if segs or not text else
                [{"start": 0.0, "end": duration, "text": text}])
        return (_vtt(cues) if fmt == "vtt" else _srt(cues)), "text/plain"
    if fmt == "verbose_json":
        def spk(item):
            n = _speaker_number(item["speaker"]) if speakers else None
            return {} if n is None else {"speaker": n}

        out = {"task": "transcribe" if task == "transcribe" else "translate",
               "language": fields["language"] or None, "duration": duration,
               "text": text,
               "segments": [{"id": i, "start": g["start"], "end": g["end"],
                             "text": g["text"], **spk(g)} for i, g in enumerate(segs)]}
        if fields["words"]:
            out["words"] = [{"word": w["word"], "start": w["start"], "end": w["end"],
                             **spk(w)} for w in words]
        if error:
            out["alignment_error"] = error
        return out, "application/json"
    if fmt == "diarized_json":
        letters: dict = {}
        turns = [] if error or not text else segments(words, by_turn=True)
        for g in turns:
            letters.setdefault(g["speaker"], _letter(len(letters)))
        out = {"task": "transcribe", "duration": duration, "text": text,
               "segments": [{"id": f"seg_{i}", "type": "transcript.text.segment",
                             "speaker": letters[g["speaker"]], "start": g["start"],
                             "end": g["end"], "text": g["text"]}
                            for i, g in enumerate(turns)]}
        if error:
            out["alignment_error"] = error
        return out, "application/json"
    return {"text": text}, "application/json"


def _letter(i: int) -> str:
    """Speaker labels A to Z, then AA, AB and on."""
    s = ""
    i += 1
    while i:
        i, r = divmod(i - 1, 26)
        s = chr(65 + r) + s
    return s


async def answer(http_request, form, model_id: str, audio_bytes: bytes,
                 filename: str, task: str):
    """``(content, media_type)`` for a transcription or translation on the
    served model ``model_id``. Raises STTRequestError for a client error."""
    import asyncio

    if not audio_bytes:
        raise STTRequestError(400, "multipart field 'file' is required")
    if adapter(model_id) == "lfm3":
        from . import native_stt_lfm3

        return await native_stt_lfm3.answer(http_request, form, model_id,
                                            audio_bytes, filename, task)
    fields = _fields(form, task)
    audio = await asyncio.to_thread(decode, audio_bytes, filename)
    result = await run(http_request, model_id, audio, task, fields)
    return format_result(result, task, fields)

