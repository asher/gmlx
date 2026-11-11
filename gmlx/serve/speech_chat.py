# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Asher Feldman
"""Audio output on the chat route.

A request with ``modalities: ["text", "audio"]`` on a model that speaks
gets its reply as audio with a transcript. The reply is made one of two
ways:

- A spoken reply: one row that writes text and speech in turn.
- A text reply, then the speech pass: a second generation on the same
  model that reads the text aloud. A request with ``tools`` takes this
  way, since a tool call is text. So does a request for a voice that only
  the speech prompt selects.

``/v1/audio/speech`` on a served model is the speech pass alone
(:mod:`gmlx.serve.native_tts`). It reaches the chat route in process
with :data:`SPEECH_ONLY` set.

The chat route calls :func:`plan` once per request and holds the
:class:`Reply` it returns.
"""

from __future__ import annotations

import base64
import logging
import threading
import time
import uuid
from collections import OrderedDict
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Iterator, Optional

import numpy as np

from . import native_tts

_log = logging.getLogger(__name__)

SAMPLE_RATE = 24000
# OpenAI's chat audio formats. A stream sends ``pcm16`` only.
FORMATS = ("wav", "mp3", "flac", "opus", "pcm16")
DEFAULT_FORMAT = "wav"
STREAM_FORMAT = "pcm16"

SPOKEN, TEXT_FIRST, ONLY = "spoken", "text_first", "only"

# How long a reply's transcript stays for a later turn that names the
# reply by its audio id, and how many the server keeps.
TRANSCRIPT_SECONDS = 3600
TRANSCRIPTS_KEPT = 1024
_TRANSCRIPTS: OrderedDict = OrderedDict()
_TRANSCRIPTS_LOCK = threading.Lock()


@dataclass(frozen=True)
class SpeechOnly:
    """A ``/v1/audio/speech`` request: the text to read, already
    sanitized, and the checked voice and language."""

    text: str
    voice: str
    language: Optional[str] = None


SPEECH_ONLY: ContextVar[Optional[SpeechOnly]] = ContextVar(
    "gmlx_speech_only", default=None)


def _refuse(message: str):
    from .patches._common import refusal_type

    return refusal_type()(status_code=400, detail=message)


def remember(audio_id: str, transcript: str) -> int:
    """Keep ``transcript`` for a later turn that names ``audio_id``.
    Returns the time it expires."""
    expires = int(time.time()) + TRANSCRIPT_SECONDS
    with _TRANSCRIPTS_LOCK:
        _TRANSCRIPTS[audio_id] = (expires, transcript)
        while len(_TRANSCRIPTS) > TRANSCRIPTS_KEPT:
            _TRANSCRIPTS.popitem(last=False)
    return expires


def history_text(message) -> Optional[str]:
    """The text of an earlier assistant turn that holds only an audio id,
    as OpenAI clients send a spoken reply back. None for any other
    message. A reply's audio is not kept, so the turn goes back to the
    model as its transcript. Raises a 400 for an id the server does not
    hold."""
    audio = getattr(message, "audio", None)
    if getattr(message, "role", None) != "assistant" \
            or not isinstance(audio, dict) or not audio.get("id"):
        return None
    if isinstance(audio.get("transcript"), str):
        return audio["transcript"]
    with _TRANSCRIPTS_LOCK:
        held = _TRANSCRIPTS.get(str(audio["id"]))
    if held is None or held[0] < time.time():
        raise _refuse(
            f"audio id {audio['id']!r} is not held by this server. Send "
            "the turn's transcript as its content")
    return held[1]


def pcm16(audio) -> bytes:
    """Float samples as 16-bit little-endian bytes."""
    samples = np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0)
    return np.round(samples * 32767.0).astype("<i2").tobytes()


def speech_prompt(processor, text: str, voice: str,
                  language: Optional[str], marker: Optional[int]) -> str:
    """The model's speech prompt for ``text``, without the mode marker
    ``marker``: the engine forces the marker as the reply's first token,
    and the model's own speech template ends in it."""
    from gmlx.models.lfm3 import speak

    tokenizer = getattr(processor, "tokenizer", processor)
    prompt = speak.tts_prompt(tokenizer, text, voice, language)
    piece = (None if marker is None
             else tokenizer.convert_ids_to_tokens(int(marker)))
    if isinstance(piece, str) and piece and prompt.endswith(piece):
        prompt = prompt[:-len(piece)]
    return prompt


def _audio_field(request) -> dict:
    audio = getattr(request, "audio", None)
    if audio is None:
        return {}
    if not isinstance(audio, dict):
        raise _refuse("audio must be an object with voice and format")
    return audio


def wanted(request) -> bool:
    """Whether the request asks for audio output."""
    modalities = getattr(request, "modalities", None)
    return isinstance(modalities, (list, tuple)) and "audio" in modalities


def plan(request, model, generator, tools) -> Optional["Reply"]:
    """The audio output of this request, or None for a text reply.
    Raises a 400 for a request the model cannot speak."""
    only = SPEECH_ONLY.get()
    if only is None and not wanted(request):
        if getattr(request, "audio", None) is not None:
            _log.warning("chat request sets audio without modalities "
                         "[\"text\", \"audio\"]; the reply is text")
        return None
    capabilities = getattr(model, "speech_capabilities", lambda: None)()
    if capabilities is None:
        raise _refuse(f"model {request.model!r} has no speech output")
    if getattr(generator, "draft_model", None) is not None:
        raise _refuse(
            f"model {request.model!r} is served with a drafter, and speech "
            "output does not run with one. Serve the model without the "
            "drafter to get audio")
    if getattr(request, "logprobs", None):
        raise _refuse("logprobs are not available with audio output")
    if getattr(request, "response_format", None) is not None:
        raise _refuse(
            "response_format is not available with audio output")
    stream = bool(getattr(request, "stream", False))
    marker = getattr(getattr(getattr(model, "config", None), "speech_config",
                             None), "audio_start_token_id", None)
    if only is not None:
        return Reply(ONLY, only.voice, STREAM_FORMAT, only.language, stream,
                     only=only, marker=marker)
    audio = _audio_field(request)
    try:
        voice = native_tts.voice_of(audio.get("voice"))
        language = native_tts.language_of(audio.get("language"))
    except ValueError as e:
        raise _refuse(str(e)) from None
    fmt = str(audio.get("format") or "").strip().lower() or (
        STREAM_FORMAT if stream else DEFAULT_FORMAT)
    if fmt not in FORMATS:
        raise _refuse(f"unsupported audio format {fmt!r} "
                      f"(supported: {', '.join(FORMATS)})")
    if stream and fmt != STREAM_FORMAT:
        raise _refuse(f"a stream sends audio as {STREAM_FORMAT}; send "
                      f"audio.format {STREAM_FORMAT!r}")
    problem = native_tts.format_problem(fmt)
    if problem:
        raise _refuse(problem)
    # Only the speech prompt selects a voice, and a tool call is text.
    spoken = not tools and voice == native_tts.voices()[0]
    return Reply(SPOKEN if spoken else TEXT_FIRST, voice, fmt, language,
                 stream, text_is_content=bool(tools), marker=marker)


class Reply:
    """One request's audio output: how the reply is made, and the audio
    and transcript so far."""

    def __init__(self, kind: str, voice: str, fmt: str,
                 language: Optional[str], stream: bool, *,
                 only: Optional[SpeechOnly] = None,
                 text_is_content: bool = False,
                 marker: Optional[int] = None):
        self.kind, self.voice, self.format = kind, voice, fmt
        self.language, self.stream, self.only = language, stream, only
        # The id of the marker that starts a speech pass.
        self.marker = marker
        # With tools the route does not know a reply is plain text until
        # it ends, so the text is the message's content.
        self.text_is_content = text_is_content
        self.id = f"audio_{uuid.uuid4().hex}"
        self.pcm = bytearray()
        self.text: list = []
        # Frames of audio, end frames left out.
        self.frames = 0
        # Forced mode markers. Each is a prompt token in the usage.
        self.markers = 0
        # The speech passes: how many ran, their prompt tokens and their
        # output tokens.
        self.passes = 0
        self.pass_prompt_tokens = 0
        self.pass_tokens = 0

    # -- before the generation ------------------------------------------

    def messages(self, messages: list) -> list:
        """The request's messages. A spoken reply with no system message
        gets the card's instruction for one."""
        if self.kind != SPOKEN or any(
                m.get("role") in ("system", "developer") for m in messages):
            return messages
        from gmlx.models.lfm3 import speak

        return [{"role": "system", "content": speak.INTERLEAVED_SYSTEM},
                *messages]

    def prompt(self, processor, prompt: str, args, generator) -> str:
        """The prompt of the first generation. For the speech route it is
        the speech prompt, and ``args`` gets its frame budget."""
        if self.kind != ONLY:
            return prompt
        assert self.only is not None
        prompt = speech_prompt(processor, self.only.text, self.voice,
                               self.language, self.marker)
        args.max_tokens = self._budget(generator, processor, prompt,
                                       self.only.text, args.max_tokens)
        return prompt

    def stamp(self, args):
        """Mark the first generation's row as a speaking row."""
        if self.kind == SPOKEN:
            args._kq_speech = {"mode": "interleaved"}
        elif self.kind == ONLY:
            args._kq_speech = {"mode": "audio", "stop_at_end": True}
        return args

    def _budget(self, generator, processor, prompt: str, text: str,
                most: Optional[int]) -> int:
        """The frames a speech pass over ``text`` gets: the text's
        allowance, under ``most`` and under the context the prompt
        leaves. Raises a 400 for a text that leaves no room."""
        from gmlx._vendor.mlx_vlm.server.generation import (
            get_configured_context_limit)

        budget = native_tts.frame_budget(text)
        if most is not None:
            budget = min(budget, int(most))
        limit = get_configured_context_limit()
        if limit is not None:
            tokenizer = getattr(processor, "tokenizer", processor)
            lock = getattr(generator, "_tokenizer_lock", None)
            if lock is not None:
                with lock:
                    used = len(tokenizer.encode(prompt))
            else:
                used = len(tokenizer.encode(prompt))
            room = int(limit) - used - 2
            if room < native_tts.MIN_FRAMES:
                raise _refuse(
                    f"the text is {used} tokens long, which leaves no room "
                    f"to speak it in this model's {limit}-token context. "
                    "Send it in parts")
            budget = min(budget, room)
        return max(budget, 0)

    # -- during a generation --------------------------------------------

    def take(self, token) -> Optional[bytes]:
        """Count one token of a speaking row. Returns its audio as 16-bit
        pcm, or None."""
        kind = getattr(token, "speech_kind", None)
        if kind == "marker":
            self.markers += 1
        elif kind == "frame":
            self.frames += 1
        audio = getattr(token, "audio", None)
        if audio is None or not getattr(audio, "size", 0):
            return None
        chunk = pcm16(audio)
        self.pcm += chunk
        return chunk

    def delta(self, token, content) -> tuple:
        """``(content, audio)`` of a stream chunk for ``token``, whose
        text the route filtered to ``content``. ``audio`` is the chunk's
        ``delta.audio`` object or None."""
        chunk = self.take(token)
        audio: dict = {}
        if content and not self.text_is_content:
            self.text.append(content)
            audio["transcript"] = content
            content = None
        elif content:
            self.text.append(content)
        if chunk is not None:
            audio["data"] = base64.b64encode(chunk).decode()
        if audio:
            audio["id"] = self.id
        return content, (audio or None)

    # -- the speech pass -------------------------------------------------

    def wants_pass(self, tool_calls: bool) -> bool:
        """Whether the finished text reply is spoken by a pass."""
        return (self.kind == TEXT_FIRST and not tool_calls
                and bool(self.transcript()))

    def speak(self, generator, processor, args) -> Iterator[bytes]:
        """Run the speech pass over the reply's text and yield its audio
        as 16-bit pcm chunks. The request's ``max_tokens`` bounds the text
        alone: the pass takes the text's own allowance, under the context.
        A text too long for one pass is spoken piece by piece. Blocks, so
        the route runs it on a thread."""
        from .tts import sanitize_speech_text

        spoken = sanitize_speech_text(self.transcript())
        for index, piece in enumerate(native_tts.split_text(spoken)):
            yield from self._speak_piece(generator, processor, args, piece,
                                         index)

    def _speak_piece(self, generator, processor, args, piece: str,
                     index: int) -> Iterator[bytes]:
        from gmlx._vendor.mlx_vlm.server.generation import GenerationArguments

        prompt = speech_prompt(processor, piece, self.voice, self.language,
                               self.marker)
        budget = self._budget(generator, processor, prompt, piece, None)
        # The pass is its own request: only the seed, the tenant and the
        # adapters of the reply carry over.
        pass_args = GenerationArguments(
            max_tokens=budget, temperature=0.0, enable_thinking=False,
            seed=None if args.seed is None else int(args.seed) + index,
            tenant_id=args.tenant_id)
        lora = getattr(args, "_kq_lora", None)
        if lora is not None:
            pass_args._kq_lora = lora  # type: ignore[attr-defined]
        pass_args._kq_speech = {  # type: ignore[attr-defined]
            "mode": "audio", "stop_at_end": True}
        ctx, tokens = generator.generate(prompt, None, None, args=pass_args)
        self.passes += 1
        self.pass_prompt_tokens += int(ctx.prompt_tokens)
        try:
            for token in tokens:
                self.pass_tokens += getattr(token, "token_count", 1)
                chunk = self.take(token)
                if chunk is not None:
                    yield chunk
                if token.finish_reason:
                    if token.finish_reason == "length":
                        _log.warning(
                            "speech pass stopped at its %d-frame allowance "
                            "for %d characters", budget, len(piece))
                    break
        finally:
            tokens.close()

    def audio_delta(self, chunk: bytes, first: bool) -> dict:
        """The ``delta.audio`` object of a pass's chunk. With the text as
        content, the first chunk carries the transcript."""
        audio: dict = {"id": self.id,
                       "data": base64.b64encode(chunk).decode()}
        if first and self.text_is_content:
            audio["transcript"] = "".join(self.text)
        return audio

    # -- after the reply --------------------------------------------------

    def transcript(self) -> str:
        return "".join(self.text).strip()

    def message_audio(self, generator, processor, args,
                      content: Optional[str],
                      tool_calls: bool) -> Optional[dict]:
        """The ``audio`` object of a message that is not streamed, whose
        text is ``content``, or None for a reply with no audio. Runs the
        speech pass when the reply takes one, and encodes the audio, so
        the route runs it on a thread."""
        self.text = [content or ""]
        if self.wants_pass(tool_calls):
            for _ in self.speak(generator, processor, args):
                pass
        if not self.pcm:
            return None
        transcript = self.transcript()
        data = native_tts.encode(bytes(self.pcm), SAMPLE_RATE, self.format)
        return {"id": self.id, "data": base64.b64encode(data).decode(),
                "expires_at": remember(self.id, transcript),
                "transcript": transcript}

    def message_content(self, content: Optional[str],
                        audio: Optional[dict]) -> Optional[str]:
        """The message's ``content``: null for a reply whose text is its
        audio's transcript, as OpenAI sends a spoken reply."""
        return content if self.text_is_content or audio is None else None

    def close_stream(self) -> Optional[dict]:
        """The last ``delta.audio`` object of a stream: the id and when
        its transcript expires."""
        if not self.pcm:
            return None
        return {"id": self.id,
                "expires_at": remember(self.id, self.transcript())}

    def usage(self, prompt_tokens: int, completion_tokens: int) -> dict:
        """The reply's token counts. A forced mode marker is a prompt
        token, as it is in the model's own prompt. The speech pass adds
        its prompt and its frames."""
        markers = self.markers
        if self.kind == ONLY:
            tts = prompt_tokens + markers
        else:
            # Each pass's prompt and its marker.
            tts = self.pass_prompt_tokens + self.passes
        prompt = prompt_tokens + self.pass_prompt_tokens + markers
        completion = max(0, completion_tokens + self.pass_tokens - markers)
        return {"prompt_tokens": prompt, "completion_tokens": completion,
                "total_tokens": prompt + completion,
                "audio_tokens": self.frames, "tts_tokens": tts}


def apply_usage(usage: Any, counts: dict) -> Any:
    """``usage`` (a UsageStats) with a reply's counts."""
    usage.prompt_tokens = counts["prompt_tokens"]
    usage.completion_tokens = counts["completion_tokens"]
    usage.total_tokens = counts["total_tokens"]
    usage.prompt_tokens_details.tts_tokens = counts["tts_tokens"]
    usage.completion_tokens_details = {"audio_tokens": counts["audio_tokens"]}
    return usage
