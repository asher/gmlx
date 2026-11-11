# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Asher Feldman
"""An LFM3 reply in text, speech, or both: the state of one reply and a
single-stream generator for run, chat and tests.

Ported from the LFM3-Omni reference ``modeling_lfm3_omni.py``
(``_generate``, Apache-2.0) and llama.cpp ``mtmd-helper-gen.cpp``
(``lfm3a_gen_audio_pipeline``, MIT, contributed by Liquid AI).

Each step of the text model fills one slot of the reply. A text slot
samples a token from the head. A speech slot samples a frame of codes
from the hidden state (:class:`~gmlx.models.lfm3.speech.DepthDecoder`),
and the frame's embedding is the text model's next input.

The reply's mode says which slots come next:

- ``text``: text slots. A sampled ``<|audio_start|>`` or
  ``<|mixed_start|>`` changes the mode.
- ``audio``: speech slots until a frame holds the end code. The reply
  then goes on in text.
- ``interleaved``: 6 text slots, then 12 speech slots, and so on.
  ``<|text_end|>`` ends the text, and the rest is speech up to the end
  code.

The mode starts from the prompt's last token.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Iterator, List, Optional, Sequence

import mlx.core as mx
import numpy as np

from .speech import CodecStream, SpeechConfig

TEXT, AUDIO, INTERLEAVED = "text", "audio", "interleaved"
MODES = (TEXT, AUDIO, INTERLEAVED)

VOICES = ("female", "male")
TTS_TEMPLATE = "tts"
# The card's speech instruction without its language.
TTS_ANY = "Perform TTS using a {voice} voice."
# The codes the tts template names a language for.
TTS_LANGUAGES = (
    "ar", "da", "de", "el", "en", "en-gb", "en-us", "es", "fi", "fr", "he",
    "it", "ja", "ko", "nl", "pl", "pt", "sv", "tr", "zh")
# The card's system prompt for a spoken reply with its transcript.
INTERLEAVED_SYSTEM = "Respond with interleaved text and audio."

DEFAULT_PREFILL_STEP = 2048
_CLEAR_EVERY = 256


class SpeechState:
    """Where one reply stands between text and speech."""

    def __init__(self, config: SpeechConfig, mode: str = TEXT, *,
                 text_only: bool = False, end_of_turn: Sequence[int] = ()):
        if mode not in MODES:
            raise ValueError(
                f"speech mode {mode!r} is not one of {', '.join(MODES)}")
        self.config = config
        self.mode = mode
        # An explicit text reply cannot start speech.
        self.text_only = text_only
        self.end_of_turn = frozenset(
            int(t) for t in end_of_turn) - {config.text_end_token_id}
        self.slots_left = config.interleaved_n_text
        self.speaking = False
        self.text_done = False
        self.finished = False
        # The frames the repetition penalty reads, oldest first.
        self.history: deque = deque(maxlen=max(0, config.repetition_window))
        self.frames = 0
        self.stream: Optional[CodecStream] = None

    @property
    def wants_frame(self) -> bool:
        """Whether the next slot is a frame of codes."""
        return not self.finished and (
            self.mode == AUDIO
            or (self.mode == INTERLEAVED and self.speaking))

    def _count(self, interleaving: bool) -> None:
        cfg = self.config
        if interleaving and not self.text_done:
            self.slots_left -= 1
            if self.slots_left == 0:
                self.speaking = not self.speaking
                self.slots_left = (cfg.interleaved_n_audio if self.speaking
                                   else cfg.interleaved_n_text)

    def take_token(self, token: int) -> None:
        """Record the token a text slot sampled."""
        cfg = self.config
        interleaving = self.mode == INTERLEAVED
        if token == cfg.text_end_token_id:
            if interleaving:
                self.text_done = self.speaking = True
            else:
                self.finished = True
        elif token in self.end_of_turn:
            self.finished = True
        elif interleaving:
            # A mode marker only opens a block. Inside one it is text.
            pass
        elif token == cfg.audio_start_token_id:
            self.mode = AUDIO
        elif token == cfg.mixed_start_token_id:
            self.mode = INTERLEAVED
            self.slots_left = cfg.interleaved_n_text
            self.speaking = self.text_done = False
        self._count(interleaving)

    def take_frame(self, codes) -> bool:
        """Record the frame a speech slot sampled. True when it holds the
        end code: the audio ends and the reply goes on in text."""
        interleaving = self.mode == INTERLEAVED
        codes = np.asarray(codes).reshape(-1)
        ended = bool((codes == self.config.eos_code).any())
        if ended:
            self.mode = TEXT
        else:
            self.history.append(codes)
            self.frames += 1
        self._count(interleaving)
        return ended


def prompt_mode(config: SpeechConfig, last_token: int) -> str:
    """The mode a prompt that ends in ``last_token`` starts in."""
    if last_token in (config.audio_start_token_id, config.audio_out_token_id):
        return AUDIO
    if last_token == config.mixed_start_token_id:
        return INTERLEAVED
    return TEXT


def start(config: SpeechConfig, last_token: int, mode: Optional[str] = None,
          end_of_turn: Sequence[int] = ()) -> tuple:
    """``(state, marker)`` for a reply to a prompt that ends in
    ``last_token``. ``mode`` None reads the mode off the prompt. A named
    mode gives the marker token the prompt still needs, else None."""
    for name in ("audio_start_token_id", "mixed_start_token_id",
                 "text_end_token_id", "audio_out_token_id"):
        if getattr(config, name) is None:
            raise ValueError(f"the speech config has no {name}")
    if mode is None:
        return SpeechState(config, prompt_mode(config, last_token),
                           end_of_turn=end_of_turn), None
    if mode not in MODES:
        raise ValueError(
            f"speech mode {mode!r} is not one of {', '.join(MODES)}")
    marker = None
    if mode != TEXT and prompt_mode(config, last_token) != mode:
        marker = (config.audio_start_token_id if mode == AUDIO
                  else config.mixed_start_token_id)
    return SpeechState(config, mode, text_only=mode == TEXT,
                       end_of_turn=end_of_turn), marker


def tts_prompt(tokenizer, text: str, voice: Optional[str] = None,
               language: Optional[str] = None) -> str:
    """The prompt that makes the model read ``text`` aloud. With a
    ``language``, a code or a name, it is the model's ``tts`` template.
    Without one it is the card's instruction with the language left out,
    and the model reads the language from the text. A reply to it starts
    in the ``audio`` mode."""
    if voice is not None and voice.lower() not in VOICES:
        raise ValueError(
            f"voice {voice!r} is not one of {', '.join(VOICES)}")
    if not language:
        return tokenizer.apply_chat_template(
            [{"role": "system",
              "content": TTS_ANY.format(voice=(voice or VOICES[0]).lower())},
             {"role": "user", "content": text}],
            add_generation_prompt=True, tokenize=False)
    source = (getattr(tokenizer, "gguf_named_templates", None) or {}).get(
        TTS_TEMPLATE)
    if not source:
        raise ValueError("this model's GGUF has no tts chat template")
    from transformers.utils.chat_template_utils import _compile_jinja_template

    return _compile_jinja_template(source).render(
        messages=[{"role": "user", "content": text}],
        bos_token=getattr(tokenizer, "bos_token", None) or "",
        language=language, **({"voice": voice} if voice else {}))


@dataclass
class SpeechEvent:
    """One slot of a reply, or its closing summary."""

    token: Optional[int] = None
    text: str = ""
    # A frame's codes, ``[codebooks]``.
    codes: Optional[np.ndarray] = None
    # Float32 samples ready to play, at the codec's sample rate.
    audio: Optional[np.ndarray] = None
    finish_reason: Optional[str] = None
    prompt_tokens: int = 0
    text_tokens: int = 0
    frames: int = 0
    prompt_seconds: float = 0.0
    first_audio_seconds: Optional[float] = None
    generation_seconds: float = 0.0
    peak_memory_gb: float = 0.0


def _greedy(logprobs: mx.array) -> mx.array:
    return mx.argmax(logprobs, axis=-1)


def speech_slots(
    model,
    input_ids: mx.array,
    inputs_embeds: mx.array,
    state: SpeechState,
    *,
    max_tokens: int = 1024,
    sampler: Optional[Callable[[mx.array], mx.array]] = None,
    logits_processors: Optional[List[Callable]] = None,
    key: Optional[mx.array] = None,
    prefill_step_size: Optional[int] = DEFAULT_PREFILL_STEP,
    max_kv_size: Optional[int] = None,
) -> Iterator[SpeechEvent]:
    """The slots of one reply to a prepared prompt: ``input_ids`` is
    ``[1, tokens]`` and ``inputs_embeds`` its rows with the media in
    place. Yields an event per slot, with no text, then the summary."""
    from gmlx._vendor.mlx_vlm.generate.common import generation_stream
    from gmlx._vendor.mlx_vlm.models import cache as vlm_cache

    lm = model.language_model
    sampler = sampler or _greedy
    processors = list(logits_processors or [])
    if key is None:
        key = mx.random.key(int(time.time_ns() & 0x7FFFFFFF))
    prompt_tokens = int(input_ids.shape[1])
    max_tokens = max(0, int(max_tokens))

    mx.reset_peak_memory()
    tic = time.perf_counter()
    with mx.stream(generation_stream):
        prompt_cache = vlm_cache.make_prompt_cache(lm, max_kv_size=max_kv_size)
        step = prefill_step_size or prompt_tokens
        done = 0
        while prompt_tokens - done > step:
            lm.model(input_ids[:, done:done + step], prompt_cache,
                     inputs_embeds[:, done:done + step])
            mx.eval([c.state for c in prompt_cache])
            done += step
            mx.clear_cache()
        hidden = lm.model(input_ids[:, done:], prompt_cache,
                          inputs_embeds[:, done:])[:, -1]
        mx.eval(hidden)
    prompt_seconds = time.perf_counter() - tic
    tic = time.perf_counter()

    seen_tokens = mx.array([], dtype=input_ids.dtype)
    text_tokens = 0
    first_audio = None
    no_ids = mx.zeros((1, 1), dtype=input_ids.dtype)
    finish_reason = "length"
    for n in range(max_tokens):
        last = n == max_tokens - 1
        if state.wants_frame:
            with mx.stream(generation_stream):
                keys = mx.random.split(key)
                key = keys[0]
                codes, embedding = model.speech_step(
                    [state], hidden, keys[1][None])
                pending = [codes, key]
                if not last:
                    hidden = lm.model(no_ids, prompt_cache, embedding)[:, -1]
                    pending.append(hidden)
                # The text model's step runs while the host decodes the
                # frame's audio.
                mx.async_eval(*pending)
            frame = np.asarray(codes)[0]
            ended = state.take_frame(frame)
            chunk = model.speech_flush(state) if ended \
                else model.speech_audio(state, frame)
            if chunk is not None and chunk.size and first_audio is None:
                first_audio = time.perf_counter() - tic
            yield SpeechEvent(codes=None if ended else frame, audio=chunk)
        else:
            with mx.stream(generation_stream):
                logits = model.speech_text_logits(state, hidden)
                for processor in processors:
                    logits = processor(seen_tokens, logits)
                logprobs = logits - mx.logsumexp(
                    logits, axis=-1, keepdims=True)
                token = int(sampler(logprobs).item())
            state.take_token(token)
            if state.finished:
                finish_reason = "stop"
                break
            text_tokens += 1
            if processors:
                seen_tokens = mx.concatenate(
                    [seen_tokens, mx.array([token], dtype=input_ids.dtype)])
            if not last:
                with mx.stream(generation_stream):
                    hidden = lm.model(
                        mx.array([[token]], dtype=input_ids.dtype),
                        prompt_cache)[:, -1]
                    mx.async_eval(hidden)
            yield SpeechEvent(token=token)
        if (n + 1) % _CLEAR_EVERY == 0:
            mx.clear_cache()

    yield SpeechEvent(
        audio=model.speech_flush(state),
        finish_reason=finish_reason,
        prompt_tokens=prompt_tokens,
        text_tokens=text_tokens,
        frames=state.frames,
        prompt_seconds=prompt_seconds,
        first_audio_seconds=first_audio,
        generation_seconds=time.perf_counter() - tic,
        peak_memory_gb=mx.get_peak_memory() / 1e9)


def stream_speech(
    model,
    processor,
    prompt: str,
    *,
    image=None,
    audio=None,
    video=None,
    mode: Optional[str] = None,
    **kwargs: Any,
) -> Iterator[SpeechEvent]:
    """Generate one reply on one row. Yields a :class:`SpeechEvent` per
    slot, then a summary with ``finish_reason`` set.

    ``prompt`` is rendered text. ``mode`` None reads the mode off the
    prompt's last token. A text slot and a frame each count as one of
    ``max_tokens``. ``sampler`` and ``logits_processors`` apply to text
    slots, with mlx-lm's contracts. The codes sample with the model's
    own settings, and ``key`` seeds them. Other keywords go to the
    processor."""
    from gmlx._vendor.mlx_vlm.generate.common import (generation_stream,
                                                      wired_limit)
    from gmlx._vendor.mlx_vlm.tokenizer_utils import make_streaming_detokenizer
    from gmlx._vendor.mlx_vlm.utils import (prepare_inputs,
                                            should_add_special_tokens)

    if getattr(model, "speech_capabilities", lambda: None)() is None:
        raise ValueError("this model has no speech output")
    slot_kwargs = {name: kwargs.pop(name) for name in (
        "max_tokens", "sampler", "logits_processors", "key",
        "prefill_step_size", "max_kv_size") if name in kwargs}
    tokenizer = getattr(processor, "tokenizer", processor)
    config: SpeechConfig = model.config.speech_config

    inputs = prepare_inputs(
        processor, images=image or None, audio=audio or None,
        videos=video or None, prompts=prompt,
        image_token_index=getattr(model.config, "image_token_index", None),
        add_special_tokens=should_add_special_tokens(
            model.config.model_type, processor),
        **kwargs)
    input_ids = inputs["input_ids"]
    data = {k: v for k, v in inputs.items()
            if k not in ("input_ids", "pixel_values", "attention_mask")}
    state, marker = model.speech_start(
        int(input_ids[0, -1].item()), mode,
        end_of_turn=getattr(tokenizer, "eos_token_ids", None) or ())
    if marker is not None:
        input_ids = mx.concatenate(
            [input_ids, mx.array([[marker]], dtype=input_ids.dtype)], axis=1)
    control = {config.audio_start_token_id, config.mixed_start_token_id,
               config.text_end_token_id}

    with wired_limit(model, [generation_stream]):
        with mx.stream(generation_stream):
            embeds = model.get_input_embeddings(
                input_ids, inputs.get("pixel_values"), **data).inputs_embeds
        detokenizer = make_streaming_detokenizer(processor)
        for event in speech_slots(model, input_ids, embeds, state,
                                  **slot_kwargs):
            if event.token is not None and event.token not in control:
                detokenizer.add_token(event.token)
                event.text = detokenizer.last_segment
            elif event.finish_reason is not None:
                detokenizer.finalize()
                event.text = detokenizer.last_segment
            yield event


@dataclass
class SpeechResult:
    text: str
    audio: np.ndarray
    codes: np.ndarray
    summary: SpeechEvent


def generate_speech(model, processor, prompt: str, **kwargs) -> SpeechResult:
    """Run :func:`stream_speech` to its end and join the pieces."""
    text: List[str] = []
    audio: List[np.ndarray] = []
    codes: List[np.ndarray] = []
    summary = SpeechEvent()
    for event in stream_speech(model, processor, prompt, **kwargs):
        text.append(event.text)
        if event.audio is not None and event.audio.size:
            audio.append(event.audio)
        if event.codes is not None:
            codes.append(event.codes)
        summary = event
    books = model.config.speech_config.num_codebooks
    return SpeechResult(
        text="".join(text),
        audio=np.concatenate(audio) if audio
        else np.zeros(0, dtype=np.float32),
        codes=np.stack(codes) if codes
        else np.zeros((0, books), dtype=np.int32),
        summary=summary)
