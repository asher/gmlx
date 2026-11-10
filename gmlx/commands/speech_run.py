# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Asher Feldman
"""``gmlx run --speech-out``: a spoken reply from a model with speech
output, or the prompt's text read aloud with ``--tts``."""

from __future__ import annotations

import os
import sys
import wave

import numpy as np

VOICES = ("female", "male")


def add_args(ap) -> None:
    group = ap.add_argument_group("speech output")
    group.add_argument(
        "--speech-out",
        default=None,
        metavar="FILE.wav",
        help="Save the reply as speech in a WAV file (a model with "
        "speech output). The reply's text prints as it comes.",
    )
    group.add_argument(
        "--tts",
        action="store_true",
        help="With --speech-out: read the prompt's text aloud and give "
        "no answer.",
    )
    group.add_argument(
        "--voice",
        default=None,
        choices=VOICES,
        help="With --tts: the voice. Default: female.",
    )
    group.add_argument(
        "--speech-language",
        default=None,
        metavar="NAME",
        help="With --tts: the language of the text, as a name or a code "
        "(French, fr). Default: the model reads it from the text.",
    )


def refusal(args) -> str | None:
    """Why this run's speech flags cannot work, before the model loads."""
    out = getattr(args, "speech_out", None)
    tts = getattr(args, "tts", False)
    extras = [flag for flag, value in (
        ("--voice", getattr(args, "voice", None)),
        ("--speech-language", getattr(args, "speech_language", None)))
        if value]
    if not out:
        flags = (["--tts"] if tts else []) + extras
        if flags:
            return f"{flags[0]} needs --speech-out FILE.wav"
        return None
    if not out.lower().endswith(".wav"):
        return "--speech-out writes a WAV file. Name it FILE.wav"
    if extras and not tts:
        return (f"{extras[0]} applies to --tts. A spoken reply uses the "
                "model's own voice and the language of the conversation")
    from gmlx.load.discovery import projector_caps

    if not projector_caps(getattr(args, "mmproj", None)).speech_out:
        return ("--speech-out needs a model with speech output, loaded "
                "with the --mmproj that holds its speech generator")
    if tts:
        if getattr(args, "image", None) or getattr(args, "audio", None):
            return "--tts reads the prompt's text and takes no --image or --audio"
        if getattr(args, "system_prompt", None):
            return "--tts uses the model's own speech prompt. Drop --system-prompt"
        if getattr(args, "no_chat_template", False):
            return "--tts uses the model's own speech prompt. Drop --no-chat-template"
    return None


def write_wav(path: str, audio: np.ndarray, sample_rate: int) -> None:
    pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(path, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(sample_rate)
        out.writeframes(pcm.tobytes())


def _prompt(args, model, processor, config, images, audios):
    """``(prompt text, mode)`` of the run."""
    from gmlx.models.lfm3 import speak

    tokenizer = getattr(processor, "tokenizer", processor)
    if args.tts:
        return speak.tts_prompt(tokenizer, args.prompt, args.voice,
                                args.speech_language), speak.AUDIO
    if args.no_chat_template:
        # The prompt's last token says how the reply starts.
        return args.prompt, None
    from typing import cast

    from gmlx._vendor.mlx_vlm.prompt_utils import apply_chat_template
    from gmlx.tui.chat import fold_thinking_flag, parse_template_config

    template_kwargs = fold_thinking_flag(
        args, parse_template_config(args.chat_template_config))
    messages = [
        {"role": "system",
         "content": args.system_prompt or speak.INTERLEAVED_SYSTEM},
        {"role": "user", "content": args.prompt},
    ]
    return cast(str, apply_chat_template(
        processor, config, messages, num_images=len(images),
        num_audios=len(audios), **template_kwargs)), speak.INTERLEAVED


def run(args) -> int:
    """Generate the reply, print its text and save its speech."""
    import mlx.core as mx

    import gmlx.load.loadlog as loadlog
    from gmlx._vendor.mlx_lm.sample_utils import (make_logits_processors,
                                                  make_sampler)
    from gmlx.commands import cli
    from gmlx.load.vlm import load_vlm_model
    from gmlx.models.lfm3 import speak
    from gmlx.tui.chat import parse_logit_bias

    logit_bias = parse_logit_bias(args.logit_bias)
    images, audios = cli._media_inputs(args)
    with loadlog.load_ui(args.verbose, args.gguf):
        model, config, processor, *_ = load_vlm_model(
            args.gguf,
            args.mmproj,
            hf_source=args.hf_source,
            arch=args.arch,
            zero_copy=not args.no_zero_copy,
            rope_override=cli.rope_override_from_args(args),
            verbose=args.verbose,
        )
    capabilities = getattr(model, "speech_capabilities", lambda: None)()
    if capabilities is None:
        print("error: gmlx has no speech output for this model.",
              file=sys.stderr)
        return 1
    cli.print_family_note(args)
    try:
        prompt, mode = _prompt(args, model, processor, config, images, audios)
    except ValueError as e:
        print(f"error: {e}.", file=sys.stderr)
        return 2

    from gmlx.stream import expert_streaming

    lm = getattr(model, "language_model", model)
    cli._apply_placement(args, lm)
    step, _defaulted = expert_streaming._resolve_prefill_step(
        lm, args.prefill_step_size)
    tokenizer = getattr(processor, "tokenizer", processor)
    why = cli.unfit_request(model, tokenizer, prompt, 0, args.max_tokens,
                            step)
    if why:
        print(f"error: {why}.", file=sys.stderr)
        return 1
    rep = args.repetition_penalty
    sampler = make_sampler(temp=args.temp, top_p=args.top_p,
                           top_k=args.top_k, min_p=args.min_p)
    processors = make_logits_processors(
        logit_bias=logit_bias or None,
        repetition_penalty=None if rep in (0.0, 1.0) else rep,
        repetition_context_size=args.repetition_context_size,
        presence_penalty=args.presence_penalty or None,
        frequency_penalty=args.frequency_penalty or None,
    )
    key = None if args.seed is None else mx.random.key(args.seed)
    if args.seed is not None:
        mx.random.seed(args.seed)
    print(f"[generate] max_tokens={cli.max_tokens_label(args)} "
          f"temp={args.temp} speech={'tts' if args.tts else mode or 'prompt'}"
          "\n")

    chunks = []
    summary = None
    for event in speak.stream_speech(
            model, processor, prompt, image=images or None,
            audio=audios or None, mode=mode, max_tokens=args.max_tokens,
            sampler=sampler, logits_processors=processors, key=key,
            prefill_step_size=step or speak.DEFAULT_PREFILL_STEP,
            max_kv_size=args.max_kv_size):
        if event.text:
            print(event.text, end="", flush=True)
        if event.audio is not None and event.audio.size:
            chunks.append(event.audio)
        summary = event
    assert summary is not None
    print()

    rate = capabilities["sample_rate"]
    audio = np.concatenate(chunks) if chunks else np.zeros(0, np.float32)
    seconds = audio.size / rate
    print("=" * 10)
    if not audio.size:
        print("The model gave no speech, so no file was written.")
    else:
        path = os.path.expanduser(args.speech_out)
        write_wav(path, audio, rate)
        print(f"Speech: {seconds:.2f} s saved to {path}")
    slots = summary.text_tokens + summary.frames
    print(f"Prompt: {summary.prompt_tokens} tokens, "
          f"{summary.prompt_tokens / max(summary.prompt_seconds, 1e-9):.3f} "
          "tokens-per-sec")
    print(f"Generation: {summary.text_tokens} text tokens and "
          f"{summary.frames} speech frames, "
          f"{slots / max(summary.generation_seconds, 1e-9):.3f} per sec")
    if seconds:
        print(f"Speech speed: {seconds / summary.generation_seconds:.1f} "
              "times real time, first audio "
              f"{summary.first_audio_seconds * 1000:.0f} ms after the prompt")
    print(f"Peak memory: {summary.peak_memory_gb:.3f} GB")
    if summary.finish_reason == "length":
        cli.warn_cap_hit(args, slots)
    return 0
