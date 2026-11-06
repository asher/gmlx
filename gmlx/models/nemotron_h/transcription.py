# SPDX-License-Identifier: Apache-2.0
# The language map, output budget and silence test follow NVIDIA
# NeMo-Speech.cpp 536b17a4 src/speechlm/salm_policy.{h,cpp} (Apache-2.0).
# cleanup_generation_loops is the model card's guardrail, adapted by NVIDIA
# from Qwen3-ASR (Copyright 2026 The Alibaba Qwen team, Apache-2.0).
"""Transcription requests to Nemotron 3.5 Transcribe: the instructions
from the model card, the language hint, the output token budget, the
silence test and the text cleanup applied to a finished transcript."""

from __future__ import annotations

import math
import re

import numpy as np

# The card's prompts (README "Prompting").
TRANSCRIBE = ("Audio transcription. Language: as spoken. Wording: verbatim, "
              "including fillers and repetitions. Formatting: punctuation and "
              "capitalization. Output: transcript only.")
SPEAKERS = ("Transcribe this audio with punctuation and capitalization. "
            "Include all spoken words and verbal fillers. Write numbers in "
            "their standard written form. Put speaker tags <spk:0> <spk:1> "
            "<spk:2> <spk:3> on the transcription.")

# The trained limits: one hour of audio in a 65,536-token context. A
# request holds at most six clips.
CONTEXT_TOKENS = 65_536
MAX_AUDIO_SECONDS = 3600.0
MAX_OUTPUT_TOKENS = 21_664
MAX_CLIPS = 6

# BCP-47 primary subtag to the name the language hint uses.
LANGUAGE_NAMES = {
    "en": "English", "ar": "Arabic", "zh": "Chinese", "fr": "French",
    "de": "German", "hi": "Hindi", "it": "Italian", "ja": "Japanese",
    "es": "Spanish", "ko": "Korean", "pt": "Portuguese", "ru": "Russian",
    "th": "Thai", "hy": "Armenian", "nn": "Norwegian", "nb": "Norwegian",
    "no": "Norwegian", "bg": "Bulgarian", "hr": "Croatian", "cs": "Czech",
    "da": "Danish", "nl": "Dutch", "et": "Estonian", "fi": "Finnish",
    "el": "Greek", "he": "Hebrew", "hu": "Hungarian", "id": "Indonesian",
    "lv": "Latvian", "lt": "Lithuanian", "mt": "Maltese", "ta": "Tamil",
    "bn": "Bengali", "pl": "Polish", "ro": "Romanian", "sk": "Slovak",
    "sl": "Slovenian", "sv": "Swedish", "tr": "Turkish", "tl": "Filipino",
    "fil": "Filipino", "uk": "Ukrainian", "vi": "Vietnamese",
}

SILENCE_PEAK = 1e-3
SILENCE_RMS = 3e-5

_CHAR_RUN = re.compile(r"(.)\1{20,}", re.DOTALL)
_BLOCK_RUN = re.compile(r"(.{1,4096}?)\1{19,}", re.DOTALL)
_STYLE_TAGS = re.compile(r"<(?:ev|sty):[^<>]*>")
_SPEAKER_TAGS = re.compile(r"<spk:\d+>")
_SPACES = re.compile(r"[ \t]{2,}")


def language_name(code: str | None) -> str | None:
    """The language a code such as ``en``, ``pt-BR`` or ``zh_Hant`` names,
    or None when the model's hint list has no such language."""
    if not code or not code.strip():
        return None
    parts = code.strip().lower().replace("_", "-").split("-")
    if len(parts) > 2 or not parts[0]:
        return None
    return LANGUAGE_NAMES.get(parts[0])


def transcription_instruction(language: str | None = None, *,
                              speakers: bool = False) -> str:
    """The user turn's text for a transcription: the card's verbatim
    prompt with a hint when ``language`` is known, or its speaker prompt."""
    if speakers:
        return SPEAKERS
    name = language_name(language)
    return f"{TRANSCRIBE} The speech is in {name}." if name else TRANSCRIBE


def translation_instruction(language: str = "English") -> str:
    return f"Please translate the given speech to {language}."


def check_clips(samples: list[int], rate: int = 16000) -> None:
    """Refuse more clips, or more audio, than the model is trained on:
    ``samples`` holds each clip's sample count."""
    if len(samples) > MAX_CLIPS:
        raise ValueError(
            f"{len(samples)} audio clips in one request; the limit is "
            f"{MAX_CLIPS}. Start a new conversation for more audio")
    seconds = sum(samples) / rate
    if seconds > MAX_AUDIO_SECONDS + 1e-6:
        raise ValueError(f"{seconds:.1f} s of audio in one request; the "
                         f"limit is {MAX_AUDIO_SECONDS:.0f} s")


def output_budget(seconds: float, prompt_tokens: int,
                  context: int = CONTEXT_TOKENS) -> int:
    """Tokens a transcription of ``seconds`` of audio may generate:
    ``max(24, ceil(6 s) + 24)``, at most :data:`MAX_OUTPUT_TOKENS` and the
    room the prompt leaves in ``context``."""
    want = max(24, math.ceil(seconds * 6.0 - 1e-9) + 24)
    return max(0, min(want, MAX_OUTPUT_TOKENS, context - prompt_tokens))


def near_silent(audio: np.ndarray) -> bool:
    """Near-digital silence: no samples, or a peak under 1e-3 (-60 dBFS)
    and an RMS under 3e-5 (-90 dBFS)."""
    x = np.asarray(audio, np.float64).reshape(-1)
    if x.size == 0:
        return True
    if np.abs(x).max() >= SILENCE_PEAK:
        return False
    return float(np.sqrt(np.mean(x * x))) < SILENCE_RMS


def _gram_repeats(cp: np.ndarray, k: int, times: int) -> bool:
    """Whether some ``k``-gram of ``cp`` occurs ``times`` times or more
    (by a 64-bit polynomial hash, so a collision can only say yes)."""
    m = cp.size - k + 1
    if m < times:
        return False
    h = np.zeros(m, np.uint64)
    base = np.uint64(1_000_003)
    with np.errstate(over="ignore"):
        for j in range(k):
            h = h * base + cp[j:j + m].astype(np.uint64)
    _, counts = np.unique(h, return_counts=True)
    return bool(counts.max() >= times)


def _block_run_spans(cp: np.ndarray) -> list[tuple[int, int]]:
    """``(first, last)`` ranges of the positions where a block of 1 to
    4,096 characters starts that repeats 20 times or more: a run of
    ``19 * L`` positions ``j`` with ``cp[j] == cp[j + L]``."""
    spans: list[tuple[int, int]] = []
    top = min(4096, cp.size // 20)

    def runs(lag: int) -> None:
        need = 19 * lag
        eq = cp[:-lag] == cp[lag:]
        if np.count_nonzero(eq) < need:
            return
        breaks = np.flatnonzero(~eq)
        starts = np.concatenate(([0], breaks + 1))
        ends = np.concatenate((breaks, [eq.size]))
        for k in np.flatnonzero(ends - starts >= need):
            spans.append((int(starts[k]), int(ends[k]) - need))

    for lag in range(1, min(top, 31) + 1):
        runs(lag)
    # A block of 32 or more characters repeated 20 times repeats each of
    # its 32-grams 20 times.
    if top >= 32 and _gram_repeats(cp, 32, 20):
        for lag in range(32, top + 1):
            runs(lag)
    return spans


def _collapse_blocks(text: str) -> str:
    """``_BLOCK_RUN.sub``, with the regex tried only where a match can
    start: it costs about 90 us per character otherwise."""
    cp = np.frombuffer(text.encode("utf-32-le"), np.uint32)
    spans = _block_run_spans(cp)
    out: list[str] = []
    pos = 0
    while spans:
        starts = [max(a, pos) for a, b in spans if b >= pos]
        if not starts:
            break
        at = min(starts)
        m = _BLOCK_RUN.match(text, at)
        if m is None:  # pragma: no cover - a span always starts a match
            break
        out += [text[pos:at], m.group(1)]
        pos = m.end()
        spans = [(a, b) for a, b in spans if b >= pos]
    out.append(text[pos:])
    return "".join(out)


def cleanup_generation_loops(text: str) -> str:
    """Collapse a character repeated 21 times or more, then a block of up
    to 4,096 characters repeated 20 times or more, to one copy."""
    return _collapse_blocks(_CHAR_RUN.sub(lambda m: m.group(1), text))


def strip_tags(text: str, *, speakers: bool = False) -> str:
    """``text`` without the model's event and style tags, and without its
    speaker tags when ``speakers``; runs of spaces collapse to one."""
    text = _STYLE_TAGS.sub("", text)
    if speakers:
        text = _SPEAKER_TAGS.sub("", text)
    return _SPACES.sub(" ", text).strip()
