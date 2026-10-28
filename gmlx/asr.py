"""The speech cascade: a text model that reads audio through a speech
companion GGUF.

A model whose arch pairs with a companion (``arch_table.ASR_COMPANION_REPOS``,
Granite Switch today) renders each audio part of a message as its audio
marker token. The companion transcribes each clip, and :func:`splice`
replaces each marker in the prompt ids with that clip's transcript ids.
One loaded companion per file serves every model and the speech-to-text
route."""

from __future__ import annotations

import os
import re
import threading
from typing import Callable, Optional

MARKER = "<|audio|>"
# A clip with no speech still fills its marker, as IBM's cascade does.
EMPTY_TRANSCRIPT = " "
# The model's own special tokens, such as <|start_of_role|>. IBM's cascade
# refuses a transcript that holds one.
_SPECIAL = re.compile(r"<\|[^<>|\s]+\|>")
_TEXT_PARTS = ("text", "input_text", "output_text")
NO_COMPANION = ("this model reads audio through a speech companion GGUF, and "
                "none is paired: pass --asr PATH, or set `asr:` in the model's "
                "config entry")


def pairs_with_companion(model_path: str | None) -> bool:
    """True when the GGUF at ``model_path`` reads audio through a companion."""
    from gmlx.load import arch_table
    from gmlx.load.discovery import header_meta

    meta = header_meta(model_path) if model_path else None
    return bool(meta) and meta.get("arch") in arch_table.ASR_COMPANION_REPOS


def companion_path(model_path: str | None, explicit: str | None = None,
                   disabled: bool = False) -> str | None:
    """The companion to pair with the model at ``model_path``: ``explicit``
    (``--asr`` or the config's ``asr:``), else the one discovery finds beside
    the model; None when ``disabled`` or there is none."""
    if disabled or not model_path:
        return None
    if explicit:
        return os.path.abspath(os.path.expanduser(explicit))
    if not pairs_with_companion(model_path):
        return None
    from gmlx.load.discovery import find_asr_companion

    return find_asr_companion(model_path)


class Companion:
    """One companion GGUF, loaded on first use."""

    def __init__(self, path: str):
        self.path = path
        self._lock = threading.Lock()
        self._speech = None

    @property
    def loaded(self) -> bool:
        return self._speech is not None

    def speech(self):
        with self._lock:
            if self._speech is None:
                from gmlx.models.granite_speech5.model import load

                self._speech = load(self.path)
            return self._speech

    @property
    def sample_rate(self) -> int:
        return self.speech().sample_rate

    def transcribe(self, wave) -> str:
        speech = self.speech()
        with self._lock:
            return speech.transcribe(wave)

    def release(self) -> None:
        with self._lock:
            self._speech = None


_companions: dict[str, Companion] = {}
_companions_lock = threading.Lock()


def companion(path: str) -> Companion:
    """The shared :class:`Companion` of ``path``."""
    path = os.path.abspath(os.path.expanduser(path))
    with _companions_lock:
        got = _companions.get(path)
        if got is None:
            got = _companions[path] = Companion(path)
        return got


def is_audio_part(part) -> bool:
    """True for a content part the Granite template renders as the marker."""
    return isinstance(part, dict) and "audio" in str(part.get("type") or "")


def has_audio(content) -> bool:
    return isinstance(content, list) and any(is_audio_part(p) for p in content)


def marked_text(content: list) -> str:
    """A content list as the Granite template flattens it: the text parts
    joined by newlines, and the marker in place of each audio part. Other
    parts render nothing."""
    out = ""
    for part in content:
        if not isinstance(part, dict):
            continue
        if part.get("type") in _TEXT_PARTS:
            if out:
                out += "\n"
            out += str(part.get("text") or "")
        elif is_audio_part(part):
            out += MARKER
    return out


def check_transcript(text: str) -> None:
    """Raise ValueError for a transcript that holds a special token."""
    hit = _SPECIAL.search(text)
    if hit:
        raise ValueError(f"a transcript holds the special token {hit.group(0)}, "
                         "which this model reads as a control token")


def marker_id(tokenizer) -> Optional[int]:
    """The audio marker's token id, or None when the vocab has none."""
    # A TokenizerWrapper holds the HF tokenizer as _tokenizer; an HF fast
    # tokenizer holds the tokenizers.Tokenizer there.
    for tok in (tokenizer, getattr(tokenizer, "_tokenizer", None)):
        convert = getattr(tok, "convert_tokens_to_ids", None)
        if convert is not None:
            tid = convert(MARKER)
            unk = getattr(tok, "unk_token_id", None)
            return None if tid is None or tid == unk else int(tid)
    return None


def splice(ids: list[int], marker: int, transcripts: list[str],
           encode: Callable[[str], list[int]]) -> list[int]:
    """``ids`` with the n-th audio marker replaced by the n-th transcript's
    ids, each tokenized alone. Raises ValueError when the marker count is
    not the clip count, or a transcript holds a special token."""
    for text in transcripts:
        check_transcript(text)
    n = sum(1 for t in ids if t == marker)
    if n != len(transcripts):
        raise ValueError(f"the prompt has {n} audio marker(s) for "
                         f"{len(transcripts)} audio clip(s)")
    out: list[int] = []
    clips = iter(transcripts)
    for t in ids:
        if t == marker:
            out += list(encode(next(clips) or EMPTY_TRANSCRIPT))
        else:
            out.append(t)
    return out
