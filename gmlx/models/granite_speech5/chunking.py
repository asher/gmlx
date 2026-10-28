# SPDX-License-Identifier: Apache-2.0
# Ported from generative-computing/granite-switch
# src/granite_switch/vllm/audio/chunking.py at 43ccd291 (Apache-2.0; see
# licenses/granite-switch-LICENSE).
"""Long-audio windows for the speech companion: overlapping windows in,
one merged transcript out. The overlap is transcribed twice, and the merge
keeps one copy of the words at each seam."""

from __future__ import annotations

import re

import numpy as np

# Bounded seam search, so the merge stays linear in transcript length.
_MAX_SEAM_WORDS = 60
WINDOW_S = 120.0
OVERLAP_S = 5.0


def split_waveform(samples: np.ndarray, sr: int, window_s: float = WINDOW_S,
                   overlap_s: float = OVERLAP_S) -> list[np.ndarray]:
    """A 1-D waveform as overlapping windows that advance by ``window_s -
    overlap_s``. A clip no longer than one window is one segment."""
    if window_s <= 0:
        raise ValueError(f"window_s must be > 0, got {window_s}")
    if not (0 <= overlap_s < window_s):
        raise ValueError(f"overlap_s must satisfy 0 <= overlap_s < window_s; got "
                         f"overlap_s={overlap_s}, window_s={window_s}")
    window = max(1, round(window_s * sr))
    step = max(1, round((window_s - overlap_s) * sr))
    n = len(samples)
    if n <= window:
        return [samples]
    segments = []
    start = 0
    while start < n:
        segments.append(samples[start:start + window])
        if start + window >= n:
            break
        start += step
    return segments


def _norm_word(word: str) -> str:
    return re.sub(r"[^\w]+", "", word).lower()


def _seam_overlap(prev: list[str], nxt: list[str]) -> int:
    """The longest k with the last k words of ``prev`` equal to the first k
    of ``nxt``, ignoring case and punctuation; 0 when none match."""
    for k in range(min(len(prev), len(nxt), _MAX_SEAM_WORDS), 0, -1):
        if [_norm_word(w) for w in prev[-k:]] == [_norm_word(w) for w in nxt[:k]]:
            return k
    return 0


def merge_transcripts(transcripts: list[str]) -> str:
    """The window transcripts joined, with each seam's repeated words once."""
    merged: list[str] = []
    for text in transcripts:
        words = text.split()
        if not words:
            continue
        if not merged:
            merged = words
            continue
        merged.extend(words[_seam_overlap(merged, words):])
    return " ".join(merged)
