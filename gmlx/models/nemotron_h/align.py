"""Word timestamps and speakers for Nemotron 3.5 Transcribe.

A port of NeMo Speech's ``MultiSpeakerSOTWordTimestampAligner`` with its
banded backend (``nemo/collections/speechlm2/parts/ctc_timestamp_utils.py``
at 979cc6c6): transcript words and speaker streams, their CTC targets, the
token-distribution speaker mapping, the anchor-banded max-sum CTC search,
word rows, calibrated boundary interpolation, diarization refinement and
the 10 ms diarization segments. Back pointers are packed at 2 bits per band
cell (NeMo keeps a byte). Pure NumPy over host arrays; the CTC head that
gives the log probabilities is :meth:`SpeechEncoder.ctc_columns`.

The checkpoint's alignment parameters and calibration are in the model's
``config.json`` (``ctc_timestamps.alignment_config``), not in the GGUF, so
:data:`PARAMETERS` pins them.
"""

from __future__ import annotations

import base64
import math
import re
import unicodedata
from bisect import bisect_left, bisect_right
from collections import defaultdict
from dataclasses import dataclass, field
from fnmatch import fnmatchcase
from functools import cached_property
from itertools import groupby
from typing import Any

import numpy as np

# HF nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B @ c1766764, config.json
# ctc_timestamps.alignment_config (schema 1, basis 1, upsampling 2).
PARAMETERS: dict[str, Any] = {
    "speaker_logprob_weight": 0.6057,
    "speaker_activity_threshold": 0.1881,
    "support_activity_threshold": 0.0225,
    "vad_threshold": 0.4641,
    "band_ratio": 0.0484,
    "maximum_token_len": 1.0364,
    "refinement_end_blend": 0.7382,
    "boundary_outside_word_duration_seconds": 0.1747,
    "onset_snap_seconds": 0.08,
    "boundary_max_correction_seconds": 0.04,
    "boundary_min_onset_spacing_seconds": 0.01,
    "boundary_min_word_duration_seconds": 0.005,
    "band_min_states": 50,
    "max_word_frames": 19,
    "band_full_below_frames": 0,
    "boundary_profile_radius_frames": 4,
    "boundary_profile_points": 33,
    "boundary_logprob_floor": -20.0,
    "boundary_logprob_ceiling": 0.0,
    "boundary_grid_seconds": 0.01,
    "boundary_interpolation": {
        "coefficients": [
            [-0.0002166376730768876, 0.0011034308334353859, -0.0023364347654229443,
             0.0004432468739614406, -0.0006112624259137253, 0.0009186590141194121,
             0.0008936531449761716, -0.00032564959098740524, 0.0009534467373316291,
             -0.0006841331962980549, 0.0002364067745016569, -7.613400442991535e-05,
             -0.00037485682802432463, 0.0010040103810179503, -0.0003161025648419653,
             0.0006077597635300055, -0.0010064009281295807, -0.0009313666476320707,
             -0.0001362872039203774, 0.0002812433651844184, 6.10944274587241e-05,
             0.0004609156208085293, 0.00010091507507042862, -0.0001028959125451627],
            [0.003366517585792393, -0.0038862878823709107, -0.004189308829413212,
             0.0025464040055689016, -0.0010541458667035343, 0.0010038389587020283,
             0.0009109164073724175, -0.0015966388571699583, 0.0007440020382137693,
             -0.0002527302196180498, 0.0002965690615668117, 0.0014428291779151339,
             -0.0007785076678433898, 0.0004340954622391335, 1.0325823582757346e-05,
             -0.00044058912087083815, -0.00011999138369963432, -0.0013171473151848026,
             0.00022320090572465454, -0.00032724805493375766, -0.0002112379836947067,
             0.000306075565007786, 3.6466920221606976e-05, 0.00036613335041990705],
        ],
        "offset": [0.02618952076580825, 0.12891473673580162],
    },
}
EPSILON = 1.0e-6
DIARIZATION_FRAME_SECONDS = 0.01
SPEAKER_TAG = re.compile(r"<spk:(\d+)>", flags=re.IGNORECASE)
IGNORED_TAGS = ("<spk:*>", "<sty:stt:*>", "<sty:end:*>")
ALIGNED_TAGS = ("<ev:*>",)
NO_PATH = "no_legal_path_in_anchor_band"
UNMATCHED = "unmatched_speaker_tag"


class AlignmentError(ValueError):
    """A transcript the aligner cannot align (NeMo's ``ValueError``)."""


@dataclass(frozen=True)
class Params:
    speaker_logprob_weight: float
    speaker_activity_threshold: float
    support_activity_threshold: float
    vad_threshold: float
    band_ratio: float
    maximum_token_len: float
    refinement_end_blend: float
    boundary_outside_word_duration_seconds: float
    onset_snap_seconds: float | None
    boundary_max_correction_seconds: float
    boundary_min_onset_spacing_seconds: float
    boundary_min_word_duration_seconds: float
    band_min_states: int
    max_word_frames: int
    band_full_below_frames: int
    boundary_profile_radius_frames: int
    boundary_profile_points: int
    boundary_logprob_floor: float
    boundary_logprob_ceiling: float
    boundary_grid_seconds: float
    boundary_interpolation: dict | None = field(default=None, hash=False)

    @cached_property
    def calibration(self) -> dict | None:
        if self.boundary_interpolation is None:
            return None
        return _expand_boundary_calibration(self.boundary_interpolation,
                                            self.boundary_profile_points)


CHECKPOINT = Params(**PARAMETERS)


# -- transcript ---------------------------------------------------------------


def _is_aligned_tag(token: str) -> bool:
    """Whether a special token gets a timestamp: excluded patterns win."""
    token = token.lower()
    return (not any(fnmatchcase(token, p) for p in IGNORED_TAGS)
            and any(fnmatchcase(token, p) for p in ALIGNED_TAGS))


def clean_transcript(transcript: str, special_tokens) -> str:
    """NeMo's ``clean_sot_transcript``: every special token of the decoding
    tokenizer except speaker tags and aligned tags becomes a space, then
    words are joined by single spaces."""
    for token in special_tokens:
        if SPEAKER_TAG.fullmatch(token) is None and not _is_aligned_tag(token):
            transcript = transcript.replace(token, " ")
    return " ".join(transcript.split())


def parse_words(transcript: str) -> list[dict]:
    """NeMo's ``parse_sot_words``. Each ``<spk:N>`` starts a speaker turn;
    words before the first tag have speaker None, and a transcript without
    tags is speaker 0. Em dashes separate words; tags other than aligned
    ones never become targets."""
    transcript = re.sub(r"(<[^<>\s]+>)", r" \1 ", transcript.replace("\u2014", " "))
    words: list[dict] = []
    parts = SPEAKER_TAG.split(transcript)
    tag: int | None = 0 if len(parts) == 1 else None
    turn: int | None = None
    for i, part in enumerate(parts):
        if i % 2:
            tag, turn = int(part), i // 2
            continue
        for word in part.split():
            if word.startswith("<") and word.endswith(">") and not _is_aligned_tag(word):
                continue
            words.append({"word": word, "speaker_tag": tag, "turn_index": turn,
                          "word_index": len(words)})
    return words


def _lexical_characters(text: str) -> list[str]:
    return [c for c in unicodedata.normalize("NFC", text)
            if unicodedata.category(c)[0] in "LMN"]


_SCRIPTS = (("ko", ("HANGUL",)), ("ja", ("HIRAGANA", "KATAKANA")),
            ("zh", ("CJK UNIFIED IDEOGRAPH", "CJK COMPATIBILITY IDEOGRAPH")),
            ("th", ("THAI",)))
_TAG = re.compile(r"<[^<>]+>")


def timestamp_units(words: list[dict], language: str | None = None) -> list[dict]:
    """NeMo's ``_timestamp_units`` with script detection: Chinese and
    Japanese words become lexical characters; other languages keep their
    words. Thai needs a word segmenter gmlx does not ship, so it is
    refused, as transformers does."""
    if language is None:
        names = [unicodedata.name(c, "") for c in
                 {c for w in words if not _TAG.fullmatch(w["word"]) for c in w["word"]}]
        language = next((lang for lang, prefixes in _SCRIPTS
                         if any(n.startswith(prefixes) for n in names)), None)
    if language == "th":
        raise AlignmentError("word timestamps do not support Thai")
    if language not in ("zh", "ja"):
        return words
    out: list[dict] = []
    for (_, _, special), group in groupby(
            words, key=lambda w: (w.get("speaker_tag"), w.get("turn_index"),
                                  bool(_TAG.fullmatch(w["word"])))):
        for word in group:
            if special:
                out.append({**word, "word_index": len(out)})
                continue
            for unit in _lexical_characters(word["word"]):
                out.append({**word, "word": unit, "word_index": len(out),
                            "alignment_unit": "character"})
    return out


class Tokenizer:
    """The CTC head's SentencePiece model (``ctc.tokenizer.spm_model``, a
    base64 model proto). NeMo's ``SentencePieceTokenizer`` with
    ``legacy=False`` is the plain model's ``encode``."""

    def __init__(self, spm_model: str | bytes):
        import sentencepiece as spm

        proto = (base64.b64decode(spm_model) if isinstance(spm_model, str)
                 else bytes(spm_model))
        # sentencepiece's methods are generated: no stubs to check.
        self.sp: Any = spm.SentencePieceProcessor(model_proto=proto)

    def text_to_ids(self, text: str) -> list[int]:
        return [int(i) for i in self.sp.encode(text)]

    def character_ids(self, character: str, blank_id: int) -> list[int]:
        """NeMo's ``_character_token_ids``: the character's own piece, else
        its pieces without bare word-boundary pieces."""
        sp = self.sp
        index = sp.piece_to_id(character)
        ids = ([index] if index != sp.unk_id() else
               [t for t in sp.encode(character) if sp.id_to_piece(t) != "\u2581"])
        if (not ids or sp.unk_id() in ids or min(ids) < 0 or max(ids) >= blank_id
                or "".join(_lexical_characters(sp.decode(ids))) != character):
            raise AlignmentError(f"unrepresentable character: {character!r}")
        return ids


def _stream_ids(tok: Tokenizer, words: list[dict], blank_id: int) -> list[list[int]]:
    """NeMo's ``_tokenize_word_stream``: token ids per word of one speaker."""
    if any(w.get("alignment_unit") == "character" for w in words):
        return [tok.character_ids(w["word"], blank_id)
                if w.get("alignment_unit") == "character"
                else _stream_ids(tok, [w], blank_id)[0] for w in words]
    ids = tok.text_to_ids(" ".join(w["word"] for w in words))
    if not ids or min(ids) < 0 or max(ids) >= blank_id:
        raise AlignmentError("tokenizer produced ids outside the CTC vocabulary")
    of: dict[str, list[int]] = {}
    for w in words:
        if w["word"] not in of:
            of[w["word"]] = tok.text_to_ids(w["word"])
    by_word = [list(of[w["word"]]) for w in words]
    if any(not x for x in by_word):
        raise AlignmentError("empty word tokenization")
    if [t for x in by_word for t in x] != ids:
        raise AlignmentError("per-word and full-stream tokenization disagree")
    return by_word


def _group(words: list[dict]) -> dict[int | None, list[dict]]:
    out: dict[int | None, list[dict]] = {}
    for w in words:
        out.setdefault(w["speaker_tag"], []).append(w)
    return out


@dataclass
class Prepared:
    """A transcript as CTC targets: its words, its speaker streams with
    labels renumbered to ``columns`` (blank first), and those columns."""

    words: list[dict]
    streams: list[dict]
    columns: list[int]


def prepare(transcript: str, tokenizer: Tokenizer, blank_id: int,
            ctc_frames: int, language: str | None = None) -> Prepared:
    """NeMo's ``_prepare`` for one cleaned transcript: targets before any
    log probability is read, so a transcript the aligner refuses costs no
    CTC head run."""
    units = timestamp_units(parse_words(transcript), language)
    tokenized: dict[int, dict] = {}
    for stream in _group(units).values():
        for w, ids in zip(stream, _stream_ids(tokenizer, stream, blank_id)):
            tokenized[w["word_index"]] = {**w, "token_ids": ids}
    words = [tokenized[w["word_index"]] for w in units]
    streams = []
    for tag, stream_words in _group(words).items():
        labels: list[int] = [blank_id]
        state_to_word: list[int | None] = [None]
        for i, w in enumerate(stream_words):
            for t in w["token_ids"]:
                labels.extend((t, blank_id))
                state_to_word.extend((i, None))
        tokens = labels[1::2]
        need = len(tokens) + sum(a == b for a, b in zip(tokens, tokens[1:]))
        if need > ctc_frames:
            raise AlignmentError(
                f"speaker {tag!r} transcript is too long for the CTC timeline")
        streams.append({"speaker_tag": tag, "words": stream_words,
                        "labels": labels, "state_to_word": state_to_word})
    columns = [blank_id] + sorted({x for s in streams for x in s["labels"]} - {blank_id})
    column_of = {label: c for c, label in enumerate(columns)}
    for s in streams:
        s["labels"] = [column_of[x] for x in s["labels"]]
    return Prepared(words=words, streams=streams, columns=columns)


# -- speaker mapping ----------------------------------------------------------


def resample_speaker_probs(probs: np.ndarray, frames: int) -> np.ndarray:
    """Linear resampling of ``(T, S)`` onto ``frames`` rows, ends aligned."""
    if probs.shape[0] == frames:
        return probs
    if probs.shape[0] == 1:
        return np.broadcast_to(probs, (frames, probs.shape[1])).copy()
    pos = np.linspace(0, probs.shape[0] - 1, frames, dtype=np.float32)
    lower = np.floor(pos).astype(np.int64)
    upper = np.ceil(pos).astype(np.int64)
    frac = (pos - lower)[:, None]
    return probs[lower] * (1 - frac) + probs[upper] * frac


def speaker_mapping(ctc: np.ndarray, streams: list[dict], speaker: np.ndarray,
                    threshold: float, greedy: np.ndarray) -> dict:
    """NeMo's ``_one_pass_speaker_mapping``: speaker tag to diarization
    column, by IDF-weighted cosine similarity of token counts and a linear
    assignment; unassigned tags map to None."""
    if not streams:
        return {}
    if speaker.shape[1] == 0:
        return {s["speaker_tag"]: None for s in streams}
    from scipy.optimize import linear_sum_assignment

    peaks = (greedy != 0) & np.r_[True, greedy[1:] != greedy[:-1]]
    observed = np.stack([
        np.bincount(greedy[peaks & (speaker[:, c] >= threshold)], minlength=ctc.shape[1])
        for c in range(speaker.shape[1])])
    expected = np.stack([np.bincount(np.asarray(s["labels"])[1::2], minlength=ctc.shape[1])
                         for s in streams])
    idf = np.log1p(len(observed) / (1 + (observed > 0).sum(axis=0)))
    a = np.sqrt(expected * idf)
    b = np.sqrt(observed * idf)
    sim = a @ b.T / np.maximum(1e-8, np.linalg.norm(a, axis=1)[:, None]
                               * np.linalg.norm(b, axis=1)[None, :])
    sim -= 0.1 * np.abs(np.log((expected.sum(axis=1)[:, None] + 1)
                               / (observed.sum(axis=1)[None, :] + 1)))
    rows, cols = linear_sum_assignment(-sim)
    mapping = {s["speaker_tag"]: None for s in streams}
    mapping.update({streams[r]["speaker_tag"]: int(c) for r, c in zip(rows, cols)})
    return mapping


# -- banded search ------------------------------------------------------------


def _trigram_positions(tokens) -> dict:
    table = defaultdict(list)
    for i in range(len(tokens) - 2):
        table[tuple(int(t) for t in tokens[i:i + 3])].append(i)
    return table


def anchor_band_starts(greedy: np.ndarray, target: np.ndarray, states: int,
                       half: int, width: int) -> np.ndarray:
    """NeMo's ``_ctc_anchor_band_starts``: band starts interpolated through
    the longest monotonic chain of trigrams unique in both the greedy
    onsets and the target."""
    frames = len(greedy)
    if not frames:
        return np.empty(0, dtype=np.int64)
    keep = (greedy != 0) & np.r_[True, greedy[1:] != greedy[:-1]]
    observed = greedy[keep]
    times = np.flatnonzero(keep)
    left, right = _trigram_positions(observed), _trigram_positions(target)
    pairs = sorted((int(times[v[0] + 1]), 2 * (right[k][0] + 1) + 1)
                   for k, v in left.items()
                   if len(v) == 1 and len(right.get(k, [])) == 1)
    tails: list[int] = []
    tail_ids: list[int] = []
    parent: list[int] = []
    for i, (_, state) in enumerate(pairs):
        j = bisect_left(tails, state)
        parent.append(tail_ids[j - 1] if j else -1)
        if j == len(tails):
            tails.append(state)
            tail_ids.append(i)
        else:
            tails[j] = state
            tail_ids[j] = i
    chosen = []
    k = tail_ids[-1] if tail_ids else -1
    while k >= 0:
        chosen.append(pairs[k])
        k = parent[k]
    chosen = chosen[::-1]
    points = ([(0, 0)] + [(t, s) for t, s in chosen if 0 < t < frames - 1]
              + [(frames - 1, states - 1)])
    center = np.interp(np.arange(frames), [p[0] for p in points], [p[1] for p in points])
    starts = np.clip(np.rint(center).astype(np.int64) - half, 0, states - width)
    steps = 2 * np.arange(frames)
    starts = np.minimum.accumulate(starts - steps) + steps
    return np.maximum(starts, np.maximum(0, states - width - 2 * (frames - 1 - np.arange(frames))))


def banded_forward(logp: np.ndarray, labels: np.ndarray, positions: np.ndarray,
                   prior: np.ndarray | None, starts: np.ndarray, width: int):
    """NeMo's ``_ctc_banded_forward_numpy``: max-sum DP over the band, ties
    preferring stay, step, then skip. Returns the last frame's scores and
    the moves, packed four 2-bit cells to a byte, ``(frames, ceil(width/4))``."""
    frames = len(positions)
    packed_width = -(-width // 4)
    previous = np.full(width, -np.inf, np.float32)
    first = min(2, width)
    previous[:first] = logp[positions[0], labels[:first]]
    if prior is not None:
        previous[:first] += np.where(labels[:first] != 0, prior[positions[0]], 0)
    pointers = np.zeros((frames, packed_width), np.uint8)
    move = np.zeros(packed_width * 4, np.uint8)
    padded = np.full(width + 4, -np.inf, np.float32)
    can_skip = (labels != 0) & (labels != np.r_[labels[:2], labels[:-2]])
    for t in range(1, frames):
        start = int(starts[t])
        delta = start - int(starts[t - 1])
        padded[2:width + 2] = previous
        best = padded[2 + delta:2 + delta + width].copy()
        one = padded[1 + delta:1 + delta + width]
        two = padded[delta:delta + width]
        move[:] = 0
        take = one > best
        np.copyto(best, one, where=take)
        move[:width][take] = 1
        take = (two > best) & can_skip[start:start + width]
        np.copyto(best, two, where=take)
        move[:width][take] = 2
        pointers[t] = move[0::4] | (move[1::4] << 2) | (move[2::4] << 4) | (move[3::4] << 6)
        target = labels[start:start + width]
        previous = best + logp[positions[t], target]
        if prior is not None:
            previous += np.where(target != 0, prior[positions[t]], 0)
    return previous, pointers


def _move(pointers: np.ndarray, t: int, j: int) -> int:
    return (int(pointers[t, j >> 2]) >> ((j & 3) * 2)) & 3


def banded_backtrace(previous: np.ndarray, pointers: np.ndarray, starts: np.ndarray,
                     states: int):
    """NeMo's ``_ctc_banded_backtrace``: the best path ending in either of
    the two final states, or all ``-1`` and ``-inf`` when none is finite."""
    frames = pointers.shape[0]
    width = previous.shape[0]
    path = np.full(frames, -1, np.int64)
    ends = np.array([states - 1, states - 2])
    local = ends - starts[-1]
    valid = (local >= 0) & (local < width)
    ends, local = ends[valid], local[valid]
    best = previous[local].argmax()
    score = float(previous[local[best]])
    if np.isfinite(score):
        state = int(ends[best])
        for t in range(frames - 1, -1, -1):
            path[t] = state
            if t:
                state -= _move(pointers, t, state - int(starts[t]))
    return path, score


def band_half_width(frames: int, states: int, p: Params) -> int:
    if frames < p.band_full_below_frames:
        return states
    return min(states, math.ceil(p.band_ratio * states + p.band_min_states))


def banded_search(logp: np.ndarray, labels, positions: np.ndarray,
                  prior: np.ndarray | None, greedy: np.ndarray | None = None,
                  p: Params = CHECKPOINT):
    """NeMo's ``_banded_ctc_search``: ``(state per searched frame, score,
    band starts)``; the path is all ``-1`` and the score ``-inf`` when no
    legal path fits the band."""
    labels = np.asarray(labels, np.int64)
    frames, states = len(positions), len(labels)
    half = band_half_width(frames, states, p)
    width = min(states, 2 * half + 1)
    starts = np.clip(np.rint(np.linspace(0, states - 1, frames)).astype(np.int64) - half,
                     0, states - width)
    target = labels[1::2]
    if frames < len(target) + np.count_nonzero(target[1:] == target[:-1]) or not frames:
        return np.full(frames, -1, np.int64), -np.inf, starts
    guide = logp[positions].argmax(axis=1) if greedy is None else greedy[positions]
    starts = anchor_band_starts(guide, target, states, half, width)
    previous, pointers = banded_forward(logp, labels, positions, prior, starts, width)
    path, score = banded_backtrace(previous, pointers, starts, states)
    return path, score, starts


# -- word rows and refinement -------------------------------------------------


def word_rows(words: list[dict], state_to_word, path: np.ndarray, ctc_step: float,
              time_offset: float, labels, maximum_token_len: float) -> list[dict]:
    """NeMo's ``_word_rows_from_path``: a word spans the frames whose state
    is one of its tokens, capped at ``maximum_token_len`` seconds."""
    by_word: list[list[int]] = [[] for _ in words]
    for frame, state in enumerate(path.tolist()):
        if state < 0:
            continue
        w = state_to_word[state]
        if w is not None:
            by_word[w].append(frame)
    rows = []
    for word, frames in zip(words, by_word):
        if not frames:
            raise AlignmentError(f"CTC path did not visit word {word['word']!r}")
        start = time_offset + frames[0] * ctc_step
        end = min(time_offset + (frames[-1] + 1) * ctc_step, start + maximum_token_len)
        rows.append({**word, "start": start, "end": end,
                     "_boundary_labels": [labels[path[frames[0]]], labels[path[frames[-1]]]]})
    return rows


def diarization_segments(labels: np.ndarray, frame_seconds: float, time_offset: float,
                         duration: float | None) -> list[dict]:
    """NeMo's ``diarization_segments``: each active run of boolean
    ``(speakers, frames)`` as a segment, ends clipped to the duration."""
    segments = []
    for speaker, activity in enumerate(labels):
        padded = np.concatenate([[0], activity.astype(np.int8), [0]])
        edges = np.diff(padded)
        for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            start = time_offset + int(a) * frame_seconds
            end = time_offset + int(b) * frame_seconds
            if duration is not None:
                end = min(end, time_offset + float(duration))
            if end > start:
                segments.append({"speaker": speaker, "start": start, "end": end})
    return sorted(segments, key=lambda s: (s["start"], s["end"], s["speaker"]))


def _public_word_timestamps(rows: list[dict]) -> dict:
    grouped: dict = {}
    for w in rows:
        tag = w["speaker_tag"]
        grouped.setdefault(tag, []).append({
            "word": w["word"], "speaker": tag, "start": w["start"], "end": w["end"],
            **({"_outside_diarization": True} if w.get("_outside_diarization") else {})})
    return grouped


def _expand_boundary_calibration(calibration: dict, points: int) -> dict:
    """NeMo's ``_expand_boundary_calibration``: DCT coefficients to full
    token/blank profile weights."""
    if "coefficients" not in calibration:
        return calibration
    coefficients = np.asarray(calibration["coefficients"], dtype=np.float64)
    if (coefficients.ndim != 2 or coefficients.shape[0] != 2
            or not 1 <= coefficients.shape[1] <= 2 * points
            or not np.isfinite(coefficients).all()):
        raise ValueError("boundary_interpolation coefficients must have shape (2, K)")
    positions = np.arange(points)[:, None]
    frequency = np.arange(points)[None, :]
    temporal = np.cos(np.pi * (positions + 0.5) * frequency / points) * np.sqrt(2 / points)
    temporal[:, 0] /= np.sqrt(2)
    basis = np.empty((2 * points, 2 * points))
    basis[:points, 0::2] = basis[:points, 1::2] = temporal / np.sqrt(2)
    basis[points:, 0::2] = temporal / np.sqrt(2)
    basis[points:, 1::2] = -temporal / np.sqrt(2)
    return {"weights": coefficients @ basis[:, :coefficients.shape[1]].T,
            "offset": calibration.get("offset")}


def _boundary_profiles(logp: np.ndarray, boundaries: np.ndarray, labels: np.ndarray,
                       step: float, offset: float, p: Params) -> np.ndarray:
    """NeMo's ``_boundary_profiles``: token and blank log probabilities
    around each word edge, linearly interpolated to the profile points."""
    radius = p.boundary_profile_radius_frames
    points = p.boundary_profile_points
    centers = (np.asarray(boundaries).T - offset) / step
    base = np.floor(centers).astype(np.int64)
    coarse = np.arange(-radius, radius + 2)
    indices = np.clip(base[..., None] + coarse, 0, len(logp) - 1)
    columns = np.stack([np.asarray(labels).T, np.zeros_like(np.asarray(labels).T)], axis=-1)
    samples = logp[indices[..., None], columns[..., None, :]].swapaxes(-1, -2)
    samples = np.clip(samples, p.boundary_logprob_floor, p.boundary_logprob_ceiling)
    fine = np.linspace(0, 2 * radius, points) + (centers - base)[..., None]
    matrix = np.maximum(1.0 - np.abs(np.arange(len(coarse))[:, None] - fine[..., None, :]), 0.0)
    return (samples @ matrix).reshape(2, len(boundaries), 2 * points)


def interpolate_word_boundaries(rows: list[dict], logp: np.ndarray, step: float,
                                offset: float, duration: float | None,
                                diarization: list[dict], mapping: dict,
                                p: Params) -> None:
    """NeMo's ``_interpolate_word_boundaries``: calibrated corrections of the
    raw word edges, before diarization refinement; in place."""
    calibration = p.calibration
    if not rows or calibration is None:
        return
    segments = defaultdict(list)
    for s in diarization:
        segments[s["speaker"]].append((s["start"], s["end"]))
    intervals = {}
    for speaker, values in segments.items():
        values.sort()
        intervals[speaker] = (np.array([s for s, _ in values]), np.array([e for _, e in values]))
    upper = offset + (duration if duration is not None else len(logp) * step)
    for word in rows:
        starts, ends = intervals.get(mapping.get(word["speaker_tag"]), ([], []))
        last = np.searchsorted(starts, word["end"], side="left") - 1
        outside = last < 0 or ends[last] <= word["start"]
        word["_outside_diarization"] = bool(outside)
        if outside:
            word["end"] = min(word["start"] + p.boundary_outside_word_duration_seconds, upper)
    boundaries = np.array([[w["start"], w["end"]] for w in rows])
    labels = np.array([w["_boundary_labels"] for w in rows])
    profiles = _boundary_profiles(logp, boundaries, labels, step, offset, p)
    if "mean" in calibration:
        profiles = profiles - np.asarray(calibration["mean"])[:, None, :]
    corrections = (profiles @ np.asarray(calibration["weights"])[..., None])[..., 0]
    corrections += np.asarray(calibration["offset"])[:, None]
    cap = p.boundary_max_correction_seconds
    grid = p.boundary_grid_seconds
    refined = boundaries + np.clip(corrections.T, -cap, cap)
    refined = np.clip(offset + np.round((refined - offset) / grid) * grid, offset, upper)
    by_speaker = defaultdict(list)
    for i, w in enumerate(rows):
        by_speaker[w["speaker_tag"]].append(i)
    min_duration = p.boundary_min_word_duration_seconds
    latest_onset = max(offset, upper - min_duration)
    for indices in by_speaker.values():
        spacing = min(p.boundary_min_onset_spacing_seconds,
                      (latest_onset - offset) / max(1, len(indices) - 1))
        g = np.arange(len(indices)) * spacing
        ordered = np.maximum.accumulate(refined[indices, 0] - g) + g
        refined[indices, 0] = np.minimum(ordered, latest_onset - (len(indices) - 1) * spacing + g)
    refined = np.clip(refined, boundaries - cap, boundaries + cap)
    refined = np.clip(refined, offset, upper)
    refined[:, 0] = np.minimum(refined[:, 0], latest_onset)
    refined[:, 1] = np.minimum(upper, np.maximum(refined[:, 0] + min_duration, refined[:, 1]))
    refined[:, 1] = np.minimum(refined[:, 1], np.clip(boundaries[:, 1] + cap, offset, upper))
    outside = np.array([w["_outside_diarization"] for w in rows], dtype=bool)
    seed = p.boundary_outside_word_duration_seconds
    refined[outside, 1] = np.minimum(upper, np.clip(
        refined[outside, 1],
        refined[outside, 0] + max(min_duration, seed - cap),
        refined[outside, 0] + max(min_duration, seed + cap)))
    for w, (start, end) in zip(rows, refined):
        w["start"], w["end"] = float(start), float(end)


def refine_word_timestamps(speaker_words: dict, diarization: list[dict], mapping: dict,
                           ctc_step: float, max_frames: int, p: Params) -> dict:
    """NeMo's ``_refine_word_timestamps``: same-speaker word edges snapped
    inside that speaker's diarization segments, durations capped at
    ``max_frames`` frames of at most 80 ms."""
    limit = max_frames * min(ctc_step, 0.08)
    snap = limit if p.onset_snap_seconds is None else p.onset_snap_seconds
    spacing = p.boundary_min_onset_spacing_seconds
    blend = p.refinement_end_blend
    segments = defaultdict(list)
    for s in diarization:
        if s["end"] > s["start"]:
            segments[int(s["speaker"])].append((s["start"], s["end"]))
    for v in segments.values():
        v.sort()
    refined = {}
    for tag, source in speaker_words.items():
        words = sorted((dict(w) for w in source), key=lambda w: w["start"])
        intervals = segments.get(mapping.get(tag), [])
        starts = [s for s, _ in intervals]
        member = []
        for w in words:
            i = bisect_right(starts, w["start"]) - 1
            member.append(i if not w.get("_outside_diarization") and i >= 0
                          and w["start"] < intervals[i][1] else None)
        seen = set()
        for i, (w, idx) in enumerate(zip(words, member)):
            if idx is None or idx in seen:
                continue
            onset = intervals[idx][0]
            if i:
                onset = max(onset, min(w["start"], words[i - 1]["start"] + spacing))
            if 0 < w["start"] - onset <= snap:
                w["start"] = onset
            seen.add(idx)
        for i, (w, idx) in enumerate(zip(words, member)):
            if w.pop("_outside_diarization", False):
                end = w["end"]
            elif idx is None:
                end = words[i + 1]["start"] if i + 1 < len(words) else w["end"]
            elif i + 1 < len(words) and member[i + 1] == idx:
                end = words[i + 1]["start"]
            else:
                end = intervals[idx][1]
            if blend != 1.0:
                end = w["end"] + blend * (end - w["end"])
            w["end"] = max(w["start"], min(end, w["start"] + limit))
        refined[tag] = words
    return refined


# -- one recording ------------------------------------------------------------


@dataclass
class Alignment:
    """Aligned words (``word``, ``start``, ``end``, ``speaker`` tag) in time
    order, 10 ms diarization segments, the tag to diarization speaker map
    and the streams with no legal path (``speaker``, ``reason``)."""

    words: list[dict]
    diarization: list[dict]
    mapping: dict
    failed: list[dict]


def align(prepared: Prepared, logp: np.ndarray, speaker_probs: np.ndarray | None,
          diarization_labels: np.ndarray | None, duration: float,
          upsampling: int = 2, p: Params = CHECKPOINT,
          time_offset: float = 0.0) -> Alignment:
    """NeMo's ``_align_banded_ctc`` for one recording.

    ``logp`` is ``(ctc frames, len(prepared.columns))`` log probabilities,
    blank in column 0; ``speaker_probs`` the 80 ms speaker probabilities
    ``(frames, speakers)``; ``diarization_labels`` the boolean 10 ms activity
    ``(speakers, frames)``; ``duration`` the audio seconds."""
    frames = logp.shape[0]
    ctc_step = float(duration) / frames
    greedy = logp.argmax(axis=1)
    tags = [s["speaker_tag"] for s in prepared.streams]
    if speaker_probs is None:
        speaker = None
        mapping = {t: None for t in tags}
    else:
        speaker = resample_speaker_probs(np.clip(speaker_probs, 0, 1).astype(np.float32), frames)
        mapping = speaker_mapping(logp, prepared.streams, speaker,
                                  p.speaker_activity_threshold, greedy)
    rows, failed = [], []
    for stream in prepared.streams:
        labels = np.asarray(stream["labels"], np.int64)
        column = mapping[stream["speaker_tag"]]
        activity = (np.ones(frames, np.float32) if speaker is None else
                    np.zeros(frames, np.float32) if column is None else speaker[:, column])
        positions = np.flatnonzero(activity >= p.support_activity_threshold)
        prior = p.speaker_logprob_weight * np.log(np.maximum(activity, EPSILON))
        path, score, _ = banded_search(logp, labels, positions, prior, greedy, p)
        if not np.isfinite(score):
            failed.append({"speaker": stream["speaker_tag"],
                           "reason": UNMATCHED if speaker is not None and column is None
                           else NO_PATH})
            continue
        full = np.full(frames, -1, np.int64)
        full[positions] = path
        rows.extend(word_rows(stream["words"], stream["state_to_word"], full, ctc_step,
                              time_offset, labels, p.maximum_token_len))
    diarization = ([] if diarization_labels is None else diarization_segments(
        diarization_labels, DIARIZATION_FRAME_SECONDS, time_offset, duration))
    if p.calibration is not None:
        interpolate_word_boundaries(rows, logp, ctc_step, time_offset, duration,
                                    diarization, mapping, p)
    speaker_words = refine_word_timestamps(
        _public_word_timestamps(rows), diarization, mapping, ctc_step,
        p.max_word_frames * upsampling, p)
    words = [{"word": w["word"], "start": w["start"], "end": w["end"], "speaker": w["speaker"]}
             for ws in speaker_words.values() for w in ws]
    words.sort(key=lambda w: (w["start"], w["end"]))
    return Alignment(words=words, diarization=diarization, mapping=mapping, failed=failed)


def seconds(value: float) -> float:
    """Times rounded to milliseconds, as NeMo's public result rounds them."""
    return round(float(value), 3)


def public_result(result: Alignment | None, error: str | None = None,
                  diarization: list[dict] | None = None) -> dict:
    """NeMo ``ctc_timestamps``' public result: ``words`` (speaker as the tag
    number in a string), ``diarization`` (speaker as the diarization index),
    ``speaker_tag_to_diarization_speaker`` and ``error``."""
    if result is None:
        return {"words": [], "diarization": [
            {"speaker": int(s["speaker"]), "start": seconds(s["start"]), "end": seconds(s["end"])}
            for s in diarization or ()],
            "speaker_tag_to_diarization_speaker": {}, "error": error}
    return {
        "words": [{"word": w["word"], "start": seconds(w["start"]), "end": seconds(w["end"]),
                   "speaker": str(w["speaker"])} for w in result.words],
        "diarization": [{"speaker": int(s["speaker"]), "start": seconds(s["start"]),
                         "end": seconds(s["end"])} for s in result.diarization],
        "speaker_tag_to_diarization_speaker": {
            str(t): None if c is None else int(c) for t, c in result.mapping.items()},
        "error": None,
    }


__all__ = ["CHECKPOINT", "PARAMETERS", "Params", "AlignmentError", "Alignment",
           "Prepared", "Tokenizer", "clean_transcript", "parse_words", "prepare",
           "align", "banded_search", "public_result", "diarization_segments"]
