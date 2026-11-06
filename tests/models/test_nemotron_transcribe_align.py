"""Word timestamps and speakers for Nemotron 3.5 Transcribe
(gmlx/models/nemotron_h/align.py) against NeMo's aligner.

The goldens come from scripts/speech_oracle/ref_align.py: NeMo Speech's
ctc_timestamp_utils.py at 979cc6c6, run on the synthetic cases of
scripts/speech_oracle/align_cases.py with the checkpoint's alignment
parameters. Pure NumPy; no model."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from gmlx.models.nemotron_h import align

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "fixtures" / "nemotron_transcribe"
NAME = "align_nemo_tiny"


def _module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


fx = _module(ROOT / "scripts" / "image_oracle" / "fixtures.py", "_align_fixtures")


@pytest.fixture(scope="module")
def golden():
    return fx.load(NAME, root=FIXTURES)


@pytest.fixture(scope="module")
def tokenizer(golden):
    arrays, _ = golden
    return align.Tokenizer(arrays["spm"].tobytes())


def _case(golden, name):
    arrays, manifest = golden
    logp = arrays[f"{name}/logp"].astype(np.float32)
    return (manifest["inputs"][name], logp, arrays.get(f"{name}/speaker_probs"),
            arrays.get(f"{name}/labels"), manifest["results"][name])


CASES = ["plain", "two", "long", "unmatched", "cjk"]


@pytest.mark.parametrize("name", CASES)
def test_alignment_matches_nemo(golden, tokenizer, name):
    inputs, logp, probs, labels, want = _case(golden, name)
    blank = tokenizer.sp.get_piece_size()
    prepared = align.prepare(inputs["transcript"], tokenizer, blank, logp.shape[0])
    got = align.align(prepared, logp[:, prepared.columns], probs, labels,
                      inputs["duration"])
    assert [(w["word"], w["speaker"]) for w in got.words] == [
        (w["word"], w["speaker"]) for w in want["words"]]
    for g, w in zip(got.words, want["words"]):
        assert g["start"] == pytest.approx(w["start"], abs=1e-9)
        assert g["end"] == pytest.approx(w["end"], abs=1e-9)
    assert len(got.diarization) == len(want["diarization"])
    for g, w in zip(got.diarization, want["diarization"]):
        assert g["speaker"] == w["speaker"]
        assert (g["start"], g["end"]) == pytest.approx((w["start"], w["end"]), abs=1e-9)
    assert {str(k): v for k, v in got.mapping.items()} == want["mapping"]
    # NeMo names every failed stream a band miss; gmlx names a tag that
    # matched no diarization speaker as such.
    assert [(f["speaker"], f["reason"]) for f in got.failed] == [
        (f["speaker"], align.UNMATCHED if want["mapping"][str(f["speaker"])] is None
         else f["reason"]) for f in want["failed"]]


def test_the_long_case_runs_in_a_band_narrower_than_its_target(golden):
    *_, want = _case(golden, "long")
    assert all(d["half_width"] * 2 + 1 < d["states"] for d in want["banded"])


def test_a_transcript_longer_than_the_timeline_is_refused(golden, tokenizer):
    inputs, logp, *_, want = _case(golden, "too_long")
    with pytest.raises(align.AlignmentError, match="too long for the CTC timeline"):
        align.prepare(inputs["transcript"], tokenizer, tokenizer.sp.get_piece_size(),
                      logp.shape[0])
    assert "too long" in want["error"]


def _viterbi(logp, labels):
    """The exact max-sum CTC path score over every state."""
    s = len(labels)
    prev = np.full(s, -np.inf)
    prev[:2] = logp[0, labels[:2]]
    for t in range(1, len(logp)):
        cur = prev.copy()
        cur[1:] = np.maximum(cur[1:], prev[:-1])
        for j in range(2, s):
            if labels[j] != 0 and labels[j] != labels[j - 2]:
                cur[j] = max(cur[j], prev[j - 2])
        prev = cur + logp[t, labels]
    return max(prev[-1], prev[-2])


@pytest.mark.parametrize("seed", range(6))
def test_a_band_that_covers_every_state_finds_the_best_path(seed):
    rng = np.random.default_rng(seed)
    tokens = rng.integers(1, 6, int(rng.integers(3, 9)))
    labels = [0]
    for t in tokens:
        labels += [int(t), 0]
    logp = np.log(rng.dirichlet(np.ones(6), 40)).astype(np.float32)
    full = align.Params(**{**align.PARAMETERS, "band_full_below_frames": 10**6})
    path, score, _ = align.banded_search(logp, labels, np.arange(40), None, p=full)
    assert score == pytest.approx(_viterbi(logp, np.array(labels)), rel=1e-5)
    assert (path >= 0).all() and path[0] in (0, 1) and path[-1] >= len(labels) - 2
    assert (np.diff(path) >= 0).all() and (np.diff(path) <= 2).all()


def test_speaker_tags_split_the_transcript_into_streams():
    words = align.parse_words(
        "hi <spk:0> one\u2014two <sty:stt:flat> <spk:1> three <ev:laughter> four")
    assert [(w["word"], w["speaker_tag"], w["turn_index"]) for w in words] == [
        ("hi", None, None), ("one", 0, 0), ("two", 0, 0), ("three", 1, 1),
        ("<ev:laughter>", 1, 1), ("four", 1, 1)]
    assert [w["speaker_tag"] for w in align.parse_words("no tags here")] == [0, 0, 0]


def test_cleaning_drops_control_tokens_but_keeps_speaker_and_aligned_tags():
    special = ["<|im_end|>", "<spk:1>", "<ev:laughter>", "<sty:end:flat>"]
    assert align.clean_transcript(
        "<spk:1> yes <ev:laughter> <sty:end:flat>no<|im_end|>", special) == (
        "<spk:1> yes <ev:laughter> no")


def test_thai_is_refused():
    with pytest.raises(align.AlignmentError, match="Thai"):
        align.timestamp_units(align.parse_words("\u0e2a\u0e27\u0e31\u0e2a\u0e14\u0e35"))


def test_the_public_result_rounds_to_milliseconds_and_names_tags_as_strings():
    result = align.Alignment(
        words=[{"word": "hi", "start": 0.12345, "end": 0.5, "speaker": 0}],
        diarization=[{"speaker": 2, "start": 0.1, "end": 0.6000001}],
        mapping={0: 2, 1: None}, failed=[])
    assert align.public_result(result) == {
        "words": [{"word": "hi", "start": 0.123, "end": 0.5, "speaker": "0"}],
        "diarization": [{"speaker": 2, "start": 0.1, "end": 0.6}],
        "speaker_tag_to_diarization_speaker": {"0": 2, "1": None},
        "error": None}
    assert align.public_result(None, "alignment_failed")["error"] == "alignment_failed"


FOLD_FRAMES = 300  # 40 ms CTC frames: 12 s


def _spread(prepared, ranges):
    """CTC log probabilities with each stream's tokens peaking at even steps
    over its ``(first, last)`` frame range, blank everywhere else, and the
    seconds at which each word's first token peaks."""
    logp = np.full((FOLD_FRAMES, len(prepared.columns)), np.log(1e-4), np.float32)
    logp[:, 0] = np.log(0.99)
    onsets = []
    for stream, (a, b) in zip(prepared.streams, ranges):
        tokens = stream["labels"][1::2]
        owners = [w for w in stream["state_to_word"][1::2]]
        frames = np.linspace(a, b, len(tokens)).round().astype(int)
        for i, (t, frame) in enumerate(zip(tokens, frames)):
            logp[frame] = np.log(1e-4)
            logp[frame, t] = np.log(0.99)
            if i == 0 or owners[i] != owners[i - 1]:
                onsets.append(frame * 0.04)
    return logp, sorted(onsets)


def _activity(spans, speakers):
    """80 ms speaker probabilities and 10 ms labels: each ``(speaker,
    first, last)`` span is active, in 40 ms frames."""
    probs = np.full((FOLD_FRAMES, speakers), 0.001, np.float32)
    for s, a, b in spans:
        probs[a:b + 1, s] = 0.95
    labels = np.repeat(probs.T > 0.5, 4, axis=1)
    return probs[::2], labels


def test_a_voice_split_over_two_diarizer_speakers_aligns_over_both(tokenizer):
    words = "red mat and ran garden at twelve open meeting closed budget seven new friday green"
    blank = tokenizer.sp.get_piece_size()
    prepared = align.prepare("<spk:0> " + words, tokenizer, blank, FOLD_FRAMES)
    logp, onsets = _spread(prepared, [(5, 290)])
    probs, labels = _activity([(0, 0, 149), (1, 150, 299)], 2)
    got = align.align(prepared, logp, probs, labels, FOLD_FRAMES * 0.04)
    starts = [w["start"] for w in got.words]
    assert len(starts) == len(words.split())
    assert np.abs(np.array(starts) - onsets).max() <= 0.2
    assert max(onsets) > 150 * 0.04
    assert len(got.mapping) == 1 and got.mapping[0] in (0, 1)
    assert {s["speaker"] for s in got.diarization} == {0, 1}


def test_a_leftover_column_joins_the_tag_whose_words_it_heard(tokenizer):
    first = "red mat and ran garden at twelve"
    second = "open meeting closed budget seven new friday green tuesday river"
    blank = tokenizer.sp.get_piece_size()
    prepared = align.prepare(f"<spk:0> {first} <spk:1> {second}", tokenizer, blank,
                             FOLD_FRAMES)
    logp, onsets = _spread(prepared, [(5, 120), (140, 290)])
    probs, labels = _activity([(0, 0, 130), (1, 131, 255), (2, 256, 299)], 3)
    speaker = align.resample_speaker_probs(probs, FOLD_FRAMES)
    mapping, extra = align.speaker_mapping(
        logp, prepared.streams, speaker, align.CHECKPOINT.speaker_activity_threshold,
        logp.argmax(axis=1))
    assert mapping == {0: 0, 1: 1} and extra == {1: [2]}
    got = align.align(prepared, logp, probs, labels, FOLD_FRAMES * 0.04)
    assert got.mapping == {0: 0, 1: 1}
    assert len(got.words) == len(onsets)
    assert np.abs(np.array([w["start"] for w in got.words]) - onsets).max() <= 0.2
    assert sum(w["speaker"] == 1 and w["start"] > 256 * 0.04 for w in got.words) >= 2
    assert {s["speaker"] for s in got.diarization} == {0, 1, 2}
