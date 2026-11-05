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
