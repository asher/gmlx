"""Word timestamps on a served model that hears audio (gmlx/serve/
word_timestamps.py): the route's alignment over a capture, a disconnect
while the word timestamp head runs, and the failures each route answers.

The model and the CTC head are stubbed; the log probabilities and the
expected words come from the NeMo golden of
tests/models/test_nemotron_transcribe_align.py."""

from __future__ import annotations

import asyncio
import importlib
import importlib.util
import types
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("gmlx._vendor.mlx_vlm")

from gmlx.models.nemotron_h import align as al  # noqa: E402
from gmlx.serve import native_stt, word_timestamps  # noqa: E402
from gmlx.serve.engine_jobs import JobCancelled, JobTimeout  # noqa: E402
from gmlx.serve.patches import native_audio  # noqa: E402
from gmlx.serve.stt import STTRequestError  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
_APP = importlib.import_module("gmlx._vendor.mlx_vlm.server.app")


def _fixtures():
    spec = importlib.util.spec_from_file_location(
        "_wt_fixtures", ROOT / "scripts" / "image_oracle" / "fixtures.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.load("align_nemo_tiny", root=ROOT / "tests" / "fixtures" / "nemotron_transcribe")


@pytest.fixture(scope="module")
def golden():
    arrays, manifest = _fixtures()
    inputs = manifest["inputs"]["two"]
    logp = arrays["two/logp"].astype(np.float32)
    capture = types.SimpleNamespace(
        states=np.zeros((logp.shape[0] // 2, 1), np.float32),
        speaker_probs=arrays["two/speaker_probs"], labels=arrays["two/labels"],
        samples=int(round(inputs["duration"] * word_timestamps.SAMPLE_RATE)))
    return {"tok": al.Tokenizer(arrays["spm"].tobytes()), "logp": logp,
            "capture": capture, "transcript": inputs["transcript"],
            "want": manifest["results"]["two"]}


@pytest.fixture
def served(golden, monkeypatch):
    """A served model whose word timestamp head answers from the golden."""
    tok = golden["tok"]
    model = types.SimpleNamespace(
        audio_tower=types.SimpleNamespace(cfg=types.SimpleNamespace(
            ctc=types.SimpleNamespace(upsampling=2, blank_id=tok.sp.get_piece_size()))),
        ctc_tokenizer=lambda: tok)
    processor = types.SimpleNamespace(tokenizer=types.SimpleNamespace(
        all_special_tokens=["<|im_end|>"]))
    monkeypatch.setattr(_APP, "get_cached_model", lambda name: (model, processor, None),
                        raising=False)
    calls = []

    def ctc_log_probs(rg, model, capture, columns, stop, timeout_s):
        calls.append(columns)
        return golden["logp"][:, columns]

    monkeypatch.setattr(native_audio, "ctc_log_probs", ctc_log_probs)
    return types.SimpleNamespace(calls=calls, golden=golden)


class _Client:
    """An HTTP request whose client stays, or leaves at once."""

    def __init__(self, leaves=False):
        self.leaves = leaves

    async def receive(self):
        if self.leaves:
            return {"type": "http.disconnect"}
        await asyncio.Event().wait()


def _align(served, transcript=None, client=None):
    g = served.golden
    return asyncio.run(word_timestamps.align(
        client or _Client(), "m", g["capture"],
        g["transcript"] if transcript is None else transcript))


def test_the_route_times_the_transcript_as_nemo_does(served):
    result, error = _align(served, served.golden["transcript"] + "<|im_end|>")
    assert error is None and not result.failed
    want = served.golden["want"]["words"]
    assert [(w["word"], w["speaker"]) for w in result.words] == [
        (w["word"], w["speaker"]) for w in want]
    for got, w in zip(result.words, want):
        assert (got["start"], got["end"]) == pytest.approx((w["start"], w["end"]), abs=1e-9)
    assert len(served.calls) == 1


def test_an_empty_transcript_asks_nothing_of_the_head(served):
    result, error = _align(served, "<|im_end|>")
    assert error is None and result.words == [] and result.diarization
    assert served.calls == []


def test_a_transcript_too_long_for_the_clip_names_why(served):
    result, error = _align(served, " ".join(["word"] * 2000))
    assert result is None and "too long for the CTC timeline" in error
    assert served.calls == []
    public = word_timestamps.public(result, error, served.golden["capture"])
    assert public["error"] == word_timestamps.FAILED and public["words"] == []


def test_a_client_that_leaves_stops_the_head(served, monkeypatch):
    def ctc_log_probs(rg, model, capture, columns, stop, timeout_s):
        assert stop.wait(30), "the disconnect never reached the job"
        raise JobCancelled("the client went away")

    monkeypatch.setattr(native_audio, "ctc_log_probs", ctc_log_probs)
    with pytest.raises(JobCancelled):
        _align(served, client=_Client(leaves=True))
    chat = asyncio.run(word_timestamps.chat_result(
        _Client(leaves=True), "m", served.golden["capture"], served.golden["transcript"]))
    assert chat["error"] == word_timestamps.FAILED
    with pytest.raises(STTRequestError) as e:
        asyncio.run(native_stt.align_transcript(
            _Client(leaves=True), "m", served.golden["capture"], served.golden["transcript"]))
    assert e.value.status_code == 499


def test_a_head_that_times_out_is_a_504_on_stt_and_an_error_on_chat(served, monkeypatch):
    def ctc_log_probs(*a):
        raise JobTimeout("the word timestamp head did not finish")

    monkeypatch.setattr(native_audio, "ctc_log_probs", ctc_log_probs)
    g = served.golden
    with pytest.raises(STTRequestError) as e:
        asyncio.run(native_stt.align_transcript(_Client(), "m", g["capture"], g["transcript"]))
    assert e.value.status_code == 504
    chat = asyncio.run(word_timestamps.chat_result(_Client(), "m", g["capture"],
                                                   g["transcript"]))
    assert chat["error"] == word_timestamps.FAILED and chat["words"] == []


def test_a_failed_stream_fails_the_whole_stt_alignment(served, monkeypatch):
    g = served.golden
    real = al.align

    def failing(prepared, *a, **k):
        out = real(prepared, *a, **k)
        out.failed.append({"speaker": 1, "reason": "no_legal_path_in_anchor_band"})
        return out

    monkeypatch.setattr(al, "align", failing)
    out = asyncio.run(native_stt.align_transcript(_Client(), "m", g["capture"],
                                                  g["transcript"]))
    assert out == {"words": [], "error": "no_legal_path_in_anchor_band"}


def test_a_stream_that_fails_sends_no_timestamps():
    async def body():
        yield 'data: {"id": "c", "choices": [{"index": 0, "delta": {"content": "Hi"}, "finish_reason": null}]}\n\n'
        yield 'data: {"error": {"message": "boom"}}\n\n'
        yield "data: [DONE]\n\n"

    async def collect():
        return [e async for e in word_timestamps._stream(body(), None, "m", "cap")]

    out = asyncio.run(collect())
    assert len(out) == 3 and "ctc_timestamps" not in "".join(out)
    assert out[-1] == "data: [DONE]\n\n"

