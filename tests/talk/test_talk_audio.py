"""talk_audio pure parts: endpointer, energy VAD, earcons, keyword-line
composition. No sounddevice, no sherpa-onnx, no network."""
from __future__ import annotations

import numpy as np
import pytest

from gmlx import talk_audio as ta
from gmlx.talk_audio import EnergyVAD, Endpointer, rms_dbfs


def _frame(level: int) -> np.ndarray:
    return np.full(ta.FRAME_SAMPLES, level, dtype=np.int16)


LOUD, QUIET = _frame(8000), _frame(10)


# rms / energy VAD
def test_rms_dbfs_scale():
    assert rms_dbfs(_frame(32767)) == pytest.approx(0.0, abs=0.1)
    assert rms_dbfs(_frame(0)) < -80
    assert rms_dbfs(np.array([], dtype=np.int16)) < -80


def test_energy_vad_gate():
    vad = EnergyVAD(threshold_dbfs=-38.0)
    assert vad.prob(LOUD) == 1.0
    assert vad.prob(QUIET) == 0.0


# Endpointer
def _ep(**kw):
    kw.setdefault("silence_ms", 240.0)     # 3 frames at 80 ms
    kw.setdefault("min_speech_ms", 160.0)  # 2 frames
    kw.setdefault("pre_roll_ms", 160.0)    # 2 frames
    kw.setdefault("min_level_dbfs", -60.0)
    return Endpointer(**kw)


def test_endpointer_start_end_with_preroll():
    ep = _ep()
    for _ in range(5):                       # idle: fills the pre-roll ring
        assert ep.feed(QUIET, 0.0) is None
    assert ep.feed(LOUD, 1.0) == ("start", None)
    for _ in range(3):
        assert ep.feed(LOUD, 1.0) is None
    for _ in range(2):
        assert ep.feed(QUIET, 0.0) is None   # hangover not yet reached
    kind, utt = ep.feed(QUIET, 0.0)
    assert kind == "end"
    # 2 pre-roll + 4 speech + 3 silence frames survived into the utterance
    assert len(utt) == 9 * ta.FRAME_SAMPLES
    assert utt.dtype == np.int16
    assert not ep.capturing                  # ready for the next utterance


def test_endpointer_drops_too_short():
    ep = _ep()
    assert ep.feed(LOUD, 1.0) == ("start", None)   # one speech frame = 80 ms
    for _ in range(2):
        assert ep.feed(QUIET, 0.0) is None
    assert ep.feed(QUIET, 0.0) == ("drop", "too short")


def test_endpointer_drops_too_quiet():
    ep = _ep(min_level_dbfs=-20.0)           # very demanding floor
    soft = _frame(200)
    assert ep.feed(soft, 1.0) == ("start", None)
    for _ in range(3):
        ep.feed(soft, 1.0)
    for _ in range(2):
        ep.feed(QUIET, 0.0)
    assert ep.feed(QUIET, 0.0) == ("drop", "too quiet")


def test_endpointer_speech_resets_hangover():
    ep = _ep()
    ep.feed(LOUD, 1.0)
    ep.feed(QUIET, 0.0)
    ep.feed(QUIET, 0.0)
    assert ep.feed(LOUD, 1.0) is None        # speech resumes, hangover clears
    for _ in range(2):
        assert ep.feed(QUIET, 0.0) is None
    kind, _ = ep.feed(QUIET, 0.0)
    assert kind == "end"


# Wake word
def test_keyword_line_composition(monkeypatch, tmp_path):
    class FakeSP:
        def load(self, path):
            pass

        def encode(self, text, out_type=str):
            assert text == "HEY GADGET"
            return ["▁HE", "Y", "▁GA", "D", "GET"]

    import sentencepiece
    monkeypatch.setattr(sentencepiece, "SentencePieceProcessor", FakeSP)
    (tmp_path / "bpe.model").write_bytes(b"")
    (tmp_path / "tokens.txt").write_text(
        "\n".join(f"{t} {i}" for i, t in
                  enumerate(["▁HE", "Y", "▁GA", "D", "GET"])))
    line = ta.SherpaKwsDetector._keyword_line(
        "hey gadget", str(tmp_path), 0.3, 2.5)
    assert line == "▁HE Y ▁GA D GET :2.5 #0.3 @HEY_GADGET"


def test_keyword_line_rejects_unspellable(monkeypatch, tmp_path):
    class FakeSP:
        def load(self, path):
            pass

        def encode(self, text, out_type=str):
            return ["▁Z", "Q"]

    import sentencepiece
    monkeypatch.setattr(sentencepiece, "SentencePieceProcessor", FakeSP)
    (tmp_path / "bpe.model").write_bytes(b"")
    (tmp_path / "tokens.txt").write_text("▁Z 0\n")
    with pytest.raises(ta.TalkAudioError, match="can't spell"):
        ta.SherpaKwsDetector._keyword_line("zq", str(tmp_path), 0.5, 2.0)


# Earcons
def test_earcons_shape_and_kinds():
    for kind in ("wake", "idle"):
        pcm, rate = ta.earcon(kind)
        assert rate == 24000 and pcm.dtype == np.int16
        assert 0 < len(pcm) <= rate            # short blip
        assert int(np.abs(pcm).max()) <= 0.3 * 32767   # polite volume
        assert abs(int(pcm[0])) < 300 and abs(int(pcm[-1])) < 300  # no clicks
    with pytest.raises(ValueError):
        ta.earcon("boing")


def test_wake_and_idle_earcons_differ():
    a, _ = ta.earcon("wake")
    b, _ = ta.earcon("idle")
    assert not np.array_equal(a, b)


# Playback gain (no PortAudio: a prepared backend with a fake output stream)
def _gain_backend():
    import threading

    b = ta.SoundDeviceBackend.__new__(ta.SoundDeviceBackend)
    b._sd = None
    b._out_spec = None
    b._out_rate = 24000
    b._out_lock = threading.Lock()
    b.gain = 1.0
    writes = []

    class Out:
        def write(self, data):
            writes.append(np.array(data, copy=True).ravel())

    b._out_stream = Out()
    return b, writes


def test_play_applies_gain_per_slice():
    b, writes = _gain_backend()
    b.gain = 0.5
    assert b.play(np.array([1000, -2000, 32767], dtype=np.int16), 24000)
    got = np.concatenate(writes)
    assert got.dtype == np.int16
    assert list(got) == [500, -1000, 16383]


def test_play_unity_gain_passes_untouched_and_overdrive_clips():
    b, writes = _gain_backend()
    pcm = np.array([1000, -32768, 32767], dtype=np.int16)
    assert b.play(pcm, 24000)
    assert list(np.concatenate(writes)) == [1000, -32768, 32767]
    writes.clear()
    b.gain = 2.0                     # attribute is unclamped: must not wrap
    assert b.play(pcm, 24000)
    assert list(np.concatenate(writes)) == [2000, -32768, 32767]


def test_play_gain_change_lands_mid_playback():
    b, writes = _gain_backend()

    class Out:
        def write(self, data):
            writes.append(np.array(data, copy=True).ravel())
            b.gain = 0.5             # a slider drag between slices

    b._out_stream = Out()
    pcm = np.full(24000, 1000, dtype=np.int16)   # 1s -> several 150ms slices
    assert b.play(pcm, 24000)
    assert len(writes) > 1
    assert writes[0][0] == 1000 and writes[-1][0] == 500
