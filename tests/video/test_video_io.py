"""Clips to files: the ffmpeg probe, the MP4 mux through a stand-in
ffmpeg (the .part file, the rename, a failure and an abort), the WAV and
PNG writers, and one real mux when the system ffmpeg has the encoders.
CPU."""
from __future__ import annotations

import os
import stat
import wave

import numpy as np
import pytest
from video_fixtures import wait_for

from gmlx.video import io

_ENCODERS = """Encoders:
 V....D h264_videotoolbox    VideoToolbox H.264 Encoder
 A....D aac                  AAC (Advanced Audio Coding)
"""


def _ffmpeg(tmp_path, body: str, encoders: str = _ENCODERS) -> str:
    """A stand-in ffmpeg: ``-encoders`` prints ``encoders``, any other
    call runs ``body`` with ``$last`` the output path."""
    (tmp_path / "encoders.txt").write_text(encoders)
    path = tmp_path / "ffmpeg"
    path.write_text(f"""#!/bin/sh
for last; do :; done
case " $* " in *" -encoders "*) cat "{tmp_path}/encoders.txt"; exit 0;; esac
{body}
""")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return str(path)


def test_probe_finds_an_ffmpeg_with_the_encoders(tmp_path):
    ff = _ffmpeg(tmp_path, "exit 0")
    assert io.probe(ff) == ff


def test_probe_names_a_missing_encoder(tmp_path):
    ff = _ffmpeg(tmp_path, "exit 0", encoders=" A....D aac  AAC\n")
    with pytest.raises(io.FfmpegError, match="has no h264_videotoolbox"):
        io.probe(ff)


def test_probe_without_ffmpeg_points_to_frames_dir(tmp_path):
    with pytest.raises(io.FfmpegError, match="brew install ffmpeg.*--frames-dir"):
        io.probe(str(tmp_path / "nothing"))


def _frames(n=3, h=4, w=6):
    return np.linspace(0, 1, n * h * w * 3, dtype=np.float32).reshape(
        n, h, w, 3)


def test_a_mux_writes_part_then_renames(tmp_path):
    ff = _ffmpeg(tmp_path, 'cat > "$last"')
    out = str(tmp_path / "clip.mp4")
    with io.Mux(out, width=6, height=4, fps=24, wav=None, ffmpeg=ff) as m:
        m.write(_frames())
        m.write(_frames(2))
        assert not os.path.exists(out)
    assert m.frames == 5 and not os.path.exists(out + ".part")
    data = np.frombuffer(open(out, "rb").read(), np.uint8)
    assert data.size == 5 * 4 * 6 * 3
    assert np.array_equal(data[:72], io.to_uint8(_frames()[0]).ravel())


def test_a_failed_mux_raises_ffmpegs_message_and_leaves_no_part(tmp_path):
    ff = _ffmpeg(tmp_path, 'cat > "$last"; echo "bad frame size" >&2; exit 1')
    out = str(tmp_path / "clip.mp4")
    m = io.Mux(out, width=6, height=4, fps=24, wav=None, ffmpeg=ff)
    m.write(_frames())
    with pytest.raises(io.FfmpegError, match="bad frame size"):
        m.close()
    assert not os.path.exists(out) and not os.path.exists(out + ".part")


def test_an_aborted_mux_kills_ffmpeg_and_leaves_no_part(tmp_path):
    ff = _ffmpeg(tmp_path, 'cat > "$last"')
    out = str(tmp_path / "clip.mp4")
    with pytest.raises(KeyboardInterrupt):
        with io.Mux(out, width=6, height=4, fps=24, wav=None, ffmpeg=ff) as m:
            m.write(_frames())
            assert wait_for(out + ".part")
            raise KeyboardInterrupt
    assert m._proc.returncode is not None
    assert not os.path.exists(out) and not os.path.exists(out + ".part")


@pytest.mark.parametrize("n, side, at", [(1, 8, "close"),
                                         (4, 512, "write")])
def test_a_write_into_an_ffmpeg_that_exited_raises_its_message(tmp_path, n,
                                                                side, at):
    """A small frame waits in the pipe's buffer until close; 3 MB meet
    the closed pipe in the write itself."""
    ff = _ffmpeg(tmp_path, 'echo "could not open output" >&2; exit 1')
    out = str(tmp_path / "clip.mp4")
    m = io.Mux(out, width=side, height=side, fps=24, wav=None, ffmpeg=ff)
    m._proc.wait()
    frames = np.zeros((n, side, side, 3), np.float32)
    try:
        if at == "close":
            m.write(frames)
        with pytest.raises(io.FfmpegError, match="could not open output"):
            if at == "write":
                m.write(frames)
            else:
                m.close()
    finally:
        m.abort()
    assert not os.path.exists(out) and not os.path.exists(out + ".part")


def test_frames_round_to_8_bits():
    x = np.array([-0.5, 0.0, 0.5 / 255, 0.5, 1.0, 2.0], np.float32)
    assert io.to_uint8(x).tolist() == [0, 0, 0, 128, 255, 255]


def test_a_wav_holds_stereo_16_bit_samples(tmp_path):
    left = np.array([0.0, 0.5, -1.0, 1.0], np.float32)
    path = io.write_wav(str(tmp_path / "a.wav"),
                        np.stack([left, -left]), 32000)
    with wave.open(path, "rb") as w:
        assert (w.getnchannels(), w.getsampwidth(), w.getframerate(),
                w.getnframes()) == (2, 2, 32000, 4)
        pcm = np.frombuffer(w.readframes(4), "<i2").reshape(4, 2)
    assert pcm[:, 0].tolist() == [0, 16384, -32767, 32767]
    assert pcm[:, 1].tolist() == [0, -16384, 32767, -32767]
    assert not os.path.exists(path + ".part")


def test_frames_are_numbered_pngs(tmp_path):
    pytest.importorskip("PIL")
    nxt = io.write_frames(str(tmp_path), _frames(2), 0)
    nxt = io.write_frames(str(tmp_path), _frames(1), nxt)
    assert nxt == 3
    assert sorted(os.listdir(tmp_path)) == [
        "frame_000001.png", "frame_000002.png", "frame_000003.png"]


def test_a_real_mux_with_audio(tmp_path):
    """The system ffmpeg, when it has VideoToolbox H.264 and AAC."""
    try:
        ff = io.probe()
    except io.FfmpegError as e:
        pytest.skip(str(e))
    wav = io.write_wav(str(tmp_path / "a.wav"),
                       np.zeros((2, 32000), np.float32), 32000)
    out = str(tmp_path / "clip.mp4")
    with io.Mux(out, width=128, height=128, fps=24, wav=wav, ffmpeg=ff) as m:
        m.write(np.zeros((24, 128, 128, 3), np.float32))
    assert os.path.getsize(out) > 0 and not os.path.exists(out + ".part")
