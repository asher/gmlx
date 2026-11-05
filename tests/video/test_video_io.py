"""Clips to files: the ffmpeg probe, the MP4 mux through a stand-in
ffmpeg (the .part file, the rename, a failure and an abort), the WAV and
PNG writers, and one real mux when the system ffmpeg has the encoders.
Guides from files through the system ffmpeg (skipped without it): a
video's frames at its own rate and rotation, cropped as they arrive,
and a file's sound as two channels at the audio VAE's rate. CPU."""
from __future__ import annotations

import os
import shutil
import stat
import subprocess
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


@pytest.mark.parametrize("frames, latents, samples", [
    (124, 207, 165333), (243, 405, 324000), (362, 603, 482667)])
def test_the_audio_is_cut_or_padded_to_the_clips_length(frames, latents,
                                                        samples):
    """The decode gives 800 samples a latent: 267 past the last frame at
    124 frames, exact at 243, and 267 short at 362, padded with zeros."""
    from gmlx.video.families.h3 import family as hf

    assert hf.latent_shapes(frames, 32, 32)[1][3] == latents
    wave = np.arange(1, 2 * latents * 800 + 1,
                     dtype=np.float32).reshape(2, -1)
    got = io.fit_audio(wave, frames=frames, fps=hf.FPS, rate=32000)
    assert got.shape == (2, samples) and got.dtype == np.float32
    kept = min(samples, wave.shape[1])
    assert (got[:, :kept] == wave[:, :kept]).all()
    assert (got[:, kept:] == 0).all()


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


@pytest.fixture
def ffmpeg():
    exe = shutil.which("ffmpeg")
    if exe is None:
        pytest.skip("no ffmpeg on PATH")
    return exe


def _make(ffmpeg, *args) -> None:
    subprocess.run([ffmpeg, "-v", "error", "-nostdin", "-y", *args],
                   check=True)


def _rgb(ffmpeg, path, w, h) -> np.ndarray:
    """Every frame of ``path`` as ffmpeg decodes it to 8-bit RGB, unturned
    and at the file's size."""
    out = subprocess.run([ffmpeg, "-v", "error", "-nostdin", "-noautorotate",
                          "-i", path, "-fps_mode", "passthrough", "-f",
                          "rawvideo", "-pix_fmt", "rgb24", "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(out, np.uint8).reshape(-1, h, w, 3)


def test_a_guide_video_streams_its_frames_at_its_own_rate(ffmpeg, tmp_path):
    """A 30 fps second of 64x48: guide_length counts what guide_frames
    reads, at most ``most`` frames and at 30 a second, and each frame is
    the decoded one cropped about its center to the canvas."""
    from gmlx.video import images

    src = str(tmp_path / "clip.mp4")
    _make(ffmpeg, "-f", "lavfi", "-i", "testsrc=size=64x48:rate=30", "-t",
          "1", "-pix_fmt", "yuv420p", src)
    assert io.guide_length(src, most=100, ffmpeg=ffmpeg) == 30
    assert io.guide_length(src, most=22, ffmpeg=ffmpeg) == 22
    got = io.guide_frames(src, width=32, height=32, most=22, ffmpeg=ffmpeg)
    want = [images.cover(f, 32, 32) for f in _rgb(ffmpeg, src, 64, 48)[:22]]
    np.testing.assert_array_equal(got, np.stack(want))


@pytest.mark.parametrize("rotation", [90, -90])
def test_a_guide_videos_rotation_turns_its_frames(ffmpeg, tmp_path,
                                                  rotation):
    """A phone video stores its frames on their side with a rotation;
    each frame turns by it, a quarter turn counter-clockwise for 90, as
    ComfyUI's LoadVideo turns it (np.rot90 by the rotation's quarters)."""
    flat, src = str(tmp_path / "flat.mp4"), str(tmp_path / "turned.mp4")
    _make(ffmpeg, "-f", "lavfi", "-i", "testsrc=size=64x48:rate=24",
          "-frames:v", "5", "-pix_fmt", "yuv420p", flat)
    _make(ffmpeg, "-display_rotation", str(rotation), "-i", flat, "-c",
          "copy", src)
    got = io.guide_frames(src, width=48, height=64, most=5, ffmpeg=ffmpeg)
    want = np.rot90(_rgb(ffmpeg, flat, 64, 48), k=rotation // 90 % 4,
                    axes=(1, 2))
    np.testing.assert_array_equal(got, want)


def _pcm(path, channels) -> np.ndarray:
    """A 16-bit WAV's samples (channels, n) as float32, as ffmpeg reads
    them."""
    with wave.open(str(path), "rb") as w:
        x = np.frombuffer(w.readframes(w.getnframes()), "<i2")
    return (x.reshape(-1, channels).T / 32768.0).astype(np.float32)


def _tones(path, rate, seconds, freqs) -> None:
    t = np.arange(int(rate * seconds)) / rate
    x = np.stack([0.5 * np.sin(2 * np.pi * f * t) for f in freqs], -1)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(len(freqs))
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes((x * 32767).astype("<i2").tobytes())


def test_a_mono_guide_sounds_on_both_channels_as_it_is(ffmpeg, tmp_path):
    """Mono on both channels at its own level, as ComfyUI pads it; a
    stereo file keeps its channels apart."""
    mono, stereo = tmp_path / "mono.wav", tmp_path / "stereo.wav"
    _tones(mono, 32000, 0.5, [440])
    _tones(stereo, 32000, 0.5, [440, 660])
    got = io.guide_audio(str(mono), rate=32000, samples=16000,
                         ffmpeg=ffmpeg)
    np.testing.assert_array_equal(got, np.repeat(_pcm(mono, 1), 2, 0))
    got = io.guide_audio(str(stereo), rate=32000, samples=16000,
                         ffmpeg=ffmpeg)
    np.testing.assert_array_equal(got, _pcm(stereo, 2))


def test_a_guide_sound_is_resampled_cut_and_mixed_down(ffmpeg, tmp_path):
    """48 kHz to 32 kHz, cut at ``samples`` or shorter when the file is,
    and a 5.1 soundtrack mixed down to two channels."""
    long, short = tmp_path / "long.wav", tmp_path / "short.wav"
    _tones(long, 48000, 1.0, [440])
    _tones(short, 48000, 0.25, [440])
    assert io.guide_audio(str(long), rate=32000, samples=20000,
                          ffmpeg=ffmpeg).shape == (2, 20000)
    assert io.guide_audio(str(short), rate=32000, samples=20000,
                          ffmpeg=ffmpeg).shape == (2, 8000)
    six = tmp_path / "six.wav"
    _tones(six, 48000, 0.25, [220, 330, 440, 550, 660, 770])
    got = io.guide_audio(str(six), rate=32000, samples=20000, ffmpeg=ffmpeg)
    assert got.shape == (2, 8000) and np.isfinite(got).all()


def test_a_guide_video_without_sound_has_none(ffmpeg, tmp_path):
    """A video guide's sound is optional; an audio guide's is not, and
    a file with no video is no video guide."""
    src = str(tmp_path / "silent.mp4")
    _make(ffmpeg, "-f", "lavfi", "-i", "testsrc=size=64x48:rate=24",
          "-frames:v", "5", "-pix_fmt", "yuv420p", src)
    assert io.guide_audio(src, rate=32000, samples=800, ffmpeg=ffmpeg,
                          optional=True) is None
    with pytest.raises(io.FfmpegError, match="silent.mp4 has no audio "
                                             "stream$"):
        io.guide_audio(src, rate=32000, samples=800, ffmpeg=ffmpeg)
    _tones(tmp_path / "a.wav", 32000, 0.1, [440])
    with pytest.raises(io.FfmpegError, match="a.wav has no video stream$"):
        io.guide_frames(str(tmp_path / "a.wav"), width=32, height=32,
                        most=5, ffmpeg=ffmpeg)
    (tmp_path / "junk.mp4").write_bytes(b"junk" * 64)
    with pytest.raises(io.FfmpegError, match="could not read .*junk.mp4: "
                                             ".*Invalid data"):
        io.guide_length(str(tmp_path / "junk.mp4"), most=5, ffmpeg=ffmpeg)
