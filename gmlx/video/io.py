"""Clips to files: an MP4 through the system ffmpeg, or PNG frames and a
WAV file without it.

``probe`` checks ffmpeg before any model loads. ``Mux`` runs one ffmpeg
child that takes raw RGB frames on its stdin a chunk at a time, as the
VAE decodes them, with the clip's audio from a WAV file, and writes
``<output>.part``, renamed to the output when ffmpeg ends. A failure or a
Ctrl-C kills the child and removes the ``.part``.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import wave

import numpy as np

ENCODERS = ("h264_videotoolbox", "aac")
# H.264 on the VideoToolbox encoder at a fixed quality, AAC audio.
VIDEO_ARGS = ("-c:v", "h264_videotoolbox", "-q:v", "65", "-pix_fmt",
              "yuv420p")
AUDIO_ARGS = ("-c:a", "aac", "-b:a", "192k")


class FfmpegError(RuntimeError):
    """ffmpeg is missing, lacks an encoder gmlx uses, or failed."""


def probe(ffmpeg: str | None = None) -> str:
    """The path of an ffmpeg that has the encoders gmlx muxes with."""
    path = shutil.which(ffmpeg or "ffmpeg")
    if path is None:
        raise FfmpegError(
            f"{ffmpeg or 'ffmpeg'} is not on PATH: install ffmpeg "
            "(brew install ffmpeg), or pass --frames-dir to write PNG "
            "frames and a WAV file")
    try:
        out = subprocess.run([path, "-hide_banner", "-encoders"],
                             capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired) as e:
        raise FfmpegError(f"{path} -encoders failed: {e}") from None
    have = {line.split()[1] for line in out.stdout.splitlines()
            if len(line.split()) > 1}
    missing = [e for e in ENCODERS if e not in have]
    if out.returncode or missing:
        raise FfmpegError(
            f"{path} has no {' or '.join(missing) or 'encoder list'}: gmlx "
            "writes H.264 through VideoToolbox and AAC audio. Pass "
            "--frames-dir to write PNG frames and a WAV file")
    return path


def to_uint8(frames) -> np.ndarray:
    """8-bit RGB of float frames in [0, 1], rounded."""
    x = np.asarray(frames, np.float32)
    return np.clip(np.round(x * 255.0), 0, 255).astype(np.uint8)


def fit_audio(waveform, *, frames: int, fps: int, rate: int):
    """The waveform (channels, samples) cut or zero-padded to
    round(frames / fps x rate) samples, the length of the video.

    The audio decode gives whole 25 ms latents, so it lands up to 12.5 ms
    either side of the video. At 124 frames, 207 latents give 165600
    samples, cut to 165333. At 362 frames, 603 latents give 482400
    samples, padded to 482667 with 8 ms of silence."""
    n = round(frames * rate / fps)
    out = waveform[:, :n]
    if out.shape[1] < n:
        out = np.pad(out, ((0, 0), (0, n - out.shape[1])))
    return out


def write_wav(path: str, waveform, rate: int) -> str:
    """A 16-bit PCM WAV of a waveform (channels, samples) in [-1, 1]."""
    x = np.asarray(waveform, np.float32)
    pcm = np.clip(np.round(x * 32767.0), -32768, 32767).astype("<i2")
    tmp = f"{path}.part"
    with wave.open(tmp, "wb") as w:
        w.setnchannels(int(x.shape[0]))
        w.setsampwidth(2)
        w.setframerate(int(rate))
        w.writeframes(np.ascontiguousarray(pcm.T).tobytes())
    os.replace(tmp, path)
    return path


class Mux:
    """``with Mux(...) as m: m.write(frames)`` writes ``output`` as an MP4
    of the frames written, at ``fps``, with the audio of ``wav``."""

    def __init__(self, output: str, *, width: int, height: int, fps: int,
                 wav: str | None, ffmpeg: str):
        self.output = output
        self.part = f"{output}.part"
        self.frames = 0
        cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
               "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", f"{width}x{height}", "-r", str(fps), "-i", "pipe:0"]
        if wav is not None:
            cmd += ["-i", wav]
        cmd += ["-map", "0:v"] + (["-map", "1:a"] if wav else [])
        cmd += [*VIDEO_ARGS] + ([*AUDIO_ARGS] if wav else [])
        cmd += ["-movflags", "+faststart", "-f", "mp4", self.part]
        # A file, not a pipe, so ffmpeg never blocks on a full stderr.
        self._err = tempfile.TemporaryFile()
        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                      stderr=self._err)

    def write(self, frames) -> None:
        """Frames (n, H, W, 3), float in [0, 1] or uint8."""
        x = np.asarray(frames)
        if x.dtype != np.uint8:
            x = to_uint8(x)
        try:
            self._proc.stdin.write(np.ascontiguousarray(x).tobytes())
        except BrokenPipeError:
            self._fail()
        self.frames += int(x.shape[0])

    def _fail(self):
        self._proc.wait()
        self._err.seek(0)
        err = self._err.read().decode(errors="replace").strip()
        self._err.close()
        self._remove()
        raise FfmpegError(f"ffmpeg failed writing {self.output}"
                          + (f": {err}" if err else ""))

    def _remove(self):
        try:
            os.unlink(self.part)
        except FileNotFoundError:
            pass

    def close(self) -> str:
        try:
            self._proc.stdin.close()
        except BrokenPipeError:
            self._fail()
        if self._proc.wait():
            self._fail()
        self._err.close()
        os.replace(self.part, self.output)
        return self.output

    def abort(self) -> None:
        if self._proc.poll() is None:
            self._proc.kill()
        self._proc.wait()
        self._err.close()
        self._remove()

    def __enter__(self):
        return self

    def __exit__(self, kind, value, tb):
        if kind is None:
            self.close()
        else:
            self.abort()
        return False


def write_frames(folder: str, frames, start: int) -> int:
    """PNG files ``frame_000001.png`` on of frames (n, H, W, 3) in [0, 1]
    from frame ``start`` + 1, and the next start."""
    from PIL import Image

    x = to_uint8(frames)
    for i, f in enumerate(x):
        path = os.path.join(folder, f"frame_{start + i + 1:06d}.png")
        Image.fromarray(f).save(path + ".part", format="PNG")
        os.replace(path + ".part", path)
    return start + int(x.shape[0])
