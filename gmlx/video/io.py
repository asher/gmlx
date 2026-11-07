"""Clips to files: an MP4 through the system ffmpeg, or PNG frames and a
WAV file without it.

``probe`` checks ffmpeg before any model loads. ``Mux`` runs one ffmpeg
child that takes raw RGB frames on its stdin a chunk at a time, as the
VAE decodes them, with the clip's audio from a WAV file, and writes
``<output>.part``, renamed to the output when ffmpeg ends. A failure or a
Ctrl-C kills the child and removes the ``.part``.

Guides are read through ffmpeg too. ``guide_frames`` streams a video's
frames at the file's own rate, turned by its rotation, and crops and
resizes each as it arrives, so a large source never sits whole in
memory; ``guide_length`` counts the frames it would read. ``guide_audio``
reads a file's first audio stream at the audio VAE's rate as two
channels: mono on both, more than two mixed down by ffmpeg.
"""

from __future__ import annotations

import contextlib
import fcntl
import os
import re
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


# The header ffmpeg's PPM encoder writes before each frame's pixels.
_PPM = re.compile(rb"P6\n(\d+) (\d+)\n255\n")
# What ffmpeg says of a map that found no stream, and of an output whose
# optional map found none.
_NO_MATCH = "matches no streams"
_NO_STREAM = "does not contain any stream"


def file_arg(path: str) -> str:
    """``path`` as ffmpeg's file protocol names it, so a name that starts
    with "-" or holds a ":" is read as a path, not an option or a
    protocol."""
    return f"file:{path}"


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
            cmd += ["-i", file_arg(wav)]
        cmd += ["-map", "0:v"] + (["-map", "1:a"] if wav else [])
        cmd += [*VIDEO_ARGS] + ([*AUDIO_ARGS] if wav else [])
        cmd += ["-movflags", "+faststart", "-f", "mp4", file_arg(self.part)]
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


_WRITTEN = re.compile(r"(frame_[0-9]{6}\.png|audio\.wav)(\.part)?")


def clear_frames(folder: str) -> None:
    """Remove the files a clip's frames take in ``folder``: the PNGs of
    :func:`write_frames`, ``audio.wav`` and their part files, so an
    earlier, longer clip leaves none past this one's end. Other files
    stay."""
    for e in os.scandir(folder):
        if _WRITTEN.fullmatch(e.name) and not e.is_dir(follow_symlinks=False):
            os.remove(e.path)


@contextlib.contextmanager
def frames_lock(folder: str, log=None):
    """Hold ``folder`` for one run's frames, so a second run on it waits
    until the first has written its last frame and ``audio.wav`` before
    it clears any of them. ``log`` gets a line when the run waits."""
    fd = os.open(folder, os.O_RDONLY)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            if log is not None:
                log(f"waiting for another run writing frames to {folder}")
            fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        os.close(fd)


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


def _ffmpeg_failed(path: str, err, kind: str) -> FfmpegError:
    err.seek(0)
    text = err.read().decode(errors="replace")
    if _NO_MATCH in text:
        return FfmpegError(f"{path} has no {kind} stream")
    why = text.strip().splitlines()
    return FfmpegError(f"ffmpeg could not read {path}"
                       + (f": {why[-1]}" if why else ""))


def _frames(ffmpeg: str, path: str, most: int, each, scale=None) -> int:
    """Call ``each`` on each of the first ``most`` frames of the video at
    ``path``, (H, W, 3) uint8 as ffmpeg decodes them, scaled to ``scale``
    (width, height) when given. Returns the frame count."""
    cmd = [ffmpeg, "-v", "error", "-nostdin", "-i", file_arg(path), "-map", "0:v:0",
           "-frames:v", str(int(most)), "-fps_mode", "passthrough"]
    if scale is not None:
        cmd += ["-vf", f"scale={scale[0]}:{scale[1]}"]
    cmd += ["-f", "image2pipe", "-c:v", "ppm", "-pix_fmt", "rgb24", "-"]
    n = 0
    with tempfile.TemporaryFile() as err:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=err)
        read = False
        try:
            out = p.stdout
            assert out is not None
            while n < most:
                head = b"".join(out.readline() for _ in range(3))
                if not head:
                    break
                m = _PPM.fullmatch(head)
                if m is None:
                    raise FfmpegError(f"ffmpeg gave no frame header for "
                                      f"{path}")
                w, h = int(m.group(1)), int(m.group(2))
                data = out.read(w * h * 3)
                if len(data) != w * h * 3:
                    raise FfmpegError(f"ffmpeg gave part of a frame of "
                                      f"{path}")
                each(np.frombuffer(data, np.uint8).reshape(h, w, 3))
                n += 1
            read = True
        finally:
            if not read and p.poll() is None:
                p.kill()
            p.stdout.close()
            rc = p.wait()
        if rc:
            raise _ffmpeg_failed(path, err, "video")
    if not n:
        raise FfmpegError(f"ffmpeg found no frames in {path}")
    return n


def guide_frames(path: str, *, width: int, height: int, most: int,
                 ffmpeg: str) -> np.ndarray:
    """The first ``most`` frames of the video at ``path``, or all of
    them, (T, height, width, 3) uint8: each cropped about its center to
    the canvas's aspect and resized as it arrives."""
    from .images import cover

    out: list = []
    _frames(ffmpeg, path, most, lambda f: out.append(cover(f, width,
                                                           height)))
    return np.stack(out)


def each_guide_frame(path: str, *, most: int, ffmpeg: str, each) -> int:
    """Call ``each`` on each of the first ``most`` frames of the video at
    ``path``, (H, W, 3) uint8 as ffmpeg decodes them, through one pipe, so
    a caller keeps only the frames it needs. Returns the frame count."""
    return _frames(ffmpeg, path, most, each)


def video_size(path: str, *, ffmpeg: str) -> tuple[int, int]:
    """The width and height of the first frame of the video at ``path``,
    from the header ffmpeg writes before the frame, which is never read."""
    cmd = [ffmpeg, "-v", "error", "-nostdin", "-i", file_arg(path), "-map", "0:v:0",
           "-frames:v", "1", "-f", "image2pipe", "-c:v", "ppm", "-pix_fmt",
           "rgb24", "-"]
    with tempfile.TemporaryFile() as err:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=err)
        try:
            out = p.stdout
            assert out is not None
            head = b"".join(out.readline() for _ in range(3))
        finally:
            if p.poll() is None:
                p.kill()
            p.stdout.close()
            rc = p.wait()
        m = _PPM.fullmatch(head)
        if m is None:
            if rc > 0:
                raise _ffmpeg_failed(path, err, "video")
            raise FfmpegError(f"ffmpeg found no frames in {path}")
    return int(m.group(1)), int(m.group(2))


def guide_length(path: str, *, most: int, ffmpeg: str) -> int:
    """How many frames ``guide_frames`` reads of ``path`` with ``most``,
    counted from a small scale of each."""
    return _frames(ffmpeg, path, most, lambda f: None, scale=(16, 16))


def _wav(data: bytes, path: str) -> np.ndarray:
    """The float32 samples (channels, n) of a WAV ffmpeg wrote to a
    pipe, whose sizes it leaves unset."""
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise FfmpegError(f"ffmpeg gave no WAV stream for {path}")
    i, channels = 12, 0
    while i + 8 <= len(data):
        tag = data[i:i + 4]
        size = int.from_bytes(data[i + 4:i + 8], "little")
        if tag == b"fmt ":
            channels = int.from_bytes(data[i + 10:i + 12], "little")
        elif tag == b"data" and channels:
            body = data[i + 8:]
            body = body[:len(body) - len(body) % (4 * channels)]
            return np.frombuffer(body, "<f4").reshape(-1, channels).T.copy()
        i += 8 + size + (size & 1)
    raise FfmpegError(f"ffmpeg gave a WAV stream with no samples for "
                      f"{path}")


def _audio(ffmpeg: str, path: str, rate: int, samples: int,
           optional: bool, mix: tuple) -> np.ndarray | None:
    cmd = [ffmpeg, "-v", "error", "-nostdin", "-i", file_arg(path), "-map",
           "0:a:0?" if optional else "0:a:0", *mix, "-af",
           f"aresample={int(rate)},atrim=end_sample={int(samples)}",
           "-c:a", "pcm_f32le", "-f", "wav", "-"]
    with tempfile.TemporaryFile() as err:
        try:
            r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=err,
                               timeout=300)
        except (OSError, subprocess.TimeoutExpired) as e:
            raise FfmpegError(f"ffmpeg could not read {path}: {e}") from None
        if r.returncode:
            err.seek(0)
            if optional and _NO_STREAM in err.read().decode(
                    errors="replace"):
                return None
            raise _ffmpeg_failed(path, err, "audio")
    return _wav(r.stdout, path)


def _samples(ffmpeg: str, path: str, rate: int, channels: int,
             mix: tuple) -> int:
    """The samples of the first audio stream of ``path`` at ``rate`` Hz,
    counted as ffmpeg decodes the whole stream, none of them kept."""
    cmd = [ffmpeg, "-v", "error", "-nostdin", "-i", file_arg(path), "-map", "0:a:0",
           *mix, "-af", f"aresample={int(rate)}", "-c:a", "pcm_f32le",
           "-f", "f32le", "-"]
    n = 0
    with tempfile.TemporaryFile() as err:
        try:
            with subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                  stderr=err) as proc:
                assert proc.stdout is not None
                while chunk := proc.stdout.read(1 << 20):
                    n += len(chunk)
        except OSError as e:
            raise FfmpegError(f"ffmpeg could not read {path}: {e}") from None
        if proc.returncode:
            raise _ffmpeg_failed(path, err, "audio")
    return n // (4 * channels)


def guide_audio(path: str, *, rate: int, samples: int, ffmpeg: str,
                hop: int, optional: bool = False) -> np.ndarray | None:
    """The first audio stream of ``path`` cut to whole latents of ``hop``
    samples about its middle, as ComfyUI's VAE encode cuts it, then to its
    first ``samples`` samples or fewer: (2, n) float32 at ``rate`` Hz, a
    mono stream on both channels as it is, more than two channels mixed
    down to two by ffmpeg. With ``optional``, None for a file with no
    audio stream."""
    mix: tuple = ()
    w = _audio(ffmpeg, path, rate, samples + hop, optional, mix)
    if w is not None and w.shape[0] > 2:
        mix = ("-ac", "2")
        w = _audio(ffmpeg, path, rate, samples + hop, optional, mix)
    if w is None:
        return None
    if not w.shape[1]:
        raise FfmpegError(f"{path} holds no audio samples")
    n = (w.shape[1] if w.shape[1] < samples + hop
         else _samples(ffmpeg, path, rate, w.shape[0], mix))
    if n < hop:
        raise FfmpegError(f"{path} holds {n} audio samples, less than one "
                          f"latent of {hop}")
    at = n % hop // 2
    w = w[:, at:at + min(n - n % hop, samples)]
    return np.repeat(w, 2, axis=0) if w.shape[0] == 1 else w
