"""The ffmpeg and ffprobe that the server runs to decode and encode audio.

mlx-audio and mlx-whisper find these programs on PATH. The server keeps
the PATH of the shell that started it, and that PATH can hold a folder
that a container client changes, such as a project's .venv/bin. So the
server looks for them only in the Homebrew and system folders, which
launch never shares by default.
"""

from __future__ import annotations

import json
import shutil
import subprocess


class ProgramMissing(RuntimeError):
    """ffmpeg or ffprobe is not in the folders where the server looks."""


def folders() -> list[str]:
    """The folders where the server looks for ffmpeg and ffprobe."""
    from gmlx.container import settings

    return settings.SYSTEM_PATH.split(":")


def folders_text() -> str:
    """The folders of :func:`folders` as one phrase, such as "/a, /b or /c"."""
    where = folders()
    return f"{', '.join(where[:-1])} or {where[-1]}" if len(where) > 1 else where[0]


def find(name: str) -> str | None:
    """The path of program ``name`` in :func:`folders`, or None."""
    return shutil.which(name, path=":".join(folders()))


def program(name: str) -> str:
    """The path of program ``name``. Raises :class:`ProgramMissing` when
    it is not in :func:`folders`."""
    path = find(name)
    if path is None:
        raise ProgramMissing(f"{name} is not in {folders_text()}, where the gmlx server "
                             "looks for it. Install it with `brew install ffmpeg`.")
    return path


def _run(argv: list[str], data: bytes | None) -> subprocess.CompletedProcess[bytes]:
    if data is None:
        return subprocess.run(argv, stdin=subprocess.DEVNULL, capture_output=True)
    return subprocess.run(argv, input=data, capture_output=True)


def decode(source) -> tuple:
    """Decode audio bytes or an audio file to 16-bit samples, and give the
    sample rate and the channel count that ffprobe reads. It takes the
    place of mlx-audio's ``_decode_ffmpeg`` and gives the same result."""
    import numpy as np

    ffmpeg, ffprobe = program("ffmpeg"), program("ffprobe")
    data = bytes(source) if isinstance(source, (bytes, bytearray)) else None
    name = "pipe:0" if data is not None else str(source)
    probe = _run([ffprobe, "-v", "quiet", "-print_format", "json", "-show_streams",
                  "-select_streams", "a:0", "-i", name], data)
    if probe.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {probe.stderr.decode(errors='replace')}")
    streams = json.loads(probe.stdout.decode(errors="replace") or "{}").get("streams")
    if not streams:
        raise RuntimeError("No audio streams found in file")
    rate = int(streams[0].get("sample_rate", 44100))
    channels = int(streams[0].get("channels", 2))
    out = _run([ffmpeg, "-v", "error", "-i", name, "-f", "s16le", "-acodec", "pcm_s16le",
                "-ar", str(rate), "-ac", str(channels), "pipe:1"], data)
    if out.returncode != 0:
        raise RuntimeError(f"ffmpeg decoding failed: {out.stderr.decode(errors='replace')}")
    return np.frombuffer(out.stdout, dtype=np.int16), rate, channels


def decode_mono(path: str, rate: int):
    """Decode an audio file to mono float32 samples at ``rate``, as
    mlx-whisper's ``load_audio`` does."""
    import numpy as np

    out = _run([program("ffmpeg"), "-v", "error", "-nostdin", "-threads", "0", "-i", path,
                "-f", "s16le", "-ac", "1", "-acodec", "pcm_s16le", "-ar", str(rate), "-"],
               None)
    if out.returncode != 0:
        raise RuntimeError(f"ffmpeg cannot decode the audio: "
                           f"{out.stderr.decode(errors='replace')}")
    return np.frombuffer(out.stdout, np.int16).astype(np.float32) / 32768.0
