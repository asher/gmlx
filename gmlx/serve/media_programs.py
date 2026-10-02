"""The ffmpeg and ffprobe that the server runs to decode and encode audio.

mlx-audio and mlx-whisper run the ffmpeg that they find on PATH. The
server's PATH can hold a folder that a container client changes, such as a
project's .venv/bin. So the server finds both programs through
:mod:`gmlx.serve.programs`, which skips such folders and refuses a program
that leads into one.
"""

from __future__ import annotations

import json
import subprocess

from . import programs
from .programs import ProgramMissing, ProgramRefused

__all__ = ["ProgramMissing", "ProgramRefused", "decode", "decode_mono", "find", "log_programs",
           "problem", "program"]

# The next step when the server cannot run ffmpeg or ffprobe. Homebrew's
# ffmpeg formula installs both.
_STEP = ("Install it with `brew install ffmpeg`, or start the server from a shell whose PATH "
         "holds your ffmpeg.")
_REFUSED_STEP = ("Remove that file, so that the server finds another one, or install ffmpeg "
                 "with `brew install ffmpeg`.")


def _step(lookup: programs.Lookup) -> str:
    return _REFUSED_STEP if lookup.refusal is not None else _STEP


def find(name: str) -> str | None:
    """The path of program ``name`` that the server runs, or None."""
    lookup = programs.look_up(name)
    return lookup.path if lookup.refusal is None else None


def problem(name: str) -> str | None:
    """Why the server cannot run program ``name``, with the next step, or
    None when it can."""
    lookup = programs.look_up(name)
    return programs.problem(lookup, _step(lookup))


def program(name: str) -> str:
    """The path of program ``name``. Raises :class:`ProgramMissing` or
    :class:`ProgramRefused` when the server cannot run it. The server log
    names the program, and each PATH entry that the search skips."""
    lookup = programs.look_up(name)
    step = _step(lookup)
    programs.log(lookup, step)
    return programs.checked(lookup, step).path or ""


def log_programs() -> None:
    """Write to the server log which ffmpeg and ffprobe the server runs."""
    for name in ("ffmpeg", "ffprobe"):
        lookup = programs.look_up(name)
        programs.log(lookup, _step(lookup))


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
