"""The ffmpeg and ffprobe that the server runs to decode and encode audio.

mlx-audio and mlx-whisper run the ffmpeg that they find on PATH. The
server's PATH can hold a folder that a container client changes, such as a
project's .venv/bin. So the server finds both programs through
:mod:`gmlx.serve.programs`, which skips such folders and refuses a program
that leads into one.

A small audio file can decode to hours of samples, so the decoders read
ffmpeg's output in parts and stop ffmpeg at
:data:`~gmlx.serve.patches.media_gate.AUDIO_MAX_SAMPLES` samples.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import subprocess
import threading

from . import programs
from .programs import ProgramMissing, ProgramRefused

__all__ = ["ProgramMissing", "ProgramRefused", "decode", "decode_mono", "find", "log_programs",
           "problem", "program"]

# The next step when the server cannot run ffmpeg or ffprobe. Homebrew's
# ffmpeg formula installs both.
_STEP = ("Install it with `brew install ffmpeg`, or start the server from a shell whose PATH "
         "holds your ffmpeg.")
# The refused program comes first in the search, so an install elsewhere does
# not help until that file is gone.
_REFUSED_STEP = "Remove that file, so that the server looks for another ffmpeg."


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


def log_programs(cfg) -> None:
    """Write to the server log which ffmpeg and ffprobe the server runs.
    The line of a missing program gives the install step only when ``cfg``
    serves transcription or speech. Otherwise the first request that needs
    the program writes the step."""
    advise = bool(getattr(cfg, "stt", None) or getattr(cfg, "tts", None))
    for name in ("ffmpeg", "ffprobe"):
        lookup = programs.look_up(name)
        programs.log(lookup, _step(lookup) if advise else "")


def _run(argv: list[str], data: bytes | None) -> subprocess.CompletedProcess[bytes]:
    if data is None:
        return subprocess.run(argv, stdin=subprocess.DEVNULL, capture_output=True)
    return subprocess.run(argv, input=data, capture_output=True)


# The most of a decoder's error output that the server keeps.
_STDERR_KEEP = 64 << 10


def _run_bounded(argv: list[str], data: bytes | None,
                 limit: int) -> tuple[int, bytearray, bytes] | None:
    """Run ``argv`` with ``data`` on its input, and give its exit code, its
    output and the start of its error output. When the output passes
    ``limit`` bytes, the program is stopped and the result is None, so a
    small input that decodes to a large output never fills the memory."""
    proc = subprocess.Popen(
        argv, stdin=subprocess.DEVNULL if data is None else subprocess.PIPE,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    errors = bytearray()

    def feed() -> None:
        try:
            proc.stdin.write(data)
        except OSError:            # the decoder stopped reading, or was stopped
            pass
        finally:
            with contextlib.suppress(OSError):
                proc.stdin.close()

    def drain() -> None:
        while chunk := os.read(proc.stderr.fileno(), _STDERR_KEEP):
            errors.extend(chunk[:max(0, _STDERR_KEEP - len(errors))])

    helpers = [threading.Thread(target=drain, daemon=True)]
    if data is not None:
        helpers.append(threading.Thread(target=feed, daemon=True))
    for t in helpers:
        t.start()
    out = bytearray()
    over = False
    try:
        while chunk := os.read(proc.stdout.fileno(), 1 << 20):
            out += chunk
            if len(out) > limit:
                over = True
                break
    except BaseException:
        proc.kill()
        raise
    finally:
        if over:
            proc.kill()
        proc.wait()
        for t in helpers:
            t.join()
        proc.stdout.close()
        proc.stderr.close()
    if over:
        return None
    return proc.returncode, out, bytes(errors)


def _limit_seconds(rate: int, channels: int) -> str:
    """ffmpeg's ``-t`` for a decode at ``rate`` Hz with ``channels``
    channels: one second past the limit, so that ffmpeg ends a long clip by
    itself and the output still shows that the clip passed the limit."""
    from .patches import media_gate as mg

    return str(math.ceil(mg.AUDIO_MAX_SAMPLES / (rate * channels)) + 1)


def decode(source, field: str = "audio") -> tuple:
    """Decode audio bytes or an audio file to 16-bit samples, and give the
    sample rate and the channel count that ffprobe reads. It takes the
    place of mlx-audio's ``_decode_ffmpeg`` and gives the same result.
    Audio over :data:`~.patches.media_gate.AUDIO_MAX_SAMPLES` samples, or at
    a sample rate over :data:`~.patches.media_gate.AUDIO_MAX_RATE`, is
    refused, and the decode stops at that count."""
    import numpy as np

    from .patches import media_gate as mg

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
    if not 0 < rate <= mg.AUDIO_MAX_RATE:
        raise mg.audio_rate_refusal(field, rate)
    if channels <= 0:
        raise RuntimeError(f"ffprobe reads {channels} channels")
    # The duration in the header refuses a long clip before the decode. It
    # can be wrong, so the decode stops at the limit too.
    try:
        seconds = float(streams[0].get("duration", "nan"))
    except (TypeError, ValueError):
        seconds = math.nan
    if seconds * rate * channels > mg.AUDIO_MAX_SAMPLES:
        raise mg.audio_too_long(field, rate, channels)
    out = _run_bounded([ffmpeg, "-v", "error", "-i", name, "-t", _limit_seconds(rate, channels),
                        "-f", "s16le", "-acodec", "pcm_s16le", "-ar", str(rate),
                        "-ac", str(channels), "pipe:1"],
                       data, 2 * mg.AUDIO_MAX_SAMPLES)
    if out is None:
        raise mg.audio_too_long(field, rate, channels)
    code, pcm, errors = out
    if code != 0:
        raise RuntimeError(f"ffmpeg decoding failed: {errors.decode(errors='replace')}")
    return np.frombuffer(pcm, dtype=np.int16), rate, channels


def decode_mono(path: str, rate: int, field: str = "audio"):
    """Decode an audio file to mono float32 samples at ``rate``, as
    mlx-whisper's ``load_audio`` does. Audio over
    :data:`~.patches.media_gate.AUDIO_MAX_SAMPLES` samples is refused, and
    the decode stops at that count."""
    import numpy as np

    from .patches import media_gate as mg

    out = _run_bounded([program("ffmpeg"), "-v", "error", "-nostdin", "-threads", "0",
                        "-i", path, "-t", _limit_seconds(rate, 1), "-f", "s16le",
                        "-ac", "1", "-acodec", "pcm_s16le", "-ar", str(rate), "-"],
                       None, 2 * mg.AUDIO_MAX_SAMPLES)
    if out is None:
        raise mg.audio_too_long(field, rate, 1, "Split the recording, and send each "
                                                "part in its own request.")
    code, pcm, errors = out
    if code != 0:
        raise RuntimeError(f"ffmpeg cannot decode the audio: "
                           f"{errors.decode(errors='replace')}")
    return np.frombuffer(pcm, np.int16).astype(np.float32) / 32768.0
