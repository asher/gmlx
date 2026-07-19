"""Speech-to-text for the server: the engine behind ``POST /v1/audio/transcriptions``.

Backed by the optional `mlx-whisper <https://pypi.org/project/mlx-whisper/>`_
package (``pip install 'gmlx[stt]'``). Whisper checkpoints are not GGUFs -
whisper.cpp uses its own ggml container and there is no community supply of
Whisper GGUF quants - so unlike the LLM path this loads MLX-format checkpoints
directly: any mlx-community Whisper repo (fp16 or pre-quantized), any HF repo
in MLX-whisper layout, or a local converted directory. mlx-whisper caches the
loaded model process-wide, so repeat requests don't reload.

This module is import-safe without mlx-whisper installed: the import is lazy
and only :func:`import_mlx_whisper` (called at server startup when STT is
configured, and per-request) touches it.
"""

from __future__ import annotations

import concurrent.futures
import os
import tempfile

from . import subservice
from .hf_cache import offline_resolve
from .subservice import SingleWorker, SubserviceRequestError

# Friendly aliases -> mlx-community MLX-format Whisper repos (all verified to
# exist). `whisper-turbo` is the recommended default: large-v3 quality at ~6x
# the decode speed; the -q4 variant is ~600 MB for memory-tight setups.
STT_ALIASES = {
    "whisper-turbo":    "mlx-community/whisper-large-v3-turbo",
    "whisper-turbo-q4": "mlx-community/whisper-large-v3-turbo-q4",
    "whisper-large":    "mlx-community/whisper-large-v3-mlx",
    "whisper-medium":   "mlx-community/whisper-medium-mlx",
    "whisper-small":    "mlx-community/whisper-small-mlx",
    "whisper-base":     "mlx-community/whisper-base-mlx",
    "whisper-tiny":     "mlx-community/whisper-tiny",
}
DEFAULT_STT_ALIAS = "whisper-turbo"

# Request `model` values that mean "whatever this server has configured".
# OpenAI clients conventionally send "whisper-1".
_CONFIGURED_NAMES = frozenset({"", "whisper-1", "default"})

RESPONSE_FORMATS = ("json", "text", "verbose_json", "srt", "vtt")

# All mlx-whisper work (the load and every transcription) routes through one
# persistent single-worker thread - see subservice.SingleWorker for why
# (per-thread MLX streams; an uncaught cross-thread abort otherwise).
_STT_WORKER = SingleWorker("stt-worker")


class STTRequestError(SubserviceRequestError):
    """A client-side problem with a transcription request (HTTP 4xx)."""


def import_mlx_whisper():
    """Import mlx-whisper, with install guidance when missing."""
    try:
        import mlx_whisper
    except ImportError as exc:
        raise ImportError(
            "speech-to-text requires the optional stt extra:\n"
            "    pip install 'gmlx[stt]'\n"
            "(installs mlx-whisper + python-multipart; audio decoding also "
            "needs ffmpeg on PATH - `brew install ffmpeg`)") from exc
    return mlx_whisper


def _load_stt_model(model_path: str) -> None:
    """Populate mlx-whisper's process-wide model cache for ``model_path``.

    Goes through the same ``ModelHolder.get_model(path, float16)`` cache and
    dtype that :func:`run_transcription` reaches, so the warmed entry is the
    exact one served (no silent reload on the first request). Runs on the STT
    worker thread (via :func:`prewarm` / :func:`run_transcription`) so the
    thread that loads the model is the same one that later transcribes with it;
    the single worker serializes loads, so a request that races the warm waits
    on this load rather than kicking off a second one.
    """
    import mlx.core as mx
    from mlx_whisper.transcribe import ModelHolder

    with offline_resolve(model_path):
        ModelHolder.get_model(model_path, mx.float16)


def prewarm(model_path: str) -> concurrent.futures.Future:
    """Background-load the configured Whisper model at server startup
    (best-effort; see :func:`subservice.prewarm`)."""
    def _load():
        import_mlx_whisper()              # install guidance if the extra is gone
        _load_stt_model(model_path)

    return subservice.prewarm(_STT_WORKER, _load, "stt")


def resolve_stt_model(value) -> str:
    """Normalize a configured STT model value to an HF repo id or local path.

    Accepts a friendly alias (``whisper-turbo``), an HF repo id
    (``mlx-community/whisper-large-v3-turbo`` - e.g. the fp16 repo when you
    don't want a q4), a local model directory, or ``True``/``"default"`` (YAML
    ``stt: true`` / bare ``--stt``) for the default alias.
    """
    return subservice.resolve_alias_or_path(
        value, aliases=STT_ALIASES, default_alias=DEFAULT_STT_ALIAS,
        default_names=("default", "true", "whisper-1"))


def effective_model(requested: str, configured: str) -> str:
    """Map a request's ``model`` field onto the configured STT model
    (conventional names pass; anything else is a 400)."""
    return subservice.effective_model(
        requested, configured, accepted_names=_CONFIGURED_NAMES,
        resolver=resolve_stt_model, error_cls=STTRequestError,
        kind="STT", hint="whisper-1")


# Response formatting (OpenAI shapes)

def _ts(seconds: float, *, sep: str) -> str:
    ms = max(0, int(round(float(seconds) * 1000)))
    h, rem = divmod(ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def _srt(segments: list) -> str:
    out = []
    for i, seg in enumerate(segments, 1):
        out.append(f"{i}\n{_ts(seg['start'], sep=',')} --> "
                   f"{_ts(seg['end'], sep=',')}\n{seg['text'].strip()}\n")
    return "\n".join(out)


def _vtt(segments: list) -> str:
    out = ["WEBVTT\n"]
    for seg in segments:
        out.append(f"{_ts(seg['start'], sep='.')} --> "
                   f"{_ts(seg['end'], sep='.')}\n{seg['text'].strip()}\n")
    return "\n".join(out)


_VERBOSE_SEGMENT_KEYS = ("id", "seek", "start", "end", "text", "tokens",
                         "temperature", "avg_logprob", "compression_ratio",
                         "no_speech_prob")


def format_result(result: dict, response_format: str, *, task: str = "transcribe"):
    """Shape an mlx-whisper result dict into ``(content, media_type)`` per the
    OpenAI ``response_format`` values: json | text | verbose_json | srt | vtt.
    ``task`` ("transcribe" or "translate") is echoed in the verbose_json body."""
    text = (result.get("text") or "").strip()
    segments = result.get("segments") or []
    if response_format == "text":
        return text + "\n", "text/plain"
    if response_format == "srt":
        return _srt(segments), "text/plain"
    if response_format == "vtt":
        return _vtt(segments), "text/plain"
    if response_format == "verbose_json":
        duration = float(segments[-1]["end"]) if segments else 0.0
        return {
            "task": task,
            "language": result.get("language"),
            "duration": duration,
            "text": text,
            "segments": [{k: seg[k] for k in _VERBOSE_SEGMENT_KEYS if k in seg}
                         for seg in segments],
        }, "application/json"
    return {"text": text}, "application/json"


# Request handling core (sync; the HTTP layer calls this in a threadpool)

def run_transcription(audio_bytes: bytes, *, filename: str, configured_model: str,
                      model: str = "", language: str = "", prompt: str = "",
                      temperature: str = "", response_format: str = "",
                      task: str = "transcribe"):
    """Validate fields, transcribe (or translate) ``audio_bytes``, and return
    ``(content, media_type)``. Raises :class:`STTRequestError` for 4xx problems.

    ``task`` is Whisper's own task: "transcribe" (same-language text) or
    "translate" (any-language audio -> English text, the /v1/audio/translations
    endpoint). All field params arrive as the raw form strings; this owns parsing.
    """
    if not audio_bytes:
        raise STTRequestError(400, "multipart field 'file' is required")
    target = effective_model(model, configured_model)
    fmt = (response_format or "json").strip().lower()
    if fmt not in RESPONSE_FORMATS:
        raise STTRequestError(
            400, f"unsupported response_format {fmt!r} "
                 f"(supported: {', '.join(RESPONSE_FORMATS)})")
    try:
        temp = float(temperature) if str(temperature).strip() else 0.0
    except ValueError:
        raise STTRequestError(400, f"temperature {temperature!r} is not a number")

    mw = import_mlx_whisper()
    decode_options = {}
    if task and task != "transcribe":
        decode_options["task"] = task        # Whisper's built-in translate task
    if (language or "").strip():
        decode_options["language"] = language.strip()
    if (prompt or "").strip():
        decode_options["initial_prompt"] = prompt

    suffix = os.path.splitext(filename or "")[1] or ".wav"
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    try:
        tmp.write(audio_bytes)
        tmp.close()

        def _job():
            # Load (if needed) and transcribe on the same dedicated thread -
            # see subservice.SingleWorker. transcribe() returns a plain dict of host
            # values, so nothing GPU-bound crosses back to the caller. The
            # offline_resolve window keeps mlx-whisper's own snapshot_download
            # cache-only (no Hub round-trip per request) once the repo is local.
            with offline_resolve(target):
                return mw.transcribe(tmp.name, path_or_hf_repo=target,
                                     temperature=temp, **decode_options)

        result = _STT_WORKER.submit(_job).result()
    except Exception as exc:
        hint = (" (audio decoding needs ffmpeg on PATH - `brew install ffmpeg`)"
                if "ffmpeg" in str(exc).lower() else "")
        raise RuntimeError(f"transcription failed: {exc}{hint}") from exc
    finally:
        tmp.close()
        try:
            os.unlink(tmp.name)
        except OSError:
            pass
    return format_result(result, fmt, task=task)
