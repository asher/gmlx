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

A second backend serves a speech companion GGUF (Granite Speech TurboCTC,
:mod:`gmlx.serve.asr_service`): ``--stt`` names its path, or the alias
``granite-speech`` for the companion a configured model pairs with. It needs
no mlx-whisper, and gives the ``json`` and ``text`` formats only.
"""

from __future__ import annotations

import concurrent.futures
import os
import sys
import tempfile

from . import media_programs, subservice
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
ALIAS_COMPANION = "granite-speech"

# Request `model` values that mean "whatever this server has configured".
# OpenAI clients conventionally send "whisper-1".
_CONFIGURED_NAMES = frozenset({"", "whisper-1", "default"})

RESPONSE_FORMATS = ("json", "text", "verbose_json", "srt", "vtt")
# A CTC transcript has no segments, timestamps or language.
COMPANION_FORMATS = ("json", "text")

# The sample rate of Whisper's input, mlx_whisper.audio.SAMPLE_RATE.
WHISPER_SAMPLE_RATE = 16000

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
        from gmlx.commands.extras import install_hint
        raise ImportError(
            "speech-to-text requires the optional stt extra:\n"
            f"    {install_hint('stt')}\n"
            "The extra installs mlx-whisper and python-multipart. Audio decoding also "
            "needs ffmpeg on the PATH of the server. Install it with `brew install ffmpeg`."
        ) from exc
    return mlx_whisper


def is_companion(model: str | None) -> bool:
    """True when ``model`` is a speech companion GGUF."""
    if not model or not str(model).endswith(".gguf"):
        return False
    from gmlx.load import arch_table
    from gmlx.load.discovery import header_meta

    meta = header_meta(model)
    return bool(meta) and meta.get("arch") in arch_table.ASR_ARCHES


def check_backend(model: str) -> None:
    """Import what serving ``model`` needs, with install guidance when it
    is missing. A companion needs only the form parser."""
    if not is_companion(model):
        import_mlx_whisper()
        return
    try:
        import python_multipart  # noqa: F401
    except ImportError:
        try:
            import multipart  # noqa: F401
        except ImportError as exc:
            from gmlx.commands.extras import install_hint
            raise ImportError(
                "speech-to-text requires the optional stt extra:\n"
                f"    {install_hint('stt')}") from exc


def paired_companion(cfg) -> str:
    """The companion GGUF a model of ``cfg`` pairs with, for the
    ``granite-speech`` alias. Raises ConfigError when none does."""
    from gmlx import asr
    from gmlx.config import ConfigError, resolve_model

    for mid in (getattr(cfg, "models", None) or {}):
        try:
            rm = resolve_model(mid, cfg)
        except Exception:  # noqa: BLE001  a skipped model pairs nothing
            continue
        path = asr.companion_path(rm.path, rm.asr, rm.no_asr)
        if path:
            return path
    raise ConfigError(
        f"server.stt: {ALIAS_COMPANION!r} names the speech companion of a "
        "configured model, and no model pairs with one. Name the companion "
        "GGUF path instead.")


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
    if is_companion(model_path):
        from . import asr_service
        return asr_service.prewarm(model_path)

    def _load():
        import_mlx_whisper()              # install guidance if the extra is gone
        _load_stt_model(model_path)

    return subservice.prewarm(_STT_WORKER, _load, "stt")


def release() -> concurrent.futures.Future:
    """Drop the cached Whisper model (see :func:`subservice.release`) and
    any loaded speech companion."""
    from . import asr_service
    asr_service.release()
    transcribe = sys.modules.get("mlx_whisper.transcribe")
    holders = (transcribe.ModelHolder,) if transcribe is not None else ()
    return subservice.release(_STT_WORKER, *holders)


def resolve_stt_model(value, cfg=None) -> str:
    """Normalize a configured STT model value to an HF repo id or local path.

    Accepts a friendly alias (``whisper-turbo``), an HF repo id
    (``mlx-community/whisper-large-v3-turbo`` - e.g. the fp16 repo when you
    don't want a q4), a local model directory, or ``True``/``"default"`` (YAML
    ``stt: true`` / bare ``--stt``) for the default alias. A speech companion
    GGUF path resolves to its absolute path, and ``granite-speech`` to the
    companion a model of ``cfg`` pairs with.
    """
    if isinstance(value, str):
        v = value.strip()
        if v.lower() == ALIAS_COMPANION:
            return paired_companion(cfg)
        expanded = os.path.abspath(os.path.expanduser(v))
        if is_companion(expanded):
            return expanded
    return subservice.resolve_alias_or_path(
        value, aliases=STT_ALIASES, default_alias=DEFAULT_STT_ALIAS,
        default_names=("default", "true", "whisper-1"))


def effective_model(requested: str, configured: str) -> str:
    """Map a request's ``model`` field onto the configured STT model
    (conventional names pass; anything else is a 400)."""
    accepted = _CONFIGURED_NAMES
    if is_companion(configured):
        accepted = accepted | {ALIAS_COMPANION}
    return subservice.effective_model(
        requested, configured, accepted_names=accepted,
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
    if is_companion(target):
        return _run_companion(audio_bytes, target, fmt, task)
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
            # The server decodes the upload itself, because mlx-whisper runs
            # the ffmpeg that it finds on PATH, and stops a decode that passes
            # the limit for one clip.
            from .patches.media_gate import MediaRefused

            try:
                audio = media_programs.decode_mono(tmp.name, WHISPER_SAMPLE_RATE,
                                                   field="the uploaded file")
            except MediaRefused as e:
                raise STTRequestError(400, str(e)) from None
            with offline_resolve(target):
                return mw.transcribe(audio, path_or_hf_repo=target,
                                     temperature=temp, **decode_options)

        result = _STT_WORKER.submit(_job).result()
    except STTRequestError:
        raise
    except Exception as exc:
        raise RuntimeError(f"transcription failed: {exc}") from exc
    finally:
        tmp.close()
        try:
            os.unlink(tmp.name)
        except OSError:
            pass
    return format_result(result, fmt, task=task)


def _run_companion(audio_bytes: bytes, target: str, fmt: str, task: str):
    """Transcribe with the speech companion. It has no translate task and
    no segments, and ignores ``language``, ``prompt`` and ``temperature``."""
    from . import asr_service

    name = os.path.basename(target)
    if task != "transcribe":
        raise STTRequestError(
            400, f"the speech-to-text model {name} transcribes only; it has no "
                 "translations")
    if fmt not in COMPANION_FORMATS:
        raise STTRequestError(
            400, f"response_format {fmt!r} is not available with {name}, which "
                 f"gives text only (supported: {', '.join(COMPANION_FORMATS)})")
    try:
        text = asr_service.transcribe(target, audio_bytes, field="the uploaded file")
    except asr_service.ASRRequestError as exc:
        raise STTRequestError(exc.status_code, str(exc)) from None
    except Exception as exc:
        raise RuntimeError(f"transcription failed: {exc}") from exc
    return format_result({"text": text}, fmt, task=task)
