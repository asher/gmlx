"""Text-to-speech for the server: the engine behind ``POST /v1/audio/speech``.

Backed by the optional `mlx-audio <https://pypi.org/project/mlx-audio/>`_
package (``pip install 'gmlx[tts]'``). TTS checkpoints are not GGUFs -
the community ships them as MLX-format (or safetensors -> MLX) repos - so like
the STT path (:mod:`gmlx.stt`) this loads MLX-format checkpoints directly:
any mlx-community TTS repo (Kokoro, Qwen3-TTS, ...) or a local converted dir.
The loaded model is cached process-wide, so repeat requests don't reload.

This module is import-safe without mlx-audio installed: the import is lazy and
only :func:`import_mlx_audio` (called at server startup when TTS is configured,
and per-request) touches it. Audio is encoded through mlx-audio's own writer
(WAV via miniaudio; mp3/flac/opus via ffmpeg) - ``pcm`` is emitted directly.
"""

from __future__ import annotations

import concurrent.futures
import io
import os
import re
import sys
import unicodedata

import numpy as np

from . import subservice
from .hf_cache import offline_resolve
from .subservice import SingleWorker, SubserviceRequestError

# Friendly aliases -> mlx-community MLX-format TTS repos. `kokoro` is the
# recommended default: 82M params, ~24 kHz, Apache-2.0, 54 voice presets.
# Qwen3-TTS is larger and multilingual with named voices (e.g. "Chelsie").
TTS_ALIASES = {
    "kokoro":          "mlx-community/Kokoro-82M-bf16",
    "kokoro-8bit":     "mlx-community/Kokoro-82M-8bit",
    "kokoro-4bit":     "mlx-community/Kokoro-82M-4bit",
    "qwen3-tts":       "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit",
    "qwen3-tts-small": "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16",
}
DEFAULT_TTS_ALIAS = "kokoro"

# Default Kokoro voice when a request omits `voice` (American-English female).
DEFAULT_VOICE = "af_heart"

# Request `model` values that mean "whatever this server has configured".
# OpenAI clients conventionally send "tts-1" / "tts-1-hd" / "gpt-4o-mini-tts".
_CONFIGURED_NAMES = frozenset(
    {"", "tts-1", "tts-1-hd", "gpt-4o-mini-tts", "default"})

# OpenAI `response_format` values we serve. wav is dependency-free (miniaudio);
# mp3/flac/opus go through mlx-audio's ffmpeg writer; pcm is raw 16-bit LE.
RESPONSE_FORMATS = ("mp3", "wav", "flac", "opus", "pcm")
_MEDIA_TYPES = {
    "mp3":  "audio/mpeg",
    "wav":  "audio/wav",
    "flac": "audio/flac",
    "opus": "audio/opus",
    "pcm":  "audio/pcm",
}

# OpenAI clamps speed to [0.25, 4.0].
_SPEED_MIN, _SPEED_MAX = 0.25, 4.0

# Qwen3-TTS has no voices/ directory; it takes named speakers. Best-effort
# known set (standard + dialect voices) for the aliases we curate.
QWEN3_TTS_VOICES = ("Chelsie", "Cherry", "Ethan", "Serena",
                    "Dylan", "Jada", "Sunny")
_VOICE_FILE_EXTS = (".safetensors", ".pt", ".npz", ".bin")

# All mlx-audio work (the load and every synthesis) routes through one
# persistent single-worker thread - see subservice.SingleWorker for why
# (per-thread MLX streams; an uncaught cross-thread abort otherwise).
_TTS_WORKER = SingleWorker("tts-worker")


class TTSRequestError(SubserviceRequestError):
    """A client-side problem with a speech request (HTTP 4xx)."""


def import_mlx_audio():
    """Import mlx-audio, with install guidance when missing."""
    try:
        import mlx_audio
    except ImportError as exc:
        raise ImportError(
            "text-to-speech requires the optional tts extra:\n"
            "    pip install 'gmlx[tts]'\n"
            "(installs mlx-audio; non-wav formats also need ffmpeg on "
            "PATH - `brew install ffmpeg`)") from exc
    return mlx_audio


# espeak-ng copies espeak_Initialize's data path into a fixed path_home buffer
# and silently keeps its compiled-in build path when the argument is longer
# than N_PATH_HOME-13 (146) chars - with the loader wheel that build path is
# the wheel CI's machine, and the first out-of-dictionary word then hard-exits
# the process. Deep venv paths (CI runners, tmpdirs) hit this.
_ESPEAK_PATH_MAX = 140


def _espeak_data_path(data_path) -> str:
    """The loader's data dir, mirrored into a short ~/.cache copy when the
    real path is too long for espeak's path_home buffer. A real copy, not a
    symlink: phonemizer's ``data_path`` property ``resolve()``s the value, so
    a symlink hands espeak the long path again. The mirror carries a stamp
    file (source path + mtime) and re-syncs when the loader wheel changes."""
    p = os.path.realpath(str(data_path))
    if len(p) <= _ESPEAK_PATH_MAX:
        return p
    mirror = os.path.expanduser("~/.cache/gmlx/espeak-ng-data")
    if len(os.path.realpath(mirror)) > _ESPEAK_PATH_MAX:
        return p                              # nothing shorter to offer
    stamp = mirror + ".src"
    want = f"{p} {int(os.stat(p).st_mtime)}"
    import shutil
    try:
        if os.path.islink(mirror):            # earlier releases symlinked
            os.unlink(mirror)
        with open(stamp) as f:
            fresh = f.read() == want
    except OSError:
        fresh = False
    try:
        if not (fresh and os.path.isdir(mirror)):
            tmp = mirror + ".tmp"
            shutil.rmtree(tmp, ignore_errors=True)
            shutil.copytree(p, tmp)
            shutil.rmtree(mirror, ignore_errors=True)
            os.rename(tmp, mirror)
            with open(stamp, "w") as f:
                f.write(want)
        return mirror
    except OSError:
        return p


def _ensure_misaki() -> None:
    """Make ``import misaki`` resolve: an installed distribution wins, else
    the vendored snapshot (see ``_vendor/__init__.py``) registers under the
    top-level name so mlx-audio's Kokoro pipeline finds it unchanged."""
    import importlib.util
    if "misaki" in sys.modules or importlib.util.find_spec("misaki"):
        return
    from ._vendor import misaki
    sys.modules["misaki"] = misaki


def _configure_espeak() -> None:
    """Point phonemizer at espeakng-loader's bundled libespeak-ng.

    misaki's English G2P needs espeak-ng as its out-of-dictionary fallback
    (without it OOD words crash the Kokoro pipeline), but only probes a
    hardcoded, version-pinned Homebrew path on macOS. The pip
    ``espeakng-loader`` wheel ships the library + data; wire it in before
    misaki's own probe runs (misaki keeps an already-set library). No-op
    when either package is missing or a library is already configured."""
    _ensure_misaki()
    try:
        import espeakng_loader
        from phonemizer.backend.espeak.wrapper import EspeakWrapper
    except ImportError:
        return
    lib = str(espeakng_loader.get_library_path())
    if EspeakWrapper._ESPEAK_LIBRARY and \
            str(EspeakWrapper._ESPEAK_LIBRARY) != lib:
        return                       # a different espeak was configured; keep it
    # misaki.espeak (re)wires both paths at import - the data path as the raw,
    # possibly-too-long one. Import it now so the settings below are final.
    try:
        import misaki.espeak  # noqa: F401
    except Exception:
        pass
    EspeakWrapper.set_library(lib)
    data = _espeak_data_path(espeakng_loader.get_data_path())
    if hasattr(EspeakWrapper, "set_data_path"):
        EspeakWrapper.set_data_path(data)
    else:
        os.environ.setdefault("ESPEAK_DATA_PATH", data)


def _patch_kokoro_sine_length() -> None:
    """Length-guard mlx-audio's Kokoro sine source (<=0.4.4).

    ``SineGen._f02sine`` round-trips the F0 track through ``interpolate``
    with ``scale_factor=1/upsample_scale`` then ``upsample_scale``; the
    reciprocal is inexact in fp, so certain lengths come back one frame
    long (e.g. 33600 -> 33900) and broadcast-fail against the uv mask,
    500ing synthesis for value-dependent utterances. Trim or edge-pad the
    sine output back to the input length. Idempotent."""
    try:
        from mlx_audio.tts.models.kokoro import istftnet
    except ImportError:
        return
    if getattr(istftnet.SineGen, "_kq_len_guard", False):
        return
    import mlx.core as mx

    orig = istftnet.SineGen._f02sine

    def f02sine(self, f0_values):
        sines = orig(self, f0_values)
        n = f0_values.shape[1]
        if sines.shape[1] > n:
            sines = sines[:, :n, :]
        elif sines.shape[1] < n:
            b, cur, d = sines.shape
            tail = mx.broadcast_to(sines[:, -1:, :], (b, n - cur, d))
            sines = mx.concatenate([sines, tail], axis=1)
        return sines

    istftnet.SineGen._f02sine = f02sine
    istftnet.SineGen._kq_len_guard = True


class _TTSModelHolder:
    """Process-wide single-model cache (mlx-audio's ``load_model`` has none of
    its own). Mirrors mlx-whisper's ModelHolder: reload only on a path change."""

    model = None
    model_path = None

    @classmethod
    def get(cls, model_path: str):
        if cls.model is None or model_path != cls.model_path:
            from mlx_audio.tts.utils import load_model
            _configure_espeak()
            _patch_kokoro_sine_length()
            with offline_resolve(model_path):
                cls.model = load_model(model_path)
            cls.model_path = model_path
        return cls.model


def _load_tts_model(model_path: str) -> None:
    """Populate the model cache for ``model_path``. Runs on the TTS worker
    thread (via :func:`prewarm` / :func:`run_synthesis`) so the thread that
    loads the model is the same one that later synthesizes with it. The single
    worker serializes loads, so a request that races a background warm waits on
    this load instead of kicking off a second one."""
    _TTSModelHolder.get(model_path)


def prewarm(model_path: str) -> concurrent.futures.Future:
    """Background-load the configured TTS model at server startup
    (best-effort; see :func:`subservice.prewarm`)."""
    def _load():
        import_mlx_audio()                # install guidance if the extra is gone
        _load_tts_model(model_path)

    return subservice.prewarm(_TTS_WORKER, _load, "tts")


def resolve_tts_model(value) -> str:
    """Normalize a configured TTS model value to an HF repo id or local path.

    Accepts a friendly alias (``kokoro``), an HF repo id
    (``mlx-community/Kokoro-82M-bf16``), a local model directory, or
    ``True``/``"default"`` (YAML ``tts: true`` / bare ``--tts``) for the
    default alias.
    """
    return subservice.resolve_alias_or_path(
        value, aliases=TTS_ALIASES, default_alias=DEFAULT_TTS_ALIAS,
        default_names=("default", "true", "tts-1", "tts-1-hd"))


def effective_model(requested: str, configured: str) -> str:
    """Map a request's ``model`` field onto the configured TTS model
    (conventional names pass; anything else is a 400)."""
    return subservice.effective_model(
        requested, configured, accepted_names=_CONFIGURED_NAMES,
        resolver=resolve_tts_model, error_cls=TTSRequestError,
        kind="TTS", hint="tts-1")


def available_voices(model_path: str) -> list:
    """Best-effort voice names for a resolved TTS model (``GET
    /v1/audio/voices``; the talk client's ``/voice`` completion). Kokoro-style
    repos enumerate their ``voices/`` directory (54 presets) from the local
    dir or HF cache - never the network; qwen3-tts models return the known
    named-speaker set; anything else returns []."""
    local = model_path if os.path.isdir(model_path) else None
    if local is None:
        try:
            from huggingface_hub import snapshot_download
            local = snapshot_download(model_path, local_files_only=True)
        except Exception:                 # noqa: BLE001 - not cached / no hub
            local = None
    if local:
        vdir = os.path.join(local, "voices")
        if os.path.isdir(vdir):
            names = sorted({os.path.splitext(f)[0] for f in os.listdir(vdir)
                            if f.endswith(_VOICE_FILE_EXTS)})
            if names:
                return names
    if "qwen3-tts" in model_path.lower():
        return list(QWEN3_TTS_VOICES)
    return []


# Audio encoding (OpenAI response_format -> bytes)

def _to_int16_pcm(audio: np.ndarray) -> bytes:
    """Float waveform in [-1, 1] -> raw little-endian 16-bit PCM bytes."""
    clipped = np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0)
    return (clipped * 32767.0).astype("<i2").tobytes()


def encode_audio(audio: np.ndarray, sample_rate: int, fmt: str) -> bytes:
    """Encode a float waveform to ``fmt`` bytes. ``pcm`` is emitted directly;
    everything else goes through mlx-audio's writer (wav: miniaudio; the rest:
    ffmpeg)."""
    if fmt == "pcm":
        return _to_int16_pcm(audio)
    from mlx_audio import audio_io
    writer = getattr(audio_io, "audio_write", None) or audio_io.write
    buf = io.BytesIO()
    writer(buf, np.asarray(audio, dtype=np.float32), sample_rate, format=fmt)
    return buf.getvalue()


# Markdown structure that survives NFKC; stripped before synthesis.
_MD_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_MD_HEADING = re.compile(r"^[ \t]{0,3}#{1,6}[ \t]+", re.MULTILINE)
_MD_QUOTE = re.compile(r"^[ \t]*>[ \t]?", re.MULTILINE)
_MD_BULLET = re.compile(r"^[ \t]*[-*+•][ \t]+", re.MULTILINE)
_WS_RUN = re.compile(r"[ \t]+")

# Sanitizer character policy. Letters, digits, and combining marks of every
# script stay (CJK / Cyrillic / Indic / accented Latin are speakable; the
# per-language G2P decides what it can say). Only genuinely unspeakable
# categories go: emoji and other symbols (So/Sk), control and format chars
# (C*, minus whitespace). Currency and math symbols (Sc/Sm) are speakable.
_DROP_CATEGORIES = frozenset({"So", "Sk", "Cc", "Cf", "Co", "Cn", "Cs"})
_MD_CHARS = frozenset("*`~#|_")
_DASHES = frozenset("–—―")  # en/em/horizontal-bar -> pause


def sanitize_speech_text(text: str) -> str:
    """Reduce ``text`` to what a TTS front-end can actually speak.

    LLM replies carry markdown and symbols that either crash or garble
    grapheme->phoneme front-ends; requests quote the model output verbatim.
    Strips markdown structure, folds compatibility forms (NFKC: fullwidth
    forms, vulgar fractions, ligatures), drops emoji/control characters, and
    turns dashes into spoken pauses - script-aware, never dropping letters.
    """
    text = unicodedata.normalize("NFKC", text)
    text = _MD_LINK.sub(r"\1", text)
    text = _MD_HEADING.sub("", text)
    text = _MD_QUOTE.sub("", text)
    text = _MD_BULLET.sub("", text)
    out = []
    for ch in text:
        if ch in _MD_CHARS:
            out.append(" ")
        elif ch in _DASHES:
            out.append(", ")
        elif ch == "⁄":                   # fraction slash (NFKC's 1/2)
            out.append("/")
        elif ch in ("\n", "\t"):
            out.append(ch)
        elif unicodedata.category(ch) in _DROP_CATEGORIES:
            out.append(" ")
        else:
            out.append(ch)
    s = _WS_RUN.sub(" ", "".join(out))
    s = re.sub(r" ?(, )+", ", ", s)            # collapse stacked pauses
    s = re.sub(r" +\n", "\n", s)
    s = re.sub(r"\n{2,}", "\n", s)
    return s.strip()


def _synthesize(model_obj, text: str, voice: str, speed: float):
    """Run mlx-audio generation and return ``(waveform, sample_rate)``,
    concatenating any streamed segments into one array."""
    chunks, sample_rate = [], None
    for result in model_obj.generate(text, voice=voice, speed=speed):
        chunks.append(np.asarray(result.audio, dtype=np.float32))
        sample_rate = getattr(result, "sample_rate", sample_rate)
    if not chunks:
        raise RuntimeError("the TTS model produced no audio")
    if sample_rate is None:
        sample_rate = int(getattr(model_obj, "sample_rate", 24000))
    audio = chunks[0] if len(chunks) == 1 else np.concatenate(chunks)
    return audio, int(sample_rate)


def run_synthesis(text: str, *, configured_model: str, model: str = "",
                  voice: str = "", response_format: str = "", speed: str = ""):
    """Validate fields, synthesize ``text``, and return ``(content, media_type)``.
    Raises :class:`TTSRequestError` for 4xx problems.

    All field params arrive as the raw request strings; this owns parsing them.
    """
    if not (text or "").strip():
        raise TTSRequestError(400, "field 'input' is required")
    text = sanitize_speech_text(text)
    if not text:
        raise TTSRequestError(400, "field 'input' has no speakable text")
    target = effective_model(model, configured_model)
    fmt = (response_format or "mp3").strip().lower()
    if fmt not in RESPONSE_FORMATS:
        raise TTSRequestError(
            400, f"unsupported response_format {fmt!r} "
                 f"(supported: {', '.join(RESPONSE_FORMATS)})")
    try:
        spd = float(speed) if str(speed).strip() else 1.0
    except ValueError:
        raise TTSRequestError(400, f"speed {speed!r} is not a number")
    if not _SPEED_MIN <= spd <= _SPEED_MAX:
        raise TTSRequestError(
            400, f"speed must be between {_SPEED_MIN} and {_SPEED_MAX}")

    import_mlx_audio()

    # The Kokoro preset default only exists on models that ship it; forcing
    # it on another family (qwen3-tts named speakers) errors or mis-voices.
    # No voice means the model's own default applies.
    v = (voice or "").strip() or None
    if v is None and DEFAULT_VOICE in available_voices(target):
        v = DEFAULT_VOICE

    def _job():
        # Load (if needed) and synthesize on the same dedicated thread - see
        # subservice.SingleWorker. The mx -> numpy conversion in _synthesize
        # forces evaluation here, so only host-side numpy crosses back to the
        # caller.
        model_obj = _TTSModelHolder.get(target)
        return _synthesize(model_obj, text, v, spd)

    try:
        audio, sample_rate = _TTS_WORKER.submit(_job).result()
    except Exception as exc:
        raise RuntimeError(f"synthesis failed: {exc}") from exc
    content = encode_audio(audio, sample_rate, fmt)        # pure numpy/io
    return content, _MEDIA_TYPES[fmt]
