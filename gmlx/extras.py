"""Detect and install gmlx's optional feature extras.

The ``init`` wizard (and ``init --with-* --install``) uses this to check whether
an optional service's package is importable and, on request, to install it.

We install the extra's concrete dependency packages directly - e.g.
``pip install mlx-whisper python-multipart`` - rather than
``pip install 'gmlx[stt]'``: gmlx may be an editable checkout or absent
from a package index, so resolving the extra by distribution name could fail or
disturb the install. The trade-off is that :data:`EXTRA_PACKAGES` must track
``pyproject.toml``'s ``[project.optional-dependencies]`` by hand.
"""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys

# Mirrors pyproject [project.optional-dependencies]. Keep in sync when those move.
# `vlm` and `embeddings` are core dependencies now (the serve engine and the
# encoder-embedding backends hard-require mlx-vlm / mlx-embeddings); their entries
# are empty - matching pyproject's back-compat empty extras - so `install_extra` is
# a no-op for them (the presence probe still uses find_spec and reports the real
# installed state).
EXTRA_PACKAGES = {
    "vlm": [],
    "chat": ["prompt_toolkit", "rich"],
    "stt": ["mlx-whisper", "python-multipart"],
    # Kokoro's misaki G2P is vendored (gmlx/_vendor); these are its
    # runtime deps - see the pyproject tts comment.
    "tts": ["mlx-audio", "spacy<4; python_version < '3.14'", "num2words",
            "addict", "phonemizer-fork", "espeakng-loader"],
    "embeddings": [],
    # pyproject's `talk` self-references gmlx[stt,tts]; here that flattens
    # to the concrete packages (see module docstring for why).
    "talk": ["sounddevice", "sherpa-onnx",
             "mlx-whisper", "python-multipart", "mlx-audio",
             "spacy<4; python_version < '3.14'", "num2words", "addict",
             "phonemizer-fork", "espeakng-loader"],
    "assistant": ["mcp"],
    # Everything: chat TUI + full voice stack + MCP assistant. Mirrors
    # pyproject's `all = gmlx[chat,talk,assistant]`, flattened.
    "all": ["prompt_toolkit", "rich", "sounddevice", "sherpa-onnx",
            "mlx-whisper", "python-multipart", "mlx-audio",
            "spacy<4; python_version < '3.14'", "num2words", "addict",
            "phonemizer-fork", "espeakng-loader", "mcp"],
}

# The feature-critical imports each extra provides - all must be importable to
# answer "is this extra installed?" (the distribution name is not the import
# name). Probing a single package misses half-installed extras - a rebuilt
# venv that kept sounddevice but lost sherpa-onnx would report `talk` as
# installed while wake-word mode silently degrades to an open mic.
_PROBE_MODULES = {
    "vlm": ("mlx_vlm",),
    "chat": ("prompt_toolkit",),
    "stt": ("mlx_whisper",),
    "tts": ("mlx_audio",),
    "embeddings": ("mlx_embeddings",),
    "talk": ("sounddevice", "sherpa_onnx", "mlx_whisper", "mlx_audio"),
    "assistant": ("mcp",),
    "all": ("prompt_toolkit", "sounddevice", "sherpa_onnx", "mlx_whisper",
            "mlx_audio", "mcp"),
}

# Extras whose runtime additionally needs ffmpeg on PATH (audio decode/encode).
FFMPEG_EXTRAS = frozenset({"stt", "tts", "talk", "all"})


def extra_packages(extra: str) -> list[str]:
    """The pip package specifiers an extra installs."""
    try:
        return list(EXTRA_PACKAGES[extra])
    except KeyError:
        raise KeyError(f"unknown extra {extra!r}") from None


def missing_extra_modules(extra: str) -> list[str]:
    """The extra's probe modules that are not importable in this interpreter
    (empty = fully installed). Names the exact hole in a half-installed extra
    so callers can say more than "reinstall everything"."""
    try:
        modules = _PROBE_MODULES[extra]
    except KeyError:
        raise KeyError(f"unknown extra {extra!r}") from None
    missing = []
    for module in modules:
        try:
            if importlib.util.find_spec(module) is None:
                missing.append(module)
        except (ImportError, ValueError):
            missing.append(module)
    return missing


def extra_installed(extra: str) -> bool:
    """True if every probe module of the extra is importable."""
    return not missing_extra_modules(extra)


def ffmpeg_present() -> bool:
    """True if an ``ffmpeg`` binary is on PATH."""
    return shutil.which("ffmpeg") is not None


def install_extra(extra: str, *, runner=None) -> bool:
    """``pip install`` the extra's dependency packages into the running
    interpreter. Returns True on success. ``runner`` overrides
    :func:`subprocess.run` (the test seam); the default streams pip's own output."""
    pkgs = extra_packages(extra)
    if not pkgs:                                  # core feature - nothing to install
        return True
    cmd = [sys.executable, "-m", "pip", "install", *pkgs]
    run = runner or subprocess.run
    print(f"[init] installing {extra}: {' '.join(pkgs)}", file=sys.stderr)
    try:
        proc = run(cmd)
    except Exception as exc:                     # pip missing / spawn failure
        print(f"[init] install failed to launch: {exc}", file=sys.stderr)
        return False
    return getattr(proc, "returncode", 1) == 0
