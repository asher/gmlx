"""Detect and install gmlx's optional feature extras.

The ``init`` wizard (and ``init --with-* --install``) uses this to check whether
an optional service's package is importable and, on request, to install it.

On the pip route we install the extra's concrete dependency packages directly -
e.g. ``pip install mlx-whisper python-multipart`` - rather than
``pip install 'gmlx[stt]'``: gmlx may be an editable checkout or absent
from a package index, so resolving the extra by distribution name could fail or
disturb the install. The trade-off is that :data:`EXTRA_PACKAGES` must track
``pyproject.toml``'s ``[project.optional-dependencies]`` by hand.

Which installer to drive is not a given: gmlx may sit in a plain venv, or in an
environment owned by ``uv tool`` or pipx. :func:`install_route` decides, and
:func:`install_hint` renders the matching command for the "not installed"
messages throughout the codebase - none of which should hardcode ``pip``.
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
    "tts": ["mlx-audio", "spacy<4", "num2words",
            "addict", "phonemizer-fork", "espeakng-loader"],
    "embeddings": [],
    # pyproject's `talk` self-references gmlx[stt,tts]; here that flattens
    # to the concrete packages (see module docstring for why).
    "talk": ["sounddevice", "sherpa-onnx",
             "mlx-whisper", "python-multipart", "mlx-audio",
             "spacy<4", "num2words", "addict",
             "phonemizer-fork", "espeakng-loader"],
    "assistant": ["mcp"],
    # Everything: chat TUI + full voice stack + MCP assistant. Mirrors
    # pyproject's `all = gmlx[chat,talk,assistant]`, flattened.
    "all": ["prompt_toolkit", "rich", "sounddevice", "sherpa-onnx",
            "mlx-whisper", "python-multipart", "mlx-audio",
            "spacy<4", "num2words", "addict",
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
    # spacy as well as mlx_audio: it backs the vendored misaki G2P that Kokoro
    # phonemizes through, so a stack missing it synthesizes nothing - probing
    # mlx_audio alone reported such an install as complete.
    "tts": ("mlx_audio", "spacy"),
    "embeddings": ("mlx_embeddings",),
    "talk": ("sounddevice", "sherpa_onnx", "mlx_whisper", "mlx_audio", "spacy"),
    "assistant": ("mcp",),
    "all": ("prompt_toolkit", "sounddevice", "sherpa_onnx", "mlx_whisper",
            "mlx_audio", "spacy", "mcp"),
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


# How gmlx itself was installed decides how an extra is added to it. Tool
# installers put gmlx in an environment they own: uv's has no pip at all, and
# anything pip-installed into pipx's is dropped by the next upgrade. Both drop
# a receipt file at the environment root - the only reliable marker, since
# their directory layouts move with UV_TOOL_DIR / PIPX_HOME.
DIST_NAME = "gmlx"
_UV_RECEIPT = "uv-receipt.toml"
_PIPX_RECEIPT = "pipx_metadata.json"

ROUTE_UV = "uv-tool"
ROUTE_PIPX = "pipx"
ROUTE_PIP = "pip"


def install_route() -> str:
    """Which installer owns this environment: ``uv-tool`` / ``pipx`` / ``pip``."""
    from pathlib import Path
    prefix = Path(sys.prefix)
    if (prefix / _UV_RECEIPT).is_file():
        return ROUTE_UV
    if (prefix / _PIPX_RECEIPT).is_file():
        return ROUTE_PIPX
    return ROUTE_PIP


# Receipt keys naming a source other than an index. A reinstall driven from a
# reconstructed `name[extras]specifier` spec would silently repoint such a
# requirement at PyPI, so a receipt carrying any of them is not rebuilt at all.
_UV_SOURCE_KEYS = ("directory", "path", "url", "git", "editable")


def _uv_requirement(req: dict) -> str | None:
    """One receipt requirement as an installable spec, or ``None`` when it
    names a non-registry source. uv records the pin separately from the name
    (``{name = "rich", specifier = "==13.7.1"}``), so both have to be put back
    or a reinstall drops the constraint."""
    name = req.get("name")
    if not name or any(k in req for k in _UV_SOURCE_KEYS):
        return None
    extras = ",".join(e for e in (req.get("extras") or ()) if e)
    return f"{name}[{extras}]{req.get('specifier', '')}" if extras \
        else f"{name}{req.get('specifier', '')}"


def _uv_tool_spec(extra: str) -> tuple[str, list[str], str | None] | None:
    """``(gmlx[...], other requirements, pinned python)`` for a uv reinstall,
    or ``None`` when the receipt cannot be reproduced faithfully.

    ``uv tool install --force`` re-resolves the tool from the spec it is given,
    so the extra has to be merged with what the receipt already records -
    installing ``gmlx[tts]`` on its own would quietly drop an earlier
    ``gmlx[chat]``, along with any ``--with`` packages and their version pins.
    The recorded interpreter is carried too: without it the reinstall picks
    uv's default Python, which need not be the one currently running. When any
    requirement came from a directory, URL, or VCS, none of that can be
    rebuilt from the receipt, and guessing would move the install to PyPI."""
    import tomllib
    from pathlib import Path
    try:
        with open(Path(sys.prefix) / _UV_RECEIPT, "rb") as f:
            tool = tomllib.load(f).get("tool", {})
    except (OSError, ValueError):
        tool = {}
    extras, others, pin = {extra}, [], ""
    for req in tool.get("requirements", []):
        spec = _uv_requirement(req)
        if spec is None:
            return None
        if req.get("name") == DIST_NAME:
            extras |= {e for e in (req.get("extras") or ()) if e}
            pin = req.get("specifier", "")
        else:
            others.append(spec)
    return (f"{DIST_NAME}[{','.join(sorted(extras))}]{pin}", others,
            tool.get("python"))


def install_command(extra: str, route: str | None = None) -> list[str]:
    """The argv that adds ``extra`` to this environment, or ``[]`` when this
    environment's installer cannot be driven safely.

    The pip route installs the extra's concrete packages rather than
    ``gmlx[extra]`` (see the module docstring); the tool routes name the extra
    itself, which is safe there because a tool install of gmlx from an index
    can be reconstructed from its receipt - one from a local checkout or a VCS
    cannot, and that is the empty case."""
    route = route or install_route()
    if route == ROUTE_UV:
        plan = _uv_tool_spec(extra)
        if plan is None:
            return []
        spec, others, python = plan
        cmd = ["uv", "tool", "install", "--force", spec]
        for req in others:
            cmd += ["--with", req]
        return cmd + (["--python", python] if python else [])
    if route == ROUTE_PIPX:
        return ["pipx", "inject", DIST_NAME, *extra_packages(extra)]
    return [sys.executable, "-m", "pip", "install", *extra_packages(extra)]


def install_hint(extra: str) -> str:
    """The command to add ``extra``, for a message telling the user to run it.

    Every "not installed" message routes through this. The right command
    depends on how gmlx was installed, and naming the wrong one is worse than
    naming none: ``pip install`` inside a uv tool environment fails with
    ``No module named pip``, which reads as a broken gmlx."""
    import shlex
    route = install_route()
    if route == ROUTE_PIP:
        return f"pip install '{DIST_NAME}[{extra}]'"
    cmd = install_command(extra, route)
    if not cmd:
        return (f"reinstall gmlx from the source this tool install came from, "
                f"with the {extra} extra added")
    return shlex.join(cmd)


def install_extra(extra: str, *, runner=None) -> bool:
    """Install the extra's dependencies into the running gmlx. Returns True on
    success. ``runner`` overrides :func:`subprocess.run` (the test seam); the
    default streams the installer's own output."""
    pkgs = extra_packages(extra)
    if not pkgs:                                  # core feature - nothing to install
        return True
    route = install_route()
    cmd = install_command(extra, route)
    if not cmd or (route != ROUTE_PIP and shutil.which(cmd[0]) is None):
        why = (f"{cmd[0]} is not on PATH" if cmd
               else "this install came from a local or VCS source")
        print(f"[init] {why} - install {extra} with:\n"
              f"    {install_hint(extra)}", file=sys.stderr)
        return False
    print(f"[init] installing {extra}: {' '.join(pkgs)}", file=sys.stderr)
    run = runner or subprocess.run
    try:
        proc = run(cmd)
    except Exception as exc:                     # missing installer / spawn failure
        print(f"[init] install failed to launch: {exc}", file=sys.stderr)
        return False
    return getattr(proc, "returncode", 1) == 0
