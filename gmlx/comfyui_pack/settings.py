# SPDX-License-Identifier: Apache-2.0
"""The settings file, and what ``gmlx comfyui install`` records in it.

The user may set ``base_url`` and ``api_key`` there. Install adds the
absolute path of the ``gmlx`` command, how gmlx was installed, the default
image model and where the pack's copy lives. This module imports neither
torch nor ComfyUI, so the stand-in nodes can use it when the full import
fails.
"""

from __future__ import annotations

import json
import os
import shlex
import shutil

INSTALL_URL = "https://asher.github.io/gmlx/installation.html"
GUIDE_URL = "https://asher.github.io/gmlx/comfyui.html"
FALLBACK_MODEL = "z-image-turbo"

# The folder ComfyUI imported the pack through: the comfyui-gmlx link in its
# custom_nodes.
PACK_DIR = os.path.dirname(os.path.abspath(__file__))


def settings_path() -> str:
    return os.environ.get("GMLX_COMFY_SETTINGS") or os.path.expanduser(
        "~/.config/gmlx/comfyui.json")


def load() -> dict:
    try:
        with open(settings_path()) as f:
            doc = json.load(f)
        return doc if isinstance(doc, dict) else {}
    except (OSError, ValueError):
        return {}


def gmlx_command(doc: dict | None = None) -> str | None:
    """The ``gmlx`` command: the one install recorded while it exists, else
    the one on PATH, else None. ComfyUI started from the Dock has no login
    shell PATH, so the recorded path comes first."""
    doc = load() if doc is None else doc
    rec = doc.get("gmlx_command")
    if isinstance(rec, str) and os.path.isabs(rec) and os.path.isfile(rec) \
            and os.access(rec, os.X_OK):
        return rec
    return shutil.which("gmlx")


def upgrade_command(doc: dict | None = None) -> str | None:
    """The command that upgrades gmlx for the recorded install kind."""
    doc = load() if doc is None else doc
    kind = doc.get("gmlx_install")
    if kind == "homebrew":
        return "brew upgrade gmlx"
    if kind == "uv":
        return "uv tool upgrade gmlx"
    if kind == "pip":
        py = doc.get("gmlx_python")
        py = shlex.quote(py) if isinstance(py, str) and py else "python"
        return f"{py} -m pip install -U gmlx pillow opencv-python"
    return None


def upgrade_step(doc: dict | None = None) -> str:
    """How to upgrade gmlx, as a clause: the command for the recorded install
    kind, else the installation page."""
    cmd = upgrade_command(doc)
    if cmd:
        return f"upgrade gmlx with `{cmd}`"
    return f"upgrade gmlx the way you installed it, as {INSTALL_URL} says"


def default_image_model(doc: dict | None = None) -> str:
    """The model a new loader node starts on: GMLX_DEFAULT_IMAGE_MODEL, then
    the recorded default image model, then z-image-turbo."""
    doc = load() if doc is None else doc
    rec = doc.get("default_image_model")
    return (os.environ.get("GMLX_DEFAULT_IMAGE_MODEL")
            or (rec if isinstance(rec, str) and rec else None) or FALLBACK_MODEL)


def leftovers(doc: dict | None = None) -> list:
    """The two paths that remove the pack by hand: its link in ComfyUI's
    custom_nodes, and the folder that holds gmlx's copy of it."""
    doc = load() if doc is None else doc
    out = [PACK_DIR]
    real = os.path.realpath(PACK_DIR)
    pack_dir = doc.get("pack_dir")
    if isinstance(pack_dir, str) and pack_dir and not doc.get("pack_source"):
        out.append(os.path.dirname(pack_dir))
    elif os.path.basename(os.path.dirname(real)) == "versions":
        out.append(os.path.dirname(os.path.dirname(real)))
    return out


def not_installed(doc: dict | None = None, cause: str | None = None) -> str:
    """The message for a pack whose gmlx is gone, or whose gmlx command
    does not run, as ``cause`` says."""
    doc = load() if doc is None else doc
    if cause is None:
        rec = doc.get("gmlx_command")
        gone = f"{rec} does not exist and " if isinstance(rec, str) and rec else ""
        cause = f"{gone}no gmlx command is on PATH"
    paths = " and ".join(leftovers(doc))
    return (f"gmlx is not installed: {cause}. Install gmlx again ({INSTALL_URL}), or remove "
            f"the gmlx nodes by deleting {paths}")
