# SPDX-License-Identifier: Apache-2.0
"""Stand-in nodes for a ComfyUI the pack's full import fails on.

A ComfyUI release can change an internal the pack uses. Then the nodes
still load under their own names, so a workflow keeps its graph, and
running one raises an error that names the change, the ComfyUI version, the
newest version this gmlx was tested on and the gmlx upgrade for this
install. This module imports neither torch nor ComfyUI.
"""

from __future__ import annotations

import logging
import re

from . import compat, discovery, settings
from .version import COMFYUI_FLOOR, COMFYUI_TESTED, __version__

log = logging.getLogger("gmlx")

CATEGORY = "gmlx"


def _nums(version: str) -> tuple:
    return tuple(int(n) for n in re.findall(r"\d+", version)[:3])


def message(error: BaseException) -> str:
    """Why the gmlx nodes did not load, and what fixes it."""
    try:
        missing = compat.first_missing()
    except Exception:  # noqa: BLE001: the error itself then names the cause
        missing = None
    cause = (f"it needs {missing}, which this ComfyUI does not have" if missing
             else f"{type(error).__name__}: {error}")
    doc = settings.load()
    version = compat.comfyui_version()
    head = (f"gmlx: the gmlx nodes of gmlx {__version__} do not load on ComfyUI "
            f"{version}: {cause}.")
    guide = ""
    if _nums(version) and _nums(version) < _nums(COMFYUI_FLOOR):
        # An older ComfyUI than the nodes support: ComfyUI is what to update.
        text = (f"{head} They need ComfyUI {COMFYUI_FLOOR} or newer. Update ComfyUI, then "
                "restart it.")
        guide = f" The ComfyUI guide: {settings.GUIDE_URL}"
    else:
        # The copy in ComfyUI changes only when a gmlx command refreshes it.
        text = (f"{head} This gmlx was tested on ComfyUI up to {COMFYUI_TESTED}. To fix it, "
                f"{settings.upgrade_step(doc)}, which brings the nodes for newer ComfyUI "
                "releases, then run `gmlx comfyui install` and restart ComfyUI.")
    if settings.gmlx_command(doc) is None:
        text += " " + settings.not_installed(doc) + "."
    # The link ends the message, so no sentence runs on from it.
    return text + guide


def _loader():
    return discovery.loader_inputs()


def _sigmas():
    return {"required": {
        "model": ("MODEL",),
        "steps": ("INT", {"default": 0, "min": 0, "max": 1000}),
        "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01})}}


# Each node: its inputs, outputs, output names and whether it is an output
# node, as nodes.py declares them.
SPECS = {
    "GmlxModelLoader": (_loader, ("MODEL",), None, False),
    "GmlxCLIPLoader": (_loader, ("CLIP",), None, False),
    "GmlxVAELoader": (_loader, ("VAE",), None, False),
    "GmlxSigmas": (_sigmas, ("SIGMAS", "FLOAT"), ("sigmas", "cfg"), False),
    "GmlxServerStatus": (_loader, ("STRING",), None, True),
}

DISPLAY_NAMES = {
    "GmlxModelLoader": "gmlx Model Loader",
    "GmlxCLIPLoader": "gmlx CLIP Loader",
    "GmlxVAELoader": "gmlx VAE Loader",
    "GmlxSigmas": "gmlx Sigmas",
    "GmlxServerStatus": "gmlx Server Status",
}


def _node(name: str, text: str):
    inputs, returns, names, output = SPECS[name]

    def run(self, **_kw):
        if output:
            return {"ui": {"text": [text]}, "result": (text,)}
        raise RuntimeError(text)

    attrs = {"INPUT_TYPES": classmethod(lambda cls: inputs()), "RETURN_TYPES": returns,
             "FUNCTION": "run", "CATEGORY": CATEGORY, "OUTPUT_NODE": output,
             "DESCRIPTION": "Not loaded: " + text, "run": run}
    if names:
        attrs["RETURN_NAMES"] = names
    if inputs is _loader:
        # A saved model name the list lacks still reaches run, which names the fix.
        attrs["VALIDATE_INPUTS"] = classmethod(lambda cls, model_name: discovery.validate(model_name))
    return type(name, (), attrs)


def nodes(error: BaseException) -> tuple[dict, dict]:
    """The stand-in NODE_CLASS_MAPPINGS and NODE_DISPLAY_NAME_MAPPINGS."""
    text = message(error)
    log.error("%s", text, exc_info=error)
    return ({name: _node(name, text) for name in SPECS},
            {name: f"{label} (not loaded)" for name, label in DISPLAY_NAMES.items()})
