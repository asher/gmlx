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

from . import compat, settings
from .version import COMFYUI_TESTED, __version__

log = logging.getLogger("gmlx")

CATEGORY = "gmlx"


def message(error: BaseException) -> str:
    """Why the gmlx nodes did not load, and what fixes it."""
    try:
        missing = compat.first_missing()
    except Exception:  # noqa: BLE001: the error itself then names the cause
        missing = None
    cause = (f"it needs {missing}, which this ComfyUI does not have" if missing
             else f"{type(error).__name__}: {error}")
    doc = settings.load()
    text = (f"gmlx: the gmlx nodes of gmlx {__version__} do not load on ComfyUI "
            f"{compat.comfyui_version()}: {cause}. This gmlx was tested on ComfyUI up to "
            f"{COMFYUI_TESTED}. To fix it, {settings.upgrade_step(doc)}, which brings the "
            "nodes for newer ComfyUI releases, then restart ComfyUI.")
    if settings.gmlx_command(doc) is None:
        text += " " + settings.not_installed(doc) + "."
    return text


def _loader():
    return {"required": {
        "model_name": ("STRING", {"default": settings.default_image_model()}),
        "server_url": ("STRING", {"default": ""}),
        "start_server": ("BOOLEAN", {"default": True})}}


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
    return type(name, (), attrs)


def nodes(error: BaseException) -> tuple[dict, dict]:
    """The stand-in NODE_CLASS_MAPPINGS and NODE_DISPLAY_NAME_MAPPINGS."""
    text = message(error)
    log.error("%s", text)
    return ({name: _node(name, text) for name in SPECS},
            {name: f"{label} (not loaded)" for name, label in DISPLAY_NAMES.items()})
