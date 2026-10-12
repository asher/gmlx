# SPDX-License-Identifier: Apache-2.0
"""gmlx nodes for ComfyUI: image model steps on a local gmlx server.

The full import needs ComfyUI internals that a ComfyUI release can change.
When it fails inside ComfyUI, stand-in nodes of the same names load instead,
and running one names the change and the gmlx upgrade for this install.
Outside ComfyUI the import fails as it is.
"""

import importlib.util

try:
    from . import compat

    compat.check()

    from .model import register_prompt_handler
    from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

    register_prompt_handler()
except Exception as error:  # noqa: BLE001: any failure gets the stand-in nodes
    if importlib.util.find_spec("comfy") is None:
        raise
    from . import stub

    NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS = stub.nodes(error)

from . import discovery  # noqa: E402

# The model list of the loaders is probed while ComfyUI starts.
discovery.refresh()

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
