# SPDX-License-Identifier: Apache-2.0
"""ComfyUI-gmlx: image model steps on a local gmlx server."""

from . import compat

compat.check()

from .model import register_prompt_handler  # noqa: E402
from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS  # noqa: E402

register_prompt_handler()

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
