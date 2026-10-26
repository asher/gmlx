# SPDX-License-Identifier: Apache-2.0
"""The pack's version and the ComfyUI versions it supports.

gmlx reads this file as text, never imports it, so it holds plain literals
only. The pack's version is gmlx's version; a gmlx test fails when they
differ.
"""

__version__ = "0.5.2"

# The oldest ComfyUI the pack loads on. `gmlx comfyui install` refuses an
# older one.
COMFYUI_FLOOR = "0.37.0"

# The newest ComfyUI release this version was tested on.
COMFYUI_TESTED = "0.41.1"
