"""The node pack's suites need torch and a ComfyUI checkout, and run in
ComfyUI's Python through run_tests.py. gmlx's own pytest collects nothing
here unless COMFYUI_DIR names a ComfyUI checkout."""

import os

collect_ignore_glob = [] if os.environ.get("COMFYUI_DIR") else ["*"]
