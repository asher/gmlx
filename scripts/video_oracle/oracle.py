"""Oracle-side helpers for the video checks, imported only by oracle
scripts in an oracle environment.

The torch helpers are those of scripts/image_oracle/oracle.py, loaded by
path: the offline guard, the CPU setup, the dtype of each mode, the
bfloat16 matmul emulation and the tensor to numpy conversion. The ComfyUI
helpers start a ComfyUI checkout on the CPU and import its ComfyUI-GGUF
custom node package, for the oracles that run ComfyUI's model code.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path


def _image_oracle():
    path = Path(__file__).resolve().parent.parent / "image_oracle" / \
        "oracle.py"
    spec = importlib.util.spec_from_file_location("_video_oracle_oracle",
                                                  path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"error: cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_img = _image_oracle()

MODES = _img.MODES
require_offline = _img.require_offline
setup = _img.setup
torch_dtype = _img.torch_dtype
emulate_bf16_matmul = _img.emulate_bf16_matmul
matmul_mode = _img.matmul_mode
to_numpy = _img.to_numpy


def git_head(path: str | Path) -> str:
    """The checkout's commit, with ``-dirty`` when a tracked file differs
    from it."""
    def git(*args):
        return subprocess.run(["git", "-C", str(path), *args],
                              capture_output=True, text=True,
                              check=True).stdout.strip()

    head = git("rev-parse", "HEAD")
    return head + "-dirty" if git("status", "--porcelain",
                                  "--untracked-files=no") else head


def start_comfy(comfy_dir: str | Path) -> str:
    """Import ComfyUI from the checkout ``comfy_dir`` with the arguments
    of ``main.py --cpu --use-pytorch-cross-attention`` and return its
    commit. Call it after the script has parsed its own arguments.

    Attention then runs through torch's scaled_dot_product_attention, as
    ComfyUI runs it on CUDA, with float32 logits in a bfloat16 pass. Its
    CPU kernel runs bfloat16 on every thread; the default sub-quadratic
    path rounds the logits to bfloat16 and multiplies on one thread."""
    root = Path(comfy_dir).expanduser().resolve()
    if not (root / "comfy" / "ldm").is_dir():
        raise SystemExit(f"error: {root} is not a ComfyUI checkout")
    argv = sys.argv
    sys.argv = [argv[0], "--cpu", "--use-pytorch-cross-attention"]
    try:
        sys.path.insert(0, str(root))
        import comfy.options

        comfy.options.enable_args_parsing()
        import comfy.cli_args  # noqa: F401  parses sys.argv on import
        import comfy.model_management  # noqa: F401
        from comfy.ldm.modules import attention
    finally:
        sys.argv = argv
    picked = attention.optimized_attention.__name__
    if picked != "attention_pytorch":
        raise SystemExit(f"error: ComfyUI picked {picked}, not "
                         "attention_pytorch")
    return git_head(root)


def load_comfy_gguf(comfy_dir: str | Path) -> str:
    """Import the checkout's custom_nodes/ComfyUI-GGUF as the package
    ``ComfyUI_GGUF`` and return its commit."""
    pkg = Path(comfy_dir).expanduser().resolve() / "custom_nodes" / \
        "ComfyUI-GGUF"
    spec = importlib.util.spec_from_file_location(
        "ComfyUI_GGUF", pkg / "__init__.py",
        submodule_search_locations=[str(pkg)])
    if spec is None or spec.loader is None:
        raise SystemExit(f"error: no ComfyUI-GGUF package at {pkg}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["ComfyUI_GGUF"] = mod
    spec.loader.exec_module(mod)
    return git_head(pkg)
