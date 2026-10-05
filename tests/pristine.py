"""Upstream model files as upstream shipped them, for reference runs.

A copied model file that gmlx owns is an alias of the owned module, so the
copy path holds no upstream body. ``tests/cache/upstream`` keeps those
files, and their sha256 is the manifest's upstream digest. ``pristine``
loads one as ``pristine.<path>``, with its relative imports resolved in the
copied package. A copied file that is not an alias loads from the copy
under the same kind of name. Each name such a module took from a module in
``ORIGINS`` is then rebound to the pristine one, so a pristine container
builds the pristine language model.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
import warnings
from pathlib import Path

UPSTREAM = Path(__file__).resolve().parent / "cache" / "upstream"
VENDOR = "gmlx._vendor."
_COPY = Path(__file__).resolve().parents[1] / "gmlx" / "_vendor"

# The aliased files with a reference test, and the copied containers over
# them. Each entry follows the modules it imports.
ORIGINS = (
    "gmlx._vendor.mlx_vlm.models.qwen3_5.language",
    "gmlx._vendor.mlx_vlm.models.gemma4.language",
    "gmlx._vendor.mlx_vlm.models.gemma4.gemma4",
    "gmlx._vendor.mlx_vlm.models.gemma4_unified.gemma4_unified",
    "gmlx._vendor.mlx_lm.models.gemma4_text",
    "gmlx._vendor.mlx_lm.models.gemma4",
)
# The entries that are aliases, which need a file under UPSTREAM.
ALIASED = tuple(o for o in ORIGINS if not o.endswith(".gemma4"))

_modules: dict[str, object] = {}


def upstream_file(origin: str) -> Path:
    """The upstream file of the copied module ``origin``."""
    rel = origin.removeprefix(VENDOR).replace(".", "/") + ".py"
    return UPSTREAM / rel if origin in ALIASED else _COPY / rel


def pristine(origin: str):
    """The upstream module of the copied path ``origin``."""
    if origin in _modules:
        return _modules[origin]
    live = importlib.import_module(origin)
    name = "pristine." + origin.removeprefix(VENDOR)
    spec = importlib.util.spec_from_file_location(name, upstream_file(origin))
    module = importlib.util.module_from_spec(spec)
    module.__package__ = origin.rpartition(".")[0]
    sys.modules[name] = module
    with warnings.catch_warnings():
        # The module's name is outside the package its imports resolve in.
        warnings.filterwarnings("ignore", "__package__ != __spec__.parent",
                                DeprecationWarning)
        spec.loader.exec_module(module)
    for other in ORIGINS[:ORIGINS.index(origin)]:
        if other not in sys.modules:
            continue
        source = sys.modules[other]
        for attr, value in list(vars(module).items()):
            if attr.startswith("__"):
                continue
            if value is source:
                setattr(module, attr, pristine(other))
            elif (vars(source).get(attr) is value
                  and getattr(value, "__module__", None) == source.__name__):
                setattr(module, attr, getattr(pristine(other), attr, value))
    assert module is not live
    _modules[origin] = module
    return module
