"""Upstream model files as upstream shipped them, for reference runs.

A copied model file that gmlx owns is an alias of the owned module, so the
copy path holds no upstream body. ``tests/cache/upstream`` keeps those
files, and their sha256 is the manifest's upstream digest. ``pristine``
loads one as ``pristine.<path>``, with its relative imports resolved in the
copied package. A copied file that is not an alias loads from the copy
under the same kind of name. While a file loads, each ``ORIGINS`` module
it imports from is the pristine one, so the objects it builds at import
are upstream's too. Each name the module took from an ``ORIGINS`` module
by another way is then rebound to the pristine one. A pristine container
builds the pristine language model, and a pristine language model runs
upstream's scans and verifier. An import that a function of the file runs
at call time resolves in the copied package, as it finds it then.
"""

from __future__ import annotations

import ast
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
    "gmlx._vendor.mlx_vlm.models.qwen3_5.gated_delta",
    "gmlx._vendor.mlx_vlm.models.qwen3_5.speculative_verifier",
    "gmlx._vendor.mlx_vlm.models.qwen3_5.language",
    "gmlx._vendor.mlx_vlm.models.gemma4.language",
    "gmlx._vendor.mlx_vlm.models.gemma4.gemma4",
    "gmlx._vendor.mlx_vlm.models.gemma4_unified.gemma4_unified",
    "gmlx._vendor.mlx_lm.models.gemma4_text",
    "gmlx._vendor.mlx_lm.models.gemma4",
)
# The entries that are aliases, which need a file under UPSTREAM. The
# others are copied files that bind names of an alias.
ALIASED = tuple(o for o in ORIGINS if not o.endswith(
    (".gemma4", ".speculative_verifier")))
_MISSING = object()

_modules: dict[str, object] = {}


def upstream_file(origin: str) -> Path:
    """The upstream file of the copied module ``origin``."""
    rel = origin.removeprefix(VENDOR).replace(".", "/") + ".py"
    return UPSTREAM / rel if origin in ALIASED else _COPY / rel


def _imports(origin: str) -> tuple[list[str], list[str]]:
    """The modules the file of ``origin`` imports as it loads, and those
    of them that are other ``ORIGINS`` entries."""
    package = origin.split(".")[:-1]
    modules, origins = [], []

    def visit(node):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            return
        if isinstance(node, ast.Import):
            modules.extend(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = package[:len(package) - node.level + 1] if node.level else []
            target = ".".join(base + ([node.module] if node.module else []))
            modules.append(target)
            for path in [target] + [f"{target}.{a.name}" for a in node.names]:
                if path in ORIGINS and path != origin and path not in origins:
                    origins.append(path)
        for child in ast.iter_child_nodes(node):
            visit(child)

    visit(ast.parse(upstream_file(origin).read_text()))
    return modules, origins


def pristine(origin: str):
    """The upstream module of the copied path ``origin``."""
    if origin in _modules:
        return _modules[origin]
    live = importlib.import_module(origin)
    name = "pristine." + origin.removeprefix(VENDOR)
    spec = importlib.util.spec_from_file_location(name, upstream_file(origin))
    module = importlib.util.module_from_spec(spec)
    module.__package__ = origin.rpartition(".")[0]
    modules, origins = _imports(origin)
    # Everything it imports is loaded first, so no module of the copy
    # loads during the swap below and keeps a pristine object.
    for other in modules:
        try:
            importlib.import_module(other)
        except ImportError:
            pass
    # The copied paths of the ORIGINS modules it imports from hold the
    # pristine ones while it loads, each of those loaded before any swap.
    swapped = []
    for other, upstream in [(o, pristine(o)) for o in origins]:
        parent, _, leaf = other.rpartition(".")
        swapped.append((other, sys.modules[other], parent, leaf,
                        getattr(sys.modules[parent], leaf, _MISSING)))
        sys.modules[other] = upstream
        setattr(sys.modules[parent], leaf, upstream)
    sys.modules[name] = module
    try:
        with warnings.catch_warnings():
            # The module's name is outside the package its imports resolve in.
            warnings.filterwarnings("ignore", "__package__ != __spec__.parent",
                                    DeprecationWarning)
            spec.loader.exec_module(module)
    finally:
        for other, copied, parent, leaf, attr in swapped:
            sys.modules[other] = copied
            if attr is _MISSING:
                delattr(sys.modules[parent], leaf)
            else:
                setattr(sys.modules[parent], leaf, attr)
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
