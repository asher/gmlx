"""Graft gmlx tool-call parsers into mlx-vlm's parser registry.

mlx-vlm resolves a parser by name from ``mlx_vlm.tools.parsers.<name>`` and
infers the name from chat-template markers (``mlx_vlm.tools.registry.SPECS``,
highest priority wins). ``register_tool_parser`` adds a gmlx module to both.
"""

from __future__ import annotations

import importlib
import sys

from gmlx.upstream.graft import graft_module

# Stock format specs sit at priority 100 and the json_tools fallback at 0.
# A gmlx format outranks any stock spec whose markers overlap its own.
_GMLX_PRIORITY = 200


def register_tool_parser(name: str, module, markers) -> None:
    """Make ``mlx_vlm.tools.parsers.<name>`` resolve to ``module``, ahead of
    any upstream parser of that name, and route templates carrying
    ``markers`` (an OR of AND-groups of strings) to it. A spec upstream
    already registers under ``name`` keeps its markers and routes to
    ``module``. Idempotent."""
    graft_module(f"mlx_vlm.tools.parsers.{name}", module)
    try:
        registry = importlib.import_module("mlx_vlm.tools.registry")
    except ImportError:
        return
    if name in registry._BY_NAME:
        return
    spec = registry.ParserSpec(
        name, tuple(tuple(group) for group in markers), priority=_GMLX_PRIORITY)
    # _infer_tool_parser reads the module global at call time. The request
    # schemas validate a tool_parser override against the package
    # re-export, a separate binding.
    registry.SPECS = (spec, *registry.SPECS)
    registry._BY_NAME[name] = spec
    package = sys.modules.get("mlx_vlm.tools")
    if package is not None and hasattr(package, "SPECS"):
        package.SPECS = registry.SPECS
