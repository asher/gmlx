"""Register gmlx tool-call parsers with mlx-vlm's parser registry.

mlx-vlm resolves a parser by name with ``load_tool_module`` and infers the
name from chat-template markers (``mlx_vlm.tools.registry.SPECS``, highest
priority wins). ``register_tool_parser`` adds a gmlx module to both: the
vendored registry's ``GMLX_PARSERS`` table, which ``load_tool_module``
reads first, and ``SPECS``. A model registers its parser when it loads, so
its markers route nothing before then.
"""

from __future__ import annotations

import importlib
import sys

# Stock format specs sit at priority 100 and the json_tools fallback at 0.
# A gmlx format outranks any stock spec whose markers overlap its own.
_GMLX_PRIORITY = 200

# Names of the specs this module added to the registry. The seam guard pins
# the stock table without them.
_REGISTERED: set[str] = set()


def registered_names() -> frozenset[str]:
    """Names of the tool-parser specs gmlx added to mlx-vlm's registry."""
    return frozenset(_REGISTERED)


def register_tool_parser(name: str, module, markers) -> None:
    """Make ``load_tool_module(name)`` return ``module``, ahead of any
    vendored parser of that name, and route templates carrying ``markers``
    (an OR of AND-groups of strings) to it. A spec the vendored registry
    already holds under ``name`` keeps its markers and routes to
    ``module``. Idempotent."""
    try:
        registry = importlib.import_module("gmlx._vendor.mlx_vlm.tools.registry")
    except ImportError:
        return
    registry.GMLX_PARSERS[name] = module
    if name in registry._BY_NAME:
        return
    spec = registry.ParserSpec(
        name, tuple(tuple(group) for group in markers), priority=_GMLX_PRIORITY)
    # _infer_tool_parser reads the module global at call time. The request
    # schemas validate a tool_parser override against the package
    # re-export, a separate binding.
    registry.SPECS = (spec, *registry.SPECS)
    registry._BY_NAME[name] = spec
    _REGISTERED.add(name)
    package = sys.modules.get("gmlx._vendor.mlx_vlm.tools")
    if package is not None and hasattr(package, "SPECS"):
        package.SPECS = registry.SPECS
