"""The stock objects the owned cache classes and SDPA helpers replace.

gmlx owns every body of mlx-vlm's ``models/cache.py`` and the mask and
SDPA helpers of the mlx-lm cache and both base modules, so the vendored
copy holds none of those stock bodies. ``upstream/`` keeps the four files
as upstream shipped them, and their sha256 is the manifest's upstream
digest. Each file loads here once, as ``stock.<path>``, with its relative
imports resolved in the vendored package. Then each name gmlx owns is
rebound to the owned object, as the vendored code sees it, and the stock
object is kept. Controls put the stock bodies on the owned classes to show
that a test fails without the owned fix.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path

from gmlx.upstream.patching import root
from owned_names import EXPECTED

UPSTREAM = Path(__file__).resolve().parent / "upstream"
VENDOR = "gmlx._vendor."
VLM_CACHE = "gmlx._vendor.mlx_vlm.models.cache"

# Base modules first: the lm cache imports its mask from the lm base.
ORIGINS = ("gmlx._vendor.mlx_lm.models.base", "gmlx._vendor.mlx_vlm.models.base",
           "gmlx._vendor.mlx_lm.models.cache", VLM_CACHE)

_modules: dict[str, object] = {}
# origin -> [(name, stock object, owned object)]
_replaced: dict[str, list[tuple[str, object, object]]] = {}


def upstream_file(origin: str) -> Path:
    """The upstream file of the vendored module ``origin``."""
    return UPSTREAM / (origin.removeprefix(VENDOR).replace(".", "/") + ".py")


def _load(origin: str):
    if origin in _modules:
        return _modules[origin]
    importlib.import_module(origin)
    name = "stock." + origin.removeprefix(VENDOR)
    spec = importlib.util.spec_from_file_location(name, upstream_file(origin))
    module = importlib.util.module_from_spec(spec)
    module.__package__ = origin.rpartition(".")[0]
    sys.modules[name] = module
    spec.loader.exec_module(module)
    pairs = []
    for attr, where in EXPECTED[origin].items():
        old = module.__dict__[attr]
        new = getattr(importlib.import_module(where), attr)
        if old is not new:
            # The owned object under any gmlx wrapper over it.
            pairs.append((attr, old, root(new)))
            setattr(module, attr, new)
    _modules[origin] = module
    _replaced[origin] = pairs
    return module


def replaced(origin: str | None = None) -> list[tuple[str, object, object]]:
    """(vendored module, stock object, owned object) for each owned name of
    ``origin``, or of all four modules."""
    out = []
    for o in ORIGINS if origin is None else (origin,):
        _load(o)
        out += [(o, old, new) for _name, old, new in _replaced[o]]
    return out


def stock_twin(owned, origin=VLM_CACHE):
    """The ``origin`` class or function whose name now holds ``owned``."""
    for _o, old, new in replaced(origin):
        if new is root(owned):
            return old
    raise LookupError(f"no {origin} twin of {owned!r}")


def stock_method(owned_cls, name, origin=VLM_CACHE):
    """The stock body of ``owned_cls.name``, as the stock class defines it."""
    return stock_twin(owned_cls, origin).__dict__[name]
