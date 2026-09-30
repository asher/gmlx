"""The attention dispatch of the owned Gemma 4 text model.

``gmlx.models.gemma4.language`` imports ``scaled_dot_product_attention``
from here. A call meets three claims, then the base, in the order gmlx
routes stock gemma4 attention:

1. the sparse top-k page route (``gmlx.upstream.sparse_sdpa``), when
   ``GMLX_SPARSE_ATTN=1`` and gemma4_text is an allowed architecture;
2. the shared-prefix cascade route (``gmlx.upstream.cascade_sdpa``), when
   the loader installed it;
3. the head_dim 512 batched row route (``batched_sdpa``), when the hd512
   wrapper is installed and ``GMLX_G4_BATCHED_SDPA`` is not 0;
4. ``gmlx.cache.attention.scaled_dot_product_attention``: TurboQuant,
   KVarN, the quantized routes and the fp path.

The first claim that answers wins. ``configure()`` reads the gates once
per load; until it runs, every call goes to the base. On a load of the
owned model it names the open routes in one log line. A KVarN view
passes the three claims untouched and reaches the KVarN branch of the
base.

The signature is positional, as in the base, since training attention
wraps this module global and calls the original positionally.
"""

from __future__ import annotations

import gmlx.upstream.attn_hd512 as attn_hd512
from gmlx.cache.attention import scaled_dot_product_attention as _base
from gmlx.envflags import env_bool
from gmlx.upstream import cascade_sdpa, sparse_sdpa

from . import batched_sdpa

_SPARSE = False
_CASCADE = False
_BATCHED = False
# The routes in claim order, by gate, as the load log names them.
_ROUTE_NAMES = (("sparse", "sparse top-k pages"),
                ("cascade", "shared-prefix cascade"),
                ("batched", "hd512 batched rows"))


def configure(model=None, log=None) -> None:
    """Read the three route gates. The loader calls this once per load,
    after the installs the gates read. When ``model`` holds the owned
    text model, ``log`` gets one line naming the open routes."""
    global _SPARSE, _CASCADE, _BATCHED
    _SPARSE = env_bool("GMLX_SPARSE_ATTN", False) and sparse_sdpa.arch_enabled(
        "gemma4_text")
    _CASCADE = bool(cascade_sdpa._installed_route)
    _BATCHED = bool(attn_hd512._installed) and env_bool(
        "GMLX_G4_BATCHED_SDPA", True)
    if log is not None and _holds_owned_model(model):
        routes = [name for gate, name in _ROUTE_NAMES if gates()[gate]]
        log("[install] gemma-4 attention routes: "
            + (", ".join(routes) if routes else "none, the base SDPA only"))


def _holds_owned_model(model) -> bool:
    from .language import Gemma4TextModel

    named = getattr(model, "named_modules", None)
    return named is not None and any(
        isinstance(m, Gemma4TextModel) for _, m in named())


def gates() -> dict:
    """The route gates configure() set."""
    return {"sparse": _SPARSE, "cascade": _CASCADE, "batched": _BATCHED}


def scaled_dot_product_attention(queries, keys, values, cache=None,
                                 scale=1.0, mask=None, sinks=None):
    if _SPARSE:
        out = sparse_sdpa._claim(queries, keys, values, cache, scale, mask,
                                 sinks)
        if out is not None:
            return out
    if _CASCADE:
        out = cascade_sdpa._claim(queries, keys, values, cache, scale, mask,
                                  sinks)
        if out is not None:
            return out
    if _BATCHED:
        out = batched_sdpa._claim(queries, keys, values, cache, scale, mask,
                                  sinks)
        if out is not None:
            return out
    return _base(queries, keys, values, cache, scale, mask, sinks)
