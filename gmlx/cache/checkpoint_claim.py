"""The APC disk tier resolves the owned cache classes by their own module.

mlx-vlm's ``_resolve_checkpoint_class`` resolves the class a checkpoint
shard records only under ``mlx_vlm.``. The owned classes keep their true
module, gmlx.cache.kv_cache, so without the claim the writer would skip
them and the reader would decline their shards. The claim accepts that one
module as well. It does not accept every gmlx module, so KVarN and
kv_policy classes keep their own disk kinds. A shard written before the
binding names mlx_vlm.models.cache, which now resolves to the owned class
too.

Installed with the other cache guards at model load. Idempotent.
"""

from __future__ import annotations

import importlib

from gmlx.upstream.patching import link, wrapped_with

OWNED_MODULE = "gmlx.cache.kv_cache"
_FLAG = "_gmlx_owned_checkpoint_class"


def _resolve_owned(qualname: str):
    value = importlib.import_module(OWNED_MODULE)
    try:
        for part in qualname.split("."):
            value = getattr(value, part)
    except AttributeError:
        return None
    return value if isinstance(value, type) else None


def install_checkpoint_class_claim() -> bool:
    """Widen ``mlx_vlm.apc._resolve_checkpoint_class`` to the owned module.

    Returns False when mlx-vlm is not installed."""
    try:
        apc = importlib.import_module("gmlx._vendor.mlx_vlm.apc")
    except ImportError:
        return False
    stock = getattr(apc, "_resolve_checkpoint_class", None)
    if stock is None:
        raise RuntimeError(
            "mlx_vlm.apc._resolve_checkpoint_class is gone; re-audit "
            "against the pinned seams (gmlx.upstream.seams)")
    if wrapped_with(stock, _FLAG):
        return True

    def resolve_checkpoint_class(module_name: str, qualname: str):
        if module_name == OWNED_MODULE and "<locals>" not in qualname:
            return _resolve_owned(qualname)
        return stock(module_name, qualname)

    setattr(apc, "_resolve_checkpoint_class",
            link(resolve_checkpoint_class, stock, _FLAG))
    return True
