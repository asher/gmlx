"""Dual-origin KV-cache class compatibility.

mlx-vlm <= 0.6.3 re-exported mlx-lm's cache classes, so a cache kind had one
class identity everywhere. 0.6.4 vendored models/cache.py: the same kind now
has two identities, and which one a live cache carries depends on who built
it (mlx-vlm's models/generate stack vs mlx-lm's). Shared code paths must not
assume either origin:

- ``cache_types(name)`` returns every loaded origin's class for isinstance
  checks. It never imports mlx-vlm: if ``mlx_vlm.models.cache`` isn't in
  ``sys.modules`` no vlm-origin cache can exist yet, so the mlx-lm-only
  tuple is complete -- and plain text-model loads stay free of the VLM
  stack.
- ``runtime_cache_module()`` returns the cache module whose classes the
  mlx-vlm runtime (apc.py isinstance gates, ar.py batch machinery) expects;
  use it to construct caches that will be handed to that machinery. On
  <= 0.6.3 it resolves to the same classes as mlx-lm's.
"""
from __future__ import annotations

import importlib
import sys


def _origin_modules():
    lm = importlib.import_module("mlx_lm.models.cache")
    vlm = sys.modules.get("mlx_vlm.models.cache")
    return (lm, vlm) if vlm is not None else (lm,)


def runtime_cache_module():
    """The cache module matching the mlx-vlm runtime's class identities."""
    return importlib.import_module("mlx_vlm.models.cache")


def construction_cache_module():
    """Origin for constructing caches in code shared between the mlx-lm and
    mlx-vlm stacks: the vlm module when the vlm runtime is loaded (its
    apc/generate machinery isinstance-gates on its own classes), else
    mlx-lm's. Never imports mlx-vlm, so pure text paths stay light."""
    lm = importlib.import_module("mlx_lm.models.cache")
    return sys.modules.get("mlx_vlm.models.cache", lm)


def rebind_to_runtime_origin(caches):
    """Rebind mlx-lm-origin plain cache entries to the mlx-vlm runtime's
    class identities via ``__class__`` swap, recursing into CacheList.

    The vendored classes are attribute-compatible (plain ``__dict__``
    layouts, no slots), so the swap is a pure identity change on an empty
    or populated cache. Entries from any other origin (gmlx pooling
    caches, already-vlm entries) pass through. No-op when mlx-vlm is not
    loaded.
    """
    vlm = sys.modules.get("mlx_vlm.models.cache")
    if vlm is None:
        return caches
    for c in caches or []:
        _rebind_entry(c, vlm)
    return caches


def _rebind_entry(c, vlm) -> None:
    for sub in getattr(c, "caches", None) or ():
        _rebind_entry(sub, vlm)
    cls = type(c)
    if cls.__module__ != "mlx_lm.models.cache":
        return
    target = getattr(vlm, cls.__name__, None)
    if isinstance(target, type) and target is not cls:
        c.__class__ = target


def ensure_runtime_origin_make_cache(model):
    """Wrap ``model.make_cache`` so served caches carry the mlx-vlm
    runtime's class identities. Since 0.6.4 apc/ar isinstance-gate on
    those; an mlx-lm-arch model served through the vlm engine otherwise
    resolves ``model_apc_mode`` to None and silently loses APC entirely.
    Idempotent; no-op for models without ``make_cache``.
    """
    make = getattr(model, "make_cache", None)
    if not callable(make) or getattr(make, "_kq_runtime_origin", False):
        return model

    def make_cache_runtime_origin():
        return rebind_to_runtime_origin(make())

    make_cache_runtime_origin._kq_runtime_origin = True
    model.make_cache = make_cache_runtime_origin
    return model


def cache_types(name: str) -> tuple[type, ...]:
    """Every loaded origin's class for cache kind ``name``, deduplicated.

    Raises AttributeError only when no origin defines the name; a kind one
    origin lacks (e.g. BufferedRotatingKVCache exists only in mlx-vlm)
    yields the tuple of origins that do define it.
    """
    out = []
    for mod in _origin_modules():
        cls = getattr(mod, name, None)
        if isinstance(cls, type) and cls not in out:
            out.append(cls)
    if not out:
        raise AttributeError(
            f"cache class {name!r} not found in mlx-lm or mlx-vlm "
            "(upstream cache module change?)")
    return tuple(out)
