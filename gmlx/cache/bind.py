# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Asher Feldman
"""The registry of gmlx's own cache classes.

gmlx owns the mlx-vlm KV cache classes and base SDPA helpers, in
gmlx.cache.kv_cache and gmlx.cache.attention. The vendored mlx-vlm
``models/cache.py`` is an alias of gmlx.cache.kv_cache, and the vendored
mlx-lm ``models/cache.py`` and both ``models/base.py`` files import the
owned objects by name, so there is one class of each name.

gmlx's own cache classes (the KVarN family, MSAKVCache, QSAKVCache and the
DeepSeek-V4 pooling caches) join by ``register``. Owned and vendored code
resolve a cache class by name in their module globals when they restore a
cache: ``CacheList.from_state`` in gmlx.cache.kv_cache and
``load_prompt_cache`` in the vendored mlx-lm ``models/cache.py``.
``register`` puts each class in those globals, and the vendored module
takes the registered classes when it loads later (``apply_registry``).

This module imports no MLX and no vendored module.
"""

from __future__ import annotations

import importlib
import sys

KV = "gmlx.cache.kv_cache"

# The modules whose globals take registered gmlx cache classes.
REGISTRY_MODULES = (KV, "gmlx._vendor.mlx_lm.models.cache")

# name -> registered gmlx cache class
_registry: dict[str, type] = {}


def register(*classes: type) -> None:
    """Make each gmlx cache class resolvable by name in the modules of
    ``REGISTRY_MODULES``, including one that loads later. The class must
    derive from the owned ``_BaseCache``, or TypeError. A name that one of
    those modules, loaded now, binds to another object, or that another
    class holds, raises ValueError, so a registered class never shadows an
    owned class or a vendored name. Registering a class again does
    nothing."""
    owned = importlib.import_module(KV)
    pending: dict[str, type] = {}
    for cls in classes:
        if not (isinstance(cls, type) and issubclass(cls, owned._BaseCache)):
            raise TypeError(
                f"cannot register {cls!r}: not a subclass of "
                f"{KV}._BaseCache")
        name = cls.__name__
        held = pending.get(name) or _registry.get(name)
        if held is not None and held is not cls:
            raise ValueError(
                f"cannot register {_qualified(cls)}: the cache class name "
                f"{name!r} is registered to {_qualified(held)}")
        for where in REGISTRY_MODULES:
            module = sys.modules.get(where)
            if module is not None and module.__dict__.get(name, cls) is not cls:
                raise ValueError(
                    f"cannot register {_qualified(cls)}: {where} defines "
                    f"{name!r}")
        pending[name] = cls
    _registry.update(pending)
    for where in REGISTRY_MODULES:
        module = sys.modules.get(where)
        if module is not None:
            apply_registry(module)


def registered() -> dict[str, type]:
    """Each registered name and its gmlx cache class."""
    return dict(_registry)


def _qualified(cls) -> str:
    return f"{cls.__module__}.{cls.__qualname__}"


def apply_registry(module) -> None:
    """Put each registered class into ``module``. A module that loads after
    a class registered, and defines the name itself, raises ImportError
    before anything is set."""
    for name, cls in _registry.items():
        if module.__dict__.get(name, cls) is not cls:
            raise ImportError(
                f"{module.__name__} defines {name!r}, the name gmlx "
                f"registered for {_qualified(cls)}; a name the module "
                f"defines is never replaced")
    for name, cls in _registry.items():
        setattr(module, name, cls)
