# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Asher Feldman
"""Bind the owned cache classes and SDPA helpers under their stock names.

gmlx owns the mlx-vlm KV cache classes and base SDPA helpers, in
gmlx.cache.kv_cache and gmlx.cache.attention. Hundreds of stock modules
import those names by name as they load, so setting attributes on the
stock modules afterwards would miss them. ``install`` puts a finder on
``sys.meta_path`` that wraps the execution of the four stock modules that
define the names. When one finishes executing, each mapped name is
rebound to the owned object before any importer reads it, so every later
by-name import and subclass statement sees the owned object.

A stock module that loaded before gmlx takes the late path instead: its
names are rebound, every loaded module global that holds a replaced
object is pointed at the owned one, each loaded direct subclass of a
replaced class is rebased onto the owned class, and the APC type tables
that were filled with stock classes are reset. An instance created before
that keeps its class.

gmlx's own cache classes (the KVarN family, MSAKVCache, QSAKVCache and the
DeepSeek-V4 pooling caches) join by ``register``. Stock and owned code
resolve a cache class by name in their module globals when they restore a
cache: ``CacheList.from_state`` in gmlx.cache.kv_cache and mlx_lm
``load_prompt_cache``. ``register`` puts each class in those globals and in
mlx_vlm.models.cache, and the finder puts it into a stock cache module that
loads later.

The install imports nothing from MLX. This module imports no upstream
package; the owned modules load when a stock module first needs them.
"""

from __future__ import annotations

import importlib
import sys

KV = "gmlx.cache.kv_cache"
ATTN = "gmlx.cache.attention"

# Every module-level name mlx_vlm.models.cache defines, private ones too.
VLM_CACHE_NAMES = (
    "_BaseCache", "CacheMemory",
    "ConcatenateKVCache", "QuantizedKVCache", "KVCache", "RotatingKVCache",
    "ArraysCache", "ChunkedKVCache", "CacheList", "BatchKVCache",
    "BatchRotatingKVCache", "BufferedRotatingKVCache",
    "BatchQuantizedKVCache", "PoolingCache", "BatchPoolingCache",
    "SimpleKVCache", "StaticPrefixKVCache",
    "should_quantize_kv_layer", "create_causal_mask", "make_prompt_cache",
    "create_attention_mask", "cache_nbytes", "_kv_memory_profile",
    "_windowed_memory_profile", "_pooling_memory_profile",
    "_dequantize_uniform", "dynamic_roll", "_select_pooling_remainder",
    "_speculative_lengths", "_record_pooling_speculative_inputs",
    "_pooling_speculative_inputs", "_pooling_speculative_nbytes",
)
# The mlx_lm names with an owned twin. The prompt cache file helpers,
# TokenBuffer, PromptTrie and LRUPromptCache stay stock.
LM_CACHE_NAMES = (
    "_BaseCache", "ConcatenateKVCache", "QuantizedKVCache", "KVCache",
    "RotatingKVCache", "ArraysCache", "ChunkedKVCache", "CacheList",
    "BatchKVCache", "BatchRotatingKVCache", "dynamic_roll",
    "make_prompt_cache", "create_attention_mask", "create_causal_mask",
)
VLM_BASE_NAMES = (
    "create_attention_mask", "create_ssm_mask",
    "align_attention_mask_to_scores",
    "quantized_scaled_dot_product_attention",
    "_turboquant_attention_applies", "scaled_dot_product_attention",
)
LM_BASE_NAMES = (
    "create_attention_mask", "create_ssm_mask",
    "quantized_scaled_dot_product_attention", "scaled_dot_product_attention",
)

# stock module -> {name: (owned module, name)}. create_attention_mask names
# a different function in the cache and base modules, and each binds to
# its own twin.
MAP: dict[str, dict[str, tuple[str, str]]] = {
    "mlx_vlm.models.cache": {n: (KV, n) for n in VLM_CACHE_NAMES},
    "mlx_lm.models.cache": {n: (KV, n) for n in LM_CACHE_NAMES},
    "mlx_vlm.models.base": {"create_causal_mask": (KV, "create_causal_mask"),
                            **{n: (ATTN, n) for n in VLM_BASE_NAMES}},
    "mlx_lm.models.base": {"create_causal_mask": (KV, "create_causal_mask"),
                           **{n: (ATTN, n) for n in LM_BASE_NAMES}},
}

# The modules whose globals take registered gmlx cache classes.
REGISTRY_MODULES = (KV, "mlx_lm.models.cache", "mlx_vlm.models.cache")

# name -> registered gmlx cache class
_registry: dict[str, type] = {}

# id(stock object) -> (stock object, owned object). Holding the stock
# object keeps its id from being reused.
_replaced: dict[int, tuple[object, object]] = {}
# Stock modules that loaded before install.
late_modules: list[str] = []


def replaced() -> list[tuple[object, object]]:
    """Each (stock object, owned object) pair the binding replaced."""
    return list(_replaced.values())


def _owned(where: str):
    return importlib.import_module(where)


def bind_module(module) -> None:
    """Rebind the mapped names of the stock ``module`` to the owned objects."""
    for name, (where, attr) in MAP[module.__name__].items():
        old = module.__dict__.get(name)
        if old is None:
            continue
        new = getattr(_owned(where), attr)
        if old is not new:
            _replaced.setdefault(id(old), (old, new))
            setattr(module, name, new)


def register(*classes: type) -> None:
    """Make each gmlx cache class resolvable by name in the modules of
    ``REGISTRY_MODULES``, including one that loads later. A name that
    gmlx.cache.kv_cache defines, or that another class holds, raises
    ValueError, so a gmlx class never shadows an owned class. Registering
    a class again does nothing."""
    owned = _owned(KV)
    for cls in classes:
        name = cls.__name__
        held = _registry.get(name)
        if held is cls:
            continue
        if held is not None:
            raise ValueError(
                f"cannot register {cls.__module__}.{cls.__qualname__}: the "
                f"cache class name {name!r} is registered to "
                f"{held.__module__}.{held.__qualname__}")
        if name in vars(owned):
            raise ValueError(
                f"cannot register {cls.__module__}.{cls.__qualname__}: "
                f"{KV} defines {name!r}")
        _registry[name] = cls
    for where in REGISTRY_MODULES:
        module = sys.modules.get(where)
        if module is not None:
            _apply_registry(module)


def registered() -> dict[str, type]:
    """Each registered name and its gmlx cache class."""
    return dict(_registry)


def _apply_registry(module) -> None:
    for name, cls in _registry.items():
        setattr(module, name, cls)


def _bind(module) -> None:
    bind_module(module)
    if module.__name__ in REGISTRY_MODULES:
        _apply_registry(module)


class _Finder:
    """Find the four stock modules through the other finders, and bind each
    one as soon as it has executed, the registered classes included."""

    def find_spec(self, fullname, path=None, target=None):
        if fullname not in MAP:
            return None
        for finder in sys.meta_path:
            if finder is self or not hasattr(finder, "find_spec"):
                continue
            spec = finder.find_spec(fullname, path, target)
            if spec is not None:
                break
        else:
            return None
        loader = spec.loader
        if loader is None or not hasattr(loader, "exec_module"):
            return spec
        stock_exec = loader.exec_module

        def exec_module(module):
            stock_exec(module)
            _bind(module)

        # The loader instance serves this one module, so its import
        # metadata (__loader__, __spec__, __file__) stays the stock one.
        loader.exec_module = exec_module
        return spec


def _sweep_globals() -> None:
    table = {k: new for k, (_old, new) in _replaced.items()}
    for mod in list(sys.modules.values()):
        d = getattr(mod, "__dict__", None)
        if not isinstance(d, dict):
            continue
        for key, value in list(d.items()):
            new = table.get(id(value))
            if new is not None and _replaced[id(value)][0] is value:
                d[key] = new


def _rebase_subclasses() -> None:
    swap: dict[int, type] = {
        id(old): new for old, new in _replaced.values()
        if isinstance(old, type) and isinstance(new, type)}
    for old, _new in list(_replaced.values()):
        if not isinstance(old, type):
            continue
        for sub in old.__subclasses__():
            if id(sub) in swap:
                continue
            bases = tuple(swap.get(id(b), b) for b in sub.__bases__)
            try:
                sub.__bases__ = bases
            except TypeError as e:
                raise ImportError(
                    f"{sub.__module__}.{sub.__qualname__} subclasses a stock "
                    f"cache class that loaded before gmlx, and cannot be "
                    f"moved onto the gmlx cache classes: {e}. Import gmlx "
                    f"before mlx_vlm and mlx_lm.") from e


def _reset_apc_tables() -> None:
    adapters = sys.modules.get("mlx_vlm.apc_adapters")
    if adapters is None:
        return
    cap = getattr(adapters, "_CAPABILITY", None)
    if isinstance(cap, dict):
        for key in list(cap):
            hit = _replaced.get(id(key))
            if hit is not None and hit[0] is key:
                cap[hit[1]] = cap.pop(key)
    for name in ("_APC_EXACT_TYPES", "_APC_BLOCK_TYPES"):
        if hasattr(adapters, name):
            setattr(adapters, name, None)


def _late_path(modules) -> None:
    for module in modules:
        _bind(module)
    _sweep_globals()
    _rebase_subclasses()
    _reset_apc_tables()


def install() -> None:
    """Put the finder first on ``sys.meta_path``, once, and take the late
    path for stock modules already loaded."""
    if any(isinstance(f, _Finder) for f in sys.meta_path):
        return
    sys.meta_path.insert(0, _Finder())
    loaded = [sys.modules[name] for name in MAP if name in sys.modules]
    if loaded:
        late_modules.extend(m.__name__ for m in loaded)
        _late_path(loaded)
