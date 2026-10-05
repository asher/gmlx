"""The vendored cache modules hand out the owned cache classes and helpers.

The vendored mlx-vlm ``models/cache.py`` is an alias of
gmlx.cache.kv_cache, and the vendored mlx-lm ``models/cache.py`` and both
``models/base.py`` files import the owned objects by name in place of
their own bodies. These tests hold that in this process, in each import
order in a child, and at the places that keep a class they saw first:
borrowed module globals, the APC tables, the checkpoint class resolver and
``CacheList.from_state``. The class registry that makes gmlx's own cache
classes resolvable by name is held here too.
"""

from __future__ import annotations

import ast
import hashlib
import importlib
import inspect
import json
import os
import subprocess
import sys
from pathlib import Path

import mlx.core as mx
import pytest

import gmlx.cache.attention as owned_attn
import gmlx.cache.kv_cache as owned
from gmlx.cache import bind
from gmlx.upstream.patching import _walk, root
from owned_names import ATTN, EXPECTED, KV, OWNED_CLASSES, VLM_BASE, VLM_CACHE
from stock_blocker import BLOCKED
from stock_twin import UPSTREAM, upstream_file

VENDOR = "gmlx._vendor."
VLM_CACHE_PATH = "gmlx._vendor.mlx_vlm.models.cache"
LM_CACHE_PATH = "gmlx._vendor.mlx_lm.models.cache"


def _mod(name):
    return importlib.import_module(name)


def _stock(path: str) -> str:
    """The stock spelling of a vendored module path."""
    return path.removeprefix(VENDOR)


def _leads_to(value, target) -> bool:
    """``value`` and ``target`` are the same object once any gmlx wrapper
    over either is followed down (the KVarN sweep also wraps the owned
    SDPA in gmlx.cache.attention)."""
    return root(value) is root(target)


@pytest.mark.parametrize("vendored", sorted(EXPECTED))
def test_every_shared_name_is_the_owned_object(vendored):
    mod = _mod(vendored)
    for name, where in EXPECTED[vendored].items():
        value = getattr(mod, name)
        assert _leads_to(value, getattr(_mod(where), name)), (
            vendored, name, [(getattr(f, "__module__", None),
                              getattr(f, "__qualname__", None))
                             for f in _walk(value)])


def test_the_vlm_cache_path_is_the_owned_module():
    import gmlx._vendor.mlx_vlm.models as vlm_models

    assert _mod(VLM_CACHE_PATH) is owned
    assert sys.modules[VLM_CACHE_PATH] is owned
    assert vlm_models.cache is owned


@pytest.mark.parametrize("vendored", sorted(set(EXPECTED) - {VLM_CACHE_PATH}))
def test_the_vendored_files_define_no_owned_body(vendored):
    """An owned name comes into each edited file by import, never from a
    body left in the copy."""
    path = _mod(vendored).__file__
    tree = ast.parse(open(path, encoding="utf-8").read())
    defined = {n.name for n in tree.body
               if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    assert not defined & set(EXPECTED[vendored]), vendored


@pytest.mark.parametrize("vendored", sorted(EXPECTED))
def test_the_upstream_reference_files_are_the_shipped_files(vendored):
    """tests/cache/upstream holds the stock bodies the owned code replaces,
    for the controls. Each file is the one the vendored manifest records."""
    vendor = Path(_mod("gmlx._vendor").__file__).parent
    manifest = json.loads((vendor / "manifest.json").read_text())
    rel = upstream_file(vendored).relative_to(UPSTREAM).as_posix()
    digest = hashlib.sha256(upstream_file(vendored).read_bytes()).hexdigest()
    assert digest == manifest["files"][rel]["upstream"], rel


def test_the_mlx_lm_only_names_stay_in_the_copy():
    lm = _mod(LM_CACHE_PATH)
    for name in ("save_prompt_cache", "load_prompt_cache",
                 "can_trim_prompt_cache", "trim_prompt_cache", "TokenBuffer",
                 "PromptTrieResult", "PromptTrie", "LRUPromptCache"):
        assert getattr(lm, name).__module__ == LM_CACHE_PATH, name
    base = _mod("gmlx._vendor.mlx_lm.models.base")
    assert base.BaseModelArgs.__module__ == "gmlx._vendor.mlx_lm.models.base"


def test_each_create_attention_mask_is_its_own_twin():
    assert owned.create_attention_mask is not owned_attn.create_attention_mask
    for name in (VLM_CACHE_PATH, LM_CACHE_PATH):
        assert _mod(name).create_attention_mask is owned.create_attention_mask
    for name in ("gmlx._vendor.mlx_vlm.models.base",
                 "gmlx._vendor.mlx_lm.models.base"):
        assert _leads_to(_mod(name).create_attention_mask,
                         owned_attn.create_attention_mask)
    cache_args = list(inspect.signature(owned.create_attention_mask).parameters)
    base_args = list(
        inspect.signature(owned_attn.create_attention_mask).parameters)
    assert cache_args[:2] == ["N", "offset"]
    assert base_args[:2] == ["h", "cache"]


def test_vendored_cache_classes_derive_from_the_owned_classes():
    from gmlx.models.hy_v4.model import HyV4KVCache
    from gmlx.models.minimax_m3 import MSAKVCache
    from gmlx.models.qwen4_exp.model import QSAKVCache

    tq = _mod("gmlx._vendor.mlx_vlm.turboquant")
    for name in ("TurboQuantKVCache", "BatchTurboQuantKVCache",
                 "HybridQuantKVCache"):
        assert owned._BaseCache in getattr(tq, name).__mro__, name
    for cls in (MSAKVCache, QSAKVCache, HyV4KVCache,
                _mod("gmlx._vendor.mlx_vlm.models.hy_v4.cache").HyV4KVCache):
        assert owned.KVCache in cls.__mro__, cls
    # No loaded class derives from a cache class of an owned name that is
    # not the owned class.
    strays = []
    for name in OWNED_CLASSES:
        for sub in _all_subclasses(getattr(owned, name)):
            for base in sub.__mro__:
                if base.__name__ in OWNED_CLASSES and \
                        base.__module__ != KV and \
                        base.__module__.startswith(VENDOR):
                    strays.append((sub, base))
    assert not strays, strays


def _all_subclasses(cls):
    todo, seen = [cls], set()
    while todo:
        c = todo.pop()
        for s in c.__subclasses__():
            if s not in seen:
                seen.add(s)
                todo.append(s)
    return seen


def test_borrowed_modules_hold_the_owned_objects():
    cs = _mod("gmlx._vendor.mlx_vlm.speculative.cache_state")
    for name in ("RotatingKVCache", "BatchRotatingKVCache", "CacheList"):
        assert getattr(cs, name) is getattr(owned, name), name
    gen = _mod("gmlx._vendor.mlx_lm.generate")
    for name in ("ArraysCache", "BatchKVCache", "BatchRotatingKVCache",
                 "CacheList", "KVCache", "QuantizedKVCache",
                 "RotatingKVCache"):
        assert getattr(gen, name) is getattr(owned, name), name
    assert gen.cache.KVCache is owned.KVCache
    adapters = _mod("gmlx._vendor.mlx_vlm.apc_adapters")
    assert adapters.CacheMemory is owned.CacheMemory
    assert adapters.cache_nbytes is owned.cache_nbytes
    assert _mod("gmlx._vendor.mlx_vlm.apc_coordinator").CacheMemory is \
        owned.CacheMemory
    assert _leads_to(_mod("gmlx._vendor.mlx_vlm.apc")._cache_nbytes,
                     owned.cache_nbytes)


# Model modules that import the SDPA by name from their package's base. The
# per-architecture decode claims (cascade, sparse) wrap these bindings, and
# the wrappers lead down to the owned SDPA. The copied gemma4 language and
# gemma4_text paths are aliases of the owned modules, whose SDPA is the
# owned dispatch.
_SDPA_BINDINGS = (
    "gmlx._vendor.mlx_lm.models.llama",
)


@pytest.mark.parametrize("module", _SDPA_BINDINGS)
def test_a_model_module_sdpa_leads_to_the_owned_sdpa(module):
    assert _leads_to(_mod(module).scaled_dot_product_attention,
                     owned_attn.scaled_dot_product_attention)


def test_leads_to_follows_a_linked_wrapper_and_refuses_another_function():
    from gmlx.upstream.patching import link

    sdpa = owned_attn.scaled_dot_product_attention

    def wrapper(*args, **kwargs):
        return sdpa(*args, **kwargs)

    assert not _leads_to(wrapper, sdpa)
    assert _leads_to(link(wrapper, sdpa, "_test_flag"), sdpa)
    assert not _leads_to(owned_attn.create_attention_mask, sdpa)


def test_the_apc_tables_key_on_owned_classes(monkeypatch):
    adapters = _mod("gmlx._vendor.mlx_vlm.apc_adapters")
    monkeypatch.setattr(adapters, "_CAPABILITY", dict(adapters._CAPABILITY))
    adapters.register_default_capabilities()
    keys = [k for k in adapters._CAPABILITY if isinstance(k, type)]
    assert owned.KVCache in keys
    assert not [k for k in keys if k.__name__ in OWNED_CLASSES
                and k is not getattr(owned, k.__name__)]
    exact, block = adapters._apc_type_tables()
    assert owned.KVCache in exact and owned.KVCache in block
    assert not [t for t in exact if t.__name__ in OWNED_CLASSES
                and t is not getattr(owned, t.__name__)]


def test_owned_objects_carry_their_true_module():
    for name in VLM_CACHE:
        value = root(getattr(owned, name))
        assert isinstance(value, type) or inspect.isfunction(value), name
        assert value.__module__ == KV, name
    for name in VLM_BASE:
        # Under any gmlx wrapper (the KVarN sweep wraps the owned SDPA).
        value = root(getattr(owned_attn, name))
        assert value.__module__ == ATTN, name


def test_the_package_root_exports_the_owned_make_prompt_cache():
    import gmlx

    assert gmlx.make_prompt_cache is owned.make_prompt_cache


# --- the APC checkpoint class resolver ---


def _no_stock_modules():
    return not [m for m in sys.modules
                if m.partition(".")[0] in BLOCKED]


def test_the_checkpoint_resolver_takes_every_spelling_of_an_owned_class():
    """The owned module, the vendored path and the stock path that a shard
    written by an earlier gmlx records all give the owned class, and no
    stock module is imported."""
    resolve = _mod("gmlx._vendor.mlx_vlm.apc")._resolve_checkpoint_class
    for module in (KV, VLM_CACHE_PATH, _stock(VLM_CACHE_PATH)):
        assert resolve(module, "QuantizedKVCache") is owned.QuantizedKVCache
        assert resolve(module, "KVCache") is owned.KVCache
    tq = _mod("gmlx._vendor.mlx_vlm.turboquant")
    for module in (tq.__name__, _stock(tq.__name__)):
        assert resolve(module, "TurboQuantKVCache") is tq.TurboQuantKVCache
    assert _no_stock_modules()


def test_the_checkpoint_resolver_refuses_the_rest():
    resolve = _mod("gmlx._vendor.mlx_vlm.apc")._resolve_checkpoint_class
    assert resolve(KV, "create_causal_mask") is None
    assert resolve(KV, "f.<locals>.C") is None
    assert resolve("gmlx.cache.kvarn_cache", "KVarNKVCache") is None
    assert resolve("gmlx.cache.kv_policy", "KvQuantPolicy") is None
    assert resolve(_stock(LM_CACHE_PATH), "KVCache") is None
    assert resolve(_stock("gmlx._vendor.mlx_vlm.no_such_module"), "C") is None
    assert resolve("gmlx._vendor", "C") is None
    assert _no_stock_modules()


@pytest.mark.parametrize("spelling", ["owned", "vendored", "stock"])
def test_a_quantized_checkpoint_round_trips_the_disk_tier(
        tmp_path, monkeypatch, spelling):
    """The vendored and stock spellings stand for shards written by earlier
    gmlx releases, which recorded the class under those modules."""
    module = {"owned": KV, "vendored": VLM_CACHE_PATH,
              "stock": _stock(VLM_CACHE_PATH)}[spelling]
    apc = _mod("gmlx._vendor.mlx_vlm.apc")
    mx.random.seed(5)
    src = owned.KVCache()
    k = mx.random.normal((1, 2, 70, 64))
    src.update_and_fetch(k, k * 0.5)
    q = src.to_quantized(group_size=64, bits=8)
    store = apc.DiskBlockStore(tmp_path, namespace="t")
    try:
        with monkeypatch.context() as m:
            m.setattr(owned.QuantizedKVCache, "__module__", module)
            snapshot = apc._DiskExactCacheSnapshot(
                cache_hash=11, token_ids=(1, 2, 3), extra_hash=0,
                prompt_cache=[q])
            path = store.dir / f"exact_{11:032x}{store.SUFFIX}"
            store._write_exact_cache_snapshot(path, snapshot)
        assert owned.QuantizedKVCache.__module__ == KV
        loaded = store.load_exact_cache(11)
        assert loaded is not None
        _ids, _extra, caches = loaded
        back = caches[0]
        assert type(back) is owned.QuantizedKVCache
        assert back.offset == q.offset and back.bits == 8
        for a, b in zip(back.state, q.state):
            for x, y in zip(a, b):
                assert mx.array_equal(x, y).item()
    finally:
        stop = getattr(store, "close", None) or getattr(store, "stop", None)
        if callable(stop):
            stop()
    assert _no_stock_modules()


def test_mlx_lm_make_cache_batches_the_owned_kvcache():
    from types import SimpleNamespace

    from gmlx.models.minimax_m3 import MSAKVCache

    gen = _mod("gmlx._vendor.mlx_lm.generate")
    model = SimpleNamespace(
        layers=[None, None],
        make_cache=lambda: [owned.KVCache(), owned.RotatingKVCache(8)])
    got = gen._make_cache(model, [0, 1], None)
    assert [type(c) for c in got] == [owned.BatchKVCache,
                                      owned.BatchRotatingKVCache]
    model = SimpleNamespace(layers=[None], make_cache=lambda: [MSAKVCache()])
    with pytest.raises(ValueError, match="does not yet support batching"):
        gen._make_cache(model, [0, 1], None)


def test_cache_list_from_state_resolves_registered_gmlx_classes():
    import gmlx.models.deepseek_v4.model as ds4
    import gmlx.models.minimax_m3 as m3
    from gmlx.models.deepseek_v4.cache import DS4PoolingCache

    ds4.ensure_registered()
    m3.ensure_registered()
    assert owned.DS4PoolingCache is DS4PoolingCache
    assert owned.PoolingCache is not DS4PoolingCache
    mx.random.seed(6)
    kv = owned.KVCache()
    kv.update_and_fetch(mx.ones((1, 1, 3, 8)), mx.ones((1, 1, 3, 8)))
    pool = DS4PoolingCache(4)
    pool.update_and_fetch(mx.random.normal((1, 5, 8)))
    msa = m3.MSAKVCache()
    msa.update_and_fetch_msa(mx.ones((1, 1, 3, 8)), mx.ones((1, 1, 3, 8)),
                             mx.ones((1, 1, 3, 4)))
    src = owned.CacheList(kv, pool, msa)
    back = owned.CacheList.from_state(src.state, src.meta_state)
    assert [type(c) for c in back.caches] == [owned.KVCache, DS4PoolingCache,
                                              m3.MSAKVCache]
    assert back.caches[1].pooled.shape == pool.pooled.shape
    assert back.caches[2].ik.shape[-2] == 3


def test_load_prompt_cache_resolves_registered_gmlx_classes(tmp_path):
    import gmlx.models.minimax_m3 as m3

    m3.ensure_registered()
    lm = _mod(LM_CACHE_PATH)
    assert lm.MSAKVCache is m3.MSAKVCache
    kv = owned.KVCache()
    kv.update_and_fetch(mx.ones((1, 1, 3, 8)), mx.ones((1, 1, 3, 8)))
    msa = m3.MSAKVCache()
    msa.update_and_fetch_msa(mx.ones((1, 1, 3, 8)), mx.ones((1, 1, 3, 8)),
                             mx.ones((1, 1, 3, 4)))
    path = str(tmp_path / "prompt.safetensors")
    lm.save_prompt_cache(path, [kv, msa])
    back = lm.load_prompt_cache(path)
    assert [type(c) for c in back] == [owned.KVCache, m3.MSAKVCache]
    assert back[1].offset == 3


# --- the class registry ---


def test_the_registry_modules_are_the_owned_and_the_vendored_lm_cache():
    assert bind.REGISTRY_MODULES == (KV, LM_CACHE_PATH)


def test_the_registry_refuses_a_name_already_taken():
    """A gmlx class never shadows an owned class, or another gmlx class,
    where cache classes resolve by name."""
    from gmlx.models.qwen4_exp.model import QSAKVCache

    shadow = type("PoolingCache", (owned._BaseCache,), {})
    with pytest.raises(ValueError, match="defines 'PoolingCache'"):
        bind.register(shadow)
    assert "PoolingCache" not in bind.registered()
    for where in bind.REGISTRY_MODULES:
        assert getattr(_mod(where), "PoolingCache", None) is not shadow, where
    assert _mod(VLM_CACHE_PATH).PoolingCache is owned.PoolingCache
    bind.register(QSAKVCache)
    bind.register(QSAKVCache)
    twin = type("QSAKVCache", (owned.KVCache,), {})
    with pytest.raises(ValueError, match="is registered to"):
        bind.register(twin)
    for where in bind.REGISTRY_MODULES:
        assert _mod(where).QSAKVCache is QSAKVCache, where


@pytest.mark.parametrize("name", ["LRUPromptCache", "tree_flatten"])
def test_the_registry_refuses_a_name_the_vendored_cache_module_defines(name):
    """The vendored mlx_lm.server holds LRUPromptCache, and
    save_prompt_cache calls tree_flatten, so neither name may change."""
    lm = _mod(LM_CACHE_PATH)
    before = getattr(lm, name)
    cls = type(name, (owned._BaseCache,), {})
    with pytest.raises(ValueError, match=f"{LM_CACHE_PATH} defines '{name}'"):
        bind.register(cls)
    assert getattr(lm, name) is before
    assert name not in bind.registered() and name not in vars(owned)


def test_the_registry_takes_only_cache_classes():
    with pytest.raises(TypeError, match="_BaseCache"):
        bind.register(type("NotACache", (object,), {}))
    assert "NotACache" not in bind.registered()


def test_a_cleared_registry_takes_the_same_class_again(monkeypatch):
    from gmlx.models.minimax_m3 import MSAKVCache

    bind.register(MSAKVCache)
    monkeypatch.setattr(bind, "_registry", {})
    bind.register(MSAKVCache)
    assert bind.registered() == {"MSAKVCache": MSAKVCache}
    twin = type("MSAKVCache", (owned.KVCache,), {})
    monkeypatch.setattr(bind, "_registry", {})
    with pytest.raises(ValueError, match="gmlx.cache.kv_cache defines"):
        bind.register(twin)


_REGISTER_FIRST_CHILD = r"""
import json
import sys

import mlx.core as mx

mx.set_default_device(mx.cpu)
from gmlx.cache import bind
from gmlx.cache.kv_cache import KVCache


class ProbeCache(KVCache):
    pass


bind.register(ProbeCache)
names = ("gmlx._vendor.mlx_vlm.models.cache", "gmlx._vendor.mlx_lm.models.cache")
loaded = [n for n in names if n in sys.modules]
held = {}
for n in names:
    __import__(n)
    held[n] = getattr(sys.modules[n], "ProbeCache", None) is ProbeCache
print(json.dumps({"loaded": loaded, "held": held}))
"""


def test_a_class_registered_before_a_cache_module_loads_is_there():
    r = subprocess.run([sys.executable, "-c", _REGISTER_FIRST_CHILD],
                       capture_output=True, text=True, timeout=600,
                       env=dict(os.environ))
    assert r.returncode == 0, r.stderr[-4000:]
    out = json.loads(r.stdout.strip().splitlines()[-1])
    assert out == {"loaded": [], "held": {VLM_CACHE_PATH: True,
                                          LM_CACHE_PATH: True}}


_REGISTER_A_VENDORED_NAME_CHILD = r"""
import json
import sys

import mlx.core as mx

mx.set_default_device(mx.cpu)
import gmlx.cache.kv_cache as owned
from gmlx.cache import bind

bind.register(type("LRUPromptCache", (owned._BaseCache,), {}))
loaded = "gmlx._vendor.mlx_lm.models.cache" in sys.modules
try:
    import gmlx._vendor.mlx_lm.models.cache  # noqa: F401
    error = None
except ImportError as e:
    error = str(e)
print(json.dumps({"loaded": loaded, "error": error}))
"""


def test_a_vendored_cache_module_that_defines_a_registered_name_fails_to_load():
    r = subprocess.run([sys.executable, "-c", _REGISTER_A_VENDORED_NAME_CHILD],
                       capture_output=True, text=True, timeout=600,
                       env=dict(os.environ))
    assert r.returncode == 0, r.stderr[-4000:]
    out = json.loads(r.stdout.strip().splitlines()[-1])
    assert out["loaded"] is False
    assert "defines 'LRUPromptCache'" in (out["error"] or ""), out


def _msa():
    from gmlx.models.minimax_m3 import MSAKVCache

    c = MSAKVCache()

    def feed(n, start):
        k = mx.full((1, 1, n, 8), float(start))
        c.update_and_fetch_msa(k, k, mx.full((1, 1, n, 4), float(start)))
    return c, feed, 2


def _qsa():
    from gmlx.models.qwen4_exp.model import QSAKVCache

    c = QSAKVCache()

    def feed(n, start):
        k = mx.full((1, 1, n, 8), float(start))
        c.update_and_fetch_qsa(k, k, mx.full((1, n, 4), float(start)))
    return c, feed, 1


def _lengths(state, ik_axis):
    return [state[0].shape[2], state[1].shape[2], state[2].shape[ik_axis]]


@pytest.mark.parametrize("make", [_msa, _qsa], ids=["MSAKVCache",
                                                    "QSAKVCache"])
def test_prefix_cache_reserve_then_state_keeps_every_stream_in_step(make):
    """The inherited reserve pads keys and values only. The index stream
    grows at its own next update, and state slices every stream to
    offset."""
    c, feed, ik_axis = make()
    feed(5, 1)
    assert c.prefix_cache_reserve(600)
    assert c.keys.shape[2] >= 600 > c.ik.shape[ik_axis]
    assert _lengths(c.state, ik_axis) == [5, 5, 5]
    feed(3, 2)
    assert _lengths(c.state, ik_axis) == [8, 8, 8]
    tail = mx.take(c.state[2], mx.arange(5, 8), axis=ik_axis)
    assert (tail == 2.0).all().item()


# --- import orders, each in a child ---

_ORDER_CHILD = r"""
import importlib
import json
import sys

import mlx.core as mx

mx.set_default_device(mx.cpu)
first, expected = json.loads(sys.argv[1]), json.loads(sys.argv[2])
early = None
for name in first:
    mod = importlib.import_module(name)
    if name == "gmlx._vendor.mlx_vlm.models.cache":
        early = mod.KVCache()
for name in ("gmlx._vendor.mlx_vlm.models.cache", "gmlx._vendor.mlx_lm.models.cache",
             "gmlx._vendor.mlx_vlm.models.base", "gmlx._vendor.mlx_lm.models.base",
             "gmlx._vendor.mlx_lm.generate", "gmlx._vendor.mlx_vlm.turboquant",
             "gmlx._vendor.mlx_vlm.models.hy_v4.cache",
             "gmlx._vendor.mlx_vlm.speculative.cache_state",
             "gmlx._vendor.mlx_vlm.apc_adapters"):
    importlib.import_module(name)
from gmlx.upstream.patching import root

owned = importlib.import_module("gmlx.cache.kv_cache")
unbound = []
for vendored, names in expected.items():
    mod = sys.modules[vendored]
    for n, where in names.items():
        target = getattr(importlib.import_module(where), n)
        if root(getattr(mod, n)) is not root(target):
            unbound.append(f"{vendored}.{n}")
derived = {}
for mod_name, cls_name in (("gmlx._vendor.mlx_vlm.turboquant", "TurboQuantKVCache"),
                           ("gmlx._vendor.mlx_vlm.turboquant", "HybridQuantKVCache"),
                           ("gmlx._vendor.mlx_vlm.models.hy_v4.cache", "HyV4KVCache")):
    cls = getattr(sys.modules[mod_name], cls_name)
    derived[f"{mod_name}.{cls_name}"] = owned._BaseCache in cls.__mro__
globals_ok = {}
for mod_name, names in (("gmlx._vendor.mlx_lm.generate", ("KVCache", "BatchKVCache")),
                        ("gmlx._vendor.mlx_vlm.speculative.cache_state",
                         ("RotatingKVCache", "CacheList"))):
    mod = sys.modules[mod_name]
    globals_ok[mod_name] = all(getattr(mod, n) is getattr(owned, n)
                               for n in names)
adapters = sys.modules["gmlx._vendor.mlx_vlm.apc_adapters"]
adapters.register_default_capabilities()
exact, block = adapters._apc_type_tables()
out = {"alias": sys.modules["gmlx._vendor.mlx_vlm.models.cache"] is owned,
       "unbound": unbound, "derived": derived, "globals": globals_ok,
       "exact": sorted(f"{c.__module__}.{c.__name__}" for c in exact),
       "block": sorted(f"{c.__module__}.{c.__name__}" for c in block),
       "eligible": adapters.apc_block_eligible(owned.KVCache()),
       "capability": adapters.resolve_capability(owned.KVCache()).name,
       "stock": sorted(m for m in sys.modules
                       if m.partition(".")[0] in ("mlx_vlm", "mlx_lm"))}
if early is not None:
    out["early"] = type(early) is owned.KVCache
print(json.dumps(out))
"""

_FIRST = {
    "owned": ["gmlx.cache.kv_cache", "gmlx.cache.attention"],
    "vlm-cache": [VLM_CACHE_PATH],
    "vlm-base": ["gmlx._vendor.mlx_vlm.models.base"],
    "lm-cache": [LM_CACHE_PATH],
    "lm-base": ["gmlx._vendor.mlx_lm.models.base"],
    "vlm-model": ["gmlx._vendor.mlx_vlm.models.hy_v4.cache"],
    "turboquant": ["gmlx._vendor.mlx_vlm.turboquant"],
    "lm-generate": ["gmlx._vendor.mlx_lm.generate"],
    "apc": ["gmlx._vendor.mlx_vlm.apc"],
}


@pytest.mark.parametrize("first", sorted(_FIRST))
def test_every_import_order_gives_the_owned_objects(first):
    r = subprocess.run([sys.executable, "-c", _ORDER_CHILD,
                        json.dumps(_FIRST[first]), json.dumps(EXPECTED)],
                       capture_output=True, text=True, timeout=600,
                       env=dict(os.environ))
    assert r.returncode == 0, r.stderr[-4000:]
    out = json.loads(r.stdout.strip().splitlines()[-1])
    assert out["alias"] is True
    assert out["unbound"] == []
    assert out["derived"] and all(out["derived"].values())
    assert all(out["globals"].values())
    kv = KV + "."
    assert out["exact"] == sorted(kv + n for n in (
        "KVCache", "BatchKVCache", "BatchRotatingKVCache",
        "BatchQuantizedKVCache", "RotatingKVCache", "ChunkedKVCache",
        "ArraysCache"))
    assert out["block"] == [kv + "KVCache"]
    assert out["eligible"] is True
    assert out["capability"] == "PAGEABLE"
    assert out["stock"] == []
    if first == "vlm-cache":
        assert out["early"] is True
