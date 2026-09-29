"""gmlx binds its own cache classes and SDPA helpers under the stock names.

``import gmlx`` puts a finder on ``sys.meta_path`` that rebinds the mapped
names of mlx_vlm.models.cache, mlx_lm.models.cache and the two base
modules as each one executes (gmlx.cache.bind). A stock module that
loaded first takes the late path. These tests hold the binding in this
process, in each import order in a child, and at the places that keep a
class they saw first: borrowed module globals, the APC tables, the
checkpoint class resolver and ``CacheList.from_state``. The class registry
that makes gmlx's own cache classes resolvable by name is held here too.
"""

from __future__ import annotations

import importlib
import inspect
import json
import os
import subprocess
import sys

import mlx.core as mx
import pytest

import gmlx.cache.attention as owned_attn
import gmlx.cache.kv_cache as owned
from gmlx.cache import bind
from gmlx.upstream.patching import _walk, root
from owned_names import ATTN, EXPECTED, KV, VLM_BASE, VLM_CACHE

pytest.importorskip("mlx_vlm.models.cache")


def _mod(name):
    return importlib.import_module(name)


def _leads_to(value, target) -> bool:
    """``value`` and ``target`` are the same object once any gmlx wrapper
    over either is followed down (the KVarN sweep also wraps the owned
    SDPA in gmlx.cache.attention)."""
    return root(value) is root(target)


@pytest.mark.parametrize("stock", sorted(EXPECTED))
def test_every_shared_name_is_the_owned_object(stock):
    mod = _mod(stock)
    for name, where in EXPECTED[stock].items():
        value = getattr(mod, name)
        assert _leads_to(value, getattr(_mod(where), name)), (
            stock, name, [(getattr(f, "__module__", None),
                           getattr(f, "__qualname__", None))
                          for f in _walk(value)])


def test_the_binding_map_names_exactly_the_shared_names():
    assert bind.MAP == {stock: {n: (where, n) for n, where in names.items()}
                        for stock, names in EXPECTED.items()}


def test_the_mlx_lm_only_names_stay_stock():
    lm = _mod("mlx_lm.models.cache")
    for name in ("save_prompt_cache", "load_prompt_cache",
                 "can_trim_prompt_cache", "trim_prompt_cache", "TokenBuffer",
                 "PromptTrieResult", "PromptTrie", "LRUPromptCache"):
        assert getattr(lm, name).__module__ == "mlx_lm.models.cache", name
    base = _mod("mlx_lm.models.base")
    assert base.BaseModelArgs.__module__ == "mlx_lm.models.base"


def test_each_create_attention_mask_binds_to_its_own_twin():
    assert owned.create_attention_mask is not owned_attn.create_attention_mask
    for name in ("mlx_vlm.models.cache", "mlx_lm.models.cache"):
        assert _mod(name).create_attention_mask is owned.create_attention_mask
    for name in ("mlx_vlm.models.base", "mlx_lm.models.base"):
        assert _leads_to(_mod(name).create_attention_mask,
                         owned_attn.create_attention_mask)
    cache_args = list(inspect.signature(owned.create_attention_mask).parameters)
    base_args = list(
        inspect.signature(owned_attn.create_attention_mask).parameters)
    assert cache_args[:2] == ["N", "offset"]
    assert base_args[:2] == ["h", "cache"]


def test_no_class_derives_from_a_replaced_stock_class():
    _mod("mlx_vlm.turboquant")
    _mod("mlx_vlm.models.qwen4_exp.language")
    _mod("mlx_vlm.models.hy_v4.cache")
    from gmlx.models.hy_v4.model import HyV4KVCache
    from gmlx.models.minimax_m3 import MSAKVCache
    from gmlx.models.qwen4_exp.model import QSAKVCache

    stock = {id(old) for old, _new in bind.replaced()
             if isinstance(old, type)}
    for old, _new in bind.replaced():
        if isinstance(old, type):
            strays = [s for s in old.__subclasses__() if id(s) not in stock]
            assert not strays, (old, strays)
    tq = _mod("mlx_vlm.turboquant")
    for name in ("TurboQuantKVCache", "BatchTurboQuantKVCache",
                 "HybridQuantKVCache"):
        assert owned._BaseCache in getattr(tq, name).__mro__, name
    for cls in (MSAKVCache, QSAKVCache, HyV4KVCache,
                _mod("mlx_vlm.models.qwen4_exp.language").QSAKVCache,
                _mod("mlx_vlm.models.hy_v4.cache").HyV4KVCache):
        assert owned.KVCache in cls.__mro__, cls


def test_borrowed_modules_hold_the_owned_objects():
    cs = _mod("mlx_vlm.speculative.cache_state")
    for name in ("RotatingKVCache", "BatchRotatingKVCache", "CacheList"):
        assert getattr(cs, name) is getattr(owned, name), name
    gen = _mod("mlx_lm.generate")
    for name in ("ArraysCache", "BatchKVCache", "BatchRotatingKVCache",
                 "CacheList", "KVCache", "QuantizedKVCache",
                 "RotatingKVCache"):
        assert getattr(gen, name) is getattr(owned, name), name
    assert gen.cache.KVCache is owned.KVCache
    adapters = _mod("mlx_vlm.apc_adapters")
    assert adapters.CacheMemory is owned.CacheMemory
    assert adapters.cache_nbytes is owned.cache_nbytes
    assert _mod("mlx_vlm.apc_coordinator").CacheMemory is owned.CacheMemory
    assert _leads_to(_mod("mlx_vlm.apc")._cache_nbytes, owned.cache_nbytes)


def test_the_apc_tables_key_on_owned_classes(monkeypatch):
    adapters = _mod("mlx_vlm.apc_adapters")
    monkeypatch.setattr(adapters, "_CAPABILITY", dict(adapters._CAPABILITY))
    adapters.register_default_capabilities()
    stock = {id(old) for old, _new in bind.replaced()}
    keys = [k for k in adapters._CAPABILITY if isinstance(k, type)]
    assert owned.KVCache in keys
    assert not [k for k in keys if id(k) in stock]
    exact, block = adapters._apc_type_tables()
    assert owned.KVCache in exact and owned.KVCache in block
    assert not [t for t in exact if id(t) in stock]


def test_owned_objects_carry_their_true_module():
    for name in VLM_CACHE:
        value = root(getattr(owned, name))
        assert isinstance(value, type) or inspect.isfunction(value), name
        assert value.__module__ == KV, name
    for name in VLM_BASE:
        # Under any gmlx wrapper (the KVarN sweep wraps the owned SDPA).
        value = root(getattr(owned_attn, name))
        assert value.__module__ == ATTN, name


@pytest.fixture
def claim(monkeypatch):
    apc = _mod("mlx_vlm.apc")
    monkeypatch.setattr(apc, "_resolve_checkpoint_class",
                        apc._resolve_checkpoint_class)
    from gmlx.cache.checkpoint_claim import install_checkpoint_class_claim

    assert install_checkpoint_class_claim()
    assert install_checkpoint_class_claim()
    return apc


def test_the_checkpoint_claim_resolves_under_both_names(claim):
    resolve = claim._resolve_checkpoint_class
    for module in ("gmlx.cache.kv_cache", "mlx_vlm.models.cache"):
        assert resolve(module, "QuantizedKVCache") is owned.QuantizedKVCache
        assert resolve(module, "KVCache") is owned.KVCache
    assert resolve("gmlx.cache.kv_cache", "create_causal_mask") is None
    assert resolve("gmlx.cache.kv_cache", "f.<locals>.C") is None
    assert resolve("gmlx.cache.kvarn_cache", "KVarNKVCache") is None
    assert resolve("gmlx.cache.kv_policy", "KvQuantPolicy") is None


@pytest.mark.parametrize("spelling", ["gmlx.cache.kv_cache",
                                      "mlx_vlm.models.cache"])
def test_a_quantized_checkpoint_round_trips_the_disk_tier(
        claim, tmp_path, monkeypatch, spelling):
    """The second spelling stands for a shard written before the binding,
    which recorded the class under mlx_vlm.models.cache."""
    apc = claim
    mx.random.seed(5)
    src = owned.KVCache()
    k = mx.random.normal((1, 2, 70, 64))
    src.update_and_fetch(k, k * 0.5)
    q = src.to_quantized(group_size=64, bits=8)
    store = apc.DiskBlockStore(tmp_path, namespace="t")
    try:
        with monkeypatch.context() as m:
            m.setattr(owned.QuantizedKVCache, "__module__", spelling)
            snapshot = apc._DiskExactCacheSnapshot(
                cache_hash=11, token_ids=(1, 2, 3), extra_hash=0,
                prompt_cache=[q])
            path = store.dir / f"exact_{11:032x}{store.SUFFIX}"
            store._write_exact_cache_snapshot(path, snapshot)
        assert owned.QuantizedKVCache.__module__ == "gmlx.cache.kv_cache"
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


def test_mlx_lm_make_cache_batches_the_owned_kvcache():
    from types import SimpleNamespace

    from gmlx.models.minimax_m3 import MSAKVCache

    gen = _mod("mlx_lm.generate")
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


# --- the class registry ---


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
    assert _mod("mlx_vlm.models.cache").PoolingCache is owned.PoolingCache
    bind.register(QSAKVCache)
    bind.register(QSAKVCache)
    twin = type("QSAKVCache", (owned.KVCache,), {})
    with pytest.raises(ValueError, match="is registered to"):
        bind.register(twin)
    for where in bind.REGISTRY_MODULES:
        assert _mod(where).QSAKVCache is QSAKVCache, where


@pytest.mark.parametrize("name", ["LRUPromptCache", "tree_flatten"])
def test_the_registry_refuses_a_name_a_stock_cache_module_defines(name):
    """mlx_lm.server holds the stock LRUPromptCache, and save_prompt_cache
    calls tree_flatten, so neither name may change."""
    lm = _mod("mlx_lm.models.cache")
    before = getattr(lm, name)
    cls = type(name, (owned._BaseCache,), {})
    with pytest.raises(ValueError, match=f"mlx_lm.models.cache defines '{name}'"):
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
import gmlx  # noqa: F401
from gmlx.models.qwen4_exp.model import QSAKVCache, ensure_registered

ensure_registered()
loaded = "mlx_vlm.models.cache" in sys.modules
import mlx_vlm.models.cache as vlm_cache

print(json.dumps({"loaded": loaded,
                  "held": getattr(vlm_cache, "QSAKVCache", None)
                  is QSAKVCache}))
"""


def test_a_class_registered_before_the_vlm_cache_module_loads_is_there():
    r = subprocess.run([sys.executable, "-c", _REGISTER_FIRST_CHILD],
                       capture_output=True, text=True, timeout=600,
                       env=dict(os.environ))
    assert r.returncode == 0, r.stderr[-4000:]
    out = json.loads(r.stdout.strip().splitlines()[-1])
    assert out == {"loaded": False, "held": True}


_REGISTER_A_STOCK_NAME_CHILD = r"""
import json
import sys

import mlx.core as mx

mx.set_default_device(mx.cpu)
import gmlx  # noqa: F401
import gmlx.cache.kv_cache as owned
from gmlx.cache import bind

bind.register(type("LRUPromptCache", (owned._BaseCache,), {}))
loaded = "mlx_lm.models.cache" in sys.modules
try:
    import mlx_lm.models.cache  # noqa: F401
    error = None
except ImportError as e:
    error = str(e)
print(json.dumps({"loaded": loaded, "error": error}))
"""


def test_a_stock_cache_module_that_defines_a_registered_name_fails_to_load():
    r = subprocess.run([sys.executable, "-c", _REGISTER_A_STOCK_NAME_CHILD],
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
mode, expected = sys.argv[1], json.loads(sys.argv[2])
early = None
if mode == "gmlx-first":
    import gmlx  # noqa: F401
    for name in ("mlx_vlm.models.cache", "mlx_lm.models.cache",
                 "mlx_vlm.models.base", "mlx_lm.models.base",
                 "mlx_lm.generate", "mlx_vlm.turboquant"):
        importlib.import_module(name)
elif mode in ("vlm-first", "vlm-model-first"):
    import mlx_vlm  # noqa: F401
    if mode == "vlm-model-first":
        importlib.import_module("mlx_vlm.models.qwen4_exp.language")
        importlib.import_module("mlx_vlm.models.hy_v4.cache")
    # A served host fills the APC type tables and the capability table
    # with the stock classes.
    adapters = importlib.import_module("mlx_vlm.apc_adapters")
    adapters._apc_type_tables()
    adapters.register_default_capabilities()
    import gmlx  # noqa: F401
elif mode == "lm-first":
    importlib.import_module("mlx_lm.generate")
    import gmlx  # noqa: F401
elif mode == "instance-first":
    import mlx_vlm.models.cache as c
    early = c.KVCache()
    import gmlx  # noqa: F401
from gmlx.cache import bind
from gmlx.upstream.patching import root

owned = importlib.import_module("gmlx.cache.kv_cache")
unbound = []
for stock, names in expected.items():
    mod = sys.modules.get(stock)
    if mod is None:
        continue
    for n, where in names.items():
        target = getattr(importlib.import_module(where), n)
        if root(getattr(mod, n)) is not root(target):
            unbound.append(f"{stock}.{n}")
stock_ids = {id(o) for o, _ in bind.replaced() if isinstance(o, type)}
strays = [f"{s.__module__}.{s.__qualname__}"
          for o, _ in bind.replaced() if isinstance(o, type)
          for s in o.__subclasses__() if id(s) not in stock_ids]
rebased = {}
for mod_name, cls_name in (("mlx_vlm.turboquant", "TurboQuantKVCache"),
                           ("mlx_vlm.turboquant", "HybridQuantKVCache"),
                           ("mlx_vlm.models.qwen4_exp.language", "QSAKVCache"),
                           ("mlx_vlm.models.hy_v4.cache", "HyV4KVCache")):
    mod = sys.modules.get(mod_name)
    if mod is not None:
        cls = getattr(mod, cls_name)
        rebased[f"{mod_name}.{cls_name}"] = owned._BaseCache in cls.__mro__
globals_ok = {}
for mod_name, names in (("mlx_lm.generate", ("KVCache", "BatchKVCache")),
                        ("mlx_vlm.speculative.cache_state",
                         ("RotatingKVCache", "CacheList"))):
    mod = sys.modules.get(mod_name)
    if mod is not None:
        globals_ok[mod_name] = all(getattr(mod, n) is getattr(owned, n)
                                   for n in names)
out = {"late": list(bind.late_modules), "unbound": unbound,
       "strays": strays, "rebased": rebased, "globals": globals_ok}
adapters = sys.modules.get("mlx_vlm.apc_adapters")
if adapters is not None:
    def names(classes):
        return sorted(f"{c.__module__}.{c.__name__}" for c in classes)

    exact, block = adapters._apc_type_tables()
    out["apc"] = {
        "exact": names(exact), "block": names(block),
        "eligible": adapters.apc_block_eligible(owned.KVCache()),
        "capability": adapters.resolve_capability(owned.KVCache()).name,
        "stock_keys": names(k for k in adapters._CAPABILITY
                            if isinstance(k, type)
                            and k.__module__.startswith(("mlx_vlm.models.cache",
                                                         "mlx_lm.models.cache")))}
if early is not None:
    fresh = sys.modules["mlx_vlm.models.cache"].KVCache()
    out["early"] = [type(early).__module__, type(fresh).__module__]
print(json.dumps(out))
"""


def _order(mode):
    r = subprocess.run([sys.executable, "-c", _ORDER_CHILD, mode,
                        json.dumps(EXPECTED)],
                       capture_output=True, text=True, timeout=600,
                       env=dict(os.environ))
    assert r.returncode == 0, r.stderr[-4000:]
    return json.loads(r.stdout.strip().splitlines()[-1])


def test_gmlx_first_binds_every_stock_module_as_it_loads():
    out = _order("gmlx-first")
    assert out["late"] == []
    assert out["unbound"] == [] and out["strays"] == []
    assert all(out["rebased"].values()) and out["rebased"]
    assert "mlx_lm.generate" in out["globals"]
    assert all(out["globals"].values())


@pytest.mark.parametrize("mode", ["vlm-first", "vlm-model-first"])
def test_vlm_first_takes_the_late_path_and_rebases_subclasses(mode):
    out = _order(mode)
    assert out["late"] == ["mlx_vlm.models.cache", "mlx_vlm.models.base"]
    assert out["unbound"] == [] and out["strays"] == []
    want = {"mlx_vlm.turboquant.TurboQuantKVCache",
            "mlx_vlm.turboquant.HybridQuantKVCache"}
    if mode == "vlm-model-first":
        want |= {"mlx_vlm.models.qwen4_exp.language.QSAKVCache",
                 "mlx_vlm.models.hy_v4.cache.HyV4KVCache"}
    assert set(out["rebased"]) == want and all(out["rebased"].values())
    # The type tables filled before gmlx refill with the owned classes.
    kv = "gmlx.cache.kv_cache."
    assert out["apc"]["exact"] == sorted(kv + n for n in (
        "KVCache", "BatchKVCache", "BatchRotatingKVCache",
        "BatchQuantizedKVCache", "RotatingKVCache", "ChunkedKVCache",
        "ArraysCache"))
    assert out["apc"]["block"] == [kv + "KVCache"]
    assert out["apc"]["eligible"] is True
    assert out["apc"]["capability"] == "PAGEABLE"
    assert out["apc"]["stock_keys"] == []


def test_mlx_lm_first_takes_the_late_path():
    out = _order("lm-first")
    assert out["late"] == ["mlx_lm.models.cache", "mlx_lm.models.base"]
    assert out["unbound"] == [] and out["strays"] == []
    assert "mlx_lm.generate" in out["globals"]
    assert all(out["globals"].values())


def test_an_instance_made_before_the_late_path_keeps_its_class():
    out = _order("instance-first")
    assert out["early"] == ["mlx_vlm.models.cache", "gmlx.cache.kv_cache"]
    assert out["unbound"] == []
