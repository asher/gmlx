"""Every public member of each stock cache class exists on its owned twin.

Borrowed mlx-lm and mlx-vlm code reaches the owned classes through the
stock names. At a version bump a stock class can gain a member or a
parameter that borrowed code starts to use, and the drift report on the
owned files cannot see that caller. So for each stock class the binding
replaced, at the pinned versions: every public class member and every
public attribute a new instance sets exists on the owned class, and each
public method and the constructor accept every parameter the stock one
names. CPU.
"""

from __future__ import annotations

import importlib
import inspect

import pytest

from gmlx.cache import bind

_ORIGINS = ("mlx_lm.models.cache", "mlx_vlm.models.cache")

# Constructor arguments that both origins and the owned classes accept.
_ARGS = {
    "RotatingKVCache": ((8,), {}),
    "ArraysCache": ((2,), {}),
    "ChunkedKVCache": ((8,), {}),
    "BatchKVCache": (([0, 1],), {}),
    "BatchRotatingKVCache": ((8, [0, 1]), {}),
    "BufferedRotatingKVCache": ((8,), {}),
    "BatchQuantizedKVCache": (([0, 1],), {}),
    "PoolingCache": ((4,), {}),
    "BatchPoolingCache": ((4, [0, 1]), {}),
    "StaticPrefixKVCache": ((16,), {}),
}


def _pairs():
    for origin in _ORIGINS:
        importlib.import_module(origin)
    return sorted(((old, new) for old, new in bind.replaced()
                   if isinstance(old, type)
                   and old.__module__ in _ORIGINS),
                  key=lambda p: (p[0].__module__, p[0].__name__))


_PAIRS = _pairs()


def _id(pair):
    old, _ = pair
    return f"{old.__module__.split('.')[0]}.{old.__name__}"


def _public(names):
    return sorted(n for n in names if not n.startswith("_"))


def _params(fn):
    try:
        return inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return None


def _lost(stock_fn, owned_fn) -> list[str]:
    """The stock parameters the owned callable does not accept."""
    old, new = _params(stock_fn), _params(owned_fn)
    if old is None or new is None:
        return []
    if any(p.kind is p.VAR_KEYWORD for p in new.values()):
        return []
    skip = (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    return [n for n, p in old.items() if p.kind not in skip and n not in new]


def _instance(cls):
    name = cls.__name__
    if name == "CacheList":
        return cls()
    args, kwargs = _ARGS.get(name, ((), {}))
    return cls(*args, **kwargs)


def test_every_stock_class_has_an_owned_twin():
    got = {(old.__module__, old.__name__) for old, _ in _PAIRS}
    assert ("mlx_lm.models.cache", "KVCache") in got
    assert ("mlx_vlm.models.cache", "BatchQuantizedKVCache") in got
    for old, new in _PAIRS:
        assert new.__module__ == "gmlx.cache.kv_cache", _id((old, new))
        assert new.__name__ == old.__name__


@pytest.mark.parametrize("pair", _PAIRS, ids=_id)
def test_class_members(pair):
    old, new = pair
    missing = [n for n in _public(dir(old)) if not hasattr(new, n)]
    assert not missing, (_id(pair), missing)


@pytest.mark.parametrize("pair", _PAIRS, ids=_id)
def test_method_parameters(pair):
    old, new = pair
    lost = {}
    for n in _public(dir(old)) + ["__init__"]:
        so = inspect.getattr_static(old, n, None)
        sn = inspect.getattr_static(new, n, None)
        so, sn = getattr(so, "__func__", so), getattr(sn, "__func__", sn)
        if isinstance(so, type) or not (callable(so) and callable(sn)):
            continue
        if _lost(so, sn):
            lost[n] = _lost(so, sn)
    assert not lost, (_id(pair), lost)


@pytest.mark.parametrize(
    "pair", [p for p in _PAIRS if p[0].__name__ != "_BaseCache"], ids=_id)
def test_instance_attributes(pair):
    old, new = pair
    stock, owned = _instance(old), _instance(new)
    missing = [n for n in _public(vars(stock))
               if n not in vars(owned) and not hasattr(new, n)]
    assert not missing, (_id(pair), missing)
