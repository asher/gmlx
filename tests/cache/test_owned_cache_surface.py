"""Every stock cache member and every stock call holds on the owned twin.

Borrowed mlx-lm and mlx-vlm code reaches the owned classes and functions
through the stock names. At a version bump a stock class or function can
gain a member or a parameter that borrowed code starts to use, and the
drift report on the owned files cannot see that caller. So for each stock
class and function the binding replaced, at the pinned versions: every
class member and every attribute a new instance sets exists on the owned
class, and each method, constructor and function takes the stock calls.
A parameter both signatures name keeps its stock kind, position and
required status, and one only the owned signature names has a default.
Calls built from the stock signature, by position and by keyword, with
and without the optional parameters, must bind each value to the owned
parameter of its stock name or to the owned ``*args`` or ``**kwargs``.
CPU.
"""

from __future__ import annotations

import importlib
import inspect

import pytest

from gmlx.cache import bind

_ORIGINS = ("gmlx._vendor.mlx_lm.models.cache", "gmlx._vendor.mlx_vlm.models.cache",
            "gmlx._vendor.mlx_lm.models.base", "gmlx._vendor.mlx_vlm.models.base")

# Owned signatures that differ from the stock one on purpose, by stock
# module and qualified name, with the reason. The parameter tests skip
# each one, and test_each_listed_difference_still_differs fails once the
# signatures agree again.
_DIFFERS: dict[str, str] = {}

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

_P = inspect.Parameter
_VAR = (_P.VAR_POSITIONAL, _P.VAR_KEYWORD)
_POSITIONAL = (_P.POSITIONAL_ONLY, _P.POSITIONAL_OR_KEYWORD)


def _replaced(kind):
    for origin in _ORIGINS:
        importlib.import_module(origin)
    return sorted(((old, new) for old, new in bind.replaced()
                   if kind(old)
                   and getattr(old, "__module__", None) in _ORIGINS),
                  key=lambda p: (p[0].__module__, p[0].__name__))


_PAIRS = _replaced(lambda old: isinstance(old, type))
_FUNCS = _replaced(inspect.isfunction)


def _id(pair):
    old, _ = pair
    return f"{old.__module__}.{old.__name__}"


def _signature(fn):
    try:
        return inspect.signature(fn)
    except (TypeError, ValueError):
        return None


def _required(p) -> bool:
    return p.kind not in _VAR and p.default is p.empty


def _place(sig, name):
    """(kind, position among the positional parameters, required)."""
    positional = [p.name for p in sig.parameters.values()
                  if p.kind in _POSITIONAL]
    p = sig.parameters[name]
    pos = positional.index(name) if p.kind in _POSITIONAL else None
    return p.kind.name, pos, _required(p)


class _Arg:
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return self.name


def _calls(sig):
    """Calls the stock signature takes: each parameter by position where
    it can be, then by keyword, with all parameters and with only the
    required ones. A stock ``*args`` or ``**kwargs`` takes one more."""
    params = [p for p in sig.parameters.values() if p.kind not in _VAR]
    kinds = {p.kind for p in sig.parameters.values()}
    out = []
    for only_required in (False, True):
        use = [p for p in params if _required(p) or not only_required]
        for by_position in (True, False):
            args, kwargs = [], {}
            for p in use:
                if p.kind is _P.POSITIONAL_ONLY or (
                        by_position and p.kind is _P.POSITIONAL_OR_KEYWORD):
                    args.append(_Arg(p.name))
                else:
                    kwargs[p.name] = _Arg(p.name)
            all_by_position = len(args) == sum(
                p.kind in _POSITIONAL for p in params)
            if _P.VAR_POSITIONAL in kinds and all_by_position:
                args.append(_Arg("*"))
            if _P.VAR_KEYWORD in kinds:
                kwargs["_extra"] = _Arg("**")
            out.append((tuple(args), kwargs))
    return out


def _lands_alike(sig, args, kwargs) -> str | None:
    """None when the owned signature binds the call with every value on
    its stock name, else what goes wrong."""
    try:
        bound = sig.bind(*args, **kwargs)
    except TypeError as e:
        return f"refuses {args} {kwargs}: {e}"
    for name, value in bound.arguments.items():
        kind = sig.parameters[name].kind
        if kind is _P.VAR_POSITIONAL:
            continue  # a stock value left to *args keeps its position
        if kind is _P.VAR_KEYWORD:
            wrong = [k for k, v in value.items()
                     if v.name != ("**" if k == "_extra" else k)]
        else:
            wrong = [name] if value.name != name else []
        if wrong:
            return f"binds {args} {kwargs} as {dict(bound.arguments)}"
    return None


def _departures(stock_fn, owned_fn) -> list[str]:
    """How the owned signature departs from the stock one."""
    old = _signature(stock_fn)
    if old is None:
        return []
    new = _signature(owned_fn)
    if new is None:
        return ["the owned signature cannot be read"]
    out = []
    for name, p in old.parameters.items():
        q = new.parameters.get(name)
        if p.kind in _VAR:
            if not any(r.kind is p.kind for r in new.parameters.values()):
                out.append(f"no {p.kind.description} parameter")
        elif q is not None and q.kind not in _VAR and \
                _place(old, name) != _place(new, name):
            out.append(f"{name}: stock {_place(old, name)}, "
                       f"owned {_place(new, name)}")
    out += [f"{name}: required, and stock has no such parameter"
            for name, q in new.parameters.items()
            if name not in old.parameters and _required(q)]
    for args, kwargs in _calls(old):
        wrong = _lands_alike(new, args, kwargs)
        if wrong:
            out.append(wrong)
    return out


def _methods(old, new):
    """(name, stock callable, owned callable) for each method of ``old``."""
    for n in sorted(dir(old)):
        so = inspect.getattr_static(old, n, None)
        sn = inspect.getattr_static(new, n, None)
        so, sn = getattr(so, "__func__", so), getattr(sn, "__func__", sn)
        if isinstance(so, type) or not (callable(so) and callable(sn)):
            continue
        yield n, so, sn


def _instance(cls):
    name = cls.__name__
    if name == "CacheList":
        return cls()
    args, kwargs = _ARGS.get(name, ((), {}))
    return cls(*args, **kwargs)


def test_every_stock_class_has_an_owned_twin():
    got = {(old.__module__, old.__name__) for old, _ in _PAIRS}
    assert ("gmlx._vendor.mlx_lm.models.cache", "KVCache") in got
    assert ("gmlx._vendor.mlx_vlm.models.cache", "BatchQuantizedKVCache") in got
    for old, new in _PAIRS:
        assert new.__module__ == bind.KV, _id((old, new))
        assert new.__name__ == old.__name__


def test_every_stock_function_has_an_owned_twin():
    got = {(old.__module__, old.__name__) for old, _ in _FUNCS}
    assert ("gmlx._vendor.mlx_vlm.models.base", "scaled_dot_product_attention") in got
    assert ("gmlx._vendor.mlx_lm.models.base", "quantized_scaled_dot_product_attention") in got
    assert ("gmlx._vendor.mlx_vlm.models.cache", "make_prompt_cache") in got
    for old, new in _FUNCS:
        assert new.__module__ in (bind.KV, bind.ATTN), _id((old, new))
        assert new.__name__ == old.__name__


@pytest.mark.parametrize("pair", _PAIRS, ids=_id)
def test_class_members(pair):
    old, new = pair
    missing = [n for n in dir(old) if not hasattr(new, n)]
    assert not missing, (_id(pair), missing)


@pytest.mark.parametrize("pair", _PAIRS, ids=_id)
def test_method_parameters(pair):
    old, new = pair
    wrong = {}
    for n, so, sn in _methods(old, new):
        got = _departures(so, sn)
        if got and f"{_id(pair)}.{n}" not in _DIFFERS:
            wrong[n] = got
    assert not wrong, (_id(pair), wrong)


@pytest.mark.parametrize("pair", _FUNCS, ids=_id)
def test_function_parameters(pair):
    old, new = pair
    if _id(pair) in _DIFFERS:
        pytest.skip(_DIFFERS[_id(pair)])
    got = _departures(old, new)
    assert not got, (_id(pair), got)


def test_each_listed_difference_still_differs():
    got = {_id(p): p for p in _FUNCS}
    for old, new in _PAIRS:
        for n, so, sn in _methods(old, new):
            got[f"{_id((old, new))}.{n}"] = (so, sn)
    for name in _DIFFERS:
        assert name in got and _departures(*got[name]), name


@pytest.mark.parametrize(
    "pair", [p for p in _PAIRS if p[0].__name__ != "_BaseCache"], ids=_id)
def test_instance_attributes(pair):
    old, new = pair
    stock, owned = _instance(old), _instance(new)
    missing = [n for n in vars(stock)
               if n not in vars(owned) and not hasattr(new, n)]
    assert not missing, (_id(pair), missing)
