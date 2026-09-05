"""Routing truth: where every cache-shape family lands, pinned.

CPU-only, runs in CI. For each family this asserts the full routing
tuple -- upstream mode resolution, ckpt-tier eligibility, and the layout
tags -- so any change to where a family lands (an upstream class-identity
move, a geometry gate change, a new cache class) fails loudly with the
family name instead of silently re-routing to a tier that stores nothing
(the 0.6.4 regression class, and how serve APC died on every hybrid).
"""

from types import SimpleNamespace

from mlx_vlm.apc import model_apc_mode

import pytest

from gmlx.cache.compat import runtime_cache_module
from gmlx.cache.snapshot import ckpt_layout, ckpt_supported

_cache = runtime_cache_module()


def _dense():
    return [_cache.KVCache() for _ in range(3)]


def _gdn():
    out = []
    for kind in ("kv", "arr", "arr", "kv", "arr"):
        out.append(_cache.KVCache() if kind == "kv"
                   else _cache.ArraysCache(size=2))
    return out


def _swa():
    out = []
    for kind in ("kv", "rot", "rot", "kv", "rot"):
        out.append(_cache.KVCache() if kind == "kv"
                   else _cache.RotatingKVCache(max_size=32))
    return out


def _pure_arr():
    return [_cache.ArraysCache(size=2) for _ in range(3)]


def _cachelist():
    return [_cache.CacheList(_cache.ArraysCache(size=2), _cache.KVCache())
            for _ in range(2)]


class MSAKVCache:
    """Stand-in for the minimax-m3 MSA indexer cache: any cache class the
    registry does not recognize must resolve to mode None (APC off with
    the stage-1 loud log), never to a tier that would half-work."""


def _msa():
    return [_cache.KVCache(), MSAKVCache()]


FAMILIES = [
    # (family, stack factory, expected mode, ckpt?, expected tags)
    ("dense-kv", _dense, "block", False, None),
    ("gdn-hybrid", _gdn, "exact", True,
     ("kv", "arr", "arr", "kv", "arr")),
    ("swa", _swa, "exact", True,
     ("kv", "rot:32:0", "rot:32:0", "kv", "rot:32:0")),
    ("pure-arr", _pure_arr, "exact", False, None),
    ("cachelist", _cachelist, "exact", False, None),
    ("msa-indexer", _msa, None, False, None),
]


@pytest.mark.parametrize(
    "family,factory,mode,ckpt,tags",
    FAMILIES, ids=[f[0] for f in FAMILIES])
def test_family_routing_pinned(family, factory, mode, ckpt, tags):
    model = SimpleNamespace(make_cache=factory)
    got_mode = model_apc_mode(model)
    assert got_mode == mode, (
        f"{family}: model_apc_mode moved to {got_mode!r} "
        f"(pinned {mode!r})")
    stack = factory()
    assert ckpt_supported(stack) == ckpt, (
        f"{family}: ckpt_supported flipped to {not ckpt}")
    got_tags = ckpt_layout(stack)
    got_tags = tuple(got_tags) if got_tags is not None else None
    assert got_tags == tags, (
        f"{family}: layout tags moved to {got_tags} (pinned {tags})")


KVARN_TAG = "kvarn:6:6:1024"


def _kvarn_gdn():
    from gmlx.cache.kvarn_cache import KVarNKVCache
    out = []
    for kind in ("kvarn", "arr", "arr", "kvarn", "arr"):
        out.append(KVarNKVCache() if kind == "kvarn"
                   else _cache.ArraysCache(size=2))
    return out


def _kvarn_swa():
    from gmlx.cache.kvarn_cache import KVarNKVCache
    out = []
    for kind in ("kvarn", "rot", "rot", "kvarn", "rot"):
        out.append(KVarNKVCache() if kind == "kvarn"
                   else _cache.RotatingKVCache(max_size=32))
    return out


def _kvarn_three_kind():
    from gmlx.cache.kvarn_cache import KVarNKVCache
    return [KVarNKVCache(), _cache.RotatingKVCache(max_size=32),
            _cache.ArraysCache(size=2), KVarNKVCache(),
            _cache.ArraysCache(size=2)]


# The three-kind (plamo2-shaped) row is validated by these unit rows
# only -- stage-6 live validation runs kvarn+arr and kvarn+rot, so the
# first live plamo2-shaped model is the one that finds anything the unit
# rows missed.
KVARN_FAMILIES = [
    ("kvarn-gdn", _kvarn_gdn, True,
     (KVARN_TAG, "arr", "arr", KVARN_TAG, "arr")),
    ("kvarn-swa", _kvarn_swa, True,
     (KVARN_TAG, "rot:32:0", "rot:32:0", KVARN_TAG, "rot:32:0")),
    ("kvarn-gdn-swa", _kvarn_three_kind, True,
     (KVARN_TAG, "rot:32:0", "arr", KVARN_TAG, "arr")),
]


@pytest.mark.parametrize(
    "family,factory,ckpt,tags",
    KVARN_FAMILIES, ids=[f[0] for f in KVARN_FAMILIES])
def test_kvarn_family_routing_pinned(kvarn_apc_arms, family, factory, ckpt, tags):
    # Production truth needs the kvarn arms installed (serve installs
    # them at boot): the model_apc_mode wrap resolves kvarn stacks to
    # "exact"; bare upstream would say None.

    from mlx_vlm import apc

    model = SimpleNamespace(make_cache=factory)
    got_mode = apc.model_apc_mode(model)
    assert got_mode == "exact", (
        f"{family}: model_apc_mode moved to {got_mode!r}")
    stack = factory()
    assert ckpt_supported(stack) == ckpt, (
        f"{family}: ckpt_supported flipped to {not ckpt}")
    got_tags = tuple(ckpt_layout(stack) or ()) or None
    assert got_tags == tags, (
        f"{family}: layout tags moved to {got_tags} (pinned {tags})")
