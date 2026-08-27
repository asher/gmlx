"""Stamp/salt scoping: both are gated on an actual-conversion probe.

CPU-only. The bug class pinned here: a kvarn-window boot of a
zero-conversion arch (deepseek4's rot+CacheList stack, recurrent_gemma's
arr+rot stack) used to get stamped and its manager salted from the env
alone -- fp16 caches running under the exact tier's kvarn salt, so every
cross-boot lookup cold-missed. The probe counts the classes
convert_prompt_cache actually converts (plain KVCache and ChunkedKVCache
-- the latter so CLI and serve agree on llama4-shaped stacks), and both
the residency salt site and the serve gate read the one stashed answer.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from gmlx.cache.compat import runtime_cache_module
from gmlx.cache.kvarn_apc import (
    _MODE_STAMP,
    apply_kvarn_salt,
    install_kvarn_apc,
    kvarn_entry_salt,
    kvarn_model_converts,
    stamp_model,
)
from gmlx.cache.kvarn_cache import KVarNKVCache, convert_prompt_cache

_cache = runtime_cache_module()
ArraysCache = _cache.ArraysCache
CacheList = _cache.CacheList
KVCache = _cache.KVCache
RotatingKVCache = _cache.RotatingKVCache


class _Args:
    def __init__(self, head_dim=128):
        self.model_type = "llama"
        self.head_dim = head_dim


class _LM:
    def __init__(self, stack, head_dim=128):
        self.args = _Args(head_dim)
        self.layers = [object()] * len(stack)
        self._stack = stack

    def make_cache(self):
        return [f() for f in self._stack]


class _DenseNoMakeCache:
    """Model without make_cache: upstream builds plain KV per layer."""

    def __init__(self):
        self.args = _Args()
        self.layers = [object()] * 2


def _hybrid(head_dim=128):
    return _LM([KVCache, lambda: ArraysCache(size=2)], head_dim)


def _rec_gemma():
    # ckpt-shaped but zero-conversion: no plain KV at all.
    return _LM([lambda: ArraysCache(size=2),
                lambda: RotatingKVCache(max_size=32)])


def _ds4():
    return _LM([lambda: RotatingKVCache(max_size=32),
                lambda: CacheList(ArraysCache(size=2))], head_dim=512)


def _llama4():
    from mlx_lm.models.cache import ChunkedKVCache

    return _LM([lambda: ChunkedKVCache(chunk_size=8192), KVCache])


@pytest.fixture
def _ops_ok(monkeypatch):
    from gmlx.cache import kvarn_sdpa

    monkeypatch.setattr(kvarn_sdpa, "_probe_result", (None,))
    for k in ("GMLX_KVARN", "GMLX_KVARN_BITS", "KV_BITS", "KV_TAIL_TOKENS"):
        monkeypatch.delenv(k, raising=False)


def test_probe_counts_convertible_layers(_ops_ok):
    assert kvarn_model_converts(_hybrid())
    assert kvarn_model_converts(_llama4())
    assert kvarn_model_converts(_DenseNoMakeCache())
    assert not kvarn_model_converts(_rec_gemma())
    assert not kvarn_model_converts(_ds4())
    assert not kvarn_model_converts(_hybrid(head_dim=64))


def test_probe_stashes_on_model(_ops_ok):
    model = _hybrid()
    assert kvarn_model_converts(model)
    model.make_cache = None  # a re-probe would now blow up
    assert kvarn_model_converts(model)


def test_convert_prompt_cache_converts_chunked(_ops_ok):
    from mlx_lm.models.cache import ChunkedKVCache as LmChunked
    from mlx_vlm.models.cache import ChunkedKVCache as VlmChunked

    caches = [LmChunked(chunk_size=8192), VlmChunked(chunk_size=8192),
              KVCache(), ArraysCache(size=2)]
    n = convert_prompt_cache(caches, k_bits=6, v_bits=6, tail_tokens=1024)
    assert n == 3
    assert all(type(c) is KVarNKVCache for c in caches[:3])
    assert type(caches[3]) is ArraysCache


def test_stamp_covers_wrapper_and_language_model(_ops_ok):
    install_kvarn_apc()
    from mlx_vlm import apc

    wrapper = SimpleNamespace(language_model=_hybrid())
    stamp_model(wrapper)
    assert getattr(wrapper, _MODE_STAMP, False)
    assert getattr(wrapper.language_model, _MODE_STAMP, False)
    assert apc.model_apc_mode(wrapper) == "exact"
    assert apc.model_apc_mode(wrapper.language_model) == "exact"
    # Check-down: a wrapper whose language model carries the stamp still
    # resolves exact when the wrapper itself is passed.
    lm_only = SimpleNamespace(language_model=_hybrid())
    stamp_model(lm_only.language_model)
    assert not getattr(lm_only, _MODE_STAMP, False)
    assert apc.model_apc_mode(lm_only) == "exact"


def test_salt_gated_on_conversion(_ops_ok, monkeypatch):
    monkeypatch.setenv("KV_QUANT_SCHEME", "kvarn")
    salt = kvarn_entry_salt()
    assert salt != 0
    man = SimpleNamespace(_exact_extra_salt=0)
    apply_kvarn_salt(man, _hybrid())
    assert man._exact_extra_salt == salt
    for model in (_rec_gemma(), _ds4()):
        man = SimpleNamespace(_exact_extra_salt=0)
        apply_kvarn_salt(man, model)
        assert man._exact_extra_salt == 0, type(model).__name__


def test_salt_zero_outside_kvarn_window(_ops_ok, monkeypatch):
    monkeypatch.delenv("KV_QUANT_SCHEME", raising=False)
    man = SimpleNamespace(_exact_extra_salt=0)
    apply_kvarn_salt(man, _hybrid())
    assert man._exact_extra_salt == 0


def test_build_apc_manager_never_salts(_ops_ok, monkeypatch):
    # The build runs pre-load with nothing to probe; the salt belongs to
    # residency's post-load pairing only.
    from gmlx.cache.apc_manager import build_apc_manager

    monkeypatch.setenv("KV_QUANT_SCHEME", "kvarn")
    monkeypatch.setenv("GMLX_APC_ENABLED", "1")
    monkeypatch.delenv("APC_DISK_PATH", raising=False)
    mgr = build_apc_manager()
    assert mgr is not None
    assert mgr._exact_extra_salt == 0


def _fake_ar():
    class FakeBG:
        def __init__(self, model, *args, **kwargs):
            self.model = model
            self.init_kwargs = kwargs

    return SimpleNamespace(BatchGenerator=FakeBG)


def test_gated_init_zero_conversion_keeps_stock_path(_ops_ok):
    # recurrent_gemma-shaped: ckpt-shaped but kvarn converts nothing --
    # no stamp, kv_bits and the manager untouched, so fp16 ckpt records
    # keep storing exactly as on a stock boot.
    from gmlx.cache.kvarn_serve import _install_apc_gate

    install_kvarn_apc()
    ar = _fake_ar()
    _install_apc_gate(ar)
    model = _rec_gemma()
    manager = object()
    bg = ar.BatchGenerator(model, apc_manager=manager,
                           kv_quant_scheme="kvarn", kv_bits=8)
    assert not getattr(model, _MODE_STAMP, False)
    assert bg.init_kwargs["kv_bits"] == 8
    assert bg.init_kwargs["apc_manager"] is manager


def test_gated_init_converting_model_stamps(_ops_ok):
    from gmlx.cache.kvarn_serve import _install_apc_gate

    install_kvarn_apc()
    ar = _fake_ar()
    _install_apc_gate(ar)
    model = _hybrid()
    bg = ar.BatchGenerator(model, apc_manager=object(),
                           kv_quant_scheme="kvarn", kv_bits=8)
    assert getattr(model, _MODE_STAMP, False)
    assert bg.init_kwargs["kv_bits"] is None
