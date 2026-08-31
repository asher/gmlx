"""kvarn serve installers: _make_cache batch construction, the APC gate,
the safe-quantization scheme arm, and the cascade declines."""

from __future__ import annotations

import numpy as np
import pytest

import mlx.core as mx

pytest.importorskip("mlx_vlm.generate.ar")

from mlx_vlm.generate import ar  # noqa: E402
from mlx_vlm.models.cache import (  # noqa: E402
    BatchKVCache,
    BatchQuantizedKVCache,
    KVCache,
)
from mlx_vlm.server import generation as gen  # noqa: E402

from gmlx.cache import kvarn_serve  # noqa: E402
from gmlx.cache.kvarn_cache import BatchKVarNKVCache  # noqa: E402

_NEEDS_GPU = pytest.mark.skipif(
    mx.default_device() != mx.gpu,
    reason="kvarn kernels are Metal-only; needs the GPU device",
)


class _Args:
    def __init__(self, **kw):
        self.model_type = kw.pop("model_type", "llama")
        self.head_dim = kw.pop("head_dim", 128)
        for k, v in kw.items():
            setattr(self, k, v)


class _LayersLM:
    def __init__(self, n=3, **kw):
        self.args = _Args(**kw)
        self.layers = [object()] * n

    def make_cache(self):
        return [KVCache() for _ in self.layers]


@pytest.fixture
def _ops_ok(monkeypatch):
    from gmlx.cache import kvarn_sdpa

    monkeypatch.setattr(kvarn_sdpa, "_probe_result", (None,))
    monkeypatch.delenv("GMLX_KVARN", raising=False)
    monkeypatch.delenv("GMLX_KVARN_BITS", raising=False)


@pytest.fixture
def restorable(monkeypatch):
    monkeypatch.setattr(ar, "_make_cache", ar._make_cache)
    monkeypatch.setattr(gen, "_make_cache", gen._make_cache)
    monkeypatch.setattr(ar.BatchGenerator, "__init__", ar.BatchGenerator.__init__)
    monkeypatch.setattr(
        ar.PromptProcessingBatch,
        "__init__",
        ar.PromptProcessingBatch.__init__,
    )
    monkeypatch.delenv("KV_BITS", raising=False)
    monkeypatch.delenv("KV_TAIL_TOKENS", raising=False)
    return monkeypatch


def _install(monkeypatch):
    monkeypatch.setenv("KV_QUANT_SCHEME", "kvarn")
    kvarn_serve.install_kvarn_serve()


def test_install_idempotent(restorable):
    kvarn_serve.install_kvarn_serve()
    wrapped = ar._make_cache
    wrapped_init = ar.PromptProcessingBatch.__init__
    kvarn_serve.install_kvarn_serve()
    assert ar._make_cache is wrapped
    assert ar.PromptProcessingBatch.__init__ is wrapped_init


def test_make_cache_builds_kvarn(restorable, _ops_ok, capsys):
    _install(restorable)
    restorable.setenv("KV_TAIL_TOKENS", "256")
    caches = ar._make_cache(_LayersLM(), [0, 4], kv_bits=None, kv_quant_scheme="kvarn")
    assert len(caches) == 3
    # The shared carve-out holds the last layer of a deep stack fp16.
    assert all(type(c) is BatchKVarNKVCache for c in caches[:-1])
    assert type(caches[-1]) is BatchKVCache
    assert all(c.tail_cap == 256 and c.k_bits == 6 for c in caches[:-1])
    assert np.array_equal(np.array(caches[0].left_padding), [0, 4])
    assert "[kv] serve batch:" in capsys.readouterr().out
    assert gen._make_cache is ar._make_cache


def test_make_cache_env_bits(restorable, _ops_ok):
    _install(restorable)
    restorable.setenv("KV_BITS", "4")
    caches = ar._make_cache(_LayersLM(), [0], kv_bits=4.0, kv_quant_scheme="kvarn")
    assert all(c.k_bits == 4 for c in caches[:-1])


def test_make_cache_declines_to_fp16(restorable, _ops_ok):
    _install(restorable)
    caches = ar._make_cache(
        _LayersLM(head_dim=64), [0], kv_bits=6.0, kv_quant_scheme="kvarn"
    )
    # Declined model: fp16 batch caches, never the affine quantized cache.
    assert all(type(c) is BatchKVCache for c in caches)


def test_make_cache_stock_paths_untouched(restorable, _ops_ok):
    _install(restorable)
    affine = ar._make_cache(_LayersLM(), [0], kv_bits=8, kv_quant_scheme="uniform")
    assert all(type(c) is BatchQuantizedKVCache for c in affine[:-1])
    bare = ar._make_cache(_LayersLM(), [0])
    assert all(type(c) is BatchKVCache for c in bare)


def _make_ppb(model, **kw):
    return ar.PromptProcessingBatch(
        model=model,
        uids=[0],
        input_ids=[[1, 2, 3]],
        max_tokens=[8],
        inputs_embeds=mx.zeros((1, 3, 8), mx.float16),
        prompt_kwargs={},
        prefill_step_size=None,
        **kw,
    )


def test_ppb_fastpath_rebuilds_kvarn(restorable, _ops_ok):
    # Stock init's B=1 fast path skips _make_cache when kv_bits is None;
    # the wrap must land kvarn caches there too.
    _install(restorable)
    restorable.setenv("KV_TAIL_TOKENS", "256")
    batch = _make_ppb(_LayersLM(), kv_bits=None, kv_quant_scheme="kvarn")
    assert all(type(c) is BatchKVarNKVCache for c in batch.prompt_cache[:-1])
    assert all(c.tail_cap == 256 for c in batch.prompt_cache[:-1])


def test_ppb_fastpath_leaves_ineligible(restorable, _ops_ok):
    _install(restorable)
    batch = _make_ppb(_LayersLM(head_dim=64), kv_bits=None, kv_quant_scheme="kvarn")
    assert all(type(c) is KVCache for c in batch.prompt_cache)


def test_spec_build_suspends_kvarn(restorable, _ops_ok):
    _install(restorable)
    with kvarn_serve.spec_cache_build():
        caches = ar._make_cache(_LayersLM(), [0], kv_bits=None, kv_quant_scheme="kvarn")
    assert all(type(c) is BatchKVCache for c in caches)
    caches = ar._make_cache(_LayersLM(), [0], kv_bits=None, kv_quant_scheme="kvarn")
    assert all(type(c) is BatchKVarNKVCache for c in caches[:-1])


def test_apc_gate(restorable, _ops_ok):
    from mlx_vlm import apc

    from gmlx.cache import kvarn_apc

    class _FakeBG:
        def __init__(self, model, *args, **kwargs):
            self.apc_manager = kwargs.get("apc_manager")
            self.kv_bits = kwargs.get("kv_bits")

    class _FakeAR:
        BatchGenerator = _FakeBG

    kvarn_serve._install_apc_gate(_FakeAR)
    sentinel = object()

    # Arms installed: eligible models keep the manager, get stamped for
    # exact mode, and lose the kv_bits kwarg (the scheme owns the width).
    restorable.setattr(apc, kvarn_apc._FLAG, True, raising=False)
    lm = _LayersLM()
    bg = _FakeBG(lm, apc_manager=sentinel, kv_quant_scheme="kvarn", kv_bits=6)
    assert bg.apc_manager is sentinel and bg.kv_bits is None
    assert getattr(lm, kvarn_apc._MODE_STAMP, False)

    # Ineligible models and other schemes pass through untouched.
    cold = _LayersLM(head_dim=64)
    bg = _FakeBG(cold, apc_manager=sentinel, kv_quant_scheme="kvarn", kv_bits=6)
    assert bg.apc_manager is sentinel and bg.kv_bits == 6
    assert not getattr(cold, kvarn_apc._MODE_STAMP, False)
    bg = _FakeBG(_LayersLM(), apc_manager=sentinel, kv_quant_scheme="uniform")
    assert bg.apc_manager is sentinel

    # Arms missing: the manager is nulled rather than half-armed.
    restorable.setattr(apc, kvarn_apc._FLAG, False, raising=False)
    bg = _FakeBG(_LayersLM(), apc_manager=sentinel, kv_quant_scheme="kvarn")
    assert bg.apc_manager is None


def test_safe_quant_kvarn_arm(restorable, _ops_ok):
    import importlib

    from gmlx.cache.apc_pooling import install_safe_kv_quantization

    # importlib, not `import mlx_vlm.generate as ...`: the package exports
    # a `generate` function that shadows the submodule attribute.
    vlm_generate = importlib.import_module("mlx_vlm.generate")
    vlm_common = importlib.import_module("mlx_vlm.generate.common")

    restorable.setattr(
        vlm_generate, "maybe_quantize_kv_cache", vlm_generate.maybe_quantize_kv_cache
    )
    restorable.setattr(
        vlm_common, "maybe_quantize_kv_cache", vlm_common.maybe_quantize_kv_cache
    )
    install_safe_kv_quantization()

    class _Convertible:
        offset = 4096
        converted = False

        def to_quantized(self, **kw):
            pytest.fail("kvarn scheme must skip the affine mid-stream pass")

    pc = [_Convertible()]
    vlm_generate.maybe_quantize_kv_cache(pc, 0, 64, 8, kv_quant_scheme="kvarn")
    assert type(pc[0]) is _Convertible


def test_cascade_declines_kvarn():
    import gmlx.upstream.cascade_sdpa as cascade_sdpa

    class _KvarnCache:
        kv_quant_scheme = "kvarn"

    q = mx.zeros((2, 8, 1, 128), mx.float16)
    assert (
        cascade_sdpa._claim(q, object(), object(), _KvarnCache(), 1.0, None, None)
        is None
    )
    assert not cascade_sdpa._stamp_caches([_KvarnCache()], [b"ab", b"ab"])
