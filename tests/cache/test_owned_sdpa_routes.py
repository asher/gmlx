"""Route locks for the owned SDPA dispatch, and the per-arch SDPA wraps
over it.

Each branch of ``scaled_dot_product_attention`` in gmlx.cache.attention
gets one call shaped for it, with the branch's target replaced by a spy:
a KVarN view goes to ``kvarn_attention``, a TurboQuant cache to its fused
decode, a quantized cache to the quantized SDPA and there to the prefill
route, the q8 decode kernel or the upstream body, and anything else to
``mx.fast.scaled_dot_product_attention``. The per-arch wraps that sit over
the owned function (cascade, sparse, gemma-4 batched rows and both
qwen3.5 verify folds) must hand a KVarN view down untouched, since the
KVarN route lives under them.
"""

from __future__ import annotations

import mlx.core as mx
import pytest

import gmlx.cache.attention as attention
from gmlx.cache import kvarn_sdpa
from gmlx.cache.kv_cache import BatchQuantizedKVCache, KVCache
from gmlx.cache.kvarn_cache import KVarNView
from stock_twin import stock_twin

_OUT = mx.zeros((1,))


class _Spy:
    def __init__(self, out=_OUT):
        self.calls = []
        self.out = out

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.out


def _qkv(B=1, hq=8, hkv=2, L=1, S=64, d=64, dtype=mx.float16):
    mx.random.seed(5)
    q = mx.random.normal((B, hq, L, d)).astype(dtype)
    k = mx.random.normal((B, hkv, S, d)).astype(dtype)
    v = mx.random.normal((B, hkv, S, d)).astype(dtype)
    return q, k, v


def _no_other_route(monkeypatch):
    """Every target but the one under test fails the call."""
    def refuse(name):
        def fail(*a, **k):
            pytest.fail(f"the call took the {name} route")
        return fail

    monkeypatch.setattr(kvarn_sdpa, "kvarn_attention", refuse("kvarn"))
    monkeypatch.setattr(attention, "_prefill_route", refuse("prefill"))


def test_a_kvarn_view_takes_the_kvarn_route(monkeypatch):
    spy = _Spy()
    monkeypatch.setattr(kvarn_sdpa, "kvarn_attention", spy)
    owner = object()
    q = mx.zeros((1, 8, 1, 64))
    out = attention.scaled_dot_product_attention(
        q, KVarNView(owner, "k"), KVarNView(owner, "v"), None, 0.5, None)
    assert out is _OUT
    assert len(spy.calls) == 1 and spy.calls[0][0][1] is owner


def test_a_turboquant_cache_takes_its_fused_decode(monkeypatch):
    tq = pytest.importorskip("gmlx._vendor.mlx_vlm.turboquant")
    _no_other_route(monkeypatch)
    spy = _Spy()
    monkeypatch.setattr(tq.TurboQuantKVCache, "decode_attention", spy)
    cache = tq.TurboQuantKVCache(bits=4.0)
    q = mx.zeros((1, 8, 1, 64))
    out = attention.scaled_dot_product_attention(
        q, object(), object(), cache, 0.5, None)
    assert out is _OUT and len(spy.calls) == 1


def test_a_quantized_cache_takes_the_quantized_sdpa(monkeypatch):
    _no_other_route(monkeypatch)
    spy = _Spy()
    monkeypatch.setattr(attention, "quantized_scaled_dot_product_attention",
                        spy)
    cache = BatchQuantizedKVCache([0], group_size=32, bits=4)
    q = mx.zeros((1, 8, 1, 64))
    out = attention.scaled_dot_product_attention(q, (), (), cache, 0.5, None)
    assert out is _OUT and len(spy.calls) == 1
    assert spy.calls[0][1] == {"scale": 0.5, "mask": None, "group_size": 32,
                               "bits": 4}


def test_prefill_width_takes_the_prefill_route(monkeypatch):
    spy = _Spy()
    monkeypatch.setattr(attention, "_prefill_route", spy)
    monkeypatch.setattr(attention, "_decode_route",
                        lambda *a: pytest.fail("prefill width took the decode route"))
    q, k, v = _qkv(L=8)
    qk = mx.quantize(k, group_size=64, bits=8)
    qv = mx.quantize(v, group_size=64, bits=8)
    out = attention.quantized_scaled_dot_product_attention(
        q, qk, qv, 0.5, "causal")
    assert out is _OUT and len(spy.calls) == 1


@pytest.mark.skipif(
    mx.default_device().type != mx.DeviceType.gpu,
    reason="the q8 decode kernel is Metal only")
def test_a_q8_decode_takes_the_kq_kernel(monkeypatch):
    _no_other_route(monkeypatch)
    spy = _Spy()
    monkeypatch.setattr(attention, "_kq_decode_kernel", lambda: spy)
    q, k, v = _qkv(L=1)
    qk = mx.quantize(k, group_size=64, bits=8)
    qv = mx.quantize(v, group_size=64, bits=8)
    out = attention.quantized_scaled_dot_product_attention(q, qk, qv, 0.5,
                                                           None)
    assert out is _OUT and len(spy.calls) == 1


def test_a_call_no_route_claims_runs_the_upstream_body(monkeypatch):
    _no_other_route(monkeypatch)
    monkeypatch.setattr(attention, "_kq_decode_kernel",
                        lambda: pytest.fail("fp32 queries took the kq route"))
    q, k, v = _qkv(L=4, dtype=mx.float32)
    qk = mx.quantize(k, group_size=64, bits=8)
    qv = mx.quantize(v, group_size=64, bits=8)
    stock = stock_twin(attention.quantized_scaled_dot_product_attention,
                       origin="gmlx._vendor.mlx_vlm.models.base")
    got = attention.quantized_scaled_dot_product_attention(
        mx.array(q), qk, qv, 0.5, "causal")
    want = stock(mx.array(q), qk, qv, 0.5, "causal")
    assert mx.array_equal(got, want).item()


def test_a_plain_cache_takes_the_fused_kernel(monkeypatch):
    _no_other_route(monkeypatch)
    spy = _Spy()
    monkeypatch.setattr(mx.fast, "scaled_dot_product_attention", spy)
    q, k, v = _qkv()
    out = attention.scaled_dot_product_attention(q, k, v, KVCache(), 0.5,
                                                 None)
    assert out is _OUT and len(spy.calls) == 1
    assert spy.calls[0][0][1] is k


def _wraps():
    from gmlx.models.gemma4 import batched_sdpa
    from gmlx.models.qwen35 import verify_fold
    from gmlx.upstream import cascade_sdpa, sparse_sdpa

    def verify(under):
        fold = verify_fold._fold_verify(None, under, None)
        return lambda q, k, v, cache, scale, mask: fold(
            q, k, v, cache=cache, scale=scale, mask=mask)

    return {
        "cascade": cascade_sdpa._make_route,
        "sparse": sparse_sdpa._make_route,
        "gemma4-rows": batched_sdpa._make_route,
        "qwen35-batch-fold": lambda under: verify_fold._fold_batch_sdpa(
            under, None),
        "qwen35-verify-fold": verify,
    }


@pytest.mark.parametrize("name", sorted(_wraps()))
@pytest.mark.parametrize("B,L", [(1, 1), (2, 1), (2, 4)])
def test_each_per_arch_wrap_hands_a_kvarn_view_down(name, B, L):
    spy = _Spy()
    wrap = _wraps()[name](spy)
    cache = type("KVarNLike", (), {"kv_quant_scheme": "kvarn",
                                   "left_padding": mx.zeros((B,), mx.int32)})()
    kv, vv = KVarNView(cache, "k"), KVarNView(cache, "v")
    q = mx.zeros((B, 8, L, 512), dtype=mx.float16)
    out = wrap(q, kv, vv, cache=cache, scale=0.5, mask="causal")
    assert out is _OUT and len(spy.calls) == 1
    args, kwargs = spy.calls[0]
    assert args[1] is kv and args[2] is vv
    assert kwargs.get("cache") is cache and kwargs.get("scale") == 0.5
