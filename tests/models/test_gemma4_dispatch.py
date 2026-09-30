"""Route locks for the owned Gemma 4 attention dispatch.

The owned text model calls ``gmlx.models.gemma4.dispatch`` for attention:
the sparse, cascade and batched hd512 claims in that order, then gmlx's
base SDPA. Each lock builds an owned attention layer, calls it, and checks
by counter or spy that the call reached the route. The stock half runs the
same calls on stock gemma4 attention in a child that applies no loader
install (gemma4_probes.py): no claim counter moves there, since only the
loader's module-global installs route stock gemma4.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import mlx.core as mx
import pytest

import gmlx.cache.attention as attention
import gmlx.upstream.attn_hd512 as hd512
from gmlx.cache import kvarn_sdpa
from gmlx.cache.kv_cache import BatchQuantizedKVCache, KVCache
from gmlx.cache.kvarn_cache import KVarNView
from gmlx.models.gemma4 import batched_sdpa, dispatch
from gmlx.models.gemma4 import language as owned
from gmlx.upstream import cascade_sdpa, sparse_sdpa

import gemma4_probes as probes

DRIVER = Path(__file__).with_name("gemma4_probes.py")
_OWNED_CLAIMS = {
    "batched_decode": 1, "batched_verify": 1, "relay": 1, "relay_cold": 0,
    "cascade": 1, "cascade_rotating": 0, "overlap_cascade": 1,
    "overlap_batched": 0, "sparse_default": 0, "sparse_opted_in": 1,
    "sparse_module_name": 0,
}
_GPU = (mx.default_device() == mx.Device(mx.gpu)
        and not os.environ.get("KQUANT_FORCE_CPU"))


@pytest.fixture(scope="module")
def stock() -> dict:
    r = subprocess.run([sys.executable, str(DRIVER), "--cpu"],
                       capture_output=True, text=True, env=dict(os.environ),
                       timeout=600)
    assert r.returncode == 0, r.stderr[-4000:]
    return json.loads(r.stdout)


@pytest.fixture
def gates(monkeypatch):
    """Every route gate open, and the gates restored afterwards."""
    for name in ("_SPARSE", "_CASCADE", "_BATCHED"):
        monkeypatch.setattr(dispatch, name, True)


def test_the_owned_model_attends_through_the_dispatch():
    assert (owned.scaled_dot_product_attention
            is dispatch.scaled_dot_product_attention)


def test_owned_attention_reaches_each_route():
    assert probes.route_claims(owned) == _OWNED_CLAIMS


def test_stock_attention_reaches_no_route(stock):
    assert stock["route_claims"] == dict.fromkeys(_OWNED_CLAIMS, 0)


def test_configure_reads_the_gates(monkeypatch):
    monkeypatch.setattr(dispatch, "_SPARSE", False)
    monkeypatch.setattr(dispatch, "_CASCADE", False)
    monkeypatch.setattr(dispatch, "_BATCHED", False)
    monkeypatch.setattr(hd512, "_installed", True)
    monkeypatch.setattr(cascade_sdpa, "_installed_route", True)
    monkeypatch.setenv("GMLX_SPARSE_ATTN", "1")
    monkeypatch.setenv("GMLX_SPARSE_ARCHS", "gemma4_text")
    monkeypatch.delenv("GMLX_G4_BATCHED_SDPA", raising=False)
    dispatch.configure()
    assert dispatch.gates() == {"sparse": True, "cascade": True,
                                "batched": True}
    monkeypatch.setenv("GMLX_G4_BATCHED_SDPA", "0")
    monkeypatch.delenv("GMLX_SPARSE_ARCHS")
    monkeypatch.setattr(cascade_sdpa, "_installed_route", False)
    dispatch.configure()
    assert dispatch.gates() == {"sparse": False, "cascade": False,
                                "batched": False}


def test_configure_names_the_open_routes_of_the_owned_model(monkeypatch):
    import mlx.nn as nn

    for name in ("_SPARSE", "_CASCADE", "_BATCHED"):
        monkeypatch.setattr(dispatch, name, False)
    monkeypatch.setattr(hd512, "_installed", True)
    monkeypatch.setattr(cascade_sdpa, "_installed_route", True)
    monkeypatch.setenv("GMLX_SPARSE_ATTN", "1")
    monkeypatch.setenv("GMLX_SPARSE_ARCHS", "gemma4_text")
    monkeypatch.delenv("GMLX_G4_BATCHED_SDPA", raising=False)
    lm, lines = probes.tiny(owned), []
    dispatch.configure(lm, lines.append)
    assert lines == ["[install] gemma-4 attention routes: sparse top-k pages, "
                     "shared-prefix cascade, hd512 batched rows"]
    monkeypatch.setenv("GMLX_G4_BATCHED_SDPA", "0")
    monkeypatch.delenv("GMLX_SPARSE_ARCHS")
    monkeypatch.setattr(cascade_sdpa, "_installed_route", False)
    dispatch.configure(lm, lines.append)
    assert lines[1:] == [
        "[install] gemma-4 attention routes: none, the base SDPA only"]
    dispatch.configure(nn.Linear(2, 2), lines.append)
    dispatch.configure(None, lines.append)
    dispatch.configure(lm)
    assert len(lines) == 2


def test_the_sparse_arch_list_opts_gemma4_in(monkeypatch):
    monkeypatch.delenv("GMLX_SPARSE_ARCHS", raising=False)
    assert sparse_sdpa.arch_enabled("llama")
    assert not sparse_sdpa.arch_enabled("gemma4_text")
    monkeypatch.setenv("GMLX_SPARSE_ARCHS", "qwen3_next, gemma4_text")
    assert sparse_sdpa.arch_enabled("gemma4_text")
    assert not sparse_sdpa.arch_enabled("language")


def test_a_call_no_route_claims_reaches_the_base_with_its_sinks(
        monkeypatch):
    seen = []
    monkeypatch.setattr(dispatch, "_base", lambda *a: seen.append(a) or a[0])
    for name in ("_SPARSE", "_CASCADE", "_BATCHED"):
        monkeypatch.setattr(dispatch, name, False)
    q = mx.zeros((1, 2, 1, 16))
    sinks = mx.zeros((2,))
    dispatch.scaled_dot_product_attention(q, q, q, None, 1.0, None,
                                          sinks=sinks)
    assert len(seen) == 1 and seen[0][-1] is sinks


def test_a_kvarn_view_passes_every_claim(gates, monkeypatch):
    """A KVarN view reaches kvarn_attention in the base, with every gate
    open and no claim counter moving."""
    calls = []
    monkeypatch.setattr(kvarn_sdpa, "kvarn_attention",
                        lambda *a: calls.append(a) or mx.zeros((1,)))
    before = (sparse_sdpa.claims(), cascade_sdpa.claims(),
              batched_sdpa.claims())
    for B, L in ((1, 1), (2, 1), (2, 4)):
        cache = type("KVarNLike", (), {
            "kv_quant_scheme": "kvarn",
            "left_padding": mx.zeros((B,), mx.int32)})()
        q = mx.zeros((B, 8, L, 512), dtype=mx.float16)
        dispatch.scaled_dot_product_attention(
            q, KVarNView(cache, "k"), KVarNView(cache, "v"), cache, 0.5,
            "causal")
        assert calls[-1][1] is cache
    assert len(calls) == 3
    assert (sparse_sdpa.claims(), cascade_sdpa.claims(),
            batched_sdpa.claims()) == before


def _refuse(name):
    def fail(*a, **k):
        pytest.fail(f"the call took the {name} route")
    return fail


def test_a_q8_batch_cache_takes_the_kv8_prefill_route(gates, monkeypatch):
    calls = []
    monkeypatch.setattr(attention, "_prefill_route",
                        lambda q, *a: calls.append(a) or mx.zeros(q.shape))
    monkeypatch.setattr(kvarn_sdpa, "kvarn_attention", _refuse("kvarn"))
    monkeypatch.setenv("GMLX_KV8_PREFILL_FLASH_MIN_L", "4")
    slide = probes.attention(owned, 0, mx.float16)
    cache = BatchQuantizedKVCache([0, 0], group_size=64, bits=8)
    slide(mx.random.normal((2, 8, 64)).astype(mx.float16), "causal", cache)
    assert len(calls) == 1
    assert calls[0][-2:] == (64, 8)


@pytest.mark.skipif(not _GPU, reason="the q8 decode kernel is Metal only")
def test_a_q8_batch_cache_takes_the_q8_decode_kernel(gates, monkeypatch):
    calls = []

    def kernel(q, *a, **kw):
        calls.append(kw)
        return mx.zeros(q.shape, q.dtype)

    monkeypatch.setattr(attention, "_kq_decode_kernel", lambda: kernel)
    monkeypatch.setattr(attention, "_prefill_route", _refuse("prefill"))
    slide = probes.attention(owned, 0, mx.float16)
    cache = BatchQuantizedKVCache([0, 2], group_size=64, bits=8)
    probes.fill(cache, 2, 64, 64, mx.float16)
    slide(mx.random.normal((2, 1, 64)).astype(mx.float16),
          cache.make_mask(1), cache)
    assert len(calls) == 1


@pytest.mark.skipif(not _GPU, reason="the hd512 routes are Metal kernels")
@pytest.mark.parametrize("width", [1, 4])
def test_the_hd512_wrapper_takes_an_owned_global_layer(width):
    assert hd512.install_hd512_sdpa()
    full = probes.attention(owned, 1, mx.bfloat16)
    cache = KVCache()
    probes.fill(cache, 1, 64, 512, mx.bfloat16)
    before = hd512.route_counts()
    x = mx.random.normal((1, width, 64)).astype(mx.bfloat16)
    mx.eval(full(x, None if width == 1 else "causal", cache)[0])
    after = hd512.route_counts()
    moved = [key for key in after if after[key] > before.get(key, 0)]
    assert moved and all(key[1:3] == (512, width) for key in moved), moved


def test_training_attention_wraps_the_owned_module_and_restores_it(
        monkeypatch):
    """install_training_attention patches the module global the owned
    layers call: a training call takes the blocked attention, an eval call
    takes the dispatch with the logits it gave before, and the restore
    gives back the dispatch."""
    import gmlx.tune.attention as tune_attention

    calls = []
    blocked = tune_attention.blocked_attention

    def spy(*a, **kw):
        calls.append(a[0].shape)
        return blocked(*a, **kw)

    monkeypatch.setattr(tune_attention, "blocked_attention", spy)
    lm = probes.tiny(owned)
    ids = mx.array([probes.PROMPT])
    want = lm(ids).logits
    restore = tune_attention.install_training_attention(lm)
    try:
        wrapped = owned.scaled_dot_product_attention
        assert wrapped is not dispatch.scaled_dot_product_attention
        assert wrapped._gmlx_orig is dispatch.scaled_dot_product_attention
        lm.train()
        mx.eval(lm(ids).logits)
        trained = len(calls)
        lm.eval()
        evald = lm(ids).logits
    finally:
        lm.eval()
        restore()
    assert trained == len(lm.model.layers)
    assert len(calls) == trained
    assert mx.array_equal(evald, want).item()
    assert (owned.scaled_dot_product_attention
            is dispatch.scaled_dot_product_attention)
