"""gemma4 hd512 batched-decode row route: claim gates, left-pad tail-slice
numerics vs the masked reference, pad memoization, and install hygiene.

Numerics run in f32 through whatever mx.fast route the platform provides
(stock on CI): the pin is that per-row tail slicing reproduces the bool
left-pad mask semantics, independent of which kernel executes the row."""

import mlx.core as mx
import pytest

pytest.importorskip("mlx_vlm.models.gemma4.language")
from mlx_vlm.models.gemma4 import language as g4

from gmlx import attn_hd512, gemma4_batched_sdpa as gb

_orig_sym = g4.scaled_dot_product_attention


def teardown_module(module):
    g4.scaled_dot_product_attention = _orig_sym
    gb._installed = False
    gb._orig = None


class _FakeBatchCache:
    def __init__(self, pads):
        self.left_padding = mx.array(pads)
        self._right_padding = None


def _install(monkeypatch=None):
    assert attn_hd512.install_hd512_sdpa()
    assert gb.install_gemma4_batched_sdpa()


def _rand(B, hq, hkv, L, d=512):
    q = mx.random.normal((B, hq, 1, d))
    k = mx.random.normal((B, hkv, L, d))
    v = mx.random.normal((B, hkv, L, d))
    mx.eval(q, k, v)
    return q, k, v


def _pad_mask(pads, L):
    pos = mx.arange(L)[None, None, None, :]
    return pos >= mx.array(pads)[:, None, None, None]


def test_route_matches_masked_reference(monkeypatch):
    _install()
    monkeypatch.setattr(attn_hd512, "_MIN_KV", 32)
    mx.random.seed(3)
    pads = [0, 5, 2]
    q, k, v = _rand(3, 8, 2, 64)
    cache = _FakeBatchCache(pads)

    n0 = gb.claims()
    got = g4.scaled_dot_product_attention(
        q, k, v, cache=cache, scale=0.125, mask=_pad_mask(pads, 64))
    assert gb.claims() == n0 + 1
    ref = gb._orig(q, k, v, cache=cache, scale=0.125,
                   mask=_pad_mask(pads, 64))
    err = mx.abs(got - ref).max().item()
    assert err < 1e-4, f"tail-slice vs mask err={err}"


def test_pads_memoized_on_array_identity(monkeypatch):
    _install()
    monkeypatch.setattr(attn_hd512, "_MIN_KV", 32)
    cache = _FakeBatchCache([1, 0])
    q, k, v = _rand(2, 4, 2, 48)
    g4.scaled_dot_product_attention(q, k, v, cache=cache, scale=0.5,
                                    mask=None)
    memo = cache._gmlx_g4_pads
    assert memo[1] == [1, 0]
    g4.scaled_dot_product_attention(q, k, v, cache=cache, scale=0.5,
                                    mask=None)
    assert cache._gmlx_g4_pads is memo  # same lp array -> no re-read
    cache.left_padding = mx.array([0, 0])  # rebinding invalidates
    g4.scaled_dot_product_attention(q, k, v, cache=cache, scale=0.5,
                                    mask=None)
    assert cache._gmlx_g4_pads is not memo


def test_non_claims_fall_through(monkeypatch):
    _install()
    monkeypatch.setattr(attn_hd512, "_MIN_KV", 32)
    seen = []
    monkeypatch.setattr(gb, "_orig",
                        lambda *a, **kw: seen.append(1) or mx.zeros(1))
    cache = _FakeBatchCache([0, 0])

    q, k, v = _rand(1, 4, 2, 48)  # B=1
    g4.scaled_dot_product_attention(q, k, v, cache=cache, scale=0.5, mask=None)
    q, k, v = _rand(2, 4, 2, 48, d=256)  # hd256
    g4.scaled_dot_product_attention(q, k, v, cache=cache, scale=0.5, mask=None)
    q, k, v = _rand(2, 4, 2, 48)
    q = mx.concatenate([q, q], axis=2)  # qL=2
    g4.scaled_dot_product_attention(q, k, v, cache=cache, scale=0.5,
                                    mask="causal")
    q, k, v = _rand(2, 4, 2, 16)  # below MIN_KV
    g4.scaled_dot_product_attention(q, k, v, cache=cache, scale=0.5, mask=None)

    class _QCache(_FakeBatchCache):
        bits = 4
        group_size = 64

    q, k, v = _rand(2, 4, 2, 48)  # quantized-style cache
    g4.scaled_dot_product_attention(q, k, v, cache=_QCache([0, 0]),
                                    scale=0.5, mask=None)
    assert len(seen) == 5


def test_install_idempotent_and_killable(monkeypatch):
    g4.scaled_dot_product_attention = _orig_sym
    gb._installed = False
    gb._orig = None
    monkeypatch.setenv("GMLX_G4_BATCHED_SDPA", "0")
    assert gb.install_gemma4_batched_sdpa() is False
    assert g4.scaled_dot_product_attention is _orig_sym

    monkeypatch.delenv("GMLX_G4_BATCHED_SDPA", raising=False)
    _install()
    patched = g4.scaled_dot_product_attention
    assert patched is not _orig_sym
    assert patched._gmlx_orig is _orig_sym
    assert gb.install_gemma4_batched_sdpa()
    assert g4.scaled_dot_product_attention is patched  # no double wrap
