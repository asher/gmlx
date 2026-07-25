"""gemma4 hd512 batched-decode row route: claim gates, left-pad tail-slice
numerics vs the masked reference, pad memoization, the producer->consumer
mask relay for kv-shared layers, dual-seam install (mlx_lm gemma4_text +
mlx_vlm gemma4 language), and install hygiene.

Numerics run in f32 through whatever mx.fast route the platform provides
(stock on CI): the pin is that per-row tail slicing reproduces the bool
left-pad mask semantics, independent of which kernel executes the row."""

import mlx.core as mx
import pytest

pytest.importorskip("mlx_vlm.models.gemma4.language")
from mlx_lm.models import gemma4_text as g4t
from mlx_vlm.models.gemma4 import language as g4v

from gmlx import attn_hd512, gemma4_batched_sdpa as gb

_orig_vlm = g4v.scaled_dot_product_attention
_orig_lm = g4t.scaled_dot_product_attention


def teardown_module(module):
    g4v.scaled_dot_product_attention = _orig_vlm
    g4t.scaled_dot_product_attention = _orig_lm
    gb._installed = False
    gb._MASK_PADS.clear()


class _FakeBatchCache:
    def __init__(self, pads):
        self.left_padding = mx.array(pads)
        self._right_padding = None


def _install():
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
    got = g4v.scaled_dot_product_attention(
        q, k, v, cache=cache, scale=0.125, mask=_pad_mask(pads, 64))
    assert gb.claims() == n0 + 1
    orig = g4v.scaled_dot_product_attention._gmlx_orig
    ref = orig(q, k, v, cache=cache, scale=0.125, mask=_pad_mask(pads, 64))
    err = mx.abs(got - ref).max().item()
    assert err < 1e-4, f"tail-slice vs mask err={err}"


def test_gemma4_text_seam_claims(monkeypatch):
    """Text-only loads build from mlx_lm.models.gemma4_text; the route must
    claim at that module's symbol too, chained to mlx_lm's own original."""
    _install()
    monkeypatch.setattr(attn_hd512, "_MIN_KV", 32)
    route = g4t.scaled_dot_product_attention
    assert getattr(route, "_gmlx_g4_route", False)
    assert route._gmlx_orig is _orig_lm
    mx.random.seed(5)
    pads = [3, 0]
    q, k, v = _rand(2, 4, 2, 48)
    n0 = gb.claims()
    got = route(q, k, v, cache=_FakeBatchCache(pads), scale=0.125,
                mask=_pad_mask(pads, 48))
    assert gb.claims() == n0 + 1
    ref = _orig_lm(q, k, v, cache=_FakeBatchCache(pads), scale=0.125,
                   mask=_pad_mask(pads, 48))
    assert mx.abs(got - ref).max().item() < 1e-4


def test_pads_memoized_on_array_identity(monkeypatch):
    _install()
    monkeypatch.setattr(attn_hd512, "_MIN_KV", 32)
    cache = _FakeBatchCache([1, 0])
    q, k, v = _rand(2, 4, 2, 48)
    g4v.scaled_dot_product_attention(q, k, v, cache=cache, scale=0.5,
                                     mask=None)
    memo = cache._gmlx_g4_pads
    assert memo[1] == [1, 0]
    g4v.scaled_dot_product_attention(q, k, v, cache=cache, scale=0.5,
                                     mask=None)
    assert cache._gmlx_g4_pads is memo  # same lp array -> no re-read
    cache.left_padding = mx.array([0, 0])  # rebinding invalidates
    g4v.scaled_dot_product_attention(q, k, v, cache=cache, scale=0.5,
                                     mask=None)
    assert cache._gmlx_g4_pads is not memo


def test_mask_relay_producer_to_consumer(monkeypatch):
    """kv-shared consumers (cache=None) claim via the producer's mask object
    and inherit its pads; numerics match the masked reference."""
    _install()
    monkeypatch.setattr(attn_hd512, "_MIN_KV", 32)
    gb._MASK_PADS.clear()
    mx.random.seed(7)
    pads = [4, 0, 9]
    mask = _pad_mask(pads, 64)
    qp, kp, vp = _rand(3, 8, 2, 64)
    n0 = gb.claims()
    g4t.scaled_dot_product_attention(qp, kp, vp, cache=_FakeBatchCache(pads),
                                     scale=0.125, mask=mask)
    assert gb.claims() == n0 + 1
    qc, kc, vc = _rand(3, 8, 2, 64)
    got = g4t.scaled_dot_product_attention(qc, kc, vc, cache=None,
                                           scale=0.125, mask=mask)
    assert gb.claims() == n0 + 2
    ref = _orig_lm(qc, kc, vc, cache=None, scale=0.125, mask=mask)
    assert mx.abs(got - ref).max().item() < 1e-4


def test_mask_relay_cold_miss_falls_through(monkeypatch):
    """cache=None with an array mask no producer registered must fall
    through -- the mask may encode padding the route cannot reconstruct."""
    monkeypatch.setattr(attn_hd512, "_MIN_KV", 32)
    gb._MASK_PADS.clear()
    seen = []
    route = gb._make_route(lambda *a, **kw: seen.append(1) or mx.zeros(1))
    q, k, v = _rand(2, 4, 2, 48)
    route(q, k, v, cache=None, scale=0.5, mask=_pad_mask([0, 0], 48))
    assert seen == [1]


def test_non_claims_fall_through(monkeypatch):
    monkeypatch.setattr(attn_hd512, "_MIN_KV", 32)
    seen = []
    route = gb._make_route(lambda *a, **kw: seen.append(1) or mx.zeros(1))
    cache = _FakeBatchCache([0, 0])

    q, k, v = _rand(1, 4, 2, 48)  # B=1
    route(q, k, v, cache=cache, scale=0.5, mask=None)
    q, k, v = _rand(2, 4, 2, 48, d=256)  # hd256
    route(q, k, v, cache=cache, scale=0.5, mask=None)
    q, k, v = _rand(2, 4, 2, 48)
    q = mx.concatenate([q, q], axis=2)  # qL=2
    route(q, k, v, cache=cache, scale=0.5, mask="causal")
    q, k, v = _rand(2, 4, 2, 16)  # below MIN_KV
    route(q, k, v, cache=cache, scale=0.5, mask=None)

    class _QCache(_FakeBatchCache):
        bits = 4
        group_size = 64

    q, k, v = _rand(2, 4, 2, 48)  # quantized-style cache
    route(q, k, v, cache=_QCache([0, 0]), scale=0.5, mask=None)
    assert len(seen) == 5


def test_install_idempotent_and_killable(monkeypatch):
    g4v.scaled_dot_product_attention = _orig_vlm
    g4t.scaled_dot_product_attention = _orig_lm
    gb._installed = False
    monkeypatch.setenv("GMLX_G4_BATCHED_SDPA", "0")
    assert gb.install_gemma4_batched_sdpa() is False
    assert g4v.scaled_dot_product_attention is _orig_vlm
    assert g4t.scaled_dot_product_attention is _orig_lm

    monkeypatch.delenv("GMLX_G4_BATCHED_SDPA", raising=False)
    _install()
    pv = g4v.scaled_dot_product_attention
    pl = g4t.scaled_dot_product_attention
    assert pv is not _orig_vlm and pv._gmlx_orig is _orig_vlm
    assert pl is not _orig_lm and pl._gmlx_orig is _orig_lm
    assert gb.install_gemma4_batched_sdpa()
    assert g4v.scaled_dot_product_attention is pv  # no double wrap
    assert g4t.scaled_dot_product_attention is pl
