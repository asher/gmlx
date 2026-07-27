"""gemma4 hd512 batched-decode route: claim gates, left-pad tail-slice
numerics vs the masked reference, pad memoization, the producer->consumer
mask relay for kv-shared layers, dual-seam install (mlx_lm gemma4_text +
mlx_vlm gemma4 language), install hygiene, and the decode-width one-call
starts path (single batched kq.sdpa_decode_gqa with per-row key starts).

Row-loop numerics run in f32 through whatever mx.fast route the platform
provides (stock on CI): the pin is that per-row tail slicing reproduces the
bool left-pad mask semantics, independent of which kernel executes the row.
f32 also keeps the one-call path out (its dtype gate mirrors the kernel),
so those tests stay valid on the forced-CPU stream; the one-call tests are
bf16 and GPU-only."""

import os

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


def _verify_mask(pads, L, qL):
    """Left-pad visibility AND end-aligned in-block causal: query j (abs
    position L-qL+j) sees keys k with k >= pad and k <= L-qL+j."""
    pos = mx.arange(L)[None, None, None, :]
    vis = pos >= mx.array(pads)[:, None, None, None]
    limit = (L - qL) + mx.arange(qL)[None, None, :, None]
    return vis & (pos <= limit)


def test_verify_width_matches_masked_reference(monkeypatch):
    """qL 2..8 (MTP verify blocks) claim with per-row tail slice + causal;
    numerics vs the orig with the pad+causal mask."""
    _install()
    monkeypatch.setattr(attn_hd512, "_MIN_KV", 32)
    mx.random.seed(9)
    pads = [6, 0, 3]
    qL = 4
    q = mx.random.normal((3, 8, qL, 512))
    k = mx.random.normal((3, 2, 64, 512))
    v = mx.random.normal((3, 2, 64, 512))
    mx.eval(q, k, v)
    mask = _verify_mask(pads, 64, qL)
    n0 = gb.claims()
    got = g4t.scaled_dot_product_attention(
        q, k, v, cache=_FakeBatchCache(pads), scale=0.125, mask=mask)
    assert gb.claims() == n0 + 1
    ref = _orig_lm(q, k, v, cache=_FakeBatchCache(pads), scale=0.125,
                   mask=mask)
    err = mx.abs(got - ref).max().item()
    assert err < 1e-4, f"verify tail-slice vs mask err={err}"


def test_verify_width_non_claims(monkeypatch):
    monkeypatch.setattr(attn_hd512, "_MIN_KV", 32)
    seen = []
    route = gb._make_route(lambda *a, **kw: seen.append(1) or mx.zeros(1))
    cache = _FakeBatchCache([0, 5])

    def qkv(qL, L=48):
        q = mx.random.normal((2, 4, qL, 512))
        k = mx.random.normal((2, 2, L, 512))
        v = mx.random.normal((2, 2, L, 512))
        return q, k, v

    q, k, v = qkv(9)  # above the verify-block range
    route(q, k, v, cache=cache, scale=0.5, mask=_verify_mask([0, 5], 48, 9))
    q, k, v = qkv(3)  # qL>1 with mask None: unmasked block, never claim
    route(q, k, v, cache=cache, scale=0.5, mask=None)
    q, k, v = qkv(3)  # mask width mismatched to qL
    route(q, k, v, cache=cache, scale=0.5, mask=_verify_mask([0, 5], 48, 1))
    q, k, v = qkv(4)  # pad overlaps the block: 46 > 48 - 4
    route(q, k, v, cache=_FakeBatchCache([0, 46]), scale=0.5,
          mask=_verify_mask([0, 46], 48, 4))
    assert len(seen) == 4


def test_non_claims_fall_through(monkeypatch):
    monkeypatch.setattr(attn_hd512, "_MIN_KV", 32)
    seen = []
    route = gb._make_route(lambda *a, **kw: seen.append(1) or mx.zeros(1))
    cache = _FakeBatchCache([0, 0])

    q, k, v = _rand(1, 4, 2, 48)  # B=1
    route(q, k, v, cache=cache, scale=0.5, mask=None)
    q, k, v = _rand(2, 4, 2, 48, d=256)  # hd256
    route(q, k, v, cache=cache, scale=0.5, mask=None)
    q, k, v = _rand(2, 4, 2, 48)  # attention sinks present
    route(q, k, v, cache=cache, scale=0.5, mask=None, sinks=mx.zeros((4,)))
    q, k, v = _rand(2, 4, 2, 16)  # below MIN_KV
    route(q, k, v, cache=cache, scale=0.5, mask=None)

    class _QCache(_FakeBatchCache):
        bits = 4
        group_size = 64

    q, k, v = _rand(2, 4, 2, 48)  # quantized-style cache
    route(q, k, v, cache=_QCache([0, 0]), scale=0.5, mask=None)
    assert len(seen) == 5


_gpu_only = pytest.mark.skipif(
    bool(os.environ.get("KQUANT_FORCE_CPU")),
    reason="kq.sdpa_decode_gqa is Metal-only; the one-call path needs it")


def _rand16(B, hq, hkv, L, qL=1, d=512):
    q = mx.random.normal((B, hq, qL, d)).astype(mx.bfloat16)
    k = mx.random.normal((B, hkv, L, d)).astype(mx.bfloat16)
    v = mx.random.normal((B, hkv, L, d)).astype(mx.bfloat16)
    mx.eval(q, k, v)
    return q, k, v


@_gpu_only
def test_onecall_starts_engagement_and_numerics(monkeypatch):
    """Decode width bf16 with ragged pads: served by ONE batched kernel
    call (onecall counter), numerics vs the masked-reference orig."""
    if not gb._HAS_STARTS:
        pytest.skip("resident mlx-kquant lacks sdpa_decode_gqa starts")
    _install()
    monkeypatch.setattr(attn_hd512, "_MIN_KV", 32)
    mx.random.seed(13)
    pads = [0, 17, 40]
    q, k, v = _rand16(3, 8, 2, 64)
    n0, o0 = gb.claims(), gb.onecall_claims()
    got = g4v.scaled_dot_product_attention(
        q, k, v, cache=_FakeBatchCache(pads), scale=0.125,
        mask=_pad_mask(pads, 64))
    assert gb.claims() == n0 + 1
    assert gb.onecall_claims() == o0 + 1
    # independent f32 oracle (kernel-vs-kernel bf16 would double the error
    # budget); rel-norm bound is the kq-suite bf16 bar
    g = q.shape[1] // k.shape[1]
    kf = mx.repeat(k.astype(mx.float32), g, axis=1)
    vf = mx.repeat(v.astype(mx.float32), g, axis=1)
    s = (q.astype(mx.float32) * 0.125) @ kf.swapaxes(-1, -2)
    s = mx.where(_pad_mask(pads, 64), s, -float("inf"))
    rf = mx.softmax(s, axis=-1, precise=True) @ vf
    gf = got.astype(mx.float32)
    rel = float(mx.linalg.norm(gf - rf) / (mx.linalg.norm(rf) + 1e-9))
    assert rel < 5e-3, f"one-call vs f32 oracle rel={rel}"


@_gpu_only
def test_onecall_pad_zero_and_kill_switch(monkeypatch):
    if not gb._HAS_STARTS:
        pytest.skip("resident mlx-kquant lacks sdpa_decode_gqa starts")
    _install()
    monkeypatch.setattr(attn_hd512, "_MIN_KV", 32)
    mx.random.seed(17)
    q, k, v = _rand16(2, 8, 2, 64)
    cache = _FakeBatchCache([0, 0])
    o0 = gb.onecall_claims()
    a = g4v.scaled_dot_product_attention(q, k, v, cache=cache, scale=0.125,
                                         mask=None)
    assert gb.onecall_claims() == o0 + 1  # pad-0: starts omitted, still one
    monkeypatch.setenv("GMLX_G4_SDPA_STARTS", "0")
    b = g4v.scaled_dot_product_attention(q, k, v, cache=cache, scale=0.125,
                                         mask=None)
    assert gb.onecall_claims() == o0 + 1  # kill switch -> row loop
    err = mx.abs(a.astype(mx.float32) - b.astype(mx.float32)).max().item()
    assert err < 2e-2, f"one-call vs row loop err={err}"


@_gpu_only
def test_onecall_verify_width_stays_rowloop(monkeypatch):
    """qL>1 blocks stay on the per-row loop (fa_verify territory)."""
    if not gb._HAS_STARTS:
        pytest.skip("resident mlx-kquant lacks sdpa_decode_gqa starts")
    _install()
    monkeypatch.setattr(attn_hd512, "_MIN_KV", 32)
    mx.random.seed(19)
    pads = [5, 0]
    q, k, v = _rand16(2, 8, 2, 64, qL=4)
    o0 = gb.onecall_claims()
    n0 = gb.claims()
    g4v.scaled_dot_product_attention(
        q, k, v, cache=_FakeBatchCache(pads), scale=0.125,
        mask=_verify_mask(pads, 64, 4))
    assert gb.claims() == n0 + 1
    assert gb.onecall_claims() == o0


def test_onecall_dtype_gate_keeps_f32_on_rowloop(monkeypatch):
    """f32 calls never enter the one-call path (this is also what keeps the
    forced-CPU stream off the Metal-only kernel)."""
    _install()
    monkeypatch.setattr(attn_hd512, "_MIN_KV", 32)
    mx.random.seed(23)
    pads = [2, 0]
    q, k, v = _rand(2, 8, 2, 64)
    o0 = gb.onecall_claims()
    n0 = gb.claims()
    g4v.scaled_dot_product_attention(
        q, k, v, cache=_FakeBatchCache(pads), scale=0.125,
        mask=_pad_mask(pads, 64))
    assert gb.claims() == n0 + 1
    assert gb.onecall_claims() == o0


def test_onecall_absent_kernel_falls_back(monkeypatch):
    _install()
    monkeypatch.setattr(attn_hd512, "_MIN_KV", 32)
    monkeypatch.setattr(gb, "_HAS_STARTS", False)
    mx.random.seed(29)
    pads = [1, 0]
    q, k, v = _rand(2, 8, 2, 64)
    o0 = gb.onecall_claims()
    n0 = gb.claims()
    g4v.scaled_dot_product_attention(
        q, k, v, cache=_FakeBatchCache(pads), scale=0.125,
        mask=_pad_mask(pads, 64))
    assert gb.claims() == n0 + 1
    assert gb.onecall_claims() == o0


def test_starts_ring_memoizes_by_value():
    gb._STARTS_RING.clear()
    a = gb._starts_for([3, 0, 7], 3)
    b = gb._starts_for([3, 0, 7], 3)
    assert a is b
    c = gb._starts_for([3, 0, 8], 3)
    assert c is not a
    for i in range(12):  # bounded ring
        gb._starts_for([i, i], 2)
    assert len(gb._STARTS_RING) <= 8


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
