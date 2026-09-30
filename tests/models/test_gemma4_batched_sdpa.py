"""gemma4 hd512 batched-decode route: claim gates, left-pad tail-slice
numerics vs the masked reference, pad memoization, the producer->consumer
mask relay for kv-shared layers, the owned attention dispatch taking the
route under its gate, and the decode-width one-call starts path (single
batched kq.sdpa_decode_gqa with per-row key starts).

Row-loop numerics run in f32 through whatever mx.fast route the platform
provides (stock on CI): the pin is that per-row tail slicing reproduces the
bool left-pad mask semantics, independent of which kernel executes the row.
f32 also keeps the one-call path out (its dtype gate mirrors the kernel),
so those tests stay valid on the forced-CPU stream; the one-call tests are
bf16 and GPU-only."""

import os

import mlx.core as mx
import pytest

import gmlx.models.gemma4.batched_sdpa as gb
import gmlx.upstream.attn_hd512 as attn_hd512
from gmlx.models.gemma4 import dispatch


@pytest.fixture(autouse=True)
def _fresh_route_state(monkeypatch):
    """An empty mask relay, and the dispatch gates back as they were."""
    for name in ("_SPARSE", "_CASCADE", "_BATCHED"):
        monkeypatch.setattr(dispatch, name, getattr(dispatch, name))
    gb._MASK_PADS.clear()
    yield
    gb._MASK_PADS.clear()


class _FakeBatchCache:
    def __init__(self, pads):
        self.left_padding = mx.array(pads)
        self._right_padding = None


def _install():
    """The hd512 wrapper, then the dispatch gates the loader reads."""
    assert attn_hd512.install_hd512_sdpa()
    dispatch.configure()
    assert dispatch.gates()["batched"]


def _route(q, k, v, cache=None, scale=1.0, mask=None, sinks=None):
    return dispatch.scaled_dot_product_attention(q, k, v, cache, scale, mask,
                                                 sinks)


def _ref(q, k, v, scale, mask):
    return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale,
                                                mask=mask)


def _declines(q, k, v, cache=None, scale=0.5, mask=None, sinks=None):
    return gb._claim(q, k, v, cache, scale, mask, sinks) is None


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
    got = _route(q, k, v, cache=cache, scale=0.125, mask=_pad_mask(pads, 64))
    assert gb.claims() == n0 + 1
    ref = _ref(q, k, v, 0.125, _pad_mask(pads, 64))
    err = mx.abs(got - ref).max().item()
    assert err < 1e-4, f"tail-slice vs mask err={err}"


def test_the_dispatch_claims_only_under_its_gate(monkeypatch):
    """The owned dispatch takes the route when configure() set its gate,
    and otherwise hands the same call to the base."""
    _install()
    monkeypatch.setattr(attn_hd512, "_MIN_KV", 32)
    mx.random.seed(5)
    pads = [3, 0]
    q, k, v = _rand(2, 4, 2, 48)
    mask = _pad_mask(pads, 48)
    n0 = gb.claims()
    got = _route(q, k, v, cache=_FakeBatchCache(pads), scale=0.125, mask=mask)
    assert gb.claims() == n0 + 1
    monkeypatch.setattr(dispatch, "_BATCHED", False)
    base = _route(q, k, v, cache=_FakeBatchCache(pads), scale=0.125,
                  mask=mask)
    assert gb.claims() == n0 + 1
    assert mx.abs(got - base).max().item() < 1e-4


def test_pads_memoized_on_array_identity(monkeypatch):
    _install()
    monkeypatch.setattr(attn_hd512, "_MIN_KV", 32)
    cache = _FakeBatchCache([1, 0])
    q, k, v = _rand(2, 4, 2, 48)
    _route(q, k, v, cache=cache, scale=0.5, mask=None)
    memo = cache._gmlx_g4_pads
    assert memo[1] == [1, 0]
    _route(q, k, v, cache=cache, scale=0.5, mask=None)
    assert cache._gmlx_g4_pads is memo  # same lp array -> no re-read
    cache.left_padding = mx.array([0, 0])  # rebinding invalidates
    _route(q, k, v, cache=cache, scale=0.5, mask=None)
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
    _route(qp, kp, vp, cache=_FakeBatchCache(pads), scale=0.125, mask=mask)
    assert gb.claims() == n0 + 1
    qc, kc, vc = _rand(3, 8, 2, 64)
    got = _route(qc, kc, vc, cache=None, scale=0.125, mask=mask)
    assert gb.claims() == n0 + 2
    ref = _ref(qc, kc, vc, 0.125, mask)
    assert mx.abs(got - ref).max().item() < 1e-4


def test_mask_relay_cold_miss_falls_through(monkeypatch):
    """cache=None with an array mask no producer registered must fall
    through -- the mask may encode padding the route cannot reconstruct."""
    monkeypatch.setattr(attn_hd512, "_MIN_KV", 32)
    gb._MASK_PADS.clear()
    q, k, v = _rand(2, 4, 2, 48)
    assert _declines(q, k, v, cache=None, scale=0.5,
                     mask=_pad_mask([0, 0], 48))


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
    got = _route(q, k, v, cache=_FakeBatchCache(pads), scale=0.125,
                 mask=mask)
    assert gb.claims() == n0 + 1
    ref = _ref(q, k, v, 0.125, mask)
    err = mx.abs(got - ref).max().item()
    assert err < 1e-4, f"verify tail-slice vs mask err={err}"


def test_verify_width_non_claims(monkeypatch):
    monkeypatch.setattr(attn_hd512, "_MIN_KV", 32)
    cache = _FakeBatchCache([0, 5])

    def qkv(qL, L=48):
        q = mx.random.normal((2, 4, qL, 512))
        k = mx.random.normal((2, 2, L, 512))
        v = mx.random.normal((2, 2, L, 512))
        return q, k, v

    q, k, v = qkv(9)  # above the verify-block range
    assert _declines(q, k, v, cache=cache, mask=_verify_mask([0, 5], 48, 9))
    q, k, v = qkv(3)  # qL>1 with mask None: unmasked block, never claim
    assert _declines(q, k, v, cache=cache, mask=None)
    q, k, v = qkv(3)  # mask width mismatched to qL
    assert _declines(q, k, v, cache=cache, mask=_verify_mask([0, 5], 48, 1))
    q, k, v = qkv(4)  # pad overlaps the block: 46 > 48 - 4
    assert _declines(q, k, v, cache=_FakeBatchCache([0, 46]),
                     mask=_verify_mask([0, 46], 48, 4))


def test_non_claims_fall_through(monkeypatch):
    monkeypatch.setattr(attn_hd512, "_MIN_KV", 32)
    cache = _FakeBatchCache([0, 0])

    q, k, v = _rand(1, 4, 2, 48)  # B=1
    assert _declines(q, k, v, cache=cache)
    q, k, v = _rand(2, 4, 2, 48, d=256)  # hd256
    assert _declines(q, k, v, cache=cache)
    q, k, v = _rand(2, 4, 2, 48)  # attention sinks present
    assert _declines(q, k, v, cache=cache, sinks=mx.zeros((4,)))
    q, k, v = _rand(2, 4, 2, 16)  # below MIN_KV
    assert _declines(q, k, v, cache=cache)

    class _QCache(_FakeBatchCache):
        bits = 4
        group_size = 64

    q, k, v = _rand(2, 4, 2, 48)  # quantized-style cache
    assert _declines(q, k, v, cache=_QCache([0, 0]))


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
    got = _route(
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
    a = _route(q, k, v, cache=cache, scale=0.125, mask=None)
    assert gb.onecall_claims() == o0 + 1  # pad-0: starts omitted, still one
    monkeypatch.setenv("GMLX_G4_SDPA_STARTS", "0")
    b = _route(q, k, v, cache=cache, scale=0.125, mask=None)
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
    _route(
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
    _route(
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
    _route(
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


def test_the_kill_switch_turns_the_route_off(monkeypatch):
    """GMLX_G4_BATCHED_SDPA=0 leaves the gate off at configure(), and the
    dispatch hands a claimable call to the base."""
    assert attn_hd512.install_hd512_sdpa()
    monkeypatch.setattr(attn_hd512, "_MIN_KV", 32)
    pads = [2, 0]
    q, k, v = _rand(2, 8, 2, 64)
    monkeypatch.setenv("GMLX_G4_BATCHED_SDPA", "0")
    dispatch.configure()
    assert not dispatch.gates()["batched"]
    n0 = gb.claims()
    _route(q, k, v, cache=_FakeBatchCache(pads), scale=0.125,
           mask=_pad_mask(pads, 64))
    assert gb.claims() == n0
    monkeypatch.delenv("GMLX_G4_BATCHED_SDPA")
    dispatch.configure()
    assert dispatch.gates()["batched"]
    _route(q, k, v, cache=_FakeBatchCache(pads), scale=0.125,
           mask=_pad_mask(pads, 64))
    assert gb.claims() == n0 + 1
