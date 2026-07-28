"""Shared-prefix cascade decode route: claim gates, fused-kernel numerics
vs the masked reference on stamped caches, ragged left pads, stamp
lifecycle (warm-multi detection, extend drop, filter survival), and
install hygiene.

Numerics require the mlx_kquant fused op (GPU); those tests skip on
platforms where the op is unavailable. Stamp-detection tests are pure
host logic and always run."""

import mlx.core as mx
import pytest

from mlx_lm.models import llama as llama_mod

from gmlx import cascade_sdpa as cs

_orig_llama = llama_mod.scaled_dot_product_attention


def _has_cascade_op():
    try:
        from mlx_kquant import sdpa_decode_gqa_cascade  # noqa: F401
    except ImportError:
        return False
    return mx.default_device() == mx.Device(mx.gpu)


def teardown_module(module):
    llama_mod.scaled_dot_product_attention = _orig_llama
    cs._installed_route = False
    cs._installed_stamp = False


class _FakeBatchCache:
    def __init__(self, pads, P=None):
        self.left_padding = mx.array(pads)
        self._right_padding = None
        if P is not None:
            self._gmlx_cascade = {"P": P}

    def extend(self, other):
        return "extended"


def _install():
    assert cs.install_cascade_sdpa()


def _shared_batch(B, hq, hkv, P, sp, pads, d=128, dtype=mx.float16):
    """[B, hkv, maxpad+P+sp, d] buffer whose rows carry one shared prefix
    copy at [pads[b], pads[b]+P) and private tails after it."""
    mx.random.seed(11)
    L = max(pads) + P + sp
    kb = mx.random.normal((B, hkv, L, d)).astype(dtype)
    vb = mx.random.normal((B, hkv, L, d)).astype(dtype)
    pk = kb[0:1, :, pads[0]:pads[0] + P]
    pv = vb[0:1, :, pads[0]:pads[0] + P]
    rk, rv = [], []
    for b in range(B):
        rk.append(mx.concatenate(
            [kb[b:b + 1, :, :pads[b]], pk, kb[b:b + 1, :, pads[b] + P:]],
            axis=2))
        rv.append(mx.concatenate(
            [vb[b:b + 1, :, :pads[b]], pv, vb[b:b + 1, :, pads[b] + P:]],
            axis=2))
    k = mx.concatenate(rk, axis=0)
    v = mx.concatenate(rv, axis=0)
    q = mx.random.normal((B, hq, 1, d)).astype(dtype)
    mx.eval(q, k, v)
    return q, k, v


def _masked_ref(q, k, v, pads, scale):
    B, hkv, L, d = k.shape
    hq = q.shape[1]
    kr = mx.repeat(k, hq // hkv, axis=1).astype(mx.float32)
    vr = mx.repeat(v, hq // hkv, axis=1).astype(mx.float32)
    s = (q.astype(mx.float32) * scale) @ kr.swapaxes(-1, -2)
    pos = mx.arange(L)[None, :]
    keep = pos >= mx.array(pads)[:, None]
    bias = mx.where(keep, mx.zeros(keep.shape),
                    mx.full(keep.shape, -mx.inf))[:, None, None, :]
    return mx.softmax(s + bias, axis=-1) @ vr


@pytest.mark.skipif(not _has_cascade_op(), reason="fused cascade op needs GPU")
@pytest.mark.parametrize("pads", [[0, 0, 0, 0], [0, 64, 128, 32]])
def test_route_matches_masked_reference(monkeypatch, pads):
    _install()
    monkeypatch.setenv("GMLX_CASCADE_MIN_P", "256")
    P, sp = 2048, 96
    q, k, v = _shared_batch(4, 16, 8, P, sp, pads)
    cache = _FakeBatchCache(pads, P=P)
    n0 = cs.claims()
    got = llama_mod.scaled_dot_product_attention(
        q, k, v, cache=cache, scale=0.125, mask=None)
    assert cs.claims() == n0 + 1
    ref = _masked_ref(q, k, v, pads, 0.125)
    err = mx.abs(got.astype(mx.float32) - ref).max().item()
    assert err < 2e-2, f"cascade vs masked ref err={err}"


@pytest.mark.skipif(not _has_cascade_op(), reason="fused cascade op needs GPU")
def test_non_claims_fall_through(monkeypatch):
    _install()
    monkeypatch.setenv("GMLX_CASCADE_MIN_P", "256")
    P = 1024
    q, k, v = _shared_batch(2, 16, 8, P, 64, [0, 0])
    n0 = cs.claims()

    # no stamp
    llama_mod.scaled_dot_product_attention(
        q, k, v, cache=_FakeBatchCache([0, 0]), scale=0.125, mask=None)
    # B == 1
    llama_mod.scaled_dot_product_attention(
        q[:1], k[:1], v[:1], cache=_FakeBatchCache([0], P=P), scale=0.125,
        mask=None)
    # quantized cache (checked via _claim: the stock fallback would then
    # route the fake into the real quantized path and fail on its own)
    qc = _FakeBatchCache([0, 0], P=P)
    qc.bits = 8
    assert cs._claim(q, k, v, qc, 0.125, None, None) is None
    # P below MIN_P
    llama_mod.scaled_dot_product_attention(
        q, k, v, cache=_FakeBatchCache([0, 0], P=128), scale=0.125, mask=None)
    # verify width (qL > 1)
    q2 = mx.concatenate([q, q], axis=2)
    llama_mod.scaled_dot_product_attention(
        q2, k, v, cache=_FakeBatchCache([0, 0], P=P), scale=0.125, mask=None)
    assert cs.claims() == n0


@pytest.mark.skipif(not _has_cascade_op(), reason="fused cascade op needs GPU")
def test_starts_memoized_on_pad_identity(monkeypatch):
    _install()
    monkeypatch.setenv("GMLX_CASCADE_MIN_P", "256")
    P = 1024
    pads = [0, 32]
    q, k, v = _shared_batch(2, 16, 8, P, 64, pads)
    cache = _FakeBatchCache(pads, P=P)
    llama_mod.scaled_dot_product_attention(
        q, k, v, cache=cache, scale=0.125, mask=None)
    memo = cache._gmlx_casc_starts
    llama_mod.scaled_dot_product_attention(
        q, k, v, cache=cache, scale=0.125, mask=None)
    assert cache._gmlx_casc_starts is memo
    cache.left_padding = mx.array(pads)  # rebinding invalidates
    llama_mod.scaled_dot_product_attention(
        q, k, v, cache=cache, scale=0.125, mask=None)
    assert cache._gmlx_casc_starts is not memo


def test_install_idempotent_and_killable(monkeypatch):
    _install()
    route = llama_mod.scaled_dot_product_attention
    assert cs.install_cascade_sdpa()
    assert llama_mod.scaled_dot_product_attention is route
    monkeypatch.setenv("GMLX_CASCADE_SDPA", "0")
    cs._installed_route = False
    assert not cs.install_cascade_sdpa()


# ---------------------------------------------------------------------------
# Stamp detection (host-only)
# ---------------------------------------------------------------------------


class _FakeBlock:
    def __init__(self, ntok):
        self.keys = [mx.zeros((1, 2, ntok, 8))]
        self.values = [mx.zeros((1, 2, ntok, 8))]


def test_common_prefix_block_identity():
    a, b, c, d = _FakeBlock(16), _FakeBlock(16), _FakeBlock(16), _FakeBlock(16)
    picks = [
        {"matched_blocks": [a, b, c], "prefix_len": 48},
        {"matched_blocks": [a, b, d], "prefix_len": 48},
    ]
    assert cs._common_prefix_tokens(picks) == 32  # a + b shared, c != d
    picks[1]["matched_blocks"] = [a, b, c]
    assert cs._common_prefix_tokens(picks) == 48
    # equal-content but distinct objects share nothing (identity only)
    picks[1]["matched_blocks"] = [_FakeBlock(16), b, c]
    assert cs._common_prefix_tokens(picks) == 0


def test_common_prefix_cold_and_scalar():
    a = _FakeBlock(16)
    assert cs._common_prefix_tokens([{"matched_blocks": [a],
                                      "prefix_len": 16}]) == 0
    assert cs._common_prefix_tokens([
        {"matched_blocks": [a], "prefix_len": 16}, None]) == 0


def test_common_prefix_exact_mode():
    warm = object()
    picks = [
        {"warm_cache": warm, "prefix_len": 512, "matched_blocks": None},
        {"warm_cache": warm, "prefix_len": 512, "matched_blocks": None},
    ]
    assert cs._common_prefix_tokens(picks) == 512
    picks[1]["warm_cache"] = object()
    assert cs._common_prefix_tokens(picks) == 0


def test_stamp_dropped_on_extend_kept_on_filter():
    cache = _FakeBatchCache([0, 0])
    cs._stamp_caches([cache], 1024)
    assert cache._gmlx_cascade == {"P": 1024}
    assert cache.extend(object()) == "extended"
    assert not hasattr(cache, "_gmlx_cascade")
    # a second extend still works (wrapper is single-shot on the attr)
    assert cache.extend(object()) == "extended"


def test_warm_multi_wrapper_stamps():
    a, b = _FakeBlock(16), _FakeBlock(16)
    picks = [
        {"matched_blocks": [a, b], "prefix_len": 32},
        {"matched_blocks": [a, b], "prefix_len": 32},
    ]
    caches = [_FakeBatchCache([0, 0]), _FakeBatchCache([0, 0])]

    def fake_orig(picks, num_layers):
        return caches, 32

    wrapped = cs._make_stamped_warm_multi(fake_orig)
    out_caches, max_prefix = wrapped(picks, 2)
    assert out_caches is caches and max_prefix == 32
    assert all(c._gmlx_cascade == {"P": 32} for c in caches)

    # cold row -> no stamp
    caches2 = [_FakeBatchCache([0, 0])]
    wrapped2 = cs._make_stamped_warm_multi(lambda p, n: (caches2, 32))
    wrapped2([picks[0], None], 1)
    assert not hasattr(caches2[0], "_gmlx_cascade")


def test_stamp_install_killable(monkeypatch):
    monkeypatch.setenv("GMLX_CASCADE_SDPA", "0")
    cs._installed_stamp = False
    assert not cs.install_cascade_stamp()
