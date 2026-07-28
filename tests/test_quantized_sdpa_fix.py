"""Upstream quantized-KV SDPA breaks on B>1 GQA array masks: 5D grouped
scores vs a 4D batch mask raises a broadcast error when B != n_kv_heads
and silently misapplies the mask (batch dim lands in the head slot) when
B == n_kv_heads. The fix inserts one mask axis; these tests pin the raise
repro, both numerics cases against a dequantized reference, the
ungrouped/B=1 pass-through, and install hygiene. The same wrapper routes
prefill-width calls (qL >= GMLX_KV8_PREFILL_FLASH_MIN_L) through
dequant + fused flash SDPA; those tests pin numerics vs both the
dequantized reference and the stock quantized body, the threshold, and
the kill switch. Decode-width calls (qL == 1) route through the fused
mlx_kquant kernel when its build carries q8 operands + starts; those
tests pin the left-pad mask -> starts conversion, the non-left-pad and
float-mask fallbacks, the identity memo, and the kill switch.

Upstream mutates `queries` in place (`queries *= scale`), so every call
here gets a fresh copy -- reusing one q across calls poisons comparisons.
"""

import mlx.core as mx
import pytest

from mlx_lm.models import base as lm_base

pytest.importorskip("mlx_vlm.models.base")
from mlx_vlm.models import base as vlm_base

from gmlx import quantized_sdpa_fix as qf

_orig_lm = lm_base.quantized_scaled_dot_product_attention
_orig_vlm = vlm_base.quantized_scaled_dot_product_attention


def teardown_module(module):
    lm_base.quantized_scaled_dot_product_attention = _orig_lm
    vlm_base.quantized_scaled_dot_product_attention = _orig_vlm
    qf._installed = False


def _case(B=2, nq=8, nkv=4, L=1, kv=64, d=64, pads=(5, 0),
          dtype=mx.float32):
    mx.random.seed(11)
    q = mx.random.normal((B, nq, L, d)).astype(dtype)
    k = mx.random.normal((B, nkv, kv, d)).astype(dtype)
    v = mx.random.normal((B, nkv, kv, d)).astype(dtype)
    qk = mx.quantize(k, group_size=64, bits=8)
    qv = mx.quantize(v, group_size=64, bits=8)
    pos = mx.arange(kv)[None, None, None, :]
    mask = pos >= mx.array(list(pads))[:B, None, None, None]
    mx.eval(q, mask, *qk, *qv)
    return q, qk, qv, mask


def _ref(q, qk, qv, mask, scale=0.125):
    kd = mx.dequantize(*qk, group_size=64, bits=8)
    vd = mx.dequantize(*qv, group_size=64, bits=8)
    return mx.fast.scaled_dot_product_attention(
        mx.array(q), kd, vd, scale=scale, mask=mask)


def test_upstream_repro_and_fix_numerics():
    # B != n_kv_heads: upstream raises on the 5D/4D broadcast
    q, qk, qv, mask = _case(B=2, nq=8, nkv=4)
    with pytest.raises(ValueError):
        _orig_lm(mx.array(q), qk, qv, 0.125, mask)

    assert qf.install_quantized_sdpa_mask_fix()
    got = lm_base.quantized_scaled_dot_product_attention(
        mx.array(q), qk, qv, 0.125, mask)
    err = mx.abs(got - _ref(q, qk, qv, mask)).max().item()
    assert err < 2e-2, f"quantized vs dequantized ref err={err}"


def test_silent_misapply_case_fixed():
    # B == n_kv_heads: upstream broadcasts WITHOUT raising, applying the
    # mask's batch dim as the kv-head dim -- silently wrong. The fix must
    # produce reference numerics on exactly this shape.
    q, qk, qv, mask = _case(B=2, nq=8, nkv=2, pads=(9, 0))
    assert qf.install_quantized_sdpa_mask_fix()
    got = lm_base.quantized_scaled_dot_product_attention(
        mx.array(q), qk, qv, 0.125, mask)
    err = mx.abs(got - _ref(q, qk, qv, mask)).max().item()
    assert err < 2e-2, f"fixed numerics err={err}"
    bad = _orig_lm(mx.array(q), qk, qv, 0.125, mask)
    assert mx.abs(bad - _ref(q, qk, qv, mask)).max().item() > 0.05


def test_vlm_twin_fixed_too():
    assert qf.install_quantized_sdpa_mask_fix()
    q, qk, qv, mask = _case(B=3, nq=8, nkv=2, pads=(7, 2, 0))
    got = vlm_base.quantized_scaled_dot_product_attention(
        mx.array(q), qk, qv, 0.125, mask)
    err = mx.abs(got - _ref(q, qk, qv, mask)).max().item()
    assert err < 2e-2, f"vlm fixed numerics err={err}"


def test_ungrouped_and_b1_pass_through(monkeypatch):
    # kq route off: this test pins the mask handling of the stock body.
    monkeypatch.setenv("GMLX_QSDPA_KQ", "0")
    assert qf.install_quantized_sdpa_mask_fix()
    # no grouping (nq == nkv): wrapper must not change the mask
    q, qk, qv, mask = _case(B=2, nq=2, nkv=2)
    got = lm_base.quantized_scaled_dot_product_attention(
        mx.array(q), qk, qv, 0.125, mask)
    ref = _orig_lm(mx.array(q), qk, qv, 0.125, mask)
    assert mx.abs(got - ref).max().item() == 0.0
    # B=1 grouped worked upstream already; fix must agree with it
    q, qk, qv, mask = _case(B=1, nq=8, nkv=4, pads=(3,))
    got = lm_base.quantized_scaled_dot_product_attention(
        mx.array(q), qk, qv, 0.125, mask)
    ref = _orig_lm(mx.array(q), qk, qv, 0.125, mask)
    assert mx.abs(got - ref).max().item() == 0.0


def test_prefill_flash_matches_dequant_reference():
    # qL >= MIN_L routes dequant+fused-flash; identical ops to _ref -> exact.
    assert qf.install_quantized_sdpa_mask_fix()
    q, qk, qv, mask = _case(B=2, nq=8, nkv=4, L=64, kv=256, pads=(5, 0))
    got = lm_base.quantized_scaled_dot_product_attention(
        mx.array(q), qk, qv, 0.125, mask)
    assert mx.abs(got - _ref(q, qk, qv, mask)).max().item() == 0.0


def test_prefill_flash_matches_stock_numerics():
    # Cross-check the flash path against the stock quantized body (mask
    # pre-expanded for the upstream 5D bug) at prefill width.
    assert qf.install_quantized_sdpa_mask_fix()
    q, qk, qv, mask = _case(B=2, nq=8, nkv=4, L=32, kv=256, pads=(5, 0))
    got = lm_base.quantized_scaled_dot_product_attention(
        mx.array(q), qk, qv, 0.125, mask)
    stock = _orig_lm(mx.array(q), qk, qv, 0.125, mask[:, None])
    err = mx.abs(got - stock).max().item()
    assert err < 2e-2, f"flash vs stock err={err}"


def test_prefill_flash_causal_str_mask():
    assert qf.install_quantized_sdpa_mask_fix()
    q, qk, qv, _ = _case(B=1, nq=8, nkv=4, L=16, kv=64)
    got = lm_base.quantized_scaled_dot_product_attention(
        mx.array(q), qk, qv, 0.125, "causal")
    stock = _orig_lm(mx.array(q), qk, qv, 0.125, "causal")
    err = mx.abs(got - stock).max().item()
    assert err < 2e-2, f"causal flash vs stock err={err}"


def test_prefill_flash_threshold_and_kill(monkeypatch):
    assert qf.install_quantized_sdpa_mask_fix()
    # below MIN_L: stock quantized path (compare exact vs orig)
    q, qk, qv, mask = _case(B=1, nq=8, nkv=4, L=4, kv=64, pads=(3,))
    got = lm_base.quantized_scaled_dot_product_attention(
        mx.array(q), qk, qv, 0.125, mask)
    ref = _orig_lm(mx.array(q), qk, qv, 0.125, mask)
    assert mx.abs(got - ref).max().item() == 0.0
    # raised threshold: prefill width falls back to stock
    monkeypatch.setenv("GMLX_KV8_PREFILL_FLASH_MIN_L", "128")
    q, qk, qv, mask = _case(B=1, nq=8, nkv=4, L=64, kv=256, pads=(3,))
    got = lm_base.quantized_scaled_dot_product_attention(
        mx.array(q), qk, qv, 0.125, mask)
    ref = _orig_lm(mx.array(q), qk, qv, 0.125, mask)
    assert mx.abs(got - ref).max().item() == 0.0
    monkeypatch.delenv("GMLX_KV8_PREFILL_FLASH_MIN_L", raising=False)
    # kill switch: flash off, stock path even at prefill width
    monkeypatch.setenv("GMLX_KV8_PREFILL_FLASH", "0")
    got = lm_base.quantized_scaled_dot_product_attention(
        mx.array(q), qk, qv, 0.125, mask)
    ref = _orig_lm(mx.array(q), qk, qv, 0.125, mask)
    assert mx.abs(got - ref).max().item() == 0.0


def test_install_idempotent_and_killable(monkeypatch):
    lm_base.quantized_scaled_dot_product_attention = _orig_lm
    vlm_base.quantized_scaled_dot_product_attention = _orig_vlm
    qf._installed = False
    monkeypatch.setenv("GMLX_QSDPA_MASK_FIX", "0")
    assert qf.install_quantized_sdpa_mask_fix() is False
    assert lm_base.quantized_scaled_dot_product_attention is _orig_lm

    monkeypatch.delenv("GMLX_QSDPA_MASK_FIX", raising=False)
    assert qf.install_quantized_sdpa_mask_fix()
    pl = lm_base.quantized_scaled_dot_product_attention
    pv = vlm_base.quantized_scaled_dot_product_attention
    assert pl._gmlx_orig is _orig_lm and pv._gmlx_orig is _orig_vlm
    assert qf.install_quantized_sdpa_mask_fix()
    assert lm_base.quantized_scaled_dot_product_attention is pl
    assert vlm_base.quantized_scaled_dot_product_attention is pv


needs_kq = pytest.mark.skipif(
    qf._kq_q8_route() is None,
    reason="mlx_kquant build without q8+starts sdpa_decode_gqa")


def _spy_wrapper():
    """A wrapper built around the real kq route with call recording."""
    real = qf._kq_q8_route()
    calls = []

    def spy(*a, **kw):
        calls.append(kw.get("starts"))
        return real(*a, **kw)

    return qf._make_fixed(_orig_lm, spy), calls


@needs_kq
def test_kq_decode_route_registered_leftpad_mask():
    # a REGISTERED boolean left-pad mask routes with its starts
    fixed, calls = _spy_wrapper()
    q, qk, qv, mask = _case(B=3, nq=8, nkv=4, kv=512, pads=(37, 5, 0),
                            dtype=mx.float16)
    qf._STARTS_MEMO.clear()
    qf._register_starts(mask, mx.array([37, 5, 0]))
    got = fixed(mx.array(q), qk, qv, 0.125, mask)
    assert len(calls) == 1 and calls[0] is not None
    assert calls[0].tolist() == [37, 5, 0]
    err = mx.abs(got - _ref(q, qk, qv, mask)).max().item()
    assert err < 2e-2, f"kq route vs dequant ref err={err}"
    qf._STARTS_MEMO.clear()


@needs_kq
def test_make_mask_registers_and_routes():
    # the real seam: BatchQuantizedKVCache.make_mask registers its mask,
    # and a wire fetched from the same cache routes end to end
    pytest.importorskip("mlx_vlm.models.cache")
    from mlx_vlm.models import cache as vcache

    assert qf._patch_make_mask()
    assert vcache.BatchQuantizedKVCache.make_mask._gmlx_starts_reg
    qf._STARTS_MEMO.clear()
    mx.random.seed(11)
    pads = [37, 5, 0]
    c = vcache.BatchQuantizedKVCache(pads, group_size=64, bits=8)
    k = mx.random.normal((3, 4, 511, 64)).astype(mx.float16)
    v = mx.random.normal((3, 4, 511, 64)).astype(mx.float16)
    c.update_and_fetch(k, v)
    # per-step order: mask first (anticipates the new token), then update
    mask = c.make_mask(1)
    qk, qv = c.update_and_fetch(
        mx.random.normal((3, 4, 1, 64)).astype(mx.float16),
        mx.random.normal((3, 4, 1, 64)).astype(mx.float16))
    ent = qf._STARTS_MEMO.get(id(mask))
    assert ent is not None and ent[0] is mask
    assert ent[1].tolist() == pads and ent[1].dtype == mx.int32
    # windowed masks are NOT registered
    wmask = c.make_mask(1, window_size=64)
    assert qf._registered_starts(wmask) is None

    fixed, calls = _spy_wrapper()
    q = mx.random.normal((3, 8, 1, 64)).astype(mx.float16)
    got = fixed(mx.array(q), qk, qv, 0.125, mask)
    assert len(calls) == 1 and calls[0].tolist() == pads
    err = mx.abs(got - _ref(q, qk, qv, mask)).max().item()
    assert err < 2e-2, f"seam route vs dequant ref err={err}"
    qf._STARTS_MEMO.clear()


@needs_kq
def test_kq_decode_route_maskless_and_causal():
    fixed, calls = _spy_wrapper()
    q, qk, qv, _ = _case(B=2, nq=8, nkv=4, kv=256, dtype=mx.float16)
    got = fixed(mx.array(q), qk, qv, 0.125, None)
    got_c = fixed(mx.array(q), qk, qv, 0.125, "causal")
    assert len(calls) == 2 and calls[0] is None and calls[1] is None
    full = mx.ones((2, 1, 1, 256), dtype=mx.bool_)
    err = mx.abs(got - _ref(q, qk, qv, full)).max().item()
    assert err < 2e-2, f"maskless route err={err}"
    assert mx.abs(got_c - got).max().item() == 0.0


@needs_kq
def test_kq_decode_route_fallbacks():
    fixed, calls = _spy_wrapper()
    q, qk, qv, mask = _case(B=2, nq=8, nkv=4, kv=256, pads=(9, 0),
                            dtype=mx.float16)
    # UNREGISTERED boolean mask (unknown provenance) -> stock body, exact
    qf._STARTS_MEMO.clear()
    got = fixed(mx.array(q), qk, qv, 0.125, mask)
    ref = _orig_lm(mx.array(q), qk, qv, 0.125, mask[:, None])
    assert not calls
    assert mx.abs(got - ref).max().item() == 0.0
    # additive float mask -> stock body, exact
    fmask = mx.where(mask, mx.array(0.0, mx.float16),
                     mx.array(-mx.inf, mx.float16))
    got = fixed(mx.array(q), qk, qv, 0.125, fmask)
    ref = _orig_lm(mx.array(q), qk, qv, 0.125, fmask[:, None])
    assert not calls
    assert mx.abs(got - ref).max().item() == 0.0
    # fp32 queries -> stock body
    q32, qk32, qv32, m32 = _case(B=2, nq=8, nkv=4, kv=256, pads=(9, 0))
    got = fixed(mx.array(q32), qk32, qv32, 0.125, m32)
    ref = _orig_lm(mx.array(q32), qk32, qv32, 0.125, m32[:, None])
    assert not calls
    assert mx.abs(got - ref).max().item() == 0.0


@needs_kq
def test_kq_decode_route_kill_switch(monkeypatch):
    fixed, calls = _spy_wrapper()
    q, qk, qv, mask = _case(B=2, nq=8, nkv=4, kv=256, pads=(9, 0),
                            dtype=mx.float16)
    qf._STARTS_MEMO.clear()
    qf._register_starts(mask, mx.array([9, 0]))
    monkeypatch.setenv("GMLX_QSDPA_KQ", "0")
    got = fixed(mx.array(q), qk, qv, 0.125, mask)
    ref = _orig_lm(mx.array(q), qk, qv, 0.125, mask[:, None])
    assert not calls
    assert mx.abs(got - ref).max().item() == 0.0
    monkeypatch.delenv("GMLX_QSDPA_KQ", raising=False)
    fixed(mx.array(q), qk, qv, 0.125, mask)
    assert len(calls) == 1
    qf._STARTS_MEMO.clear()


def test_starts_registry_identity_and_cap():
    qf._STARTS_MEMO.clear()
    pos = mx.arange(64)[None, None, None, :]
    mask = pos >= mx.array([7, 0])[:, None, None, None]
    qf._register_starts(mask, mx.array([7, 0]))
    a = qf._registered_starts(mask)
    b = qf._registered_starts(mask)
    assert a is b and a.tolist() == [7, 0] and a.dtype == mx.int32
    assert len(qf._STARTS_MEMO) == 1
    # unregistered object -> None, even with identical contents
    twin = pos >= mx.array([7, 0])[:, None, None, None]
    assert qf._registered_starts(twin) is None
    for i in range(qf._STARTS_MEMO_CAP + 2):
        m = pos >= mx.array([i, 0])[:, None, None, None]
        qf._register_starts(m, mx.array([i, 0]))
    assert len(qf._STARTS_MEMO) == qf._STARTS_MEMO_CAP
    qf._STARTS_MEMO.clear()
