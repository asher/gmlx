"""Upstream quantized-KV SDPA breaks on B>1 GQA array masks: 5D grouped
scores vs a 4D batch mask raises a broadcast error when B != n_kv_heads
and silently misapplies the mask (batch dim lands in the head slot) when
B == n_kv_heads. The fix inserts one mask axis; these tests pin the raise
repro, both numerics cases against a dequantized reference, the
ungrouped/B=1 pass-through, and install hygiene. The same wrapper routes
prefill-width calls (qL >= GMLX_KV8_PREFILL_FLASH_MIN_L) through
dequant + fused flash SDPA; those tests pin numerics vs both the
dequantized reference and the stock quantized body, the threshold, and
the kill switch.

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


def _case(B=2, nq=8, nkv=4, L=1, kv=64, d=64, pads=(5, 0)):
    mx.random.seed(11)
    q = mx.random.normal((B, nq, L, d))
    k = mx.random.normal((B, nkv, kv, d))
    v = mx.random.normal((B, nkv, kv, d))
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


def test_ungrouped_and_b1_pass_through():
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
