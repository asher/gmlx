"""Left-padded batched MTP verify attention: one masked SDPA call must
reproduce the stock per-pad-group loop (mx.take of full-depth K/V per
group + per-query-position SDPA) exactly. Pins the route decisions too:
all-zero pads no longer bail to upstream (which declines them and lands
in stock slow attention), L==1 with real pads still bails (that is the
ragged decode kernel's case, not verify's), and GMLX_VERIFY_RAGGED_MASK=0
restores the stock group loop per call.
"""

import os
from types import SimpleNamespace

import mlx.core as mx
import pytest

pytest.importorskip("mlx_vlm.models.qwen3_5.language")
import mlx_vlm.models.qwen3_5.language as q35l

from gmlx import gdn_patches


def _install():
    gdn_patches._patch_batched_verify_sdpa()
    patched = q35l._target_verify_left_padded_attention
    stock = gdn_patches._STOCK_VERIFY_LEFT_PADDED
    assert stock is not None and patched is not stock
    return patched, stock


def _case(pads, B=4, nq=8, nkv=2, L=4, T=96, d=128, dtype=mx.bfloat16):
    mx.random.seed(7)
    q = mx.random.normal((B, nq, L, d)).astype(dtype)
    k = mx.random.normal((B, nkv, T, d)).astype(dtype)
    v = mx.random.normal((B, nkv, T, d)).astype(dtype)
    cache = SimpleNamespace()
    if pads is not None:
        cache._qwen3_5_decode_left_padding = list(pads)
    mx.eval(q, k, v)
    return q, k, v, cache


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float32])
def test_ragged_mask_matches_stock_loop(dtype):
    patched, stock = _install()
    q, k, v, cache = _case([0, 7, 33, 12], dtype=dtype)
    got = patched(q, k, v, cache=cache, scale=0.125, mask=None)
    ref = stock(mx.array(q), k, v, cache=cache, scale=0.125, mask=None)
    assert got is not None and ref is not None
    mx.eval(got, ref)
    assert got.dtype == ref.dtype
    if dtype == mx.bfloat16:
        if os.environ.get("KQUANT_FORCE_CPU"):
            # The CPU sdpa fallback rounds bf16 differently; bit-exactness
            # is a Metal serve-path claim.
            assert mx.allclose(got, ref, atol=2e-2, rtol=2e-2).item()
        else:
            # Production dtype: bit-exact against the group loop.
            assert mx.array_equal(got, ref).item()
    else:
        # fp32 differs by 1-2 ulps (reduction order of the masked full-T
        # kernel vs the sliced per-position calls), max 1.8e-7 measured.
        assert mx.allclose(got, ref, atol=1e-6, rtol=1e-5).item()


def test_zero_pads_takes_fast_path():
    patched, stock = _install()
    q, k, v, cache = _case([0, 0, 0, 0])
    # Stock declines the all-zero case entirely (returns None -> caller
    # falls back); the patched fn must answer with the batched call.
    assert stock(mx.array(q), k, v, cache=cache, scale=0.125, mask=None) is None
    got = patched(q, k, v, cache=cache, scale=0.125, mask=None)
    assert got is not None
    ref = mx.fast.scaled_dot_product_attention(
        q, k, v, scale=0.125, mask="causal")
    assert mx.array_equal(got, ref).item()


@pytest.mark.skipif(bool(os.environ.get("KQUANT_FORCE_CPU")),
                    reason="stock L==1 answers with the ragged decode Metal kernel")
def test_l1_with_pads_still_bails_to_stock():
    patched, stock = _install()
    q, k, v, cache = _case([0, 7, 33, 12], L=1)
    got = patched(q, k, v, cache=cache, scale=0.125, mask=None)
    ref = stock(mx.array(q), k, v, cache=cache, scale=0.125, mask=None)
    # Whatever stock does at L==1 (ragged decode kernel or group loop),
    # the patch must not intercept it.
    if ref is None:
        assert got is None
    else:
        mx.eval(got, ref)
        assert mx.array_equal(got, ref).item()


def test_env_kill_restores_stock_loop(monkeypatch):
    patched, stock = _install()
    q, k, v, cache = _case([0, 7, 33, 12])
    monkeypatch.setenv("GMLX_VERIFY_RAGGED_MASK", "0")
    got = patched(q, k, v, cache=cache, scale=0.125, mask=None)
    ref = stock(mx.array(q), k, v, cache=cache, scale=0.125, mask=None)
    mx.eval(got, ref)
    assert mx.array_equal(got, ref).item()


def test_array_mask_with_pads_bails():
    patched, stock = _install()
    q, k, v, cache = _case([0, 7, 33, 12])
    arr_mask = mx.zeros((1, 1, 4, 96), dtype=q.dtype)
    got = patched(q, k, v, cache=cache, scale=0.125, mask=arr_mask)
    ref = stock(mx.array(q), k, v, cache=cache, scale=0.125, mask=arr_mask)
    if ref is None:
        assert got is None
    else:
        mx.eval(got, ref)
        assert mx.array_equal(got, ref).item()


def test_pad_len_mismatch_bails():
    # A pads list shorter than B violates an upstream invariant; stock
    # raises on it. The patch must forward, not silently mask over it.
    patched, stock = _install()
    q, k, v, cache = _case([0, 7])  # 2 pads for B=4
    with pytest.raises(KeyError):
        stock(mx.array(q), k, v, cache=cache, scale=0.125, mask=None)
    with pytest.raises(KeyError):
        patched(q, k, v, cache=cache, scale=0.125, mask=None)
