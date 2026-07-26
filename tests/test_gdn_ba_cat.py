#!/usr/bin/env python3
"""The b/a concat for the fused gated-delta decode step.

On k-quant files the z projection is quantized, so the zba merge can never
fire; b/a load as tiny plain bf16 Linears whose per-step matvecs fall onto
MLX's steel GEMM tile at batched decode (M>=2), costing ~70 us wall each
for ~1 MB of weights. `_gdn_try_cat_ba` concatenates them into one
[2*Hv, K] weight the decode body routes through the M-stationary head
kernel. The cat owns the rows and the original modules' weights become
row-slice views, so prefill and every stock path are untouched.
"""

from __future__ import annotations

import importlib
import os

import mlx.core as mx
import mlx.nn as nn
import pytest

from mlx_lm.models.qwen3_5 import GatedDeltaNet

patches = importlib.import_module("gmlx.gdn_patches")


def _bare_gdn(dtype=mx.float32, hv=2, k=4):
    gdn = GatedDeltaNet.__new__(GatedDeltaNet)
    nn.Module.__init__(gdn)
    gdn.value_dim = hv * 4
    gdn.num_v_heads = hv
    gdn.in_proj_z = nn.Linear(k, hv * 4, bias=False)
    gdn.in_proj_b = nn.Linear(k, hv, bias=False)
    gdn.in_proj_a = nn.Linear(k, hv, bias=False)
    if dtype != mx.float32:
        gdn.in_proj_b.weight = gdn.in_proj_b.weight.astype(dtype)
        gdn.in_proj_a.weight = gdn.in_proj_a.weight.astype(dtype)
    return gdn


def test_cat_content_and_views():
    gdn = _bare_gdn()
    wb, wa = gdn.in_proj_b.weight, gdn.in_proj_a.weight
    assert patches._gdn_try_cat_ba(gdn)
    expect = mx.concatenate([wb, wa], axis=0)
    assert mx.array_equal(gdn._gdn_ba_weight, expect)
    # originals stay usable as row-slice views (stock paths untouched)
    assert mx.array_equal(gdn.in_proj_b.weight, wb)
    assert mx.array_equal(gdn.in_proj_a.weight, wa)


def test_cat_refuses_after_zba_merge():
    """A merged instance's b/a weights are already views into the zba
    weight; a second concat would duplicate rows."""
    gdn = _bare_gdn()
    assert patches._gdn_try_merge_zba(gdn)
    assert not patches._gdn_try_cat_ba(gdn)
    assert getattr(gdn, "_gdn_ba_weight", None) is None


def test_cat_refusals():
    gdn = _bare_gdn()
    gdn.in_proj_b = nn.Linear(4, 2, bias=True)
    assert not patches._gdn_try_cat_ba(gdn)  # bias

    gdn = _bare_gdn()
    gdn.in_proj_a.weight = gdn.in_proj_a.weight.astype(mx.bfloat16)
    assert not patches._gdn_try_cat_ba(gdn)  # dtype mismatch

    gdn = _bare_gdn()
    gdn.in_proj_b = nn.Linear(4, 3, bias=False)
    assert not patches._gdn_try_cat_ba(gdn)  # rows != num_v_heads

    class _Quantish(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = mx.zeros((2, 4))

    gdn = _bare_gdn()
    gdn.in_proj_b = _Quantish()
    assert not patches._gdn_try_cat_ba(gdn)  # not a plain nn.Linear


def test_cat_default_on_and_kill_switch(monkeypatch):
    if patches._gdn_fused_decode_kernel is None:
        pytest.skip("fused gdn decode kernel unavailable")
    saved_call = GatedDeltaNet.__call__
    saved_installed = patches._FUSED_DECODE_PATCH.installed
    saved_stock = patches._FUSED_DECODE_PATCH.stock
    catted = []
    monkeypatch.setattr(
        patches, "_gdn_try_cat_ba", lambda m: bool(catted.append(m)))
    gdn = GatedDeltaNet.__new__(GatedDeltaNet)

    class _One:
        def modules(self):
            return [gdn]

    try:
        monkeypatch.delenv("GMLX_GDN_ZBA", raising=False)
        monkeypatch.setenv("GMLX_GDN_BA_CAT", "0")
        patches._patch_gated_delta_fused_decode(_One())
        assert not catted  # kill switch
        monkeypatch.delenv("GMLX_GDN_BA_CAT", raising=False)
        patches._patch_gated_delta_fused_decode(_One())
        assert len(catted) == 1  # default on
    finally:
        GatedDeltaNet.__call__ = saved_call
        patches._FUSED_DECODE_PATCH.installed = saved_installed
        patches._FUSED_DECODE_PATCH.stock = saved_stock


def test_zba_merge_wins_over_cat(monkeypatch):
    """Per instance the two are exclusive: with the merge opted in, the cat
    branch must not run (its rows are inside the merged weight)."""
    if patches._gdn_fused_decode_kernel is None:
        pytest.skip("fused gdn decode kernel unavailable")
    saved_call = GatedDeltaNet.__call__
    saved_installed = patches._FUSED_DECODE_PATCH.installed
    saved_stock = patches._FUSED_DECODE_PATCH.stock
    calls = []
    monkeypatch.setattr(
        patches, "_gdn_try_merge_zba",
        lambda m: (calls.append("zba"), True)[1])
    monkeypatch.setattr(
        patches, "_gdn_try_cat_ba",
        lambda m: (calls.append("ba"), True)[1])
    gdn = GatedDeltaNet.__new__(GatedDeltaNet)

    class _One:
        def modules(self):
            return [gdn]

    try:
        monkeypatch.setenv("GMLX_GDN_ZBA", "1")
        monkeypatch.delenv("GMLX_GDN_BA_CAT", raising=False)
        patches._patch_gated_delta_fused_decode(_One())
        assert calls == ["zba"]
    finally:
        GatedDeltaNet.__call__ = saved_call
        patches._FUSED_DECODE_PATCH.installed = saved_installed
        patches._FUSED_DECODE_PATCH.stock = saved_stock


def test_verify_patcher_cats_vlm_instances(monkeypatch):
    from mlx_vlm.models.qwen3_5.language import Qwen3_5GatedDeltaNet

    if patches._gdn_fused_verify_kernel is None:
        pytest.skip("fused gdn verify kernel unavailable")
    saved_call = Qwen3_5GatedDeltaNet.__call__
    saved_installed = patches._FUSED_VERIFY_PATCH.installed
    saved_stock = patches._FUSED_VERIFY_PATCH.stock
    catted = []
    monkeypatch.setattr(
        patches, "_gdn_try_cat_ba", lambda m: bool(catted.append(m)))
    gdn = Qwen3_5GatedDeltaNet.__new__(Qwen3_5GatedDeltaNet)

    class _One:
        def modules(self):
            return [gdn]

    try:
        monkeypatch.delenv("GMLX_GDN_BA_CAT", raising=False)
        patches._patch_gated_delta_fused_verify(_One())
        assert len(catted) == 1
        monkeypatch.setenv("GMLX_GDN_BA_CAT", "0")
        catted.clear()
        patches._patch_gated_delta_fused_verify(_One())
        assert not catted
    finally:
        Qwen3_5GatedDeltaNet.__call__ = saved_call
        patches._FUSED_VERIFY_PATCH.installed = saved_installed
        patches._FUSED_VERIFY_PATCH.stock = saved_stock


@pytest.mark.skipif(bool(os.environ.get("KQUANT_FORCE_CPU")),
                    reason="_f16_head_gemv is a Metal kernel")
@pytest.mark.parametrize("m", [2, 4, 8])
def test_head_gemv_route_parity(m):
    """The decode-body route: cat weight through _f16_head_gemv must match
    the two separate linears (f32-accum on both sides; bf16 storage)."""
    if patches._F16_HEAD_GEMV is None:
        pytest.skip("f16 head gemv kernel unavailable")
    k, hv = 256, 4
    gdn = _bare_gdn(dtype=mx.bfloat16, hv=hv, k=k)
    assert patches._gdn_try_cat_ba(gdn)
    x = mx.random.normal((m, 1, k)).astype(mx.bfloat16)
    got = patches._f16_head_gemv(
        x.reshape(1, m, k), gdn._gdn_ba_weight).reshape(m, 1, 2 * hv)
    b_ref = gdn.in_proj_b(x)
    a_ref = gdn.in_proj_a(x)
    ref = mx.concatenate([b_ref, a_ref], axis=-1)
    assert mx.allclose(got.astype(mx.float32), ref.astype(mx.float32),
                       atol=1e-2, rtol=1e-2)


def test_cat_matmul_fallback_parity():
    """The M>8 (or no-kernel) fallback: inputs @ cat.T equals the separate
    projections exactly up to matmul accumulation order."""
    gdn = _bare_gdn(dtype=mx.bfloat16, hv=4, k=256)
    assert patches._gdn_try_cat_ba(gdn)
    x = mx.random.normal((16, 1, 256)).astype(mx.bfloat16)
    got = x @ gdn._gdn_ba_weight.T
    ref = mx.concatenate([gdn.in_proj_b(x), gdn.in_proj_a(x)], axis=-1)
    assert mx.allclose(got.astype(mx.float32), ref.astype(mx.float32),
                       atol=1e-2, rtol=1e-2)
