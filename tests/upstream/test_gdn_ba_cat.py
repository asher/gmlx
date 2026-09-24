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

patches = importlib.import_module("gmlx.upstream.gdn_patches")


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


def _adapted_gdn(leaves, merge_zba=False):
    """A one-layer tree whose GatedDeltaNet was armed at load, then took a
    GGUF adapter on ``leaves``."""
    from mlx_lm.models.qwen3_5 import TextModelArgs

    from gmlx.load.adapter import LoraAdapter, LoraModule
    from gmlx.load.modules import install_lora_adapter

    args = TextModelArgs(
        model_type="qwen3_5", hidden_size=64, intermediate_size=128,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
        rms_norm_eps=1e-6, vocab_size=32, linear_num_value_heads=2,
        linear_num_key_heads=2, linear_key_head_dim=32,
        linear_value_head_dim=32, linear_conv_kernel_dim=4,
        full_attention_interval=4, head_dim=32, rope_theta=1e4,
        max_position_embeddings=128)

    class Root(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = [nn.Module()]
            self.layers[0].linear_attn = GatedDeltaNet(args)

    mx.random.seed(0)
    root = Root()
    gdn = root.layers[0].linear_attn
    mx.eval(root.parameters())
    if merge_zba:
        assert patches._gdn_try_merge_zba(gdn)
    else:
        assert patches._gdn_try_cat_ba(gdn)
    r = 4
    modules = {}
    for leaf in leaves:
        path = f"layers.0.linear_attn.{leaf}"
        out = getattr(gdn, leaf).weight.shape[0]
        modules[path] = LoraModule(
            path, mx.random.normal((r, 64)) * 0.5,
            mx.random.normal((out, r)) * 0.5, r, 1.0)
    install_lora_adapter(root, LoraAdapter(alpha=float(r), arch="qwen35",
                                           modules=modules))
    return gdn


@pytest.mark.parametrize("merge_zba,leaves", [
    (False, ("in_proj_b",)),
    (False, ("in_proj_a",)),
    (True, ("in_proj_z",)),
    (True, ("in_proj_b", "in_proj_a")),
])
def test_an_adapter_on_z_b_or_a_reaches_the_fused_decode(monkeypatch, merge_zba, leaves):
    """The cat and the merge are built at load. An adapter installed later
    on one of their members clears them, so decode calls the wrapper."""
    from mlx_lm.models.cache import ArraysCache

    gdn = _adapted_gdn(leaves, merge_zba=merge_zba)
    assert getattr(gdn, "_gdn_ba_weight", None) is None
    assert getattr(gdn, "_gdn_zba_weight", None) is None
    seen = {}

    def kernel(inputs, template, grid, threadgroup, output_shapes, output_dtypes):
        seen["a"], seen["b"], seen["z"] = inputs[2], inputs[3], inputs[7]
        return [mx.zeros(output_shapes[0], output_dtypes[0]), inputs[6]]

    monkeypatch.setattr(patches, "_gdn_fused_decode_kernel", kernel)
    x = mx.random.normal((1, 1, 64))
    cache = ArraysCache(size=2)
    cache[0] = mx.zeros((1, 3, gdn.conv_dim))
    cache[1] = mx.zeros((1, 2, 32, 32))
    patches._gdn_fused_decode_body(gdn, x, cache)
    assert mx.allclose(seen["b"], gdn.in_proj_b(x).reshape(seen["b"].shape))
    assert mx.allclose(seen["a"], gdn.in_proj_a(x).reshape(seen["a"].shape))
    assert mx.allclose(seen["z"], gdn.in_proj_z(x).reshape(seen["z"].shape))


def test_an_adapter_elsewhere_keeps_the_cat():
    gdn = _adapted_gdn(("in_proj_qkv",))
    assert getattr(gdn, "_gdn_ba_weight", None) is not None


def test_the_fused_verify_calls_adapted_b_a_and_out(monkeypatch):
    """The verify body reads each projection's weight for the head GEMV. A
    wrapper has none, so it is called instead."""
    gdn = _adapted_gdn(("in_proj_b", "in_proj_a", "out_proj"))
    seen = {}

    def kernel(inputs, template, grid, threadgroup, output_shapes, output_dtypes):
        seen["ba"] = inputs[2]
        return [mx.zeros(s, d) for s, d in zip(output_shapes, output_dtypes)]

    monkeypatch.setattr(patches, "_gdn_fused_verify_kernel", kernel)
    monkeypatch.setattr(patches, "_F16_HEAD_GEMV", object())
    monkeypatch.setattr(patches, "_f16_head_gemv", lambda x, w: x @ w.T)
    x = mx.random.normal((1, 3, 64))
    out = patches._gdn_fused_verify_body(gdn, x, None, None, [])
    want = mx.concatenate([gdn.in_proj_b(x), gdn.in_proj_a(x)], axis=-1)
    assert mx.allclose(seen["ba"], want)
    assert mx.allclose(out, gdn.out_proj(mx.zeros((1, 3, gdn.value_dim))))
