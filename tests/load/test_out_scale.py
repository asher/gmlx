#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Asher Feldman
"""Per-tensor weight scales: ``install_out_scales`` multiplies a module's
output by its ``out_scale``, keeps float16 activations finite under a small
scale, applies an lm_head scale only to an NVFP4 head, refuses per-expert
scales, and the wire fusions skip a scaled projection."""

from __future__ import annotations

import os

import numpy as np
import pytest

import mlx.core as mx
import mlx.nn as nn

from gmlx.load.out_scale import has_out_scale, install_out_scales


class _Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(16, 8, bias=False)
        self.other = nn.Linear(16, 8, bias=False)
        self.lm_head = nn.Linear(16, 8, bias=False)


def test_scales_the_output_of_named_modules_only():
    model = _Tiny()
    weights = {"proj.out_scale": mx.array([0.25])}
    assert install_out_scales(model, weights) == 1
    model.load_weights(list(weights.items()), strict=False)
    x = mx.random.normal((2, 16))
    np.testing.assert_allclose(
        np.array(model.proj(x)), 0.25 * np.array(x @ model.proj.weight.T),
        rtol=1e-6)
    assert has_out_scale(model.proj) and not has_out_scale(model.other)
    assert isinstance(model.proj, nn.Linear)
    assert "out_scale" not in model.trainable_parameters()["proj"]


def test_install_is_idempotent():
    model = _Tiny()
    weights = {"proj.out_scale": mx.array([0.5])}
    install_out_scales(model, weights)
    cls = type(model.proj)
    install_out_scales(model, weights)
    assert type(model.proj) is cls


def test_float16_product_past_range_stays_finite():
    # A 1e-4 scale with weights 1e4 larger: the unscaled product overflows
    # float16, the scaled one does not.
    model = _Tiny()
    model.proj.weight = mx.full((8, 16), 1e4)
    weights = {"proj.out_scale": mx.array([1e-4])}
    install_out_scales(model, weights)
    model.load_weights(list(weights.items()), strict=False)
    y = model.proj(mx.full((1, 16), 1.0, dtype=mx.float16))
    assert y.dtype == mx.float16
    np.testing.assert_allclose(np.array(y.astype(mx.float32)), 16.0, rtol=1e-2)


def test_lm_head_scale_needs_an_nvfp4_head():
    model = _Tiny()
    weights = {"lm_head.out_scale": mx.array([0.5])}
    assert install_out_scales(model, weights, {"lm_head.weight": "q8_0"}) == 0
    assert "lm_head.out_scale" not in weights
    assert not has_out_scale(model.lm_head)
    weights = {"lm_head.out_scale": mx.array([0.5])}
    assert install_out_scales(model, weights, {"lm_head.weight": "nvfp4"}) == 1


def test_per_expert_scale_fails_loud():
    model = _Tiny()
    with pytest.raises(NotImplementedError, match="per-expert"):
        install_out_scales(model, {"proj.out_scale": mx.ones((4,))})


def test_wire_fusions_skip_a_scaled_projection():
    from mlx_kquant.nn import KQuantLinear

    from gmlx.load.modules import _kq_dense
    from gmlx.upstream.occupancy_fuse import _same_codec_kquant

    class Mlp(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_proj = KQuantLinear(64, 32, False, "q8_0")
            self.up_proj = KQuantLinear(64, 32, False, "q8_0")

    m = Mlp()
    assert _kq_dense(m.gate_proj, {"q8_0"})
    assert _same_codec_kquant([m.gate_proj, m.up_proj])
    install_out_scales(m, {"gate_proj.out_scale": mx.array([0.5])})
    assert isinstance(m.gate_proj, KQuantLinear)
    assert not _kq_dense(m.gate_proj, {"q8_0"})
    assert not _same_codec_kquant([m.gate_proj, m.up_proj])


class _Moe(nn.Module):
    def __init__(self):
        super().__init__()
        from gmlx._vendor.mlx_lm.models.switch_layers import SwitchGLU

        self.experts = SwitchGLU(256, 256, 8)


@pytest.mark.parametrize("proj", ["gate_proj", "up_proj", "down_proj"])
def test_fused_moe_skips_a_scaled_expert_stack(proj):
    """The fused MoE gathers read the expert stacks' wire bytes, so a
    stack with a weight scale keeps the stock gathers, which run the
    module's own call and apply the scale."""
    import mlx_kquant as kq

    from gmlx.load import modules

    if not hasattr(kq, "moe_glu_gather_kq"):
        pytest.skip("mlx_kquant without fused MoE kernels")

    def build():
        m = _Moe()
        m.eval()
        modules.install_kquant_modules(m, {
            f"experts.{p}.weight": "q8_0"
            for p in ("gate_proj", "up_proj", "down_proj")})
        return m

    assert modules.install_fused_moe_glu(build()) == 1
    m = build()
    install_out_scales(m, {f"experts.{proj}.out_scale": mx.array([0.5])})
    assert modules.install_fused_moe_glu(m) == 0


needs_gpu = pytest.mark.skipif(
    bool(os.environ.get("KQUANT_FORCE_CPU")) or not mx.metal.is_available(),
    reason="the fused MoE kernels need a real GPU")


def _q8(leaf, shape, rng):
    import mlx_kquant as kq

    w = mx.array((rng.standard_normal(shape) * 0.05).astype(np.float32))
    leaf.weight, leaf.scales = kq.quantize(w, "q8_0")
    return leaf


def _q8_glu(glu, e, d, inter, rng):
    from mlx_kquant.nn import KQuantSwitchLinear

    for name, (o, i) in (("gate_proj", (inter, d)), ("up_proj", (inter, d)),
                         ("down_proj", (d, inter))):
        setattr(glu, name, _q8(
            KQuantSwitchLinear(e, o, i, False, "q8_0"), (e, o, i), rng))


def _scaled_matches_stock(shell, path, call, monkeypatch):
    """Scale ``path`` by 0.5, install the fused MoE paths, and compare the
    block's output with the stock forward."""
    from gmlx.load import modules

    weights = {f"{path}.out_scale": mx.array([0.5])}
    assert install_out_scales(shell, weights) == 1
    shell.load_weights(list(weights.items()), strict=False)
    assert dict(shell.named_modules())[path]["out_scale"].item() == 0.5
    modules.install_fused_moe_glu(shell)
    mx.eval(shell.parameters())
    y = call()
    mx.eval(y)
    with monkeypatch.context() as m:
        m.setattr(modules, "_FUSED_MOE_ENABLED", False)
        ref = call()
        mx.eval(ref)
    a, b = np.array(y.astype(mx.float32)), np.array(ref.astype(mx.float32))
    assert np.linalg.norm(a - b) / np.linalg.norm(b) < 2e-2


@needs_gpu
@pytest.mark.parametrize("path", ["mlp.switch_mlp.down_proj",
                                  "mlp.switch_mlp.gate_proj"])
def test_fused_moe_block_applies_a_routed_stack_scale(path, monkeypatch):
    from types import SimpleNamespace

    from mlx_kquant.nn import KQuantLinear

    from gmlx._vendor.mlx_lm.models.qwen3_next import Qwen3NextSparseMoeBlock

    d, inter, e = 256, 256, 4
    mx.random.seed(0)
    rng = np.random.default_rng(0)
    blk = Qwen3NextSparseMoeBlock(SimpleNamespace(
        hidden_size=d, moe_intermediate_size=inter,
        shared_expert_intermediate_size=inter, norm_topk_prob=True,
        num_experts=e, num_experts_per_tok=2))
    _q8_glu(blk.switch_mlp, e, d, inter, rng)
    for name, (o, i) in (("gate_proj", (inter, d)), ("up_proj", (inter, d)),
                         ("down_proj", (d, inter))):
        setattr(blk.shared_expert, name,
                _q8(KQuantLinear(i, o, False, "q8_0"), (o, i), rng))
    blk.eval()

    class Shell(nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = blk

    shell = Shell()
    x = mx.array(rng.standard_normal((1, 1, d)).astype(np.float32)).astype(mx.bfloat16)
    _scaled_matches_stock(shell, path, lambda: shell.mlp(x), monkeypatch)


@needs_gpu
@pytest.mark.parametrize("path", ["experts.switch_glu.down_proj",
                                  "experts.switch_glu.gate_proj",
                                  "experts.switch_glu.up_proj"])
def test_fused_gemma_experts_apply_a_stack_scale(path, monkeypatch):
    from types import SimpleNamespace

    from gmlx.models.gemma4.language import Experts

    d, inter, e, k = 512, 256, 8, 2
    rng = np.random.default_rng(3)
    ex = Experts(SimpleNamespace(hidden_size=d, moe_intermediate_size=inter,
                                 num_experts=e))
    _q8_glu(ex.switch_glu, e, d, inter, rng)
    ex.eval()

    class Shell(nn.Module):
        def __init__(self):
            super().__init__()
            self.experts = ex

    shell = Shell()
    x = mx.array(rng.standard_normal((1, 1, d)).astype(np.float32)).astype(mx.bfloat16)
    inds = mx.array(rng.integers(0, e, (1, 1, k)).astype(np.uint32))
    w = mx.array(rng.random((1, 1, k)).astype(np.float32)).astype(mx.bfloat16)
    _scaled_matches_stock(shell, path, lambda: shell.experts(x, inds, w), monkeypatch)
