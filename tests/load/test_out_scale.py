#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Asher Feldman
"""Per-tensor and per-expert weight scales: ``install_out_scales`` gives
an nvfp4 kq module its scale for the kernels and wraps every other module,
the result is ``s * (W x) + b`` either way, an lm_head scale applies only
to an NVFP4 head, the fusions that read wire bytes pass an nvfp4 scale to
their ops, and they skip a wrapped module."""

from __future__ import annotations

import os

import numpy as np
import pytest

import mlx.core as mx
import mlx.nn as nn

from gmlx.load.out_scale import has_out_scale, install_out_scales, weight_scale


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
    assert "post_scale" not in model.trainable_parameters()["proj"]


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


def test_several_values_on_a_dense_module_fail_loud():
    model = _Tiny()
    with pytest.raises(NotImplementedError, match="one per expert"):
        install_out_scales(model, {"proj.out_scale": mx.ones((4,))})


def test_wrapper_scales_before_the_bias():
    model = _Tiny()
    model.proj = nn.Linear(16, 8, bias=True)
    model.proj.bias = mx.full((8,), 3.0)
    weights = {"proj.out_scale": mx.array([0.25])}
    install_out_scales(model, weights)
    model.load_weights(list(weights.items()), strict=False)
    x = mx.random.normal((2, 16))
    ref = 0.25 * np.array(x @ model.proj.weight.T) + 3.0
    np.testing.assert_allclose(np.array(model.proj(x)), ref, rtol=1e-5,
                               atol=1e-5)


def test_wrapper_reads_a_per_expert_scale_at_the_expert_ids():
    from gmlx._vendor.mlx_lm.models.switch_layers import SwitchLinear

    class Moe(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = SwitchLinear(16, 8, 4, bias=True)

    model = Moe()
    scale = mx.array([0.5, 0.25, 2.0, 4.0])
    weights = {"proj.out_scale": scale}
    install_out_scales(model, weights)
    model.load_weights(list(weights.items()), strict=False)
    x = mx.random.normal((3, 1, 1, 16))
    idx = mx.array([[0, 2], [1, 3], [3, 0]])
    w, b = np.array(model.proj.weight), np.array(model.proj.bias)
    ref = np.empty((3, 2, 1, 8), np.float32)
    for t in range(3):
        for j in range(2):
            e = int(idx[t, j])
            ref[t, j, 0] = (float(scale[e]) * (w[e] @ np.array(x[t, 0, 0]))
                            + b[e])
    np.testing.assert_allclose(np.array(model.proj(x, idx)), ref, rtol=1e-5,
                               atol=1e-5)


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
    assert weight_scale(dict(shell.named_modules())[path]).item() == 0.5
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


# nvfp4: the scale rides the kq call and the fused ops.

def _nvfp4_wire(rng, shape):
    """Random nvfp4 wire bytes for weights of ``shape`` (last dim K): 36
    bytes per 64 weights, sub-scale bytes in a moderate range."""
    *lead, k = shape
    raw = rng.integers(0, 256, size=(*lead, k // 64, 36)).astype(np.uint8)
    raw[..., :4] = rng.integers(0x30, 0x41, size=raw[..., :4].shape)
    return mx.array(raw.reshape(*lead, k // 64 * 36))


def _deq(leaf):
    """The float64 weights of a kq leaf, (..., N, K)."""
    import mlx_kquant as kq

    w = leaf["weight"]
    flat = kq.dequantize(w.reshape(-1, w.shape[-1]), leaf["scales"],
                         leaf.kquant_type)
    return np.array(flat.astype(mx.float32), np.float64).reshape(
        *w.shape[:-1], -1)


def _needs_kq_scales():
    from gmlx.load.out_scale import kq_applies_out_scale

    if not kq_applies_out_scale():
        pytest.skip("mlx_kquant without in-kernel weight scales")


def _install_and_load(shell, weights):
    n = install_out_scales(shell, weights)
    shell.load_weights(list(weights.items()), strict=False)
    return n


def _rel(a, b):
    a = np.array(a.astype(mx.float32), np.float64)
    return np.linalg.norm(a - b) / np.linalg.norm(b)


@pytest.mark.parametrize("m_rows", [1, 4, 32])
@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
def test_nvfp4_linear_takes_its_scale_in_the_kernel(m_rows, dtype):
    from mlx_kquant.nn import KQuantLinear

    from gmlx.load.out_scale import wrapped_scale

    _needs_kq_scales()
    rng = np.random.default_rng(1)

    class Shell(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = KQuantLinear(256, 64, True, "nvfp4")

    shell = Shell()
    shell.proj.weight = _nvfp4_wire(rng, (64, 256))
    shell.proj.bias = mx.array(rng.standard_normal(64).astype(np.float32)
                               ).astype(dtype)
    weights = {"proj.out_scale": mx.array([2e-4])}
    assert _install_and_load(shell, weights) == 1
    assert type(shell.proj) is KQuantLinear and not wrapped_scale(shell.proj)
    # Large enough that the unscaled product passes the float16 range.
    x = rng.standard_normal((m_rows, 256)).astype(np.float32) * 3e3
    y = shell.proj(mx.array(x).astype(dtype))
    ref = (2e-4 * (np.array(mx.array(x).astype(dtype).astype(mx.float32),
                            np.float64) @ _deq(shell.proj).T)
           + np.array(shell.proj.bias.astype(mx.float32), np.float64))
    assert y.dtype == dtype
    assert np.isfinite(np.array(y.astype(mx.float32))).all()
    assert _rel(y, ref) < 1e-2


def test_scale_lever_off_keeps_the_wrapper(monkeypatch):
    from mlx_kquant.nn import KQuantLinear

    from gmlx.load.modules import _kq_dense
    from gmlx.load.out_scale import wrapped_scale

    _needs_kq_scales()
    monkeypatch.setenv("GMLX_KQ_OUT_SCALE", "0")
    rng = np.random.default_rng(2)

    class Shell(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = KQuantLinear(256, 64, False, "nvfp4")

    shell = Shell()
    shell.proj.weight = _nvfp4_wire(rng, (64, 256))
    _install_and_load(shell, {"proj.out_scale": mx.array([0.5])})
    assert wrapped_scale(shell.proj)
    assert not _kq_dense(shell.proj, {"nvfp4"})
    x = mx.array(rng.standard_normal((2, 256)).astype(np.float32))
    ref = 0.5 * np.array(x, np.float64) @ _deq(shell.proj).T
    assert _rel(shell.proj(x), ref) < 1e-2


@pytest.mark.parametrize("n_tok", [3, 300])
def test_nvfp4_stack_takes_one_scale_per_expert(n_tok):
    """Decode gathers, and the sorted segment GEMM at prefill widths."""
    from mlx_kquant.nn import KQuantSwitchLinear

    _needs_kq_scales()
    rng = np.random.default_rng(3)
    e, k = 4, 2

    class Shell(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = KQuantSwitchLinear(e, 64, 256, False, "nvfp4")

    shell = Shell()
    shell.proj.weight = _nvfp4_wire(rng, (e, 64, 256))
    scale = np.array([0.5, 0.125, 2.0, 0.03125], np.float32)
    _install_and_load(shell, {"proj.out_scale": mx.array(scale)})
    assert type(shell.proj) is KQuantSwitchLinear
    assert shell.proj["out_scale"].shape == (e, 1)
    x = rng.standard_normal((n_tok, 1, 1, 256)).astype(np.float32)
    idx = rng.integers(0, e, (n_tok, k)).astype(np.uint32)
    deq = _deq(shell.proj)
    ref = np.einsum("tk,tkn->tkn", scale[idx].astype(np.float64),
                    np.einsum("tc,tknc->tkn", x[:, 0, 0], deq[idx]))
    xs = mx.array(x).astype(mx.bfloat16)
    if n_tok * k >= 512:
        flat = mx.array(idx.reshape(-1))
        order = mx.argsort(flat)
        y = shell.proj(xs, flat[order], sorted_indices=True,
                       x_rows=order // k)
        inv = np.argsort(np.array(order))
        y = np.array(y.astype(mx.float32))[inv].reshape(n_tok, k, -1)
        assert np.linalg.norm(y - ref) / np.linalg.norm(ref) < 1e-2
    else:
        y = shell.proj(xs, mx.array(idx))
        assert _rel(y.squeeze(-2), ref) < 1e-2


def _nvfp4_glu(glu, e, d, inter, rng):
    from mlx_kquant.nn import KQuantSwitchLinear

    for name, (o, i) in (("gate_proj", (inter, d)), ("up_proj", (inter, d)),
                         ("down_proj", (d, inter))):
        leaf = KQuantSwitchLinear(e, o, i, False, "nvfp4")
        leaf.weight = _nvfp4_wire(rng, (e, o, i))
        setattr(glu, name, leaf)


def _expert_scales(prefix, e, rng):
    """Distinct per-expert scales on every stack, so a fused path that
    drops one, or swaps gate and up, misses the reference."""
    return {f"{prefix}.{p}.out_scale": mx.array(
        rng.uniform(lo, 2 * lo, e).astype(np.float32))
        for p, lo in (("gate_proj", 0.02), ("up_proj", 0.1),
                      ("down_proj", 0.05))}


def _fused_matches_stock(call, monkeypatch, flags=("_FUSED_MOE_ENABLED",)):
    from gmlx.load import modules

    y = call()
    mx.eval(y)
    with monkeypatch.context() as m:
        for f in flags:
            m.setattr(modules, f, False)
        ref = call()
        mx.eval(ref)
    ref = np.array(ref.astype(mx.float32), np.float64)
    assert _rel(y, ref) < 2e-2


@needs_gpu
@pytest.mark.parametrize("n_tok", [2, 48])
def test_fused_switch_glu_applies_nvfp4_expert_scales(n_tok, monkeypatch):
    """Decode (fused gathers) and prefill (the gate+up concat)."""
    from gmlx._vendor.mlx_lm.models.switch_layers import SwitchGLU
    from gmlx.load import modules

    _needs_kq_scales()
    # The simulated test host is too small for the concat's headroom.
    monkeypatch.setattr(modules, "_GATEUP_CONCAT_HEADROOM_GB", 0)
    d, inter, e, k = 256, 256, 8, 2
    rng = np.random.default_rng(4)
    shell = _Moe()
    _nvfp4_glu(shell.experts, e, d, inter, rng)
    shell.eval()
    _install_and_load(shell, _expert_scales("experts", e, rng))
    assert modules.install_fused_moe_glu(shell) == 1
    assert type(shell.experts) is not SwitchGLU
    x = mx.array(rng.standard_normal((n_tok, d)).astype(np.float32) * 0.1
                 ).astype(mx.bfloat16)
    idx = mx.array(rng.integers(0, e, (n_tok, k)).astype(np.uint32))
    _fused_matches_stock(lambda: shell.experts(x, idx), monkeypatch,
                         ("_FUSED_MOE_ENABLED", "_GATEUP_CONCAT_ENABLED"))
    if n_tok * k >= 64:
        assert getattr(shell.experts, "_kq_gate_up", None) is not None


@needs_gpu
def test_fused_moe_block_applies_nvfp4_scales(monkeypatch):
    """qwen3-next block: expert stacks and the shared expert."""
    from types import SimpleNamespace

    from mlx_kquant.nn import KQuantLinear

    from gmlx._vendor.mlx_lm.models.qwen3_next import Qwen3NextSparseMoeBlock
    from gmlx.load import modules

    _needs_kq_scales()
    d, inter, e = 256, 256, 4
    rng = np.random.default_rng(5)
    blk = Qwen3NextSparseMoeBlock(SimpleNamespace(
        hidden_size=d, moe_intermediate_size=inter,
        shared_expert_intermediate_size=inter, norm_topk_prob=True,
        num_experts=e, num_experts_per_tok=2))
    _nvfp4_glu(blk.switch_mlp, e, d, inter, rng)
    for name, (o, i) in (("gate_proj", (inter, d)), ("up_proj", (inter, d)),
                         ("down_proj", (d, inter))):
        leaf = KQuantLinear(i, o, False, "nvfp4")
        leaf.weight = _nvfp4_wire(rng, (o, i))
        setattr(blk.shared_expert, name, leaf)
    blk.eval()

    class Shell(nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = blk

    shell = Shell()
    weights = _expert_scales("mlp.switch_mlp", e, rng)
    weights.update({f"mlp.shared_expert.{p}.out_scale": mx.array([s])
                    for p, s in (("gate_proj", 0.03), ("up_proj", 0.2),
                                 ("down_proj", 0.07))})
    _install_and_load(shell, weights)
    modules.install_fused_moe_glu(shell)
    assert type(shell.mlp).__name__ == "_FusedKQuantMoeBlock"
    x = mx.array(rng.standard_normal((1, 2, d)).astype(np.float32) * 0.1
                 ).astype(mx.bfloat16)
    _fused_matches_stock(lambda: shell.mlp(x), monkeypatch)


@needs_gpu
def test_fused_gemma_experts_apply_nvfp4_scales(monkeypatch):
    from types import SimpleNamespace

    from gmlx.load import modules
    from gmlx.models.gemma4.language import Experts

    _needs_kq_scales()
    monkeypatch.setenv("GMLX_GEMMA_FUSED_GLU", "1")
    d, inter, e, k = 512, 256, 8, 2
    rng = np.random.default_rng(6)
    ex = Experts(SimpleNamespace(hidden_size=d, moe_intermediate_size=inter,
                                 num_experts=e))
    _nvfp4_glu(ex.switch_glu, e, d, inter, rng)
    ex.eval()

    class Shell(nn.Module):
        def __init__(self):
            super().__init__()
            self.experts = ex

    shell = Shell()
    _install_and_load(shell, _expert_scales("experts.switch_glu", e, rng))
    modules.install_fused_moe_glu(shell)
    assert type(shell.experts).__name__ == "_FusedKQuantGemmaExperts"
    x = mx.array(rng.standard_normal((1, 1, d)).astype(np.float32) * 0.1
                 ).astype(mx.bfloat16)
    inds = mx.array(rng.integers(0, e, (1, 1, k)).astype(np.uint32))
    w = mx.array(rng.random((1, 1, k)).astype(np.float32)).astype(mx.bfloat16)
    _fused_matches_stock(lambda: shell.experts(x, inds, w), monkeypatch)


def _nvfp4_linears(rng, rows, scales, k=256):
    from mlx_kquant.nn import KQuantLinear

    out = []
    for n, s in zip(rows, scales):
        lin = KQuantLinear(k, n, False, "nvfp4")
        lin.weight = _nvfp4_wire(rng, (n, k))
        if s is not None:
            lin.out_scale = mx.array([s], dtype=mx.float32)
            lin.freeze(recurse=False, keys=["out_scale"])
        out.append(lin)
    return out


@pytest.mark.parametrize("site", ["gdn_ba", "gdn_multi", "dsv4_pair"])
def test_row_concat_fusions_keep_each_member_scale(site):
    """The fusions that join scaled projections into one wire, or one
    dispatch, give each member's rows that member's scale."""
    _needs_kq_scales()
    rng = np.random.default_rng(7)
    x = mx.array(rng.standard_normal((3, 256)).astype(np.float32)
                 ).astype(mx.bfloat16)
    if site == "gdn_ba":
        from gmlx.upstream.gdn_patches import _gdn_try_cat_ba_kq

        b, a = _nvfp4_linears(rng, (16, 16), (0.03, 0.3))
        want = mx.concatenate([b(x), a(x)], axis=-1)
        gdn = nn.Module()
        gdn.num_v_heads, gdn.hidden_size = 16, 256
        assert _gdn_try_cat_ba_kq(gdn, [b, a])
        got = gdn._gdn_ba_kq(x)
    elif site == "gdn_multi":
        from gmlx.upstream.gdn_patches import kquant_multi

        lins = _nvfp4_linears(rng, (64, 32, 32), (0.03, None, 0.3))
        want = mx.concatenate([lin(x) for lin in lins], axis=-1)
        out = kquant_multi(lins, x)
        assert out is not None and all(o is not None for o in out)
        got = mx.concatenate(list(out), axis=-1)
    else:
        from gmlx.models.deepseek_v4.model import _fuse_kquant_pair

        a, b = _nvfp4_linears(rng, (64, 32), (None, 0.3))
        want = mx.concatenate([a(x), b(x)], axis=-1)
        fused = _fuse_kquant_pair(a, b, 256)
        assert fused is not None
        got = fused(x)
    assert _rel(got, np.array(want.astype(mx.float32), np.float64)) < 1e-2


@pytest.mark.parametrize("batch", [2, 12])
def test_occupancy_fused_mlp_keeps_each_projection_scale(batch, monkeypatch):
    """gate and up joined in one wire, and down split in two K halves at
    the wider batch."""
    from types import SimpleNamespace

    from gmlx._vendor.mlx_lm.models.llama import MLP
    import gmlx.upstream.occupancy_fuse as of

    _needs_kq_scales()
    rng = np.random.default_rng(8)

    class Shell(nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = MLP(SimpleNamespace(hidden_size=256,
                                           intermediate_size=512))

    shell = Shell()
    g, u = _nvfp4_linears(rng, (512, 512), (None, None))
    (dn,) = _nvfp4_linears(rng, (256,), (None,), k=512)
    shell.mlp.gate_proj, shell.mlp.up_proj, shell.mlp.down_proj = g, u, dn
    _install_and_load(shell, {"mlp.gate_proj.out_scale": mx.array([0.02]),
                              "mlp.up_proj.out_scale": mx.array([0.1]),
                              "mlp.down_proj.out_scale": mx.array([0.05])})
    assert of.install_occupancy_fuse(shell) == 1
    x = mx.array(rng.standard_normal((batch, 1, 256)).astype(np.float32) * 0.1
                 ).astype(mx.bfloat16)
    got = shell.mlp(x)
    assert shell.mlp._kq_os
    if batch >= 12:
        assert shell.mlp._kq_wdn is not None
    monkeypatch.setenv("GMLX_OCCUPANCY_FUSE", "0")
    want = shell.mlp(x)
    assert _rel(got, np.array(want.astype(mx.float32), np.float64)) < 2e-2
