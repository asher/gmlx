#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Asher Feldman
"""Per-tensor weight scales: ``install_out_scales`` multiplies a module's
output by its ``out_scale``, keeps float16 activations finite under a small
scale, applies an lm_head scale only to an NVFP4 head, refuses per-expert
scales, and the wire fusions skip a scaled projection."""

from __future__ import annotations

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
