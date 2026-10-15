#!/usr/bin/env python3
"""The gpt-oss fused GLU (swiglu_clamp over biased mxfp4 wire stacks) runs
with the biases the weights load, not the zeros the modules are built with:
the fusion installs before the weights load. The CPU row fakes the kq
kernels and reads the biases each call hands them. The GPU row compares the
fused forward with the stock one."""
from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from helpers import _real_apple_gpu

from gmlx._vendor.mlx_lm.models.gpt_oss import SwiGLU
from gmlx._vendor.mlx_lm.models.switch_layers import SwitchGLU

import gmlx.load.modules as modules

D, INTER, E, K = 256, 256, 4, 2
PROJS = ("gate_proj", "up_proj", "down_proj")


class _Holder(nn.Module):
    def __init__(self, glu):
        super().__init__()
        self.experts = glu


def _fused():
    from mlx_kquant.nn import KQuantSwitchLinear

    glu = SwitchGLU(D, INTER, E, activation=SwiGLU(), bias=True)
    for name, (o, i) in (("gate_proj", (INTER, D)), ("up_proj", (INTER, D)),
                         ("down_proj", (D, INTER))):
        setattr(glu, name, KQuantSwitchLinear(E, o, i, True, "mxfp4"))
    glu.eval()
    model = _Holder(glu)
    if modules.install_fused_moe_glu(model) != 1:
        pytest.skip("this mlx-kquant build has no fused swiglu_clamp GLU")
    assert model.experts._kq_glu_act == "swiglu_clamp"
    return model.experts


def _load_biases(glu, seed):
    rng = np.random.default_rng(seed)
    w = {f"{n}.bias": mx.array(rng.standard_normal(
        getattr(glu, n).bias.shape).astype(np.float32)).astype(mx.bfloat16)
         for n in PROJS}
    glu.load_weights(list(w.items()), strict=False)
    return w


@pytest.fixture
def handed(monkeypatch):
    import mlx_kquant as kq

    got = {}

    def glu(x, gw, uw, ktype, idx, **kw):
        got["gate_proj"], got["up_proj"] = kw["gate_bias"], kw["up_bias"]
        return mx.zeros((idx.shape[0], idx.shape[1], INTER), mx.float32)

    def down(h, dw, ktype, idx, **kw):
        got["down_proj"] = kw["bias"]
        return mx.zeros((h.shape[0], h.shape[1], D), mx.float32)

    glu.__doc__ = kq.moe_glu_gather_kq.__doc__    # the install's feature sniff
    monkeypatch.setattr(kq, "moe_glu_gather_kq", glu)
    monkeypatch.setattr(kq, "gather_qmv_kq", down)
    monkeypatch.setattr(modules, "_kq_fused_device_ok", lambda *m: True)
    return got


def test_each_fused_call_takes_the_biases_of_the_last_load(handed):
    glu = _fused()
    x = mx.ones((1, 1, D), mx.bfloat16)
    idx = mx.array([[[0, 3]]], mx.uint32)
    for seed in (1, 2):          # a load, then a load of other biases
        want = _load_biases(glu, seed)
        glu(x, idx)
        for name in PROJS:
            assert handed[name].dtype == mx.float32
            np.testing.assert_array_equal(
                np.array(handed[name]),
                np.array(want[f"{name}.bias"].astype(mx.float32)))


@pytest.mark.skipif(not _real_apple_gpu(),
                    reason="the fused kq kernels are Metal-only")
def test_the_fused_forward_matches_the_stock_forward():
    glu = _fused()
    rng = np.random.default_rng(0)
    weights = []
    for name in PROJS:
        # mxfp4 blocks of 17 bytes: an E8M0 scale of 2^-6, then 32 fp4 values.
        w = rng.integers(0, 256, getattr(glu, name).weight.shape, np.uint8)
        w[..., ::17] = 121
        weights.append((f"{name}.weight", mx.array(w)))
    glu.load_weights(weights, strict=False)
    _load_biases(glu, 1)
    x = mx.array(0.5 * rng.standard_normal((1, 3, D)), mx.bfloat16)
    idx = mx.array(rng.integers(0, E, (1, 3, K)), mx.uint32)

    fused = np.array(glu(x, idx).astype(mx.float32))
    stock = np.array(super(type(glu), glu).__call__(x, idx).astype(mx.float32))
    assert fused.shape == stock.shape == (1, 3, K, D)
    scale = float(np.abs(stock).max())
    np.testing.assert_allclose(fused, stock, rtol=0, atol=0.03 * scale)
