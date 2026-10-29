#!/usr/bin/env python3
"""The gpt-oss fused GLU (swiglu_clamp over biased mxfp4 wire stacks) runs
with the biases the weights load, not the zeros the modules are built with:
the fusion installs before the weights load. The CPU row fakes the kq
kernels and reads the biases each call hands them. The GPU rows compare the
fused forward with the stock one, without and with a down-projection LoRA."""
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


def _load_wire(glu, rng):
    weights = []
    for name in PROJS:
        # mxfp4 blocks of 17 bytes: an E8M0 scale of 2^-6, then 32 fp4 values.
        w = rng.integers(0, 256, getattr(glu, name).weight.shape, np.uint8)
        w[..., ::17] = 121
        weights.append((f"{name}.weight", mx.array(w)))
    glu.load_weights(weights, strict=False)
    _load_biases(glu, 1)


@pytest.mark.skipif(not _real_apple_gpu(),
                    reason="the fused kq kernels are Metal-only")
def test_the_fused_forward_matches_the_stock_forward():
    glu = _fused()
    rng = np.random.default_rng(0)
    _load_wire(glu, rng)
    x = mx.array(0.5 * rng.standard_normal((1, 3, D)), mx.bfloat16)
    idx = mx.array(rng.integers(0, E, (1, 3, K)), mx.uint32)

    fused = np.array(glu(x, idx).astype(mx.float32))
    stock = np.array(super(type(glu), glu).__call__(x, idx).astype(mx.float32))
    assert fused.shape == stock.shape == (1, 3, K, D)
    scale = float(np.abs(stock).max())
    np.testing.assert_allclose(fused, stock, rtol=0, atol=0.03 * scale)


@pytest.mark.skipif(not _real_apple_gpu(),
                    reason="the fused kq kernels are Metal-only")
def test_a_down_lora_adds_its_delta_on_the_fused_forward():
    """An --adapter LoRA on the routed down projection of a gpt-oss expert
    stack takes the fused exit, which adds the expert bias and the delta."""
    import gmlx.load.adapter as adapter
    from gmlx import lora_rows

    glu = _fused()
    rng = np.random.default_rng(0)
    _load_wire(glu, rng)
    x = mx.array(0.5 * rng.standard_normal((1, 3, D)), mx.bfloat16)
    idx = mx.array(rng.integers(0, E, (1, 3, K)), mx.uint32)
    base = np.array(SwitchGLU.__call__(glu, x, idx).astype(mx.float32))
    xe = mx.expand_dims(x, (-2, -3))
    h = np.array(glu.activation(glu.up_proj(xe, idx), glu.gate_proj(xe, idx))
                 .astype(mx.float32))[0, :, :, 0]                # [3, K, I]
    r = 4
    a = rng.standard_normal((E, r, INTER)).astype(np.float32)
    b = rng.standard_normal((E, D, r)).astype(np.float32)
    delta = np.zeros_like(base)
    for t in range(3):
        for j in range(K):
            e = int(np.array(idx)[0, t, j])
            delta[0, t, j] = (h[t, j] @ a[e].T) @ b[e].T
    # Delta as large as the base output, so a dropped delta fails.
    c = float(np.abs(base).max() / np.abs(delta).max())
    b, delta = b * c, delta * c
    path = "experts.down_proj"
    plan = adapter.LoraAdapter(alpha=float(r), arch="gpt-oss", modules={
        path: adapter.LoraModule(module_path=path,
                                 a=mx.array(a).astype(mx.float16),
                                 b=mx.array(b).astype(mx.float16), rank=r,
                                 scale=1.0, experts=True)})
    lora_rows.configure("static", 1)
    try:
        modules.install_lora_adapter(_Holder(glu), plan)
        assert type(glu).__name__ == "_LoRASwitchGLU"
        assert glu._fused_ok(x, idx)
        y = np.array(glu(x, idx).astype(mx.float32))
    finally:
        lora_rows.configure("static", 1)
    want = base + delta
    assert y.shape == want.shape == (1, 3, K, D)
    np.testing.assert_allclose(y, want, rtol=0,
                               atol=0.03 * float(np.abs(want).max()))
