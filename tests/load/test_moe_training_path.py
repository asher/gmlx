"""A K-quant MoE block in training mode: the fused kernels (which have no
gradient) are never called, the stock gather path runs on kq's gather
matmul, and the gradient with respect to the block input matches a float
reference built from the dequantized expert stacks. CPU."""
from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from mlx_lm.models.switch_layers import SwitchGLU

import gmlx.load.modules as modules


class _Holder(nn.Module):
    def __init__(self, glu):
        super().__init__()
        self.switch_mlp = glu


@pytest.fixture(autouse=True)
def _cpu():
    dev = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(dev)


def _quantized_glu(d=64, inter=96, experts=4, seed=0):
    import mlx_kquant as kq
    from mlx_kquant.nn import KQuantSwitchLinear

    r = np.random.default_rng(seed)
    glu = SwitchGLU(d, inter, experts)
    ref = SwitchGLU(d, inter, experts)
    for name, (o, i) in (("gate_proj", (inter, d)), ("up_proj", (inter, d)),
                         ("down_proj", (d, inter))):
        w = mx.array((r.standard_normal((experts, o, i)) * 0.2).astype(np.float32))
        wire, scales = kq.quantize(w, "q8_0")
        leaf = KQuantSwitchLinear(experts, o, i, False, "q8_0")
        leaf.weight = wire
        leaf.scales = scales
        setattr(glu, name, leaf)
        getattr(ref, name).weight = kq.dequantize(wire, scales, "q8_0", mx.float32)
    return glu, ref


def test_training_mode_skips_fused_kernels_and_has_input_gradient(monkeypatch):
    import mlx_kquant as kq

    glu, ref = _quantized_glu()
    model = _Holder(glu)
    if modules.install_fused_moe_glu(model) != 1:
        pytest.skip("fused MoE glu not installable in this build")
    monkeypatch.setattr(modules, "_kq_fused_device_ok", lambda *m: True)
    for name in ("moe_glu_gather_kq", "gather_qmv_kq", "gather_qmv_mix_kq",
                 "moe_glu_gather_shexp_kq"):
        if hasattr(kq, name):
            monkeypatch.setattr(kq, name, lambda *a, **k: (_ for _ in ()).throw(
                AssertionError("fused kernel called in training mode")))

    model.train()
    r = np.random.default_rng(1)
    x = mx.array(r.standard_normal((2, 3, 64)).astype(np.float32))
    idx = mx.array(r.integers(0, 4, (2, 3, 2)), dtype=mx.uint32)
    w = mx.array(r.standard_normal((2, 3, 2, 64)).astype(np.float32))

    def loss(x):
        return (model.switch_mlp(x, idx) * w).sum()

    def loss_ref(x):
        return (ref(x, idx) * w).sum()

    y = model.switch_mlp(x, idx)
    y_ref = ref(x, idx)
    gx = mx.grad(loss)(x)
    gx_ref = mx.grad(loss_ref)(x)
    mx.eval(y, gx, y_ref, gx_ref)

    def rel(a, b):
        a = np.array(a.astype(mx.float32))
        b = np.array(b.astype(mx.float32))
        return float(np.linalg.norm(a - b) / np.linalg.norm(b))

    # the gather path computes in bfloat16, so the block output and its
    # input gradient sit within bf16 rounding of the float reference
    assert y.dtype == mx.bfloat16
    assert np.isfinite(np.array(gx)).all()
    assert rel(y, y_ref) < 0.03
    assert rel(gx, gx_ref) < 0.03
