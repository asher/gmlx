"""Kernel choices for a training run: a K-quant expert call routes more
rows than kq's segment GEMM threshold in any real training batch, and
that GEMM has no backward."""
from __future__ import annotations

import os

import mlx.core as mx
import numpy as np
import pytest

from gmlx.tune.kernels import install_training_switch_gemm


def test_install_sets_zero_and_restore_puts_back_what_was_there(monkeypatch):
    monkeypatch.setenv("KQ_SWITCH_GEMM_MIN_ROWS", "256")
    restore = install_training_switch_gemm()
    assert os.environ["KQ_SWITCH_GEMM_MIN_ROWS"] == "0"
    inner = install_training_switch_gemm()
    inner()
    assert os.environ["KQ_SWITCH_GEMM_MIN_ROWS"] == "0"
    restore()
    assert os.environ["KQ_SWITCH_GEMM_MIN_ROWS"] == "256"
    monkeypatch.delenv("KQ_SWITCH_GEMM_MIN_ROWS")
    install_training_switch_gemm()()
    assert "KQ_SWITCH_GEMM_MIN_ROWS" not in os.environ


@pytest.mark.skipif(os.environ.get("KQUANT_FORCE_CPU") == "1" or not mx.metal.is_available()
                    or mx.default_device() != mx.gpu, reason="kq's segment GEMM runs on the GPU only")
def test_a_kquant_expert_backward_over_many_routed_rows(monkeypatch):
    """1280 sorted routed rows: kq would take its segment GEMM, whose
    backward is not implemented, without the training setting."""
    import mlx_kquant as kq
    from mlx_kquant.nn import KQuantSwitchLinear
    from mlx_lm.models.switch_layers import SwitchGLU

    monkeypatch.delenv("KQ_SWITCH_GEMM_MIN_ROWS", raising=False)
    d, inter, experts = 256, 512, 8
    r = np.random.default_rng(0)
    glu, ref = SwitchGLU(d, inter, experts), SwitchGLU(d, inter, experts)
    for name, (o, i) in (("gate_proj", (inter, d)), ("up_proj", (inter, d)), ("down_proj", (d, inter))):
        w = mx.array((r.standard_normal((experts, o, i)) * 0.05).astype(np.float32))
        wire, scales = kq.quantize(w, "q4_k")
        leaf = KQuantSwitchLinear(experts, o, i, False, "q4_k")
        leaf.weight, leaf.scales = wire, scales
        setattr(glu, name, leaf)
        getattr(ref, name).weight = kq.dequantize(wire, scales, "q4_k", mx.float32)
    glu.train()
    x = mx.array(r.standard_normal((4, 160, d)).astype(np.float32))
    idx = mx.array(r.integers(0, experts, (4, 160, 2)), dtype=mx.uint32)

    def loss(m):
        return lambda x: (m(x, idx) ** 2).sum()

    restore = install_training_switch_gemm()
    try:
        g = mx.grad(loss(glu))(x)
        mx.eval(g)
    finally:
        restore()
    g_ref = mx.grad(loss(ref))(x)
    rel = float(mx.abs(g - g_ref).max() / mx.abs(g_ref).max())
    assert rel < 2e-2, rel
