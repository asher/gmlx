#!/usr/bin/env python3
"""The fused K-quant SwitchGLU forward split into ``_fused_ok`` (the gate),
``_fused_h`` (gate/up gather) and ``_fused_down`` (three exits). Pins the
gate predicate, the flat return ranks the caller discriminates on (mixed
``(t, N)`` vs unmixed ``(t, k, N)``), and the batched shapes on every exit.
CPU-safe: the kq kernels are monkeypatched fakes."""
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
        self.experts = glu


def _fused(codec="q8_0", d=256, inter=256, experts=4):
    from mlx_kquant.nn import KQuantSwitchLinear

    glu = SwitchGLU(d, inter, experts)
    for name, (o, i) in (("gate_proj", (inter, d)), ("up_proj", (inter, d)),
                         ("down_proj", (d, inter))):
        setattr(glu, name, KQuantSwitchLinear(experts, o, i, False, codec))
    glu.eval()
    model = _Holder(glu)
    n = modules.install_fused_moe_glu(model)
    if n != 1:
        pytest.skip("fused MoE glu not installable in this build")
    return model.experts


@pytest.fixture
def fakes(monkeypatch):
    import mlx_kquant as kq
    seen = {}

    def glu(x, gw, uw, ktype, idx, **kw):
        seen["h"] = (x.shape, idx.shape)
        return mx.full((idx.shape[0], idx.shape[1], 256), 2.0, mx.float32)

    def mix_ns(h, dw, ktype, idx, sc):
        seen["mix_ns"] = (h.shape, tuple(sc.shape))
        return mx.full((h.shape[0], 256), 7.0, mx.float32)

    def plain(h, dw, ktype, idx, **kw):
        seen["plain"] = h.shape
        return mx.full((h.shape[0], h.shape[1], 256), 3.0, mx.float32)

    monkeypatch.setattr(kq, "moe_glu_gather_kq", glu)
    monkeypatch.setattr(kq, "gather_qmv_mix_ns_kq", mix_ns)
    monkeypatch.setattr(kq, "gather_qmv_kq", plain)
    monkeypatch.setattr(modules, "_kq_fused_device_ok", lambda *m: True)
    return seen


def test_fused_ok_predicate(monkeypatch):
    glu = _fused()
    monkeypatch.setattr(modules, "_kq_fused_device_ok", lambda *m: True)
    x = mx.zeros((1, 1, 256), mx.bfloat16)
    assert glu._fused_ok(x, mx.zeros((1, 1, 2), mx.uint32))
    assert not glu._fused_ok(x, mx.zeros((1, 32, 2), mx.uint32))     # >= 64
    assert not glu._fused_ok(x.astype(mx.float32), mx.zeros((1, 1, 2), mx.uint32))
    glu.train()
    assert not glu._fused_ok(x, mx.zeros((1, 1, 2), mx.uint32))
    glu.eval()
    monkeypatch.setattr(modules, "_kq_fused_device_ok", lambda *m: False)
    assert not glu._fused_ok(x, mx.zeros((1, 1, 2), mx.uint32))


def test_scores_none_takes_plain_exit_unmixed(fakes):
    glu = _fused()
    x = mx.zeros((2, 1, 256), mx.bfloat16)
    idx = mx.zeros((2, 1, 2), mx.uint32)
    y = glu(x, idx)
    assert y.shape == (2, 1, 2, 256)          # unmixed [..., k, N]
    assert np.allclose(np.array(y.astype(mx.float32)), 3.0)
    assert "plain" in fakes and "mix_ns" not in fakes
    assert fakes["h"] == ((2, 256), (2, 2))


def test_scores_with_mix_ns_returns_mixed(fakes):
    glu = _fused()
    if not glu._kq_mix_scores:
        pytest.skip("build without gather_qmv_mix_ns_kq")
    x = mx.zeros((2, 1, 256), mx.bfloat16)
    idx = mx.zeros((2, 1, 2), mx.uint32)
    sc = mx.full((2, 1, 2), 0.5, mx.bfloat16)
    y = glu(x, idx, sc)
    assert y.shape == (2, 1, 256)             # mixed in the kernel
    assert np.allclose(np.array(y.astype(mx.float32)), 7.0)
    assert fakes["mix_ns"] == ((2, 2, 256), (2, 2))
    assert "plain" not in fakes


def test_flat_return_ranks_of_fused_down(fakes):
    glu = _fused()
    idx = mx.zeros((3, 2), mx.uint32)
    h = mx.zeros((3, 2, 256), mx.float32)
    assert glu._fused_down(h, idx, None).ndim == 3          # (t, k, N)
    if glu._kq_mix_scores:
        sc = mx.full((3, 2), 0.5, mx.bfloat16)
        assert glu._fused_down(h, idx, sc).ndim == 2       # (t, N)


def test_stock_fallback_mixes_python_side(monkeypatch):
    glu = _fused()
    monkeypatch.setattr(modules, "_kq_fused_device_ok", lambda *m: False)
    calls = {}
    parent = type(glu).__mro__[1]

    def stock(self, x, indices):
        calls["stock"] = indices.shape
        return mx.full(indices.shape + (256,), 4.0, mx.bfloat16)
    monkeypatch.setattr(parent, "__call__", stock)
    x = mx.zeros((1, 1, 256), mx.bfloat16)
    idx = mx.zeros((1, 1, 2), mx.uint32)
    sc = mx.full((1, 1, 2), 0.5, mx.bfloat16)
    y = glu(x, idx, sc)
    assert calls["stock"] == (1, 1, 2)
    assert y.shape == (1, 1, 256)
    assert np.allclose(np.array(y.astype(mx.float32)), 4.0)   # 0.5*4 + 0.5*4
