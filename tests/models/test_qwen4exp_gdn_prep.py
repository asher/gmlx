"""The gated-delta prefill kernels (prepare and gate) against the op
chain they replace: the same bits, and a layer takes them only on calls
they serve."""
from __future__ import annotations

import os
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from gmlx.models.qwen4_exp import gdn_prep

if os.environ.get("KQUANT_FORCE_CPU") or not mx.metal.is_available():
    pytest.skip("the prefill kernels need a real GPU and a GPU default device",
                allow_module_level=True)

HK, HV, D, K, EPS = 2, 6, 128, 4, 1e-6
C = (2 * HK + HV) * D


def _rand(rng, shape, dtype, scale=1.0):
    return mx.array((rng.standard_normal(shape) * scale).astype(np.float32)).astype(dtype)


def _same(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and bool(mx.array_equal(a, b).item())


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
@pytest.mark.parametrize("t", [3, 16, 45])
@pytest.mark.parametrize("tile", [1, HV // HK])
def test_prepare_has_the_bits_of_the_op_chain(dtype, t, tile):
    rng = np.random.default_rng(t)
    conv = nn.Conv1d(in_channels=C, out_channels=C, kernel_size=K, groups=C,
                     padding=0, bias=False)
    conv.weight = _rand(rng, (C, K, 1), dtype, 0.5)
    state = _rand(rng, (1, K - 1, C), dtype)
    rows = _rand(rng, (1, t, C), dtype)
    out = nn.silu(conv(mx.concatenate([state, rows], axis=1)))
    q, k, v = [x.reshape(1, t, h, D) for x, h in
               zip(mx.split(out, [HK * D, 2 * HK * D], axis=-1), (HK, HK, HV))]
    s = D ** -0.5
    q = mx.tile((s ** 2) * mx.fast.rms_norm(q, None, EPS), [1, 1, tile, 1])
    k = mx.tile(s * mx.fast.rms_norm(k, None, EPS), [1, 1, tile, 1])
    assert gdn_prep.prepare_supported(state, rows, conv.weight, D)
    got = gdn_prep.prepare(state, rows, conv.weight, HK, EPS, s ** 2, s, tile=tile)
    assert _same(got[0], q) and _same(got[1], k) and _same(got[2], v)


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
@pytest.mark.parametrize("t", [3, 45])
def test_gate_has_the_bits_of_the_op_chain(dtype, t):
    rng = np.random.default_rng(t)
    y = _rand(rng, (1, t, HV, D), dtype, 3.0)
    z = _rand(rng, (1, t, HV, D), dtype, 4.0)
    w = mx.array((1 + 0.2 * rng.standard_normal(D)).astype(np.float32)).astype(dtype)
    want = mx.fast.rms_norm(y, w, EPS) * mx.sigmoid(z)
    assert gdn_prep.gate_supported(y, z, w)
    assert _same(gdn_prep.gate(y, z, w, EPS), want)


def _layer():
    import gmlx.models.qwen4_exp.model as qm

    args = SimpleNamespace(
        hidden_size=256, linear_num_value_heads=HV, linear_num_key_heads=HK,
        linear_key_head_dim=D, linear_value_head_dim=D,
        linear_conv_kernel_dim=K, rms_norm_eps=EPS)
    mx.random.seed(0)
    layer = qm.GatedDeltaNet(args)
    layer.norm.weight = 1 + 0.2 * mx.random.normal((D,))
    layer.set_dtype(mx.bfloat16)
    layer.eval()
    return layer


def _spy(monkeypatch):
    calls = {"prepare": 0, "gate": 0}

    def wrap(name):
        real = getattr(gdn_prep, name)

        def spy(*a, **kw):
            calls[name] += 1
            return real(*a, **kw)
        monkeypatch.setattr(gdn_prep, name, spy)
    wrap("prepare")
    wrap("gate")
    return calls


@pytest.mark.parametrize("scan", ["0", "1"])
def test_layer_prefill_in_the_kernels_has_the_bits_of_the_op_chain(monkeypatch, scan):
    """Two chunks through one cache, with the stock scan and the staged
    one: outputs, conv state and scan state bit for bit."""
    layer = _layer()
    calls = _spy(monkeypatch)
    monkeypatch.setenv("GMLX_GDN_STAGED_SCAN", scan)
    x = (mx.random.normal((1, 40, 256)) * 0.5).astype(mx.bfloat16)

    def run():
        cache = [None, None]
        outs = [layer(x[:, :23], cache=cache), layer(x[:, 23:], cache=cache)]
        mx.eval(outs, cache)
        return outs, cache

    monkeypatch.setenv("GMLX_Q4_GDN_PREFILL_FUSE", "0")
    want, want_c = run()
    assert calls == {"prepare": 0, "gate": 0}
    monkeypatch.delenv("GMLX_Q4_GDN_PREFILL_FUSE")
    got, got_c = run()
    assert calls == {"prepare": 2, "gate": 2}
    assert all(_same(g, w) for g, w in zip(got, want))
    assert _same(got_c[0], want_c[0]) and _same(got_c[1], want_c[1])


def test_calls_the_prefill_kernels_do_not_take(monkeypatch):
    """A masked call, one shorter than the conv state, a float32 one and a
    verify forward run the op chain."""
    layer = _layer()
    calls = _spy(monkeypatch)
    x = (mx.random.normal((1, 8, 256)) * 0.5).astype(mx.bfloat16)
    mx.eval(layer(x, mask=mx.ones((1, 8), dtype=mx.bool_), cache=[None, None]))
    mx.eval(layer(x[:, :2], cache=[None, None]))
    mx.eval(layer(x, cache=[None, None], gdn_sink=[]))
    layer.set_dtype(mx.float32)
    mx.eval(layer(x.astype(mx.float32), cache=[None, None]))
    assert calls == {"prepare": 0, "gate": 0}
