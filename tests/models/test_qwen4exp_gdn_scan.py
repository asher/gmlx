"""The staged prefill scan of the gated-delta recurrence against the stock
scan: the same sums in the same order on untiled q and k, so the same bits."""
from __future__ import annotations

import os

import mlx.core as mx
import numpy as np
import pytest

from gmlx._vendor.mlx_lm.models.gated_delta import gated_delta_kernel
from gmlx.models.qwen4_exp import gdn_scan

if os.environ.get("KQUANT_FORCE_CPU") or not mx.metal.is_available():
    pytest.skip("the scan kernels need a real GPU and a GPU default device",
                allow_module_level=True)

HK, HV, D = 2, 6, 128


def _inputs(t, seed=0, dtype=mx.bfloat16):
    rng = np.random.default_rng(seed)

    def unit(x):
        return x / np.sqrt((x * x).mean(-1, keepdims=True) + 1e-6)

    q = mx.array(unit(rng.standard_normal((1, t, HK, D))) / D).astype(dtype)
    k = mx.array(unit(rng.standard_normal((1, t, HK, D))) / D ** 0.5).astype(dtype)
    v = mx.array(rng.standard_normal((1, t, HV, D))).astype(dtype)
    g = mx.array(np.exp(-np.exp(rng.standard_normal((1, t, HV)) - 2)).astype(np.float32))
    beta = mx.array(1 / (1 + np.exp(-rng.standard_normal((1, t, HV))))).astype(dtype)
    s0 = mx.array((rng.standard_normal((1, HV, D, D)) * 0.1).astype(np.float32))
    return q, k, v, g, beta, s0


def _np(a):
    mx.eval(a)
    return np.array(a.astype(mx.float32))


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16, mx.float32])
@pytest.mark.parametrize("t", [2, 16, 17, 45])
@pytest.mark.parametrize("tiled", [True, False])
def test_staged_scan_has_the_bits_of_the_stock_scan(t, tiled, dtype):
    q, k, v, g, beta, s0 = _inputs(t, seed=t, dtype=dtype)
    r = HV // HK
    if tiled:
        qt, kt = mx.tile(q, [1, 1, r, 1]), mx.tile(k, [1, 1, r, 1])
    else:
        qt, kt = mx.repeat(q, r, -2), mx.repeat(k, r, -2)
    want_y, want_s = gated_delta_kernel(qt, kt, v, g, beta, s0, None)
    assert gdn_scan.supported(q, v, g, s0)
    got_y, got_s = gdn_scan.scan_staged(q, k, v, g, beta, s0, tiled=tiled)
    assert got_y.dtype == want_y.dtype and got_s.dtype == want_s.dtype
    assert mx.array_equal(got_y, want_y).item()
    assert mx.array_equal(got_s, want_s).item()


def test_staged_scan_in_two_chunks_has_the_bits_of_one():
    """The state carried between chunks: a split at a token that is not a
    multiple of the staged block gives the outputs and state of one call."""
    q, k, v, g, beta, s0 = _inputs(45, seed=3)
    y, s = gdn_scan.scan_staged(q, k, v, g, beta, s0)
    cut = 21
    y1, s1 = gdn_scan.scan_staged(q[:, :cut], k[:, :cut], v[:, :cut],
                                  g[:, :cut], beta[:, :cut], s0)
    y2, s2 = gdn_scan.scan_staged(q[:, cut:], k[:, cut:], v[:, cut:],
                                  g[:, cut:], beta[:, cut:], s1)
    assert np.array_equal(_np(mx.concatenate([y1, y2], axis=1)), _np(y))
    assert np.array_equal(_np(s2), _np(s))


def test_shapes_the_staged_scan_does_not_take():
    q, k, v, g, beta, s0 = _inputs(4)
    two = lambda a: mx.concatenate([a, a], axis=0)
    assert not gdn_scan.supported(two(q), two(v), two(g), two(s0))
    assert not gdn_scan.supported(q[..., :64], v, g, s0)
    assert not gdn_scan.supported(q, v, mx.zeros((1, 4, HV, D)), s0)


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16, mx.float32])
@pytest.mark.parametrize("t", [2, 17, 45, 300])
@pytest.mark.parametrize("tiled", [True, False])
def test_scan_gates_have_the_bits_of_compute_g_and_sigmoid(t, tiled, dtype):
    """The gates computed in the kernel from the joined b | a rows, over
    softplus inputs from deep negative to deep positive."""
    from gmlx._vendor.mlx_lm.models.gated_delta import compute_g

    q, k, v, _, _, s0 = _inputs(t, seed=t + 7, dtype=dtype)
    rng = np.random.default_rng(t)
    ba = rng.standard_normal((1, t, 2 * HV)) * 4
    ba[0, :, ::5] = rng.uniform(-40, 40, ba[0, :, ::5].shape)
    ba[0, 0, :] = 0.0
    ba = mx.array(ba.astype(np.float32)).astype(dtype)
    A_log = mx.array(rng.uniform(-3, 3, HV).astype(np.float32))
    dt_bias = mx.array(rng.uniform(-5, 5, HV).astype(np.float32))
    b, a = ba[..., :HV], ba[..., HV:]
    assert gdn_scan.gates_supported(q, v, ba, A_log, dt_bias)
    want_y, want_s = gdn_scan.scan_staged(
        q, k, v, compute_g(A_log, a, dt_bias), mx.sigmoid(b), s0, tiled=tiled)
    got_y, got_s = gdn_scan.scan_staged_gates(q, k, v, ba, A_log, dt_bias, s0,
                                              tiled=tiled)
    assert mx.array_equal(got_y, want_y).item()
    assert mx.array_equal(got_s, want_s).item()


def test_scan_gates_decline_a_bf16_dt_bias():
    """compute_g adds a bf16 dt_bias to a in bf16, which the kernel does not."""
    q, k, v, _, _, _ = _inputs(4)
    ba = mx.zeros((1, 4, 2 * HV), dtype=mx.bfloat16)
    A_log = mx.zeros((HV,))
    assert not gdn_scan.gates_supported(q, v, ba, A_log,
                                        mx.zeros((HV,), dtype=mx.bfloat16))
    assert not gdn_scan.gates_supported(q, v, ba[..., :HV], A_log,
                                        mx.zeros((HV,)))


def _gdn_layer(hidden=256):
    from types import SimpleNamespace

    import gmlx.models.qwen4_exp.model as qm

    args = SimpleNamespace(
        hidden_size=hidden, linear_num_value_heads=HV, linear_num_key_heads=HK,
        linear_key_head_dim=D, linear_value_head_dim=D,
        linear_conv_kernel_dim=4, rms_norm_eps=1e-6)
    mx.random.seed(0)
    layer = qm.GatedDeltaNet(args)
    layer.set_dtype(mx.bfloat16)
    # A GGUF's decay weights stay float32.
    layer.A_log = mx.random.uniform(-2, 2, (HV,))
    layer.dt_bias = mx.random.uniform(-2, 2, (HV,))
    layer.eval()
    return layer


@pytest.mark.parametrize("rows", [1, 2, 5, 16, 40, 300, 871])
def test_joined_ba_product_has_the_bits_of_two(rows):
    import gmlx.upstream.gdn_patches as gp

    layer = _gdn_layer()
    x = (mx.random.normal((1, rows, 256), key=mx.random.key(rows)) * 0.5
         ).astype(mx.bfloat16)
    want_b, want_a = layer.in_proj_b(x), layer.in_proj_a(x)
    assert layer._ba(x)[2] is None
    assert gp._gdn_try_cat_ba(layer)
    b, a, ba = layer._ba(x)
    assert ba is not None and ba.shape == (1, rows, 2 * HV)
    assert mx.array_equal(b, want_b).item()
    assert mx.array_equal(a, want_a).item()


def test_layer_prefill_computes_the_gates_in_the_scan(monkeypatch):
    """With the b | a concatenation, a prefill takes scan_staged_gates, with
    the bits of the gates as ops (GMLX_GDN_SCAN_GATES=0) and of the stock
    scan (GMLX_GDN_STAGED_SCAN=0)."""
    import gmlx.upstream.gdn_patches as gp

    layer = _gdn_layer()
    assert gp._gdn_try_cat_ba(layer)
    x = (mx.random.normal((1, 40, 256)) * 0.5).astype(mx.bfloat16)
    calls = []
    real = gdn_scan.scan_staged_gates
    monkeypatch.setattr(gdn_scan, "scan_staged_gates",
                        lambda *a, **kw: calls.append(1) or real(*a, **kw))
    for var in ("GMLX_GDN_SCAN_GATES", "GMLX_GDN_STAGED_SCAN"):
        monkeypatch.delenv(var, raising=False)
    runs = []
    for env in ({}, {"GMLX_GDN_SCAN_GATES": "0"}, {"GMLX_GDN_STAGED_SCAN": "0"}):
        for var, val in env.items():
            monkeypatch.setenv(var, val)
        c = [None, None]
        out = layer(x, cache=c)
        mx.eval(out, c)
        runs.append((out, c))
        for var in env:
            monkeypatch.delenv(var)
    assert len(calls) == 1
    got, cg = runs[0]
    for want, cw in runs[1:]:
        assert mx.array_equal(got, want).item()
        assert mx.array_equal(cg[0], cw[0]).item()
        assert mx.array_equal(cg[1], cw[1]).item()


def test_layer_prefill_takes_the_staged_scan_with_the_same_bits(monkeypatch):
    """GatedDeltaNet at a prefill width takes the staged scan; a one-row
    step and GMLX_GDN_STAGED_SCAN=0 take the stock one, with the same
    output and state."""
    from types import SimpleNamespace

    import gmlx.models.qwen4_exp.model as qm

    args = SimpleNamespace(
        hidden_size=256, linear_num_value_heads=HV, linear_num_key_heads=HK,
        linear_key_head_dim=D, linear_value_head_dim=D,
        linear_conv_kernel_dim=4, rms_norm_eps=1e-6)
    mx.random.seed(0)
    layer = qm.GatedDeltaNet(args)
    layer.set_dtype(mx.bfloat16)
    layer.eval()
    x = (mx.random.normal((1, 40, 256)) * 0.5).astype(mx.bfloat16)
    calls = []
    real = gdn_scan.scan_staged
    monkeypatch.setattr(gdn_scan, "scan_staged",
                        lambda *a, **kw: calls.append(1) or real(*a, **kw))
    monkeypatch.delenv("GMLX_GDN_STAGED_SCAN", raising=False)
    c1 = [None, None]
    got = layer(x, cache=c1)
    mx.eval(got, c1)
    assert len(calls) == 1
    mx.eval(layer(x[:, :1], cache=[None, None]))
    assert len(calls) == 1
    monkeypatch.setenv("GMLX_GDN_STAGED_SCAN", "0")
    c0 = [None, None]
    want = layer(x, cache=c0)
    mx.eval(want, c0)
    assert len(calls) == 1
    assert mx.array_equal(got, want).item()
    assert mx.array_equal(c1[0], c0[0]).item()
    assert mx.array_equal(c1[1], c0[1]).item()
