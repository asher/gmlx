"""The qwen4exp stream mixer with float down and up weights runs as fused
kernels at decode width. They must agree with the eager op chain to within
its own rounding, in its dtypes."""
from __future__ import annotations

import os

import mlx.core as mx
import numpy as np
import pytest

import gmlx.models.qwen4_exp.model as q4

if os.environ.get("KQUANT_FORCE_CPU") or not mx.metal.is_available():
    pytest.skip("the kernels need a real GPU and a GPU default device", allow_module_level=True)
if q4._kq_hc() is None:
    pytest.skip("needs the kq hyper-connection norm", allow_module_level=True)


def _mixer(D, LR, half, inject_dtype, seed=0):
    mx.random.seed(seed)
    m = q4.HyperConnection(D, 4, LR, 1e-6)
    K = 4 * D
    m.norm.weight = (1.0 + 0.1 * mx.random.normal((K,))).astype(half)
    m.down.weight = (0.02 * mx.random.normal((LR, K))).astype(half)
    m.up.weight = (0.05 * mx.random.normal((K, LR))).astype(half)
    m.inject.weight = (0.02 * mx.random.normal((4, K))).astype(inject_dtype)
    m.eval()
    mx.eval(m.parameters())
    return m


def _reference(m, h):
    f = lambda a: np.array(a.astype(mx.float32)).astype(np.float64)
    hh = f(h)
    D = hh.shape[-1]
    xn = hh / np.sqrt((hh ** 2).mean(-1, keepdims=True) + m.norm.eps) * f(m.norm.weight).reshape(4, D)
    xf = xn.reshape(*hh.shape[:2], 4 * D)
    t = xf @ f(m.down.weight).T * 0.25
    lo = t / (1 + np.exp(-t))
    gate = 1 / (1 + np.exp(-(lo @ f(m.up.weight).T))).reshape(xn.shape)
    inj = 2 / (1 + np.exp(-(xf @ f(m.inject.weight).T) * 0.25))
    return (gate * xn).mean(axis=2), inj


def _eager(m, h, monkeypatch):
    with monkeypatch.context() as mp:
        mp.setattr(q4, "_hc_float_kerns_cache", False)
        out = m(h)
        mx.eval(out)
    return out


def _err(got, ref):
    return float(np.abs(np.array(got.astype(mx.float32)) - ref).max() / np.abs(ref).max())


@pytest.mark.parametrize("rows", [1, 2, 4, 8])
@pytest.mark.parametrize("half", [mx.bfloat16, mx.float16])
@pytest.mark.parametrize("inject_f32", [False, True])
@pytest.mark.parametrize("h_f32", [False, True])
def test_fused_mixer_agrees_with_the_eager_chain(monkeypatch, rows, half, inject_f32, h_f32):
    m = _mixer(128, 32, half, mx.float32 if inject_f32 else half)
    h = mx.random.normal((1, rows, 4, 128)).astype(mx.float32 if h_f32 else half)
    assert m._hcfl_ok(h.dtype)
    mixed, inj = m(h)
    e_mixed, e_inj = _eager(m, h, monkeypatch)
    assert mixed.dtype == e_mixed.dtype and inj.dtype == e_inj.dtype
    assert mixed.shape == e_mixed.shape and inj.shape == e_inj.shape
    r_mixed, r_inj = _reference(m, h)
    assert _err(mixed, r_mixed) <= 2 * _err(e_mixed, r_mixed) + 1e-5
    assert _err(inj, r_inj) <= 2 * _err(e_inj, r_inj) + 1e-5


def test_fused_mixer_at_the_model_shape(monkeypatch):
    m = _mixer(2560, 320, mx.bfloat16, mx.bfloat16)
    h = mx.random.normal((1, 4, 4, 2560)).astype(mx.bfloat16)
    mixed, inj = m(h)
    e_mixed, e_inj = _eager(m, h, monkeypatch)
    r_mixed, r_inj = _reference(m, h)
    assert _err(mixed, r_mixed) <= 2 * _err(e_mixed, r_mixed) + 1e-5
    assert _err(inj, r_inj) <= 2 * _err(e_inj, r_inj) + 1e-5


def test_rows_batched_match_rows_alone():
    """A row's result does not depend on the rows beside it."""
    m = _mixer(128, 32, mx.bfloat16, mx.bfloat16)
    h = mx.random.normal((1, 4, 4, 128)).astype(mx.bfloat16)
    mixed, inj = m(h)
    for r in range(4):
        one_mixed, one_inj = m(h[:, r:r + 1])
        assert mx.array_equal(one_mixed[0, 0], mixed[0, r]).item()
        assert mx.array_equal(one_inj[0, 0], inj[0, r]).item()


def test_prefill_width_and_odd_shapes_keep_the_eager_chain(monkeypatch):
    m = _mixer(128, 32, mx.bfloat16, mx.bfloat16)
    calls = []
    monkeypatch.setattr(q4, "_hc_float_mix", lambda *a: calls.append(1))
    m(mx.random.normal((1, 9, 4, 128)).astype(mx.bfloat16))
    odd = _mixer(128, 30, mx.bfloat16, mx.bfloat16)
    assert not odd._hcfl_ok(mx.bfloat16)
    odd(mx.random.normal((1, 1, 4, 128)).astype(mx.bfloat16))
    assert not calls


def test_env_switch_turns_the_route_off(monkeypatch):
    monkeypatch.setenv("GMLX_Q4_HC_FLOAT_KERN", "0")
    monkeypatch.setattr(q4, "_hc_float_kerns_cache", None)
    m = _mixer(128, 32, mx.bfloat16, mx.bfloat16)
    assert q4._hc_float_kerns() is None
    assert not m._hcfl_ok(mx.bfloat16)
    mixed, _ = m(mx.random.normal((1, 1, 4, 128)).astype(mx.bfloat16))
    assert mixed.shape == (1, 1, 128)
