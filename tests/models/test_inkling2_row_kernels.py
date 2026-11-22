"""Inkling 2 row kernels: the one-dispatch L2 clip and float32 RMS norm
against the ops they replace.

The kernels sum a row in a different order, so float32 results match to
rounding. A bfloat16 norm output is bit-equal on these inputs.
"""
from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

import gmlx.models.inkling2.model as M
from gmlx.models.inkling2 import row_kernels

pytestmark = pytest.mark.skipif(
    not row_kernels.enabled(), reason="the row kernels run on the GPU")


@pytest.fixture
def ops_path(monkeypatch):
    def run(fn, *args):
        monkeypatch.setenv("GMLX_INKLING2_ROW_KERNEL", "0")
        out = fn(*args)
        mx.eval(out)
        monkeypatch.delenv("GMLX_INKLING2_ROW_KERNEL")
        return out
    return run


def _f32(a):
    return np.array(a.astype(mx.float32))


@pytest.mark.parametrize("shape", [(1, 1, 32, 128), (2, 5, 4, 16)])
@pytest.mark.parametrize("scaled", [False, True])
@pytest.mark.parametrize("tau_rows", [0, 1, 2])
@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
def test_the_clip_matches_the_ops_path(shape, scaled, tau_rows, dtype,
                                       ops_path):
    B, L, H, D = shape
    mx.random.seed(1)
    # Rows sliced out of a wider buffer, as a row-fused projection gives
    # them: the kernel reads them through their strides.
    wide = mx.random.normal((B, L, H * D + 24)) * 3
    x = wide[..., 8:8 + H * D].reshape(B, L, H, D).astype(mx.bfloat16)
    scale = mx.random.uniform(shape=(H, D)) + 0.5 if scaled else None
    tau = (mx.random.uniform(shape=(min(tau_rows, B), L, 1, 1)) + 1
           if tau_rows else None)
    assert row_kernels.clip_eligible(x, scale, tau)
    got = M.clip(x, scale, 8.0, tau, dtype)
    want = ops_path(M.clip, x, scale, 8.0, tau, dtype)
    assert got.dtype == dtype
    tol = 1e-6 if dtype == mx.float32 else 1e-2
    np.testing.assert_allclose(_f32(got), _f32(want), rtol=tol, atol=tol)


@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
def test_the_norm_matches_the_ops_path(dtype, ops_path):
    mx.random.seed(2)
    norm = M._Norm(2304)
    norm.weight = mx.random.normal((2304,))
    x = (mx.random.normal((2, 3, 2304)) * 2).astype(dtype)
    got, want = norm(x), ops_path(norm, x)
    assert got.dtype == dtype
    np.testing.assert_allclose(_f32(got), _f32(want), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(_f32(M._rms(x)), _f32(ops_path(M._rms, x)),
                               rtol=1e-6, atol=1e-6)


def test_gradients_come_from_the_ops_path(ops_path):
    # A kernel has no VJP; training reads the ops path's gradient.
    mx.random.seed(3)
    x = mx.random.normal((1, 3, 4, 16))
    scale = mx.random.uniform(shape=(4, 16)) + 0.5
    tau = mx.random.uniform(shape=(1, 3, 1, 1)) + 1
    norm = M._Norm(512)
    norm.weight = mx.random.normal((512,))
    y = mx.random.normal((2, 512))

    def grads():
        return (mx.grad(lambda x: M.clip(x, scale, 2.0, tau).sum())(x),
                mx.grad(lambda y: norm(y).sum())(y))

    for got, want in zip(grads(), ops_path(grads)):
        np.testing.assert_allclose(np.array(got), np.array(want),
                                   rtol=1e-5, atol=1e-6)
