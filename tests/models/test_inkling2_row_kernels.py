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


@pytest.mark.parametrize("biased", [False, True])
@pytest.mark.parametrize("rows", [1, 7])
@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
def test_the_router_picks_the_ops_paths_experts(biased, rows, dtype):
    mx.random.seed(4)
    E, k, scale = 128, 8, 8.0
    logits = mx.random.normal((1, rows, E)) * 3
    bias = mx.random.normal((E,)) * 0.1 if biased else None
    inds, weights = row_kernels.router(logits, bias, k, scale, dtype)
    want_i, want_w = M._router(k, scale, biased, dtype)(logits, bias)
    got_i, got_w = np.array(inds), _f32(weights)
    want_i, want_w = np.array(want_i), _f32(want_w)
    # Same experts and the weight of each; the kernel lists them by score.
    for r in range(rows):
        go, wo = np.argsort(got_i[0, r]), np.argsort(want_i[0, r])
        np.testing.assert_array_equal(got_i[0, r][go], want_i[0, r][wo])
        tol = 1e-6 if dtype == mx.float32 else 1e-2
        np.testing.assert_allclose(got_w[0, r][go], want_w[0, r][wo],
                                   rtol=tol, atol=tol)
    sel = np.array(mx.sigmoid(logits)) + (np.array(bias) if biased else 0)
    picked = np.take_along_axis(sel, got_i.astype(np.int64), -1)
    assert (np.diff(picked, axis=-1) <= 0).all()


def test_the_router_gradient_comes_from_the_ops_path():
    mx.random.seed(5)
    logits = mx.random.normal((1, 3, 128))
    probe = mx.random.normal((1, 3, 8))

    def loss(fn):
        def run(x):
            inds, w = fn(x)
            order = mx.argsort(inds, axis=-1)  # compare in id order
            return (mx.take_along_axis(w, order, axis=-1) * probe).sum()
        return mx.grad(run)(logits)

    got = loss(lambda x: row_kernels.router(x, None, 8, 8.0, mx.float32))
    want = loss(lambda x: M._router(8, 8.0, False, mx.float32)(x, None))
    np.testing.assert_allclose(np.array(got), np.array(want), rtol=1e-5,
                               atol=1e-6)


@pytest.mark.parametrize("n_blocks", [0, 1, 7])
@pytest.mark.parametrize("nq", [1, 3])
@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
def test_the_depth_mix_matches_the_ops_path(n_blocks, nq, dtype, ops_path):
    mx.random.seed(6)
    shape = (2, 3, 512)
    blocks = [M._rms(mx.random.normal(shape) * 2) for _ in range(n_blocks)]
    cur = (mx.random.normal(shape) * 3).astype(dtype)
    query = mx.random.normal((nq, 512))
    assert row_kernels.mix_eligible(blocks, cur, query)
    got = M.depth_mix(blocks, cur, query)
    want = ops_path(M.depth_mix, blocks, cur, query)
    assert got.shape == (2, 3, nq, 512) and got.dtype == mx.float32
    np.testing.assert_allclose(_f32(got), _f32(want), rtol=1e-5, atol=1e-5)


def test_the_depth_mix_gradient_comes_from_the_ops_path(ops_path):
    mx.random.seed(7)
    blocks = [M._rms(mx.random.normal((1, 2, 256))) for _ in range(3)]
    cur = mx.random.normal((1, 2, 256))
    query = mx.random.normal((2, 256))

    def grads():
        return mx.grad(lambda c, q: M.depth_mix(blocks, c, q).sum(),
                       argnums=(0, 1))(cur, query)

    for got, want in zip(grads(), ops_path(grads)):
        np.testing.assert_allclose(np.array(got), np.array(want),
                                   rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("rows", [1, 4])
@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
def test_the_controller_matches_the_ops_path(rows, dtype, ops_path):
    mx.random.seed(8)
    ctrl = M.Controller(512, 64)
    ctrl.update({"down": {"weight": mx.random.normal((64, 1024)) * 0.05},
                 "gain": {"weight": mx.random.normal((512, 64)) * 0.1},
                 "forget": {"weight": mx.random.normal((512, 64)) * 0.1}})
    cur = (mx.random.normal((1, rows, 512)) * 2).astype(dtype)
    msg = mx.random.normal((1, rows, 512))
    w = (ctrl.down.weight, ctrl.gain.weight, ctrl.forget.weight)
    assert row_kernels.control_eligible(cur, msg, *w)
    got, want = ctrl(cur, msg), ops_path(ctrl, cur, msg)
    assert got.dtype == dtype
    tol = 1e-5 if dtype == mx.float32 else 1e-2
    np.testing.assert_allclose(_f32(got), _f32(want), rtol=tol, atol=tol)


def test_the_controller_gradient_comes_from_the_ops_path(ops_path):
    mx.random.seed(9)
    ctrl = M.Controller(256, 32)
    ctrl.update({"down": {"weight": mx.random.normal((32, 512)) * 0.05},
                 "gain": {"weight": mx.random.normal((256, 32)) * 0.1},
                 "forget": {"weight": mx.random.normal((256, 32)) * 0.1}})
    cur = mx.random.normal((1, 2, 256))
    msg = mx.random.normal((1, 2, 256))

    def grads():
        return mx.grad(lambda c, m: ctrl(c, m).sum(), argnums=(0, 1))(cur, msg)

    for got, want in zip(grads(), ops_path(grads)):
        np.testing.assert_allclose(np.array(got), np.array(want),
                                   rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("L", [1, 4])
@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
def test_the_short_conv_matches_the_ops_path(L, dynamic, dtype):
    mx.random.seed(10)
    B, C, width, heads = 2, 64, 4, 8
    state = mx.random.normal((B, width - 1, C))
    x = mx.random.normal((B, L, C)).astype(dtype)
    w = mx.random.normal((C, width))
    dyn = (mx.random.normal((B, L, heads * width)).astype(dtype)
           if dynamic else None)
    assert row_kernels.conv_eligible(state, x, w, dyn)
    ops = M._conv_dynamic if dynamic else M._conv_static
    out, kept, window = row_kernels.conv(state, x, w, dyn, True, ops)
    want_window, want = ops(state, x, w, *([dyn] if dynamic else []))
    tol = 1e-5 if dtype == mx.float32 else 1e-2
    np.testing.assert_allclose(_f32(out), _f32(want), rtol=tol, atol=tol)
    np.testing.assert_array_equal(np.array(window), np.array(want_window))
    np.testing.assert_array_equal(np.array(kept),
                                  np.array(want_window[:, L:]))


def test_the_short_conv_gradient_comes_from_the_ops_path():
    mx.random.seed(11)
    state = mx.random.normal((1, 3, 32))
    x = mx.random.normal((1, 2, 32))
    w = mx.random.normal((32, 4))
    dyn = mx.random.normal((1, 2, 16))

    def loss(x, d):
        out, kept = row_kernels.conv(state, x, w, d, False, M._conv_dynamic)
        return out.sum() + kept.sum()

    def ref(x, d):
        window, out = M._conv_dynamic(state, x, w, d)
        return out.sum() + window[:, 2:].sum()

    got = mx.grad(loss, argnums=(0, 1))(x, dyn)
    want = mx.grad(ref, argnums=(0, 1))(x, dyn)
    for g, r in zip(got, want):
        np.testing.assert_allclose(np.array(g), np.array(r), rtol=1e-5,
                                   atol=1e-6)


@pytest.mark.parametrize("ring", [False, True])
@pytest.mark.parametrize("L", [1, 4])
@pytest.mark.parametrize("window", [None, 6])
def test_the_bias_mask_matches_the_ops_path(ring, L, window):
    mx.random.seed(12)
    B, H, RL, extent, K = 1, 4, 16, 8, 13
    r = mx.random.normal((B, L, H, RL))
    sink = mx.random.normal((B, L, H))
    table = mx.pad(mx.random.normal((extent, RL)), [(0, 1), (0, 0)])
    keys = mx.zeros((B, 2, K, 8))
    idx = 5 if ring else None
    got = row_kernels.bias_mask(r, sink, table, keys, window, idx,
                                mx.float32)
    want = row_kernels._mask_ops(r, sink, table, mx.array([K, idx or 0]),
                                 K, window or 0, ring, mx.float32)
    g, w = np.array(got), np.array(want)
    np.testing.assert_array_equal(np.isinf(g), np.isinf(w))
    fin = ~np.isinf(w)
    np.testing.assert_allclose(g[fin], w[fin], rtol=1e-5, atol=1e-5)
    # Distances past the table take the pad row: the bias is -sink alone.
    assert fin.any()


def test_the_bias_mask_gradient_comes_from_the_ops_path():
    mx.random.seed(13)
    r = mx.random.normal((1, 2, 2, 16))
    sink = mx.random.normal((1, 2, 2))
    table = mx.pad(mx.random.normal((8, 16)), [(0, 1), (0, 0)])
    keys = mx.zeros((1, 1, 10, 8))
    probe = mx.random.normal((1, 2, 2, 10))
    params = mx.array([10, 0])

    def loss(fn):
        def run(r, s):
            m = fn(r, s)
            return mx.where(mx.isinf(m), 0, m * probe).sum()
        return mx.grad(run, argnums=(0, 1))(r, sink)

    got = loss(lambda r, s: row_kernels.bias_mask(r, s, table, keys, None,
                                                  None, mx.float32))
    want = loss(lambda r, s: row_kernels._mask_ops(r, s, table, params, 10, 0,
                                                   False, mx.float32))
    for g, w in zip(got, want):
        np.testing.assert_allclose(np.array(g), np.array(w), rtol=1e-5,
                                   atol=1e-6)
