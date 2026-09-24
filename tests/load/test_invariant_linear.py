"""Batch-invariant small float projections: the kernel gives one row the
same result at any batch size, matches an fp64 reference as closely as
stock, and the install swaps only plain float Linears under the width."""
from __future__ import annotations

import os

import numpy as np
import mlx.core as mx
import mlx.nn as nn
import pytest

from gmlx.load import invariant_linear as il

if os.environ.get("KQUANT_FORCE_CPU") or not mx.metal.is_available():
    # the kernels are Metal and the CI runners' paravirtual GPU is not a
    # dependable place to compile them; FORCE_CPU keeps them off it
    pytest.skip("Metal kernels need a real GPU and a GPU default device", allow_module_level=True)


def _rows(rng, m, k, dtype):
    return mx.array(rng.standard_normal((1, m, k)).astype(np.float32)).astype(dtype)


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16, mx.float32])
@pytest.mark.parametrize("n,k", [(256, 2048), (32, 2048), (1, 2048), (7, 40)])
def test_row_result_independent_of_batch(dtype, n, k):
    rng = np.random.default_rng(1)
    w = mx.array(rng.standard_normal((n, k)).astype(np.float32)).astype(dtype)
    x = _rows(rng, 64, k, dtype)
    y64 = il.invariant_linear(x, w)
    y1 = il.invariant_linear(x[:, :1], w)
    y3 = il.invariant_linear(x[:, :3], w)
    mx.eval(y64, y1, y3)
    f32 = lambda a: np.array(a.astype(mx.float32))  # noqa: E731
    assert np.array_equal(f32(y64[:, :1]), f32(y1))
    assert np.array_equal(f32(y64[:, :3]), f32(y3))


def test_matches_reference_like_stock():
    rng = np.random.default_rng(2)
    n, k = 256, 2048
    wf = rng.standard_normal((n, k)).astype(np.float32)
    xf = rng.standard_normal((1, 512, k)).astype(np.float32)
    w = mx.array(wf).astype(mx.bfloat16)
    x = mx.array(xf).astype(mx.bfloat16)
    b = mx.array(rng.standard_normal(n).astype(np.float32)).astype(mx.bfloat16)
    ref = (np.array(x.astype(mx.float32), np.float64)
           @ np.array(w.astype(mx.float32), np.float64).T
           + np.array(b.astype(mx.float32), np.float64))
    ours = il.invariant_linear(x, w, b)
    stock = x @ w.T + b
    mx.eval(ours, stock)
    assert ours.dtype == mx.bfloat16
    e_ours = np.abs(np.array(ours.astype(mx.float32), np.float64) - ref).mean()
    e_stock = np.abs(np.array(stock.astype(mx.float32), np.float64) - ref).mean()
    assert e_ours <= e_stock * 1.05


def test_element_and_bf16_paths_agree():
    """The 16-byte bf16 path and the element loop sum in the same order,
    so a K that is not a multiple of 8 (element loop) and one that is
    (bf16 loads) give the same value on the shared prefix of K."""
    rng = np.random.default_rng(3)
    w = mx.array(rng.standard_normal((16, 64)).astype(np.float32)).astype(mx.bfloat16)
    x = _rows(rng, 5, 64, mx.bfloat16)
    fast = il.invariant_linear(x, w)
    slow = il._element_kernel(
        inputs=[x.reshape(-1, 64), w], template=[("T", mx.bfloat16)],
        grid=(5, 16, 1), threadgroup=(5, 8, 1),
        output_shapes=[(5, 16)], output_dtypes=[mx.float32])[0]
    mx.eval(fast, slow)
    assert np.array_equal(np.array(fast.astype(mx.float32)),
                          np.array(slow.astype(mx.bfloat16).astype(mx.float32)).reshape(1, 5, 16))


class _Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.router = nn.Linear(64, 8, bias=False)
        self.gate = nn.Linear(64, 4, bias=True)
        self.wide = nn.Linear(64, 1024, bias=False)
        self.quant = nn.QuantizedLinear(64, 8, bias=False)


def test_install_swaps_only_small_float_linears(monkeypatch):
    monkeypatch.setenv("GMLX_BATCH_INVARIANT", "1")
    m = _Toy()
    assert il.install_batch_invariant_linears(m) == 2
    assert type(m.router) is il.BatchInvariantLinear
    assert type(m.gate) is il.BatchInvariantLinear
    assert type(m.wide) is nn.Linear
    assert type(m.quant) is nn.QuantizedLinear
    x = mx.random.normal((2, 3, 64)).astype(mx.bfloat16)
    m.router.weight = m.router.weight.astype(mx.bfloat16)
    y = m.router(x)
    mx.eval(y)
    assert y.shape == (2, 3, 8) and y.dtype == mx.bfloat16
    # an input wider than the weight takes the stock forward
    y32 = m.router(x.astype(mx.float32))
    mx.eval(y32)
    assert y32.dtype == mx.float32


def test_float32_weight_takes_a_bf16_input_and_training_skips_the_kernel(monkeypatch):
    """The loader keeps MoE routers in float32 while activations are bf16:
    the kernel runs on the promoted input, as stock promotion would, and a
    module in training mode never enters the kernel (it has no gradient)."""
    monkeypatch.setenv("GMLX_BATCH_INVARIANT", "1")
    m = _Toy()
    m.eval()  # a fresh module is in training mode
    il.install_batch_invariant_linears(m)
    x = mx.random.normal((2, 3, 64)).astype(mx.bfloat16)
    calls = []
    orig = il.invariant_linear
    monkeypatch.setattr(il, "invariant_linear", lambda *a, **k: calls.append(1) or orig(*a, **k))
    y = m.router(x)
    mx.eval(y)
    assert y.dtype == mx.float32 and y.shape == (2, 3, 8) and calls == [1]
    m.train()
    y_train = m.router(x)
    mx.eval(y_train)
    assert y_train.dtype == mx.float32 and calls == [1]


def test_install_off_by_default(monkeypatch):
    monkeypatch.delenv("GMLX_BATCH_INVARIANT", raising=False)
    m = _Toy()
    assert il.install_batch_invariant_linears(m) == 0
    assert type(m.router) is nn.Linear


def test_install_width_env(monkeypatch):
    monkeypatch.setenv("GMLX_BATCH_INVARIANT", "1")
    monkeypatch.setenv("GMLX_BATCH_INVARIANT_MAX_OUT", "4")
    m = _Toy()
    assert il.install_batch_invariant_linears(m) == 1
    assert type(m.gate) is il.BatchInvariantLinear
    assert type(m.router) is nn.Linear
