"""The float32 products of gmlx.image.float32, which hold float32
precision in a process that multiplies float32 at TF32."""

from __future__ import annotations

import numpy as np
import pytest
from block_gates import tf32

import mlx.core as mx

from gmlx.image import float32 as f32


def test_a_split_product_is_float32_in_this_process():
    """``exact_matmul`` holds float32 precision whatever MLX's TF32 switch
    reads in this process, at the magnitudes of the tower's residual
    stream, for ``a @ b`` and ``a @ b.T``."""
    rng = np.random.default_rng(0)
    a = (rng.standard_normal((96, 160)) * 3000).astype(np.float32)
    b = rng.standard_normal((160, 72)).astype(np.float32) / 12
    want = a.astype(np.float64) @ b.astype(np.float64)
    for got in (f32.exact_matmul(mx.array(a), mx.array(b)),
                f32.exact_matmul(mx.array(a), mx.array(b.T.copy()),
                                 b_t=True)):
        assert got.dtype == mx.float32
        gap = np.linalg.norm(np.array(got, np.float64) - want) \
            / np.linalg.norm(want)
        assert gap < 2e-6, gap
    # The high part keeps 10 mantissa bits, and the parts sum to the input.
    high, low = f32.split(mx.array(a))
    assert not (np.array(high).view(np.uint32) & 0x1FFF).any()
    assert np.array_equal(np.array(high + low), a)
    # A matrix read from bfloat16 is its own high part: the product that
    # leaves it whole is the split product, bit for bit.
    b16 = mx.array(b).astype(mx.bfloat16).astype(mx.float32)
    assert f32.tf32_exact(b16).item() and not f32.tf32_exact(
        mx.array(b)).item()
    for t in (False, True):
        w = b16.T if t else b16
        assert mx.array_equal(
            f32.exact_matmul(mx.array(a), w, b_t=t, b_exact=True),
            f32.exact_matmul(mx.array(a), w, b_t=t)).item()


def test_the_product_checks_read_the_device(monkeypatch):
    """Each check is kept per device, so a CPU stream's answer is never
    the GPU's."""
    monkeypatch.setattr(f32, "_EXACT", {})
    monkeypatch.setattr(f32, "_CONV_EXACT", {})
    with mx.stream(mx.cpu):
        cpu = str(mx.default_device())
        assert f32.f32_products_exact() is True
        assert f32.conv1d_exact(8, 8) is True
    assert f32._EXACT == {cpu: True}
    assert f32._CONV_EXACT == {(cpu, 8, 8, 1): True}
    if tf32():
        assert f32.conv1d_exact(8, 8) is False


def _gap(got, want: np.ndarray) -> float:
    return float(np.linalg.norm(np.array(got, np.float64) - want)
                 / np.linalg.norm(want))


def _transposed(x: np.ndarray, w: np.ndarray, stride: int,
                padding: int) -> np.ndarray:
    """A float64 transposed convolution: each input sample adds its kernel
    times itself at ``stride`` steps, less ``padding`` on each side."""
    b, n, _ = x.shape
    k = w.shape[1]
    out = np.zeros((b, (n - 1) * stride + k, w.shape[0]))
    for j in range(k):
        out[:, j:j + (n - 1) * stride + 1:stride] += x @ w[:, j].T
    return out[:, padding:out.shape[1] - padding]


@pytest.mark.parametrize("cin,cout,k,stride,padding,n",
                         [(64, 32, 9, 5, 2, 40), (32, 16, 4, 2, 1, 300),
                          (1, 1, 12, 2, 0, 500)])
def test_a_transposed_conv_is_float32_in_this_process(
        cin, cout, k, stride, padding, n):
    """``conv_transpose1d`` holds float32 precision whatever MLX's TF32
    switch reads: under twice the error of the stock op on the CPU."""
    rng = np.random.default_rng(k)
    x = rng.standard_normal((2, n, cin)).astype(np.float32)
    w = (rng.standard_normal((cout, k, cin)) / np.sqrt(cin * k)).astype(
        np.float32)
    want = _transposed(x.astype(np.float64), w.astype(np.float64), stride,
                       padding)
    got = f32.conv_transpose1d(mx.array(x), mx.array(w), stride, padding)
    cpu = mx.conv_transpose1d(mx.array(x), mx.array(w), stride=stride,
                              padding=padding, stream=mx.cpu)
    assert got.shape == want.shape
    assert _gap(got, want) < 2 * _gap(cpu, want), (_gap(got, want),
                                                   _gap(cpu, want))


def _conv(x: np.ndarray, w: np.ndarray, padding: int,
          dilation: int) -> np.ndarray:
    """A float64 convolution with zero padding, as mx.conv1d computes."""
    x = np.pad(x, [(0, 0), (padding, padding), (0, 0)])
    n = x.shape[1] - dilation * (w.shape[1] - 1)
    return sum(x[:, j * dilation:j * dilation + n] @ w[:, j].T
               for j in range(w.shape[1]))


@pytest.mark.parametrize("split", [False, True], ids=["as is", "split"])
@pytest.mark.parametrize("cin,cout", [(8, 8), (16, 8), (64, 8), (16, 24)])
def test_a_conv_is_float32_in_this_process(monkeypatch, split, cin, cout):
    """``conv1d`` holds float32 precision whatever MLX's TF32 switch reads,
    under 1.5 times the error of the stock op on the CPU, at shapes MLX
    convolves at TF32 there (8 to 8, 16 to 24) and at others, and its
    split form is the same product. On a TF32 device two parts read about
    twice the CPU's error."""
    if split:
        monkeypatch.setattr(f32, "conv1d_exact", lambda *a: False)
    rng = np.random.default_rng(cin + cout)
    x = rng.standard_normal((2, 300, cin)).astype(np.float32)
    w = (rng.standard_normal((cout, 7, cin)) / np.sqrt(7 * cin)).astype(
        np.float32)
    want = _conv(x.astype(np.float64), w.astype(np.float64), 9, 3)
    got = f32.conv1d(mx.array(x), mx.array(w), padding=9, dilation=3)
    cpu = mx.conv1d(mx.array(x), mx.array(w), padding=9, dilation=3,
                    stream=mx.cpu)
    assert _gap(got, want) < 1.5 * _gap(cpu, want), (_gap(got, want),
                                                     _gap(cpu, want))
