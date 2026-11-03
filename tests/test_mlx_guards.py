"""``gmlx.mlx_guards`` against the stock mlx defects of
docs/upstream-defects.md.

``addmm``: a first argument of one column that is a view with a stride is
misread on a GPU stream. The LoRA wrappers of the text and the image path
add their delta through the guard, so a layer with one output, or a band
of one output column, is right over a base output that is a column of a
wider array. On the CPU device the stock op is right and the tests pass
with or without the guard.

``concatenate`` and ``pad``: past 2^31 output elements the stock ops
misplace inputs on a GPU stream. The tests here hold the routing on small
arrays with the limit lowered. The tests of each caller past the real
limit are beside the caller's other tests.
"""
from __future__ import annotations

import os

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from gmlx import mlx_guards

needs_gpu = pytest.mark.skipif(
    bool(os.environ.get("KQUANT_FORCE_CPU")) or not mx.metal.is_available(),
    reason="the stock addmm defect is on GPU streams only")

DTYPES = (mx.float32, mx.float16, mx.bfloat16)


def _np(a: mx.array) -> np.ndarray:
    return np.array(a.astype(mx.float32)).astype(np.float64)


def _wide(rows: int, cols: int, dtype) -> mx.array:
    # Whole numbers, so every dtype holds them and the sums exactly.
    return mx.arange(rows * cols).reshape(rows, cols).astype(dtype)


def _views(dtype) -> dict:
    """A c of one column and several rows, as each kind of strided view."""
    return {
        "column of a wider array": _wide(3, 7, dtype)[:, 3:4],
        "every second row": _wide(6, 1, dtype)[::2],
        "reversed rows": _wide(3, 1, dtype)[::-1],
        "transposed row": _wide(1, 3, dtype).T,
        "column of a transposed array": _wide(7, 3, dtype).T[:, 3:4],
        "batched column": _wide(6, 7, dtype).reshape(2, 3, 7)[..., 5:6],
    }


@needs_gpu
@pytest.mark.parametrize("dtype", DTYPES, ids=lambda d: str(d).split(".")[-1])
@pytest.mark.parametrize("kind", list(_views(mx.float32)))
@pytest.mark.parametrize("lazy", (False, True), ids=("evaluated", "lazy"))
def test_addmm_reads_a_strided_column(dtype, kind, lazy):
    c = _views(dtype)[kind]
    a = (mx.arange(c.size * 2).reshape(*c.shape[:-1], 2) % 3).astype(dtype)
    b = mx.array([[2.0], [1.0]]).astype(dtype)
    if not lazy:
        mx.eval(c, a, b)
    want = _np(c) + _np(a) @ _np(b)
    got = mlx_guards.addmm(c, a, b)
    assert got.dtype == dtype
    np.testing.assert_array_equal(_np(got), want)


class _Column(nn.Module):
    """A layer of one output whose result is a column of a wider product,
    as a head that reads one output of a wider layer is."""

    def __init__(self, wide: nn.Linear, column: int):
        super().__init__()
        self.wide, self.column = wide, column

    def __call__(self, x):
        return self.wide(x)[..., self.column:self.column + 1]


def _factors(rank: int, dims: int, dtype):
    a = ((mx.arange(rank * dims).reshape(rank, dims) % 5) - 2).astype(dtype)
    b = (mx.arange(rank).reshape(1, rank) + 1).astype(dtype)
    return a, b


def _base(dims: int, outs: int, dtype) -> nn.Linear:
    lin = nn.Linear(dims, outs, bias=False)
    lin.weight = ((mx.arange(outs * dims).reshape(outs, dims) % 7)
                  - 3).astype(dtype)
    return lin


@needs_gpu
@pytest.mark.parametrize("dtype", DTYPES, ids=lambda d: str(d).split(".")[-1])
def test_a_text_lora_layer_of_one_output_over_a_column_view(dtype):
    from gmlx import lora_rows
    from gmlx.load.modules import LoRAKQuantLinear

    lora_rows.configure("static", 1)
    wide = _base(4, 7, dtype)
    a, b = _factors(2, 4, dtype)
    layer = LoRAKQuantLinear(_Column(wide, 3), a, b, 1.0)
    x = ((mx.arange(5 * 4).reshape(5, 4) % 4) - 1).astype(dtype)
    xn = _np(x)
    want = (xn @ _np(wide.weight).T)[:, 3:4] + xn @ _np(a).T @ _np(b).T
    np.testing.assert_array_equal(_np(layer(x)), want)
    # A second adapter on the same target adds through the same guard.
    layer.add_slot(a, b, 1.0, 1)
    lora_rows.configure("static", 2)
    try:
        np.testing.assert_array_equal(
            _np(layer(x)), want + xn @ _np(a).T @ _np(b).T)
    finally:
        lora_rows.configure("static", 1)


@needs_gpu
@pytest.mark.parametrize("dtype", DTYPES, ids=lambda d: str(d).split(".")[-1])
def test_an_image_lora_band_of_one_output_column(dtype):
    from gmlx.image.lora import install

    wide = _base(4, 7, dtype)
    a, b = _factors(2, 4, dtype)
    a_t, b_t = mx.contiguous(a.T), mx.contiguous(b.T)
    x = ((mx.arange(5 * 4).reshape(5, 4) % 4) - 1).astype(dtype)
    xn = _np(x)
    delta = xn @ _np(a).T @ _np(b).T
    whole = xn @ _np(wide.weight).T
    # One pair on output column 3 of a layer of seven outputs.
    band = install._Slice([(3, 4, a_t, b_t)], 7)
    want = whole.copy()
    want[:, 3:4] += delta
    np.testing.assert_array_equal(_np(band(wide, x)), want)
    # One pair over the whole weight of a layer of one output.
    block = install._Block(a_t, b_t)
    np.testing.assert_array_equal(_np(block(_Column(wide, 3), x)),
                                  whole[:, 3:4] + delta)


def test_the_guard_changes_the_graph_for_one_column_only(monkeypatch):
    """A c of two or more columns, or of one row, reaches the stock op as
    the array it is, so the guard adds no op at the widths a model runs.
    A c of one column and several rows goes in as another array, the
    dense one."""
    seen = []
    stock = mx.addmm

    def spy(c, a, b):
        seen.append(c)
        return stock(c, a, b)

    monkeypatch.setattr(mlx_guards.mx, "addmm", spy)
    wide = _wide(3, 7, mx.float32)
    for c in (wide[:, 2:4], wide[:, 1:], wide, wide[1:2, 3:4],
              wide.reshape(3, 1, 7)[:, :, ::2]):
        seen.clear()
        mlx_guards.addmm(c, mx.ones((*c.shape[:-1], 2)),
                         mx.ones((2, c.shape[-1])))
        assert seen[0] is c, c.shape
    column = wide[:, 3:4]
    seen.clear()
    out = mlx_guards.addmm(column, mx.ones((3, 2)), mx.ones((2, 1)))
    assert seen[0] is not column
    np.testing.assert_array_equal(_np(seen[0]), _np(column))
    np.testing.assert_array_equal(_np(out), _np(column) + 2.0)


# The callers' layouts, at a few elements.
CONCATS = {
    # [old, zeros] on the token axis of a [B, H, S, D] KV regrow.
    "kv regrow": ([(2, 2, 6, 4), (2, 2, 2, 4)], 2),
    # [rows, hid] on the last axis of a FLUX.2 single block.
    "single block": ([(1, 5, 4), (1, 5, 12)], -1),
}
PADS = {
    # A batch's keys padded on the token axis to the joint width.
    "batch extend": ((3, 2, 5, 4), [(0, 0), (0, 0), (2, 1), (0, 0)]),
    # A VAE activation padded on the bottom and the right.
    "vae downsample": ((1, 5, 7, 4), [(0, 0), (0, 1), (0, 1), (0, 0)]),
}


def _counts(shape, start: int) -> mx.array:
    # Whole numbers from 1 to 250, which bfloat16 holds exactly, so no
    # input element is the fill.
    n = int(np.prod(shape))
    return ((mx.arange(n) + start) % 250 + 1).reshape(shape).astype(mx.bfloat16)


def _spy(monkeypatch, name: str) -> list:
    """The arrays the stock ``mx.<name>`` returns from now on."""
    calls = []
    stock = getattr(mx, name)

    def spy(*args, **kwargs):
        calls.append(stock(*args, **kwargs))
        return calls[-1]

    monkeypatch.setattr(mlx_guards.mx, name, spy)
    return calls


@pytest.mark.parametrize("kind", list(CONCATS))
def test_concatenate_takes_a_slice_update_only_past_the_limit(monkeypatch, kind):
    """An output at the limit is the stock op's array, so the graph of the
    shapes models run today is the stock one. One element more, the inputs
    are written into zeros and the values are the stock op's."""
    shapes, axis = CONCATS[kind]
    arrays = [_counts(s, 100 * i) for i, s in enumerate(shapes)]
    mx.eval(arrays)
    want = np.concatenate([_np(a) for a in arrays], axis=axis)
    calls = _spy(monkeypatch, "concatenate")
    monkeypatch.setattr(mlx_guards, "_STOCK_MAX", want.size)
    out = mlx_guards.concatenate(arrays, axis=axis)
    assert len(calls) == 1 and out is calls[0]
    monkeypatch.setattr(mlx_guards, "_STOCK_MAX", want.size - 1)
    out = mlx_guards.concatenate(arrays, axis=axis)
    assert len(calls) == 1
    assert out.dtype == mx.bfloat16
    np.testing.assert_array_equal(_np(out), want)


@pytest.mark.parametrize("kind", list(PADS))
def test_pad_takes_a_slice_update_only_past_the_limit(monkeypatch, kind):
    """As for concatenate: the stock op's array at the limit, and past it
    the input written into zeros, with the stock op's values."""
    shape, widths = PADS[kind]
    x = _counts(shape, 0)
    mx.eval(x)
    want = np.pad(_np(x), widths)
    calls = _spy(monkeypatch, "pad")
    monkeypatch.setattr(mlx_guards, "_STOCK_MAX", want.size)
    out = mlx_guards.pad(x, widths)
    assert len(calls) == 1 and out is calls[0]
    monkeypatch.setattr(mlx_guards, "_STOCK_MAX", want.size - 1)
    out = mlx_guards.pad(x, widths)
    assert len(calls) == 1
    assert out.dtype == mx.bfloat16
    np.testing.assert_array_equal(_np(out), want)


def test_a_bool_pad_past_the_limit_stays_bool(monkeypatch):
    """Past the limit the output keeps x's dtype, as the stock op's does:
    a bool mask padded with False is bool, where a fill built from the
    constant alone is int32."""
    x = mx.arange(24).reshape(2, 3, 4) % 3 == 0
    widths = [(0, 0), (1, 2), (0, 0)]
    want = mx.pad(x, widths)
    monkeypatch.setattr(mlx_guards, "_STOCK_MAX", want.size - 1)
    out = mlx_guards.pad(x, widths)
    assert (out.dtype, want.dtype) == (mx.bool_, mx.bool_)
    np.testing.assert_array_equal(np.array(out), np.array(want))
