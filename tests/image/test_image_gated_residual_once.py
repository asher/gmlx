"""ops.gated_residual_once, the gated residual of MiniMax-H3's blocks:
x + gate * y in float32, rounded once, under one shapeless compile, and
in blocks of rows into its output where mx.compile does not compile."""
from __future__ import annotations

import mlx.core as mx
import numpy as np

from gmlx.image import ops
from gmlx.image.ops import epilogues


def _inputs(rows: int, width: int = 256, seed: int = 0) -> tuple:
    rng = np.random.default_rng(seed)
    x = mx.array(40 + rng.standard_normal((1, rows, width)), mx.bfloat16)
    y = mx.array(-2e4 * rng.standard_normal((1, rows, width)), mx.bfloat16)
    g = mx.array(rng.uniform(-1, 1, (width,)), mx.bfloat16)
    mx.eval(x, y, g)
    return x, g, y


def _rounded_once(x, g, y):
    f = mx.float32
    return (x.astype(f) + g.astype(f) * y.astype(f)).astype(x.dtype)


def test_one_compile_rounds_once_at_each_segment_size():
    """A clip calls the seam at each segment's row count in turn."""
    for rows in (8, 256, 16, 74):
        x, g, y = _inputs(rows, seed=rows)
        assert mx.array_equal(ops.gated_residual_once(x, g, y),
                              _rounded_once(x, g, y)), rows
    x, g, y = (v.astype(mx.float32) for v in _inputs(32))
    assert mx.array_equal(ops.gated_residual_once(x, g, y),
                          ops.gated_residual(x, g, y))


def test_with_compile_off_a_call_holds_one_block_of_float32_copies(
        monkeypatch):
    """Under MLX_DISABLE_COMPILE or mx.disable_compile the float32 copies
    of the whole call would take 8 bytes an element beside its output."""
    monkeypatch.setattr(epilogues, "ONCE_ROWS", 256)
    x, g, y = _inputs(4096, width=1024)
    want = ops.gated_residual_once(x, g, y)
    mx.eval(want)
    mx.synchronize()
    mx.clear_cache()
    base = mx.get_active_memory()
    mx.reset_peak_memory()
    mx.disable_compile()
    try:
        got = ops.gated_residual_once(x, g, y)
        mx.eval(got)
        mx.synchronize()
    finally:
        mx.enable_compile()
    peak = mx.get_peak_memory() - base
    assert mx.array_equal(got, want)
    # The output and one block's copies; all at once it is 5 outputs.
    assert peak < 2 * x.nbytes, (peak, x.nbytes)


def test_with_compile_off_each_block_writes_into_the_one_output(
        monkeypatch):
    """MLX frees what a command buffer used when the buffer's completed
    handler runs, which can be after mx.eval returns. A block update that
    finds the output still held copies the whole output. Over many blocks
    and calls, a handler is late at some block boundary."""
    monkeypatch.setattr(epilogues, "ONCE_ROWS", 16)
    x, g, y = _inputs(4096, width=64)
    want = _rounded_once(x, g, y)
    mx.eval(want)
    peaks = []
    for _ in range(20):
        mx.synchronize()
        mx.clear_cache()
        base = mx.get_active_memory()
        mx.reset_peak_memory()
        mx.disable_compile()
        try:
            got = ops.gated_residual_once(x, g, y)
            mx.eval(got)
            mx.synchronize()
        finally:
            mx.enable_compile()
        peaks.append(mx.get_peak_memory() - base)
        assert mx.array_equal(got, want)
        del got
    assert max(peaks) < 2 * x.nbytes, (max(peaks), x.nbytes)
