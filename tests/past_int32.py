"""Inputs, checks and gating for the tests of a pad or a concatenate whose
output passes 2^31 elements (gmlx.mlx_guards).

Each input element holds a bit pattern from a formula of its row r, its
token s and its column d, a finite and nonzero bfloat16. An element left
unwritten, or written at another element's place, then differs from the
formula. The checks compare spans of far fewer than 2^31 elements.

Each test holds several GiB, so the modules take MARKS: the nightly
marker, a skip without a real Apple GPU or with a working set under
FLOOR_GIB, and the settled fixture, which runs one of these tests at a
time across xdist workers. The routing tests in tests/test_mlx_guards.py
hold the guards on every run.
"""
from __future__ import annotations

import fcntl
import gc
import os
import sys
import tempfile

import mlx.core as mx
import pytest

import host_box
from helpers import _real_apple_gpu

# The largest MLX peak of these tests, measured on an M5 Max: the batch
# extend. The floor is twice that, so the rest of a run fits beside it.
PEAK_GIB = 13.8
FLOOR_GIB = 2 * PEAK_GIB
_LOCK = os.path.join(tempfile.gettempdir(), "gmlx-past-int32.lock")


def _working_set_gib() -> float:
    """The real GPU's recommended working set in GiB, 0 without one."""
    try:
        info = host_box.real_device_info()
        return info.get("max_recommended_working_set_size", 0) / 2**30
    except Exception:
        return 0.0


@pytest.fixture
def settled():
    """Holds a lock file every pytest process on this machine shares, so
    one of these tests runs at a time. No freed buffer stays in MLX's
    cache, so a test's memory is back before the lock passes on."""
    with open(_LOCK, "w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            mx.synchronize()
            mx.set_cache_limit(0)
            mx.clear_cache()
            yield
            # A failure's traceback holds the test's arrays until the
            # worker's next test, past the lock.
            for name in ("last_type", "last_value", "last_traceback", "last_exc"):
                if hasattr(sys, name):
                    delattr(sys, name)
            gc.collect()
            mx.synchronize()
            mx.clear_cache()
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


_WS = _working_set_gib()
MARKS = [
    pytest.mark.nightly,
    pytest.mark.skipif(not _real_apple_gpu(),
                       reason="the stock pad and concatenate defect is on GPU streams only"),
    pytest.mark.skipif(_WS < FLOOR_GIB, reason=(
        f"needs a recommended GPU working set of {FLOOR_GIB:.1f} GiB, twice the "
        f"{PEAK_GIB} GiB peak of the largest of these tests; this GPU has {_WS:.1f} GiB")),
    pytest.mark.usefixtures("settled"),
]


def pattern(r0: int, rows: int, s0: int, tokens: int, width: int,
            salt: int = 0) -> mx.array:
    """The uint16 bit patterns of rows r0.. and tokens s0.., as
    [rows, tokens, width]. Only this array is written at full size."""
    r = mx.arange(r0, r0 + rows)[:, None]
    s = mx.arange(s0, s0 + tokens)[None, :]
    hi = ((r * 37 + s * 31 + salt) % 126 + 1).astype(mx.uint16) * 256
    lo = ((mx.arange(width)[None, :] + 3 * r + salt) % 256).astype(mx.uint16)
    return hi[:, :, None] + lo[:, None, :]


def bf16(shape: tuple, *, r0: int = 0, s0: int = 0, salt: int = 0) -> mx.array:
    """``pattern`` as an evaluated bfloat16 array of ``shape``, its last
    two axes the tokens and the width and the others the rows."""
    rows = 1
    for n in shape[:-2]:
        rows *= n
    a = pattern(r0, rows, s0, shape[-2], shape[-1], salt).reshape(shape)
    a = a.view(mx.bfloat16)
    mx.eval(a)
    return a


def placed(r: int, s0: int, n: int, width: int, lo: int, hi: int,
           shift: int = 0, salt: int = 0) -> mx.array:
    """Tokens s0..s0+n of row r of an output that holds the pattern of row
    r at tokens lo..hi, token s from input token s - shift, and zeros at
    every other token, as uint16 [n, width]."""
    p = pattern(r, 1, s0 - shift, n, width, salt)[0]
    s = mx.arange(s0, s0 + n)[:, None]
    return mx.where((s >= lo) & (s < hi), p, mx.zeros_like(p))


def wrong(a: mx.array, shape3: tuple, expect,
          chunk: int = 1 << 26) -> dict[tuple[int, int], int]:
    """{(row, first token of a span): elements that differ} of ``a``
    viewed as uint16 ``shape3`` = [rows, tokens, width], against
    ``expect(r, s0, n)``, the uint16 [n, width] tokens s0..s0+n of row r
    should hold."""
    u = a.view(mx.uint16).reshape(shape3)
    rows, tokens, width = shape3
    step = max(1, chunk // width)
    out = {}
    for r in range(rows):
        for s0 in range(0, tokens, step):
            n = min(step, tokens - s0)
            bad = int((u[r, s0:s0 + n] != expect(r, s0, n)).sum().item())
            if bad:
                out[(r, s0)] = bad
    return out
