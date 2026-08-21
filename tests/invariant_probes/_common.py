"""Shared pieces for the pinned-invariant probes.

Each probe runs as its own subprocess, one arm per process: several
outcomes under test are process-fatal (GPU-timeout aborts) or
permanent hangs, so isolation and external timeouts are part of the
methodology, not hygiene. Known-answer graphs throughout: a wrong
value is a failure even when the process lives.

The oversized allocation that trips the allocator is sized from the
device's reported max buffer length, so the throw happens on any box
rather than only where a hard-coded shape exceeds the limit. The
trip tensor depends on the survivor chain, so the survivors are
encoded before the throw by topological order, not traversal luck.
"""

import mlx.core as mx

N = 2048
MID_VAL = 1.0 * 1.0001 + 1.0  # the survivor's per-element value


def oversized_rows() -> int:
    """Rows r such that an (r, N, N) float32 buffer exceeds the
    device max buffer length."""
    limit = mx.device_info()["max_buffer_length"]
    per_row = N * N * 4
    return int(limit * 1.2) // per_row + 1


def survivor_and_trip():
    """Build (survivor, trip) on the CURRENT default stream.

    survivor is a small (N,) array two ops deep; trip is an oversized
    tensor computed FROM it, so evaluating trip encodes the survivor
    first and then throws at the oversized output allocation. The
    materializer must be mx.contiguous: binary ops with a scalar give
    the output the broadcast input's strides and allocate only its
    data_size, so they never reach the oversized malloc.
    """
    x = mx.full((N,), 1.0)
    mid = x * 1.0001 + 1.0
    big = mx.contiguous(mx.broadcast_to(mid, (oversized_rows(), N, N)))
    return mid, big


def check(got: float, want: float) -> str:
    ok = abs(got - want) < 1e-3 * abs(want)
    return "CLEAN" if ok else "WRONG"
