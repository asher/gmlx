"""Upstream-bug fix: mx.fast.rope int-offset batch corruption at T == 1.

The Metal rope kernel in stock mlx 0.31.2 mis-addresses batch rows when
called with a plain int offset on a single-position input (B, *, 1, D)
with B > 1: row 0 is correct, every later batch row is rotated from
out-of-bounds memory (allocator-dependent garbage, occasionally NaN).
The CPU path is correct, T > 1 is correct, B == 1 is correct, and the
per-row array-offset path is correct -- which is why batched serving
(BatchKVCache passes offset arrays) and single-stream chat never see it,
while any plain-KVCache batched decode chain silently corrupts every
row past the first.

The fix routes the broken case onto the healthy kernel: when offset is
an int, the input is 3-D or 4-D with shape[-2] == 1 and shape[0] > 1,
the offset is expanded to a per-row int32 array of length shape[0].
All other calls pass through untouched, so B == 1 single-stream decode
keeps bit-identical behavior.

Install is idempotent; GMLX_ROPE_BATCH_FIX=0 disables (read per call;
A/B safe). Patched at mx.fast.rope so every arch's rope -- module
classes and direct calls alike -- is covered.
"""

from __future__ import annotations

import mlx.core as mx

from .envflags import env_bool

_installed = False
_orig_rope = None


def _fixed_rope(a, dims, *, traditional, base, scale, offset, freqs=None,
                stream=None):
    if (
        env_bool("GMLX_ROPE_BATCH_FIX", True)
        and type(offset) is int
        and a.ndim in (3, 4)
        and a.shape[-2] == 1
        and a.shape[0] > 1
    ):
        offset = mx.full((a.shape[0],), offset, dtype=mx.int32)
    return _orig_rope(a, dims, traditional=traditional, base=base,
                      scale=scale, offset=offset, freqs=freqs,
                      stream=stream)


def install_rope_batch_fix() -> bool:
    """Wrap mx.fast.rope with the int-offset batch-row fix.
    Idempotent; no-op when GMLX_ROPE_BATCH_FIX=0. Returns True when active."""
    global _orig_rope, _installed
    if not env_bool("GMLX_ROPE_BATCH_FIX", True):
        return False
    if _installed:
        return True
    cur = mx.fast.rope
    _orig_rope = getattr(cur, "_gmlx_orig_rope", cur)
    _fixed_rope._gmlx_orig_rope = _orig_rope
    mx.fast.rope = _fixed_rope
    _installed = True
    return True
