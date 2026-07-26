"""Upstream-bug fix: mx.fast.rope scalar-offset batch corruption at T == 1.

The Metal rope kernel in stock mlx 0.31.2 mis-addresses batch rows when
called with a scalar offset on a single-position input (B, *, 1, D)
with B > 1: row 0 is correct, every later batch row is rotated from
out-of-bounds memory (allocator-dependent garbage, occasionally NaN).
Scalar means fewer offset entries than batch rows: a plain int, a 0-d
mx.array (the ``mx.array(cache.offset)`` wrap in mlx-lm's gemma4_text
produces exactly this), or a size-1 array. The CPU path is correct,
T > 1 is correct, B == 1 is correct, and the full per-row array path is
correct -- which is why batched serving (BatchKVCache passes size-B
offset arrays) and single-stream chat never see it, while any
plain-KVCache batched decode chain silently corrupts every row past
the first.

The fix routes the broken case onto the healthy kernel: when the input
is 3-D or 4-D with shape[-2] == 1 and shape[0] > 1 and the offset
carries fewer entries than shape[0], the offset is expanded to a
per-row int32 array of length shape[0]. All other calls pass through
untouched, so B == 1 single-stream decode keeps bit-identical behavior.

Verification note: the array-offset variants only demonstrate the bug in
a FRESH process. The OOB read lands on the buffer a previous expanded
offset left behind, so an in-process A/B that ran any fixed call first
reads primed memory and looks clean. The regression tripwires therefore
run per-variant subprocesses.

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
        and a.ndim in (3, 4)
        and a.shape[-2] == 1
        and a.shape[0] > 1
        and (type(offset) is int
             or (isinstance(offset, mx.array) and offset.size < a.shape[0]))
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
