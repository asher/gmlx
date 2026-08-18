"""Decode concurrency control (GMLX_DECODE_BATCH).

Bounds how many requests decode together in one batch step. Upstream
hard-wires 32; aggregate throughput saturates well below that at depth
while per-stream latency keeps degrading.

Knobs:
    GMLX_DECODE_BATCH   decode concurrency (default 8; 0 = upstream
                        default)
"""

from __future__ import annotations

import os

DEFAULT_DECODE_BATCH = 8


def _upstream_default() -> int:
    try:
        from mlx_vlm.generate.ar import DEFAULT_COMPLETION_BATCH_SIZE
        return int(DEFAULT_COMPLETION_BATCH_SIZE)
    except Exception:
        return 32


def decode_batch() -> int:
    """Effective serve-path decode concurrency, always positive."""
    raw = os.environ.get("GMLX_DECODE_BATCH", "").strip()
    if raw:
        try:
            v = int(raw)
        except ValueError:
            return DEFAULT_DECODE_BATCH
        if v > 0:
            return v
        if v == 0:
            return _upstream_default()
    return DEFAULT_DECODE_BATCH
