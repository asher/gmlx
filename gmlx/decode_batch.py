"""Decode concurrency control (GMLX_DECODE_BATCH).

completion_batch_size bounds how many requests decode together in one
BatchGenerator step. Upstream defaults it to 32 and the server never
passes the kwarg, so 32 is hard-wired on the serve path. Measured
aggregate decode throughput saturates far below that on this hardware
class (bandwidth-bound batched attention; see the lab batched-decode
ladder): rows admitted past the knee add latency for every active
request without adding tokens per second. The serve default here is
deliberately small; raise it per deployment when the model and depths
in play still measure gains past it.

Knobs:
    GMLX_DECODE_BATCH   decode concurrency (default 8; 0 = upstream
                        default, currently 32)
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
