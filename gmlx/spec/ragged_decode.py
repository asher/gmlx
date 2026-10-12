"""Mixed-depth batched ragged decode for the base qwen3.5 fallback.

The base ``_qwen3_5_ragged_decode_attention`` (the vendored mlx-vlm
0.6.15 text model in ``gmlx.models.qwen35.language``) requires every row
of a batch to land in the same sdpa-vector plan bucket
(``len(set(plans)) != 1 -> None``, buckets flip at effective lengths
8k/32k/64k). Streams at different session
depths -- the normal continuous-batching case -- then fall back to a per-row
python loop in ``_target_verify_left_padded_attention``: full-depth
``mx.take`` gather + slice + separate SDPA + concat, EVERY decode step.

The bail is unnecessary: both metal kernels partition the PADDED ``k_size``
and mask per-row via ``pads``, so one plan computed from ``k_size`` is
correct for every row (shorter rows just early-exit more blocks). This seam
rebinds the module global with the unified-plan dispatch, which lives in
``gmlx.models.qwen35.attn`` (the owned attention calls it directly).

No loader calls the installer for an MTP target, which is always built
on the owned classes. It is the serve-parity oracle arm of the
owned-forward tests.

Env: GMLX_RAGGED_UNIFIED_PLAN=0 disables (also read per call by the
dispatch, which then keeps the base strict-bucket bail).
"""

from __future__ import annotations

import logging
import os

_log = logging.getLogger(__name__)


def install_unified_ragged_plan() -> None:
    """Rebind qwen3_5 ragged decode with plan-unify fallback. Idempotent.

    Pre-M3 rebinds even when ``GMLX_RAGGED_UNIFIED_PLAN=0``: upstream's own
    ragged kernels launch 1024-thread groups with no ceiling gate, and the
    owned dispatch (which steps aside there) is what keeps that launch off
    the device. The dispatch still honors the flag for the plan choice."""
    from gmlx.models.qwen35.attn import (
        _wide_threadgroups_ok,
        ragged_decode_attention,
    )

    if (os.environ.get("GMLX_RAGGED_UNIFIED_PLAN", "1") == "0"
            and _wide_threadgroups_ok()):
        return
    from gmlx.models.qwen35 import language as _lang

    # Identity check instead of a latch: no attribute stamped onto the
    # owned function, and a test that restores the upstream global gets
    # a clean reinstall on the next call.
    if _lang._qwen3_5_ragged_decode_attention is ragged_decode_attention:
        return

    _lang._qwen3_5_ragged_decode_attention = ragged_decode_attention
    _log.info("unified ragged-plan decode installed (qwen3_5 family)")


def install_pre_m3_ragged_guard() -> None:
    """Rebind the base ragged decode on pre-M3 GPUs purely for the owned
    dispatch's threadgroup gate. Covers the loads that never build the
    owned MTP classes: plain VLM builds. No-op on M3+, where the base
    kernels fit and plain loads keep the base behavior."""
    from gmlx.models.qwen35.attn import _wide_threadgroups_ok

    if _wide_threadgroups_ok():
        return
    install_unified_ragged_plan()
