"""Mixed-depth batched ragged decode for the stock qwen3.5 fallback.

Stock ``_qwen3_5_ragged_decode_attention`` requires every row of a batch to
land in the same sdpa-vector plan bucket (``len(set(plans)) != 1 -> None``,
buckets flip at effective lengths 8k/32k/64k). Streams at different session
depths -- the normal continuous-batching case -- then fall back to a per-row
python loop in ``_target_verify_left_padded_attention``: full-depth
``mx.take`` gather + slice + separate SDPA + concat, EVERY decode step.

The bail is unnecessary: both metal kernels partition the PADDED ``k_size``
and mask per-row via ``pads``, so one plan computed from ``k_size`` is
correct for every row (shorter rows just early-exit more blocks). This seam
rebinds the module global with the unified-plan dispatch, which lives in
``gmlx.qwen35_attn`` (the owned attention calls it directly).

No production path installs this rebind anymore: the ``GMLX_QWEN_OWNED=0``
fallback runs genuinely stock (strict-bucket bail included). The installer
stays in-tree for the serve-parity oracle arm that owned-forward probes
and tests compose.

Env: GMLX_RAGGED_UNIFIED_PLAN=0 disables (also read per call by the
dispatch, which then keeps the stock strict-bucket bail).
"""

from __future__ import annotations

import logging
import os

_log = logging.getLogger(__name__)

_INSTALLED_FLAG = "_kq_gguf_unified_ragged_plan"


def install_unified_ragged_plan() -> None:
    """Rebind qwen3_5 ragged decode with plan-unify fallback. Idempotent."""
    if os.environ.get("GMLX_RAGGED_UNIFIED_PLAN", "1") == "0":
        return
    from mlx_vlm.models.qwen3_5 import language as _lang

    if getattr(
        _lang._qwen3_5_ragged_decode_attention, _INSTALLED_FLAG, False
    ):
        return

    from .qwen35_attn import ragged_decode_attention

    setattr(ragged_decode_attention, _INSTALLED_FLAG, True)
    _lang._qwen3_5_ragged_decode_attention = ragged_decode_attention
    _log.info("unified ragged-plan decode installed (qwen3_5 family)")
