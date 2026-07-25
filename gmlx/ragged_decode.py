"""Mixed-depth batched ragged decode: unify the kernel plan, never bail.

Stock ``_qwen3_5_ragged_decode_attention`` requires every row of a batch to
land in the same sdpa-vector plan bucket (``len(set(plans)) != 1 -> None``,
buckets flip at effective lengths 8k/32k/64k). Streams at different session
depths -- the normal continuous-batching case -- then fall back to a per-row
python loop in ``_target_verify_left_padded_attention``: full-depth
``mx.take`` gather + slice + separate SDPA + concat, EVERY decode step.

The bail is unnecessary: both metal kernels partition the PADDED ``k_size``
and mask per-row via ``pads``, so one plan computed from ``k_size`` is
correct for every row (shorter rows just early-exit more blocks). This seam
rebinds the module global with a version that falls back to the
``k_size``-derived plan when the per-row buckets disagree, keeping the
per-row plan (identical to stock) when they agree.

Env: GMLX_RAGGED_UNIFIED_PLAN=0 disables.
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional

import mlx.core as mx

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

    _plan = _lang._qwen3_5_sdpa_vector_plan
    _one_pass = _lang._qwen3_5_ragged_sdpa_one_pass_kernel
    _two_pass_1 = _lang._qwen3_5_ragged_sdpa_two_pass_1_kernel
    _two_pass_2 = _lang._qwen3_5_ragged_sdpa_two_pass_2_kernel
    _i32 = _lang._qwen3_5_cached_i32_array
    _scalars = _lang._qwen3_5_cached_sdpa_scalars

    def _ragged_decode_attention(
        queries: mx.array,
        keys: mx.array,
        values: mx.array,
        pads: List[int],
        scale: float,
    ) -> Optional[mx.array]:
        if not mx.metal.is_available():
            return None
        if (
            queries.ndim != 4
            or keys.ndim != 4
            or values.ndim != 4
            or queries.shape[2] != 1
            or queries.dtype not in (mx.bfloat16, mx.float16)
            or keys.dtype != queries.dtype
            or values.dtype != queries.dtype
        ):
            return None

        batch, q_heads, _, d_size = queries.shape
        pads = tuple(int(p) for p in pads)
        if len(pads) != batch or any(p < 0 for p in pads):
            return None
        kv_heads = keys.shape[1]
        k_size = keys.shape[2]
        v_size = values.shape[-1]
        if (
            q_heads % kv_heads != 0
            or d_size != v_size
            or d_size not in (64, 96, 128, 256)
            or any(p >= k_size for p in pads)
        ):
            return None

        plans = {_plan(k_size - pad, q_heads, kv_heads) for pad in pads}
        if len(plans) == 1:
            mode, blocks = next(iter(plans))
        else:
            # Rows straddle plan buckets: both kernels partition the padded
            # k_size and mask per-row via pads, so the k_size-derived plan
            # is valid for every row.
            mode, blocks = _plan(k_size, q_heads, kv_heads)

        queries = mx.contiguous(queries)
        keys = mx.contiguous(keys)
        values = mx.contiguous(values)
        pads_array = _i32(pads)
        scale_array, k_size_array = _scalars(float(scale), int(k_size))
        template = [
            ("T", queries.dtype),
            ("D_SIZE", int(d_size)),
            ("V_SIZE", int(v_size)),
            ("NUM_Q_HEADS", int(q_heads)),
            ("NUM_KV_HEADS", int(kv_heads)),
            ("GQA_FACTOR", int(q_heads // kv_heads)),
        ]

        if mode == "one_pass":
            kernel = _one_pass(queries.dtype, d_size, v_size)
            return kernel(
                inputs=[
                    queries, keys, values, pads_array, scale_array,
                    k_size_array,
                ],
                template=template,
                grid=(1024, batch * q_heads, 1),
                threadgroup=(1024, 1, 1),
                output_shapes=[(batch, q_heads, 1, v_size)],
                output_dtypes=[queries.dtype],
            )[0]

        kernel_1 = _two_pass_1(queries.dtype, d_size, v_size, blocks)
        partials, sums, maxs = kernel_1(
            inputs=[
                queries, keys, values, pads_array, scale_array, k_size_array,
            ],
            template=[*template, ("BLOCKS", int(blocks))],
            grid=(32 * kv_heads, (q_heads // kv_heads) * batch, blocks),
            threadgroup=(32, q_heads // kv_heads, 1),
            output_shapes=[
                (batch, q_heads, 1, blocks, v_size),
                (batch, q_heads, 1, blocks),
                (batch, q_heads, 1, blocks),
            ],
            output_dtypes=[queries.dtype, mx.float32, mx.float32],
        )
        kernel_2 = _two_pass_2(queries.dtype, v_size, blocks)
        return kernel_2(
            inputs=[partials, sums, maxs],
            template=[
                ("T", queries.dtype),
                ("D_SIZE", int(v_size)),
                ("BLOCKS", int(blocks)),
            ],
            grid=(1024, batch * q_heads, 1),
            threadgroup=(1024, 1, 1),
            output_shapes=[(batch, q_heads, 1, v_size)],
            output_dtypes=[queries.dtype],
        )[0]

    setattr(_ragged_decode_attention, _INSTALLED_FLAG, True)
    _lang._qwen3_5_ragged_decode_attention = _ragged_decode_attention
    _log.info("unified ragged-plan decode installed (qwen3_5 family)")
