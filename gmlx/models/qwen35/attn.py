"""Owned full-attention forward for qwen3.5/3.6 MTP targets.

Subclasses mlx-vlm's ``Qwen3_5Attention``; the owned constructors
build it directly (``rebind_attn`` arms stock-built trees in tests).
The owned ``__call__`` composes three formerly patch-installed routes:

- ragged left-padded S=1 decode through the in-tree ragged SDPA
  kernels, with the unified-plan fallback for rows straddling plan
  buckets;
- MTP verify: single masked-SDPA call for left-padded batches, one
  "causal" call for B=1 and unpadded batches (quantized KV included),
  per-row "causal" fold for deep batched verify widths;
- the stock per-pad-group loop as the fallback.

The ragged kernel sources, plan/block tables, and cached-array helpers
are verbatim in-tree copies, equality-tested against the pinned
mlx-vlm release. Projections go through ``qwen35_verify_linear``, the
rotary apply through ``qwen35_rope``; the quantized-KV routing calls
the base module symbol at call time so the batch-mask fix stays
engaged under its own ledger pin.

Owned loads never read the qwen3_5 module globals; the
``GMLX_QWEN_OWNED=0`` fallback runs stock, and the patch modules stay
in-tree as test oracles. ``GMLX_QWEN35_VERIFY_FOLD=0``,
``GMLX_VERIFY_RAGGED_MASK=0`` and ``GMLX_RAGGED_UNIFIED_PLAN=0``
disable the corresponding routes, read per call.
"""

from functools import lru_cache
from typing import Any, Optional

import mlx.core as mx

from mlx_vlm.models import base as _B
from mlx_vlm.models.qwen3_5 import language as _L

import gmlx.load.loadlog as loadlog
from gmlx.envflags import env_bool
from .gdn import owned_gdn_active as owned_attn_active  # one switch
from .owned import _qwen3_5_left_padding_info
from .rope import apply_multimodal_rotary_pos_emb as _apply_mrope
from .rope import rope_apply_rotary
from .verify_fold import _pads_list
from .verify_linear import verify_linear, verify_linears

__all__ = ["owned_attn_active", "rebind_attn", "OwnedQwen3_5Attention"]


# --- verbatim upstream copies (source-equality-tested; see
#     tests/models/test_qwen35_attn.py) ---

_QWEN3_5_RAGGED_SDPA_ONE_PASS_SOURCE = r"""
    uint q_batch_head_idx = threadgroup_position_in_grid.y;
    uint simd_gid = simdgroup_index_in_threadgroup;
    uint simd_lid = thread_index_in_simdgroup;

    constexpr int BN = 32;
    constexpr int BD = 32;
    constexpr int qk_per_thread = D_SIZE / BD;
    constexpr int v_per_thread = V_SIZE / BD;

    typedef float U;
    thread U q[qk_per_thread];
    thread U k[qk_per_thread];
    thread U o[v_per_thread];
    threadgroup U outputs[BN * BD];
    threadgroup U max_scores[BN];
    threadgroup U sum_exp_scores[BN];

    int K_SIZE = int(k_size[0]);
    int batch_idx = int(q_batch_head_idx) / NUM_Q_HEADS;
    int q_head_idx = int(q_batch_head_idx) - batch_idx * NUM_Q_HEADS;
    int kv_head_idx = q_head_idx / GQA_FACTOR;
    int pad = int(pads[batch_idx]);
    int N = K_SIZE - pad;

    const device T* qptr =
        queries + int(q_batch_head_idx) * D_SIZE + int(simd_lid) * qk_per_thread;
    const device T* kptr =
        keys + (batch_idx * NUM_KV_HEADS + kv_head_idx) * K_SIZE * D_SIZE +
        (pad + int(simd_gid)) * D_SIZE + int(simd_lid) * qk_per_thread;
    const device T* vptr =
        values + (batch_idx * NUM_KV_HEADS + kv_head_idx) * K_SIZE * V_SIZE +
        (pad + int(simd_gid)) * V_SIZE + int(simd_lid) * v_per_thread;
    device T* optr =
        out + int(q_batch_head_idx) * V_SIZE + int(simd_gid) * v_per_thread;

    U s = U(scale[0]);
    for (int i = 0; i < qk_per_thread; i++) {
        q[i] = s * qptr[i];
    }
    for (int i = 0; i < v_per_thread; i++) {
        o[i] = 0;
    }

    U max_score = -3.4028234663852886e38f;
    U sum_exp_score = 0;

    for (int i = int(simd_gid); i < N; i += BN) {
        for (int j = 0; j < qk_per_thread; j++) {
            k[j] = kptr[j];
        }

        U score = 0;
        for (int j = 0; j < qk_per_thread; j++) {
            score += q[j] * k[j];
        }
        score = simd_sum(score);

        U new_max = max(max_score, score);
        U factor = fast::exp(max_score - new_max);
        U exp_score = fast::exp(score - new_max);

        max_score = new_max;
        sum_exp_score = sum_exp_score * factor + exp_score;

        for (int j = 0; j < v_per_thread; j++) {
            o[j] = o[j] * factor + exp_score * vptr[j];
        }

        kptr += BN * D_SIZE;
        vptr += BN * V_SIZE;
    }

    if (simd_lid == 0) {
        max_scores[simd_gid] = max_score;
        sum_exp_scores[simd_gid] = sum_exp_score;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    max_score = max_scores[simd_lid];
    U new_max = simd_max(max_score);
    U factor = fast::exp(max_score - new_max);
    sum_exp_score = simd_sum(sum_exp_scores[simd_lid] * factor);

    for (int i = 0; i < v_per_thread; i++) {
        outputs[simd_lid * BD + simd_gid] = o[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        o[i] = simd_sum(outputs[simd_gid * BD + simd_lid] * factor);
        o[i] = sum_exp_score == 0 ? o[i] : (o[i] / sum_exp_score);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (simd_lid == 0) {
        for (int i = 0; i < v_per_thread; i++) {
            optr[i] = static_cast<T>(o[i]);
        }
    }
"""


_QWEN3_5_RAGGED_SDPA_TWO_PASS_1_SOURCE = r"""
    uint simd_lid = thread_index_in_simdgroup;
    uint kv_head_idx = threadgroup_position_in_grid.x;
    uint batch_idx = threadgroup_position_in_grid.y;
    uint block_idx = threadgroup_position_in_grid.z;
    uint gqa_idx = thread_position_in_threadgroup.y;

    constexpr int BD = 32;
    constexpr int qk_per_thread = D_SIZE / BD;
    constexpr int v_per_thread = V_SIZE / BD;

    typedef float U;
    thread U q[qk_per_thread];
    thread U o[v_per_thread] = {0};

    int K_SIZE = int(k_size[0]);
    int q_head_idx = int(GQA_FACTOR * kv_head_idx + gqa_idx);
    int q_batch_head_idx = int(batch_idx) * NUM_Q_HEADS + q_head_idx;
    int pad = int(pads[batch_idx]);
    int N = K_SIZE - pad;

    const device T* qptr =
        queries + q_batch_head_idx * D_SIZE + int(simd_lid) * qk_per_thread;
    const device T* kptr =
        keys + (int(batch_idx) * NUM_KV_HEADS + int(kv_head_idx)) *
                   K_SIZE * D_SIZE +
        (pad + int(block_idx)) * D_SIZE + int(simd_lid) * qk_per_thread;
    const device T* vptr =
        values + (int(batch_idx) * NUM_KV_HEADS + int(kv_head_idx)) *
                     K_SIZE * V_SIZE +
        (pad + int(block_idx)) * V_SIZE + int(simd_lid) * v_per_thread;
    device T* optr =
        partials + q_batch_head_idx * BLOCKS * V_SIZE +
        int(block_idx) * V_SIZE + int(simd_lid) * v_per_thread;
    device float* sump =
        sums + q_batch_head_idx * BLOCKS + int(block_idx);
    device float* maxp =
        maxs + q_batch_head_idx * BLOCKS + int(block_idx);

    U s = U(scale[0]);
    for (int i = 0; i < qk_per_thread; i++) {
        q[i] = s * qptr[i];
    }

    U max_score = -3.4028234663852886e38f;
    U sum_exp_score = 0;

    for (int i = int(block_idx); i < N; i += BLOCKS) {
        U score = 0;
        for (int j = 0; j < qk_per_thread; j++) {
            score += q[j] * kptr[j];
        }
        score = simd_sum(score);

        U new_max = max(max_score, score);
        U factor = fast::exp(max_score - new_max);
        U exp_score = fast::exp(score - new_max);

        max_score = new_max;
        sum_exp_score = sum_exp_score * factor + exp_score;

        for (int j = 0; j < v_per_thread; j++) {
            o[j] = o[j] * factor + exp_score * vptr[j];
        }

        kptr += BLOCKS * D_SIZE;
        vptr += BLOCKS * V_SIZE;
    }

    if (simd_lid == 0) {
        sump[0] = sum_exp_score;
        maxp[0] = max_score;
    }
    for (int i = 0; i < v_per_thread; i++) {
        optr[i] = static_cast<T>(o[i]);
    }
"""


_QWEN3_5_RAGGED_SDPA_TWO_PASS_2_SOURCE = r"""
    uint q_batch_head_idx = threadgroup_position_in_grid.y;
    uint simd_gid = simdgroup_index_in_threadgroup;
    uint simd_lid = thread_index_in_simdgroup;

    constexpr int BN = 32;
    constexpr int BD = 32;
    constexpr int elem_per_thread = D_SIZE / BD;

    typedef float U;
    thread U o[elem_per_thread] = {0};
    threadgroup U outputs[BN * BD];

    const device T* part =
        partials + int(q_batch_head_idx) * BLOCKS * D_SIZE +
        int(simd_gid) * D_SIZE + int(simd_lid) * elem_per_thread;
    const device float* sump = sums + int(q_batch_head_idx) * BLOCKS;
    const device float* maxp = maxs + int(q_batch_head_idx) * BLOCKS;
    device T* optr =
        out + int(q_batch_head_idx) * D_SIZE + int(simd_gid) * elem_per_thread;

    U sum_exp_score = 0.0;
    U max_score = -3.4028234663852886e38f;

    for (int b = 0; b < BLOCKS / BN; ++b) {
        max_score = max(max_score, maxp[int(simd_lid) + BN * b]);
    }
    max_score = simd_max(max_score);

    for (int b = 0; b < BLOCKS / BN; ++b) {
        U factor = fast::exp(maxp[int(simd_lid) + BN * b] - max_score);
        sum_exp_score += factor * sump[int(simd_lid) + BN * b];
    }
    sum_exp_score = simd_sum(sum_exp_score);

    for (int b = 0; b < BLOCKS / BN; ++b) {
        U factor = fast::exp(maxp[int(simd_gid)] - max_score);
        for (int i = 0; i < elem_per_thread; i++) {
            o[i] += factor * static_cast<U>(part[i]);
        }
        maxp += BN;
        sump += BN;
        part += BN * D_SIZE;
    }

    for (int i = 0; i < elem_per_thread; i++) {
        outputs[simd_lid * BD + simd_gid] = o[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        o[i] = simd_sum(outputs[simd_gid * BD + simd_lid]);
        o[i] = sum_exp_score == 0 ? o[i] : (o[i] / sum_exp_score);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (simd_lid == 0) {
        for (int i = 0; i < elem_per_thread; i++) {
            optr[i] = static_cast<T>(o[i]);
        }
    }
"""


@lru_cache(maxsize=1)
def _qwen3_5_device_arch_suffix() -> str:
    info = mx.device_info() if hasattr(mx, "device_info") else mx.metal.device_info()
    return str(info.get("architecture", ""))[-1:]


@lru_cache(maxsize=1)
def _wide_threadgroups_ok() -> bool:
    """Whether this GPU's compiled pipelines take the 1024-thread launches."""
    if env_bool("GMLX_FORCE_RAGGED_SDPA", False):
        return True
    from gmlx.load.dtypes import gpu_arch_gen

    gen = gpu_arch_gen()
    return gen == 0 or gen >= 15


def _qwen3_5_sdpa_vector_blocks(seq_len: int, gqa_factor: int) -> int:
    devc = _qwen3_5_device_arch_suffix()
    n_simds = gqa_factor
    if devc == "s":
        blocks = 64
        if seq_len > 1024 and n_simds > 4:
            if seq_len <= 8192:
                blocks = 128
            elif seq_len <= 32768:
                blocks = 256
            elif seq_len <= 65536:
                blocks = 512
            else:
                blocks = 1024
        return blocks
    if devc == "d":
        blocks = 128
        if n_simds <= 2 and seq_len > 8192:
            blocks = 256
        elif n_simds >= 6:
            if 16384 <= seq_len < 65536:
                blocks = 512
            elif seq_len >= 65536:
                blocks = 1024
        return blocks
    if n_simds >= 4:
        return 64
    return 32


def _qwen3_5_sdpa_vector_plan(seq_len: int, q_heads: int, kv_heads: int):
    devc = _qwen3_5_device_arch_suffix()
    if (devc in {"d", "s"} and seq_len >= 1024) or (
        kv_heads < q_heads and seq_len >= 4096
    ):
        return ("two_pass", _qwen3_5_sdpa_vector_blocks(seq_len, q_heads // kv_heads))
    return ("one_pass", 0)


@lru_cache(maxsize=None)
def _qwen3_5_ragged_sdpa_one_pass_kernel(dtype, d_size, v_size):
    dtype_name = {mx.bfloat16: "bf16", mx.float16: "fp16"}.get(dtype, "unk")
    return mx.fast.metal_kernel(
        name=f"qwen3_5_ragged_sdpa_1p_{dtype_name}_d{d_size}_v{v_size}",
        input_names=["queries", "keys", "values", "pads", "scale", "k_size"],
        output_names=["out"],
        header="#include <metal_simdgroup>\nusing namespace metal;\n",
        source=_QWEN3_5_RAGGED_SDPA_ONE_PASS_SOURCE,
    )


@lru_cache(maxsize=None)
def _qwen3_5_ragged_sdpa_two_pass_1_kernel(dtype, d_size, v_size, blocks):
    dtype_name = {mx.bfloat16: "bf16", mx.float16: "fp16"}.get(dtype, "unk")
    return mx.fast.metal_kernel(
        name=(
            f"qwen3_5_ragged_sdpa_2p1_{dtype_name}_" f"d{d_size}_v{v_size}_b{blocks}"
        ),
        input_names=["queries", "keys", "values", "pads", "scale", "k_size"],
        output_names=["partials", "sums", "maxs"],
        header="#include <metal_simdgroup>\nusing namespace metal;\n",
        source=_QWEN3_5_RAGGED_SDPA_TWO_PASS_1_SOURCE,
    )


@lru_cache(maxsize=None)
def _qwen3_5_ragged_sdpa_two_pass_2_kernel(dtype, v_size, blocks):
    dtype_name = {mx.bfloat16: "bf16", mx.float16: "fp16"}.get(dtype, "unk")
    return mx.fast.metal_kernel(
        name=f"qwen3_5_ragged_sdpa_2p2_{dtype_name}_v{v_size}_b{blocks}",
        input_names=["partials", "sums", "maxs"],
        output_names=["out"],
        header="#include <metal_simdgroup>\nusing namespace metal;\n",
        source=_QWEN3_5_RAGGED_SDPA_TWO_PASS_2_SOURCE,
    )


@lru_cache(maxsize=128)
def _qwen3_5_cached_i32_array(values):
    return mx.array(values, dtype=mx.int32)


@lru_cache(maxsize=128)
def _qwen3_5_cached_sdpa_scalars(scale: float, k_size: int):
    return (
        mx.array([scale], dtype=mx.float32),
        mx.array([k_size], dtype=mx.int32),
    )


def apply_multimodal_rotary_pos_emb(q, k, cos, sin, unqueeze_dim=1):
    return _apply_mrope(q, k, cos, sin, style="interleaved", unsqueeze_dim=unqueeze_dim)


# --- owned dispatch (gmlx-authored) ---


def ragged_decode_attention(queries, keys, values, pads, scale):
    """Ragged left-padded S=1 decode through the in-tree kernels.

    Body matches the upstream dispatch except the plan choice: when the
    per-row effective lengths straddle plan buckets, fall back to the
    plan for the padded k_size (both kernels partition the padded
    k_size and mask per-row via pads, so that plan is correct for every
    row) instead of returning None. GMLX_RAGGED_UNIFIED_PLAN=0 restores
    the strict per-row-bucket bail.
    """
    if not mx.metal.is_available():
        return None
    # Both kernels launch 1024-thread groups, and metal_kernel emits no
    # max_total_threads_per_threadgroup bound, so a pre-M3 pipeline can
    # compile with a lower ceiling (measured 640 on an M1 Max) and the
    # launch throws at eval. None routes the caller to its per-pad-group
    # mx.fast SDPA fallback. GMLX_FORCE_RAGGED_SDPA=1 overrides for A/B.
    if not _wide_threadgroups_ok():
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

    plans = {_qwen3_5_sdpa_vector_plan(k_size - pad, q_heads, kv_heads) for pad in pads}
    if len(plans) == 1:
        mode, blocks = next(iter(plans))
    elif env_bool("GMLX_RAGGED_UNIFIED_PLAN", True):
        mode, blocks = _qwen3_5_sdpa_vector_plan(k_size, q_heads, kv_heads)
    else:
        return None

    queries = mx.contiguous(queries)
    keys = mx.contiguous(keys)
    values = mx.contiguous(values)
    pads_array = _qwen3_5_cached_i32_array(pads)
    scale_array, k_size_array = _qwen3_5_cached_sdpa_scalars(float(scale), int(k_size))
    template = [
        ("T", queries.dtype),
        ("D_SIZE", int(d_size)),
        ("V_SIZE", int(v_size)),
        ("NUM_Q_HEADS", int(q_heads)),
        ("NUM_KV_HEADS", int(kv_heads)),
        ("GQA_FACTOR", int(q_heads // kv_heads)),
    ]

    if mode == "one_pass":
        kernel = _qwen3_5_ragged_sdpa_one_pass_kernel(queries.dtype, d_size, v_size)
        return kernel(
            inputs=[queries, keys, values, pads_array, scale_array, k_size_array],
            template=template,
            grid=(1024, batch * q_heads, 1),
            threadgroup=(1024, 1, 1),
            output_shapes=[(batch, q_heads, 1, v_size)],
            output_dtypes=[queries.dtype],
        )[0]

    kernel_1 = _qwen3_5_ragged_sdpa_two_pass_1_kernel(
        queries.dtype, d_size, v_size, blocks
    )
    partials, sums, maxs = kernel_1(
        inputs=[queries, keys, values, pads_array, scale_array, k_size_array],
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
    kernel_2 = _qwen3_5_ragged_sdpa_two_pass_2_kernel(queries.dtype, v_size, blocks)
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


def _sdpa_dispatch(queries, keys, values, *, cache, scale, mask, sinks=None):
    """Owned SDPA dispatch tail: quantized-KV routing, then mx.fast.

    Mirrors mlx-vlm's base dispatch minus the TurboQuant branches. gmlx
    never builds those caches, but both TurboQuant cache classes set
    ``bits``, so raise before the affine quantized branch misreads
    their packing. The quantized branch resolves the base module symbol
    at call time so the batch-mask fix patch stays engaged (pinned
    under the quantized-SDPA ledger).
    """
    if isinstance(cache, (_B.TurboQuantKVCache, _B.BatchTurboQuantKVCache)):
        raise RuntimeError(
            "TurboQuant KV cache reached the owned qwen3.5 SDPA dispatch; "
            "gmlx does not build these caches and the affine quantized "
            "branch cannot read their packing"
        )
    if hasattr(cache, "bits"):
        if sinks is not None:
            raise ValueError("Quantized SDPA does not support attention sinks.")
        return _B.quantized_scaled_dot_product_attention(
            queries,
            keys,
            values,
            scale=scale,
            mask=mask,
            group_size=cache.group_size,
            bits=cache.bits,
        )
    return mx.fast.scaled_dot_product_attention(
        queries,
        keys,
        values,
        scale=scale,
        mask=mask,
        sinks=sinks,
    )


def _sdpa(queries, keys, values, *, cache, scale, mask, sinks=None):
    """SDPA with the deep batched-verify per-row fold, then the owned tail.

    The fold claims verify-width (2 <= qL <= 8) batched calls on plain
    deep caches and splits per row past each row's pad, so every row is
    a plain bottom-right-causal verify on fused kernels. Everything
    else lands on the owned dispatch tail (quantized-KV routing
    included, through the batch-mask fix patch).
    """
    if not isinstance(keys, (tuple, list)) and (
        sinks is None
        and queries.ndim == 4
        and keys.ndim == 4
        and queries.shape[0] >= 2
        and 2 <= queries.shape[2] <= 8
        and keys.shape[2] >= 4096
        and not hasattr(cache, "bits")
        and env_bool("GMLX_QWEN35_VERIFY_FOLD", True)
    ):
        causal = mask is None or (isinstance(mask, str) and mask == "causal")
        pads = _pads_list(cache)
        if pads is None and causal:
            pads = [0] * queries.shape[0]
        padded = (
            pads is not None
            and isinstance(mask, mx.array)
            and mask.shape[-1] == keys.shape[2]
        )
        if pads is not None and (causal or padded):
            # An array mask on a batch cache is left-pad + causal by
            # construction (BatchKVCache.make_mask); per-row slices past
            # the pad make each row a plain bottom-right-causal verify.
            rows = [
                _sdpa_dispatch(
                    queries[r : r + 1],
                    keys[r : r + 1, :, p:, :],
                    values[r : r + 1, :, p:, :],
                    cache=cache,
                    scale=scale,
                    mask="causal",
                )
                for r, p in enumerate(pads)
            ]
            return mx.concatenate(rows, axis=0)
    return _sdpa_dispatch(
        queries, keys, values, cache=cache, scale=scale, mask=mask, sinks=sinks
    )


def _left_padded_attention(queries, keys, values, *, cache, scale, mask):
    """Stock per-pad-group left-padded attention under owned control.

    Body mirrors upstream's ``_target_verify_left_padded_attention``
    with the ragged dispatch and SDPA routed to the owned copies (the
    memo read uses the owned padding-info builder).
    """
    if hasattr(cache, "bits") or queries.ndim != 4 or keys.ndim != 4:
        return None

    pads = getattr(cache, "_qwen3_5_decode_left_padding", None)
    if pads is None:
        left_padding_info = _qwen3_5_left_padding_info(cache)
        if left_padding_info is None or left_padding_info[1] <= 0:
            return None
        pads = list(left_padding_info[0])
    if max(pads) <= 0:
        return None

    if (
        queries.shape[0] > 1
        and queries.shape[2] == 1
        and env_bool("GMLX_CASCADE_SDPA", True)
    ):
        from gmlx.upstream.cascade_sdpa import _claim as _cascade_claim

        output = _cascade_claim(queries, keys, values, cache, scale,
                                None, None)
        if output is not None:
            return output

    output = ragged_decode_attention(queries, keys, values, pads, scale)
    if output is not None:
        return output

    row_outputs = {}
    for pad in sorted(set(pads)):
        rows = [i for i, row_pad in enumerate(pads) if row_pad == pad]
        row_idx = mx.array(rows, dtype=mx.int32)
        group_queries = mx.take(queries, row_idx, axis=0)
        group_keys = mx.take(keys, row_idx, axis=0)[:, :, pad:, :]
        group_values = mx.take(values, row_idx, axis=0)[:, :, pad:, :]

        if group_queries.shape[2] > 1:
            prefix_len = group_keys.shape[-2] - group_queries.shape[2]
            group_output = mx.concatenate(
                [
                    _sdpa(
                        group_queries[:, :, i : i + 1, :],
                        group_keys[:, :, : prefix_len + i + 1, :],
                        group_values[:, :, : prefix_len + i + 1, :],
                        cache=None,
                        scale=scale,
                        mask=None,
                    )
                    for i in range(group_queries.shape[2])
                ],
                axis=2,
            )
        else:
            group_output = _sdpa(
                group_queries,
                group_keys,
                group_values,
                cache=None,
                scale=scale,
                mask=None,
            )

        for j, row in enumerate(rows):
            row_outputs[row] = group_output[j : j + 1]

    return mx.concatenate([row_outputs[i] for i in range(queries.shape[0])], axis=0)


_ragged_noted = set()


def _note_ragged(route, B, L, T, pads):
    key = (route, B, L)
    if key in _ragged_noted:
        return
    _ragged_noted.add(key)
    # warn, not verbose_print: live verify traffic should never reach
    # this route, so the one-shot must survive quiet serve logs.
    loadlog.warn(f"[verify] ragged-pads route: {route} B={B} S={L} T={T} "
                 f"max_pad={max(pads)}")


def _folded_or_group_attention(queries, keys, values, *, cache, scale, mask):
    """B=1 verify fold, then the group-loop fallback.

    Mirrors the composed inner patch chain: one "causal" call claims
    single-row verify blocks (packed quantized KV included, routed
    through the quantized-SDPA dispatch); array masks and batches fall
    to the per-pad-group loop.
    """
    quant = isinstance(keys, (tuple, list))
    claimed = (
        env_bool("GMLX_QWEN35_VERIFY_FOLD", True)
        and queries.ndim == 4
        and (hasattr(cache, "bits") if quant
             else keys.ndim == 4 and not hasattr(cache, "bits"))
        and queries.shape[0] == 1
        and queries.shape[2] > 1
        and (mask is None or (isinstance(mask, str) and mask == "causal"))
    )
    if claimed:
        return _sdpa(
            queries, keys, values, cache=cache, scale=scale, mask="causal"
        )
    return _left_padded_attention(
        queries, keys, values, cache=cache, scale=scale, mask=mask
    )


def _verify_attention(queries, keys, values, *, cache, scale, mask):
    """Composed verify/left-padded-decode attention resolver.

    Route order mirrors the patch chain the stock fallback installs:
    the batched masked-SDPA claim first (padded L>1 single call,
    unpadded L>1 "causal" call), then the B=1 fold, then the group
    loop. Returns None for unpadded S=1 decode so the caller's general
    path takes it, exactly like the stock helper contract.
    """
    if hasattr(cache, "bits") or queries.ndim != 4 or keys.ndim != 4:
        return _folded_or_group_attention(
            queries, keys, values, cache=cache, scale=scale, mask=mask
        )
    pads = getattr(cache, "_qwen3_5_decode_left_padding", None)
    padded = pads is not None and len(pads) > 0 and max(pads) > 0
    L = queries.shape[-2]
    if (
        L > 1
        and queries.shape[0] > 1
        and env_bool("GMLX_CASCADE_SDPA", True)
    ):
        # Layer masks on this tree are always cache-derived (the owned
        # model call drops any caller mask), so at verify width they
        # encode pad visibility + end-aligned causal -- exactly what
        # the fused cascade op reproduces from cache.left_padding. The
        # claim re-derives pads itself and declines right padding and
        # width-mismatched arrays; None means "causal" on this
        # resolver, so map it before the claim.
        from gmlx.upstream.cascade_sdpa import _claim as _cascade_claim

        output = _cascade_claim(
            queries, keys, values, cache, scale,
            mask if isinstance(mask, mx.array) else "causal", None)
        if output is not None:
            return output
    if padded and (
        L <= 1
        or len(pads) != queries.shape[0]
        or isinstance(mask, mx.array)
        or not env_bool("GMLX_VERIFY_RAGGED_MASK", True)
    ):
        if L > 1:
            _note_ragged("group-loop", queries.shape[0], L,
                         keys.shape[-2], pads)
        return _folded_or_group_attention(
            queries, keys, values, cache=cache, scale=scale, mask=mask
        )
    if L <= 1:
        return None
    if padded:
        _note_ragged("masked-sdpa", queries.shape[0], L,
                     keys.shape[-2], pads)
        T = keys.shape[-2]
        t = mx.arange(T)[None, None, :]
        end = (T - L) + mx.arange(L)[None, :, None]
        pad_arr = mx.array(pads, dtype=mx.int32)[:, None, None]
        sdpa_mask = ((t >= pad_arr) & (t <= end))[:, None, :, :]
    else:
        sdpa_mask = mask if isinstance(mask, mx.array) else "causal"
    return mx.fast.scaled_dot_product_attention(
        queries, keys, values, scale=scale, mask=sdpa_mask
    )


class OwnedQwen3_5Attention(_L.Qwen3_5Attention):
    """Built by the owned constructors (``rebind_attn`` remains for
    arming a stock-built module tree in tests).

    ``__call__`` mirrors upstream's body; the attention dispatch goes
    through the owned resolvers instead of the patched module globals.
    """

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        position_ids: Optional[mx.array] = None,
        position_embeddings: Optional[tuple[mx.array, mx.array]] = None,
        target_verify: bool = False,
    ) -> mx.array:
        B, L, D = x.shape
        q_proj_output, keys, values = verify_linears(
            (self.q_proj, self.k_proj, self.v_proj), x, target_verify
        )
        queries, gate = mx.split(
            q_proj_output.reshape(B, L, self.num_attention_heads, -1), 2, axis=-1
        )
        gate = gate.reshape(B, L, -1)

        queries = self.q_norm(queries).transpose(0, 2, 1, 3)
        keys = self.k_norm(keys.reshape(B, L, self.num_key_value_heads, -1)).transpose(
            0, 2, 1, 3
        )
        values = values.reshape(B, L, self.num_key_value_heads, -1).transpose(
            0, 2, 1, 3
        )

        kv_seq_len = keys.shape[-2]

        if position_ids is None:
            cache_offset = cache.offset
            if isinstance(cache_offset, mx.array) and cache_offset.ndim > 0:
                offsets = mx.maximum(cache_offset[:B], 0)
                kv_seq_len = kv_seq_len + offsets + 1
                position_ids = offsets[:, None] + mx.arange(L)[None, :]
                position_ids = mx.expand_dims(position_ids, axis=0)
                position_ids = mx.tile(position_ids, (3, 1, 1))
            else:
                if isinstance(cache_offset, mx.array):
                    cache_offset = int(cache_offset.item())
                kv_seq_len += cache_offset + 1
                position_ids = mx.arange(cache_offset, cache_offset + L)
                position_ids = mx.expand_dims(position_ids, axis=0)
                position_ids = mx.tile(position_ids, (3, 1, 1))
        else:
            kv_seq_len += cache.offset + 1 if cache is not None else 0

        if position_embeddings is None:
            queries, keys = rope_apply_rotary(
                self.rotary_emb,
                queries,
                keys,
                position_ids,
                unsqueeze_dim=1,
            )
        else:
            cos, sin = position_embeddings
            queries, keys = apply_multimodal_rotary_pos_emb(queries, keys, cos, sin)

        if mask is not None and isinstance(mask, mx.array):
            if (
                cache is not None
                and hasattr(cache, "_idx")
                and hasattr(cache, "left_padding")
            ):
                kv_seq_len = int(cache._idx) + L
            elif isinstance(kv_seq_len, mx.array):
                kv_seq_len = kv_seq_len.max().item()
            mask = mask[..., : int(kv_seq_len)]

        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        left_padded_decode = (
            mask == "left_padded_decode" if isinstance(mask, str) else False
        )
        if left_padded_decode:
            mask = None

        if (target_verify and L > 1) or left_padded_decode:
            output = _verify_attention(
                queries, keys, values, cache=cache, scale=self.scale, mask=mask
            )
        else:
            output = None

        if output is None and target_verify and L > 1:
            prefix_len = keys.shape[-2] - L
            output = mx.concatenate(
                [
                    _sdpa(
                        queries[:, :, i : i + 1, :],
                        keys[:, :, : prefix_len + i + 1, :],
                        values[:, :, : prefix_len + i + 1, :],
                        cache=cache,
                        scale=self.scale,
                        mask=(
                            mask[..., i : i + 1, : prefix_len + i + 1]
                            if isinstance(mask, mx.array) and mask.ndim >= 4
                            else None
                        ),
                    )
                    for i in range(L)
                ],
                axis=2,
            )
        elif output is None:
            output = _sdpa(
                queries, keys, values, cache=cache, scale=self.scale, mask=mask
            )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)

        return verify_linear(
            self.o_proj, output * mx.sigmoid(gate), target_verify
        )


def rebind_attn(model) -> int:
    """Rebind every stock Qwen3_5Attention in ``model`` to the owned
    class. Real loads build owned classes at construction; this is the
    install path for stock-built toy trees in tests. Returns the
    rebind count."""
    lm = getattr(model, "language_model", model)
    n = 0
    for m in lm.modules():
        if type(m) is _L.Qwen3_5Attention:
            m.__class__ = OwnedQwen3_5Attention
            n += 1
    loadlog.verbose_print(f"[build] attention: owned forward on {n} layers")
    return n
