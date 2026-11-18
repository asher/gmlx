# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Asher Feldman
"""Fused Metal kernels for the key row and the query rows of a decode step.

The ops path builds a key row in three dispatches (the latent's RMSNorm,
rope on the rope half, the join of the two) and the query rows in two
(rope on the rope half, the join with the absorbed latent half). Each is
one dispatch here. Both repeat the arithmetic of ``mx.fast.rms_norm`` and
``mx.fast.rope`` step for step, so the rows carry the ops path's bits;
``tests/models/test_hy_v4_flash_row_kernels.py`` holds them to that.

``GMLX_HY4_ROW_KERNEL=0`` restores the ops path.
"""

import ctypes
import functools
import math

import mlx.core as mx
import mlx.nn as nn

from gmlx.envflags import env_bool

# Rows per call: a decode step or an MTP verify block.
MAX_ROWS = 8
# The latent width fills a threadgroup of 128 at four values a thread, and
# the rope half's 32 pairs take the first 32 threads.
LATENT = 512
ROPE = 64

_KV_ROW = """
        uint tid  = thread_position_in_threadgroup.x;
        uint row  = threadgroup_position_in_grid.x;
        uint lane = thread_index_in_simdgroup;
        uint sg   = simdgroup_index_in_threadgroup;

        constexpr int W = R + P;

        const device T* xr = x_in + row * W;
        device T* orow = out + row * W;

        threadgroup float sums[32];
        threadgroup float inv_sh[1];

        // mx.fast.rms_norm's reduction: four values a thread, a sum per
        // simdgroup, then a sum of those.
        float acc = 0;
        for (int i = 0; i < 4; i++) {
            float xi = xr[tid * 4 + i];
            acc += xi * xi;
        }
        acc = simd_sum(acc);
        if (sg == 0) {
            sums[lane] = 0;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lane == 0) {
            sums[sg] = acc;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0) {
            acc = simd_sum(sums[lane]);
            if (lane == 0) {
                inv_sh[0] = metal::precise::rsqrt(acc / uint(R) + consts[1]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int i = 0; i < 4; i++) {
            orow[tid * 4 + i] =
                w[tid * 4 + i] * static_cast<T>(xr[tid * 4 + i] * inv_sh[0]);
        }

        // mx.fast.rope on the rope half, traditional pairs.
        if (tid < P / 2) {
            float d = static_cast<float>(tid) / static_cast<float>(P / 2);
            float inv_freq = metal::exp2(-d * consts[0]);
            float theta = static_cast<float>(pos[row % L]) * inv_freq;
            float costheta = metal::fast::cos(theta);
            float sintheta = metal::fast::sin(theta);
            float x1 = static_cast<float>(xr[R + 2 * tid]);
            float x2 = static_cast<float>(xr[R + 2 * tid + 1]);
            orow[R + 2 * tid] = static_cast<T>(x1 * costheta - x2 * sintheta);
            orow[R + 2 * tid + 1] =
                static_cast<T>(x1 * sintheta + x2 * costheta);
        }
"""

_Q_ROW = """
        uint gid = thread_position_in_grid.x;
        uint row = gid / 32;
        uint t   = gid % 32;
        // emb is [B, H, L, R]; q_in is [B, L, H, QD].
        uint l = row % L;
        uint h = (row / L) % H;
        uint b = row / (L * H);

        constexpr int W = R + P;
        const device T* er = emb + row * R;
        const device T* qr = q_in + ((b * L + l) * H + h) * QD + (QD - P);
        device T* orow = out + row * W;

        for (uint i = t; i < R; i += 32) {
            orow[i] = er[i];
        }

        float d = static_cast<float>(t) / static_cast<float>(P / 2);
        float inv_freq = metal::exp2(-d * consts[0]);
        float theta = static_cast<float>(pos[l]) * inv_freq;
        float costheta = metal::fast::cos(theta);
        float sintheta = metal::fast::sin(theta);
        float x1 = static_cast<float>(qr[2 * t]);
        float x2 = static_cast<float>(qr[2 * t + 1]);
        orow[R + 2 * t] = static_cast<T>(x1 * costheta - x2 * sintheta);
        orow[R + 2 * t + 1] = static_cast<T>(x1 * sintheta + x2 * costheta);
"""


@functools.cache
def _kernels():
    if mx.default_device() != mx.gpu or not mx.metal.is_available():
        return None
    kv = mx.fast.metal_kernel(
        name="hy4_flash_kv_row",
        input_names=["x_in", "w", "pos", "consts"],
        output_names=["out"],
        source=_KV_ROW,
        ensure_row_contiguous=True,
    )
    q = mx.fast.metal_kernel(
        name="hy4_flash_q_row",
        input_names=["emb", "q_in", "pos", "consts"],
        output_names=["out"],
        source=_Q_ROW,
        ensure_row_contiguous=True,
    )
    return kv, q


def _log2f(x: float) -> float:
    """log2 in fp32 as mx.fast.rope takes it."""
    try:
        fn = ctypes.CDLL(None).log2f
        fn.restype = ctypes.c_float
        fn.argtypes = [ctypes.c_float]
        return float(fn(x))
    except (OSError, AttributeError):  # pragma: no cover - libm is there
        return math.log2(x)


@functools.cache
def _consts(theta: float, eps: float) -> mx.array:
    """log2 of the rope base and the norm's eps, as the kernels read them."""
    out = mx.array([_log2f(theta), eps], dtype=mx.float32)
    mx.eval(out)
    return out


def _positions(offset: int, L: int) -> mx.array:
    return mx.array([offset + j for j in range(L)], dtype=mx.int32)


def eligible(x: mx.array, norm, rope, latent: int, rope_dims: int) -> bool:
    """A decode-width call on the GPU in a 16-bit type, with the plain
    traditional rope and a norm weight in the row's type."""
    return (
        x.ndim == 3
        and x.shape[1] <= MAX_ROWS
        and latent == LATENT
        and rope_dims == ROPE
        and x.dtype in (mx.float16, mx.bfloat16)
        and type(rope) is nn.RoPE
        and rope.traditional
        and rope.scale == 1.0
        and "weight" in norm
        and norm.weight.dtype == x.dtype
        and mx.default_device() == mx.gpu
        and env_bool("GMLX_HY4_ROW_KERNEL", True)
        and _kernels() is not None
    )


def kv_row(compressed, norm, rope, offset: int) -> mx.array:
    """``compressed`` [B, L, 512 + 64] -> the key rows [B, 1, L, 576]: the
    latent through ``norm``, the rope half rotated at ``offset`` on."""
    B, L, W = compressed.shape
    kernel = _kernels()[0]
    return kernel(
        inputs=[compressed, norm.weight, _positions(offset, L),
                _consts(float(rope.base), float(norm.eps))],
        template=[
            ("T", compressed.dtype),
            ("R", LATENT),
            ("P", ROPE),
            ("L", L),
        ],
        grid=(B * L * 128, 1, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(B, 1, L, W)],
        output_dtypes=[compressed.dtype],
    )[0]


def q_row(emb, q_flat, rope, offset: int) -> mx.array:
    """The query rows [B, H, L, 576]: ``emb`` [B, H, L, 512] beside the
    rope half of ``q_flat`` [B, L, H * QD], rotated at ``offset`` on."""
    B, H, L, R = emb.shape
    qd = q_flat.shape[-1] // H
    kernel = _kernels()[1]
    return kernel(
        inputs=[emb, q_flat, _positions(offset, L),
                _consts(float(rope.base), 0.0)],
        template=[
            ("T", emb.dtype),
            ("R", LATENT),
            ("P", ROPE),
            ("H", H),
            ("L", L),
            ("QD", qd),
        ],
        grid=(B * H * L * 32, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(B, H, L, R + ROPE)],
        output_dtypes=[emb.dtype],
    )[0]
