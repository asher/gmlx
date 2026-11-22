# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Asher Feldman
"""One-dispatch Metal kernels for two small Inkling 2 row ops.

A decode step runs the L2 row clip about 144 times and the float32 RMS norm
about 154 times. The compiled ops path takes four dispatches per clip (the
reduction splits the chain) and three per norm (the casts on either side).
Each is one dispatch here. The arithmetic follows the ops path; only the
order of the row sum differs, so results match it to float32 rounding.

``GMLX_INKLING2_ROW_KERNEL=0`` restores the ops path.
"""

import functools

import mlx.core as mx

from gmlx.envflags import env_bool

# Rows wider than this take the ops path: the clip keeps a row in registers.
CLIP_MAX = 256
NORM_THREADS = 256

_CLIP = """
        uint lane = thread_index_in_simdgroup;
        uint row = threadgroup_position_in_grid.x * (threads_per_threadgroup.x / 32)
                   + simdgroup_index_in_threadgroup;
        if (row >= N) {
            return;
        }
        uint h = row % H;
        uint l = (row / H) % L;
        uint b = row / (H * L);
        const device T* xr = x + b * x_strides[0] + l * x_strides[1]
                               + h * x_strides[2];
        constexpr int PER = (D + 31) / 32;
        float vals[PER];
        float acc = 0.0f;
        for (int i = 0; i < PER; i++) {
            int d = lane + i * 32;
            float v = 0.0f;
            if (d < D) {
                v = static_cast<float>(xr[d * x_strides[3]]);
                if (SCALED) {
                    v = v * static_cast<float>(scale[h * D + d]);
                }
            }
            vals[i] = v;
            acc += v * v;
        }
        float norm = metal::sqrt(simd_sum(acc)) + 1e-6f;
        float f = metal::clamp(threshold[0] / norm, 0.0f, 1.0f);
        float t = TAU ? tau[(TB == 1 ? 0 : b) * L + l] : 1.0f;
        device O* orow = out + row * D;
        for (int i = 0; i < PER; i++) {
            int d = lane + i * 32;
            if (d < D) {
                float v = vals[i] * f;
                if (TAU) {
                    v = v * t;
                }
                orow[d] = static_cast<O>(v);
            }
        }
"""

_NORM = """
        uint tid = thread_position_in_threadgroup.x;
        uint row = threadgroup_position_in_grid.x;
        uint lane = thread_index_in_simdgroup;
        uint sg = simdgroup_index_in_threadgroup;
        uint n_sg = threads_per_threadgroup.x / 32;
        const device T* xr = x + row * D;
        device O* orow = out + row * D;
        threadgroup float sums[32];
        threadgroup float inv_sh[1];
        float acc = 0.0f;
        for (uint i = tid; i < D; i += threads_per_threadgroup.x) {
            float v = static_cast<float>(xr[i]);
            acc += v * v;
        }
        acc = simd_sum(acc);
        if (sg == 0) {
            sums[lane] = 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lane == 0) {
            sums[sg] = acc;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0) {
            acc = simd_sum(lane < n_sg ? sums[lane] : 0.0f);
            if (lane == 0) {
                inv_sh[0] = metal::precise::rsqrt(acc / float(D) + eps[0]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float inv = inv_sh[0];
        for (uint i = tid; i < D; i += threads_per_threadgroup.x) {
            float v = static_cast<float>(xr[i]) * inv;
            if (WEIGHTED) {
                v = static_cast<float>(w[i]) * v;
            }
            orow[i] = static_cast<O>(v);
        }
"""


@functools.cache
def _kernels():
    clip = mx.fast.metal_kernel(
        name="inkling2_row_clip",
        input_names=["x", "scale", "threshold", "tau"],
        output_names=["out"],
        source=_CLIP,
        ensure_row_contiguous=False,
    )
    norm = mx.fast.metal_kernel(
        name="inkling2_row_norm",
        input_names=["x", "w", "eps"],
        output_names=["out"],
        source=_NORM,
        ensure_row_contiguous=True,
    )
    return clip, norm


@functools.cache
def _scalar(value: float) -> mx.array:
    out = mx.array([value], dtype=mx.float32)
    mx.eval(out)
    return out


def enabled() -> bool:
    """The kernels run: on the GPU, unless the switch turns them off."""
    return (mx.default_device() == mx.gpu and mx.metal.is_available()
            and env_bool("GMLX_INKLING2_ROW_KERNEL", True))


def clip_eligible(x: mx.array, scale, tau) -> bool:
    return (x.ndim == 4 and x.shape[-1] <= CLIP_MAX
            and x.dtype in (mx.float32, mx.float16, mx.bfloat16)
            and (scale is None or scale.shape == x.shape[-2:])
            and (tau is None or (tau.ndim == 4 and tau.shape[2:] == (1, 1)
                                 and tau.shape[1] == x.shape[1]
                                 and tau.shape[0] in (1, x.shape[0])))
            and enabled())


def _clip_kernel(x, scale, threshold, tau, dtype, scaled, tb):
    B, L, H, D = x.shape
    n = B * L * H
    rows_per_group = 4
    return _kernels()[0](
        inputs=[x, scale, _scalar(float(threshold)), tau.reshape(-1)],
        template=[("T", x.dtype), ("O", dtype), ("D", D), ("H", H),
                  ("L", L), ("N", n), ("TB", tb), ("SCALED", scaled),
                  ("TAU", tb > 0)],
        grid=(-(-n // rows_per_group) * rows_per_group * 32, 1, 1),
        threadgroup=(rows_per_group * 32, 1, 1),
        output_shapes=[x.shape],
        output_dtypes=[dtype],
    )[0]


@functools.lru_cache(maxsize=None)
def _clip_fn(ops, threshold: float, dtype, scaled: bool, tb: int):
    # A kernel has no VJP: training differentiates the ops path instead.
    @mx.custom_function
    def run(x, scale, tau):
        return _clip_kernel(x, scale, threshold, tau, dtype, scaled, tb)

    @run.vjp
    def run_vjp(primals, cotangent, _output):
        x, scale, tau = primals
        _, grads = mx.vjp(
            lambda x, s, t: ops(x, s if scaled else None,
                                t if tb else None),
            [x, scale, tau], [cotangent])
        return tuple(grads)

    return run


def clip(x: mx.array, scale, threshold: float, tau, dtype, ops) -> mx.array:
    """``model.clip`` in one dispatch: ``x`` [B, L, H, D]; ``scale``
    [H, D] or None; ``tau`` [B or 1, L, 1, 1] or None. ``ops(x, scale,
    tau)`` is the ops path, which gives the gradient."""
    tb = 0 if tau is None else tau.shape[0]
    run = _clip_fn(ops, float(threshold), dtype, scale is not None, tb)
    return run(x, _scalar(1.0) if scale is None else scale,
               _scalar(1.0) if tau is None else tau)


def norm_eligible(x: mx.array, weight) -> bool:
    return (x.ndim >= 1 and x.shape[-1] >= NORM_THREADS
            and x.dtype in (mx.float32, mx.float16, mx.bfloat16)
            and (weight is None or weight.shape == x.shape[-1:])
            and enabled())


def _norm_kernel(x, weight, eps, dtype, weighted):
    D = x.shape[-1]
    rows = x.size // D
    return _kernels()[1](
        inputs=[x, weight, _scalar(float(eps))],
        template=[("T", x.dtype), ("O", dtype), ("D", D),
                  ("WEIGHTED", weighted)],
        grid=(rows * NORM_THREADS, 1, 1),
        threadgroup=(NORM_THREADS, 1, 1),
        output_shapes=[x.shape],
        output_dtypes=[dtype],
    )[0]


@functools.lru_cache(maxsize=None)
def _norm_fn(eps: float, dtype, weighted: bool):
    @mx.custom_function
    def run(x, weight):
        return _norm_kernel(x, weight, eps, dtype, weighted)

    @run.vjp
    def run_vjp(primals, cotangent, _output):
        x, weight = primals

        def ops(x, w):
            y = mx.fast.rms_norm(x.astype(mx.float32),
                                 w.astype(mx.float32) if weighted else None,
                                 eps)
            return y.astype(dtype)

        _, grads = mx.vjp(ops, [x, weight], [cotangent])
        return tuple(grads)

    return run


def rms_norm(x: mx.array, weight, eps: float, dtype) -> mx.array:
    """``mx.fast.rms_norm`` of ``x`` in float32, times the optional
    ``weight``, cast to ``dtype``, in one dispatch."""
    run = _norm_fn(float(eps), dtype, weight is not None)
    return run(x, _scalar(1.0) if weight is None else weight)
