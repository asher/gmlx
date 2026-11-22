# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Asher Feldman
"""Metal kernels for small Inkling 2 row ops.

Per decode step, the ops path runs these chains many times:

- the L2 row clip, about 144 times, four dispatches each;
- the float32 RMS norm, about 154 times, three each (the casts);
- the router, 48 times, six each;
- the depth mix, about 120 times, four each, plus a cast and a copy of
  each query's strided slice (the kernel writes query-major rows in the
  caller's dtype);
- a loop controller, 96 times, seven each;
- the short convolution, 192 times, about four each with its cache copy;
- the attention bias mask, 48 times, about twelve each;
- the output gate, 48 times, two each with the copy of its transposed input.

Here each is one dispatch, and a controller two. The arithmetic follows
the ops path; only the order of a row sum differs, so results match it to
float32 rounding. The router returns the same experts in score order,
where the ops path returns argpartition's order.

``GMLX_INKLING2_ROW_KERNEL=0`` restores the ops path.
"""

import contextlib
import functools
import struct

import mlx.core as mx

from gmlx.envflags import env_bool

# Rows wider than this take the ops path: the clip keeps a row in registers.
CLIP_MAX = 256
NORM_THREADS = 256

_CLIP = """
        uint lane = thread_index_in_simdgroup;
        uint row = threadgroup_position_in_grid.x * (threads_per_threadgroup.x / 32)
                   + simdgroup_index_in_threadgroup;
        const uint L = x_shape[1];
        const uint N = x_shape[0] * L * H;
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


_ROUTER = """
        uint lane = thread_index_in_simdgroup;
        uint row = threadgroup_position_in_grid.x * (threads_per_threadgroup.x / 32)
                   + simdgroup_index_in_threadgroup;
        const uint N = logits_shape[0];
        if (row >= N) {
            return;
        }
        constexpr int PER = E / 32;
        const device float* lr = logits + row * E;
        float score[PER];
        float sel[PER];
        for (int i = 0; i < PER; i++) {
            int e = lane + i * 32;
            float x = lr[e];
            // mx.sigmoid, as MLX's Metal unary op computes it.
            float y = 1 / (1 + metal::precise::exp(metal::abs(x)));
            score[i] = (x < 0) ? y : 1 - y;
            sel[i] = BIASED ? score[i] + bias[e] : score[i];
        }
        float w[K];
        uint picked[K];
        float total = 0.0f;
        for (int r = 0; r < K; r++) {
            float best = -INFINITY;
            uint best_e = E;
            for (int i = 0; i < PER; i++) {
                uint e = lane + i * 32;
                if (sel[i] > best || (sel[i] == best && e < best_e)) {
                    best = sel[i];
                    best_e = e;
                }
            }
            float top = simd_max(best);
            uint chosen = simd_min(best == top ? best_e : uint(E));
            float s = -INFINITY;
            for (int i = 0; i < PER; i++) {
                if (lane + i * 32 == chosen) {
                    s = score[i];
                    sel[i] = -INFINITY;
                }
            }
            s = simd_max(s);
            picked[r] = chosen;
            w[r] = s;
            total += s;
        }
        if (lane == 0) {
            float denom = total + 1e-20f;
            for (int r = 0; r < K; r++) {
                inds[row * K + r] = picked[r];
                weights[row * K + r] = static_cast<O>((w[r] / denom) * scale[0]);
            }
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
    router = mx.fast.metal_kernel(
        name="inkling2_router",
        input_names=["logits", "bias", "scale"],
        output_names=["inds", "weights"],
        source=_ROUTER,
        ensure_row_contiguous=True,
    )
    return clip, norm, router


@functools.cache
def _scalar(value: float) -> mx.array:
    out = mx.array([value], dtype=mx.float32)
    mx.eval(out)
    return out


_held: list = [None]


def _live() -> bool:
    return (mx.default_device() == mx.gpu and mx.metal.is_available()
            and env_bool("GMLX_INKLING2_ROW_KERNEL", True))


def enabled() -> bool:
    """The kernels run: on the GPU, unless the switch turns them off.
    Inside :func:`held` the answer is the one read when it began."""
    on = _held[0]
    return _live() if on is None else on


@contextlib.contextmanager
def held():
    """Read the switch and device once for a forward, which asks
    ``enabled`` several hundred times."""
    saved = _held[0]
    _held[0] = _live()
    try:
        yield
    finally:
        _held[0] = saved


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
                  ("TB", tb), ("SCALED", scaled), ("TAU", tb > 0)],
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


def router_eligible(logits: mx.array, k: int) -> bool:
    E = logits.shape[-1]
    return (E % 32 == 0 and E <= 1024 and 0 < k <= 32 and k <= E
            and enabled())


def _router_kernel(logits, bias, k, scale, dtype, biased):
    E = logits.shape[-1]
    n = logits.size // E
    rows_per_group = 4
    shape = (*logits.shape[:-1], k)
    return _kernels()[2](
        inputs=[logits.astype(mx.float32).reshape(n, E), bias,
                _scalar(float(scale))],
        template=[("O", dtype), ("E", E), ("K", k), ("BIASED", biased)],
        grid=(-(-n // rows_per_group) * rows_per_group * 32, 1, 1),
        threadgroup=(rows_per_group * 32, 1, 1),
        output_shapes=[shape, shape],
        output_dtypes=[mx.uint32, dtype],
    )


@functools.lru_cache(maxsize=None)
def _router_fn(k: int, scale: float, dtype, biased: bool):
    @mx.custom_function
    def run(logits, bias):
        return tuple(_router_kernel(logits, bias, k, scale, dtype, biased))

    @run.vjp
    def run_vjp(primals, cotangents, outputs):
        # The ops path's weights at the kernel's picks give the gradient.
        logits, bias = primals
        inds = outputs[0]

        def weights(logits):
            scores = mx.sigmoid(logits.astype(mx.float32))
            w = mx.take_along_axis(scores, inds, axis=-1)
            w = w / (mx.sum(w, axis=-1, keepdims=True) + 1e-20)
            return (w * scale).astype(dtype)

        _, (g,) = mx.vjp(weights, [logits], [cotangents[1]])
        return g, mx.zeros_like(bias)

    return run


def router(logits: mx.array, bias, k: int, scale: float, dtype):
    """The top-``k`` expert ids of sigmoid routing (selected on score plus
    ``bias``) and their normalized weights times ``scale``, in one
    dispatch. Ids within a row come in score order, not argpartition's."""
    run = _router_fn(k, float(scale), dtype, bias is not None)
    inds, weights = run(logits, _scalar(0.0) if bias is None else bias)
    return mx.stop_gradient(inds), weights


# Depth mix: one threadgroup per row. Pass one sums cur^2 and every query
# dot; pass two writes the softmax-weighted states.
MIX_THREADS = 256
MIX_MAX_STATES = 16
MIX_MAX_QUERIES = 4

_MIX = """
        uint row = threadgroup_position_in_grid.x;
        uint tid = thread_position_in_threadgroup.x;
        uint lane = thread_index_in_simdgroup;
        uint sg = simdgroup_index_in_threadgroup;
        uint n_sg = threads_per_threadgroup.x / 32;
        constexpr int NS = NB + 1;
        constexpr int NA = NQ * NS + 1;
        const uint ROWS = cur_shape[0] * cur_shape[1];
        STATES
        const device T* c = cur + row * D;
        threadgroup float part[8][NA];
        threadgroup float total[NA];
        float acc[NA];
        for (int i = 0; i < NA; i++) {
            acc[i] = 0.0f;
        }
        for (uint d = tid; d < D; d += threads_per_threadgroup.x) {
            float cv = static_cast<float>(c[d]);
            acc[NA - 1] += cv * cv;
            for (int j = 0; j < NQ; j++) {
                float qv = query[j * D + d];
                for (int b = 0; b < NB; b++) {
                    acc[j * NS + b] += qv * s[b][d];
                }
                acc[j * NS + NB] += qv * cv;
            }
        }
        for (int i = 0; i < NA; i++) {
            float v = simd_sum(acc[i]);
            if (lane == 0) {
                part[sg][i] = v;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < NA) {
            float v = 0.0f;
            for (uint g = 0; g < n_sg; g++) {
                v += part[g][tid];
            }
            total[tid] = v;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float inv = metal::precise::rsqrt(total[NA - 1] / float(D) + eps[0]);
        float scale = metal::precise::rsqrt(float(D));
        float p[NQ][NS];
        for (int j = 0; j < NQ; j++) {
            float top = -INFINITY;
            for (int b = 0; b < NS; b++) {
                float v = total[j * NS + b] * scale;
                if (b == NB) {
                    v = v * inv;
                }
                p[j][b] = v;
                top = metal::max(top, v);
            }
            float z = 0.0f;
            for (int b = 0; b < NS; b++) {
                p[j][b] = metal::precise::exp(p[j][b] - top);
                z += p[j][b];
            }
            for (int b = 0; b < NS; b++) {
                p[j][b] = p[j][b] / z;
            }
        }
        // Query-major output: query j's rows are one contiguous block.
        for (uint d = tid; d < D; d += threads_per_threadgroup.x) {
            float cv = static_cast<float>(c[d]) * inv;
            for (int j = 0; j < NQ; j++) {
                float v = p[j][NB] * cv;
                for (int b = 0; b < NB; b++) {
                    v += p[j][b] * s[b][d];
                }
                out[(j * ROWS + row) * D + d] = static_cast<O>(v);
            }
        }
"""


@functools.lru_cache(maxsize=None)
def _mix_kernel_for(n_blocks: int):
    names = [f"b{i}" for i in range(n_blocks)]
    if n_blocks:
        states = ("const device float* s[NB] = {"
                  + ", ".join(f"{n} + row * D" for n in names) + "};")
    else:
        states = "const device float* const* s = nullptr;"
    return mx.fast.metal_kernel(
        name=f"inkling2_depth_mix_{n_blocks}",
        input_names=[*names, "cur", "query", "eps"],
        output_names=["out"],
        source=_MIX.replace("STATES", states),
        ensure_row_contiguous=True,
    )


def mix_eligible(blocks, cur: mx.array, query: mx.array) -> bool:
    D = cur.shape[-1]
    return (cur.ndim == 3 and len(blocks) < MIX_MAX_STATES
            and query.ndim == 2 and query.shape[-1] == D
            and 0 < query.shape[0] <= MIX_MAX_QUERIES
            and cur.dtype in (mx.float32, mx.float16, mx.bfloat16)
            and all(b.dtype == mx.float32 and b.shape == cur.shape
                    for b in blocks)
            and enabled())


def _mix_kernel(blocks, cur, query, eps, dtype):
    B, L, D = cur.shape
    nq = query.shape[0]
    return _mix_kernel_for(len(blocks))(
        inputs=[*blocks, cur, query.astype(mx.float32), _scalar(float(eps))],
        template=[("T", cur.dtype), ("O", dtype), ("D", D),
                  ("NB", len(blocks)), ("NQ", nq)],
        grid=(B * L * MIX_THREADS, 1, 1),
        threadgroup=(MIX_THREADS, 1, 1),
        output_shapes=[(nq, B, L, D)],
        output_dtypes=[dtype],
    )[0]


@functools.lru_cache(maxsize=None)
def _mix_fn(ops, eps: float, n_blocks: int, dtype):
    @mx.custom_function
    def run(*args):
        *blocks, cur, query = args
        return _mix_kernel(blocks, cur, query, eps, dtype)

    @run.vjp
    def run_vjp(primals, cotangent, _output):
        def f(*args):
            *blocks, cur, query = args
            return ops(tuple(blocks), cur, query).transpose(
                2, 0, 1, 3).astype(dtype)

        _, grads = mx.vjp(f, list(primals), [cotangent])
        return tuple(grads)

    return run


def depth_mix(blocks, cur: mx.array, query: mx.array, eps: float, ops,
              dtype=mx.float32):
    """``model._mix_slab`` in one dispatch: ``blocks`` rms-normed float32
    [B, L, D] states, ``cur`` [B, L, D] (normed here), ``query`` [nq, D].
    Returns [nq, B, L, D] in ``dtype``. ``ops(blocks, cur, query)`` is the
    ops path ([B, L, nq, D] float32), which gives the gradient."""
    return _mix_fn(ops, float(eps), len(blocks), dtype)(*blocks, cur, query)


# Loop controller: kernel one runs one threadgroup per rank row r and
# returns h = down [rms(cur); rms(msg)] for every input row; kernel two
# runs one thread per output element and blends.
CTRL_MAX_ROWS = 8

_CTRL_DOWN = """
        uint r = threadgroup_position_in_grid.x;
        uint tid = thread_position_in_threadgroup.x;
        uint lane = thread_index_in_simdgroup;
        uint sg = simdgroup_index_in_threadgroup;
        uint n_sg = threads_per_threadgroup.x / 32;
        constexpr int NA = 4 * MR;
        threadgroup float part[8][NA];
        threadgroup float total[NA];
        float acc[NA];
        for (int i = 0; i < NA; i++) {
            acc[i] = 0.0f;
        }
        const device float* wc = down + r * 2 * D;
        const device float* wm = wc + D;
        for (uint d = tid; d < D; d += threads_per_threadgroup.x) {
            float a = wc[d];
            float b = wm[d];
            for (int m = 0; m < MR; m++) {
                float c = static_cast<float>(cur[m * D + d]);
                float g = static_cast<float>(msg[m * D + d]);
                acc[4 * m] += c * c;
                acc[4 * m + 1] += g * g;
                acc[4 * m + 2] += a * c;
                acc[4 * m + 3] += b * g;
            }
        }
        for (int i = 0; i < NA; i++) {
            float v = simd_sum(acc[i]);
            if (lane == 0) {
                part[sg][i] = v;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < NA) {
            float v = 0.0f;
            for (uint s = 0; s < n_sg; s++) {
                v += part[s][tid];
            }
            total[tid] = v;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < MR) {
            int m = tid;
            float ic = metal::precise::rsqrt(total[4 * m] / float(D) + eps[0]);
            float im = metal::precise::rsqrt(total[4 * m + 1] / float(D) + eps[0]);
            h[m * R + r] = total[4 * m + 2] * ic + total[4 * m + 3] * im;
        }
"""

_CTRL_BLEND = """
        uint i = thread_position_in_grid.x;
        if (i >= MR * D) {
            return;
        }
        uint m = i / D;
        uint d = i % D;
        const device float* hr = h + m * R;
        const device float* gr = gain + d * R;
        const device float* fr = forget + d * R;
        float g = 0.0f;
        float f = 0.0f;
        for (int k = 0; k < R; k++) {
            g += gr[k] * hr[k];
            f += fr[k] * hr[k];
        }
        g = metal::precise::tanh(g);
        f = metal::precise::tanh(f);
        float c = static_cast<float>(cur[i]);
        out[i] = static_cast<T>(c - f * c + g * static_cast<float>(msg[i]));
"""


@functools.cache
def _ctrl_kernels():
    down = mx.fast.metal_kernel(
        name="inkling2_ctrl_down",
        input_names=["cur", "msg", "down", "eps"],
        output_names=["h"],
        source=_CTRL_DOWN,
        ensure_row_contiguous=True,
    )
    blend = mx.fast.metal_kernel(
        name="inkling2_ctrl_blend",
        input_names=["cur", "msg", "h", "gain", "forget"],
        output_names=["out"],
        source=_CTRL_BLEND,
        ensure_row_contiguous=True,
    )
    return down, blend


def control_eligible(cur: mx.array, msg: mx.array, down: mx.array,
                     gain: mx.array, forget: mx.array) -> bool:
    D = cur.shape[-1]
    R = down.shape[0]
    rows = cur.size // max(D, 1)
    return (cur.shape == msg.shape and 0 < rows <= CTRL_MAX_ROWS
            and down.shape == (R, 2 * D) and gain.shape == (D, R)
            and forget.shape == (D, R) and R <= 256
            and all(w.dtype == mx.float32 for w in (down, gain, forget))
            and cur.dtype in (mx.float32, mx.float16, mx.bfloat16)
            and msg.dtype in (mx.float32, mx.float16, mx.bfloat16)
            and enabled())


def _control_kernel(cur, msg, down, gain, forget, eps):
    D = cur.shape[-1]
    R = down.shape[0]
    rows = cur.size // D
    k_down, k_blend = _ctrl_kernels()
    h = k_down(
        inputs=[cur, msg, down, _scalar(float(eps))],
        template=[("D", D), ("R", R), ("MR", rows)],
        grid=(R * NORM_THREADS, 1, 1),
        threadgroup=(NORM_THREADS, 1, 1),
        output_shapes=[(rows, R)],
        output_dtypes=[mx.float32],
    )[0]
    n = rows * D
    return k_blend(
        inputs=[cur, msg, h, gain, forget],
        template=[("T", cur.dtype), ("D", D), ("R", R), ("MR", rows)],
        grid=(-(-n // 256) * 256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[cur.shape],
        output_dtypes=[cur.dtype],
    )[0]


@functools.lru_cache(maxsize=None)
def _control_fn(ops, eps: float):
    @mx.custom_function
    def run(cur, msg, down, gain, forget):
        return _control_kernel(cur, msg, down, gain, forget, eps)

    @run.vjp
    def run_vjp(primals, cotangent, _output):
        _, grads = mx.vjp(ops, list(primals), [cotangent])
        return tuple(grads)

    return run


def control(cur, msg, down, gain, forget, eps: float, ops):
    """A loop controller in two dispatches:
    ``(1 - tanh(F h)) * cur + tanh(G h) * msg`` with
    ``h = D [rms(cur); rms(msg)]``, in ``cur``'s dtype. ``ops`` is the ops
    path, which gives the gradient."""
    return _control_fn(ops, float(eps))(cur, msg, down, gain, forget)


# Short convolution: one thread per (row, window slot, channel). A slot
# copies its input to the window and the kept state, and a slot past the
# kept inputs also writes its output row.
_CONV = """
        uint i = thread_position_in_grid.x;
        constexpr int NK = WIDTH - 1;
        const int B = x_shape[0];
        const int L = x_shape[1];
        const int S = NK + L;
        if (i >= uint(B * S * C)) {
            return;
        }
        uint c = i % C;
        uint j = (i / C) % S;
        uint b = i / (C * S);
        const device float* st = state + b * NK * C;
        const device T* xr = x + b * L * C;
        float v = (j < NK) ? st[j * C + c] : static_cast<float>(xr[(j - NK) * C + c]);
        WINDOW_STORE
        if (int(j) >= L) {
            kept[(b * NK + (j - L)) * C + c] = v;
        }
        if (j < NK) {
            return;
        }
        int l = j - NK;
        // Window slot t + k feeds tap k of output row l; t = l here.
        float out = 0.0f;
        for (int k = 0; k < WIDTH; k++) {
            int s = l + k;
            float wv = (s < NK) ? st[s * C + c] : static_cast<float>(xr[(s - NK) * C + c]);
            float term = wv * w[c * WIDTH + k];
            out = (k == 0) ? term : out + term;
        }
        float scale = sc[0];
        DYN_BEGIN {
            constexpr int HC = C / NH;
            const device auto* tr = dyn + (b * L + l) * NH * WIDTH + (c / HC) * WIDTH;
            for (int lag = 0; lag < WIDTH; lag++) {
                int s = NK - lag + l;
                float pv = (s < NK) ? st[s * C + c] : static_cast<float>(xr[(s - NK) * C + c]);
                out = out + pv * static_cast<float>(tr[lag]);
            }
        }
        out_y[(b * L + l) * C + c] = static_cast<T>((v + out) * scale);
"""


def _strip_dyn(src: str) -> str:
    """``src`` without the dynamic-tap block, which names an input the
    static kernel lacks."""
    start = src.index("        DYN_BEGIN {")
    end = src.index("        out_y[")
    return src[:start] + src[end:]


@functools.lru_cache(maxsize=None)
def _conv_kernel_for(dyn: bool, with_window: bool):
    names = ["state", "x", "w", "sc"] + (["dyn"] if dyn else [])
    outs = ["out_y", "kept"] + (["window"] if with_window else [])
    src = _CONV.replace("DYN_BEGIN", "if (true)") if dyn else _strip_dyn(_CONV)
    return mx.fast.metal_kernel(
        name=f"inkling2_conv_{int(dyn)}{int(with_window)}",
        input_names=names,
        output_names=outs,
        source=src.replace("WINDOW_STORE",
                           "window[i] = v;" if with_window else ""),
        ensure_row_contiguous=True,
    )


def conv_eligible(state, x: mx.array, w: mx.array, dynamic) -> bool:
    if x.ndim != 3 or w.ndim != 2:
        return False
    B, L, C = x.shape
    width = w.shape[1]
    return (state is not None and state.shape == (B, width - 1, C)
            and state.dtype == mx.float32 and w.shape[0] == C
            and w.dtype == mx.float32
            and x.dtype in (mx.float32, mx.float16, mx.bfloat16)
            and (dynamic is None
                 or (dynamic.ndim == 3 and dynamic.shape[:2] == (B, L)
                     and dynamic.shape[-1] % width == 0
                     and C % (dynamic.shape[-1] // width) == 0))
            and enabled())


def _conv_kernel(state, x, w, dynamic, with_window: bool):
    B, L, C = x.shape
    width = w.shape[1]
    dyn = dynamic is not None
    nh = dynamic.shape[-1] // width if dyn else 1
    scale = (2 * width + 1) ** -0.5 if dyn else (width + 1) ** -0.5
    n = B * (width - 1 + L) * C
    shapes = [x.shape, state.shape] + ([(B, width - 1 + L, C)]
                                        if with_window else [])
    return _conv_kernel_for(dyn, with_window)(
        inputs=[state, x, w, _scalar(float(scale))]
        + ([dynamic] if dyn else []),
        template=[("T", x.dtype), ("C", C), ("WIDTH", width), ("NH", nh)],
        grid=(-(-n // 256) * 256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=shapes,
        output_dtypes=[x.dtype, mx.float32] + ([mx.float32]
                                               if with_window else []),
    )


@functools.lru_cache(maxsize=None)
def _conv_fn(ops, dyn: bool, with_window: bool):
    def from_ops(state, x, w, *dynamic):
        window, out = ops(state, x, w, *dynamic)
        kept = window[:, x.shape[1]:]
        return (out, kept, window) if with_window else (out, kept)

    @mx.custom_function
    def run(state, x, w, *dynamic):
        return tuple(_conv_kernel(state, x, w, dynamic[0] if dyn else None,
                                  with_window))

    @run.vjp
    def run_vjp(primals, cotangents, _outputs):
        _, grads = mx.vjp(from_ops, list(primals), list(cotangents))
        return tuple(grads)

    return run


def conv(state, x, w, dynamic, with_window: bool, ops):
    """The short convolution plus its residual, scaled, in ``x``'s dtype;
    the kept state (the last ``width - 1`` window rows, float32); and with
    ``with_window`` the whole float32 window, in one dispatch. ``ops`` is
    ``_conv_static`` or ``_conv_dynamic``, which gives the gradient."""
    args = (state, x, w) + (() if dynamic is None else (dynamic,))
    return _conv_fn(ops, dynamic is not None, with_window)(*args)


# Attention bias mask at decode and verify widths: one thread per mask
# element, the key slot on x and the (batch, head, query) row on y, so no
# thread divides by the run-time key count. The key distance comes from the
# slot (time order, or the ring order of a rotating cache), and the bias
# from a dot with the table row.
MASK_MAX_ROWS = 16

_MASK = """
        int s = thread_position_in_grid.x;
        uint q = thread_position_in_grid.y;
        const int L = r_shape[1];
        int K = params[0];
        if (s >= K) {
            return;
        }
        uint i = q * uint(K) + uint(s);
        int l = q % L;
        int h = (q / L) % H;
        int b = q / (L * H);
        int back = L - 1 - l;
        int d;
        if (RING) {
            // s < K and 1 <= params[1] <= K, so t > -K.
            int t = params[1] - 1 - s;
            d = (t < 0 ? t + K : t) - back;
        } else {
            d = (K - 1 - s) - back;
        }
        bool ok = d >= 0 && (WIN == 0 || d < WIN);
        if (!ok) {
            out[i] = static_cast<O>(-INFINITY);
            return;
        }
        int row = (b * L + l) * H + h;
        float e = 0.0f;
        if (d < EXT) {
            const device auto* rr = r + row * RL;
            const device float* tr = table + d * RL;
            for (int k = 0; k < RL; k++) {
                e += static_cast<float>(rr[k]) * tr[k];
            }
        }
        out[i] = static_cast<O>(e - static_cast<float>(sink[row]));
"""


@functools.lru_cache(maxsize=None)
def _mask_kernel_for(ring: bool):
    return mx.fast.metal_kernel(
        name=f"inkling2_bias_mask_{int(ring)}",
        input_names=["r", "sink", "table", "params"],
        output_names=["out"],
        source=_MASK,
        ensure_row_contiguous=True,
    )


def mask_eligible(r: mx.array, n_keys: int) -> bool:
    return (r.ndim == 4 and r.shape[1] <= MASK_MAX_ROWS
            and r.shape[0] * r.shape[1] * r.shape[2] * n_keys < (1 << 31)
            and enabled())


def _mask_kernel(r, sink, table, params, n_keys, window, ring, dtype):
    B, L, H, RL = r.shape
    return _mask_kernel_for(ring)(
        inputs=[r, sink, table, params],
        template=[("O", dtype), ("H", H), ("RL", RL),
                  ("EXT", table.shape[0] - 1), ("WIN", int(window or 0)),
                  ("RING", ring)],
        grid=(-(-n_keys // 256) * 256, B * H * L, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(B, H, L, n_keys)],
        output_dtypes=[dtype],
    )[0]


def _mask_ops(r, sink, table, params, n_keys, window, ring, dtype):
    """The kernel's mask from ops, for the gradient."""
    L = r.shape[1]
    extent = table.shape[0] - 1
    s = mx.arange(n_keys)[None, :]
    back = (L - 1 - mx.arange(L))[:, None]
    if ring:
        d = mx.remainder(params[1] - 1 - s, n_keys) - back
    else:
        d = (n_keys - 1 - s) - back
    valid = d >= 0
    if window:
        valid = valid & (d < window)
    scores = r.astype(mx.float32) @ table.T - sink[..., None]
    index = mx.where((d >= 0) & (d < extent), d, extent)
    bias = mx.take_along_axis(scores, index[None, :, None, :], axis=-1)
    return mx.where(valid, bias.transpose(0, 2, 1, 3), -mx.inf).astype(dtype)


@functools.lru_cache(maxsize=None)
def _mask_fn(window: int, ring: bool, dtype):
    # ``keys`` only carries the key count, which grows every step; reading
    # it from a shape keeps one function for every length.
    @mx.custom_function
    def run(r, sink, table, params, keys):
        return _mask_kernel(r, sink, table, params, keys.shape[2], window,
                            ring, dtype)

    @run.vjp
    def run_vjp(primals, cotangent, _output):
        r, sink, table, params, keys = primals
        _, grads = mx.vjp(
            lambda r, s, t: _mask_ops(r, s, t, params, keys.shape[2], window,
                                      ring, dtype),
            [r, sink, table], [cotangent])
        return (*grads, mx.zeros_like(params), mx.zeros_like(keys))

    return run


def bias_mask(r, sink, table, keys, window, ring_idx, dtype):
    """[B, H, L, K] additive mask in one dispatch, K the key count of
    ``keys`` [B, H_kv, K, D]: ``R . table[d] - sink`` where the key is
    visible, with the zero pad row past the table, and -inf elsewhere.
    ``d`` runs back from the newest key in time order, or in ring order
    from ``ring_idx`` for a rotating cache."""
    ring = ring_idx is not None
    params = mx.array([keys.shape[2], int(ring_idx) if ring else 0],
                      dtype=mx.int32)
    return _mask_fn(int(window or 0), ring, dtype)(r, sink, table, params,
                                                   keys)


# Attention at decode and verify widths, with the bias built in: pass one
# runs a threadgroup per (batch, KV head, key chunk) over every query row of
# the KV head, so each key and value row is read once. Four key blocks of 32
# are in flight, each on one simdgroup, or on two that split the rows. Lanes
# run over keys for the scores and the bias ``R . table[d] - sink``, over
# the head dim for the value sum, with an online softmax per row. Pass two
# merges the chunks and adds the zero sink logit; one chunk merges in pass
# one. Keys and values are read through their strides, so a cache view is
# not copied.
ATTN_MAX_ROWS = 12
_ATTN_BLOCKS = 4
# Masked scores: finite, so fast math keeps the comparisons.
_ATTN_HEAD = """
constant constexpr float KQ_NEG = -1.0e30f;

// Eight or four 16-bit elements, loaded as one word group, to float.
template <typename T> inline void kq_unpack8(uint4 w, thread float4& a,
                                             thread float4& b);
template <> inline void kq_unpack8<bfloat16_t>(uint4 w, thread float4& a,
                                               thread float4& b) {
    a = float4(as_type<float>(w.x << 16), as_type<float>(w.x & 0xFFFF0000u),
               as_type<float>(w.y << 16), as_type<float>(w.y & 0xFFFF0000u));
    b = float4(as_type<float>(w.z << 16), as_type<float>(w.z & 0xFFFF0000u),
               as_type<float>(w.w << 16), as_type<float>(w.w & 0xFFFF0000u));
}
template <> inline void kq_unpack8<half>(uint4 w, thread float4& a,
                                         thread float4& b) {
    a = float4(float2(as_type<half2>(w.x)), float2(as_type<half2>(w.y)));
    b = float4(float2(as_type<half2>(w.z)), float2(as_type<half2>(w.w)));
}
template <typename T> inline float4 kq_unpack4(uint2 w);
template <> inline float4 kq_unpack4<bfloat16_t>(uint2 w) {
    return float4(as_type<float>(w.x << 16), as_type<float>(w.x & 0xFFFF0000u),
                  as_type<float>(w.y << 16), as_type<float>(w.y & 0xFFFF0000u));
}
template <> inline float4 kq_unpack4<half>(uint2 w) {
    return float4(float2(as_type<half2>(w.x)), float2(as_type<half2>(w.y)));
}
"""

_ATTN = """
        constexpr int NT = 32 * NSG;
        constexpr int D4 = D / 4;
        constexpr int RR = R / RG;
        constexpr int NKB = NSG / RG;
        int lane = thread_position_in_threadgroup.x;
        int sg = thread_position_in_threadgroup.y;
        int tid = sg * 32 + lane;
        int rg = sg % RG;
        int kbi = sg / RG;
        int split = threadgroup_position_in_grid.y;
        int bk = threadgroup_position_in_grid.z;
        int b = bk / KVH;
        int kvh = bk % KVH;
        const int K = params[0];
        const int idx = params[1];
        const int chunk = params[2];
        const int splits = params[3];
        const int c0 = split * chunk;
        const device T* kb = k + b * k_strides[0] + kvh * k_strides[1];
        const device T* vb = v + b * v_strides[0] + kvh * v_strides[1];
        const int64_t ks = k_strides[2];
        const int64_t vs = v_strides[2];
        // Keys and values are read in words: a unit head-dim stride and
        // aligned rows, as cache views have. Other views read nothing and
        // give NaN.
        const bool words = k_strides[3] == 1 && v_strides[3] == 1
            && ks % 8 == 0 && vs % 4 == 0
            && (reinterpret_cast<ulong>(kb) & 15) == 0
            && (reinterpret_cast<ulong>(vb) & 7) == 0;
        const int c1 = words ? min(K, c0 + chunk) : c0;
        const float scale = as_type<float>(params[4]);

        threadgroup float tg[TGN];
        threadgroup float* qs = tg;
        threadgroup float* rs = tg + R * D;
        threadgroup float* ss = rs + R * RL;
        // Row l * G + g is query l of head kvh * G + g.
        for (int i = tid; i < R * D; i += NT) {
            int rr = i / D;
            int dd = i % D;
            int h = kvh * G + rr % G;
            int l = rr / G;
            qs[i] = static_cast<float>(q[b * q_strides[0] + h * q_strides[1]
                                         + l * q_strides[2]
                                         + dd * q_strides[3]]) * scale;
        }
        for (int i = tid; i < R * RL; i += NT) {
            int rr = i / RL;
            int h = kvh * G + rr % G;
            int l = rr / G;
            rs[i] = r[((b * L + l) * H + h) * RL + i % RL];
        }
        for (int i = tid; i < R; i += NT) {
            ss[i] = sink[(b * L + i / G) * H + kvh * G + i % G];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const int r0 = rg * RR;
        float m[RR];
        float den[RR];
        float4 o[RR];
        for (int j = 0; j < RR; j++) {
            m[j] = KQ_NEG;
            den[j] = 0.0f;
            o[j] = float4(0.0f);
        }
        threadgroup float* ps = tg + PSO + sg * RR * 32;
        for (int blk = c0 + kbi * 32; blk < c1; blk += NKB * 32) {
            int s = blk + lane;
            bool in = s < c1;
            float sc[RR];
            for (int j = 0; j < RR; j++) {
                sc[j] = 0.0f;
            }
            if (in) {
                for (int e = 0; e < D4 / 2; e++) {
                    float4 ka, kc;
                    kq_unpack8<T>(
                        ((const device uint4*)(kb + s * ks))[e], ka, kc);
                    for (int j = 0; j < RR; j++) {
                        const threadgroup float4* qr =
                            (const threadgroup float4*)(qs + (r0 + j) * D);
                        sc[j] += dot(qr[2 * e], ka) + dot(qr[2 * e + 1], kc);
                    }
                }
            }
            // Distance of key s back from the newest key, in time or ring order.
            int back0;
            if (RING) {
                int t = idx - 1 - s;
                back0 = t < 0 ? t + K : t;
            } else {
                back0 = K - 1 - s;
            }
            for (int j0 = 0; j0 < RR; j0 += G) {
                int l = (r0 + j0) / G;
                int d = back0 - (L - 1 - l);
                bool ok = in && d >= 0 && (WIN == 0 || d < WIN);
                bool near = ok && d < EXT;
                float tb[RL];
                for (int e = 0; e < RL; e++) {
                    tb[e] = near ? table[d * RL + e] : 0.0f;
                }
                for (int g = 0; g < G; g++) {
                    int j = j0 + g;
                    float bias = 0.0f;
                    for (int e = 0; e < RL; e++) {
                        bias += rs[(r0 + j) * RL + e] * tb[e];
                    }
                    sc[j] = ok ? sc[j] + bias - ss[r0 + j] : KQ_NEG;
                }
            }
            for (int j = 0; j < RR; j++) {
                bool ok = sc[j] > 0.5f * KQ_NEG;
                float mn = max(m[j], simd_max(sc[j]));
                float p = ok ? metal::exp(sc[j] - mn) : 0.0f;
                float corr = metal::exp(m[j] - mn);
                den[j] = den[j] * corr + simd_sum(p);
                o[j] *= corr;
                m[j] = mn;
                ps[j * 32 + lane] = p;
            }
            // The weights go through threadgroup memory, four keys a read;
            // past the last key they are zero and the value row repeats.
            simdgroup_barrier(mem_flags::mem_threadgroup);
            const int last = c1 - 1;
            for (int t = 0; t < 32 && blk + t <= last; t += 4) {
                float4 v0 = kq_unpack4<T>(((const device uint2*)(
                    vb + min(blk + t, last) * vs))[lane]);
                float4 v1 = kq_unpack4<T>(((const device uint2*)(
                    vb + min(blk + t + 1, last) * vs))[lane]);
                float4 v2 = kq_unpack4<T>(((const device uint2*)(
                    vb + min(blk + t + 2, last) * vs))[lane]);
                float4 v3 = kq_unpack4<T>(((const device uint2*)(
                    vb + min(blk + t + 3, last) * vs))[lane]);
                for (int j = 0; j < RR; j++) {
                    float4 p4 = ((const threadgroup float4*)(ps + j * 32))[t / 4];
                    o[j] += p4.x * v0 + p4.y * v1 + p4.z * v2 + p4.w * v3;
                }
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }


        // Merge the key blocks: rows of [NKB][R][DP], the value sum, then
        // the running max and the denominator.
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int j = 0; j < RR; j++) {
            threadgroup float* row = tg + (kbi * R + r0 + j) * DP;
            ((threadgroup float4*)row)[lane] = o[j];
            if (lane == 0) {
                row[D] = m[j];
                row[D + 1] = den[j];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int i = tid; i < R * D; i += NT) {
            int rr = i / D;
            int dd = i % D;
            float mx_ = KQ_NEG;
            for (int j = 0; j < NKB; j++) {
                mx_ = max(mx_, tg[(j * R + rr) * DP + D]);
            }
            float num = 0.0f;
            float dsum = 0.0f;
            for (int j = 0; j < NKB; j++) {
                const threadgroup float* row = tg + (j * R + rr) * DP;
                float w = metal::exp(row[D] - mx_);
                num += w * row[dd];
                dsum += w * row[D + 1];
            }
            num = words ? num : NAN;
            if (FINAL) {
                // The zero sink logit joins the denominator.
                float top = max(mx_, 0.0f);
                float w = metal::exp(mx_ - top);
                int h = kvh * G + rr % G;
                int l = rr / G;
                out[((b * H + h) * L + l) * D + dd] = static_cast<OT>(
                    num * w / (dsum * w + metal::exp(-top)));
            } else {
                device float* pr = (device float*)out
                    + ((bk * splits + split) * R + rr) * DP;
                pr[dd] = num;
                if (dd == 0) {
                    pr[D] = mx_;
                    pr[D + 1] = dsum;
                }
            }
        }
"""

_ATTN_MERGE = """
        int dd = thread_position_in_grid.x;
        int rr = thread_position_in_grid.y;
        int bk = thread_position_in_grid.z;
        const int splits = params[3];
        int b = bk / KVH;
        int kvh = bk % KVH;
        const device float* base = part + (bk * splits * R + rr) * DP;
        float mx_ = KQ_NEG;
        for (int s = 0; s < splits; s++) {
            mx_ = max(mx_, base[s * R * DP + D]);
        }
        // The zero sink logit joins the denominator.
        float top = max(mx_, 0.0f);
        float num = 0.0f;
        float dsum = metal::exp(-top);
        for (int s = 0; s < splits; s++) {
            const device float* pr = base + s * R * DP;
            float w = metal::exp(pr[D] - top);
            num += w * pr[dd];
            dsum += w * pr[D + 1];
        }
        int h = kvh * G + rr % G;
        int l = rr / G;
        out[((b * H + h) * L + l) * D + dd] = static_cast<OT>(num / dsum);
"""


@functools.cache
def _attn_kernels():
    one = mx.fast.metal_kernel(
        name="inkling2_attend",
        input_names=["q", "k", "v", "r", "sink", "table", "params"],
        output_names=["out"],
        source=_ATTN,
        header=_ATTN_HEAD,
        ensure_row_contiguous=False,
    )
    merge = mx.fast.metal_kernel(
        name="inkling2_attend_merge",
        input_names=["part", "params"],
        output_names=["out"],
        source=_ATTN_MERGE,
        header=_ATTN_HEAD,
        ensure_row_contiguous=True,
    )
    return one, merge


def attend_eligible(q: mx.array, k: mx.array, v: mx.array,
                    r: mx.array) -> bool:
    if q.ndim != 4 or k.ndim != 4 or r.ndim != 4:
        return False
    B, H, L, D = q.shape
    KVH = k.shape[1]
    return (D == 128 and k.shape[0] == B and k.shape[3] == D
            and v.shape == k.shape and KVH > 0 and H % KVH == 0
            and (H // KVH) * L <= ATTN_MAX_ROWS and k.shape[2] > 0
            and q.dtype in (mx.float16, mx.bfloat16)
            and k.dtype == q.dtype and v.dtype == q.dtype
            and r.shape[:3] == (B, L, H) and r.dtype == mx.float32
            and enabled())


def _attn_split(n_keys: int, rows: int) -> tuple:
    """(chunk, splits): chunks of one key block per simdgroup pass, at most
    8 passes, and about 80 threadgroups where the keys allow it."""
    unit = 32 * _ATTN_BLOCKS
    want = max(1, -(-80 // rows))
    passes = min(8, max(1, -(-n_keys // (unit * want))))
    chunk = unit * passes
    return chunk, -(-n_keys // chunk)


def _attend_kernel(q, k, v, r, sink, table, params, window, ring):
    B, H, L, D = q.shape
    KVH = k.shape[1]
    _, splits = _attn_split(k.shape[2], B * KVH)
    G = H // KVH
    R = G * L
    RL = r.shape[-1]
    DP = D + 4
    # Two simdgroups share a key block from 8 rows, each with whole query
    # rows of the G heads.
    rg = 2 if R >= 8 and (R // 2) % G == 0 else 1
    nsg = _ATTN_BLOCKS * rg
    pso = -(-(R * D + R * RL + R) // 4) * 4
    tgn = max(pso + R * 32 * _ATTN_BLOCKS, _ATTN_BLOCKS * R * DP)
    one, merge = _attn_kernels()
    final = splits == 1
    common = [("H", H), ("KVH", KVH), ("G", G), ("L", L), ("R", R),
              ("D", D), ("DP", DP)]
    out = one(
        inputs=[q, k, v, r, sink, table, params],
        template=[("T", q.dtype), ("OT", q.dtype if final else mx.float32),
                  *common, ("RL", RL), ("EXT", table.shape[0] - 1),
                  ("WIN", window), ("RING", ring), ("FINAL", final),
                  ("NSG", nsg), ("RG", rg), ("TGN", tgn), ("PSO", pso)],
        grid=(32, nsg * splits, B * KVH),
        threadgroup=(32, nsg, 1),
        output_shapes=[(B, H, L, D) if final
                       else (B * KVH, splits, R, DP)],
        output_dtypes=[q.dtype if final else mx.float32],
    )[0]
    if final:
        return out
    return merge(
        inputs=[out, params],
        template=[("OT", q.dtype), *common],
        grid=(D, R, B * KVH),
        threadgroup=(D, 1, 1),
        output_shapes=[(B, H, L, D)],
        output_dtypes=[q.dtype],
    )[0]


def _attend_ops(q, k, v, r, sink, table, params, window, ring, scale):
    """The kernel's attention from ops, for the gradient."""
    mask = _mask_ops(r, sink, table, params, k.shape[2], window, ring,
                     q.dtype)
    return mx.fast.scaled_dot_product_attention(
        q, k, v, scale=scale, mask=mask,
        sinks=mx.zeros((q.shape[1],), dtype=q.dtype))


@functools.lru_cache(maxsize=None)
def _attend_fn(window: int, ring: bool, scale: float):
    @mx.custom_function
    def run(q, k, v, r, sink, table, params):
        return _attend_kernel(q, k, v, r, sink, table, params, window, ring)

    @run.vjp
    def run_vjp(primals, cotangent, _output):
        q, k, v, r, sink, table, params = primals
        _, grads = mx.vjp(
            lambda *a: _attend_ops(*a, params, window, ring, scale),
            [q, k, v, r, sink, table], [cotangent])
        return (*grads, mx.zeros_like(params))

    return run


def attend(q, k, v, r, sink, table, window, ring_idx, scale: float):
    """Attention at decode and verify widths with the folded bias and the
    zero sink logit, in one or two dispatches and no mask: ``q`` [B, H, L,
    D], ``k`` and ``v`` [B, H_kv, K, D] (strided views read in place),
    ``r`` [B, L, H, RL] and ``sink`` [B, L, H] in float32, ``table`` the
    padded bias table. Distances run back from the newest key in time
    order, or in ring order from ``ring_idx``. Returns [B, H, L, D]."""
    ring = ring_idx is not None
    n_keys = k.shape[2]
    chunk, splits = _attn_split(n_keys, q.shape[0] * k.shape[1])
    scale = float(scale)
    bits = struct.unpack("<i", struct.pack("<f", scale))[0]
    params = mx.array([n_keys, int(ring_idx) if ring else 0, chunk, splits,
                       bits], dtype=mx.int32)
    return _attend_fn(int(window or 0), ring, scale)(
        q, k, v, mx.contiguous(r), mx.contiguous(sink), table, params)


# Output gate: reads the attention output [B, H, L, D] and the gate through
# their strides and writes the gated rows [B, L, H * D] contiguous, so the
# output projection needs no copy.
_GATE = """
        uint i = thread_position_in_grid.x;
        const uint B = attn_shape[0];
        const uint L = attn_shape[2];
        if (i >= B * L * H * D) {
            return;
        }
        uint dd = i % D;
        uint h = (i / D) % H;
        uint l = (i / (D * H)) % L;
        uint b = i / (D * H * L);
        float o = static_cast<float>(attn[b * attn_strides[0] + h * attn_strides[1]
                                          + l * attn_strides[2] + dd * attn_strides[3]]);
        float x = static_cast<float>(
            g[b * g_strides[0] + l * g_strides[1] + (h * D + dd) * g_strides[2]]);
        // mx.sigmoid, as MLX's Metal unary op computes it.
        float y = 1 / (1 + metal::precise::exp(metal::abs(x)));
        float s = static_cast<float>(static_cast<T>((x < 0) ? y : 1 - y));
        out[i] = static_cast<T>(o * s);
"""


@functools.cache
def _gate_kernel():
    return mx.fast.metal_kernel(
        name="inkling2_out_gate",
        input_names=["attn", "g"],
        output_names=["out"],
        source=_GATE,
        ensure_row_contiguous=False,
    )


def gate_eligible(attn: mx.array, g: mx.array) -> bool:
    if attn.ndim != 4 or g.ndim != 3:
        return False
    B, H, L, D = attn.shape
    return (g.shape == (B, L, H * D)
            and attn.dtype in (mx.float16, mx.bfloat16, mx.float32)
            and enabled())


@functools.lru_cache(maxsize=None)
def _gate_fn(ops):
    @mx.custom_function
    def run(attn, g):
        B, H, L, D = attn.shape
        n = B * L * H * D
        return _gate_kernel()(
            inputs=[attn, g],
            template=[("T", attn.dtype), ("H", H), ("D", D)],
            grid=(-(-n // 256) * 256, 1, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[(B, L, H * D)],
            output_dtypes=[attn.dtype],
        )[0]

    @run.vjp
    def run_vjp(primals, cotangent, _output):
        def f(attn, g):
            B, H, L, D = attn.shape
            return ops(attn.transpose(0, 2, 1, 3).reshape(B, L, -1), g)

        _, grads = mx.vjp(f, list(primals), [cotangent])
        return tuple(grads)

    return run


def gate(attn: mx.array, g: mx.array, ops) -> mx.array:
    """``ops(attn as [B, L, H * D], g)``, the sigmoid output gate, in one
    dispatch with a contiguous result. ``ops`` gives the gradient."""
    return _gate_fn(ops)(attn, g)
