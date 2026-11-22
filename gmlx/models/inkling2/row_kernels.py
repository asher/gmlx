# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Asher Feldman
"""Metal kernels for small Inkling 2 row ops.

Per decode step, the ops path runs these chains many times:

- the L2 row clip, about 144 times, four dispatches each;
- the float32 RMS norm, about 154 times, three each (the casts);
- the router, 48 times, six each;
- the depth mix, about 120 times, four each;
- a loop controller, 96 times, seven each.

Here each is one dispatch, and a controller two. The arithmetic follows
the ops path; only the order of a row sum differs, so results match it to
float32 rounding. The router returns the same experts in score order,
where the ops path returns argpartition's order.

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


_ROUTER = """
        uint lane = thread_index_in_simdgroup;
        uint row = threadgroup_position_in_grid.x * (threads_per_threadgroup.x / 32)
                   + simdgroup_index_in_threadgroup;
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
        inputs=[logits.astype(mx.float32), bias, _scalar(float(scale))],
        template=[("O", dtype), ("E", E), ("K", k), ("N", n),
                  ("BIASED", biased)],
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
        device float* o = out + row * NQ * D;
        for (uint d = tid; d < D; d += threads_per_threadgroup.x) {
            float cv = static_cast<float>(c[d]) * inv;
            for (int j = 0; j < NQ; j++) {
                float v = p[j][NB] * cv;
                for (int b = 0; b < NB; b++) {
                    v += p[j][b] * s[b][d];
                }
                o[j * D + d] = v;
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


def _mix_kernel(blocks, cur, query, eps):
    B, L, D = cur.shape
    nq = query.shape[0]
    return _mix_kernel_for(len(blocks))(
        inputs=[*blocks, cur, query.astype(mx.float32), _scalar(float(eps))],
        template=[("T", cur.dtype), ("D", D), ("NB", len(blocks)),
                  ("NQ", nq)],
        grid=(B * L * MIX_THREADS, 1, 1),
        threadgroup=(MIX_THREADS, 1, 1),
        output_shapes=[(B, L, nq, D)],
        output_dtypes=[mx.float32],
    )[0]


@functools.lru_cache(maxsize=None)
def _mix_fn(ops, eps: float, n_blocks: int):
    @mx.custom_function
    def run(*args):
        *blocks, cur, query = args
        return _mix_kernel(blocks, cur, query, eps)

    @run.vjp
    def run_vjp(primals, cotangent, _output):
        def f(*args):
            *blocks, cur, query = args
            return ops(tuple(blocks), cur, query)

        _, grads = mx.vjp(f, list(primals), [cotangent])
        return tuple(grads)

    return run


def depth_mix(blocks, cur: mx.array, query: mx.array, eps: float, ops):
    """``model.depth_mix`` in one dispatch: ``blocks`` rms-normed float32
    [B, L, D] states, ``cur`` [B, L, D] (normed here), ``query`` [nq, D].
    Returns [B, L, nq, D] float32. ``ops(blocks, cur, query)`` is the ops
    path, which gives the gradient."""
    return _mix_fn(ops, float(eps), len(blocks))(*blocks, cur, query)


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
