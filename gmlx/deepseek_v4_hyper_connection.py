# Copyright (c) 2026 Apple Inc.
#
# Manifold-constrained hyper-connections for DeepSeek V4 (vendored).
# Copied near-verbatim from mlx-lm PR 1192 by way of omlx's
# patches/deepseek_v4/hyper_connection.py port (fused Metal
# sinkhorn+collapse kernel via mx.fast.metal_kernel with a compiled
# pure-ops fallback for CPU/training). Registered into the
# mlx_lm.models namespace by deepseek_v4_model.ensure_registered().
# Delete once installed mlx-lm ships models/hyper_connection.py.

import os
from typing import Tuple

import mlx.core as mx
import mlx.nn as nn

# Prefer the mlx-kquant C++ ports of the M=1 glue kernels when the
# installed kq has them: same kernels, bit-exact, ~5x cheaper per call on
# the host. GMLX_HC_KQ=0 forces the in-repo JIT kernels for A/Bs.
try:
    import mlx_kquant as _kq

    _KQ_HC = hasattr(_kq, "hc_front_reduce") and os.environ.get(
        "GMLX_HC_KQ", "1") != "0"
except ImportError:
    _kq = None
    _KQ_HC = False


def _make_hc_sinkhorn_collapse_kernel():
    """Fused rmsnorm + mixes matmul + sinkhorn + collapse.

    One dispatch per HC cycle. The front (fp32 cast, rms_norm over the
    flattened HC*D row, z @ fn_t) is computed in-kernel: rms_norm commutes
    with the matmul (mixes = rrms * (y @ fn_t)), so one pass over the row
    accumulates sum-of-squares and the MIX dot products together. This
    replaces an M x 16384 x 24 fp32 GEMM that falls off the GEMV fast path
    at M >= 2 (46us vs 7us on M5 Max) plus two more dispatches.

    1. ONE-PASS FRONT: 256 threads stride the HC*D row; each accumulates
       ssq + MIX fma partials from coalesced float4 fn_t loads; simd_sum
       then a cross-simdgroup reduce in threadgroup memory.
    2. BRANCHLESS SINKHORN: all 32 lanes in simd group 0 execute identical
       instructions. Lanes >= HC use multiplicative mask (active=0) instead
       of divergent branches - eliminates SIMD serialization.
    3. PARALLEL SINKHORN: lanes 0-3 each own one comb row. Column norm
       via simd_sum() - free SIMD shuffle.
    4. NATIVE bfloat4 LOADS: single 64-bit load yields 4 bfloat16 values;
       cast to float4 is a free hardware conversion.
    5. FMA CHAINS: collapse uses fused multiply-add for 3 of 4 terms.
    """
    if mx.default_device() != mx.gpu or not mx.metal.is_available():
        return None

    source = """
        uint tid  = thread_position_in_threadgroup.x;
        uint row  = threadgroup_position_in_grid.x;
        uint lane = tid % 32;
        uint sg   = tid / 32;

        constexpr int MIX      = (2 + HC) * HC;
        constexpr int BASE_OFF = 2 * HC;
        constexpr int KD       = HC * D;
        constexpr float EPS  = EPS_INT * 1e-9;
        constexpr float NEPS = NEPS_INT * 1e-9;

        device float* post_out = (device float*)post + row * HC;
        device float* comb_out = (device float*)comb + row * HC * HC;

        threadgroup float pre_shared[HC];
        threadgroup float mix_sh[MIX];
        threadgroup float red_sh[8][MIX];
        threadgroup float ssq_sh[8];

        // ================================================================
        // PHASE 0: rmsnorm + mixes in one pass over the row.
        //   mixes = rms_norm(y) @ fn_t = rrms * (y @ fn_t), so raw dots
        //   and sum-of-squares accumulate together; rrms scales at the end.
        // ================================================================
        {
            const device T*      xr  = (const device T*)x_in + row * KD;
            const device float4* fr  = (const device float4*)fn_t;

            float acc[MIX] = {0.0f};
            float ssq = 0.0f;
            for (uint i = tid; i < (uint)KD; i += 256) {
                float xv = (float)xr[i];
                ssq = metal::fma(xv, xv, ssq);
                const device float4* f4 = fr + i * (MIX / 4);
                #pragma clang loop unroll(full)
                for (int j = 0; j < MIX / 4; ++j) {
                    float4 f = f4[j];
                    acc[4*j+0] = metal::fma(xv, f.x, acc[4*j+0]);
                    acc[4*j+1] = metal::fma(xv, f.y, acc[4*j+1]);
                    acc[4*j+2] = metal::fma(xv, f.z, acc[4*j+2]);
                    acc[4*j+3] = metal::fma(xv, f.w, acc[4*j+3]);
                }
            }
            ssq = simd_sum(ssq);
            #pragma clang loop unroll(full)
            for (int j = 0; j < MIX; ++j) {
                acc[j] = simd_sum(acc[j]);
            }
            if (lane == 0) {
                ssq_sh[sg] = ssq;
                for (int j = 0; j < MIX; ++j) {
                    red_sh[sg][j] = acc[j];
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (sg == 0) {
                float s = (lane < 8) ? ssq_sh[lane] : 0.0f;
                s = simd_sum(s);
                float rrms = metal::precise::rsqrt(s / (float)KD + NEPS);
                if (lane < (uint)MIX) {
                    float tot = 0.0f;
                    for (int g = 0; g < 8; ++g) {
                        tot += red_sh[g][lane];
                    }
                    mix_sh[lane] = tot * rrms;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // ================================================================
        // PHASE 1: Branchless sinkhorn on simd group 0
        //   All 32 lanes execute identical instructions. Lanes >= HC
        //   compute on clamped indices but multiply by active=0, so they
        //   contribute zero to simd_sum. No divergent branches in the loop.
        // ================================================================
        if (sg == 0) {
            const float pre_scale  = scale[0];
            const float post_scale = scale[1];
            const float comb_scale = scale[2];

            const float active = (lane < (uint)HC) ? 1.0f : 0.0f;
            const uint  llane  = metal::min(lane, (uint)(HC - 1));

            // Pre/post sigmoids: all lanes compute, only active lanes write
            float pre_z  = mix_sh[llane]      * pre_scale  + base[llane];
            float post_z = mix_sh[HC + llane] * post_scale + base[HC + llane];
            float pre_v  = 1.0f / (1.0f + metal::fast::exp(-pre_z)) + EPS;
            float post_v = 2.0f / (1.0f + metal::fast::exp(-post_z));

            if (lane < (uint)HC) {
                pre_shared[lane] = pre_v;
                post_out[lane]   = post_v;
            }

            // Comb softmax: load + mask. Inactive lanes load row 0 (safe)
            // but multiply by active=0 so they hold zeros.
            float4 m = float4(mix_sh[BASE_OFF + llane * HC + 0],
                              mix_sh[BASE_OFF + llane * HC + 1],
                              mix_sh[BASE_OFF + llane * HC + 2],
                              mix_sh[BASE_OFF + llane * HC + 3]);
            float4 v = (m * comb_scale
                      + *(const device float4*)(base + BASE_OFF + llane * HC))
                     * active;

            float row_max = metal::max(metal::max(v.x, v.y),
                                       metal::max(v.z, v.w));
            float4 e = metal::fast::exp(v - row_max) * active;
            float4 r = e * (1.0f / (e.x + e.y + e.z + e.w + EPS))
                     + EPS * active;

            // Initial column normalization
            float4 col_inv = 1.0f / (float4(
                simd_sum(r.x), simd_sum(r.y),
                simd_sum(r.z), simd_sum(r.w)
            ) + EPS);
            r *= col_inv;

            // Sinkhorn iterations: zero branches in the loop body
            for (int iter = 1; iter < ITERS; ++iter) {
                // Row norm + re-clamp inactive lanes
                r *= (1.0f / (r.x + r.y + r.z + r.w + EPS)) * active;

                // Col norm via simd_sum
                col_inv = 1.0f / (float4(
                    simd_sum(r.x), simd_sum(r.y),
                    simd_sum(r.z), simd_sum(r.w)
                ) + EPS);
                r *= col_inv;
            }

            if (lane < (uint)HC) {
                *(device float4*)(comb_out + lane * HC) = r;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ================================================================
        // PHASE 2: Collapse - all 256 threads, vectorized
        // ================================================================
        const float p0 = pre_shared[0];
        const float p1 = pre_shared[1];
        const float p2 = pre_shared[2];
        const float p3 = pre_shared[3];

        const device T* x_row  = (const device T*)x_in
                                         + row * (HC * D);
        device U*       out_row = (device U*)collapsed
                                         + row * D;

        using T4 = vec<T, 4>;
        using U4 = vec<U, 4>;
        const device T4* x_row0 = (const device T4*)(x_row + 0*D);
        const device T4* x_row1 = (const device T4*)(x_row + 1*D);
        const device T4* x_row2 = (const device T4*)(x_row + 2*D);
        const device T4* x_row3 = (const device T4*)(x_row + 3*D);
        device U4*       out4   = (device U4*)out_row;

        constexpr uint D4 = (uint)D / 4;

        for (uint d4 = tid; d4 < D4; d4 += 256) {
            float4 x0 = float4(x_row0[d4]);
            float4 x1 = float4(x_row1[d4]);
            float4 x2 = float4(x_row2[d4]);
            float4 x3 = float4(x_row3[d4]);

            float4 result = fma(float4(p0), x0,
                            fma(float4(p1), x1,
                            fma(float4(p2), x2, float4(p3) * x3)));

            out4[d4] = U4(result);
        }

        // Scalar tail for D not divisible by 4
        #if (D % 4) != 0
        for (uint d = D4 * 4 + tid; d < (uint)D; d += 256) {
            float val = p0*(float)x_row[0*D+d] + p1*(float)x_row[1*D+d]
                      + p2*(float)x_row[2*D+d] + p3*(float)x_row[3*D+d];
            out_row[d] = (U)val;
        }
        #endif
    """

    return mx.fast.metal_kernel(
        name="hc_norm_mix_sinkhorn_collapse",
        input_names=["x_in", "fn_t", "scale", "base"],
        output_names=["collapsed", "post", "comb"],
        source=source,
        ensure_row_contiguous=True,
    )


_hc_sinkhorn_collapse_kernel = _make_hc_sinkhorn_collapse_kernel()


def _hc_kernel(x, fn_t, scale, base, hc_mult, sinkhorn_iters, eps, norm_eps):
    """Requires ``hc_mult == 4`` despite the HC template arg (the kernel body
    unrolls 4 streams); callers route other widths to :func:`_hc_ops`."""
    B, L, H, D = x.shape

    return _hc_sinkhorn_collapse_kernel(
        inputs=[x, fn_t, scale, base],
        template=[
            ("T", x.dtype),
            ("U", x.dtype),
            ("HC", hc_mult),
            ("ITERS", sinkhorn_iters),
            ("D", D),
            ("EPS_INT", round(eps / 1e-9)),
            ("NEPS_INT", round(norm_eps / 1e-9)),
        ],
        grid=(B * L * 256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(B, L, D), (B, L, hc_mult), (B, L, hc_mult, hc_mult)],
        output_dtypes=[x.dtype, mx.float32, mx.float32],
    )


@mx.compile
def _hc_split_sinkhorn_ops(
    mixes: mx.array,
    scale: mx.array,
    base: mx.array,
    hc_mult: int,
    sinkhorn_iters: int,
    eps: float,
) -> Tuple[mx.array, mx.array, mx.array]:
    mixes = mixes.astype(mx.float32)
    scale = scale.astype(mx.float32)
    base = base.astype(mx.float32)
    pre_scale, post_scale, comb_scale = scale[0], scale[1], scale[2]

    pre = mx.sigmoid(mixes[..., :hc_mult] * pre_scale + base[:hc_mult]) + eps
    post = 2 * mx.sigmoid(
        mixes[..., hc_mult : 2 * hc_mult] * post_scale + base[hc_mult : 2 * hc_mult]
    )
    comb = mixes[..., 2 * hc_mult :].reshape(
        *mixes.shape[:-1], hc_mult, hc_mult
    ) * comb_scale + base[2 * hc_mult :].reshape(hc_mult, hc_mult)
    comb = mx.softmax(comb, axis=-1, precise=True) + eps
    comb = comb / (comb.sum(axis=-2, keepdims=True) + eps)
    for _ in range(max(sinkhorn_iters - 1, 0)):
        comb = comb / (comb.sum(axis=-1, keepdims=True) + eps)
        comb = comb / (comb.sum(axis=-2, keepdims=True) + eps)
    return pre, post, comb


def _hc_ops(x, y, mixes, scale, base, hc_mult, sinkhorn_iters, eps):
    pre, post, comb = _hc_split_sinkhorn_ops(
        mixes, scale, base, hc_mult, sinkhorn_iters, eps
    )
    return (pre[..., None] * y).sum(axis=2).astype(x.dtype), post, comb


class HyperConnection(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hc_mult = config.hc_mult
        self.sinkhorn_iters = config.hc_sinkhorn_iters
        self.hc_eps = config.hc_eps
        self.norm_eps = config.rms_norm_eps

        mix = (2 + self.hc_mult) * self.hc_mult
        self.fn = mx.zeros((mix, self.hc_mult * config.hidden_size), dtype=mx.float32)
        self.base = mx.zeros((mix,), dtype=mx.float32)
        self.scale = mx.ones((3,), dtype=mx.float32)
        self._m1_ok = None

    def m1_fused_ok(self, x):
        """True when the 3-kernel M=1 route applies to this call."""
        ok = self._m1_ok
        if ok is None:
            # static conditions only; training is re-checked per call
            ok = (
                _HC_M1_ENABLED
                and self.hc_mult == 4
                and (_KQ_HC or _hc_front_reduce_kernel is not None)
                and mx.default_device() == mx.gpu
                and (self.fn.shape[1] // self.hc_mult) % 1024 == 0
            )
            self._m1_ok = ok
        return ok and not self.training and x.shape[0] * x.shape[1] == 1

    def _collapse_m1(self, x, mixes_raw, ssq, norm_weight):
        B, L, H, D = x.shape
        if _KQ_HC:
            return _kq.hc_sinkhorn_collapse(
                x, mixes_raw, ssq, self.scale, self.base, norm_weight,
                iters=self.sinkhorn_iters, hc_eps=self.hc_eps,
                norm_eps=self.norm_eps)
        return _hc_sinkhorn_collapse_m1_kernel(
            inputs=[x, mixes_raw, ssq, self.scale, self.base, norm_weight],
            template=[
                ("T", x.dtype),
                ("U", x.dtype),
                ("HC", H),
                ("ITERS", self.sinkhorn_iters),
                ("D", D),
                ("EPS_INT", round(self.hc_eps / 1e-9)),
                ("NEPS_INT", round(self.norm_eps / 1e-9)),
            ],
            grid=(B * L * 256, 1, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[(B, L, D), (B, L, H), (B, L, H, H)],
            output_dtypes=[x.dtype, mx.float32, mx.float32],
        )

    def fused_m1(self, x, norm_weight):
        """Two-dispatch M=1 front: front_reduce then sinkhorn+collapse with
        the sublayer RMSNorm (weight ``norm_weight``) folded into the output.
        Returns (normed_collapsed, post, comb); matches __call__ + RMSNorm to
        rounding (fp reduction order differs)."""
        B, L, H, D = x.shape
        if _KQ_HC:
            mixes_raw, ssq = _kq.hc_front_reduce(x, self.fn)
            return self._collapse_m1(x, mixes_raw, ssq, norm_weight)
        mix = (2 + H) * H
        mixes_raw, ssq = _hc_front_reduce_kernel(
            inputs=[x, self.fn],
            template=[("T", x.dtype), ("HC", H), ("D", D)],
            grid=(B * L * (mix + 1) * 256, 1, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[(B, L, mix), (B, L, 1)],
            output_dtypes=[mx.float32, mx.float32],
        )
        return self._collapse_m1(x, mixes_raw, ssq, norm_weight)

    def fused_m1_expand(self, carry, norm_weight):
        """Step-2 front: consume the previous cycle's pending expand
        (x_sub, residual, post, comb) inside the front reduction, one
        dispatch for expand+cast+rms+mix GEMV. Returns (h, front) where h
        is the materialized expanded stream and front is (normed_collapsed,
        post, comb) for this cycle. Bit-identical to
        hc_expand_m1 followed by fused_m1."""
        x_sub, resid, post, comb = carry
        B, L, D = x_sub.shape
        H = resid.shape[2]
        if _KQ_HC:
            h, mixes_raw, ssq = _kq.hc_front_expand_reduce(
                x_sub, resid, post, comb, self.fn)
            return h, self._collapse_m1(h, mixes_raw, ssq, norm_weight)
        mix = (2 + H) * H
        h, mixes_raw, ssq = _hc_front_expand_reduce_kernel(
            inputs=[x_sub, resid, post, comb, self.fn],
            template=[("T", x_sub.dtype), ("HC", H), ("D", D)],
            grid=(B * L * (mix + 1) * 256, 1, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[(B, L, H, D), (B, L, mix), (B, L, 1)],
            output_dtypes=[x_sub.dtype, mx.float32, mx.float32],
        )
        return h, self._collapse_m1(h, mixes_raw, ssq, norm_weight)

    def __call__(self, x: mx.array):
        # fn is frozen; materialize its transpose once instead of adding a
        # Transpose node per layer per step (underscored: not a parameter).
        fn_t = getattr(self, "_fn_t", None)
        if fn_t is None:
            fn_t = mx.contiguous(self.fn.T)
            mx.eval(fn_t)
            self._fn_t = fn_t

        use_ops = (
            self.training
            or mx.default_device() != mx.gpu
            # The fused kernel's float4 row loads and pre_shared[0..3] /
            # x_row0..3 unrolls assume 4 streams; _hc_ops is generic.
            or self.hc_mult != 4
            or not mx.metal.is_available()
        )
        if not use_ops:
            return _hc_kernel(
                x,
                fn_t,
                self.scale,
                self.base,
                self.hc_mult,
                self.sinkhorn_iters,
                self.hc_eps,
                self.norm_eps,
            )

        y = x.astype(mx.float32)
        z = mx.fast.rms_norm(y.flatten(-2), None, self.norm_eps)
        mixes = z @ fn_t
        return _hc_ops(
            x,
            y,
            mixes,
            self.scale,
            self.base,
            self.hc_mult,
            self.sinkhorn_iters,
            self.hc_eps,
        )


@mx.compile
def _hc_expand_op(x, residual, post, comb):
    y = post[..., None] * x[:, :, None, :].astype(mx.float32)
    y = y + mx.matmul(comb.swapaxes(-1, -2), residual.astype(mx.float32))
    return y.astype(x.dtype)


def hc_expand(x, residual, post, comb):
    return _hc_expand_op(x, residual, post, comb)



# --- fused M=1 decode route ------------------------------------------------
# At B*L == 1 the HC glue is dispatch-latency bound: the eager front
# (f32 cast -> rms_norm -> mix GEMV) plus the sinkhorn+collapse kernel,
# the sublayer RMSNorm, and the compiled expand cost ~7 dispatches per
# cycle. The route below folds the cycle into 3 deterministic kernels:
#   front_reduce:      raw mix GEMV + stream sum-of-squares, one dispatch
#                      (rms_norm(y) @ fn.T == (y @ fn.T) * rsqrt(mean(y^2)
#                      + eps) because the norm is weightless, so the scalar
#                      factor is deferred to the collapse kernel's scales)
#   sinkhorn_collapse_m1: existing kernel + deferred rms factor + the
#                      sublayer RMSNorm folded into the output pass
#   expand_m1:         post/comb expand as one elementwise kernel
# Gated by GMLX_HC_M1_FUSED (default on); any other width or device uses
# the original path. fp reduction order differs from the eager front, so
# outputs match to rounding, not bit-exactly.

_HC_M1_ENABLED = os.environ.get("GMLX_HC_M1_FUSED", "1") != "0"


def _make_hc_front_reduce_kernel():
    if mx.default_device() != mx.gpu or not mx.metal.is_available():
        return None

    source = """
        uint tid  = thread_position_in_threadgroup.x;
        uint tg   = threadgroup_position_in_grid.x;
        uint lane = tid % 32;
        uint sg   = tid / 32;

        constexpr int MIX  = (2 + HC) * HC;
        constexpr int KTOT = HC * D;

        uint row = tg / (MIX + 1);
        uint m   = tg % (MIX + 1);

        const device T* xr = (const device T*)x_in + row * KTOT;
        using T4 = vec<T, 4>;
        const device T4* x4 = (const device T4*)xr;

        float acc = 0.0f;
        if (m < (uint)MIX) {
            const device float4* f4 = (const device float4*)(fn + m * KTOT);
            for (uint k = tid; k < (uint)(KTOT / 4); k += 256) {
                float4 xv = float4(x4[k]);
                float4 fv = f4[k];
                acc = fma(xv.x, fv.x, acc);
                acc = fma(xv.y, fv.y, acc);
                acc = fma(xv.z, fv.z, acc);
                acc = fma(xv.w, fv.w, acc);
            }
        } else {
            for (uint k = tid; k < (uint)(KTOT / 4); k += 256) {
                float4 xv = float4(x4[k]);
                acc = fma(xv.x, xv.x, acc);
                acc = fma(xv.y, xv.y, acc);
                acc = fma(xv.z, xv.z, acc);
                acc = fma(xv.w, xv.w, acc);
            }
        }

        threadgroup float partial[8];
        acc = simd_sum(acc);
        if (lane == 0) partial[sg] = acc;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0) {
            float v = (lane < 8) ? partial[lane] : 0.0f;
            v = simd_sum(v);
            if (lane == 0) {
                if (m < (uint)MIX) mixes_raw[row * MIX + m] = v;
                else               sumsq[row] = v;
            }
        }
    """
    return mx.fast.metal_kernel(
        name="hc_front_reduce",
        input_names=["x_in", "fn"],
        output_names=["mixes_raw", "sumsq"],
        source=source,
        ensure_row_contiguous=True,
    )


def _make_hc_sinkhorn_collapse_m1_kernel():
    """The stock sinkhorn+collapse kernel with two extensions for the fused
    M=1 route: the deferred rms factor multiplies the three mix scales, and
    the collapse result is RMSNorm-ed (weight ``w``) before the single
    rounding to the output dtype."""
    if mx.default_device() != mx.gpu or not mx.metal.is_available():
        return None

    source = """
        uint tid  = thread_position_in_threadgroup.x;
        uint row  = threadgroup_position_in_grid.x;
        uint lane = tid % 32;
        uint sg   = tid / 32;

        constexpr int MIX      = (2 + HC) * HC;
        constexpr int BASE_OFF = 2 * HC;
        constexpr float EPS  = EPS_INT * 1e-9;
        constexpr float NEPS = NEPS_INT * 1e-9;

        const device float* mix      = (const device float*)mixes_raw
                                       + row * MIX;
        device float*       post_out = (device float*)post + row * HC;
        device float*       comb_out = (device float*)comb + row * HC * HC;

        const float factor = metal::rsqrt(
            sumsq[row] / (float)(HC * D) + NEPS);

        threadgroup float pre_shared[HC];
        threadgroup float ssq_shared[8];
        threadgroup float inv_shared[1];

        if (sg == 0) {
            const float pre_scale  = scale[0] * factor;
            const float post_scale = scale[1] * factor;
            const float comb_scale = scale[2] * factor;

            const float active = (lane < (uint)HC) ? 1.0f : 0.0f;
            const uint  llane  = metal::min(lane, (uint)(HC - 1));

            float pre_z  = mix[llane]      * pre_scale  + base[llane];
            float post_z = mix[HC + llane] * post_scale + base[HC + llane];
            float pre_v  = 1.0f / (1.0f + metal::fast::exp(-pre_z)) + EPS;
            float post_v = 2.0f / (1.0f + metal::fast::exp(-post_z));

            if (lane < (uint)HC) {
                pre_shared[lane] = pre_v;
                post_out[lane]   = post_v;
            }

            float4 v = (*(const device float4*)(mix  + BASE_OFF + llane * HC)
                            * comb_scale
                      + *(const device float4*)(base + BASE_OFF + llane * HC))
                     * active;

            float row_max = metal::max(metal::max(v.x, v.y),
                                       metal::max(v.z, v.w));
            float4 e = metal::fast::exp(v - row_max) * active;
            float4 r = e * (1.0f / (e.x + e.y + e.z + e.w + EPS))
                     + EPS * active;

            float4 col_inv = 1.0f / (float4(
                simd_sum(r.x), simd_sum(r.y),
                simd_sum(r.z), simd_sum(r.w)
            ) + EPS);
            r *= col_inv;

            for (int iter = 1; iter < ITERS; ++iter) {
                r *= (1.0f / (r.x + r.y + r.z + r.w + EPS)) * active;
                col_inv = 1.0f / (float4(
                    simd_sum(r.x), simd_sum(r.y),
                    simd_sum(r.z), simd_sum(r.w)
                ) + EPS);
                r *= col_inv;
            }

            if (lane < (uint)HC) {
                *(device float4*)(comb_out + lane * HC) = r;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        const float p0 = pre_shared[0];
        const float p1 = pre_shared[1];
        const float p2 = pre_shared[2];
        const float p3 = pre_shared[3];

        const device T* x_row  = (const device T*)x_in + row * (HC * D);
        device U*       out_row = (device U*)collapsed + row * D;

        using T4 = vec<T, 4>;
        using U4 = vec<U, 4>;
        const device T4* x_row0 = (const device T4*)(x_row + 0*D);
        const device T4* x_row1 = (const device T4*)(x_row + 1*D);
        const device T4* x_row2 = (const device T4*)(x_row + 2*D);
        const device T4* x_row3 = (const device T4*)(x_row + 3*D);
        device U4*       out4   = (device U4*)out_row;

        constexpr uint D4 = (uint)D / 4;
        constexpr uint CHUNKS = (D4 + 255) / 256;

        float4 vals[CHUNKS];
        float ssq = 0.0f;
        for (uint c = 0; c < CHUNKS; ++c) {
            uint d4 = c * 256 + tid;
            float4 result = float4(0.0f);
            if (d4 < D4) {
                float4 x0 = float4(x_row0[d4]);
                float4 x1 = float4(x_row1[d4]);
                float4 x2 = float4(x_row2[d4]);
                float4 x3 = float4(x_row3[d4]);
                result = fma(float4(p0), x0,
                         fma(float4(p1), x1,
                         fma(float4(p2), x2, float4(p3) * x3)));
                ssq += result.x * result.x + result.y * result.y
                     + result.z * result.z + result.w * result.w;
            }
            vals[c] = result;
        }

        ssq = simd_sum(ssq);
        if (lane == 0) ssq_shared[sg] = ssq;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0) {
            float v = (lane < 8) ? ssq_shared[lane] : 0.0f;
            v = simd_sum(v);
            if (lane == 0) {
                inv_shared[0] = metal::rsqrt(v / (float)D + NEPS);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float inv = inv_shared[0];

        for (uint c = 0; c < CHUNKS; ++c) {
            uint d4 = c * 256 + tid;
            if (d4 < D4) {
                uint d = d4 * 4;
                float4 wv = float4((float)w[d],     (float)w[d + 1],
                                   (float)w[d + 2], (float)w[d + 3]);
                out4[d4] = U4(vals[c] * inv * wv);
            }
        }
    """
    return mx.fast.metal_kernel(
        name="hc_sinkhorn_collapse_m1",
        input_names=["x_in", "mixes_raw", "sumsq", "scale", "base", "w"],
        output_names=["collapsed", "post", "comb"],
        source=source,
        ensure_row_contiguous=True,
    )


def _make_hc_front_expand_reduce_kernel():
    """Step-2 fusion: the previous cycle's expand computed inline ahead of
    the front reduction, one dispatch for both. Every threadgroup recomputes
    the expand for the chunks its dot needs (identical fp order to the
    expand_m1 kernel, rounded to T before use, so h and the mixes are
    bit-identical to the unmerged pair); the sumsq threadgroup also writes
    h out for the residual chain and captures."""
    if mx.default_device() != mx.gpu or not mx.metal.is_available():
        return None

    source = """
        uint tid  = thread_position_in_threadgroup.x;
        uint tg   = threadgroup_position_in_grid.x;
        uint lane = tid % 32;
        uint sg   = tid / 32;

        constexpr int MIX  = (2 + HC) * HC;
        constexpr int KTOT = HC * D;
        constexpr uint D4q = (uint)D / 4;

        uint row = tg / (MIX + 1);
        uint m   = tg % (MIX + 1);

        const device T* xs = (const device T*)x_sub + row * D;
        const device T* rr = (const device T*)resid + row * KTOT;
        device T* hout = (device T*)h_out + row * KTOT;

        const uint pb = row * 4, cb = row * 16;

        using T4 = vec<T, 4>;
        const device T4* xs4 = (const device T4*)xs;
        const device T4* rr4 = (const device T4*)rr;
        device T4* h4 = (device T4*)hout;
        const device float4* f4 = (const device float4*)(fn + m * KTOT);

        float acc = 0.0f;
        for (uint i = 0; i < (uint)HC; ++i) {
            const float pi = post[pb + i];
            const float c0 = comb[cb + 0 * 4 + i];
            const float c1 = comb[cb + 1 * 4 + i];
            const float c2 = comb[cb + 2 * 4 + i];
            const float c3 = comb[cb + 3 * 4 + i];
            for (uint d4 = tid; d4 < D4q; d4 += 256) {
                uint k = i * D4q + d4;
                float4 xv = float4(xs4[d4]);
                float4 r0 = float4(rr4[0 * D4q + d4]);
                float4 r1 = float4(rr4[1 * D4q + d4]);
                float4 r2 = float4(rr4[2 * D4q + d4]);
                float4 r3 = float4(rr4[3 * D4q + d4]);
                float4 e = fma(float4(pi), xv,
                           fma(float4(c0), r0, fma(float4(c1), r1,
                           fma(float4(c2), r2, float4(c3) * r3))));
                T4 hv = T4(e);
                float4 hf = float4(hv);
                if (m < (uint)MIX) {
                    float4 fv = f4[k];
                    acc = fma(hf.x, fv.x, acc);
                    acc = fma(hf.y, fv.y, acc);
                    acc = fma(hf.z, fv.z, acc);
                    acc = fma(hf.w, fv.w, acc);
                } else {
                    h4[k] = hv;
                    acc = fma(hf.x, hf.x, acc);
                    acc = fma(hf.y, hf.y, acc);
                    acc = fma(hf.z, hf.z, acc);
                    acc = fma(hf.w, hf.w, acc);
                }
            }
        }

        threadgroup float partial[8];
        acc = simd_sum(acc);
        if (lane == 0) partial[sg] = acc;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0) {
            float v = (lane < 8) ? partial[lane] : 0.0f;
            v = simd_sum(v);
            if (lane == 0) {
                if (m < (uint)MIX) mixes_raw[row * MIX + m] = v;
                else               sumsq[row] = v;
            }
        }
    """
    return mx.fast.metal_kernel(
        name="hc_front_expand_reduce",
        input_names=["x_sub", "resid", "post", "comb", "fn"],
        output_names=["h_out", "mixes_raw", "sumsq"],
        source=source,
        ensure_row_contiguous=True,
    )


def _make_hc_expand_m1_kernel():
    if mx.default_device() != mx.gpu or not mx.metal.is_available():
        return None

    source = """
        uint tid = thread_position_in_threadgroup.x;
        uint tg  = threadgroup_position_in_grid.x;
        uint row = tg / NTG;
        uint sub = tg % NTG;

        const device T* xr = (const device T*)x_in + row * D;
        const device T* rr = (const device T*)resid + row * 4 * D;
        device T* orow = (device T*)out + row * 4 * D;

        const uint pb = row * 4, cb = row * 16;
        float p0 = post[pb + 0], p1 = post[pb + 1];
        float p2 = post[pb + 2], p3 = post[pb + 3];
        // comb is [j][i]; expand applies comb^T: sum_j comb[j][i] * res[j]
        float c00 = comb[cb + 0],  c01 = comb[cb + 1];
        float c02 = comb[cb + 2],  c03 = comb[cb + 3];
        float c10 = comb[cb + 4],  c11 = comb[cb + 5];
        float c12 = comb[cb + 6],  c13 = comb[cb + 7];
        float c20 = comb[cb + 8],  c21 = comb[cb + 9];
        float c22 = comb[cb + 10], c23 = comb[cb + 11];
        float c30 = comb[cb + 12], c31 = comb[cb + 13];
        float c32 = comb[cb + 14], c33 = comb[cb + 15];

        constexpr uint SPAN = D / NTG;
        uint d0 = sub * SPAN;
        using T4 = vec<T, 4>;
        const device T4* x4  = (const device T4*)(xr + d0);
        const device T4* r04 = (const device T4*)(rr + 0*D + d0);
        const device T4* r14 = (const device T4*)(rr + 1*D + d0);
        const device T4* r24 = (const device T4*)(rr + 2*D + d0);
        const device T4* r34 = (const device T4*)(rr + 3*D + d0);
        device T4* o04 = (device T4*)(orow + 0*D + d0);
        device T4* o14 = (device T4*)(orow + 1*D + d0);
        device T4* o24 = (device T4*)(orow + 2*D + d0);
        device T4* o34 = (device T4*)(orow + 3*D + d0);
        for (uint k = tid; k < SPAN / 4; k += 256) {
            float4 xv = float4(x4[k]);
            float4 r0 = float4(r04[k]), r1 = float4(r14[k]);
            float4 r2 = float4(r24[k]), r3 = float4(r34[k]);
            o04[k] = T4(fma(float4(p0), xv,
                  fma(float4(c00), r0, fma(float4(c10), r1,
                  fma(float4(c20), r2, float4(c30) * r3)))));
            o14[k] = T4(fma(float4(p1), xv,
                  fma(float4(c01), r0, fma(float4(c11), r1,
                  fma(float4(c21), r2, float4(c31) * r3)))));
            o24[k] = T4(fma(float4(p2), xv,
                  fma(float4(c02), r0, fma(float4(c12), r1,
                  fma(float4(c22), r2, float4(c32) * r3)))));
            o34[k] = T4(fma(float4(p3), xv,
                  fma(float4(c03), r0, fma(float4(c13), r1,
                  fma(float4(c23), r2, float4(c33) * r3)))));
        }
    """
    return mx.fast.metal_kernel(
        name="hc_expand_m1",
        input_names=["x_in", "resid", "post", "comb"],
        output_names=["out"],
        source=source,
        ensure_row_contiguous=True,
    )


_hc_front_reduce_kernel = _make_hc_front_reduce_kernel()
_hc_front_expand_reduce_kernel = _make_hc_front_expand_reduce_kernel()
_hc_sinkhorn_collapse_m1_kernel = _make_hc_sinkhorn_collapse_m1_kernel()
_hc_expand_m1_kernel = _make_hc_expand_m1_kernel()


def hc_expand_m1(x, residual, post, comb):
    if _KQ_HC:
        return _kq.hc_expand(x, residual, post, comb)
    B, L, D = x.shape
    H = residual.shape[2]
    return _hc_expand_m1_kernel(
        inputs=[x, residual, post, comb],
        template=[("T", x.dtype), ("D", D), ("NTG", 2)],
        grid=(B * L * 2 * 256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(B, L, H, D)],
        output_dtypes=[x.dtype],
    )[0]




def _make_hc_expand_collapse_kernel():
    """Fused expand + rmsnorm + mixes + sinkhorn + collapse.

    One dispatch replaces the hc_expand between a block's attention output
    and its ffn HyperConnection plus that HyperConnection's whole front.
    The expanded h is recomputed on the fly for the collapse pass (a
    device barrier makes the h_out writes visible in-threadgroup), and the
    norm/dot math consumes the T-rounded h so the numerics match the
    unfused expand -> collapse chain exactly.
    """
    if mx.default_device() != mx.gpu or not mx.metal.is_available():
        return None

    source = """
        uint tid  = thread_position_in_threadgroup.x;
        uint row  = threadgroup_position_in_grid.x;
        uint lane = tid % 32;
        uint sg   = tid / 32;

        constexpr int MIX      = (2 + HC) * HC;
        constexpr int BASE_OFF = 2 * HC;
        constexpr int KD       = HC * D;
        constexpr float EPS  = EPS_INT * 1e-9;
        constexpr float NEPS = NEPS_INT * 1e-9;

        device float* post_out = (device float*)post2 + row * HC;
        device float* comb_out = (device float*)comb2 + row * HC * HC;

        threadgroup float pre_shared[HC];
        threadgroup float mix_sh[MIX];
        threadgroup float red_sh[8][MIX];
        threadgroup float ssq_sh[8];

        // ================================================================
        // PHASE 0: expand + rmsnorm + mixes in one pass over the row.
        //   h_i[d] = post_i * x[d] + sum_j comb[j][i] * r_j[d], written
        //   T-rounded; the rounded value feeds ssq and the MIX dots so
        //   the math matches the unfused expand -> collapse chain.
        // ================================================================
        {
            const device T*      xr = (const device T*)x_in + row * D;
            const device T*      rr = (const device T*)residual + row * KD;
            device T*            hr = (device T*)h_out + row * KD;
            const device float4* fr = (const device float4*)fn_t;

            // post_in / comb_in are indexed directly: at qL=1 they are
            // small enough that metal_kernel binds them in the constant
            // address space, where a device-pointer alias fails to compile.
            const uint pbase = row * HC;
            const uint cbase = row * HC * HC;
            const float p0 = post_in[pbase + 0];
            const float p1 = post_in[pbase + 1];
            const float p2 = post_in[pbase + 2];
            const float p3 = post_in[pbase + 3];

            float acc[MIX] = {0.0f};
            float ssq = 0.0f;
            for (uint d = tid; d < (uint)D; d += 256) {
                const float xd = (float)xr[d];
                const float r0 = (float)rr[0 * D + d];
                const float r1 = (float)rr[1 * D + d];
                const float r2 = (float)rr[2 * D + d];
                const float r3 = (float)rr[3 * D + d];
                #pragma clang loop unroll(full)
                for (int i = 0; i < HC; ++i) {
                    float y = (i == 0 ? p0 : i == 1 ? p1 : i == 2 ? p2 : p3)
                              * xd;
                    y = metal::fma(comb_in[cbase + 0 * HC + i], r0, y);
                    y = metal::fma(comb_in[cbase + 1 * HC + i], r1, y);
                    y = metal::fma(comb_in[cbase + 2 * HC + i], r2, y);
                    y = metal::fma(comb_in[cbase + 3 * HC + i], r3, y);
                    const T yt = (T)y;
                    hr[i * D + d] = yt;
                    const float yr = (float)yt;
                    ssq = metal::fma(yr, yr, ssq);
                    const device float4* f4 = fr + (i * D + d) * (MIX / 4);
                    #pragma clang loop unroll(full)
                    for (int j = 0; j < MIX / 4; ++j) {
                        float4 f = f4[j];
                        acc[4*j+0] = metal::fma(yr, f.x, acc[4*j+0]);
                        acc[4*j+1] = metal::fma(yr, f.y, acc[4*j+1]);
                        acc[4*j+2] = metal::fma(yr, f.z, acc[4*j+2]);
                        acc[4*j+3] = metal::fma(yr, f.w, acc[4*j+3]);
                    }
                }
            }
            ssq = simd_sum(ssq);
            #pragma clang loop unroll(full)
            for (int j = 0; j < MIX; ++j) {
                acc[j] = simd_sum(acc[j]);
            }
            if (lane == 0) {
                ssq_sh[sg] = ssq;
                for (int j = 0; j < MIX; ++j) {
                    red_sh[sg][j] = acc[j];
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup
                                | mem_flags::mem_device);
            if (sg == 0) {
                float s = (lane < 8) ? ssq_sh[lane] : 0.0f;
                s = simd_sum(s);
                float rrms = metal::precise::rsqrt(s / (float)KD + NEPS);
                if (lane < (uint)MIX) {
                    float tot = 0.0f;
                    for (int g = 0; g < 8; ++g) {
                        tot += red_sh[g][lane];
                    }
                    mix_sh[lane] = tot * rrms;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // ================================================================
        // PHASE 1: Branchless sinkhorn on simd group 0 (see the collapse
        // kernel above for the masking scheme).
        // ================================================================
        if (sg == 0) {
            const float pre_scale  = scale[0];
            const float post_scale = scale[1];
            const float comb_scale = scale[2];

            const float active = (lane < (uint)HC) ? 1.0f : 0.0f;
            const uint  llane  = metal::min(lane, (uint)(HC - 1));

            float pre_z  = mix_sh[llane]      * pre_scale  + base[llane];
            float post_z = mix_sh[HC + llane] * post_scale + base[HC + llane];
            float pre_v  = 1.0f / (1.0f + metal::fast::exp(-pre_z)) + EPS;
            float post_v = 2.0f / (1.0f + metal::fast::exp(-post_z));

            if (lane < (uint)HC) {
                pre_shared[lane] = pre_v;
                post_out[lane]   = post_v;
            }

            float4 m = float4(mix_sh[BASE_OFF + llane * HC + 0],
                              mix_sh[BASE_OFF + llane * HC + 1],
                              mix_sh[BASE_OFF + llane * HC + 2],
                              mix_sh[BASE_OFF + llane * HC + 3]);
            float4 v = (m * comb_scale
                      + *(const device float4*)(base + BASE_OFF + llane * HC))
                     * active;

            float row_max = metal::max(metal::max(v.x, v.y),
                                       metal::max(v.z, v.w));
            float4 e = metal::fast::exp(v - row_max) * active;
            float4 r = e * (1.0f / (e.x + e.y + e.z + e.w + EPS))
                     + EPS * active;

            float4 col_inv = 1.0f / (float4(
                simd_sum(r.x), simd_sum(r.y),
                simd_sum(r.z), simd_sum(r.w)
            ) + EPS);
            r *= col_inv;

            for (int iter = 1; iter < ITERS; ++iter) {
                r *= (1.0f / (r.x + r.y + r.z + r.w + EPS)) * active;
                col_inv = 1.0f / (float4(
                    simd_sum(r.x), simd_sum(r.y),
                    simd_sum(r.z), simd_sum(r.w)
                ) + EPS);
                r *= col_inv;
            }

            if (lane < (uint)HC) {
                *(device float4*)(comb_out + lane * HC) = r;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ================================================================
        // PHASE 2: Collapse from the T-rounded h written in phase 0.
        // ================================================================
        const float q0 = pre_shared[0];
        const float q1 = pre_shared[1];
        const float q2 = pre_shared[2];
        const float q3 = pre_shared[3];

        const device T* h_row  = (const device T*)h_out + row * KD;
        device U*       out_row = (device U*)collapsed + row * D;

        using T4 = vec<T, 4>;
        using U4 = vec<U, 4>;
        const device T4* h_row0 = (const device T4*)(h_row + 0*D);
        const device T4* h_row1 = (const device T4*)(h_row + 1*D);
        const device T4* h_row2 = (const device T4*)(h_row + 2*D);
        const device T4* h_row3 = (const device T4*)(h_row + 3*D);
        device U4*       out4   = (device U4*)out_row;

        constexpr uint D4 = (uint)D / 4;

        for (uint d4 = tid; d4 < D4; d4 += 256) {
            float4 h0 = float4(h_row0[d4]);
            float4 h1 = float4(h_row1[d4]);
            float4 h2 = float4(h_row2[d4]);
            float4 h3 = float4(h_row3[d4]);

            float4 result = fma(float4(q0), h0,
                            fma(float4(q1), h1,
                            fma(float4(q2), h2, float4(q3) * h3)));

            out4[d4] = U4(result);
        }

        #if (D % 4) != 0
        for (uint d = D4 * 4 + tid; d < (uint)D; d += 256) {
            float val = q0*(float)h_row[0*D+d] + q1*(float)h_row[1*D+d]
                      + q2*(float)h_row[2*D+d] + q3*(float)h_row[3*D+d];
            out_row[d] = (U)val;
        }
        #endif
    """

    return mx.fast.metal_kernel(
        name="hc_expand_norm_mix_sinkhorn_collapse",
        input_names=[
            "x_in", "residual", "post_in", "comb_in", "fn_t", "scale", "base"
        ],
        output_names=["h_out", "collapsed", "post2", "comb2"],
        source=source,
        ensure_row_contiguous=True,
    )


_hc_expand_collapse_kernel = _make_hc_expand_collapse_kernel()


def hc_expand_collapse(hc, x, residual, post, comb):
    """hc_expand(x, residual, post, comb) followed by hc(h), fused into
    one dispatch on the GPU kernel path; returns (h, collapsed, post2,
    comb2). Falls back to the eager pair off-GPU or at hc_mult != 4."""
    use_ops = (
        hc.training
        or mx.default_device() != mx.gpu
        or hc.hc_mult != 4
        or not mx.metal.is_available()
    )
    if use_ops:
        h = hc_expand(x, residual, post, comb)
        collapsed, post2, comb2 = hc(h)
        return h, collapsed, post2, comb2

    fn_t = getattr(hc, "_fn_t", None)
    if fn_t is None:
        fn_t = mx.contiguous(hc.fn.T)
        mx.eval(fn_t)
        hc._fn_t = fn_t

    B, L, H, D = residual.shape
    return _hc_expand_collapse_kernel(
        inputs=[x, residual, post, comb, fn_t, hc.scale, hc.base],
        template=[
            ("T", x.dtype),
            ("U", x.dtype),
            ("HC", hc.hc_mult),
            ("ITERS", hc.sinkhorn_iters),
            ("D", D),
            ("EPS_INT", round(hc.hc_eps / 1e-9)),
            ("NEPS_INT", round(hc.norm_eps / 1e-9)),
        ],
        grid=(B * L * 256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[
            (B, L, H, D),
            (B, L, D),
            (B, L, hc.hc_mult),
            (B, L, hc.hc_mult, hc.hc_mult),
        ],
        output_dtypes=[x.dtype, x.dtype, mx.float32, mx.float32],
    )


_KQ_SKINNY = {"ok": None}


def _kq_skinny_available() -> bool:
    ok = _KQ_SKINNY["ok"]
    if ok is None:
        try:
            import mlx_kquant as kq

            ok = (
                mx.metal.is_available()
                and hasattr(kq, "skinny_matmul")
                and os.environ.get("GMLX_KQ_SKINNY", "1") != "0"
            )
        except Exception:  # noqa: BLE001 - optional dependency
            ok = False
        _KQ_SKINNY["ok"] = ok
    return ok



class HyperHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hc_mult = config.hc_mult
        self.norm_eps = config.rms_norm_eps
        self.hc_eps = config.hc_eps
        self.fn = mx.zeros(
            (self.hc_mult, self.hc_mult * config.hidden_size), dtype=mx.float32
        )
        self.base = mx.zeros((self.hc_mult,), dtype=mx.float32)
        self.scale = mx.ones((1,), dtype=mx.float32)

    def __call__(self, x: mx.array):
        y = x.astype(mx.float32)
        z = mx.fast.rms_norm(y.flatten(-2), None, self.norm_eps)
        if (
            2 <= z.shape[-2] <= 16
            and mx.default_device() == mx.gpu
            and _kq_skinny_available()
        ):
            import mlx_kquant as kq

            mixes = kq.skinny_matmul(z, self.fn)
        else:
            fn_t = getattr(self, "_fn_t", None)
            if fn_t is None:
                fn_t = mx.contiguous(self.fn.T)
                mx.eval(fn_t)
                self._fn_t = fn_t
            mixes = z @ fn_t
        pre = mx.sigmoid(mixes * self.scale + self.base) + self.hc_eps
        return (pre[..., None] * y).sum(axis=2).astype(x.dtype)
