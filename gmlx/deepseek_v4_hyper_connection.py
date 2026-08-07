# Copyright (c) 2026 Apple Inc.
#
# Manifold-constrained hyper-connections for DeepSeek V4 (vendored).
# Copied near-verbatim from mlx-lm PR 1192 by way of omlx's
# patches/deepseek_v4/hyper_connection.py port (fused Metal
# sinkhorn+collapse kernel via mx.fast.metal_kernel with a compiled
# pure-ops fallback for CPU/training). Registered into the
# mlx_lm.models namespace by deepseek_v4_model.ensure_registered().
# Delete once installed mlx-lm ships models/hyper_connection.py.

from typing import Tuple

import mlx.core as mx
import mlx.nn as nn


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
        fn_t = getattr(self, "_fn_t", None)
        if fn_t is None:
            fn_t = mx.contiguous(self.fn.T)
            mx.eval(fn_t)
            self._fn_t = fn_t
        mixes = z @ fn_t
        pre = mx.sigmoid(mixes * self.scale + self.base) + self.hc_eps
        return (pre[..., None] * y).sum(axis=2).astype(x.dtype)
