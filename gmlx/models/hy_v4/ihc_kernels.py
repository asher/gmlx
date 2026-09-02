# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Asher Feldman
# Portions copyright (c) 2026 Apple Inc. (mlx-lm PR 1192 hyper-connection
# kernels, MIT), by way of gmlx.models.deepseek_v4.hyper_connection.
"""Fused Metal kernels for the HY4 iHC cycle.

Two dispatches replace the ops path's eight. ``front_collapse`` reads the
[hc, D] stream row once for the mix dots and the sum of squares together
(rms_norm commutes with the mix matmul, so the rms factor scales the dots
at the end), then collapses the streams and folds the sublayer RMSNorm
into a single rounding. ``expand`` writes the streams back.

The kernels are not bit-identical to the ops path: the reduction order
differs and the fused norm rounds once where the ops path rounds twice.
Both differences move the result toward the fp32 reference, which
``tests/models/test_hy_v4_ihc_kernels.py`` asserts.

The mix matrix has 2*hc rows here against deepseek_v4's (2+hc)*hc, so the
mlx-kquant ports of that family do not apply: they fix the row count at 24
and reaching them needs a zero-padded mix matrix, whose extra resident
bytes and extra reads cost more than the ports' lower per-call overhead
returns.
"""

import os

import mlx.core as mx

_HC_ENABLED = os.environ.get("GMLX_HY4_IHC_KERNEL", "1") != "0"


def _make_front_collapse_kernel():
    if mx.default_device() != mx.gpu or not mx.metal.is_available():
        return None

    source = """
        uint tid  = thread_position_in_threadgroup.x;
        uint row  = threadgroup_position_in_grid.x;
        uint lane = tid % 32;
        uint sg   = tid / 32;

        constexpr int MIX = 2 * HC;
        constexpr int KD  = HC * D;
        constexpr float EPS  = EPS_INT * 1e-9;
        constexpr float NEPS = NEPS_INT * 1e-9;
        constexpr float MAG  = MAG_INT * 1e-6;

        device float* post_out = (device float*)post + row * HC;

        threadgroup float pre_sh[HC];
        threadgroup float mix_sh[MIX];
        threadgroup float red_sh[8][MIX];
        threadgroup float ssq_sh[8];
        threadgroup float inv_sh[1];

        // Mix dots and the row sum of squares in one pass: mixes =
        // rms_norm(y) @ fn_t = rrms * (y @ fn_t).
        {
            const device T* xr = (const device T*)x_in + row * KD;
            const device float4* fr = (const device float4*)fn_t;

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

        if (sg == 0 && lane < (uint)HC) {
            float pre_z  = mix_sh[lane]      * scale[0] + base[lane];
            float post_z = mix_sh[HC + lane] * scale[1] + base[HC + lane];
            pre_sh[lane] = 1.0f / (1.0f + metal::precise::exp(-pre_z)) + EPS;
            post_out[lane] =
                MAG / (1.0f + metal::precise::exp(-post_z)) + EPS;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const float p0 = pre_sh[0];
        const float p1 = pre_sh[1];
        const float p2 = pre_sh[2];
        const float p3 = pre_sh[3];

        const device T* x_row = (const device T*)x_in + row * KD;
        device U* out_row = (device U*)collapsed + row * D;

        using T4 = vec<T, 4>;
        using U4 = vec<U, 4>;
        const device T4* x_row0 = (const device T4*)(x_row + 0*D);
        const device T4* x_row1 = (const device T4*)(x_row + 1*D);
        const device T4* x_row2 = (const device T4*)(x_row + 2*D);
        const device T4* x_row3 = (const device T4*)(x_row + 3*D);
        device U4* out4 = (device U4*)out_row;

        constexpr uint D4 = (uint)D / 4;
        constexpr uint CHUNKS = (D4 + 255) / 256;

        // The collapsed row stays in registers between the two passes the
        // sublayer norm needs, so the second pass costs no reload.
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
        if (lane == 0) ssq_sh[sg] = ssq;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0) {
            float v = (lane < 8) ? ssq_sh[lane] : 0.0f;
            v = simd_sum(v);
            if (lane == 0) {
                inv_sh[0] = metal::precise::rsqrt(v / (float)D + NEPS);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float inv = inv_sh[0];

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
        name="ihc_front_collapse",
        input_names=["x_in", "fn_t", "scale", "base", "w"],
        output_names=["collapsed", "post"],
        source=source,
        ensure_row_contiguous=True,
    )


def _make_expand_kernel():
    if mx.default_device() != mx.gpu or not mx.metal.is_available():
        return None

    source = """
        uint tid = thread_position_in_threadgroup.x;
        uint tg  = threadgroup_position_in_grid.x;
        uint row = tg / NTG;
        uint sub = tg % NTG;

        const device T* xr = (const device T*)x_in + row * D;
        const device T* rr = (const device T*)resid + row * HC * D;
        device T* orow = (device T*)out + row * HC * D;

        const uint pb = row * HC;
        float p0 = post[pb + 0], p1 = post[pb + 1];
        float p2 = post[pb + 2], p3 = post[pb + 3];

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
            o04[k] = T4(fma(float4(p0), xv, float4(r04[k])));
            o14[k] = T4(fma(float4(p1), xv, float4(r14[k])));
            o24[k] = T4(fma(float4(p2), xv, float4(r24[k])));
            o34[k] = T4(fma(float4(p3), xv, float4(r34[k])));
        }
    """
    return mx.fast.metal_kernel(
        name="ihc_expand",
        input_names=["x_in", "resid", "post"],
        output_names=["out"],
        source=source,
        ensure_row_contiguous=True,
    )


_front_collapse_kernel = _make_front_collapse_kernel()
_expand_kernel = _make_expand_kernel()

_NTG = 2


def eligible(x: mx.array, hc_mult: int) -> bool:
    """The kernels unroll 4 streams and load the row as float4."""
    return (
        _HC_ENABLED
        and _front_collapse_kernel is not None
        and mx.default_device() == mx.gpu
        and hc_mult == 4
        and x.ndim == 4
        and x.shape[-1] % (4 * _NTG) == 0
        and x.dtype in (mx.float16, mx.bfloat16)
    )


def front_collapse(x, fn_t, scale, base, w, hc_eps, norm_eps, magnitude):
    """``x`` [B, L, hc, D] -> (rms_norm(collapse(x), w), post [B, L, hc])."""
    B, L, H, D = x.shape
    return _front_collapse_kernel(
        inputs=[x, fn_t, scale, base, w],
        template=[
            ("T", x.dtype),
            ("U", x.dtype),
            ("HC", H),
            ("D", D),
            ("EPS_INT", round(hc_eps / 1e-9)),
            ("NEPS_INT", round(norm_eps / 1e-9)),
            ("MAG_INT", round(magnitude / 1e-6)),
        ],
        grid=(B * L * 256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(B, L, D), (B, L, H)],
        output_dtypes=[x.dtype, mx.float32],
    )


def expand(y, residual, post):
    """``residual`` [B, L, hc, D] + ``post`` [B, L, hc] * ``y`` [B, L, D]."""
    B, L, D = y.shape
    H = residual.shape[2]
    return _expand_kernel(
        inputs=[y, residual, post],
        template=[("T", y.dtype), ("HC", H), ("D", D), ("NTG", _NTG)],
        grid=(B * L * _NTG * 256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(B, L, H, D)],
        output_dtypes=[y.dtype],
    )[0]
