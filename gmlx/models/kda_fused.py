# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Asher Feldman
"""Fused single-token KDA decode step (kimi_k3 / glm5_next linear layers).

At M=1 the KDA sublayer is dispatch-latency bound: three short convs (each
a concat, conv1d, silu and state slice), two l2 norms, the decay and beta
elementwise chain, the recurrence kernel, the output RMSNorm and the
sigmoid gate come to roughly 25 small dispatches per layer on top of the
weight matvecs. This kernel folds everything between the input projections
and o_proj into one dispatch per step:

  conv(q,k,v) + silu -> l2norm(q,k) with the folded scales -> per-channel
  decay exp(lb * sigmoid(exp(A_log) * (a + dt_bias))) -> beta = sigmoid(b)
  -> gated delta rule update of the [Dv, Dk] state -> out = state . q
  -> RMSNorm over Dv (weight w) * sigmoid(gate)

and emits the shifted conv tails plus the new recurrent state. One
threadgroup per (batch, head): SG simdgroups split the Dv rows, each lane
owns Dk / 32 key channels, so the Dk reductions are simd_sums and the Dv
reduction for the output norm goes through threadgroup memory. All math is
f32 (the reference evaluation order); the eager path rounds the conv and
norm outputs to bf16 between ops, so results agree to bf16 noise, not
bit-exactly.

Only the plain decode shape is fused (T == 1, no ssm mask, no per-row
lengths). ``GMLX_KDA_FUSED=0`` disables the route.
"""

import os

import mlx.core as mx

_ENABLED = os.environ.get("GMLX_KDA_FUSED", "1") != "0"

_SOURCE = """
    uint tid  = thread_position_in_threadgroup.x;
    uint lane = tid % 32;
    uint sg   = tid / 32;
    uint n    = threadgroup_position_in_grid.z;   // b * H + h
    uint b    = n / H;
    uint h    = n % H;

    constexpr int NPT  = DK / 32;      // key channels per lane
    constexpr int ROWS = DV / SG;      // value rows per simdgroup
    constexpr int C    = H * DK;       // projection width (Dk == Dv)
    constexpr int NS   = KW - 1;       // carried conv taps

    const float lb       = params[0];
    const float scale    = params[1];
    const float l2_eps   = params[2];
    const float norm_eps = params[3];

    // ---- q / k: conv + silu, l2 norm over Dk, decay -------------------
    float qn[NPT], kn[NPT], g[NPT];
    float qss = 0.0f, kss = 0.0f;
    for (int i = 0; i < NPT; ++i) {
        int dk = lane * NPT + i;
        int c  = h * DK + dk;
        float aq = 0.0f, ak = 0.0f;
        for (int t = 0; t < NS; ++t) {
            aq = fma(float(wq[c * KW + t]), float(sq[(b * NS + t) * C + c]), aq);
            ak = fma(float(wk[c * KW + t]), float(sk[(b * NS + t) * C + c]), ak);
        }
        aq = fma(float(wq[c * KW + NS]), float(xq[b * C + c]), aq);
        ak = fma(float(wk[c * KW + NS]), float(xk[b * C + c]), ak);
        aq = aq / (1.0f + metal::exp(-aq));
        ak = ak / (1.0f + metal::exp(-ak));
        qn[i] = aq; kn[i] = ak;
        qss = fma(aq, aq, qss);
        kss = fma(ak, ak, kss);
        float a = float(a_raw[b * C + c]) + dt_bias[c];
        float s = 1.0f / (1.0f + metal::exp(a_folded[h] * a));  // sigmoid(-a_folded * a)
        g[i] = metal::exp(lb * s);
    }
    qss = simd_sum(qss);
    kss = simd_sum(kss);
    float qf = metal::rsqrt(qss / float(DK) + l2_eps) * scale * scale;
    float kf = metal::rsqrt(kss / float(DK) + l2_eps) * scale;
    for (int i = 0; i < NPT; ++i) { qn[i] *= qf; kn[i] *= kf; }

    float beta = 1.0f / (1.0f + metal::exp(-float(b_logit[b * H + h])));

    // ---- delta rule over this simdgroup's Dv rows ---------------------
    threadgroup float outbuf[DV];
    threadgroup float red[SG];
    const device float* st_in  = state_in  + (size_t)(n * DV) * DK;
    device float*       st_out = state_out + (size_t)(n * DV) * DK;
    for (int r = 0; r < ROWS; ++r) {
        int dv = sg * ROWS + r;
        int cv = h * DV + dv;
        float av = 0.0f;
        for (int t = 0; t < NS; ++t)
            av = fma(float(wv[cv * KW + t]), float(sv[(b * NS + t) * C + cv]), av);
        av = fma(float(wv[cv * KW + NS]), float(xv[b * C + cv]), av);
        av = av / (1.0f + metal::exp(-av));

        float s[NPT];
        float kv_mem = 0.0f;
        for (int i = 0; i < NPT; ++i) {
            int dk = lane * NPT + i;
            s[i] = st_in[dv * DK + dk] * g[i];
            kv_mem = fma(s[i], kn[i], kv_mem);
        }
        kv_mem = simd_sum(kv_mem);
        float delta = (av - kv_mem) * beta;
        float o = 0.0f;
        for (int i = 0; i < NPT; ++i) {
            int dk = lane * NPT + i;
            s[i] = fma(kn[i], delta, s[i]);
            o = fma(s[i], qn[i], o);
            st_out[dv * DK + dk] = s[i];
        }
        o = simd_sum(o);
        if (lane == 0) outbuf[dv] = o;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- output RMSNorm over Dv, sigmoid gate --------------------------
    float ss = 0.0f;
    for (int dv = tid; dv < DV; dv += 32 * SG) ss = fma(outbuf[dv], outbuf[dv], ss);
    ss = simd_sum(ss);
    if (lane == 0) red[sg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = 0.0f;
    for (int i = 0; i < SG; ++i) tot += red[i];
    float of = metal::rsqrt(tot / float(DV) + norm_eps);
    for (int dv = tid; dv < DV; dv += 32 * SG) {
        int cv = h * DV + dv;
        float gt = 1.0f / (1.0f + metal::exp(-float(gate[b * C + cv])));
        y[b * C + cv] = T(outbuf[dv] * of * float(w[dv]) * gt);
    }

    // ---- shifted conv tails -------------------------------------------
    if (sg == 0) {
        for (int i = 0; i < NPT; ++i) {
            int c = h * DK + lane * NPT + i;
            for (int t = 0; t < NS; ++t) {
                size_t o_ = (b * NS + t) * C + c;
                if (t + 1 < NS) {
                    size_t i_ = (b * NS + t + 1) * C + c;
                    sq_out[o_] = sq[i_]; sk_out[o_] = sk[i_]; sv_out[o_] = sv[i_];
                } else {
                    sq_out[o_] = xq[b * C + c]; sk_out[o_] = xk[b * C + c];
                    sv_out[o_] = xv[b * C + c];
                }
            }
        }
    }
"""

_KERNEL = None


def _kernel():
    global _KERNEL
    if _KERNEL is None:
        _KERNEL = mx.fast.metal_kernel(
            name="kda_decode_fused",
            input_names=["xq", "xk", "xv", "sq", "sk", "sv", "wq", "wk", "wv",
                         "a_raw", "dt_bias", "a_folded", "b_logit", "gate",
                         "state_in", "w", "params"],
            output_names=["y", "sq_out", "sk_out", "sv_out", "state_out"],
            source=_SOURCE,
            ensure_row_contiguous=True,
        )
    return _KERNEL


def fused_ok(x, mask, cache) -> bool:
    return (
        _ENABLED
        and cache is not None
        and mask is None
        and x.shape[1] == 1
        and getattr(cache, "lengths", None) is None
        and mx.default_device() == mx.gpu
        and mx.metal.is_available()
    )


def kda_decode_fused(
    xq, xk, xv, sq, sk, sv, wq, wk, wv, a_raw, dt_bias, a_folded, b_logit,
    gate, state, w, *, lb: float, scale: float, l2_eps: float,
    norm_eps: float, num_heads: int, head_dim: int, conv_kernel: int,
):
    """One fused decode step. Shapes: xq/xk/xv/a_raw/gate [B, 1, H*D],
    sq/sk/sv [B, KW-1, H*D], wq/wk/wv [H*D, KW, 1], dt_bias [H*D],
    a_folded [H], b_logit [B, 1, H], state [B, H, D, D] f32, w [D].
    Returns (y [B, 1, H*D], sq', sk', sv', state')."""
    B = xq.shape[0]
    H, D, KW = num_heads, head_dim, conv_kernel
    C = H * D
    dtype = xq.dtype
    if sq is None:
        z = mx.zeros((B, KW - 1, C), dtype=dtype)
        sq = sk = sv = z
    if state is None:
        state = mx.zeros((B, H, D, D), dtype=mx.float32)
    sg = 16 if B == 1 else 32
    params = mx.array([lb, scale, l2_eps, norm_eps], dtype=mx.float32)
    outs = _kernel()(
        inputs=[
            xq.reshape(B, C), xk.reshape(B, C), xv.reshape(B, C),
            sq, sk, sv, wq, wk, wv,
            a_raw.reshape(B, C), dt_bias.astype(mx.float32),
            a_folded.astype(mx.float32), b_logit.reshape(B, H),
            gate.reshape(B, C), state.astype(mx.float32), w, params,
        ],
        template=[("T", dtype), ("H", H), ("DK", D), ("DV", D),
                  ("KW", KW), ("SG", sg)],
        grid=(32 * sg, 1, B * H),
        threadgroup=(32 * sg, 1, 1),
        output_shapes=[(B, C), (B, KW - 1, C), (B, KW - 1, C),
                       (B, KW - 1, C), (B, H, D, D)],
        output_dtypes=[dtype, dtype, dtype, dtype, mx.float32],
    )
    y, sq2, sk2, sv2, st2 = outs
    return y.reshape(B, 1, C), sq2, sk2, sv2, st2
