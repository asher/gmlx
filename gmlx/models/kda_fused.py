# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Asher Feldman
"""Fused KDA decode step (kimi_k3 / glm5_next linear layers), one token or
a short block of tokens per dispatch.

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

The simdgroup's state rows are loaded into registers up front (one float4
per lane per row at head dim 128) and stay there across the NT tokens of
a block, so an MTP verify block of NT tokens is one dispatch: the state is
read and written once, the conv taps for a token come from the carried
tails or the block's earlier tokens, and the state after every token but
the last is written to a side output for the verify sink. The single-token
kernel is the NT = 1 instance. Latency dominates this dispatch (the state
pass is under 15 us at bandwidth), so the block form roughly halves the
KDA cost of a two-token verify.

Only the decode shape is fused (T <= max_t, no ssm mask, no per-row
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

    threadgroup float outbuf[DV];
    threadgroup float red[SG];
    const device float* st_in  = state_in  + (size_t)(n * DV) * DK;
    device float*       st_out = state_out + (size_t)(n * DV) * DK;

    // This simdgroup's state rows stay in registers for the whole block;
    // every row's load is issued before any token's math.
    __ROWS_DECL__

    for (int t = 0; t < NT; ++t) {
        // ---- q / k: conv (carried tails, then the block's earlier
        // tokens), silu, l2 norm over Dk, per-channel decay ----------
        float qn[NPT], kn[NPT], g[NPT];
        float qss = 0.0f, kss = 0.0f;
        for (int i = 0; i < NPT; ++i) {
            int dk = lane * NPT + i;
            int c  = h * DK + dk;
            float aq = 0.0f, ak = 0.0f;
            for (int j = 0; j < NS; ++j) {
                int m = t - NS + j;
                float vq = (m < 0) ? float(sq[(b * NS + NS + m) * C + c])
                                   : float(xq[(b * NT + m) * C + c]);
                float vk = (m < 0) ? float(sk[(b * NS + NS + m) * C + c])
                                   : float(xk[(b * NT + m) * C + c]);
                aq = fma(float(wq[c * KW + j]), vq, aq);
                ak = fma(float(wk[c * KW + j]), vk, ak);
            }
            aq = fma(float(wq[c * KW + NS]), float(xq[(b * NT + t) * C + c]), aq);
            ak = fma(float(wk[c * KW + NS]), float(xk[(b * NT + t) * C + c]), ak);
            aq = aq / (1.0f + metal::exp(-aq));
            ak = ak / (1.0f + metal::exp(-ak));
            qn[i] = aq; kn[i] = ak;
            qss = fma(aq, aq, qss);
            kss = fma(ak, ak, kss);
            float a = float(a_raw[(b * NT + t) * C + c]) + dt_bias[c];
            float s_ = 1.0f / (1.0f + metal::exp(a_folded[h] * a));  // sigmoid(-a_folded * a)
            g[i] = metal::exp(lb * s_);
        }
        qss = simd_sum(qss);
        kss = simd_sum(kss);
        float qf = metal::rsqrt(qss / float(DK) + l2_eps) * scale * scale;
        float kf = metal::rsqrt(kss / float(DK) + l2_eps) * scale;
        for (int i = 0; i < NPT; ++i) { qn[i] *= qf; kn[i] *= kf; }
        float beta = 1.0f / (1.0f + metal::exp(-float(b_logit[(b * NT + t) * H + h])));

        // ---- v conv + silu for this simdgroup's rows -----------------
        float av[ROWS];
        for (int r = 0; r < ROWS; ++r) {
            int cv = h * DV + sg * ROWS + r;
            float a = 0.0f;
            for (int j = 0; j < NS; ++j) {
                int m = t - NS + j;
                float vv = (m < 0) ? float(sv[(b * NS + NS + m) * C + cv])
                                   : float(xv[(b * NT + m) * C + cv]);
                a = fma(float(wv[cv * KW + j]), vv, a);
            }
            a = fma(float(wv[cv * KW + NS]), float(xv[(b * NT + t) * C + cv]), a);
            av[r] = a / (1.0f + metal::exp(-a));
        }

        // ---- delta rule over the register rows -----------------------
        __DELTA_RULE__
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ---- output RMSNorm over Dv, sigmoid gate --------------------
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
            float gt = 1.0f / (1.0f + metal::exp(-float(gate[(b * NT + t) * C + cv])));
            y[(b * NT + t) * C + cv] = T(outbuf[dv] * of * float(w[dv]) * gt);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    __ROWS_STORE__

    // ---- conv tails after the block ----------------------------------
    if (sg == 0) {
        for (int i = 0; i < NPT; ++i) {
            int c = h * DK + lane * NPT + i;
            for (int j = 0; j < NS; ++j) {
                int m = NT - NS + j;
                size_t o_ = (b * NS + j) * C + c;
                if (m < 0) {
                    size_t i_ = (b * NS + NS + m) * C + c;
                    sq_out[o_] = sq[i_]; sk_out[o_] = sk[i_]; sv_out[o_] = sv[i_];
                } else {
                    size_t i_ = (b * NT + m) * C + c;
                    sq_out[o_] = xq[i_]; sk_out[o_] = xk[i_]; sv_out[o_] = xv[i_];
                }
            }
        }
    }
"""

# Head dim 128: one float4 of state per lane per row (the vector loads
# and dots are what make the block form pay; the scalar form is the same
# math for other head dims).
_VEC = {
    "__ROWS_DECL__": """float4 s[ROWS];
    for (int r = 0; r < ROWS; ++r)
        s[r] = ((const device float4*)(st_in + (sg * ROWS + r) * DK))[lane];""",
    "__DELTA_RULE__": """float4 gv = float4(g[0], g[1], g[2], g[3]);
        float4 kv = float4(kn[0], kn[1], kn[2], kn[3]);
        float4 qv = float4(qn[0], qn[1], qn[2], qn[3]);
        for (int r = 0; r < ROWS; ++r) {
            int dv = sg * ROWS + r;
            s[r] *= gv;
            float kv_mem = simd_sum(dot(s[r], kv));
            float delta = (av[r] - kv_mem) * beta;
            s[r] = fma(kv, float4(delta), s[r]);
            float o = simd_sum(dot(s[r], qv));
            if (lane == 0) outbuf[dv] = o;
            if (t + 1 < NT)
                ((device float4*)(state_mid + ((size_t)t * B * H + n) * DV * DK + dv * DK))[lane] = s[r];
        }""",
    "__ROWS_STORE__": """for (int r = 0; r < ROWS; ++r)
        ((device float4*)(st_out + (sg * ROWS + r) * DK))[lane] = s[r];""",
}

_SCALAR = {
    "__ROWS_DECL__": """float s[ROWS][NPT];
    for (int r = 0; r < ROWS; ++r)
        for (int i = 0; i < NPT; ++i)
            s[r][i] = st_in[(sg * ROWS + r) * DK + lane * NPT + i];""",
    "__DELTA_RULE__": """for (int r = 0; r < ROWS; ++r) {
            int dv = sg * ROWS + r;
            float kv_mem = 0.0f;
            for (int i = 0; i < NPT; ++i) {
                s[r][i] *= g[i];
                kv_mem = fma(s[r][i], kn[i], kv_mem);
            }
            kv_mem = simd_sum(kv_mem);
            float delta = (av[r] - kv_mem) * beta;
            float o = 0.0f;
            for (int i = 0; i < NPT; ++i) {
                s[r][i] = fma(kn[i], delta, s[r][i]);
                o = fma(s[r][i], qn[i], o);
            }
            o = simd_sum(o);
            if (lane == 0) outbuf[dv] = o;
            if (t + 1 < NT) {
                device float* mid = state_mid + ((size_t)t * B * H + n) * DV * DK + dv * DK;
                for (int i = 0; i < NPT; ++i) mid[lane * NPT + i] = s[r][i];
            }
        }""",
    "__ROWS_STORE__": """for (int r = 0; r < ROWS; ++r)
        for (int i = 0; i < NPT; ++i)
            st_out[(sg * ROWS + r) * DK + lane * NPT + i] = s[r][i];""",
}

_KERNELS = {}


def _kernel(vec: bool):
    if vec not in _KERNELS:
        src = _SOURCE
        for k, v in (_VEC if vec else _SCALAR).items():
            src = src.replace(k, v)
        _KERNELS[vec] = mx.fast.metal_kernel(
            name="kda_decode_fused_block" + ("" if vec else "_s"),
            input_names=["xq", "xk", "xv", "sq", "sk", "sv", "wq", "wk", "wv",
                         "a_raw", "dt_bias", "a_folded", "b_logit", "gate",
                         "state_in", "w", "params"],
            output_names=["y", "sq_out", "sk_out", "sv_out", "state_out",
                          "state_mid"],
            source=src,
            ensure_row_contiguous=True,
        )
    return _KERNELS[vec]


def fused_ok(x, mask, cache, max_t: int = 1) -> bool:
    return (
        _ENABLED
        and cache is not None
        and mask is None
        and x.shape[1] <= max_t
        and getattr(cache, "lengths", None) is None
        and mx.default_device() == mx.gpu
        and mx.metal.is_available()
    )


def kda_decode_fused_block(
    xq, xk, xv, sq, sk, sv, wq, wk, wv, a_raw, dt_bias, a_folded, b_logit,
    gate, state, w, *, lb: float, scale: float, l2_eps: float,
    norm_eps: float, num_heads: int, head_dim: int, conv_kernel: int,
):
    """One fused decode block of NT tokens. Shapes: xq/xk/xv/a_raw/gate
    [B, NT, H*D], sq/sk/sv [B, KW-1, H*D], wq/wk/wv [H*D, KW, 1], dt_bias
    [H*D], a_folded [H], b_logit [B, NT, H], state [B, H, D, D] f32, w [D].
    Returns (y [B, NT, H*D], sq', sk', sv', state' after the block,
    states_mid [NT-1, B, H, D, D] f32: the state after each token but the
    last, for the verify sink)."""
    B, NT, C = xq.shape
    H, D, KW = num_heads, head_dim, conv_kernel
    assert C == H * D
    dtype = xq.dtype
    if sq is None:
        z = mx.zeros((B, KW - 1, C), dtype=dtype)
        sq = sk = sv = z
    if state is None:
        state = mx.zeros((B, H, D, D), dtype=mx.float32)
    sg = 16 if B == 1 else 32
    params = mx.array([lb, scale, l2_eps, norm_eps], dtype=mx.float32)
    outs = _kernel(D == 128)(
        inputs=[
            xq, xk, xv, sq, sk, sv, wq, wk, wv,
            a_raw, dt_bias.astype(mx.float32),
            a_folded.astype(mx.float32), b_logit.reshape(B, NT, H),
            gate, state.astype(mx.float32), w, params,
        ],
        template=[("T", dtype), ("B", B), ("H", H), ("DK", D), ("DV", D),
                  ("KW", KW), ("SG", sg), ("NT", NT)],
        grid=(32 * sg, 1, B * H),
        threadgroup=(32 * sg, 1, 1),
        output_shapes=[(B, NT, C), (B, KW - 1, C), (B, KW - 1, C),
                       (B, KW - 1, C), (B, H, D, D),
                       (NT - 1, B, H, D, D) if NT > 1 else (1,)],
        output_dtypes=[dtype, dtype, dtype, dtype, mx.float32, mx.float32],
    )
    return outs


def kda_decode_fused(
    xq, xk, xv, sq, sk, sv, wq, wk, wv, a_raw, dt_bias, a_folded, b_logit,
    gate, state, w, *, lb: float, scale: float, l2_eps: float,
    norm_eps: float, num_heads: int, head_dim: int, conv_kernel: int,
):
    """One fused decode step (the NT = 1 block). Shapes: xq/xk/xv/a_raw/gate
    [B, 1, H*D], sq/sk/sv [B, KW-1, H*D], wq/wk/wv [H*D, KW, 1], dt_bias
    [H*D], a_folded [H], b_logit [B, 1, H], state [B, H, D, D] f32, w [D].
    Returns (y [B, 1, H*D], sq', sk', sv', state')."""
    B = xq.shape[0]
    C = num_heads * head_dim
    y, sq2, sk2, sv2, st2, _ = kda_decode_fused_block(
        xq.reshape(B, 1, C), xk.reshape(B, 1, C), xv.reshape(B, 1, C),
        sq, sk, sv, wq, wk, wv, a_raw.reshape(B, 1, C), dt_bias, a_folded,
        b_logit.reshape(B, 1, num_heads), gate.reshape(B, 1, C), state, w,
        lb=lb, scale=scale, l2_eps=l2_eps, norm_eps=norm_eps,
        num_heads=num_heads, head_dim=head_dim, conv_kernel=conv_kernel)
    return y, sq2, sk2, sv2, st2
