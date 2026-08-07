# Copyright (c) 2026 Apple Inc.
#
# Fused decode-width attention front ops for DeepSeek V4: per-head RMSNorm
# + interleaved (traditional) RoPE in one dispatch, replacing the eager
# rms_norm -> transpose -> mx.fast.rope chain at L == 1 (where the
# transpose is a free reshape). The rope math mirrors mx.fast.rope with a
# freqs array: theta_i = position / freqs[i]; inf-freq entries (the nope
# prefix DeepseekV4RoPE builds for full-head-dim calls) give theta 0 and
# pass values through the norm unrotated.

import os

import mlx.core as mx

# Opt-in: GPU-side certified (oracle-favored, bit-exact at small
# offsets) but a probe-level regression as python-invoked JIT kernels
# (~129 metal_kernel calls/step of host overhead vs the cheap stock
# ops they replace). Revival path: port to mlx-kquant C++ ops.
_ATTN_M1_ENABLED = os.environ.get("GMLX_ATTN_M1_FUSED", "0") == "1"

# One device upload per distinct scaled offset per step: every layer asks
# for the same few values, so a tiny rolling memo kills the per-call
# host-to-device array creation.
_pos_memo = {}


def _pos_array(offset):
    arr = _pos_memo.get(offset)
    if arr is None:
        if len(_pos_memo) > 8:
            _pos_memo.clear()
        arr = mx.array([float(offset)], dtype=mx.float32)
        _pos_memo[offset] = arr
    return arr


def _make_q_norm_rope_kernel():
    if mx.default_device() != mx.gpu or not mx.metal.is_available():
        return None

    source = """
        uint tid = thread_position_in_threadgroup.x;
        uint row = threadgroup_position_in_grid.x;

        constexpr float NEPS = NEPS_INT * 1e-9;
        constexpr uint PAIRS = (uint)D / 2;

        const device T* xr = (const device T*)x_in + row * D;
        device T* orow = (device T*)out + row * D;

        threadgroup float sh[D];
        threadgroup float partial[D / 32];

        float v = (tid < (uint)D) ? (float)xr[tid] : 0.0f;
        float ss = simd_sum(v * v);
        uint lane = tid % 32, sg = tid / 32;
        if (lane == 0) partial[sg] = ss;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float tot = 0.0f;
        for (uint i = 0; i < (uint)(D / 32); ++i) tot += partial[i];
        const float inv = metal::rsqrt(tot / (float)D + NEPS);
        if (tid < (uint)D) sh[tid] = v * inv;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid < PAIRS) {
            float a = sh[2 * tid];
            float b = sh[2 * tid + 1];
            float f = freqs[tid];
            float c = 1.0f, s = 0.0f;
            if (metal::isfinite(f)) {
                float theta = pos[0] / f;
                c = metal::precise::cos(theta);
                s = metal::precise::sin(theta);
            }
            orow[2 * tid]     = (T)(a * c - b * s);
            orow[2 * tid + 1] = (T)(a * s + b * c);
        }
    """
    return mx.fast.metal_kernel(
        name="dsv4_q_norm_rope",
        input_names=["x_in", "freqs", "pos"],
        output_names=["out"],
        source=source,
        ensure_row_contiguous=True,
    )


def _make_kv_norm_rope_kernel():
    """Weighted RMSNorm + traditional rope for the single decode KV row."""
    if mx.default_device() != mx.gpu or not mx.metal.is_available():
        return None

    source = """
        uint tid = thread_position_in_threadgroup.x;
        uint row = threadgroup_position_in_grid.x;

        constexpr float NEPS = NEPS_INT * 1e-9;
        constexpr uint PAIRS = (uint)D / 2;

        const device T* xr = (const device T*)x_in + row * D;
        device T* orow = (device T*)out + row * D;

        threadgroup float sh[D];
        threadgroup float partial[D / 32];

        float v = (tid < (uint)D) ? (float)xr[tid] : 0.0f;
        float ss = simd_sum(v * v);
        uint lane = tid % 32, sg = tid / 32;
        if (lane == 0) partial[sg] = ss;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float tot = 0.0f;
        for (uint i = 0; i < (uint)(D / 32); ++i) tot += partial[i];
        const float inv = metal::rsqrt(tot / (float)D + NEPS);
        if (tid < (uint)D) sh[tid] = v * inv * (float)w[tid];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid < PAIRS) {
            float a = sh[2 * tid];
            float b = sh[2 * tid + 1];
            float f = freqs[tid];
            float c = 1.0f, s = 0.0f;
            if (metal::isfinite(f)) {
                float theta = pos[0] / f;
                c = metal::precise::cos(theta);
                s = metal::precise::sin(theta);
            }
            orow[2 * tid]     = (T)(a * c - b * s);
            orow[2 * tid + 1] = (T)(a * s + b * c);
        }
    """
    return mx.fast.metal_kernel(
        name="dsv4_kv_norm_rope",
        input_names=["x_in", "w", "freqs", "pos"],
        output_names=["out"],
        source=source,
        ensure_row_contiguous=True,
    )


def _make_invrope_regroup_kernel():
    """Inverse rope + o_groups regroup in one dispatch: reads the attention
    output (B, H, L, HD), applies the negated-freqs rotation, and writes
    straight into wo_a's grouped input layout (B, G, L, (H/G)*HD),
    replacing the rope dispatch plus a materialized 5-D permute copy."""
    if mx.default_device() != mx.gpu or not mx.metal.is_available():
        return None

    source = """
        uint gid = thread_position_in_grid.x;
        constexpr uint PAIRS_PER_HEAD = (uint)HD / 2;
        constexpr uint TOTAL = (uint)H * PAIRS_PER_HEAD;
        if (gid >= (uint)ROWS * TOTAL) return;

        uint row = gid / TOTAL;
        uint p   = gid % TOTAL;
        uint h   = p / PAIRS_PER_HEAD;
        uint pp  = p % PAIRS_PER_HEAD;

        const device T* xr = (const device T*)x_in
            + (row * H + h) * (uint)HD;
        float a = (float)xr[2 * pp];
        float b = (float)xr[2 * pp + 1];

        float f = freqs[pp];
        float c = 1.0f, s = 0.0f;
        if (metal::isfinite(f)) {
            float theta = pos[0] / f;
            c = metal::precise::cos(theta);
            s = metal::precise::sin(theta);
        }
        float oa = a * c - b * s;
        float ob = a * s + b * c;

        uint gi   = h / (uint)HG;
        uint hpos = h % (uint)HG;
        device T* orow = (device T*)out
            + ((row * (uint)G + gi) * (uint)HG + hpos) * (uint)HD;
        orow[2 * pp]     = (T)oa;
        orow[2 * pp + 1] = (T)ob;
    """
    return mx.fast.metal_kernel(
        name="dsv4_invrope_regroup",
        input_names=["x_in", "freqs", "pos"],
        output_names=["out"],
        source=source,
        ensure_row_contiguous=True,
    )


_q_norm_rope_kernel = _make_q_norm_rope_kernel()
_kv_norm_rope_kernel = _make_kv_norm_rope_kernel()
_invrope_regroup_kernel = _make_invrope_regroup_kernel()


def invrope_regroup_m1(out, rope, offset, o_groups):
    """Fused inverse rope + grouped regroup for (B, H, 1, HD) attention
    output. Returns (B, G, 1, (H/G)*HD) or None when not applicable."""
    if not _ATTN_M1_ENABLED or _invrope_regroup_kernel is None:
        return None
    B, H, L, HD = out.shape
    if B * L != 1 or HD % 2 or H % o_groups or not isinstance(offset, int):
        return None
    freqs = rope._get_freqs(HD, True)
    if rope.freq_scale != 1:
        offset = offset // rope.freq_scale
    pos = _pos_array(offset)
    HG = H // o_groups
    total = B * L * H * (HD // 2)
    return _invrope_regroup_kernel(
        inputs=[out, freqs, pos],
        template=[("T", out.dtype), ("H", H), ("HD", HD),
                  ("G", o_groups), ("HG", HG), ("ROWS", B * L)],
        grid=(((total + 255) // 256) * 256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(B, o_groups, L, HG * HD)],
        output_dtypes=[out.dtype],
    )[0]


def kv_norm_rope_m1(kv_out, norm_weight, rope, offset, eps):
    """Fused weighted RMSNorm + rope for the (B, L, D) decode KV row.
    Returns (B, 1, L, D) or None when the route does not apply."""
    if not _ATTN_M1_ENABLED or _kv_norm_rope_kernel is None:
        return None
    B, L, D = kv_out.shape
    if B * L != 1 or D % 64 != 0 or not isinstance(offset, int):
        return None
    freqs = rope._get_freqs(D, False)
    if rope.freq_scale != 1:
        offset = offset // rope.freq_scale
    pos = _pos_array(offset)
    out = _kv_norm_rope_kernel(
        inputs=[kv_out, norm_weight, freqs, pos],
        template=[("T", kv_out.dtype), ("D", D),
                  ("NEPS_INT", round(eps / 1e-9))],
        grid=(B * L * D, 1, 1),
        threadgroup=(D, 1, 1),
        output_shapes=[(B, 1, L, D)],
        output_dtypes=[kv_out.dtype],
    )[0]
    return out


def q_norm_rope_m1(q, rope, offset, eps):
    """Fused per-head weightless RMSNorm + rope for (B, 1, H, D) decode
    queries. Returns (B, H, 1, D) or None when the route does not apply."""
    if not _ATTN_M1_ENABLED or _q_norm_rope_kernel is None:
        return None
    B, L, H, D = q.shape
    if B * L != 1 or D % 64 != 0 or not isinstance(offset, int):
        return None
    freqs = rope._get_freqs(D, False)
    if rope.freq_scale != 1:
        offset = offset // rope.freq_scale
    pos = _pos_array(offset)
    out = _q_norm_rope_kernel(
        inputs=[q, freqs, pos],
        template=[("T", q.dtype), ("D", D),
                  ("NEPS_INT", round(eps / 1e-9))],
        grid=(B * L * H * D, 1, 1),
        threadgroup=(D, 1, 1),
        output_shapes=[(B, H, L, D)],
        output_dtypes=[q.dtype],
    )[0]
    return out
