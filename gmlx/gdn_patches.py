"""Runtime patches for the gated-delta (Qwen3-Next / Qwen3.5) family.

Split-GDN module fixup, the fused decode/verify Metal kernels, the qwen3.5
MTP-verify guards, and the tiled V-head K mapping rewrite. Each installer is
install-once and kill-switchable via its GMLX_* env flag.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


from . import loadlog
from .envflags import env_bool
from .patching import ClassPatch


# GGUF V-head tiling fixup (runtime patch)


def _needs_tiled_v_patch(config: dict) -> bool:
    """Check for asymmetric linear-attention K/V heads.

    convert_hf_to_gguf reorders V heads from grouped to tiled when
    ``linear_num_key_heads != linear_num_value_heads``. mlx-lm's gated_delta
    assumes grouped layout, so we must patch the K->V mapping.

    Note: the synthesized config also carries ``kv_head_layout="tiled"`` for
    forward-compat, but the *installed* mlx-lm has no consumer for that field -
    this monkey-patch is the load-bearing mechanism today.

    qwen3_next is EXCLUDED despite its asymmetric heads (16 k / 32 v): its
    GGUFs keep V heads in HF grouped order (the legacy fused ssm_in layout is
    the raw HF tensor, and the newer split converter's de-interleave preserves
    group order), and mlx-lm's qwen3_next consumes grouped directly - the
    tiled fixup would corrupt its K->V mapping.
    """
    if config.get("model_type") == "qwen3_next":
        return False
    k = config.get("linear_num_key_heads", 0)
    v = config.get("linear_num_value_heads", 0)
    return k > 0 and v > 0 and k != v


def _patch_qwen3next_split_gdn(model) -> None:
    """Restructure qwen3_next GDN modules for the split GGUF wire layout.

    The current llama.cpp converter de-interleaves each gated-delta
    ``in_proj_qkvz`` into ``attn_qkv`` (q|k|v rows, flat head-major with group
    order preserved) and ``attn_gate`` (z). mlx-lm's class wants the fused
    per-k-head-interleaved tensor; re-fusing would be a row re-interleave on
    quantized blocks, so instead each GDN gets split ``in_proj_qkv`` /
    ``in_proj_z`` Linears (the qwen3.5 names the canonical remap targets) and
    a forward that skips the runtime de-interleave: the split projection's
    output already is the ``mixed_qkv`` concat the stock forward rebuilds, and
    z comes straight out in ``(B, S, num_v_heads, head_v_dim)`` group order.
    ``in_proj_ba`` stays fused on both sides and keeps its stock split.

    Per-instance and structural: ``__class__`` is swapped to a subclass of the
    stock GDN; no mlx-lm module globals are touched (unlike the tiled-V
    patch), so it cannot leak into qwen3.5/mlx-vlm loads in the same process.
    Must run before sanitize/leaf-swap so the remapped weights land on the
    new modules and the fused module leaves no unfilled params.
    """
    from mlx_lm.models.gated_delta import gated_delta_update
    from mlx_lm.models.qwen3_next import Qwen3NextGatedDeltaNet

    class _SplitGDN(Qwen3NextGatedDeltaNet):
        def __call__(self, inputs, mask=None, cache=None):
            B, S, _ = inputs.shape
            mixed_qkv = self.in_proj_qkv(inputs)
            z = self.in_proj_z(inputs).reshape(B, S, -1, self.head_v_dim)
            mixed_ba = self.in_proj_ba(inputs).reshape(B, S, self.num_k_heads, -1)
            b, a = mx.split(mixed_ba, [self.num_v_heads // self.num_k_heads], axis=-1)
            b = b.reshape(B, S, self.num_v_heads)
            a = a.reshape(B, S, self.num_v_heads)

            # From here on: byte-identical to the stock __call__ tail.
            if cache is not None and cache[0] is not None:
                conv_state = cache[0]
            else:
                conv_state = mx.zeros(
                    (B, self.conv_kernel_size - 1, self.conv_dim),
                    dtype=inputs.dtype,
                )

            if mask is not None:
                mixed_qkv = mx.where(mask[..., None], mixed_qkv, 0)
            conv_input = mx.concatenate([conv_state, mixed_qkv], axis=1)

            if cache is not None:
                n_keep = self.conv_kernel_size - 1
                if cache.lengths is not None:
                    ends = mx.clip(cache.lengths, 0, S)
                    positions = (ends[:, None] + mx.arange(n_keep))[..., None]
                    cache[0] = mx.take_along_axis(conv_input, positions, axis=1)
                else:
                    cache[0] = mx.contiguous(conv_input[:, -n_keep:, :])

            conv_out = nn.silu(self.conv1d(conv_input))

            q, k, v = [
                t.reshape(B, S, h, d)
                for t, h, d in zip(
                    mx.split(conv_out, [self.key_dim, 2 * self.key_dim], -1),
                    [self.num_k_heads, self.num_k_heads, self.num_v_heads],
                    [self.head_k_dim, self.head_k_dim, self.head_v_dim],
                )
            ]

            state = cache[1] if cache else None
            inv_scale = k.shape[-1] ** -0.5
            q = (inv_scale**2) * mx.fast.rms_norm(q, None, 1e-6)
            k = inv_scale * mx.fast.rms_norm(k, None, 1e-6)

            out, state = gated_delta_update(
                q,
                k,
                v,
                a,
                b,
                self.A_log,
                self.dt_bias,
                state,
                mask,
                use_kernel=not self.training,
            )

            if cache is not None:
                cache[1] = state
                cache.advance(S)

            out = self.norm(out, z)
            return self.out_proj(out.reshape(B, S, -1))

    n = 0
    for m in model.modules():
        if not isinstance(m, Qwen3NextGatedDeltaNet):
            continue
        m.in_proj_qkv = nn.Linear(
            m.hidden_size, 2 * m.key_dim + m.value_dim, bias=False
        )
        m.in_proj_z = nn.Linear(m.hidden_size, m.value_dim, bias=False)
        del m.in_proj_qkvz  # leaves no unfilled params for strict-load
        m.__class__ = _SplitGDN
        n += 1
    loadlog.verbose_print(
        f"[patch] qwen3_next: split GDN in_proj (attn_qkv/attn_gate wire "
        f"layout) on {n} layers"
    )


# Fused gated-delta decode step (runtime patch)
#
# The gated-delta decode step (S=1) is a strictly serial chain of tiny ops:
# depthwise causal conv (K taps) -> silu -> split q/k/v -> rms_norm(q) -> rms_norm(k)
# -> compute g/beta -> single recurrent scan step -> gated output norm. Each op is
# a separate Metal launch whose launch latency sits on the critical path.
#
# Fusing the whole chain into one kernel collapses ~6 serial launches into one.
# The win is launch-latency-bound, so it only materialises when the surrounding
# per-layer matmuls are too cheap to hide those launches behind -- i.e. on MoE
# models, whose decode activates only a few experts per token (sparse, small
# matmuls). On dense models the full per-layer matmuls already overlap the chain's
# launches, so the fused kernel is a wash there and is left opt-in per instance.
#
# The kernel runs the conv/silu/norms in fp32 and matches the recurrence op order
# of the stock scan kernel; it is fp32-accurate (it tracks an fp32 reference to
# ~1e-7 on the state) rather than bit-identical to the stock bf16-intermediate
# path, so greedy output can differ from stock only at near-ties.

_GDN_FUSED_DECODE_SRC = r"""
    constexpr int n_per_t = Dk / 32;
    constexpr int key_dim = Hk * Dk;
    constexpr int value_dim = Hv * Dv;
    constexpr int conv_dim = 2 * key_dim + value_dim;
    constexpr int n_dv = Dv / SG;
    const float inv_scale = rsqrt((float)Dk);

    uint lane = thread_position_in_threadgroup.x;
    uint sg   = thread_position_in_threadgroup.y;
    uint n = thread_position_in_grid.z;
    uint b_idx = n / Hv;
    uint hv_idx = n % Hv;
    uint hk_idx = TILED ? (hv_idx % Hk) : (hv_idx / (Hv / Hk));

    threadgroup float sumsq_sh[SG];

    float ab = (float)a[b_idx * Hv + hv_idx] + dt_bias[hv_idx];
    float sp = ab > 20.0f ? ab : log(1.0f + exp(ab));
    float g = exp(-exp(A_log[hv_idx]) * sp);
    float beta = 1.0f / (1.0f + exp(-(float)b[b_idx * Hv + hv_idx]));

    auto ci = conv_input + b_idx * K_size * conv_dim;
    float qn[n_per_t], kn[n_per_t];
    float qsq = 0.0f, ksq = 0.0f;
    for (int i = 0; i < n_per_t; ++i) {
        int dk = n_per_t * lane + i;
        int qch = hk_idx * Dk + dk;
        int kch = key_dim + hk_idx * Dk + dk;
        float qa = 0.0f, ka = 0.0f;
        for (int w = 0; w < K_size; ++w) {
            qa += (float)ci[w * conv_dim + qch] * conv_weight[w * conv_dim + qch];
            ka += (float)ci[w * conv_dim + kch] * conv_weight[w * conv_dim + kch];
        }
        if (CONV_BF16) { qa = (float)(InT)qa; ka = (float)(InT)ka; }
        float qs = qa * (1.0f / (1.0f + exp(-qa)));
        float ks = ka * (1.0f / (1.0f + exp(-ka)));
        qn[i] = qs; kn[i] = ks; qsq += qs * qs; ksq += ks * ks;
    }
    qsq = simd_sum(qsq); ksq = simd_sum(ksq);
    float qscale = inv_scale * inv_scale * rsqrt(qsq / (float)Dk + 1e-6f);
    float kscale = inv_scale * rsqrt(ksq / (float)Dk + 1e-6f);
    for (int i = 0; i < n_per_t; ++i) { qn[i] *= qscale; kn[i] *= kscale; }

    float my_out[n_dv];
    float my_sumsq = 0.0f;
    for (int td = 0; td < n_dv; ++td) {
        int dv = sg * n_dv + td;
        auto i_state = state_in + ((uint)(n * Dv + dv)) * Dk;
        auto o_state = state_out + ((uint)(n * Dv + dv)) * Dk;
        float state[n_per_t];
        for (int i = 0; i < n_per_t; ++i) state[i] = (float)i_state[n_per_t * lane + i];
        int vch = 2 * key_dim + hv_idx * Dv + dv;
        float va = 0.0f;
        for (int w = 0; w < K_size; ++w)
            va += (float)ci[w * conv_dim + vch] * conv_weight[w * conv_dim + vch];
        if (CONV_BF16) va = (float)(InT)va;
        float v_val = va * (1.0f / (1.0f + exp(-va)));
        float kv_mem = 0.0f;
        for (int i = 0; i < n_per_t; ++i) { state[i] *= g; kv_mem += state[i] * kn[i]; }
        kv_mem = simd_sum(kv_mem);
        float delta = (v_val - kv_mem) * beta;
        float out = 0.0f;
        for (int i = 0; i < n_per_t; ++i) { state[i] += kn[i] * delta; out += state[i] * qn[i]; }
        out = simd_sum(out);
        for (int i = 0; i < n_per_t; ++i) o_state[n_per_t * lane + i] = (StT)state[i];
        my_out[td] = out;
        if (lane == 0) my_sumsq += out * out;
    }

    if (lane == 0) sumsq_sh[sg] = my_sumsq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = 0.0f;
    for (int s = 0; s < SG; ++s) tot += sumsq_sh[s];
    float rms_denom = rsqrt(tot / (float)Dv + 1e-6f);

    if (lane == 0) {
        for (int td = 0; td < n_dv; ++td) {
            int dv = sg * n_dv + td;
            float xn = my_out[td] * rms_denom * (float)norm_weight[dv];
            float zv = (float)z[(b_idx * Hv + hv_idx) * Dv + dv];
            float gate = zv * (1.0f / (1.0f + exp(-zv)));
            y[(b_idx * Hv + hv_idx) * Dv + dv] = (InT)(gate * xn);
        }
    }
"""

_gdn_fused_decode_kernel = (
    mx.fast.metal_kernel(
        name="gmlx_gated_delta_fused_decode",
        input_names=[
            "conv_input",
            "conv_weight",
            "a",
            "b",
            "A_log",
            "dt_bias",
            "state_in",
            "z",
            "norm_weight",
        ],
        output_names=["y", "state_out"],
        source=_GDN_FUSED_DECODE_SRC,
    )
    if mx.metal.is_available()
    else None
)

_FUSED_DECODE_PATCH = ClassPatch()


def _gdn_fused_decode_body(self, inputs, cache, *, vlm_cache_advance=False):
    """Shared fused S=1 gated-delta decode step. mlx-lm's ``GatedDeltaNet`` and
    mlx-vlm's ``Qwen3_5GatedDeltaNet`` share the attribute + cache layout, so both
    the text path and the (non-MTP) decode through the mlx-vlm target reuse this:
    the conv -> silu -> q/k rms-norm -> recurrent scan -> gated output norm launch
    chain collapses into one kernel. Caller guarantees ``S == 1``, a recurrent
    state in ``cache[1]``, no array mask, and head dims that tile the kernel grid.

    ``vlm_cache_advance`` replays the left-padding/lengths advance bookkeeping the
    stock mlx-vlm decode maintains alongside ``cache.advance`` (the mlx-lm cache
    has no such metadata, so the text path leaves it False)."""
    B, S, _ = inputs.shape
    Dv = self.head_v_dim
    SG = 16 if B == 1 else 32

    qkv = self.in_proj_qkv(inputs)
    zba_w = getattr(self, "_gdn_zba_weight", None)
    if zba_w is not None:
        zba = inputs @ zba_w.T
        vd, hv = self.value_dim, self.num_v_heads
        z = zba[..., :vd].reshape(B, S, hv, Dv)
        b = zba[..., vd:vd + hv]
        a = zba[..., vd + hv:]
    else:
        z = self.in_proj_z(inputs).reshape(
            B, S, self.num_v_heads, self.head_v_dim)
        b = self.in_proj_b(inputs)
        a = self.in_proj_a(inputs)

    conv_state = (
        cache[0]
        if (cache is not None and cache[0] is not None)
        else mx.zeros((B, self.conv_kernel_size - 1, self.conv_dim), dtype=inputs.dtype)
    )
    conv_input = mx.concatenate([conv_state, qkv], axis=1)
    if cache is not None:
        n_keep = self.conv_kernel_size - 1
        if cache.lengths is not None:
            ends = mx.clip(cache.lengths, 0, S)
            positions = (ends[:, None] + mx.arange(n_keep))[..., None]
            cache[0] = mx.take_along_axis(conv_input, positions, axis=1)
        else:
            cache[0] = mx.contiguous(conv_input[:, -n_keep:, :])

    cw = getattr(self, "_gdn_decode_conv_weight", None)
    if cw is None or cw[0] != id(self.conv1d.weight):
        w = self.conv1d.weight[:, :, 0].T.astype(mx.float32)
        mx.eval(w)
        cw = (id(self.conv1d.weight), w)
        self._gdn_decode_conv_weight = cw
    conv_weight = cw[1]

    state = cache[1]
    tiled = int(self.num_k_heads != self.num_v_heads)
    y, state = _gdn_fused_decode_kernel(
        inputs=[
            conv_input,
            conv_weight,
            a,
            b,
            self.A_log,
            self.dt_bias,
            state,
            z,
            self.norm.weight,
        ],
        template=[
            ("InT", conv_input.dtype),
            ("StT", state.dtype),
            ("Dk", self.head_k_dim),
            ("Dv", Dv),
            ("Hk", self.num_k_heads),
            ("Hv", self.num_v_heads),
            ("K_size", self.conv_kernel_size),
            ("SG", SG),
            ("TILED", tiled),
            ("CONV_BF16", 1),
        ],
        grid=(32, SG, B * self.num_v_heads),
        threadgroup=(32, SG, 1),
        output_shapes=[(B, 1, self.num_v_heads, Dv), state.shape],
        output_dtypes=[conv_input.dtype, state.dtype],
    )
    if cache is not None:
        cache[1] = state
        cache.advance(S)
        if vlm_cache_advance:
            import mlx_vlm.models.qwen3_5.language as _L

            _L._qwen3_5_advance_left_padding_info(cache, S)
            _L._qwen3_5_advance_lengths_info(cache, S)
    return self.out_proj(y.reshape(B, S, -1))


def _gdn_fused_decode_call(self, inputs, mask=None, cache=None):
    """Patched mlx-lm ``GatedDeltaNet.__call__`` routing the S=1 decode step through
    the fused kernel. Falls back to the stock implementation for prefill (S>1), when
    the fused path is disabled on this instance, when no recurrent state exists yet,
    or when the head dims don't tile evenly onto the kernel grid."""
    B, S, _ = inputs.shape
    state = cache[1] if cache else None
    Dv = self.head_v_dim
    SG = 16 if B == 1 else 32
    if (
        not getattr(self, "_gdn_fused", False)
        or S != 1
        or mask is not None
        or state is None
        or _gdn_fused_decode_kernel is None
        or Dv % SG != 0
        or self.head_k_dim % 32 != 0
    ):
        return _FUSED_DECODE_PATCH.stock(self, inputs, mask, cache)
    return _gdn_fused_decode_body(self, inputs, cache)


def _patch_gated_delta_fused_decode(model) -> None:
    """Enable the fused gated-delta decode kernel on every GatedDeltaNet in
    ``model``. Installs the class-level dispatch once (a no-op for any instance
    without the per-instance ``_gdn_fused`` flag, so unrelated/dense loads in the
    same process are untouched) and flags this model's instances."""
    from mlx_lm.models.qwen3_5 import GatedDeltaNet

    if (
        _gdn_fused_decode_kernel is None
        or not env_bool("GMLX_FUSED_GDN", True)
    ):
        return
    _FUSED_DECODE_PATCH.install(GatedDeltaNet, "__call__", _gdn_fused_decode_call)
    n = 0
    # z/b/a matvec merge: token-exact but an E2E wash on qwen3.6-27b
    # (-0.4%, within noise) -- launch gaps are already hidden inside the
    # step, so fewer dispatches buy nothing there. Off by default; opt-in
    # for re-measure at other geometries.
    merge_zba = env_bool("GMLX_GDN_ZBA", False)
    n_zba = 0
    for m in model.modules():
        if isinstance(m, GatedDeltaNet):
            m._gdn_fused = True
            n += 1
            if merge_zba and _gdn_try_merge_zba(m):
                n_zba += 1
    zba = f", z/b/a matvecs merged on {n_zba}" if n_zba else ""
    loadlog.verbose_print(
        f"[patch] gated_delta: fused decode kernel enabled on {n} layers{zba}")


def _gdn_try_merge_zba(gdn) -> bool:
    """Collapse the three float in-projections (z, b, a) into one matvec for
    the fused decode step. The merged weight owns the storage; the original
    modules' weights become row-slice views into it, so the stock path
    (prefill, S>1 fallback, verify) is untouched and no memory is duplicated.
    Only applies when all three are plain unbiased Linears of one dtype
    (quantized z, e.g. pure-K-quant files, keeps the separate path)."""
    import mlx.nn as nn

    mods = [gdn.in_proj_z, gdn.in_proj_b, gdn.in_proj_a]
    for m in mods:
        if type(m) is not nn.Linear or "bias" in m:
            return False
    if len({m.weight.dtype for m in mods}) != 1:
        return False
    if gdn.in_proj_z.weight.shape[0] != gdn.value_dim:
        return False
    merged = mx.concatenate([m.weight for m in mods], axis=0)
    mx.eval(merged)
    v, h = gdn.value_dim, gdn.num_v_heads
    gdn.in_proj_z.weight = merged[:v]
    gdn.in_proj_b.weight = merged[v:v + h]
    gdn.in_proj_a.weight = merged[v + h:]
    gdn._gdn_zba_weight = merged
    return True


# The multi-position (S>1) analog of the decode kernel above, for the MTP
# speculative-verify forward. The same serial chain - depthwise causal conv ->
# silu -> q/k rms-norm -> recurrent scan -> gated output norm - runs once per
# verify position, with the scan additionally emitting the per-position
# intermediate states the rejection rollback indexes into. Fusing the whole
# per-position chain into one kernel collapses the launches the same way; the
# win is again launch-latency-bound and so materialises on MoE targets, whose
# cheap per-token expert matmuls leave the chain exposed. It is *larger* than the
# decode win because at S>1 the chain runs per position while the verify's
# expert matmuls stay small, so the chain is a bigger fraction of the round.
#
# Same fp32-accurate (not bit-identical) recurrence as the decode kernel; greedy
# verify output can differ from the stock path only at near-ties.

_GDN_FUSED_VERIFY_SRC = r"""
    constexpr int n_per_t = Dk / 32;
    constexpr int key_dim = Hk * Dk;
    constexpr int value_dim = Hv * Dv;
    constexpr int conv_dim = 2 * key_dim + value_dim;
    constexpr int n_dv = Dv / SG;
    const float inv_scale = rsqrt((float)Dk);

    uint lane = thread_position_in_threadgroup.x;
    uint sg   = thread_position_in_threadgroup.y;
    uint n    = thread_position_in_grid.z;
    uint b_idx = n / Hv;
    uint hv_idx = n % Hv;
    uint hk_idx = TILED ? (hv_idx % Hk) : (hv_idx / (Hv / Hk));
    uint Tn = (uint)T;

    threadgroup float sumsq_sh[SG];

    float A = exp((float)A_log[hv_idx]);
    float dtb = (float)dt_bias[hv_idx];

    uint Tin = Tn + (K_size - 1);
    auto ci_b = conv_input + (uint)b_idx * Tin * conv_dim;

    float st[n_dv][n_per_t];
    for (int td = 0; td < n_dv; ++td) {
        int dv = sg * n_dv + td;
        auto i_state = state_in + ((uint)(n * Dv + dv)) * Dk;
        for (int i = 0; i < n_per_t; ++i) st[td][i] = (float)i_state[n_per_t * lane + i];
    }

    for (uint t = 0; t < Tn; ++t) {
        float ab = (float)a[(b_idx * Tn + t) * Hv + hv_idx] + dtb;
        float sp = ab > 20.0f ? ab : log(1.0f + exp(ab));
        float g = exp(-A * sp);
        float beta = 1.0f / (1.0f + exp(-(float)b[(b_idx * Tn + t) * Hv + hv_idx]));

        auto ci_t = ci_b + t * conv_dim;
        float qn[n_per_t], kn[n_per_t];
        float qsq = 0.0f, ksq = 0.0f;
        for (int i = 0; i < n_per_t; ++i) {
            int dk = n_per_t * lane + i;
            int qch = hk_idx * Dk + dk;
            int kch = key_dim + hk_idx * Dk + dk;
            float qa = 0.0f, ka = 0.0f;
            for (int w = 0; w < K_size; ++w) {
                qa += (float)ci_t[w * conv_dim + qch] * conv_weight[w * conv_dim + qch];
                ka += (float)ci_t[w * conv_dim + kch] * conv_weight[w * conv_dim + kch];
            }
            if (CONV_BF16) { qa = (float)(InT)qa; ka = (float)(InT)ka; }
            float qs = qa * (1.0f / (1.0f + exp(-qa)));
            float ks = ka * (1.0f / (1.0f + exp(-ka)));
            qn[i] = qs; kn[i] = ks; qsq += qs * qs; ksq += ks * ks;
        }
        qsq = simd_sum(qsq); ksq = simd_sum(ksq);
        float qscale = inv_scale * inv_scale * rsqrt(qsq / (float)Dk + 1e-6f);
        float kscale = inv_scale * rsqrt(ksq / (float)Dk + 1e-6f);
        for (int i = 0; i < n_per_t; ++i) { qn[i] *= qscale; kn[i] *= kscale; }

        float out_t[n_dv];
        float my_sumsq = 0.0f;
        for (int td = 0; td < n_dv; ++td) {
            int dv = sg * n_dv + td;
            int vch = 2 * key_dim + hv_idx * Dv + dv;
            float va = 0.0f;
            for (int w = 0; w < K_size; ++w)
                va += (float)ci_t[w * conv_dim + vch] * conv_weight[w * conv_dim + vch];
            if (CONV_BF16) va = (float)(InT)va;
            float v_val = va * (1.0f / (1.0f + exp(-va)));
            float kv_mem = 0.0f;
            for (int i = 0; i < n_per_t; ++i) { st[td][i] *= g; kv_mem += st[td][i] * kn[i]; }
            kv_mem = simd_sum(kv_mem);
            float delta = (v_val - kv_mem) * beta;
            float out = 0.0f;
            for (int i = 0; i < n_per_t; ++i) { st[td][i] += kn[i] * delta; out += st[td][i] * qn[i]; }
            out = simd_sum(out);
            auto states_t = states + ((((uint)b_idx * Tn + t) * Hv + hv_idx) * Dv + dv) * Dk;
            for (int i = 0; i < n_per_t; ++i) states_t[n_per_t * lane + i] = (StT)st[td][i];
            out_t[td] = out;
            if (lane == 0) my_sumsq += out * out;
        }

        if (lane == 0) sumsq_sh[sg] = my_sumsq;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float tot = 0.0f;
        for (int s = 0; s < SG; ++s) tot += sumsq_sh[s];
        float rms_denom = rsqrt(tot / (float)Dv + 1e-6f);
        if (lane == 0) {
            for (int td = 0; td < n_dv; ++td) {
                int dv = sg * n_dv + td;
                float xn = out_t[td] * rms_denom * (float)norm_weight[dv];
                float zv = (float)z[(((uint)b_idx * Tn + t) * Hv + hv_idx) * Dv + dv];
                float gate = zv * (1.0f / (1.0f + exp(-zv)));
                y[(((uint)b_idx * Tn + t) * Hv + hv_idx) * Dv + dv] = (InT)(gate * xn);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (int td = 0; td < n_dv; ++td) {
        int dv = sg * n_dv + td;
        auto o_state = state_out + ((uint)(n * Dv + dv)) * Dk;
        for (int i = 0; i < n_per_t; ++i) o_state[n_per_t * lane + i] = (StT)st[td][i];
    }
"""

_gdn_fused_verify_kernel = (
    mx.fast.metal_kernel(
        name="gmlx_gated_delta_fused_verify",
        input_names=[
            "conv_input",
            "conv_weight",
            "a",
            "b",
            "A_log",
            "dt_bias",
            "state_in",
            "z",
            "norm_weight",
            "T",
        ],
        output_names=["y", "state_out", "states"],
        source=_GDN_FUSED_VERIFY_SRC,
    )
    if mx.metal.is_available()
    else None
)

# Dense (bf16/f16) lm_head verify projection. mx.matmul runs the F16 head at
# ~68% of peak for verify widths M=2-8 (tuned for M=1 GEMV and large-M GEMM, not
# the small-batch middle). This M-stationary GEMV-ext -- one output row per
# thread, LPR lanes reducing K, M register accumulators reused across the single
# weight read -- holds ~95% of peak flat across M=2..8, like the quantized
# mv_ext the K-quant linears use. GMLX_F16_HEAD_KERNEL=0 falls back to matmul.
_F16_HEAD_GEMV = (
    mx.fast.metal_kernel(
        name="gmlx_f16_head_gemv_ext",
        input_names=["x", "w"],
        output_names=["y"],
        header="#include <metal_simdgroup>\nusing namespace metal;\n",
        source=r"""
        constexpr int LPR = 8;                 // lanes per output row (K reduction)
        constexpr int ROWS_PER_SG = 32 / LPR;  // output rows per simdgroup
        constexpr int NSG = 2;
        constexpr int BN = ROWS_PER_SG * NSG;
        constexpr int VPT = 8;                 // K values per lane per step
        constexpr int KSTEP = LPR * VPT;
        uint n_tile = threadgroup_position_in_grid.y;
        uint b_idx = threadgroup_position_in_grid.z;
        uint simd_gid = simdgroup_index_in_threadgroup;
        uint simd_lid = thread_index_in_simdgroup;
        int row_in_sg = int(simd_lid) / LPR;
        int lane_in_row = int(simd_lid) % LPR;
        int out_row = int(n_tile) * BN + int(simd_gid) * ROWS_PER_SG + row_in_sg;
        const device TX* x_base = x + (int(b_idx) * VERIFY_T) * K;
        const device TW* w_base = (out_row < N_SIZE) ? (w + out_row * K) : w;
        float result[VERIFY_T];
        for (int t = 0; t < VERIFY_T; ++t) result[t] = 0.0f;
        for (int k0 = 0; k0 < K; k0 += KSTEP) {
            int k = k0 + lane_in_row * VPT;
            float wv[VPT];
            for (int j = 0; j < VPT; ++j) wv[j] = (k + j < K) ? float(w_base[k + j]) : 0.0f;
            for (int t = 0; t < VERIFY_T; ++t) {
                float acc = 0.0f;
                for (int j = 0; j < VPT; ++j)
                    acc += wv[j] * ((k + j < K) ? float(x_base[t * K + k + j]) : 0.0f);
                result[t] += acc;
            }
        }
        for (int t = 0; t < VERIFY_T; ++t) {
            float s = result[t];
            for (int off = LPR / 2; off >= 1; off >>= 1) s += simd_shuffle_down(s, off);
            if (lane_in_row == 0 && out_row < N_SIZE)
                y[(int(b_idx) * VERIFY_T + t) * N_SIZE + out_row] = TX(s);
        }
        """,
    )
    if mx.metal.is_available()
    else None
)


def _f16_head_gemv(x, w):
    """x [B, M, K], w [N, K] (dense bf16/f16) -> [B, M, N] == x @ w.T.
    M-stationary GEMV-ext; ~95% of peak at M=2..8 where mx.matmul is ~68%."""
    B, M, Kd = x.shape
    Nd = w.shape[0]
    BN = (32 // 8) * 2  # ROWS_PER_SG * NSG
    n_groups = (Nd + BN - 1) // BN
    return _F16_HEAD_GEMV(
        inputs=[mx.contiguous(x), w],
        template=[("TX", x.dtype), ("TW", w.dtype), ("VERIFY_T", int(M)),
                  ("K", int(Kd)), ("N_SIZE", int(Nd))],
        grid=(32, 2 * n_groups, B),
        threadgroup=(32, 2, 1),
        output_shapes=[(B, M, Nd)],
        output_dtypes=[x.dtype],
    )[0]


def _bf16_verify_linear(linear, x):
    """Route bf16 nn.Linear through _f16_head_gemv at M>1.  At M=1 or when the
    Metal kernel is unavailable, falls back to the stock nn.Linear forward."""
    if x.shape[1] > 1 and _F16_HEAD_GEMV is not None and not hasattr(linear, "scales"):
        out = _f16_head_gemv(x, linear.weight)
        b = getattr(linear, "bias", None)
        if b is not None:
            out = out + b
        return out
    return linear(x)


_FUSED_VERIFY_PATCH = ClassPatch()


def _gdn_fused_verify_call(
    self, inputs, mask=None, cache=None, gdn_sink=None, target_verify=False
):
    """Patched mlx-vlm ``Qwen3_5GatedDeltaNet.__call__`` routing the multi-position
    MTP verify branch (``gdn_sink`` set, S>1) through the fused kernel. Every other
    path (prefill, S=1 decode, no-sink) and any instance without the per-instance
    ``_gdn_fused_verify`` flag falls back to the stock implementation."""
    B, S, _ = inputs.shape
    target_verify = target_verify or gdn_sink is not None
    Dv = self.head_v_dim
    # S=1 plain decode (no MTP verify sink): route through the fused decode kernel -
    # the same launch-collapse the mlx-lm text path already gets. Without this,
    # text decode through the mlx-vlm class (VLM generation, MTP base/first-token
    # decode) runs the unfused conv->norm->scan->gate chain, ~7% slower per token on
    # MoE gated-delta hybrids (dense is a wash). Same fp32-accurate (not
    # bit-identical) kernel as the text path; greedy diverges only at near-ties.
    if (
        S == 1
        and gdn_sink is None
        and getattr(self, "_gdn_fused_verify", False)
        and _gdn_fused_decode_kernel is not None
        and (mask is None or not isinstance(mask, mx.array))
        and cache is not None
        and cache[1] is not None
        and Dv % (16 if B == 1 else 32) == 0
        and self.head_k_dim % 32 == 0
    ):
        return _gdn_fused_decode_body(self, inputs, cache, vlm_cache_advance=True)
    if (
        not getattr(self, "_gdn_fused_verify", False)
        or gdn_sink is None
        or S <= 1
        or _gdn_fused_verify_kernel is None
        or Dv % 16 != 0
        or self.head_k_dim % 32 != 0
    ):
        return _FUSED_VERIFY_PATCH.stock(
            self, inputs, mask, cache, gdn_sink, target_verify
        )

    import mlx_vlm.models.qwen3_5.language as _L

    mixed_qkv = self.in_proj_qkv(inputs)
    z = _bf16_verify_linear(self.in_proj_z, inputs)
    b = _bf16_verify_linear(self.in_proj_b, inputs)
    a = _bf16_verify_linear(self.in_proj_a, inputs)
    z = z.reshape(B, S, self.num_v_heads, Dv)

    conv_state = cache[0] if (cache is not None and cache[0] is not None) else None
    if conv_state is None or conv_state.shape[0] != B:
        conv_state = mx.zeros(
            (B, self.conv_kernel_size - 1, self.conv_dim), dtype=inputs.dtype
        )
    if mask is not None and mask.shape[0] == B:
        mixed_qkv = mx.where(mask[..., None], mixed_qkv, 0)
    conv_input = mx.concatenate([conv_state, mixed_qkv], axis=1)
    if cache is not None:
        n_keep = self.conv_kernel_size - 1
        if getattr(cache, "lengths", None) is not None:
            ends = mx.clip(cache.lengths, 0, S)
            positions = (ends[:, None] + mx.arange(n_keep))[..., None]
            cache[0] = mx.take_along_axis(conv_input, positions, axis=1)
        else:
            cache[0] = mx.contiguous(conv_input[:, -n_keep:, :])

    init_state = cache[1] if cache else None
    state = init_state
    if state is None or state.shape[0] != B:
        state = mx.zeros((B, self.num_v_heads, Dv, self.head_k_dim), dtype=mx.float32)

    cw = getattr(self, "_gdn_verify_conv_weight", None)
    if cw is None or cw[0] != id(self.conv1d.weight):
        w = self.conv1d.weight[:, :, 0].T.astype(mx.float32)
        mx.eval(w)
        cw = (id(self.conv1d.weight), w)
        self._gdn_verify_conv_weight = cw
    conv_weight = cw[1]

    SG = 16 if B == 1 else 32
    tiled = int(self.num_k_heads != self.num_v_heads)
    y, new_state, inter = _gdn_fused_verify_kernel(
        inputs=[
            conv_input,
            conv_weight,
            a,
            b,
            self.A_log,
            self.dt_bias,
            state,
            z,
            self.norm.weight,
            S,
        ],
        template=[
            ("InT", conv_input.dtype),
            ("StT", state.dtype),
            ("Dk", self.head_k_dim),
            ("Dv", Dv),
            ("Hk", self.num_k_heads),
            ("Hv", self.num_v_heads),
            ("K_size", self.conv_kernel_size),
            ("SG", SG),
            ("TILED", tiled),
            ("CONV_BF16", 1),
        ],
        grid=(32, SG, B * self.num_v_heads),
        threadgroup=(32, SG, 1),
        output_shapes=[
            (B, S, self.num_v_heads, Dv),
            state.shape,
            (B, S, self.num_v_heads, Dv, self.head_k_dim),
        ],
        output_dtypes=[conv_input.dtype, state.dtype, state.dtype],
    )

    if cache is not None:
        cache[1] = new_state
        if hasattr(cache, "advance"):
            cache.advance(S)
            _L._qwen3_5_advance_left_padding_info(cache, S)
            _L._qwen3_5_advance_lengths_info(cache, S)

    if gdn_sink is not None:
        # The rejection rollback indexes the per-position intermediate states and
        # the conv input directly (snapshot rollback); the q/k/v entries are only
        # read by the no-intermediate-states fallback, which this path never hits.
        gdn_sink.append(
            (
                None,
                None,
                None,
                a,
                b,
                self.A_log,
                self.dt_bias,
                init_state,
                mask,
                conv_input,
                self.conv_kernel_size,
                inter,
            )
        )

    return _bf16_verify_linear(self.out_proj, y.reshape(B, S, -1))


_BATCHED_VERIFY_SDPA_PATCHED = False


def _patch_batched_verify_sdpa() -> None:
    """Replace the per-position SDPA loop in _target_verify_left_padded_attention
    with a single batched call using a causal mask.

    For MTP verify (S=K+1 query positions), the stock code does S separate
    mx.fast.scaled_dot_product_attention calls per attention layer.  A single
    call with mask='causal' is numerically identical (verified bit-exact for
    GQA, B=1..4, d up to 4096) and saves S-1 kernel launches per layer."""
    global _BATCHED_VERIFY_SDPA_PATCHED
    if _BATCHED_VERIFY_SDPA_PATCHED:
        return

    import mlx_vlm.models.qwen3_5.language as _L

    _orig_left_padded = _L._target_verify_left_padded_attention

    def _batched_verify_attention(queries, keys, values, *, cache, scale, mask):
        if hasattr(cache, "bits") or queries.ndim != 4 or keys.ndim != 4:
            return _orig_left_padded(
                queries, keys, values, cache=cache, scale=scale, mask=mask
            )
        pads = getattr(cache, "_qwen3_5_decode_left_padding", None)
        if pads is not None:
            return _orig_left_padded(
                queries, keys, values, cache=cache, scale=scale, mask=mask
            )
        L = queries.shape[-2]
        if L <= 1:
            return None
        sdpa_mask = mask if isinstance(mask, mx.array) else "causal"
        return mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=scale, mask=sdpa_mask
        )

    _L._target_verify_left_padded_attention = _batched_verify_attention
    _BATCHED_VERIFY_SDPA_PATCHED = True


_BF16_VERIFY_LINEAR_PATCHED = False


def _patch_bf16_verify_linear() -> None:
    """Route bf16 nn.Linear through the M-stationary GEMV-ext at verify width.

    MLX's bf16 matmul at small M (2..8) dispatches through a general GEMM path
    that runs at ~65-70% of peak BW. The GEMV-ext kernel (same weight read,
    M register accumulators) holds ~95% flat across M=2..8. This patches
    _target_verify_linear in mlx-vlm so every bf16 projection at verify width
    (MLP gate/up/down, attention q/k/v/o in UD-protected layers) benefits."""
    global _BF16_VERIFY_LINEAR_PATCHED
    if _BF16_VERIFY_LINEAR_PATCHED or _F16_HEAD_GEMV is None:
        return
    import mlx_vlm.models.qwen3_5.language as _L

    _stock_tvl = _L._target_verify_linear

    def _fast_target_verify_linear(linear, x, target_verify):
        if target_verify and x.shape[1] > 1 and not hasattr(linear, "scales"):
            out = _f16_head_gemv(x, linear.weight)
            b = getattr(linear, "bias", None)
            if b is not None:
                out = out + b
            return out
        return _stock_tvl(linear, x, target_verify)

    _L._target_verify_linear = _fast_target_verify_linear
    _BF16_VERIFY_LINEAR_PATCHED = True


_S0_MODEL_GUARD_INSTALLED = False


def _patch_qwen35_empty_sequence_guard() -> None:
    """Guard Qwen3_5Model.__call__ against S=0 (empty sequence) inputs.

    mlx-vlm's batched-prefill per-row loop can produce S=0 when a row's
    left-padding consumes the entire chunk. MLX's reshape(-1) raises on
    empty arrays, crashing gated-delta and attention layers. The guard
    returns self.norm(empty) immediately -- correct because padding tokens
    produce zero hidden states, and the caller zero-pads the output."""
    global _S0_MODEL_GUARD_INSTALLED
    if _S0_MODEL_GUARD_INSTALLED:
        return
    from mlx_vlm.models.qwen3_5.language import Qwen3_5Model

    _orig_model_call = Qwen3_5Model.__call__

    def _s0_guarded_call(self, inputs, inputs_embeds=None, **kw):
        if inputs_embeds is not None and inputs_embeds.shape[1] == 0:
            return self.norm(inputs_embeds)
        return _orig_model_call(self, inputs, inputs_embeds=inputs_embeds, **kw)

    Qwen3_5Model.__call__ = _s0_guarded_call
    _S0_MODEL_GUARD_INSTALLED = True


def _patch_gated_delta_fused_verify(model) -> None:
    """Enable the fused gated-delta verify kernel on every Qwen3_5GatedDeltaNet in
    the MTP target. Installs the class-level dispatch once (a no-op for instances
    without the per-instance ``_gdn_fused_verify`` flag) and flags this model's
    instances."""
    from mlx_vlm.models.qwen3_5.language import Qwen3_5GatedDeltaNet

    _patch_qwen35_empty_sequence_guard()

    if (
        _gdn_fused_verify_kernel is None
        or not env_bool("GMLX_FUSED_GDN", True)
    ):
        return
    _FUSED_VERIFY_PATCH.install(
        Qwen3_5GatedDeltaNet, "__call__", _gdn_fused_verify_call)
    n = 0
    lm = getattr(model, "language_model", model)
    for m in lm.modules():
        if isinstance(m, Qwen3_5GatedDeltaNet):
            m._gdn_fused_verify = True
            n += 1
    loadlog.verbose_print(f"[patch] gated_delta: fused verify kernel enabled on {n} layers")


def _patch_dense_head_verify(model) -> None:
    """Route the MTP verify-width (M=2..8) dense-head projection through the
    M-stationary GEMV-ext kernel (``_f16_head_gemv``). mx.matmul runs a dense
    bf16/f16 head at ~68% of peak for those widths; this lifts it to ~95%.

    Arch-agnostic: instead of hooking a per-arch logits method, it runtime-
    subclasses the head module and gates on the verify width, so it fires for any
    model whose head is a dense bf16/f16 untied Linear -- qwen native-head, the
    gemma assistant's target head, llama, future arches alike. Only the verify
    branch is intercepted (decode is M=1, prefill projects M=1); every other call
    falls through to the stock matmul, so non-MTP loads are unaffected even though
    the head is subclassed. Lossless modulo f32-accum reduction order (validated
    token-identical). GMLX_F16_HEAD_KERNEL=0 disables (matmul fallback)."""
    if _F16_HEAD_GEMV is None or not env_bool("GMLX_F16_HEAD_KERNEL", True):
        return
    lm = getattr(model, "language_model", model)
    head = getattr(lm, "lm_head", None)
    w = getattr(head, "weight", None)
    if (w is None or not hasattr(w, "ndim") or w.ndim != 2
            or w.dtype not in (mx.bfloat16, mx.float16)
            or getattr(head, "bias", None) is not None):
        return  # quantized (packed-uint weight), tied, or biased -> stock path
    if getattr(type(head), "_f16_gemv_kernel", False):
        return  # already subclassed in this process

    base = type(head)

    class _HeadWithGemvVerify(base):
        _f16_gemv_kernel = True
        _f16_gemv_on = True  # flip on the class for A/B; install gate is the env

        def __call__(self, x):
            if (type(self)._f16_gemv_on and x.ndim == 3 and 2 <= x.shape[1] <= 8
                    and x.shape[-1] == self.weight.shape[1]):
                return _f16_head_gemv(x, self.weight)
            return super().__call__(x)

    head.__class__ = _HeadWithGemvVerify
    loadlog.verbose_print(f"[patch] dense head: M-stationary GEMV-ext verify kernel "
          f"(vocab={w.shape[0]}, dtype={w.dtype})")


def _tiled_v_patch_applied() -> bool:
    """True once ``_patch_gated_delta_tiled_v`` has rewritten mlx-lm's
    gated_delta module globals in this process (marker set by the patch)."""
    import sys

    gd = sys.modules.get("mlx_lm.models.gated_delta")
    return bool(getattr(gd, "_gmlx_tiled_v_patched", False))


def _patch_gated_delta_tiled_v():
    """Monkey-patch gated_delta to use tiled V-head K mapping.

    GGUF stores V heads in tiled order for ggml broadcast:
        grouped (HF): [G0_v0 G0_v1 G0_v2 G1_v0 G1_v1 G1_v2 ...]
        tiled (GGUF):  [G0_v0 G1_v0 ... GN_v0 G0_v1 G1_v1 ... GN_v1 ...]

    The only difference is K->V head indexing:
        grouped: K[hv // (Hv/Hk)]
        tiled:   K[hv %  Hk]
    """
    from mlx_lm.models import gated_delta as gd

    gd._gmlx_tiled_v_patched = True  # read back by _tiled_v_patch_applied

    # Metal kernels: hv_idx / (Hv / Hk)  ->  hv_idx % Hk
    def _make_tiled_kernel(has_mask=False, vectorized=False):
        if not mx.metal.is_available():
            return None
        mask_source = "mask[b_idx * T + t]" if has_mask else "true"
        if vectorized:
            g_comment = "// g: [B, T, Hv, Dk]"
            g_setup = "auto g_ = g + (b_idx * T * Hv + hv_idx) * Dk;"
            g_access = "g_[s_idx]"
            g_advance = "g_ += Hv * Dk;"
        else:
            g_comment = "// g: [B, T, Hv]"
            g_setup = "auto g_ = g + b_idx * T * Hv;"
            g_access = "g_[hv_idx]"
            g_advance = "g_ += Hv;"

        source = f"""
            auto n = thread_position_in_grid.z;
            auto b_idx = n / Hv;
            auto hv_idx = n % Hv;
            auto hk_idx = hv_idx % Hk;
            constexpr int n_per_t = Dk / 32;

            auto q_ = q + b_idx * T * Hk * Dk + hk_idx * Dk;
            auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;

            auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv;
            y += b_idx * T * Hv * Dv + hv_idx * Dv;

            auto dk_idx = thread_position_in_threadgroup.x;
            auto dv_idx = thread_position_in_grid.y;

            auto i_state = state_in + (n * Dv + dv_idx) * Dk;
            auto o_state = state_out + (n * Dv + dv_idx) * Dk;

            float state[n_per_t];
            for (int i = 0; i < n_per_t; ++i) {{
              auto s_idx = n_per_t * dk_idx + i;
              state[i] = static_cast<float>(i_state[s_idx]);
            }}

            {g_comment}
            {g_setup}
            auto beta_ = beta + b_idx * T * Hv;

            for (int t = 0; t < T; ++t) {{
              if ({mask_source}) {{
                float kv_mem = 0.0f;
                for (int i = 0; i < n_per_t; ++i) {{
                  auto s_idx = n_per_t * dk_idx + i;
                  state[i] = state[i] * {g_access};
                  kv_mem += state[i] * k_[s_idx];
                }}
                kv_mem = simd_sum(kv_mem);

                auto delta = (v_[dv_idx] - kv_mem) * beta_[hv_idx];

                float out = 0.0f;
                for (int i = 0; i < n_per_t; ++i) {{
                  auto s_idx = n_per_t * dk_idx + i;
                  state[i] = state[i] + k_[s_idx] * delta;
                  out += state[i] * q_[s_idx];
                }}
                out = simd_sum(out);
                if (thread_index_in_simdgroup == 0) {{
                  y[dv_idx] = static_cast<InT>(out);
                }}
              }} else {{
                y[dv_idx] = static_cast<InT>(0);
              }}
              q_ += Hk * Dk;
              k_ += Hk * Dk;
              v_ += Hv * Dv;
              y += Hv * Dv;
              {g_advance}
              beta_ += Hv;
            }}
            for (int i = 0; i < n_per_t; ++i) {{
              auto s_idx = n_per_t * dk_idx + i;
              // Implicit narrowing to o_state's element type: naming it would
              // need an StT template arg, which mlx-lm's caller (which owns the
              // template list) only supplies from 0.31 on.
              o_state[s_idx] = state[i];
            }}
        """
        inputs = ["q", "k", "v", "g", "beta", "state_in", "T"]
        if has_mask:
            inputs.append("mask")
        suffix = ""
        if vectorized:
            suffix += "_vec"
        if has_mask:
            suffix += "_mask"
        return mx.fast.metal_kernel(
            name=f"gated_delta_step_tiled{suffix}",
            input_names=inputs,
            output_names=["y", "state_out"],
            source=source,
        )

    gd._gated_delta_kernel = _make_tiled_kernel(False, False)
    gd._gated_delta_kernel_masked = _make_tiled_kernel(True, False)
    gd._gated_delta_kernel_vec = _make_tiled_kernel(False, True)
    gd._gated_delta_kernel_vec_masked = _make_tiled_kernel(True, True)

    # Ops fallback: mx.repeat -> mx.tile
    _orig_step = gd._gated_delta_step_ops

    def _tiled_gated_delta_ops(q, k, v, g, beta, state=None, mask=None):
        B, T, Hk, Dk = q.shape
        Hv, Dv = v.shape[-2:]
        if state is None:
            state = mx.zeros((B, Hv, Dv, Dk), dtype=mx.float32)
        if (r := Hv // Hk) > 1:
            q = mx.tile(q, [1, 1, r, 1])
            k = mx.tile(k, [1, 1, r, 1])
        ys = []
        for t in range(T):
            y, state = _orig_step(
                q[:, t],
                k[:, t],
                v[:, t],
                g[:, t],
                beta[:, t],
                state,
                None if mask is None else mask[:, t],
            )
            ys.append(y)
        return mx.stack(ys, axis=1), state

    gd.gated_delta_ops = _tiled_gated_delta_ops
    loadlog.verbose_print("[patch] gated_delta: K->V head mapping set to tiled (GGUF layout)")


# - mlx-vlm's own gated_delta (MTP target uses these; seam 3)


def _tiled_gd_with_states_ops(q, k, v, g, beta, state, mask=None):
    """Tiled-V copy of mlx-vlm's ``_gated_delta_with_states_ops``.

    Identical to the upstream op except the grouped K/V expand (``mx.repeat``)
    becomes the tiled one (``mx.tile``) - GGUF stores V heads in tiled order, so
    V head ``hv`` reads K head ``hv % Hk`` rather than ``hv // (Hv/Hk)``.
    """
    B, T, Hk, _Dk = q.shape
    Hv = v.shape[-2]
    if (r := Hv // Hk) > 1:
        q = mx.tile(q, [1, 1, r, 1])
        k = mx.tile(k, [1, 1, r, 1])
    ys, states = [], []
    for t in range(T):
        old_state = state
        decay = g[:, t, :, None, None]
        state = state * decay
        kv_mem = (state * k[:, t, :, None, :]).sum(axis=-1)
        delta = (v[:, t] - kv_mem) * beta[:, t, :, None]
        state = state + k[:, t, :, None, :] * delta[..., None]
        y = (state * q[:, t, :, None, :]).sum(axis=-1)
        if mask is not None:
            valid = mask[:, t]
            state = mx.where(valid[:, None, None, None], state, old_state)
            y = mx.where(valid[:, None, None], y, 0)
        ys.append(y.astype(q.dtype))
        states.append(state)
    return mx.stack(ys, axis=1), state, mx.stack(states, axis=1)


def _tiled_gd_state_ops(k, v, g, beta, state, steps, mask=None):
    """Tiled-V copy of mlx-vlm's ``_gated_delta_state_ops`` (repeat -> tile)."""
    _B, T, Hk, _Dk = k.shape
    Hv = v.shape[-2]
    if (r := Hv // Hk) > 1:
        k = mx.tile(k, [1, 1, r, 1])
    for t in range(T):
        old_state = state
        decay = g[:, t, :, None, None]
        state = state * decay
        kv_mem = (state * k[:, t, :, None, :]).sum(axis=-1)
        delta = (v[:, t] - kv_mem) * beta[:, t, :, None]
        state = state + k[:, t, :, None, :] * delta[..., None]
        valid = steps > t
        if mask is not None:
            valid = valid & mask[:, t]
        state = mx.where(valid[:, None, None, None], state, old_state)
    return state


def _patch_mlxvlm_gated_delta_tiled_v() -> None:
    """Extend the tiled-V fixup to mlx-vlm's qwen3_5 gated_delta (seam 3).

    mlx-vlm's ``models/qwen3_5/gated_delta.py`` has its own grouped K->V mapping
    that ``_patch_gated_delta_tiled_v`` (which only touches mlx-lm) never reaches:

    * the regular prefill path (``gated_delta_update``) calls the names it
      ``from``-imported from mlx-lm at module load - so the patched mlx-lm
      attribute is shadowed for the *ops* fallback (the *kernel* function reads
      its module global at call time and is already tiled). Rebind both copies.
    * the MTP state-capturing paths (``gated_delta_update_with_states`` /
      ``gated_delta_state_update``) ship their own grouped Metal kernels + ops.
      They run only over the tiny draft block (small T), so we route them to the
      tiled ops and disable the grouped kernels rather than duplicate ~150 lines
      of Metal that we cannot validate without GPU.

    Must run after ``_patch_gated_delta_tiled_v`` (depends on its tiled mlx-lm
    ``gated_delta_ops`` / ``_gated_delta_kernel``).
    """
    import importlib
    from mlx_lm.models import gated_delta as gd

    try:
        vgd = importlib.import_module("mlx_vlm.models.qwen3_5.gated_delta")
    except ImportError:
        return

    # The rebind is blind assignment; verify the seams still exist first so
    # an upstream rename fails here, not as unvalidated grouped-kernel output.
    for _name in ("gated_delta_ops", "gated_delta_kernel",
                  "_gated_delta_with_states_ops", "_gated_delta_state_ops"):
        if not hasattr(vgd, _name):
            raise RuntimeError(
                f"mlx-vlm gated_delta seam {_name!r} is gone - re-audit "
                f"against the pinned seams (gmlx.upstream_seams)")

    # Regular prefill: rebind the import-time copies to the tiled mlx-lm ones.
    vgd.gated_delta_ops = gd.gated_delta_ops
    vgd.gated_delta_kernel = gd.gated_delta_kernel

    # MTP state-capture paths: tiled ops + disable the grouped kernels so the
    # `if kernel is None: return <ops>` fallback fires (small-T, cheap).
    vgd._gated_delta_with_states_ops = _tiled_gd_with_states_ops
    vgd._gated_delta_state_ops = _tiled_gd_state_ops
    vgd._gated_delta_with_states_kernel = None
    vgd._gated_delta_with_states_kernel_masked = None
    vgd._gated_delta_state_kernel = None
    vgd._gated_delta_state_kernel_masked = None
    loadlog.verbose_print("[patch] mlx-vlm gated_delta: K->V mapping set to tiled (MTP target)")
