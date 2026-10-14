"""Runtime patches for the gated-delta (Qwen3-Next / Qwen3.5) family.

Split-GDN module fixup, the fused decode/verify Metal kernels, the qwen3.5
MTP-verify guards, and the check that a GGUF's gated-delta layers carry
the tiled V-head order. Each installer is install-once and kill-switchable
via its GMLX_* env flag.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


import atexit
import sys
from functools import lru_cache
from typing import Callable, NamedTuple, cast

import gmlx.load.loadlog as loadlog
from gmlx.envflags import env_bool, env_int
from gmlx.load.hadamard_modules import is_folded, shared_linears
from .patching import ClassPatch, link


def _metal_kernel(**kwargs) -> Callable[..., list[mx.array]]:
    """``mx.fast.metal_kernel``, typed: the mlx stub returns ``object``."""
    return cast(Callable[..., list[mx.array]], mx.fast.metal_kernel(**kwargs))


def gpu_active() -> bool:
    """Metal kernels dispatch only while the default device is the GPU."""
    return mx.default_device() == mx.gpu


@lru_cache(maxsize=None)
def gdn_sg(B: int) -> int:
    """Simdgroups per (batch, head) tile of the fused GDN decode kernel.

    ``mx.fast.metal_kernel`` emits no ``max_total_threads_per_threadgroup``
    launch bound, so the compiled pipeline's thread ceiling is register-
    allocation dependent; the M1 Max compiled the 16-simdgroup verify
    pipeline to a 448-thread ceiling and the 512-thread launch threw at
    eval. Note the ceiling is a register budget, not a thread count: a
    smaller SG spends more registers per thread (``my_out[Dv / SG]``
    doubles when SG halves), so its pipeline has its own, lower ceiling.
    SG=8 shrinks threads x regs by ~30% at Dv=128 - directionally safe but
    unvalidated on M1/M2 hardware; if it still throws, GMLX_GDN_SG=4 is the
    next step (restart required: the cache reads the env once per process).
    M2 is clamped on the M1 evidence, not its own measurement.
    """
    forced = env_int("GMLX_GDN_SG", 0)
    if forced > 0:
        return forced
    from gmlx.load.dtypes import gpu_arch_gen

    gen = gpu_arch_gen()
    if 0 < gen < 15:
        return 8
    return 16 if B == 1 else 32


# One-shot routing dump for the GDN call: which branch each distinct
# (S, B, sink, mask) shape takes. GMLX_GDN_ROUTE_DEBUG=1.
_ROUTE_DEBUG = env_bool("GMLX_GDN_ROUTE_DEBUG", False)
_route_logged: set = set()
# Call counts per (which, S, B, mask, branch). The decode-width histogram this
# yields is the only record of live batch width on the NON-MTP arm, which has
# no round log -- without it a gated-vs-baseline comparison cannot tell a real
# regression from an arm that simply never reached the same width.
_route_counts: dict = {}


_route_calls = [0]
# Dump cadence in GDN calls. The harness SIGTERMs servers, which does not run
# atexit, so the histogram has to land in the log while the process is alive.
_ROUTE_DUMP_EVERY = 2000


def _log_gdn_route_once(which, S, B, sink, mask, taken):
    key = (which, S, B, sink is not None, type(mask).__name__, taken)
    _route_counts[key] = _route_counts.get(key, 0) + 1
    _route_calls[0] += 1
    if _route_calls[0] % _ROUTE_DUMP_EVERY == 0:
        _dump_gdn_route_counts()
    if key in _route_logged:
        return
    _route_logged.add(key)
    print(f"[gdn] route {which}: S={S} B={B} sink={sink is not None} "
          f"mask={type(mask).__name__} -> {taken}", file=sys.stderr, flush=True)


# Single-slot memo for "does this mask exclude anything". Every GDN layer in a
# step is handed the SAME mask object, so one host sync per step covers all of
# them; holding the reference keeps the identity check honest against id reuse.
_noop_mask_memo: tuple = (None, False)


def _mask_excludes_nothing(mask) -> bool:
    """True when `mask` admits every entry, so a kernel that ignores masks is
    still correct. At S=1 the recurrent ssm mask is all-true by construction
    (each live row contributes one real token; left padding is history, not
    input), but this is verified rather than assumed -- a row that is fully
    padded would make it false and must keep the stock path."""
    global _noop_mask_memo
    if _noop_mask_memo[0] is mask:
        return _noop_mask_memo[1]
    if mask.dtype == mx.bool_:
        verdict = bool(mx.all(mask).item())
    else:
        verdict = bool(mx.all(mask >= 0).item())
    _noop_mask_memo = (mask, verdict)
    return verdict


_mask_described: set = set()


def _describe_blocking_mask_once(which, S, B, mask):
    """When a mask is what costs this layer the fused kernel, report whether it
    actually masks anything. An all-true mask at S=1 blocks the fast path while
    excluding nothing."""
    if not isinstance(mask, mx.array):
        return
    key = (which, S, B, tuple(mask.shape), str(mask.dtype))
    if key in _mask_described:
        return
    _mask_described.add(key)
    if mask.dtype == mx.bool_:
        blocked = int((~mask).sum())
    else:
        blocked = int((mask < 0).sum())
    print(f"[gdn] blocking mask {which}: S={S} B={B} shape={tuple(mask.shape)} "
          f"dtype={mask.dtype} excluded_entries={blocked} of {mask.size} "
          f"({'NO-OP -> fused path lost for nothing' if blocked == 0 else 'real'})",
          file=sys.stderr, flush=True)


@atexit.register
def _dump_gdn_route_counts():
    if not _ROUTE_DEBUG or not _route_counts:
        return
    # Decode steps only (S == 1); per-layer calls, so divide by layer count for
    # rounds. Prefill chunks (S > 1) are reported separately.
    print("[gdn] route histogram (calls, S=1 decode):", file=sys.stderr)
    for key in sorted(_route_counts, key=lambda k: (-_route_counts[k], str(k))):
        which, S, B, sink, mask, taken = key
        if S != 1:
            continue
        print(f"[gdn]   {which} B={B} sink={sink} mask={mask} -> {taken}: "
              f"{_route_counts[key]} calls", file=sys.stderr)
    print("[gdn] route histogram (calls, S>1 prefill):", file=sys.stderr)
    for key in sorted(_route_counts, key=lambda k: (-_route_counts[k], str(k))):
        which, S, B, sink, mask, taken = key
        if S == 1:
            continue
        print(f"[gdn]   {which} S={S} B={B} -> {taken}: "
              f"{_route_counts[key]} calls", file=sys.stderr, flush=True)


# GGUF V-head order


def gguf_v_heads_tiled(config: dict) -> bool:
    """Whether a GGUF of this config stores its gated-delta V heads tiled.

    convert_hf_to_gguf reorders V heads from grouped to tiled when
    ``linear_num_key_heads != linear_num_value_heads``:
        grouped (HF): [G0_v0 G0_v1 G0_v2 G1_v0 G1_v1 G1_v2 ...]
        tiled (GGUF): [G0_v0 G1_v0 ... GN_v0 G0_v1 G1_v1 ... GN_v1 ...]
    V head hv then reads K head hv % Hk, not hv // (Hv / Hk). The
    synthesized config carries ``kv_head_layout="tiled"``, each gated-delta
    layer reads it into ``gdn_kv_tiled`` and passes it to its scan, so the
    order belongs to the model and a process holds models of both orders.

    qwen3_next is excluded despite its asymmetric heads (16 k / 32 v): its
    GGUFs keep V heads in HF grouped order (the legacy fused ssm_in layout is
    the raw HF tensor, and the newer split converter's de-interleave preserves
    group order). qwen4_exp is excluded too: its V heads are tiled on the
    wire, and its GatedDeltaNet tiles q/k itself before the scan.
    """
    if config.get("model_type") in ("qwen3_next", "qwen4_exp"):
        return False
    k = config.get("linear_num_key_heads", 0)
    v = config.get("linear_num_value_heads", 0)
    return k > 0 and v > 0 and k != v


def require_tiled_gdn(model, config: dict) -> None:
    """Refuse a GGUF load whose gated-delta layers would pair their heads
    in the grouped order. Such a load gives wrong text with no error: the
    no-cache forward looks fine and decode degrades to repetition. It
    happens when the config lost ``kv_head_layout`` on its way to the model
    classes, or a layer class does not read it. A container config holds
    the head counts under ``text_config``."""
    if "linear_num_key_heads" not in config:
        text = config.get("text_config")
        if isinstance(text, dict):
            config = text
    if not gguf_v_heads_tiled(config):
        return
    grouped = sorted({
        f"{type(m).__module__}.{type(m).__name__}"
        for m in model.modules()
        if hasattr(m, "num_k_heads") and hasattr(m, "num_v_heads")
        and m.num_k_heads != m.num_v_heads
        and not getattr(m, "gdn_kv_tiled", False)})
    if grouped:
        raise RuntimeError(
            "this GGUF stores its gated-delta V heads tiled "
            f"({config.get('linear_num_key_heads')} K heads, "
            f"{config.get('linear_num_value_heads')} V heads), and "
            f"{', '.join(grouped)} would pair them in the grouped order: "
            "kv_head_layout did not reach the layer from the config "
            f"(the config has {config.get('kv_head_layout')!r})")


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
    stock GDN; no mlx-lm module globals are touched, so it cannot leak into
    qwen3.5/mlx-vlm loads in the same process.
    Must run before sanitize/leaf-swap so the remapped weights land on the
    new modules and the fused module leaves no unfilled params.
    """
    from gmlx._vendor.mlx_lm.models.gated_delta import gated_delta_update
    from gmlx._vendor.mlx_lm.models.qwen3_next import Qwen3NextGatedDeltaNet

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

_GDN_FUSED_DECODE_TEMPLATE = r"""
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
/*REPLAY*/
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
            float sig = 1.0f / (1.0f + exp(-zv));
            float gate = GATE_SIGMOID ? sig : zv * sig;
            y[(b_idx * Hv + hv_idx) * Dv + dv] = (InT)(gate * xn);
        }
    }
"""

# The decode step after a verify round whose state is still a pending
# replay (see set_gdn_replay): each state row first replays the R recorded
# positions, the same float ops the verify scan ran.
_GDN_DECODE_REPLAY = r"""
        for (int p = 0; p < (int)R; ++p) {
            uint rb = (b_idx * (uint)rd_shape[1] + p) * Hv + hv_idx;
            float gp = rg[rb];
            float dp = rd[rb * Dv + dv];
            for (int i = 0; i < n_per_t; ++i) {
                state[i] *= gp;
                state[i] += rk[rb * Dk + n_per_t * lane + i] * dp;
            }
        }
"""

_GDN_FUSED_DECODE_SRC = _GDN_FUSED_DECODE_TEMPLATE.replace("/*REPLAY*/\n", "")
_GDN_FUSED_DECODE_REPLAY_SRC = _GDN_FUSED_DECODE_TEMPLATE.replace(
    "/*REPLAY*/\n", _GDN_DECODE_REPLAY)

_gdn_fused_decode_kernel = (
    _metal_kernel(
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

_gdn_fused_decode_replay_kernel = (
    _metal_kernel(
        name="gmlx_gated_delta_fused_decode_replay",
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
            "rd",
            "rk",
            "rg",
            "R",
        ],
        output_names=["y", "state_out"],
        source=_GDN_FUSED_DECODE_REPLAY_SRC,
    )
    if mx.metal.is_available()
    else None
)

_FUSED_DECODE_PATCH = ClassPatch()


def _live_cat(gdn, slot, members):
    """The concatenated weight in ``slot`` while every member it was cut
    from is still the plain Linear the concat took, else None. A wrapper
    such as an in-process LoRA replaces the member and has to be called."""
    w = getattr(gdn, slot, None)
    if w is None:
        return None
    for name in members:
        if type(getattr(gdn, name, None)) is not nn.Linear:
            return None
    return w


def _kv_heads_tiled(gdn) -> int:
    """1 when V head hv reads K head hv % Hk (the GGUF tiled order), 0 when
    it reads K head hv // (Hv / Hk). A module sets ``gdn_kv_tiled`` from
    its own config."""
    if gdn.num_k_heads == gdn.num_v_heads:
        return 0
    return int(bool(getattr(gdn, "gdn_kv_tiled", False)))


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
    SG = gdn_sg(B)

    zba_w = _live_cat(self, "_gdn_zba_weight", ("in_proj_z", "in_proj_b", "in_proj_a"))
    if zba_w is not None:
        qkv = self.in_proj_qkv(inputs)
        zba = inputs @ zba_w.T
        vd, hv = self.value_dim, self.num_v_heads
        z = zba[..., :vd].reshape(B, S, hv, Dv)
        b = zba[..., vd:vd + hv]
        a = zba[..., vd + hv:]
    else:
        # One rotation feeds both projections on a Hadamard-folded file.
        qkv, z = shared_linears((self.in_proj_qkv, self.in_proj_z), inputs)
        z = z.reshape(B, S, self.num_v_heads, self.head_v_dim)
        ba_w = _live_cat(self, "_gdn_ba_weight", ("in_proj_b", "in_proj_a"))
        if ba_w is not None:
            # One [2*Hv, K] matvec for the two tiny decay/gate rows. At
            # M=B*S in [2, 8] the M-stationary head kernel matters: stock
            # mlx routes each [M, K] @ [K, Hv] through a steel GEMM tile
            # that costs ~70 us wall for ~1 MB of weights (measured B=2:
            # b/a alone add 6.9 ms/step across 48 layers; this path
            # recovers all of it, cat-without-kernel only ~1.7 ms).
            if 2 <= B * S <= 8 and _F16_HEAD_GEMV is not None:
                ba = _f16_head_gemv(
                    inputs.reshape(1, B * S, -1), ba_w
                ).reshape(B, S, -1)
            else:
                ba = inputs @ ba_w.T
            hv = self.num_v_heads
            b = ba[..., :hv]
            a = ba[..., hv:]
        elif (ba_kq := _gdn_ba_kq(self)) is not None:
            ba = ba_kq(inputs)
            hv = self.num_v_heads
            b = ba[..., :hv]
            a = ba[..., hv:]
        else:
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
    kernel, replay = _gdn_fused_decode_kernel, []
    pend = take_gdn_replay(cache) if cache is not None else None
    if pend is not None:
        state, records, steps = pend
        kernel, replay = _gdn_fused_decode_replay_kernel, [*records, steps]
    tiled = _kv_heads_tiled(self)
    y, state = kernel(
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
            *replay,
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
            # output gate: silu(z) (qwen3.5) or sigmoid(z) (qwen4exp)
            ("GATE_SIGMOID", int(bool(getattr(self, "gdn_gate_sigmoid", False)))),
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
            from gmlx.models.qwen35.owned import (
                _qwen3_5_advance_left_padding_info,
                _qwen3_5_advance_lengths_info,
            )

            _qwen3_5_advance_left_padding_info(cache, S)
            _qwen3_5_advance_lengths_info(cache, S)
    return self.out_proj(y.reshape(B, S, -1))


def _gdn_fused_decode_call(self, inputs, mask=None, cache=None):
    """Patched mlx-lm ``GatedDeltaNet.__call__`` routing the S=1 decode step through
    the fused kernel. Falls back to the stock implementation for prefill (S>1), when
    the fused path is disabled on this instance, when no recurrent state exists yet,
    or when the head dims don't tile evenly onto the kernel grid."""
    B, S, _ = inputs.shape
    state = cache[1] if cache else None
    Dv = self.head_v_dim
    SG = gdn_sg(B)
    # A mask that excludes nothing must not cost the fused kernel: BatchGenerator
    # hands every batched decode step an ssm mask built from the cache's left
    # padding, and at S=1 that mask is all-true, so bailing on `mask is not None`
    # dropped all 30 gated-delta layers onto the unfused chain for nothing.
    if (
        not getattr(self, "_gdn_fused", False)
        or S != 1
        or (mask is not None
            and not (isinstance(mask, mx.array) and _mask_excludes_nothing(mask)))
        or state is None
        or _gdn_fused_decode_kernel is None
        or Dv % SG != 0
        or self.head_k_dim % 32 != 0
    ):
        if _ROUTE_DEBUG:
            _log_gdn_route_once("lm", S, B, None, mask, "STOCK-unfused")
            _describe_blocking_mask_once("lm", S, B, mask)
        return _FUSED_DECODE_PATCH.stock(self, inputs, mask, cache)
    if _ROUTE_DEBUG:
        _log_gdn_route_once("lm", S, B, None, mask, "fused-decode")
    return _gdn_fused_decode_body(self, inputs, cache)


def _patch_gated_delta_fused_decode(model) -> None:
    """Enable the fused gated-delta decode kernel on every GatedDeltaNet in
    ``model``. Installs the class-level dispatch once (a no-op for any instance
    without the per-instance ``_gdn_fused`` flag, so unrelated/dense loads in the
    same process are untouched) and flags this model's instances."""
    from gmlx._vendor.mlx_lm.models.qwen3_5 import GatedDeltaNet

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
    cat_ba = env_bool("GMLX_GDN_BA_CAT", True)
    n_zba = 0
    n_ba = 0
    for m in model.modules():
        if isinstance(m, GatedDeltaNet):
            m._gdn_fused = True
            n += 1
            if merge_zba and _gdn_try_merge_zba(m):
                n_zba += 1
            elif cat_ba and _gdn_try_cat_ba(m):
                n_ba += 1
    zba = f", z/b/a matvecs merged on {n_zba}" if n_zba else ""
    ba = f", b/a matvecs concatenated on {n_ba}" if n_ba else ""
    loadlog.verbose_print(
        f"[patch] gated_delta: fused decode kernel enabled on {n} "
        f"layers{zba}{ba}")


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
    # Plain instance attr: nn.Module.__setattr__ would file the merged
    # array into the parameter tree, aliasing the three views above and
    # double-counting any parameter-size audit.
    object.__setattr__(gdn, "_gdn_zba_weight", merged)
    return True


def _gdn_try_cat_ba(gdn) -> bool:
    """Concatenate the two tiny float decay/gate in-projections (b, a) into
    one [2*Hv, K] weight for the fused decode step. This is the k-quant
    complement of the zba merge above: on k-quant files z is quantized (the
    merge can never fire), while b/a load as plain bf16 Linears whose
    per-step matvecs collapse onto MLX's steel GEMM tile at batched decode
    (M>=2) at ~70 us wall each for ~1 MB of weights. The decode body routes
    the concatenated weight through the M-stationary head kernel instead.
    Same storage discipline as the merge: the cat owns the rows, the
    original modules' weights become row-slice views, so prefill and every
    stock path are untouched and no memory is duplicated."""
    import mlx.nn as nn

    if getattr(gdn, "_gdn_zba_weight", None) is not None:
        return False  # zba merge already owns these rows
    if getattr(gdn, "_gdn_ba_weight", None) is not None:
        return False  # already cat: arming may run on both load paths
    if getattr(gdn, "_gdn_ba_kq", None) is not None:
        return False
    mods = [getattr(gdn, "in_proj_b", None), getattr(gdn, "in_proj_a", None)]
    if _gdn_try_cat_ba_kq(gdn, mods):
        return True
    for m in mods:
        if m is None or type(m) is not nn.Linear or "bias" in m:
            return False
    if len({m.weight.dtype for m in mods}) != 1:
        return False
    h = getattr(gdn, "num_v_heads", None)
    if not h or any(m.weight.shape[0] != h for m in mods):
        return False
    cat = mx.concatenate([m.weight for m in mods], axis=0)
    mx.eval(cat)
    gdn.in_proj_b.weight = cat[:h]
    gdn.in_proj_a.weight = cat[h:]
    # Same reason as the zba merge: keep the cat out of the parameter
    # tree, it only aliases the two view weights above.
    object.__setattr__(gdn, "_gdn_ba_weight", cat)
    return True



def _gdn_try_cat_ba_kq(gdn, mods) -> bool:
    """The k-quant form of the b/a cat, for files that store b and a in a
    block codec: one KQuantLinear over b's rows then a's, so each step
    runs one quantized matmul where it ran two. Block-codec rows are
    self-contained, so the cat is bit-exact against the two calls. Same
    storage discipline: the cat owns the wire rows and b and a become
    row-slice views of it."""
    KQuantLinear = _kquant_linear_cls()
    b, a = mods
    if type(b) is not KQuantLinear or type(a) is not KQuantLinear:
        return False
    if "bias" in b or "bias" in a or b.kquant_type != a.kquant_type:
        return False
    h = getattr(gdn, "num_v_heads", None)
    if (not h or b.weight.ndim != 2 or a.weight.ndim != 2
            or b.weight.shape != a.weight.shape or b.weight.shape[0] != h):
        return False
    in_dims = int(getattr(gdn, "hidden_size", 0) or 0)
    if not in_dims:
        return False
    cat = KQuantLinear(in_dims, 2 * h, bias=False, codec=b.kquant_type)
    if cat.weight.shape != (2 * h, b.weight.shape[1]):
        return False
    cat.weight = mx.concatenate([b.weight, a.weight], axis=0)
    cat.scales = b.scales
    mx.eval(cat.weight)
    b.weight = cat.weight[:h]
    a.weight = cat.weight[h:]
    # Kept out of the parameter tree like the float cat: its weight is
    # the storage b and a view.
    object.__setattr__(gdn, "_gdn_ba_kq", cat)
    return True



def _kquant_linear_cls():
    import importlib

    return importlib.import_module("mlx_kquant.nn").KQuantLinear


def kquant_multi(linears, x: mx.array):
    """The outputs of ``linears`` on ``x`` from one
    ``kq.quantized_matmul_multi`` call, or None where it does not apply.

    It applies at 2 to 8 rows of ``x``, the verify widths, to a group of
    plain unbiased KQuantLinears of one codec: the op runs up to four
    weights that share ``x`` in one dispatch, each read from its own
    buffer. None on an mlx-kquant without the op, so callers keep their
    per-weight calls, and for any member an adapter or a fold wraps."""
    import importlib

    multi = getattr(importlib.import_module("mlx_kquant"),
                    "quantized_matmul_multi", None)
    if multi is None or len(linears) < 2 or x.ndim < 2:
        return None
    if not 2 <= x.size // x.shape[-1] <= 8:
        return None
    cls = _kquant_linear_cls()
    codec = getattr(linears[0], "kquant_type", None)
    for lin in linears:
        if type(lin) is not cls or "bias" in lin or lin.kquant_type != codec:
            return None
    return tuple(multi(x, [lin["weight"] for lin in linears], codec))

def _gdn_ba_kq(gdn):
    """The k-quant b/a cat while b and a are still the modules it was cut
    from, else None. A wrapper such as an in-process LoRA replaces a
    member and has to be called."""
    cat = getattr(gdn, "_gdn_ba_kq", None)
    if cat is None:
        return None
    for name in ("in_proj_b", "in_proj_a"):
        if type(getattr(gdn, name, None)) is not type(cat):
            return None
    return cat

# The multi-position (S>1) analog of the decode kernel above, for the MTP
# speculative-verify forward, in two dispatches. The scan kernel runs the
# serial per-position chain (depthwise causal conv, silu, q/k rms-norm,
# recurrent scan) for RPS state rows per simdgroup and emits the pre-norm
# outputs plus what the rejection rollback needs. Its threadgroups never
# synchronize, so one head spreads over Dv / (SGS * RPS) of them and a layer
# over every core, where the one-threadgroup-per-head form left most of the
# GPU idle. The gate kernel then applies the gated output norm per
# (position, head) row.
#
# The scan has two forms. The states form writes the final state and the
# state after every position but the last, which the rollback indexes. The
# records form (B=1) writes, per position, the three values a step adds to
# the state: the decay g, the scaled key and the delta row. A state is then
# its start state plus a replay of the first R records, and the next kernel
# to read the state replays them in registers before its own first step
# (set_gdn_replay). That drops the per-position state store, 7 full states
# per layer at S=8 (22 MB on the 27B), for about 0.4 MB of records.
#
# Same fp32-accurate (not bit-identical) recurrence as the decode kernel;
# greedy verify output can differ from the stock path only at near-ties. A
# replay runs the scan's own float ops in the same order, so a replayed
# state equals the stored one bit for bit.

_GDN_VERIFY_SGS = 4
_GDN_VERIFY_RPS = 4
_GDN_VERIFY_ROWS = _GDN_VERIFY_SGS * _GDN_VERIFY_RPS

_GDN_VERIFY_SCAN_TEMPLATE = r"""
    constexpr int n_per_t = Dk / 32;
    constexpr int key_dim = Hk * Dk;
    constexpr int value_dim = Hv * Dv;
    constexpr int conv_dim = 2 * key_dim + value_dim;
    constexpr int rows_per_tg = SGS * RPS;
    constexpr int NTG = Dv / rows_per_tg;
    const float inv_scale = rsqrt((float)Dk);

    uint lane = thread_position_in_threadgroup.x;
    uint sgl  = thread_position_in_threadgroup.y;
    uint tgz  = threadgroup_position_in_grid.z;
    uint part = tgz % NTG;
    uint n    = tgz / NTG;
    uint b_idx = n / Hv;
    uint hv_idx = n % Hv;
    uint hk_idx = TILED ? (hv_idx % Hk) : (hv_idx / (Hv / Hk));
    uint Tn = (uint)T;
    int row0 = (int)(part * rows_per_tg + sgl * RPS);

    float A = exp((float)A_log[hv_idx]);
    float dtb = (float)dt_bias[hv_idx];

    uint Tin = Tn + (K_size - 1);
    auto ci_b = conv_input + (uint)b_idx * Tin * conv_dim;

    float st[RPS][n_per_t];
    for (int r = 0; r < RPS; ++r) {
        auto i_state = state_in + ((uint)(n * Dv + row0 + r)) * Dk;
        for (int i = 0; i < n_per_t; ++i) st[r][i] = (float)i_state[n_per_t * lane + i];
    }
/*PROLOGUE*/
    for (uint t = 0; t < Tn; ++t) {
        // ba packs the b then a projections per position ([B, T, 2, Hv]):
        // one combined gemv upstream instead of two tiny dispatches.
        uint ba_base = ((b_idx * Tn + t) * 2) * Hv;
        float ab = (float)ba[ba_base + Hv + hv_idx] + dtb;
        float sp = ab > 20.0f ? ab : log(1.0f + exp(ab));
        float g = exp(-A * sp);
        float beta = 1.0f / (1.0f + exp(-(float)ba[ba_base + hv_idx]));

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
/*STEP*/
        for (int r = 0; r < RPS; ++r) {
            int dv = row0 + r;
            int vch = 2 * key_dim + hv_idx * Dv + dv;
            float va = 0.0f;
            for (int w = 0; w < K_size; ++w)
                va += (float)ci_t[w * conv_dim + vch] * conv_weight[w * conv_dim + vch];
            if (CONV_BF16) va = (float)(InT)va;
            float v_val = va * (1.0f / (1.0f + exp(-va)));
            float kv_mem = 0.0f;
            for (int i = 0; i < n_per_t; ++i) { st[r][i] *= g; kv_mem += st[r][i] * kn[i]; }
            kv_mem = simd_sum(kv_mem);
            float delta = (v_val - kv_mem) * beta;
            float out = 0.0f;
            for (int i = 0; i < n_per_t; ++i) { st[r][i] += kn[i] * delta; out += st[r][i] * qn[i]; }
            out = simd_sum(out);
/*ROW*/
            if (lane == 0) pre[(((uint)b_idx * Tn + t) * Hv + hv_idx) * Dv + dv] = out;
        }
    }
/*EPILOGUE*/
"""

# States form. Rollback targets are positions 0..Tn-2 only (the state after
# all Tn positions is state_out and rollback never fires on full accept),
# so the last position is not recorded.
_GDN_VERIFY_STATES_ROW = r"""
            if (t + 1 < Tn) {
                auto states_t = states + ((((uint)b_idx * (Tn - 1) + t) * Hv + hv_idx) * Dv + dv) * Dk;
                for (int i = 0; i < n_per_t; ++i) states_t[n_per_t * lane + i] = (StT)st[r][i];
            }
"""
_GDN_VERIFY_STATES_EPILOGUE = r"""
    for (int r = 0; r < RPS; ++r) {
        auto o_state = state_out + ((uint)(n * Dv + row0 + r)) * Dk;
        for (int i = 0; i < n_per_t; ++i) o_state[n_per_t * lane + i] = (StT)st[r][i];
    }
"""

# Records form. The replay loop is shared with the replay-only kernel below.
_GDN_REPLAY_ROWS = r"""
    for (int p = 0; p < (int)R; ++p) {
        uint rb = ((uint)b_idx * (uint)rd_shape[1] + p) * Hv + hv_idx;
        float gp = rg[rb];
        float kp[n_per_t];
        for (int i = 0; i < n_per_t; ++i) kp[i] = rk[rb * Dk + n_per_t * lane + i];
        for (int r = 0; r < RPS; ++r) {
            float dp = rd[rb * Dv + row0 + r];
            for (int i = 0; i < n_per_t; ++i) { st[r][i] *= gp; st[r][i] += kp[i] * dp; }
        }
    }
"""
# A kernel that replayed pending records starts from a state no buffer
# holds, so it stores that state as the base of its own records.
_GDN_VERIFY_REC_PROLOGUE = _GDN_REPLAY_ROWS + r"""
    if (R > 0) {
        for (int r = 0; r < RPS; ++r) {
            auto s_state = start + ((uint)(n * Dv + row0 + r)) * Dk;
            for (int i = 0; i < n_per_t; ++i) s_state[n_per_t * lane + i] = (StT)st[r][i];
        }
    }
"""
_GDN_VERIFY_REC_STEP = r"""
        if (part == 0 && sgl == 0) {
            uint rt = ((uint)b_idx * Tn + t) * Hv + hv_idx;
            for (int i = 0; i < n_per_t; ++i) kstore[rt * Dk + n_per_t * lane + i] = kn[i];
            if (lane == 0) gstore[rt] = g;
        }
"""
_GDN_VERIFY_REC_ROW = r"""
            if (lane == 0) dstore[(((uint)b_idx * Tn + t) * Hv + hv_idx) * Dv + dv] = delta;
"""


def _verify_scan_src(prologue="", step="", row="", epilogue=""):
    return (_GDN_VERIFY_SCAN_TEMPLATE
            .replace("/*PROLOGUE*/\n", prologue)
            .replace("/*STEP*/\n", step)
            .replace("/*ROW*/\n", row)
            .replace("/*EPILOGUE*/\n", epilogue))


_GDN_VERIFY_SCAN_SRC = _verify_scan_src(
    row=_GDN_VERIFY_STATES_ROW, epilogue=_GDN_VERIFY_STATES_EPILOGUE)
_GDN_VERIFY_REC_SRC = _verify_scan_src(
    prologue=_GDN_VERIFY_REC_PROLOGUE, step=_GDN_VERIFY_REC_STEP,
    row=_GDN_VERIFY_REC_ROW)

# Materializes a pending replay for a reader that is not a fused kernel:
# the start state plus R records, one thread tile per verify-scan tile.
_GDN_REPLAY_SRC = r"""
    constexpr int n_per_t = Dk / 32;
    constexpr int rows_per_tg = SGS * RPS;
    constexpr int NTG = Dv / rows_per_tg;
    uint lane = thread_position_in_threadgroup.x;
    uint sgl  = thread_position_in_threadgroup.y;
    uint tgz  = threadgroup_position_in_grid.z;
    uint part = tgz % NTG;
    uint n    = tgz / NTG;
    uint b_idx = n / Hv;
    uint hv_idx = n % Hv;
    int row0 = (int)(part * rows_per_tg + sgl * RPS);

    float st[RPS][n_per_t];
    for (int r = 0; r < RPS; ++r) {
        auto i_state = state_in + ((uint)(n * Dv + row0 + r)) * Dk;
        for (int i = 0; i < n_per_t; ++i) st[r][i] = (float)i_state[n_per_t * lane + i];
    }
""" + _GDN_REPLAY_ROWS + r"""
    for (int r = 0; r < RPS; ++r) {
        auto o_state = state_out + ((uint)(n * Dv + row0 + r)) * Dk;
        for (int i = 0; i < n_per_t; ++i) o_state[n_per_t * lane + i] = (StT)st[r][i];
    }
"""

_GDN_VERIFY_GATE_SRC = r"""
    constexpr int n_per_t = Dv / 32;
    uint lane = thread_position_in_threadgroup.x;
    uint row = thread_position_in_grid.y;
    if (row >= (uint)nrows) return;
    auto prow = pre + (size_t)row * Dv + n_per_t * lane;
    auto zrow = z + (size_t)row * Dv + n_per_t * lane;
    float v[n_per_t];
    float ss = 0.0f;
    for (int i = 0; i < n_per_t; ++i) { v[i] = prow[i]; ss += v[i] * v[i]; }
    ss = simd_sum(ss);
    float rms_denom = rsqrt(ss / (float)Dv + 1e-6f);
    for (int i = 0; i < n_per_t; ++i) {
        float xn = v[i] * rms_denom * (float)norm_weight[n_per_t * lane + i];
        float zv = (float)zrow[i];
        float sig = 1.0f / (1.0f + exp(-zv));
        float gate = GATE_SIGMOID ? sig : zv * sig;
        y[(size_t)row * Dv + n_per_t * lane + i] = (InT)(gate * xn);
    }
"""

_gdn_fused_verify_kernel = (
    _metal_kernel(
        name="gmlx_gated_delta_verify_scan",
        input_names=[
            "conv_input",
            "conv_weight",
            "ba",
            "A_log",
            "dt_bias",
            "state_in",
            "T",
        ],
        output_names=["pre", "state_out", "states"],
        source=_GDN_VERIFY_SCAN_SRC,
    )
    if mx.metal.is_available()
    else None
)

_gdn_verify_rec_kernel = (
    _metal_kernel(
        name="gmlx_gated_delta_verify_scan_rec",
        input_names=[
            "conv_input",
            "conv_weight",
            "ba",
            "A_log",
            "dt_bias",
            "state_in",
            "T",
            "rd",
            "rk",
            "rg",
            "R",
        ],
        output_names=["pre", "start", "dstore", "kstore", "gstore"],
        source=_GDN_VERIFY_REC_SRC,
    )
    if mx.metal.is_available()
    else None
)

_gdn_replay_kernel = (
    _metal_kernel(
        name="gmlx_gated_delta_replay",
        input_names=["state_in", "rd", "rk", "rg", "R"],
        output_names=["state_out"],
        source=_GDN_REPLAY_SRC,
    )
    if mx.metal.is_available()
    else None
)

_gdn_verify_gate_kernel = (
    _metal_kernel(
        name="gmlx_gated_delta_verify_gate",
        input_names=["pre", "z", "norm_weight", "nrows"],
        output_names=["y"],
        source=_GDN_VERIFY_GATE_SRC,
    )
    if mx.metal.is_available()
    else None
)

class GdnRecords(NamedTuple):
    """What a records-form verify leaves in the sink for the rollback: the
    state before the round and the per-position (delta, key, g) records."""

    base: mx.array
    records: tuple


_NO_RECORDS: list = []


def _no_records() -> tuple:
    """Placeholder records for a kernel called with R=0, which reads none."""
    if not _NO_RECORDS:
        z = (mx.zeros((1, 1, 1, 1)), mx.zeros((1, 1, 1, 1)), mx.zeros((1, 1, 1)))
        mx.eval(z)
        _NO_RECORDS.append(z)
    return _NO_RECORDS[0]


def gdn_replay_state(base: mx.array, records: tuple, steps: int) -> mx.array:
    """The recurrent state after the first ``steps`` recorded positions,
    replayed onto ``base``. Lazy: a fused kernel that takes the replay
    itself leaves this array unevaluated."""
    B, Hv, Dv, Dk = base.shape
    SGS, RPS = _GDN_VERIFY_SGS, _GDN_VERIFY_RPS
    (state,) = _gdn_replay_kernel(
        inputs=[base, *records, steps],
        template=[("StT", base.dtype), ("Dk", Dk), ("Dv", Dv), ("Hv", Hv),
                  ("SGS", SGS), ("RPS", RPS)],
        grid=(32, SGS, B * Hv * (Dv // _GDN_VERIFY_ROWS)),
        threadgroup=(32, SGS, 1),
        output_shapes=[base.shape],
        output_dtypes=[base.dtype],
    )
    return state


def set_gdn_replay(cache, base: mx.array, records: tuple, steps: int) -> None:
    """Leave ``cache``'s recurrent state as ``base`` plus a replay of
    ``steps`` records. ``cache[1]`` becomes the lazy replay, so any reader
    gets the right state, and the fused kernels take the pending replay
    into their own prologue instead (take_gdn_replay)."""
    state = gdn_replay_state(base, records, steps)
    cache[1] = state
    cache._gdn_replay = (state, base, records, steps)


def take_gdn_replay(cache):
    """``(base, records, steps)`` when ``cache[1]`` is still the replay
    set_gdn_replay left, else None. Clears the pending record either way.
    A writer that replaces ``cache[1]`` voids the record by identity; one
    that updated the array in place would not, and none does."""
    pend = getattr(cache, "_gdn_replay", None)
    if pend is None:
        return None
    cache._gdn_replay = None
    if pend[0] is not cache[1]:
        return None
    return pend[1], pend[2], pend[3]


# Dense (bf16/f16) lm_head verify projection. mx.matmul runs the F16 head at
# ~68% of peak for verify widths M=2-8 (tuned for M=1 GEMV and large-M GEMM, not
# the small-batch middle). This M-stationary GEMV-ext -- one output row per
# thread, LPR lanes reducing K, M register accumulators reused across the single
# weight read -- holds ~95% of peak flat across M=2..8, like the quantized
# mv_ext the K-quant linears use. GMLX_F16_HEAD_KERNEL=0 falls back to matmul.
_F16_HEAD_GEMV = (
    _metal_kernel(
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
    if not gpu_active():
        return x @ w.T
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


def _float_weight(linear):
    """The float ``weight`` a head GEMV can read, or None for a quantized
    linear or an adapter wrapper, which have to be called."""
    if hasattr(linear, "scales"):
        return None
    w = getattr(linear, "weight", None)
    return w if isinstance(w, mx.array) else None


def _bf16_verify_linear(linear, x):
    """Route bf16 nn.Linear through _f16_head_gemv at M>1.  At M=1, when the
    Metal kernel is unavailable, or for a linear without a float weight,
    falls back to the module's own forward."""
    if x.shape[1] > 1 and _F16_HEAD_GEMV is not None and _float_weight(linear) is not None:
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
        and Dv % gdn_sg(B) == 0
        and self.head_k_dim % 32 == 0
    ):
        if _ROUTE_DEBUG:
            _log_gdn_route_once("vlm", S, B, gdn_sink, mask, "fused-decode")
        return _gdn_fused_decode_body(self, inputs, cache, vlm_cache_advance=True)
    if (
        not getattr(self, "_gdn_fused_verify", False)
        or gdn_sink is None
        or S <= 1
        or _gdn_fused_verify_kernel is None
        or Dv % _GDN_VERIFY_ROWS != 0
        or self.head_k_dim % 32 != 0
    ):
        if _ROUTE_DEBUG:
            _log_gdn_route_once("vlm", S, B, gdn_sink, mask, "STOCK-unfused")
        return _FUSED_VERIFY_PATCH.stock(
            self, inputs, mask, cache, gdn_sink, target_verify
        )
    if _ROUTE_DEBUG:
        _log_gdn_route_once("vlm", S, B, gdn_sink, mask, "fused-verify")

    return _gdn_fused_verify_body(self, inputs, mask, cache, gdn_sink)


def _gdn_fused_verify_body(self, inputs, mask, cache, gdn_sink, *, records=False):
    """Fused multi-position verify forward, shared by the class patch
    (stock fallback loads) and the owned GatedDeltaNet class. Callers
    guarantee the route conditions: sink set, S>1, kernel available,
    Dv%16==0, Dk%32==0.

    ``records`` runs the records form of the scan at B=1: the sink carries
    a GdnRecords and the cache a pending replay of all S positions, so the
    caller's rollback must know GdnRecords (the owned qwen3.5 one does)."""
    B, S, _ = inputs.shape
    Dv = self.head_v_dim

    multi = None
    if is_folded(self.in_proj_z):
        mixed_qkv, z = shared_linears((self.in_proj_qkv, self.in_proj_z), inputs)
    else:
        # A q4_0 file runs qkv, z and the b/a pair as one multi-weight
        # matmul at verify widths.
        ba_kq = _gdn_ba_kq(self)
        multi = kquant_multi(
            (self.in_proj_qkv, self.in_proj_z)
            + ((ba_kq,) if ba_kq is not None
               else (self.in_proj_b, self.in_proj_a)),
            inputs)
        if multi is not None:
            mixed_qkv, z = multi[0], multi[1]
        else:
            mixed_qkv = self.in_proj_qkv(inputs)
            z = _bf16_verify_linear(self.in_proj_z, inputs)
    # b and a are tiny [D -> Hv] dense rows; two separate dispatches cost
    # ~2x the combined one at these sizes, so run them as one [2 * Hv]
    # gemv against the concatenated weight (row-independent, bit-exact).
    # Keyed on the modules as well: an adapter wrapper replaces the module
    # and has no weight of its own.
    wb = _float_weight(self.in_proj_b)
    wa = _float_weight(self.in_proj_a)
    ba_key = (id(self.in_proj_b), id(self.in_proj_a), id(wb), id(wa))
    cba = getattr(self, "_gdn_verify_ba_weight", None)
    if cba is None or cba[0] != ba_key:
        fuse_ok = (
            _F16_HEAD_GEMV is not None
            and wb is not None
            and wa is not None
            and getattr(self.in_proj_b, "bias", None) is None
            and getattr(self.in_proj_a, "bias", None) is None
            and wb.dtype == wa.dtype
        )
        wba = None
        if fuse_ok and wb is not None and wa is not None:
            wba = mx.concatenate([wb, wa], axis=0)
            mx.eval(wba)
        cba = (ba_key, wba)
        self._gdn_verify_ba_weight = cba
    if multi is not None:
        ba = (multi[2] if len(multi) == 3
              else mx.concatenate(list(multi[2:]), axis=-1))
    elif cba[1] is not None:
        ba = _f16_head_gemv(inputs, cba[1])
    elif (ba_kq := _gdn_ba_kq(self)) is not None:
        ba = ba_kq(inputs)
    else:
        ba = mx.concatenate(
            [
                _bf16_verify_linear(self.in_proj_b, inputs),
                _bf16_verify_linear(self.in_proj_a, inputs),
            ],
            axis=-1,
        )
    b = ba[..., : self.num_v_heads]
    a = ba[..., self.num_v_heads :]
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

    SGS, RPS = _GDN_VERIFY_SGS, _GDN_VERIFY_RPS
    Hv, Dk = self.num_v_heads, self.head_k_dim
    tiled = _kv_heads_tiled(self)
    template = [
        ("InT", conv_input.dtype),
        ("StT", state.dtype),
        ("Dk", Dk),
        ("Dv", Dv),
        ("Hk", self.num_k_heads),
        ("Hv", Hv),
        ("K_size", self.conv_kernel_size),
        ("SGS", SGS),
        ("RPS", RPS),
        ("TILED", tiled),
        ("CONV_BF16", 1),
    ]
    grid = (32, SGS, B * Hv * (Dv // _GDN_VERIFY_ROWS))
    records = records and B == 1 and _gdn_verify_rec_kernel is not None
    if records:
        pend = take_gdn_replay(cache) if cache is not None else None
        base, replay, steps = pend if pend is not None else (state, _no_records(), 0)
        pre, start, rd, rk, rg = _gdn_verify_rec_kernel(
            inputs=[conv_input, conv_weight, ba, self.A_log, self.dt_bias, base, S,
                    *replay, steps],
            template=template,
            grid=grid,
            threadgroup=(32, SGS, 1),
            output_shapes=[
                (B, S, Hv, Dv),
                # Written only after a replay; else the base is the start.
                base.shape if steps else (1,),
                (B, S, Hv, Dv),
                (B, S, Hv, Dk),
                (B, S, Hv),
            ],
            output_dtypes=[mx.float32, base.dtype, mx.float32, mx.float32,
                           mx.float32],
        )
        rollback = GdnRecords(start if steps else base, (rd, rk, rg))
        if cache is not None:
            # Full accept; the rollback shortens the replay on a rejection.
            set_gdn_replay(cache, rollback.base, rollback.records, S)
    else:
        pre, new_state, rollback = _gdn_fused_verify_kernel(
            inputs=[
                conv_input,
                conv_weight,
                ba,
                self.A_log,
                self.dt_bias,
                state,
                S,
            ],
            template=template,
            grid=grid,
            threadgroup=(32, SGS, 1),
            output_shapes=[
                (B, S, Hv, Dv),
                state.shape,
                # Per-position rollback states for positions 0..S-2;
                # position S-1 would duplicate state_out and no rollback
                # path reads it (every engine call site guards accepted <
                # bs - 1).
                (B, S - 1, Hv, Dv, Dk),
            ],
            output_dtypes=[mx.float32, state.dtype, state.dtype],
        )
        if cache is not None:
            cache[1] = new_state
    nrows = B * S * self.num_v_heads
    (y,) = _gdn_verify_gate_kernel(
        inputs=[pre, z, self.norm.weight, nrows],
        template=[
            ("InT", conv_input.dtype),
            ("Dv", Dv),
            # output gate: silu(z) (qwen3.5) or sigmoid(z) (qwen4exp)
            ("GATE_SIGMOID", int(bool(getattr(self, "gdn_gate_sigmoid", False)))),
        ],
        grid=(32, nrows, 1),
        threadgroup=(32, 8, 1),
        output_shapes=[(B, S, self.num_v_heads, Dv)],
        output_dtypes=[conv_input.dtype],
    )

    if cache is not None:
        if hasattr(cache, "advance"):
            cache.advance(S)
            from gmlx.models.qwen35.owned import (
                _qwen3_5_advance_left_padding_info,
                _qwen3_5_advance_lengths_info,
            )

            _qwen3_5_advance_left_padding_info(cache, S)
            _qwen3_5_advance_lengths_info(cache, S)

    if gdn_sink is not None:
        # The rejection rollback indexes the per-position intermediate states
        # (or replays the records) and slices the conv input; the q/k/v
        # entries are only read by the no-intermediate-states fallback, which
        # this path never hits.
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
                rollback,
            )
        )

    return _bf16_verify_linear(self.out_proj, y.reshape(B, S, -1))


_BATCHED_VERIFY_SDPA_PATCHED = False
# Stock upstream _target_verify_left_padded_attention, captured at patch
# install; tests use it as the bit-exact reference.
_STOCK_VERIFY_LEFT_PADDED = None


def _patch_batched_verify_sdpa() -> None:
    """Replace the per-position SDPA loop in _target_verify_left_padded_attention
    with a single batched call.

    For MTP verify (S=K+1 query positions), the stock code does S separate
    mx.fast.scaled_dot_product_attention calls per attention layer.  A single
    call with mask='causal' is numerically identical (verified bit-exact for
    GQA, B=1..4, d up to 4096) and saves S-1 kernel launches per layer.

    Left-padded batches (ragged prompt lengths) get the same treatment with
    an explicit [B, 1, S, T] boolean mask that encodes the per-row pad and
    causality together.  The stock path for that case groups rows by pad
    value and does an mx.take of full-depth K/V per group plus a
    per-query-position SDPA, so at depth every verify step re-reads the
    whole KV once more; the masked call touches each key exactly once.
    Masked columns contribute exp(-inf) = 0 to reductions, so the result is
    bit-exact against the group loop.  GMLX_VERIFY_RAGGED_MASK=0 restores
    the stock group-loop path (read per call)."""
    global _BATCHED_VERIFY_SDPA_PATCHED, _STOCK_VERIFY_LEFT_PADDED
    if _BATCHED_VERIFY_SDPA_PATCHED:
        return

    import os

    import gmlx.models.qwen35.language as _L

    _orig_left_padded = _L._target_verify_left_padded_attention
    _STOCK_VERIFY_LEFT_PADDED = _orig_left_padded
    _ragged_noted = set()

    def _note_ragged(route, B, L, T, pads):
        key = (route, B, L)
        if key in _ragged_noted:
            return
        _ragged_noted.add(key)
        # warn, not verbose_print: live verify traffic should never reach
        # this seam, so the one-shot must survive quiet serve logs.
        loadlog.warn(f"[verify] ragged-pads route: {route} B={B} S={L} T={T} "
                     f"max_pad={max(pads)}")

    def _batched_verify_attention(queries, keys, values, *, cache, scale, mask):
        if hasattr(cache, "bits") or queries.ndim != 4 or keys.ndim != 4:
            return _orig_left_padded(
                queries, keys, values, cache=cache, scale=scale, mask=mask
            )
        pads = getattr(cache, "_qwen3_5_decode_left_padding", None)
        if pads is None:
            pads = []
        padded = len(pads) > 0 and max(pads) > 0
        L = queries.shape[-2]
        if padded and (
            L <= 1
            or len(pads) != queries.shape[0]
            or isinstance(mask, mx.array)
            or os.environ.get("GMLX_VERIFY_RAGGED_MASK", "1") == "0"
        ):
            if L > 1:
                _note_ragged("stock-loop", queries.shape[0], L,
                             keys.shape[-2], pads)
            return _orig_left_padded(
                queries, keys, values, cache=cache, scale=scale, mask=mask
            )
        if L <= 1:
            return None
        if padded:
            _note_ragged("masked-sdpa", queries.shape[0], L,
                         keys.shape[-2], pads)
            T = keys.shape[-2]
            t = mx.arange(T)[None, None, :]
            end = (T - L) + mx.arange(L)[None, :, None]
            pad_arr = mx.array(pads, dtype=mx.int32)[:, None, None]
            sdpa_mask = ((t >= pad_arr) & (t <= end))[:, None, :, :]
        else:
            sdpa_mask = mask if isinstance(mask, mx.array) else "causal"
        return mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=scale, mask=sdpa_mask
        )

    _L._target_verify_left_padded_attention = link(_batched_verify_attention,
                                                   _orig_left_padded)
    _BATCHED_VERIFY_SDPA_PATCHED = True


_BF16_VERIFY_LINEAR_PATCHED = False


def _patch_bf16_verify_linear() -> None:
    """Route bf16 nn.Linear through the M-stationary GEMV-ext at verify width.

    MLX's bf16 matmul at small M (2..8) dispatches through a general GEMM path
    that runs at ~65-70% of peak BW. The GEMV-ext kernel (same weight read,
    M register accumulators) holds ~95% flat across M=2..8. This patches
    _target_verify_linear in the vendored qwen3.5 base so every bf16
    projection at verify width (MLP gate/up/down, attention q/k/v/o in
    UD-protected layers) benefits."""
    global _BF16_VERIFY_LINEAR_PATCHED
    if _BF16_VERIFY_LINEAR_PATCHED or _F16_HEAD_GEMV is None:
        return
    import gmlx.models.qwen35.language as _L

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
    from gmlx.models.qwen35.language import Qwen3_5Model

    _orig_model_call = Qwen3_5Model.__call__

    def _s0_guarded_call(self, inputs, inputs_embeds=None, **kw):
        if inputs_embeds is not None and inputs_embeds.shape[1] == 0:
            return self.norm(inputs_embeds)
        return _orig_model_call(self, inputs, inputs_embeds=inputs_embeds, **kw)

    Qwen3_5Model.__call__ = link(_s0_guarded_call, _orig_model_call)  # pyright: ignore[reportAttributeAccessIssue]
    _S0_MODEL_GUARD_INSTALLED = True


def _patch_gated_delta_fused_verify(model) -> None:
    """Enable the fused gated-delta verify kernel on every Qwen3_5GatedDeltaNet in
    the MTP target. Installs the class-level dispatch once (a no-op for instances
    without the per-instance ``_gdn_fused_verify`` flag) and flags this model's
    instances."""
    from gmlx.models.qwen35.language import Qwen3_5GatedDeltaNet

    _patch_qwen35_empty_sequence_guard()

    if (
        _gdn_fused_verify_kernel is None
        or not env_bool("GMLX_FUSED_GDN", True)
    ):
        return
    _FUSED_VERIFY_PATCH.install(
        Qwen3_5GatedDeltaNet, "__call__", _gdn_fused_verify_call)
    n = 0
    n_ba = 0
    cat_ba = env_bool("GMLX_GDN_BA_CAT", True)
    lm = getattr(model, "language_model", model)
    for m in lm.modules():
        if isinstance(m, Qwen3_5GatedDeltaNet):
            m._gdn_fused_verify = True
            n += 1
            # The S=1 decode branch above routes these instances through
            # _gdn_fused_decode_body too, so they need the b/a cat as much
            # as the mlx-lm text path does.
            if cat_ba and getattr(m, "_gdn_ba_weight", None) is None \
                    and _gdn_try_cat_ba(m):
                n_ba += 1
    ba = f", b/a matvecs concatenated on {n_ba}" if n_ba else ""
    loadlog.verbose_print(
        f"[patch] gated_delta: fused verify kernel enabled on {n} layers{ba}")


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

    base = cast(type[nn.Module], type(head))

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
