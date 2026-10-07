# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Asher Feldman
# Portions copyright (c) 2024 Apple Inc. (mlx-lm qwen3_next/qwen3_5 skeleton, MIT)
"""Vendored mlx-lm-style model for Qwen3.8-Flash-Next (GGUF arch ``qwen4exp``).

Neither pinned mlx-lm nor pinned mlx-vlm ships a qwen4_exp class; this module
is the runtime for llama.cpp's ``LLM_ARCH_QWEN4EXP`` conversions. It is the
qwen3.5 hybrid (gated DeltaNet on three of every four layers, gated full
attention on the fourth, a 512-expert top-10 MoE with a sigmoid-gated shared
expert on every layer) plus three mechanisms of its own:

  1. Hyper-connections. The residual is four parallel streams ``[T, 4, D]``
     initialised from the token embedding. Each sub-layer reads one mixed
     ``[T, D]`` row (grouped RMSNorm per stream, low-rank silu down / sigmoid
     up gate, mean over the streams) and writes back ``out * inject``, where
     ``inject = 2 * sigmoid(W xn / 4)`` is one scalar per stream. A final
     mixer without inject replaces output_norm.
  2. QSA sparse attention. A 4-head indexer scores mean-pooled blocks of
     ``compress_ratio`` keys (k_norm + rope at the block start position,
     ``sum_h relu(q_h . k_b)``), keeps the ``budget / ratio`` best complete
     blocks plus the incomplete tail, and attention is masked to that set.
     Dense while at most ``budget / ratio`` complete blocks exist.
  3. PLE n-gram hash embeddings on one layer: 2-gram and 3-gram hashes of the
     token history (EOS resets the context) index a huge quantized table; the
     rows gate into all four streams plus a dilated depthwise conv branch.

Wire facts the forward relies on: every norm weight is already ``(1 + w)``
baked (plain RMSNorm); the GDN output gate is sigmoid; GDN V heads are tiled
(V head ``hv`` reads K head ``hv % Hk``), so q/k are tiled explicitly before
the scan; attention ``q_proj`` carries ``[q | gate]`` per head; rotary is 64
of 256 dims with interleaved mrope sections, which for text-only positions is
plain NEOX rope.
"""

import importlib
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Any, List, Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.base import (
    BaseModelArgs,
    create_attention_mask,
    create_ssm_mask,
    scaled_dot_product_attention,
)
from mlx_lm.models.cache import ArraysCache, KVCache
from mlx_lm.models.gated_delta import gated_delta_update

from gmlx.models import owned
from gmlx.envflags import env_bool
from gmlx.tune.attention import blocked_attention
from gmlx.tune.gdn import training_gated_delta_update
from mlx_lm.models.switch_layers import SwitchGLU


def ensure_registered() -> None:
    """Make ``import mlx_lm.models.qwen4_exp`` resolve to this module,
    and expose ``QSAKVCache`` on the cache modules (prompt-cache save/load
    resolves cache classes by name there; mlx-vlm >= 0.6.4 vendors its own
    models/cache.py, so register on both when it is loaded)."""
    import mlx_lm.models.cache as _mlx_cache

    vlm_cache = sys.modules.get("mlx_vlm.models.cache")
    for mod in (_mlx_cache, vlm_cache):
        if mod is not None and not hasattr(mod, "QSAKVCache"):
            mod.QSAKVCache = QSAKVCache
    owned.install("mlx_lm.models.qwen4_exp", __name__)


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "qwen4_exp"
    hidden_size: int = 2560
    num_hidden_layers: int = 48
    vocab_size: int = 248320
    rms_norm_eps: float = 1e-6
    num_attention_heads: int = 24
    num_key_value_heads: int = 2
    head_dim: int = 256
    full_attention_interval: int = 4
    layer_types: Optional[List[str]] = None
    linear_num_value_heads: int = 48
    linear_num_key_heads: int = 16
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_conv_kernel_dim: int = 4
    rope_theta: float = 10000000.0
    partial_rotary_factor: float = 0.25
    mrope_section: List[int] = field(default_factory=lambda: [11, 11, 10])
    max_position_embeddings: int = 262144
    num_experts: int = 512
    num_experts_per_tok: int = 10
    moe_intermediate_size: int = 640
    shared_expert_intermediate_size: int = 640
    norm_topk_prob: bool = True
    hc_count: int = 4
    hc_lowrank: int = 320
    indexer_n_heads: int = 4
    indexer_head_dim: int = 128
    indexer_budget: int = 2048
    compress_ratios: Optional[List[int]] = None
    ple_layer_ids: List[int] = field(default_factory=list)
    ple_ngram_size: int = 3
    ple_heads_per_ngram: int = 8
    ple_conv_kernel: int = 4
    ple_eos_token_id: int = 0
    ple_image_token_id: Optional[int] = None
    ple_embed_dim: int = 160
    ple_table_rows: int = 0
    ple_layer_multipliers: List[int] = field(default_factory=list)
    ple_head_offsets: List[int] = field(default_factory=list)
    ple_head_vocab_sizes: List[int] = field(default_factory=list)
    tie_word_embeddings: bool = False
    kv_head_layout: str = "tiled"

    def __post_init__(self):
        if self.layer_types is None:
            self.layer_types = [
                "full_attention"
                if (i + 1) % self.full_attention_interval == 0
                else "linear_attention"
                for i in range(self.num_hidden_layers)
            ]
        if self.compress_ratios is None:
            self.compress_ratios = [0] * self.num_hidden_layers


# Norms


def _grouped_rms_norm(x: mx.array, weight: mx.array, eps: float) -> mx.array:
    """RMSNorm over the last axis of ``[..., hc, D]`` with a ``[hc * D]``
    gamma: one statistic per stream, one gain per element."""
    hc, d = x.shape[-2], x.shape[-1]
    return mx.fast.rms_norm(x, None, eps) * weight.reshape(hc, d)


class GroupedRMSNorm(nn.Module):
    """RMSNorm applied per residual stream, gamma over all streams."""

    def __init__(self, dims: int, eps: float):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        return _grouped_rms_norm(x, self.weight, self.eps)


class RMSNormGatedSigmoid(nn.Module):
    """``rms_norm(x) * sigmoid(gate)``: the GDN output norm (qwen3.5 uses
    silu here; this is the one numerical difference in the GDN)."""

    def __init__(self, dims: int, eps: float):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps

    def __call__(self, x: mx.array, gate: mx.array) -> mx.array:
        return mx.fast.rms_norm(x, self.weight, self.eps) * mx.sigmoid(gate)


# Hyper-connections


class HyperConnection(nn.Module):
    """Low-rank sigmoid-gated stream mixer (+ optional per-stream inject)."""

    def __init__(self, hidden: int, hc: int, lowrank: int, eps: float,
                 inject: bool = True):
        super().__init__()
        self.hc = hc
        self.hidden = hidden
        self.norm = GroupedRMSNorm(hc * hidden, eps)
        self.down = nn.Linear(hc * hidden, lowrank, bias=False)
        self.up = nn.Linear(lowrank, hc * hidden, bias=False)
        if inject:
            self.inject = nn.Linear(hc * hidden, hc, bias=False)

    def __call__(self, h: mx.array):
        B, T, hc, D = h.shape
        # A LoRA wrapper (training, or a served adapter) has no weight of
        # its own, so the kernels that read one fall back to calling it.
        down, up = self.down, self.up
        inject = self.inject if "inject" in self else None
        plain = ("weight" in down and "weight" in up
                 and (inject is None or "weight" in inject))
        # The kernels have no backward: training takes the ops.
        kern = not self.training
        if kern and B * T <= 8 and plain and self._hclr_ok(h.dtype):
            norm, front, epi = _kq_hc()
            xn = norm(h, self.norm.weight, self.norm.eps)
            lo, inj = front(xn, down.weight, inject.weight,
                            self.norm.weight.dtype)
            return epi(lo, up.weight, xn), inj
        xn = _hc_norm_kern(h, self.norm.weight, self.norm.eps) if kern else None
        if xn is None:
            xn = self.norm(h)
        xf = xn.reshape(B, T, hc * D)
        lo = nn.silu(down(xf) * (1.0 / hc))
        up_out = up(lo)
        if inject is None:
            return _hc_mix(up_out, xn)
        inj_out = (_hc_inject_kern(xf, inject.weight)
                   if kern and "weight" in inject else None)
        if inj_out is None:
            inj_out = inject(xf)
        r = _hc_epi_kern(up_out, xn, inj_out) if kern else None
        if r is not None:
            return r
        # inj stays eager: compiling this sigmoid shifts its fp32 lsb and
        # the drift compounds across 96 combines per token
        return _hc_mix(up_out, xn), 2.0 * mx.sigmoid(inj_out * (1.0 / hc))

    def _hclr_ok(self, h_dtype) -> bool:
        """Fused-path eligibility: kq hc_lowrank ops present, q8_0 down/up
        wire, f32 inject, half norm gain, kernel-aligned shapes. Cached per
        module after the first call (weights are frozen post-load)."""
        ok = self.__dict__.get("_hclr_cache")
        if ok is None:
            ok = (
                self.hc == 4
                and "inject" in self
                and getattr(self.down, "kquant_type", None) == "q8_0"
                and getattr(self.up, "kquant_type", None) == "q8_0"
                and self.inject.weight.dtype == mx.float32
                and self.norm.weight.dtype in (mx.float16, mx.bfloat16)
                and self.hidden % 64 == 0
                and self.down.weight.shape[0] % 32 == 0
                and self.down.weight.shape[0] <= 512
            )
            self.__dict__["_hclr_cache"] = ok
        return (ok and _kq_hc() is not None
                and (h_dtype == mx.float32
                     or h_dtype == self.norm.weight.dtype))


# Prefill-width HC kernels: the compiled epilogues still run the
# [B, T, hc, D] elementwise chains at ~1/4 of achievable bandwidth (the
# mean-over-streams and broadcast-combine patterns defeat mx.compile's
# fusion), and the inject GEMV's [hc, hc*D] fp32 weight forces the
# stock matmul onto a promoted fp32 path that re-reads x. Four
# single-pass kernels cover the norm, the mix+inject epilogue, the
# combine, and the inject GEMV; each falls back to the op path when
# ineligible. GMLX_Q4_HC_PREFILL_KERN=0 disables all four.

_HC_NORM_SRC = r"""
    uint lane = thread_position_in_threadgroup.x;
    uint sg   = thread_position_in_threadgroup.y;
    uint row  = thread_position_in_grid.z;      // b*T*hc + ...
    uint s_idx = row % HC;
    uint flat = sg * 32 + lane;
    auto x_ = x + (size_t)row * D;
    auto y_ = y + (size_t)row * D;
    auto g_ = gamma + (size_t)s_idx * D;

    threadgroup float ssq[SGS];
    float acc = 0.0f;
    for (uint d = flat; d < (uint)D; d += SGS * 32) {
        float xv = (float)x_[d];
        acc += xv * xv;
    }
    acc = simd_sum(acc);
    if (lane == 0) ssq[sg] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = 0.0f;
    for (int s = 0; s < SGS; ++s) tot += ssq[s];
    float scale = rsqrt(tot / (float)D + eps[0]);
    // normalized value rounds to InT before the gain multiply, matching
    // mx.fast.rms_norm's rounding points
    for (uint d = flat; d < (uint)D; d += SGS * 32)
        y_[d] = (InT)((float)(InT)((float)x_[d] * scale) * (float)g_[d]);
"""

_HC_EPI_SRC = r"""
    uint gid = thread_position_in_grid.x;       // b*T*D + ...
    uint bt = gid / D;
    uint d = gid % D;
    auto u_ = upo + (size_t)bt * HC * D + d;
    auto x_ = xn + (size_t)bt * HC * D + d;
    // InT casts mirror the eager op chain's per-op rounding (sigmoid ->
    // InT gate, gate*xn -> InT product, fp32 mean); without them the
    // fp32 drift compounds across the 96 combines per token
    float acc = 0.0f;
    for (int s = 0; s < HC; ++s) {
        float uv = (float)u_[(size_t)s * D];
        float sig = (float)(InT)(1.0f / (1.0f + metal::precise::exp(-uv)));
        acc += (float)(InT)(sig * (float)x_[(size_t)s * D]);
    }
    mixed[gid] = (InT)(acc * (1.0f / (float)HC));
    if (d < (uint)HC) {
        float iv = (float)(InT)(injo[(size_t)bt * HC + d]
                                * (1.0f / (float)HC));
        inj[(size_t)bt * HC + d] =
            2.0f * (float)(InT)(1.0f / (1.0f + metal::precise::exp(-iv)));
    }
"""

_HC_COMBINE_SRC = r"""
    uint gid = thread_position_in_grid.x;       // b*T*hc*D + ...
    uint d = gid % D;
    uint s_idx = (gid / D) % HC;
    uint bt = gid / (HC * D);
    // product rounds to InT before the add, mirroring the eager ops
    yn[gid] = (InT)((float)h[gid]
        + (float)(InT)((float)out[(size_t)bt * D + d]
                       * inj[(size_t)bt * HC + s_idx]));
"""

_HC_INJECT_SRC = r"""
    uint lane = thread_position_in_threadgroup.x;
    uint sg   = thread_position_in_threadgroup.y;
    uint bt   = thread_position_in_grid.z;      // b*T + ...
    uint flat = sg * 32 + lane;
    auto x_ = x + (size_t)bt * N;

    // fp32 accumulation over fp32 weights, matching the stock promoted
    // matmul; only the reduction order differs (fp32 lsb class, rounded
    // to InT by the epi kernel before the sigmoid)
    threadgroup float part[SGS * HC];
    float acc[HC];
    for (int s = 0; s < HC; ++s) acc[s] = 0.0f;
    for (uint i = flat; i < (uint)N; i += SGS * 32) {
        float xv = (float)x_[i];
        for (int s = 0; s < HC; ++s)
            acc[s] += xv * w[(size_t)s * N + i];
    }
    for (int s = 0; s < HC; ++s) {
        float v = simd_sum(acc[s]);
        if (lane == 0) part[sg * HC + s] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0 && lane < (uint)HC) {
        float tot = 0.0f;
        for (int g = 0; g < SGS; ++g) tot += part[g * HC + lane];
        y[(size_t)bt * HC + lane] = tot;
    }
"""

_hc_prefill_kerns = None


def _hc_kerns():
    """The prefill-width HC kernels (norm, epi, combine, inject), or None."""
    global _hc_prefill_kerns
    if _hc_prefill_kerns is None:
        from gmlx.envflags import env_bool
        if (env_bool("GMLX_Q4_HC_PREFILL_KERN", True)
                and mx.metal.is_available()
                and mx.default_device().type == mx.DeviceType.gpu):
            _hc_prefill_kerns = (
                mx.fast.metal_kernel(
                    name="gmlx_hc_norm",
                    input_names=["x", "gamma", "eps"],
                    output_names=["y"], source=_HC_NORM_SRC),
                mx.fast.metal_kernel(
                    name="gmlx_hc_epi",
                    input_names=["upo", "xn", "injo"],
                    output_names=["mixed", "inj"], source=_HC_EPI_SRC),
                mx.fast.metal_kernel(
                    name="gmlx_hc_combine",
                    input_names=["h", "out", "inj"],
                    output_names=["yn"], source=_HC_COMBINE_SRC),
                mx.fast.metal_kernel(
                    name="gmlx_hc_inject",
                    input_names=["x", "w"],
                    output_names=["y"], source=_HC_INJECT_SRC),
            )
        else:
            _hc_prefill_kerns = False
    return _hc_prefill_kerns or None


def _hc_norm_kern(x: mx.array, weight: mx.array, eps: float) -> mx.array:
    B, T, hc, D = x.shape
    kerns = _hc_kerns()
    SGS = 8
    if kerns is None or B * T <= 8 or D % (SGS * 32) or x.dtype not in (
            mx.float16, mx.bfloat16):
        return None
    return kerns[0](
        inputs=[x, weight.astype(x.dtype), mx.array([eps], dtype=mx.float32)],
        template=[("InT", x.dtype), ("D", D), ("HC", hc), ("SGS", SGS)],
        grid=(32, SGS, B * T * hc), threadgroup=(32, SGS, 1),
        output_shapes=[x.shape], output_dtypes=[x.dtype])[0]


def _hc_epi_kern(up_out: mx.array, xn: mx.array, inj_out: mx.array):
    B, T, hc, D = xn.shape
    kerns = _hc_kerns()
    if (kerns is None or B * T <= 8 or D < hc
            or xn.dtype not in (mx.float16, mx.bfloat16)):
        return None
    return kerns[1](
        inputs=[up_out, xn, inj_out.astype(mx.float32)],
        template=[("InT", xn.dtype), ("D", D), ("HC", hc)],
        grid=(B * T * D, 1, 1), threadgroup=(min(256, B * T * D), 1, 1),
        output_shapes=[(B, T, D), (B, T, hc)],
        output_dtypes=[xn.dtype, mx.float32])


def _hc_inject_kern(xf: mx.array, weight: mx.array):
    B, T, N = xf.shape
    kerns = _hc_kerns()
    SGS = 8
    if (kerns is None or B * T <= 8 or N % (SGS * 32)
            or weight.dtype != mx.float32
            or xf.dtype not in (mx.float16, mx.bfloat16)):
        return None
    hc = weight.shape[0]
    return kerns[3](
        inputs=[xf, weight],
        template=[("InT", xf.dtype), ("N", N), ("HC", hc), ("SGS", SGS)],
        grid=(32, SGS, B * T), threadgroup=(32, SGS, 1),
        output_shapes=[(B, T, hc)], output_dtypes=[mx.float32])[0]


def _hc_combine_kern(h: mx.array, out: mx.array, inject: mx.array):
    B, T, hc, D = h.shape
    kerns = _hc_kerns()
    if (kerns is None or h.dtype not in (mx.float16, mx.bfloat16)
            or out.dtype != h.dtype or inject.dtype != mx.float32
            or inject.shape != (B, T, hc)):
        return None
    n = B * T * hc * D
    return kerns[2](
        inputs=[h, out, inject],
        template=[("InT", h.dtype), ("D", D), ("HC", hc)],
        grid=(n, 1, 1), threadgroup=(min(256, n), 1, 1),
        output_shapes=[h.shape], output_dtypes=[h.dtype])[0]


@mx.compile
def _hc_mix(up_out: mx.array, xn: mx.array) -> mx.array:
    B, T, hc, D = xn.shape
    gate = mx.sigmoid(up_out).reshape(B, T, hc, D)
    return (gate * xn).mean(axis=2)


@mx.compile
def _hc_combine_ops(h: mx.array, out: mx.array, inject: mx.array) -> mx.array:
    return h + out[:, :, None, :] * inject[..., None]


def _hc_combine(h: mx.array, out: mx.array, inject: mx.array,
                kern: bool = True) -> mx.array:
    """``kern`` False keeps to the ops, which have a backward."""
    if kern and h.shape[0] * h.shape[1] > 8:
        y = _hc_combine_kern(h, out, inject)
        if y is not None:
            return y
    return _hc_combine_ops(h, out, inject)


# Gated DeltaNet


class GatedDeltaNet(nn.Module):
    """qwen3.5 gated DeltaNet with a sigmoid output gate and the GGUF tiled
    K->V head pairing applied explicitly. Attribute and cache layout match
    mlx-lm's ``GatedDeltaNet`` so the fused S=1 decode kernel body in
    ``gdn_patches`` (conv -> silu -> q/k norm -> scan -> gated norm in one
    launch) runs unchanged; ``gdn_gate_sigmoid`` selects its gate."""

    gdn_gate_sigmoid = True

    def __init__(self, args: ModelArgs):
        super().__init__()
        self._fused_decode = False
        self._fused_verify = False
        self.hidden_size = args.hidden_size
        self.num_v_heads = args.linear_num_value_heads
        self.num_k_heads = args.linear_num_key_heads
        self.head_k_dim = args.linear_key_head_dim
        self.head_v_dim = args.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        self.conv_kernel_size = args.linear_conv_kernel_dim
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.eps = args.rms_norm_eps

        self.in_proj_qkv = nn.Linear(self.hidden_size, self.conv_dim, bias=False)
        self.in_proj_z = nn.Linear(self.hidden_size, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
        self.in_proj_a = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            padding=0,
            bias=False,
        )
        self.dt_bias = mx.ones((self.num_v_heads,))
        self.A_log = mx.zeros((self.num_v_heads,))
        self.norm = RMSNormGatedSigmoid(self.head_v_dim, eps=self.eps)
        self.out_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)

    def _fused_decode_ok(self, B, S, mask, cache) -> bool:
        if not (self._fused_decode and S == 1 and cache is not None
                and cache[1] is not None and cache[1].shape[0] == B):
            return False
        if mask is not None and isinstance(mask, mx.array):
            return False
        import gmlx.upstream.gdn_patches as _gp

        return (
            _gp._gdn_fused_decode_kernel is not None and _gp.gpu_active()
            and self.head_v_dim % _gp.gdn_sg(B) == 0
            and self.head_k_dim % 32 == 0
        )

    def _fused_verify_ok(self, B, mask, cache) -> bool:
        if not (self._fused_verify and cache is not None):
            return False
        if mask is not None and not (
            isinstance(mask, mx.array) and mask.ndim == 2 and mask.shape[0] == B
        ):
            return False
        import gmlx.upstream.gdn_patches as _gp

        return (
            _gp._gdn_fused_verify_kernel is not None and _gp.gpu_active()
            and self.head_v_dim % _gp.gdn_sg(B) == 0
            and self.head_k_dim % 32 == 0
        )

    def __call__(self, inputs: mx.array, mask: Optional[mx.array] = None,
                 cache=None, gdn_sink=None) -> mx.array:
        """``gdn_sink`` (a list) marks the MTP verify forward: every call
        appends the record ``rollback_verify_sink`` needs to rewind this
        layer's cache to a shorter accepted prefix."""
        B, S, _ = inputs.shape
        if self._fused_decode_ok(B, S, mask, cache):
            import gmlx.upstream.gdn_patches as _gp

            return _gp._gdn_fused_decode_body(self, inputs, cache)
        pre = (cache[0], cache[1]) if cache is not None else (None, None)
        if gdn_sink is not None and S > 1 and self._fused_verify_ok(B, mask, cache):
            import gmlx.upstream.gdn_patches as _gp

            rec: list = []
            out = _gp._gdn_fused_verify_body(self, inputs, mask, cache, rec)
            gdn_sink.append({
                "kind": "gdn", "layer": self, "cache": cache, "pre": pre,
                "inputs": inputs, "mask": mask, "conv_input": rec[-1][9],
                "inter": rec[-1][11], "K": self.conv_kernel_size,
            })
            return out
        qkv = self.in_proj_qkv(inputs)
        z = self.in_proj_z(inputs).reshape(B, S, self.num_v_heads, self.head_v_dim)
        b = self.in_proj_b(inputs)
        a = self.in_proj_a(inputs)

        n_keep = self.conv_kernel_size - 1
        conv_state = None
        if cache is not None and cache[0] is not None:
            conv_state = cache[0]
            if conv_state.shape[0] != B:
                conv_state = None
        if conv_state is None:
            conv_state = mx.zeros((B, n_keep, self.conv_dim), dtype=inputs.dtype)

        if mask is not None:
            if mask.shape[0] != B:
                mask = None
            else:
                qkv = mx.where(mask[..., None], qkv, 0)
        conv_input = mx.concatenate([conv_state, qkv], axis=1)
        if cache is not None:
            lengths = getattr(cache, "lengths", None)
            if lengths is not None:
                ends = mx.clip(lengths, 0, S)
                positions = (ends[:, None] + mx.arange(n_keep))[..., None]
                cache[0] = mx.take_along_axis(conv_input, positions, axis=1)
            else:
                cache[0] = mx.contiguous(conv_input[:, -n_keep:, :])
        if gdn_sink is not None:
            gdn_sink.append({
                "kind": "gdn", "layer": self, "cache": cache, "pre": pre,
                "inputs": inputs, "mask": mask, "conv_input": conv_input,
                "inter": None, "K": self.conv_kernel_size,
            })
        conv_out = nn.silu(self.conv1d(conv_input))

        q, k, v = [
            t.reshape(B, S, h, d)
            for t, h, d in zip(
                mx.split(conv_out, [self.key_dim, 2 * self.key_dim], axis=-1),
                (self.num_k_heads, self.num_k_heads, self.num_v_heads),
                (self.head_k_dim, self.head_k_dim, self.head_v_dim),
            )
        ]
        inv_scale = self.head_k_dim ** -0.5
        q = (inv_scale ** 2) * mx.fast.rms_norm(q, None, self.eps)
        k = inv_scale * mx.fast.rms_norm(k, None, self.eps)
        if self.num_v_heads != self.num_k_heads:
            # GGUF tiled V layout: V head hv pairs with K head hv % Hk.
            r = self.num_v_heads // self.num_k_heads
            q = mx.tile(q, [1, 1, r, 1])
            k = mx.tile(k, [1, 1, r, 1])

        state = cache[1] if cache is not None else None
        if state is not None and state.shape[0] != B:
            state = None
        if self.training and cache is None:
            out, state = training_gated_delta_update(
                q, k, v, a, b, self.A_log, self.dt_bias, state, mask
            )
        else:
            out, state = gated_delta_update(
                q, k, v, a, b, self.A_log, self.dt_bias, state, mask,
                use_kernel=not self.training,
            )
        if cache is not None:
            cache[1] = state
            if hasattr(cache, "advance"):
                cache.advance(S)
        out = self.norm(out, z)
        return self.out_proj(out.reshape(B, S, -1))


_KQ_TOPK_UNSET = object()
_kq_topk_fn = _KQ_TOPK_UNSET
_kq_score_fn = _KQ_TOPK_UNSET
_kq_hc_fns = _KQ_TOPK_UNSET
_kq_paged_fn = _KQ_TOPK_UNSET
_kq_bs_fn = _KQ_TOPK_UNSET


def _kq_hc():
    """kq fused low-rank hyper-connection (norm, front, epilogue) ops, or
    None (eager op chain). ``GMLX_Q4_HC_FUSED=0`` disables."""
    global _kq_hc_fns
    if _kq_hc_fns is _KQ_TOPK_UNSET:
        from gmlx.envflags import env_bool
        fns = None
        if env_bool("GMLX_Q4_HC_FUSED", True):
            try:
                import mlx_kquant as _kq
                norm = getattr(_kq, "hc_lowrank_norm", None)
                front = getattr(_kq, "hc_lowrank_front", None)
                epi = getattr(_kq, "hc_lowrank_epilogue", None)
                if (norm is not None and front is not None
                        and epi is not None
                        and mx.default_device().type == mx.DeviceType.gpu):
                    fns = (norm, front, epi)
            except ImportError:
                pass
        _kq_hc_fns = fns
    return _kq_hc_fns


def _kq_paged():
    """kq page-gather decode sdpa with 4-row pages at head_dim 256, or
    None (gathered-KV eager path). ``GMLX_Q4_QSA_PAGED_SDPA=0`` disables.
    Requires a kq build whose sdpa_decode_gqa_paged accepts tile_c=4
    (probed once with a dry call)."""
    global _kq_paged_fn
    if _kq_paged_fn is _KQ_TOPK_UNSET:
        from gmlx.envflags import env_bool
        fn = None
        if env_bool("GMLX_Q4_QSA_PAGED_SDPA", True):
            try:
                import mlx_kquant as _kq
                cand = getattr(_kq, "sdpa_decode_gqa_paged", None)
                if cand is not None and mx.default_device().type == mx.DeviceType.gpu:
                    try:
                        cand(mx.zeros((1, 2, 1, 256), dtype=mx.bfloat16),
                             mx.zeros((1, 1, 8, 256), dtype=mx.bfloat16),
                             mx.zeros((1, 1, 8, 256), dtype=mx.bfloat16),
                             1.0, mx.zeros((1, 1, 2), dtype=mx.int32),
                             tile_c=4)
                        fn = cand
                    except (TypeError, ValueError):
                        fn = None  # older kq: fixed 16-row pages
            except ImportError:
                pass
        _kq_paged_fn = fn
    return _kq_paged_fn


def _kq_bs_prefill():
    """kq block-sparse FA prefill over QSA-selected 4-row pages, or None
    (dense-masked stock FA). ``GMLX_Q4_QSA_BS_PREFILL=0`` disables."""
    global _kq_bs_fn
    if _kq_bs_fn is _KQ_TOPK_UNSET:
        from gmlx.envflags import env_bool
        fn = None
        if env_bool("GMLX_Q4_QSA_BS_PREFILL", True):
            try:
                import mlx_kquant as _kq
                if mx.default_device().type == mx.DeviceType.gpu:
                    fn = getattr(_kq, "sdpa_prefill_block_sparse", None)
            except ImportError:
                pass
        _kq_bs_fn = fn
    return _kq_bs_fn


def _kq_topk():
    """kq radix top-k for the QSA block selection, or None (stock
    argpartition). ``GMLX_Q4_QSA_KQ_TOPK=0`` disables."""
    global _kq_topk_fn
    if _kq_topk_fn is _KQ_TOPK_UNSET:
        from gmlx.envflags import env_bool
        fn = None
        if env_bool("GMLX_Q4_QSA_KQ_TOPK", True):
            try:
                import mlx_kquant as _kq
                if mx.default_device().type == mx.DeviceType.gpu:
                    fn = getattr(_kq, "dsa_topk_indices", None)
            except ImportError:
                pass
        _kq_topk_fn = fn
    return _kq_topk_fn


def _kq_score():
    """kq fused decode-width indexer score (4-head band), or None.
    ``GMLX_Q4_QSA_KQ_SCORE=0`` disables. Requires a kq build whose
    dsa_indexer_score_decode accepts 4 heads (probed once with a dry
    shape check on the host validator)."""
    global _kq_score_fn
    if _kq_score_fn is _KQ_TOPK_UNSET:
        from gmlx.envflags import env_bool
        fn = None
        if env_bool("GMLX_Q4_QSA_KQ_SCORE", True):
            try:
                import mlx_kquant as _kq
                cand = getattr(_kq, "dsa_indexer_score_decode", None)
                if cand is not None and mx.default_device().type == mx.DeviceType.gpu:
                    try:
                        cand(mx.zeros((1, 4, 1, 128), dtype=mx.bfloat16),
                             mx.zeros((1, 8, 128), dtype=mx.bfloat16),
                             mx.zeros((1, 1, 4), dtype=mx.bfloat16), 0, 4)
                        fn = cand
                    except ValueError:
                        fn = None  # older kq: 64-head band only
            except ImportError:
                pass
        _kq_score_fn = fn
    return _kq_score_fn


# mrope (vision positions). Text-only calls pass positions=None and take the
# scalar-offset mx.fast.rope path; the two agree exactly when the three
# position streams are equal (interleaved sections tile the same frequency
# ladder). The VLM wrapper passes [3, B, L] t/h/w position ids.


def _mrope_selector(sections: List[int], half: int) -> mx.array:
    """``[half]`` stream index per frequency: freq ``j`` reads stream
    ``j % 3`` while ``j < sections[j % 3] * 3`` (HF interleaved layout)."""
    sel = []
    for j in range(half):
        src = j % 3
        assert j < sections[src] * 3, (j, sections)
        sel.append(src)
    return mx.array(sel, dtype=mx.int32)


def _mrope_cos_sin(positions: mx.array, rotary_dims: int, base: float,
                   selector: mx.array):
    """``positions [3, B, L]`` -> ``(cos, sin)`` each ``[B, L, rotary_dims]``
    in the non-traditional (rotate-half) layout."""
    half = rotary_dims // 2
    inv = mx.power(base, -mx.arange(0, half, dtype=mx.float32) / half)
    freqs = positions.astype(mx.float32)[..., None] * inv  # [3, B, L, half]
    _, B, L, _ = freqs.shape
    idx = mx.broadcast_to(selector[None, None, None, :], (1, B, L, half))
    freqs = mx.take_along_axis(freqs, idx.astype(mx.uint32), axis=0)[0]
    emb = mx.concatenate([freqs, freqs], axis=-1)
    return mx.cos(emb), mx.sin(emb)


def _apply_rope_cos_sin(x: mx.array, cos: mx.array, sin: mx.array,
                        rotary_dims: int) -> mx.array:
    """Rotate the first ``rotary_dims`` of ``x [B, H, L, D]`` with
    ``cos/sin [B, L, rotary_dims]`` (rotate-half pairing)."""
    half = rotary_dims // 2
    xr, xp = x[..., :rotary_dims], x[..., rotary_dims:]
    rot = mx.concatenate([-xr[..., half:], xr[..., :half]], axis=-1)
    c, sn = cos[:, None].astype(x.dtype), sin[:, None].astype(x.dtype)
    return mx.concatenate([xr * c + rot * sn, xp], axis=-1)


# GMLX_DECODE_LAYER_PROFILE=1: eval after each decode-layer part (ple, hc,
# attn, gdn, ffn) and print each part's wall per token at exit. Attribution
# only: the syncs slow the run and stop one step overlapping the next, so a
# cost that needs that overlap does not show. Level 2 also evals inside
# attention: a.proj (q, k, v), a.cache (append), a.blocks (pooled index
# keys), a.score, a.topk, a.core (gather and attention), a.out.
# GMLX_LAYER_PROFILE_PREFILL=1 also marks steps wider than one token.
_LAYER_PROFILE_LEVEL = int(os.environ.get("GMLX_DECODE_LAYER_PROFILE", "0") or 0)
_LAYER_PROFILE = _LAYER_PROFILE_LEVEL >= 1
_SUB_PROFILE = _LAYER_PROFILE_LEVEL >= 2
_PROFILE_WIDE = os.environ.get("GMLX_LAYER_PROFILE_PREFILL", "0") == "1"
_PROF: dict = {}
_PROF_CALLS = [0]
_PROF_LOG: list = []


def _prof_on(width: int) -> bool:
    return _LAYER_PROFILE and (width == 1 or _PROFILE_WIDE)


def _subprof_on(width: int) -> bool:
    return _SUB_PROFILE and (width == 1 or _PROFILE_WIDE)


def _prof_mark(key: str, li: int, arr, t0: float) -> float:
    import time

    mx.eval(arr)
    t1 = time.perf_counter()
    _PROF[(key, li)] = _PROF.get((key, li), 0.0) + (t1 - t0)
    if _SUB_PROFILE:
        w1 = time.time()
        _PROF_LOG.append((key, li, w1 - (t1 - t0), w1))
    return t1


def _prof_reset() -> None:
    _PROF.clear()
    _PROF_LOG.clear()
    _PROF_CALLS[0] = 0


def _prof_summary() -> dict:
    """Per-token milliseconds by part, and by layer for each part."""
    n = _PROF_CALLS[0]
    if not n:
        return {}
    parts: dict = {}
    layers: dict = {}
    for (k, li), v in _PROF.items():
        parts[k] = parts.get(k, 0.0) + 1e3 * v / n
        layers.setdefault(k, {})[li] = 1e3 * v / n
    return {"tokens": n, "parts": parts, "layers": layers}


def _prof_dump() -> None:
    s = _prof_summary()
    if not s:
        return
    parts = s["parts"]
    top = [k for k in ("ple", "hc", "attn", "gdn", "ffn", "head") if k in parts]
    print(
        "[layerprof] per token ms: "
        + " | ".join(f"{k} {parts[k]:.1f}" for k in top)
        + f" | total {sum(parts[k] for k in top):.1f} over {s['tokens']} tokens",
        flush=True,
    )
    for k in ("attn", "gdn", "ffn"):
        rows = sorted(s["layers"].get(k, {}).items())
        if rows:
            print(
                f"[layerprof] {k} by layer ms/token: "
                + " ".join(f"{li}:{v:.2f}" for li, v in rows),
                flush=True,
            )
    subs = sorted(k for k in parts if "." in k)
    if subs:
        print(
            "[layerprof] sub per token ms: "
            + " | ".join(f"{k} {parts[k]:.1f}" for k in subs),
            flush=True,
        )
        for k in subs:
            rows = sorted(s["layers"][k].items())
            print(
                f"[layerprof] {k} by layer ms/token: "
                + " ".join(f"{li}:{v:.2f}" for li, v in rows),
                flush=True,
            )
    path = os.environ.get("GMLX_DECODE_LAYER_PROFILE_LOG")
    if path and _PROF_LOG:
        with open(path, "w") as f:
            for key, li, w0, w1 in _PROF_LOG:
                f.write(f"{key} {li} {w0:.6f} {w1:.6f}\n")


if _LAYER_PROFILE:
    import atexit

    atexit.register(_prof_dump)


# QSA: cache, indexer, attention


def _kv_tail_rows() -> int:
    """Rows in the decode tail of a QSAKVCache (``GMLX_Q4_KV_TAIL``); 0
    appends every step to the one buffer."""
    from gmlx.envflags import env_int
    return max(0, env_int("GMLX_Q4_KV_TAIL", 1024))


def _make_kv_gather_kernel():
    """One-dispatch K and V row gather over two segments: row ``r`` reads
    the base when ``r < meta[0]``, else tail row ``r - meta[0]``. Copies
    only. rows is [B, n] int32; outputs are [B, H, n, D], D a multiple
    of 16."""
    source = """
        const uint gid = thread_position_in_grid.x;
        const int H = kb_shape[1];
        const int D = kb_shape[3];
        const int n = rows_shape[1];
        const int DC = D / 16;
        const int c = gid % DC;
        const int i = (gid / DC) % n;
        const int bh = gid / (uint)(DC * n);
        const int row = rows[(bh / H) * n + i];
        const int bl = meta[0];
        const size_t o = ((size_t)bh * n + i) * D + c * 16;
        if (row < bl) {
            const size_t s = ((size_t)bh * kb_shape[2] + row) * D + c * 16;
            for (int j = 0; j < 16; j++) {
                ko[o + j] = kb[s + j];
                vo[o + j] = vb[s + j];
            }
        } else {
            const size_t s =
                ((size_t)bh * kt_shape[2] + (row - bl)) * D + c * 16;
            for (int j = 0; j < 16; j++) {
                ko[o + j] = kt[s + j];
                vo[o + j] = vt[s + j];
            }
        }
    """
    return mx.fast.metal_kernel(
        name="q4_kv_gather",
        input_names=["kb", "kt", "vb", "vt", "rows", "meta"],
        output_names=["ko", "vo"],
        source=source,
        ensure_row_contiguous=True,
    )


_kv_gather_kernel = None


def _kv_gather():
    """The fused two-segment gather kernel, or None (take and where)."""
    global _kv_gather_kernel
    if _kv_gather_kernel is None:
        if mx.default_device().type == mx.DeviceType.gpu:
            _kv_gather_kernel = _make_kv_gather_kernel()
        else:
            _kv_gather_kernel = False
    return _kv_gather_kernel or None


class QSAKVCache(KVCache):
    """KVCache extended with the QSA indexer key stream.

    ``ik`` holds one raw (pre-norm, pre-rope) indexer key per cached token,
    appended in lockstep with K/V so ``offset`` covers all three streams.
    ``blocks`` caches the finished (mean-pooled, normed, roped) block keys
    derived from ``ik``; it is a pure function of the raw stream, so trim and
    state restore just invalidate it. Quantizing would drop the stream, so
    ``to_quantized`` is refused.

    Each stream is stored as two segments. The base holds rows
    ``[0, offset - _tl)`` and changes only when a prefill chunk lands or the
    tail is folded in. Decode steps append to the tail, a buffer of
    ``GMLX_Q4_KV_TAIL`` rows. Serve dispatches a step before the one
    before it has finished, and that pending step still holds the buffers it
    read, so MLX cannot update them in place and copies them whole. With
    one buffer per stream that copy is the full K and V of every layer on
    every token. ``keys``, ``values`` and ``ik`` fold the tail in and read as
    the whole stream.

    ``blocks`` is split the same way: a block that a decode step finishes
    joins a small tail, so the base, one key per block of the whole context,
    is not copied for it.
    """

    kv_quant_unsupported = True

    _kb = _vb = _ib = None  # base segments
    _kt = _vt = _it = None  # decode tails
    _tl = 0                 # rows live in the tails
    _tail_cap = None
    _bt = None              # finished blocks after the first _nbb
    _nbb = 0                # finished blocks live in ``blocks``

    def __init__(self, ratio: int = 4):
        super().__init__()
        self.ratio = ratio
        self.pos = None      # [3, B, cap] mrope positions (VLM loads only)
        self.blocks = None   # [B, n_blocks, index_dim]
        self.n_blocks = 0

    @property
    def keys(self):
        self._fold()
        return self._kb

    @keys.setter
    def keys(self, v):
        self._kb = None if v is None else mx.contiguous(v)
        self._tl = 0

    @property
    def values(self):
        self._fold()
        return self._vb

    @values.setter
    def values(self, v):
        self._vb = None if v is None else mx.contiguous(v)
        self._tl = 0

    @property
    def ik(self):
        self._fold()
        return self._ib

    @ik.setter
    def ik(self, v):
        self._ib = None if v is None else mx.contiguous(v)
        self._tl = 0

    def _tail_rows(self) -> int:
        if self._tail_cap is None:
            self._tail_cap = _kv_tail_rows()
        return self._tail_cap

    @staticmethod
    def _cap_for(need: int) -> int:
        spare = min(16384, max(256, need // 16))
        return -(-(need + spare) // 256) * 256

    @classmethod
    def _write(cls, base, start: int, rows):
        """``base`` with ``rows`` at ``start`` on the token axis, grown
        when it is short."""
        n = rows.shape[-2]
        if base is None or start + n > base.shape[-2]:
            shape = list(rows.shape)
            shape[-2] = cls._cap_for(start + n) - (start + n)
            parts = [rows, mx.zeros(shape, rows.dtype)]
            if base is not None and start:
                parts.insert(0, base[..., :start, :])
            return mx.concatenate(parts, axis=-2)
        base[..., start:start + n, :] = rows
        return base

    def _fold(self) -> None:
        """Move the tail rows into the base."""
        tl = self._tl
        if not tl:
            return
        bl = self.offset - tl
        # A pending step holds the base and would force a copy of it.
        mx.synchronize()
        self._kb = self._write(self._kb, bl, self._kt[..., :tl, :])
        self._vb = self._write(self._vb, bl, self._vt[..., :tl, :])
        self._ib = self._write(self._ib, bl, self._it[..., :tl, :])
        self._tl = 0

    def append_qsa(self, keys, values, ik, pos=None) -> bool:
        """Append one call's rows to the three streams. True when they went
        to the tail; the caller then reads K and V through ``gather_kv`` or
        ``kv_full``."""
        prev = self.offset
        B, _, n, _ = keys.shape
        cap = self._tail_rows()
        tail = self._kb is not None and n <= min(8, cap)
        if tail:
            if self._kt is None or self._kt.shape[0] != B:
                self._kt = mx.zeros(keys.shape[:2] + (cap, keys.shape[3]),
                                    keys.dtype)
                self._vt = mx.zeros(values.shape[:2] + (cap, values.shape[3]),
                                    values.dtype)
                self._it = mx.zeros((B, cap, ik.shape[2]), ik.dtype)
            if self._tl + n > cap:
                self._fold()
            tl = self._tl
            self._kt[..., tl:tl + n, :] = keys
            self._vt[..., tl:tl + n, :] = values
            self._it[:, tl:tl + n, :] = ik
            self._tl = tl + n
        else:
            self._fold()
            self._kb = self._write(self._kb, prev, keys)
            self._vb = self._write(self._vb, prev, values)
            self._ib = self._write(self._ib, prev, ik)
        self.offset = prev + n
        if pos is not None:
            if self.pos is None or prev + n > self.pos.shape[2]:
                # Rows past prev are stale after a verify rollback.
                head = None if self.pos is None else self.pos[:, :, :prev]
                width = self._cap_for(prev + n) - (0 if head is None else prev)
                newp = mx.zeros((3, B, width), mx.int32)
                self.pos = (newp if head is None else
                            mx.concatenate([head, newp], axis=2))
            self.pos[:, :, prev:prev + n] = pos.astype(mx.int32)
        return tail

    def update_and_fetch_qsa(self, keys, values, ik, pos=None):
        self.append_qsa(keys, values, ik, pos=pos)
        n = self.offset
        return (self.keys[..., :n, :], self.values[..., :n, :],
                self.ik[:, :n, :])

    def kv_full(self):
        """K and V over all ``offset`` rows. The tail is not folded: with
        rows in it the result is a copy."""
        tl = self._tl
        bl = self.offset - tl
        k, v = self._kb[..., :bl, :], self._vb[..., :bl, :]
        if tl:
            k = mx.concatenate([k, self._kt[..., :tl, :]], axis=2)
            v = mx.concatenate([v, self._vt[..., :tl, :]], axis=2)
        return k, v

    def gather_kv(self, rows: mx.array):
        """K and V at the absolute rows ``[B, n]`` int32, each read from
        the segment that holds it: two ``[B, Hkv, n, D]`` arrays."""
        bl = self.offset - self._tl
        kb, vb, kt, vt = self._kb, self._vb, self._kt, self._vt
        B, H, _, D = kb.shape
        n = rows.shape[1]
        kern = _kv_gather()
        if (kern is not None and D % 16 == 0 and vb.shape == kb.shape
                and kt.dtype == kb.dtype and vt.dtype == vb.dtype):
            total = B * H * n * (D // 16)
            ko, vo = kern(
                inputs=[kb, kt, vb, vt, rows,
                        mx.array([bl], dtype=mx.int32)],
                grid=(total, 1, 1),
                threadgroup=(min(256, total), 1, 1),
                output_shapes=[(B, H, n, D), (B, H, n, D)],
                output_dtypes=[kb.dtype, vb.dtype],
            )
            return ko, vo
        in_base = (rows < bl)[:, None, :, None]
        rb = mx.minimum(rows, max(bl - 1, 0))[:, None, :, None]
        rt = mx.clip(rows - bl, 0, kt.shape[2] - 1)[:, None, :, None]
        out = []
        for base, tail in ((kb, kt), (vb, vt)):
            ib = mx.broadcast_to(rb, (B, H, n, 1))
            it = mx.broadcast_to(rt, (B, H, n, 1))
            out.append(mx.where(
                in_base, mx.take_along_axis(base, ib, axis=2),
                mx.take_along_axis(tail, it, axis=2).astype(base.dtype)))
        return out[0], out[1]

    def _ik_rows(self, a: int, b: int) -> mx.array:
        bl = self.offset - self._tl
        if b <= bl:
            return self._ib[:, a:b, :]
        if a >= bl:
            return self._it[:, a - bl:b - bl, :]
        return mx.concatenate(
            [self._ib[:, a:bl, :], self._it[:, :b - bl, :]], axis=1)

    def block_segments(self, n_blocks: int, finish):
        """Block keys as ``(base [B, nb, D], tail [B, n_blocks - nb, D] or
        None)``; ``finish(raw, start_block)`` turns ``[B, m * ratio, D]`` raw
        keys into ``[B, m, D]`` finished ones."""
        r = self.ratio
        if self.blocks is not None and self.blocks.shape[0] != self._ib.shape[0]:
            self.blocks, self._bt, self.n_blocks, self._nbb = None, None, 0, 0
        if n_blocks > self.n_blocks:
            raw = self._ik_rows(self.n_blocks * r, n_blocks * r)
            new = finish(raw, self.n_blocks)
            nb, nt = self._nbb, self.n_blocks - self._nbb
            m = new.shape[1]
            held = [self.blocks[:, :nb]] if nb else []
            if nt:
                held.append(self._bt[:, :nt])
            if nb and m <= 2 and nt + m <= self._tail_rows() // r:
                self._bt = mx.concatenate(held[1:] + [new], axis=1)
            else:
                self.blocks = mx.concatenate(held + [new], axis=1)
                self._bt, self._nbb = None, n_blocks
            self.n_blocks = n_blocks
        nb = min(self._nbb, n_blocks)
        base = self.blocks[:, :nb]
        return base, (self._bt[:, :n_blocks - nb] if n_blocks > nb else None)

    def finished_blocks(self, n_blocks: int, finish):
        """``block_segments`` as one ``[B, n_blocks, D]`` array."""
        base, tail = self.block_segments(n_blocks, finish)
        return base if tail is None else mx.concatenate([base, tail], axis=1)

    @property
    def state(self):
        k, v, ik = self.keys, self.values, self.ik
        n = self.offset
        if n != k.shape[2]:
            k, v = k[..., :n, :], v[..., :n, :]
        base = (k, v, ik[:, :n])
        return base if self.pos is None else base + (self.pos[:, :, :n],)

    @state.setter
    def state(self, v):
        if len(v) == 4:
            self.keys, self.values, self.ik, self.pos = v
        else:
            self.keys, self.values, self.ik = v
            self.pos = None
        self.offset = self._kb.shape[2]
        self.blocks, self._bt, self.n_blocks, self._nbb = None, None, 0, 0

    @classmethod
    def from_state(cls, state, meta_state):
        # mlx-vlm's APC clone prefers this over type(c).__new__, which would
        # skip __init__ and leave ik/pos/blocks attributes unset.
        c = cls()
        c.meta_state = meta_state
        c.state = state
        return c

    @classmethod
    def merge(cls, entries, prefix_lens=None):
        # Two callers share this name. mlx_lm's batch lift calls
        # merge(caches): delegate to the inherited KVCache.merge; the
        # resulting BatchKVCache drops ik, so that request decodes dense
        # (exact math, sparse-attention perf lost). mlx-vlm's APC
        # merge_cache_entries calls merge(entries, prefix_lens) and
        # dispatches on the class's own __dict__, so the arm must live
        # here, not on a parent. Single-request batches only; multi-row
        # joins decline to the cold path rather than guess at row padding
        # for the indexer stream.
        if prefix_lens is None:
            return super().merge(entries)
        if len(entries) != 1:
            return None
        e, p = entries[0], int(prefix_lens[0])
        state = e.state
        if p < e.offset:
            head = (state[0][..., :p, :], state[1][..., :p, :],
                    state[2][:, :p])
            state = head if len(state) == 3 else head + (
                state[3][:, :, :p],)
        return cls.from_state(state, e.meta_state)

    @property
    def meta_state(self):
        return str(self.ratio)

    @meta_state.setter
    def meta_state(self, v):
        if v:
            self.ratio = int(v)

    def trim(self, n):
        n = min(self.offset, n)
        self.offset -= n
        self._tl = max(0, self._tl - n)
        self.n_blocks = min(self.n_blocks, self.offset // self.ratio)
        self._nbb = min(self._nbb, self.n_blocks)
        return n

    def to_quantized(self, group_size: int = 64, bits: int = 4):
        raise NotImplementedError(
            "QSAKVCache cannot quantize: the QSA indexer key stream has no "
            "quantized form (drop --kv-bits for qwen4exp)")

    @property
    def nbytes(self):
        return sum(
            a.nbytes for a in (self._kb, self._vb, self._ib, self._kt,
                               self._vt, self._it, self.blocks, self._bt)
            if a is not None)


class QSAIndexer(nn.Module):
    """Block selector for one attention layer."""

    def __init__(self, args: ModelArgs, ratio: int, rotary_dims: int):
        super().__init__()
        self.n_heads = args.indexer_n_heads
        self.head_dim = args.indexer_head_dim
        self.ratio = ratio
        self.block_topk = args.indexer_budget // ratio
        self.rotary_dims = rotary_dims
        self.rope_theta = args.rope_theta
        self.q_proj = nn.Linear(args.hidden_size, self.n_heads * self.head_dim,
                                bias=False)
        self.k_proj = nn.Linear(args.hidden_size, self.head_dim, bias=False)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self._li = -1

    def _rope(self, x, offset, scale=1.0):
        return mx.fast.rope(x, self.rotary_dims, traditional=False,
                            base=self.rope_theta, scale=scale, offset=offset)

    def _pooled(self, raw: mx.array):
        B, n, D = raw.shape
        m = n // self.ratio
        pooled = raw[:, :m * self.ratio].reshape(B, m, self.ratio, D)
        pooled = pooled.astype(mx.float32).mean(axis=2).astype(raw.dtype)
        return self.k_norm(pooled)

    def finish_blocks(self, raw: mx.array, start_block: int) -> mx.array:
        """Mean-pool ``[B, m * r, D]`` raw keys into ``[B, m, D]`` block keys,
        k_norm, then rope at the block start positions (``scale=r`` makes
        position ``i`` of the call land on token ``(start_block + i) * r``)."""
        return self._rope(self._pooled(raw)[:, None], start_block,
                          scale=float(self.ratio))[:, 0]

    def finish_blocks_at(self, raw: mx.array, start_block: int,
                         pos_all: mx.array, selector: mx.array) -> mx.array:
        """``finish_blocks`` with explicit mrope positions: block ``i`` is
        roped at the cached position of its start token
        (``pos_all[:, :, (start_block + i) * r]``), matching the reference's
        ``full_cos.index_select(group_starts)``."""
        pooled = self._pooled(raw)
        m = pooled.shape[1]
        starts = (start_block + mx.arange(m, dtype=mx.int32)) * self.ratio
        pos = mx.take(pos_all, starts.astype(mx.uint32), axis=2)  # [3, B, m]
        cos, sin = _mrope_cos_sin(pos, self.rotary_dims, self.rope_theta,
                                  selector)
        return _apply_rope_cos_sin(pooled[:, None], cos, sin,
                                   self.rotary_dims)[:, 0]

    def _queries(self, x: mx.array, offset: int, cos, sin) -> mx.array:
        """Normed, roped indexer queries ``[B, H, L, D]``."""
        B, L, _ = x.shape
        q = self.q_proj(x).reshape(B, L, self.n_heads, self.head_dim)
        q = self.q_norm(q).transpose(0, 2, 1, 3)
        if cos is not None:
            return _apply_rope_cos_sin(q, cos, sin, self.rotary_dims)
        return self._rope(q, offset)

    def scores(self, x: mx.array, blocks: mx.array, offset: int,
               cos=None, sin=None) -> mx.array:
        """``[B, L, n_blocks]`` f32 block scores for the queries in ``x``."""
        q = self._queries(x, offset, cos, sin)
        s = q.astype(mx.float32) @ blocks.astype(mx.float32)[:, None].transpose(0, 1, 3, 2)
        s = mx.maximum(s, 0).sum(axis=1)
        return s * (1.0 / math.sqrt(self.head_dim))

    def select(self, x: mx.array, ik_all: mx.array, cache, offset: int,
               cos=None, sin=None, pos_all=None, selector=None,
               key_len=None):
        """Top-k complete blocks per query, or None when every query is
        still dense. Returns ``(blocks [B, L, topk], complete_counts [L])``."""
        B, L, _ = x.shape
        if key_len is None:
            key_len = ik_all.shape[1]
        r = self.ratio
        n_blocks = key_len // r
        if n_blocks <= self.block_topk:
            return None
        if pos_all is not None:
            def finish(raw, start_block):
                return self.finish_blocks_at(raw, start_block, pos_all,
                                             selector)
        else:
            finish = self.finish_blocks
        btail = None
        if cache is not None:
            blocks, btail = cache.block_segments(n_blocks, finish)
        else:
            blocks = finish(ik_all[:, :n_blocks * r], 0)
        prof = _subprof_on(L)
        if prof:
            import time
            t0 = _prof_mark("a.blocks", self._li,
                            [blocks] if btail is None else [blocks, btail],
                            time.perf_counter())
        query_ends = offset + mx.arange(L) + 1
        complete = query_ends // r
        k = self.block_topk
        topk = _kq_topk()
        kq_ok = (topk is not None and k == 512 and n_blocks >= k
                 and x.dtype in (mx.float16, mx.bfloat16))
        if kq_ok and L <= 4 and _kq_score() is not None:
            # Fused decode/verify path: one kernel scores every pooled block
            # (relu dots summed over the 4 heads, per-row visibility baked as
            # finite_min) and the radix top-k consumes its 16-bit rows
            # directly. Replaces the astype/matmul/relu/sum/where chain.
            q = self._queries(x, offset, cos, sin).astype(x.dtype)
            w = mx.full((B, L, self.n_heads),
                        1.0 / math.sqrt(self.head_dim), dtype=x.dtype)
            s16 = _kq_score()(q, blocks.astype(x.dtype), w, offset, self.ratio)
            if btail is not None:
                # The scorer numbers blocks from 0, so the tail's call moves
                # the query back by the base's tokens to keep visibility.
                s16 = mx.concatenate(
                    [s16, _kq_score()(q, btail.astype(x.dtype), w,
                                      offset - blocks.shape[1] * r, r)],
                    axis=-1)
            if prof:
                t0 = _prof_mark("a.score", self._li, s16, t0)
            sel = mx.stop_gradient(topk(s16, k, True)[:, 0].astype(mx.int64))
            if prof:
                _prof_mark("a.topk", self._li, sel, t0)
            return sel, complete
        if btail is not None:
            blocks = mx.concatenate([blocks, btail], axis=1)
        s = self.scores(x, blocks, offset, cos=cos, sin=sin)
        valid = mx.arange(n_blocks)[None, None, :] < complete[None, :, None]
        s = mx.where(valid, s, -mx.inf)
        if kq_ok:
            # kq radix top-k (one threadgroup per row) replaces the
            # sort-based argpartition. Selection is set-equivalent up to
            # ties at the threshold, and the mask/gather consumers are
            # order-insensitive. Scores narrow to the activation dtype for
            # the kernel's 16-bit wire.
            sel = topk(s.astype(x.dtype)[:, None], k, True)[:, 0]
            sel = mx.stop_gradient(sel.astype(mx.int64))
        else:
            sel = mx.argpartition(s, kth=-k, axis=-1)[..., -k:]
        return sel, complete


def _qsa_token_mask(sel, complete, offset, L, key_len, ratio, topk):
    """Boolean ``[B, 1, L, key_len]`` attention mask from the block selection:
    selected block members plus the incomplete tail, or plain causal for
    queries with at most ``topk`` complete blocks."""
    B = sel.shape[0]
    n_blocks = key_len // ratio
    blk = mx.zeros((B, L, n_blocks + 1), dtype=mx.bool_)
    blk = mx.put_along_axis(blk, sel, mx.array(True), axis=-1)[..., :n_blocks]
    tok_sel = mx.repeat(blk, ratio, axis=-1)
    pad = key_len - n_blocks * ratio
    if pad:
        tok_sel = mx.concatenate(
            [tok_sel, mx.zeros((B, L, pad), dtype=mx.bool_)], axis=-1)
    tok = mx.arange(key_len)[None, :]
    query_ends = (offset + mx.arange(L) + 1)[:, None]
    tail_start = (complete * ratio)[:, None]
    causal = tok < query_ends
    tail = (tok >= tail_start) & causal
    use_sparse = (complete > topk)[:, None]
    m = mx.where(use_sparse[None], tok_sel | tail[None], causal[None])
    return m[:, None]


class Attention(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.num_attention_heads = args.num_attention_heads
        self.num_key_value_heads = args.num_key_value_heads
        self.head_dim = args.head_dim
        self.scale = self.head_dim ** -0.5
        self.rotary_dims = int(self.head_dim * args.partial_rotary_factor)
        self.rope_theta = args.rope_theta
        H, Hkv, D = self.num_attention_heads, self.num_key_value_heads, self.head_dim
        self.q_proj = nn.Linear(args.hidden_size, H * D * 2, bias=False)
        self.k_proj = nn.Linear(args.hidden_size, Hkv * D, bias=False)
        self.v_proj = nn.Linear(args.hidden_size, Hkv * D, bias=False)
        self.o_proj = nn.Linear(H * D, args.hidden_size, bias=False)
        self.q_norm = nn.RMSNorm(D, eps=args.rms_norm_eps)
        self.k_norm = nn.RMSNorm(D, eps=args.rms_norm_eps)
        self.ratio = int(args.compress_ratios[layer_idx])
        self._mrope_selector = _mrope_selector(
            list(args.mrope_section), self.rotary_dims // 2)
        if self.ratio > 0:
            self.indexer = QSAIndexer(args, self.ratio, self.rotary_dims)
            self.indexer._li = layer_idx
        self._li = layer_idx

    def _rope(self, x, offset):
        return mx.fast.rope(x, self.rotary_dims, traditional=False,
                            base=self.rope_theta, scale=1.0, offset=offset)

    def _gathered_attention(self, q, k, v, sel, complete, offset, L,
                            gather=None):
        """Decode / verify: gather the selected keys per query and run one
        ``B * L``-batched qL=1 sdpa over ``topk * r + r`` keys. ``gather``
        reads the rows from a segmented cache in place of ``k`` and ``v``."""
        B, H, _, D = q.shape
        Hkv = self.num_key_value_heads
        r, topk = self.ratio, self.indexer.block_topk
        members = (sel[..., None] * r + mx.arange(r)).reshape(B, L, topk * r)
        tail_start = (complete * r)[None, :, None]
        tail = mx.broadcast_to(
            tail_start + mx.arange(r)[None, None, :], (B, L, r))
        query_ends = (offset + mx.arange(L) + 1)[None, :, None]
        tail_ok = tail < query_ends
        idx = mx.concatenate([members, mx.minimum(tail, query_ends - 1)], axis=-1)
        W = idx.shape[-1]
        bias = mx.concatenate(
            [mx.ones((B, L, topk * r), dtype=mx.bool_), tail_ok], axis=-1)
        if gather is not None:
            k_sel, v_sel = gather(idx.reshape(B, L * W).astype(mx.int32))
        else:
            flat = mx.broadcast_to(
                idx.reshape(B, 1, L * W, 1), (B, Hkv, L * W, 1))
            k_sel = mx.take_along_axis(k, flat, axis=2)
            v_sel = mx.take_along_axis(v, flat, axis=2)
        k_sel = k_sel.reshape(B, Hkv, L, W, D)
        v_sel = v_sel.reshape(B, Hkv, L, W, D)
        k_sel = k_sel.transpose(0, 2, 1, 3, 4).reshape(B * L, Hkv, W, D)
        v_sel = v_sel.transpose(0, 2, 1, 3, 4).reshape(B * L, Hkv, W, D)
        q_l = q.transpose(0, 2, 1, 3).reshape(B * L, H, 1, D)
        out = mx.fast.scaled_dot_product_attention(
            q_l, k_sel, v_sel, scale=self.scale,
            mask=bias.reshape(B * L, 1, 1, W))
        return out.reshape(B, L, H, D).transpose(0, 2, 1, 3)

    def _block_sparse_prefill(self, q, k, v, sel, offset, key_len, bs):
        """Prefill chunk with every query sparse: fold queries into 4-wide
        windows and walk each window's own page list (union of its queries'
        selected blocks plus the window's tail span) through the kq
        block-sparse FA kernel. One host sync reads the widest union;
        prefill graphs rebuild per chunk anyway."""
        L = q.shape[2]
        r = self.ratio
        n_qt = L // r
        nb_total = (key_len + r - 1) // r
        topk = self.indexer.block_topk
        g = sel[0].astype(mx.int32).reshape(n_qt, r * topk)
        w4 = offset + mx.arange(n_qt, dtype=mx.int32) * r
        span = mx.stack(
            [w4 // r, mx.minimum((w4 + r) // r, nb_total - 1)], axis=1)
        srt = mx.sort(mx.concatenate([g, span], axis=1), axis=1)
        newv = mx.concatenate(
            [mx.ones((n_qt, 1), dtype=mx.bool_), srt[:, 1:] != srt[:, :-1]],
            axis=1)
        counts = newv.sum(axis=1).astype(mx.int32)
        max_p = int(counts.max())  # host sync
        slot = mx.cumsum(newv.astype(mx.int32), axis=1) - 1
        pages = mx.put_along_axis(
            mx.full((n_qt, max_p), -1, dtype=mx.int32), slot, srt, axis=1)
        lut = mx.put_along_axis(
            mx.zeros((n_qt, nb_total), dtype=mx.int32), srt, slot, axis=1)
        pm = mx.zeros((n_qt, max_p), dtype=mx.uint16)
        selq = sel[0].astype(mx.int32).reshape(n_qt, r, topk)
        for qi in range(r):
            slots_q = mx.take_along_axis(lut, selq[:, qi], axis=1)
            pm = pm | mx.put_along_axis(
                mx.zeros((n_qt, max_p), dtype=mx.uint16), slots_q,
                mx.array(1 << qi, dtype=mx.uint16), axis=1)
        return bs(q, k, v, self.scale, pages, pm, counts, offset)

    def _split_regime_prefill(self, q, k, v, sel, complete, offset, L,
                              key_len, bs):
        """Mixed prefill window spanning the causal/sparse boundary
        (a query is sparse once its complete-block count exceeds
        block_topk): split the query axis exactly at the boundary and give
        each span its native path -- flash-causal SDPA for the causal
        prefix, the L<=8 gathered path for the (at most ratio-1) ragged
        queries up to the next 4-aligned window, and the block-sparse FA
        kernel for the aligned sparse tail. Replaces the dense-masked SDPA
        fallback that materialized an [L, key_len] token mask for the
        whole window."""
        r, topk = self.ratio, self.indexer.block_topk
        c = min(L, max(0, r * (topk + 1) - 1 - offset))
        s = min(L, -(-c // r) * r)
        outs = []
        if c > 0:
            outs.append(mx.fast.scaled_dot_product_attention(
                q[..., :c, :], k[..., :offset + c, :],
                v[..., :offset + c, :], scale=self.scale, mask="causal"))
        if s > c:
            outs.append(self._gathered_attention(
                q[..., c:s, :], k, v, sel[:, c:s], complete[c:s],
                offset + c, s - c))
        if L > s:
            outs.append(self._block_sparse_prefill(
                q[..., s:, :], k, v, sel[:, s:], offset + s, key_len, bs))
        return mx.concatenate(outs, axis=2) if len(outs) > 1 else outs[0]

    def _segmented_attention(self, q, cache, sel, complete, offset, L,
                             key_len, paged):
        """Decode / verify over a cache with rows in its tail: gather the
        selected rows from both segments in one dispatch, then attend over
        the compact copy. At L=1 the paged kernel walks it as consecutive
        pages, the same rows in the same order as ``_paged_attention``."""
        if paged is None or L != 1:
            return self._gathered_attention(
                q, None, None, sel, complete, offset, L,
                gather=cache.gather_kv)
        B = q.shape[0]
        r = self.ratio
        rows = (sel[:, 0, :, None] * r + mx.arange(r)).reshape(B, -1)
        rows = rows.astype(mx.int32)
        rem = key_len % r
        if rem:
            tail = mx.arange(key_len - rem, key_len, dtype=mx.int32)
            rows = mx.concatenate(
                [rows, mx.broadcast_to(tail[None], (B, rem))], axis=-1)
        k_sel, v_sel = cache.gather_kv(rows)
        n_pages = -(-rows.shape[1] // r)
        pages = mx.broadcast_to(
            mx.arange(n_pages, dtype=mx.int32)[None, None],
            (B, self.num_key_value_heads, n_pages))
        return paged(q, k_sel, v_sel, self.scale, pages, tile_c=4)

    def _paged_attention(self, q, k, v, sel, key_len, paged):
        """Decode (L=1): kq page-gather sdpa straight over the KV cache
        with the selected 4-row blocks as pages -- no per-token K/V copy.
        The partial tail block is appended when present; a full tail
        contributes nothing in the gathered path and is omitted here."""
        B = q.shape[0]
        Hkv = k.shape[1]
        pages = sel[:, 0, :].astype(mx.int32)
        if key_len % self.ratio:
            tail = mx.full((B, 1), key_len // self.ratio, dtype=mx.int32)
            pages = mx.concatenate([pages, tail], axis=-1)
        pages = mx.broadcast_to(pages[:, None, :], (B, Hkv, pages.shape[-1]))
        return paged(q, k, v, self.scale, pages, tile_c=4)

    def __call__(self, x: mx.array, mask=None, cache=None,
                 positions=None) -> mx.array:
        """``positions`` is the ``[3, B, L]`` mrope position block for this
        window (VLM loads); ``None`` takes the scalar-offset text path."""
        B, L, _ = x.shape
        H, Hkv, D = self.num_attention_heads, self.num_key_value_heads, self.head_dim
        qg = self.q_proj(x).reshape(B, L, H, 2 * D)
        q, gate = mx.split(qg, 2, axis=-1)
        gate = gate.reshape(B, L, -1)
        q = self.q_norm(q).transpose(0, 2, 1, 3)
        k = self.k_norm(self.k_proj(x).reshape(B, L, Hkv, D)).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, L, Hkv, D).transpose(0, 2, 1, 3)

        offset = cache.offset if cache is not None else 0
        cos = sin = None
        if positions is not None:
            cos, sin = _mrope_cos_sin(positions, self.rotary_dims,
                                      self.rope_theta, self._mrope_selector)
            q = _apply_rope_cos_sin(q, cos, sin, self.rotary_dims)
            k = _apply_rope_cos_sin(k, cos, sin, self.rotary_dims)
        else:
            q = self._rope(q, offset)
            k = self._rope(k, offset)

        prof = _subprof_on(L)
        if prof:
            import time
            t0 = _prof_mark("a.proj", self._li, (q, k, v, gate),
                            time.perf_counter())
        selection = None
        seg = False
        if "indexer" in self:
            ik = self.indexer.k_proj(x)
            if cache is not None and hasattr(cache, "update_and_fetch_qsa"):
                seg = cache.append_qsa(k, v, ik, pos=positions)
                if prof:
                    t0 = _prof_mark(
                        "a.cache", self._li,
                        (cache._kt, cache._vt, cache._it) if seg else
                        (cache._kb, cache._vb, cache._ib), t0)
                selection = self.indexer.select(
                    x, None, cache, offset, cos=cos, sin=sin,
                    pos_all=cache.pos, selector=self._mrope_selector,
                    key_len=cache.offset)
                if not seg:
                    k, v = cache.kv_full()
            elif cache is not None:
                k, v = cache.update_and_fetch(k, v)
            else:
                selection = self.indexer.select(
                    x, ik, None, 0, cos=cos, sin=sin,
                    pos_all=(positions.astype(mx.int32)
                             if positions is not None else None),
                    selector=self._mrope_selector)
        elif cache is not None:
            k, v = cache.update_and_fetch(k, v)

        if selection is None:
            if seg:
                k, v = cache.kv_full()
            out = scaled_dot_product_attention(
                q, k, v, cache=cache, scale=self.scale, mask=mask)
        else:
            sel, complete = selection
            key_len = cache.offset if seg else k.shape[2]
            all_sparse = (offset + 1) // self.ratio > self.indexer.block_topk
            # the kq sparse kernels have no backward: a training forward
            # takes the token-mask branch
            kern = not (self.training and cache is None)
            paged = _kq_paged() if (
                kern and L == 1 and self.ratio == 4 and D == 256
                and q.dtype in (mx.bfloat16, mx.float16)) else None
            seg_ok = (seg and kern and L <= 8 and all_sparse
                      and not isinstance(mask, mx.array))
            if seg and not seg_ok:
                k, v = cache.kv_full()
            if seg_ok:
                out = self._segmented_attention(
                    q, cache, sel, complete, offset, L, key_len, paged)
            elif paged is not None and all_sparse and not isinstance(mask, mx.array):
                out = self._paged_attention(q, k, v, sel, key_len, paged)
            elif kern and L <= 8 and all_sparse and not isinstance(mask, mx.array):
                out = self._gathered_attention(q, k, v, sel, complete, offset, L)
            elif (kern and B == 1 and all_sparse and not isinstance(mask, mx.array)
                  and L % 4 == 0 and self.ratio == 4 and D == 256
                  and H == 12 * Hkv and q.dtype in (mx.bfloat16, mx.float16)
                  and (bs := _kq_bs_prefill()) is not None):
                out = self._block_sparse_prefill(q, k, v, sel, offset,
                                                 key_len, bs)
            elif (kern and B == 1 and not all_sparse and L > 8
                  and not isinstance(mask, mx.array)
                  and offset % 4 == 0 and L % 4 == 0
                  and self.ratio == 4 and D == 256 and H == 12 * Hkv
                  and q.dtype in (mx.bfloat16, mx.float16)
                  and (bs := _kq_bs_prefill()) is not None):
                out = self._split_regime_prefill(q, k, v, sel, complete,
                                                 offset, L, key_len, bs)
            elif (kern and B == 1 and L > 8 and L % 4 != 0
                  and not isinstance(mask, mx.array)
                  and (all_sparse or offset % 4 == 0)
                  and self.ratio == 4 and D == 256 and H == 12 * Hkv
                  and q.dtype in (mx.bfloat16, mx.float16)
                  and (offset + (L - L % 4) + 1) // self.ratio
                  > self.indexer.block_topk
                  and (bs := _kq_bs_prefill()) is not None):
                # Ragged window (serve one-shots whole prompts, so L rarely
                # lands on a multiple of 4): 4-aligned head through the
                # kernel paths, the <=3 tail queries through the gathered
                # path. Without this the whole chunk pays the dense token
                # mask.
                Lh = L - L % 4
                if all_sparse:
                    head = self._block_sparse_prefill(
                        q[..., :Lh, :], k, v, sel[:, :Lh], offset,
                        key_len, bs)
                else:
                    head = self._split_regime_prefill(
                        q[..., :Lh, :], k, v, sel[:, :Lh], complete[:Lh],
                        offset, Lh, key_len, bs)
                tail = self._gathered_attention(
                    q[..., Lh:, :], k, v, sel[:, Lh:], complete[Lh:],
                    offset + Lh, L - Lh)
                out = mx.concatenate([head, tail], axis=2)
            else:
                qsa = _qsa_token_mask(sel, complete, offset, L, key_len,
                                      self.ratio, self.indexer.block_topk)
                if isinstance(mask, mx.array):
                    if mask.dtype == mx.bool_:
                        qsa = qsa & mask
                    else:
                        qsa = mask + mx.where(qsa, 0.0, -mx.inf).astype(mask.dtype)
                if (self.training and cache is None
                        and env_bool("GMLX_TRAIN_BLOCKED_ATTN", True)):
                    # MLX's unfused backward keeps every layer's
                    # [B, H, L, L] softmax on the tape
                    out = blocked_attention(q, k, v, scale=self.scale,
                                            mask=qsa)
                else:
                    out = mx.fast.scaled_dot_product_attention(
                        q, k, v, scale=self.scale, mask=qsa)
        if prof:
            t0 = _prof_mark("a.core", self._li, out, time.perf_counter())
        out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)
        out = self.o_proj(out * mx.sigmoid(gate))
        if prof:
            _prof_mark("a.out", self._li, out, t0)
        return out


# MoE (qwen3-next shape: router + SwitchGLU + sigmoid-gated shared expert)


class MLP(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)

    def __call__(self, x) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class SparseMoeBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        dim = args.hidden_size
        self.norm_topk_prob = args.norm_topk_prob
        self.num_experts = args.num_experts
        self.top_k = args.num_experts_per_tok
        self.gate = nn.Linear(dim, self.num_experts, bias=False)
        self.switch_mlp = SwitchGLU(dim, args.moe_intermediate_size, self.num_experts)
        self.shared_expert = MLP(dim, args.shared_expert_intermediate_size)
        self.shared_expert_gate = nn.Linear(dim, 1, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        gates = self.gate(x)
        gates = mx.softmax(gates, axis=-1, precise=True)
        k = self.top_k
        inds = mx.stop_gradient(mx.argpartition(gates, kth=-k, axis=-1)[..., -k:])
        scores = mx.take_along_axis(gates, inds, axis=-1)
        if self.norm_topk_prob:
            scores = scores / scores.sum(axis=-1, keepdims=True)
        # Routing math runs fp32 (kept router weights); the weighted sum
        # returns to the activation dtype so the fp32 scores do not
        # promote the residual stream for every downstream layer.
        scores = scores.astype(x.dtype)
        y = self.switch_mlp(x, inds)
        y = (y * scores[..., None]).sum(axis=-2)
        shared_y = self.shared_expert(x)
        shared_y = mx.sigmoid(self.shared_expert_gate(x)) * shared_y
        return y + shared_y


# PLE n-gram hash embeddings


def _make_ple_hash_kernel():
    """One-dispatch n-gram row hash: thread (b, t, head) recomputes the
    eager chain (shift-with-EOS-cut, 64-bit mix, mod + offset) for its
    head. Integer ALU only. hist is [B, T + CTX] int64; out [B, T, NH]
    int32 (table rows stay below 2^31)."""
    source = """
        const uint gid = thread_position_in_grid.x;
        const int T = hist_shape[1] - CTX;
        const int nh = gid % NH;
        const int t = (gid / NH) % T;
        const int b = gid / (uint)(NH * T);
        const size_t hbase = (size_t)b * (T + CTX);
        const int p = CTX + t;
        const int n = 2 + nh / HPN;
        ulong mixed = (ulong)hist[hbase + p] * mults[0];
        bool eos_seen = false;
        for (int sh = 1; sh < n; sh++) {
            const long tok = hist[hbase + p - sh];
            eos_seen = eos_seen || (tok == EOS);
            mixed ^= (eos_seen ? (ulong)EOS : (ulong)tok) * mults[sh];
        }
        out[gid] = (int)(mixed % sizes[nh] + offsets[nh]);
    """
    return mx.fast.metal_kernel(
        name="ple_hash_rows",
        input_names=["hist", "mults", "sizes", "offsets"],
        output_names=["out"],
        source=source,
        ensure_row_contiguous=True,
    )


_ple_hash_kernel = None


def _ple_hash():
    """The fused PLE row-hash kernel, or None (eager op chain).
    ``GMLX_Q4_PLE_FUSED_HASH=0`` disables."""
    global _ple_hash_kernel
    if _ple_hash_kernel is None:
        from gmlx.envflags import env_bool
        if (env_bool("GMLX_Q4_PLE_FUSED_HASH", True)
                and mx.default_device().type == mx.DeviceType.gpu):
            _ple_hash_kernel = _make_ple_hash_kernel()
        else:
            _ple_hash_kernel = False
    return _ple_hash_kernel or None


class PLEEmbedding(nn.Module):
    """Hash the token history into per-head row ids and gather the rows.

    Head ``h`` of n-gram order ``n`` (2 <= n <= ngram_size) hashes
    ``ctx[0] * m[0] ^ ... ^ ctx[n-1] * m[n-1]`` (uint64 wraparound) into
    ``mixed % vocab[h] + offset[h]``. ``ctx[s]`` is the token ``s`` positions
    back, or EOS when an EOS sits anywhere in between (the token's own EOS
    does not cut its context). The multipliers are 45-bit and token ids
    18-bit, so the products never reach the sign bit and signed (HF) and
    unsigned (llama.cpp) arithmetic agree.
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.ngram_size = args.ple_ngram_size
        self.heads_per_ngram = args.ple_heads_per_ngram
        self.context_len = self.ngram_size - 1
        self.n_heads = self.context_len * self.heads_per_ngram
        self.eos_token_id = args.ple_eos_token_id
        self.embed_dim = args.ple_embed_dim
        self._mults = [int(m) for m in args.ple_layer_multipliers]
        self._mults_u64 = mx.array(self._mults, dtype=mx.uint64)
        self._sizes = mx.array([int(s) for s in args.ple_head_vocab_sizes],
                               dtype=mx.uint64)
        self._offsets = mx.array([int(o) for o in args.ple_head_offsets],
                                 dtype=mx.uint64)

    def _shift_right_ignore_eos(self, tokens: mx.array, shift: int) -> mx.array:
        if shift == 0:
            return tokens
        B, T = tokens.shape
        positions = mx.arange(T, dtype=mx.int64)
        eos_pos = mx.where(tokens == self.eos_token_id, positions[None], -1)
        prev_eos_incl = mx.cummax(eos_pos, axis=1)
        prev_eos = mx.concatenate(
            [mx.full((B, 1), -1, dtype=mx.int64), prev_eos_incl[:, :-1]], axis=1)
        src = positions - shift
        gathered = mx.take_along_axis(
            tokens, mx.broadcast_to(mx.maximum(src, 0)[None], (B, T)), axis=1)
        valid = (src[None] > prev_eos) & (src[None] >= 0)
        return mx.where(valid, gathered, self.eos_token_id)

    def prev_history(self, cache, B: int) -> mx.array:
        """The ``[B, context_len]`` token history before this call (EOS
        filled when the cache is fresh or was built for another batch)."""
        if cache is not None and cache[3] is not None and cache[3].shape[0] == B:
            return cache[3]
        return mx.full((B, self.context_len), self.eos_token_id, dtype=mx.int64)

    def row_ids(self, input_ids: mx.array, cache, prev=None) -> mx.array:
        """``[B, T, n_heads]`` table rows for the tokens in ``input_ids``."""
        ids = input_ids.astype(mx.int64)
        B, T = ids.shape
        if prev is None:
            prev = self.prev_history(cache, B)
        hist = mx.concatenate([prev, ids], axis=1)
        if cache is not None:
            cache[3] = mx.contiguous(hist[:, -self.context_len:])
        kern = _ple_hash() if self.ngram_size <= 3 else None
        if kern is not None:
            nh = self.n_heads
            return kern(
                inputs=[hist, self._mults_u64, self._sizes, self._offsets],
                template=[("CTX", self.context_len),
                          ("HPN", self.heads_per_ngram),
                          ("NH", nh),
                          ("EOS", self.eos_token_id)],
                grid=(B * T * nh, 1, 1),
                threadgroup=(min(256, B * T * nh), 1, 1),
                output_shapes=[(B, T, nh)],
                output_dtypes=[mx.int32],
            )[0]
        shifted = [
            self._shift_right_ignore_eos(hist, s).astype(mx.uint64)
            for s in range(self.ngram_size)
        ]
        mults = [mx.array(m, dtype=mx.uint64) for m in self._mults]
        blocks = []
        for n in range(2, self.ngram_size + 1):
            start = (n - 2) * self.heads_per_ngram
            end = start + self.heads_per_ngram
            mixed = shifted[0] * mults[0]
            for p in range(1, n):
                mixed = mx.bitwise_xor(mixed, shifted[p] * mults[p])
            rows = mixed[..., None] % self._sizes[start:end] + self._offsets[start:end]
            blocks.append(rows)
        rows = mx.concatenate(blocks, axis=-1)[:, -T:]
        return rows.astype(mx.int32)

    def __call__(self, input_ids: mx.array, cache, table, prev=None) -> mx.array:
        """``table`` is the model-level row table (``model.ple_embed``); it is
        passed in rather than owned so the 320M-row weight has one parameter
        path independent of which layer hosts the PLE block."""
        rows = self.row_ids(input_ids, cache, prev=prev)
        emb = table(rows)
        return emb.reshape(*emb.shape[:-2], -1)


class PLELayer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.hidden_size = args.hidden_size
        self.hc = args.hc_count
        hc_dim = self.hidden_size * self.hc
        eps = args.rms_norm_eps
        self.embedding = PLEEmbedding(args)
        emb_dim = self.embedding.n_heads * args.ple_embed_dim
        self.key_proj = nn.Linear(emb_dim, hc_dim, bias=False)
        self.value_proj = nn.Linear(emb_dim, self.hidden_size, bias=False)
        self.norm_key = GroupedRMSNorm(hc_dim, eps)
        self.norm_query = GroupedRMSNorm(hc_dim, eps)
        self.norm_conv = GroupedRMSNorm(hc_dim, eps)
        self.conv_dilation = args.ple_ngram_size
        self.conv_kernel = args.ple_conv_kernel
        self.conv_state_len = (self.conv_kernel - 1) * self.conv_dilation
        self.conv1d = nn.Conv1d(
            hc_dim, hc_dim, kernel_size=self.conv_kernel,
            dilation=self.conv_dilation, groups=hc_dim, bias=False)

    def _conv(self, x: mx.array, cache) -> mx.array:
        B = x.shape[0]
        state = None
        if cache is not None and cache[2] is not None:
            state = cache[2]
            if state.shape[0] != B:
                state = None
        if state is None:
            state = mx.zeros((B, self.conv_state_len, x.shape[-1]), dtype=x.dtype)
        conv_input = mx.concatenate([state, x], axis=1)
        if cache is not None:
            cache[2] = mx.contiguous(conv_input[:, -self.conv_state_len:])
        return nn.silu(self.conv1d(conv_input)), conv_input

    def __call__(self, h: mx.array, input_ids: mx.array, cache, table,
                 mask=None, gdn_sink=None) -> mx.array:
        B, T, hc, D = h.shape
        prev = self.embedding.prev_history(cache, B)
        emb = self.embedding(input_ids, cache, table, prev=prev)
        keys = self.norm_key(self.key_proj(emb).reshape(B, T, hc, D))
        values = self.value_proj(emb)
        queries = self.norm_query(h)
        gate = (keys.astype(mx.float32) * queries.astype(mx.float32)).sum(
            axis=-1, keepdims=True) * (1.0 / math.sqrt(D))
        gate = mx.sign(gate) * mx.sqrt(mx.maximum(mx.abs(gate), 1e-6))
        gate = mx.sigmoid(gate).astype(h.dtype)
        gated = gate * values[:, :, None, :]
        normed = self.norm_conv(gated).reshape(B, T, hc * D)
        gated = gated.reshape(B, T, hc * D)
        if isinstance(mask, mx.array) and mask.ndim == 2 and mask.shape[0] == B:
            gated = mx.where(mask[..., None], gated, 0)
            normed = mx.where(mask[..., None], normed, 0)
        conv, conv_input = self._conv(normed, cache)
        if gdn_sink is not None:
            gdn_sink.append({
                "kind": "ple", "cache": cache, "prev": prev, "ids": input_ids,
                "ctx": self.embedding.context_len, "conv_input": conv_input,
                "L": self.conv_state_len,
            })
        return h + (gated + conv).reshape(B, T, hc, D)


def prepare_runtime(model) -> dict:
    """Arm the owned kernel routes after the weights are loaded: the fused
    S=1 GDN decode kernel (``GMLX_FUSED_GDN=0`` disables) and the
    concatenated b/a decay matvec it consumes. Returns counts for the load
    log."""
    import gmlx.upstream.gdn_patches as _gp
    from gmlx.envflags import env_bool

    counts = {"gdn_fused": 0, "gdn_ba_cat": 0, "gdn_fused_verify": 0}
    want = env_bool("GMLX_FUSED_GDN", True)
    fused = want and _gp._gdn_fused_decode_kernel is not None
    fused_verify = want and _gp._gdn_fused_verify_kernel is not None
    cat_ba = env_bool("GMLX_GDN_BA_CAT", True)
    for m in model.modules():
        if isinstance(m, GatedDeltaNet):
            m._fused_decode = fused
            m._fused_verify = fused_verify
            counts["gdn_fused"] += int(fused)
            counts["gdn_fused_verify"] += int(fused_verify)
            if fused and cat_ba and _gp._gdn_try_cat_ba(m):
                counts["gdn_ba_cat"] += 1
    return counts


def rollback_verify_sink(sink, n: int) -> None:
    """Rewind the recurrent caches after an MTP verify forward over ``S``
    positions to the state after its first ``n`` (the accepted prefix).

    GDN: the conv state is a window of the recorded conv input and the scan
    state is the fused verify kernel's per-position intermediate; without
    intermediates (unfused path) the layer is re-run over the prefix from
    its pre-verify state. PLE: both the token history and the conv state are
    windows of recorded arrays. KV caches are trimmed by the caller.
    """
    for e in sink:
        cache = e["cache"]
        if cache is None:
            continue
        if e["kind"] == "gdn":
            K = e["K"]
            if e["inter"] is not None:
                cache[0] = mx.contiguous(e["conv_input"][:, n:n + K - 1, :])
                cache[1] = mx.contiguous(e["inter"][:, n - 1])
                continue
            cache[0], cache[1] = e["pre"]
            mask = e["mask"]
            if isinstance(mask, mx.array):
                mask = mask[:, :n]
            e["layer"](e["inputs"][:, :n], mask=mask, cache=cache)
        elif e["kind"] == "ple":
            hist = mx.concatenate([e["prev"], e["ids"][:, :n].astype(mx.int64)],
                                  axis=1)
            cache[3] = mx.contiguous(hist[:, -e["ctx"]:])
            cache[2] = mx.contiguous(e["conv_input"][:, n:n + e["L"], :])


# Layers and model


class DecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.is_linear = args.layer_types[layer_idx] == "linear_attention"
        eps = args.rms_norm_eps
        self.hc_attn = HyperConnection(args.hidden_size, args.hc_count,
                                       args.hc_lowrank, eps)
        self.hc_ffn = HyperConnection(args.hidden_size, args.hc_count,
                                      args.hc_lowrank, eps)
        if self.is_linear:
            self.linear_attn = GatedDeltaNet(args)
        else:
            self.self_attn = Attention(args, layer_idx)
        self.mlp = SparseMoeBlock(args)
        if layer_idx in args.ple_layer_ids:
            self.ple = PLELayer(args)

    def __call__(self, h: mx.array, input_ids: mx.array, mask=None, cache=None,
                 ple_table=None, gdn_sink=None, positions=None):
        prof = _prof_on(h.shape[1])
        li = self.layer_idx
        if prof:
            import time
            if li == 0:
                _PROF_CALLS[0] += 1
            mx.eval(h)
            t0 = time.perf_counter()
        if "ple" in self:
            h = self.ple(h, input_ids, cache, ple_table, mask, gdn_sink=gdn_sink)
            if prof:
                t0 = _prof_mark("ple", li, h, t0)
        mixed, inject = self.hc_attn(h)
        if prof:
            t0 = _prof_mark("hc", li, (mixed, inject), t0)
        if self.is_linear:
            out = self.linear_attn(mixed, mask=mask, cache=cache, gdn_sink=gdn_sink)
        else:
            out = self.self_attn(mixed, mask=mask, cache=cache,
                                 positions=positions)
        if prof:
            t0 = _prof_mark("gdn" if self.is_linear else "attn", li, out, t0)
        h = _hc_combine(h, out, inject, kern=not self.training)
        mixed, inject = self.hc_ffn(h)
        if prof:
            t0 = _prof_mark("hc", li, (h, mixed, inject), t0)
        out = self.mlp(mixed)
        if prof:
            t0 = _prof_mark("ffn", li, out, t0)
        h = _hc_combine(h, out, inject, kern=not self.training)
        if prof:
            _prof_mark("hc", li, h, t0)
        return h


class Qwen4ExpModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.hc = args.hc_count
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [DecoderLayer(args, i) for i in range(args.num_hidden_layers)]
        self.hc_head = HyperConnection(args.hidden_size, args.hc_count,
                                       args.hc_lowrank, args.rms_norm_eps,
                                       inject=False)
        if args.ple_layer_ids:
            # The PLE row table (wire bytes at load; KQuantEmbedding gathers
            # and dequantizes only the touched rows).
            self.ple_embed = nn.Embedding(max(int(args.ple_table_rows), 1),
                                          args.ple_embed_dim)
        self.ssm_idx = next(
            (i for i, t in enumerate(args.layer_types) if t == "linear_attention"), 0)
        self.fa_idx = next(
            (i for i, t in enumerate(args.layer_types) if t == "full_attention"), 0)

    def __call__(self, inputs: mx.array, cache=None, input_embeddings=None,
                 return_streams: bool = False, gdn_sink=None, position_ids=None):
        h = input_embeddings if input_embeddings is not None else self.embed_tokens(inputs)
        B, T, D = h.shape
        h = mx.broadcast_to(h[:, :, None, :], (B, T, self.hc, D))
        if cache is None:
            cache = [None] * len(self.layers)
        probe = h[:, :, 0, :]
        fa_mask = create_attention_mask(probe, cache[self.fa_idx])
        ssm_mask = create_ssm_mask(probe, cache[self.ssm_idx])
        table = self.ple_embed if "ple_embed" in self else None
        for layer, c in zip(self.layers, cache):
            mask = ssm_mask if layer.is_linear else fa_mask
            h = layer(h, inputs, mask=mask, cache=c, ple_table=table,
                      gdn_sink=gdn_sink, positions=position_ids)
        out = self.hc_head(h)
        if return_streams:
            return out, h
        return out


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = Qwen4ExpModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(self, inputs: mx.array, cache=None, input_embeddings=None):
        out = self.model(inputs, cache, input_embeddings=input_embeddings)
        if self.args.tie_word_embeddings:
            return self.model.embed_tokens.as_linear(out)
        return self.lm_head(out)

    def sanitize(self, weights):
        return weights

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        caches: List[Any] = []
        for layer in self.layers:
            if layer.is_linear:
                caches.append(ArraysCache(size=4 if "ple" in layer else 2))
            elif layer.self_attn.ratio > 0:
                caches.append(QSAKVCache(layer.self_attn.ratio))
            else:
                caches.append(KVCache())
        return caches


# Arch-default prefill chunk once the split-regime QSA path is armed (8192
# measured 913 vs 758 tps at d8192 over the stock 2048 chunk: MoE gather
# segments grow ~19 -> ~35 TF and the split path keeps QSA out of the dense
# token-mask quadratic). Without the kq block-sparse prefill kernel the big
# window falls back to that dense mask and loses, so the profile only arms
# with the kernel present. Score transient: the indexer's [B, heads, L,
# depth/ratio] fp32 block scores. GMLX_Q4_PREFILL_STEP overrides in either
# direction; an explicit PREFILL_STEP_SIZE (flag/config/env) wins over both,
# and the decode-pressure tick term shrinks live chunks regardless of base.
_Q4_BASE_STEP: Optional[int] = 8192


def _q4_base_step() -> Optional[int]:
    env = os.environ.get("GMLX_Q4_PREFILL_STEP")
    if env:
        try:
            n = int(env)
            return n if n > 0 else None
        except ValueError:
            return None
    return _Q4_BASE_STEP


_prefill_decay = importlib.import_module("gmlx.gen.prefill_decay")

_prefill_score_profile = _prefill_decay.build_score_profile(
    profile=lambda: _prefill_decay.ScoreTransientProfile(
        heads=4, bytes_per_elem=4, depth_divisor=4,
        base_step=_q4_base_step()),
    kernels_armed=lambda: _kq_bs_prefill() is not None,
    require_cache=QSAKVCache,
)


_prefill_decay.register_score_profile("qwen4_exp", _prefill_score_profile)
