# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Asher Feldman
# Portions copyright (c) 2024 Apple Inc. (mlx-lm qwen3_next/qwen3_5 skeleton, MIT)
"""Vendored mlx-lm-style model for Qwen3.8-Flash-Next (GGUF arch ``qwen4exp``).

mlx-lm ships no qwen4_exp class. mlx-vlm ships a qwen4_exp package from
0.7.3, and the vendored vision model (:mod:`gmlx.models.qwen4_exp.vlm_model`)
loads ahead of it. This module is the runtime for llama.cpp's
``LLM_ARCH_QWEN4EXP`` conversions. It is the qwen3.5 hybrid (gated DeltaNet
on three of every four layers, gated full attention on the fourth, a
512-expert top-10 MoE with a sigmoid-gated shared expert on every layer)
plus three mechanisms of its own:

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
from dataclasses import dataclass, field
from typing import Any, List, Optional

import mlx.core as mx
import mlx.nn as nn

from gmlx._vendor.mlx_lm.models.base import BaseModelArgs
from gmlx._vendor.mlx_lm.models.gated_delta import gated_delta_update

from gmlx.cache import bind, refusals
from gmlx.cache.attention import (
    create_attention_mask,
    create_ssm_mask,
    scaled_dot_product_attention,
)
from gmlx.cache.kv_cache import ArraysCache, KVCache, _BaseCache
from gmlx.envflags import env_bool, env_int
from gmlx.gen import deferred_clear
from gmlx.tune.attention import blocked_attention
from gmlx.tune.gdn import training_gated_delta_update
from gmlx._vendor.mlx_lm.models.switch_layers import SwitchGLU


def ensure_registered() -> None:
    """Register this module's ``QSAKVCache`` (gmlx.cache.bind.register):
    prompt-cache save and load resolve cache classes by name, and the APC
    QSA arms check for this class. A cache module that loads later takes it
    too."""
    bind.register(QSAKVCache)


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
    # Resolved rope scaling (gmlx.load.rope_scaling), present only when it
    # changes the rope.
    gmlx_rope_scaling: Optional[dict] = None

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
        if kern and B * T <= 8 and plain:
            types = self._hcmix_types(h.dtype)
            if types is not None:
                # One op with no copies, for the weights of any codec it
                # reads.
                return _kq_hc_mix()(h, self.norm.weight, down.weight,
                                    inject.weight, up.weight, self.norm.eps,
                                    **types)
        if kern and B * T <= 8 and plain and self._hclr_ok(h.dtype):
            norm, front, epi = _kq_hc()
            xn = norm(h, self.norm.weight, self.norm.eps)
            lo, inj = front(xn, down.weight, inject.weight,
                            self.norm.weight.dtype)
            return epi(lo, up.weight, xn), inj
        if kern and B * T <= 8 and plain and self._hcfl_ok(h.dtype):
            return _hc_float_mix(self, h)
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

    def _hcmix_types(self, h_dtype):
        """The codec kwargs of kq hc_lowrank_mix for this mixer's weights,
        or None when it does not take them: a weight codec it does not
        read, up blocks that do not divide the rank, or no inject. Cached
        per module like ``_hclr_ok``."""
        types = self.__dict__.get("_hcmix_cache")
        if types is None:
            types = False
            codecs = _kq_hc_mix_codecs()
            half = self.norm.weight.dtype
            lr = self.down.weight.shape[0]
            if (codecs and self.hc == 4 and "inject" in self
                    and half in (mx.float16, mx.bfloat16)
                    and self.hidden % 64 == 0 and lr % 32 == 0
                    and lr <= 512):
                found = {}
                for name, mod in (("down", self.down),
                                  ("inject", self.inject), ("up", self.up)):
                    c = getattr(mod, "kquant_type", None)
                    if c is None:
                        c = _HC_FLOAT_CODECS.get(mod.weight.dtype)
                    if c is None or c not in codecs:
                        found = None
                        break
                    found[name] = c
                if found is not None and lr % _hc_block(found["up"]) == 0:
                    # A mixer that reads only q8_0 and f32 takes no codec
                    # kwargs.
                    types = {} if codecs == _HC_Q8_MIX else {
                        f"{k}_type": "" if v in _HC_FLOAT_CODEC_NAMES else v
                        for k, v in found.items()}
            self.__dict__["_hcmix_cache"] = types
        if types is False or not (h_dtype == mx.float32
                                  or h_dtype == self.norm.weight.dtype):
            return None
        return types

    def _hcfl_ok(self, h_dtype) -> bool:
        """Eligibility of the float-weight fused path: half down, up and
        norm gain of one dtype, an inject of that dtype or float32,
        kernel-aligned shapes. Cached per module like ``_hclr_ok``."""
        ok = self.__dict__.get("_hcfl_cache")
        if ok is None:
            half = self.norm.weight.dtype
            k = self.hc * self.hidden
            ok = (
                self.hc == 4
                and "inject" in self
                and half in (mx.float16, mx.bfloat16)
                and getattr(self.down, "kquant_type", None) is None
                and getattr(self.up, "kquant_type", None) is None
                and self.down.weight.dtype == half
                and self.up.weight.dtype == half
                and self.inject.weight.dtype in (half, mx.float32)
                and self.hidden % 64 == 0
                and k % 256 == 0
                and self.down.weight.shape[0] % 4 == 0
                and self.down.weight.shape == (self.down.weight.shape[0], k)
                and self.up.weight.shape == (k, self.down.weight.shape[0])
            )
            self.__dict__["_hcfl_cache"] = ok
        return (ok and _kq_hc() is not None and _hc_float_kerns() is not None
                and (h_dtype == mx.float32
                     or h_dtype == self.norm.weight.dtype))


# Decode-width HC kernels for float down and up weights (a GGUF that keeps
# the low-rank pair at bf16). The eager chain runs each mixer as a dozen
# ops around three matmuls; here it is the kq norm and two dispatches that
# read each weight once for every row. Rounding points are the eager
# chain's: a dot rounds to the result dtype before its activation, and so
# does each op after it. Accumulation is fp32. GMLX_Q4_HC_FLOAT_KERN=0
# disables.

# Front: 4 output rows per threadgroup, one simdgroup per K segment, 8
# values per lane per 256-column chunk. The last threadgroup does the 4
# inject rows.
_HC_FLOAT_FRONT_SRC = r"""
    uint lane = thread_position_in_threadgroup.x;
    uint sg = thread_position_in_threadgroup.y;
    uint grp = thread_position_in_grid.y / SG;
    bool down = grp < (uint)(LR / 4);
    int out0 = down ? (int)grp * 4 : 0;
    float result[R * 4];
    for (int i = 0; i < R * 4; i++) result[i] = 0.0f;
    int k0 = (int)sg * (K / SG);
    for (int k = k0; k < k0 + K / SG; k += 256) {
        int kg = k + (int)lane * 8;
        float wt[4][8];
        for (int o = 0; o < 4; o++) {
            if (down) {
                auto wr = wd + (size_t)(out0 + o) * K + kg;
                for (int i = 0; i < 8; i++) wt[o][i] = (float)wr[i];
            } else {
                auto wr = wi + (size_t)o * K + kg;
                for (int i = 0; i < 8; i++) wt[o][i] = (float)wr[i];
            }
        }
        for (int r = 0; r < R; r++) {
            auto xr = xn + (size_t)r * K + kg;
            float xt[8];
            for (int i = 0; i < 8; i++) xt[i] = (float)xr[i];
            for (int o = 0; o < 4; o++) {
                float p = 0.0f;
                for (int i = 0; i < 8; i++) p = metal::fma(xt[i], wt[o][i], p);
                result[r * 4 + o] += p;
            }
        }
    }
    threadgroup float part[SG * R * 4];
    for (int i = 0; i < R * 4; i++) {
        float v = simd_sum(result[i]);
        if (lane == 0) part[sg * R * 4 + i] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0 && lane < (uint)(R * 4)) {
        float v = 0.0f;
        for (int g = 0; g < SG; g++) v += part[g * R * 4 + lane];
        int r = (int)lane / 4, o = (int)lane % 4;
        if (down) {
            float t = (float)(LoT)v * 0.25f;
            float sig = (float)(LoT)(1.0f / (1.0f + metal::precise::exp(-t)));
            lo[(size_t)r * LR + out0 + o] = (LoT)(t * sig);
        } else {
            float t = (float)(InjT)v * 0.25f;
            inj[(size_t)r * 4 + o] = (InjT)(2.0f * (float)(InjT)(
                1.0f / (1.0f + metal::precise::exp(-t))));
        }
    }
"""

# Epilogue: one lane per (d, stream) walks its up row once for every row
# of lo. The 4 stream lanes of a d are neighbors in a simdgroup, and each
# reads the other three for the mean.
_HC_FLOAT_EPI_SRC = r"""
    uint gid = thread_position_in_grid.x;
    uint lane = thread_index_in_simdgroup;
    int s = (int)(gid % 4u);
    int d = (int)(gid / 4u);
    bool live = d < D;
    int dd = live ? d : 0;
    auto w = wu + (size_t)(s * D + dd) * LR;
    float acc[R];
    for (int r = 0; r < R; r++) acc[r] = 0.0f;
    for (int j = 0; j < LR; j += 4) {
        float w0 = (float)w[j], w1 = (float)w[j + 1];
        float w2 = (float)w[j + 2], w3 = (float)w[j + 3];
        for (int r = 0; r < R; r++) {
            auto lor = lo + (size_t)r * LR + j;
            acc[r] += (float)lor[0] * w0 + (float)lor[1] * w1
                    + (float)lor[2] * w2 + (float)lor[3] * w3;
        }
    }
    ushort base = (ushort)(lane & ~3u);
    for (int r = 0; r < R; r++) {
        float a = (float)(LoT)acc[r];
        float g = (float)(LoT)(1.0f / (1.0f + metal::precise::exp(-a)));
        float p = (float)(LoT)(g * (float)xn[(size_t)r * 4 * D + s * D + dd]);
        float m = simd_shuffle(p, base);
        m += simd_shuffle(p, (ushort)(base + 1));
        m += simd_shuffle(p, (ushort)(base + 2));
        m += simd_shuffle(p, (ushort)(base + 3));
        if (s == 0 && live) mixed[(size_t)r * D + d] = (LoT)(m * 0.25f);
    }
"""

_hc_float_kerns_cache = None


def _hc_float_kerns():
    """The decode-width float-weight HC kernels (front, epilogue), or None."""
    global _hc_float_kerns_cache
    if _hc_float_kerns_cache is None:
        from gmlx.envflags import env_bool
        if (env_bool("GMLX_Q4_HC_FLOAT_KERN", True)
                and mx.metal.is_available()
                and mx.default_device().type == mx.DeviceType.gpu):
            _hc_float_kerns_cache = (
                mx.fast.metal_kernel(
                    name="gmlx_hc_float_front",
                    input_names=["xn", "wd", "wi"],
                    output_names=["lo", "inj"], source=_HC_FLOAT_FRONT_SRC),
                mx.fast.metal_kernel(
                    name="gmlx_hc_float_epi",
                    input_names=["lo", "wu", "xn"],
                    output_names=["mixed"], source=_HC_FLOAT_EPI_SRC),
            )
        else:
            _hc_float_kerns_cache = False
    return _hc_float_kerns_cache or None


def _hc_float_mix(m, h: mx.array):
    """``HyperConnection.__call__`` for float down and up weights at
    decode width: (mixed, inject) in the eager chain's dtypes."""
    B, T, hc, D = h.shape
    R, K = B * T, hc * D
    LR = m.down.weight.shape[0]
    front, epi = _hc_float_kerns()
    xn = _kq_hc()[0](h, m.norm.weight, m.norm.eps)
    wi = m.inject.weight
    injt = mx.float32 if mx.float32 in (xn.dtype, wi.dtype) else xn.dtype
    sg = next(n for n in (5, 4, 2, 1) if K % (n * 256) == 0)
    lo, inj = front(
        inputs=[xn, m.down.weight, wi],
        template=[("LoT", xn.dtype), ("InjT", injt), ("K", K), ("LR", LR),
                  ("R", R), ("SG", sg)],
        grid=(32, sg * (LR // 4 + 1), 1), threadgroup=(32, sg, 1),
        output_shapes=[(B, T, LR), (B, T, hc)],
        output_dtypes=[xn.dtype, injt])
    mixed = epi(
        inputs=[lo, m.up.weight, xn],
        template=[("LoT", xn.dtype), ("D", D), ("LR", LR), ("R", R)],
        grid=(-(-hc * D // 256) * 256, 1, 1), threadgroup=(256, 1, 1),
        output_shapes=[(B, T, D)], output_dtypes=[xn.dtype])[0]
    return mixed, inj


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


def _conv_state_in(state: mx.array, dtype) -> mx.array:
    """A conv state in the activation dtype of the call it feeds. Decode
    runs the hc residual float32, and a conv state whose input follows it
    is float32: on the real model the PLE's, and a GDN's after a step
    wider than the hc kernel's 8 rows (the kq low-rank mixer emits bf16
    up to that, the eager hc path float32). A retire record keeps them. A
    prefill after its restore would concatenate them into its bf16 rows
    and run float32 from there on, where the hc prefill kernels decline;
    the cast keeps it in the dtype a fresh prefill runs. A bf16 call
    always runs bf16, and a float32 call (decode, a verify block) gets
    the state the concatenation promoted before."""
    return state if state.dtype == dtype else state.astype(dtype)


class GatedDeltaNet(nn.Module):
    """qwen3.5 gated DeltaNet with a sigmoid output gate and the GGUF tiled
    K->V head pairing applied explicitly. Attribute and cache layout match
    mlx-lm's ``GatedDeltaNet`` so the fused S=1 decode kernel body in
    ``gdn_patches`` (conv -> silu -> q/k norm -> scan -> gated norm in one
    launch) runs unchanged; ``gdn_gate_sigmoid`` selects its gate."""

    gdn_gate_sigmoid = True
    # The fused kernels pair V head hv with K head hv % Hk.
    gdn_kv_tiled = True

    def __init__(self, args: ModelArgs):
        super().__init__()
        self._fused_decode = False
        self._fused_verify = False
        self._verify_v2 = False
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

    def _staged_scan_on(self, S, mask) -> bool:
        """Whether a call of S rows may take the staged prefill scan."""
        from gmlx.models.qwen4_exp import gdn_scan
        return not (self.training or mask is not None or S < 2
                    or not gdn_scan.enabled()
                    or mx.default_device() != mx.gpu)

    def _prefill_fused(self, qkv, conv_state, mask, cache, gdn_sink) -> bool:
        """Whether the prefill kernels of ``gdn_prep`` take this call: one
        unmasked sequence outside a verify forward, with the same bits as
        the op chain."""
        if (self.training or mask is not None or gdn_sink is not None
                or getattr(cache, "lengths", None) is not None
                or mx.default_device() != mx.gpu):
            return False
        from gmlx.models.qwen4_exp import gdn_prep
        return gdn_prep.enabled() and gdn_prep.prepare_supported(
            conv_state, qkv, self.conv1d.weight, self.head_k_dim)

    def _staged_scan(self, q, k, v, a, b, state, mask, ba=None):
        """The staged prefill scan on untiled q and k, or None when it does
        not take this call (the stock scan then). ``ba`` is the joined
        b | a product, from which the scan computes its gates itself."""
        from gmlx.models.qwen4_exp import gdn_scan
        if not self._staged_scan_on(q.shape[1], mask):
            return None
        # The decay has the shape of a: one scalar gate a value head.
        if not gdn_scan.supported(q, v, a, state):
            return None
        if state is None:
            state = mx.zeros((1, self.num_v_heads, self.head_v_dim,
                              self.head_k_dim), dtype=mx.float32)
        if (ba is not None and gdn_scan.gates_enabled()
                and gdn_scan.gates_supported(q, v, ba, self.A_log,
                                             self.dt_bias)):
            return gdn_scan.scan_staged_gates(q, k, v, ba, self.A_log,
                                              self.dt_bias, state,
                                              tiled=self.gdn_kv_tiled)
        from gmlx._vendor.mlx_lm.models.gated_delta import compute_g
        g = compute_g(self.A_log, a, self.dt_bias)
        return gdn_scan.scan_staged(q, k, v, g, mx.sigmoid(b), state,
                                    tiled=self.gdn_kv_tiled)

    def _ba(self, inputs):
        """b and a, and their joined product when one runs (None else): the
        decode path's b | a concatenation, float or k-quant, while b and a
        are still the modules it was cut from. Training calls them apart."""
        import gmlx.upstream.gdn_patches as _gp
        hv = self.num_v_heads
        if not self.training:
            cat = _gp._live_cat(self, "_gdn_ba_weight", ("in_proj_b", "in_proj_a"))
            if cat is not None:
                ba = inputs @ cat.T
                return ba[..., :hv], ba[..., hv:], ba
            if (kcat := _gp._gdn_ba_kq(self)) is not None:
                ba = kcat(inputs)
                return ba[..., :hv], ba[..., hv:], ba
        return self.in_proj_b(inputs), self.in_proj_a(inputs), None

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
            out = _gp._gdn_fused_verify_body(self, inputs, mask, cache, rec,
                                             v2=self._verify_v2)
            entry = {
                "kind": "gdn", "layer": self, "cache": cache, "pre": pre,
                "inputs": inputs, "mask": mask, "conv_input": rec[-1][9],
                "inter": rec[-1][11], "K": self.conv_kernel_size,
            }
            if isinstance(rec[-1][11], _gp.GdnRecords):
                # The one-kernel form: the rollback replays records.
                entry.update(conv_input=None, inter=None,
                             conv=rec[-1][9], records=rec[-1][11])
            gdn_sink.append(entry)
            return out
        from gmlx.upstream.gdn_patches import kquant_multi_or

        qz = kquant_multi_or(
            (self.in_proj_qkv, self.in_proj_z), inputs,
            lambda left: [lin(inputs) for lin in left], prefill=True)
        qkv, z = qz if qz is not None else (self.in_proj_qkv(inputs),
                                            self.in_proj_z(inputs))
        z = z.reshape(B, S, self.num_v_heads, self.head_v_dim)
        b, a, ba = self._ba(inputs)

        n_keep = self.conv_kernel_size - 1
        conv_state = None
        if cache is not None and cache[0] is not None:
            conv_state = cache[0]
            if conv_state.shape[0] != B:
                conv_state = None
        if conv_state is None:
            conv_state = mx.zeros((B, n_keep, self.conv_dim), dtype=inputs.dtype)
        conv_state = _conv_state_in(conv_state, inputs.dtype)

        if mask is not None:
            if mask.shape[0] != B:
                mask = None
            else:
                qkv = mx.where(mask[..., None], qkv, 0)
        inv_scale = self.head_k_dim ** -0.5
        state = cache[1] if cache is not None else None
        if state is not None and state.shape[0] != B:
            state = None
        fused = self._prefill_fused(qkv, conv_state, mask, cache, gdn_sink)
        if fused:
            # The prepare kernel: no concat, and q and k leave it tiled
            # when the stock scan runs.
            from gmlx.models.qwen4_exp import gdn_prep
            scan = self._staged_scan_on(S, mask)
            q, k, v = gdn_prep.prepare(
                conv_state, qkv, self.conv1d.weight, self.num_k_heads,
                self.eps, inv_scale ** 2, inv_scale,
                tile=1 if scan else self.num_v_heads // self.num_k_heads)
            v = v.reshape(B, S, self.num_v_heads, self.head_v_dim)
            if cache is not None:
                cache[0] = mx.contiguous(qkv[:, -n_keep:, :])
            staged = (self._staged_scan(q, k, v, a, b, state, mask, ba)
                      if scan else None)
            return self._scan_and_gate(q, k, v, z, a, b, state, mask, cache,
                                       staged, fused=True)
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
        q = (inv_scale ** 2) * mx.fast.rms_norm(q, None, self.eps)
        k = inv_scale * mx.fast.rms_norm(k, None, self.eps)
        staged = self._staged_scan(q, k, v, a, b, state, mask, ba)
        return self._scan_and_gate(q, k, v, z, a, b, state, mask, cache,
                                   staged, fused=False)

    def _scan_and_gate(self, q, k, v, z, a, b, state, mask, cache, staged,
                       fused):
        """The scan on prepared q, k, v, the cache write and the gated
        output norm. ``fused`` marks a call the prefill kernels take."""
        B, S = v.shape[:2]
        if (staged is None and self.num_v_heads != self.num_k_heads
                and q.shape[2] != self.num_v_heads):
            # GGUF tiled V layout: V head hv pairs with K head hv % Hk.
            r = self.num_v_heads // self.num_k_heads
            q = mx.tile(q, [1, 1, r, 1])
            k = mx.tile(k, [1, 1, r, 1])

        if staged is not None:
            out, state = staged
        elif self.training and cache is None:
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
        if fused:
            from gmlx.models.qwen4_exp import gdn_prep
            if gdn_prep.gate_supported(out, z, self.norm.weight):
                out = gdn_prep.gate(out, z, self.norm.weight, self.norm.eps)
                return self.out_proj(out.reshape(B, S, -1))
        out = self.norm(out, z)
        return self.out_proj(out.reshape(B, S, -1))


_KQ_TOPK_UNSET = object()
_kq_topk_fn = _KQ_TOPK_UNSET
_kq_score_fn = _KQ_TOPK_UNSET
_kq_hc_fns = _KQ_TOPK_UNSET
_kq_hc_mix_fn = _KQ_TOPK_UNSET
_kq_paged_fn = _KQ_TOPK_UNSET
_kq_bs_fn = _KQ_TOPK_UNSET
_kq_rows_fn = _KQ_TOPK_UNSET


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


def _kq_hc_mix():
    """kq one-op low-rank hyper-connection mixer (hc_lowrank_mix), or None
    on a kq build without it (the three ops then). ``GMLX_Q4_HC_MIX=0``
    disables."""
    global _kq_hc_mix_fn
    if _kq_hc_mix_fn is _KQ_TOPK_UNSET:
        from gmlx.envflags import env_bool
        fn = None
        if _kq_hc() is not None and env_bool("GMLX_Q4_HC_MIX", True):
            import mlx_kquant as _kq
            fn = getattr(_kq, "hc_lowrank_mix", None)
        _kq_hc_mix_fn = fn
    return _kq_hc_mix_fn


# A float weight's codec name for hc_lowrank_mix.
_HC_FLOAT_CODECS = {mx.float32: "f32", mx.float16: "f16", mx.bfloat16: "bf16"}
_HC_FLOAT_CODEC_NAMES = frozenset(_HC_FLOAT_CODECS.values())


_HC_Q8_MIX = frozenset(("q8_0", "f32"))


def _kq_hc_mix_codecs() -> frozenset:
    """The weight codecs the kq mixer reads: its list on a kq build that
    has one, q8_0 and f32 (_HC_Q8_MIX) on a build whose mixer reads only
    those, and none without the mixer."""
    if _kq_hc_mix() is None:
        return frozenset()
    import mlx_kquant as _kq
    fn = getattr(_kq, "hc_lowrank_mix_codecs", None)
    return frozenset(fn()) if fn is not None else _HC_Q8_MIX


def _hc_block(codec: str) -> int:
    """Values in one block of a mixer weight codec (16 for a float)."""
    if codec in _HC_FLOAT_CODEC_NAMES:
        return 16
    from mlx_kquant.codec_geometry import CODEC_GEOMETRY
    return CODEC_GEOMETRY[codec][3]


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


# page sizes kq.sdpa_paged_rows walks, by head dim
_KQ_ROWS_PAGES = {64: (32,), 128: (32,), 256: (4, 16), 512: (8,)}


def _kq_paged_rows():
    """kq paged sdpa over a block of queries, each with its own page list,
    K and V read in place from the cache base and its tail, or None (the
    gathered path). ``GMLX_Q4_QSA_ROWS_SDPA=0`` disables."""
    global _kq_rows_fn
    if _kq_rows_fn is _KQ_TOPK_UNSET:
        from gmlx.envflags import env_bool
        fn = None
        if env_bool("GMLX_Q4_QSA_ROWS_SDPA", True):
            try:
                import mlx_kquant as _kq
                if mx.default_device().type == mx.DeviceType.gpu:
                    fn = getattr(_kq, "sdpa_paged_rows", None)
            except ImportError:
                pass
        _kq_rows_fn = fn
    return _kq_rows_fn


_kq_nax_ok = None


def _kq_rows_prefill() -> bool:
    """True when a prefill chunk past the sparse boundary takes the kq
    paged-rows kernel, each query walking its own pages, instead of the
    block-sparse prefill kernel and its window unions: on a GPU with
    tensor-op units, where its tensor-op form is the faster of the two.
    ``GMLX_Q4_QSA_ROWS_PREFILL=0`` keeps the block-sparse kernel."""
    global _kq_nax_ok
    from gmlx.envflags import env_bool
    if not env_bool("GMLX_Q4_QSA_ROWS_PREFILL", True):
        return False
    if _kq_nax_ok is None:
        try:
            import mlx_kquant as _kq
            fn = getattr(_kq, "nax_available", None)
            _kq_nax_ok = bool(fn()) if fn is not None else False
        except ImportError:
            _kq_nax_ok = False
    return _kq_nax_ok


_kq_score_w: dict = {}


def _kq_score_weights(B: int, L: int, n_heads: int, head_dim: int, dtype):
    """Head weights of the fused scorer, ``[B, L, H]`` of
    ``1 / sqrt(head_dim)``. Kept per shape: a fill per call is one more GPU
    dispatch in every indexer layer of every token."""
    key = (B, L, n_heads, head_dim, dtype)
    w = _kq_score_w.get(key)
    if w is None:
        w = mx.full((B, L, n_heads), 1.0 / math.sqrt(head_dim), dtype=dtype)
        _kq_score_w[key] = w
    return w


_kq_score_wide_ok = None


def _kq_score_wide() -> bool:
    """True when the kq fused scorer takes 5 to 8 query rows and a block
    key tail in one call. Probed once with a call that builds no work."""
    global _kq_score_wide_ok
    if _kq_score_wide_ok is None:
        ok = False
        try:
            import mlx_kquant as _kq
            fn = getattr(_kq, "dsa_indexer_score_decode", None)
            if fn is not None:
                k = mx.zeros((1, 8, 128), dtype=mx.bfloat16)
                fn(mx.zeros((1, 4, 8, 128), dtype=mx.bfloat16), k,
                   mx.zeros((1, 8, 4), dtype=mx.bfloat16), 40, 4,
                   keys_tail=k)
                ok = True
        except (ImportError, TypeError, ValueError):
            ok = False
        _kq_score_wide_ok = ok
    return _kq_score_wide_ok


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


_kq_skinny_dtypes: dict = {}


def _kq_skinny(xdt, wdt):
    """kq.skinny_matmul when it takes an ``xdt`` input against a ``wdt``
    weight (probed once per pair with a dry call), else None.
    ``GMLX_KQ_SKINNY=0`` disables."""
    key = (xdt, wdt)
    if key not in _kq_skinny_dtypes:
        from gmlx.envflags import env_bool
        fn = None
        if (env_bool("GMLX_KQ_SKINNY", True)
                and mx.default_device().type == mx.DeviceType.gpu):
            try:
                import mlx_kquant as _kq
                cand = getattr(_kq, "skinny_matmul", None)
                if cand is not None:
                    cand(mx.zeros((1, 8), xdt), mx.zeros((8, 8), wdt))
                    fn = cand
            except (ImportError, ValueError):
                fn = None
        _kq_skinny_dtypes[key] = fn
    return _kq_skinny_dtypes[key]


def _indexer_proj(lin, x: mx.array) -> mx.array:
    """``lin(x)`` for an indexer projection. Up to 16 tokens take
    kq.skinny_matmul: MLX's float32 GEMM leaves its GEMV path at 2 rows
    and runs these weights several times under their bytes at verify
    widths. The kernel computes a row as its one-row call does, so a
    row's block selection does not depend on the verify width. The
    indexer only feeds the selection, under stop_gradient, so the kernel
    needs no vjp."""
    if (type(lin) is nn.Linear and "bias" not in lin
            and x.ndim >= 2 and x.shape[-2] <= 16 and x.shape[-1] % 4 == 0
            and (fn := _kq_skinny(x.dtype, lin.weight.dtype)) is not None):
        return fn(x, lin.weight)
    return lin(x)


_kq_score_geom: dict = {}


def _kq_score_fits(n_heads: int, head_dim: int) -> bool:
    """Whether the fused scorer takes an indexer of this many heads of this
    dim. Its host validator decides, once per shape."""
    fn = _kq_score()
    if fn is None:
        return False
    ok = _kq_score_geom.get((n_heads, head_dim))
    if ok is None:
        try:
            fn(mx.zeros((1, n_heads, 1, head_dim), dtype=mx.bfloat16),
               mx.zeros((1, 8, head_dim), dtype=mx.bfloat16),
               mx.zeros((1, 1, n_heads), dtype=mx.bfloat16), 0, 4)
            ok = True
        except ValueError:
            ok = False
        _kq_score_geom[(n_heads, head_dim)] = ok
    return ok


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
                   selector: mx.array, rope=None):
    """``positions [3, B, L]`` -> ``(cos, sin)`` each ``[B, L, rotary_dims]``
    in the non-traditional (rotate-half) layout. ``rope`` (a ``_RopeTable``)
    replaces the plain table and scales cos and sin by its magnitude."""
    half = rotary_dims // 2
    if rope is not None and rope.inv is not None:
        inv = rope.inv
    else:
        inv = mx.power(base, -mx.arange(0, half, dtype=mx.float32) / half)
    freqs = positions.astype(mx.float32)[..., None] * inv  # [3, B, L, half]
    _, B, L, _ = freqs.shape
    idx = mx.broadcast_to(selector[None, None, None, :], (1, B, L, half))
    freqs = mx.take_along_axis(freqs, idx.astype(mx.uint32), axis=0)[0]
    emb = mx.concatenate([freqs, freqs], axis=-1)
    if rope is not None and rope.mscale != 1.0:
        return mx.cos(emb) * rope.mscale, mx.sin(emb) * rope.mscale
    return mx.cos(emb), mx.sin(emb)


class _RopeTable:
    """The scaled rope of one site (gmlx.load.rope_scaling): the
    ``mx.fast.rope`` table, its reciprocal for the cos/sin path, and the
    magnitude llama.cpp multiplies cos and sin by (ggml rope_yarn)."""

    def __init__(self, raw: dict, rotary_dims: int, base: float):
        from gmlx.load import rope_scaling as rs

        scaling = rs.RopeScaling(**raw)
        if scaling.freq_base_override and scaling.freq_base is not None:
            base = scaling.freq_base
        self.dims = rotary_dims
        self.freqs = rs.freqs(rotary_dims, base, scaling)
        self.inv = mx.array(rs.inv_freq(rotary_dims, base, scaling))
        self.mscale = scaling.mscale

    def __call__(self, x, offset, scale=1.0):
        if self.mscale != 1.0:
            d = self.dims
            x = mx.concatenate([x[..., :d] * self.mscale, x[..., d:]], axis=-1)
        return mx.fast.rope(x, self.dims, traditional=False, base=None,
                            scale=scale, offset=offset, freqs=self.freqs)


def _rope_table(args, rotary_dims: int) -> "_RopeTable | None":
    raw = getattr(args, "gmlx_rope_scaling", None)
    return _RopeTable(raw, rotary_dims, args.rope_theta) if raw else None


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


_norm_rope_kernel = None


def _norm_rope_source() -> str:
    """Source of the indexer's rms-norm and rope in one dispatch, bit for bit with
    the op chain: mx.fast.rms_norm's reduction and rounding, then
    mx.fast.rope (not traditional) on the first DIMS features.

    NH 0 finishes blocks (QSAIndexer.finish_blocks): each threadgroup
    mean-pools the R raw rows of one block in float32, rounds to the raw
    dtype, then norms and ropes at the block's start position. raw is
    [B, m * R, D]; out [B, m, D].

    NH > 0 makes the queries (QSAIndexer._queries then the cast to the
    scorer's dtype): R is 1, raw is the q projection [B, m, NH * D], row
    l is roped at position start + l, and out is [B, NH, m, D] in T_OUT."""
    return """
    const uint row = threadgroup_position_in_grid.x;
    const uint lid = thread_position_in_threadgroup.x;
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const int m = meta[0];
    const int start = meta[1];
    const float eps = fp[0];
    const float scale = fp[1];
    const float base_l2 = fp[2];
    const float inv_r = fp[3];
    constexpr int HALF = DIMS / 2;
    int b, i;
    size_t o_row;
    const device T_IN* src;
    if (NH == 0) {
      b = row / m;
      i = row % m;
      src = raw + ((size_t)b * raw_shape[1] + (size_t)i * R) * D;
      o_row = (size_t)b * m + i;
    } else {
      b = row / (m * NH);
      i = (row / NH) % m;
      src = raw + (size_t)row * D;
      o_row = ((size_t)b * NH + row % NH) * m + i;
    }

    threadgroup float local_sums[32];
    threadgroup float local_inv[1];
    threadgroup T ys[D];

    T xv[4];
    float acc = 0;
    for (int k = 0; k < 4; k++) {
      const int e = lid * 4 + k;
      if (e < D) {
        float s = 0;
        if (R == 1) {
          s = static_cast<float>(src[e]);
        } else {
          for (int rr = 0; rr < R; rr++) {
            s = s + static_cast<float>(src[rr * D + e]);
          }
        }
        const T_IN p = static_cast<T_IN>(s * inv_r);
        xv[k] = static_cast<T>(p);
        float xi = xv[k];
        acc += xi * xi;
      }
    }
    acc = simd_sum(acc);
    if (sg == 0) {
      local_sums[lane] = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) {
      local_sums[sg] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
      acc = simd_sum(local_sums[lane]);
      if (lane == 0) {
        local_inv[0] = metal::precise::rsqrt(acc / (uint)D + eps);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int k = 0; k < 4; k++) {
      const int e = lid * 4 + k;
      if (e < D) {
        ys[e] = w[e] * static_cast<T>(xv[k] * local_inv[0]);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int k = 0; k < 4; k++) {
      const int e = lid * 4 + k;
      if (e >= D) {
        continue;
      }
      T o;
      if (e < DIMS) {
        const int j = e < HALF ? e : e - HALF;
        const float d = static_cast<float>(j) / static_cast<float>(HALF);
        const float inv_freq = metal::exp2(-d * base_l2);
        const float L = scale * static_cast<float>(i + start);
        const float theta = L * inv_freq;
        const float c = metal::fast::cos(theta);
        const float sn = metal::fast::sin(theta);
        const float x1 = static_cast<float>(ys[j]);
        const float x2 = static_cast<float>(ys[j + HALF]);
        float rx;
        if (e < HALF) {
          rx = x1 * c - x2 * sn;
        } else {
          rx = x1 * sn + x2 * c;
        }
        o = static_cast<T>(rx);
      } else {
        o = ys[e];
      }
      out[o_row * D + e] = static_cast<T_OUT>(o);
    }
    """


def _make_norm_rope_kernel():
    return mx.fast.metal_kernel(
        name="q4_norm_rope",
        input_names=["raw", "w", "meta", "fp"],
        output_names=["out"],
        source=_norm_rope_source(),
        ensure_row_contiguous=True,
    )


def _log2f(x: float) -> float:
    """log2 of float32 ``x`` in float32: the libm ``log2f`` mx.fast.rope
    calls on its base, or the rounded double result without it."""
    import ctypes
    import struct

    try:
        fn = ctypes.CDLL(None).log2f
        fn.restype, fn.argtypes = ctypes.c_float, [ctypes.c_float]
        return float(fn(x))
    except (OSError, AttributeError):
        f32 = struct.unpack("f", struct.pack("f", x))[0]
        return struct.unpack("f", struct.pack("f", math.log2(f32)))[0]


def _norm_rope():
    """The fused indexer norm and rope kernel, or None (the op chains).
    ``GMLX_Q4_QSA_NORM_ROPE=0`` disables."""
    global _norm_rope_kernel
    if _norm_rope_kernel is None:
        from gmlx.envflags import env_bool
        if (mx.default_device().type == mx.DeviceType.gpu
                and env_bool("GMLX_Q4_QSA_NORM_ROPE", True)):
            _norm_rope_kernel = _make_norm_rope_kernel()
        else:
            _norm_rope_kernel = False
    return _norm_rope_kernel or None


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
    # The indexer keys and the block keys grow beside K and V; the config
    # prices them as this layer's pools (mem_preflight.pool_bytes_per_token).
    holds_config_pools = True
    # Kept out of batching and the stock KV quant arms (gmlx.cache.refusals).
    to_batch = refusals.refuse_batch
    preserve_auxiliary_kv_state = True
    extract = refusals.no_extract

    _kb = _vb = _ib = None  # base segments
    _kt = _vt = _it = None  # decode tails
    _tl = 0                 # rows live in the tails
    _tail_cap = None
    _bt = None              # finished blocks after the first _nbb
    _nbb = 0                # finished blocks live in ``blocks``
    _fence = None           # copies to wait for before the next write
    # Set by a model whose decode batch cannot keep its rows on their own
    # caches (Qwen4ExpModel._qsa_dense_batch): merge lifts such rows dense.
    dense_lift = False

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
        self._kb = self._owned(v)
        self._tl = 0

    @property
    def values(self):
        self._fold()
        return self._vb

    @values.setter
    def values(self, v):
        self._vb = self._owned(v)
        self._tl = 0

    @property
    def ik(self):
        self._fold()
        return self._ib

    @ik.setter
    def ik(self, v):
        self._ib = self._owned(v)
        self._tl = 0

    @staticmethod
    def _owned(v):
        """``v`` as one compact buffer, evaluated. The prompt cache
        evaluates what it assigns and then reads the cache from its disk
        writer thread, which cannot evaluate an array of this thread."""
        if v is None:
            return None
        v = mx.contiguous(v)
        mx.eval(v)
        return v

    def hold_writes(self, arrays: list) -> None:
        """Wait for ``arrays`` before the next append: a copy still in
        flight holds the buffers it reads, and MLX would copy a base whole
        rather than write it in place. The list may grow until then."""
        self._fence = arrays

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
        when it is short. A grown base keeps its dtype, as the write in
        place does: a decode-width step runs float32 off the hc residual,
        and its rows must not widen the whole stream."""
        n = rows.shape[-2]
        if base is None or start + n > base.shape[-2]:
            dt = rows.dtype if base is None else base.dtype
            shape = list(rows.shape)
            shape[-2] = cls._cap_for(start + n) - (start + n)
            parts = [rows.astype(dt), mx.zeros(shape, dt)]
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
        fence = self._fence
        if fence is not None:
            self._fence = None
            if fence:
                mx.eval(fence)
        prev = self.offset
        B, _, n, _ = keys.shape
        cap = self._tail_rows()
        tail = self._kb is not None and n <= min(8, cap)
        if tail:
            # The tails take the base's dtype, as _write does, so rows a
            # float32 step wrote after a restore keep gather_kv's kernel.
            kdt = self._kb.dtype
            if (self._kt is None or self._kt.shape[0] != B
                    or self._kt.dtype != kdt):
                self._fold()
                idt = ik.dtype if self._ib is None else self._ib.dtype
                self._kt = mx.zeros(keys.shape[:2] + (cap, keys.shape[3]),
                                    kdt)
                self._vt = mx.zeros(values.shape[:2] + (cap, values.shape[3]),
                                    self._vb.dtype)
                self._it = mx.zeros((B, cap, ik.shape[2]), idt)
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
            if held and new.dtype != held[0].dtype:
                new = new.astype(held[0].dtype)
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
        # An aligned prompt-cache skeleton holds no K/V rows (the block
        # chain holds them), only ik.
        k, v, ik = self.keys, self.values, self.ik
        n = self.offset
        if k is not None and n != k.shape[2]:
            k, v = k[..., :n, :], v[..., :n, :]
        base = (k, v, None if ik is None else ik[:, :n])
        return base if self.pos is None else base + (self.pos[:, :, :n],)

    @state.setter
    def state(self, v):
        if len(v) == 4:
            self.keys, self.values, self.ik, self.pos = v
        else:
            self.keys, self.values, self.ik = v
            self.pos = None
        rows = self._kb if self._kb is not None else self._ib
        self.offset = 0 if rows is None else int(rows.shape[-2])
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
        # Two callers share this name. The decode batch's lift calls
        # merge(caches): the rows join a BatchQSAKVCache as they are, each
        # with its indexer stream, so every batched row runs the
        # single-row sparse attention. mlx-vlm's APC merge_cache_entries
        # calls merge(entries, prefix_lens) and dispatches on the class's
        # own __dict__, so the arm must live here, not on a parent.
        # Single-request batches only; multi-row joins decline to the cold
        # path rather than guess at row padding for the indexer stream.
        if prefix_lens is None:
            if any(getattr(e, "dense_lift", False) for e in entries):
                # The padded batch class: keys and values only, each row
                # dense over its whole context.
                return super().merge(entries)
            return BatchQSAKVCache.merge(entries)
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


class BatchQSAKVCache(_BaseCache):
    """A decode batch's cache on a QSA layer: one single-row QSAKVCache per
    batch row, in batch order.

    Each row keeps its indexer key stream, so the layer can run the
    single-row sparse attention on each row. A join, a leave or a filter
    moves rows in the list and copies no array, and no row is padded.
    ``left_padding`` is all zeros: it marks the cache as batched for the
    admission lift, and no row has any. The list is ``rows``, not
    ``caches``: readers that walk a CacheList's members would take one
    row's offset for the layer's, and the admission lift would find a
    member without ``left_padding`` and merge the cache again.
    """

    kv_quant_unsupported = True
    holds_config_pools = True
    preserve_auxiliary_kv_state = True
    holds_row_caches = True

    def __init__(self, rows=()):
        self.rows = list(rows)

    @property
    def left_padding(self):
        return mx.zeros((len(self.rows),), mx.int32)

    @property
    def offset(self):
        return mx.array([r.offset for r in self.rows], mx.int32)

    @classmethod
    def merge(cls, caches):
        """Single rows and containers joined, in order, with no copy."""
        rows = []
        for c in caches:
            rows.extend(c.rows if isinstance(c, BatchQSAKVCache) else [c])
        return cls(rows)

    def extend(self, other):
        self.rows.extend(other.rows if isinstance(other, BatchQSAKVCache)
                         else [other])

    def filter(self, batch_indices):
        keep = (batch_indices.tolist() if isinstance(batch_indices, mx.array)
                else list(batch_indices))
        self.rows = [self.rows[int(i)] for i in keep]

    def extract(self, idx):
        """Row ``idx`` as a single-row QSAKVCache on the row's buffers, so
        its record tags "qsa:R" as the lookups sign and a batched row's
        retirement stores.

        The row's tail folds in first, in place: a fold on the copy would
        rewrite the whole base while the live row still holds it. The copy
        takes new handles on the buffers, not the row's arrays: a slice
        assignment updates the array object it writes, so two caches
        holding one array object would each write into the other's rows.
        With a handle each, a write on either side lands in a copy. The
        block keys are rebuilt from the indexer stream when read."""
        row = self.rows[idx]
        row._fold()
        out = QSAKVCache(row.ratio)
        out._kb, out._vb, out._ib, out.pos = (
            None if a is None else mx.contiguous(a)
            for a in (row._kb, row._vb, row._ib, row.pos))
        out.offset = row.offset
        return out

    def is_trimmable(self):
        return True

    def trim(self, n):
        """Every row by the same ``n`` (no batched path rolls rows back by
        different counts)."""
        n = min([n] + [r.offset for r in self.rows])
        for r in self.rows:
            r.trim(n)
        return n

    def make_mask(self, N, return_array=False, window_size=None, **kwargs):
        # Each row's attention builds its own mask.
        return None if N == 1 else "causal"

    @property
    def state(self):
        """Every row's buffers as they are held, with no fold and no slice:
        what the batch evaluates before it changes."""
        return [tuple(a for a in (r._kb, r._vb, r._ib, r._kt, r._vt, r._it,
                                  r.blocks, r._bt, r.pos) if a is not None)
                for r in self.rows]

    def size(self):
        return max((r.offset for r in self.rows), default=0)

    def empty(self):
        return all(r.offset == 0 for r in self.rows)

    @property
    def nbytes(self):
        return sum(r.nbytes for r in self.rows)

    def to_quantized(self, group_size: int = 64, bits: int = 4):
        raise NotImplementedError(
            "BatchQSAKVCache cannot quantize: the QSA indexer key stream has "
            "no quantized form (drop --kv-bits for qwen4exp)")


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
        self._rope_scaled = _rope_table(args, rotary_dims)
        self.q_proj = nn.Linear(args.hidden_size, self.n_heads * self.head_dim,
                                bias=False)
        self.k_proj = nn.Linear(args.hidden_size, self.head_dim, bias=False)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self._li = -1

    def _rope(self, x, offset, scale=1.0):
        if self._rope_scaled is not None:
            return self._rope_scaled(x, offset, scale)
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
        fused = self._fused_finish(raw, start_block)
        if fused is not None:
            return fused
        return self._chain_finish(raw, start_block)

    def _chain_finish(self, raw: mx.array, start_block: int) -> mx.array:
        return self._rope(self._pooled(raw)[:, None], start_block,
                          scale=float(self.ratio))[:, 0]

    def _fused_finish(self, raw: mx.array, start_block: int):
        """``finish_blocks`` in one kernel, or None where the kernel does
        not apply: a scaled rope table, a dtype other than float32 and the
        16-bit floats, or an odd rope width. The kernel has no vjp; the
        uncached (training) forward takes the op chain."""
        B, n, D = raw.shape
        m = n // self.ratio
        w = self.k_norm.weight
        float_types = (mx.float32, mx.float16, mx.bfloat16)
        if (m == 0 or self._rope_scaled is not None
                or raw.dtype not in float_types or w.dtype not in float_types
                or D % 4 or self.rotary_dims % 2 or self.rotary_dims > D
                or (kern := _norm_rope()) is None):
            return None
        out_t = mx.float32 if mx.float32 in (raw.dtype, w.dtype) else raw.dtype
        fp = getattr(self, "_finish_fp", None)
        if fp is None:
            # eps, rope scale, log2 of the rope base and 1 / ratio, each
            # rounded to float32 as the stock kernels round them
            fp = mx.array(
                [self.k_norm.eps, float(self.ratio),
                 _log2f(self.rope_theta), 1.0 / self.ratio],
                dtype=mx.float32)
            self._finish_fp = fp
        tg = 32 * (((D + 3) // 4 + 31) // 32)
        return kern(
            inputs=[raw, w.astype(out_t),
                    mx.array([m, start_block], dtype=mx.int32), fp],
            template=[("T_IN", raw.dtype), ("T", out_t), ("T_OUT", out_t),
                      ("D", D), ("DIMS", self.rotary_dims),
                      ("R", self.ratio), ("NH", 0)],
            grid=(tg * B * m, 1, 1),
            threadgroup=(tg, 1, 1),
            output_shapes=[(B, m, D)],
            output_dtypes=[out_t],
        )[0]

    def _fused_queries(self, x: mx.array, offset: int, dtype):
        """``_queries(x, offset).astype(dtype)`` with the norm, rope and
        cast in one kernel, or None where the kernel does not apply (as
        ``_fused_finish``)."""
        B, L, _ = x.shape
        D, H = self.head_dim, self.n_heads
        w = self.q_norm.weight
        float_types = (mx.float32, mx.float16, mx.bfloat16)
        if (self._rope_scaled is not None or x.dtype not in float_types
                or w.dtype not in float_types or dtype not in float_types
                or D % 4 or self.rotary_dims % 2 or self.rotary_dims > D
                or (kern := _norm_rope()) is None):
            return None
        q = _indexer_proj(self.q_proj, x)
        t = mx.float32 if mx.float32 in (q.dtype, w.dtype) else q.dtype
        fp = getattr(self, "_query_fp", None)
        if fp is None:
            fp = mx.array([self.q_norm.eps, 1.0, _log2f(self.rope_theta), 1.0],
                          dtype=mx.float32)
            self._query_fp = fp
        tg = 32 * (((D + 3) // 4 + 31) // 32)
        return kern(
            inputs=[q, w.astype(t), mx.array([L, offset], dtype=mx.int32), fp],
            template=[("T_IN", q.dtype), ("T", t), ("T_OUT", dtype),
                      ("D", D), ("DIMS", self.rotary_dims), ("R", 1),
                      ("NH", H)],
            grid=(tg * B * L * H, 1, 1),
            threadgroup=(tg, 1, 1),
            output_shapes=[(B, H, L, D)],
            output_dtypes=[dtype],
        )[0]

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
                                  selector, self._rope_scaled)
        return _apply_rope_cos_sin(pooled[:, None], cos, sin,
                                   self.rotary_dims)[:, 0]

    def _queries(self, x: mx.array, offset: int, cos, sin) -> mx.array:
        """Normed, roped indexer queries ``[B, H, L, D]``."""
        B, L, _ = x.shape
        q = _indexer_proj(self.q_proj, x).reshape(B, L, self.n_heads,
                                                  self.head_dim)
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
        elif pos_all is not None:
            blocks = finish(ik_all[:, :n_blocks * r], 0)
        else:
            blocks = self._chain_finish(ik_all[:, :n_blocks * r], 0)
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
        half = (mx.float16, mx.bfloat16)
        kq_shape = topk is not None and k == 512 and n_blocks >= k
        kq_ok = kq_shape and x.dtype in half
        # The fused scorer reads 16-bit queries and block keys. Its dtype is
        # the block keys' when they are 16-bit (the cache's), else a 16-bit
        # layer input's. A float32 layer input keeps its query projection,
        # norm and rope in float32, and only the finished queries narrow.
        sdt = blocks.dtype if blocks.dtype in half else (
            x.dtype if x.dtype in half else None)
        if (kq_shape and sdt is not None and L <= 8
                and _kq_score_fits(self.n_heads, self.head_dim)):
            # Fused decode/verify path: one kernel scores every pooled block
            # (relu dots summed over the 4 heads, per-row visibility baked as
            # finite_min) and the radix top-k consumes its 16-bit rows
            # directly. Replaces the astype/matmul/relu/sum/where chain. A kq
            # build that takes 8 queries and the block key tail scores the
            # whole block in one call; an older one takes 4 queries a call
            # and one key segment.
            q = self._fused_queries(x, offset, sdt) if cos is None else None
            if q is None:
                q = self._queries(x, offset, cos, sin).astype(sdt)
            blocks = blocks.astype(sdt)
            if btail is not None:
                btail = btail.astype(sdt)
            parts = []
            if _kq_score_wide():
                w = _kq_score_weights(B, L, self.n_heads, self.head_dim, sdt)
                parts.append(_kq_score()(q, blocks, w, offset, r,
                                         keys_tail=btail))
            for a in range(0, L if not parts else 0, 4):
                qa = q if L <= 4 else q[:, :, a:a + 4]
                w = _kq_score_weights(B, qa.shape[2], self.n_heads,
                                      self.head_dim, sdt)
                sa = _kq_score()(qa, blocks, w, offset + a, self.ratio)
                if btail is not None:
                    # The scorer numbers blocks from 0, so the tail's call
                    # moves the query back by the base's tokens to keep
                    # visibility.
                    sa = mx.concatenate(
                        [sa, _kq_score()(qa, btail, w,
                                         offset + a - blocks.shape[1] * r, r)],
                        axis=-1)
                parts.append(sa)
            s16 = parts[0] if len(parts) == 1 else mx.concatenate(parts, axis=2)
            if prof:
                t0 = _prof_mark("a.score", self._li, s16, t0)
            # Block ids sit below 2**31: the top-k's uint32 is read as int32
            # in place, the dtype every consumer indexes with.
            sel = mx.stop_gradient(topk(s16, k, True)[:, 0].view(mx.int32))
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
        self._rope_scaled = _rope_table(args, self.rotary_dims)
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
        if self._rope_scaled is not None:
            return self._rope_scaled(x, offset)
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

    def _rows_kernel(self, q, k, v, cache, seg, L):
        """The kq paged-rows sdpa when it serves this block, else None: a
        verify-width block or a prefill chunk (_kq_rows_prefill), a page
        size and head grouping the kernel is built for, and q, K and V in
        one 16-bit float dtype."""
        H, Hkv, D = self.num_attention_heads, self.num_key_value_heads, self.head_dim
        if not ((2 <= L <= 16 or (L > 16 and _kq_rows_prefill()))
                and self.ratio in _KQ_ROWS_PAGES.get(D, ())
                and H % Hkv == 0 and H // Hkv <= 16
                and q.dtype in (mx.bfloat16, mx.float16)):
            return None
        if seg:
            kv = (cache._kb, cache._vb, cache._kt, cache._vt)
        else:
            kv = (k, v)
        if any(a.dtype != q.dtype for a in kv):
            return None
        return _kq_paged_rows()

    def _rows_attention(self, q, k, v, cache, seg, sel, complete, offset, L,
                        rows):
        """Verify block: each query walks its own selected blocks and its
        own tail block as pages, in one dispatch, with K and V read in place
        (the cache base and its tail when ``seg``). No per-query K/V copy."""
        B = q.shape[0]
        # Each query's own tail block, ends // ratio, is the page the kernel
        # adds to its list.
        pages = sel.astype(mx.int32)
        ends = mx.broadcast_to(
            (offset + 1 + mx.arange(L, dtype=mx.int32))[None], (B, L))
        if seg:
            return rows(q, cache._kb, cache._vb, self.scale, pages, ends,
                        cache._kt, cache._vt, cache.offset - cache._tl,
                        page_rows=self.ratio, tail_page=True)
        return rows(q, k, v, self.scale, pages, ends, page_rows=self.ratio,
                    tail_page=True)

    def _block_sparse_prefill(self, q, k, v, sel, offset, key_len, bs):
        """Prefill chunk with every query sparse: fold queries into 4-wide
        windows and walk each window's own page list (union of its queries'
        selected blocks plus the window's tail span) through the kq
        block-sparse FA kernel. The page lists take the widest union a
        window can have (its queries' blocks and the tail span, at most
        every block), so no host read stops the forward;
        ``GMLX_Q4_QSA_BS_STATIC_PAGES=0`` reads the widest one instead."""
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
        if _bs_static_pages():
            max_p = min(r * topk + 2, nb_total)
        else:
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
        rows = (self._rows_kernel(q, k, v, None, False, L - c)
                if L - c > 16 else None)
        if rows is not None:
            outs.append(self._rows_attention(
                q[..., c:, :], k, v, None, False, sel[:, c:], complete[c:],
                offset + c, L - c, rows))
            return mx.concatenate(outs, axis=2) if len(outs) > 1 else outs[0]
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

    def _project(self, x: mx.array):
        """The layer's projections of ``x``: q (normed, not roped), the
        output gate, k (normed, not roped), v, and the raw indexer keys
        (None on a layer without an indexer)."""
        B, L, _ = x.shape
        H, Hkv, D = self.num_attention_heads, self.num_key_value_heads, self.head_dim
        from gmlx.upstream.gdn_patches import kquant_multi_or

        lins = (self.q_proj, self.k_proj, self.v_proj)
        qkv = kquant_multi_or(lins, x, lambda left: [lin(x) for lin in left],
                              prefill=True)
        qg, k, v = qkv if qkv is not None else (lin(x) for lin in lins)
        qg = qg.reshape(B, L, H, 2 * D)
        q, gate = mx.split(qg, 2, axis=-1)
        gate = gate.reshape(B, L, -1)
        q = self.q_norm(q).transpose(0, 2, 1, 3)
        k = self.k_norm(k.reshape(B, L, Hkv, D)).transpose(0, 2, 1, 3)
        v = v.reshape(B, L, Hkv, D).transpose(0, 2, 1, 3)
        ik = (_indexer_proj(self.indexer.k_proj, x) if "indexer" in self
              else None)
        return q, gate, k, v, ik

    def __call__(self, x: mx.array, mask=None, cache=None,
                 positions=None) -> mx.array:
        """``positions`` is the ``[3, B, L]`` mrope position block for this
        window (VLM loads); ``None`` takes the scalar-offset text path."""
        B, L, _ = x.shape
        q, gate, k, v, ik = self._project(x)
        if isinstance(cache, BatchQSAKVCache):
            out = self._attend_rows(x, q, k, v, ik, cache, positions)
        else:
            out = self._attend(x, q, k, v, ik, mask, cache, positions)
        prof = _subprof_on(L)
        if prof:
            import time
            t0 = time.perf_counter()
        out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)
        out = self.o_proj(out * mx.sigmoid(gate))
        if prof:
            _prof_mark("a.out", self._li, out, t0)
        return out

    def _attend_rows(self, x, q, k, v, ik, cache, positions) -> mx.array:
        """A decode batch on a BatchQSAKVCache: each row's attention is the
        single-row path on that row's slice of the projections, over the
        row's own cache with the row's own mask, so it computes what the
        row alone computes. The model's batch mask is not read."""
        rows = cache.rows
        if len(rows) != x.shape[0]:
            raise ValueError(
                f"BatchQSAKVCache holds {len(rows)} rows for a batch of "
                f"{x.shape[0]}")
        outs = []
        for i, row in enumerate(rows):
            s = slice(i, i + 1)
            xi = x[s]
            outs.append(self._attend(
                xi, q[s], k[s], v[s], None if ik is None else ik[s],
                create_attention_mask(xi, row), row,
                None if positions is None else positions[:, s]))
        return outs[0] if len(outs) == 1 else mx.concatenate(outs, axis=0)

    def _attend(self, x, q, k, v, ik, mask, cache, positions) -> mx.array:
        """Rope, the cache append, the block selection and the attention
        core over one cache: ``[B, H, L, D]``. ``x`` is the layer input the
        indexer queries project from."""
        B, L, _ = x.shape
        H, Hkv, D = self.num_attention_heads, self.num_key_value_heads, self.head_dim
        offset = cache.offset if cache is not None else 0
        cos = sin = None
        if positions is not None:
            cos, sin = _mrope_cos_sin(positions, self.rotary_dims,
                                      self.rope_theta, self._mrope_selector,
                                      self._rope_scaled)
            q = _apply_rope_cos_sin(q, cos, sin, self.rotary_dims)
            k = _apply_rope_cos_sin(k, cos, sin, self.rotary_dims)
        else:
            q = self._rope(q, offset)
            k = self._rope(k, offset)

        prof = _subprof_on(L)
        if prof:
            import time
            t0 = _prof_mark("a.proj", self._li, (q, k, v),
                            time.perf_counter())
        selection = None
        seg = False
        if "indexer" in self:
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
            rows = self._rows_kernel(q, k, v, cache, seg, L) if (
                kern and cache is not None and all_sparse
                and not isinstance(mask, mx.array)) else None
            if seg and not seg_ok and rows is None:
                k, v = cache.kv_full()
            if rows is not None:
                out = self._rows_attention(q, k, v, cache, seg, sel,
                                           complete, offset, L, rows)
            elif seg_ok:
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
            _prof_mark("a.core", self._li, out, time.perf_counter())
        return out


# MoE (qwen3-next shape: router + SwitchGLU + sigmoid-gated shared expert)


class MLP(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)

    def __call__(self, x) -> mx.array:
        from gmlx.upstream.gdn_patches import kquant_multi_or

        gu = kquant_multi_or((self.gate_proj, self.up_proj), x,
                             lambda left: [lin(x) for lin in left], prefill=True)
        g, u = gu if gu is not None else (self.gate_proj(x), self.up_proj(x))
        return self.down_proj(nn.silu(g) * u)


class SparseMoeBlock(nn.Module):
    # The fused k-quant block runs up to this many rows. Its default bound
    # (64 gathered rows) stops at 6 with top_k 10; a speculative verify of
    # 7 to 16 rows is faster fused than as prefill gathers.
    kq_block_rows = max(0, env_int("GMLX_Q4_MOE_BLOCK_ROWS", 16))

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
        y = None
        if self.switch_mlp.__dict__.get("_kq_wide_store") is not None:
            # A full expert store's wide form mixes in its last kernel.
            from gmlx.load import expert_store
            y = expert_store.wide_glu_mixed(self.switch_mlp, x, inds, scores)
        if y is None:
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


@dataclass(frozen=True)
class PLEHostHash:
    """``PLEEmbedding.row_ids`` in numpy, for a reader that needs a
    prompt's table rows before its prefill runs (the table read-ahead).
    Called with ``[B, context_len + T]`` token ids, the first
    ``context_len`` of them the history before the T tokens, it returns
    their ``[B, T, n_heads]`` rows as int64. Plain values only, so it
    holds nothing of the model."""

    context_len: int
    ngram_size: int
    heads_per_ngram: int
    eos_token_id: int
    mults: tuple
    sizes: tuple
    offsets: tuple

    def __call__(self, hist):
        import numpy as np

        hist = np.asarray(hist, dtype=np.int64)
        if hist.ndim == 1:
            hist = hist[None]
        B, L = hist.shape
        eos = self.eos_token_id
        pos = np.arange(L)
        eos_pos = np.where(hist == eos, pos[None], -1)
        prev_eos = np.concatenate(
            [np.full((B, 1), -1, np.int64),
             np.maximum.accumulate(eos_pos, axis=1)[:, :-1]], axis=1)
        shifted = []
        for s in range(self.ngram_size):
            src = pos - s
            g = hist[:, np.maximum(src, 0)]
            valid = (src[None] > prev_eos) & (src[None] >= 0)
            shifted.append(np.where(valid, g, eos).astype(np.uint64))
        mults = [np.uint64(m) for m in self.mults]
        sizes = np.asarray(self.sizes, dtype=np.uint64)
        offs = np.asarray(self.offsets, dtype=np.uint64)
        blocks = []
        with np.errstate(over="ignore"):
            for n in range(2, self.ngram_size + 1):
                lo = (n - 2) * self.heads_per_ngram
                hi = lo + self.heads_per_ngram
                mixed = shifted[0] * mults[0]
                for p in range(1, n):
                    mixed = mixed ^ (shifted[p] * mults[p])
                blocks.append(mixed[..., None] % sizes[lo:hi] + offs[lo:hi])
        rows = np.concatenate(blocks, axis=-1)[:, self.context_len:]
        return rows.astype(np.int64)


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

    def host_hash(self) -> PLEHostHash:
        """This hash on the host (``PLEHostHash``)."""
        return PLEHostHash(
            self.context_len, self.ngram_size, self.heads_per_ngram,
            int(self.eos_token_id), tuple(int(m) for m in self._mults),
            tuple(int(v) for v in self._sizes.tolist()),
            tuple(int(v) for v in self._offsets.tolist()))

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

    def note_history(self, input_ids: mx.array, cache, prev) -> None:
        """Advance the cache's token history as row_ids does, for a call
        that takes its rows from elsewhere."""
        if cache is not None:
            hist = mx.concatenate([prev, input_ids.astype(mx.int64)], axis=1)
            cache[3] = mx.contiguous(hist[:, -self.context_len:])

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


_PLE_EARLY = None


def _bs_static_pages() -> bool:
    return os.environ.get("GMLX_Q4_QSA_BS_STATIC_PAGES", "1") != "0"


# Rows from which a forward submits each layer as it is built.
_PIPELINE_MIN_ROWS = 32


def _prefill_pipeline() -> bool:
    """Whether a forward of prefill width submits each layer as it is
    built, so the GPU runs it while the host builds the next one.
    ``GMLX_Q4_PREFILL_PIPELINE=0`` builds the whole forward first."""
    return os.environ.get("GMLX_Q4_PREFILL_PIPELINE", "1") != "0"


def _ple_early() -> bool:
    """Whether a verify submits its table rows as its PLE layer is built,
    and a prefill forward before its first layer. ``GMLX_Q4_PLE_EARLY=0``
    leaves them to the submission of the layer that reads them."""
    global _PLE_EARLY
    if _PLE_EARLY is None:
        from gmlx.envflags import env_bool

        _PLE_EARLY = env_bool("GMLX_Q4_PLE_EARLY", True)
    return _PLE_EARLY


# Bumped by invalidate_verify_compile: a layer's recorded verify graphs of
# an older epoch are dropped on their next use.
_VERIFY_EPOCH = 0
# Set for the process once a weight swap runs around a layer call (streamed
# experts): the swap happens inside the body a trace would freeze.
_VERIFY_BLOCKED = False


def invalidate_verify_compile(block: bool = False) -> None:
    """Drop every recorded verify graph: a module of the model was replaced
    or wrapped (a LoRA install). ``block`` also turns the recorded path
    off for the process (weights swapped around each layer call)."""
    global _VERIFY_EPOCH, _VERIFY_BLOCKED
    _VERIFY_EPOCH += 1
    if block:
        _VERIFY_BLOCKED = True


def _layer_leaves(node, out: list) -> list:
    """``(container, key, value)`` for every array and module under
    ``node``: what a trace of the layer froze."""
    items = (node.items() if isinstance(node, dict)
             else enumerate(node) if isinstance(node, (list, tuple)) else ())
    for k, v in items:
        if isinstance(v, mx.array):
            out.append((node, k, v))
        elif isinstance(v, (dict, list, tuple)):
            if isinstance(v, nn.Module):
                out.append((node, k, v))
            _layer_leaves(v, out)
    return out


def _layer_has_lora(layer) -> bool:
    """Whether a module under ``layer`` carries a LoRA: a wrapped linear,
    or an expert stack with a stamped adapter. Their deltas read a row
    vector published per forward, which a trace would freeze."""
    from gmlx.load.modules import LoRAKQuantEmbedding, LoRAKQuantLinear

    for m in layer.modules():
        if isinstance(m, (LoRAKQuantLinear, LoRAKQuantEmbedding)):
            return True
        if getattr(m, "_kq_lora", None) is not None:
            return True
    return False


# Slot for the speculative round loop (gmlx.spec.verify_submit): called as
# hook(layer index, stream) ahead of each layer of a verify forward and
# returns the stream, which it may pass through a kq commit mark.
_VERIFY_LAYER_HOOK = None

_VERIFY_FUSED = None


def _verify_fused() -> bool:
    """Whether a verify runs its gated-delta layers as recorded kq graphs
    (``GMLX_Q4_VERIFY_FUSED``): the layer's graph is recorded once per
    input layout and replayed as one primitive, which on a GPU stream
    replays the layer's Metal calls. On by default (0 turns it off). Needs
    a kq build with fuse_graph; the kq lever cache goes on with it, so a
    KQ_* lever is read once."""
    global _VERIFY_FUSED
    if _VERIFY_FUSED is None:
        on = env_bool("GMLX_Q4_VERIFY_FUSED", True)
        if on:
            import mlx_kquant as kq

            on = hasattr(kq, "fuse_graph") and hasattr(kq, "env_cache")
            if on:
                kq.env_cache(True)
        _VERIFY_FUSED = on
    return _VERIFY_FUSED and not _VERIFY_BLOCKED


_VERIFY_FUSED_QSA = None


def _verify_fused_qsa() -> bool:
    """Whether the fused verify also covers the attention layers
    (``GMLX_Q4_VERIFY_FUSED_QSA=0`` keeps them on the plain call)."""
    global _VERIFY_FUSED_QSA
    if _VERIFY_FUSED_QSA is None:
        _VERIFY_FUSED_QSA = env_bool("GMLX_Q4_VERIFY_FUSED_QSA", True)
    return _VERIFY_FUSED_QSA and _verify_fused()


# Layers that hold records, for the bound on the records of the process.
_FUSED_LAYERS: list = []


def _fused_release(table: dict) -> None:
    import mlx_kquant as kq

    for rec in table.values():
        if isinstance(rec, tuple):
            for handle in rec[:2]:
                if isinstance(handle, int):
                    kq.fuse_release(handle)
                    _FUSED_HELD[0] -= 1


def drop_verify_records(layers) -> None:
    """Release the recorded verify graphs of ``layers`` now, for arrays
    they hold as constants that were let go (an expert store released);
    each layer records again from its next use."""
    for layer in layers:
        recs = vars(layer).get("_verify_fused")
        if recs is not None:
            _fused_release(recs[1])
            # An epoch no use matches: the next one starts a table.
            object.__setattr__(layer, "_verify_fused", (-1, {}, []))


_VERIFY_FUSED_CAP = None


def _verify_fused_cap() -> bool:
    """Whether the gated-delta layers of a fused verify write their
    records and conv rows at a fixed row capacity, so a record's layout
    depends on its own round's width and not on the round before it
    (``GMLX_Q4_VERIFY_FUSED_CAP=0`` keeps the per-width arrays)."""
    global _VERIFY_FUSED_CAP
    if _VERIFY_FUSED_CAP is None:
        _VERIFY_FUSED_CAP = env_bool("GMLX_Q4_VERIFY_FUSED_CAP", True)
    return _VERIFY_FUSED_CAP


def _fused_kw() -> dict:
    """Keywords of a record that replays its Metal calls, its kernels in
    barrier groups (fewer barriers than the walk's order, same bits)."""
    return {"direct": True, "level": 2}


# Record handles held by the process, handles made since the last bound
# check, records evicted; and the use counter the records are aged by.
_FUSED_HELD = [0, 0, 0]
_FUSED_TICK = [0]
_FUSED_LIMITS = None
# Handles made between two checks of the byte bound.
_FUSED_CHECK_EVERY = 64


def _fused_limits() -> tuple:
    """The bound on the records of the process: handles
    (``GMLX_Q4_VERIFY_FUSED_MAX_RECORDS``) and the bytes their programs
    keep (``GMLX_Q4_VERIFY_FUSED_MAX_MB``)."""
    global _FUSED_LIMITS
    if _FUSED_LIMITS is None:
        from gmlx.envflags import env_int

        _FUSED_LIMITS = (
            max(2, env_int("GMLX_Q4_VERIFY_FUSED_MAX_RECORDS", 4096)),
            max(1, env_int("GMLX_Q4_VERIFY_FUSED_MAX_MB", 256)) << 20)
    return _FUSED_LIMITS


def _fused_use() -> list:
    """The age stamp of a new record: [last use, bytes once known]."""
    _FUSED_TICK[0] += 1
    return [_FUSED_TICK[0], None]


def _fused_note_new(handles: int) -> None:
    """Count ``handles`` new record handles and keep the process inside
    its bound: over it, the least recently used records are released and
    their layouts go back to the state before a record (the next use runs
    plain and records again)."""
    _FUSED_HELD[0] += handles
    _FUSED_HELD[1] += handles
    if (_FUSED_HELD[0] > _fused_limits()[0]
            or _FUSED_HELD[1] >= _FUSED_CHECK_EVERY):
        _fused_bound()


def _fused_bytes(rec) -> int:
    """Bytes the programs of one table entry keep; cached on the entry
    once its programs exist (they are made at the first replay)."""
    import mlx_kquant as kq

    use = rec[-1]
    if use[1] is None:
        total, known = 0, True
        for handle in (rec[:2] if isinstance(rec[2], bool) else rec[:1]):
            info = kq.fused_direct_info(handle)
            total += info["program_bytes"] + info["kept_bytes"]
            known = known and info["program_bytes"] > 0
        if not known:
            return total
        use[1] = total
    return use[1]


def _fused_bound() -> None:
    import mlx_kquant as kq

    max_handles, max_bytes = _fused_limits()
    _FUSED_HELD[1] = 0
    held = []
    for ref in _FUSED_LAYERS:
        layer = ref()
        recs = vars(layer).get("_verify_fused") if layer is not None else None
        if recs is None:
            continue
        for key, rec in recs[1].items():
            if isinstance(rec, tuple):
                held.append((rec[-1][0], _fused_bytes(rec), recs[1], key, rec))
    handles = 2 * len(held)
    total = sum(h[1] for h in held)
    _FUSED_HELD[0] = handles
    if handles <= max_handles and total <= max_bytes:
        return
    held.sort(key=lambda h: h[0])
    for _, nbytes, table, key, rec in held:
        if handles <= max_handles * 3 // 4 and total <= max_bytes * 3 // 4:
            break
        for handle in rec[:2]:
            if isinstance(handle, int):
                kq.fuse_release(handle)
        table[key] = False
        handles -= 2
        total -= nbytes
        _FUSED_HELD[2] += 1
    _FUSED_HELD[0] = handles


_FUSED_STEPS: dict = {}


def _fused_steps(S: int, B: int) -> tuple:
    """The evaluated step counts of a full accept at ``S`` rows, one pair
    per shape for the process: evaluating one per record stopped the build
    of a verify for a GPU round trip at every layer. Two arrays, the
    pending conv window's and the replay's: the next round's record takes
    both as inputs, and one array cannot be two inputs."""
    steps = _FUSED_STEPS.get((S, B))
    if steps is None:
        import gmlx.upstream.gdn_patches as _gp

        steps = (_gp.replay_steps(S, B), _gp.replay_steps(S, B))
        mx.eval(steps)
        _FUSED_STEPS[(S, B)] = steps
    return steps


def _fused_gdn_verify(layer, h, c, gdn_sink):
    """One gated-delta layer of a verify through its recorded graph, or
    None where the plain call must run: the first use of a layout, a layer
    with a LoRA, a cache that holds a non-array, or a layer whose verify
    is not the one-kernel records form. The second use of a layout runs
    plain and records the graph it built; later uses replay it. The record
    holds the layer's arrays as constants and is dropped once one of the
    layer's arrays or modules was replaced. Every array of the request the layer may read
    (the stream, the cache slots, a pending replay and a pending conv
    window) is an input of the record, never a constant."""
    import mlx_kquant as kq

    import gmlx.upstream.gdn_patches as _gp

    # A PLE layer's cache holds the PLE state behind the two slots read
    # here; its PLE block has run by now (see the caller).
    slots = list(c.cache)[:2]
    if (len(slots) != 2 or not all(isinstance(a, mx.array) for a in slots)
            or _prof_on(h.shape[1])):
        return None
    recs = vars(layer).get("_verify_fused")
    if (recs is None or recs[0] != _VERIFY_EPOCH
            or not all(d[k] is v for d, k, v in recs[2])):
        if recs is not None:
            _fused_release(recs[1])
        else:
            import weakref

            _FUSED_LAYERS.append(weakref.ref(layer))
        recs = (_VERIFY_EPOCH, {}, _layer_leaves(layer, []))
        object.__setattr__(layer, "_verify_fused", recs)
        if _verify_fused_cap():
            # From here on the layer's one-kernel verify leaves records
            # and conv rows of one shape for every width.
            layer.linear_attn._verify_row_cap = True
    table = recs[1]
    # The pending records as the layer's take_ calls would see them.
    rp = getattr(c, "_gdn_replay", None)
    if rp is not None and rp[0] is not slots[1]:
        rp = None
    cv = getattr(c, "_gdn_conv", None)
    if cv is not None and cv[0] is not slots[0]:
        cv = None
    cand = [h, slots[0], slots[1]]
    if rp is not None:
        cand += [rp[1], *rp[2], rp[3]]
    if cv is not None:
        cand += [cv[1], cv[2], cv[3]]
    key = (rp is not None, cv is not None,
           tuple((a.shape, a.dtype) for a in cand))
    rec = table.get(key)
    if rec is None:
        table[key] = "plain" if _layer_has_lora(layer) else False
        return None
    if rec == "plain":
        return None
    S = int(h.shape[1])
    if rec is False:
        table[key] = "plain"
        sink: list = []
        # The layer call from behind its PLE block on.
        kern = not layer.training
        mixed, inject = layer.hc_attn(h)
        out = _hc_combine(h, layer.linear_attn(
            mixed, mask=None, cache=c, gdn_sink=sink), inject, kern=kern)
        mixed, inject = layer.hc_ffn(out)
        out = _hc_combine(out, layer.mlp(mixed), inject, kern=kern)
        gdn_sink.extend(sink)
        e = sink[-1] if len(sink) == 1 else {}
        r, conv, left = e.get("records"), e.get("conv"), getattr(
            c, "_gdn_replay", None)
        # The fixed row capacity form leaves the conv window pending.
        left_cv = getattr(c, "_gdn_conv", None)
        if (r is None or conv is None or left is None or left[0] is not c[1]
                or left[1] is not r.base or len(r.records) != 3
                or any(a is not b for a, b in zip(left[2], r.records))
                or (left_cv is not None and (
                    left_cv[0] is not c[0] or left_cv[1] is not conv.base
                    or left_cv[2] is not conv.rows))):
            return out
        # What the call leaves behind: the output, the conv state, and the
        # records and conv rows of the rollback.
        want = [out, c[0], r.base, *r.records, conv.base, conv.rows]
        ins = {id(a): i for i, a in enumerate(cand)}
        if len(ins) != len(cand):
            return out
        outs: list = []
        refs = []
        for a in want:
            if id(a) in ins:
                refs.append((0, ins[id(a)]))
                continue
            j = next((j for j, o in enumerate(outs) if o is a), len(outs))
            if j == len(outs):
                outs.append(a)
            refs.append((1, j))
        try:
            probe = kq.fuse_graph(cand, outs)
            used = kq.fused_info(probe)[2]
            kq.fuse_release(probe)
            pick = [i for i, u in enumerate(used) if u or i == 0]
            handle = kq.fuse_graph([cand[i] for i in pick], outs,
                                   **_fused_kw())
        except (ValueError, RuntimeError, TypeError):
            return out
        # The state after a full accept, which set_gdn_replay left lazy
        # in the cache: its own record, so a replay builds one node for it
        # and a reader that evaluates it runs the replay kernel as before.
        try:
            lazy = kq.fuse_graph([left[1], *left[2]], [left[0]])
        except (ValueError, RuntimeError):
            kq.fuse_release(handle)
            return out
        table[key] = (handle, lazy, pick, refs, e["K"],
                      _fused_steps(S, int(h.shape[0])), left_cv is not None,
                      _fused_use())
        _fused_note_new(2)
        return out
    handle, lazy, pick, refs, K, steps, pend_cv, use = rec
    _FUSED_TICK[0] += 1
    use[0] = _FUSED_TICK[0]
    res = kq.fused_call(handle, [cand[i] for i in pick])
    out, conv_out, rbase, rd, rk, rg, cbase, crows = (
        (cand if kind == 0 else res)[i] for kind, i in refs)
    if pend_cv:
        c._gdn_conv = (conv_out, cbase, crows, steps[0])
    elif hasattr(c, "_gdn_conv"):
        c._gdn_conv = None
    c[0] = conv_out
    # As set_gdn_replay(c, rbase, (rd, rk, rg), S) leaves the cache.
    state = kq.fused_call(lazy, [rbase, rd, rk, rg])[0]
    c[1] = state
    c._gdn_replay = (state, rbase, (rd, rk, rg), steps[1])
    if hasattr(c, "advance"):
        # As the one-kernel verify leaves the cache (_gdn_verify_v2).
        from gmlx.models.qwen35.owned import (
            _qwen3_5_advance_left_padding_info,
            _qwen3_5_advance_lengths_info,
        )

        c.advance(S)
        _qwen3_5_advance_left_padding_info(c, S)
        _qwen3_5_advance_lengths_info(c, S)
    gdn_sink.append({
        "kind": "gdn", "layer": layer.linear_attn, "cache": c,
        "pre": (slots[0], slots[1]), "inputs": None, "mask": None,
        "conv_input": None, "inter": None, "K": K,
        "conv": _gp.GdnConvRows(cbase, crows),
        "records": _gp.GdnRecords(rbase, (rd, rk, rg))})
    return out


def _fused_qsa_verify(layer, h, c, mask, positions):
    """One attention layer of a verify with its two ends as recorded
    graphs, or None where the plain call must run. The first record is
    the stream mix and the projections, the second the output projection,
    the combines and the MoE; both depend on the row count alone. Between
    them the rope, the cache append, the block selection and the
    attention core are built as the plain call builds them: they depend
    on the cache offset and length. Records follow the rules of
    _fused_gdn_verify."""
    import mlx_kquant as kq

    if layer.training or _prof_on(h.shape[1]):
        return None
    recs = vars(layer).get("_verify_fused")
    if (recs is None or recs[0] != _VERIFY_EPOCH
            or not all(d[k] is v for d, k, v in recs[2])):
        if recs is not None:
            _fused_release(recs[1])
        else:
            import weakref

            _FUSED_LAYERS.append(weakref.ref(layer))
        recs = (_VERIFY_EPOCH, {}, _layer_leaves(layer, []))
        object.__setattr__(layer, "_verify_fused", recs)
    table = recs[1]
    key = (h.shape, h.dtype)
    rec = table.get(key)
    if rec is None:
        table[key] = "plain" if _layer_has_lora(layer) else False
        return None
    if rec == "plain":
        return None
    attn = layer.self_attn
    B, L = h.shape[0], h.shape[1]

    def tail(h, core, gate, inject):
        out = core.transpose(0, 2, 1, 3).reshape(B, L, -1)
        out = attn.o_proj(out * mx.sigmoid(gate))
        h = _hc_combine(h, out, inject, kern=True)
        mixed, inj = layer.hc_ffn(h)
        return _hc_combine(h, layer.mlp(mixed), inj, kern=True)

    if rec is False:
        table[key] = "plain"
        mixed, inject = layer.hc_attn(h)
        q, gate, k, v, ik = attn._project(mixed)
        first = [mixed, inject, q, gate, k, v] + ([] if ik is None else [ik])
        try:
            ha = kq.fuse_graph([h], first, **_fused_kw())
        except (ValueError, RuntimeError, TypeError):
            ha = None
        core = attn._attend(mixed, q, k, v, ik, mask, c, positions)
        out = tail(h, core, gate, inject)
        if ha is None:
            return out
        try:
            hb = kq.fuse_graph([h, core, gate, inject], [out], **_fused_kw())
        except (ValueError, RuntimeError, TypeError):
            kq.fuse_release(ha)
            return out
        table[key] = (ha, hb, ik is None, (core.shape, core.dtype),
                      _fused_use())
        _fused_note_new(2)
        return out
    ha, hb, no_ik, core_key, use = rec
    _FUSED_TICK[0] += 1
    use[0] = _FUSED_TICK[0]
    r = kq.fused_call(ha, [h])
    mixed, inject, q, gate, k, v = r[:6]
    ik = None if no_ik else r[6]
    core = attn._attend(mixed, q, k, v, ik, mask, c, positions)
    if (core.shape, core.dtype) != core_key:
        return tail(h, core, gate, inject)
    return kq.fused_call(hb, [h, core, gate, inject])[0]


def verify_fused_stats(model) -> dict:
    """Counts and bytes of the recorded verify graphs under ``model``:
    records, how many replay Metal calls directly, and the bytes their
    programs keep (scratch pool counted once per record that uses it)."""
    import mlx_kquant as kq

    out = {"records": 0, "direct": 0, "replays": 0, "plain_runs": 0,
           "kept_bytes": 0, "program_bytes": 0, "scratch_bytes": 0,
           "pool_bytes": 0, "layouts": 0, "never_replayed": 0,
           "evicted": _FUSED_HELD[2], "refused": {}}
    for m in model.modules():
        recs = vars(m).get("_verify_fused")
        if recs is None:
            continue
        out["layouts"] += len(recs[1])
        for rec in recs[1].values():
            if not isinstance(rec, tuple):
                continue
            # The lazy state record of a gated-delta layer is not direct.
            direct = (rec[:2] if isinstance(rec[2], bool) else rec[:1])
            for handle in direct:
                out["records"] += 1
                info = kq.fused_direct_info(handle)
                out["direct"] += int(info["replays"] > 0)
                out["never_replayed"] += int(info["replays"] == 0)
                out["replays"] += info["replays"]
                out["plain_runs"] += info["plain_runs"]
                for k in ("kept_bytes", "program_bytes", "scratch_bytes"):
                    out[k] += info[k]
                out["pool_bytes"] = max(out["pool_bytes"], info["pool_bytes"])
                if info["reason"]:
                    why = info["reason"] + " " + ",".join(info["primitives"])
                    out["refused"][why] = out["refused"].get(why, 0) + 1
    return out


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

    def _conv(self, x: mx.array, cache, fused: bool = False):
        """silu(conv) of the state and ``x``, and the conv input, which a
        ``fused`` call (one kernel, no conv input built) returns as None."""
        B = x.shape[0]
        state = None
        if cache is not None and cache[2] is not None:
            state = cache[2]
            if state.shape[0] != B:
                state = None
        if state is None:
            state = mx.zeros((B, self.conv_state_len, x.shape[-1]), dtype=x.dtype)
        state = _conv_state_in(state, x.dtype)
        if fused:
            from gmlx.models.qwen4_exp import ple_prep
            if ple_prep.enabled() and ple_prep.conv_supported(
                    state, x, self.conv1d.weight, self.conv_dilation):
                L, T = self.conv_state_len, x.shape[1]
                if cache is not None:
                    cache[2] = (mx.contiguous(x[:, -L:]) if T >= L
                                else mx.concatenate([state[:, T:], x], axis=1))
                return ple_prep.conv_silu(state, x, self.conv1d.weight,
                                          self.conv_dilation), None
        conv_input = mx.concatenate([state, x], axis=1)
        if cache is not None:
            cache[2] = mx.contiguous(conv_input[:, -self.conv_state_len:])
        return nn.silu(self.conv1d(conv_input)), conv_input

    def rows_ahead(self, input_ids: mx.array, cache, table) -> None:
        """Read the table rows of the call that follows with these ids and
        this cache, ahead of it. A prefill forward calls this before it
        builds its first layer: a streamed table reads the rows on the
        table stream, beside the layers ahead of this one."""
        prev = self.embedding.prev_history(cache, input_ids.shape[0])
        emb = self.embedding(input_ids, cache, table, prev=prev)
        object.__setattr__(self, "_rows_ahead", (input_ids, emb))
        mx.async_eval(emb)

    def __call__(self, h: mx.array, input_ids: mx.array, cache, table,
                 mask=None, gdn_sink=None) -> mx.array:
        B, T, hc, D = h.shape
        ahead = vars(self).pop("_rows_ahead", None)
        prev = self.embedding.prev_history(cache, B)
        # The rows of the last verify, kept on the layer (a cache snapshot
        # restore would drop them from the cache): (history, rows).
        kept = vars(self).get("_kept_rows") if gdn_sink is not None else None
        if ahead is not None and ahead[0] is input_ids and gdn_sink is None:
            # rows_ahead read them and moved the cache's token history on.
            emb = ahead[1]
        elif (kept is not None and vars(self).get("_take_rows") == id(cache)
                and kept[0] is prev and T <= kept[1].shape[1]):
            # The caller said this block is a leading part of the one the
            # last verify ran from this state (reuse_verify_rows): row t
            # depends on the tokens up to t, so its rows are those rows.
            self.embedding.note_history(input_ids, cache, prev)
            emb = kept[1][:, :T]
        else:
            emb = self.embedding(input_ids, cache, table, prev=prev)
            if gdn_sink is not None:
                object.__setattr__(self, "_kept_rows", (prev, emb))
        if gdn_sink is not None:
            object.__setattr__(self, "_take_rows", None)
        if (gdn_sink is not None and _ple_early()
                and not vars(self).get("_hold_rows")):
            # A verify: the rows depend on the tokens alone, and a
            # streamed table reads them on the table stream. Submitted
            # here, the read runs while the host builds and encodes the
            # rest of the forward, which otherwise stops the GPU for it.
            # Not while the caller holds the rows (hold_verify_rows): a
            # verify built ahead that may be dropped reads them with its
            # own submission.
            mx.async_eval(emb)
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
        conv, conv_input = self._conv(
            normed, cache, fused=gdn_sink is None and not self.training)
        if gdn_sink is not None:
            gdn_sink.append({
                "kind": "ple", "cache": cache, "prev": prev, "ids": input_ids,
                "ctx": self.embedding.context_len, "conv_input": conv_input,
                "L": self.conv_state_len,
            })
        return h + (gated + conv).reshape(B, T, hc, D)


def prepare_runtime(model) -> dict:
    """Arm the owned kernel routes after the weights are loaded: the fused
    S=1 GDN decode kernel (``GMLX_FUSED_GDN=0`` disables), the
    concatenated b/a decay matvec it consumes, and the one-kernel verify
    (``GMLX_GDN_VERIFY_V2=0`` keeps the two-kernel states form). Returns
    counts for the load log."""
    import gmlx.upstream.gdn_patches as _gp
    from gmlx.envflags import env_bool

    global _VERIFY_BLOCKED
    # Each load starts with the recorded verify on: a model that swaps its
    # weights around its layer calls turns it off again at its next call.
    _VERIFY_BLOCKED = False
    counts = {"gdn_fused": 0, "gdn_ba_cat": 0, "gdn_fused_verify": 0}
    want = env_bool("GMLX_FUSED_GDN", True)
    fused = want and _gp._gdn_fused_decode_kernel is not None
    fused_verify = want and _gp._gdn_fused_verify_kernel is not None
    cat_ba = env_bool("GMLX_GDN_BA_CAT", True)
    for m in model.modules():
        if isinstance(m, GatedDeltaNet):
            m._fused_decode = fused
            m._fused_verify = fused_verify
            m._verify_v2 = fused_verify and _gp.gdn_verify_v2_default()
            counts["gdn_fused"] += int(fused)
            counts["gdn_fused_verify"] += int(fused_verify)
            if fused and cat_ba and _gp._gdn_try_cat_ba(m):
                counts["gdn_ba_cat"] += 1
    return counts


def rollback_verify_sink(sink, n: int) -> None:
    """Rewind the recurrent caches after an MTP verify forward over ``S``
    positions to the state after its first ``n`` (the accepted prefix).

    GDN: after the one-kernel verify the cache is left as the block's start
    plus a replay of its first ``n`` records and conv rows, which the next
    fused kernel runs in its prologue. After the two-kernel verify the conv
    state is a window of the recorded conv input and the scan state is the
    kernel's per-position intermediate; without either (unfused path) the
    layer is re-run over the prefix from its pre-verify state. PLE: both the token history and the conv state are
    windows of recorded arrays. KV caches are trimmed by the caller.
    """
    for e in sink:
        cache = e["cache"]
        if cache is None:
            continue
        if e["kind"] == "gdn":
            K = e["K"]
            if e.get("records") is not None:
                import gmlx.upstream.gdn_patches as _gp

                rec, conv = e["records"], e["conv"]
                _gp.set_gdn_replay(cache, rec.base, rec.records, n)
                _gp.set_gdn_conv(cache, conv.base, conv.rows, n)
                continue
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
        attn = [layer.self_attn for layer in self.layers if not layer.is_linear]
        self._all_qsa = all(a.ratio > 0 for a in attn)
        # The batch mask comes from the first attention layer's cache. Where
        # that layer is QSA and another is plain, a cache of one cache per
        # row on it would build no mask for the plain layer's padding, so
        # the QSA rows join the padded batch class (QSAKVCache.dense_lift).
        self._qsa_dense_batch = (bool(attn) and attn[0].ratio > 0
                                 and not self._all_qsa)

    def __call__(self, inputs: mx.array, cache=None, input_embeddings=None,
                 return_streams: bool = False, gdn_sink=None, position_ids=None,
                 capture=None):
        """``capture`` is ``(layer_ids, sink)``: the output of each listed
        layer, averaged over its streams, is appended to ``sink`` (the
        DFlash drafter's taps, llama.cpp's hc_stream_mean)."""
        cap_ids, cap_sink = capture if capture is not None else ((), None)
        h = input_embeddings if input_embeddings is not None else self.embed_tokens(inputs)
        B, T, D = h.shape
        h = mx.broadcast_to(h[:, :, None, :], (B, T, self.hc, D))
        if cache is None:
            cache = [None] * len(self.layers)
        if self._qsa_dense_batch:
            # A restored row's QSA caches are rebuilt from its record, not
            # by make_cache, and every row runs at least its last prompt
            # token here before it joins a batch.
            for c in cache:
                if isinstance(c, QSAKVCache):
                    c.dense_lift = True
        if isinstance(cache[self.fa_idx], BatchQSAKVCache) and not self._all_qsa:
            # The batch mask comes from that layer's cache, which builds
            # none: a plain attention layer would attend its padding.
            raise NotImplementedError(
                "batched decode needs every attention layer on the QSA cache")
        probe = h[:, :, 0, :]
        fa_mask = create_attention_mask(probe, cache[self.fa_idx])
        ssm_mask = create_ssm_mask(probe, cache[self.ssm_idx])
        table = self.ple_embed if "ple_embed" in self else None
        pipeline = (gdn_sink is None and T >= _PIPELINE_MIN_ROWS
                    and not self.training and _prefill_pipeline())
        if pipeline and _ple_early():
            for layer, c in zip(self.layers, cache):
                if "ple" in layer:
                    layer.ple.rows_ahead(inputs, c, table)
        for li, (layer, c) in enumerate(zip(self.layers, cache)):
            mask = ssm_mask if layer.is_linear else fa_mask
            if _VERIFY_LAYER_HOOK is not None and gdn_sink is not None:
                h = _VERIFY_LAYER_HOOK(li, h)
            if (gdn_sink is not None and layer.is_linear and mask is None
                    and B == 1 and "ple" in layer and _verify_fused()
                    and not _prof_on(T)):
                # The PLE block as the plain call runs it, then the rest
                # of the layer through its record.
                hp = layer.ple(h, inputs, c, table, mask, gdn_sink=gdn_sink)
                out = _fused_gdn_verify(layer, hp, c, gdn_sink)
                if out is None:
                    kern = not layer.training
                    mixed, inject = layer.hc_attn(hp)
                    out = _hc_combine(hp, layer.linear_attn(
                        mixed, mask=None, cache=c, gdn_sink=gdn_sink),
                        inject, kern=kern)
                    mixed, inject = layer.hc_ffn(out)
                    out = _hc_combine(out, layer.mlp(mixed), inject,
                                      kern=kern)
                h = out
                if li in cap_ids:
                    cap_sink.append(h.mean(axis=2))
                continue
            if (gdn_sink is not None and layer.is_linear and mask is None
                    and B == 1 and "ple" not in layer
                    and _verify_fused()):
                out = _fused_gdn_verify(layer, h, c, gdn_sink)
                if out is not None:
                    h = out
                    if li in cap_ids:
                        cap_sink.append(h.mean(axis=2))
                    continue
            elif (gdn_sink is not None and not layer.is_linear and B == 1
                    and "ple" not in layer and type(c) is QSAKVCache
                    and "indexer" in layer.self_attn and _verify_fused_qsa()):
                out = _fused_qsa_verify(layer, h, c, mask, position_ids)
                if out is not None:
                    h = out
                    if li in cap_ids:
                        cap_sink.append(h.mean(axis=2))
                    continue
            h = layer(h, inputs, mask=mask, cache=c, ple_table=table,
                      gdn_sink=gdn_sink, positions=position_ids)
            tap = h.mean(axis=2) if li in cap_ids else None
            if tap is not None:
                cap_sink.append(tap)
            if pipeline:
                # A tap goes with its layer: left lazy, it would hold the
                # layer's streams to the forward's end.
                mx.async_eval(*((h,) if tap is None else (h, tap)))
                # The clear a chunk before this one left for the GPU's
                # next work (gmlx.gen.deferred_clear).
                deferred_clear.run()
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

    def table_row_hashes(self) -> dict:
        """The host hash of each lookup table's rows, by the table's path:
        a reader computes what a prompt's prefill reads from it."""
        ple = next((layer.ple for layer in self.layers if "ple" in layer),
                   None)
        if ple is None or "ple_embed" not in self.model:
            return {}
        return {"model.ple_embed": ple.embedding.host_hash()}

    def make_cache(self):
        caches: List[Any] = []
        for layer in self.layers:
            if layer.is_linear:
                caches.append(ArraysCache(size=4 if "ple" in layer else 2))
            elif layer.self_attn.ratio > 0:
                c = QSAKVCache(layer.self_attn.ratio)
                c.dense_lift = self.model._qsa_dense_batch
                caches.append(c)
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
