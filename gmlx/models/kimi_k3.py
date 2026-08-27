# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Asher Feldman
# Portions copyright (c) 2026 Apple Inc. (mlx-lm kimi_linear skeleton, MIT)
"""Vendored mlx-lm-style model for Moonshot Kimi-K3 (GGUF arch ``kimi-k3``).

mlx-lm has no kimi_k3 class; this module is the runtime for llama.cpp PR
26185 / unsloth PR 48 conversions, built from the pinned mlx-lm 0.31.3
``kimi_linear`` skeleton (KDA + nope-only MLA + sigmoid MoE) plus the five
K3-only mechanisms:

  1. cross-layer attention residuals (``attn_res_block_size``): the residual
     stream is banked every N layers; before attention and before the FFN the
     current stream is replaced by a softmax-weighted convex mix of all banked
     checkpoints + itself (scores = rms(x) . score vector, mix over RAW
     values). On bank layers the residual RESTARTS from the attention output.
  2. latent MoE (``routed_expert_hidden_size``): routed experts run behind
     down/up projections at latent width; the router reads full width.
  3. situ activation everywhere (replaces SwiGLU).
  4. MLA sigmoid output gate before o_proj.
  5. full-rank KDA gate (single ``g_proj`` instead of g_a/g_b).

The KDA recurrence rides mlx-lm's ``gated_delta`` kernel/ops, which already
support per-key-channel decay (g: [B, T, H, Dk]); only the decay activation
differs: K3 uses ``exp(lb * sigmoid(exp(A_log) * (f(x) + dt_bias)))`` when
``kda_gate_lower_bound`` is set (the GGUF ``ssm_a`` tensor stores the folded
``-exp(A_log)``, kept folded here as ``a_folded``).

``GMLX_KIMI_ATTNRES=0`` disables the residual mixing for A/B debugging (the
score-vector params still exist and load).
"""

import importlib
import os
import sys
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Dict, List, Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.base import (
    BaseModelArgs,
    create_attention_mask,
    create_ssm_mask,
    scaled_dot_product_attention,
)
from mlx_lm.models.cache import ArraysCache, KVCache
from mlx_lm.models.gated_delta import gated_delta_kernel, gated_delta_ops
from mlx_lm.models.mla import MultiLinear
from mlx_lm.models.switch_layers import SwitchGLU


_MOE_MIX_SCORES = os.environ.get("GMLX_K3_MOE_MIX", "1") != "0"
_ATTNRES = os.environ.get("GMLX_KIMI_ATTNRES", "1") != "0"


def ensure_registered() -> None:
    """Make ``import mlx_lm.models.kimi_k3`` resolve, preferring upstream."""
    if "mlx_lm.models.kimi_k3" not in sys.modules:
        try:
            importlib.import_module("mlx_lm.models.kimi_k3")  # upstream wins
        except ImportError:
            sys.modules["mlx_lm.models.kimi_k3"] = sys.modules[__name__]


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    vocab_size: int
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    intermediate_size: int
    rms_norm_eps: float
    # per-layer schedule: "linear_attention" (KDA) | "full_attention" (MLA)
    layer_types: List[str] = field(default_factory=list)
    # KDA
    kda_head_dim: int = 128
    ssm_conv_kernel: int = 4
    # None = kimi-linear softplus decay; set (K3: -5.0) = sigmoid gate form.
    kda_gate_lower_bound: Optional[float] = None
    # MLA (nope-only: no rope fields at all)
    q_lora_rank: Optional[int] = None
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    # MoE
    num_experts: int = 0
    num_experts_per_tok: int = 1
    moe_intermediate_size: int = 0
    num_shared_experts: int = 0
    first_k_dense_replace: int = 0
    routed_scaling_factor: float = 1.0
    moe_renormalize: bool = True
    routed_expert_hidden_size: Optional[int] = None
    has_routed_norm: bool = False
    # situ activation
    situ_beta: float = 4.0
    situ_linear_beta: float = 25.0
    # cross-layer attention residuals (0 disables)
    attn_res_block_size: int = 0
    max_position_embeddings: int = 1048576
    tie_word_embeddings: bool = False


@partial(mx.compile, shapeless=True)
def _situ(gate: mx.array, up: mx.array, beta: float, lb: float) -> mx.array:
    # situ(gate, up) = [beta*tanh(gate/beta)*sigmoid(gate)] * [lb*tanh(up/lb)]
    # lb <= 0 leaves the up branch untransformed.
    a = beta * mx.tanh(gate / beta) * mx.sigmoid(gate)
    if lb > 0.0:
        up = lb * mx.tanh(up / lb)
    return a * up


class SituActivation(nn.Module):
    """SwitchGLU-compatible situ: called as ``activation(x_up, x_gate)``."""

    def __init__(self, beta: float, linear_beta: float):
        super().__init__()
        self._beta = beta
        self._lb = linear_beta

    def __call__(self, x_up: mx.array, x_gate: mx.array) -> mx.array:
        return _situ(x_gate, x_up, self._beta, self._lb)


class KimiK3MLP(nn.Module):
    """Dense situ MLP (layer-0 dense FFN and the shared experts)."""

    def __init__(self, args: ModelArgs, intermediate_size: Optional[int] = None):
        super().__init__()
        dim = args.hidden_size
        hidden = intermediate_size or args.intermediate_size
        self.gate_proj = nn.Linear(dim, hidden, bias=False)
        self.up_proj = nn.Linear(dim, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, dim, bias=False)
        self._beta = args.situ_beta
        self._lb = args.situ_linear_beta

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(
            _situ(self.gate_proj(x), self.up_proj(x), self._beta, self._lb))


class KimiK3MoE(nn.Module):
    """Latent MoE: router at full width, routed experts at latent width.

    Reference ordering (llama.cpp build_latent_moe): routed_down -> experts
    (situ) -> weighted sum AT LATENT WIDTH -> routed_norm -> routed_up ->
    + shared experts (full width, unprojected input).
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        hidden = args.hidden_size
        experts = args.num_experts
        latent = args.routed_expert_hidden_size or hidden

        self.gate = nn.Linear(hidden, experts, bias=False)
        self.e_score_correction_bias = mx.zeros((experts,), dtype=mx.float32)

        if args.routed_expert_hidden_size:
            self.routed_down = nn.Linear(hidden, latent, bias=False)
            self.routed_up = nn.Linear(latent, hidden, bias=False)
        else:
            self.routed_down = None
            self.routed_up = None
        self.routed_norm = (nn.RMSNorm(latent, eps=args.rms_norm_eps)
                            if args.has_routed_norm else None)

        self.switch_mlp = SwitchGLU(
            latent, args.moe_intermediate_size, experts,
            activation=SituActivation(args.situ_beta, args.situ_linear_beta))

        if args.num_shared_experts:
            self.shared_experts = KimiK3MLP(
                args, args.moe_intermediate_size * args.num_shared_experts)
        else:
            self.shared_experts = None

    def __call__(self, x: mx.array) -> mx.array:
        # Router: full-width input, fp32 logits (sigmoid top-16-of-896 with a
        # correction bias is near-tie-heavy).
        scores = mx.sigmoid(self.gate(x.astype(mx.float32)))
        orig_scores = scores
        scores = scores + self.e_score_correction_bias

        k = self.args.num_experts_per_tok
        inds = mx.argpartition(-scores, kth=k - 1, axis=-1)[..., :k]
        weights = mx.take_along_axis(orig_scores, inds, axis=-1)
        if k > 1 and self.args.moe_renormalize:
            weights = weights / (mx.sum(weights, axis=-1, keepdims=True) + 1e-20)
        weights = (weights * self.args.routed_scaling_factor).astype(x.dtype)

        # Expert-controls seam (probe / expert-mass): moe_experts targets
        # this block directly rather than swapping the forward.
        if (getattr(self, "_kq_expert_probe", None) is not None
                or getattr(self, "_kq_expert_mass", None) is not None):
            from gmlx.stream.moe_experts import _apply_expert_controls

            inds, weights = _apply_expert_controls(self, inds, weights)

        if getattr(self.switch_mlp, "_kq_lookahead", None) is not None:
            # Latent MoE: the wrapped expert container sees routed_down's
            # latent-width output, but the router replica needs the
            # full-width block input - hand it over out of band.
            object.__setattr__(self.switch_mlp, "_kq_la_input", x)
        y_in = self.routed_down(x) if self.routed_down is not None else x
        # Mix seam: hand the routing weights to the swapped kquant module when
        # it accepts them; an unmixed return keeps the python-side sum here.
        if _MOE_MIX_SCORES and (
                getattr(self.switch_mlp, "_kq_mix_scores", False)
                or getattr(self.switch_mlp, "_kq_scores_sink", False)):
            y = self.switch_mlp(y_in, inds, weights)
        else:
            y = self.switch_mlp(y_in, inds)
        if y.ndim == weights.ndim + 1:
            y = (y * weights[..., None]).sum(axis=-2)

        if self.routed_norm is not None:
            y = self.routed_norm(y)
        if self.routed_up is not None:
            y = self.routed_up(y)
        if self.shared_experts is not None:
            y = y + self.shared_experts(x)
        return y


class KimiK3MLAAttention(nn.Module):
    """Nope-only MLA (no rope anywhere) with K3's sigmoid output gate.

    Module names follow mlx-lm deepseek_v3/kimi_linear so the DEEPSEEK2-style
    remap rows and KQuantMultiLinear (embed_q/unembed_out) engage unchanged.
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.num_heads = args.num_attention_heads
        self.qk_nope_head_dim = args.qk_nope_head_dim
        self.qk_rope_head_dim = args.qk_rope_head_dim
        self.q_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.v_head_dim = args.v_head_dim
        self.kv_lora_rank = args.kv_lora_rank
        self.scale = self.q_head_dim**-0.5

        hidden = args.hidden_size
        if args.q_lora_rank:
            self.q_a_proj = nn.Linear(hidden, args.q_lora_rank, bias=False)
            self.q_a_layernorm = nn.RMSNorm(args.q_lora_rank,
                                            eps=args.rms_norm_eps)
            self.q_b_proj = nn.Linear(
                args.q_lora_rank, self.num_heads * self.q_head_dim, bias=False)
        else:
            self.q_proj = nn.Linear(
                hidden, self.num_heads * self.q_head_dim, bias=False)

        self.kv_a_proj_with_mqa = nn.Linear(
            hidden, args.kv_lora_rank + self.qk_rope_head_dim, bias=False)
        self.kv_a_layernorm = nn.RMSNorm(args.kv_lora_rank,
                                         eps=args.rms_norm_eps)
        self.embed_q = MultiLinear(
            self.qk_nope_head_dim, args.kv_lora_rank, self.num_heads)
        self.unembed_out = MultiLinear(
            args.kv_lora_rank, self.v_head_dim, self.num_heads)
        # K3: sigmoid output gate on the attention output, before o_proj.
        # Reads the same (normed, post-res-mix) input the projections read.
        self.attn_gate = nn.Linear(
            hidden, self.num_heads * self.v_head_dim, bias=False)
        self.o_proj = nn.Linear(
            self.num_heads * self.v_head_dim, hidden, bias=False)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[KVCache] = None,
    ) -> mx.array:
        B, L, _ = x.shape

        if hasattr(self, "q_a_proj"):
            q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(x)))
        else:
            q = self.q_proj(x)
        q = q.reshape(B, L, self.num_heads, self.q_head_dim).transpose(0, 2, 1, 3)
        q_nope, q_pe = mx.split(q, [self.qk_nope_head_dim], axis=-1)

        compressed_kv = self.kv_a_proj_with_mqa(x)
        compressed_kv, k_pe = mx.split(compressed_kv, [self.kv_lora_rank], axis=-1)
        # nope-only: k_pe joins the score path un-rotated.
        k_pe = k_pe.reshape(B, L, 1, self.qk_rope_head_dim).transpose(0, 2, 1, 3)
        kv_latent = mx.expand_dims(self.kv_a_layernorm(compressed_kv), axis=1)

        if cache is not None:
            kv_latent, k_pe = cache.update_and_fetch(kv_latent, k_pe)

        pe_scores = (q_pe * self.scale) @ k_pe.swapaxes(-1, -2)
        if mask is not None:
            pe_scores = mx.where(
                mask, pe_scores,
                mx.array(mx.finfo(pe_scores.dtype).min, pe_scores.dtype))

        if L == 1:
            q_nope = self.embed_q(q_nope)
            k = v = kv_latent
        else:
            k = self.embed_q(kv_latent, transpose=False)
            v = self.unembed_out(kv_latent)

        out = scaled_dot_product_attention(
            q_nope, k, v, cache=cache, scale=self.scale, mask=pe_scores)

        if L == 1:
            out = self.unembed_out(out)

        out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)
        out = out * mx.sigmoid(self.attn_gate(x))
        return self.o_proj(out)


class ShortConv1d(nn.Module):
    """Depthwise causal short conv with carried state (from kimi_linear)."""

    def __init__(self, channels: int, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(
            in_channels=channels, out_channels=channels,
            kernel_size=kernel_size, bias=False, groups=channels, padding=0)

    def __call__(self, x, state, mask, lengths):
        if mask is not None:
            x = mx.where(mask[..., None], x, 0)
        if state is None:
            state = mx.zeros(
                (x.shape[0], self.kernel_size - 1, x.shape[-1]), dtype=x.dtype)
        conv_input = mx.concatenate([state, x], axis=1)
        out = nn.silu(self.conv(conv_input))
        n_keep = self.kernel_size - 1
        if lengths is not None:
            ends = mx.clip(lengths, 0, x.shape[1])
            positions = (ends[:, None] + mx.arange(n_keep))[..., None]
            new_state = mx.take_along_axis(conv_input, positions, axis=1)
        else:
            new_state = mx.contiguous(conv_input[:, -n_keep:, :])
        return out, new_state


@partial(mx.compile, shapeless=True)
def _kda_decay_lb(a_folded, a_raw, dt_bias, lb):
    # K3 gate form: decay = exp(lb * sigmoid(exp(A_log) * (a + dt_bias)));
    # a_folded stores -exp(A_log), so exp(A_log) = -a_folded. All fp32.
    a = a_raw.astype(mx.float32) + dt_bias
    return mx.exp(lb * mx.sigmoid((-a_folded)[..., None] * a))


@partial(mx.compile, shapeless=True)
def _kda_decay_softplus(a_folded, a_raw, dt_bias):
    # kimi-linear form: decay = exp(-exp(A_log) * softplus(a + dt_bias)).
    a = a_raw.astype(mx.float32) + dt_bias
    return mx.exp(a_folded[..., None] * nn.softplus(a))


class KimiK3DeltaAttention(nn.Module):
    """KDA with K3's full-rank gate; recurrence via mlx-lm gated_delta."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.num_heads = args.num_attention_heads
        self.head_dim = args.kda_head_dim
        self.conv_kernel = args.ssm_conv_kernel
        self.gate_lower_bound = args.kda_gate_lower_bound
        self.projection_dim = self.num_heads * self.head_dim
        self.scale = float(self.head_dim) ** -0.5
        hidden = args.hidden_size

        self.q_proj = nn.Linear(hidden, self.projection_dim, bias=False)
        self.k_proj = nn.Linear(hidden, self.projection_dim, bias=False)
        self.v_proj = nn.Linear(hidden, self.projection_dim, bias=False)
        self.q_conv = ShortConv1d(self.projection_dim, self.conv_kernel)
        self.k_conv = ShortConv1d(self.projection_dim, self.conv_kernel)
        self.v_conv = ShortConv1d(self.projection_dim, self.conv_kernel)

        self.f_a_proj = nn.Linear(hidden, self.head_dim, bias=False)
        self.f_b_proj = nn.Linear(self.head_dim, self.projection_dim, bias=False)
        self.b_proj = nn.Linear(hidden, self.num_heads, bias=False)
        # Wire ssm_a: the folded -exp(A_log), [num_heads], fp32. Kept folded.
        self.a_folded = -mx.ones((self.num_heads,))
        self.dt_bias = mx.zeros((self.projection_dim,))
        # K3: single full-rank gate (kimi-linear factors this as g_b(g_a(x))).
        self.g_proj = nn.Linear(hidden, self.projection_dim, bias=False)
        self.o_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.o_proj = nn.Linear(self.projection_dim, hidden, bias=False)

        # The metal kernel needs Dk % 32 == 0; fall back to ops otherwise.
        self._can_kernel = (self.head_dim % 32 == 0) and mx.metal.is_available()

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, T, _ = x.shape
        dtype = x.dtype

        if cache is not None:
            q_state, k_state, v_state, ssm_state = cache
            lengths = cache.lengths
        else:
            q_state = k_state = v_state = ssm_state = None
            lengths = None

        q_conv, q_state = self.q_conv(self.q_proj(x), q_state, mask, lengths)
        k_conv, k_state = self.k_conv(self.k_proj(x), k_state, mask, lengths)
        v_conv, v_state = self.v_conv(self.v_proj(x), v_state, mask, lengths)
        if cache is not None:
            cache[0] = q_state
            cache[1] = k_state
            cache[2] = v_state

        q = q_conv.reshape(B, T, self.num_heads, self.head_dim)
        k = k_conv.reshape(B, T, self.num_heads, self.head_dim)
        v = v_conv.reshape(B, T, self.num_heads, self.head_dim)

        # l2-norm with the attention scale folded in (kimi_linear convention:
        # l2norm(x) = rms_norm(x)/sqrt(d), q additionally carries 1/sqrt(d)).
        q = (self.scale**2) * mx.fast.rms_norm(q, None, 1e-6)
        k = self.scale * mx.fast.rms_norm(k, None, 1e-6)

        a_raw = self.f_b_proj(self.f_a_proj(x)).reshape(
            B, T, self.num_heads, self.head_dim)
        dt = self.dt_bias.reshape(self.num_heads, self.head_dim)
        if self.gate_lower_bound is not None:
            g = _kda_decay_lb(self.a_folded, a_raw, dt, self.gate_lower_bound)
        else:
            g = _kda_decay_softplus(self.a_folded, a_raw, dt)
        beta = mx.sigmoid(self.b_proj(x).reshape(B, T, self.num_heads))

        if ssm_state is None:
            ssm_state = mx.zeros(
                (B, self.num_heads, self.head_dim, self.head_dim),
                dtype=mx.float32)

        if self._can_kernel and mx.default_device() == mx.gpu and not self.training:
            out, ssm_state = gated_delta_kernel(q, k, v, g, beta, ssm_state, mask)
        else:
            out, ssm_state = gated_delta_ops(q, k, v, g, beta, ssm_state, mask)

        if cache is not None:
            cache[3] = ssm_state
            cache.advance(T)

        gate = self.g_proj(x).reshape(B, T, self.num_heads, self.head_dim)
        out = (self.o_norm(out.reshape(B, T, self.num_heads, self.head_dim))
               * mx.sigmoid(gate)).reshape(B, T, -1)
        return self.o_proj(out.astype(dtype))


class _ResidualMixer:
    """Per-forward cross-layer residual state (no params, never cached).

    Banks the RAW residual stream at bank layers plus a weightless-rms-normed
    fp32 copy (scores read normed values, the mix reads raw ones). ``mix``
    softmaxes [banked..., current] scores in fp32 and returns the convex
    combination. Checkpoint memory: C x B x T x hidden per forward chunk.
    """

    def __init__(self, eps: float, enabled: bool):
        self.eps = eps
        self.enabled = enabled
        self.raw: List[mx.array] = []
        self.normed: List[mx.array] = []

    def _rms(self, x: mx.array) -> mx.array:
        x32 = x.astype(mx.float32)
        return x32 * mx.rsqrt(x32.square().mean(-1, keepdims=True) + self.eps)

    def push(self, x: mx.array) -> None:
        if not self.enabled:
            return
        self.raw.append(x)
        self.normed.append(self._rms(x))

    def mix(self, cur: mx.array, score_w: mx.array) -> mx.array:
        if not self.enabled or not self.raw:
            return cur
        w = score_w.astype(mx.float32)
        s_bank = mx.stack([n @ w for n in self.normed], axis=-1)  # [B,T,C]
        s_cur = self._rms(cur) @ w                                # [B,T]
        p = mx.softmax(
            mx.concatenate([s_bank, s_cur[..., None]], axis=-1),
            axis=-1, precise=True)
        c = len(self.raw)
        out = cur * p[..., c:].astype(cur.dtype)
        for i, r in enumerate(self.raw):
            out = out + r * p[..., i:i + 1].astype(cur.dtype)
        return out


class KimiK3DecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.is_linear = args.layer_types[layer_idx] == "linear_attention"
        res_bs = args.attn_res_block_size
        self.banked = res_bs > 0 and layer_idx % res_bs == 0

        if self.is_linear:
            self.self_attn = KimiK3DeltaAttention(args)
        else:
            self.self_attn = KimiK3MLAAttention(args)

        if layer_idx < args.first_k_dense_replace or args.num_experts == 0:
            self.mlp = KimiK3MLP(args)
        else:
            self.mlp = KimiK3MoE(args)

        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps)
        if res_bs > 0:
            self.attn_res_score = mx.zeros((args.hidden_size,))
            self.ffn_res_score = mx.zeros((args.hidden_size,))

    def __call__(
        self,
        x: mx.array,
        mixer: _ResidualMixer,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        # Reference flow (llama.cpp kimi-k3.cpp): mix BEFORE banking; bank the
        # RAW layer input; the residual stream RESTARTS from the attention
        # output on bank layers (the banked input re-enters via the mix).
        has_res = self.banked or hasattr(self, "attn_res_score")
        cur = mixer.mix(x, self.attn_res_score) if has_res else x
        if self.banked:
            mixer.push(x)
        a = self.self_attn(self.input_layernorm(cur), mask, cache)
        h = a if (self.banked and mixer.enabled) else x + a
        cur = mixer.mix(h, self.ffn_res_score) if has_res else h
        return h + self.mlp(self.post_attention_layernorm(cur))


class KimiK3Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [KimiK3DecoderLayer(args, i)
                       for i in range(args.num_hidden_layers)]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        if args.attn_res_block_size > 0:
            self.output_res_score = mx.zeros((args.hidden_size,))

        types = args.layer_types
        self.ssm_idx = types.index("linear_attention")
        self.attn_idx = types.index("full_attention")

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[List[Any]] = None,
    ) -> mx.array:
        h = self.embed_tokens(inputs)
        if cache is None:
            cache = [None] * len(self.layers)

        ssm_mask = create_ssm_mask(h, cache[self.ssm_idx])
        attn_mask = create_attention_mask(
            h, cache[self.attn_idx], return_array=True)

        mixer = _ResidualMixer(
            self.args.rms_norm_eps,
            enabled=self.args.attn_res_block_size > 0 and _ATTNRES)
        for layer, layer_cache in zip(self.layers, cache):
            mask = ssm_mask if layer.is_linear else attn_mask
            h = layer(h, mixer, mask=mask, cache=layer_cache)

        score = getattr(self, "output_res_score", None)
        if score is not None:
            h = mixer.mix(h, score)
        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = KimiK3Model(args)
        if args.tie_word_embeddings:
            self.lm_head = None
        else:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size,
                                     bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[List[Any]] = None,
    ) -> mx.array:
        out = self.model(inputs, cache)
        if self.lm_head is None:
            return self.model.embed_tokens.as_linear(out)
        return self.lm_head(out)

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        # KDA layers: slots 0..2 = q/k/v conv states, 3 = ssm state (fp32).
        # Heterogeneous list => the hybrid cache is non-trimmable and the
        # server falls back to full re-prefill on partial prefix reuse
        # (qwen3.5 precedent). Snapshots round-trip on the upstream classes.
        return [ArraysCache(size=4) if layer.is_linear else KVCache()
                for layer in self.layers]

    def sanitize(self, weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        # GGUF loads arrive pre-remapped; this only covers HF-style leftovers:
        # vision tower / projector (text-only), MTP-less trunk, tied head.
        weights = {k: v for k, v in weights.items()
                   if not k.startswith(("vision_tower.", "mm_projector.",
                                        "model.mtp"))}
        if self.args.tie_word_embeddings:
            weights.pop("lm_head.weight", None)
        return weights

    @property
    def cast_predicate(self):
        def predicate(path: str):
            if "e_score_correction_bias" in path:
                return False
            if path.endswith(("a_folded", "dt_bias")):
                return False
            if path.endswith(("attn_res_score", "ffn_res_score",
                              "output_res_score")):
                return False
            return True

        return predicate

    @property
    def quant_predicate(self):
        def predicate(path, _):
            if path.endswith("mlp.gate"):
                return {"group_size": 64, "bits": 8}
            return True

        return predicate
