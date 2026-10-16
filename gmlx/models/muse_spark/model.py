# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Asher Feldman
"""Model for Meta Muse Spark (GGUF arch ``muse-spark``).

This is the runtime for llama.cpp's ``LLM_ARCH_MUSE_SPARK`` conversions. The
attention comes from Muse Glimmer: an output gate, per-head QK-norm, RoPE on
the sliding-window layers only and NoPE on the global layers, an unweighted
norm on the token embeddings, and a logit scale with a tanh softcap. Spark
adds these parts:

  1. A weightless per-head RMSNorm on the attention output, before the gate.
  2. Weightless post-attention and post-FFN norms at ``post_norm_eps``.
  3. Norm-preserving residuals: ``x * alpha + branch * beta`` with per-channel
     alpha and beta, resolved at conversion.
  4. A latent MoE on every layer. The router reads the full-width FFN input
     (sigmoid, a selection-only bias, top-k, renormalized). The experts run
     SwiGLU at ``moe_latent_size`` between ``latent_down`` and ``latent_up``.
     Each expert output is scaled by ``post_exp_norm`` and normalized before
     the router weights apply. There is no shared expert.

The converter folds the +1 into the block norms, so every norm weight loads
as a plain gain. Q/K arrive in the interleaved layout (rope type NORM), so
rope runs ``traditional=True``.
"""

from dataclasses import dataclass
from typing import Any, List, Optional

import mlx.core as mx
import mlx.nn as nn

from gmlx._vendor.mlx_lm.models.base import BaseModelArgs
from gmlx._vendor.mlx_lm.models.rope_utils import initialize_rope
from gmlx._vendor.mlx_lm.models.switch_layers import SwitchGLU

from gmlx.cache.attention import (
    create_attention_mask,
    scaled_dot_product_attention,
)
from gmlx.cache.kv_cache import KVCache, RotatingKVCache
from gmlx.models.muse_glimmer.model import scale_and_softcap


# llama.cpp clamps the routing weight sum to the smallest normal f16.
_WEIGHT_SUM_FLOOR = 6.103515625e-5


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    layer_types: List[str]
    num_experts: int
    num_experts_per_tok: int
    moe_intermediate_size: int
    moe_latent_size: int
    sliding_window: int = 2048
    rms_norm_eps: float = 1e-5
    post_norm_eps: float = 1e-8
    rope_theta: float = 500000.0
    rope_parameters: Optional[dict] = None
    max_position_embeddings: int = 1048576
    output_multiplier: float = 1.0
    final_logit_softcapping: float = 0.0
    routed_scaling_factor: float = 1.0
    norm_topk_prob: bool = True
    tie_word_embeddings: bool = False


class ScaledRMSNorm(nn.Module):
    """``rms(x * weight)``: the gain applies before the normalization."""

    def __init__(self, dims: int, eps: float):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = float(eps)

    def __call__(self, x: mx.array) -> mx.array:
        return mx.fast.rms_norm(x * self.weight, None, self.eps)


class Attention(nn.Module):
    def __init__(self, args: ModelArgs, use_sliding: bool):
        super().__init__()
        dim = args.hidden_size
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.head_dim = args.head_dim
        self.use_sliding = use_sliding
        self.scale = self.head_dim**-0.5
        self.eps = float(args.rms_norm_eps)

        self.q_proj = nn.Linear(dim, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, dim, bias=False)
        self.gate_proj = nn.Linear(dim, self.n_heads * self.head_dim, bias=False)

        self.q_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)

        self.rope = (
            initialize_rope(
                self.head_dim,
                args.rope_theta,
                True,
                args.rope_parameters,
                args.max_position_embeddings,
            )
            if use_sliding
            else None
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, _ = x.shape

        queries = self.q_proj(x).reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = self.k_proj(x).reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = self.v_proj(x).reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        queries = self.q_norm(queries)
        keys = self.k_norm(keys)

        if self.rope is not None:
            offset = cache.offset if cache is not None else 0
            queries = self.rope(queries, offset=offset)
            keys = self.rope(keys, offset=offset)

        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )
        output = mx.fast.rms_norm(output, None, self.eps)
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        output = output * mx.sigmoid(self.gate_proj(x))
        return self.o_proj(output)


class MuseSparkMoE(nn.Module):
    """Latent MoE with a full-width sigmoid router and a selection-only bias.

    The expert container is called unmixed, so the per-expert norm runs on
    every expert output before the weighted sum.
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        dim, latent = args.hidden_size, args.moe_latent_size
        experts = args.num_experts
        self.num_experts_per_tok = int(args.num_experts_per_tok)
        self.norm_topk_prob = bool(args.norm_topk_prob)
        self.routed_scaling_factor = float(args.routed_scaling_factor)

        self.gate = nn.Linear(dim, experts, bias=False)
        self.e_score_correction_bias = mx.zeros((experts,), dtype=mx.float32)

        self.latent_down = nn.Linear(dim, latent, bias=False)
        self.pre_exp_norm = nn.RMSNorm(latent, eps=args.rms_norm_eps)
        self.switch_mlp = SwitchGLU(latent, args.moe_intermediate_size, experts)
        self.post_exp_norm = ScaledRMSNorm(latent, args.post_norm_eps)
        self.latent_up = nn.Linear(latent, dim, bias=False)
        self.post_ffw_norm = ScaledRMSNorm(dim, args.post_norm_eps)

    def __call__(self, x: mx.array) -> mx.array:
        scores = mx.sigmoid(self.gate(x.astype(mx.float32)))
        choice = scores + self.e_score_correction_bias
        k = self.num_experts_per_tok

        def weights_at(inds):
            w = mx.take_along_axis(scores, inds, axis=-1)
            if self.norm_topk_prob:
                w = w / mx.maximum(w.sum(axis=-1, keepdims=True), _WEIGHT_SUM_FLOOR)
            if self.routed_scaling_factor not in (0.0, 1.0):
                w = w * self.routed_scaling_factor
            return w.astype(x.dtype)

        inds = mx.stop_gradient(mx.argpartition(-choice, kth=k - 1, axis=-1)[..., :k])
        weights = weights_at(inds)

        from gmlx.stream.moe_experts import expert_controls_active

        if expert_controls_active(self):
            from gmlx.stream.moe_experts import _apply_expert_controls

            inds, weights = _apply_expert_controls(self, inds, weights, weights_at)

        if getattr(self.switch_mlp, "_kq_lookahead", None) is not None:
            # The router replica needs the full-width input. It runs only on
            # one-token steps, so a prefill chunk is not kept.
            la_x = x if x.size == x.shape[-1] else None
            object.__setattr__(self.switch_mlp, "_kq_la_input", la_x)
        z = self.pre_exp_norm(self.latent_down(x))
        y = self.post_exp_norm(self.switch_mlp(z, inds))
        y = (y * weights[..., None]).sum(axis=-2)
        return self.post_ffw_norm(self.latent_up(y))


class DecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, use_sliding: bool):
        super().__init__()
        dim = args.hidden_size
        self.use_sliding = use_sliding
        self.post_norm_eps = float(args.post_norm_eps)
        self.self_attn = Attention(args, use_sliding)
        self.mlp = MuseSparkMoE(args)

        self.input_layernorm = nn.RMSNorm(dim, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(dim, eps=args.rms_norm_eps)
        self.attn_resid_alpha = mx.ones((dim,), dtype=mx.float32)
        self.attn_resid_beta = mx.ones((dim,), dtype=mx.float32)
        self.ffn_resid_alpha = mx.ones((dim,), dtype=mx.float32)
        self.ffn_resid_beta = mx.ones((dim,), dtype=mx.float32)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        r = self.self_attn(self.input_layernorm(x), mask, cache)
        r = mx.fast.rms_norm(r, None, self.post_norm_eps)
        h = (x * self.attn_resid_alpha + r * self.attn_resid_beta).astype(x.dtype)
        r = self.mlp(self.post_attention_layernorm(h))
        return (h * self.ffn_resid_alpha + r * self.ffn_resid_beta).astype(x.dtype)


class MuseSparkModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.sliding_window = args.sliding_window
        self.layer_types = args.layer_types

        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            DecoderLayer(args, layer_type == "sliding_attention")
            for layer_type in args.layer_types
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

        self.fa_idx = next(
            (i for i, t in enumerate(args.layer_types) if t != "sliding_attention"),
            None,
        )
        self.swa_idx = next(
            (i for i, t in enumerate(args.layer_types) if t == "sliding_attention"),
            None,
        )

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
        inputs_embeds: Optional[mx.array] = None,
    ):
        h = self.embed_tokens(inputs) if inputs_embeds is None else inputs_embeds
        h = mx.fast.rms_norm(h, None, self.args.rms_norm_eps)

        if cache is None:
            cache = [None] * len(self.layers)

        fa_mask = swa_mask = None
        if self.fa_idx is not None:
            fa_mask = create_attention_mask(h, cache[self.fa_idx])
        if self.swa_idx is not None:
            swa_mask = create_attention_mask(
                h, cache[self.swa_idx], window_size=self.sliding_window
            )

        for layer, c in zip(self.layers, cache):
            h = layer(h, swa_mask if layer.use_sliding else fa_mask, cache=c)

        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = MuseSparkModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def head(self, h: mx.array) -> mx.array:
        if self.args.tie_word_embeddings:
            out = self.model.embed_tokens.as_linear(h)
        else:
            out = self.lm_head(h)
        return scale_and_softcap(
            out, self.args.output_multiplier, self.args.final_logit_softcapping
        )

    def __call__(self, inputs: mx.array, cache: Optional[Any] = None):
        return self.head(self.model(inputs, cache))

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        return [
            RotatingKVCache(max_size=self.model.sliding_window)
            if layer.use_sliding
            else KVCache()
            for layer in self.layers
        ]

    @property
    def cast_predicate(self):
        def predicate(path: str):
            if "e_score_correction_bias" in path:
                return False
            if path.endswith(".mlp.gate.weight"):
                return False
            if path.endswith(("_resid_alpha", "_resid_beta")):
                return False
            return True

        return predicate
