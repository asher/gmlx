# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Asher Feldman
"""Model for IFM K2 Horizon (GGUF arch ``k2-horizon``).

This is the runtime for llama.cpp's ``LLM_ARCH_K2_HORIZON`` conversions,
dense and MoVA. The attention is GQA with NeoX RoPE. These parts differ
from a Llama block:

  1. Every norm is a group RMSNorm: one RMS statistic per equal slice of
     the hidden width, then one full-width gain.
  2. An optional output gate scales the attention output before ``o_proj``
     by ``log2(1 + 2^g)``, with ``g`` projected from the normed input.
  3. An optional per-head Q/K RMSNorm runs before RoPE.
  4. MoVA layers route each token over value experts. V is the weighted
     sum of ``silu(W_k x)`` over the selected experts. The router shares
     the FFN router's gating, normalization and scale.
  5. MoE layers route the FFN over sigmoid-scored experts with a
     selection-only bias, plus one shared expert.
"""

import math
from dataclasses import dataclass
from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

from gmlx._vendor.mlx_lm.models.base import BaseModelArgs
from gmlx._vendor.mlx_lm.models.rope_utils import initialize_rope
from gmlx._vendor.mlx_lm.models.switch_layers import (
    SwitchGLU,
    SwitchLinear,
    _gather_sort,
    _scatter_unsort,
)

from gmlx.cache.attention import (
    create_attention_mask,
    scaled_dot_product_attention,
)
from gmlx.cache.kv_cache import KVCache


# llama.cpp clamps the routing weight sum to the smallest normal f16.
_WEIGHT_SUM_FLOOR = 6.103515625e-5
_LN2 = math.log(2.0)


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    hidden_size: int
    num_hidden_layers: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    rms_norm_eps: float = 1e-6
    norm_groups: int = 1
    rope_theta: float = 1e7
    rope_dim: Optional[int] = None
    rope_scaling: Optional[dict] = None
    max_position_embeddings: int = 524288
    attn_output_gate: bool = False
    qk_norm: bool = False
    num_experts: int = 0
    num_experts_per_tok: int = 0
    moe_intermediate_size: int = 0
    shared_expert_intermediate_size: int = 0
    first_k_dense_replace: int = 0
    num_value_experts: int = 0
    num_value_experts_per_tok: int = 0
    scoring_func: str = "sigmoid"
    norm_topk_prob: bool = False
    routed_scaling_factor: float = 1.0
    tie_word_embeddings: bool = False


def _route_weights(scores, inds, norm, scale):
    """Weights at ``inds`` from the unbiased scores, as llama.cpp computes
    them: renormalized over a clamped sum, then scaled unless the scale is
    0 or 1."""
    w = mx.take_along_axis(scores, inds, axis=-1)
    if norm:
        w = w / mx.maximum(w.sum(axis=-1, keepdims=True), _WEIGHT_SUM_FLOOR)
    if scale not in (0.0, 1.0):
        w = w * scale
    return w


def _route(logits, bias, top_k, scoring_func, norm, scale):
    """``(inds, weights)``: top-k on the scores plus ``bias``, weights from
    the unbiased scores."""
    logits = logits.astype(mx.float32)
    if scoring_func == "softmax":
        scores = mx.softmax(logits, axis=-1, precise=True)
    else:
        scores = mx.sigmoid(logits)
    choice = scores if bias is None else scores + bias
    inds = mx.stop_gradient(
        mx.argpartition(-choice, kth=top_k - 1, axis=-1)[..., :top_k])
    return inds, _route_weights(scores, inds, norm, scale)


class GroupRMSNorm(nn.Module):
    """RMS over ``groups`` equal slices of the last axis, then one
    full-width gain."""

    def __init__(self, dims: int, eps: float, groups: int = 1):
        super().__init__()
        if dims % groups:
            raise ValueError(f"width {dims} is not divisible by {groups} groups")
        self.weight = mx.ones((dims,))
        self.eps = float(eps)
        self.groups = int(groups)

    def __call__(self, x: mx.array) -> mx.array:
        if self.groups == 1:
            return mx.fast.rms_norm(x, self.weight, self.eps)
        shape = x.shape
        y = x.reshape(*shape[:-1], self.groups, shape[-1] // self.groups)
        y = mx.fast.rms_norm(y, None, self.eps).reshape(shape)
        return y * self.weight


class HeadRMSNorm(nn.Module):
    """RMSNorm over the head dim with a separate gain for every head. The
    gains load flat, as llama.cpp stores them."""

    def __init__(self, n_heads: int, head_dim: int, eps: float):
        super().__init__()
        self.weight = mx.ones((n_heads * head_dim,))
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.eps = float(eps)

    def __call__(self, x: mx.array) -> mx.array:
        # x: [B, L, H, D]
        w = self.weight.reshape(self.n_heads, self.head_dim)
        return mx.fast.rms_norm(x, None, self.eps) * w


class ValueExperts(nn.Module):
    """MoVA: V = sum_k w_k * silu(W_k x) over the routed value experts."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        dim = args.hidden_size
        out = args.num_key_value_heads * args.head_dim
        n = args.num_value_experts
        self.top_k = int(args.num_value_experts_per_tok)
        self.scoring_func = args.scoring_func
        self.norm_topk_prob = bool(args.norm_topk_prob)
        self.routed_scaling_factor = float(args.routed_scaling_factor)
        self.v_router = nn.Linear(dim, n, bias=False)
        # The wire's attn_v_gate.bias. llama.cpp adds it to the scores for
        # selection only, never to the logits.
        self.selection_bias = mx.zeros((n,), dtype=mx.float32)
        self.v_experts = SwitchLinear(dim, out, n, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        inds, weights = _route(
            self.v_router(x.astype(mx.float32)), self.selection_bias,
            self.top_k, self.scoring_func, self.norm_topk_prob,
            self.routed_scaling_factor)

        h = mx.expand_dims(x, (-2, -3))
        do_sort = inds.size >= 64
        idx, inv_order = inds, None
        if do_sort:
            h, idx, inv_order = _gather_sort(h, inds)
        v = self.v_experts(h, idx, sorted_indices=do_sort)
        if do_sort:
            v = _scatter_unsort(v, inv_order, inds.shape)
        v = nn.silu(v.squeeze(-2))
        return (v * weights[..., None].astype(v.dtype)).sum(axis=-2)


class Attention(nn.Module):
    def __init__(self, args: ModelArgs, mova: bool):
        super().__init__()
        dim = args.hidden_size
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.head_dim = args.head_dim
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(dim, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        if mova:
            self.value_experts = ValueExperts(args)
        else:
            self.v_proj = nn.Linear(
                dim, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, dim, bias=False)
        if args.attn_output_gate:
            self.gate_proj = nn.Linear(
                dim, self.n_heads * self.head_dim, bias=False)
        if args.qk_norm:
            self.q_norm = HeadRMSNorm(
                self.n_heads, self.head_dim, args.rms_norm_eps)
            self.k_norm = HeadRMSNorm(
                self.n_kv_heads, self.head_dim, args.rms_norm_eps)

        self.rope = initialize_rope(
            args.rope_dim or self.head_dim,
            args.rope_theta,
            False,
            args.rope_scaling,
            args.max_position_embeddings,
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, _ = x.shape

        queries = self.q_proj(x).reshape(B, L, self.n_heads, -1)
        keys = self.k_proj(x).reshape(B, L, self.n_kv_heads, -1)
        if "value_experts" in self:
            values = self.value_experts(x)
        else:
            values = self.v_proj(x)
        values = values.reshape(B, L, self.n_kv_heads, -1)

        if "q_norm" in self:
            queries = self.q_norm(queries)
            keys = self.k_norm(keys)

        queries = queries.transpose(0, 2, 1, 3)
        keys = keys.transpose(0, 2, 1, 3)
        values = values.transpose(0, 2, 1, 3)

        offset = cache.offset if cache is not None else 0
        queries = self.rope(queries, offset=offset)
        keys = self.rope(keys, offset=offset)
        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        if "gate_proj" in self:
            # log2(1 + 2^g), the softplus with beta ln 2.
            g = self.gate_proj(x).astype(mx.float32)
            gate = nn.softplus(g * _LN2) / _LN2
            output = output * gate.astype(output.dtype)
        return self.o_proj(output)


class MLP(nn.Module):
    """SwiGLU MLP: the dense layers and the shared expert."""

    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden, bias=False)
        self.up_proj = nn.Linear(dim, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class K2HorizonMoEGate(nn.Module):
    """FFN router: sigmoid scores, a selection-only correction bias, then
    renormalized and scaled weights.

    A gate submodule returning ``(inds, weights)`` with an int ``top_k``,
    the shape stream/moe_experts and stream/lookahead duck-type on.
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.top_k = int(args.num_experts_per_tok)
        self.scoring_func = args.scoring_func
        self.norm_topk_prob = bool(args.norm_topk_prob)
        self.routed_scaling_factor = float(args.routed_scaling_factor)
        # fp32 like the F32 wire tensors; llama.cpp routes in fp32.
        self.weight = mx.zeros(
            (args.num_experts, args.hidden_size), dtype=mx.float32)
        self.e_score_correction_bias = mx.zeros(
            (args.num_experts,), dtype=mx.float32)

    def _logits(self, x: mx.array) -> mx.array:
        return x.astype(mx.float32) @ self.weight.T.astype(mx.float32)

    def __call__(self, x: mx.array):
        return _route(
            self._logits(x), self.e_score_correction_bias, self.top_k,
            self.scoring_func, self.norm_topk_prob, self.routed_scaling_factor)

    def _kq_route_weights(self, x, inds):
        # Route replay (stream/moe_routes): the weight branch of _route at
        # caller-chosen ids.
        logits = self._logits(x)
        if self.scoring_func == "softmax":
            scores = mx.softmax(logits, axis=-1, precise=True)
        else:
            scores = mx.sigmoid(logits)
        return _route_weights(
            scores, inds, self.norm_topk_prob, self.routed_scaling_factor)


class K2HorizonMoE(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.gate = K2HorizonMoEGate(args)
        self.switch_mlp = SwitchGLU(
            args.hidden_size, args.moe_intermediate_size, args.num_experts)
        if args.shared_expert_intermediate_size:
            self.shared_experts = MLP(
                args.hidden_size, args.shared_expert_intermediate_size)

    def __call__(self, x: mx.array) -> mx.array:
        inds, scores = self.gate(x)
        if getattr(self.switch_mlp, "_kq_mix_scores", False):
            y = self.switch_mlp(x, inds, scores)
        else:
            y = self.switch_mlp(x, inds)
            if y.ndim == scores.ndim + 1:
                y = (y * scores[..., None].astype(y.dtype)).sum(-2)
        if "shared_experts" in self:
            y = y + self.shared_experts(x)
        return y


class DecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        dim = args.hidden_size
        moe = args.num_experts > 0 and layer_idx >= args.first_k_dense_replace
        mova = moe and args.num_value_experts > 0
        self.self_attn = Attention(args, mova)
        self.mlp = K2HorizonMoE(args) if moe else MLP(dim, args.intermediate_size)
        self.input_layernorm = GroupRMSNorm(
            dim, args.rms_norm_eps, args.norm_groups)
        self.post_attention_layernorm = GroupRMSNorm(
            dim, args.rms_norm_eps, args.norm_groups)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        h = x + self.self_attn(self.input_layernorm(x), mask, cache)
        return h + self.mlp(self.post_attention_layernorm(h))


class K2HorizonModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            DecoderLayer(args, i) for i in range(args.num_hidden_layers)]
        self.norm = GroupRMSNorm(
            args.hidden_size, args.rms_norm_eps, args.norm_groups)

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
        input_embeddings: Optional[mx.array] = None,
    ):
        h = self.embed_tokens(inputs) if input_embeddings is None \
            else input_embeddings
        if cache is None:
            cache = [None] * len(self.layers)
        mask = create_attention_mask(h, cache[0])
        for layer, c in zip(self.layers, cache):
            h = layer(h, mask, c)
        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = K2HorizonModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
        input_embeddings: Optional[mx.array] = None,
    ):
        out = self.model(inputs, cache, input_embeddings)
        if self.args.tie_word_embeddings:
            return self.model.embed_tokens.as_linear(out)
        return self.lm_head(out)

    def sanitize(self, weights):
        if self.args.tie_word_embeddings:
            weights.pop("lm_head.weight", None)
        return weights

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        return [KVCache() for _ in self.layers]

    @property
    def cast_predicate(self):
        def predicate(path: str):
            if path.endswith((".mlp.gate.weight", ".e_score_correction_bias",
                              ".selection_bias")):
                return False
            return True

        return predicate
