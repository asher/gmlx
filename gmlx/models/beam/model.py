# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Asher Feldman
"""Model for Reflection Beam-1.0 (GGUF arch ``beam``).

This is the runtime for llama.cpp's ``LLM_ARCH_BEAM`` conversions. The
attention is GQA with a per-head Q/K RMSNorm. These parts differ from a
Llama block:

  1. A sigmoid output gate scales the attention output before ``o_proj``.
     The gate is projected from the normed input over every head, plus one
     bias per head.
  2. The sliding-window layers run NeoX RoPE. The full layers are NoPE and
     scale q by ``(1 + (pos + 1) / length) ** scale``.
  3. Sandwich norms: the attention and FFN outputs are normed again before
     the residual add.
  4. Depth-scaled residuals: ``branch / sqrt(i) + x * sqrt((i - 1) / i)``,
     with ``i = 2 (l + 1)`` after attention and ``2 (l + 1) + 1`` after the
     FFN.
  5. The layers after the leading dense ones route a sigmoid MoE with a
     selection-only bias and grouped selection: only the groups with the
     best sum of their top-2 biased scores compete for the top-k. There is
     no shared expert.
  6. A tanh softcap on the logits.
"""

import math
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
from gmlx.load.dtypes import (
    NORM_WEIGHT_KEY,
    activation_dtype,
    fp32_norms,
    wider_float,
)
from gmlx.models.muse_glimmer.model import scale_and_softcap


# llama.cpp clamps the routing weight sum to the smallest normal f16.
_WEIGHT_SUM_FLOOR = 6.103515625e-5


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
    layer_types: List[str]
    num_experts: int
    num_experts_per_tok: int
    moe_intermediate_size: int
    first_k_dense_replace: int = 0
    n_group: int = 1
    topk_group: int = 1
    sliding_window: int = 512
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    rope_parameters: Optional[dict] = None
    max_position_embeddings: int = 1048576
    attn_temperature_scale: float = 0.1
    attn_temperature_length: int = 2048
    routed_scaling_factor: float = 1.0
    norm_topk_prob: bool = True
    final_logit_softcapping: float = 0.0
    tie_word_embeddings: bool = False


def nope_query_scale(offset, length: int, temp_scale: float,
                     temp_length: int) -> mx.array:
    """The NoPE layers' query scale ``(1 + (pos + 1) / temp_length) **
    temp_scale`` for ``length`` tokens after ``offset``, in fp32.

    ``offset`` is the cache offset: an int, or one int per row on a batch
    cache, where a left-padded row starts below zero. Positions clamp at 0,
    since the power form has no real value below ``-temp_length - 1``.
    Shaped ``[L, 1]``, or ``[B, 1, L, 1]`` for per-row offsets, to multiply
    ``[B, H, L, D]`` queries.
    """
    steps = mx.arange(length)
    per_row = isinstance(offset, mx.array) and offset.ndim > 0
    pos = offset[:, None] + steps if per_row else offset + steps
    pos = mx.maximum(pos, 0).astype(mx.float32)
    scale = mx.exp(temp_scale * mx.log1p((pos + 1.0) / temp_length))
    return scale[:, None, :, None] if per_row else scale[:, None]


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


@mx.compile
def _group_select(scores, bias, top_k, n_group, topk_group):
    """Top-k ids over the biased scores, competing only within the
    ``topk_group`` groups whose top-2 biased scores sum highest. The other
    groups are masked to -inf, as llama.cpp does, so a negative bias can
    never pull in an expert from a dropped group."""
    choice = scores + bias
    if n_group > 1:
        shp = choice.shape
        grouped = choice.reshape(*shp[:-1], n_group, shp[-1] // n_group)
        group_scores = mx.topk(grouped, 2, axis=-1).sum(axis=-1, keepdims=True)
        drop = mx.argpartition(
            group_scores, kth=n_group - topk_group - 1, axis=-2,
        )[..., : n_group - topk_group, :]
        grouped = mx.put_along_axis(
            grouped, mx.stop_gradient(drop), mx.array(-mx.inf, grouped.dtype),
            axis=-2)
        choice = grouped.reshape(shp)
    return mx.stop_gradient(
        mx.argpartition(-choice, kth=top_k - 1, axis=-1)[..., :top_k])


class BeamGate(nn.Module):
    """FFN router: sigmoid scores, a selection-only correction bias and
    grouped selection, then renormalized and scaled weights.

    A gate submodule returning ``(inds, weights)`` with an int ``top_k``,
    the shape stream/moe_experts and stream/lookahead duck-type on.
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.top_k = int(args.num_experts_per_tok)
        self.n_group = int(args.n_group)
        self.topk_group = int(args.topk_group)
        self.norm_topk_prob = bool(args.norm_topk_prob)
        self.routed_scaling_factor = float(args.routed_scaling_factor)
        # fp32 like the F32 wire tensors; llama.cpp routes in fp32.
        self.weight = mx.zeros(
            (args.num_experts, args.hidden_size), dtype=mx.float32)
        self.e_score_correction_bias = mx.zeros(
            (args.num_experts,), dtype=mx.float32)

    def _scores(self, x: mx.array) -> mx.array:
        return mx.sigmoid(
            x.astype(mx.float32) @ self.weight.T.astype(mx.float32))

    def __call__(self, x: mx.array):
        scores = self._scores(x)
        inds = _group_select(
            scores, self.e_score_correction_bias, self.top_k, self.n_group,
            self.topk_group)
        return inds, _route_weights(
            scores, inds, self.norm_topk_prob, self.routed_scaling_factor)

    def _kq_route_weights(self, x, inds):
        # Route replay (stream/moe_routes): the weight branch at
        # caller-chosen ids.
        return _route_weights(
            self._scores(x), inds, self.norm_topk_prob,
            self.routed_scaling_factor)


class BeamMoE(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.gate = BeamGate(args)
        self.switch_mlp = SwitchGLU(
            args.hidden_size, args.moe_intermediate_size, args.num_experts)

    def __call__(self, x: mx.array) -> mx.array:
        inds, scores = self.gate(x)
        scores = scores.astype(x.dtype)
        # Hand the routing weights to a swapped expert module that mixes
        # them itself or feeds them to the streaming miss-shed.
        if (getattr(self.switch_mlp, "_kq_mix_scores", False)
                or getattr(self.switch_mlp, "_kq_scores_sink", False)):
            y = self.switch_mlp(x, inds, scores)
        else:
            y = self.switch_mlp(x, inds)
        if y.ndim == scores.ndim + 1:
            y = (y * scores[..., None]).sum(axis=-2)
        return y


class BeamMLP(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden, bias=False)
        self.up_proj = nn.Linear(dim, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class _Norm(nn.RMSNorm):
    """RMSNorm whose weight the loader keeps float32. The output takes the
    wider of the input and activation dtypes, as a weight in the activation
    dtype gives, so the float32 weight never widens a bf16 stream."""

    OUTPUT_IN_INPUT_DTYPE = True

    def __init__(self, dims: int, eps: float = 1e-5):
        super().__init__(dims, eps=eps)
        self._act = activation_dtype("beam")

    def __call__(self, x):
        out = wider_float(x.dtype, self._act)
        return mx.fast.rms_norm(x, self.weight, self.eps).astype(out)


class Attention(nn.Module):
    def __init__(self, args: ModelArgs, use_sliding: bool):
        super().__init__()
        dim = args.hidden_size
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.head_dim = args.head_dim
        self.use_sliding = use_sliding
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(dim, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, dim, bias=False)
        self.gate_proj = nn.Linear(dim, self.n_heads * self.head_dim, bias=False)
        # One bias per head, fp32 like the F32 wire tensor.
        self.gate_bias = mx.zeros((self.n_heads,), dtype=mx.float32)

        self.q_norm = _Norm(self.head_dim, eps=args.rms_norm_eps)
        self.k_norm = _Norm(self.head_dim, eps=args.rms_norm_eps)

        self.rope = (
            initialize_rope(
                self.head_dim,
                args.rope_theta,
                False,
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
        q_scale: Optional[mx.array] = None,
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
        elif q_scale is not None:
            queries = queries * q_scale.astype(queries.dtype)

        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )
        gate = self.gate_proj(x).reshape(B, L, self.n_heads, -1)
        gate = mx.sigmoid(gate + self.gate_bias[:, None].astype(gate.dtype))
        output = output.transpose(0, 2, 1, 3) * gate
        return self.o_proj(output.reshape(B, L, -1))


class DecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        dim = args.hidden_size
        self.use_sliding = args.layer_types[layer_idx] == "sliding_attention"
        self.self_attn = Attention(args, self.use_sliding)
        if layer_idx < args.first_k_dense_replace:
            self.mlp = BeamMLP(dim, args.intermediate_size)
        else:
            self.mlp = BeamMoE(args)
        self.input_layernorm = _Norm(dim, eps=args.rms_norm_eps)
        self.post_attn_norm = _Norm(dim, eps=args.rms_norm_eps)
        # The FFN pre-norm, under the name stream/lookahead reads.
        self.post_attention_layernorm = _Norm(dim, eps=args.rms_norm_eps)
        self.post_ffw_norm = _Norm(dim, eps=args.rms_norm_eps)

        i = 2 * (layer_idx + 1)
        self.attn_scales = (1.0 / math.sqrt(i), math.sqrt((i - 1) / i))
        self.ffn_scales = (1.0 / math.sqrt(i + 1), math.sqrt(i / (i + 1)))

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        q_scale: Optional[mx.array] = None,
    ) -> mx.array:
        r = self.self_attn(self.input_layernorm(x), mask, cache, q_scale)
        b, s = self.attn_scales
        h = self.post_attn_norm(r) * b + x * s
        r = self.mlp(self.post_attention_layernorm(h))
        b, s = self.ffn_scales
        return self.post_ffw_norm(r) * b + h * s


class BeamModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.sliding_window = args.sliding_window
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            DecoderLayer(args, i) for i in range(args.num_hidden_layers)]
        self.norm = _Norm(args.hidden_size, eps=args.rms_norm_eps)

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
        input_embeddings: Optional[mx.array] = None,
    ):
        h = self.embed_tokens(inputs) if input_embeddings is None \
            else input_embeddings
        if cache is None:
            cache = [None] * len(self.layers)

        fa_mask = swa_mask = q_scale = None
        if self.fa_idx is not None:
            fa_cache = cache[self.fa_idx]
            fa_mask = create_attention_mask(h, fa_cache)
            q_scale = nope_query_scale(
                fa_cache.offset if fa_cache is not None else 0, h.shape[1],
                self.args.attn_temperature_scale,
                self.args.attn_temperature_length)
        if self.swa_idx is not None:
            swa_mask = create_attention_mask(
                h, cache[self.swa_idx], window_size=self.sliding_window
            )

        for layer, c in zip(self.layers, cache):
            if layer.use_sliding:
                h = layer(h, swa_mask, c)
            else:
                h = layer(h, fa_mask, c, q_scale)

        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = BeamModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def head(self, h: mx.array) -> mx.array:
        if self.args.tie_word_embeddings:
            out = self.model.embed_tokens.as_linear(h)
        else:
            out = self.lm_head(h)
        return scale_and_softcap(out, 1.0, self.args.final_logit_softcapping)

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
        input_embeddings: Optional[mx.array] = None,
    ):
        return self.head(self.model(inputs, cache, input_embeddings))

    def sanitize(self, weights):
        if self.args.tie_word_embeddings:
            weights.pop("lm_head.weight", None)
        return weights

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
        # Kept in step with the loader's fp32 pins (loader.fp32_keep_for).
        norms = fp32_norms("beam")

        def predicate(path: str):
            if path.endswith((".mlp.gate.weight", ".e_score_correction_bias",
                              ".gate_bias")):
                return False
            return not (norms and NORM_WEIGHT_KEY in path)

        return predicate
