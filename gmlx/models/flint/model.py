# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Asher Feldman
"""Model for RecursiveSI Flint (GGUF arch ``flint``).

This is the runtime for llama.cpp's ``LLM_ARCH_FLINT`` conversions. The
attention is GQA with a per-head Q/K RMSNorm. These parts differ from a
Llama block:

  1. Every layer adds per-head learnable attention sinks to the softmax.
  2. The sliding-window layers run NeoX RoPE. The full layers are NoPE.
  3. The layers after the leading dense ones route a sigmoid MoE with an
     optional selection-only bias, renormalized weights and one ungated
     shared expert.
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
from gmlx.load.dtypes import (
    NORM_WEIGHT_KEY,
    activation_dtype,
    fp32_norms,
    wider_float,
)
from gmlx.models.k2_horizon.model import _route, _route_weights


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
    shared_expert_intermediate_size: int = 0
    first_k_dense_replace: int = 0
    router_bias: bool = False
    sliding_window: int = 513
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    rope_parameters: Optional[dict] = None
    max_position_embeddings: int = 262144
    routed_scaling_factor: float = 1.0
    norm_topk_prob: bool = True
    tie_word_embeddings: bool = False


class FlintGate(nn.Module):
    """FFN router: sigmoid scores, an optional selection-only bias, then
    renormalized and scaled weights.

    A gate submodule returning ``(inds, weights)`` with an int ``top_k``,
    the shape stream/moe_experts and stream/lookahead duck-type on.
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.top_k = int(args.num_experts_per_tok)
        self.norm_topk_prob = bool(args.norm_topk_prob)
        self.routed_scaling_factor = float(args.routed_scaling_factor)
        # fp32 like the F32 wire tensors; llama.cpp routes in fp32.
        self.weight = mx.zeros(
            (args.num_experts, args.hidden_size), dtype=mx.float32)
        if args.router_bias:
            self.e_score_correction_bias = mx.zeros(
                (args.num_experts,), dtype=mx.float32)

    def _logits(self, x: mx.array) -> mx.array:
        return x.astype(mx.float32) @ self.weight.T.astype(mx.float32)

    def __call__(self, x: mx.array):
        bias = (self.e_score_correction_bias
                if "e_score_correction_bias" in self else None)
        return _route(
            self._logits(x), bias, self.top_k, "sigmoid", self.norm_topk_prob,
            self.routed_scaling_factor, fused=not self.training)

    def _kq_route_weights(self, x, inds):
        # Route replay (stream/moe_routes): the weight branch of _route at
        # caller-chosen ids.
        return _route_weights(
            mx.sigmoid(self._logits(x)), inds, self.norm_topk_prob,
            self.routed_scaling_factor)


class MLP(nn.Module):
    """SwiGLU MLP: the dense layers and the shared expert."""

    # Plain SiLU. The fused shared-expert fold reads this to accept the
    # shared expert as more slots of the expert gathers.
    swiglu_limit = 0.0

    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden, bias=False)
        self.up_proj = nn.Linear(dim, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class FlintMoE(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.gate = FlintGate(args)
        self.switch_mlp = SwitchGLU(
            args.hidden_size, args.moe_intermediate_size, args.num_experts)
        if args.shared_expert_intermediate_size:
            self.shared_experts = MLP(
                args.hidden_size, args.shared_expert_intermediate_size)

    def __call__(self, x: mx.array) -> mx.array:
        inds, scores = self.gate(x)
        # Hand the routing weights to a swapped expert module that mixes
        # them itself or feeds them to the streaming miss-shed.
        if (getattr(self.switch_mlp, "_kq_mix_scores", False)
                or getattr(self.switch_mlp, "_kq_scores_sink", False)):
            y = self.switch_mlp(x, inds, scores)
        else:
            y = self.switch_mlp(x, inds)
        if y.ndim == scores.ndim + 1:
            y = (y * scores[..., None].astype(y.dtype)).sum(-2)
        elif getattr(self.switch_mlp, "_kq_shexp_mod", None) is not None:
            # A mixed return from the stamped shared-expert fold has it.
            return y
        if "shared_experts" in self:
            y = y + self.shared_experts(x)
        return y


class _Norm(nn.RMSNorm):
    """RMSNorm whose weight the loader may keep float32. The output takes
    the wider of the input and activation dtypes, as a weight in the
    activation dtype gives, so a float32 weight never widens a bf16
    stream."""

    OUTPUT_IN_INPUT_DTYPE = True

    def __init__(self, dims: int, eps: float = 1e-5):
        super().__init__(dims, eps=eps)
        self._act = activation_dtype("flint")

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
        # Per-head sink logits. mx.fast SDPA takes them in the output dtype,
        # so they cast with the weights at load.
        self.sinks = mx.zeros((self.n_heads,))

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
            queries, keys, values, cache=cache, scale=self.scale, mask=mask,
            sinks=self.sinks.astype(queries.dtype),
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class DecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        dim = args.hidden_size
        self.use_sliding = args.layer_types[layer_idx] == "sliding_attention"
        self.self_attn = Attention(args, self.use_sliding)
        if layer_idx < args.first_k_dense_replace:
            self.mlp = MLP(dim, args.intermediate_size)
        else:
            self.mlp = FlintMoE(args)
        self.input_layernorm = _Norm(dim, eps=args.rms_norm_eps)
        self.post_attention_layernorm = _Norm(dim, eps=args.rms_norm_eps)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        # A float32 stream in a float16 graph (FlintModel._WIDE_FROM): the
        # sublayers take float16, and the MoE takes bfloat16 for its range.
        act = self.self_attn.sinks.dtype
        if x.dtype == act or x.dtype != mx.float32:
            h = x + self.self_attn(self.input_layernorm(x), mask, cache)
            return h + self.mlp(self.post_attention_layernorm(h))
        h = x + self.self_attn(
            self.input_layernorm(x).astype(act), mask, cache)
        y = self.post_attention_layernorm(h)
        y = y.astype(mx.bfloat16 if isinstance(self.mlp, FlintMoE) else act)
        return h + self.mlp(y)


class FlintModel(nn.Module):
    # A float16 graph holds the residual stream in float32. Number tokens
    # push it past 40000 in the last layers, and their routed SwiGLU output
    # past 300000, beyond float16's 65504.
    _WIDE_FROM = mx.float16

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

        fa_mask = swa_mask = None
        if self.fa_idx is not None:
            fa_mask = create_attention_mask(h, cache[self.fa_idx])
        if self.swa_idx is not None:
            swa_mask = create_attention_mask(
                h, cache[self.swa_idx], window_size=self.sliding_window
            )
        # The sinks carry the graph's activation dtype.
        wide = self.layers[0].self_attn.sinks.dtype == self._WIDE_FROM
        if wide:
            h = h.astype(mx.float32)

        for layer, c in zip(self.layers, cache):
            h = layer(h, swa_mask if layer.use_sliding else fa_mask, c)

        out = self.norm(h)
        return out.astype(self._WIDE_FROM) if wide else out


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = FlintModel(args)
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
        return [
            RotatingKVCache(max_size=self.model.sliding_window)
            if layer.use_sliding
            else KVCache()
            for layer in self.layers
        ]

    @property
    def cast_predicate(self):
        # Kept in step with the loader's fp32 pins (loader.fp32_keep_for).
        norms = fp32_norms("flint")

        def predicate(path: str):
            if path.endswith((".mlp.gate.weight", ".e_score_correction_bias")):
                return False
            return not (norms and NORM_WEIGHT_KEY in path)

        return predicate
