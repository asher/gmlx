# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Asher Feldman
"""Model for Ricursive Ripple (GGUF arch ``ricursive``).

This is the runtime for llama.cpp's ``LLM_ARCH_RICURSIVE`` conversions
(``src/models/ricursive.cpp``). The block is a dense GQA decoder with
per-head Q/K RMSNorm and four norms per layer. These parts differ from a
Llama block:

  1. The sliding-window layers run NeoX RoPE. The full layers are NoPE.
  2. The token embeddings are scaled by ``embedding_scale``. That scaled
     embedding is x0, and each layer's V input is
     ``lerp(x0, attn_norm(x), sigmoid(lambda))`` (the value residual).
  3. A sigmoid output gate from the attention input multiplies the
     attention output before ``o_proj``.
  4. The SwiGLU clamps the gate from above and the up projection on both
     sides before the activation (``ggml_swiglu_clamp``).
"""

from dataclasses import dataclass
from typing import Any, List, Optional

import mlx.core as mx
import mlx.nn as nn

from gmlx._vendor.mlx_lm.models.base import BaseModelArgs
from gmlx._vendor.mlx_lm.models.rope_utils import initialize_rope

from gmlx.cache import window_sdpa
from gmlx.cache.attention import (
    create_attention_mask,
    scaled_dot_product_attention,
)
from gmlx.cache.kv_cache import KVCache, RotatingKVCache
from gmlx.models.deepseek_v4.model import _limited_swiglu


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
    sliding_window: int = 1024
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    rope_parameters: Optional[dict] = None
    max_position_embeddings: int = 2097152
    embedding_scale: float = 1.0
    # One clamp per layer; 0 means a plain SwiGLU on that layer.
    swiglu_limits: Optional[List[float]] = None
    use_gated_attention: bool = True
    use_valres: bool = False
    tie_word_embeddings: bool = False


@mx.compile
def _gate(out: mx.array, g: mx.array) -> mx.array:
    # The reference takes the sigmoid in float32 (gate_full_precision).
    return out * mx.sigmoid(g.astype(mx.float32)).astype(out.dtype)


@mx.compile
def _value_input(x: mx.array, x0: mx.array, lam: mx.array) -> mx.array:
    mix = mx.sigmoid(lam.astype(mx.float32)).astype(x.dtype)
    return x0 + (x - x0) * mix


class Attention(nn.Module):
    def __init__(self, args: ModelArgs, use_sliding: bool):
        super().__init__()
        dim = args.hidden_size
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.head_dim = args.head_dim
        self.use_sliding = use_sliding
        self.window = args.sliding_window if use_sliding else None
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(dim, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, dim, bias=False)
        if args.use_gated_attention:
            self.gate_proj = nn.Linear(
                dim, self.n_heads * self.head_dim, bias=False)
        if args.use_valres:
            # float32 like the F32 wire tensor (the loader pins it).
            self.vres_lambda = mx.zeros((1,), dtype=mx.float32)

        self.q_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)

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
        x0: Optional[mx.array],
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, _ = x.shape

        v_in = x
        if "vres_lambda" in self:
            v_in = _value_input(x, x0, self.vres_lambda)

        queries = self.q_proj(x).reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = self.k_proj(x).reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = self.v_proj(v_in).reshape(
            B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        queries = self.q_norm(queries)
        keys = self.k_norm(keys)

        if self.rope is not None:
            offset = cache.offset if cache is not None else 0
            queries = self.rope(queries, offset=offset)
            keys = self.rope(keys, offset=offset)

        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        output = window_sdpa.claim(
            queries, keys, values, cache, self.scale, self.window
        )
        if output is None:
            output = scaled_dot_product_attention(
                queries, keys, values, cache=cache, scale=self.scale, mask=mask
            )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        if "gate_proj" in self:
            output = _gate(output, self.gate_proj(x))
        return self.o_proj(output)


class MLP(nn.Module):
    def __init__(self, dim: int, hidden: int, limit: float):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden, bias=False)
        self.up_proj = nn.Linear(dim, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, dim, bias=False)
        self.swiglu_limit = float(limit)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(_limited_swiglu(
            self.gate_proj(x), self.up_proj(x), self.swiglu_limit))


class DecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        dim, eps = args.hidden_size, args.rms_norm_eps
        self.use_sliding = args.layer_types[layer_idx] == "sliding_attention"
        self.self_attn = Attention(args, self.use_sliding)
        limits = args.swiglu_limits or []
        limit = limits[layer_idx] if layer_idx < len(limits) else 0.0
        self.mlp = MLP(dim, args.intermediate_size, limit)
        self.input_layernorm = nn.RMSNorm(dim, eps=eps)
        self.post_attention_layernorm = nn.RMSNorm(dim, eps=eps)
        self.pre_feedforward_layernorm = nn.RMSNorm(dim, eps=eps)
        self.post_feedforward_layernorm = nn.RMSNorm(dim, eps=eps)

    def __call__(
        self,
        x: mx.array,
        x0: Optional[mx.array],
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        r = self.self_attn(self.input_layernorm(x), x0, mask, cache)
        h = x + self.post_attention_layernorm(r)
        r = self.mlp(self.pre_feedforward_layernorm(h))
        return h + self.post_feedforward_layernorm(r)


class RicursiveModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.sliding_window = args.sliding_window
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            DecoderLayer(args, i) for i in range(args.num_hidden_layers)]
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
        capture_layers: Optional[tuple] = None,
        inputs_embeds: Optional[mx.array] = None,
    ):
        # The scale applies to given embeddings too, as in the reference,
        # and x0 is the scaled embedding for every layer.
        h = self.embed_tokens(inputs) if inputs_embeds is None else inputs_embeds
        if self.args.embedding_scale != 1.0:
            h = h * mx.array(self.args.embedding_scale, dtype=h.dtype)
        x0 = h if self.args.use_valres else None

        if cache is None:
            cache = [None] * len(self.layers)

        fa_mask = swa_mask = None
        if self.fa_idx is not None:
            fa_mask = create_attention_mask(h, cache[self.fa_idx])
        if self.swa_idx is not None:
            swa_mask = create_attention_mask(
                h, cache[self.swa_idx], window_size=self.sliding_window
            )

        captures = []
        cap_set = capture_layers or ()
        for idx, (layer, c) in enumerate(zip(self.layers, cache)):
            h = layer(h, x0, swa_mask if layer.use_sliding else fa_mask, c)
            if idx in cap_set:
                captures.append(h)

        if capture_layers is not None:
            return self.norm(h), captures
        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = RicursiveModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def head(self, h: mx.array) -> mx.array:
        if self.args.tie_word_embeddings:
            return self.model.embed_tokens.as_linear(h)
        return self.lm_head(h)

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
        input_embeddings: Optional[mx.array] = None,
    ):
        return self.head(
            self.model(inputs, cache, inputs_embeds=input_embeddings))

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
        def predicate(path: str):
            return not path.endswith(".vres_lambda")

        return predicate
