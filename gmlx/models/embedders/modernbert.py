# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2024 Answer.AI, LightOn, and contributors, and the HuggingFace Inc. team. All rights reserved.
# Ported from transformers 5.15.1
# src/transformers/models/modernbert/modeling_modernbert.py (the embeddings,
# the rotary attention with global and local layers, the GeGLU MLP and the
# pre-norm encoder layer) and modified for gmlx (Apache-2.0; see
# licenses/transformers-LICENSE).
"""ModernBERT sentence encoder.

Every ``global_attn_every_n_layers``-th layer, starting at layer 0, attends
over the whole sequence; the others attend within ``local_attention // 2``
tokens on either side. Global and local layers use their own RoPE bases.
Layer 0 has no attention norm. The MLP is a GeGLU: the first half of
``Wi`` goes through the activation and gates the second half.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
from gmlx._vendor.mlx_lm.models.base import BaseModelArgs

from .base import (
    EncoderOutput,
    activation,
    additive_mask,
    combine_masks,
    l2_normalize,
    padding_mask,
    pool,
    window_mask,
)


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    hidden_size: int = 768
    intermediate_size: int = 1152
    num_hidden_layers: int = 22
    num_attention_heads: int = 12
    vocab_size: int = 50368
    norm_eps: float = 1e-5
    norm_bias: bool = False
    attention_bias: bool = False
    mlp_bias: bool = False
    hidden_activation: str = "gelu"
    global_attn_every_n_layers: int | None = 3
    local_attention: int = 128
    global_rope_theta: float | None = 160000.0
    local_rope_theta: float | None = 10000.0
    layer_types: list | None = None
    rope_parameters: dict | None = None
    pooling: str = "mean"

    def __post_init__(self):
        # A transformers 5 config writes layer_types and rope_parameters and
        # leaves the older fields null.
        if self.layer_types is None:
            n = self.global_attn_every_n_layers or 3
            self.layer_types = [
                "full_attention" if i % n == 0 else "sliding_attention"
                for i in range(self.num_hidden_layers)]
        rp = self.rope_parameters or {}
        full = (rp.get("full_attention") or {}).get("rope_theta")
        local = (rp.get("sliding_attention") or {}).get("rope_theta")
        self.global_rope_theta = full or self.global_rope_theta or 160000.0
        self.local_rope_theta = local or self.local_rope_theta or 10000.0


class Embeddings(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.hidden_size)
        self.norm = nn.LayerNorm(args.hidden_size, eps=args.norm_eps,
                                 bias=args.norm_bias)

    def __call__(self, input_ids: mx.array) -> mx.array:
        return self.norm(self.tok_embeddings(input_ids))


class Attention(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.n_heads = args.num_attention_heads
        self.head_dim = args.hidden_size // args.num_attention_heads
        self.scale = self.head_dim**-0.5
        self.is_global = args.layer_types[layer_idx] == "full_attention"
        base = args.global_rope_theta if self.is_global else args.local_rope_theta
        self.Wqkv = nn.Linear(args.hidden_size, 3 * args.hidden_size,
                              bias=args.attention_bias)
        self.Wo = nn.Linear(args.hidden_size, args.hidden_size,
                            bias=args.attention_bias)
        self.rope = nn.RoPE(self.head_dim, traditional=False, base=base)

    def __call__(self, x: mx.array, mask: mx.array | None) -> mx.array:
        B, L, _ = x.shape

        def heads(t):
            return t.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)

        q, k, v = (heads(t) for t in mx.split(self.Wqkv(x), 3, axis=-1))
        q, k = self.rope(q), self.rope(k)
        o = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale,
                                                 mask=mask)
        return self.Wo(o.transpose(0, 2, 1, 3).reshape(B, L, -1))


@functools.cache
def _glu(name: str):
    """``act(h) * gate`` for the activation ``name``, compiled once."""
    act = activation(name)
    return mx.compile(lambda h, gate: act(h) * gate, shapeless=True)


class MLP(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.Wi = nn.Linear(args.hidden_size, 2 * args.intermediate_size,
                            bias=args.mlp_bias)
        self.Wo = nn.Linear(args.intermediate_size, args.hidden_size,
                            bias=args.mlp_bias)
        self.act_name = args.hidden_activation

    def __call__(self, x: mx.array) -> mx.array:
        h, gate = mx.split(self.Wi(x), 2, axis=-1)
        return self.Wo(_glu(self.act_name)(h, gate))


class Layer(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        if layer_idx > 0:
            self.attn_norm = nn.LayerNorm(args.hidden_size, eps=args.norm_eps,
                                          bias=args.norm_bias)
        self.attn = Attention(args, layer_idx)
        self.mlp_norm = nn.LayerNorm(args.hidden_size, eps=args.norm_eps,
                                     bias=args.norm_bias)
        self.mlp = MLP(args)

    def __call__(self, x: mx.array, mask: mx.array | None) -> mx.array:
        a = self.attn_norm(x) if "attn_norm" in self else x
        x = x + self.attn(a, mask)
        return x + self.mlp(self.mlp_norm(x))


class ModernBertModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.embeddings = Embeddings(args)
        self.layers = [Layer(args, i) for i in range(args.num_hidden_layers)]
        self.final_norm = nn.LayerNorm(args.hidden_size, eps=args.norm_eps,
                                       bias=args.norm_bias)

    def __call__(self, input_ids: mx.array,
                 attention_mask: mx.array | None) -> mx.array:
        h = self.embeddings(input_ids)
        pad = padding_mask(attention_mask)
        local = window_mask(input_ids.shape[1], self.args.local_attention // 2)
        # Additive masks, built once for all layers.
        global_mask = additive_mask(pad, h.dtype)
        local_mask = additive_mask(combine_masks(pad, local), h.dtype)
        for layer in self.layers:
            h = layer(h, global_mask if layer.attn.is_global else local_mask)
        return self.final_norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = ModernBertModel(args)

    def __call__(self, input_ids: mx.array,
                 attention_mask: mx.array | None = None) -> EncoderOutput:
        hidden = self.model(input_ids, attention_mask)
        embeds = l2_normalize(pool(hidden, attention_mask, self.args.pooling))
        return EncoderOutput(last_hidden_state=hidden, text_embeds=embeds)

    @staticmethod
    def sanitize(weights: dict) -> dict:
        """Home a bare ModernBertModel tree under ``model.`` and drop the
        masked-LM head."""
        out = {}
        for k, v in weights.items():
            if k.startswith(("head.", "decoder.")):
                continue
            if not k.startswith("model."):
                k = "model." + k
            out[k] = v
        return out
