# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2018 The Google AI Language Team Authors, Facebook AI Research and The HuggingFace Inc. team
# Ported from transformers 5.15.1 src/transformers/models/bert/modeling_bert.py
# and src/transformers/models/xlm_roberta/modeling_xlm_roberta.py (the
# embeddings, the self-attention and the post-norm encoder layer) and modified
# for gmlx (Apache-2.0; see licenses/transformers-LICENSE).
"""BERT and XLM-RoBERTa sentence encoders.

The two share one encoder. They differ in position ids: BERT numbers
positions from 0, and XLM-RoBERTa numbers real tokens from
``pad_token_id + 1`` and gives padding the ``pad_token_id`` position.
Attention is bidirectional over the real tokens of each row.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
from gmlx._vendor.mlx_lm.models.base import BaseModelArgs

from .base import EncoderOutput, activation, l2_normalize, padding_mask, pool

XLM_ROBERTA_TYPES = frozenset({"xlm-roberta", "xlm_roberta"})


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    vocab_size: int = 30522
    max_position_embeddings: int = 512
    type_vocab_size: int = 2
    layer_norm_eps: float = 1e-12
    hidden_act: str = "gelu"
    pad_token_id: int = 0
    position_embedding_type: str = "absolute"
    pooling: str = "mean"

    def __post_init__(self):
        if self.position_embedding_type != "absolute":
            raise ValueError(
                f"{self.model_type}: position_embedding_type "
                f"{self.position_embedding_type!r} is not supported "
                "(only 'absolute')")


class Embeddings(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.fairseq_positions = args.model_type in XLM_ROBERTA_TYPES
        self.pad_token_id = args.pad_token_id
        self.word_embeddings = nn.Embedding(args.vocab_size, args.hidden_size)
        self.position_embeddings = nn.Embedding(
            args.max_position_embeddings, args.hidden_size)
        self.token_type_embeddings = nn.Embedding(
            args.type_vocab_size, args.hidden_size)
        self.LayerNorm = nn.LayerNorm(args.hidden_size, eps=args.layer_norm_eps)

    def __call__(self, input_ids: mx.array) -> mx.array:
        B, L = input_ids.shape
        if self.fairseq_positions:
            real = (input_ids != self.pad_token_id).astype(mx.int32)
            position_ids = mx.cumsum(real, axis=1) * real + self.pad_token_id
        else:
            position_ids = mx.broadcast_to(mx.arange(L), (B, L))
        token_type_ids = mx.zeros((B, L), dtype=mx.int32)
        h = self.word_embeddings(input_ids)
        h = h + self.token_type_embeddings(token_type_ids)
        h = h + self.position_embeddings(position_ids)
        return self.LayerNorm(h)


class SelfAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.num_attention_heads
        self.head_dim = args.hidden_size // args.num_attention_heads
        self.scale = self.head_dim**-0.5
        self.query = nn.Linear(args.hidden_size, args.hidden_size)
        self.key = nn.Linear(args.hidden_size, args.hidden_size)
        self.value = nn.Linear(args.hidden_size, args.hidden_size)

    def __call__(self, x: mx.array, mask: mx.array | None) -> mx.array:
        B, L, _ = x.shape

        def heads(t):
            return t.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)

        q, k, v = heads(self.query(x)), heads(self.key(x)), heads(self.value(x))
        o = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale,
                                                 mask=mask)
        return o.transpose(0, 2, 1, 3).reshape(B, L, -1)


class SelfOutput(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.dense = nn.Linear(args.hidden_size, args.hidden_size)
        self.LayerNorm = nn.LayerNorm(args.hidden_size, eps=args.layer_norm_eps)

    def __call__(self, x: mx.array, residual: mx.array) -> mx.array:
        return self.LayerNorm(self.dense(x) + residual)


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.self = SelfAttention(args)
        self.output = SelfOutput(args)

    def __call__(self, x: mx.array, mask: mx.array | None) -> mx.array:
        return self.output(self.self(x, mask), x)


class Intermediate(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.dense = nn.Linear(args.hidden_size, args.intermediate_size)
        self.act = activation(args.hidden_act)

    def __call__(self, x: mx.array) -> mx.array:
        return self.act(self.dense(x))


class Output(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.dense = nn.Linear(args.intermediate_size, args.hidden_size)
        self.LayerNorm = nn.LayerNorm(args.hidden_size, eps=args.layer_norm_eps)

    def __call__(self, x: mx.array, residual: mx.array) -> mx.array:
        return self.LayerNorm(self.dense(x) + residual)


class Layer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.attention = Attention(args)
        self.intermediate = Intermediate(args)
        self.output = Output(args)

    def __call__(self, x: mx.array, mask: mx.array | None) -> mx.array:
        a = self.attention(x, mask)
        return self.output(self.intermediate(a), a)


class Encoder(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.layer = [Layer(args) for _ in range(args.num_hidden_layers)]

    def __call__(self, x: mx.array, mask: mx.array | None) -> mx.array:
        for layer in self.layer:
            x = layer(x, mask)
        return x


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.embeddings = Embeddings(args)
        self.encoder = Encoder(args)

    def __call__(self, input_ids: mx.array,
                 attention_mask: mx.array | None = None) -> EncoderOutput:
        hidden = self.encoder(self.embeddings(input_ids),
                              padding_mask(attention_mask))
        embeds = l2_normalize(pool(hidden, attention_mask, self.args.pooling))
        return EncoderOutput(last_hidden_state=hidden, text_embeds=embeds)

    @staticmethod
    def sanitize(weights: dict) -> dict:
        """Drop an HF task-model prefix and the tensors no encoder uses: the
        position id buffer, the pooler head and any task head."""
        out = {}
        for k, v in weights.items():
            for prefix in ("roberta.", "bert.", "xlm_roberta.", "model."):
                if k.startswith(prefix):
                    k = k[len(prefix):]
                    break
            if k.endswith("position_ids") or k.startswith(
                    ("pooler.", "lm_head.", "cls.")):
                continue
            out[k] = v
        return out
