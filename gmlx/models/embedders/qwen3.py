# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2025 The Qwen team, Alibaba Group and the HuggingFace Inc. team
# Ported from transformers 5.15.1 src/transformers/models/qwen3/modeling_qwen3.py
# (the Qwen3Model causal and padding masks) and the last-token pooling of the
# Qwen3-Embedding sentence-transformers configuration, and modified for gmlx
# (Apache-2.0; see licenses/transformers-LICENSE).
"""Qwen3-Embedding: the Qwen3 decoder as a sentence encoder.

The blocks are mlx-lm's ``qwen3`` blocks, the same ones the Qwen3 GGUF
loads into. Attention is causal, and a padding token is never a key. The
sentence vector is the last hidden state of each text's last real token,
L2-normalized. The tokenizer ends every text with ``<|endoftext|>``, so
that token is the one pooled.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
from gmlx._vendor.mlx_lm.models import qwen3

from .base import EncoderOutput, additive_mask, l2_normalize, padding_mask, pool


@dataclass
class ModelArgs(qwen3.ModelArgs):
    # A transformers 5 config keeps the RoPE base in rope_parameters.
    rope_theta: float | None = None
    head_dim: int | None = None
    # The encoder has no output projection, so the flag only matters to the
    # decoder. Configs that omit it load.
    tie_word_embeddings: bool = True
    pooling: str = "last"
    rope_parameters: dict | None = None

    def __post_init__(self):
        rp = self.rope_parameters or {}
        if self.rope_theta is None:
            self.rope_theta = rp.get("rope_theta")
        if self.rope_theta is None:
            raise ValueError("qwen3: the config sets no rope_theta")
        if self.rope_scaling is None and rp.get("rope_type", "default") != "default":
            self.rope_scaling = rp
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = qwen3.Qwen3Model(args)

    def __call__(self, input_ids: mx.array,
                 attention_mask: mx.array | None = None) -> EncoderOutput:
        trunk = self.model
        h = trunk.embed_tokens(input_ids)
        pad = padding_mask(attention_mask)
        if pad is None:
            mask = "causal"
        else:
            n = input_ids.shape[1]
            pos = mx.arange(n)
            causal = pos[:, None] >= pos[None, :]
            mask = additive_mask(mx.logical_and(pad, causal), h.dtype)
        for layer in trunk.layers:
            h = layer(h, mask, None)
        hidden = trunk.norm(h)
        x = pool(hidden, attention_mask, self.args.pooling)
        return EncoderOutput(last_hidden_state=hidden,
                             text_embeds=l2_normalize(x))

    @property
    def layers(self):
        return self.model.layers

    @staticmethod
    def sanitize(weights: dict) -> dict:
        """Home a bare Qwen3Model tree under ``model.`` and drop an LM head."""
        out = {}
        for k, v in weights.items():
            if k.startswith("lm_head."):
                continue
            out[k if k.startswith("model.") else f"model.{k}"] = v
        return out
