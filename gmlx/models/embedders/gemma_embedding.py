# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2025 Google Inc. HuggingFace Inc. team
# Ported from transformers 5.15.1
# src/transformers/models/gemma3/modeling_gemma3.py and
# configuration_gemma3.py (the Gemma3TextModel masks under
# use_bidirectional_attention) and modified for gmlx (Apache-2.0; see
# licenses/transformers-LICENSE).
"""EmbeddingGemma: a gemma3 text backbone run as a bidirectional encoder.

The blocks are mlx-lm's ``gemma3_text`` blocks. The encoder differs from the
decoder only in its masks: a global layer attends to every real token, and a
sliding layer attends to the real tokens at most ``sliding_window // 2``
positions away on either side. The sentence vector is the mean of the last
hidden states over the real tokens, passed through the bias-free dense
layers and L2-normalized.

Safetensors checkpoints (``model_type`` ``gemma3_text``) and the GGUF
(``gemma_embedding``, built by the GGUF loader) share this class.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
from gmlx._vendor.mlx_lm.models import gemma3_text

from .base import (
    EncoderOutput,
    combine_masks,
    l2_normalize,
    padding_mask,
    pool,
    window_mask,
)


@dataclass
class ModelArgs(gemma3_text.ModelArgs):
    # Output widths of the dense layers after pooling, in order. None means
    # the EmbeddingGemma head: 4 * hidden_size, then hidden_size.
    dense_dims: list | None = None
    pooling: str = "mean"
    # The transformers 5 config fields. layer_types sets the sliding layer
    # pattern and rope_parameters the two RoPE bases.
    layer_types: list | None = None
    rope_parameters: dict | None = None

    def __post_init__(self):
        if self.dense_dims is None:
            self.dense_dims = [4 * self.hidden_size, self.hidden_size]
        if self.layer_types:
            full = [t == "full_attention" for t in self.layer_types]
            n = full.index(True) + 1 if any(full) else len(full) + 1
            if full != [(i + 1) % n == 0 for i in range(len(full))]:
                raise ValueError(
                    "gemma3_text: layer_types must repeat a run of sliding "
                    "layers closed by one full attention layer")
            self.sliding_window_pattern = n
        rp = self.rope_parameters or {}
        full_theta = (rp.get("full_attention") or {}).get("rope_theta")
        local_theta = (rp.get("sliding_attention") or {}).get("rope_theta")
        if full_theta is not None:
            self.rope_theta = full_theta
        if local_theta is not None:
            self.rope_local_base_freq = local_theta


class Encoder(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [gemma3_text.TransformerBlock(args, layer_idx=i)
                       for i in range(args.num_hidden_layers)]
        self.norm = gemma3_text.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(self, input_ids: mx.array,
                 attention_mask: mx.array | None) -> mx.array:
        h = self.embed_tokens(input_ids)
        # The scale is rounded to the weight dtype first, as the reference does.
        h = h * mx.array(self.args.hidden_size**0.5, h.dtype)
        pad = padding_mask(attention_mask)
        local = window_mask(input_ids.shape[1], self.args.sliding_window // 2)
        global_mask = pad
        local_mask = combine_masks(pad, local)
        for layer in self.layers:
            mask = local_mask if layer.self_attn.is_sliding else global_mask
            h = layer(h, mask, None)
        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = Encoder(args)
        dims = [args.hidden_size, *(args.dense_dims or [])]
        self.dense = [nn.Linear(i, o, bias=False)
                      for i, o in zip(dims[:-1], dims[1:])]

    def __call__(self, input_ids: mx.array,
                 attention_mask: mx.array | None = None) -> EncoderOutput:
        hidden = self.model(input_ids, attention_mask)
        x = pool(hidden, attention_mask, self.args.pooling)
        for layer in self.dense:
            x = layer(x.astype(hidden.dtype)).astype(mx.float32)
        return EncoderOutput(last_hidden_state=hidden,
                             text_embeds=l2_normalize(x))

    @property
    def layers(self):
        return self.model.layers

    @staticmethod
    def sanitize(weights: dict) -> dict:
        """Drop a stray LM head: the encoder has no output projection."""
        return {k: v for k, v in weights.items() if not k.startswith("lm_head.")}
