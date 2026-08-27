# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Asher Feldman
# Portions copyright (c) 2024 Apple Inc. (mlx-lm afmoe skeleton, MIT)
"""Vendored mlx-lm-style model for Meta Muse Glimmer (GGUF arch ``muse-glimmer``).

mlx-lm has no muse_glimmer class; this module is the runtime for llama.cpp's
``LLM_ARCH_MUSE_GLIMMER`` conversions, built from the pinned mlx-lm 0.31.3
``afmoe`` skeleton - which already has the attention output gate, per-head
QK-norm, sandwich norms and the sliding/full ``layer_types`` split - with the
MoE stripped and the Glimmer-only mechanics added:

  1. RoPE rides the sliding-window layers only; full-attention layers are NoPE.
     That is the inverse of the usual arrangement, and the reason the declared
     131072 context has no extrapolation ceiling: the largest positional offset
     ever resolved is the 2048 window.
  2. an unweighted RMSNorm on the token embeddings, before layer 0.
  3. two norm epsilons: ``rms_norm_eps`` (1e-5) on the pre-norms and the final
     norm, ``post_norm_eps`` (1e-8) on the two post-norms.
  4. ``output_multiplier`` on the logits, then a gemma-style tanh softcap.
  5. Q/K arrive interleaved (llama.cpp tags the arch LLAMA_ROPE_TYPE_NORM and
     the converter un-permutes HF's rotate_half layout), so rope runs
     ``traditional=True`` rather than permuting the wire bytes on load.

The four per-layer norm weights arrive with the +1 already folded in at
conversion, so they load as plain ``nn.RMSNorm`` weights with no gemma-style
unbake; the final norm is not baked. The QK-norm weights are synthesized at
conversion to absorb ``qk_scale_factor`` (q_norm is a uniform 3.87, k_norm is
ones), which is also why the interleaved Q/K layout is safe: a uniform per-head
norm is invariant under the rope permutation.
"""

import importlib
import sys
from dataclasses import dataclass
from typing import Any, List, Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.activations import swiglu
from mlx_lm.models.base import (
    BaseModelArgs,
    create_attention_mask,
    scaled_dot_product_attention,
)
from mlx_lm.models.cache import KVCache, RotatingKVCache
from mlx_lm.models.rope_utils import initialize_rope


def ensure_registered() -> None:
    """Make ``import mlx_lm.models.muse_glimmer`` resolve, preferring upstream."""
    if "mlx_lm.models.muse_glimmer" not in sys.modules:
        try:
            importlib.import_module("mlx_lm.models.muse_glimmer")  # upstream wins
        except ImportError:
            sys.modules["mlx_lm.models.muse_glimmer"] = sys.modules[__name__]


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    layer_types: List[str]
    sliding_window: int = 2048
    rms_norm_eps: float = 1e-5
    # Post-attention / post-FFN norms only (llama.cpp muse-glimmer.cpp:68,
    # HF text_config.post_norm_eps). Not carried in the GGUF.
    post_norm_eps: float = 1e-8
    rope_theta: float = 500000.0
    rope_parameters: Optional[dict] = None
    max_position_embeddings: int = 131072
    output_multiplier: float = 1.0
    final_logit_softcapping: float = 0.0
    tie_word_embeddings: bool = False


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

        self.q_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)

        # Interleaved Q/K wire layout => traditional rope. Full-attention
        # layers carry no rope at all.
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
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        output = output * mx.sigmoid(self.gate_proj(x))
        return self.o_proj(output)


class MLP(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        dim, hidden = args.hidden_size, args.intermediate_size
        self.gate_proj = nn.Linear(dim, hidden, bias=False)
        self.up_proj = nn.Linear(dim, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


class DecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, use_sliding: bool):
        super().__init__()
        self.use_sliding = use_sliding
        self.self_attn = Attention(args, use_sliding)
        self.mlp = MLP(args)

        dim, eps, post_eps = args.hidden_size, args.rms_norm_eps, args.post_norm_eps
        self.input_layernorm = nn.RMSNorm(dim, eps=eps)
        self.post_attention_layernorm = nn.RMSNorm(dim, eps=post_eps)
        self.pre_feedforward_layernorm = nn.RMSNorm(dim, eps=eps)
        self.post_feedforward_layernorm = nn.RMSNorm(dim, eps=post_eps)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        r = self.self_attn(self.input_layernorm(x), mask, cache)
        h = x + self.post_attention_layernorm(r)
        r = self.mlp(self.pre_feedforward_layernorm(h))
        return h + self.post_feedforward_layernorm(r)


class MuseGlimmerModel(nn.Module):
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
        capture_layers: Optional[tuple] = None,
        inputs_embeds: Optional[mx.array] = None,
    ):
        # The embedding norm sits after llama.cpp's build_inp_embd, so injected
        # multimodal embeddings are normed alongside token embeddings.
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

        captures = []
        cap_set = capture_layers or ()
        for idx, (layer, c) in enumerate(zip(self.layers, cache)):
            h = layer(h, swa_mask if layer.use_sliding else fa_mask, cache=c)
            if idx in cap_set:
                captures.append(h)

        if capture_layers is not None:
            return self.norm(h), captures
        return self.norm(h)


def scale_and_softcap(out: mx.array, multiplier: float, cap: float) -> mx.array:
    """Logit tail shared with the vision-language wrapper: the output multiplier
    then the gemma-style tanh softcap. Computed in fp32 - llama.cpp's parity
    oracle scales and softcaps an fp32 ``result_output``, and the softcap is
    nonlinear enough that bf16 rounding moves argmax at depth."""
    out = out.astype(mx.float32) * multiplier
    if cap:
        out = mx.tanh(out / cap) * cap
    return out


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = MuseGlimmerModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def head(self, h: mx.array) -> mx.array:
        """Logits from a final-normed hidden state: lm_head, output multiplier,
        tanh softcap."""
        if self.args.tie_word_embeddings:
            out = self.model.embed_tokens.as_linear(h)
        else:
            out = self.lm_head(h)
        return scale_and_softcap(
            out, self.args.output_multiplier, self.args.final_logit_softcapping)

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
