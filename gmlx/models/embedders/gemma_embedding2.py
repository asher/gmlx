# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 Google Inc. HuggingFace Inc. team. All rights reserved.
# Ported from transformers 5.22.2
# src/transformers/models/embedding_gemma2/modeling_embedding_gemma2.py and
# configuration_embedding_gemma2.py, and modified for gmlx (Apache-2.0; see
# licenses/transformers-LICENSE).
"""EmbeddingGemma 2: a bidirectional Gemma 4-style encoder with per-layer
inputs, and optional Gemma 4 vision and audio towers.

The text backbone has sliding layers (head_dim 256, a symmetric window) and
full layers (head_dim 512, one KV head). Each layer adds a per-layer input
block after its MLP. The per-layer inputs are projected from the input
embeddings alone, so media soft tokens feed them too. The sentence vector is
the mean of the normed last hidden states over every token, projected to
``embedding_dim`` and L2-normalized.

Attention runs in query blocks: a sliding layer attends to a key band of the
block plus the window on each side, and a full layer attends to every key
one block at a time. This bounds the score memory at full context.

GGUF loads (``gemma_embedding2``, built by the GGUF loader) and MLX
checkpoints (``embedding_gemma2``) share this class. Attribute names follow
transformers, so an MLX checkpoint loads without renames.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from .base import EncoderOutput, l2_normalize

# Query rows per attention block. Tests lower it to cross block edges with
# short inputs.
Q_BLOCK = 512

MODEL_TYPES = ("gemma_embedding2", "embedding_gemma2")


@dataclass
class TextArgs:
    vocab_size: int = 262_144
    hidden_size: int = 512
    intermediate_size: int = 2048
    num_hidden_layers: int = 24
    num_attention_heads: int = 4
    num_key_value_heads: int = 2
    head_dim: int = 256
    hidden_activation: str = "gelu_pytorch_tanh"
    max_position_embeddings: int = 8192
    rms_norm_eps: float = 1e-6
    pad_token_id: int = 0
    rope_parameters: dict | None = None
    attention_bias: bool = False
    # Keys a sliding-layer query sees on each side, as in transformers'
    # bidirectional sliding mask (|q - k| <= sliding_window).
    sliding_window: int = 512
    layer_types: list | None = None
    hidden_size_per_layer_input: int = 512
    embedding_dim: int = 768
    # Per-layer overrides keyed by layer index (int or a "05"-style string).
    per_layer_config: dict | None = None
    global_head_dim: int = 512
    num_global_key_value_heads: int = 1

    def __post_init__(self):
        if self.layer_types is None:
            self.layer_types = [
                "sliding_attention" if (i + 1) % 6 else "full_attention"
                for i in range(self.num_hidden_layers)]
        if self.rope_parameters is None:
            self.rope_parameters = {
                "sliding_attention": {"rope_type": "default", "rope_theta": 10_000.0},
                "full_attention": {"rope_type": "default", "rope_theta": 1_000_000.0},
            }
        for kind, rp in self.rope_parameters.items():
            if (rp or {}).get("rope_type", "default") != "default":
                raise ValueError(
                    f"embedding_gemma2: rope type {rp['rope_type']!r} on "
                    f"{kind} layers is not supported")
        if self.per_layer_config is None:
            self.per_layer_config = {
                i: {"head_dim": self.global_head_dim,
                    "num_key_value_heads": self.num_global_key_value_heads}
                for i, t in enumerate(self.layer_types) if t == "full_attention"}
        else:
            self.per_layer_config = {int(k): dict(v)
                                     for k, v in self.per_layer_config.items()}

    def layer(self, i: int) -> tuple[int, int]:
        """``(head_dim, num_key_value_heads)`` of layer ``i``."""
        over = self.per_layer_config.get(i, {})
        return (int(over.get("head_dim", self.head_dim)),
                int(over.get("num_key_value_heads", self.num_key_value_heads)))

    @classmethod
    def from_dict(cls, d: dict) -> "TextArgs":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})


@dataclass
class ModelArgs:
    model_type: str = "gemma_embedding2"
    text_config: dict = field(default_factory=dict)
    vision_config: dict | None = None
    audio_config: dict | None = None
    boi_token_id: int = 255_999
    eoi_token_id: int = 258_882
    image_token_id: int = 258_880
    video_token_id: int = 258_884
    boa_token_id: int = 256_000
    eoa_token_index: int = 258_883
    audio_token_id: int = 258_881
    # The most tokens an item may have, whatever the backbone's rope reaches.
    max_tokens: int = 8192

    def __post_init__(self):
        self.text = TextArgs.from_dict(self.text_config or {})

    @property
    def max_position_embeddings(self) -> int:
        return min(self.max_tokens, self.text.max_position_embeddings)

    @classmethod
    def from_dict(cls, d: dict) -> "ModelArgs":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})


def _rms_norm(x: mx.array, weight: mx.array | None, eps: float) -> mx.array:
    """transformers' EmbeddingGemma2RMSNorm: normalize in float32, scale by
    the weight itself (no 1 + w), and return the input dtype."""
    out = mx.fast.rms_norm(x.astype(mx.float32), None, eps)
    if weight is not None:
        out = out * weight.astype(mx.float32)
    return out.astype(x.dtype)


class RMSNorm(nn.Module):
    def __init__(self, dims: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        return _rms_norm(x, self.weight, self.eps)


def _rope(x: mx.array, base: float, offset: int = 0) -> mx.array:
    """Full-width default RoPE (rotate-half), positions from ``offset``. The
    reference rounds cos and sin to the activation dtype."""
    dims = x.shape[-1]
    inv = 1.0 / (base ** (mx.arange(0, dims, 2, dtype=mx.float32) / dims))
    pos = mx.arange(offset, offset + x.shape[-2], dtype=mx.float32)
    ang = pos[:, None] * inv[None, :]
    ang = mx.concatenate([ang, ang], axis=-1)
    cos = mx.cos(ang).astype(x.dtype)
    sin = mx.sin(ang).astype(x.dtype)
    half = dims // 2
    rot = mx.concatenate([-x[..., half:], x[..., :half]], axis=-1)
    return x * cos + rot * sin


def _band_mask(q0: int, q1: int, k0: int, k1: int, radius: int) -> mx.array:
    """``[q1 - q0, k1 - k0]`` boolean mask allowing ``|q - k| <= radius``."""
    q = mx.arange(q0, q1)[:, None]
    k = mx.arange(k0, k1)[None, :]
    return mx.abs(q - k) <= radius


def _key_pad(pad: mx.array | None, k0: int, k1: int) -> mx.array | None:
    return None if pad is None else pad[:, :, :, k0:k1]


def _combine(a: mx.array | None, b: mx.array | None) -> mx.array | None:
    if a is None:
        return b
    if b is None:
        return a
    return mx.logical_and(a, b)


class Attention(nn.Module):
    def __init__(self, args: TextArgs, layer_idx: int):
        super().__init__()
        self.is_sliding = args.layer_types[layer_idx] == "sliding_attention"
        self.head_dim, self.n_kv = args.layer(layer_idx)
        self.n_heads = args.num_attention_heads
        self.eps = args.rms_norm_eps
        kind = "sliding_attention" if self.is_sliding else "full_attention"
        self.rope_base = float(args.rope_parameters[kind]["rope_theta"])
        self.window = args.sliding_window
        h, d, bias = args.hidden_size, self.head_dim, args.attention_bias
        self.q_proj = nn.Linear(h, self.n_heads * d, bias=bias)
        self.k_proj = nn.Linear(h, self.n_kv * d, bias=bias)
        self.v_proj = nn.Linear(h, self.n_kv * d, bias=bias)
        self.o_proj = nn.Linear(self.n_heads * d, h, bias=bias)
        self.q_norm = RMSNorm(d, eps=self.eps)
        self.k_norm = RMSNorm(d, eps=self.eps)

    def __call__(self, x: mx.array, pad: mx.array | None) -> mx.array:
        B, L, _ = x.shape
        d = self.head_dim
        q = self.q_norm(self.q_proj(x).reshape(B, L, self.n_heads, d))
        k = self.k_norm(self.k_proj(x).reshape(B, L, self.n_kv, d))
        v = _rms_norm(self.v_proj(x).reshape(B, L, self.n_kv, d), None, self.eps)
        q = _rope(q.transpose(0, 2, 1, 3), self.rope_base)
        k = _rope(k.transpose(0, 2, 1, 3), self.rope_base)
        v = v.transpose(0, 2, 1, 3)
        if self.is_sliding and L > self.window + 1:
            out = self._banded(q, k, v, pad)
        else:
            out = self._tiled(q, k, v, pad)
        out = out.transpose(0, 2, 1, 3).reshape(B, L, self.n_heads * d)
        return self.o_proj(out)

    def _tiled(self, q, k, v, pad):
        """Every key, one query block at a time."""
        L = q.shape[2]
        mask = pad
        outs = []
        for q0 in range(0, L, Q_BLOCK):
            q1 = min(L, q0 + Q_BLOCK)
            outs.append(mx.fast.scaled_dot_product_attention(
                q[:, :, q0:q1], k, v, scale=1.0, mask=mask))
        return outs[0] if len(outs) == 1 else mx.concatenate(outs, axis=2)

    def _banded(self, q, k, v, pad):
        """A query block against the keys within the window of any of its
        rows."""
        L, w = q.shape[2], self.window
        outs = []
        for q0 in range(0, L, Q_BLOCK):
            q1 = min(L, q0 + Q_BLOCK)
            k0, k1 = max(0, q0 - w), min(L, q1 + w)
            mask = _combine(_band_mask(q0, q1, k0, k1, w), _key_pad(pad, k0, k1))
            outs.append(mx.fast.scaled_dot_product_attention(
                q[:, :, q0:q1], k[:, :, k0:k1], v[:, :, k0:k1],
                scale=1.0, mask=mask))
        return outs[0] if len(outs) == 1 else mx.concatenate(outs, axis=2)


class MLP(nn.Module):
    def __init__(self, args: TextArgs):
        super().__init__()
        h, i = args.hidden_size, args.intermediate_size
        self.gate_proj = nn.Linear(h, i, bias=False)
        self.up_proj = nn.Linear(h, i, bias=False)
        self.down_proj = nn.Linear(i, h, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.gelu_approx(self.gate_proj(x)) * self.up_proj(x))


class PerLayerInputs(nn.Module):
    """Projection-only per-layer inputs: ``rmsnorm(proj(x) / sqrt(hidden))``
    split into one ``hidden_size_per_layer_input`` slice per layer.

    ``layer_input`` computes one layer's slice from that layer's rows of the
    projection. The projection is linear and the norm acts on each slice, so
    this equals slicing the full projection, without holding all layers at
    once."""

    def __init__(self, args: TextArgs):
        super().__init__()
        self.n_layers = args.num_hidden_layers
        self.width = args.hidden_size_per_layer_input
        self.scale = args.hidden_size ** -0.5
        self.eps = args.rms_norm_eps
        self.per_layer_model_projection = nn.Linear(
            args.hidden_size, self.n_layers * self.width, bias=False)
        self.per_layer_projection_norm = RMSNorm(self.width, eps=self.eps)

    def layer_input(self, x: mx.array, i: int) -> mx.array:
        lo, hi = i * self.width, (i + 1) * self.width
        proj = self.per_layer_model_projection
        if type(proj) is nn.Linear:
            y = x @ proj.weight[lo:hi].T
        elif type(proj) is nn.QuantizedLinear:
            y = mx.quantized_matmul(
                x, proj.weight[lo:hi], proj.scales[lo:hi],
                None if proj.biases is None else proj.biases[lo:hi],
                transpose=True, group_size=proj.group_size, bits=proj.bits,
                mode=getattr(proj, "mode", "affine"))
        elif getattr(proj, "mode", None) == "kquant" and "out_scale" not in proj:
            # Wire bytes hold one row per output, so a row range slices.
            import mlx_kquant as kq
            y = kq.quantized_matmul(x, proj["weight"][lo:hi], proj["scales"],
                                    proj.kquant_type, transpose=True)
        else:
            # A codec whose rows cannot be sliced here: the full projection.
            y = proj(x)[..., lo:hi]
        return self.per_layer_projection_norm(y * self.scale)


class PerLayerBlock(nn.Module):
    def __init__(self, args: TextArgs):
        super().__init__()
        h, w = args.hidden_size, args.hidden_size_per_layer_input
        self.per_layer_input_gate = nn.Linear(h, w, bias=False)
        self.per_layer_projection = nn.Linear(w, h, bias=False)
        self.post_per_layer_input_norm = RMSNorm(h, eps=args.rms_norm_eps)

    def __call__(self, x: mx.array, per_layer_input: mx.array) -> mx.array:
        g = nn.gelu_approx(self.per_layer_input_gate(x)) * per_layer_input
        return x + self.post_per_layer_input_norm(self.per_layer_projection(g))


class EncoderLayer(nn.Module):
    def __init__(self, args: TextArgs, layer_idx: int):
        super().__init__()
        h, eps = args.hidden_size, args.rms_norm_eps
        self.self_attn = Attention(args, layer_idx)
        self.mlp = MLP(args)
        self.input_layernorm = RMSNorm(h, eps=eps)
        self.post_attention_layernorm = RMSNorm(h, eps=eps)
        self.pre_feedforward_layernorm = RMSNorm(h, eps=eps)
        self.post_feedforward_layernorm = RMSNorm(h, eps=eps)
        self.ple_block = PerLayerBlock(args)
        self.layer_scalar = mx.ones((1,))

    def __call__(self, x, pad, per_layer_input):
        x = x + self.post_attention_layernorm(
            self.self_attn(self.input_layernorm(x), pad))
        x = x + self.post_feedforward_layernorm(
            self.mlp(self.pre_feedforward_layernorm(x)))
        x = self.ple_block(x, per_layer_input)
        return x * self.layer_scalar.astype(x.dtype)


class TextModel(nn.Module):
    def __init__(self, args: TextArgs):
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.ple = PerLayerInputs(args)
        self.layers = [EncoderLayer(args, i)
                       for i in range(args.num_hidden_layers)]
        self.norm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.embedding_projection = nn.Linear(
            args.hidden_size, args.embedding_dim, bias=False)

    def embed(self, input_ids: mx.array) -> mx.array:
        """Token embeddings times sqrt(hidden_size), the scale rounded to the
        activation dtype first, as transformers rounds it to the weight dtype.
        In float32 the scale is exact, as in llama.cpp."""
        h = self.embed_tokens(input_ids)
        return h * mx.array(self.args.hidden_size ** 0.5, h.dtype)

    def __call__(self, inputs_embeds: mx.array,
                 attention_mask: mx.array | None = None) -> mx.array:
        """The normed last hidden states ``[B, L, hidden_size]``."""
        pad = None
        if attention_mask is not None:
            pad = (attention_mask != 0)[:, None, None, :]
        h = inputs_embeds
        for i, layer in enumerate(self.layers):
            h = layer(h, pad, self.ple.layer_input(inputs_embeds, i))
        return self.norm(h)


class Towers(nn.Module):
    """The Gemma 4 vision and audio towers and their projections into the
    text width. Attribute names follow transformers, so checkpoint keys
    ``vision_tower.*``, ``embed_vision.*``, ``audio_tower.*`` and
    ``embed_audio.*`` load into this module unchanged."""

    def __init__(self, text_hidden: int, vision_config: dict | None,
                 audio_config: dict | None):
        super().__init__()
        from gmlx._vendor.mlx_vlm.models.gemma4.audio import AudioEncoder
        from gmlx._vendor.mlx_vlm.models.gemma4.config import (
            AudioConfig,
            VisionConfig,
        )
        from gmlx._vendor.mlx_vlm.models.gemma4.gemma4 import MultimodalEmbedder
        from gmlx._vendor.mlx_vlm.models.gemma4.vision import VisionModel

        self.vision_config = self.audio_config = None
        if vision_config is not None:
            vc = VisionConfig.from_dict(vision_config)
            self.vision_config = vc
            self.vision_tower = VisionModel(vc)
            self.embed_vision = MultimodalEmbedder(
                vc.hidden_size, text_hidden, eps=vc.rms_norm_eps)
        if audio_config is not None:
            ac = AudioConfig.from_dict(audio_config)
            self.audio_config = ac
            self.audio_tower = AudioEncoder(ac)
            self.embed_audio = MultimodalEmbedder(
                ac.output_proj_dims or ac.hidden_size, text_hidden,
                eps=ac.rms_norm_eps)

    def sanitize(self, weights: dict[str, mx.array]) -> dict[str, mx.array]:
        """transformers tower weights in this module's layout: the audio
        convolutions move their input channels last, as MLX convolutions
        take them. A weight already in MLX layout passes unchanged."""
        out = {}
        for k, v in weights.items():
            if "rotary_emb" in k:
                continue
            if (k.endswith("conv.weight") and "subsample_conv_projection" in k
                    and v.ndim == 4):
                want_in = (1 if ".layer0." in k
                           else self.audio_config.subsampling_conv_channels[0])
                if v.shape[-1] != want_in:
                    v = v.transpose(0, 2, 3, 1)
            elif k.endswith("depthwise_conv1d.weight") and v.ndim == 3:
                if v.shape[-1] != 1:
                    v = v.transpose(0, 2, 1)
            out[k] = v
        return out

    @property
    def has_vision(self) -> bool:
        return self.vision_config is not None

    @property
    def has_audio(self) -> bool:
        return self.audio_config is not None

    def encode_images(self, pixel_values: mx.array) -> mx.array:
        """Soft tokens ``[n, text_hidden]`` of same-sized images or video
        frames ``[B, 3, H, W]`` (values in [0, 1]), in image order."""
        feats = self.vision_tower(pixel_values)          # [1, n, vision_hidden]
        return self.embed_vision(feats)[0]

    def encode_audio(self, features: mx.array, valid: mx.array) -> mx.array:
        """Soft tokens ``[k, text_hidden]`` of one clip: ``features`` is
        ``[1, frames, mel]`` and ``valid`` ``[1, frames]``, True at real
        frames. Padded positions are dropped."""
        enc, pad = self.audio_tower(features, mx.logical_not(valid))
        keep = int(mx.logical_not(pad[0]).astype(mx.int32).sum().item())
        # The tower's mask is a prefix of real frames, then padding.
        return self.embed_audio(enc[:, :keep])[0]


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.language_model = TextModel(args.text)
        self.towers: Towers | None = None

    @property
    def placeholder_ids(self) -> tuple[int, int, int]:
        """The image, video and audio placeholder token ids."""
        a = self.args
        return a.image_token_id, a.video_token_id, a.audio_token_id

    def merge_media(self, input_ids: mx.array,
                    media: dict[int, mx.array]) -> mx.array:
        """Input embeddings ``[1, L, hidden]`` of ``input_ids`` ``[1, L]``
        with each placeholder id's rows replaced, in order, by the rows of
        ``media[id]`` (soft tokens from the towers, which are not scaled).
        The placeholders first embed the pad id, as in transformers."""
        mm = mx.zeros(input_ids.shape, dtype=mx.bool_)
        for tok in self.placeholder_ids:
            mm = mx.logical_or(mm, input_ids == tok)
        text_ids = mx.where(mm, self.args.text.pad_token_id, input_ids)
        h = self.language_model.embed(text_ids)
        for tok, feats in media.items():
            where = (input_ids[0] == tok)
            n = int(where.astype(mx.int32).sum().item())
            if n != feats.shape[0]:
                raise ValueError(
                    f"{n} placeholder tokens {tok} for {feats.shape[0]} soft tokens")
            if n == 0:
                continue
            # Row r of h takes soft token (number of placeholders before r).
            order = mx.cumsum(where.astype(mx.int32)) - 1
            src = feats.astype(h.dtype)[mx.maximum(order, 0)]
            h = mx.where(where[None, :, None], src[None], h)
        return h

    def __call__(self, input_ids: mx.array | None = None,
                 attention_mask: mx.array | None = None, *,
                 inputs_embeds: mx.array | None = None,
                 media: dict[int, mx.array] | None = None) -> EncoderOutput:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("pass exactly one of input_ids or inputs_embeds")
        if inputs_embeds is None:
            assert input_ids is not None
            if media:
                inputs_embeds = self.merge_media(input_ids, media)
            else:
                inputs_embeds = self.language_model.embed(input_ids)
        hidden = self.language_model(inputs_embeds, attention_mask)
        # Mean over the tokens in float32, then the projection: it is
        # linear, so this equals projecting each token and taking the mean.
        h = hidden.astype(mx.float32)
        if attention_mask is None:
            pooled = h.mean(axis=1)
        else:
            m = attention_mask.astype(mx.float32)[..., None]
            pooled = (h * m).sum(axis=1) / mx.maximum(m.sum(axis=1), 1e-9)
        # A float projection runs in float32; a quantized one takes the
        # activation dtype its kernels expect.
        proj = self.language_model.embedding_projection
        x = pooled if type(proj) is nn.Linear else pooled.astype(hidden.dtype)
        vec = proj(x).astype(mx.float32)
        return EncoderOutput(last_hidden_state=hidden,
                             text_embeds=l2_normalize(vec))

    @property
    def layers(self):
        return self.language_model.layers

    @staticmethod
    def sanitize(weights: dict[str, Any]) -> dict[str, Any]:
        return weights
