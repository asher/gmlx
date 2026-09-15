# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Asher Feldman
"""Vendored mlx-vlm model for DeepSeek-V4.1-Flash-Vision (mmproj arch
``deepseek4-vision``).

Pairs the V4.1 text tower (:mod:`gmlx.models.deepseek_v41.model`) with the
DeepSeek ViT + 3x3 aligner (:mod:`gmlx.models.deepseek_v4.vision`, whose
math V4.1 shares) in the ``language_model`` / ``vision_tower`` shape
mlx-vlm's generate stack expects.

Image turns (reference ``inference/image_processor.py``): the processor
expands each ``<|deepseek_image|>`` placeholder into

    START (IMAGE * n_llm_w + NEWLINE) * n_llm_h END

as sentinel ids past the vocab, one per block type. Aligner rows land on
the IMAGE slots in reading order and the other three take their learned
vectors. Every position in the block is an image token to the text tower,
which is what the reference means by ``image_mask = token_types >= 0``:
the MoE reads its VL routing bias there and the engram layers write no
n-gram. V4.1 attends causally, so unlike V4 a block needs no lead pads,
no interleave and no in-block mask, and it may be split across prefill
chunks.
"""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass, field
from typing import List, Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_vlm.models.base import BaseModelConfig, InputEmbeddingsFeatures

from ..deepseek_v4.vision import DeepseekV4VisionModel, VisionConfig
from .image_block import (
    IMAGE,
    IMAGE_END,
    IMAGE_NEW_LINE,
    IMAGE_START,
    N_BLOCK_TYPES,
)
from .model import DeepseekV41Model, ModelArgs
from .model import Model as _TextModel
from .model import ensure_registered as _text_ensure_registered

TextConfig = ModelArgs

MODEL_TYPE = "deepseek_v41_vl"

__all__ = [
    "LanguageModel",
    "MODEL_TYPE",
    "Model",
    "ModelConfig",
    "TextConfig",
    "VisionConfig",
    "ensure_registered",
]


def ensure_registered() -> None:
    """Make ``mlx_vlm.models.deepseek_v41_vl`` resolve, preferring upstream."""
    _text_ensure_registered()
    name = f"mlx_vlm.models.{MODEL_TYPE}"
    if name not in sys.modules:
        try:
            importlib.import_module(name)  # upstream wins
        except ImportError:
            sys.modules[name] = sys.modules[__name__]


@dataclass
class ModelConfig(BaseModelConfig):
    text_config: TextConfig = None
    vision_config: VisionConfig = field(default_factory=VisionConfig)
    model_type: str = MODEL_TYPE
    image_token_id: int = 129264
    image_token_index: Optional[int] = None
    vocab_size: int = 129280
    eos_token_id: Optional[List[int]] = None
    media_token_ids: List[int] = field(default_factory=list)

    def __post_init__(self):
        if self.image_token_index is None:
            self.image_token_index = self.image_token_id
        if not self.media_token_ids:
            self.media_token_ids = [
                int(self.vocab_size) + t for t in range(N_BLOCK_TYPES)]
        tc = self.text_config
        if isinstance(tc, dict):
            tc["media_token_ids"] = list(self.media_token_ids)
        elif tc is not None and hasattr(tc, "media_token_ids"):
            tc.media_token_ids = list(self.media_token_ids)


class LanguageModel(nn.Module):
    """The V4.1 text tower under mlx-vlm's language-model contract."""

    def __init__(self, config: TextConfig):
        super().__init__()
        self.config = config
        self.args = config
        self.model_type = config.model_type
        self.model = DeepseekV41Model(config)
        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(
                config.hidden_size, config.vocab_size, bias=False)

    def __call__(self, inputs, cache=None, inputs_embeds=None, mask=None,
                 n_to_process=None, **kwargs):
        # mlx-vlm's chunked prefill calls this directly, chunk by chunk, so
        # the image mask comes from the ids rather than from the container.
        # Decode ids are sampled, so they never carry a block sentinel; a
        # text chunk passes no mask at all and takes the text-only paths.
        del mask, n_to_process, kwargs
        image_mask = None
        if inputs.shape[-1] > 1:
            media = inputs >= self.config.vocab_size
            if bool(mx.any(media)):
                image_mask = media
        out = self.model(inputs, cache, input_embeddings=inputs_embeds,
                         image_mask=image_mask)
        logits = (self.model.embed_tokens.as_linear(out)
                  if self.config.tie_word_embeddings else self.lm_head(out))
        from mlx_vlm.models.base import LanguageModelOutput

        return LanguageModelOutput(logits=logits)

    @property
    def layers(self):
        return self.model.layers

    @property
    def cast_predicate(self):
        return _TextModel.cast_predicate.fget(self)

    def make_cache(self):
        return _TextModel.make_cache(self)

    def sanitize(self, weights):
        return _TextModel.sanitize(self, weights)


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.vision_tower = DeepseekV4VisionModel(config.vision_config)
        self.language_model = LanguageModel(config.text_config)
        tc = self.language_model.config
        if hasattr(tc, "media_token_ids") and not tc.media_token_ids:
            tc.media_token_ids = list(config.media_token_ids)
        dim = config.text_config.hidden_size
        self.image_start = mx.zeros((dim,))
        self.image_newline = mx.zeros((dim,))
        self.image_end = mx.zeros((dim,))

    # --- image features -----------------------------------------------------

    def _image_features(self, pixel_values: mx.array, image_meta) -> mx.array:
        """Patches + per-image (n_vit_h, n_vit_w, n_llm_h, n_llm_w) ->
        aligner rows for every IMAGE slot, in stream order.

        Images run one at a time: the rope tables and the aligner grid are
        per-image."""
        dtype = self.vision_tower.patch_embed.proj.weight.dtype
        feats = []
        start = 0
        for row in image_meta:
            n_vit_h, n_vit_w, n_llm_h, n_llm_w = (int(v) for v in row)
            n = n_vit_h * n_vit_w
            patches = pixel_values[start:start + n].astype(dtype)
            start += n
            rows = self.vision_tower(patches, n_vit_h, n_vit_w)
            if rows.shape[0] != n_llm_h * n_llm_w:
                raise ValueError(
                    f"aligner produced {rows.shape[0]} rows for a "
                    f"{n_llm_h}x{n_llm_w} grid")
            feats.append(rows)
        return feats[0] if len(feats) == 1 else mx.concatenate(feats, axis=0)

    def get_input_embeddings(
        self,
        input_ids: Optional[mx.array] = None,
        pixel_values: Optional[mx.array] = None,
        image_meta=None,
        **kwargs,
    ):
        vocab = int(self.config.vocab_size)
        is_media = input_ids >= vocab
        embed = self.language_model.model.embed_tokens
        if pixel_values is None:
            if bool(mx.any(is_media)):
                raise ValueError(
                    "image block ids in the prompt but no pixel_values")
            return InputEmbeddingsFeatures(inputs_embeds=embed(input_ids))
        if image_meta is None:
            raise ValueError(
                "deepseek_v41_vl needs image_meta alongside pixel_values")
        features = kwargs.get("cached_image_features")
        if features is None:
            features = self._image_features(pixel_values, image_meta)
        embeds = embed(mx.where(is_media, mx.zeros_like(input_ids), input_ids))
        embeds = self.merge_input_ids_with_image_features(
            vocab + IMAGE, features, embeds, input_ids)
        for tid, vec in (
            (IMAGE_START, self.image_start),
            (IMAGE_NEW_LINE, self.image_newline),
            (IMAGE_END, self.image_end),
        ):
            embeds = mx.where(
                (input_ids == vocab + tid)[..., None],
                vec.astype(embeds.dtype), embeds)
        return InputEmbeddingsFeatures(inputs_embeds=embeds)

    @staticmethod
    def merge_input_ids_with_image_features(
        image_token_index, image_features, inputs_embeds, input_ids
    ):
        """Scatter ``image_features`` onto the IMAGE slots, in order."""
        if image_features.ndim == 3 and image_features.shape[0] == 1:
            image_features = image_features.squeeze(0)
        positions = input_ids == image_token_index
        n_slots = int(mx.sum(positions).item())
        if n_slots != image_features.shape[0]:
            raise ValueError(
                f"{n_slots} IMAGE slots but {image_features.shape[0]} "
                "aligner rows")
        if n_slots == 0:
            return inputs_embeds
        features = image_features.astype(inputs_embeds.dtype)
        rank = mx.cumsum(positions.astype(mx.int32).reshape(-1)) - 1
        gathered = mx.take(features, mx.maximum(rank, 0), axis=0)
        gathered = gathered.reshape(inputs_embeds.shape)
        return mx.where(positions[..., None], gathered, inputs_embeds)

    # --- container plumbing -------------------------------------------------

    _LM_PREFIX = "language_model."

    def sanitize(self, weights):
        """Final (remapped) names in, final names out: the text tower's
        sanitize runs on the ``language_model.`` subtree; the vision tower
        and the sentinel embeddings pass through."""
        p = self._LM_PREFIX
        text = {k[len(p):]: v for k, v in weights.items() if k.startswith(p)}
        out = {k: v for k, v in weights.items() if not k.startswith(p)}
        for k, v in self.language_model.sanitize(text).items():
            out[p + k] = v
        return out

    @property
    def layers(self):
        return self.language_model.model.layers

    def make_cache(self):
        return self.language_model.make_cache()

    def __call__(
        self,
        input_ids: mx.array,
        pixel_values: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache=None,
        **kwargs,
    ):
        features = self.get_input_embeddings(input_ids, pixel_values, **kwargs)
        return self.language_model(
            input_ids, cache=cache, inputs_embeds=features.inputs_embeds)
