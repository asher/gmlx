# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Asher Feldman
"""Hy4 with its vision tower (an mmproj with projector ``hy_v4_vl``).

Pairs the hy_v4 text tower (:mod:`gmlx.models.hy_v4.model`) with the
vision tower and projector Hy4-flash uses
(:mod:`gmlx.models.hy_v4_flash.vision`), which the mmproj sizes: a
16 px patch ViT over a learned position grid, then RMSNorm, a 2x2 merge,
GELU and a 1x1 projection onto the text width, with one newline row
closing every row of merged cells.

The processor spells each image slot as an id past the vocab (vocab_size
+ slot kind, :mod:`gmlx.models.hy_v4_flash.positions`), so a prefill
chunk can split an image. Image slots rotate at their (t, h, w), the
interleaved M-RoPE of the VL conversion, and the walk's state rides the
last member of layer 0's cache entry.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import mlx.core as mx
import mlx.nn as nn

from gmlx._vendor.mlx_vlm.models.base import (
    BaseModelConfig,
    InputEmbeddingsFeatures,
    LanguageModelOutput,
)
from gmlx.cache.kv_cache import ArraysCache, CacheList
from gmlx.models.hy_v4_flash.positions import N_SLOT_KINDS, TEXT
from gmlx.models.hy_v4_flash.vision import VisionConfig, VisionModel

from .model import HyV4Model, ModelArgs
from .model import Model as _TextModel
from .model import ensure_registered as _text_ensure_registered

TextConfig = ModelArgs

MODEL_TYPE = "hy_v4_vl"
IMAGE_PAD_TOKEN_ID = 120120

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
    """Register the text model and its tool parser before a build."""
    import gmlx.models.hy_v4.tools as hy_v4_tools

    _text_ensure_registered()
    hy_v4_tools.ensure_registered()


@dataclass
class ModelConfig(BaseModelConfig):
    text_config: TextConfig = None
    vision_config: VisionConfig = field(default_factory=VisionConfig)
    model_type: str = MODEL_TYPE
    # The image pad token: the chat template's placeholder.
    image_token_id: int = IMAGE_PAD_TOKEN_ID
    image_token_index: Optional[int] = None
    vocab_size: int = 120832
    eos_token_id: Optional[List[int]] = None
    media_token_ids: List[int] = field(default_factory=list)

    def __post_init__(self):
        if self.image_token_index is None:
            self.image_token_index = self.image_token_id
        if not self.media_token_ids:
            self.media_token_ids = [
                int(self.vocab_size) + k for k in range(N_SLOT_KINDS)]


class LanguageModel(nn.Module):
    """The hy_v4 text tower under mlx-vlm's language-model contract."""

    def __init__(self, config: TextConfig,
                 image_token_id: int = IMAGE_PAD_TOKEN_ID):
        super().__init__()
        self.config = config
        self.args = config
        self.model_type = config.model_type
        self.image_token_id = int(image_token_id)
        self.model = HyV4Model(config)
        if config.tie_word_embeddings:
            self.lm_head = None
        else:
            self.lm_head = nn.Linear(
                config.hidden_size, config.vocab_size, bias=False)

    def __call__(self, inputs, cache=None, inputs_embeds=None, mask=None,
                 n_to_process=None, **kwargs):
        # mlx-vlm's chunked prefill calls this chunk by chunk, so the slot
        # kinds come from the ids. Only a prefill call carries embeddings:
        # a decode step's ids are sampled, never image slots.
        skip_logits = kwargs.pop("skip_logits", False)
        del mask, n_to_process, kwargs
        kinds = None
        if inputs_embeds is not None:
            vocab = int(self.config.vocab_size)
            host = inputs.reshape(-1).tolist()
            if max(host) >= vocab:
                if inputs.shape[0] != 1:
                    raise ValueError("hy_v4_vl image prompts run single-row")
                kinds = [t - vocab if t >= vocab else TEXT for t in host]
                inputs = mx.array(
                    [[self.image_token_id if t >= vocab else t
                      for t in host]], dtype=inputs.dtype)
        out = self.model(inputs, cache, input_embeddings=inputs_embeds,
                         slot_kinds=kinds)
        if skip_logits:
            return LanguageModelOutput(logits=None)
        if self.lm_head is None:
            logits = self.model.embed_tokens.as_linear(out)
        else:
            logits = self.lm_head(out)
        return LanguageModelOutput(logits=logits)

    @property
    def layers(self):
        return self.model.layers

    @property
    def cast_predicate(self):
        return _TextModel.cast_predicate.fget(self)

    @property
    def quant_predicate(self):
        return _TextModel.quant_predicate.fget(self)

    def make_cache(self):
        """The text model's cache, with the image-position walk as the last
        member of layer 0's entry. Its first slot holds a placeholder row,
        so the cache's state and batch operations never meet an empty
        member. Layer 0 has no indexer on a file without the DSA indexer."""
        caches = _TextModel.make_cache(self)
        walk = ArraysCache(2)
        walk[0] = mx.zeros((1, 1), dtype=mx.int32)
        inner = getattr(caches[0], "caches", None)
        caches[0] = CacheList(*(inner if inner is not None else (caches[0],)),
                              walk)
        return caches

    def sanitize(self, weights):
        return _TextModel.sanitize(self, weights)


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.vision_tower = VisionModel(config.vision_config)
        self.language_model = LanguageModel(
            config.text_config, config.image_token_id)
        # The serve engine reads media ids off the language model's config
        # (gmlx.cache.apc_media).
        self.language_model.config.media_token_ids = list(
            config.media_token_ids)

    def _image_features(self, pixel_values: mx.array, grids) -> mx.array:
        """Patches and each image's patch grid (n_h, n_w) -> the rows of
        every image slot, in prompt order, one image at a time."""
        dtype = self.vision_tower.patch_embed.proj.weight.dtype
        feats = []
        start = 0
        for row in grids:
            n_h, n_w = (int(v) for v in row)
            n = n_h * n_w
            feats.append(self.vision_tower(
                pixel_values[start:start + n].astype(dtype), n_h, n_w))
            start += n
        return feats[0] if len(feats) == 1 else mx.concatenate(feats, axis=0)

    def get_input_embeddings(
        self,
        input_ids: Optional[mx.array] = None,
        pixel_values: Optional[mx.array] = None,
        image_patch_grid=None,
        **kwargs,
    ):
        vocab = int(self.config.vocab_size)
        is_slot = input_ids >= vocab
        embed = self.language_model.model.embed_tokens
        if pixel_values is None:
            if bool(mx.any(is_slot)):
                raise ValueError(
                    "image slot ids in the prompt but no pixel_values")
            return InputEmbeddingsFeatures(inputs_embeds=embed(input_ids))
        if image_patch_grid is None:
            raise ValueError(
                "hy_v4_vl needs image_patch_grid alongside pixel_values")
        features = kwargs.get("cached_image_features")
        if features is None:
            features = self._image_features(pixel_values, image_patch_grid)
        embeds = embed(mx.where(is_slot, mx.zeros_like(input_ids), input_ids))
        return InputEmbeddingsFeatures(
            inputs_embeds=self._scatter(features, embeds, is_slot))

    @staticmethod
    def _scatter(features, inputs_embeds, is_slot):
        """``features`` onto the image slots, in order."""
        n_slots = int(mx.sum(is_slot).item())
        if n_slots != features.shape[0]:
            raise ValueError(
                f"{n_slots} image slots but {features.shape[0]} rows from "
                "the vision tower")
        if n_slots == 0:
            return inputs_embeds
        rank = mx.cumsum(is_slot.astype(mx.int32).reshape(-1)) - 1
        rows = mx.take(features.astype(inputs_embeds.dtype),
                       mx.maximum(rank, 0), axis=0)
        return mx.where(is_slot[..., None],
                        rows.reshape(inputs_embeds.shape), inputs_embeds)

    _LM_PREFIX = "language_model."

    def sanitize(self, weights):
        """Final (remapped) names in, final names out: the text tower's
        sanitize runs on the ``language_model.`` subtree and the vision
        tower passes through."""
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
