# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Asher Feldman
"""Inkling 2 with its vision tower (an mmproj with projector ``inkling2v``).

Pairs the text model (:mod:`gmlx.models.inkling2.model`, with the
speculative hooks of :mod:`gmlx.models.inkling2.mtp`) with the vision tower
(:mod:`gmlx.models.inkling2.vision`) in the ``language_model`` /
``vision_tower`` shape mlx-vlm's generate stack expects.

The chat template writes ``<|content_image|>`` and one placeholder token
per image, in a block of its own. The processor repeats the placeholder
once per image row. Text rows take the embedding norm and image rows go
in as the tower made them (``inkling2.cpp``), so the container builds the
whole input and its language model reads those rows as they are.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import mlx.core as mx
import mlx.nn as nn

from gmlx._vendor.mlx_vlm.models.base import (
    BaseModelConfig,
    InputEmbeddingsFeatures,
)

from .model import ModelArgs
from .mtp import Inkling2SpecLM
from .vision import VisionConfig, VisionModel

TextConfig = ModelArgs

MODEL_TYPE = "inkling2_vl"
# The template's image placeholder and its id in the released vocabulary.
IMAGE_TOKEN = "<|unused_200054|>"
IMAGE_TOKEN_ID = 200054

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
    """Register the tool parser before a build."""
    import gmlx.models.inkling2.tools as inkling2_tools

    inkling2_tools.ensure_registered()


@dataclass
class ModelConfig(BaseModelConfig):
    text_config: TextConfig = None
    vision_config: VisionConfig = field(default_factory=VisionConfig)
    model_type: str = MODEL_TYPE
    image_token_id: int = IMAGE_TOKEN_ID
    image_token_index: Optional[int] = None
    vocab_size: int = 201024
    eos_token_id: Optional[List[int]] = None

    def __post_init__(self):
        if self.image_token_index is None:
            self.image_token_index = self.image_token_id


class LanguageModel(Inkling2SpecLM):
    """The Inkling 2 text model under mlx-vlm's language-model contract,
    with the speculative hooks, so an mmproj and a drafter load on one
    model."""

    def __init__(self, config: TextConfig):
        super().__init__(config)
        self.config = config

    def __call__(self, inputs, cache=None, inputs_embeds=None, **kwargs):
        # The container's rows are final: text rows already took the norm.
        if inputs_embeds is not None and kwargs.get("embed_mask") is None:
            kwargs["embed_mask"] = mx.zeros(inputs_embeds.shape[:-1],
                                            dtype=mx.bool_)
        return super().__call__(inputs, cache, inputs_embeds, **kwargs)


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.vision_tower = VisionModel(config.vision_config)
        self.language_model = LanguageModel(config.text_config)

    def _image_features(self, pixel_values, image_sizes) -> mx.array:
        """A padded batch [N, C, H, W] and each image's (h, w) -> the rows
        of every image, in prompt order. Images run one at a time."""
        feats = []
        for i, (h, w) in enumerate(image_sizes):
            image = pixel_values[i, :, :h, :w].astype(mx.float32)
            feats.append(self.vision_tower(image))
        return feats[0] if len(feats) == 1 else mx.concatenate(feats, axis=0)

    def get_input_embeddings(
        self,
        input_ids: Optional[mx.array] = None,
        pixel_values: Optional[mx.array] = None,
        **kwargs,
    ):
        lm = self.language_model.model
        embeds = lm.embed_norm(lm.embed_tokens(input_ids))
        if pixel_values is None:
            return InputEmbeddingsFeatures(inputs_embeds=embeds)
        features = kwargs.get("cached_image_features")
        if features is None:
            sizes = kwargs.get("image_sizes")
            if sizes is None:
                sizes = [pixel_values.shape[-2:]] * pixel_values.shape[0]
            features = self._image_features(
                pixel_values, [(int(h), int(w)) for h, w in sizes])
        return InputEmbeddingsFeatures(inputs_embeds=self._scatter(
            features, embeds, input_ids == self.config.image_token_index))

    @staticmethod
    def _scatter(features, inputs_embeds, is_image):
        """``features`` onto the image positions, in order."""
        if features.ndim == 3 and features.shape[0] == 1:
            features = features.squeeze(0)
        n_slots = int(mx.sum(is_image).item())
        if n_slots != features.shape[0]:
            raise ValueError(
                f"{n_slots} image placeholder tokens but {features.shape[0]} "
                "rows from the vision tower")
        if n_slots == 0:
            return inputs_embeds
        rank = mx.cumsum(is_image.astype(mx.int32).reshape(-1)) - 1
        rows = mx.take(features.astype(inputs_embeds.dtype),
                       mx.maximum(rank, 0), axis=0)
        return mx.where(is_image[..., None],
                        rows.reshape(inputs_embeds.shape), inputs_embeds)

    # Container plumbing.

    _LM_PREFIX = "language_model."

    def sanitize(self, weights):
        """Final (remapped) names in, final names out: the text model's
        sanitize runs on the ``language_model.`` subtree."""
        p = self._LM_PREFIX
        text = {k[len(p):]: v for k, v in weights.items() if k.startswith(p)}
        out = {k: v for k, v in weights.items() if not k.startswith(p)}
        for k, v in self.language_model.sanitize(text).items():
            out[p + k] = v
        return out

    @property
    def layers(self):
        return self.language_model.layers

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
