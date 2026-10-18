# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Asher Feldman
"""Nemotron-3.5-Bolt with image input (mmproj projector ``nemotron_omni``).

The text half is the class the text-only path uses
(:mod:`gmlx.models.nemotron_h.model`), in the ``language_model`` shape
mlx-vlm's generate stack expects. The vision half is the C-RADIO tower and
projector (:mod:`gmlx.models.nemotron_h.vision`).

The processor expands each ``<image>`` the chat template writes into
``<img>``, one ``<image>`` per soft token and ``</img>``, the stream
llama.cpp's mtmd builds around the embeddings. The features replace the
``<image>`` rows of the token embeddings, and the text model runs
unchanged: it has no position encoding to adjust.
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

from .model import ModelArgs
from .mtp import NemotronHSpecLM
from .vision import VisionConfig, VisionModel

TextConfig = ModelArgs

__all__ = ["LanguageModel", "Model", "ModelConfig", "TextConfig",
           "VisionConfig"]


@dataclass
class ModelConfig(BaseModelConfig):
    text_config: TextConfig = None  # pyright: ignore[reportAssignmentType]
    vision_config: VisionConfig = field(default_factory=VisionConfig)
    model_type: str = "nemotron_h_omni"
    image_token_id: int = 18
    image_token_index: Optional[int] = None
    vocab_size: int = 131072
    eos_token_id: Optional[List[int]] = None

    def __post_init__(self):
        if self.image_token_index is None:
            self.image_token_index = self.image_token_id


class LanguageModel(NemotronHSpecLM):
    """The text model with the speculative hooks, so a drafter runs on text
    turns. Image turns take the plain path with the merged embeddings."""

    def __init__(self, config: TextConfig):
        super().__init__(config)
        # The serve engine and the APC guard read the text config here.
        self.config = config

    def __call__(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        inputs: mx.array,
        inputs_embeds: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache=None,
        **kwargs,
    ):
        # The backbone builds its own attention and SSM masks from the caches.
        del mask
        if kwargs.get("return_hidden") or kwargs.get("return_shared_kv"):
            return super().__call__(inputs, cache=cache, **kwargs)
        skip_logits = kwargs.get("skip_logits", False)
        h = self.backbone(inputs, cache=cache, input_embeddings=inputs_embeds)
        return LanguageModelOutput(
            logits=None if skip_logits else self.lm_head(h))


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.vision_tower = VisionModel(config.vision_config)
        self.language_model = LanguageModel(config.text_config)

    def _image_features(self, pixel_values: mx.array, grid_thw) -> mx.array:
        """Flattened patches of every image, in image order, to their
        projected features, concatenated in the same order."""
        feats = []
        start = 0
        for _t, gh, gw in (tuple(int(v) for v in row) for row in grid_thw):
            n = gh * gw
            feats.append(self.vision_tower(
                pixel_values[start:start + n], (gh, gw)))
            start += n
        return feats[0] if len(feats) == 1 else mx.concatenate(feats, axis=0)

    def get_input_embeddings(
        self,
        input_ids: Optional[mx.array] = None,
        pixel_values: Optional[mx.array] = None,
        **kwargs,
    ):
        embeds = self.language_model.backbone.embeddings(input_ids)
        if pixel_values is None:
            return InputEmbeddingsFeatures(inputs_embeds=embeds)
        features = kwargs.get("cached_image_features")
        if features is None:
            grid_thw = kwargs.get("image_grid_thw")
            if grid_thw is None:
                raise ValueError(
                    "nemotron_h_omni vision needs image_grid_thw alongside "
                    "pixel_values")
            features = self._image_features(pixel_values, grid_thw)
        return InputEmbeddingsFeatures(
            inputs_embeds=self.merge_input_ids_with_image_features(
                self.config.image_token_index, features, embeds, input_ids))

    @staticmethod
    def merge_input_ids_with_image_features(
        image_token_index, image_features, inputs_embeds, input_ids
    ):
        """Scatter ``image_features`` onto the placeholder rows, in order."""
        if image_features.ndim == 3 and image_features.shape[0] == 1:
            image_features = image_features.squeeze(0)
        positions = input_ids == image_token_index
        n_slots = int(mx.sum(positions).item())
        if n_slots != image_features.shape[0]:
            raise ValueError(
                f"{n_slots} image placeholder tokens but "
                f"{image_features.shape[0]} image features")
        if n_slots == 0:
            return inputs_embeds
        features = image_features.astype(inputs_embeds.dtype)
        rank = mx.cumsum(positions.astype(mx.int32).reshape(-1)) - 1
        gathered = mx.take(features, mx.maximum(rank, 0), axis=0)
        gathered = gathered.reshape(inputs_embeds.shape)
        return mx.where(positions[..., None], gathered, inputs_embeds)

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
