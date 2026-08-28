# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Asher Feldman
"""Vendored mlx-vlm model for GLM-5.3-Flash vision (mmproj ``glm5next``).

mlx-vlm has no glm5_next package; this container pairs the vendored text
tower (:mod:`gmlx.models.glm5_next.model`) with the GLM-OCR ViT + projector
(:mod:`gmlx.models.glm5_next.vision`) in the ``language_model`` /
``vision_tower`` shape mlx-vlm's generate stack expects.

The text tower is NoPE, so unlike the qwen2vl-family containers there is no
position/M-RoPE machinery: image features splice into the embedding stream
at the ``<|image|>`` placeholder positions and the text model runs
unchanged. The processor pre-shuffles patches into consecutive-4 2x2
blocks; ``image_grid_thw`` rides along for the tower's rope + merger.
"""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass, field
from typing import List, Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_vlm.models.base import (
    BaseModelConfig,
    InputEmbeddingsFeatures,
    LanguageModelOutput,
)

from .model import Glm5NextModel, ModelArgs
from .model import Model as _TextModel
from .model import ensure_registered as _text_ensure_registered
from .mtp import Glm5NextSpecHooks, _SpecOutput
from .vision import Glm5NextVisionModel, VisionConfig

TextConfig = ModelArgs

__all__ = [
    "LanguageModel",
    "Model",
    "ModelConfig",
    "TextConfig",
    "VisionConfig",
    "ensure_registered",
]


def ensure_registered() -> None:
    """Make ``mlx_vlm.models.glm5_next`` resolve, preferring upstream."""
    _text_ensure_registered()
    if "mlx_vlm.models.glm5_next" not in sys.modules:
        try:
            importlib.import_module("mlx_vlm.models.glm5_next")  # upstream wins
        except ImportError:
            sys.modules["mlx_vlm.models.glm5_next"] = sys.modules[__name__]


@dataclass
class ModelConfig(BaseModelConfig):
    text_config: TextConfig = None
    vision_config: VisionConfig = field(default_factory=VisionConfig)
    model_type: str = "glm5_next"
    image_token_id: int = 154854
    image_token_index: Optional[int] = None
    vocab_size: int = 154880
    eos_token_id: Optional[List[int]] = None

    def __post_init__(self):
        if self.image_token_index is None:
            self.image_token_index = self.image_token_id


class LanguageModel(Glm5NextSpecHooks, nn.Module):
    """The vendored text tower under mlx-vlm's language-model contract,
    carrying the same speculative hooks as the text-only MTP target so
    ``--mmproj`` and text-only MTP compose on one model."""

    def __init__(self, config: TextConfig):
        super().__init__()
        self.config = config
        self.args = config
        self.model_type = config.model_type
        self.model = Glm5NextModel(config)
        if config.tie_word_embeddings:
            self.lm_head = None
        else:
            self.lm_head = nn.Linear(
                config.hidden_size, config.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        inputs_embeds: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache=None,
        return_hidden: bool = False,
        return_shared_kv: bool = False,
        **kwargs,
    ):
        # The backbone builds its own KDA/MLA masks from the caches.
        del mask, kwargs
        if return_hidden or return_shared_kv:
            normed, raw = self.model(
                inputs, cache, return_raw_hidden=True,
                input_embeddings=inputs_embeds)
            return _SpecOutput(logits=self._head(normed), hidden_states=[raw])
        h = self.model(inputs, cache=cache, input_embeddings=inputs_embeds)
        return LanguageModelOutput(logits=self._head(h))

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        return _TextModel.make_cache(self)


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.vision_tower = Glm5NextVisionModel(config.vision_config)
        self.language_model = LanguageModel(config.text_config)

    def _image_features(self, pixel_values: mx.array, grid_thw) -> mx.array:
        """Pre-shuffled patch vectors + their grids -> spliceable features.

        Images run one at a time (the rope positions and the merger's
        consecutive-4 grouping are per-image), features concatenate in
        image order - the same order the placeholders appear in."""
        dtype = self.vision_tower.patch_embed.proj.weight.dtype
        feats = []
        start = 0
        for t, gh, gw in (tuple(int(v) for v in row) for row in grid_thw):
            n = t * gh * gw
            patches = pixel_values[start:start + n].astype(dtype)
            start += n
            feats.append(self.vision_tower(patches, [(t, gh, gw)]))
        return feats[0] if len(feats) == 1 else mx.concatenate(feats, axis=0)

    def get_input_embeddings(
        self,
        input_ids: Optional[mx.array] = None,
        pixel_values: Optional[mx.array] = None,
        **kwargs,
    ):
        embeds = self.language_model.model.embed_tokens(input_ids)
        if pixel_values is None:
            return InputEmbeddingsFeatures(inputs_embeds=embeds)

        features = kwargs.get("cached_image_features")
        if features is None:
            grid_thw = kwargs.get("image_grid_thw")
            if grid_thw is None:
                raise ValueError(
                    "glm5_next vision needs image_grid_thw alongside "
                    "pixel_values")
            features = self._image_features(pixel_values, grid_thw)
        return InputEmbeddingsFeatures(
            inputs_embeds=self.merge_input_ids_with_image_features(
                self.config.image_token_index, features, embeds, input_ids))

    @staticmethod
    def merge_input_ids_with_image_features(
        image_token_index, image_features, inputs_embeds, input_ids
    ):
        """Scatter ``image_features`` onto the placeholder positions, in order."""
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
