# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Asher Feldman
"""LFM3-VL and LFM3-Omni with image and video input (mmproj projector
``lfm3vl``).

The text half is the class the text-only path uses
(:mod:`gmlx.models.lfm3.model`), in the ``language_model`` shape mlx-vlm's
generate stack expects. The vision half is the tower and projector of
:mod:`gmlx.models.lfm3.vision`.

The processor expands each ``<|image|>`` and ``<|video|>`` the chat
template writes into one placeholder per soft token
(:mod:`gmlx.models.lfm3.processing`). The projected features replace the
placeholder rows of the token embeddings, images first and then videos,
each in prompt order. The text model runs unchanged: its positions are
the token positions.
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

from .model import Model as TextModel
from .model import ModelArgs
from .vision import (CLIP_PATCHES, MultiModalProjector, VisionConfig,
                     VisionModel)

TextConfig = ModelArgs

MODEL_TYPES = ("lfm3_omni", "lfm3_vl")

__all__ = ["LanguageModel", "Model", "ModelConfig", "TextConfig",
           "VisionConfig"]


@dataclass
class ModelConfig(BaseModelConfig):
    text_config: TextConfig = None  # pyright: ignore[reportAssignmentType]
    vision_config: VisionConfig = field(default_factory=VisionConfig)
    model_type: str = "lfm3_omni"
    image_token_id: int = 125030
    video_token_id: int = 125029
    image_token_index: Optional[int] = None
    vocab_size: int = 128000
    eos_token_id: Optional[List[int]] = None

    def __post_init__(self):
        if self.image_token_index is None:
            self.image_token_index = self.image_token_id


class LanguageModel(TextModel):
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
        # The backbone builds its own attention and conv masks from the
        # caches.
        del mask
        h = self.model(inputs, cache, inputs_embeds)
        if kwargs.get("skip_logits", False):
            return LanguageModelOutput(logits=None)
        return LanguageModelOutput(logits=self.head(h))


def scatter_features(token_id: int, features: mx.array,
                     inputs_embeds: mx.array, input_ids: mx.array,
                     what: str) -> mx.array:
    """Put ``features`` on the rows of ``token_id``, in order."""
    positions = input_ids == token_id
    slots = int(mx.sum(positions).item())
    if slots != features.shape[0]:
        raise ValueError(
            f"{slots} {what} placeholder tokens but {features.shape[0]} "
            f"{what} features")
    if slots == 0:
        return inputs_embeds
    features = features.astype(inputs_embeds.dtype)
    rank = mx.cumsum(positions.astype(mx.int32).reshape(-1)) - 1
    gathered = mx.take(features, mx.maximum(rank, 0), axis=0)
    gathered = gathered.reshape(inputs_embeds.shape)
    return mx.where(positions[..., None], gathered, inputs_embeds)


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.vision_tower = VisionModel(config.vision_config)
        self.multi_modal_projector = MultiModalProjector(config.vision_config)
        # The serve engine runs the language model alone and reads the
        # media ids off its config (gmlx.cache.apc_media).
        for name in ("image_token_id", "video_token_id"):
            setattr(config.text_config, name, getattr(config, name))
        self.language_model = LanguageModel(config.text_config)

    def vision_features(self, pixel_values: mx.array, grid_thw) -> mx.array:
        """The flattened patches of every image or video, in order, to
        their projected features, concatenated in the same order. Each
        image runs on its own, and a clip a few slices at a time. Each
        pass is evaluated before the next starts."""
        feats = []
        start = 0
        rows = [tuple(int(v) for v in row) for row in
                (grid_thw.tolist() if hasattr(grid_thw, "tolist")
                 else grid_thw)]
        for t, gh, gw in rows:
            frame = gh * gw
            step = max(1, CLIP_PATCHES // frame)
            for t0 in range(0, t, step):
                count = min(step, t - t0)
                hidden = self.vision_tower(
                    pixel_values[start:start + count * frame],
                    (count, gh, gw))
                feats.append(self.multi_modal_projector(hidden))
                start += count * frame
                if len(rows) > 1 or t > step:
                    mx.eval(feats[-1])
        if start != pixel_values.shape[0]:
            raise ValueError(
                f"the grids hold {start} patches but {pixel_values.shape[0]} "
                "were given")
        return feats[0] if len(feats) == 1 else mx.concatenate(feats, axis=0)

    def get_input_embeddings(
        self,
        input_ids: Optional[mx.array] = None,
        pixel_values: Optional[mx.array] = None,
        **kwargs,
    ):
        embeds = self.language_model.model.embed_tokens(input_ids)
        videos = kwargs.get("pixel_values_videos")
        if pixel_values is None and videos is None:
            return InputEmbeddingsFeatures(inputs_embeds=embeds)
        if pixel_values is not None:
            features = kwargs.get("cached_image_features")
            if features is None:
                grid = kwargs.get("image_grid_thw")
                if grid is None:
                    raise ValueError(
                        "LFM3 vision needs image_grid_thw alongside "
                        "pixel_values")
                features = self.vision_features(pixel_values, grid)
            embeds = scatter_features(
                self.config.image_token_id, features, embeds, input_ids,
                "image")
        if videos is not None:
            grid = kwargs.get("video_grid_thw")
            if grid is None:
                raise ValueError(
                    "LFM3 vision needs video_grid_thw alongside "
                    "pixel_values_videos")
            embeds = scatter_features(
                self.config.video_token_id,
                self.vision_features(videos, grid), embeds, input_ids,
                "video")
        return InputEmbeddingsFeatures(inputs_embeds=embeds)

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
