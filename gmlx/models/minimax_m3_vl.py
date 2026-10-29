# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Asher Feldman
"""MiniMax-M3.1-Flash with image input (mmproj projector ``minimax_m3``).

The text half is the class the text-only path uses
(:class:`gmlx.models.minimax_m3.Model`), so sparse attention, the kquant
swap and expert streaming work as they do without ``--mmproj``. The vision
half is the vendored mlx-vlm 0.7.3 ViT (``minimax_m3_vl.vision``) and its
two projectors, in llama.cpp's order (``tools/mtmd/models/minimax-m3.cpp``):
the ViT with no post-norm, ``mm.1`` -> GELU -> ``mm.2`` on each patch, then
the four patches of each 2x2 window concatenated through
``mm.merger.fc1`` -> GELU -> ``fc2``.

The processor writes the patches of each 2x2 window next to each other, so
the merge is a reshape. Each ``]<]image[>[`` the chat template writes
becomes ``]<]start of image[>[``, one ``]<]image[>[`` per merged token and
``]<]end of image[>[``. The features replace the ``]<]image[>[`` rows, and
the text tower runs on plain 1-D positions, as with text.

The container registers under ``minimax_m3_vision``: the vendored copy
ships its own ``minimax_m3_vl`` package (a text tower with another MoE
layout), and an owned type never shares a name with a vendored package.
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
from gmlx._vendor.mlx_vlm.models.minimax_m3_vl.config import VisionConfig
from gmlx._vendor.mlx_vlm.models.minimax_m3_vl.minimax_m3_vl import (
    MiniMaxProjector,
)
from gmlx._vendor.mlx_vlm.models.minimax_m3_vl.vision import VisionModel

from . import minimax_m3 as _text
from .minimax_m3 import ModelArgs

TextConfig = ModelArgs

MODEL_TYPE = "minimax_m3_vision"

__all__ = ["LanguageModel", "MODEL_TYPE", "Model", "ModelConfig",
           "TextConfig", "VisionConfig", "ensure_registered"]


def ensure_registered() -> None:
    """Register the text model's cache classes before a build."""
    _text.ensure_registered()


@dataclass
class ModelConfig(BaseModelConfig):
    text_config: TextConfig = None  # pyright: ignore[reportAssignmentType]
    vision_config: VisionConfig = field(default_factory=VisionConfig)
    model_type: str = MODEL_TYPE
    # ]<]image[>[, ]<]video[>[, ]<]start of image[>[, ]<]end of image[>[
    image_token_id: int = 200025
    image_token_index: Optional[int] = None
    video_token_id: int = 200026
    vision_start_token_id: int = 200029
    vision_end_token_id: int = 200030
    projector_hidden_size: int = 6144
    projector_hidden_act: str = "gelu"
    vocab_size: int = 200064
    eos_token_id: Optional[List[int]] = None

    def __post_init__(self):
        if self.image_token_index is None:
            self.image_token_index = self.image_token_id


class LanguageModel(_text.Model):
    """The text model in mlx-vlm's ``language_model`` shape. A forward with
    ``logits_to_keep`` runs the head on the last columns only."""

    supports_logits_to_keep = True

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
        # The trunk builds its causal mask from the cache; mlx-vlm's 2-D
        # padding mask is not one.
        del mask
        h = self.model(inputs, None, cache, input_embeddings=inputs_embeds)
        if kwargs.get("skip_logits", False):
            return LanguageModelOutput(logits=None)  # pyright: ignore[reportArgumentType]
        keep = kwargs.get("logits_to_keep")
        if keep and h.shape[1] > int(keep):
            h = h[:, -int(keep):]
        if self.args.tie_word_embeddings:
            return LanguageModelOutput(logits=self.model.embed_tokens.as_linear(h))
        return LanguageModelOutput(logits=self.lm_head(h))


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        vc = config.vision_config
        hidden = config.text_config.hidden_size
        self.vision_tower = VisionModel(vc)
        self.language_model = LanguageModel(config.text_config)
        self.multi_modal_projector = MiniMaxProjector(
            vc.hidden_size, config.projector_hidden_size, hidden, True,
            config.projector_hidden_act)
        self.patch_merge_mlp = MiniMaxProjector(
            hidden * vc.spatial_merge_size ** 2, hidden, hidden, True,
            config.projector_hidden_act)

    def image_features(self, pixel_values: mx.array, grid_thw) -> mx.array:
        """Flattened patches of every image (merge-window order) to one
        feature row per merged token, in image order."""
        grid = [tuple(int(v) for v in row) for row in
                (grid_thw.tolist() if hasattr(grid_thw, "tolist") else grid_thw)]
        dtype = self.vision_tower.vision_model.embeddings.patch_embedding.weight.dtype
        hidden = self.vision_tower(pixel_values.astype(dtype), mx.array(grid))
        hidden = self.multi_modal_projector(hidden)
        window = self.config.vision_config.spatial_merge_size ** 2
        return self.patch_merge_mlp(
            hidden.reshape(-1, window * hidden.shape[-1]))

    def get_input_embeddings(
        self,
        input_ids: Optional[mx.array] = None,
        pixel_values: Optional[mx.array] = None,
        **kwargs,
    ):
        if kwargs.get("pixel_values_videos") is not None:
            raise ValueError(
                "MiniMax-M3.1 video input is not supported; pass still images")
        embeds = self.language_model.model.embed_tokens(input_ids)
        features = kwargs.get("cached_image_features")
        if pixel_values is None and features is None:
            return InputEmbeddingsFeatures(inputs_embeds=embeds)
        if features is None:
            grid_thw = kwargs.get("image_grid_thw")
            if grid_thw is None:
                raise ValueError(
                    f"{MODEL_TYPE} vision needs image_grid_thw alongside "
                    "pixel_values")
            features = self.image_features(pixel_values, grid_thw)
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

    _LM_PREFIX = "language_model."

    def sanitize(self, weights):
        """Final (remapped) names in, final names out: the text sanitize
        (indexer gating and narrowing) runs on the ``language_model.``
        subtree; the tower and projectors pass through."""
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
            input_ids, inputs_embeds=features.inputs_embeds, cache=cache,
            logits_to_keep=kwargs.get("logits_to_keep"))
