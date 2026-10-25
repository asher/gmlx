# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Asher Feldman
"""MiniCPM-V 4.6 / 4.7 VLM container (mmproj ``minicpmv4_6`` /
``minicpmv4_7`` on LLM arch ``qwen35`` or ``qwen35moe``).

The vision tower (:mod:`.vision`) turns the packed crop patches into text
width tokens, which replace the ``<|image_pad|>`` placeholders in order.
Positions come from :mod:`.language`; ``get_input_embeddings`` always
returns ``[3, 1, L]`` positions and ``[1, 1]`` deltas, text-only rows
included, so server batches never mix position shapes.

A 4.7 mmproj carries ``v.tok_embd_sep``: the F32 text embeddings of
``</image>``, ``<slice>``, ``</slice>`` and the newline. llama.cpp appends
them after each crop, so they replace the K-quant text embedding of the
separators inside an image block.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from gmlx._vendor.mlx_vlm.models.base import (
    BaseModelConfig,
    InputEmbeddingsFeatures,
)
from gmlx._vendor.mlx_vlm.models.qwen3_vl.qwen3_vl import masked_scatter

from .language import TextConfig, canvas_tokens, language_model_class
from .rope import _visual_groups
from .vision import VisionConfig, VisionModel

__all__ = ["LanguageModel", "Model", "ModelConfig", "TextConfig", "VisionConfig"]


@dataclass
class ModelConfig(BaseModelConfig):
    text_config: object = None
    vision_config: VisionConfig = None
    model_type: str = "minicpmv4_7"
    image_token_id: int = 248056
    image_token_index: Optional[int] = None
    video_token_id: Optional[int] = None
    image_start_id: int = 248078
    image_end_id: int = 248079
    slice_start_id: int = 248088
    slice_end_id: int = 248089
    image_id_start_id: int = 248090
    image_id_end_id: int = 248091
    newline_id: int = 198
    # "canvas" for 4.7; None keeps 4.6's sequential positions.
    mrope_mode: Optional[str] = None
    sep_embeds: bool = False
    vocab_size: int = 248320
    eos_token_id: Optional[Union[int, List[int]]] = None

    def __post_init__(self):
        if self.image_token_index is None:
            self.image_token_index = self.image_token_id


def LanguageModel(text_config, config):
    """The text tower for ``text_config.model_type``."""
    return language_model_class(text_config.model_type)(text_config, config)


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.vision_tower = VisionModel(config.vision_config)
        self.language_model = LanguageModel(config.text_config, config)
        if config.sep_embeds:
            self.sep_embeds = mx.zeros((4, config.text_config.hidden_size))

    @property
    def layers(self):
        return self.language_model.model.layers

    def make_cache(self):
        return self.language_model.make_cache()

    def sanitize(self, weights):
        return weights

    def _separator_rows(self, ids: np.ndarray):
        """``(flat positions, sep_embeds rows)`` of the separators llama.cpp
        appends after each crop: ``</image>`` and ``<slice>`` after the
        overview, ``</slice>`` (with ``\\n`` and ``<slice>`` between) after
        each slice. The ``<image>`` opener stays a text token."""
        cfg = self.config
        row_of = {cfg.image_end_id: 0, cfg.slice_start_id: 1,
                  cfg.slice_end_id: 2, cfg.newline_id: 3}
        toks = canvas_tokens(cfg)
        flat, rows = [], []
        b, length = ids.shape
        for r in range(b):
            seq = ids[r].tolist()
            for _g0, _g1, frames in _visual_groups(seq, toks):
                for _t0, t_end, _f_end, slices in frames:
                    stop = (slices[-1][1] if slices else t_end) + 1
                    for j in range(t_end, min(stop, length)):
                        row = row_of.get(seq[j])
                        if row is not None:
                            flat.append(r * length + j)
                            rows.append(row)
        return flat, rows

    def get_input_embeddings(
        self,
        input_ids: Optional[mx.array] = None,
        pixel_values: Optional[mx.array] = None,
        **kwargs,
    ):
        lm = self.language_model
        lm._position_ids = None
        lm._rope_deltas = None
        mask = kwargs.get("mask", None)
        target_sizes = kwargs.get("target_sizes", None)
        embeds = lm.model.embed_tokens(input_ids)

        if pixel_values is not None or kwargs.get("cached_image_features") is not None:
            features = kwargs.get("cached_image_features", None)
            vision_cache = kwargs.get("vision_cache", None)
            key = kwargs.get("_image_key", None)
            if features is None and vision_cache is not None:
                features = vision_cache.get(key)
            if features is None:
                if target_sizes is None:
                    raise ValueError("MiniCPM-V images need target_sizes")
                features = self.vision_tower(pixel_values, target_sizes)
                if vision_cache is not None and key is not None:
                    mx.eval(features)
                    vision_cache.put(key, features)
            image_mask = input_ids == self.config.image_token_id
            n_tokens = int(image_mask.sum().item())
            if n_tokens != int(features.shape[0]):
                raise ValueError(
                    f"Image features and image tokens do not match: tokens: "
                    f"{n_tokens}, features {int(features.shape[0])}")
            image_mask = mx.broadcast_to(image_mask[..., None], embeds.shape)
            embeds = masked_scatter(embeds, image_mask, features.astype(embeds.dtype))
            if self.config.sep_embeds:
                embeds = self._splice_separators(embeds, np.asarray(input_ids))

        position_ids, rope_deltas = lm.get_rope_index(
            input_ids, None, None, mask, target_sizes=target_sizes)
        return InputEmbeddingsFeatures(
            inputs_embeds=embeds,
            position_ids=position_ids,
            rope_deltas=rope_deltas,
        )

    def _splice_separators(self, embeds: mx.array, ids: np.ndarray) -> mx.array:
        flat, rows = self._separator_rows(ids)
        if not flat:
            return embeds
        b, length, d = embeds.shape
        out = embeds.reshape(b * length, d)
        idx = mx.array(flat, dtype=mx.uint32)
        out[idx] = self.sep_embeds[mx.array(rows, dtype=mx.uint32)].astype(out.dtype)
        return out.reshape(b, length, d)

    def __call__(
        self,
        input_ids: mx.array,
        pixel_values: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache=None,
        **kwargs,
    ):
        features = self.get_input_embeddings(
            input_ids, pixel_values, mask=mask, **kwargs)
        kwargs.pop("position_ids", None)
        kwargs.pop("rope_deltas", None)
        return self.language_model(
            input_ids, inputs_embeds=features.inputs_embeds, mask=mask,
            cache=cache, position_ids=features.position_ids,
            rope_deltas=features.rope_deltas)
