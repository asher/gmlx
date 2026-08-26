# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Asher Feldman
"""Vendored mlx-vlm model for Qwen3.8-Flash-Next VLM (mmproj ``qwen3vl_merger``
on LLM arch ``qwen4exp``).

mlx-vlm ships no qwen4_exp package. The vision half is the same Qwen3-VL ViT
the qwen3.5/3.6 pair uses (mlx-vlm's ``qwen3_5.vision.VisionModel``; the GGUF
disables deepstack), so this module reuses it wholesale; the text half is the
vendored :mod:`gmlx.qwen4_exp_model`, wrapped in the ``language_model`` shape
mlx-vlm's generate stack expects.

Positions: text-only rows keep the scalar-offset fast rope; when images are
present, ``get_rope_index`` (borrowed unbound from mlx-vlm's qwen3_5
``LanguageModel``, identical interleaved-mrope semantics) produces the
``[3, B, L]`` t/h/w ids and the wrapper threads them into the vendored
attention + QSA indexer (which cache per-token positions for block-key rope).
Decode positions continue at ``cache_offset + rope_delta`` on all three
streams. The PLE hash layer sees the raw token ids either way (image
placeholder positions hash the image token id, as in the reference).

v1 is single-image-turn oriented and B=1-serve aligned like the other
vendored VLM pairs; MTP + VLM composition is not wired (text-only MTP is).
"""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from typing import List, Optional, Union

import mlx.core as mx
import mlx.nn as nn

from mlx_vlm.models.base import (
    BaseModelConfig,
    InputEmbeddingsFeatures,
    LanguageModelOutput,
)
from mlx_vlm.models.qwen3_5.config import VisionConfig
from mlx_vlm.models.qwen3_5.language import LanguageModel as _Q35LanguageModel
from mlx_vlm.models.qwen3_5.vision import VisionModel
from mlx_vlm.models.qwen3_vl.qwen3_vl import Model as _Qwen3VLModel

from . import qwen4_exp_model as q4
from .qwen4_exp_model import ModelArgs as TextConfig


def ensure_registered() -> None:
    """Make ``mlx_vlm.models.qwen4_exp`` resolve, preferring upstream."""
    q4.ensure_registered()
    if "mlx_vlm.models.qwen4_exp" not in sys.modules:
        try:
            importlib.import_module("mlx_vlm.models.qwen4_exp")  # upstream wins
        except ImportError:
            sys.modules["mlx_vlm.models.qwen4_exp"] = sys.modules[__name__]


@dataclass
class ModelConfig(BaseModelConfig):
    text_config: TextConfig = None
    vision_config: VisionConfig = None
    model_type: str = "qwen4_exp"
    ignore_index: int = -100
    image_token_id: int = 248056
    video_token_id: int = 248057
    image_token_index: Optional[int] = None
    video_token_index: Optional[int] = None
    vision_start_token_id: int = 248053
    vision_end_token_id: int = 248054
    vocab_size: int = 248320
    eos_token_id: Optional[Union[int, List[int]]] = None

    def __post_init__(self):
        if self.image_token_index is None:
            self.image_token_index = self.image_token_id
        if self.video_token_index is None:
            self.video_token_index = self.video_token_id


class LanguageModel(nn.Module):
    """The vendored text tower in mlx-vlm's ``language_model`` shape:
    resolves mrope position ids (prefill from ``get_rope_index`` via
    ``_position_ids``; decode from ``rope_deltas``) and threads them into
    the qwen4exp backbone."""

    # Same interleaved-mrope position derivation as qwen3.5 (the configs
    # share every field it reads).
    get_rope_index = _Q35LanguageModel.get_rope_index

    def __init__(self, text_config: TextConfig, config: ModelConfig):
        super().__init__()
        self.config = config
        self.args = text_config
        self.model_type = text_config.model_type
        self.model = q4.Qwen4ExpModel(text_config)
        if not text_config.tie_word_embeddings:
            self.lm_head = nn.Linear(
                text_config.hidden_size, text_config.vocab_size, bias=False)
        self._position_ids = None
        self._rope_deltas = None

    def _head(self, out: mx.array) -> mx.array:
        if self.args.tie_word_embeddings:
            return self.model.embed_tokens.as_linear(out)
        return self.lm_head(out)

    def _resolve_positions(self, inputs: mx.array, cache,
                           position_ids, rope_deltas):
        """The ``[3, B, L]`` mrope block for this window, or ``None`` for the
        pure-text fast path (no vision in this request yet)."""
        B, L = inputs.shape
        if position_ids is not None:
            if position_ids.ndim == 2:
                position_ids = mx.broadcast_to(position_ids[None], (3, B, L))
            return position_ids
        offset = 0
        if cache is not None and cache[self.model.fa_idx] is not None:
            offset = cache[self.model.fa_idx].offset
        pids = self._position_ids
        if pids is not None and pids.ndim == 3 and pids.shape[1] == B \
                and pids.shape[-1] >= offset + L:
            return pids[:, :, offset:offset + L]
        deltas = rope_deltas if rope_deltas is not None else self._rope_deltas
        if deltas is None:
            return None  # text-only: scalar-offset rope
        pos = offset + deltas.astype(mx.int32).reshape(B, 1) \
            + mx.arange(L, dtype=mx.int32)[None]
        return mx.broadcast_to(pos[None], (3, B, L))

    def __call__(
        self,
        inputs: mx.array,
        inputs_embeds: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache=None,
        position_ids: Optional[mx.array] = None,
        rope_deltas: Optional[mx.array] = None,
        **kwargs,
    ):
        del mask, kwargs  # the backbone builds masks from the caches
        positions = self._resolve_positions(
            inputs, cache, position_ids, rope_deltas)
        out = self.model(inputs, cache, input_embeddings=inputs_embeds,
                         position_ids=positions)
        return LanguageModelOutput(logits=self._head(out))

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        # Same per-layer cache classes as the text-only Model.make_cache.
        shim = type("_S", (), {"args": self.args, "model": self.model,
                               "layers": self.model.layers})
        return q4.Model.make_cache(shim)


class Model(_Qwen3VLModel):
    """Qwen3-VL merge/vision-cache plumbing over the vendored halves."""

    def __init__(self, config: ModelConfig):
        nn.Module.__init__(self)
        self.config = config
        self.vision_tower = VisionModel(config.vision_config)
        self.language_model = LanguageModel(config.text_config, config)

    def get_input_embeddings(
        self,
        input_ids: Optional[mx.array] = None,
        pixel_values: Optional[mx.array] = None,
        **kwargs,
    ):
        if pixel_values is None:
            pixel_values = kwargs.get("pixel_values_videos", None)
        image_grid_thw = kwargs.get("image_grid_thw", None)
        video_grid_thw = kwargs.get("video_grid_thw", None)
        mask = kwargs.get("mask", None)
        grid_thw = image_grid_thw if image_grid_thw is not None else video_grid_thw

        embeds = self.language_model.model.embed_tokens(input_ids)
        if pixel_values is None:
            # Text-only turn: leave positions unset so the backbone keeps the
            # scalar-offset rope (identical numerics, no mrope overhead).
            self.language_model._position_ids = None
            self.language_model._rope_deltas = None
            return InputEmbeddingsFeatures(inputs_embeds=embeds)

        dtype = self.vision_tower.patch_embed.proj.weight.dtype
        cached = kwargs.get("cached_image_features", None)
        vision_cache = kwargs.get("vision_cache", None)
        if cached is None and vision_cache is not None:
            cached = vision_cache.get(kwargs.get("_image_key"))
        if cached is not None:
            hidden_states = cached
        else:
            hidden_states, _ = self.vision_tower(
                pixel_values.astype(dtype), grid_thw)
            if vision_cache is not None and kwargs.get("_image_key") is not None:
                mx.eval(hidden_states)
                vision_cache.put(kwargs["_image_key"], hidden_states)

        inputs_embeds, _ = self.merge_input_ids_with_image_features(
            hidden_states, embeds, input_ids,
            self.config.image_token_index, self.config.video_token_index)
        position_ids, rope_deltas = self.language_model.get_rope_index(
            input_ids, image_grid_thw, video_grid_thw, mask)
        self.language_model._position_ids = position_ids
        self.language_model._rope_deltas = rope_deltas
        return InputEmbeddingsFeatures(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            rope_deltas=rope_deltas,
        )

    def __call__(
        self,
        input_ids: mx.array,
        pixel_values: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache=None,
        **kwargs,
    ):
        features = self.get_input_embeddings(input_ids, pixel_values, **kwargs)
        kwargs.pop("position_ids", None)
        kwargs.pop("rope_deltas", None)
        return self.language_model(
            input_ids, inputs_embeds=features.inputs_embeds, mask=mask,
            cache=cache, position_ids=features.position_ids,
            rope_deltas=features.rope_deltas)

    def sanitize(self, weights):
        return weights

    @property
    def layers(self):
        return self.language_model.model.layers

    def make_cache(self):
        return self.language_model.make_cache()
