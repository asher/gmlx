# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Asher Feldman
"""Vendored mlx-vlm model for DeepSeek-V4-Flash-Vision-Exp (mmproj ``deepseek4v``).

Pairs the vendored V4 text tower (:mod:`gmlx.models.deepseek_v4.model`)
with the DeepSeek ViT + aligner (:mod:`gmlx.models.deepseek_v4.vision`) in
the ``language_model`` / ``vision_tower`` shape mlx-vlm's generate stack
expects. The container registers under its own model_type
(``deepseek_v4_vl``): mlx-vlm ships a text-only ``deepseek_v4`` shim that
would otherwise win the upstream-first registration.

Image turns (reference ``inference/model.py``): the processor expands each
``<|deepseek_image|>`` placeholder into a block of sentinel ids past the
vocab (lead PADs, START, the aligner grid as IMAGE/NEWLINE rows in the
interleaved N-layout, PADs, END) and reports the block offsets
(``image_spans``). Here the aligner rows land on the IMAGE slots in
``perm`` order, the four sentinels take their learned vectors, text keeps
its token embedding, and ``image_spans`` rides through to the text tower
(image-token router bias, in-block attention lookahead). Everything the
tower decides about a chunk it decides from ``image_spans`` on the host;
decode tokens and text chunks run the unchanged text paths.
"""

from __future__ import annotations

import importlib
import logging
import sys
from dataclasses import dataclass, field
from typing import List, Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_vlm.models.base import BaseModelConfig, InputEmbeddingsFeatures

from .image_block import (
    IMAGE,
    IMAGE_END,
    IMAGE_NEW_LINE,
    IMAGE_PAD,
    IMAGE_START,
    N_BLOCK_TYPES,
    build_image_block,
)
from .model import DeepseekV4Model, ModelArgs
from .model import Model as _TextModel
from .model import ensure_registered as _text_ensure_registered
from .mtp import DeepseekV4SpecHooks
from .vision import DeepseekV4VisionModel, VisionConfig

TextConfig = ModelArgs

MODEL_TYPE = "deepseek_v4_vl"
_log = logging.getLogger(__name__)

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
    """Make ``mlx_vlm.models.deepseek_v4_vl`` resolve, preferring upstream."""
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
    # <|deepseek_image|>, the per-image placeholder the chat template emits.
    image_token_id: int = 129264
    image_token_index: Optional[int] = None
    vocab_size: int = 129280
    eos_token_id: Optional[List[int]] = None
    # The expanded stream's media ids (vocab_size + block type). mlx-vlm's
    # APC keeps reusable prefixes clear of these runs once
    # gmlx.cache.apc_media folds the field into its media-id reader; the
    # serve engine reads the config off the language model, so the same
    # list is mirrored onto the text config.
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


class LanguageModel(DeepseekV4SpecHooks, nn.Module):
    """The vendored text tower under mlx-vlm's language-model contract,
    carrying the same speculative hooks as the text-only MTP target so
    ``--mmproj`` and text-only MTP compose on one model."""

    def __init__(self, config: TextConfig):
        super().__init__()
        self.config = config
        self.args = config
        self.model_type = config.model_type
        self.model = DeepseekV4Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self._init_spec_hooks()

    @property
    def layers(self):
        return self.model.pipeline_layers

    @property
    def cast_predicate(self):
        return _TextModel.cast_predicate.fget(self)

    def make_cache(self):
        return _TextModel.make_cache(self)

    def sanitize(self, weights):
        # The text tower's own sanitize (kquant scale placeholders, tid2eid
        # cast, wo_a 2D->3D, MTP-block drop), on language_model-relative
        # names.
        return _TextModel.sanitize(self, weights)

    def chunked_prefill_policy(self, **kwargs):
        # mlx-vlm's run/chat prefill loop walks fixed-size chunks it never
        # re-plans, so a boundary inside an image block is only avoidable
        # by not chunking: one forward keeps every block whole (the
        # attention transient is bounded by the span rows, not the chunk).
        # Serve chunkers steer per chunk instead (gmlx.gen.media_spans).
        from gmlx.gen.media_spans import (
            chunk_boundaries_cut,
            prefill_step_stamp,
        )

        spans = (kwargs.get("prefill_kwargs") or {}).get("image_spans")
        step = prefill_step_stamp(self)
        if spans and step:
            embeds = kwargs.get("inputs_embeds")
            ids = kwargs.get("input_ids")
            length = (embeds.shape[1] if embeds is not None
                      else ids.shape[-1] if ids is not None else 0)
            if chunk_boundaries_cut(spans, step, length):
                _log.info(
                    "prefill step %d would cut an image block in a "
                    "%d-token prompt: prefilling in one chunk", step, length)
                return False
        return super().chunked_prefill_policy(**kwargs)


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.vision_tower = DeepseekV4VisionModel(config.vision_config)
        self.language_model = LanguageModel(config.text_config)
        # The serve engine reads media ids off the language model's config.
        tc = self.language_model.config
        if hasattr(tc, "media_token_ids") and not tc.media_token_ids:
            tc.media_token_ids = list(config.media_token_ids)
        dim = config.text_config.hidden_size
        # Learned sentinel embeddings (mmproj v.token_embd.img_* and
        # v.image_newline).
        self.image_start = mx.zeros((dim,))
        self.image_pad = mx.zeros((dim,))
        self.image_newline = mx.zeros((dim,))
        self.image_end = mx.zeros((dim,))

    # --- image features -----------------------------------------------------

    def _image_features(self, pixel_values: mx.array, image_meta) -> mx.array:
        """Patches + per-image (n_vit_h, n_vit_w, n_llm_h, n_llm_w) ->
        aligner rows for every IMAGE slot, in stream order.

        Images run one at a time (rope tables and the aligner grid are
        per-image); each image's rows are permuted into the N-layout slot
        order before concatenation, so a single in-order scatter fills the
        IMAGE slots of every block."""
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
            _, perm = build_image_block(n_llm_h, n_llm_w, 0)
            feats.append(rows[mx.array(perm)])
        return feats[0] if len(feats) == 1 else mx.concatenate(feats, axis=0)

    def get_input_embeddings(
        self,
        input_ids: Optional[mx.array] = None,
        pixel_values: Optional[mx.array] = None,
        image_meta=None,
        image_spans=None,
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
                "deepseek_v4_vl needs image_meta alongside pixel_values")
        features = kwargs.get("cached_image_features")
        if features is None:
            features = self._image_features(pixel_values, image_meta)
        embeds = embed(mx.where(is_media, mx.zeros_like(input_ids), input_ids))
        embeds = self.merge_input_ids_with_image_features(
            vocab + IMAGE, features, embeds, input_ids)
        for tid, vec in (
            (IMAGE_START, self.image_start),
            (IMAGE_PAD, self.image_pad),
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
        sanitize runs on the ``language_model.`` subtree, the vision tower
        and sentinel embeddings pass through untouched."""
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
            input_ids, cache=cache, inputs_embeds=features.inputs_embeds,
            image_spans=kwargs.get("image_spans"))
