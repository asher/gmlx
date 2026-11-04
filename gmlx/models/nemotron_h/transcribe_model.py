# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Asher Feldman
"""Nemotron 3.5 Transcribe: the nemotron_h text model with audio input from
its ``pee`` speech encoder GGUF (:mod:`gmlx.models.nemotron_h.speech`).

The processor turns each ``<|audio|>`` the prompt holds into one
``<SPECIAL_19>`` per 80 ms of its clip and hands the clips' log-mel
spectrograms over as ``input_features``. The encoder's rows replace those
placeholder rows of the token embeddings, and the text model runs
unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from gmlx._vendor.mlx_vlm.models.base import (
    BaseModelConfig,
    InputEmbeddingsFeatures,
)

from .model import ModelArgs
from .speech import SpeechConfig, SpeechEncoder
from .vlm_model import LanguageModel

TextConfig = ModelArgs

# The placeholder of one audio token: a control token the template and
# the model's own output never use.
AUDIO_PLACEHOLDER = "<SPECIAL_19>"

__all__ = ["AudioConfig", "LanguageModel", "Model", "ModelConfig",
           "TextConfig"]


@dataclass
class AudioConfig(BaseModelConfig):
    """The encoder GGUF's ``pee.*`` and ``ctc.*`` header fields."""
    fields: dict = field(default_factory=dict)
    model_type: str = "nemotron_h_transcribe_audio"


@dataclass
class ModelConfig(BaseModelConfig):
    text_config: TextConfig = None  # pyright: ignore[reportAssignmentType]
    audio_config: AudioConfig = field(default_factory=AudioConfig)
    model_type: str = "nemotron_h_transcribe"
    audio_token_id: int = 19
    vocab_size: int = 131072
    eos_token_id: Optional[List[int]] = None
    media_token_ids: List[int] = field(default_factory=list)

    def __post_init__(self):
        if not self.media_token_ids:
            self.media_token_ids = [int(self.audio_token_id)]


def _scatter(token_id: int, features: mx.array, embeds: mx.array,
             input_ids: mx.array) -> mx.array:
    """Put ``features`` on the ``token_id`` rows of ``embeds``, in order."""
    positions = input_ids == token_id
    rank = mx.cumsum(positions.astype(mx.int32).reshape(-1)) - 1
    gathered = mx.take(features.astype(embeds.dtype), mx.maximum(rank, 0), axis=0)
    return mx.where(positions[..., None], gathered.reshape(embeds.shape), embeds)


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.audio_tower = SpeechEncoder(
            SpeechConfig.from_gguf(config.audio_config.fields))
        self.language_model = LanguageModel(config.text_config)
        # The APC guard reads the media ids off the text config: a reused
        # prefix then holds every audio slot of the prompt or none.
        self.language_model.config.media_token_ids = list(config.media_token_ids)

    def clip_lengths(self, mask) -> list[tuple[int, int]]:
        """``(valid frames, audio tokens)`` of each clip of a
        ``[clips, frames]`` feature mask."""
        f = self.audio_tower.cfg.asr.subsampling
        valid = np.asarray(mask).astype(bool).sum(axis=-1).reshape(-1)
        return [(int(v), -(-int(v) // f)) for v in valid]

    def audio_features(self, input_features, mask, n_slots: int) -> mx.array:
        """The encoder rows of the last ``n_slots`` audio tokens of the
        prompt's clips. Clips wholly before them, in a reused prefix, are
        not encoded."""
        feats = np.asarray(input_features, np.float32)
        if feats.ndim == 2:
            feats = feats[None]
        lengths = self.clip_lengths(mask if mask is not None
                                    else np.ones(feats.shape[:2], bool))
        total = sum(t for _, t in lengths)
        if n_slots > total:
            raise ValueError(f"{n_slots} audio placeholder tokens but "
                             f"{total} audio features")
        rows, need = [], n_slots
        f = self.audio_tower.cfg.asr.subsampling
        for i in range(len(lengths) - 1, -1, -1):
            if need <= 0:
                break
            valid, tokens = lengths[i]
            out = self.audio_tower(feats[i, : tokens * f], valid, tokens)
            rows.append(out.embeds[max(tokens - need, 0):])
            need -= tokens
        rows.reverse()
        return rows[0] if len(rows) == 1 else mx.concatenate(rows, axis=0)

    def get_input_embeddings(
        self,
        input_ids: Optional[mx.array] = None,
        pixel_values: Optional[mx.array] = None,
        **kwargs,
    ):
        if pixel_values is not None:
            raise ValueError("Nemotron 3.5 Transcribe takes no image input")
        embeds = self.language_model.backbone.embeddings(input_ids)
        feats = kwargs.get("input_features")
        n_slots = (int(mx.sum(input_ids == self.config.audio_token_id).item())
                   if feats is not None else 0)
        if n_slots:
            features = self.audio_features(
                feats, kwargs.get("input_features_mask"), n_slots)
            embeds = _scatter(self.config.audio_token_id, features, embeds,
                              input_ids)
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
