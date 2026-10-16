# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Asher Feldman
"""Muse Spark with image and audio input (mmproj projectors ``muse-glimmer``
and ``muse-spark-audio``).

The vision half is the Muse Glimmer tower and adapter
(:mod:`gmlx.models.muse_glimmer.vlm_model`); the Spark mmproj ships the same
tower with a projection into the 8192-wide Spark residual. The audio half is
:mod:`gmlx.models.muse_spark.audio`. The text half is the class the
text-only path uses (:mod:`gmlx.models.muse_spark.model`), in the
``language_model`` shape mlx-vlm's generate stack expects.

The processor expands each image into ``<|image_start|>``, one ``<|patch|>``
per soft token and ``<|image_end|>``, and each audio clip into
``<|soft_audio_start|>``, one ``<|audio|>`` per 1920 samples and
``<|soft_audio_end|>``, as llama.cpp's mtmd brackets them. The features
replace the ``<|patch|>`` and ``<|audio|>`` rows of the token embeddings.
The text tower's embedding norm then runs on every row, as in llama.cpp.
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
from gmlx.models.muse_glimmer.vlm_model import (
    Model as _GlimmerModel,
    VisionAdapter,
    VisionConfig,
    VisionModel,
)

from .audio import AudioConfig, AudioModel
from .model import Model as _SparkLM

__all__ = ["AudioConfig", "LanguageModel", "Model", "ModelConfig",
           "TextConfig", "VisionConfig"]


@dataclass
class TextConfig(BaseModelConfig):
    model_type: str = "muse_spark"
    hidden_size: int = 8192
    num_hidden_layers: int = 62
    num_attention_heads: int = 128
    num_key_value_heads: int = 16
    head_dim: int = 64
    vocab_size: int = 202048
    layer_types: List[str] = field(default_factory=list)
    num_experts: int = 256
    num_experts_per_tok: int = 8
    moe_intermediate_size: int = 4096
    moe_latent_size: int = 4096
    sliding_window: int = 2048
    rms_norm_eps: float = 1e-5
    post_norm_eps: float = 1e-8
    rope_theta: float = 500000.0
    rope_parameters: Optional[dict] = None
    max_position_embeddings: int = 1048576
    output_multiplier: float = 1.0
    final_logit_softcapping: float = 0.0
    routed_scaling_factor: float = 1.0
    norm_topk_prob: bool = True
    tie_word_embeddings: bool = False


@dataclass
class ModelConfig(BaseModelConfig):
    text_config: TextConfig = field(default_factory=TextConfig)
    vision_config: Optional[VisionConfig] = None
    audio_config: Optional[AudioConfig] = None
    model_type: str = "muse_spark"
    image_token_id: int = 200092
    image_token_index: Optional[int] = None
    audio_token_id: int = 200038
    vocab_size: int = 202048
    eos_token_id: Optional[List[int]] = None

    def __post_init__(self):
        if self.image_token_index is None:
            self.image_token_index = self.image_token_id


class LanguageModel(_SparkLM):
    def __init__(self, config: TextConfig):
        super().__init__(config)  # pyright: ignore[reportArgumentType]
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
        # The backbone builds its own sliding and full masks from the caches.
        del mask, kwargs
        h = self.model(inputs, cache=cache, inputs_embeds=inputs_embeds)
        return LanguageModelOutput(logits=self.head(h))

    @property
    def head_dim(self):
        return self.args.head_dim

    @property
    def n_kv_heads(self):
        return self.args.num_key_value_heads


def _scatter(token_id: int, features: mx.array, embeds: mx.array,
             input_ids: mx.array, kind: str) -> mx.array:
    """Put ``features`` on the ``token_id`` rows of ``embeds``, in order."""
    positions = input_ids == token_id
    n_slots = int(mx.sum(positions).item())
    if n_slots != features.shape[0]:
        raise ValueError(
            f"{n_slots} {kind} placeholder tokens but {features.shape[0]} "
            f"{kind} features")
    if n_slots == 0:
        return embeds
    rank = mx.cumsum(positions.astype(mx.int32).reshape(-1)) - 1
    gathered = mx.take(features.astype(embeds.dtype), mx.maximum(rank, 0), axis=0)
    return mx.where(positions[..., None], gathered.reshape(embeds.shape), embeds)


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        vc = config.vision_config
        if vc is not None:
            self.vision_tower = VisionModel(vc)
            self.vision_adapter = VisionAdapter(vc)
            self.vision_projection = nn.Linear(
                vc.adapter_hidden_size, vc.projection_dim, bias=False)
        if config.audio_config is not None:
            self.audio_tower = AudioModel(config.audio_config)
        self.language_model = LanguageModel(config.text_config)

    _image_features = _GlimmerModel._image_features

    def _audio_features(self, input_features: mx.array,
                        mask: Optional[mx.array]) -> mx.array:
        """Padded ``[N, samples]`` waveforms plus a ``[N, tokens]`` mask ->
        ``[total_tokens, text_hidden]``. Each clip runs on its own rows."""
        per = self.config.audio_config.samples_per_token
        if input_features.ndim == 1:
            input_features = input_features[None]
        feats = []
        for i in range(input_features.shape[0]):
            n = (int(mask[i].astype(mx.int32).sum().item()) if mask is not None
                 else input_features.shape[1] // per)
            wave = input_features[i, : n * per].astype(mx.float32)
            feats.append(self.audio_tower(wave))
        return mx.concatenate(feats, axis=0)

    def get_input_embeddings(
        self,
        input_ids: Optional[mx.array] = None,
        pixel_values: Optional[mx.array] = None,
        **kwargs,
    ):
        embeds = self.language_model.model.embed_tokens(input_ids)
        if pixel_values is not None:
            if not hasattr(self, "vision_tower"):
                raise ValueError(
                    "image input needs an mmproj with a vision encoder")
            features = kwargs.get("cached_image_features")
            if features is None:
                sizes = kwargs.get("image_sizes")
                if sizes is None:
                    sizes = [pixel_values.shape[-2:]] * pixel_values.shape[0]
                features = self._image_features(
                    pixel_values, [(int(h), int(w)) for h, w in sizes])
            embeds = _scatter(self.config.image_token_index, features, embeds,
                              input_ids, "image")
        waves = kwargs.get("input_features")
        n_slots = (int(mx.sum(input_ids == self.config.audio_token_id).item())
                   if waves is not None else 0)
        if n_slots:
            if not hasattr(self, "audio_tower"):
                raise ValueError(
                    "audio input needs an mmproj with an audio encoder")
            features = self._audio_features(
                waves, kwargs.get("input_features_mask"))
            # A reused prompt prefix leaves only the last placeholders in
            # input_ids, and they take the last features.
            if n_slots < features.shape[0]:
                features = features[features.shape[0] - n_slots:]
            embeds = _scatter(self.config.audio_token_id, features, embeds,
                              input_ids, "audio")
        return InputEmbeddingsFeatures(inputs_embeds=embeds)

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
