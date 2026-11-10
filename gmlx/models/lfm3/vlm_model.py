# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Asher Feldman
"""LFM3-VL and LFM3-Omni with image and video input (mmproj projector
``lfm3vl``), and LFM3-Audio, which has the speech encoder alone
(``lfm3a``).

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

import os
from dataclasses import dataclass, field
from typing import List, Optional

import mlx.core as mx
import mlx.nn as nn

from gmlx._vendor.mlx_vlm.models.base import (
    BaseModelConfig,
    InputEmbeddingsFeatures,
    LanguageModelOutput,
)

from . import speak
from .audio import AudioConfig, AudioEncoder, AudioProjector
from .model import Model as TextModel
from .model import ModelArgs
from .speech import CodecDecoder, DepthDecoder, SpeechConfig
from .vision import (CLIP_PATCHES, MultiModalProjector, VisionConfig,
                     VisionModel)

TextConfig = ModelArgs

MODEL_TYPES = ("lfm3_omni", "lfm3_vl", "lfm3_audio")

__all__ = ["AudioConfig", "LanguageModel", "Model", "ModelConfig",
           "TextConfig", "VisionConfig"]

# "1" runs the vision tower, the speech encoder and the speech decoders
# in a float32 stream on their 16-bit weights, to tell a 16-bit rounding
# effect from a defect.
TOWER_FP32_ENV = "GMLX_LFM3_TOWER_FP32"


@dataclass
class ModelConfig(BaseModelConfig):
    text_config: TextConfig = None  # pyright: ignore[reportAssignmentType]
    # None on a model with the speech encoder alone (``lfm3_audio``).
    vision_config: Optional[VisionConfig] = field(
        default_factory=VisionConfig)
    # The speech encoder of a model that hears, else None.
    audio_config: Optional[AudioConfig] = None
    # The speech generator of a model that speaks, else None.
    speech_config: Optional[SpeechConfig] = None
    model_type: str = "lfm3_omni"
    image_token_id: Optional[int] = 125030
    video_token_id: Optional[int] = 125029
    audio_token_id: Optional[int] = None
    image_token_index: Optional[int] = None
    vocab_size: int = 128000
    eos_token_id: Optional[List[int]] = None

    def __post_init__(self):
        if self.image_token_index is None:
            self.image_token_index = self.image_token_id
        if self.model_type == "lfm3_audio":
            self.vision_config = None
        # An absent encoder arrives as an empty dict from the loader.
        if not self.audio_config:
            self.audio_config = None
        elif isinstance(self.audio_config, dict):
            self.audio_config = AudioConfig.from_dict(self.audio_config)
        if not self.speech_config:
            self.speech_config = None
        elif isinstance(self.speech_config, dict):
            self.speech_config = SpeechConfig.from_dict(self.speech_config)


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


def scatter_features(token_id: Optional[int], features: mx.array,
                     inputs_embeds: mx.array, input_ids: mx.array,
                     what: str) -> mx.array:
    """Put ``features`` on the rows of ``token_id``, in order."""
    if token_id is None:
        raise ValueError(f"this LFM3 model has no {what} token")
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
        fp32 = os.environ.get(TOWER_FP32_ENV) == "1"
        if config.vision_config is not None:
            self.vision_tower = VisionModel(config.vision_config)
            self.multi_modal_projector = MultiModalProjector(
                config.vision_config)
            if fp32:
                self.vision_tower.stream_dtype = mx.float32
        if config.audio_config is not None:
            self.audio_encoder = AudioEncoder(config.audio_config)
            self.audio_projector = AudioProjector(config.audio_config)
            if fp32:
                self.audio_encoder.stream_dtype = mx.float32
        self._speech_biases: dict = {}
        if config.speech_config is not None:
            self.depth_decoder = DepthDecoder(config.speech_config)
            self.codec_decoder = CodecDecoder(config.speech_config)
            if fp32:
                self.depth_decoder.stream_dtype = mx.float32
                self.codec_decoder.stream_dtype = mx.float32
        # The serve engine runs the language model alone and reads the
        # media ids off its config (gmlx.cache.apc_media).
        for name in ("image_token_id", "video_token_id"):
            setattr(config.text_config, name, getattr(config, name))
        config.text_config.media_token_ids = (
            [] if config.audio_token_id is None
            else [int(config.audio_token_id)])
        self.language_model = LanguageModel(config.text_config)

    def vision_features(self, pixel_values: mx.array, grid_thw) -> mx.array:
        """The flattened patches of every image or video, in order, to
        their projected features, concatenated in the same order. Each
        image runs on its own, and a clip a few slices at a time. Each
        pass is evaluated before the next starts."""
        if getattr(self, "vision_tower", None) is None:
            raise ValueError("this LFM3 model has no vision tower")
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

    def audio_features(self, input_features: mx.array,
                       feature_attention_mask) -> mx.array:
        """The spectrograms of every clip, ``[clips, mel bins, frames]``
        right-padded, to their projected features, concatenated in clip
        order. Each clip runs alone on its own frames, and is evaluated
        before the next starts."""
        if getattr(self, "audio_encoder", None) is None:
            raise ValueError("this LFM3 model has no audio encoder")
        lengths = [int(n) for n in
                   mx.array(feature_attention_mask).sum(axis=-1).tolist()]
        feats = []
        for i, frames in enumerate(lengths):
            hidden = self.audio_encoder(input_features[i, :, :frames])
            feats.append(self.audio_projector(hidden))
            if len(lengths) > 1:
                mx.eval(feats[-1])
        return feats[0] if len(feats) == 1 else mx.concatenate(feats, axis=0)

    def get_input_embeddings(
        self,
        input_ids: Optional[mx.array] = None,
        pixel_values: Optional[mx.array] = None,
        **kwargs,
    ):
        embeds = self.language_model.model.embed_tokens(input_ids)
        videos = kwargs.get("pixel_values_videos")
        audio = kwargs.get("input_features")
        if audio is not None:
            mask = kwargs.get("feature_attention_mask")
            if mask is None:
                raise ValueError(
                    "LFM3 audio needs feature_attention_mask alongside "
                    "input_features")
            embeds = scatter_features(
                self.config.audio_token_id,
                self.audio_features(audio, mask), embeds, input_ids,
                "audio")
        if pixel_values is None and videos is None:
            return InputEmbeddingsFeatures(inputs_embeds=embeds)
        if getattr(self, "vision_tower", None) is None:
            raise ValueError("this LFM3 model has no vision tower")
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

    # -- speech output ---------------------------------------------------
    # One reply's state is a speak.SpeechState. Each step of the text
    # model gives a hidden state, [rows, hidden], that fills one slot.

    def speech_capabilities(self) -> Optional[dict]:
        """What the model's speech output is, or None when it has none."""
        cfg = self.config.speech_config
        if cfg is None:
            return None
        return {
            "sample_rate": cfg.sample_rate,
            "frame_samples": cfg.frame_samples,
            "voices": list(speak.VOICES),
            "modes": list(speak.MODES),
        }

    def speech_start(self, last_token: int, mode: Optional[str] = None,
                     end_of_turn=()):
        """``(state, marker)`` for a reply to a prompt that ends in
        ``last_token``: see :func:`speak.start`."""
        if self.config.speech_config is None:
            raise ValueError("this LFM3 model has no speech generator")
        return speak.start(self.config.speech_config, last_token, mode,
                           end_of_turn)

    def _speech_bias(self, text_only: bool) -> mx.array:
        """The head bias of a reply that may speak: the placeholders are
        masked, and the mode markers only in an explicit text reply."""
        bias = self._speech_biases.get(text_only)
        if bias is None:
            cfg = self.config
            speech = cfg.speech_config
            ids = [cfg.image_token_id, cfg.video_token_id,
                   cfg.audio_token_id, speech.audio_out_token_id]
            if text_only:
                ids += [speech.audio_start_token_id,
                        speech.mixed_start_token_id]
            bias = mx.zeros((cfg.text_config.vocab_size,), dtype=mx.float32)
            bias[mx.array([int(i) for i in ids if i is not None])] = -mx.inf
            mx.eval(bias)
            self._speech_biases[text_only] = bias
        return bias

    def speech_text_logits(self, state, hidden: mx.array) -> mx.array:
        """The logits of a text slot, ``[rows, vocab]`` float32."""
        logits = self.language_model.head(hidden, suppress=False)
        return logits.astype(mx.float32) + self._speech_bias(state.text_only)

    def speech_step(self, state, hidden: mx.array, key: mx.array):
        """A speech slot: ``(codes, embedding)``. ``codes`` is ``[rows,
        codebooks]``, with every code the end code once one is.
        ``embedding`` is the text model's next input, ``[rows, 1,
        hidden]``. Nothing is evaluated here."""
        depth = self.depth_decoder
        end = depth.config.eos_code
        codes = depth.sample_frame(
            hidden, key, depth.penalty_mask(state.history))
        codes = mx.where(
            mx.any(codes == end, axis=-1, keepdims=True), end, codes)
        return codes, depth.feedback_embedding(codes)[:, None, :]

    def speech_audio(self, state, codes):
        """The samples one more frame of ``codes`` adds to the reply."""
        if state.stream is None:
            state.stream = self.codec_decoder.new_stream()
        return self.codec_decoder.decode(codes, state.stream, finalize=False)

    def speech_flush(self, state):
        """The samples held back for the frames that never came, or None
        when no audio is open."""
        stream, state.stream = state.stream, None
        if stream is None or not stream.started:
            return None
        return self.codec_decoder.decode(
            [], stream, finalize=True)

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
