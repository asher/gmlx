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
from .speech import SpeechConfig, SpeechEncoder, num_audio_tokens
from .transcription import CONTEXT_TOKENS
from .vlm_model import LanguageModel

TextConfig = ModelArgs

# The placeholder of one audio token: a control token the template and
# the model's own output never use.
AUDIO_PLACEHOLDER = "<SPECIAL_19>"

__all__ = ["AudioConfig", "EncodedClip", "LanguageModel", "Model",
           "ModelConfig", "TextConfig"]

# The encoder's peak over its weights, one window per step: two copies of
# the larger of a token's fp32 ASR state and its 16-bit row while windows
# and fusion steps join, plus one window's activations. Measured for an
# hour: 0.53 GiB.
_STEP_BYTES = 160 << 20
# An encoder output row is 16-bit.
_ROW_BYTES = 2


class EncodedClip(np.ndarray):
    """A clip the speech encoder has read, in place of its samples: a
    zero-copy array as long as the clip (the processor counts its audio
    tokens by its length) that carries the clip's digest, its encoder rows
    and its 80 ms speaker probabilities."""

    digest: str = ""
    embeds: mx.array | None = None
    speaker_probs: np.ndarray | None = None

    @classmethod
    def of(cls, samples: int, digest: str, embeds: mx.array,
           speaker_probs: np.ndarray) -> "EncodedClip":
        out = np.broadcast_to(np.float32(0), (int(samples),)).view(cls)
        out.digest, out.embeds, out.speaker_probs = digest, embeds, speaker_probs
        return out

    def __array_finalize__(self, obj):
        for name in ("digest", "embeds", "speaker_probs"):
            setattr(self, name, getattr(obj, name, getattr(EncodedClip, name)))


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


def _drop_rows(parts: list, n: int) -> list:
    """``parts`` without their first ``n`` rows."""
    out = []
    for p in parts:
        if n >= p.shape[0]:
            n -= p.shape[0]
            continue
        out.append(p[n:] if n else p)
        n = 0
    return out


class PromptRows:
    """A prompt's embedded rows, built a chunk at a time: text rows from
    the token embedding, audio rows from the encoder's parts, each part
    freed once the prompt is past it. Whole merged rows would hold 2 bytes
    per hidden unit for every prompt token through the whole prefill.

    The prefill loop reads it as it reads an array of rows:
    ``rows[:, :n]`` builds the next ``n`` rows and ``rows[:, n:]`` moves
    past them."""

    def __init__(self, embed, input_ids: mx.array, token_id: int, parts: list):
        self._embed = embed
        self._ids = input_ids
        self._tok = int(token_id)
        self._parts = list(parts)
        self._start = 0
        # Audio rows before the first part, and before the start.
        self._dropped = 0
        self._audio = 0
        probe = embed(input_ids[:, :1])
        self.dtype = probe.dtype
        self._hidden = int(probe.shape[-1])

    @property
    def shape(self) -> tuple[int, int, int]:
        return (int(self._ids.shape[0]), int(self._ids.shape[1]) - self._start,
                self._hidden)

    def __getitem__(self, key):
        rows, cols = key
        if not (rows == slice(None) and isinstance(cols, slice)
                and cols.step is None):
            raise IndexError("prompt rows take [:, :n] or [:, n:]")
        if cols.start in (None, 0):
            n = self.shape[1] if cols.stop is None else int(cols.stop)
            return self._build(min(n, self.shape[1]))
        if cols.stop is None:
            self._advance(int(cols.start))
            return self
        raise IndexError("prompt rows take [:, :n] or [:, n:]")

    def _audio_rows(self, k: int) -> mx.array:
        a = self._audio - self._dropped
        got, pos = [], 0
        for p in self._parts:
            n = int(p.shape[0])
            lo, hi = max(a - pos, 0), min(a + k - pos, n)
            if lo < hi:
                got.append(p[lo:hi])
            pos += n
            if pos >= a + k:
                break
        return got[0] if len(got) == 1 else mx.concatenate(got, axis=0)

    def _build(self, n: int) -> mx.array:
        ids = self._ids[:, self._start:self._start + n]
        embeds = self._embed(ids)
        k = int(mx.sum(ids == self._tok).item())
        if k:
            embeds = _scatter(self._tok, self._audio_rows(k), embeds, ids)
        return embeds

    def _advance(self, n: int) -> None:
        ids = self._ids[:, self._start:self._start + n]
        self._audio += int(mx.sum(ids == self._tok).item())
        self._start += n
        while self._parts and (self._dropped + int(self._parts[0].shape[0])
                               <= self._audio):
            self._dropped += int(self._parts.pop(0).shape[0])


def drain(steps):
    """Run a step generator to the end and give its return value."""
    while True:
        try:
            next(steps)
        except StopIteration as done:
            return done.value


def _scatter(token_id: int, features: mx.array, embeds: mx.array,
             input_ids: mx.array) -> mx.array:
    """Put ``features`` on the ``token_id`` rows of ``embeds``, in order."""
    positions = input_ids == token_id
    rank = mx.cumsum(positions.astype(mx.int32).reshape(-1)) - 1
    gathered = mx.take(features.astype(embeds.dtype), mx.maximum(rank, 0), axis=0)
    return mx.where(positions[..., None], gathered.reshape(embeds.shape), embeds)


class Model(nn.Module):
    # generate_step asks for PromptRows in place of whole merged rows.
    builds_prompt_rows = True
    # The trained context, which bounds an output budget.
    context_tokens = CONTEXT_TOKENS

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

    def encode_steps(self, feats: np.ndarray, valid: int, tokens: int, *,
                     keep_states: bool = False, join: bool = True):
        """One clip's encoder steps (:meth:`SpeechEncoder.steps`), one
        window each, its rows in the dtype of the token embeddings."""
        # A quantized table's weight is packed: ask an embedding row.
        dtype = self.language_model.backbone.embeddings(mx.array([0])).dtype
        return self.audio_tower.steps(feats, valid, tokens, batch=1,
                                      dtype=dtype, keep_states=keep_states,
                                      join=join)

    def audio_parts(self, input_features, mask, n_slots: int) -> list:
        """:meth:`audio_features` as the encoder's fusion-step parts, in
        prompt order, not joined."""
        feats = np.asarray(input_features, np.float32)
        if feats.ndim == 2:
            feats = feats[None]
        lengths = self.clip_lengths(mask if mask is not None
                                    else np.ones(feats.shape[:2], bool))
        if n_slots > sum(t for _, t in lengths):
            raise ValueError(f"{n_slots} audio placeholder tokens but "
                             f"{sum(t for _, t in lengths)} audio features")
        clips, need = [], n_slots
        f = self.audio_tower.cfg.asr.subsampling
        for i in range(len(lengths) - 1, -1, -1):
            if need <= 0:
                break
            valid, tokens = lengths[i]
            parts = drain(self.encode_steps(
                feats[i, : tokens * f], valid, tokens, join=False)).parts
            clips.append(_drop_rows(parts, max(tokens - need, 0)))
            need -= tokens
        return [p for clip in reversed(clips) for p in clip]

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
            out = drain(self.encode_steps(feats[i, : tokens * f], valid, tokens))
            rows.append(out.embeds[max(tokens - need, 0):])
            need -= tokens
        rows.reverse()
        return rows[0] if len(rows) == 1 else mx.concatenate(rows, axis=0)

    def audio_preflight(self, clips) -> tuple[int, int]:
        """The audio tokens of ``clips`` (samples, or :class:`EncodedClip`)
        and the bytes encoding the ones not yet encoded holds at its peak,
        one window per step."""
        tokens = [num_audio_tokens(len(c)) for c in clips]
        fresh = [t for c, t in zip(clips, tokens)
                 if not isinstance(c, EncodedClip)]
        if not fresh:
            return sum(tokens), 0
        cfg = self.audio_tower.cfg
        per_token = 2 * max(cfg.proj_dim * 2, cfg.asr.d_model * 4)
        return sum(tokens), max(fresh) * per_token + _STEP_BYTES

    def audio_held_bytes(self, audio_tokens: int) -> int:
        """The bytes of the encoder output for ``audio_tokens``."""
        return int(audio_tokens) * self.audio_tower.cfg.proj_dim * _ROW_BYTES

    def encoded_rows(self, clips, n_slots: int) -> mx.array:
        """The rows of the last ``n_slots`` audio tokens of the prompt's
        :class:`EncodedClip` clips."""
        rows = [c.embeds for c in clips]
        total = sum(int(r.shape[0]) for r in rows)
        if n_slots > total:
            raise ValueError(f"{n_slots} audio placeholder tokens but "
                             f"{total} audio features")
        joined = rows[0] if len(rows) == 1 else mx.concatenate(rows, axis=0)
        return joined[total - n_slots:]

    def get_input_embeddings(
        self,
        input_ids: Optional[mx.array] = None,
        pixel_values: Optional[mx.array] = None,
        **kwargs,
    ):
        if pixel_values is not None:
            raise ValueError("Nemotron 3.5 Transcribe takes no image input")
        feats = kwargs.get("input_features")
        clips = kwargs.get("audio_clips")
        n_slots = (int(mx.sum(input_ids == self.config.audio_token_id).item())
                   if feats is not None or clips else 0)
        embed = self.language_model.backbone.embeddings
        if kwargs.get("prompt_rows"):
            parts = []
            if n_slots:
                parts = ([self.encoded_rows(clips, n_slots)] if clips else
                         self.audio_parts(feats, kwargs.get("input_features_mask"),
                                          n_slots))
            return InputEmbeddingsFeatures(inputs_embeds=PromptRows(  # pyright: ignore[reportArgumentType]
                embed, input_ids, self.config.audio_token_id, parts))
        embeds = embed(input_ids)
        if n_slots:
            features = (self.encoded_rows(clips, n_slots) if clips else
                        self.audio_features(
                            feats, kwargs.get("input_features_mask"), n_slots))
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
