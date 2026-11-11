# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Asher Feldman
"""LFM3 speech output (mmproj projector ``lfm3a_gen``): the depth decoder
that turns one hidden state of the text model into a frame of 12 codes,
and the codec decoder that turns frames into 24 kHz audio.

Ported from llama.cpp ``tools/mtmd/models/lfm3a-gen.cpp`` and
``mtmd-helper-gen.cpp`` (MIT, contributed by Liquid AI) and from the
LFM3-Omni reference ``modeling_lfm3_omni.py`` and
``modeling_liquid_codec.py`` (Apache-2.0).

A frame is 80 ms of speech, 12.5 frames per second. The depth decoder
gives each codebook one position: a slice of the projected hidden state
plus the embedding of its code, or a learned mask embedding while the
code is unknown. Three bidirectional LFM3 layers read the 12 positions.
The codes are revealed in a random order over a fixed schedule of passes
(MaskGIT), each pass scoring the codebooks still masked. The frame's
codes go back to the text model as one embedding, the sum of one row per
codebook.

The codec decoder is causal: a code vector becomes 6 positions of a
small LFM3 stack with a sliding attention window, and each position one
STFT frame in polar form. The inverse STFT and the overlap-add run on
the host in float32.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from gmlx._vendor.mlx_vlm.models.base import BaseModelConfig
from gmlx.cache.attention import create_attention_mask
from gmlx.cache.kv_cache import ArraysCache, RotatingKVCache

from .model import CONV, FULL, SLIDING, DecoderLayer, ModelArgs, _Norm


def _codec_layer_types() -> List[str]:
    return [SLIDING if i in (2, 4, 6) else CONV for i in range(8)]


@dataclass
class SpeechConfig(BaseModelConfig):
    model_type: str = "lfm3_speech"
    # The depth decoder.
    hidden_size: int = 1024
    intermediate_size: int = 2816
    num_hidden_layers: int = 3
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    norm_eps: float = 1e-5
    rope_theta: float = 1000000.0
    projection_dim: int = 2048
    num_codebooks: int = 12
    # The last code of every codebook ends the audio.
    codebook_size: int = 2017
    # Codebooks revealed per pass.
    schedule: List[int] = field(default_factory=lambda: [4, 4, 4])
    # The codec decoder.
    codec_hidden_size: int = 512
    codec_intermediate_size: int = 2304
    codec_num_attention_heads: int = 8
    codec_num_key_value_heads: int = 2
    codec_layer_types: List[str] = field(default_factory=_codec_layer_types)
    codec_window: int = 30
    codec_conv_L_cache: int = 7
    codec_norm_eps: float = 1e-5
    codec_rope_theta: float = 1000000.0
    quantizer_dim: int = 256
    fsq_dim: int = 4
    frame_stack: int = 6
    n_fft: int = 1280
    hop_length: int = 320
    sample_rate: int = 24000
    # The card's sampling of the codes.
    temperature: float = 0.4
    top_k: int = 0
    repetition_penalty: float = 1.2
    repetition_window: int = 64
    # An interleaved reply alternates this many text tokens and frames.
    interleaved_n_text: int = 6
    interleaved_n_audio: int = 12
    # The tokens that steer a reply between text and speech, from the
    # text model's vocabulary.
    audio_start_token_id: Optional[int] = None
    mixed_start_token_id: Optional[int] = None
    text_end_token_id: Optional[int] = None
    audio_out_token_id: Optional[int] = None

    @property
    def eos_code(self) -> int:
        return self.codebook_size - 1

    @property
    def frame_samples(self) -> int:
        return self.hop_length * self.frame_stack


# -- depth decoder ------------------------------------------------------------

class DepthDecoder(nn.Module):
    def __init__(self, config: SpeechConfig):
        super().__init__()
        self.config = config
        dim, books = config.hidden_size, config.num_codebooks
        args = ModelArgs(
            model_type="lfm3", hidden_size=dim,
            num_hidden_layers=config.num_hidden_layers,
            intermediate_size=config.intermediate_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            head_dim=dim // config.num_attention_heads, vocab_size=1,
            layer_types=[FULL] * config.num_hidden_layers,
            rms_norm_eps=config.norm_eps, rope_theta=config.rope_theta,
            use_sandwich_norm=False)
        self.proj_in = nn.Linear(config.projection_dim, books * dim)
        self.mask_embedding = mx.zeros((dim,))
        # One table per codebook, the input embedding and the tied head.
        self.embeddings = mx.zeros((books, config.codebook_size, dim))
        self.layers = [DecoderLayer(args, i)
                       for i in range(config.num_hidden_layers)]
        self.norm = _Norm(dim, eps=config.norm_eps)
        # One norm weight per codebook, ahead of its head.
        self.head_norm = mx.ones((books, dim))
        # The rows a frame's codes add up to for the text model:
        # code + codebook * codebook_size.
        self.feedback = mx.zeros(
            (books * config.codebook_size, config.projection_dim))
        # The stream follows the hidden state's dtype unless this is set.
        self.stream_dtype = None

    def _rows(self, codes: mx.array) -> mx.array:
        books, size = self.config.num_codebooks, self.config.codebook_size
        return codes + mx.arange(books, dtype=codes.dtype) * size

    def logits(self, hidden: mx.array, codes: mx.array,
               masked: mx.array) -> mx.array:
        """One pass. ``hidden`` is ``[rows, projection_dim]``, the text
        model's hidden state before each frame. ``codes`` is ``[rows,
        codebooks]``, and ``masked`` marks the codebooks to predict, whose
        codes are not read. Returns ``[rows, codebooks, codebook_size]``
        in float32."""
        cfg = self.config
        rows, books, dim = hidden.shape[0], cfg.num_codebooks, cfg.hidden_size
        if self.stream_dtype is not None:
            hidden = hidden.astype(self.stream_dtype)
        x = self.proj_in(hidden).reshape(rows, books, dim)
        table = self.embeddings.reshape(-1, dim)
        known = mx.take(table, self._rows(mx.where(masked, 0, codes)), axis=0)
        x = x + mx.where(masked[..., None],
                         self.mask_embedding.astype(known.dtype), known)
        # Every codebook reads every other: no mask and no cache.
        for layer in self.layers:
            x = layer(x, None, None)
        x = self.norm(x)
        # The head norm scales in float32 and rounds once, as the
        # reference's heads were trained.
        h = x.astype(mx.float32)
        h = h * mx.rsqrt(mx.mean(h * h, axis=-1, keepdims=True)
                         + cfg.norm_eps)
        h = (h * self.head_norm.astype(mx.float32)).astype(x.dtype)
        # [rows, books, 1, dim] x [books, dim, size]
        out = h[:, :, None, :] @ self.embeddings.transpose(0, 2, 1)
        return out[:, :, 0, :].astype(mx.float32)

    def feedback_embedding(self, codes: mx.array) -> mx.array:
        """``[rows, projection_dim]``: what a frame of ``codes`` puts in
        the text model's next position. The rows add up in float32 and
        round once: a 16-bit running sum rounds 11 times."""
        rows = mx.take(self.feedback, self._rows(codes), axis=0)
        return rows.astype(mx.float32).sum(axis=-2).astype(rows.dtype)

    def penalty_mask(self, history: Optional[Sequence]) -> Optional[mx.array]:
        """``[codebooks, codebook_size]`` bool: the codes each codebook
        took in the last ``repetition_window`` frames of ``history``
        (frames of codes, oldest first). None when nothing is penalized."""
        cfg = self.config
        if (cfg.repetition_penalty <= 1.0 or cfg.repetition_window <= 0
                or history is None or len(history) == 0):
            return None
        recent = np.asarray(history, dtype=np.int64).reshape(
            -1, cfg.num_codebooks)[-cfg.repetition_window:]
        seen = np.zeros((cfg.num_codebooks, cfg.codebook_size), dtype=bool)
        seen[np.arange(cfg.num_codebooks)[None, :], recent] = True
        return mx.array(seen)

    def sample_frame(self, hidden: mx.array, key: mx.array,
                     seen: Optional[mx.array] = None, *,
                     temperature: Optional[float] = None,
                     top_k: Optional[int] = None,
                     order: Optional[mx.array] = None) -> mx.array:
        """The codes of one frame per row of ``hidden``, ``[rows,
        codebooks]`` int32. ``key`` seeds the reveal order and every
        draw. One key draws for all rows. ``[rows, 2]`` keys give each
        row its own stream, so a row's codes do not depend on the rows
        beside it, and one row draws what the single key draws. ``seen``
        is :meth:`penalty_mask` (one for all rows, or ``[rows, codebooks,
        codebook_size]``). ``order`` fixes the reveal order, ``[rows,
        codebooks]``, for tests.

        The passes build as one graph: nothing is evaluated here."""
        cfg = self.config
        rows, books = hidden.shape[0], cfg.num_codebooks
        temperature = cfg.temperature if temperature is None else temperature
        top_k = cfg.top_k if top_k is None else top_k
        draws = len(cfg.schedule) + 1
        if key.ndim == 2:
            if key.shape[0] != rows:
                raise ValueError(
                    f"{key.shape[0]} keys for {rows} rows of hidden states")
            row_keys = [mx.random.split(key[i], draws) for i in range(rows)]

            def uniform():
                return mx.concatenate([
                    mx.random.uniform(shape=(1, books), key=k[0])
                    for k in row_keys], axis=0)

            def categorical(scores, index):
                return mx.concatenate([
                    mx.random.categorical(scores[i:i + 1], key=k[index + 1])
                    for i, k in enumerate(row_keys)], axis=0)
        else:
            keys = mx.random.split(key, draws)

            def uniform():
                return mx.random.uniform(shape=(rows, books), key=keys[0])

            def categorical(scores, index):
                return mx.random.categorical(scores, key=keys[index + 1])
        if order is None:
            order = mx.argsort(uniform(), axis=-1)
        # The pass that reveals each codebook, from its place in the order.
        place = mx.argsort(order, axis=-1)
        ends = np.cumsum(cfg.schedule)
        if int(ends[-1]) != books:
            raise ValueError(
                f"the schedule {cfg.schedule} does not cover {books} "
                "codebooks")
        reveal_pass = mx.zeros(place.shape, dtype=mx.int32)
        for end in ends[:-1]:
            reveal_pass = reveal_pass + (place >= int(end)).astype(mx.int32)

        codes = mx.zeros((rows, books), dtype=mx.int32)
        for index in range(len(cfg.schedule)):
            scores = self.logits(hidden, codes, reveal_pass >= index)
            if seen is not None:
                # The CTRL penalty.
                scores = mx.where(
                    seen, mx.where(scores > 0,
                                   scores / cfg.repetition_penalty,
                                   scores * cfg.repetition_penalty), scores)
            if temperature <= 0:
                drawn = mx.argmax(scores, axis=-1)
            else:
                if 0 < top_k < cfg.codebook_size:
                    # Codes scoring at least the k-th best, ties included.
                    kth = mx.sort(scores, axis=-1)[..., -top_k][..., None]
                    scores = mx.where(scores < kth, -mx.inf, scores)
                drawn = categorical(scores * (1.0 / temperature), index)
            codes = mx.where(reveal_pass == index, drawn.astype(mx.int32),
                             codes)
        return codes


# -- codec decoder ------------------------------------------------------------

def hann_window(length: int) -> np.ndarray:
    """The periodic Hann window, float32."""
    n = np.arange(length, dtype=np.float64)
    return (0.5 - 0.5 * np.cos(2.0 * np.pi * n / length)).astype(np.float32)


def overlap_add(frames: np.ndarray, hop: int) -> np.ndarray:
    """Sum ``[count, length]`` frames placed ``hop`` apart. ``length`` is
    a multiple of ``hop``."""
    count, length = frames.shape
    if length % hop:
        raise ValueError(f"frame length {length} is not a multiple of {hop}")
    out = np.zeros((count - 1) * hop + length, dtype=frames.dtype)
    for k in range(length // hop):
        # Slice k of consecutive frames lands end to end.
        out[k * hop:(k + count) * hop] += frames[
            :, k * hop:(k + 1) * hop].reshape(-1)
    return out


class CodecStream:
    """What the codec decoder carries between the chunks of one stream:
    its layer caches and the overlap-add tail later frames still add to."""

    def __init__(self, cache: list):
        self.cache = cache
        self.audio: Optional[np.ndarray] = None
        self.envelope: Optional[np.ndarray] = None
        self.started = False
        self.finished = False


class CodecDecoder(nn.Module):
    def __init__(self, config: SpeechConfig):
        super().__init__()
        self.config = config
        dim = config.codec_hidden_size
        args = ModelArgs(
            model_type="lfm3", hidden_size=dim,
            num_hidden_layers=len(config.codec_layer_types),
            intermediate_size=config.codec_intermediate_size,
            num_attention_heads=config.codec_num_attention_heads,
            num_key_value_heads=config.codec_num_key_value_heads,
            head_dim=dim // config.codec_num_attention_heads, vocab_size=1,
            layer_types=list(config.codec_layer_types),
            sliding_window=config.codec_window,
            conv_L_cache=config.codec_conv_L_cache,
            rms_norm_eps=config.codec_norm_eps,
            rope_theta=config.codec_rope_theta,
            rope_theta_swa=config.codec_rope_theta, use_sandwich_norm=True)
        latent = config.fsq_dim * config.num_codebooks
        # The value of every code on the FSQ levels, shared by the
        # codebooks. The end code has no row.
        self.codebook = mx.zeros((config.codebook_size - 1, config.fsq_dim))
        self.quant_out = nn.Linear(latent, config.quantizer_dim)
        self.in_proj = nn.Linear(config.quantizer_dim, dim)
        self.layers = [DecoderLayer(args, i)
                       for i in range(len(config.codec_layer_types))]
        self.norm = _Norm(dim, eps=config.codec_norm_eps)
        self.out_proj = nn.Linear(dim, 2 * (config.n_fft // 2 + 1))
        self._window = hann_window(config.n_fft)
        # The stream follows the weights' dtype unless this is set.
        self.stream_dtype = None

    def make_cache(self) -> list:
        return [ArraysCache(size=1) if layer.layer_type == CONV
                else RotatingKVCache(max_size=self.config.codec_window)
                for layer in self.layers]

    def new_stream(self) -> CodecStream:
        return CodecStream(self.make_cache())

    def polar(self, codes: mx.array, cache: Optional[list] = None) -> mx.array:
        """``[frames, codebooks]`` codes to ``[frames * frame_stack,
        2 * bins]``: the log magnitude and the angle of each STFT frame.
        ``cache`` continues a stream."""
        cfg = self.config
        frames = codes.shape[0]
        latents = mx.take(self.codebook, codes, axis=0).reshape(frames, -1)
        x = self.quant_out(latents.astype(
            self.stream_dtype or self.quant_out.weight.dtype))
        x = self.in_proj(x)
        # Each frame becomes frame_stack positions.
        h = mx.repeat(x, cfg.frame_stack, axis=0)[None]
        caches = cache if cache is not None else [None] * len(self.layers)
        sliding = next((c for layer, c in zip(self.layers, caches)
                        if layer.layer_type == SLIDING), None)
        mask = create_attention_mask(h, sliding, window_size=cfg.codec_window)
        for layer, c in zip(self.layers, caches):
            h = layer(h, None if layer.layer_type == CONV else mask, c)
        return self.out_proj(self.norm(h))[0].astype(mx.float32)

    def _synthesize(self, polar: np.ndarray):
        """The overlap-added signal and window envelope of ``[positions,
        2 * bins]`` polar frames, both float32."""
        cfg = self.config
        bins = cfg.n_fft // 2 + 1
        log_magnitude, angle = polar[:, :bins], polar[:, bins:]
        spec = np.exp(log_magnitude) * np.exp(1j * angle.astype(np.float64))
        frames = np.fft.irfft(spec, n=cfg.n_fft, axis=-1).astype(np.float32)
        frames = frames * self._window
        audio = overlap_add(frames, cfg.hop_length)
        envelope = overlap_add(
            np.broadcast_to(self._window ** 2, frames.shape).copy(),
            cfg.hop_length)
        return audio, envelope

    def decode(self, codes, stream: Optional[CodecStream] = None,
               finalize: bool = True) -> np.ndarray:
        """24 kHz float32 samples of ``[frames, codebooks]`` codes, 1,920
        per frame. With ``stream``, the chunks of one stream decode to the
        audio of the joined codes: until ``finalize``, the samples later
        frames still add to are held back."""
        cfg = self.config
        codes = np.asarray(codes, dtype=np.int64).reshape(
            -1, cfg.num_codebooks)
        if codes.size and (codes.min() < 0 or codes.max() >= cfg.eos_code):
            raise ValueError(
                f"codes must be in [0, {cfg.eos_code}): the end code has no "
                "audio")
        if stream is not None and stream.finished:
            raise ValueError("this stream is finalized; start a new one")
        trim = (cfg.n_fft - cfg.hop_length) // 2
        tail = 2 * trim
        frames = codes.shape[0]
        if not frames and not (finalize and stream is not None
                               and stream.started):
            if stream is not None and finalize:
                stream.finished = True
            return np.zeros(0, dtype=np.float32)
        if frames:
            polar = self.polar(mx.array(codes.astype(np.int32)),
                               None if stream is None else stream.cache)
            audio, envelope = self._synthesize(np.asarray(polar))
        else:
            audio = np.zeros(tail, dtype=np.float32)
            envelope = np.zeros(tail, dtype=np.float32)
        if stream is None:
            audio, envelope = audio[trim:-trim], envelope[trim:-trim]
        else:
            if stream.audio is not None:
                audio = np.concatenate(
                    [audio[:tail] + stream.audio, audio[tail:]])
                envelope = np.concatenate(
                    [envelope[:tail] + stream.envelope, envelope[tail:]])
            if not stream.started:
                audio, envelope = audio[trim:], envelope[trim:]
                stream.started = True
            if finalize:
                audio, envelope = audio[:-trim], envelope[:-trim]
                stream.audio = stream.envelope = None
                stream.finished = True
            else:
                stream.audio, stream.envelope = audio[-tail:], envelope[-tail:]
                audio, envelope = audio[:-tail], envelope[:-tail]
        return (audio / envelope).astype(np.float32)
