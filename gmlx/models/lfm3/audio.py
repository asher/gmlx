# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Asher Feldman
"""Speech encoder and projector of LFM3-Omni and LFM3-Audio (mmproj
projector ``lfm3a``).

Ported from Liquid AI's ``feature_extraction_lfm3_omni.py`` and
``modeling_lfm3_omni.py`` (Apache-2.0) and checked against llama.cpp
``tools/mtmd/models/lfm3a.cpp`` (MIT, contributed by Liquid AI):

- a log-mel spectrogram of 16 kHz audio: 400-sample periodic Hann
  windows every 160 samples with no centering, 128 Slaney mel bins,
  ``log10`` floored at -12, then ``(x + 4) / 4``;
- three stride-2 convolutions over the (mel, time) plane, so 128 mel
  bins and 8 frames become one vector: 12.5 tokens per second;
- a causal LFM2 stack: gated short convolutions and grouped-query
  attention layers, each with a SwiGLU feed-forward and pre-norms. A
  position attends to itself and the 31 before it;
- a final RMSNorm, then the projector: LayerNorm and a GELU MLP into the
  text width.

A clip is encoded whole, from a zero state, as the reference's offline
path does. The encoder runs in the activation dtype on 16-bit weights.
The spectrogram, the norms, RoPE and the convolution sums run in float32.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import List, Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from gmlx._vendor.mlx_vlm.models.base import BaseModelConfig

from .model import MLP, ShortConv, _Norm
from .vision import _rotate

CONV = "conv"
FULL = "full_attention"

SAMPLE_RATE = 16000
N_FFT = 400
HOP_LENGTH = 160
LOG_FLOOR = -12.0
NORMALIZATION_OFFSET = 4.0
NORMALIZATION_DIVISOR = 4.0
# Spectrogram frames per encoder token.
STRIDE = 8

# Spectrogram frames per FFT pass, which bounds the frame matrix.
_STFT_FRAMES = 8192
# Tokens the downsampler makes per pass. Its first convolution widens the
# spectrogram 128 times, so a clip runs a stretch at a time.
DOWNSAMPLE_TOKENS = 64
# Query rows per attention call, with the keys of their window. The
# scores of a call are rows x (rows + window), so the block stays near
# the window's size.
Q_BLOCK = 256
# Tokens a layer and the projector take per pass on a long clip. Each
# pass is evaluated, so the peak does not grow with the clip.
LAYER_TOKENS = 1024


def _default_layer_types() -> List[str]:
    return [CONV, CONV, FULL] * 5 + [FULL, FULL]


@dataclass
class AudioConfig(BaseModelConfig):
    model_type: str = "lfm3_audio_encoder"
    hidden_size: int = 768
    intermediate_size: int = 1792
    num_hidden_layers: int = 17
    num_attention_heads: int = 8
    num_key_value_heads: int = 4
    conv_L_cache: int = 3
    conv_channels: int = 256
    num_mel_bins: int = 128
    sliding_window: int = 32
    norm_eps: float = 1e-5
    rope_theta: float = 1000000.0
    layer_types: List[str] = field(default_factory=_default_layer_types)
    projection_dim: int = 2048
    projector_layer_norm_eps: float = 1e-5


# -- features -----------------------------------------------------------------

def pad_length(num_samples: int) -> int:
    """Samples of a clip once right-padded to whole frames."""
    if num_samples < N_FFT:
        return N_FFT
    return num_samples + (-(num_samples - N_FFT) % HOP_LENGTH)


def num_frames(num_samples: int) -> int:
    """Spectrogram frames of a clip."""
    return 1 + (pad_length(num_samples) - N_FFT) // HOP_LENGTH


def num_audio_tokens(num_samples: int) -> int:
    """Encoder tokens of a clip: one per ``STRIDE`` frames, the last
    group possibly incomplete."""
    return -(num_frames(num_samples) // -STRIDE)


@lru_cache(maxsize=4)
def _mel_filters(num_mel_bins: int) -> np.ndarray:
    """``[num_mel_bins, N_FFT // 2 + 1]`` float32, the reference's Slaney
    filter bank."""
    from transformers.audio_utils import mel_filter_bank

    return np.ascontiguousarray(mel_filter_bank(
        num_frequency_bins=N_FFT // 2 + 1, num_mel_filters=num_mel_bins,
        min_frequency=0.0, max_frequency=SAMPLE_RATE / 2,
        sampling_rate=SAMPLE_RATE, norm="slaney", mel_scale="slaney",
    ).T.astype(np.float32))


@lru_cache(maxsize=1)
def _window() -> np.ndarray:
    """The periodic Hann window, as ``torch.hann_window`` gives it."""
    n = np.arange(N_FFT, dtype=np.float64)
    return (0.5 - 0.5 * np.cos(2.0 * np.pi * n / N_FFT)).astype(np.float32)


def log_mel(wave: np.ndarray, num_mel_bins: int = 128) -> np.ndarray:
    """The normalized log-mel spectrogram of a mono 16 kHz clip,
    ``[num_mel_bins, frames]`` float32. NumPy, so request threads can
    call it."""
    x = np.asarray(wave, dtype=np.float32).reshape(-1)
    x = np.pad(x, (0, pad_length(x.size) - x.size))
    frames = 1 + (x.size - N_FFT) // HOP_LENGTH
    filters, window = _mel_filters(num_mel_bins), _window()
    out = np.empty((num_mel_bins, frames), dtype=np.float32)
    for start in range(0, frames, _STFT_FRAMES):
        count = min(_STFT_FRAMES, frames - start)
        idx = (np.arange(N_FFT)[None, :]
               + HOP_LENGTH * (start + np.arange(count))[:, None])
        spec = np.fft.rfft(x[idx] * window, n=N_FFT, axis=-1)
        power = (spec.real.astype(np.float32) ** 2
                 + spec.imag.astype(np.float32) ** 2)
        with np.errstate(divide="ignore"):
            mel = np.log10(filters @ power.T)
        out[:, start:start + count] = np.maximum(mel, np.float32(LOG_FLOOR))
    return ((out + np.float32(NORMALIZATION_OFFSET))
            / np.float32(NORMALIZATION_DIVISOR))


class AudioFeatureExtractor:
    """Clips to spectrograms, right-padded to the longest, with the mask
    of each clip's own frames."""

    sampling_rate = SAMPLE_RATE
    model_input_names = ["input_features", "feature_attention_mask"]

    def __init__(self, num_mel_bins: int = 128):
        self.num_mel_bins = num_mel_bins

    @staticmethod
    def tokens(num_samples: int) -> int:
        return num_audio_tokens(num_samples)

    def __call__(self, clips) -> dict:
        mels = [log_mel(clip, self.num_mel_bins) for clip in clips]
        longest = max(m.shape[1] for m in mels)
        features = np.zeros((len(mels), self.num_mel_bins, longest),
                            dtype=np.float32)
        mask = np.zeros((len(mels), longest), dtype=np.int32)
        for i, m in enumerate(mels):
            features[i, :, :m.shape[1]] = m
            mask[i, :m.shape[1]] = 1
        return {"input_features": features, "feature_attention_mask": mask}


# -- encoder ------------------------------------------------------------------

def _rope_tables(length: int, head_dim: int,
                 theta: float) -> Tuple[mx.array, mx.array]:
    """``cos`` and ``sin``, each ``[length, head_dim]`` float32, for the
    positions of one clip. Not cached: clip lengths rarely repeat."""
    inv_freq = 1.0 / (theta ** (np.arange(0, head_dim, 2, dtype=np.float32)
                                / np.float32(head_dim)))
    freq = np.arange(length, dtype=np.float32)[:, None] * inv_freq
    freq = np.concatenate([freq, freq], axis=-1)
    return mx.array(np.cos(freq)), mx.array(np.sin(freq))


class _Conv2d(nn.Module):
    """A stride-2 3x3 or a 1x1 convolution in the MLX layout,
    ``[out, k, k, in / groups]``, with a bias. It runs in float32."""

    def __init__(self, channels_in: int, channels_out: int, kernel: int,
                 groups: int = 1):
        super().__init__()
        self.groups = groups
        self.stride, self.padding = (2, 1) if kernel == 3 else (1, 0)
        self.weight = mx.zeros(
            (channels_out, kernel, kernel, channels_in // groups))
        self.bias = mx.zeros((channels_out,))

    def __call__(self, x: mx.array) -> mx.array:
        out = mx.conv2d(
            x.astype(mx.float32), self.weight.astype(mx.float32),
            stride=self.stride, padding=self.padding, groups=self.groups)
        return out + self.bias.astype(mx.float32)


class Downsampler(nn.Module):
    """Three stride-2 convolutions over the (mel, time) plane, then a
    linear layer over the channels and the 16 mel bins that remain."""

    def __init__(self, config: AudioConfig):
        super().__init__()
        ch = config.conv_channels
        self.conv_in = _Conv2d(1, ch, 3)
        self.conv_mid = _Conv2d(ch, ch, 3, groups=ch)
        self.lin_mid = _Conv2d(ch, ch, 1)
        self.conv_end = _Conv2d(ch, ch, 3, groups=ch)
        self.lin_end = _Conv2d(ch, ch, 1)
        self.lin_out = nn.Linear(config.num_mel_bins * ch // STRIDE,
                                 config.hidden_size)

    def _convolve(self, mel: mx.array) -> mx.array:
        """``[mel bins, frames]`` to ``[tokens, channels * mel bins / 8]``
        float32, channel by channel."""
        x = mel.astype(mx.float32)[None, :, :, None]
        x = nn.silu(self.conv_in(x))
        x = nn.silu(self.lin_mid(self.conv_mid(x)))
        x = nn.silu(self.lin_end(self.conv_end(x)))
        _, bins, tokens, ch = x.shape
        return x.transpose(0, 2, 3, 1).reshape(tokens, ch * bins)

    def __call__(self, mel: mx.array, dtype: mx.Dtype) -> mx.array:
        """``mel`` is ``[mel bins, frames]``. Returns ``[tokens, hidden]``
        in ``dtype``. Token ``t`` reads frames ``8t - 7`` to ``8t + 7``,
        so a stretch runs with one token of margin before it, and the
        result is the whole clip's."""
        frames = mel.shape[1]
        tokens = -(frames // -STRIDE)
        parts = []
        for t0 in range(0, tokens, DOWNSAMPLE_TOKENS):
            t1 = min(tokens, t0 + DOWNSAMPLE_TOKENS)
            lead = 1 if t0 else 0
            end = min(frames, t1 * STRIDE)
            flat = self._convolve(mel[:, (t0 - lead) * STRIDE:end])
            part = self.lin_out(flat[lead:lead + t1 - t0].astype(dtype))
            if tokens > DOWNSAMPLE_TOKENS:
                mx.eval(part)
            parts.append(part)
        return parts[0] if len(parts) == 1 else mx.concatenate(parts, axis=0)


class AudioAttention(nn.Module):
    """Causal grouped-query attention over a sliding window."""

    def __init__(self, config: AudioConfig):
        super().__init__()
        dim = config.hidden_size
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        self.head_dim = dim // self.n_heads
        self.scale = self.head_dim ** -0.5
        self.window = config.sliding_window
        self.q_proj = nn.Linear(dim, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim,
                                bias=False)
        self.v_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim,
                                bias=False)
        self.out_proj = nn.Linear(self.n_heads * self.head_dim, dim,
                                  bias=False)
        self.q_layernorm = _Norm(self.head_dim, eps=config.norm_eps)
        self.k_layernorm = _Norm(self.head_dim, eps=config.norm_eps)

    def _mask(self, q0: int, q1: int, k0: int) -> mx.array:
        """True where the query at ``q0 + i`` reads the key at ``k0 + j``:
        itself and the ``window - 1`` positions before it."""
        q = mx.arange(q0, q1)[:, None]
        k = mx.arange(k0, q1)[None, :]
        return (k <= q) & (k > q - self.window)

    def __call__(self, x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
        _, n, _ = x.shape

        def heads(t: mx.array, count: int) -> mx.array:
            return t.reshape(1, n, count, self.head_dim).transpose(0, 2, 1, 3)

        q = _rotate(self.q_layernorm(heads(self.q_proj(x), self.n_heads)),
                    cos, sin)
        k = _rotate(self.k_layernorm(heads(self.k_proj(x), self.n_kv_heads)),
                    cos, sin)
        v = heads(self.v_proj(x), self.n_kv_heads)
        outs = []
        for q0 in range(0, n, Q_BLOCK):
            q1 = min(n, q0 + Q_BLOCK)
            k0 = max(0, q0 - self.window + 1)
            out = mx.fast.scaled_dot_product_attention(
                q[:, :, q0:q1], k[:, :, k0:q1], v[:, :, k0:q1],
                scale=self.scale, mask=self._mask(q0, q1, k0))
            if n > Q_BLOCK:
                mx.eval(out)
            outs.append(out)
        out = outs[0] if len(outs) == 1 else mx.concatenate(outs, axis=2)
        return self.out_proj(out.transpose(0, 2, 1, 3).reshape(1, n, -1))


class AudioLayer(nn.Module):
    def __init__(self, config: AudioConfig, layer_idx: int):
        super().__init__()
        self.layer_type = config.layer_types[layer_idx]
        if self.layer_type == CONV:
            self.conv = ShortConv(config)  # pyright: ignore[reportArgumentType]
        else:
            self.self_attn = AudioAttention(config)
        self.feed_forward = MLP(config.hidden_size, config.intermediate_size)
        self.operator_norm = _Norm(config.hidden_size, eps=config.norm_eps)
        self.ffn_norm = _Norm(config.hidden_size, eps=config.norm_eps)
        # Earlier positions an output reads.
        self.reach = (config.conv_L_cache if self.layer_type == CONV
                      else config.sliding_window) - 1

    def _pass(self, x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
        h = self.operator_norm(x)
        if self.layer_type == CONV:
            x = x + self.conv(h)
        else:
            x = x + self.self_attn(h, cos, sin)
        return x + self.feed_forward(self.ffn_norm(x))

    def __call__(self, x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
        """A long clip runs ``LAYER_TOKENS`` at a time. Each pass takes
        the positions its first outputs read from the pass before, so
        the result is one pass's."""
        tokens = x.shape[1]
        if tokens <= LAYER_TOKENS:
            return self._pass(x, cos, sin)
        parts = []
        for t0 in range(0, tokens, LAYER_TOKENS):
            lead = min(t0, self.reach)
            span = slice(t0 - lead, t0 + LAYER_TOKENS)
            part = self._pass(x[:, span], cos[span], sin[span])[:, lead:]
            mx.eval(part)
            parts.append(part)
        return mx.concatenate(parts, axis=1)


class AudioEncoder(nn.Module):
    def __init__(self, config: AudioConfig):
        super().__init__()
        self.config = config
        self.downsample = Downsampler(config)
        self.layers = [AudioLayer(config, i)
                       for i in range(config.num_hidden_layers)]
        self.embedding_norm = _Norm(config.hidden_size, eps=config.norm_eps)
        # The stream follows the weights' dtype unless this is set.
        # Float32 here runs the 16-bit weights as llama.cpp does.
        self.stream_dtype = None

    def __call__(self, mel: mx.array) -> mx.array:
        """``mel`` is the ``[mel bins, frames]`` spectrogram of one clip.
        Returns the last hidden state, ``[tokens, hidden]``."""
        cfg = self.config
        if mel.ndim != 2 or mel.shape[0] != cfg.num_mel_bins:
            raise ValueError(
                f"expected a spectrogram of shape ({cfg.num_mel_bins}, "
                f"frames), got {tuple(mel.shape)}")
        dtype = self.stream_dtype or self.downsample.lin_out.weight.dtype
        x = self.downsample(mel, dtype)[None]
        tokens = x.shape[1]
        cos, sin = _rope_tables(
            tokens, cfg.hidden_size // cfg.num_attention_heads,
            float(cfg.rope_theta))
        for layer in self.layers:
            x = layer(x, cos, sin)
        return self.embedding_norm(x)[0]


class AudioProjector(nn.Module):
    def __init__(self, config: AudioConfig):
        super().__init__()
        dim, out = config.hidden_size, config.projection_dim
        self.layer_norm = nn.LayerNorm(
            dim, eps=config.projector_layer_norm_eps)
        self.linear_1 = nn.Linear(dim, out)
        self.linear_2 = nn.Linear(out, out)

    def _pass(self, x: mx.array) -> mx.array:
        dtype = x.dtype
        x = mx.fast.layer_norm(
            x.astype(mx.float32), self.layer_norm.weight.astype(mx.float32),
            self.layer_norm.bias.astype(mx.float32),
            self.layer_norm.eps).astype(dtype)
        return self.linear_2(nn.gelu(self.linear_1(x)))

    def __call__(self, x: mx.array) -> mx.array:
        """``x`` is ``[tokens, hidden]``. A long clip runs
        ``LAYER_TOKENS`` at a time."""
        tokens = x.shape[0]
        if tokens <= LAYER_TOKENS:
            return self._pass(x)
        parts = []
        for t0 in range(0, tokens, LAYER_TOKENS):
            parts.append(self._pass(x[t0:t0 + LAYER_TOKENS]))
            mx.eval(parts[-1])
        return mx.concatenate(parts, axis=0)
