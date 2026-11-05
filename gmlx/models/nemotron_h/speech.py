# SPDX-License-Identifier: Apache-2.0
# Ported from Hugging Face transformers 9b283429f9bf
# models/nemotron3_5_transcribe/modeling_nemotron3_5_transcribe.py,
# models/nemotron3_diarization/modeling_nemotron3_diarization.py and
# models/nemotron_asr_streaming/feature_extraction_nemotron_asr_streaming.py,
# and modified for gmlx (Apache-2.0; see licenses/transformers-LICENSE).
"""The audio encoder of Nemotron 3.5 Transcribe: the ``pee`` GGUF that
NVIDIA's NeMo-Speech.cpp writes beside the decoder.

Two transformers read the same log-mel spectrogram. The speech recognition
branch reads it normalized per mel bin, in 40 s windows with 4 s of context
on each side. The speaker branch reads it raw, a 40 s chunk at a time,
attending to a cache of earlier frames that keeps up to eight speakers
apart across the recording. Each 80 ms frame then carries the recognition
state plus a fixed embedding of the speakers active in it, projected to the
language model's width: one row per audio token of the prompt.

The CTC head of the same file reads the recognition states before that
fusion, upsampled to 40 ms frames, and scores the pieces of its own
SentencePiece vocabulary for word timestamps.

Everything runs one recording at a time, so no attention mask is needed:
the spectrogram is cut to its valid frames and zero-padded to a multiple
of 8, which the reference's masks make equivalent.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

ARCH = "pee"

# NeMo ParallelExpertEncoder defaults that the GGUF does not record
# (transformers convert_nemotron3_5_transcribe_to_hf.py), in encoder frames.
FIFO_LENGTH = 40
SPEAKER_CACHE_UPDATE_PERIOD = 300
SPEAKER_CACHE_LENGTH = 188

# The checkpoint's speaker cache policy (sortformer_modules in config.json).
SILENCE_FRAMES_PER_SPEAKER = 1
PREDICTION_SCORE_THRESHOLD = 0.25
LATEST_FRAMES_SCORE_BOOST = 0.05
STRONG_BOOST_RATE = 0.75
WEAK_BOOST_RATE = 1.5
MIN_POSITIVE_SCORES_RATE = 0.5

# The feature extractor (processor_config.json).
PREEMPHASIS = 0.97
WIN_LENGTH = 400
LOG_ZERO_GUARD = 2.0 ** -24
LAYER_NORM_EPS = 1e-5

# ASR windows run in batches of at most this many (an hour has 90).
MAX_WINDOW_BATCH = 8
# Audio tokens fused per step: the fusion of a whole hour at once holds
# about 1.4 GB of fp32 intermediates.
FUSE_ROWS = 4096


@dataclass
class BranchConfig:
    feat_in: int
    d_model: int
    n_heads: int
    n_layers: int
    ff_hidden: int
    subsampling: int
    rotary_dims: int
    qk_norm: bool

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @classmethod
    def from_gguf(cls, fields: dict, prefix: str) -> "BranchConfig":
        return cls(feat_in=int(fields[f"{prefix}.feat_in"]),
                   d_model=int(fields[f"{prefix}.d_model"]),
                   n_heads=int(fields[f"{prefix}.n_heads"]),
                   n_layers=int(fields[f"{prefix}.n_layers"]),
                   ff_hidden=int(fields[f"{prefix}.ff_hidden"]),
                   subsampling=int(fields.get(f"{prefix}.subsampling_factor", 8)),
                   rotary_dims=int(fields[f"{prefix}.rotary_dims"]),
                   qk_norm=bool(fields.get(f"{prefix}.qk_norm", False)))


@dataclass
class CTCConfig:
    d_model: int
    n_heads: int
    n_layers: int
    ff_hidden: int
    rotary_dims: int
    vocab_size: int
    upsampling: int
    residual_scale: float
    qk_norm: bool

    @property
    def blank_id(self) -> int:
        return self.vocab_size - 1

    @classmethod
    def from_gguf(cls, fields: dict) -> "CTCConfig | None":
        if "ctc.d_model" not in fields:
            return None
        return cls(d_model=int(fields["ctc.d_model"]),
                   n_heads=int(fields["ctc.n_heads"]),
                   n_layers=int(fields["ctc.n_layers"]),
                   ff_hidden=int(fields["ctc.ff_hidden"]),
                   rotary_dims=int(fields["ctc.rotary_dims"]),
                   vocab_size=int(fields["ctc.vocab_size"]),
                   upsampling=int(fields["ctc.upsampling_factor"]),
                   residual_scale=float(fields["ctc.residual_scale"]),
                   qk_norm=bool(fields.get("ctc.qk_norm", True)))


@dataclass
class SpeechConfig:
    asr: BranchConfig
    diar: BranchConfig
    num_speakers: int
    head_dim: int
    proj_dim: int
    threshold: float
    chunk_length: int
    left_context: int
    right_context: int
    sample_rate: int
    n_fft: int
    hop_length: int
    n_mels: int
    ctc: CTCConfig | None = None
    fifo_length: int = FIFO_LENGTH
    cache_update_period: int = SPEAKER_CACHE_UPDATE_PERIOD
    cache_length: int = SPEAKER_CACHE_LENGTH

    @classmethod
    def from_gguf(cls, fields: dict) -> "SpeechConfig":
        sr = int(fields.get("pee.preprocessor.sample_rate", 16000))
        return cls(
            asr=BranchConfig.from_gguf(fields, "pee.asr"),
            diar=BranchConfig.from_gguf(fields, "pee.diar"),
            num_speakers=int(fields["pee.diar.num_speakers"]),
            head_dim=int(fields["pee.diar.tf_d_model"]),
            proj_dim=int(fields["pee.fusion.proj_dim"]),
            threshold=float(fields.get("pee.fusion.speaker_activity_threshold", 0.5)),
            chunk_length=int(fields["pee.online.inference_length"]),
            left_context=int(fields["pee.online.chunk_left_context"]),
            right_context=int(fields["pee.online.chunk_right_context"]),
            sample_rate=sr,
            n_fft=int(fields.get("pee.preprocessor.n_fft", 512)),
            hop_length=round(float(fields.get("pee.preprocessor.window_stride", 0.01)) * sr),
            n_mels=int(fields.get("pee.preprocessor.features", 128)),
            ctc=CTCConfig.from_gguf(fields),
        )


# -- features -----------------------------------------------------------------


def _window(n_fft: int) -> np.ndarray:
    """``hann_window(400, periodic=False)`` rounded to bfloat16, centered in
    ``n_fft`` samples as torch.stft pads it."""
    n = np.arange(WIN_LENGTH, dtype=np.float64)
    w = (0.5 - 0.5 * np.cos(2 * np.pi * n / (WIN_LENGTH - 1))).astype(np.float32)
    w = (w.view(np.uint32) + 0x7FFF + ((w.view(np.uint32) >> 16) & 1)) & 0xFFFF0000
    w = w.astype(np.uint32).view(np.float32)
    left = (n_fft - WIN_LENGTH) // 2
    out = np.zeros(n_fft, np.float32)
    out[left:left + WIN_LENGTH] = w
    return out


def num_frames(samples: int, hop_length: int = 160) -> int:
    """Valid spectrogram frames of a clip: ``floor(samples / hop)``."""
    return samples // hop_length


def num_audio_tokens(samples: int, hop_length: int = 160,
                     subsampling: int = 8) -> int:
    """Prompt tokens of a clip: one per ``subsampling`` frames, the last
    group possibly incomplete."""
    return -(-num_frames(samples, hop_length) // subsampling)


def features(audio: np.ndarray, mel_filters: np.ndarray, *, n_fft: int = 512,
             hop_length: int = 160, subsampling: int = 8) -> np.ndarray:
    """The raw log-mel spectrogram of a 16 kHz clip, ``(frames, n_mels)``
    float32: the valid frames, zero-padded to a multiple of
    ``subsampling``. NumPy, so request threads can call it."""
    x = np.asarray(audio, np.float32).reshape(-1)
    y = np.empty_like(x)
    if x.size:
        y[0] = x[0]
        y[1:] = x[1:] - np.float32(PREEMPHASIS) * x[:-1]
    pad = n_fft // 2
    y = np.pad(y, (pad, pad))
    n = num_frames(x.size, hop_length)
    total = -(-n // subsampling) * subsampling
    out = np.zeros((total, mel_filters.shape[0]), np.float32)
    if n == 0:
        return out
    idx = np.arange(n_fft)[None, :] + hop_length * np.arange(n)[:, None]
    frames = y[idx] * _window(n_fft)
    spec = np.fft.rfft(frames, n=n_fft, axis=-1)
    power = (spec.real.astype(np.float32) ** 2 + spec.imag.astype(np.float32) ** 2)
    mel = power @ mel_filters.T.astype(np.float32)
    out[:n] = np.log(mel + np.float32(LOG_ZERO_GUARD))
    return out


def normalize_per_feature(feats: np.ndarray, valid: int) -> np.ndarray:
    """Each mel bin to zero mean and unit variance over the valid frames
    (unbiased variance, 1e-5 added to the deviation); padding stays 0."""
    out = np.zeros_like(feats)
    if valid == 0:
        return out
    v = feats[:valid].astype(np.float32)
    mean = v.sum(0, keepdims=True) / np.float32(valid)
    centered = v - mean
    std = np.sqrt((centered ** 2).sum(0, keepdims=True)
                  / np.float32(max(valid - 1, 1))) + np.float32(1e-5)
    out[:valid] = centered / std
    return out


# -- transformer blocks -------------------------------------------------------


class _Attention(nn.Module):
    def __init__(self, d: int, heads: int, rotary: int, qk_norm: bool):
        super().__init__()
        self.heads, self.head_dim, self.rotary = heads, d // heads, rotary
        self.w_qkv = nn.Linear(d, 3 * d, bias=False)
        self.out_proj = nn.Linear(d, d, bias=True)
        if qk_norm:
            self.q_norm = nn.LayerNorm(self.head_dim, eps=LAYER_NORM_EPS)
            self.k_norm = nn.LayerNorm(self.head_dim, eps=LAYER_NORM_EPS)
        self.qk_norm = qk_norm

    def __call__(self, x: mx.array) -> mx.array:
        b, t, d = x.shape
        q, k, v = mx.split(self.w_qkv(x), 3, axis=-1)
        q = q.reshape(b, t, self.heads, self.head_dim)
        k = k.reshape(b, t, self.heads, self.head_dim)
        if self.qk_norm:
            q, k = self.q_norm(q), self.k_norm(k)
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.reshape(b, t, self.heads, self.head_dim).transpose(0, 2, 1, 3)
        q = mx.fast.rope(q, self.rotary, traditional=False, base=10000.0,
                         scale=1.0, offset=0)
        k = mx.fast.rope(k, self.rotary, traditional=False, base=10000.0,
                         scale=1.0, offset=0)
        o = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=self.head_dim ** -0.5)
        return self.out_proj(o.transpose(0, 2, 1, 3).reshape(b, t, d))


class _FFN(nn.Module):
    def __init__(self, d: int, hidden: int):
        super().__init__()
        # NeMo's Sequential: Linear, GELU, Dropout, Linear.
        self.net = [nn.Linear(d, hidden), nn.Identity(), nn.Identity(),
                    nn.Linear(hidden, d)]

    def __call__(self, x: mx.array) -> mx.array:
        return self.net[3](nn.gelu(self.net[0](x)))


class _Layer(nn.Module):
    def __init__(self, d, heads, ff, rotary, qk_norm):
        super().__init__()
        self.norm1 = nn.LayerNorm(d, eps=LAYER_NORM_EPS)
        self.attn = _Attention(d, heads, rotary, qk_norm)
        self.norm2 = nn.LayerNorm(d, eps=LAYER_NORM_EPS)
        self.ffn = _FFN(d, ff)

    def __call__(self, x: mx.array) -> mx.array:
        x = x + self.attn(self.norm1(x))
        return x + self.ffn(self.norm2(x))


class _PreEncode(nn.Module):
    """Stacks ``subsampling`` frames and projects them."""

    def __init__(self, feat_in: int, d: int, subsampling: int):
        super().__init__()
        self.subsampling = subsampling
        self.proj = nn.Linear(feat_in * subsampling, d, bias=False)

    def __call__(self, feats: mx.array) -> mx.array:
        b, t, m = feats.shape
        return self.proj(feats.reshape(b, t // self.subsampling,
                                       m * self.subsampling))


class RopeEncoder(nn.Module):
    """A NeMo TransformerEncoder with rotary positions: input LayerNorm,
    pre-LN blocks, final LayerNorm. Positions start at 0 for each call."""

    def __init__(self, d, heads, layers, ff, rotary, qk_norm, *,
                 feat_in: int = 0, subsampling: int = 0):
        super().__init__()
        if feat_in:
            self.pre_encode = _PreEncode(feat_in, d, subsampling)
        self.embed_norm = nn.LayerNorm(d, eps=LAYER_NORM_EPS)
        self.layers = [_Layer(d, heads, ff, rotary, qk_norm)
                       for _ in range(layers)]
        self.final_norm = nn.LayerNorm(d, eps=LAYER_NORM_EPS)

    def embed(self, feats: mx.array) -> mx.array:
        return self.pre_encode(feats)

    def __call__(self, embeds: mx.array) -> mx.array:
        h = self.embed_norm(embeds)
        for layer in self.layers:
            h = layer(h)
            # Weights stay F16 and activations run in fp32. MLX computes
            # the fp32 copies of the weights ahead of their matmuls, so
            # without an eval per layer a call holds about 1.3 GB of them.
            mx.eval(h)
        return self.final_norm(h)

    @classmethod
    def branch(cls, c: BranchConfig) -> "RopeEncoder":
        return cls(c.d_model, c.n_heads, c.n_layers, c.ff_hidden,
                   c.rotary_dims, c.qk_norm, feat_in=c.feat_in,
                   subsampling=c.subsampling)


# -- speaker branch -----------------------------------------------------------


class SpeakerHead(nn.Module):
    """Sortformer head: encoder frames to 10 ms speaker logits."""

    def __init__(self, d_in: int, d: int, speakers: int, factor: int):
        super().__init__()
        self.factor = factor
        self.encoder_proj = nn.Linear(d_in, d)
        self.subpixel_upsample = nn.Conv1d(d, d * factor, kernel_size=3,
                                           padding=1)
        self.first_hidden_to_hidden = nn.Linear(d, d)
        self.single_hidden_to_spks = nn.Linear(d, speakers)
        self.learnable_sil_emb = mx.zeros((d_in,))

    def __call__(self, h: mx.array) -> mx.array:
        b, t, _ = h.shape
        h = self.subpixel_upsample(self.encoder_proj(h))
        h = h.reshape(b, t * self.factor, -1)
        h = self.first_hidden_to_hidden(nn.relu(h))
        return self.single_hidden_to_spks(nn.relu(h))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return (1.0 / (1.0 + np.exp(-x.astype(np.float64)))).astype(np.float32)


def _top(scores: np.ndarray, k: int) -> np.ndarray:
    """Indices of the ``k`` largest scores along axis 0, the lowest index
    first among equal scores. A frame kept for two speakers is cached
    twice, so later compressions see exact ties; the reference leaves
    their order to ``torch.topk``."""
    return np.argsort(-scores, axis=0, kind="stable")[:k]


class SpeakerCache:
    """The Arrival-Order Speaker Cache and FIFO of the speaker branch's
    offline pass (``Nemotron3DiarizationSpeakerCache``), on the host:
    pre-encoded frames every chunk attends to, compressed to
    ``cache_length`` frames grouped by speaker when it overflows."""

    def __init__(self, cfg: SpeechConfig, hidden: int):
        self.fifo_length = cfg.fifo_length
        self.update_period = cfg.cache_update_period
        self.cache_length = cfg.cache_length
        self.speakers = cfg.num_speakers
        self.factor = cfg.diar.subsampling
        budget = self.cache_length // self.speakers - SILENCE_FRAMES_PER_SPEAKER
        self.min_positive = math.floor(budget * MIN_POSITIVE_SCORES_RATE)
        self.strong = math.floor(budget * STRONG_BOOST_RATE)
        self.weak = math.floor(budget * WEAK_BOOST_RATE)
        self.embeds = np.zeros((0, hidden), np.float32)
        self.probs = np.zeros((0, self.speakers), np.float32)
        self.fifo = np.zeros((0, hidden), np.float32)
        self.compressed = False

    def rows(self) -> np.ndarray:
        return np.concatenate([self.embeds, self.fifo], axis=0)

    def _pool(self, logits: np.ndarray) -> np.ndarray:
        p = _sigmoid(logits)
        n = p.shape[0] // self.factor
        return p[: n * self.factor].reshape(n, self.factor, -1).mean(1)

    def update(self, rows: np.ndarray, logits: np.ndarray, silence: np.ndarray,
               n_chunk: int, lookback: int) -> None:
        n_cache, n_fifo = self.embeds.shape[0], self.fifo.shape[0]
        probs = self._pool(logits)
        start = n_cache + n_fifo + lookback
        chunk = rows[start:start + n_chunk]
        fifo = np.concatenate([self.fifo, chunk], axis=0)
        popped = 0
        if fifo.shape[0] > self.fifo_length:
            popped = min(max(self.update_period, fifo.shape[0] - self.fifo_length),
                         fifo.shape[0])
        if popped:
            fifo_probs = np.concatenate([probs[n_cache:n_cache + n_fifo],
                                         probs[start:start + n_chunk]], axis=0)
            stored = self.probs if self.compressed else probs[:n_cache]
            cache = np.concatenate([self.embeds, fifo[:popped]], axis=0)
            cache_probs = np.concatenate([stored, fifo_probs[:popped]], axis=0)
            fifo = fifo[popped:]
            if cache.shape[0] > self.cache_length:
                cache, cache_probs = self._compress(cache, cache_probs, silence)
                self.compressed = True
            self.embeds, self.probs = cache, cache_probs
        self.fifo = fifo

    def _scores(self, probs: np.ndarray) -> np.ndarray:
        th = PREDICTION_SCORE_THRESHOLD
        p = probs.astype(np.float32)
        log_p = np.log(np.maximum(p, th))
        log_c = np.log(np.maximum(1.0 - p, th))
        scores = log_p - log_c + log_c.sum(-1, keepdims=True) - np.float32(math.log(0.5))
        speech = p > 0.5
        scores = np.where(speech, scores, -np.inf)
        positive = scores > 0
        enough = positive.sum(0, keepdims=True) >= self.min_positive
        return np.where(~positive & speech & enough, -np.inf, scores).astype(np.float32)

    @staticmethod
    def _boost(scores: np.ndarray, n: int, boost: float) -> np.ndarray:
        if n <= 0:
            return scores
        idx = _top(scores, n)
        out = scores.copy()
        np.add.at(out, (idx, np.arange(scores.shape[1])[None, :]), np.float32(boost))
        return out

    def _compress(self, embeds, probs, silence):
        n_frames, n_spk = probs.shape
        scores = self._scores(probs)
        scores[self.cache_length:] += np.float32(LATEST_FRAMES_SCORE_BOOST)
        scores = self._boost(scores, self.strong, -2.0 * math.log(0.5))
        scores = self._boost(scores, self.weak, -math.log(0.5))
        scores = np.concatenate(
            [scores, np.full((SILENCE_FRAMES_PER_SPEAKER, n_spk), np.inf,
                             np.float32)], axis=0)
        embeds = np.concatenate([embeds, silence[None, :].astype(np.float32)], axis=0)
        probs = np.concatenate([probs, np.zeros((1, n_spk), np.float32)], axis=0)
        n_scored = n_frames + SILENCE_FRAMES_PER_SPEAKER
        sentinel = n_scored * n_spk
        flat = scores.T.reshape(-1)
        top = _top(flat, self.cache_length)
        top = np.where(flat[top] == -np.inf, sentinel, top)
        top = np.sort(top)
        frames = np.where(top == sentinel, n_frames,
                          np.minimum(top % n_scored, n_frames))
        return embeds[frames], probs[frames]


class SpeakerBranch(nn.Module):
    """The diarization model: a transformer over pre-encoded frames and the
    Sortformer head, run chunk by chunk with the speaker cache."""

    def __init__(self, cfg: SpeechConfig):
        super().__init__()
        c = cfg.diar
        self.cfg = cfg
        self.encoder = RopeEncoder.branch(c)
        self.head = SpeakerHead(c.d_model, cfg.head_dim, cfg.num_speakers,
                                c.subsampling)

    def chunks(self, n: int):
        c = self.cfg
        for start in range(0, n, c.chunk_length):
            end = min(start + c.chunk_length, n)
            ctx = max(start - c.left_context, 0)
            yield start, end, ctx, min(end + c.right_context, n)

    def logit_steps(self, feats: np.ndarray, out: list[np.ndarray]):
        """Append the 10 ms speaker logits ``(frames, speakers)`` of each
        chunk of one recording's raw spectrogram ``(frames, n_mels)`` to
        ``out``, yielding after each chunk."""
        dtype = self.encoder.embed_norm.weight.dtype
        step = self.cfg.diar.subsampling * self.cfg.chunk_length
        rows = np.concatenate([
            np.array(self.encoder.embed(mx.array(feats[a:a + step], dtype=dtype)[None])[0]
                     .astype(mx.float32))
            for a in range(0, feats.shape[0], step)] or [
                np.zeros((0, self.cfg.diar.d_model), np.float32)])
        cache = SpeakerCache(self.cfg, rows.shape[1])
        silence = np.array(self.head.learnable_sil_emb.astype(mx.float32))
        f = self.cfg.diar.subsampling
        for start, end, ctx, stop in self.chunks(rows.shape[0]):
            cached = cache.rows()
            x = np.concatenate([cached, rows[ctx:stop]], axis=0)
            h = self.encoder(mx.array(x, dtype=dtype)[None])
            logits = np.array(self.head(h)[0].astype(mx.float32))
            lookback = start - ctx
            cache.update(x, logits, silence, end - start, lookback)
            s = (cached.shape[0] + lookback) * f
            out.append(logits[s:s + (end - start) * f])
            yield

    def logits(self, feats: np.ndarray) -> np.ndarray:
        """10 ms speaker logits ``(frames, speakers)`` of one recording."""
        out: list[np.ndarray] = []
        for _ in self.logit_steps(feats, out):
            pass
        return np.concatenate(out, axis=0) if out else np.zeros(
            (0, self.cfg.num_speakers), np.float32)


# -- the encoder --------------------------------------------------------------


@dataclass
class EncoderOutput:
    embeds: mx.array | None  # (frames, proj_dim): one row per audio token
    asr_states: mx.array | None  # (frames, d_model): the CTC head's input
    speaker_probs: np.ndarray  # (frames, speakers): 80 ms means
    parts: list | None = None  # the rows in fusion steps, when not joined


class _Fusion(nn.Module):
    def __init__(self, d: int, speakers: int, proj_dim: int):
        super().__init__()
        self.asr_norm = nn.LayerNorm(d, eps=LAYER_NORM_EPS)
        self.diar_norm = nn.LayerNorm(speakers, eps=LAYER_NORM_EPS)
        self.diar_kernel = mx.zeros((speakers, d))
        self.proj = nn.Linear(d, proj_dim)


class CTCHead(nn.Module):
    """The word timestamp head: ASR states upsampled to 40 ms frames, two
    transformer layers with a scaled residual, and a classifier over the
    alignment vocabulary plus the blank (last class)."""

    def __init__(self, c: CTCConfig):
        super().__init__()
        self.c = c
        self.upsample = nn.Linear(c.d_model, c.d_model * c.upsampling)
        self.transformer = RopeEncoder(c.d_model, c.n_heads, c.n_layers,
                                       c.ff_hidden, c.rotary_dims, c.qk_norm)
        self.classifier = nn.Linear(c.d_model, c.vocab_size)

    def hidden(self, states: mx.array) -> mx.array:
        """``(frames, d)`` ASR states to ``(frames * upsampling, d)``."""
        t, d = states.shape
        x = self.upsample(states).reshape(t * self.c.upsampling, d)
        h = self.transformer(x[None])[0]
        return x + self.c.residual_scale * h


class SpeechEncoder(nn.Module):
    def __init__(self, cfg: SpeechConfig):
        super().__init__()
        self.cfg = cfg
        self.asr = RopeEncoder.branch(cfg.asr)
        self.diar = SpeakerBranch(cfg)
        self.fusion = _Fusion(cfg.asr.d_model, cfg.num_speakers, cfg.proj_dim)
        if cfg.ctc is not None:
            self.ctc = CTCHead(cfg.ctc)
        # The file's mel filterbank (preprocessor.fb), for :func:`features`.
        self.mel_filters = mx.zeros((cfg.n_mels, cfg.n_fft // 2 + 1))

    @property
    def dtype(self):
        return self.asr.embed_norm.weight.dtype

    def windows(self, n: int) -> list[tuple[int, int, int, int]]:
        """``(context start, context stop, keep start, keep stop)`` of each
        window, in encoder frames (``_get_chunks`` of the reference)."""
        c = self.cfg
        out = []
        for start in range(0, n, c.chunk_length):
            end = min(start + c.chunk_length, n)
            ctx0 = max(start - c.left_context, 0)
            ctx1 = min(end + c.right_context, n)
            out.append((ctx0, ctx1, start - ctx0, end - ctx0))
        return out

    def asr_steps(self, normalized: np.ndarray, pieces: dict[int, mx.array],
                  batch: int = MAX_WINDOW_BATCH):
        """Run the ASR branch over the per-feature normalized spectrogram,
        storing each window's kept frames in ``pieces`` by window index.
        Up to ``batch`` same-length windows run together; yields after
        each batch."""
        f = self.cfg.asr.subsampling
        wins = self.windows(normalized.shape[0] // f)
        by_len: dict[int, list[int]] = {}
        for i, (a, b, _, _) in enumerate(wins):
            by_len.setdefault(b - a, []).append(i)
        for idxs in by_len.values():
            for s in range(0, len(idxs), batch):
                group = idxs[s:s + batch]
                x = mx.array(np.stack([normalized[wins[i][0] * f:wins[i][1] * f]
                                       for i in group]), dtype=self.dtype)
                h = self.asr(self.asr.embed(x))
                for j, i in enumerate(group):
                    pieces[i] = h[j, wins[i][2]:wins[i][3]]
                mx.eval([pieces[i] for i in group])
                yield

    def _pool(self, logits: list[np.ndarray], n: int) -> np.ndarray:
        """80 ms speaker probabilities ``(n, speakers)``: the mean of the
        10 ms sigmoids of each encoder frame."""
        f = self.cfg.diar.subsampling
        flat = np.concatenate(logits, axis=0) if logits else np.zeros(
            (0, self.cfg.num_speakers), np.float32)
        return _sigmoid(flat[: n * f]).reshape(n, f, -1).mean(1)

    def fuse(self, asr_states: mx.array, speaker_probs: np.ndarray) -> mx.array:
        fu = self.fusion
        active = mx.array((speaker_probs > self.cfg.threshold).astype(np.float32),
                          dtype=asr_states.dtype)
        spk = fu.diar_norm(active) @ fu.diar_kernel.astype(asr_states.dtype)
        return fu.proj(fu.asr_norm(asr_states) + spk)

    def steps(self, feats: np.ndarray, valid: int, tokens: int,
              batch: int = MAX_WINDOW_BATCH, dtype=None,
              keep_states: bool = True, join: bool = True):
        """Encode one recording a window at a time: a generator that yields
        after each ASR batch of up to ``batch`` windows, each speaker
        chunk and each fusion step, and returns the
        :class:`EncoderOutput`, its rows in ``dtype`` (the encoder's when
        None) and its ASR states only with ``keep_states``; without
        ``join`` the rows stay in their fusion steps, in ``parts``. ``feats``
        comes from :func:`features`, ``valid`` is its valid frames,
        ``tokens`` the prompt's audio tokens for it."""
        n = feats.shape[0] // self.cfg.asr.subsampling
        pieces: dict[int, mx.array] = {}
        yield from self.asr_steps(normalize_per_feature(feats, valid), pieces,
                                  batch)
        states = (mx.concatenate([pieces[i] for i in range(len(pieces))], axis=0)
                  if pieces else mx.zeros((0, self.cfg.asr.d_model), self.dtype))
        pieces.clear()
        states = states[:tokens]
        mx.eval(states)
        logits: list[np.ndarray] = []
        yield from self.diar.logit_steps(feats, logits)
        probs = self._pool(logits, n)[:tokens]
        rows = []
        for a in range(0, tokens, FUSE_ROWS):
            part = self.fuse(states[a:a + FUSE_ROWS], probs[a:a + FUSE_ROWS])
            rows.append(part.astype(dtype or part.dtype))
            mx.eval(rows[-1])
            yield
        if not keep_states:
            states = None
        if not join:
            return EncoderOutput(embeds=None, asr_states=states,
                                 speaker_probs=probs, parts=rows)
        embeds = (rows[0] if len(rows) == 1 else mx.concatenate(rows)
                  if rows else mx.zeros((0, self.cfg.proj_dim), dtype or self.dtype))
        rows.clear()
        mx.eval(embeds)
        return EncoderOutput(embeds=embeds, asr_states=states,
                             speaker_probs=probs)

    def __call__(self, feats: np.ndarray, valid: int, tokens: int) -> EncoderOutput:
        """Encode one recording in one call; see :meth:`steps`."""
        gen = self.steps(feats, valid, tokens)
        while True:
            try:
                next(gen)
            except StopIteration as done:
                return done.value

    def ctc_logits(self, asr_states: mx.array):
        """Yield ``(first 40 ms frame, logits)`` per window of the CTC head:
        the kept frames' raw logits ``(frames, vocab)``, never the whole
        recording's at once."""
        c = self.cfg.ctc
        if c is None:
            raise ValueError("this encoder GGUF has no CTC timestamp head")
        u = c.upsampling
        for a, b, k0, k1 in self.windows(asr_states.shape[0]):
            h = self.ctc.hidden(asr_states[a:b])[k0 * u:k1 * u]
            yield (a + k0) * u, self.ctc.classifier(h)

    def ctc_columns(self, asr_states: mx.array, columns) -> dict:
        """Per 40 ms frame of the CTC head: the greedy class, the log
        probability of the blank and of each class in ``columns``."""
        c = self.cfg.ctc
        cols = mx.array(np.asarray(columns, np.int32))
        argmax, blank, picked = [], [], []
        for _, logits in self.ctc_logits(asr_states):
            logits = logits.astype(mx.float32)
            lse = mx.logsumexp(logits, axis=-1, keepdims=True)
            argmax.append(mx.argmax(logits, axis=-1))
            blank.append(logits[:, c.blank_id] - lse[:, 0])
            picked.append(mx.take(logits, cols, axis=1) - lse)
            mx.eval(argmax[-1], blank[-1], picked[-1])
        return {"argmax": np.array(mx.concatenate(argmax)).astype(np.int32),
                "blank": np.array(mx.concatenate(blank)),
                "columns": np.array(mx.concatenate(picked))}


# -- loading ------------------------------------------------------------------


def remap_tensor(name: str, arr: mx.array) -> tuple[str, mx.array]:
    """``(module path, array)`` in :class:`SpeechEncoder` of GGUF tensor
    ``name``, in numpy order as read."""
    if name == "preprocessor.fb":
        return "mel_filters", arr.astype(mx.float32)
    if name == "diar.head.subpixel_upsample.weight":
        # torch Conv1d (out, in, k) -> MLX (out, k, in).
        arr = arr.transpose(0, 2, 1)
    elif name == "ctc.classifier.weight" and arr.ndim == 3:
        arr = arr[:, :, 0]
    if name.startswith("diar.") and not name.startswith("diar.head."):
        name = "diar.encoder." + name[len("diar."):]
    return name, arr


def load(path, dtype=mx.float32) -> tuple[SpeechEncoder, np.ndarray, dict]:
    """``(encoder, mel filterbank, header fields)`` of the encoder GGUF at
    ``path``. Weights load as stored (F16 or F32); activations run in
    ``dtype``."""
    from gguf import GGMLQuantizationType as Q
    from gguf import GGUFReader

    from gmlx.load.gguf_meta import as_kv_dict

    reader = GGUFReader(str(Path(path).expanduser()))
    fields = as_kv_dict(reader)
    if fields.get("general.architecture") != ARCH:
        raise ValueError(f"{path}: not a {ARCH} (Nemotron speech encoder) GGUF")
    cfg = SpeechConfig.from_gguf(fields)
    enc = SpeechEncoder(cfg)
    weights = []
    for t in reader.tensors:
        if t.tensor_type not in (Q.F32, Q.F16):
            raise ValueError(f"{t.name}: unsupported tensor type {t.tensor_type.name}")
        shape = tuple(int(d) for d in reversed(t.shape))
        weights.append(remap_tensor(t.name, mx.array(np.asarray(t.data).reshape(shape))))
    enc.load_weights(weights, strict=True)
    if dtype != mx.float32:
        enc.set_dtype(dtype)
    mx.eval(enc.parameters())
    return enc, np.array(enc.mel_filters.astype(mx.float32)), fields


__all__ = ["ARCH", "SpeechConfig", "SpeechEncoder", "EncoderOutput",
           "features", "normalize_per_feature", "num_audio_tokens",
           "num_frames", "load"]
