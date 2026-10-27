# SPDX-License-Identifier: Apache-2.0
# Ported from Hugging Face transformers 5.16.1
# models/granite_speech5/modeling_granite_speech5.py and
# feature_extraction_granite_speech5.py, and modified for gmlx (Apache-2.0;
# see licenses/transformers-LICENSE).
"""Granite Speech 5.0 TurboCTC: the speech companion that transcribes audio
for Granite Switch.

A conformer CTC encoder over stacked log-mel and delta features. Each block
runs a half feed-forward, attention within 128-frame blocks with Shaw
relative positions, a GLU and depthwise convolution with BatchNorm, and a
second half feed-forward. Layers 0 and 1 halve the frame rate. The middle
layer feeds its CTC posteriors back into the hidden states. The transcript
is the greedy CTC path: each frame's top token, repeats collapsed, blanks
dropped.

The weights come from the GGUF that ``scripts/convert_granite_speech5_ctc.py``
writes, which also carries the mel filterbank and the STFT window."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from . import chunking

ARCH = "granite-speech5-ctc"
# SentencePiece's word-start mark.
_SPACE = "\u2581"


@dataclass
class Config:
    hidden_size: int = 1024
    intermediate_size: int = 4096
    num_hidden_layers: int = 16
    num_attention_heads: int = 8
    head_dim: int = 128
    context_size: int = 128
    max_position_embeddings: int = 512
    conv_kernel_size: int = 7
    conv_expansion_factor: int = 2
    subsample_layers: tuple = (0, 1)
    vocab_size: int = 16384
    num_mel_bins: int = 80
    layer_norm_eps: float = 1e-5
    batch_norm_eps: float = 1e-5
    sample_rate: int = 16000
    n_fft: int = 512
    hop_length: int = 160
    delta_win_length: int = 3
    logmel_floor_db: float = 8.0
    frame_stacking: int = 2
    blank_id: int = 0

    @classmethod
    def from_gguf(cls, fields) -> "Config":
        p = ARCH + "."
        get = fields.get
        return cls(
            hidden_size=get(p + "embedding_length", 1024),
            intermediate_size=get(p + "feed_forward_length", 4096),
            num_hidden_layers=get(p + "block_count", 16),
            num_attention_heads=get(p + "attention.head_count", 8),
            head_dim=get(p + "attention.key_length", 128),
            context_size=get(p + "attention.context_size", 128),
            max_position_embeddings=get(p + "attention.max_relative_position", 512),
            conv_kernel_size=get(p + "conv.kernel_size", 7),
            conv_expansion_factor=get(p + "conv.expansion_factor", 2),
            subsample_layers=tuple(get(p + "subsample_layers", (0, 1))),
            vocab_size=get(p + "vocab_size", 16384),
            num_mel_bins=get(p + "audio.num_mel_bins", 80),
            layer_norm_eps=get(p + "attention.layer_norm_epsilon", 1e-5),
            batch_norm_eps=get(p + "conv.batch_norm_epsilon", 1e-5),
            sample_rate=get(p + "audio.sample_rate", 16000),
            n_fft=get(p + "audio.n_fft", 512),
            hop_length=get(p + "audio.hop_length", 160),
            delta_win_length=get(p + "audio.delta_win_length", 3),
            logmel_floor_db=get(p + "audio.logmel_floor_db", 8.0),
            frame_stacking=get(p + "audio.frame_stacking", 2),
            blank_id=get(p + "ctc.blank_token_id", 0),
        )


# -- frontend ----------------------------------------------------------------

def _deltas(x: mx.array, win_length: int) -> mx.array:
    """torchaudio's ``compute_deltas`` along axis 0, edges replicated."""
    n = (win_length - 1) // 2
    denom = 2 * sum(i * i for i in range(1, n + 1))
    pad = mx.concatenate([mx.repeat(x[:1], n, axis=0), x, mx.repeat(x[-1:], n, axis=0)])
    T = x.shape[0]
    out = mx.zeros_like(x)
    for i in range(1, n + 1):
        out = out + i * (pad[n + i:n + i + T] - pad[n - i:n - i + T])
    return out / denom


def features(wave: mx.array, mel_filters: mx.array, window: mx.array,
             cfg: Config) -> mx.array:
    """The encoder input for one mono clip at ``cfg.sample_rate``: log-mel
    frames, floored ``logmel_floor_db`` below the clip's peak, each joined
    with its delta, then stacked in pairs. ``(frames, 4 * mel_bins)``
    float32; no frames for a clip shorter than one hop."""
    wave = wave.reshape(-1).astype(mx.float32)
    hop, n_fft, stack = cfg.hop_length, cfg.n_fft, cfg.frame_stacking
    mel_frames = wave.shape[0] // hop
    num_frames = stack * -(-mel_frames // stack)
    if num_frames == 0:
        return mx.zeros((0, 4 * cfg.num_mel_bins), dtype=mx.float32)
    need = (num_frames - 1) * hop + 1
    if wave.shape[0] < need:
        wave = mx.pad(wave, [(0, need - wave.shape[0])])
    half = n_fft // 2
    # torch.stft(center=True) reflects the clip at both ends.
    left = wave[1:half + 1][::-1]
    right = wave[-half - 1:-1][::-1]
    padded = mx.concatenate([left, wave, right])
    n = 1 + (padded.shape[0] - n_fft) // hop
    frames = mx.as_strided(padded, shape=(n, n_fft), strides=(hop, 1))[:num_frames]
    spec = mx.fft.rfft(frames * window, axis=-1)
    power = mx.square(mx.abs(spec))
    logmel = mx.log10(mx.maximum(power @ mel_filters, 1e-10))
    logmel = mx.maximum(logmel, mx.max(logmel) - cfg.logmel_floor_db) / 4 + 1
    feats = mx.concatenate([logmel, _deltas(logmel, cfg.delta_win_length)], axis=-1)
    return feats.reshape(-1, stack * feats.shape[-1])


# -- encoder -----------------------------------------------------------------

class FeedForward(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.linear1 = nn.Linear(cfg.hidden_size, cfg.intermediate_size)
        self.linear2 = nn.Linear(cfg.intermediate_size, cfg.hidden_size)

    def __call__(self, x):
        return self.linear2(nn.silu(self.linear1(x)))


class Attention(nn.Module):
    """Attention within blocks of ``context_size`` frames, with Shaw's
    relative position embeddings as an additive bias."""

    def __init__(self, cfg: Config):
        super().__init__()
        inner = cfg.num_attention_heads * cfg.head_dim
        self.heads, self.head_dim, self.ctx = cfg.num_attention_heads, cfg.head_dim, cfg.context_size
        self.scale = cfg.head_dim ** -0.5
        self.q_proj = nn.Linear(cfg.hidden_size, inner, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, inner, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, inner, bias=False)
        self.o_proj = nn.Linear(inner, cfg.hidden_size)
        self.rel_pos_emb = nn.Embedding(2 * cfg.max_position_embeddings + 1, cfg.head_dim)

    def __call__(self, x: mx.array, dists: mx.array) -> mx.array:
        _, L, D = x.shape
        C, H, hd = self.ctx, self.heads, self.head_dim
        pad = -L % C
        if pad:
            x = mx.pad(x, [(0, 0), (0, pad), (0, 0)])
        nb = x.shape[1] // C

        def heads(t):
            return t.reshape(nb, C, H, hd).transpose(0, 2, 1, 3)

        q, k, v = heads(self.q_proj(x)), heads(self.k_proj(x)), heads(self.v_proj(x))
        rel = self.rel_pos_emb(dists) * self.scale                    # (C, C, hd)
        qs = q.transpose(2, 0, 1, 3).reshape(C, nb * H, hd)
        bias = (qs @ rel.transpose(0, 2, 1)).reshape(C, nb, H, C).transpose(1, 2, 0, 3)
        if pad:
            keep = (mx.arange(nb * C) < L).reshape(nb, 1, 1, C)
            bias = mx.where(keep, bias, mx.array(mx.finfo(bias.dtype).min, bias.dtype))
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale,
                                                   mask=bias.astype(q.dtype))
        out = out.transpose(0, 2, 1, 3).reshape(1, nb * C, H * hd)
        return self.o_proj(out[:, :L])


class ConvModule(nn.Module):
    """GLU, then a depthwise convolution (``SAME`` padding, stride 1 or 2),
    BatchNorm in float32 folded to a scale and shift, SiLU, and a pointwise
    projection."""

    def __init__(self, cfg: Config, stride: int = 1):
        super().__init__()
        inner = cfg.hidden_size * cfg.conv_expansion_factor
        self.stride, self.kernel = stride, cfg.conv_kernel_size
        self.pointwise_lin1 = nn.Linear(cfg.hidden_size, inner * 2)
        self.depthwise = mx.zeros((cfg.conv_kernel_size, inner))
        self.norm_scale = mx.ones((inner,))
        self.norm_shift = mx.zeros((inner,))
        self.pointwise_lin2 = nn.Linear(inner, cfg.hidden_size)

    def __call__(self, x: mx.array) -> mx.array:
        dt = x.dtype
        a, b = mx.split(self.pointwise_lin1(x), 2, axis=-1)
        x = (a * mx.sigmoid(b)).astype(mx.float32)
        L, K, s = x.shape[1], self.kernel, self.stride
        p = (K - 1) // 2
        x = mx.pad(x, [(0, 0), (p, p), (0, 0)])
        n = (L + 2 * p - K) // s + 1
        w = self.depthwise.astype(mx.float32)
        y = sum(x[:, k:k + s * (n - 1) + 1:s] * w[k] for k in range(K))
        y = nn.silu(y * self.norm_scale + self.norm_shift)
        return self.pointwise_lin2(y.astype(dt))


class Block(nn.Module):
    def __init__(self, cfg: Config, subsample: bool = False):
        super().__init__()
        self.subsample = subsample
        self.feed_forward1 = FeedForward(cfg)
        self.self_attn = Attention(cfg)
        self.conv = ConvModule(cfg, stride=2 if subsample else 1)
        self.feed_forward2 = FeedForward(cfg)
        eps = cfg.layer_norm_eps
        self.norm_feed_forward1 = nn.LayerNorm(cfg.hidden_size, eps=eps)
        self.norm_self_att = nn.LayerNorm(cfg.hidden_size, eps=eps)
        self.norm_conv = nn.LayerNorm(cfg.hidden_size, eps=eps)
        self.norm_feed_forward2 = nn.LayerNorm(cfg.hidden_size, eps=eps)
        self.norm_out = nn.LayerNorm(cfg.hidden_size, eps=eps)

    def __call__(self, x, dists):
        x = x + 0.5 * self.feed_forward1(self.norm_feed_forward1(x))
        x = x + self.self_attn(self.norm_self_att(x), dists)
        conv = self.conv(self.norm_conv(x))
        if self.subsample:
            half = x.shape[1] // 2
            x = x[:, :2 * half].reshape(1, half, 2, -1).mean(axis=2) + conv[:, :half]
        else:
            x = x + conv
        x = x + 0.5 * self.feed_forward2(self.norm_feed_forward2(x))
        return self.norm_out(x)


class Encoder(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.input_linear = nn.Linear(cfg.num_mel_bins * 4, cfg.hidden_size)
        self.layers = [Block(cfg, subsample=i in cfg.subsample_layers)
                       for i in range(cfg.num_hidden_layers)]
        self.out = nn.Linear(cfg.hidden_size, cfg.vocab_size)
        self.out_mid = nn.Linear(cfg.vocab_size, cfg.hidden_size)
        pos = mx.arange(cfg.context_size)
        rel = pos[:, None] - pos[None, :]
        self._dists = mx.clip(rel, -cfg.context_size, cfg.context_size) + cfg.max_position_embeddings

    def __call__(self, feats: mx.array, dtype=mx.bfloat16) -> mx.array:
        """CTC logits ``(frames, vocab)`` for one clip's features."""
        x = self.input_linear(feats[None].astype(dtype))
        mid = self.cfg.num_hidden_layers // 2
        for i, layer in enumerate(self.layers):
            x = layer(x, self._dists)
            if i + 1 == mid:
                probs = mx.softmax(self.out(x).astype(mx.float32), axis=-1).astype(x.dtype)
                x = x + self.out_mid(probs)
        return self.out(x)[0]


# -- tokens ------------------------------------------------------------------

class CTCVocab:
    """The CTC vocabulary: greedy path collapse and the tokenizer's decode
    (U+2581 to a space, byte pieces joined, one leading space dropped),
    special tokens skipped."""

    def __init__(self, pieces: list[str], types: list[int], blank_id: int = 0):
        self.pieces = pieces
        self.blank_id = blank_id
        self.skip = {i for i, t in enumerate(types) if t == 3}       # CONTROL
        self.byte = {i: int(p[3:5], 16) for i, (p, t) in enumerate(zip(pieces, types))
                     if t == 6}                                         # BYTE

    def collapse(self, path: list[int]) -> list[int]:
        out, prev = [], None
        for t in path:
            if t != prev and t != self.blank_id:
                out.append(t)
            prev = t
        return out

    def decode(self, ids: list[int]) -> str:
        buf = bytearray()
        for i in ids:
            if i in self.skip:
                continue
            if i in self.byte:
                buf.append(self.byte[i])
            else:
                buf += self.pieces[i].replace(_SPACE, " ").encode("utf-8")
        text = buf.decode("utf-8", errors="replace")
        return text[1:] if text.startswith(" ") else text


# -- loading -----------------------------------------------------------------

def _gguf_fields(reader) -> dict:
    from gmlx.load.gguf_meta import as_kv_dict

    return as_kv_dict(reader)


def _tensor(t, dtype) -> mx.array:
    from gguf import GGMLQuantizationType as Q

    data = np.asarray(t.data)
    if t.tensor_type == Q.BF16:
        return mx.array(data.view(np.uint16)).view(mx.bfloat16).astype(dtype)
    if t.tensor_type in (Q.F32, Q.F16):
        return mx.array(data).astype(dtype)
    raise ValueError(f"{t.name}: unsupported tensor type {t.tensor_type.name}")


def _kquant_linear(t, bias: bool):
    from mlx_kquant.nn import KQuantLinear

    return KQuantLinear(in_dims=int(t.shape[0]), out_dims=int(t.shape[1]),
                        bias=bias, codec="q8_0")


class SpeechCTC:
    """The loaded companion: features, encoder, greedy CTC and the long-clip
    windows, behind :meth:`transcribe`."""

    def __init__(self, cfg: Config, encoder: Encoder, vocab: CTCVocab,
                 mel_filters: mx.array, window: mx.array, dtype):
        self.cfg, self.encoder, self.vocab = cfg, encoder, vocab
        self.mel_filters, self.window, self.dtype = mel_filters, window, dtype

    @property
    def sample_rate(self) -> int:
        return self.cfg.sample_rate

    def logits(self, wave) -> mx.array:
        feats = features(mx.array(wave), self.mel_filters, self.window, self.cfg)
        if feats.shape[0] == 0:
            return mx.zeros((0, self.cfg.vocab_size))
        return self.encoder(feats, self.dtype)

    def _one(self, wave) -> str:
        logits = self.logits(wave)
        if logits.shape[0] == 0:
            return ""
        path = mx.argmax(logits, axis=-1).tolist()
        return self.vocab.decode(self.vocab.collapse(path)).strip()

    def transcribe(self, wave: np.ndarray) -> str:
        """The transcript of a mono float32 clip at :attr:`sample_rate`.
        Clips past two minutes run in overlapping windows."""
        segments = chunking.split_waveform(np.asarray(wave, dtype=np.float32),
                                           self.cfg.sample_rate)
        return chunking.merge_transcripts([self._one(s) for s in segments]).strip()


def is_companion(path) -> bool:
    """True for a GGUF this module loads."""
    from gmlx.load.gguf_meta import read_string

    from gguf import GGUFReader

    try:
        return read_string(GGUFReader(str(path)), "general.architecture") == ARCH
    except Exception:  # noqa: BLE001 - any unreadable file is not one
        return False


def load(path, dtype=None) -> SpeechCTC:
    """The companion GGUF at ``path``, with weights in ``dtype`` (bfloat16 on
    the GPU and float32 on the CPU by default)."""
    from gguf import GGMLQuantizationType as Q
    from gguf import GGUFReader

    reader = GGUFReader(str(Path(path).expanduser()))
    fields = _gguf_fields(reader)
    if fields.get("general.architecture") != ARCH:
        raise ValueError(f"{path}: not a {ARCH} GGUF")
    if dtype is None:
        dtype = mx.float32 if mx.default_device() == mx.cpu else mx.bfloat16
    cfg = Config.from_gguf(fields)
    tensors = {t.name: t for t in reader.tensors}
    enc = Encoder(cfg)
    weights, quantized = [], {}
    for name, t in tensors.items():
        if name.startswith("frontend.") or ".conv.norm." in name:
            continue
        if name.endswith(".conv.depthwise_conv.weight"):
            w = _tensor(t, mx.float32).reshape(-1, cfg.conv_kernel_size).T
            weights.append((name.replace("depthwise_conv.weight", "depthwise"), w))
        elif t.tensor_type == Q.Q8_0:
            quantized[name[:-len(".weight")]] = t
        else:
            weights.append((name, _tensor(t, dtype)))
    eps = cfg.batch_norm_eps
    for i in range(cfg.num_hidden_layers):
        pre = f"layers.{i}.conv.norm."
        g, b, m, v = (_tensor(tensors[pre + k], mx.float32)
                      for k in ("weight", "bias", "running_mean", "running_var"))
        scale = g / mx.sqrt(v + eps)
        weights += [(f"layers.{i}.conv.norm_scale", scale),
                    (f"layers.{i}.conv.norm_shift", b - m * scale)]
    for prefix, t in quantized.items():
        parent_path, _, leaf = prefix.rpartition(".")
        parent = enc
        for part in parent_path.split(".") if parent_path else []:
            parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
        lin = _kquant_linear(t, prefix + ".bias" in tensors)
        setattr(parent, leaf, lin)
        weights.append((prefix + ".weight", mx.array(np.asarray(t.data))))
        weights += [(f"{prefix}.{k}", v) for k, v in lin.parameters().items()
                    if k not in ("weight", "bias")]
    enc.load_weights(weights)
    mx.eval(enc.parameters())
    vocab = CTCVocab(list(fields["tokenizer.ggml.tokens"]),
                     list(fields["tokenizer.ggml.token_type"]), cfg.blank_id)
    mel = _tensor(tensors["frontend.mel_filters"], mx.float32)
    window = _tensor(tensors["frontend.window"], mx.float32)
    return SpeechCTC(cfg, enc, vocab, mel, window, dtype)


__all__ = ["Config", "Encoder", "SpeechCTC", "CTCVocab", "features", "load",
           "is_companion"]
