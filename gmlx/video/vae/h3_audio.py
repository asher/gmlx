"""MiniMax-H3's audio VAE, from Comfy-Org's safetensors file as published.

A DAC encoder and an attention head give 32 latent channels per 800
samples of 32 kHz audio, 40 latents a second. A BigVGAN decoder turns
them back into a waveform clamped to [-1, 1]. Each stereo channel runs
through the mono model on its own, as a batch row. The file holds the
weight-norm folded into plain conv weights, and the anti-aliasing filters
and latent statistics as tensors. The model runs in float32: in bfloat16
it decodes about 20 dB too quiet. Its products stay float32 in a process
that multiplies float32 at TF32 (gmlx.image.float32), where TF32 products
would put its decode 38 dB from the float32 one.

Latents are (B, 32, 2, T) in model space, waveforms (B, 2, samples).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from gmlx.image.float32 import (conv1d, conv_transpose1d, exact_matmul,
                                f32_products_exact)

HOP = 800
SAMPLE_RATE = 32000
QUERY_BLOCK = 512


@dataclass(frozen=True)
class Config:
    encoder_dim: int = 64
    encoder_rates: tuple = (2, 4, 4, 5, 5)
    latent_dim: int = 2048
    z: int = 32
    heads: int = 8
    decoder_dim: int = 1024
    upsample_rates: tuple = (5, 5, 2, 2, 2, 2, 2)
    upsample_kernels: tuple = (9, 9, 4, 4, 4, 4, 4)
    resblock_kernels: tuple = (3, 7, 11)
    resblock_dilations: tuple = (1, 3, 5)
    filter_size: int = 12

    @property
    def latent_channels(self) -> int:
        return self.z


def config_from_shapes(shapes: dict):
    """The config of the H3 audio VAE whose torch shapes are ``shapes``,
    or None for another tree."""
    try:
        enc = shapes["encoder.block.0.weight"][0]
        latent = shapes["dec_in_proj.weight"][0]
        z = shapes["mean_proj.weight"][0]
        dec = shapes["decoder.conv_pre.weight"][0]
        ups = 1 + max(int(n.split(".")[2]) for n in shapes
                      if n.startswith("decoder.ups."))
        blocks = 1 + max(int(n.split(".")[2]) for n in shapes
                         if n.startswith("encoder.block.")) - 3
    except (KeyError, ValueError, IndexError):
        return None
    cfg = Config(encoder_dim=enc, latent_dim=latent, z=z, decoder_dim=dec)
    if (ups != len(cfg.upsample_rates) or blocks != len(cfg.encoder_rates)
            or "pre_block.attn.zero_k_bias" not in shapes
            or math.prod(cfg.encoder_rates) != HOP):
        return None
    return cfg


class Conv1d(nn.Conv1d):
    """A convolution that is float32 in a TF32 process."""

    def __call__(self, x: mx.array) -> mx.array:
        y = conv1d(x, self.weight, self.stride, self.padding, self.dilation,
                   self.groups)
        return y + self.bias if "bias" in self else y


def snake(x: mx.array, alpha: mx.array, beta: mx.array) -> mx.array:
    t = mx.sin(alpha * x)
    return x + (t * t) * mx.reciprocal(beta + 1e-9)


class Snake(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.alpha = mx.ones((channels,))

    def __call__(self, x: mx.array) -> mx.array:
        return snake(x, self.alpha, self.alpha)


class SnakeBeta(nn.Module):
    """Snake with alpha and beta held as logarithms."""

    def __init__(self, channels: int):
        super().__init__()
        self.alpha = mx.zeros((channels,))
        self.beta = mx.zeros((channels,))

    def __call__(self, x: mx.array) -> mx.array:
        return snake(x, mx.exp(self.alpha), mx.exp(self.beta))


def _edge(x: mx.array, before: int, after: int) -> mx.array:
    return mx.pad(x, [(0, 0), (before, after), (0, 0)], mode="edge")


def _per_channel(x: mx.array):
    b, n, c = x.shape
    return x.transpose(0, 2, 1).reshape(b * c, n, 1), (b, c)


def _back(y: mx.array, bc) -> mx.array:
    b, c = bc
    return y.reshape(b, c, y.shape[1]).transpose(0, 2, 1)


class UpSample(nn.Module):
    """x2 upsampling by a Kaiser-windowed sinc, the same filter on every
    channel."""

    def __init__(self, size: int):
        super().__init__()
        self.filter = mx.zeros((1, size, 1))
        self.pad = size // 2 - 1
        self.crop = self.pad * 2 + (size - 2) // 2

    def __call__(self, x: mx.array) -> mx.array:
        y, bc = _per_channel(_edge(x, self.pad, self.pad))
        y = conv_transpose1d(y, self.filter, stride=2) * 2
        y = y[:, self.crop:y.shape[1] - self.crop]
        return _back(y, bc)


class DownSample(nn.Module):
    """x2 downsampling after a Kaiser-windowed sinc low pass."""

    def __init__(self, size: int):
        super().__init__()
        self.filter = mx.zeros((1, size, 1))
        self.left = size // 2 - int(size % 2 == 0)
        self.right = size // 2

    def __call__(self, x: mx.array) -> mx.array:
        y, bc = _per_channel(_edge(x, self.left, self.right))
        return _back(conv1d(y, self.filter, stride=2), bc)


class Lowpass(nn.Module):
    def __init__(self, size: int):
        super().__init__()
        self.lowpass = DownSample(size)

    def __call__(self, x: mx.array) -> mx.array:
        return self.lowpass(x)


class Activation(nn.Module):
    """Upsample, SnakeBeta, downsample."""

    def __init__(self, channels: int, size: int):
        super().__init__()
        self.act = SnakeBeta(channels)
        self.upsample = UpSample(size)
        self.downsample = Lowpass(size)

    def __call__(self, x: mx.array) -> mx.array:
        return self.downsample(self.act(self.upsample(x)))


class Seq(nn.Module):
    def __init__(self, *layers):
        super().__init__()
        self.block = list(layers)

    def __call__(self, x: mx.array) -> mx.array:
        for layer in self.block:
            x = layer(x)
        return x


class ResidualUnit(nn.Module):
    def __init__(self, dim: int, dilation: int):
        super().__init__()
        self.block = [Snake(dim),
                      Conv1d(dim, dim, 7, dilation=dilation,
                                padding=3 * dilation),
                      Snake(dim), Conv1d(dim, dim, 1)]

    def __call__(self, x: mx.array) -> mx.array:
        y = x
        for layer in self.block:
            y = layer(y)
        return y + x


def _encoder_block(dim: int, stride: int) -> Seq:
    half = dim // 2
    return Seq(ResidualUnit(half, 1), ResidualUnit(half, 3),
               ResidualUnit(half, 9), Snake(half),
               Conv1d(half, dim, 2 * stride, stride=stride,
                         padding=math.ceil(stride / 2)))


class Encoder(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        d = cfg.encoder_dim
        layers = [Conv1d(1, d, 7, padding=3)]
        for s in cfg.encoder_rates:
            d *= 2
            layers.append(_encoder_block(d, s))
        layers += [Snake(d), Conv1d(d, cfg.latent_dim, 3, padding=1)]
        self.block = layers

    def __call__(self, x: mx.array) -> mx.array:
        for layer in self.block:
            x = layer(x)
        return x


def _matmul(a: mx.array, b: mx.array) -> mx.array:
    return a @ b if f32_products_exact() else exact_matmul(a, b)


class Linear(nn.Linear):
    """A linear layer whose product is float32 in a TF32 process."""

    def __call__(self, x: mx.array) -> mx.array:
        if f32_products_exact():
            return super().__call__(x)
        y = exact_matmul(x, self.weight, b_t=True)
        return y + self.bias if "bias" in self else y


def causal_attention(q: mx.array, k: mx.array, v: mx.array,
                     scale: float) -> mx.array:
    """Causal attention over (B, heads, n, d), QUERY_BLOCK query rows at a
    time over the keys up to the block's end, each block evaluated before
    the next, so its scores stay a block wide. Its products stay float32
    in a TF32 process."""
    exact = f32_products_exact()
    n = q.shape[2]
    out = []
    for i in range(0, n, QUERY_BLOCK):
        j = min(i + QUERY_BLOCK, n)
        if exact:
            # A causal mask with fewer queries than keys ends at the last key.
            out.append(mx.fast.scaled_dot_product_attention(
                q[:, :, i:j], k[:, :, :j], v[:, :, :j], scale=scale,
                mask="causal"))
        else:
            s = exact_matmul(q[:, :, i:j], k[:, :, :j].swapaxes(-1, -2)) \
                * scale
            s = s + mx.triu(mx.full((j - i, j), -mx.inf, s.dtype), k=i + 1)
            out.append(exact_matmul(mx.softmax(s, axis=-1, precise=True),
                                    v[:, :, :j]))
        if j < n:
            mx.eval(out[-1])
    return out[0] if len(out) == 1 else mx.concatenate(out, axis=2)


def adaptive_pool(x: mx.array, out: int) -> mx.array:
    """torch's adaptive_avg_pool1d over the last axis: output i averages
    inputs floor(i n / out) to ceil((i + 1) n / out)."""
    n = x.shape[-1]
    if n % out == 0:
        return x.reshape(*x.shape[:-1], out, n // out).mean(axis=-1)
    pool = [[0.0] * out for _ in range(n)]
    for i in range(out):
        lo, hi = (i * n) // out, -(-(i + 1) * n // out)
        for j in range(lo, hi):
            pool[j][i] = 1.0 / (hi - lo)
    return _matmul(x, mx.array(pool, x.dtype))


class CausalAttention(nn.Module):
    def __init__(self, dim: int, out: int, heads: int):
        super().__init__()
        self.heads, self.out = heads, out
        self.qkv = Linear(dim, 3 * dim, bias=False)
        self.q_bias = mx.zeros((dim,))
        self.zero_k_bias = mx.zeros((dim,))
        self.v_bias = mx.zeros((dim,))
        self.proj = Linear(out, out)

    def __call__(self, x: mx.array) -> mx.array:
        b, n, c = x.shape
        hd = c // self.heads
        bias = mx.concatenate([self.q_bias, self.zero_k_bias, self.v_bias])
        qkv = (self.qkv(x) + bias).reshape(b, n, 3, self.heads, hd)
        q, k, v = (qkv[:, :, i].transpose(0, 2, 1, 3) for i in range(3))
        o = causal_attention(q, k, v, hd ** -0.5)
        o = o.mean(axis=1)
        return self.proj(adaptive_pool(o, self.out))


class GeGluMlp(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.w0 = Linear(dim, hidden)
        self.w1 = Linear(dim, hidden)
        self.w2 = Linear(hidden, dim)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.norm(x)
        return self.w2(nn.gelu_approx(self.w0(x)) * self.w1(x))


class AttnProjection(nn.Module):
    def __init__(self, dim: int, out: int, heads: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = CausalAttention(dim, out, heads)
        self.proj = Linear(dim, out)
        self.norm3 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(out)
        self.mlp = GeGluMlp(out, 2 * out)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.proj(self.norm3(x)) + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class AMPBlock(nn.Module):
    def __init__(self, ch: int, kernel: int, dilations: tuple, size: int):
        super().__init__()
        pad = (kernel - 1) // 2
        self.convs1 = [Conv1d(ch, ch, kernel, dilation=d, padding=d * pad)
                       for d in dilations]
        self.convs2 = [Conv1d(ch, ch, kernel, padding=pad)
                       for _ in dilations]
        self.activations = [Activation(ch, size)
                            for _ in range(2 * len(dilations))]

    def __call__(self, x: mx.array) -> mx.array:
        acts = self.activations
        for c1, c2, a1, a2 in zip(self.convs1, self.convs2, acts[::2],
                                  acts[1::2]):
            x = x + c2(a2(c1(a1(x))))
        return x


class Upsample(nn.Module):
    """A ConvTranspose1d whose padding crops each side."""

    def __init__(self, cin: int, cout: int, kernel: int, stride: int):
        super().__init__()
        self.stride, self.crop = stride, (kernel - stride) // 2
        self.weight = mx.zeros((cout, kernel, cin))
        self.bias = mx.zeros((cout,))

    def __call__(self, x: mx.array) -> mx.array:
        y = conv_transpose1d(x, self.weight, self.stride, self.crop)
        return y + self.bias


class BigVGAN(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        ch = cfg.decoder_dim
        self.conv_pre = Conv1d(cfg.latent_dim, ch, 7, padding=3)
        self.ups, self.resblocks = [], []
        for u, k in zip(cfg.upsample_rates, cfg.upsample_kernels):
            self.ups.append([Upsample(ch, ch // 2, k, u)])
            ch //= 2
            for rk in cfg.resblock_kernels:
                self.resblocks.append(AMPBlock(ch, rk, cfg.resblock_dilations,
                                               cfg.filter_size))
        self.activation_post = Activation(ch, cfg.filter_size)
        self.conv_post = Conv1d(ch, 1, 7, padding=3, bias=False)
        self.kernels = len(cfg.resblock_kernels)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.conv_pre(x)
        for i, (up,) in enumerate(self.ups):
            x = up(x)
            blocks = self.resblocks[i * self.kernels:(i + 1) * self.kernels]
            xs = blocks[0](x)
            for block in blocks[1:]:
                xs = xs + block(x)
            x = xs / self.kernels
            mx.eval(x)
        x = self.activation_post(x)
        return mx.clip(self.conv_post(x), -1.0, 1.0)


class H3AudioVAE(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.encoder = Encoder(cfg)
        self.pre_block = AttnProjection(cfg.latent_dim, cfg.z, cfg.heads)
        self.mean_proj = Conv1d(cfg.z, cfg.z, 1)
        self.dec_in_proj = Conv1d(cfg.z, cfg.latent_dim, 1)
        self.decoder = BigVGAN(cfg)
        self.latents_mean = mx.zeros((cfg.z,))
        self.latents_std = mx.ones((cfg.z,))

    def encode(self, waveform: mx.array) -> mx.array:
        """Model-space latents (B, 32, 2, T) of a stereo waveform (B, 2, n)
        in [-1, 1], right-padded with zeros to a whole latent."""
        b, s, n = waveform.shape
        pad = -n % HOP
        x = mx.pad(waveform.astype(mx.float32), [(0, 0), (0, 0), (0, pad)])
        x = x.reshape(b * s, n + pad, 1)
        x = self.pre_block(self.encoder(x))
        z = (self.mean_proj(x) - self.latents_mean) / self.latents_std
        return z.reshape(b, s, z.shape[1], self.cfg.z).transpose(0, 3, 1, 2)

    def decode(self, latents: mx.array) -> mx.array:
        """The stereo waveform (B, 2, T * 800) of model-space latents
        (B, 32, 2, T)."""
        b, c, s, t = latents.shape
        z = latents.astype(mx.float32).transpose(0, 2, 3, 1).reshape(
            b * s, t, c)
        z = z * self.latents_std + self.latents_mean
        y = self.decoder(self.dec_in_proj(z))
        return y.reshape(b, s, -1)
