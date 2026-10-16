# SPDX-License-Identifier: Apache-2.0 AND MIT
# SPDX-FileCopyrightText: Copyright 2026 The Qwen Team and The HuggingFace Team
# Ported from diffusers'
# src/diffusers/models/autoencoders/autoencoder_kl_qwenimage21.py at
# e0abab83, for one frame, and modified for gmlx (Apache-2.0; see
# licenses/diffusers-LICENSE).
"""The Qwen-Image 2.1 autoencoder.

Module paths equal the official tensor names, and activations are channels
last (B, H, W, C). The image has four channels, RGBA, and a latent pixel is
16 image pixels. The encoder has widths 96 times [1, 1, 2, 4, 8, 8] and two
resnets per down block, the decoder 144 times [8, 8, 8, 4, 2, 1] and three
per up block, and each mid block is a resnet, one attention head and a
resnet. ``quant_conv`` follows the encoder and ``post_quant_conv`` precedes
the decoder.

The official class is the image case of a video autoencoder, and this
module computes what it does for its one frame.

- Every residual block adds a shortcut without parameters. On the way down
  it is space to depth and a mean over groups of consecutive channels, and
  where the video layout halves time the frame is preceded by a frame of
  zeros. On the way up it repeats channels and is depth to space, and
  where the video layout doubles time only the last frame is kept.
- ``time_conv`` never runs for one frame, so the file's twelve
  ``time_conv`` tensors are left unread, and a file may leave them out.
  No tensor the model reads depends on where the video layout halves
  time, so a file's time flags are the official config's.
- A norm divides by the L2 norm over the channels, computed in float32,
  and multiplies by sqrt(C) and ``gamma``. ``gamma`` keeps the shape it has
  in the file, (C, 1, 1, 1) in a resnet and (C, 1, 1) in the attention.
- Downsampling pads zeros on the bottom and right and then runs a stride 2
  conv. Upsampling is nearest 2x and then a conv.
- ``decode`` clamps to [-1, 1]. ``encode`` gives an RGB image an alpha of
  ones, and the mean is the first half of the encoder's channels.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from gmlx.image import ops, pacing

from .conv import Conv2d
from .flux1_ae import channels_last, chunked_sdpa

# What no tensor the model reads determines, under its name in the official
# vae/config.json of Qwen/Qwen-Image-2.1: the downsamples at which the video
# layout halves time too, and the latent statistics.
CONSTANTS = {
    "temperal_downsample": (False, True, True, True),
    "latents_mean": (
        0.5126, 0.7721, -0.0631, 1.3506, -0.7855, -2.1025, -0.3458, 1.3722,
        1.8873, -1.7177, -0.651, 0.2732, 0.7562, -0.6163, -1.0277, 3.8363,
        2.021, 0.0472, 0.932, 2.0087, 2.4954, -0.1391, -1.4249, 1.8464,
        -0.5236, 1.2826, 3.7046, -1.3035, 2.7286, -1.4518, -1.9036, -1.9955,
        -0.0342, -1.0265, -0.7636, 3.0555, 0.0746, -3.0751, -0.1076, 1.7376,
        -1.0914, -1.9435, -0.2784, -1.368, 0.4809, -0.4433, 0.3764, 0.5729,
        -2.0595, 1.096, -1.326, -2.0211, -5.0179, 0.5275, 4.0162, 1.8505,
        0.3026, 1.9373, 1.4937, 0.2632, 0.5547, -1.7121, -0.1562, 0.0304),
    "latents_std": (
        3.2001, 3.2936, 3.4321, 3.0091, 3.1061, 4.0379, 4.0705, 3.791,
        3.0785, 3.65, 3.9308, 3.0904, 2.8778, 3.7675, 3.732, 5.0756,
        3.2864, 4.0397, 3.1317, 4.0443, 2.9249, 3.9454, 3.0988, 4.2489,
        3.4896, 3.8513, 3.9323, 3.4719, 3.7498, 4.283, 3.5694, 4.2467,
        3.9037, 3.2947, 5.077, 3.5075, 3.27, 3.4767, 2.8063, 5.1125,
        3.5327, 4.7833, 3.1286, 4.1819, 3.8527, 3.8312, 3.5605, 4.3875,
        3.9624, 4.0168, 3.5643, 4.055, 5.5614, 4.2963, 4.408, 3.4959,
        3.8747, 3.7608, 3.5735, 3.149, 3.7662, 3.6746, 3.4563, 3.8161),
}
CONSTANTS_SOURCE = ("Qwen/Qwen-Image-2.1",
                    "790c92633540aa0cb11d9abf19eb46d861714758")

LATENT_CHANNELS = 64
IMAGE_CHANNELS = 4

# The reference divides by max(|x|, NORM_FLOOR), |x| the L2 norm over the
# C channels. mx.fast.rms_norm adds its eps to the mean of squares,
# |x|^2 / C, so an eps of NORM_FLOOR^2 / C puts the same floor under |x|.
# The two agree at |x| = 0 and far below the floor, and within 5e-5 from
# |x| = 1e-10 up. At the floor the sum gives 1 / sqrt(2) of the max. The
# sum stays because it is one fused pass, and the max needs float32 copies
# of the activation.
NORM_FLOOR = 1e-12

# Zero padding before a stride 2 conv: one row on the bottom and one column
# on the right.
DOWN_PAD = ((0, 0), (0, 1), (0, 1), (0, 0))

# Up blocks of the decoder that a tiled decode runs in tiles: all but the
# first. The convs of the whole up path read 13 latent pixels across a cut,
# and a 512 pixel tile cropped at 13 would keep a fifth of each side. From
# the second up block on they read 7, and from the third on 3. The part
# before the tiles runs on the whole image, and its bytes grow fourfold
# with each up block it takes: on the official file at 1024 pixels in
# bfloat16 it measured 0.63 GiB through the first up block and 1.53 GiB
# through the second.
TILED_UP_BLOCKS = 4


@dataclass(frozen=True)
class QwenImage21VAEConfig:
    in_channels: int
    out_channels: int
    latent_channels: int
    base_dim: int
    decoder_base_dim: int
    dim_mult: tuple
    num_res_blocks: int
    temporal: tuple                 # per downsample: the video layout
                                    # halves time there too
    latents_mean: tuple = CONSTANTS["latents_mean"]
    latents_std: tuple = CONSTANTS["latents_std"]

    @property
    def scale(self) -> int:
        """Pixels per latent pixel along each axis."""
        return 2 ** (len(self.dim_mult) - 1)

    @property
    def encoder_dims(self) -> tuple:
        return tuple(self.base_dim * m for m in (1, *self.dim_mult))

    @property
    def decoder_dims(self) -> tuple:
        return tuple(self.decoder_base_dim * m
                     for m in (self.dim_mult[-1], *reversed(self.dim_mult)))


def whole_up_blocks(cfg: QwenImage21VAEConfig) -> int:
    """Up blocks of the decoder that a tiled decode runs on the whole
    image, before the ``TILED_UP_BLOCKS`` it runs in tiles."""
    return max(0, len(cfg.dim_mult) - TILED_UP_BLOCKS)


def expected_unused(cfg: QwenImage21VAEConfig) -> dict:
    """The time_conv tensors of the video layout, which a file may hold and
    one frame never runs, each with its torch shape: a weight and a bias
    at each resample where the config halves or doubles time. A downsample
    keeps its channels, and an upsample doubles them for the two frames it
    makes."""
    n = len(cfg.dim_mult)
    up = cfg.temporal[::-1]
    at = [(f"encoder.down_blocks.{i}.downsampler", cfg.encoder_dims[i + 1], 1)
          for i in range(n - 1) if cfg.temporal[i]]
    at += [(f"decoder.up_blocks.{i}.upsampler", cfg.decoder_dims[i + 1], 2)
           for i in range(n - 1) if up[i]]
    out = {}
    for name, c, grow in at:
        out[f"{name}.time_conv.weight"] = (grow * c, c, 1, 1)
        out[f"{name}.time_conv.bias"] = (grow * c,)
    return out


def config_disagreements(config: dict) -> list:
    """The fields of a vae/config.json that differ from ``CONSTANTS``. A
    field the config leaves out is none."""
    return [k for k, v in CONSTANTS.items()
            if k in config and (not isinstance(config[k], list)
                                or config[k] != list(v))]


def _count(shapes: dict, pattern: str) -> int:
    rx = re.compile(pattern)
    found = {int(m.group(1)) for n in shapes if (m := rx.match(n))}
    return max(found) + 1 if found else 0


_CONVS = ("encoder.conv_in.weight", "encoder.conv_out.weight",
          "quant_conv.weight", "post_quant_conv.weight",
          "decoder.conv_in.weight", "decoder.conv_out.weight",
          "decoder.mid_block.attentions.0.to_qkv.weight")


def config_from_shapes(shapes: dict) -> QwenImage21VAEConfig | None:
    """The config from official-named torch shapes, or None when the names
    are not this autoencoder's with conv kernels of rank 4 and the official
    number of levels. The time flags are the official ones, whichever
    time_conv tensors the file holds."""
    need = (*_CONVS, "encoder.norm_out.gamma", "decoder.norm_out.gamma")
    if any(n not in shapes for n in need):
        return None
    if any(len(shapes[n]) != 4 for n in _CONVS):
        return None
    levels = _count(shapes, r"encoder\.down_blocks\.(\d+)\.")
    per_block = _count(shapes, r"encoder\.down_blocks\.0\.resnets\.(\d+)\.")
    if levels < 1 or per_block < 1:
        return None
    if _count(shapes, r"decoder\.up_blocks\.(\d+)\.") != levels:
        return None
    base = int(shapes["encoder.conv_in.weight"][0])
    mult = []
    for i in range(levels):
        s = shapes.get(f"encoder.down_blocks.{i}.resnets.0.conv1.weight")
        if s is None or len(s) != 4 or int(s[0]) % base:
            return None
        mult.append(int(s[0]) // base)
    temporal = CONSTANTS["temperal_downsample"]
    if levels - 1 != len(temporal):
        return None
    # The decoder's last width is its base times the first multiplier.
    last = int(shapes["decoder.conv_out.weight"][1])
    if last % mult[0]:
        return None
    cfg = QwenImage21VAEConfig(
        in_channels=int(shapes["encoder.conv_in.weight"][1]),
        out_channels=int(shapes["decoder.conv_out.weight"][0]),
        latent_channels=int(shapes["decoder.conv_in.weight"][1]),
        base_dim=base,
        decoder_base_dim=last // mult[0],
        dim_mult=tuple(mult),
        num_res_blocks=per_block,
        temporal=temporal,
    )
    return cfg if _shortcuts_divide(cfg) else None


def _shortcuts_divide(cfg: QwenImage21VAEConfig) -> bool:
    """Whether every shortcut maps its channels in whole groups: C * f over
    the output channels on the way down, and the output channels times f
    over C on the way up."""
    n = len(cfg.dim_mult)
    d, e = cfg.encoder_dims, cfg.decoder_dims
    up = cfg.temporal[::-1]
    for i in range(n):
        f = (2 if cfg.temporal[i] else 1) * 4 if i < n - 1 else 1
        if d[i] * f % d[i + 1]:
            return False
        if i < n - 1 and e[i + 1] * (2 if up[i] else 1) * 4 % e[i]:
            return False
    return True


def relative(shapes: dict) -> str | None:
    """The name of a VAE with this module tree that is not this one, for a
    refusal, or None."""
    probe = shapes.get("encoder.conv_in.weight")
    if probe is None or "decoder.norm_out.gamma" not in shapes:
        return None
    if len(probe) == 5:
        z = shapes.get("decoder.conv_in.weight", (0, 0))[1]
        if "encoder.down_blocks.0.resnets.0.conv1.weight" in shapes:
            return f"a video VAE in the Wan 2.2 layout, {z} latent channels"
        return (f"a video VAE in the Wan 2.1 and Qwen-Image 20B layout, {z} "
                "latent channels")
    return None


def avg_down(x: mx.array, cout: int, factor_t: int, factor_s: int) -> mx.array:
    """The down shortcut. Space to depth by ``factor_s`` puts input channel
    c at channels c * f + t * s * s + dy * s + dx, with f = factor_t * s * s,
    and the mean over each run of C * f / cout channels is one output
    channel. With ``factor_t`` 2 the frame is t = 1 and t = 0 is zeros."""
    b, h, w, c = x.shape
    s = factor_s
    f = factor_t * s * s
    if f == 1 and c == cout:
        return x
    x = x.reshape(b, h // s, s, w // s, s, c).transpose(0, 1, 3, 5, 2, 4)
    x = x.reshape(b, h // s, w // s, c, s * s)
    if factor_t > 1:
        pad = mx.zeros((b, h // s, w // s, c, (factor_t - 1) * s * s),
                       dtype=x.dtype)
        x = mx.concatenate([pad, x], axis=-1)
    return x.reshape(b, h // s, w // s, cout, c * f // cout).mean(axis=-1)


def dup_up(x: mx.array, cout: int, factor_t: int, factor_s: int) -> mx.array:
    """The up shortcut. Each input channel repeats cout * f / C times, with
    f = factor_t * s * s, and depth to space reads output channel o at
    offset (dy, dx) of frame t from repeated channel
    o * f + t * s * s + dy * s + dx. The frame kept is t = factor_t - 1."""
    b, h, w, c = x.shape
    s = factor_s
    f = factor_t * s * s
    at = (np.arange(cout)[:, None] * f + (factor_t - 1) * s * s
          + np.arange(s * s)[None, :])
    src = mx.array((at // (cout * f // c)).reshape(-1).astype(np.int32))
    x = mx.take(x, src, axis=-1).reshape(b, h, w, cout, s, s)
    return x.transpose(0, 1, 4, 2, 5, 3).reshape(b, h * s, w * s, cout)


def upsample_nearest(x: mx.array) -> mx.array:
    """Nearest 2x: each pixel repeats over a 2x2 block."""
    b, h, w, c = x.shape
    x = mx.broadcast_to(x[:, :, None, :, None, :], (b, h, 2, w, 2, c))
    return x.reshape(b, 2 * h, 2 * w, c)


def rms_norm(x: mx.array, gamma: mx.array) -> mx.array:
    """x / |x| over the channels, in float32, times sqrt(C) * gamma. The
    quotient is rounded to the dtype of ``x`` before the gamma product."""
    return ops.rms_norm(x, gamma.reshape(-1), NORM_FLOOR ** 2 / x.shape[-1])


def clamp(x: mx.array) -> mx.array:
    """The decoded image in float32, clamped to [-1, 1]."""
    return mx.clip(x.astype(mx.float32), -1.0, 1.0)


class RMSNorm(nn.Module):
    """``gamma`` has shape (C,) followed by ``ones`` ones, as in the file."""

    def __init__(self, c: int, ones: int):
        super().__init__()
        self.gamma = mx.ones((c,) + (1,) * ones)

    def __call__(self, x: mx.array) -> mx.array:
        return rms_norm(x, self.gamma)


class ResidualBlock(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.norm1 = RMSNorm(cin, 3)
        self.conv1 = Conv2d(cin, cout, 3, padding=1)
        self.norm2 = RMSNorm(cout, 3)
        self.conv2 = Conv2d(cout, cout, 3, padding=1)
        self.conv_shortcut = Conv2d(cin, cout, 1) if cin != cout else None

    def __call__(self, x: mx.array) -> mx.array:
        h = self.conv1(nn.silu(self.norm1(x)))
        h = self.conv2(nn.silu(self.norm2(h)))
        if self.conv_shortcut is not None:
            x = self.conv_shortcut(x)
        return x + h


class ZeroPad(nn.Module):
    def __call__(self, x: mx.array) -> mx.array:
        return mx.pad(x, DOWN_PAD)


class Nearest2x(nn.Module):
    def __call__(self, x: mx.array) -> mx.array:
        return upsample_nearest(x)


class Resample(nn.Module):
    """``resample`` is a step without parameters and then a conv, so the
    conv is ``resample.1`` as in the file."""

    def __init__(self, c: int, down: bool):
        super().__init__()
        if down:
            self.resample = [ZeroPad(), Conv2d(c, c, 3, stride=2, padding=0)]
        else:
            self.resample = [Nearest2x(), Conv2d(c, c, 3, padding=1)]

    def __call__(self, x: mx.array) -> mx.array:
        return self.resample[1](self.resample[0](x))


class AttentionBlock(nn.Module):
    """One head over all pixels, with a norm in front and a residual."""

    def __init__(self, c: int):
        super().__init__()
        self.norm = RMSNorm(c, 2)
        self.to_qkv = Conv2d(c, 3 * c, 1)
        self.proj = Conv2d(c, c, 1)

    def __call__(self, x: mx.array) -> mx.array:
        b, h, w, c = x.shape
        qkv = self.to_qkv(self.norm(x)).reshape(b, 1, h * w, 3 * c)
        q, k, v = mx.split(qkv, 3, axis=-1)
        o = chunked_sdpa(q, k, v)
        return x + self.proj(o.reshape(b, h, w, c))


class MidBlock(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.resnets = [ResidualBlock(c, c), ResidualBlock(c, c)]
        self.attentions = [AttentionBlock(c)]

    def __call__(self, x: mx.array) -> mx.array:
        x = self.resnets[0](x)
        return self.resnets[1](self.attentions[0](x))


class ResidualDownBlock(nn.Module):
    def __init__(self, cin: int, cout: int, layers: int, temporal: bool,
                 down: bool):
        super().__init__()
        self.resnets = [ResidualBlock(cin if i == 0 else cout, cout)
                        for i in range(layers)]
        self.downsampler = Resample(cout, down=True) if down else None
        self._shortcut = (cout, 2 if temporal else 1, 2 if down else 1)

    def __call__(self, x: mx.array) -> mx.array:
        h = x
        for r in self.resnets:
            h = r(h)
        if self.downsampler is not None:
            h = self.downsampler(h)
        return h + avg_down(x, *self._shortcut)


class ResidualUpBlock(nn.Module):
    def __init__(self, cin: int, cout: int, layers: int, temporal: bool,
                 up: bool):
        super().__init__()
        self.resnets = [ResidualBlock(cin if i == 0 else cout, cout)
                        for i in range(layers)]
        self.upsampler = Resample(cout, down=False) if up else None
        self._shortcut = (cout, 2 if temporal else 1, 2)

    def __call__(self, x: mx.array) -> mx.array:
        h = x
        for r in self.resnets:
            h = r(h)
        if self.upsampler is None:
            return h
        return self.upsampler(h) + dup_up(x, *self._shortcut)


class Encoder(nn.Module):
    def __init__(self, cfg: QwenImage21VAEConfig):
        super().__init__()
        d, n = cfg.encoder_dims, len(cfg.dim_mult)
        self.conv_in = Conv2d(cfg.in_channels, d[0], 3, padding=1)
        self.down_blocks = [
            ResidualDownBlock(d[i], d[i + 1], cfg.num_res_blocks,
                              i < n - 1 and cfg.temporal[i], i < n - 1)
            for i in range(n)]
        self.mid_block = MidBlock(d[-1])
        self.norm_out = RMSNorm(d[-1], 3)
        self.conv_out = Conv2d(d[-1], 2 * cfg.latent_channels, 3, padding=1)

    def __call__(self, x: mx.array) -> mx.array:
        return self.tail(self.head(x))

    def head(self, x: mx.array) -> mx.array:
        """The pixel-space part: conv_in and the down blocks."""
        x = self.conv_in(x)
        for blk in self.down_blocks:
            x = blk(x)
        return x

    def tail(self, x: mx.array) -> mx.array:
        """The latent-space part: the mid block and the output conv."""
        x = self.mid_block(x)
        return self.conv_out(nn.silu(self.norm_out(x)))


class Decoder(nn.Module):
    def __init__(self, cfg: QwenImage21VAEConfig):
        super().__init__()
        d, n = cfg.decoder_dims, len(cfg.dim_mult)
        temporal = cfg.temporal[::-1]
        self.conv_in = Conv2d(cfg.latent_channels, d[0], 3, padding=1)
        self.mid_block = MidBlock(d[0])
        self.up_blocks = [
            ResidualUpBlock(d[i], d[i + 1], cfg.num_res_blocks + 1,
                            i < n - 1 and temporal[i], i < n - 1)
            for i in range(n)]
        self.norm_out = RMSNorm(d[-1], 3)
        self.conv_out = Conv2d(d[-1], cfg.out_channels, 3, padding=1)
        self.whole = whole_up_blocks(cfg)

    def __call__(self, z: mx.array) -> mx.array:
        return self.tail(self.head(z))

    def head(self, z: mx.array) -> mx.array:
        """The part a tiled decode runs on the whole image: conv_in, the
        mid block and the first up blocks. Its result has ``2 ** whole``
        pixels per latent pixel."""
        x = self.mid_block(self.conv_in(z))
        for blk in self.up_blocks[:self.whole]:
            x = blk(x)
            pacing.point(x)
        return x

    def tail(self, x: mx.array) -> mx.array:
        """The last up blocks and the output conv."""
        for blk in self.up_blocks[self.whole:]:
            x = blk(x)
            pacing.point(x)
        return self.conv_out(nn.silu(self.norm_out(x)))


class QwenImage21VAE(nn.Module):
    """``decode`` maps VAE-space latents (B, h, w, 64) to an RGBA image
    (B, 16h, 16w, 4) in float32, clamped to [-1, 1]. ``encode`` takes an
    RGB or RGBA image on the [-1, 1] scale and returns the mean and log
    variance of the latent distribution, the log variance clipped to
    [-30, 20]. Both run in the dtype of the weights. Model space, which the
    DiT works in, is (z - mean) / std per latent channel.

    No norm takes a statistic of the whole image, so a tiled pass runs
    the part that holds most of the bytes one tile at a time:
    ``decode_tail`` on tiles of ``decode_head``'s output, and
    ``encode_head`` on tiles of the image. A result farther from every cut
    than ``reach`` is that of the whole pass."""

    lockstep = False

    def __init__(self, cfg: QwenImage21VAEConfig):
        super().__init__()
        z = cfg.latent_channels
        if len(cfg.latents_mean) != z or len(cfg.latents_std) != z:
            raise ValueError(f"the latent statistics hold "
                             f"{len(cfg.latents_mean)} means and "
                             f"{len(cfg.latents_std)} deviations for {z} "
                             "latent channels")
        self.config = cfg
        self.encoder = Encoder(cfg)
        self.quant_conv = Conv2d(2 * z, 2 * z, 1)
        self.post_quant_conv = Conv2d(z, z, 1)
        self.decoder = Decoder(cfg)

    @property
    def dtype(self) -> mx.Dtype:
        return self.decoder.conv_in.weight.dtype

    def channels(self, op: str) -> int:
        """Channels at full resolution in ``op``, "decode" or "encode"."""
        cfg = self.config
        return cfg.decoder_base_dim if op == "decode" else cfg.base_dim

    def decode(self, z: mx.array) -> mx.array:
        return self.finish(self.decode_tail(self.decode_head(
            z.astype(self.dtype))))

    def decode_head(self, z: mx.array) -> mx.array:
        """The part a tiled decode runs on the whole image:
        post_quant_conv, conv_in, the mid block and the first up blocks."""
        return self.decoder.head(self.post_quant_conv(z))

    def decode_tail(self, x: mx.array) -> mx.array:
        """The last up blocks and the output conv, unclamped."""
        return self.decoder.tail(x)

    def finish(self, x: mx.array) -> mx.array:
        return clamp(x)

    def to_vae_space(self, latents: mx.array) -> mx.array:
        """Model-space (B, C, h, w) to VAE-space (B, h, w, C), in float32."""
        c = self.config.latent_channels
        if latents.ndim != 4 or latents.shape[1] != c:
            raise ValueError(f"latents of shape {tuple(latents.shape)} are "
                             f"not (B, {c}, h, w)")
        z = latents.astype(mx.float32).transpose(0, 2, 3, 1)
        return (z * mx.array(self.config.latents_std)
                + mx.array(self.config.latents_mean))

    def to_model_space(self, z: mx.array) -> mx.array:
        """VAE-space (B, h, w, C) to model-space (B, C, h, w), in float32."""
        z = z.astype(mx.float32) - mx.array(self.config.latents_mean)
        return (z / mx.array(self.config.latents_std)).transpose(0, 3, 1, 2)

    def to_vae_layout(self, latents: mx.array) -> mx.array:
        """An array in the layout of model-space latents (B, C, h, w) in
        the layout of VAE space (B, h, w, C), in float32, with no mean or
        deviation."""
        return channels_last(latents, self.config.latent_channels)

    def with_alpha(self, x: mx.array) -> mx.array:
        """``x`` with an alpha of ones when it is RGB."""
        c = self.config.in_channels
        if x.shape[-1] == c:
            return x
        if x.shape[-1] != c - 1:
            raise ValueError(f"the image has {x.shape[-1]} channels, and "
                             f"this VAE encodes {c - 1} or {c}")
        return mx.concatenate(
            [x, mx.ones((*x.shape[:-1], 1), dtype=x.dtype)], axis=-1)

    def encode(self, x: mx.array) -> tuple[mx.array, mx.array]:
        return self.split_moments(self.encode_tail(self.encode_head(x)))

    def encode_head(self, x: mx.array) -> mx.array:
        """The pixel-space part: conv_in and the down blocks."""
        return self.encoder.head(self.with_alpha(x).astype(self.dtype))

    def encode_tail(self, x: mx.array) -> mx.array:
        """The part at latent resolution: the mid block, the output conv
        and quant_conv."""
        return self.quant_conv(self.encoder.tail(x))

    def split_moments(self, m: mx.array) -> tuple[mx.array, mx.array]:
        """The moments as the mean and the clipped log variance."""
        c = self.config.latent_channels
        return m[..., :c], mx.clip(m[..., c:], -30.0, 20.0)


def reach(cfg: QwenImage21VAEConfig, op: str) -> int:
    """Latent pixels over which the tiled part of ``op``, "decode" or
    "encode", reads its input: a result at a latent pixel depends on no
    input farther than this from it. Each 3x3 conv reads one pixel at its
    own resolution. The tiled part of a decode is ``Decoder.tail``, the
    up blocks from ``whole_up_blocks`` on, and that of an encode is
    ``Encoder.head``."""
    return crop_reach(len(cfg.dim_mult), cfg.num_res_blocks,
                      whole_up_blocks(cfg), op)


def crop_reach(n: int, res: int, whole: int, op: str) -> int:
    """``reach`` of a VAE with ``n`` levels, ``res`` resnets per down block
    and one more per up block, whose tiled decode starts at up block
    ``whole``."""
    r = 0.0
    if op == "decode":
        for i in range(whole, n):
            r += 2 * (res + 1) / 2 ** i
            if i < n - 1:
                r += 1 / 2 ** (i + 1)
        r += 1 / 2 ** (n - 1)
    else:
        r += 1 / 2 ** (n - 1)
        for i in range(n):
            j = n - 1 - i
            r += 2 * res / 2 ** j
            if i < n - 1:
                r += 2 / 2 ** j
    return math.ceil(r)
