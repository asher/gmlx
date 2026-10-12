# SPDX-License-Identifier: Apache-2.0 AND MIT
# SPDX-FileCopyrightText: Copyright 2026 The Qwen Team and The HuggingFace Team
# Ported from diffusers'
# src/diffusers/models/autoencoders/autoencoder_kl_qwenimage.py at commit
# e0abab83, for one frame, and modified for gmlx (Apache-2.0; see
# licenses/diffusers-LICENSE).
"""The Qwen-Image autoencoder, which is the Wan 2.1 video VAE, and the VAE
of Krea 2.

Module paths equal the diffusers tensor names, and activations are
channels last (B, H, W, C). The image is RGB, and a latent pixel is 8
image pixels with 16 channels. The encoder has widths 96 times
[1, 1, 2, 4, 4] and two resnets per level, and its blocks are one flat
list of resnets and downsamplers. The decoder has widths 96 times
[4, 4, 4, 2, 1] and three resnets per up block, and each upsampler halves
the channels it doubles the pixels of. Each mid block is a resnet, one
attention head and a resnet.

The official class is the image case of a video autoencoder, and this
module computes what it does for its one frame.

- A causal 3D conv pads two zero frames before the frame and none after,
  so one frame reads the last time slice of its kernel alone. The reader
  keeps that slice of a kernel of rank 5.
- ``time_conv`` never runs for one frame, so the file's eight
  ``time_conv`` tensors are left unread, and a file may leave them out.
- A norm divides by the L2 norm over the channels, computed in float32,
  and multiplies by sqrt(C) and ``gamma``, as in the Qwen-Image 2.1
  autoencoder, whose resnet, mid block and norm this module takes.
- Downsampling pads zeros on the bottom and right and then runs a stride 2
  conv. Upsampling is nearest 2x and then a conv to half the channels.
- ``decode`` clamps to [-1, 1]. The mean is the first half of the
  encoder's channels.

``original_name`` maps the original Wan 2.1 layout, as in ComfyUI's
``qwen_image_vae.safetensors``, to diffusers names.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from gmlx.image import pacing

from .conv import Conv2d
from .qwen_image21_vae import (MidBlock, Nearest2x, ResidualBlock, RMSNorm,
                               ZeroPad, clamp, crop_reach)

# What no tensor the model reads determines: the downsamples at which the
# video layout halves time too, and the latent statistics, the defaults of
# diffusers' AutoencoderKLQwenImage and the values of the vae/config.json
# of Qwen/Qwen-Image.
CONSTANTS = {
    "temperal_downsample": (False, True, True),
    "latents_mean": (
        -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
        0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921),
    "latents_std": (
        2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
        3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.916),
}
CONSTANTS_SOURCE = ("huggingface/diffusers",
                    "e0abab83b5df05de9e7abd788643c1a7c1e42e28")

LATENT_CHANNELS = 16
IMAGE_CHANNELS = 3

# Up blocks of the decoder that a tiled decode runs in tiles: all but the
# first. The convs of the whole up path read 13 latent pixels across a cut,
# and from the second up block on 6. The first up block runs at twice the
# latent resolution, so the part run whole holds 96 values a latent pixel.
TILED_UP_BLOCKS = 3


@dataclass(frozen=True)
class QwenImageVAEConfig:
    in_channels: int
    out_channels: int
    latent_channels: int
    base_dim: int
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
        return tuple(self.base_dim * m
                     for m in (self.dim_mult[-1], *reversed(self.dim_mult)))

    def up_in(self, i: int) -> int:
        """Input channels of up block ``i``: the channels of the upsampler
        before it, which halves its block's width."""
        d = self.decoder_dims
        return d[i] if i == 0 else d[i] // 2


def whole_up_blocks(cfg: QwenImageVAEConfig) -> int:
    """Up blocks of the decoder that a tiled decode runs on the whole
    image, before the ``TILED_UP_BLOCKS`` it runs in tiles."""
    return max(0, len(cfg.dim_mult) - TILED_UP_BLOCKS)


def encoder_layout(cfg: QwenImageVAEConfig) -> list:
    """The flat encoder list: per entry ("res", cin, cout) or ("down", c,
    temporal)."""
    d, n = cfg.encoder_dims, len(cfg.dim_mult)
    out = []
    for i in range(n):
        for j in range(cfg.num_res_blocks):
            out.append(("res", d[i] if j == 0 else d[i + 1], d[i + 1]))
        if i < n - 1:
            out.append(("down", d[i + 1], cfg.temporal[i]))
    return out


def expected_unused(cfg: QwenImageVAEConfig) -> dict:
    """The time_conv tensors of the video layout, which a file may hold and
    one frame never runs, each with its torch shape: a weight and a bias
    at each resample where the config halves or doubles time. A downsample
    keeps its channels, and an upsample doubles them for the two frames it
    makes."""
    out = {}
    for k, row in enumerate(encoder_layout(cfg)):
        if row[0] == "down" and row[2]:
            c = row[1]
            out[f"encoder.down_blocks.{k}.time_conv.weight"] = (c, c, 3, 1, 1)
            out[f"encoder.down_blocks.{k}.time_conv.bias"] = (c,)
    up = cfg.temporal[::-1]
    for i in range(len(cfg.dim_mult) - 1):
        if up[i]:
            c = cfg.decoder_dims[i + 1]
            name = f"decoder.up_blocks.{i}.upsamplers.0.time_conv"
            out[f"{name}.weight"] = (2 * c, c, 3, 1, 1)
            out[f"{name}.bias"] = (2 * c,)
    return out


def config_disagreements(config: dict) -> list:
    """The fields of a vae/config.json that differ from ``CONSTANTS``. A
    field the config leaves out is none."""
    return [k for k, v in CONSTANTS.items()
            if k in config and (not isinstance(config[k], list)
                                or config[k] != list(v))]


# --- the original Wan 2.1 layout ---------------------------------------

_RESIDUAL = ((".residual.0.", ".norm1."), (".residual.2.", ".conv1."),
             (".residual.3.", ".norm2."), (".residual.6.", ".conv2."),
             (".shortcut.", ".conv_shortcut."))
_MIDDLE = ((".middle.0.", ".mid_block.resnets.0."),
           (".middle.1.", ".mid_block.attentions.0."),
           (".middle.2.", ".mid_block.resnets.1."),
           (".head.0.", ".norm_out."), (".head.2.", ".conv_out."))
_TOP = {"conv1.": "quant_conv.", "conv2.": "post_quant_conv.",
        "encoder.conv1.": "encoder.conv_in.",
        "decoder.conv1.": "decoder.conv_in."}
_UP = re.compile(r"^decoder\.upsamples\.(\d+)\.")


def is_original_layout(names) -> bool:
    return any(n.startswith(("encoder.downsamples.", "decoder.upsamples.",
                             "decoder.middle.")) for n in names)


def original_levels(names) -> tuple:
    """The flat indices of the decoder's upsamplers in the original
    layout, which end its up blocks."""
    out = set()
    for n in names:
        m = _UP.match(n)
        if m and (".resample." in n or ".time_conv." in n):
            out.add(int(m.group(1)))
    return tuple(sorted(out))


def original_name(name: str, ups: tuple) -> str:
    """The diffusers name of an original-layout tensor. ``ups`` are the
    flat indices of the decoder's upsamplers, ``original_levels``."""
    for old, new in _TOP.items():
        if name.startswith(old):
            return new + name[len(old):]
    m = _UP.match(name)
    if m:
        k = int(m.group(1))
        level = sum(u < k for u in ups)
        rest = name[m.end():]
        if k in ups:
            name = f"decoder.up_blocks.{level}.upsamplers.0.{rest}"
        else:
            start = ups[level - 1] + 1 if level else 0
            name = f"decoder.up_blocks.{level}.resnets.{k - start}.{rest}"
    name = name.replace("encoder.downsamples.", "encoder.down_blocks.")
    for old, new in _RESIDUAL + _MIDDLE:
        name = name.replace(old, new)
    return name


def diffusers_names(names: dict) -> dict:
    """``names`` (stripped name -> file name) under diffusers names, when
    they are in the original layout."""
    if not is_original_layout(names):
        return names
    ups = original_levels(names)
    return {original_name(n, ups): f for n, f in names.items()}


# --- identification ------------------------------------------------------

def _count(shapes: dict, pattern: str) -> int:
    rx = re.compile(pattern)
    found = {int(m.group(1)) for n in shapes if (m := rx.match(n))}
    return max(found) + 1 if found else 0


# The causal 3D convs, whose kernels are (O, I, T, kH, kW) in the video
# layout with T = kH, or (O, I, kH, kW) in a file that kept the last time
# slice.
_CAUSAL = ("encoder.conv_in.weight", "encoder.conv_out.weight",
           "quant_conv.weight", "post_quant_conv.weight",
           "decoder.conv_in.weight", "decoder.conv_out.weight")


def _causal_ok(shapes: dict) -> bool:
    for n, s in shapes.items():
        if len(s) == 5 and not n.endswith(".time_conv.weight") and (
                s[2] != s[3] or s[3] != s[4]):
            return False
    return True


def config_from_shapes(shapes: dict) -> QwenImageVAEConfig | None:
    """The config from diffusers-named torch shapes, or None when the names
    are not this autoencoder's. The time flags are the official ones,
    whichever time_conv tensors the file holds."""
    need = (*_CAUSAL, "encoder.norm_out.gamma", "decoder.norm_out.gamma",
            "encoder.down_blocks.0.norm1.gamma",
            "decoder.up_blocks.0.resnets.0.norm1.gamma")
    if any(n not in shapes for n in need):
        return None
    if any(len(shapes[n]) not in (4, 5) for n in _CAUSAL):
        return None
    if not _causal_ok(shapes):
        return None
    levels = _count(shapes, r"decoder\.up_blocks\.(\d+)\.")
    per_block = _count(shapes, r"decoder\.up_blocks\.0\.resnets\.(\d+)\.") - 1
    temporal = CONSTANTS["temperal_downsample"]
    if levels - 1 != len(temporal) or per_block < 1:
        return None
    base = int(shapes["encoder.conv_in.weight"][0])
    mult = []
    for i in range(levels):
        k = i * (per_block + 1)
        s = shapes.get(f"encoder.down_blocks.{k}.conv1.weight")
        if s is None or int(s[0]) % base:
            return None
        mult.append(int(s[0]) // base)
    cfg = QwenImageVAEConfig(
        in_channels=int(shapes["encoder.conv_in.weight"][1]),
        out_channels=int(shapes["decoder.conv_out.weight"][0]),
        latent_channels=int(shapes["decoder.conv_in.weight"][1]),
        base_dim=base, dim_mult=tuple(mult), num_res_blocks=per_block,
        temporal=temporal)
    if _count(shapes, r"encoder\.down_blocks\.(\d+)\.") != len(
            encoder_layout(cfg)):
        return None
    if int(shapes["decoder.conv_out.weight"][1]) != cfg.decoder_dims[-1]:
        return None
    return cfg


# --- modules ---------------------------------------------------------------

class Resample(nn.Module):
    """``resample`` is a step without parameters and then a conv, so the
    conv is ``resample.1`` as in the file. An upsample's conv halves the
    channels."""

    def __init__(self, c: int, down: bool):
        super().__init__()
        if down:
            self.resample = [ZeroPad(), Conv2d(c, c, 3, stride=2, padding=0)]
        else:
            self.resample = [Nearest2x(), Conv2d(c, c // 2, 3, padding=1)]

    def __call__(self, x: mx.array) -> mx.array:
        return self.resample[1](self.resample[0](x))


class UpBlock(nn.Module):
    def __init__(self, cin: int, cout: int, layers: int, up: bool):
        super().__init__()
        self.resnets = [ResidualBlock(cin if i == 0 else cout, cout)
                        for i in range(layers)]
        self.upsamplers = [Resample(cout, down=False)] if up else None

    def __call__(self, x: mx.array) -> mx.array:
        for r in self.resnets:
            x = r(x)
        if self.upsamplers is not None:
            x = self.upsamplers[0](x)
        return x


class Encoder(nn.Module):
    def __init__(self, cfg: QwenImageVAEConfig):
        super().__init__()
        d = cfg.encoder_dims
        self.conv_in = Conv2d(cfg.in_channels, d[0], 3, padding=1)
        self.down_blocks = [
            ResidualBlock(row[1], row[2]) if row[0] == "res"
            else Resample(row[1], down=True) for row in encoder_layout(cfg)]
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
    def __init__(self, cfg: QwenImageVAEConfig):
        super().__init__()
        d, n = cfg.decoder_dims, len(cfg.dim_mult)
        self.conv_in = Conv2d(cfg.latent_channels, d[0], 3, padding=1)
        self.mid_block = MidBlock(d[0])
        self.up_blocks = [UpBlock(cfg.up_in(i), d[i + 1],
                                  cfg.num_res_blocks + 1, i < n - 1)
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


class QwenImageVAE(nn.Module):
    """``decode`` maps VAE-space latents (B, h, w, 16) to an RGB image
    (B, 8h, 8w, 3) in float32, clamped to [-1, 1]. ``encode`` takes an RGB
    image on the [-1, 1] scale and returns the mean and log variance of
    the latent distribution, the log variance clipped to [-30, 20]. Both
    run in the dtype of the weights. Model space, which the DiT works in,
    is (z - mean) / std per latent channel.

    No norm takes a statistic of the whole image, so a tiled pass runs
    the part that holds most of the bytes one tile at a time, as the
    Qwen-Image 2.1 autoencoder does, and a result farther from every cut
    than ``reach`` is that of the whole pass."""

    lockstep = False

    def __init__(self, cfg: QwenImageVAEConfig):
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
        return self.config.base_dim

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

    def encode(self, x: mx.array) -> tuple[mx.array, mx.array]:
        return self.split_moments(self.encode_tail(self.encode_head(x)))

    def encode_head(self, x: mx.array) -> mx.array:
        """The pixel-space part: conv_in and the down blocks."""
        c = self.config.in_channels
        if x.shape[-1] != c:
            raise ValueError(f"the image has {x.shape[-1]} channels, and "
                             f"this VAE encodes {c}")
        return self.encoder.head(x.astype(self.dtype))

    def encode_tail(self, x: mx.array) -> mx.array:
        """The part at latent resolution: the mid block, the output conv
        and quant_conv."""
        return self.quant_conv(self.encoder.tail(x))

    def split_moments(self, m: mx.array) -> tuple[mx.array, mx.array]:
        """The moments as the mean and the clipped log variance."""
        c = self.config.latent_channels
        return m[..., :c], mx.clip(m[..., c:], -30.0, 20.0)


def reach(cfg: QwenImageVAEConfig, op: str) -> int:
    """Latent pixels over which the tiled part of ``op``, "decode" or
    "encode", reads its input, as ``qwen_image21_vae.reach`` counts it:
    the up blocks from ``whole_up_blocks`` on and the output conv, or the
    input conv and the down blocks."""
    return crop_reach(len(cfg.dim_mult), cfg.num_res_blocks,
                      whole_up_blocks(cfg), op)
