# SPDX-License-Identifier: Apache-2.0 AND MIT
# SPDX-FileCopyrightText: Copyright 2026 The HuggingFace Team
# Ported from diffusers' src/diffusers/models/autoencoders/vae.py (Encoder
# and Decoder), src/diffusers/models/resnet.py (ResnetBlock2D),
# src/diffusers/models/unet_2d_blocks.py (UNetMidBlock2D, DownEncoderBlock2D
# and UpDecoderBlock2D), the padded stride 2 conv of downsampling.py, the
# nearest 2x upsample of upsampling.py and the AttnProcessor2_0 path of
# attention_processor.py at e0abab83, and modified for gmlx (Apache-2.0;
# see licenses/diffusers-LICENSE).
"""The FLUX.1 autoencoder, which is the Z-Image VAE.

Module paths equal the diffusers tensor names, and activations are
channels last (B, H, W, C). Channels per level are [128, 256, 512, 512],
each down block has two resnets and each up block three, the mid block is
a resnet, one attention head of 512 and a resnet, and there are no quant
convs. GroupNorm is the owned contiguous-group wrapper. ``tree_config``
reads the decoder's widths from its shapes, for a decoder narrower than
the encoder.

``original_name`` maps the original FLUX.1 layout, as in ``ae.safetensors``,
to diffusers names, and the FLUX.2 one, which keeps its quant convs inside
the encoder and the decoder. Up blocks there count from the output side,
and the mid attention projections are 1x1 convs, which the reader reshapes
to Linear weights.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from gmlx.image import ops, pacing

from .conv import Conv2d
from .group_norm import GroupNorm

# Config values no tensor shape determines, from the official
# vae/config.json of Tongyi-MAI/Z-Image-Turbo.
CONSTANTS = {"norm_num_groups": 32, "eps": 1e-6}
CONSTANTS_SOURCE = ("Tongyi-MAI/Z-Image-Turbo",
                    "f332072aa78be7aecdf3ee76d5c247082da564a6")

# Peak bytes of one chunk of attention scores, counted at 4 bytes each. The
# mid attention runs over every latent pixel, so full scores are 8 GiB in
# bf16 at 2048x2048 wherever they materialize.
ATTN_SCORE_BYTES = 256 << 20


@dataclass(frozen=True)
class Flux1AEConfig:
    in_channels: int
    out_channels: int
    latent_channels: int
    block_out_channels: tuple
    layers_per_block: int
    norm_num_groups: int
    eps: float
    scaling_factor: float = 1.0     # model space is (z - shift) * scaling
    shift_factor: float = 0.0
    decoder_block_out_channels: tuple = ()  # () when the encoder's

    @property
    def decoder_channels(self) -> tuple:
        """Channels per level of the decoder, from the full resolution
        side."""
        return self.decoder_block_out_channels or self.block_out_channels

    @property
    def scale(self) -> int:
        """Pixels per latent pixel along each axis."""
        return 2 ** (len(self.block_out_channels) - 1)


def config_from_shapes(shapes: dict) -> Flux1AEConfig | None:
    """The config from diffusers-named torch shapes, or None when the names
    are not this autoencoder's. A tree with both quant convs and the
    BatchNorm statistics is the FLUX.2 autoencoder's; with some of them it
    is this one's with tensors the load does not expect. The decoder is as
    wide as the encoder."""
    if all(n in shapes for n in FLUX2_MARKS):
        return None
    cfg = tree_config(shapes)
    return None if cfg is None or cfg.decoder_block_out_channels else cfg


# Tensors the FLUX.1 autoencoder lacks and the FLUX.2 one has.
FLUX2_MARKS = ("quant_conv.weight", "post_quant_conv.weight",
               "bn.running_mean", "bn.running_var")


def tree_config(shapes: dict) -> Flux1AEConfig | None:
    """The config of the encoder and decoder tree in diffusers-named torch
    shapes, or None when the names are not that tree's."""
    need = ("encoder.conv_in.weight", "encoder.conv_out.weight",
            "decoder.conv_in.weight", "decoder.conv_out.weight",
            "decoder.mid_block.attentions.0.to_q.weight")
    if any(n not in shapes for n in need):
        return None
    levels = _count(shapes, r"encoder\.down_blocks\.(\d+)\.")
    per_block = _count(shapes, r"encoder\.down_blocks\.0\.resnets\.(\d+)\.")
    if levels < 1 or per_block < 1:
        return None
    chans = []
    for i in range(levels):
        s = shapes.get(f"encoder.down_blocks.{i}.resnets.0.conv1.weight")
        if s is None or len(s) != 4:
            return None
        chans.append(int(s[0]))
    # Up block i of diffusers counts from the latent side.
    if _count(shapes, r"decoder\.up_blocks\.(\d+)\.") != levels:
        return None
    dec = []
    for i in reversed(range(levels)):
        s = shapes.get(f"decoder.up_blocks.{i}.resnets.0.conv1.weight")
        if s is None or len(s) != 4:
            return None
        dec.append(int(s[0]))
    return Flux1AEConfig(
        in_channels=int(shapes["encoder.conv_in.weight"][1]),
        out_channels=int(shapes["decoder.conv_out.weight"][0]),
        latent_channels=int(shapes["decoder.conv_in.weight"][1]),
        block_out_channels=tuple(chans),
        layers_per_block=per_block,
        norm_num_groups=CONSTANTS["norm_num_groups"],
        eps=CONSTANTS["eps"],
        decoder_block_out_channels=() if dec == chans else tuple(dec),
    )


def _count(shapes: dict, pattern: str) -> int:
    rx = re.compile(pattern)
    found = {int(m.group(1)) for n in shapes if (m := rx.match(n))}
    return max(found) + 1 if found else 0


_ORIGINAL_PARTS = (
    (".block.", ".resnets."),
    (".nin_shortcut.", ".conv_shortcut."),
    (".downsample.", ".downsamplers.0."),
    (".upsample.", ".upsamplers.0."),
    (".mid.block_1.", ".mid_block.resnets.0."),
    (".mid.block_2.", ".mid_block.resnets.1."),
    (".mid.attn_1.norm.", ".mid_block.attentions.0.group_norm."),
    (".mid.attn_1.q.", ".mid_block.attentions.0.to_q."),
    (".mid.attn_1.k.", ".mid_block.attentions.0.to_k."),
    (".mid.attn_1.v.", ".mid_block.attentions.0.to_v."),
    (".mid.attn_1.proj_out.", ".mid_block.attentions.0.to_out.0."),
    (".norm_out.", ".conv_norm_out."),
)
# The FLUX.2 autoencoder's original layout keeps its quant convs inside the
# encoder and the decoder.
_ORIGINAL_TOP = (("encoder.quant_conv.", "quant_conv."),
                 ("decoder.post_quant_conv.", "post_quant_conv."))
_ORIGINAL_UP = re.compile(r"^decoder\.up\.(\d+)\.")
_ORIGINAL_DOWN = re.compile(r"^encoder\.down\.(\d+)\.")


def is_original_layout(names) -> bool:
    return any(n.startswith(("encoder.down.", "decoder.up.")) or
               ".mid.attn_1." in n for n in names)


def original_name(name: str, levels: int) -> str:
    """The diffusers name of an original-layout tensor. ``levels`` is the
    number of up blocks, since the original counts them from the output."""
    for old, new in _ORIGINAL_TOP:
        if name.startswith(old):
            return new + name[len(old):]
    name = _ORIGINAL_UP.sub(
        lambda m: f"decoder.up_blocks.{levels - 1 - int(m.group(1))}.", name)
    name = _ORIGINAL_DOWN.sub(r"encoder.down_blocks.\1.", name)
    for old, new in _ORIGINAL_PARTS:
        name = name.replace(old, new)
    return name


def original_levels(names) -> int:
    found = {int(m.group(1)) for n in names if (m := _ORIGINAL_UP.match(n))}
    return max(found) + 1 if found else 0


class ResnetBlock(nn.Module):
    def __init__(self, cin: int, cout: int, groups: int, eps: float):
        super().__init__()
        self.norm1 = GroupNorm(groups, cin, eps)
        self.conv1 = Conv2d(cin, cout, 3, padding=1)
        self.norm2 = GroupNorm(groups, cout, eps)
        self.conv2 = Conv2d(cout, cout, 3, padding=1)
        self.conv_shortcut = Conv2d(cin, cout, 1) if cin != cout else None

    def __call__(self, x: mx.array) -> mx.array:
        h = self.conv1(nn.silu(self.norm1(x)))
        h = self.conv2(nn.silu(self.norm2(h)))
        if self.conv_shortcut is not None:
            x = self.conv_shortcut(x)
        return x + h

    def steps(self, x: mx.array):
        h = yield from self.norm1.steps(x)
        h = self.conv1(nn.silu(h))
        h = yield from self.norm2.steps(h)
        h = self.conv2(nn.silu(h))
        if self.conv_shortcut is not None:
            x = self.conv_shortcut(x)
        return x + h


class Downsample(nn.Module):
    """Zero padding on the bottom and right, then a stride 2 conv."""

    def __init__(self, c: int):
        super().__init__()
        self.conv = Conv2d(c, c, 3, stride=2, padding=0)

    def __call__(self, x: mx.array) -> mx.array:
        return self.conv(mx.pad(x, [(0, 0), (0, 1), (0, 1), (0, 0)]))


class Upsample(nn.Module):
    """Nearest 2x, then a conv."""

    def __init__(self, c: int):
        super().__init__()
        self.conv = Conv2d(c, c, 3, padding=1)

    def __call__(self, x: mx.array) -> mx.array:
        b, h, w, c = x.shape
        x = mx.broadcast_to(x[:, :, None, :, None, :], (b, h, 2, w, 2, c))
        return self.conv(x.reshape(b, 2 * h, 2 * w, c))


def channels_last(latents: mx.array, channels: int) -> mx.array:
    """(B, ``channels``, h, w) as (B, h, w, ``channels``) in float32.
    Raises ValueError for another shape."""
    if latents.ndim != 4 or latents.shape[1] != channels:
        raise ValueError(f"an array of shape {tuple(latents.shape)} is not "
                         f"in the layout (B, {channels}, h, w)")
    return latents.astype(mx.float32).transpose(0, 2, 3, 1)


def chunked_sdpa(q: mx.array, k: mx.array, v: mx.array,
                 score_bytes: int = ATTN_SCORE_BYTES) -> mx.array:
    """``ops.sdpa`` over row chunks of q (B, H, L, D), so one chunk of
    scores stays under ``score_bytes``. Rows are independent, so chunking
    changes no value."""
    b, h, n, _ = q.shape
    rows = max(1, score_bytes // (b * h * k.shape[2] * 4))
    if rows >= n:
        return ops.sdpa(q, k, v)
    return mx.concatenate([ops.sdpa(q[:, :, i:i + rows], k, v)
                           for i in range(0, n, rows)], axis=2)


class MidAttention(nn.Module):
    """One head over all pixels, with GroupNorm in front and a residual."""

    def __init__(self, c: int, groups: int, eps: float):
        super().__init__()
        self.group_norm = GroupNorm(groups, c, eps)
        self.to_q = nn.Linear(c, c)
        self.to_k = nn.Linear(c, c)
        self.to_v = nn.Linear(c, c)
        self.to_out = [nn.Linear(c, c)]

    def __call__(self, x: mx.array) -> mx.array:
        b, h, w, c = x.shape
        y = self.group_norm(x).reshape(b, 1, h * w, c)
        o = chunked_sdpa(self.to_q(y), self.to_k(y), self.to_v(y))
        return x + self.to_out[0](o).reshape(b, h, w, c)


class MidBlock(nn.Module):
    def __init__(self, c: int, groups: int, eps: float):
        super().__init__()
        self.resnets = [ResnetBlock(c, c, groups, eps),
                        ResnetBlock(c, c, groups, eps)]
        self.attentions = [MidAttention(c, groups, eps)]

    def __call__(self, x: mx.array) -> mx.array:
        x = self.resnets[0](x)
        return self.resnets[1](self.attentions[0](x))


class DownBlock(nn.Module):
    def __init__(self, cin: int, cout: int, layers: int, down: bool,
                 groups: int, eps: float):
        super().__init__()
        self.resnets = [ResnetBlock(cin if i == 0 else cout, cout, groups, eps)
                        for i in range(layers)]
        self.downsamplers = [Downsample(cout)] if down else []

    def __call__(self, x: mx.array) -> mx.array:
        for r in self.resnets:
            x = r(x)
        for d in self.downsamplers:
            x = d(x)
        return x

    def steps(self, x: mx.array):
        for r in self.resnets:
            x = yield from r.steps(x)
        for d in self.downsamplers:
            x = d(x)
        return x


class UpBlock(nn.Module):
    def __init__(self, cin: int, cout: int, layers: int, up: bool,
                 groups: int, eps: float):
        super().__init__()
        self.resnets = [ResnetBlock(cin if i == 0 else cout, cout, groups, eps)
                        for i in range(layers)]
        self.upsamplers = [Upsample(cout)] if up else []

    def __call__(self, x: mx.array) -> mx.array:
        for r in self.resnets:
            x = r(x)
        for u in self.upsamplers:
            x = u(x)
        return x

    def steps(self, x: mx.array):
        for r in self.resnets:
            x = yield from r.steps(x)
        for u in self.upsamplers:
            x = u(x)
        return x


class Encoder(nn.Module):
    def __init__(self, cfg: Flux1AEConfig):
        super().__init__()
        ch, g, eps = cfg.block_out_channels, cfg.norm_num_groups, cfg.eps
        self.conv_in = Conv2d(cfg.in_channels, ch[0], 3, padding=1)
        self.down_blocks = [
            DownBlock(ch[max(i - 1, 0)], ch[i], cfg.layers_per_block,
                      i < len(ch) - 1, g, eps) for i in range(len(ch))]
        self.mid_block = MidBlock(ch[-1], g, eps)
        self.conv_norm_out = GroupNorm(g, ch[-1], eps)
        self.conv_out = Conv2d(ch[-1], 2 * cfg.latent_channels, 3,
                               padding=1)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.conv_in(x)
        for blk in self.down_blocks:
            x = blk(x)
        return self.tail(x)

    def head_steps(self, x: mx.array):
        """The pixel-space part, conv_in and the down blocks, as steps."""
        x = self.conv_in(x)
        for blk in self.down_blocks:
            x = yield from blk.steps(x)
        return x

    def tail(self, x: mx.array) -> mx.array:
        """The latent-space part: the mid block and the output conv."""
        x = self.mid_block(x)
        return self.conv_out(nn.silu(self.conv_norm_out(x)))


class Decoder(nn.Module):
    def __init__(self, cfg: Flux1AEConfig):
        super().__init__()
        ch = tuple(reversed(cfg.decoder_channels))
        g, eps = cfg.norm_num_groups, cfg.eps
        self.conv_in = Conv2d(cfg.latent_channels, ch[0], 3, padding=1)
        self.mid_block = MidBlock(ch[0], g, eps)
        self.up_blocks = [
            UpBlock(ch[max(i - 1, 0)], ch[i], cfg.layers_per_block + 1,
                    i < len(ch) - 1, g, eps) for i in range(len(ch))]
        self.conv_norm_out = GroupNorm(g, ch[-1], eps)
        self.conv_out = Conv2d(ch[-1], cfg.out_channels, 3, padding=1)

    def __call__(self, z: mx.array) -> mx.array:
        x = self.head(z)
        for blk in self.up_blocks:
            x = blk(x)
            pacing.point(x)
        return self.conv_out(nn.silu(self.conv_norm_out(x)))

    def head(self, z: mx.array) -> mx.array:
        """The latent-space part: conv_in and the mid block."""
        return self.mid_block(self.conv_in(z))

    def tail_steps(self, x: mx.array):
        """The up blocks and the output conv, as steps."""
        for blk in self.up_blocks:
            x = yield from blk.steps(x)
        h = yield from self.conv_norm_out.steps(x)
        return self.conv_out(nn.silu(h))


class Flux1AE(nn.Module):
    """``decode`` maps VAE-space latents (B, h, w, C) to an image
    (B, 8h, 8w, 3) on the [-1, 1] scale, unclamped. ``encode`` returns the
    mean and log variance of the latent distribution, the log variance
    clipped to [-30, 20]. Both run in the dtype of the weights. The family's
    scaling and shift relate VAE space to the model space the DiT works
    in.

    GroupNorm takes its statistics over the whole image, so the tiles of a
    tiled pass advance together (``lockstep``)."""

    lockstep = True

    def __init__(self, cfg: Flux1AEConfig):
        super().__init__()
        self.config = cfg
        self.encoder = Encoder(cfg)
        self.decoder = Decoder(cfg)

    @property
    def dtype(self) -> mx.Dtype:
        return self.decoder.conv_in.weight.dtype

    def channels(self, op: str) -> int:
        """Channels at full resolution in ``op``, "decode" or "encode"."""
        if op == "decode":
            return self.config.decoder_channels[0]
        return self.config.block_out_channels[0]

    def decode(self, z: mx.array) -> mx.array:
        return self.decoder(z.astype(self.dtype))

    def decode_head(self, z: mx.array) -> mx.array:
        """The part of the decode at latent resolution, which a tiled
        decode runs whole."""
        return self.decoder.head(z)

    def to_vae_space(self, latents: mx.array) -> mx.array:
        """Model-space (B, C, h, w) to VAE-space (B, h, w, C), in float32."""
        z = latents.astype(mx.float32).transpose(0, 2, 3, 1)
        return z / self.config.scaling_factor + self.config.shift_factor

    def to_model_space(self, z: mx.array) -> mx.array:
        """VAE-space (B, h, w, C) to model-space (B, C, h, w), in float32."""
        z = z.astype(mx.float32) - self.config.shift_factor
        return (z * self.config.scaling_factor).transpose(0, 3, 1, 2)

    def to_vae_layout(self, latents: mx.array) -> mx.array:
        """An array in the layout of model-space latents (B, C, h, w) in
        the layout of VAE space (B, h, w, C), in float32, with no scale
        or shift."""
        return channels_last(latents, self.config.latent_channels)

    def encode(self, x: mx.array) -> tuple[mx.array, mx.array]:
        return self.split_moments(self.encoder(x.astype(self.dtype)))

    def split_moments(self, m: mx.array) -> tuple[mx.array, mx.array]:
        """The encoder output as the mean and the clipped log variance."""
        c = self.config.latent_channels
        return m[..., :c], mx.clip(m[..., c:], -30.0, 20.0)
