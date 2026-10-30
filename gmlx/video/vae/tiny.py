"""TAEH3, the tiny MiniMax-H3 video decoder that previews a clip, from
madebyollin's ``taeh3.safetensors``.

A stack of 3x3 convs over each latent frame, in which each memory block
also reads its input of the frame before, grows 2x2 pixel patches and 4
frames per latent. Each run of 5 latents gives 20 frames whose first 3
are dropped, as the full decoder's runs do, and the last 12 frames of
the clip are dropped. It decodes model-space latents, the ones the DiT
gives.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from gmlx.image.vae.conv import Conv2d

from .h3_video import owned

LATENT = 24
WIDTHS = (256, 128, 64, 64)
TIME_GROW = (1, 2, 2)
PATCH = 2
RUN = 5
DROP = 3
ENCODER_WIDTH = 64
ENCODER_TIME = (2, 2, 1)


@dataclass(frozen=True)
class Config:
    latent_channels: int = LATENT


def config_from_shapes(shapes: dict):
    """The TAEH3 config, or None for another tree."""
    if (shapes.get("decoder.1.weight") != (WIDTHS[0], LATENT, 3, 3)
            or shapes.get("decoder.22.weight")
            != (3 * PATCH ** 2, WIDTHS[3], 3, 3)):
        return None
    return Config()


def expected_unused(cfg) -> dict:
    """The file's encoder, which previews do not run, by torch shape."""
    w = ENCODER_WIDTH
    out = {"encoder.0.weight": (w, 3 * PATCH ** 2, 3, 3),
           "encoder.0.bias": (w,)}
    i = 2
    for stride in ENCODER_TIME:
        out[f"encoder.{i}.conv.weight"] = (w, w * stride, 1, 1)
        out[f"encoder.{i + 1}.weight"] = (w, w, 3, 3)
        for j in range(i + 2, i + 5):
            for k, cin in ((0, 2 * w), (2, w), (4, w)):
                out[f"encoder.{j}.conv.{k}.weight"] = (w, cin, 3, 3)
                out[f"encoder.{j}.conv.{k}.bias"] = (w,)
        i += 5
    out[f"encoder.{i}.weight"] = (LATENT, w, 3, 3)
    out[f"encoder.{i}.bias"] = (LATENT,)
    return out


def _conv(cin: int, cout: int, bias: bool = True) -> Conv2d:
    return Conv2d(cin, cout, 3, padding=1, bias=bias)


class Clamp(nn.Module):
    def __call__(self, x: mx.array) -> mx.array:
        return mx.tanh(x / 3) * 3


class ReLU(nn.Module):
    def __call__(self, x: mx.array) -> mx.array:
        return nn.relu(x)


class Upsample(nn.Module):
    def __call__(self, x: mx.array) -> mx.array:
        return mx.repeat(mx.repeat(x, 2, axis=1), 2, axis=2)


class MemBlock(nn.Module):
    """Three convs over a frame and the same block's input of the frame
    before, with a skip."""

    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.conv = [_conv(2 * cin, cout), ReLU(), _conv(cout, cout), ReLU(),
                     _conv(cout, cout)]
        if cin != cout:
            self.skip = Conv2d(cin, cout, 1, bias=False)

    def __call__(self, x: mx.array, frames: int,
                 before: mx.array | None = None) -> mx.array:
        """``x`` is (B * frames, H, W, C), frames in order per clip, and
        ``before`` this block's input of the frame before the first, (B,
        H, W, C), or zeros when None."""
        b = x.shape[0] // frames
        clip = x.reshape(b, frames, *x.shape[1:])
        head = mx.zeros_like(clip[:, :1]) if before is None \
            else before[:, None]
        past = mx.concatenate([head, clip[:, :-1]], axis=1).reshape(x.shape)
        h = mx.concatenate([x, past], axis=-1)
        for layer in self.conv:
            h = layer(h)
        return nn.relu(h + (self.skip(x) if "skip" in self else x))


class TGrow(nn.Module):
    """A 1x1 conv to ``stride`` frames of each frame."""

    def __init__(self, channels: int, stride: int):
        super().__init__()
        self.stride = stride
        self.conv = Conv2d(channels, channels * stride, 1, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        n, h, w, c = x.shape
        y = self.conv(x).reshape(n, h, w, self.stride, c)
        return y.transpose(0, 3, 1, 2, 4).reshape(n * self.stride, h, w, c)


class TAEH3(nn.Module):
    def __init__(self, cfg: Config | None = None):
        super().__init__()
        f = WIDTHS
        layers: list = [Clamp(), _conv(LATENT, f[0]), ReLU()]
        for i in range(3):
            layers += [MemBlock(f[i], f[i]) for _ in range(3)]
            layers += [Upsample(), TGrow(f[i], TIME_GROW[i]),
                       _conv(f[i], f[i + 1], bias=False)]
        layers += [ReLU(), _conv(f[3], 3 * PATCH ** 2)]
        self.decoder = layers

    def _run(self, latents: mx.array, carry: list) -> mx.array:
        """Every frame (B, n, H, W, 3) in [0, 1] that a run of latents
        grows to. ``carry`` holds each MemBlock's input of the frame before
        the run, and takes the run's last."""
        b, _, t = latents.shape[:3]
        dt = self.decoder[1].weight.dtype
        x = latents.transpose(0, 2, 3, 4, 1).reshape(
            b * t, *latents.shape[3:], LATENT).astype(dt)
        frames, k = t, 0
        for layer in self.decoder:
            if isinstance(layer, MemBlock):
                y = layer(x, frames, carry[k])
                carry[k] = x.reshape(b, frames, *x.shape[1:])[:, -1]
                x, k = y, k + 1
            else:
                x = layer(x)
                if isinstance(layer, TGrow):
                    frames *= layer.stride
        n, h, w, _ = x.shape
        x = x.reshape(n, h, w, 3, PATCH, PATCH).transpose(0, 1, 4, 2, 5, 3)
        return mx.clip(x.reshape(b, frames, h * PATCH, w * PATCH, 3)
                       .astype(mx.float32), 0.0, 1.0)

    def decode_frames(self, latents: mx.array):
        """Yield float32 frames (B, n, H, W, 3) in [0, 1] of model-space
        latents (B, 24, T, H, W), a run of RUN latents at a time, in
        order."""
        t = latents.shape[2]
        carry: list = [None] * sum(isinstance(m, MemBlock)
                                   for m in self.decoder)
        grow = math.prod(TIME_GROW)
        if t == 1:
            out = self._run(latents, carry)[:, grow - 1:]
            mx.eval(out)
            yield out
            return
        # A short last run is padded with zero frames to a whole run, and
        # the last DROP latents' frames of the clip are dropped.
        keep = RUN * grow - (grow - 1)
        left = -(-t // RUN) * keep - DROP * grow
        for i in range(0, t, RUN):
            out = self._run(latents[:, :, i:i + RUN], carry)[:, grow - 1:]
            out = mx.pad(out, [(0, 0), (0, keep - out.shape[1]), (0, 0),
                               (0, 0), (0, 0)])[:, :left]
            left -= out.shape[1]
            carry = [owned(c) for c in carry]
            mx.eval(out)
            yield out

    def decode(self, latents: mx.array) -> mx.array:
        """``decode_frames`` concatenated, (B, T, H, W, 3)."""
        return mx.concatenate(list(self.decode_frames(latents)), axis=1)
