# SPDX-License-Identifier: Apache-2.0 AND MIT
# SPDX-FileCopyrightText: Copyright 2026 The HuggingFace Team
# SPDX-FileCopyrightText: Copyright 2026 Black Forest Labs, Inc.
# Ported from diffusers' src/diffusers/models/autoencoders/
# autoencoder_kl_flux2.py (the quant convs and the BatchNorm statistics) at
# e0abab83, and from BFL's FLUX.3 klein reference at commit 45e57c9,
# src/flux3/autoencoder.py (the 2x2 patchify and the BatchNorm of the
# latents in AutoEncoder.encode and AutoEncoder.decode), and modified for
# gmlx (Apache-2.0; see licenses/diffusers-LICENSE and
# licenses/flux-3-image-klein-LICENSE).
"""The FLUX.2 autoencoder, which is the FLUX.3 klein VAE.

The encoder and decoder are the FLUX.1 autoencoder's at 32 latent
channels. A 1x1 quant conv follows the encoder's output conv, and a 1x1
post quant conv precedes the decoder's input conv. VAE space is what the
decoder takes, 32 channels at 8x. Model space patchifies it 2x2 into 128
channels at 16x, channel ``c * 4 + dy * 2 + dx``, and normalizes each
channel with the BatchNorm running statistics in eval mode,
``(z - mean) / sqrt(var + eps)``. Both maps run in float32, and the
statistics stay float32 whatever the decode dtype.

The decoder may run narrower than the encoder, as the MOSS-Image one
does: ``tree_config`` reads its widths from the decoder's shapes, and
``config_from_shapes`` takes only a decoder as wide as the encoder.

Module paths equal the diffusers tensor names. The original layout, as in
BFL's ``ae.safetensors``, keeps the quant convs inside the encoder and the
decoder, and ``flux1_ae.original_name`` maps it.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from . import flux1_ae
from .conv import Conv2d

# Config values no tensor shape determines, from the official
# vae/config.json of black-forest-labs/flux-3-image-klein. The GroupNorm eps
# is the reference's (autoencoder.py) and diffusers', which the config does
# not carry.
CONSTANTS = {"norm_num_groups": 32, "eps": 1e-6, "batch_norm_eps": 1e-4,
             "patch_size": (2, 2), "use_quant_conv": True,
             "use_post_quant_conv": True, "mid_block_add_attention": True,
             "act_fn": "silu"}
CONSTANTS_SOURCE = ("black-forest-labs/flux-3-image-klein",
                    "8454017f9ebaada30c998fccd9f67a80c7580f68")
BN_STATS = ("bn.running_mean", "bn.running_var")


@dataclass(frozen=True)
class Flux2AEConfig(flux1_ae.Flux1AEConfig):
    patch: int = 2                  # model space patchifies patch x patch
    bn_eps: float = 1e-4

    @property
    def model_channels(self) -> int:
        return self.patch * self.patch * self.latent_channels


def tree_config(shapes: dict) -> Flux2AEConfig | None:
    """The config of the FLUX.1 tree with both quant convs and the
    BatchNorm statistics, at the decoder widths its shapes give, or None
    when the names are not that tree's."""
    if any(n not in shapes for n in flux1_ae.FLUX2_MARKS):
        return None
    tree = flux1_ae.tree_config(shapes)
    if tree is None:
        return None
    patch = CONSTANTS["patch_size"]
    return Flux2AEConfig(
        **{k: getattr(tree, k) for k in (
            "in_channels", "out_channels", "latent_channels",
            "block_out_channels", "layers_per_block",
            "decoder_block_out_channels")},
        norm_num_groups=CONSTANTS["norm_num_groups"], eps=CONSTANTS["eps"],
        patch=patch[0], bn_eps=CONSTANTS["batch_norm_eps"])


def config_from_shapes(shapes: dict) -> Flux2AEConfig | None:
    """The config from diffusers-named torch shapes, or None when the names
    are not this autoencoder's: the tree of ``tree_config`` with a decoder
    as wide as its encoder."""
    cfg = tree_config(shapes)
    return None if cfg is None or cfg.decoder_block_out_channels else cfg


def expected_unused(cfg: Flux2AEConfig) -> dict:
    """The tensor a file may hold that stays unread: the batch count of the
    BatchNorm, an int64 scalar."""
    return {"bn.num_batches_tracked": ()}


def config_disagreements(config: dict) -> list:
    """The fields of a vae/config.json that differ from ``CONSTANTS``, and
    decoder widths other than the encoder's. A field the config leaves
    out is none, and so are null decoder widths."""
    out = constant_disagreements(config, CONSTANTS)
    dec = config.get("decoder_block_out_channels")
    if dec is not None and dec != config.get("block_out_channels", dec):
        out.append("decoder_block_out_channels")
    return out


def constant_disagreements(config: dict, constants: dict) -> list:
    """The fields of ``config`` that differ from ``constants``."""
    out = []
    for k, v in constants.items():
        if k not in config:
            continue
        got = config[k]
        if isinstance(v, tuple):
            same = isinstance(got, list) and got == list(v)
        elif isinstance(v, bool):
            same = got is v
        else:
            same = got == v
        if not same:
            out.append(k)
    return out


class BatchStats(nn.Module):
    """The running statistics of the latent BatchNorm, which has no affine
    weights."""

    def __init__(self, channels: int):
        super().__init__()
        self.running_mean = mx.zeros((channels,))
        self.running_var = mx.ones((channels,))


class Flux2AE(flux1_ae.Flux1AE):
    """``decode`` maps VAE-space latents (B, h, w, 32) to an image
    (B, 8h, 8w, 3) on the [-1, 1] scale, unclamped, through the post quant
    conv. ``encode`` returns the mean and the clipped log variance of the
    quant conv's output. ``to_model_space`` and ``to_vae_space`` are the
    patchify and the BatchNorm and their inverse."""

    def __init__(self, cfg: Flux2AEConfig):
        super().__init__(cfg)
        c = cfg.latent_channels
        self.quant_conv = Conv2d(2 * c, 2 * c, 1)
        self.post_quant_conv = Conv2d(c, c, 1)
        self.bn = BatchStats(cfg.model_channels)

    def decode(self, z: mx.array) -> mx.array:
        return self.decoder(self.post_quant_conv(z.astype(self.dtype)))

    def decode_head(self, z: mx.array) -> mx.array:
        return self.decoder.head(self.post_quant_conv(z))

    def split_moments(self, m: mx.array) -> tuple[mx.array, mx.array]:
        return super().split_moments(self.quant_conv(m))

    def stats(self, n: int) -> tuple[mx.array, mx.array]:
        """The BatchNorm mean and standard deviation, (1, n, 1, 1) in
        float32."""
        bn = self.bn
        mean = bn.running_mean.astype(mx.float32).reshape(1, n, 1, 1)
        var = bn.running_var.astype(mx.float32).reshape(1, n, 1, 1)
        return mean, mx.sqrt(var + self.config.bn_eps)

    def to_vae_space(self, latents: mx.array) -> mx.array:
        """Model-space (B, 4C, h, w) to VAE-space (B, 2h, 2w, C), in
        float32."""
        p, c = self.config.patch, self.config.latent_channels
        z = latents.astype(mx.float32)
        n = z.shape[1]
        if n != p * p * c:
            raise ValueError(f"model-space latents have {n} channels, and "
                             f"this VAE takes {p * p * c}")
        mean, std = self.stats(n)
        return unpatchify(z * std + mean, p)

    def to_model_space(self, z: mx.array) -> mx.array:
        """VAE-space (B, H, W, C) to model-space (B, 4C, H/2, W/2), in
        float32. H and W must be even."""
        p = self.config.patch
        z = z.astype(mx.float32)
        hh, ww = z.shape[1:3]
        if hh % p or ww % p:
            raise ValueError(f"VAE-space latents of {hh}x{ww} do not patchify "
                             f"{p}x{p}; the image must be a multiple of "
                             f"{p * self.config.scale} pixels")
        z = patchify(z, p)
        mean, std = self.stats(z.shape[1])
        return (z - mean) / std

    def to_vae_layout(self, latents: mx.array) -> mx.array:
        """An array in the layout of model-space latents (B, 4C, h, w) in
        the layout of VAE space (B, 2h, 2w, C), in float32: the
        unpatchify alone, with no BatchNorm."""
        p, c = self.config.patch, self.config.latent_channels
        if latents.ndim != 4 or latents.shape[1] != p * p * c:
            raise ValueError(f"an array of shape {tuple(latents.shape)} is "
                             f"not in the layout (B, {p * p * c}, h, w)")
        return unpatchify(latents.astype(mx.float32), p)


def unpatchify(z: mx.array, p: int) -> mx.array:
    """(B, p * p * C, h, w) to (B, p * h, p * w, C). Channel
    ``c * p * p + dy * p + dx`` goes to row ``y * p + dy`` and column
    ``x * p + dx``."""
    b, n, h, w = z.shape
    c = n // (p * p)
    z = z.reshape(b, c, p, p, h, w).transpose(0, 4, 2, 5, 3, 1)
    return z.reshape(b, h * p, w * p, c)


def patchify(z: mx.array, p: int) -> mx.array:
    """(B, H, W, C) to (B, p * p * C, H / p, W / p), the inverse of
    ``unpatchify``."""
    b, hh, ww, c = z.shape
    h, w = hh // p, ww // p
    z = z.reshape(b, h, p, w, p, c).transpose(0, 5, 2, 4, 1, 3)
    return z.reshape(b, c * p * p, h, w)
