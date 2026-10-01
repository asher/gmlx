# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Asher Feldman
"""VAE decode and encode, whole or in tiles.

Tiles are laid out in latent space. The default tile is 512 pixels, which
is 64 latent pixels at 8x, and neighbors overlap by at least 64 pixels.
Tiles are spread evenly so the last one ends at the edge, and no tile is
cut short. Each tile is weighted by a cosine ramp across every overlap
with a neighbor, and the weighted sum is divided by the summed weights.
Accumulation is on the host in float32, so the peak is one tile's decode.

Tiles see different GroupNorm statistics, so a tiled decode is close to
the whole decode, not equal to it.
"""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np

TILE = 512
OVERLAP = 64


def tile_starts(size: int, tile: int, overlap: int) -> list[int]:
    """Start offsets of tiles of length ``min(tile, size)`` that cover
    ``[0, size)``, with the last one ending at ``size`` and neighbors
    overlapping by at least ``overlap``."""
    if overlap >= tile:
        raise ValueError(f"overlap {overlap} must be below the tile {tile}")
    if size <= tile:
        return [0]
    n = math.ceil((size - overlap) / (tile - overlap))
    return [round(i * (size - tile) / (n - 1)) for i in range(n)]


def _ramp(n: int) -> np.ndarray:
    j = np.arange(n, dtype=np.float64)
    return (0.5 - 0.5 * np.cos(np.pi * (j + 0.5) / n)).astype(np.float32)


def axis_weights(starts: list[int], i: int, tile: int,
                 scale: int) -> np.ndarray:
    """Output weights along one axis for tile ``i``, in output pixels."""
    w = np.ones(tile * scale, np.float32)
    s = starts[i]
    if i > 0:
        n = (starts[i - 1] + tile - s) * scale
        w[:n] *= _ramp(n)
    if i + 1 < len(starts):
        n = (s + tile - starts[i + 1]) * scale
        w[tile * scale - n:] *= _ramp(n)[::-1]
    return w


def run_tiled(fn, x: mx.array, *, in_scale: int, out_scale: int,
              tile: int, overlap: int) -> np.ndarray:
    """``fn`` over tiles of channels-last ``x``, blended, as float32 numpy.

    ``tile`` and ``overlap`` are in latent pixels. A latent pixel is
    ``in_scale`` input pixels and ``out_scale`` output pixels.
    """
    b, h, w = x.shape[0], x.shape[1] // in_scale, x.shape[2] // in_scale
    th, tw = min(tile, h), min(tile, w)
    ys, xs = tile_starts(h, tile, overlap), tile_starts(w, tile, overlap)
    out = wsum = None
    for i, y0 in enumerate(ys):
        wy = axis_weights(ys, i, th, out_scale)
        for j, x0 in enumerate(xs):
            wx = axis_weights(xs, j, tw, out_scale)
            piece = fn(x[:, y0 * in_scale:(y0 + th) * in_scale,
                         x0 * in_scale:(x0 + tw) * in_scale])
            piece = np.array(piece.astype(mx.float32))
            if out is None:
                out = np.zeros((b, h * out_scale, w * out_scale,
                                piece.shape[-1]), np.float32)
                wsum = np.zeros((h * out_scale, w * out_scale), np.float32)
            wt = wy[:, None] * wx[None, :]
            ry = slice(y0 * out_scale, (y0 + th) * out_scale)
            rx = slice(x0 * out_scale, (x0 + tw) * out_scale)
            out[:, ry, rx] += piece * wt[None, :, :, None]
            wsum[ry, rx] += wt
    return out / wsum[None, :, :, None]


def _fits(h: int, w: int, tile: int) -> bool:
    return h <= tile and w <= tile


def decode(vae, latents, *, tiled: bool = False, tile: int = TILE,
           overlap: int = OVERLAP) -> mx.array:
    """Model-space latents (B, C, h, w) to an image (B, H, W, C) in float32
    on the [-1, 1] scale, unclamped. ``tile`` and ``overlap`` are in output
    pixels."""
    z = vae.to_vae_space(mx.array(latents))
    s = vae.config.scale
    if not tiled or _fits(z.shape[1], z.shape[2], tile // s):
        return vae.decode(z).astype(mx.float32)
    return mx.array(run_tiled(vae.decode, z, in_scale=1, out_scale=s,
                              tile=tile // s, overlap=overlap // s))


def encode(vae, image, *, tiled: bool = False, mode: str = "mode",
           noise=None, tile: int = TILE, overlap: int = OVERLAP) -> mx.array:
    """An image (B, H, W, C) on the [-1, 1] scale to model-space latents
    (B, C, h, w) in float32. ``mode="mode"`` takes the mean, and
    ``mode="sample"`` adds ``noise`` (B, C, h, w) scaled by the standard
    deviation."""
    if mode not in ("mode", "sample"):
        raise ValueError(f"encode mode must be 'mode' or 'sample', not {mode!r}")
    x = mx.array(image)
    s = vae.config.scale
    if x.shape[1] % s or x.shape[2] % s:
        raise ValueError(f"image size {x.shape[1]}x{x.shape[2]} is not a "
                         f"multiple of {s}")
    c = vae.config.latent_channels

    def moments(t):
        mean, logvar = vae.encode(t)
        return mx.concatenate([mean, logvar], axis=-1)

    if not tiled or _fits(x.shape[1] // s, x.shape[2] // s, tile // s):
        m = moments(x).astype(mx.float32)
    else:
        m = mx.array(run_tiled(moments, x, in_scale=s, out_scale=1,
                               tile=tile // s, overlap=overlap // s))
    mean, logvar = m[..., :c], m[..., c:]
    if mode == "sample":
        if noise is None:
            raise ValueError("encode mode 'sample' needs noise")
        eps = mx.array(noise).astype(mx.float32).transpose(0, 2, 3, 1)
        mean = mean + mx.exp(0.5 * logvar) * eps
    return vae.to_model_space(mean)
