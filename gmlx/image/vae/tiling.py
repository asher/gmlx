# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Asher Feldman
"""VAE decode and encode, whole or in tiles.

Tiles are laid out in latent space. The default tile is at most 512
pixels, which is 64 latent pixels at 8x and 32 at 16x, and neighbors
overlap by at least 64 pixels. A side takes the fewest tiles of that size
that cover it, and the tiles shrink to the shortest equal length that
keeps the overlap, so a 1024 pixel side takes three 384 pixel tiles.
Tiles are spread evenly so the last one ends at the edge. Each tile is
weighted by a cosine ramp across every overlap with a neighbor, and the
weighted sum is divided by the summed weights. Accumulation is on the host
in float32.

GroupNorm normalizes over the whole image, so a tile normalized with its
own statistics drifts from the whole pass: at 1024 pixels in 512 pixel
tiles the FLUX.1 decode falls to 33 dB against the whole decode. The VAE
therefore runs the part at latent resolution whole (conv_in and the mid
block of the decoder, the mid block and output of the encoder), and the
rest in tiles that advance together. At each GroupNorm every tile yields
its input, and the statistics of the image are merged from the region of
each tile that is nearer its own center than any neighbor's. All tiles'
activations at one GroupNorm are held at once, which is about the size of
that activation for the whole image times the overlap.

The FLUX.2 decoder reads farther across a cut than the FLUX.1 one, so its
VAE row sets the least overlap of its tiles to 128 pixels
(``memfit.tile_overlap``). At 1024 pixels in tiles that overlap by 64, its
tiled decode is 52 dB from the whole decode, where the FLUX.1 decode holds
75 dB.

The Qwen-Image 2.1 autoencoder normalizes each pixel over its channels and
takes no statistic of the whole image. The part with its mid attention
runs whole too, so the attention sees every latent pixel, and its tiles
run one at a time, each evaluated before the next. A tile's convs pad
zeros where the whole pass reads the neighboring tile. A tile therefore
equals the whole pass at every pixel farther from each cut tile edge than
the convs of the tiled part reach (``qwen_image21_vae.reach``), and
differs nearer.

Its tiles are cropped and not blended, in a decode and in an encode.
Neighbors overlap by at least twice the reach, each tile gives the pixels
nearer its center than a neighbor's, and no pixel within the reach of a
cut is kept. The result is the whole pass up to the rounding of the
device, which can differ between input shapes. A blend of the two tiles
of an overlap does not give that: on texture and on text the tiles of
the decode disagree by 8 to 15 levels of 255 in the mean before a blend.
A decode runs conv_in, the mid block and the first up block on the whole
image, and the other four up blocks in tiles, which read 7 latent pixels
across a cut. An encode runs conv_in and the down blocks in tiles, which
read 10. A cropped tile is longer than twice the reach, so a tile is 240
pixels or more in a decode and 336 or more in an encode.

``decode`` and ``encode`` evaluate their result inside
``memfit.bounded_cache``, so the buffer cache stays bounded for the whole
pass. ``memfit.vae_tiling`` prices the pass by the VAE row of the family
and refuses one that does not fit as chosen. With ``tiled=None`` a VAE
runs whole when the pass fits the free working set and in tiles
otherwise. A VAE whose row takes tiles first, as every unmeasured row
does, runs every image larger than one tile in tiles, and so does any VAE
when the device reports no free working set, and a row that bounds its
whole passes runs a pass of more pixels in tiles. A request with a memory
plan passes the plan's choice as ``tiled`` and the room the plan leaves
for the pass as ``free``. Every VAE pass goes through them, a warm-up at
a new size included. The module's own ``decode`` and
``encode`` build a graph that evaluates under whatever limit is set when
it runs.
"""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np

from .. import memfit
from .group_norm import combine_moments, group_moments

TILE = memfit.TILE_PX
OVERLAP = memfit.OVERLAP_PX


def tile_length(size: int, tile: int, overlap: int) -> int:
    """The shortest length that covers ``size`` with as few tiles of at most
    ``tile`` as possible, each overlapping the next by at least
    ``overlap``."""
    if overlap >= tile:
        raise ValueError(f"overlap {overlap} must be below the tile {tile}")
    if size <= tile:
        return size
    n = math.ceil((size - overlap) / (tile - overlap))
    return min(tile, -(-(size + (n - 1) * overlap) // n))


def tile_starts(size: int, tile: int, overlap: int) -> list[int]:
    """Start offsets of tiles of length ``min(tile, size)`` that cover
    ``[0, size)``, with the last one ending at ``size`` and neighbors
    overlapping by at least ``overlap``."""
    if size <= tile:
        return [0]
    if overlap >= tile:
        raise ValueError(f"overlap {overlap} must be below the tile {tile}")
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


def kept(starts: list[int], i: int, tile: int, size: int, scale: int,
         reach: int) -> slice:
    """Output pixels along one axis that a cropped pass takes from tile
    ``i``, in the tile's own coordinates: those nearer its center than a
    neighbor's. Neighbors are cut at the middle of their overlap, which
    must be at least twice ``reach`` latent pixels, so a kept pixel is at
    least ``reach`` from each cut edge of its tile."""
    for k in range(len(starts) - 1):
        got = starts[k] + tile - starts[k + 1]
        if got < 2 * reach:
            raise ValueError(f"tiles overlap by {got} latent pixels, and a "
                             f"crop at a reach of {reach} needs {2 * reach}")
    return _owned(starts, i, tile, size, scale)


def run_tiled(fn, x: mx.array, *, in_scale: int, out_scale: int,
              tile: int, overlap: int, reach: int = 0) -> np.ndarray:
    """``fn`` over tiles of channels-last ``x``, blended, as float32 numpy.

    ``tile`` and ``overlap`` are in latent pixels. A latent pixel is
    ``in_scale`` input pixels and ``out_scale`` output pixels.

    With a ``reach`` the tiles are cropped and not blended: the result
    copies from each tile the pixels nearer its center than a neighbor's,
    which lie at least ``reach`` latent pixels from its cut edges. When
    ``fn`` reads no input farther than that from a result, the result is
    ``fn`` of the whole input, bit for bit where ``fn`` rounds a value the
    same way in a tile as in the whole input. MLX does not always: a conv
    can round differently from one input shape to another, by a last bit
    of float32 on the CPU and by about 1e-3 of the values on a GPU that
    multiplies float32 matrices at TF32 precision.
    """
    b, h, w = x.shape[0], x.shape[1] // in_scale, x.shape[2] // in_scale
    th, tw = tile_length(h, tile, overlap), tile_length(w, tile, overlap)
    ys, xs = tile_starts(h, th, overlap), tile_starts(w, tw, overlap)
    out = wsum = None
    for i, y0 in enumerate(ys):
        wy = None if reach else axis_weights(ys, i, th, out_scale)
        for j, x0 in enumerate(xs):
            piece = fn(x[:, y0 * in_scale:(y0 + th) * in_scale,
                         x0 * in_scale:(x0 + tw) * in_scale])
            piece = np.array(piece.astype(mx.float32))
            if out is None:
                out = np.zeros((b, h * out_scale, w * out_scale,
                                piece.shape[-1]), np.float32)
                if not reach:
                    wsum = np.zeros((h * out_scale, w * out_scale),
                                    np.float32)
            if reach:
                ky = kept(ys, i, th, h, out_scale, reach)
                kx = kept(xs, j, tw, w, out_scale, reach)
                out[:, y0 * out_scale + ky.start:y0 * out_scale + ky.stop,
                    x0 * out_scale + kx.start:x0 * out_scale + kx.stop] = (
                        piece[:, ky, kx])
                continue
            wx = axis_weights(xs, j, tw, out_scale)
            wt = wy[:, None] * wx[None, :]
            ry = slice(y0 * out_scale, (y0 + th) * out_scale)
            rx = slice(x0 * out_scale, (x0 + tw) * out_scale)
            out[:, ry, rx] += piece * wt[None, :, :, None]
            wsum[ry, rx] += wt
    if reach:
        return out
    return out / wsum[None, :, :, None]


def _owned(starts: list[int], i: int, tile: int, size: int,
           r: float) -> slice:
    """Rows of tile ``i``, at ``r`` rows per latent pixel, that are nearer
    its center than a neighbor's. Neighbors share the boundary, so the
    regions of all tiles partition the image."""
    def edge(k):            # boundary between tiles k - 1 and k
        return round((starts[k - 1] + tile + starts[k]) / 2 * r)
    lo = 0 if i == 0 else edge(i)
    hi = round(size * r) if i + 1 == len(starts) else edge(i + 1)
    base = round(starts[i] * r)
    return slice(lo - base, hi - base)


def run_lockstep(steps, x: mx.array, *, in_scale: int, out_scale: int,
                 tile: int, overlap: int) -> np.ndarray:
    """``steps`` over tiles of channels-last ``x``, blended, as float32
    numpy, with every GroupNorm normalized over the whole image.

    ``steps(t)`` is a generator that yields ``(norm, h)`` before each
    GroupNorm, receives the ``(mean, var)`` to apply, and returns the tile's
    output. ``tile`` and ``overlap`` are in latent pixels, as in
    ``run_tiled``.
    """
    h, w = x.shape[1] // in_scale, x.shape[2] // in_scale
    th, tw = tile_length(h, tile, overlap), tile_length(w, tile, overlap)
    ys, xs = tile_starts(h, th, overlap), tile_starts(w, tw, overlap)
    boxes = [(i, j) for i in range(len(ys)) for j in range(len(xs))]
    gens = [steps(x[:, ys[i] * in_scale:(ys[i] + th) * in_scale,
                    xs[j] * in_scale:(xs[j] + tw) * in_scale])
            for i, j in boxes]
    outs: list = [None] * len(gens)

    def advance(k, sent):
        try:
            norm, a = gens[k].send(sent)
        except StopIteration as stop:
            outs[k] = stop.value
            mx.eval(outs[k])
            return None
        i, j = boxes[k]
        own = a[:, _owned(ys, i, th, h, a.shape[1] / th),
                _owned(xs, j, tw, w, a.shape[2] / tw)]
        moments = group_moments(own, norm.groups)
        mx.eval(a, moments[1], moments[2])
        return norm, moments

    reqs = [advance(k, None) for k in range(len(gens))]
    while reqs[0] is not None:
        norm = reqs[0][0]
        if any(r is None or r[0] is not norm for r in reqs):
            raise RuntimeError("tiles reached different GroupNorms")
        stats = combine_moments([r[1] for r in reqs])
        reqs = [advance(k, stats) for k in range(len(gens))]
    if any(r is not None for r in reqs):
        raise RuntimeError("tiles reached different GroupNorms")
    pieces = iter(outs)
    return run_tiled(lambda t: next(pieces), x, in_scale=in_scale,
                     out_scale=out_scale, tile=tile, overlap=overlap)


def row_family(vae) -> str:
    """The family whose VAE row in ``memfit`` prices ``vae``: the one it
    was loaded for, else the first family that decodes with its class."""
    family = getattr(vae, "family", None)
    if family is None:
        from .reader import VAES

        family = next((name for name, row in VAES.items()
                       if isinstance(vae, row.build)), None)
    if family is None:
        raise ValueError(f"no image family decodes with "
                         f"{type(vae).__name__}")
    return family


def price_args(vae, op: str, batch: int) -> dict:
    """What ``memfit.vae_peak`` and ``memfit.vae_tiling`` take from
    ``vae`` for ``op``."""
    return dict(family=row_family(vae), itemsize=vae.dtype.size,
                channels=vae.channels(op), config=vae.config, batch=batch,
                conv=getattr(vae, "conv", None))


def crop_layout(vae, op: str, tile: int,
                overlap: int) -> tuple[int, int, int]:
    """The tile, the overlap and the reach of a tiled ``op``, in latent
    pixels, for ``tile`` and ``overlap`` in image pixels, as the VAE row
    of the family gives them. Where the row crops the tiles of the op the
    reach is that of a cut and the overlap at least twice it. Where it
    blends them the reach is 0."""
    s = vae.config.scale
    family = row_family(vae)
    lap = memfit.tile_overlap(family, op, tile, overlap, vae.config)
    return (tile // s, lap // s, memfit.crop_reach(family, op, vae.config))


def _plan(vae, op: str, b: int, height: int, width: int, tiled,
          tile: int, overlap: int, free) -> bool:
    return memfit.vae_tiling(op, height, width, tiled=tiled, tile=tile,
                             overlap=overlap, free=free,
                             **price_args(vae, op, b))


def decode(vae, latents, *, tiled: bool | None = None, alpha: bool = False,
           tile: int = TILE, overlap: int = OVERLAP,
           free: float | None = None) -> mx.array:
    """Model-space latents (B, C, h, w) to an image (B, H, W, 3) in float32
    on the [-1, 1] scale, evaluated. The FLUX.1 autoencoder returns it
    unclamped, and the Qwen-Image 2.1 one clamped to [-1, 1]. ``alpha``
    keeps the fourth channel of a VAE that decodes RGBA. ``tile`` and
    ``overlap`` are in output pixels. ``tiled=None`` is the automatic
    choice of ``memfit.vae_tiling`` against ``free`` bytes, which default
    to the free working set. A request with a memory plan passes the two
    values of ``MemoryPlan.vae_pass``."""
    rgba = vae.config.out_channels == 4
    if alpha and not rgba:
        raise ValueError("this VAE decodes no alpha channel")
    z = vae.to_vae_space(mx.array(latents))
    s = vae.config.scale
    b, h, w = z.shape[0], z.shape[1] * s, z.shape[2] * s
    in_tiles = _plan(vae, "decode", b, h, w, tiled, tile, overlap, free)
    with memfit.bounded_cache():
        if not in_tiles:
            out = vae.decode(z).astype(mx.float32)
        elif vae.lockstep:
            t, lap, _ = crop_layout(vae, "decode", tile, overlap)
            head = vae.decode_head(z.astype(vae.dtype))
            mx.eval(head)
            out = mx.array(run_lockstep(
                vae.decoder.tail_steps, head, in_scale=1, out_scale=s,
                tile=t, overlap=lap))
        else:
            t, lap, r = crop_layout(vae, "decode", tile, overlap)
            head = vae.decode_head(z.astype(vae.dtype))
            mx.eval(head)
            out = vae.finish(mx.array(run_tiled(
                vae.decode_tail, head, in_scale=head.shape[1] // z.shape[1],
                out_scale=s, tile=t, overlap=lap, reach=r)))
        if rgba and not alpha:
            out = out[..., :3]
        mx.eval(out)
        return out


def encode(vae, image, *, tiled: bool | None = None, mode: str = "mode",
           noise=None, tile: int = TILE, overlap: int = OVERLAP,
           free: float | None = None) -> mx.array:
    """An image (B, H, W, C) on the [-1, 1] scale to model-space latents
    (B, C, h, w) in float32, evaluated. A VAE that encodes RGBA takes an
    RGB image as opaque. ``mode="mode"`` takes the mean, and
    ``mode="sample"`` adds ``noise`` (B, C, h, w) scaled by the standard
    deviation. ``tiled`` and ``free`` work as in ``decode``."""
    if mode not in ("mode", "sample"):
        raise ValueError(f"encode mode must be 'mode' or 'sample', not {mode!r}")
    if mode == "sample" and noise is None:
        raise ValueError("encode mode 'sample' needs noise")
    x = mx.array(image)
    s = vae.config.scale
    if x.shape[1] % s or x.shape[2] % s:
        raise ValueError(f"image size {x.shape[1]}x{x.shape[2]} is not a "
                         f"multiple of {s}")
    in_tiles = _plan(vae, "encode", x.shape[0], x.shape[1], x.shape[2],
                     tiled, tile, overlap, free)

    with memfit.bounded_cache():
        if not in_tiles:
            mean, logvar = vae.encode(x)
        elif vae.lockstep:
            t, lap, _ = crop_layout(vae, "encode", tile, overlap)
            head = run_lockstep(vae.encoder.head_steps, x.astype(vae.dtype),
                                in_scale=s, out_scale=1, tile=t, overlap=lap)
            mean, logvar = vae.split_moments(
                vae.encoder.tail(mx.array(head).astype(vae.dtype)))
        else:
            t, lap, r = crop_layout(vae, "encode", tile, overlap)
            head = run_tiled(vae.encode_head, x, in_scale=s, out_scale=1,
                             tile=t, overlap=lap, reach=r)
            mean, logvar = vae.split_moments(
                vae.encode_tail(mx.array(head).astype(vae.dtype)))
        mean, logvar = mean.astype(mx.float32), logvar.astype(mx.float32)
        if mode == "sample":
            eps = mx.array(noise).astype(mx.float32).transpose(0, 2, 3, 1)
            mean = mean + mx.exp(0.5 * logvar) * eps
        out = vae.to_model_space(mean)
        mx.eval(out)
    return out
