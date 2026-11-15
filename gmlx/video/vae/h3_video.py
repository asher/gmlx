"""MiniMax-H3's video VAE, from Comfy-Org's safetensors file as published.

The encoder is a causal Conv3d CNN: reflect padding in space, zero frames
in front in time, GroupNorm over each frame. A clip is encoded in pieces
of 17 frames, a short last piece padded by repeating its last frame, and
the latents of the pieces lose their last 3. A single image is encoded
alone and gives one latent. The decoder is a ViT over 256 px tiles of 7
latents, each latent token 4 frames of 16x16 pixels, with 4 register
tokens and one zero token after the patch tokens. Tiles blend across
their overlaps, and runs of 7 latents at a stride of 5 crossfade over the
5 frames two runs share (``chunking``). A single latent decodes as the
first latent of a clip of 5 copies of it, which keeps the first frame.

Frames are channels last, (B, T, H, W, 3): in [-1, 1] into the encoder
and float32 in [0, 1] out of the decoder. Latents are (B, 24, T, H, W) in
model space, the layout the DiT takes. Convs run as the flipped kernel
with ``flip=True`` (``image.vae.conv``), whose products are summed in
float32 and rounded once. The decoder holds its residual stream in
float32 and runs its products in the weights' dtype. The layers
MiniMax-H3-Image's VAE shares are in ``gmlx.image.h3.vit``.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from gmlx.image.h3.tiles import MIN_OVERLAP, TILE, split_tiles  # noqa: F401
from gmlx.image.h3.vit import (  # noqa: F401
    GROUP_EPS, GROUPS, HEAD_DIM, IMAGENET_MEAN, IMAGENET_STD, Attention, Block,
    CausalConv3d, Config, Downsample, Encoder, FeedForward, FrameGroupNorm,
    Level, PointConv, ResnetBlock, RMSNorm, _take, _triple, blend, conv3d,
    normalized, owned, reflect_pad, rope_tables, run_blocks, split_half_rope,
    tiled_moments, token_coords)

from .chunking import TimeChunks

CHUNKS = TimeChunks(clip=17, ratio=4, drop=3)


def config_from_shapes(shapes: dict):
    """The config of the H3 video VAE whose torch shapes are ``shapes``,
    or None for another tree. The level strides and the head width are
    the reference's; the shapes give the widths and counts."""
    try:
        ch = shapes["encoder.conv_in.weight"][0]
        levels = 1 + max(int(n.split(".")[2]) for n in shapes
                         if n.startswith("encoder.down."))
        blocks = 1 + max(int(n.split(".")[4]) for n in shapes
                         if n.startswith("encoder.down.0.block."))
        mult = tuple(shapes[f"encoder.down.{i}.block.0.conv1.weight"][0] // ch
                     for i in range(levels))
        z = shapes["quant_conv.weight"][0] // 2
        layers = 1 + max(int(n.split(".")[2]) for n in shapes
                         if n.startswith("decoder.transformer_blocks."))
        dim = shapes["decoder.x_embedder.weight"][0]
        ff = shapes["decoder.transformer_blocks.0.ff.w2.weight"][1]
        registers = shapes["decoder.register_tokens"][1]
    except (KeyError, ValueError, IndexError):
        return None
    cfg = Config(ch=ch, ch_mult=mult, res_blocks=blocks, z=z, layers=layers,
                 dim=dim, ff=ff, registers=registers)
    down = tuple(f"encoder.down.{i}.downsample.conv.weight" in shapes
                 for i in range(levels))
    if (levels != len(cfg.space_down) or dim % HEAD_DIM
            or down != tuple(s * t > 1 for s, t in
                             zip(cfg.space_down, cfg.time_down))
            or shapes["decoder.proj_out.weight"][0]
            != 3 * cfg.frames * cfg.spatial ** 2):
        return None
    return cfg


class Decoder(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.x_embedder = nn.Linear(cfg.z, cfg.dim)
        self.register_tokens = mx.zeros((1, cfg.registers, cfg.dim))
        self.transformer_blocks = [Block(cfg) for _ in range(cfg.layers)]
        self.norm_out = nn.LayerNorm(cfg.dim, eps=cfg.eps)
        self.proj_out = nn.Linear(cfg.dim, 3 * cfg.frames * cfg.spatial ** 2)

    def __call__(self, z: mx.array) -> mx.array:
        """Raw pixels (B, 4 T, 16 H, 16 W, 3) of latents (B, T, H, W, C)."""
        cfg = self.cfg
        b, t, hh, ww, c = z.shape
        dt = self.x_embedder.weight.dtype
        n = t * hh * ww
        x = self.x_embedder(z.reshape(b, n, c).astype(dt)).astype(mx.float32)
        regs = mx.broadcast_to(self.register_tokens.astype(mx.float32),
                               (b, cfg.registers, cfg.dim))
        x = mx.concatenate([x, regs, mx.zeros((b, 1, cfg.dim))], axis=1)
        coords = mx.concatenate([token_coords(t, hh, ww),
                                 mx.zeros((cfg.registers + 1, 3))], axis=0)
        cos, sin = rope_tables(cfg, coords)
        x = run_blocks(self.transformer_blocks, x, cos, sin)
        x = self.norm_out(x[:, :n]).astype(dt)
        out = self.proj_out(x).astype(mx.float32)
        p, f = cfg.spatial, cfg.frames
        out = out.reshape(b, t, hh, ww, 3, f, p, p)
        out = out.transpose(0, 1, 5, 2, 6, 3, 7, 4)
        return out.reshape(b, t * f, hh * p, ww * p, 3)


def drop_latents(m: mx.array) -> mx.array:
    """The moments of a clip's pieces (B, T, H, W, C) less the last
    ``CHUNKS.drop`` frames."""
    return m[:, :m.shape[1] - CHUNKS.drop]


class H3VideoVAE(nn.Module):
    """The encoder, the decoder and the latent statistics."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.encoder = Encoder(cfg)
        self.quant_conv = PointConv(2 * cfg.z, 2 * cfg.z)
        self.post_quant_conv = PointConv(cfg.z, cfg.z)
        self.decoder = Decoder(cfg)
        self.latents_mean = mx.zeros((cfg.z,))
        self.latents_std = mx.ones((cfg.z,))

    # encode

    def _normalized(self, frames: mx.array) -> mx.array:
        return normalized(frames, self.encoder.conv_in.weight.dtype)

    def _moments(self, x: mx.array) -> mx.array:
        return self.quant_conv(self.encoder(x)).astype(mx.float32)

    def _tiled_moments(self, x: mx.array) -> mx.array:
        """Moments of one piece, spatial tiles blended with the tiles
        above and to the left as each was encoded."""
        return tiled_moments(x, self._moments, self.cfg.spatial)

    def encode(self, frames: mx.array, drop: bool = True) -> mx.array:
        """Model-space latents (B, 24, T, H, W) of frames (B, T, H, W, 3)
        in [-1, 1]: the posterior mean, normalized per channel. Each piece
        encodes alone, so a run of whole pieces with ``drop`` False gives
        the latents a whole clip's encode gives those pieces."""
        t = frames.shape[1]
        if t == 1:
            m = self._tiled_moments(self._normalized(frames))[:, -1:]
        else:
            clip = CHUNKS.clip
            pieces = []
            for i in range(-(-t // clip)):
                piece = frames[:, i * clip:(i + 1) * clip]
                if piece.shape[1] < clip:
                    last = mx.repeat(piece[:, -1:], clip - piece.shape[1],
                                     axis=1)
                    piece = mx.concatenate([piece, last], axis=1)
                pieces.append(self._tiled_moments(self._normalized(piece)))
            m = mx.concatenate(pieces, axis=1)
            if drop:
                m = drop_latents(m)
        mean = m[..., :self.cfg.z]
        z = ((mean - self.latents_mean.astype(mx.float32))
             / self.latents_std.astype(mx.float32))
        return z.transpose(0, 4, 1, 2, 3)

    # decode

    def _tile(self, z: mx.array) -> mx.array:
        out = self.decoder(self.post_quant_conv(
            z.astype(self.decoder.x_embedder.weight.dtype)))
        mx.eval(out)
        return out

    def _tiled_decode(self, z: mx.array,
                      frames: slice = slice(None)) -> mx.array:
        """Raw pixels of one run of latents (B, T, H, W, C), the decoded
        ``frames`` of it alone. Each tile blends with the rows above as
        they were blended and with the tile to its left as it was
        blended."""
        sp = self.cfg.spatial
        h, w = z.shape[2] * sp, z.shape[3] * sp
        ys, ylen, yo = split_tiles(h, TILE, MIN_OVERLAP, sp)
        xs, xlen, xo = split_tiles(w, TILE, MIN_OVERLAP, sp)
        canvas = None
        for i, (y, yl) in enumerate(zip(ys, ylen)):
            above = None if canvas is None else owned(
                canvas[:, :, y:y + yo[i - 1]])
            for j, (x, xl) in enumerate(zip(xs, xlen)):
                tile = self._tile(z[:, :, y // sp:(y + yl) // sp,
                                    x // sp:(x + xl) // sp])[:, frames]
                if canvas is None:
                    canvas = mx.zeros((tile.shape[0], tile.shape[1], h, w, 3))
                if above is not None:
                    tile = blend(above[:, :, :, x:x + xl], tile, yo[i - 1],
                                 axis=2)
                if j > 0:
                    tile = blend(owned(canvas[:, :, y:y + yl,
                                              x:x + xo[j - 1]]),
                                 tile, xo[j - 1], axis=3)
                canvas[:, :, y:y + yl, x:x + xl] = tile
                del tile
                mx.eval(canvas)
        return canvas

    def _finish(self, raw: mx.array) -> mx.array:
        std = mx.array(IMAGENET_STD, mx.float32)
        mean = mx.array(IMAGENET_MEAN, mx.float32)
        return mx.clip(raw * std + mean, 0.0, 1.0)

    def decode_frames(self, latents: mx.array):
        """Yield float32 frames (B, n, H, W, 3) in [0, 1] of model-space
        latents (B, 24, T, H, W), a run at a time, in order."""
        z = latents.astype(mx.float32).transpose(0, 2, 3, 4, 1)
        z = z * self.latents_std.astype(mx.float32) + self.latents_mean.astype(
            mx.float32)
        t = z.shape[1]
        c = CHUNKS
        if t == 1:
            # A latent alone is the first latent of a clip of its copies.
            first = slice(c.pre_frames, c.pre_frames + 1)
            yield self._finish(self._tiled_decode(
                mx.repeat(z, c.tokens, axis=1), first))
            return
        pad, runs = c.runs(t)
        if pad:
            z = mx.concatenate([z, mx.repeat(z[:, -1:], pad, axis=1)], axis=1)
        left = c.frames(t)
        head = c.tokens * c.ratio
        tail = None
        for i in range(runs):
            raw = self._tiled_decode(z[:, i * c.tokens:
                                       i * c.tokens + c.tokens + c.overlap])
            part = raw[:, c.pre_frames:head]
            if tail is not None:
                part = blend(tail, part, c.fade, axis=1)
            tail = owned(raw[:, head + c.pre_frames:2 * head]) \
                if c.drop else None
            done = []
            for out in (part, tail if i == runs - 1 else None):
                if out is None or left <= 0:
                    continue
                done.append(self._finish(out[:, :left]))
                left -= done[-1].shape[1]
            mx.eval(done)
            del raw, part, out
            while done:
                yield done.pop(0)

    def decode(self, latents: mx.array) -> mx.array:
        """``decode_frames`` concatenated, (B, T, H, W, 3)."""
        return mx.concatenate(list(self.decode_frames(latents)), axis=1)
