"""MiniMax-H3-Image's VAE, from Comfy-Org's safetensors file as published.

The encoder is MiniMax-H3 video's causal Conv3d CNN on one frame, with its
quant_conv. The decoder is a 2D ViT over one token per latent pixel, each
token 16x16 pixels, with 4 register tokens and one zero token after the
patch tokens: 24 blocks of width 1024, 16 heads of 64. RoPE rotates the
first 32 dims of each head as split halves, 8 frequencies an axis over each
tile's own token grid, and the registers and the zero token sit at 0. The
file's post_quant_conv is a 1x1 Conv2d, read as the video VAE's point conv.
The file holds no latent statistics: model space takes the video VAE's.

The VAE owns its tiles, as ComfyUI's class does: a pass past one tile runs
on tiles of 256 px that overlap by at least 64, whatever the caller asks.
Decoded tiles blend with the tiles above and to the left as those were
blended, ``TILES_PER_FORWARD`` tiles of a row through the decoder at a time.
Encoded tiles blend as encoded (``vit.tiled_moments``).

``decode`` maps VAE-space latents (B, h, w, 24) to an image (B, 16h, 16w, 3)
in float32 on [-1, 1]: the decoder's pixels times the ImageNet deviation
plus its mean, clamped to [0, 1]. ``encode`` takes an image on [-1, 1] and
returns the mean and log variance of the latent distribution, the log
variance clamped to [-30, 20], so a sampled encode is ComfyUI's.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from gmlx.image.h3 import vit
from gmlx.image.h3.tiles import MIN_OVERLAP, TILE, split_tiles

from .reader import VaeRow

LATENTS_MEAN = (
    0.858090341091156, -0.9606591463088989, 1.0661640167236328,
    -0.5090325474739075, -0.2727581858634949, -1.3675414323806763,
    -0.2553254961967468, -0.26907554268836975, -0.5376840829849243,
    -0.0464097298681736, 0.6657370328903198, 0.19690127670764923,
    -0.5460608005523682, -0.4035342037677765, -0.23683024942874908,
    0.25928452610969543, -0.30133944749832153, 0.211341992020607,
    -1.1206848621368408, 0.3581933379173279, -0.04225143790245056,
    0.2604829967021942, 0.22864092886447906, 0.7056031823158264)
LATENTS_STD = (
    1.2223774194717407, 1.2767263650894165, 1.6831774711608887,
    1.7549455165863037, 1.5636216402053833, 2.194143533706665,
    0.9653137922286987, 1.0569885969161987, 0.841948926448822,
    0.7729952931404114, 1.8955937623977661, 0.946841835975647,
    0.7996809482574463, 0.44988900423049927, 0.7197399735450745,
    0.6936293244361877, 2.961095094680786, 2.7694199085235596,
    3.0496184825897217, 2.1088054180145264, 3.276226282119751,
    3.1627357006073, 2.2816812992095947, 2.6127843856811523)
LOGVAR_RANGE = (-30.0, 20.0)
TILES_PER_FORWARD = 4


@dataclass(frozen=True)
class H3ImageVAEConfig(vit.Config):
    layers: int = 24
    dim: int = 1024
    ff: int = 4096
    rope_ratio: float = 0.5

    @property
    def scale(self) -> int:
        """Pixels per latent pixel along each axis."""
        return self.spatial

    @property
    def in_channels(self) -> int:
        return 3

    @property
    def out_channels(self) -> int:
        return 3


def config_from_shapes(shapes: dict) -> H3ImageVAEConfig | None:
    """The config of the MiniMax-H3-Image VAE whose torch shapes are
    ``shapes``, or None for another tree. The level strides and the head
    width are the reference's; the shapes give the widths and counts."""
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
        post = tuple(shapes["post_quant_conv.weight"])
        out = shapes["decoder.proj_out.weight"][0]
    except (KeyError, ValueError, IndexError):
        return None
    cfg = H3ImageVAEConfig(ch=ch, ch_mult=mult, res_blocks=blocks, z=z,
                           layers=layers, dim=dim, ff=ff, registers=registers)
    down = tuple(f"encoder.down.{i}.downsample.conv.weight" in shapes
                 for i in range(levels))
    if (levels != len(cfg.space_down) or dim % vit.HEAD_DIM
            or down != tuple(s * t > 1 for s, t in
                             zip(cfg.space_down, cfg.time_down))
            or post != (z, z, 1, 1) or out != 3 * cfg.spatial ** 2):
        return None
    return cfg


def layout(name: str, shape: tuple):
    """A Conv3d kernel (O, I, T, kH, kW) keeps its frames as (O, T, kH, kW,
    I); the 1x1 Conv2d post_quant_conv (O, I, 1, 1) becomes the point
    conv's (O, 1, 1, 1, I)."""
    if len(shape) == 5:
        o, i, t, h, w = shape
        return (0, 2, 3, 4, 1), (o, t, h, w, i)
    if len(shape) == 4:
        o, i = shape[:2]
        return None, (o, 1, 1, 1, i)
    return None, tuple(shape)


def expected_unused(cfg: H3ImageVAEConfig) -> dict:
    return {"decoder.mask_token": (1, 1, cfg.dim)}


class Decoder(nn.Module):
    def __init__(self, cfg: H3ImageVAEConfig):
        super().__init__()
        self.cfg = cfg
        self.x_embedder = nn.Linear(cfg.z, cfg.dim)
        self.register_tokens = mx.zeros((1, cfg.registers, cfg.dim))
        self.transformer_blocks = [vit.Block(cfg) for _ in range(cfg.layers)]
        self.norm_out = nn.LayerNorm(cfg.dim, eps=cfg.eps)
        self.proj_out = nn.Linear(cfg.dim, 3 * cfg.spatial ** 2)

    def __call__(self, z: mx.array) -> mx.array:
        """Raw pixels (B, 16 h, 16 w, 3) of latents (B, h, w, C)."""
        cfg = self.cfg
        b, hh, ww, c = z.shape
        dt = self.x_embedder.weight.dtype
        n = hh * ww
        x = self.x_embedder(z.reshape(b, n, c).astype(dt)).astype(mx.float32)
        regs = mx.broadcast_to(self.register_tokens.astype(mx.float32),
                               (b, cfg.registers, cfg.dim))
        x = mx.concatenate([x, regs, mx.zeros((b, 1, cfg.dim))], axis=1)
        coords = mx.concatenate([vit.token_coords(hh, ww),
                                 mx.zeros((cfg.registers + 1, 2))], axis=0)
        cos, sin = vit.rope_tables(cfg, coords)
        x = vit.run_blocks(self.transformer_blocks, x, cos, sin)
        x = self.norm_out(x[:, :n]).astype(dt)
        out = self.proj_out(x).astype(mx.float32)
        p = cfg.spatial
        out = out.reshape(b, hh, ww, 3, p, p).transpose(0, 1, 4, 2, 5, 3)
        return out.reshape(b, hh * p, ww * p, 3)


class H3ImageVAE(nn.Module):
    """The encoder, the decoder and their point convs. Model space, which
    the DiT works in, is (B, 24, h, w), (z - mean) / std per channel."""

    lockstep = False

    def __init__(self, cfg: H3ImageVAEConfig):
        super().__init__()
        self.config = cfg
        self.encoder = vit.Encoder(cfg)
        self.quant_conv = vit.PointConv(2 * cfg.z, 2 * cfg.z)
        self.post_quant_conv = vit.PointConv(cfg.z, cfg.z)
        self.decoder = Decoder(cfg)

    @property
    def dtype(self) -> mx.Dtype:
        return self.decoder.x_embedder.weight.dtype

    def channels(self, op: str) -> int:
        """The width ``op`` runs at: the decoder's tokens, the encoder's
        first level."""
        return self.config.dim if op == "decode" else self.config.ch

    # decode

    def decode(self, z: mx.array,
               per_forward: int = TILES_PER_FORWARD) -> mx.array:
        """An image (B, 16h, 16w, 3) in float32 on [-1, 1] of VAE-space
        latents (B, h, w, C), evaluated, ``per_forward`` tiles of a row
        through the decoder at a time."""
        out = self.finish(self.tiled_decode(z, per_forward))
        mx.eval(out)
        return out

    def finish(self, raw: mx.array) -> mx.array:
        std = mx.array(vit.IMAGENET_STD, mx.float32)
        mean = mx.array(vit.IMAGENET_MEAN, mx.float32)
        return mx.clip(raw * std + mean, 0.0, 1.0) * 2.0 - 1.0

    def _forward(self, z: mx.array) -> mx.array:
        out = self.decoder(self.post_quant_conv(z.astype(self.dtype)))
        mx.eval(out)
        return out

    def _row(self, z: mx.array, xs: list, xlen: list, per_forward: int):
        """The raw tiles of one row of latents (B, h, w, C), in order."""
        sp, b = self.config.spatial, z.shape[0]
        tiles = [z[:, :, x // sp:(x + xl) // sp] for x, xl in zip(xs, xlen)]
        for k in range(0, len(tiles), max(1, per_forward)):
            group = tiles[k:k + max(1, per_forward)]
            out = self._forward(mx.concatenate(group, axis=0))
            for g in range(len(group)):
                yield out[g * b:(g + 1) * b]
            del out

    def tiled_decode(self, z: mx.array,
                     per_forward: int = TILES_PER_FORWARD) -> mx.array:
        """Raw pixels (B, 16h, 16w, 3) of latents (B, h, w, C). Each tile
        blends with the rows above as they were blended and with the tile
        to its left as it was blended."""
        sp = self.config.spatial
        b, h, w = z.shape[0], z.shape[1] * sp, z.shape[2] * sp
        ys, ylen, yo = split_tiles(h, TILE, MIN_OVERLAP, sp)
        xs, xlen, xo = split_tiles(w, TILE, MIN_OVERLAP, sp)
        canvas = mx.zeros((b, h, w, 3))
        for i, (y, yl) in enumerate(zip(ys, ylen)):
            above = None if i == 0 else vit.owned(canvas[:, y:y + yo[i - 1]])
            tiles = self._row(z[:, y // sp:(y + yl) // sp], xs, xlen,
                              per_forward)
            for j, (x, xl) in enumerate(zip(xs, xlen)):
                tile = next(tiles)
                if above is not None:
                    tile = vit.blend(above[:, :, x:x + xl], tile, yo[i - 1],
                                     axis=1)
                if j > 0:
                    tile = vit.blend(vit.owned(canvas[:, y:y + yl,
                                                      x:x + xo[j - 1]]),
                                     tile, xo[j - 1], axis=2)
                canvas[:, y:y + yl, x:x + xl] = tile
                del tile
                mx.eval(canvas)
            del tiles, above
        return canvas

    # encode

    def encode(self, x: mx.array) -> tuple[mx.array, mx.array]:
        """The mean and the clamped log variance (B, h, w, C), float32, of
        an image (B, H, W, 3) on [-1, 1]."""
        frames = vit.normalized(x[:, None], self.dtype)
        m = vit.tiled_moments(frames, self._moments, self.config.spatial)
        return self.split_moments(m[:, 0])

    def _moments(self, x: mx.array) -> mx.array:
        return self.quant_conv(self.encoder(x)).astype(mx.float32)

    def split_moments(self, m: mx.array) -> tuple[mx.array, mx.array]:
        z = self.config.z
        return m[..., :z], mx.clip(m[..., z:], *LOGVAR_RANGE)

    # spaces

    def to_vae_space(self, latents: mx.array) -> mx.array:
        """Model-space (B, C, h, w) to VAE-space (B, h, w, C), in float32."""
        z = self.to_vae_layout(latents)
        return z * mx.array(LATENTS_STD) + mx.array(LATENTS_MEAN)

    def to_model_space(self, z: mx.array) -> mx.array:
        """VAE-space (B, h, w, C) to model-space (B, C, h, w), in float32."""
        z = z.astype(mx.float32) - mx.array(LATENTS_MEAN)
        return (z / mx.array(LATENTS_STD)).transpose(0, 3, 1, 2)

    def to_vae_layout(self, latents: mx.array) -> mx.array:
        """An array in the layout of model-space latents (B, C, h, w) in
        the layout of VAE space (B, h, w, C), in float32, with no mean or
        deviation."""
        c = self.config.z
        if latents.ndim != 4 or latents.shape[1] != c:
            raise ValueError(f"an array of shape {tuple(latents.shape)} is "
                             f"not in the layout (B, {c}, h, w)")
        return latents.astype(mx.float32).transpose(0, 2, 3, 1)


ROW = VaeRow("the MiniMax-H3-Image VAE", config_from_shapes, H3ImageVAE, 24,
             expected_unused=expected_unused, layout=layout,
             dtypes=(mx.float16, mx.float32), own_names=True)
