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
float32 and runs its products in the weights' dtype.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from gmlx.image.vae.group_norm import group_norm

from .chunking import TimeChunks, split_tiles

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
TILE = 256
MIN_OVERLAP = 64
CHUNKS = TimeChunks(clip=17, ratio=4, drop=3)
GROUPS = 32
GROUP_EPS = 1e-6
HEAD_DIM = 64


@dataclass(frozen=True)
class Config:
    ch: int = 128
    ch_mult: tuple = (1, 2, 2, 4, 4, 8)
    res_blocks: int = 2
    space_down: tuple = (2, 2, 2, 2, 1, 1)
    time_down: tuple = (1, 2, 2, 1, 1, 1)
    z: int = 24
    layers: int = 36
    dim: int = 2048
    ff: int = 8192
    registers: int = 4
    rope_theta: float = 100.0
    rope_ratio: float = 0.75
    eps: float = 1e-5

    @property
    def latent_channels(self) -> int:
        return self.z

    @property
    def spatial(self) -> int:
        return math.prod(self.space_down)

    @property
    def frames(self) -> int:
        return math.prod(self.time_down)

    @property
    def heads(self) -> int:
        return self.dim // HEAD_DIM


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


def _triple(v) -> tuple:
    return tuple(v) if isinstance(v, (tuple, list)) else (v, v, v)


def reflect_pad(x: mx.array, axis: int, before: int, after: int) -> mx.array:
    """``x`` padded along ``axis`` by mirroring it about its edge rows,
    which stay single, as torch's reflect mode does."""
    if not before and not after:
        return x
    n = x.shape[axis]
    idx = (list(range(before, 0, -1)) + list(range(n))
           + list(range(n - 2, n - 2 - after, -1)))
    return mx.take(x, mx.array(idx), axis=axis)


def conv3d(x: mx.array, w: mx.array, stride) -> mx.array:
    return mx.conv_general(x, w[:, ::-1, ::-1, ::-1, :], stride=stride,
                           flip=True)


class CausalConv3d(nn.Module):
    """A Conv3d on (B, T, H, W, C) whose frames are padded in front with
    zeros and whose sides are padded by reflection. A single frame runs
    on the kernel's last time tap alone."""

    def __init__(self, cin: int, cout: int, kernel, stride=1, pad=0):
        super().__init__()
        self.stride = _triple(stride)
        self.pad = _triple(pad)
        self.weight = mx.zeros((cout, *_triple(kernel), cin))
        self.bias = mx.zeros((cout,))

    def __call__(self, x: mx.array, sides=None) -> mx.array:
        pt, ph, pw = self.pad
        top, bottom, left, right = sides or (ph, ph, pw, pw)
        x = reflect_pad(reflect_pad(x, 2, top, bottom), 3, left, right)
        w = self.weight
        if pt and x.shape[1] == 1:
            w = w[:, -1:]
        elif pt:
            x = mx.pad(x, [(0, 0), (2 * pt, 0), (0, 0), (0, 0), (0, 0)])
        y = conv3d(x, w, self.stride)
        return y + self.bias.astype(y.dtype)


class FrameGroupNorm(nn.Module):
    """GroupNorm over each frame on its own."""

    def __init__(self, channels: int):
        super().__init__()
        self.weight = mx.ones((channels,))
        self.bias = mx.zeros((channels,))

    def __call__(self, x: mx.array) -> mx.array:
        b, t = x.shape[:2]
        y = group_norm(x.reshape(b * t, *x.shape[2:]), self.weight,
                       self.bias, GROUPS, GROUP_EPS)
        return y.reshape(x.shape)


class ResnetBlock(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.norm1 = FrameGroupNorm(cin)
        self.conv1 = CausalConv3d(cin, cout, 3, pad=1)
        self.norm2 = FrameGroupNorm(cout)
        self.conv2 = CausalConv3d(cout, cout, 3, pad=1)
        if cin != cout:
            self.nin_shortcut = CausalConv3d(cin, cout, 1)

    def __call__(self, x: mx.array) -> mx.array:
        h = self.conv1(nn.silu(self.norm1(x)))
        if "nin_shortcut" in self:
            x = self.nin_shortcut(x)
        return self.conv2(nn.silu(self.norm2(h))) + x


class Downsample(nn.Module):
    def __init__(self, channels: int, time: int, space: int):
        super().__init__()
        self.space = space
        self.conv = CausalConv3d(channels, channels, 3,
                                 stride=(time, space, space), pad=(1, 0, 0))

    def __call__(self, x: mx.array) -> mx.array:
        return self.conv(x, (0, 1, 0, 1) if self.space == 2 else None)


class Level(nn.Module):
    def __init__(self, cin: int, cout: int, n: int, time: int, space: int):
        super().__init__()
        self.block = [ResnetBlock(cin if i == 0 else cout, cout)
                      for i in range(n)]
        if time * space > 1:
            self.downsample = Downsample(cout, time, space)


class Encoder(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        widths = [cfg.ch * m for m in cfg.ch_mult]
        self.conv_in = CausalConv3d(3, widths[0], 3, pad=1)
        self.down = [Level(widths[max(i - 1, 0)], widths[i], cfg.res_blocks,
                           cfg.time_down[i], cfg.space_down[i])
                     for i in range(len(widths))]
        self.norm_out = FrameGroupNorm(widths[-1])
        self.conv_out = CausalConv3d(widths[-1], 2 * cfg.z, 3, pad=1)

    def __call__(self, x: mx.array) -> mx.array:
        h = self.conv_in(x)
        for level in self.down:
            for block in level.block:
                h = block(h)
            if "downsample" in level:
                h = level.downsample(h)
        return self.conv_out(nn.silu(self.norm_out(h)))


class PointConv(nn.Module):
    """A 1x1x1 Conv3d."""

    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.weight = mx.zeros((cout, 1, 1, 1, cin))
        self.bias = mx.zeros((cout,))

    def __call__(self, x: mx.array) -> mx.array:
        w = self.weight.reshape(self.weight.shape[0], -1)
        return x @ w.T.astype(x.dtype) + self.bias.astype(x.dtype)


def token_coords(t: int, h: int, w: int) -> mx.array:
    """Each patch token's position on the three axes, (t * h * w, 3), each
    axis centered on [-1, 1]."""
    axes = [(mx.arange(n, dtype=mx.float32) + 0.5) / n * 2.0 - 1.0
            for n in (t, h, w)]
    grid = mx.meshgrid(*axes, indexing="ij")
    return mx.stack([g.reshape(-1) for g in grid], axis=-1)


def rope_tables(cfg: Config, coords: mx.array) -> tuple[mx.array, mx.array]:
    """cos and sin (S, pairs) of the rotary angles at ``coords`` (S, 3):
    per axis ``pairs / 3`` frequencies, axis-major."""
    rot = int(HEAD_DIM * cfg.rope_ratio)
    step = 2 * 3 / rot
    inv = 1.0 / cfg.rope_theta ** mx.arange(0.0, 1.0, step, dtype=mx.float32)
    angles = 2 * math.pi * coords[:, :, None] * inv[None, None, :]
    angles = angles.reshape(coords.shape[0], -1)
    return mx.cos(angles), mx.sin(angles)


def split_half_rope(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    """Rotate pairs (i, i + pairs) of the first 2 * pairs dims of each
    head of ``x`` (B, S, heads, head_dim); the other dims pass through."""
    p = cos.shape[-1]
    c, s = cos[None, :, None, :], sin[None, :, None, :]
    x1, x2 = x[..., :p], x[..., p:2 * p]
    return mx.concatenate([x1 * c - x2 * s, x1 * s + x2 * c, x[..., 2 * p:]],
                          axis=-1)


class Attention(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.to_qkv = nn.Linear(dim, 3 * dim)
        self.to_out = nn.Linear(dim, dim)


class FeedForward(nn.Module):
    def __init__(self, dim: int, ff: int):
        super().__init__()
        self.w1 = nn.Linear(dim, 2 * ff)
        self.w2 = nn.Linear(ff, dim)


class RMSNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.weight = mx.ones((dim,))


class Block(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.heads, self.eps = cfg.heads, cfg.eps
        self.norm1 = RMSNorm(cfg.dim)
        self.attn = Attention(cfg.dim)
        self.scale1 = mx.zeros((cfg.dim,))
        self.norm2 = RMSNorm(cfg.dim)
        self.ff = FeedForward(cfg.dim, cfg.ff)
        self.scale2 = mx.zeros((cfg.dim,))

    def __call__(self, x: mx.array, cos: mx.array, sin: mx.array):
        """``x`` is the float32 residual stream (B, S, dim)."""
        dt = self.attn.to_qkv.weight.dtype
        b, s, _ = x.shape
        h = mx.fast.rms_norm(x, self.norm1.weight.astype(mx.float32),
                             self.eps).astype(dt)
        qkv = self.attn.to_qkv(h).reshape(b, s, self.heads, 3 * HEAD_DIM)
        q, k, v = mx.split(qkv, 3, axis=-1)
        q, k = (split_half_rope(mx.fast.rms_norm(t.astype(mx.float32), None,
                                                 self.eps), cos, sin)
                .astype(dt) for t in (q, k))
        o = mx.fast.scaled_dot_product_attention(
            q.transpose(0, 2, 1, 3), k.transpose(0, 2, 1, 3),
            v.transpose(0, 2, 1, 3), scale=HEAD_DIM ** -0.5)
        o = self.attn.to_out(o.transpose(0, 2, 1, 3).reshape(b, s, -1))
        x = x + self.scale1.astype(mx.float32) * o.astype(mx.float32)
        h = mx.fast.rms_norm(x, self.norm2.weight.astype(mx.float32),
                             self.eps).astype(dt)
        gate, up = mx.split(self.ff.w1(h), 2, axis=-1)
        o = self.ff.w2(nn.silu(gate) * up)
        return x + self.scale2.astype(mx.float32) * o.astype(mx.float32)


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
        for block in self.transformer_blocks:
            x = block(x, cos, sin)
        x = self.norm_out(x[:, :n]).astype(dt)
        out = self.proj_out(x).astype(mx.float32)
        p, f = cfg.spatial, cfg.frames
        out = out.reshape(b, t, hh, ww, 3, f, p, p)
        out = out.transpose(0, 1, 5, 2, 6, 3, 7, 4)
        return out.reshape(b, t * f, hh * p, ww * p, 3)


def owned(x: mx.array) -> mx.array:
    """``x`` in a buffer of its own, evaluated, so a slice does not keep
    the array it was cut from alive."""
    y = x + mx.zeros((), x.dtype)
    mx.eval(y)
    return y


def drop_latents(m: mx.array) -> mx.array:
    """The moments of a clip's pieces (B, T, H, W, C) less the last
    ``CHUNKS.drop`` frames."""
    return m[:, :m.shape[1] - CHUNKS.drop]


def blend(a: mx.array, b: mx.array, extent: int, axis: int) -> mx.array:
    """``b`` with its first ``extent`` rows along ``axis`` crossfaded
    from the last ``extent`` rows of ``a``, linearly from all ``a``."""
    extent = min(a.shape[axis], b.shape[axis], extent)
    if extent <= 0:
        return b
    shape = [1] * b.ndim
    shape[axis] = extent
    wb = (mx.arange(extent, dtype=mx.float32) / extent).reshape(shape)
    head = (_take(a, axis, a.shape[axis] - extent, a.shape[axis]) * (1 - wb)
            + _take(b, axis, 0, extent) * wb)
    return mx.concatenate([head, _take(b, axis, extent, b.shape[axis])],
                          axis=axis)


def _take(x: mx.array, axis: int, lo: int, hi: int) -> mx.array:
    idx = [slice(None)] * x.ndim
    idx[axis] = slice(lo, hi)
    return x[tuple(idx)]


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
        mean = mx.array(IMAGENET_MEAN, mx.float32)
        std = mx.array(IMAGENET_STD, mx.float32)
        x = (frames.astype(mx.float32) + 1.0) * 0.5
        return ((x - mean) / std).astype(self.encoder.conv_in.weight.dtype)

    def _moments(self, x: mx.array) -> mx.array:
        return self.quant_conv(self.encoder(x)).astype(mx.float32)

    def _tiled_moments(self, x: mx.array) -> mx.array:
        """Moments of one piece, spatial tiles blended with the tiles
        above and to the left as each was encoded."""
        sp = self.cfg.spatial
        ys, ylen, yo = split_tiles(x.shape[2], TILE, MIN_OVERLAP, sp)
        xs, xlen, xo = split_tiles(x.shape[3], TILE, MIN_OVERLAP, sp)
        rows = []
        for i, il in zip(ys, ylen):
            row = []
            for j, jl in zip(xs, xlen):
                m = self._moments(x[:, :, i:i + il, j:j + jl])
                mx.eval(m)
                row.append(m)
            rows.append(row)
        out_rows = []
        for i, row in enumerate(rows):
            parts = []
            for j, m in enumerate(row):
                if i > 0:
                    m = blend(rows[i - 1][j], m, yo[i - 1] // sp, axis=2)
                if j > 0:
                    m = blend(row[j - 1], m, xo[j - 1] // sp, axis=3)
                if i < len(rows) - 1:
                    m = m[:, :, :m.shape[2] - yo[i] // sp]
                if j < len(row) - 1:
                    m = m[:, :, :, :m.shape[3] - xo[j] // sp]
                parts.append(m)
            out_rows.append(mx.concatenate(parts, axis=3))
        return mx.concatenate(out_rows, axis=2)

    def encode(self, frames: mx.array) -> mx.array:
        """Model-space latents (B, 24, T, H, W) of frames (B, T, H, W, 3)
        in [-1, 1]: the posterior mean, normalized per channel."""
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
            m = drop_latents(mx.concatenate(pieces, axis=1))
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
