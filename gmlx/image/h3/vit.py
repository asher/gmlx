"""The layers of MiniMax-H3's video VAE that MiniMax-H3-Image's VAE shares.

The encoder is a causal Conv3d CNN: reflect padding in space, zero frames
in front in time, GroupNorm over each frame. The decoders are ViTs over
patch tokens with 4 register tokens and one zero token after them: each
block holds its residual stream in float32, normalizes q and k per head
and rotates the first part of each head as split halves, and runs its
products in the weights' dtype. ``run_blocks`` evaluates one block at a
time. A frame is encoded as tiles whose moments blend with the tiles above
and to the left (``tiled_moments``).

Frames are channels last, (B, T, H, W, 3). Convs run as the flipped kernel
with ``flip=True`` (``image.vae.conv``), whose products are summed in
float32 and rounded once.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from gmlx.image import pacing

from .tiles import MIN_OVERLAP, TILE, split_tiles

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
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
        # gmlx.image.vae reads this module through the image VAE's row.
        from gmlx.image.vae.group_norm import group_norm

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


def token_coords(*sizes: int) -> mx.array:
    """Each patch token's position on the axes of ``sizes``, (prod(sizes),
    len(sizes)), each axis centered on [-1, 1]."""
    axes = [(mx.arange(n, dtype=mx.float32) + 0.5) / n * 2.0 - 1.0
            for n in sizes]
    grid = mx.meshgrid(*axes, indexing="ij")
    return mx.stack([g.reshape(-1) for g in grid], axis=-1)


def rope_tables(cfg: Config, coords: mx.array) -> tuple[mx.array, mx.array]:
    """cos and sin (S, pairs) of the rotary angles at ``coords`` (S, axes):
    per axis ``pairs / axes`` frequencies, axis-major."""
    rot = int(HEAD_DIM * cfg.rope_ratio)
    step = 2 * coords.shape[1] / rot
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


def run_blocks(blocks, x: mx.array, cos: mx.array,
               sin: mx.array) -> mx.array:
    """``x`` through ``blocks`` in turn, evaluated after each block and
    then given to a pacing point, so a forward holds one block's working
    set at a time."""
    for block in blocks:
        x = block(x, cos, sin)
        mx.eval(x)
        pacing.point(x)
    return x


def owned(x: mx.array) -> mx.array:
    """``x`` in a buffer of its own, evaluated, so a slice does not keep
    the array it was cut from alive."""
    y = x + mx.zeros((), x.dtype)
    mx.eval(y)
    return y


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


def normalized(frames: mx.array, dtype) -> mx.array:
    """Frames in [-1, 1] as the encoder takes them: normalized by the
    ImageNet statistics on [0, 1], in ``dtype``."""
    mean = mx.array(IMAGENET_MEAN, mx.float32)
    std = mx.array(IMAGENET_STD, mx.float32)
    x = (frames.astype(mx.float32) + 1.0) * 0.5
    return ((x - mean) / std).astype(dtype)


def tiled_moments(x: mx.array, moments, spatial: int) -> mx.array:
    """``moments`` of ``x`` (B, T, H, W, C) over spatial tiles, each tile
    blended with the tiles above and to the left as each was encoded.
    ``spatial`` is the pixels a latent spans."""
    sp = spatial
    ys, ylen, yo = split_tiles(x.shape[2], TILE, MIN_OVERLAP, sp)
    xs, xlen, xo = split_tiles(x.shape[3], TILE, MIN_OVERLAP, sp)
    rows = []
    for i, il in zip(ys, ylen):
        row = []
        for j, jl in zip(xs, xlen):
            m = moments(x[:, :, i:i + il, j:j + jl])
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
