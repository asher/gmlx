# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Asher Feldman
"""Memory of a Qwen-Image 2.1 VAE pass, by arithmetic: a floor.

``activation_terms`` walks a decode or an encode op by op, as
``qwen_image21_vae`` computes it, and gives the activation bytes live on
the device at each op: the op's inputs, its output, and every tensor held
for a later op, which is a resnet's input until its sum and a block's
input until its shortcut. ``activation_peak`` is the largest term, the
op-order floor, for a whole pass or a tiled one. Each op counts as run to
completion before the next.

``memory_floor`` adds what else arithmetic gives, each as a term of its
own beside the op-order floor:

- ``weights``: every weight of the VAE at the pass dtype. The encoder's
  and the decoder's are both loaded during either pass.
- ``attention_softmax``: the softmax output of one chunk of attention
  rows, which is as large as the chunk of scores it is taken from.
- ``host``: the float32 buffers that a tiled pass keeps on the host.

The terms are those at the step where their sum is largest, and
``memory_terms`` gives them for every step. The result is a floor, and a
measured footprint is larger. It leaves out what the lazy graph keeps
alive after the last op that reads it, the freed buffers that MLX's cache
keeps, and conv scratch. Nothing here is fitted to a measurement. A norm
counts its output at the stream dtype, since the float32 sums stay inside
the kernel. Attention scores count one chunk of ``flux1_ae.chunked_sdpa``
at the stream dtype.

A tiled pass runs one part whole and the rest one tile at a time. The
whole part is the mid block with what precedes it and, in a decode, the
up blocks before the tiled ones. The device then holds one tile beside
what stays held, which is the whole part's output in a decode and the
image in an encode. The tiles are cropped, so the host holds the result
in float32 and one tile beside it. The encode's terms still count the
buffers of a blend, the result twice and a tile twice, which is more than
its crop holds.
"""

from __future__ import annotations

from .flux1_ae import ATTN_SCORE_BYTES
from .qwen_image21_vae import (CONSTANTS, IMAGE_CHANNELS, LATENT_CHANNELS,
                               QwenImage21VAEConfig, whole_up_blocks)

# The config of the official file, which the memory plan prices before a
# file is read. The widths are those of its header.
OFFICIAL = QwenImage21VAEConfig(
    in_channels=IMAGE_CHANNELS, out_channels=IMAGE_CHANNELS,
    latent_channels=LATENT_CHANNELS, base_dim=96, decoder_base_dim=144,
    dim_mult=(1, 2, 4, 8, 8), num_res_blocks=2,
    temporal=CONSTANTS["temperal_downsample"])


class _Live:
    """Live bytes per op. A tensor is an (id, bytes, shape) triple, and
    ``held`` counts the holders of each tensor kept for a later op."""

    def __init__(self, itemsize: int, batch: int):
        self.itemsize, self.batch = itemsize, batch
        self.terms: list = []           # (op name, live bytes)
        self.shapes: dict = {}          # op name -> (C, H, W) of its output
        self.held: dict = {}            # id -> [bytes, holders]
        self.softmax: dict = {}         # op name -> bytes of its softmax
        self._n = 0

    def new(self, c: int, h: int, w: int, itemsize: int | None = None):
        self._n += 1
        size = self.batch * c * h * w * (itemsize or self.itemsize)
        return (self._n, size, (c, h, w))

    def raw(self, size: int):
        self._n += 1
        return (self._n, size, None)

    def op(self, name: str, ins, out):
        live = {k: v[0] for k, v in self.held.items()}
        live.update((t[0], t[1]) for t in ins)
        live[out[0]] = out[1]
        self.terms.append((name, sum(live.values())))
        self.shapes[name] = out[2]
        return out

    def hold(self, t) -> None:
        self.held.setdefault(t[0], [t[1], 0])[1] += 1

    def drop(self, t) -> None:
        self.held[t[0]][1] -= 1
        if not self.held[t[0]][1]:
            del self.held[t[0]]


def _resnet(tp: _Live, name: str, x, cin: int, cout: int, h: int, w: int):
    r = x
    if cin != cout:
        r = tp.op(f"{name}.conv_shortcut", [x], tp.new(cout, h, w))
    tp.hold(r)
    a = tp.op(f"{name}.norm1", [x], tp.new(cin, h, w))
    a = tp.op(f"{name}.silu1", [a], tp.new(cin, h, w))
    a = tp.op(f"{name}.conv1", [a], tp.new(cout, h, w))
    a = tp.op(f"{name}.norm2", [a], tp.new(cout, h, w))
    a = tp.op(f"{name}.silu2", [a], tp.new(cout, h, w))
    a = tp.op(f"{name}.conv2", [a], tp.new(cout, h, w))
    tp.drop(r)
    return tp.op(f"{name}.add", [a, r], tp.new(cout, h, w))


def _mid(tp: _Live, name: str, x, c: int, h: int, w: int):
    x = _resnet(tp, f"{name}.resnets.0", x, c, c, h, w)
    att = f"{name}.attentions.0"
    tp.hold(x)
    a = tp.op(f"{att}.norm", [x], tp.new(c, h, w))
    qkv = tp.op(f"{att}.to_qkv", [a], tp.new(3 * c, h, w))
    n = h * w
    rows = min(n, max(1, ATTN_SCORE_BYTES // (tp.batch * n * 4)))
    scores = tp.raw(tp.batch * rows * n * tp.itemsize)
    o = tp.op(f"{att}.sdpa", [qkv, scores], tp.new(c, h, w))
    tp.softmax[f"{att}.sdpa"] = scores[1]
    a = tp.op(f"{att}.proj", [o], tp.new(c, h, w))
    tp.drop(x)
    x = tp.op(f"{att}.add", [x, a], tp.new(c, h, w))
    return _resnet(tp, f"{name}.resnets.1", x, c, c, h, w)


def _decode_head(tp: _Live, cfg, h: int, w: int):
    z, e = cfg.latent_channels, cfg.decoder_dims
    x = tp.new(z, h, w)
    x = tp.op("post_quant_conv", [x], tp.new(z, h, w))
    x = tp.op("decoder.conv_in", [x], tp.new(e[0], h, w))
    x = _mid(tp, "decoder.mid_block", x, e[0], h, w)
    return _up_blocks(tp, cfg, x, h, w, 0, whole_up_blocks(cfg))[0]


def _up_blocks(tp: _Live, cfg, x, h: int, w: int, lo: int, hi: int):
    """Up blocks ``lo`` to ``hi`` on ``x`` of ``h`` x ``w`` pixels: the
    result with its height and width."""
    e, n = cfg.decoder_dims, len(cfg.dim_mult)
    for i in range(lo, hi):
        blk = f"decoder.up_blocks.{i}"
        up = i < n - 1
        if up:
            tp.hold(x)
        a = x
        for j in range(cfg.num_res_blocks + 1):
            a = _resnet(tp, f"{blk}.resnets.{j}", a,
                        e[i] if j == 0 else e[i + 1], e[i + 1], h, w)
        if up:
            c = e[i + 1]
            a = tp.op(f"{blk}.upsampler.nearest", [a],
                      tp.new(c, 2 * h, 2 * w))
            a = tp.op(f"{blk}.upsampler.resample.1", [a],
                      tp.new(c, 2 * h, 2 * w))
            tp.hold(a)
            s = tp.op(f"{blk}.shortcut.take", [x], tp.new(c, 2 * h, 2 * w))
            tp.drop(x)
            s = tp.op(f"{blk}.shortcut.depth_to_space", [s],
                      tp.new(c, 2 * h, 2 * w))
            tp.drop(a)
            a = tp.op(f"{blk}.add", [a, s], tp.new(c, 2 * h, 2 * w))
            h, w = 2 * h, 2 * w
        x = a
    return x, h, w


def _decode_tail(tp: _Live, cfg, x, h: int, w: int):
    """The tiled part of a decode on ``x``, the whole part's output over
    ``h`` x ``w`` latent pixels."""
    e, n = cfg.decoder_dims, len(cfg.dim_mult)
    k = whole_up_blocks(cfg)
    x, h, w = _up_blocks(tp, cfg, x, h << k, w << k, k, n)
    c = e[-1]
    a = tp.op("decoder.norm_out", [x], tp.new(c, h, w))
    a = tp.op("decoder.silu_out", [a], tp.new(c, h, w))
    a = tp.op("decoder.conv_out", [a], tp.new(cfg.out_channels, h, w))
    return tp.op("clamp", [a], tp.new(cfg.out_channels, h, w, 4))


def _encode_head(tp: _Live, cfg, h: int, w: int):
    d, n = cfg.encoder_dims, len(cfg.dim_mult)
    x = tp.new(cfg.in_channels - 1, h, w, 4)
    x = tp.op("with_alpha", [x], tp.new(cfg.in_channels, h, w))
    x = tp.op("encoder.conv_in", [x], tp.new(d[0], h, w))
    for i in range(n):
        blk = f"encoder.down_blocks.{i}"
        down = i < n - 1
        cin, cout = d[i], d[i + 1]
        tp.hold(x)
        a = x
        for j in range(cfg.num_res_blocks):
            a = _resnet(tp, f"{blk}.resnets.{j}", a,
                        cin if j == 0 else cout, cout, h, w)
        if not down:
            # The shortcut is the block's input itself.
            tp.drop(x)
            x = tp.op(f"{blk}.add", [a, x], tp.new(cout, h, w))
            continue
        a = tp.op(f"{blk}.downsampler.pad", [a], tp.new(cout, h + 1, w + 1))
        a = tp.op(f"{blk}.downsampler.resample.1", [a],
                  tp.new(cout, h // 2, w // 2))
        tp.hold(a)
        s = tp.op(f"{blk}.shortcut.space_to_depth", [x], tp.new(cin, h, w))
        tp.drop(x)
        if cfg.temporal[i]:
            s = tp.op(f"{blk}.shortcut.zero_frame", [s],
                      tp.new(2 * cin, h, w))
        s = tp.op(f"{blk}.shortcut.mean", [s], tp.new(cout, h // 2, w // 2))
        tp.drop(a)
        x = tp.op(f"{blk}.add", [a, s], tp.new(cout, h // 2, w // 2))
        h, w = h // 2, w // 2
    return x


def _encode_tail(tp: _Live, cfg, x, h: int, w: int):
    c, z = cfg.encoder_dims[-1], cfg.latent_channels
    x = _mid(tp, "encoder.mid_block", x, c, h, w)
    a = tp.op("encoder.norm_out", [x], tp.new(c, h, w))
    a = tp.op("encoder.silu_out", [a], tp.new(c, h, w))
    a = tp.op("encoder.conv_out", [a], tp.new(2 * z, h, w))
    return tp.op("quant_conv", [a], tp.new(2 * z, h, w))


def _check(cfg, op: str, height: int, width: int) -> int:
    if op not in ("decode", "encode"):
        raise ValueError(f"op must be 'decode' or 'encode', not {op!r}")
    s = cfg.scale
    if height % s or width % s:
        raise ValueError(f"image size {height}x{width} is not a multiple "
                         f"of {s}")
    return s


def activation_terms(cfg: QwenImage21VAEConfig, op: str, height: int,
                     width: int, *, itemsize: int, batch: int = 1) -> list:
    """(op name, live bytes) for each op of a whole ``op``, "decode" or
    "encode", of ``batch`` images of ``height`` x ``width`` pixels at
    ``itemsize`` bytes per activation."""
    return _walk(cfg, op, height, width, itemsize, batch).terms


def _walk(cfg, op: str, height: int, width: int, itemsize: int,
          batch: int) -> _Live:
    s = _check(cfg, op, height, width)
    tp = _Live(itemsize, batch)
    if op == "decode":
        h, w = height // s, width // s
        _decode_tail(tp, cfg, _decode_head(tp, cfg, h, w), h, w)
    else:
        x = _encode_head(tp, cfg, height, width)
        _encode_tail(tp, cfg, x, height // s, width // s)
    return tp


def output_shapes(cfg: QwenImage21VAEConfig, op: str, height: int,
                  width: int) -> dict:
    """Op name to the (C, H, W) of its output in a whole pass."""
    return {k: v for k, v in _walk(cfg, op, height, width, 1, 1)
            .shapes.items() if v is not None}


def weight_count(cfg: QwenImage21VAEConfig) -> int:
    """Weights the module holds: every conv and gamma of the encoder, the
    decoder and the two convs between them."""
    def conv(cin, cout, k):
        return cout * cin * k * k + cout

    def resnet(cin, cout):
        return (cin + conv(cin, cout, 3) + cout + conv(cout, cout, 3)
                + (conv(cin, cout, 1) if cin != cout else 0))

    def mid(c):
        return 2 * resnet(c, c) + c + conv(c, 3 * c, 1) + conv(c, c, 1)

    d, e, n = cfg.encoder_dims, cfg.decoder_dims, len(cfg.dim_mult)
    z, res = cfg.latent_channels, cfg.num_res_blocks
    total = conv(cfg.in_channels, d[0], 3) + mid(d[-1]) + d[-1]
    total += conv(d[-1], 2 * z, 3) + conv(2 * z, 2 * z, 1) + conv(z, z, 1)
    total += conv(z, e[0], 3) + mid(e[0]) + e[-1]
    total += conv(e[-1], cfg.out_channels, 3)
    for i in range(n):
        total += sum(resnet(d[i] if j == 0 else d[i + 1], d[i + 1])
                     for j in range(res))
        total += sum(resnet(e[i] if j == 0 else e[i + 1], e[i + 1])
                     for j in range(res + 1))
        if i < n - 1:
            total += conv(d[i + 1], d[i + 1], 3) + conv(e[i + 1], e[i + 1], 3)
    return total


def memory_terms(cfg: QwenImage21VAEConfig, op: str, height: int,
                 width: int, *, itemsize: int, batch: int = 1,
                 tiled: bool = False, tile: int = 512,
                 overlap: int = 64) -> list:
    """(step name, device bytes, softmax bytes, host bytes) for each step
    of ``op``. The device bytes are the activations by op order. ``tile``
    and ``overlap`` are in pixels, as in ``tiling``. An image no larger
    than a tile runs whole."""
    from .tiling import tile_length

    s = _check(cfg, op, height, width)
    h, w = height // s, width // s
    th, tw = h, w
    if tiled:
        th = tile_length(h, tile // s, overlap // s)
        tw = tile_length(w, tile // s, overlap // s)
    if th == h and tw == w:
        tp = _walk(cfg, op, height, width, itemsize, batch)
        return [(k, v, tp.softmax.get(k, 0), 0) for k, v in tp.terms]
    tp, one = _Live(itemsize, batch), _Live(itemsize, batch)
    if op == "decode":
        head = _decode_head(tp, cfg, h, w)[1]
        steps = [(k, v, tp.softmax.get(k, 0), 0) for k, v in tp.terms]
        c = cfg.out_channels
        image = 4 * batch * height * width * c
        piece = 4 * batch * th * s * tw * s * c
        # A tile of the whole part's output: the channels its last up
        # block ends on, at that block's resolution.
        whole = whole_up_blocks(cfg)
        _decode_tail(one, cfg, one.new(cfg.decoder_dims[whole], th << whole,
                                       tw << whole), th, tw)
        # The host holds the result, and a tile while its kept part is
        # copied in.
        steps += [(f"tile {k}", v + head, 0, image) for k, v in one.terms]
        steps += [
            ("tile crop", head, 0, image + piece),
            ("image to the device", head + image, 0, image),
            ("clamp", head + 2 * image, 0, 0)]
        return steps
    c = cfg.encoder_dims[-1]
    image = 4 * batch * height * width * (cfg.in_channels - 1)
    latent, piece = 4 * batch * h * w * c, 4 * batch * th * tw * c
    blend = latent + 4 * h * w
    _encode_head(one, cfg, th * s, tw * s)
    steps = [(f"tile {k}", v + image, 0, blend) for k, v in one.terms]
    steps += [
        ("tile blend", image, 0, blend + 2 * piece),
        ("blend divide", image, 0, blend + latent),
        ("head to the device", image + latent, 0, latent)]
    _encode_tail(tp, cfg, tp.new(c, h, w), h, w)
    # The blended head stays on the host while the rest runs.
    return steps + [(k, v + image, tp.softmax.get(k, 0), latent)
                    for k, v in tp.terms]


def activation_peak(cfg: QwenImage21VAEConfig, op: str, height: int,
                    width: int, *, itemsize: int, batch: int = 1,
                    tiled: bool = False, tile: int = 512,
                    overlap: int = 64) -> tuple[int, str]:
    """The op-order floor: the largest activation bytes live on the device
    in ``op``, and the step that holds them."""
    steps = memory_terms(cfg, op, height, width, itemsize=itemsize,
                         batch=batch, tiled=tiled, tile=tile,
                         overlap=overlap)
    size, name = max((v, k) for k, v, _, _ in steps)
    return size, name


def memory_floor(cfg: QwenImage21VAEConfig, op: str, height: int,
                 width: int, *, itemsize: int, batch: int = 1,
                 tiled: bool = False, tile: int = 512,
                 overlap: int = 64) -> dict:
    """A floor under the bytes that ``op`` needs: the weights and the
    largest sum of the other terms over the steps of the pass. ``at`` is
    that step, ``op_order``, ``attention_softmax`` and ``host`` are its
    terms, and ``total`` is their sum with ``weights``. Whatever the lazy
    graph and MLX's cache keep comes on top."""
    steps = memory_terms(cfg, op, height, width, itemsize=itemsize,
                         batch=batch, tiled=tiled, tile=tile,
                         overlap=overlap)
    _, name, device, softmax, host = max(
        (v + a + n, k, v, a, n) for k, v, a, n in steps)
    weights = weight_count(cfg) * itemsize
    return {"at": name, "op_order": device, "attention_softmax": softmax,
            "host": host, "weights": weights,
            "total": device + softmax + host + weights}
