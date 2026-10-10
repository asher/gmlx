# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Asher Feldman
"""Memory of a Qwen-Image VAE pass, by arithmetic: a floor.

The walk and its terms are those of ``qwen_image21_memory``, over the ops
of ``qwen_image_vae``: no block shortcut, an encoder of resnets and
downsamplers in one list, upsamplers that halve the channels, and an RGB
image. ``memory_terms`` gives (step, device bytes, softmax bytes, host
bytes) for a whole pass or a tiled one, which holds one tile at a time
beside the part run whole and crops its tiles.
"""

from __future__ import annotations

from .qwen_image21_memory import _check, _Live, _mid, _resnet
from .qwen_image_vae import (CONSTANTS, IMAGE_CHANNELS, LATENT_CHANNELS,
                             QwenImageVAEConfig, encoder_layout,
                             whole_up_blocks)

# The config of the official file, which the memory plan prices before a
# file is read.
OFFICIAL = QwenImageVAEConfig(
    in_channels=IMAGE_CHANNELS, out_channels=IMAGE_CHANNELS,
    latent_channels=LATENT_CHANNELS, base_dim=96, dim_mult=(1, 2, 4, 4),
    num_res_blocks=2, temporal=CONSTANTS["temperal_downsample"])


def _up_blocks(tp: _Live, cfg, x, h: int, w: int, lo: int, hi: int):
    e, n = cfg.decoder_dims, len(cfg.dim_mult)
    for i in range(lo, hi):
        blk = f"decoder.up_blocks.{i}"
        for j in range(cfg.num_res_blocks + 1):
            x = _resnet(tp, f"{blk}.resnets.{j}", x,
                        cfg.up_in(i) if j == 0 else e[i + 1], e[i + 1], h, w)
        if i < n - 1:
            c = e[i + 1]
            x = tp.op(f"{blk}.upsamplers.0.nearest", [x],
                      tp.new(c, 2 * h, 2 * w))
            h, w = 2 * h, 2 * w
            x = tp.op(f"{blk}.upsamplers.0.resample.1", [x],
                      tp.new(c // 2, h, w))
    return x, h, w


def _decode_head(tp: _Live, cfg, h: int, w: int):
    z, e = cfg.latent_channels, cfg.decoder_dims
    x = tp.new(z, h, w)
    x = tp.op("post_quant_conv", [x], tp.new(z, h, w))
    x = tp.op("decoder.conv_in", [x], tp.new(e[0], h, w))
    x = _mid(tp, "decoder.mid_block", x, e[0], h, w)
    return _up_blocks(tp, cfg, x, h, w, 0, whole_up_blocks(cfg))[0]


def _decode_tail(tp: _Live, cfg, x, h: int, w: int):
    e, n = cfg.decoder_dims, len(cfg.dim_mult)
    k = whole_up_blocks(cfg)
    x, h, w = _up_blocks(tp, cfg, x, h << k, w << k, k, n)
    c = e[-1]
    a = tp.op("decoder.norm_out", [x], tp.new(c, h, w))
    a = tp.op("decoder.silu_out", [a], tp.new(c, h, w))
    a = tp.op("decoder.conv_out", [a], tp.new(cfg.out_channels, h, w))
    return tp.op("clamp", [a], tp.new(cfg.out_channels, h, w, 4))


def _encode_head(tp: _Live, cfg, h: int, w: int):
    d = cfg.encoder_dims
    x = tp.new(cfg.in_channels, h, w, 4)
    x = tp.op("cast", [x], tp.new(cfg.in_channels, h, w))
    x = tp.op("encoder.conv_in", [x], tp.new(d[0], h, w))
    for k, row in enumerate(encoder_layout(cfg)):
        name = f"encoder.down_blocks.{k}"
        if row[0] == "res":
            x = _resnet(tp, name, x, row[1], row[2], h, w)
            continue
        c = row[1]
        x = tp.op(f"{name}.pad", [x], tp.new(c, h + 1, w + 1))
        h, w = h // 2, w // 2
        x = tp.op(f"{name}.resample.1", [x], tp.new(c, h, w))
    return x


def _encode_tail(tp: _Live, cfg, x, h: int, w: int):
    c, z = cfg.encoder_dims[-1], cfg.latent_channels
    x = _mid(tp, "encoder.mid_block", x, c, h, w)
    a = tp.op("encoder.norm_out", [x], tp.new(c, h, w))
    a = tp.op("encoder.silu_out", [a], tp.new(c, h, w))
    a = tp.op("encoder.conv_out", [a], tp.new(2 * z, h, w))
    return tp.op("quant_conv", [a], tp.new(2 * z, h, w))


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


def output_shapes(cfg: QwenImageVAEConfig, op: str, height: int,
                  width: int) -> dict:
    """Op name to the (C, H, W) of its output in a whole pass."""
    return {k: v for k, v in _walk(cfg, op, height, width, 1, 1)
            .shapes.items() if v is not None}


def weight_count(cfg: QwenImageVAEConfig) -> int:
    """Weights the module holds, the time convs it leaves unread aside."""
    def conv(cin, cout, k):
        return cout * cin * k * k + cout

    def resnet(cin, cout):
        return (cin + conv(cin, cout, 3) + cout + conv(cout, cout, 3)
                + (conv(cin, cout, 1) if cin != cout else 0))

    def mid(c):
        return 2 * resnet(c, c) + c + conv(c, 3 * c, 1) + conv(c, c, 1)

    d, e, n = cfg.encoder_dims, cfg.decoder_dims, len(cfg.dim_mult)
    z = cfg.latent_channels
    total = conv(cfg.in_channels, d[0], 3) + mid(d[-1]) + d[-1]
    total += conv(d[-1], 2 * z, 3) + conv(2 * z, 2 * z, 1) + conv(z, z, 1)
    total += conv(z, e[0], 3) + mid(e[0]) + e[-1]
    total += conv(e[-1], cfg.out_channels, 3)
    for row in encoder_layout(cfg):
        total += (resnet(row[1], row[2]) if row[0] == "res"
                  else conv(row[1], row[1], 3))
    for i in range(n):
        total += sum(resnet(cfg.up_in(i) if j == 0 else e[i + 1], e[i + 1])
                     for j in range(cfg.num_res_blocks + 1))
        if i < n - 1:
            total += conv(e[i + 1], e[i + 1] // 2, 3)
    return total


def memory_terms(cfg: QwenImageVAEConfig, op: str, height: int,
                 width: int, *, itemsize: int, batch: int = 1,
                 tiled: bool = False, tile: int = 512,
                 overlap: int = 64) -> list:
    """(step name, device bytes, softmax bytes, host bytes) for each step
    of ``op``, as ``qwen_image21_memory.memory_terms`` gives them."""
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
        whole = whole_up_blocks(cfg)
        _decode_tail(one, cfg, one.new(cfg.up_in(whole), th << whole,
                                       tw << whole), th, tw)
        steps += [(f"tile {k}", v + head, 0, image) for k, v in one.terms]
        steps += [
            ("tile crop", head, 0, image + piece),
            ("image to the device", head + image, 0, image),
            ("clamp", head + 2 * image, 0, 0)]
        return steps
    c = cfg.encoder_dims[-1]
    image = 4 * batch * height * width * cfg.in_channels
    latent, piece = 4 * batch * h * w * c, 4 * batch * th * tw * c
    _encode_head(one, cfg, th * s, tw * s)
    steps = [(f"tile {k}", v + image, 0, latent) for k, v in one.terms]
    steps += [
        ("tile crop", image, 0, latent + piece),
        ("head to the device", image + latent, 0, latent)]
    _encode_tail(tp, cfg, tp.new(c, h, w), h, w)
    return steps + [(k, v + image, tp.softmax.get(k, 0), latent)
                    for k, v in tp.terms]
