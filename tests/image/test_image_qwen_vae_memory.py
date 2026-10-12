"""The memory arithmetic of the Qwen-Image 2.1 VAE at the widths of the
official header: single terms, the op-order peaks and the floor with its
named terms, whole and tiled at 1024 and 2048 pixels, against sums written
out here, and the walk's shapes against the modules of a tiny model. No
array at an official width is allocated. CPU."""
from __future__ import annotations

import math

import numpy as np
import pytest
from image_fixtures import load_json
from qwen_vae_fixtures import module_config, tiny

import mlx.core as mx
from mlx.utils import tree_flatten

from gmlx.image.vae import qwen_image21_memory as mem
from gmlx.image.vae import qwen_image21_vae as q

GiB = 1 << 30
WEIGHTS = 330987188


def _header() -> dict:
    head = load_json("qwen_image21_vae.header.json")
    return {n: tuple(s) for n, s, _ in head["tensors"]}


@pytest.fixture(scope="module")
def cfg():
    return q.config_from_shapes(_header())


def test_single_terms_are_the_tensors_live_at_the_op(cfg):
    px = 32 * 48
    for isz in (2, 4):
        enc = dict(mem.activation_terms(cfg, "encode", 32, 48, itemsize=isz))
        blk = "encoder.down_blocks.0"
        # The RGB image in float32 and the RGBA copy in the stream dtype.
        assert enc["with_alpha"] == px * (3 * 4 + 4 * isz)
        assert enc["encoder.conv_in"] == px * isz * (4 + 96)
        # A resnet holds its input until the sum, and the block holds its
        # own input until the shortcut.
        assert enc[f"{blk}.resnets.0.norm1"] == px * isz * (96 + 96)
        assert enc[f"{blk}.resnets.0.conv1"] == px * isz * (96 + 96 + 96)
        assert enc[f"{blk}.resnets.1.silu1"] == px * isz * 4 * 96
        assert enc[f"{blk}.downsampler.pad"] == isz * 96 * (
            2 * px + 33 * 49)
        # The block's input after space to depth, its mean and the
        # downsampled branch.
        assert enc[f"{blk}.shortcut.mean"] == isz * 96 * (px + px // 2)
        assert f"{blk}.shortcut.zero_frame" not in enc
        # The second block takes 96 channels to 192 at half resolution and
        # halves time in the video layout. Its first resnet holds the
        # block's input and the 192 of conv_shortcut, and silu1 reads and
        # writes 96.
        blk = "encoder.down_blocks.1"
        assert enc[f"{blk}.resnets.0.silu1"] == px // 4 * isz * (
            96 + 192 + 96 + 96)
        # The frame of zeros doubles the block's input beside it, and the
        # downsampled branch is held.
        assert enc[f"{blk}.shortcut.zero_frame"] == px // 4 * isz * (
            192 // 4 + 96 + 2 * 96)
        assert enc[f"{blk}.shortcut.mean"] == px // 4 * isz * (
            192 // 4 + 2 * 96 + 192 // 4)
        assert enc["quant_conv"] == 6 * isz * (128 + 128)

        dec = dict(mem.activation_terms(cfg, "decode", 32, 48, itemsize=isz))
        blk = "decoder.up_blocks.3"
        assert dec["post_quant_conv"] == 6 * isz * (64 + 64)
        assert dec["decoder.conv_in"] == 6 * isz * (64 + 1152)
        # 1152 channels in and 576 out at 4 times the latent resolution:
        # the block's input, the 576 of conv_shortcut, and the 1152 that
        # silu1 reads and writes.
        assert dec["decoder.up_blocks.2.resnets.0.silu1"] == px // 16 * isz * (
            1152 + 576 + 1152 + 1152)
        # 576 channels in at half resolution, 288 out at full resolution:
        # the block's input, the conv's input and its output.
        assert dec[f"{blk}.upsampler.resample.1"] == px * isz * (
            576 // 4 + 288 + 288)
        # The upsampled branch, the gathered channels and their copy in
        # image order.
        assert dec[f"{blk}.shortcut.depth_to_space"] == px * isz * 3 * 288
        assert dec[f"{blk}.add"] == px * isz * 3 * 288
        assert dec["decoder.conv_out"] == px * isz * (144 + 4)
        assert dec["clamp"] == px * 4 * (isz + 4)


def test_the_attention_term_holds_one_chunk_of_scores(cfg):
    # 128 x 128 latent pixels: 16384 rows of scores in chunks of 4096.
    n = 128 * 128
    dec = dict(mem.activation_terms(cfg, "decode", 2048, 2048, itemsize=2))
    assert dec["decoder.mid_block.attentions.0.sdpa"] == 2 * n * (
        1152 + 3 * 1152 + 4096 + 1152) == 322961408
    small = dict(mem.activation_terms(cfg, "decode", 64, 64, itemsize=2))
    assert small["decoder.mid_block.attentions.0.sdpa"] == 2 * 16 * (
        1152 + 3 * 1152 + 16 + 1152)


def _decode_tile(size: int, t: int, isz: int) -> tuple:
    """(device, softmax, host) bytes of a tiled decode with one tile at
    its peak."""
    px, lat = size * size, (size // 16) ** 2
    return (864 * t * t * isz           # one tile at its peak
            + 4 * 1152 * lat * isz,     # the first up block's output, held
            0,
            4 * px * 4)                 # the image the tiles are copied to


def _encode_tile(size: int, t: int, isz: int) -> tuple:
    px, lat = size * size, (size // 16) ** 2
    return (384 * t * t * isz           # one tile at its peak
            + 4 * px * 3,               # the image
            0,
            4 * lat * 768 + 4 * lat)    # the blended head and its weights


def _attention(c: int, isz: int, beside: int = 0, host: int = 0) -> tuple:
    """(device, softmax, host) bytes of the mid attention over the 16384
    latent pixels of a 2048 pixel image: the block's input, the 3 C of
    to_qkv, a chunk of 4096 rows of scores and the output, and the softmax
    of that chunk."""
    n = 16384
    return (isz * n * (5 * c + 4096) + beside, isz * n * 4096, host)


SDPA = "mid_block.attentions.0.sdpa"
UP = "decoder.up_blocks.3.shortcut.depth_to_space"
DOWN = "encoder.down_blocks.0.resnets.1.silu2"
IMAGE_2048, HEAD_2048 = 4 * 2048 * 2048 * 3, 4 * 16384 * 768

# (op, pixels a side, itemsize, whole op-order bytes, and for the tiled
# pass the op-order bytes and the floor's step with its device, softmax
# and host bytes). At the overlap of 64 pixels that these calls leave to
# its default, the tile is 384 pixels at 1024 and 464 at 2048. In a tiled
# encode at 2048 the mid attention, which runs whole, holds more than a
# tile beside the image and the blended head.
CELLS = [
    ("decode", 1024, 2, 864 * 1024 * 1024 * 2,
     _decode_tile(1024, 384, 2)[0], f"tile {UP}", _decode_tile(1024, 384, 2)),
    ("decode", 1024, 4, 864 * 1024 * 1024 * 4,
     _decode_tile(1024, 384, 4)[0], f"tile {UP}", _decode_tile(1024, 384, 4)),
    ("decode", 2048, 2, 864 * 2048 * 2048 * 2,
     _decode_tile(2048, 464, 2)[0], f"tile {UP}", _decode_tile(2048, 464, 2)),
    ("decode", 2048, 4, 864 * 2048 * 2048 * 4,
     _decode_tile(2048, 464, 4)[0], f"tile {UP}", _decode_tile(2048, 464, 4)),
    ("encode", 1024, 2, 384 * 1024 * 1024 * 2,
     _encode_tile(1024, 384, 2)[0], f"tile {DOWN}",
     _encode_tile(1024, 384, 2)),
    ("encode", 1024, 4, 384 * 1024 * 1024 * 4,
     _encode_tile(1024, 384, 4)[0], f"tile {DOWN}",
     _encode_tile(1024, 384, 4)),
    ("encode", 2048, 2, 384 * 2048 * 2048 * 2,
     _attention(768, 2, IMAGE_2048)[0], f"encoder.{SDPA}",
     _attention(768, 2, IMAGE_2048, HEAD_2048)),
    ("encode", 2048, 4, 384 * 2048 * 2048 * 4,
     _attention(768, 4, IMAGE_2048)[0], f"encoder.{SDPA}",
     _attention(768, 4, IMAGE_2048, HEAD_2048)),
]


@pytest.mark.parametrize("op,size,isz,whole,tiled,at,terms", CELLS)
def test_the_floor_at_the_official_widths(cfg, op, size, isz, whole, tiled,
                                          at, terms):
    step = UP if op == "decode" else DOWN
    assert mem.activation_peak(cfg, op, size, size, itemsize=isz) == (
        whole, step)
    assert mem.memory_floor(cfg, op, size, size, itemsize=isz) == {
        "at": step, "op_order": whole, "attention_softmax": 0, "host": 0,
        "weights": WEIGHTS * isz, "total": whole + WEIGHTS * isz}
    got, name = mem.activation_peak(cfg, op, size, size, itemsize=isz,
                                    tiled=True)
    assert got == tiled
    floor = mem.memory_floor(cfg, op, size, size, itemsize=isz, tiled=True)
    device, softmax, host = terms
    assert floor == {
        "at": at, "op_order": device, "attention_softmax": softmax,
        "host": host, "weights": WEIGHTS * isz,
        "total": device + softmax + host + WEIGHTS * isz}
    print(f"[memory] {op} {size} itemsize {isz}: whole op order "
          f"{whole / GiB:.3f} GiB, floor {(whole + WEIGHTS * isz) / GiB:.3f}; "
          f"tiled op order {tiled / GiB:.3f} GiB at {name}, floor "
          f"{floor['total'] / GiB:.3f} at {at}")


def test_the_table_in_gib(cfg):
    def gib(op, size, isz, tiled):
        f = mem.memory_floor(cfg, op, size, size, itemsize=isz, tiled=tiled)
        return round(f["total"] / GiB, 3)

    table = {(op, size, isz): (round(whole / GiB, 3),
                               gib(op, size, isz, False),
                               round(tiled / GiB, 3),
                               gib(op, size, isz, True))
             for op, size, isz, whole, tiled, _, _ in CELLS}
    # Whole: the op-order peak and the floor. Tiled: the same two.
    assert table == {
        ("decode", 1024, 2): (1.688, 2.304, 0.272, 0.905),
        ("decode", 1024, 4): (3.375, 4.608, 0.545, 1.794),
        ("decode", 2048, 2): (6.750, 7.367, 0.487, 1.166),
        ("decode", 2048, 4): (13.500, 14.733, 0.974, 2.270),
        ("encode", 1024, 2): (0.750, 1.367, 0.117, 0.745),
        ("encode", 1024, 4): (1.500, 2.733, 0.223, 1.467),
        ("encode", 2048, 2): (3.000, 3.617, 0.289, 1.077),
        ("encode", 2048, 4): (6.000, 7.233, 0.531, 2.061),
    }
    assert round(WEIGHTS * 2 / GiB, 3) == 0.617
    assert round(WEIGHTS * 4 / GiB, 3) == 1.233


def test_the_weights_are_every_parameter_of_the_module(cfg):
    held = {n: s for n, s in _header().items() if ".time_conv." not in n}
    assert len(held) == 226
    assert mem.weight_count(cfg) == sum(
        math.prod(s) for s in held.values()) == WEIGHTS
    with mx.stream(mx.cpu):
        for name in ("tiny", *tiny.CASES):
            config = tiny.TINY if name == "tiny" else tiny.case_config(name)
            small = module_config(config)
            want = sum(math.prod(s) for n, s in tiny.shapes(config).items()
                       if ".time_conv." not in n)
            assert mem.weight_count(small) == want, name
            assert want == sum(v.size for _, v in tree_flatten(
                q.QwenImage21VAE(small).parameters())), name


def test_the_attention_softmax_is_a_second_chunk_of_scores(cfg):
    for op, side, c in (("decode", "decoder", 1152),
                        ("encode", "encoder", 768)):
        for size, n, rows in ((2048, 16384, 4096), (1024, 4096, 4096),
                              (64, 16, 16)):
            for isz in (2, 4):
                steps = {k: (v, a, h) for k, v, a, h in mem.memory_terms(
                    cfg, op, size, size, itemsize=isz)}
                device, softmax, host = steps[f"{side}.{SDPA}"]
                assert device == isz * n * (5 * c + rows)
                assert softmax == isz * n * rows and host == 0
                assert sum(a for _, a, _ in steps.values()) == softmax
                assert not any(h for _, _, h in steps.values())
                assert [(k, v) for k, (v, _, _) in steps.items()] == \
                    mem.activation_terms(cfg, op, size, size, itemsize=isz)
    # Two images halve the rows of a chunk and not its bytes.
    two = {k: a for k, _, a, _ in mem.memory_terms(
        cfg, "decode", 2048, 2048, itemsize=2, batch=2)}
    assert two[f"decoder.{SDPA}"] == 2 * 2 * 16384 * 2048


def test_the_steps_of_a_tiled_pass_after_its_tiles():
    # The tiny widths, a 256 pixel image and 32 pixel tiles, so the image
    # is larger than a tile.
    small = module_config(tiny.TINY)
    for isz in (2, 4):
        # The whole part ends on the 48 channels of the first up block at
        # 2 pixels a latent pixel.
        head = 48 * 32 * 32 * isz
        image, piece = 4 * 256 * 256 * 4, 4 * 32 * 32 * 4
        kw = dict(itemsize=isz, tiled=True, tile=32, overlap=16)
        steps = mem.memory_terms(small, "decode", 256, 256, **kw)
        # A decode crops its tiles: the host holds the image and one tile.
        assert steps[-3:] == [
            ("tile crop", head, 0, image + piece),
            ("image to the device", head + image, 0, image),
            ("clamp", head + 2 * image, 0, 0)]
        whole = [s[0] for s in steps if not s[0].startswith("tile ")][:-2]
        assert whole[0] == "post_quant_conv"
        assert whole[-1] == "decoder.up_blocks.0.add"
        assert not any(k.startswith("decoder.up_blocks.1") for k in whole)
        tiles = [s for s in steps if s[0].startswith("tile decoder.")]
        assert tiles[0][0] == "tile decoder.up_blocks.1.resnets.0.norm1"
        assert tiles and all(s[3] == image for s in tiles)
        assert max(s[1] for s in tiles) == 36 * 32 * 32 * isz + head
        floor = mem.memory_floor(small, "decode", 256, 256, **kw)
        assert (floor["at"], floor["op_order"], floor["host"]) == (
            "image to the device", head + image, image)
        assert mem.activation_peak(small, "decode", 256, 256, **kw) == (
            head + 2 * image, "clamp")

        image, latent = 4 * 256 * 256 * 3, 4 * 16 * 16 * 32
        piece, blend = 4 * 2 * 2 * 32, latent + 4 * 16 * 16
        steps = mem.memory_terms(small, "encode", 256, 256, **kw)
        at = [s[0] for s in steps].index("tile blend")
        assert steps[at:at + 3] == [
            ("tile blend", image, 0, blend + 2 * piece),
            ("blend divide", image, 0, blend + latent),
            ("head to the device", image + latent, 0, latent)]
        assert steps[0][0] == "tile with_alpha"
        assert all(s[0].startswith("tile ") and s[3] == blend
                   for s in steps[:at])
        # The rest runs whole beside the image, with the blended head on
        # the host.
        rest = steps[at + 3:]
        whole = mem.activation_terms(small, "encode", 256, 256, itemsize=isz)
        tail = [k for k, _ in whole].index("encoder.mid_block.resnets.0.norm1")
        assert [s[0] for s in rest] == [k for k, _ in whole[tail:]]
        assert all(s[3] == latent for s in rest)
        assert rest[-1][:2] == ("quant_conv", 256 * isz * 256 + image)


def test_a_tiled_decode_at_2048_by_tile_size(cfg):
    def peak(tile, isz=2):
        return mem.activation_peak(cfg, "decode", 2048, 2048, itemsize=isz,
                                   tiled=True, tile=tile)

    def floor(tile, isz=2):
        f = mem.memory_floor(cfg, "decode", 2048, 2048, itemsize=isz,
                             tiled=True, tile=tile)
        return f["at"], f["total"] - f["weights"]

    # Under 512 pixel tiles the part that runs whole holds the most, and
    # a smaller tile lowers nothing. On the device that is the first up
    # block: its upsampled branch, the channels its shortcut gathers and
    # their copy in image order, each 1152 channels at 2 pixels a latent
    # pixel. With the softmax counted it is the mid attention.
    attn = f"decoder.{SDPA}"
    first = "decoder.up_blocks.0.shortcut.depth_to_space"
    assert peak(256) == peak(128) == peak(384) == (
        3 * 4 * 1152 * 16384 * 2, first)
    assert _decode_tile(2048, 352, 2)[0] < 452984832 == peak(256)[0]
    assert floor(256) == floor(384) == (attn, 322961408 + 2 * 16384 * 4096)
    # Seven 352 pixel tiles a side at 384, five 464 pixel tiles at 512.
    assert peak(512) == (_decode_tile(2048, 464, 2)[0], f"tile {UP}")
    assert peak(512)[0] == 523026432
    assert floor(512) == (f"tile {UP}", 523026432 + 4 * 2048 * 2048 * 4)
    # Tiles of at most 768 and of at most 1024 pixels both lay out three
    # 736 pixel tiles a side.
    assert peak(768) == peak(1024)
    assert peak(768)[0] == _decode_tile(2048, 736, 2)[0]
    sizes = [peak(t)[0] for t in (256, 384, 512, 768)]
    assert sizes == sorted(sizes)
    assert peak(2048) == mem.activation_peak(cfg, "decode", 2048, 2048,
                                             itemsize=2)


def test_peaks_scale_with_the_batch_and_run_whole_inside_one_tile(cfg):
    one = mem.activation_peak(cfg, "decode", 1024, 1024, itemsize=2)
    two = mem.activation_peak(cfg, "decode", 1024, 1024, itemsize=2,
                              batch=2)
    assert two == (2 * one[0], one[1])
    f1 = mem.memory_floor(cfg, "decode", 1024, 1024, itemsize=2)
    f2 = mem.memory_floor(cfg, "decode", 1024, 1024, itemsize=2, batch=2)
    assert f2["weights"] == f1["weights"]
    assert f2["total"] == 2 * f1["op_order"] + f1["weights"]
    for op in ("decode", "encode"):
        for fn in (mem.activation_peak, mem.memory_floor, mem.memory_terms):
            assert fn(cfg, op, 512, 384, itemsize=2, tiled=True) == fn(
                cfg, op, 512, 384, itemsize=2)
    wide = mem.memory_floor(cfg, "decode", 512, 1024, itemsize=2, tiled=True)
    assert wide["op_order"] == 864 * 512 * 384 * 2 + 4608 * 32 * 64 * 2
    assert wide["host"] == 4 * 512 * 1024 * 4
    for fn in (mem.activation_peak, mem.memory_floor, mem.memory_terms):
        with pytest.raises(ValueError, match="'decode' or 'encode'"):
            fn(cfg, "both", 512, 512, itemsize=2)
        with pytest.raises(ValueError,
                           match="1000x1024 is not a multiple of 16"):
            fn(cfg, "decode", 1000, 1024, itemsize=2)
        with pytest.raises(ValueError,
                           match="1024x1000 is not a multiple of 16"):
            fn(cfg, "encode", 1024, 1000, itemsize=2)
    with pytest.raises(ValueError, match="1024x1000 is not a multiple of 16"):
        mem.activation_terms(cfg, "decode", 1024, 1000, itemsize=2)


def test_the_walk_has_the_shapes_of_the_modules():
    with mx.stream(mx.cpu):
        small = q.config_from_shapes(tiny.shapes(tiny.TINY))
        vae = q.QwenImage21VAE(small)

        def chw(x):
            return (x.shape[3], x.shape[1], x.shape[2])

        want = mem.output_shapes(small, "decode", 32, 48)
        x = vae.post_quant_conv(mx.zeros((1, 2, 3, 64)))
        assert chw(x) == want["post_quant_conv"]
        x = vae.decoder.conv_in(x)
        assert chw(x) == want["decoder.conv_in"]
        x = vae.decoder.mid_block(x)
        assert chw(x) == want["decoder.mid_block.resnets.1.add"]
        for i, blk in enumerate(vae.decoder.up_blocks):
            up = blk.upsampler is not None
            last = (f"decoder.up_blocks.{i}.add" if up
                    else f"decoder.up_blocks.{i}.resnets.2.add")
            h = x
            for j, r in enumerate(blk.resnets):
                h = r(h)
                assert chw(h) == want[f"decoder.up_blocks.{i}.resnets.{j}.add"]
            if up:
                assert chw(blk.upsampler(h)) == want[
                    f"decoder.up_blocks.{i}.upsampler.resample.1"]
            x = blk(x)
            assert chw(x) == want[last]
        assert chw(vae.decode(mx.zeros((1, 2, 3, 64)))) == want["clamp"] \
            == (4, 32, 48)

        want = mem.output_shapes(small, "encode", 32, 48)
        x = vae.with_alpha(mx.zeros((1, 32, 48, 3)))
        assert chw(x) == want["with_alpha"]
        x = vae.encoder.conv_in(x)
        assert chw(x) == want["encoder.conv_in"]
        for i, blk in enumerate(vae.encoder.down_blocks):
            if blk.downsampler is not None:
                h = x
                for r in blk.resnets:
                    h = r(h)
                pad = blk.downsampler.resample[0](h)
                assert chw(pad) == want[
                    f"encoder.down_blocks.{i}.downsampler.pad"]
                assert chw(blk.downsampler(h)) == want[
                    f"encoder.down_blocks.{i}.downsampler.resample.1"]
            x = blk(x)
            assert chw(x) == want[f"encoder.down_blocks.{i}.add"]
        x = vae.encode_tail(x)
        assert chw(x) == want["quant_conv"] == (128, 2, 3)
        names = [k for k, _ in mem.activation_terms(small, "encode", 32, 48,
                                                    itemsize=4)]
        assert len(names) == len(set(names))
        assert np.all(np.array([v for _, v in mem.activation_terms(
            small, "decode", 32, 48, itemsize=4)]) > 0)
