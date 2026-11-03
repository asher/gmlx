"""The FLUX.2 klein single block's concatenate and the FLUX.1 VAE's
downsample pad at outputs past 2^31 elements.

On a GPU stream of mlx 0.32.3 a pad or a concatenate on a later axis
misplaces input elements past 2^31 output elements (docs/upstream-defects.md).
SingleBlock.out and Downsample go through gmlx.mlx_guards. Each test runs
the call at the first shape past the limit, bfloat16, with the product or
the conv after it replaced by a capture of its input, and checks every
element of that input and of the call's own inputs against the formula of
tests/past_int32.py.
"""
from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx

import past_int32 as p32
from gmlx.image.families.flux2_klein.model import SingleBlock
from gmlx.image.vae.flux1_ae import Downsample

pytestmark = p32.MARKS
# The fixture MARKS uses, where pytest finds it.
settled = p32.settled


def _single_block(dim: int, mlp: int, length: int) -> tuple:
    """SingleBlock.out over ``length`` rows of a block of width ``dim``
    and feed-forward width ``mlp``: the shape linear2 gets, the wrong
    elements of its last row, of all its rows, and of rows and hid."""
    rows = p32.bf16((1, length, dim))
    hid = p32.bf16((1, length, mlp), salt=77)
    seen = {}

    def linear2(y):
        seen["y"] = y
        return mx.zeros((1, 1, dim), mx.float32)

    # A float32 residual and gate keep the call off the fused kernel, so
    # linear2 gets the concatenate's output.
    x = mx.zeros((1, 1, dim), mx.float32)
    mx.eval(SingleBlock.out(SimpleNamespace(linear2=linear2), x, x, rows, hid))
    y = seen.pop("y")

    def joined(r, s0, n):
        return mx.concatenate([p32.pattern(0, 1, s0, n, dim)[0],
                               p32.pattern(0, 1, s0, n, mlp, 77)[0]], axis=1)

    width = dim + mlp
    last = p32.wrong(y[:, length - 1:], (1, 1, width),
                     lambda r, s0, n: joined(r, length - 1, n))
    whole = p32.wrong(y, (1, length, width), joined)
    own = {**p32.wrong(rows, (1, length, dim),
                       lambda r, s0, n: p32.pattern(0, 1, s0, n, dim)[0]),
           **{("hid", *k): v for k, v in p32.wrong(
               hid, (1, length, mlp),
               lambda r, s0, n: p32.pattern(0, 1, s0, n, mlp, 77)[0]).items()}}
    return tuple(y.shape), last, whole, own


def test_a_single_block_past_2_31_elements_keeps_every_row():
    """FLUX.2 klein 9B joins attention rows of 4096 and feed-forward rows
    of 12,288. At 131,073 rows, the first count past the limit, the stock
    concatenate left the last row unwritten and wrote it 8 GiB below, over
    the start of the attention rows."""
    shape, last, whole, own = _single_block(4096, 12_288, 131_073)
    assert shape == (1, 131_073, 16_384)
    # {(row, first token of a span): wrong elements}, all three at once.
    assert {"last row": last, "joined": whole, "inputs": own} == {
        "last row": {}, "joined": {}, "inputs": {}}


C = 128


def _image_rows(h: int, w: int, valid_h: int, valid_w: int):
    """expect(r, s0, n) of an image (1, h, w, C) seen as one row of h
    tokens of w x C: the pattern inside valid_h x valid_w, zeros
    elsewhere."""
    def expect(r, h0, n):
        p = p32.pattern(h0, n, 0, w, C)
        hh = mx.arange(h0, h0 + n)[:, None, None]
        ww = mx.arange(w)[None, :, None]
        keep = (hh < valid_h) & (ww < valid_w)
        return mx.where(keep, p, mx.zeros_like(p)).reshape(n, w * C)
    return expect


def _downsample(h: int, w: int, w0: int) -> tuple:
    """Downsample over an (1, h, w, 128) activation: the padded shape, the
    wrong elements of its last image row from column w0, of the whole pad
    and of x."""
    x = p32.bf16((1, h, w, C))
    out = Downsample.__call__(SimpleNamespace(conv=lambda y: y), x)
    mx.eval(out)
    last = p32.wrong(out[:, h - 1:h, w0:w], (1, w - w0, C),
                     lambda r, s0, n: p32.pattern(h - 1, 1, w0 + s0, n, C)[0])
    chunk = 1 << 26
    whole = p32.wrong(out, (1, h + 1, (w + 1) * C), _image_rows(h + 1, w + 1, h, w), chunk)
    own = p32.wrong(x, (1, h, w * C), _image_rows(h, w, h, w), chunk)
    return tuple(out.shape), last, whole, own


def test_a_vae_downsample_past_2_31_elements_keeps_every_pixel():
    """The FLUX.1 VAE encoder pads 128 channels of a 4080x4112 canvas on
    the bottom and the right. x holds fewer than INT32_MAX elements and
    its place in the output passes INT32_MAX. The stock pad left the last
    image row from column 289 at the fill and wrote it 8 GiB below, over
    the start of x."""
    shape, last, whole, own = _downsample(4080, 4112, 289)
    assert shape == (1, 4081, 4113, C)
    assert {"region": last, "padded": whole, "x": own} == {
        "region": {}, "padded": {}, "x": {}}
