"""The size limits of the image families: MiniMax-H3-Image's own pixels,
side and shapes, the refusal that names the smallest size of a shape under
its least pixels, and the fitted and edit sizes each family scales into.
Every other family keeps the image limits. CPU, no mlx arrays."""
from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import pytest

from gmlx.image import registry
from gmlx.image.registry import (MAX_PIXELS, MAX_SIDE, edit_size, families,
                                 fit_size, get_family, size_refusal)

H3 = get_family("h3_image")


def test_only_h3_image_sets_limits_of_its_own():
    assert (H3.max_pixels, H3.max_side, H3.min_pixels, H3.aspect) == (
        20_000_000, 7712, 655_360, ((1, 3), (3, 1)))
    assert registry.size_limits(H3) == (20_000_000, 7712)
    for fam in families():
        if fam is H3:
            continue
        assert not registry.limited(fam), fam.name
        assert registry.size_limits(fam) == (MAX_PIXELS, MAX_SIDE)
        assert size_refusal(fam, 8192, 8192) is None
        assert size_refusal(fam, 16, 16) is None


@pytest.mark.parametrize("size", [
    (7712, 2592), (2592, 7712), (7680, 2560), (4096, 4096), (4448, 4448),
    (2048, 2048), (1024, 640), (832, 832)])
def test_h3_image_takes_the_sizes_in_its_limits(size):
    assert size_refusal(H3, *size) is None


@pytest.mark.parametrize("size, why", [
    ((7744, 2592), "7744x2592 is 20,072,448 pixels, over the 20,000,000 "
                   "that MiniMax-H3-Image takes"),
    ((4480, 4480), "4480x4480 is 20,070,400 pixels, over the 20,000,000"),
    ((8000, 2000), "8000x2000 has a side over the 7712 pixels that "
                   "MiniMax-H3-Image takes"),
    ((1024, 320), "1024x320 is outside the shapes MiniMax-H3-Image takes, "
                  "1:3 to 3:1"),
    ((320, 1024), "outside the shapes"),
    ((512, 512), "512x512 is 262,144 pixels, under the 655,360 that "
                 "MiniMax-H3-Image takes; the smallest size of its shape "
                 "is 832x832"),
    ((800, 800), "the smallest size of its shape is 832x832"),
    ((1056, 352), "the smallest size of its shape is 1376x480"),
])
def test_h3_image_refuses_a_size_past_its_limits_by_name(size, why):
    assert why in size_refusal(H3, *size)


def test_the_smallest_size_named_is_taken_and_one_step_less_is_not():
    for w, h in ((512, 512), (1024, 512), (480, 1440), (608, 352)):
        sw, sh = registry.smallest_size(H3, w, h)
        assert sw % 32 == 0 and sh % 32 == 0
        assert size_refusal(H3, sw, sh) is None, (w, h)
        assert (sw - 32) * (sh - 32) < H3.min_pixels, (w, h)


def test_a_fitted_size_scales_down_only_and_keeps_inside_the_shapes():
    # Exactly 3:1 floored at the side would leave 7712x2560, past 3:1.
    assert fit_size(9000, 3000, 32, H3) == (7680, 2560)
    assert fit_size(3000, 9000, 32, H3) == (2560, 7680)
    assert fit_size(9000, 9000, 32, H3) == (4448, 4448)
    assert fit_size(12000, 4100, 32, H3) == (7648, 2592)
    assert fit_size(1024, 1024, 32, H3) == (1024, 1024)
    assert fit_size(512, 512, 32, H3) == (512, 512)
    # Without a limited family, the image limits, as before.
    assert fit_size(8192, 128, 16) == (4096, 64)
    assert fit_size(4096, 4096, 16, get_family("z_image")) == (2048, 2048)


def _sized(size):
    return dataclasses.replace(H3, edit=SimpleNamespace(
        size=lambda refs: size))


def test_an_edit_size_is_scaled_into_the_family_limits_or_refused():
    refs = [object()]
    assert edit_size(_sized((1024, 1024)), refs, 32) == (1024, 1024)
    assert edit_size(_sized((6000, 6000)), refs, 32) == (4448, 4448)
    assert edit_size(_sized((9000, 3000)), refs, 32) == (7680, 2560)
    assert edit_size(_sized((1024, 1024)), [], 32) is None
    with pytest.raises(ValueError, match="the first reference sizes the edit "
                       "at 4000x1000: 4000x1000 is outside the shapes "
                       "MiniMax-H3-Image takes, 1:3 to 3:1; give the edit "
                       "a size"):
        edit_size(_sized((4000, 1000)), refs, 32)
    with pytest.raises(ValueError, match="the smallest size of its shape is "
                       "832x832; give the edit a size"):
        edit_size(_sized((512, 512)), refs, 32)
