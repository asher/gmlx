"""Keyframe and guide images at the canvas size, as ComfyUI's MiniMax-H3
nodes resize them.

The first frame is stretched to the canvas. The last frame and each guide
frame are cropped about their center to the canvas's aspect, then resized.
Both resize with Lanczos from 8-bit pixels: ComfyUI's lanczos takes its
float image back to uint8, and float32(v / 255) x 255 truncates back to v
for every 8-bit v, so starting from the file's own uint8 pixels gives the
same input. The crop offset rounds as Python's round does.

``read`` takes an image file as the CLI reads it: Pillow opens it, turns
it by its EXIF orientation, and keeps its RGB channels. ComfyUI's
LoadImage decodes an image through PyAV, and Pillow only when PyAV cannot.
The pixels of an 8-bit PNG are the same either way. Those of a JPEG are
not, since the two decoders upsample its color planes differently: a few
levels apart on a smooth photo, and far more at sharp color edges.

Importing this module loads no mlx.
"""

from __future__ import annotations

import numpy as np


def read(path: str) -> np.ndarray:
    """The first frame of an image file, (H, W, 3) uint8, upright."""
    from PIL import Image, ImageOps

    with Image.open(path) as img:
        img.seek(0)
        return np.asarray(ImageOps.exif_transpose(img).convert("RGB"),
                          np.uint8).copy()


def _check(image: np.ndarray) -> None:
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"an image is (H, W, 3) uint8, not {image.dtype} "
                         f"{image.shape}")


def _lanczos(image: np.ndarray, width: int, height: int) -> np.ndarray:
    from PIL import Image

    out = Image.fromarray(image).resize((width, height),
                                        resample=Image.Resampling.LANCZOS)
    return np.asarray(out, np.uint8).copy()


def crop_box(old_w: int, old_h: int, width: int, height: int) -> tuple:
    """(x, y): the columns and rows cut from each side to bring an image
    of ``old_w`` x ``old_h`` to the aspect of ``width`` x ``height``, as
    comfy.utils.common_upscale's center crop takes them."""
    old_aspect = old_w / old_h
    new_aspect = width / height
    x = y = 0
    if old_aspect > new_aspect:
        x = round((old_w - old_w * (new_aspect / old_aspect)) / 2)
    elif old_aspect < new_aspect:
        y = round((old_h - old_h * (old_aspect / new_aspect)) / 2)
    return x, y


def stretch(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """``image`` resized to ``width`` x ``height``, its aspect let go, as
    the first frame is."""
    _check(image)
    return _lanczos(image, width, height)


def cover(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """``image`` cropped about its center to the aspect of ``width`` x
    ``height`` and resized to it, as the last frame and guide frames
    are."""
    _check(image)
    h, w = image.shape[:2]
    x, y = crop_box(w, h, width, height)
    return _lanczos(np.ascontiguousarray(image[y:h - y, x:w - x]), width,
                    height)
