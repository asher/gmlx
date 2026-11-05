"""Keyframe and guide images at the canvas size, as ComfyUI's MiniMax-H3
nodes resize them.

The first frame is stretched to the canvas. The last frame and each guide
frame are cropped about their center to the canvas's aspect, then resized.
Both resize with Lanczos from 8-bit pixels: ComfyUI's lanczos takes its
float image back to uint8, and float32(v / 255) x 255 truncates back to v
for every 8-bit v, so starting from the file's own uint8 pixels gives the
same input. The crop offset rounds as Python's round does.

``read`` takes an image file as the CLI reads it. ComfyUI's LoadImage
decodes an image through PyAV, which is FFmpeg's decoder, and Pillow only
when PyAV cannot. With ffmpeg on PATH, ``read`` decodes through it too,
so a JPEG's pixels are LoadImage's; else Pillow decodes, and a JPEG can
come out a few levels apart on a photo, more at sharp color edges, since
the two decoders upsample its color planes differently. The pixels of an
8-bit PNG are the same either way. Either way the image is turned by its
EXIF orientation as Pillow's exif_transpose turns it, and its RGB
channels are kept. LoadImage turns a rotated image the same, and a
mirrored one (orientations 2, 4, 5 and 7) by a rotation alone.

Importing this module loads no mlx.
"""

from __future__ import annotations

import re
import shutil
import subprocess

import numpy as np

# The turn of each EXIF orientation, as Pillow's exif_transpose applies it.
_TURNS = {2: "FLIP_LEFT_RIGHT", 3: "ROTATE_180", 4: "FLIP_TOP_BOTTOM",
          5: "TRANSPOSE", 6: "ROTATE_270", 7: "TRANSVERSE", 8: "ROTATE_90"}
_PPM = re.compile(rb"P6\s+(\d+)\s+(\d+)\s+255\s")


def read(path: str, ffmpeg: str | None = "ffmpeg") -> np.ndarray:
    """The first frame of an image file, (H, W, 3) uint8, upright: decoded
    by ``ffmpeg`` when it is on PATH, else by Pillow (``ffmpeg`` None
    picks Pillow)."""
    from PIL import Image, ImageOps

    exe = shutil.which(ffmpeg) if ffmpeg else None
    with Image.open(path) as img:
        if exe is None:
            img.seek(0)
            return np.asarray(ImageOps.exif_transpose(img).convert("RGB"),
                              np.uint8).copy()
        turn = _TURNS.get(img.getexif().get(0x0112, 1))
    pixels = _decode(exe, path)
    if turn is None:
        return pixels
    return np.asarray(Image.fromarray(pixels).transpose(
        getattr(Image.Transpose, turn)), np.uint8).copy()


def _decode(exe: str, path: str) -> np.ndarray:
    """The first frame of ``path`` as ffmpeg decodes it, unturned, in
    8-bit RGB."""
    try:
        r = subprocess.run([exe, "-v", "error", "-noautorotate", "-i", path,
                            "-frames:v", "1", "-f", "image2pipe",
                            "-c:v", "ppm", "-pix_fmt", "rgb24", "-"],
                           capture_output=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired) as e:
        raise ValueError(f"ffmpeg could not read {path}: {e}") from None
    m = _PPM.match(r.stdout)
    if r.returncode or m is None:
        why = r.stderr.decode(errors="replace").strip().splitlines()
        raise ValueError(f"ffmpeg could not read {path}"
                         + (f": {why[-1]}" if why else ""))
    w, h = int(m.group(1)), int(m.group(2))
    data = r.stdout[m.end():m.end() + w * h * 3]
    if len(data) != w * h * 3:
        raise ValueError(f"ffmpeg gave {len(data)} bytes of a {w}x{h} "
                         f"frame of {path}")
    return np.frombuffer(data, np.uint8).reshape(h, w, 3).copy()


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
