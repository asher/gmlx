# SPDX-License-Identifier: Apache-2.0 AND MIT
# SPDX-FileCopyrightText: Copyright 2026 Black Forest Labs, Inc.
# Ported from BFL's FLUX.3 klein reference at commit 45e57c9: the
# reference canvas, the single and multi reference preparation and the
# default size of src/flux3/pipeline.py, and the choice between them and
# the crop of the output in generate() of src/flux3/generate.py, and
# modified for gmlx (Apache-2.0; see licenses/flux-3-image-klein-LICENSE).
"""The reference images of a FLUX.3 klein edit, by the rules of the
reference's pipeline.py and generate.py.

The encoder reads the prompt alone. The references go to the DiT only,
as VAE latents in the image_cond stream. One reference is fitted to at
most MATCH_MAX_PIXELS of canvas, the fitted size rounded up to the 16
pixel grid with sides of 64 to 4096, and padded to that canvas by
reflection, centered. Each of several references is fitted under
MULTI_REFERENCE_MAX_PIXELS with its sides floored and cropped to the
grid, centered. A request without a size takes the size of the first
prepared reference, and the output of one padded reference is cropped
back to the box of the fitted image."""
from __future__ import annotations

import math

import numpy as np

from gmlx.image.conditioning import Reference

from .family import REFERENCES


class Prepared(tuple):
    """The prepared references of a request, in order, with ``crop``: the
    (left, top, right, bottom) box of the fitted image inside the canvas
    of one padded reference, else None."""

    crop: tuple | None = None

    def __new__(cls, refs, crop=None):
        out = super().__new__(cls, refs)
        out.crop = crop
        return out


class BadReference(ValueError):
    """A reference image the rules refuse. ``index`` is its place in the
    request."""

    def __init__(self, message: str, index: int):
        super().__init__(message)
        self.index = index


def _grid(v: int) -> int:
    g = REFERENCES.values["grid"]
    return -(-v // g) * g


def _rgb(image):
    """``image`` upright and in RGB, as the reference reads it: the EXIF
    orientation applied, and an alpha channel dropped, not composited."""
    from PIL import Image, ImageOps

    if isinstance(image, Reference):
        image = image.pixels
    if isinstance(image, np.ndarray):
        a = np.asarray(image)
        if a.dtype != np.uint8 or a.ndim != 3 or a.shape[2] not in (3, 4):
            raise ValueError("a reference array must be (H, W, 3) or "
                             f"(H, W, 4) uint8, not {a.dtype} {a.shape}")
        image = Image.fromarray(a, "RGBA" if a.shape[2] == 4 else "RGB")
    image = ImageOps.exif_transpose(image).convert("RGB")
    if min(image.size) <= 0:
        raise ValueError("a reference image has no pixels")
    return image


def _reference(image) -> Reference:
    px = np.asarray(image, np.uint8)
    alpha = np.full(px.shape[:2] + (1,), 255, np.uint8)
    return Reference(np.concatenate([px, alpha], axis=2))


def reference_canvas(width: int, height: int) -> tuple:
    """(fitted size, canvas) of one reference of ``width`` x ``height``
    pixels, each as (width, height)."""
    r = REFERENCES.values
    most, low, high = r["match_max_pixels"], r["min_side"], r["max_side"]
    scale = min(1.0, math.sqrt(most / (width * height)),
                high / max(width, height))
    while True:
        fit = (max(1, round(width * scale)), max(1, round(height * scale)))
        canvas = tuple(min(high, max(low, _grid(v))) for v in fit)
        if canvas[0] * canvas[1] <= most:
            return fit, canvas
        scale *= 0.995


def prepare_reference(image) -> tuple:
    """One reference: (the Reference on its canvas, the box of the fitted
    image inside it or None when it fills it)."""
    from PIL import Image

    image = _rgb(image)
    fit, (w, h) = reference_canvas(image.width, image.height)
    if fit != image.size:
        image = image.resize(fit, Image.Resampling.LANCZOS)
    if (w, h) == image.size:
        return _reference(image), None
    left, top = (w - image.width) // 2, (h - image.height) // 2
    pad = ((top, h - image.height - top), (left, w - image.width - left),
           (0, 0))
    # numpy's reflect needs a pad shorter than the side.
    mode = "reflect" if min(image.size) > REFERENCES.values["grid"] \
        else "symmetric"
    px = np.pad(np.asarray(image), pad, mode=mode)
    return _reference(px), (left, top, left + image.width,
                            top + image.height)


def prepare_multi_reference(image) -> Reference:
    """Each reference of several: fitted under the pixels of one and
    cropped to the grid, centered."""
    from PIL import Image

    image = _rgb(image)
    width, height = image.size
    most = REFERENCES.values["multi_max_pixels"]
    if width * height > most:
        scale = math.sqrt(most / (width * height))
        width, height = int(width * scale), int(height * scale)
        image = image.resize((width, height), Image.Resampling.LANCZOS)
    g = REFERENCES.values["grid"]
    w, h = width // g * g, height // g * g
    if not (w and h):
        raise ValueError(f"a reference of {width}x{height} pixels has no "
                         f"{g} x {g} pixel tile to keep")
    left, top = (width - w) // 2, (height - h) // 2
    return _reference(image.crop((left, top, left + w, top + h)))


def prepare(images) -> Prepared:
    """The references of a request by the rule of their count. An image
    the rules refuse raises BadReference."""
    images = list(images)
    if len(images) > REFERENCES.values["max_refs"]:
        raise ValueError(f"at most {REFERENCES.values['max_refs']} "
                         f"references, and the request has {len(images)}")
    out, box = [], None
    for i, image in enumerate(images):
        try:
            if len(images) == 1:
                ref, box = prepare_reference(image)
            else:
                ref = prepare_multi_reference(image)
        except ValueError as e:
            raise BadReference(str(e), i) from None
        out.append(ref)
    return Prepared(tuple(out), box)


def default_size(refs) -> tuple | None:
    """(width, height) of the first prepared reference, None without."""
    return (refs[0].width, refs[0].height) if refs else None
