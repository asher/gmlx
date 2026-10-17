# SPDX-License-Identifier: Apache-2.0 AND MIT
# SPDX-FileCopyrightText: Copyright 2025 Black Forest Labs, Inc.
# Ported from BFL's FLUX.2 reference at commit 50fe516: the preparation of
# reference images of encode_image_refs and default_prep, with
# cap_min_pixels, cap_pixels and center_crop_to_multiple_of_x, in
# src/flux2/sampling.py, and modified for gmlx (Apache-2.0; see
# licenses/flux2-LICENSE).
"""The reference images of a FLUX.2 klein edit, by the rules of the
reference's encode_image_refs.

The encoder reads the prompt alone. The references go to the DiT only,
as VAE latents after the image tokens. A reference must have sides of
at least 64 pixels and an aspect ratio of at most 8. One reference is
fitted under 2024 x 2024 pixels and each of several under 1024 x 1024,
with the sides of the fitted size floored, and each is then cropped to
the 16 pixel grid, centered. A request without a size takes the size of
the first prepared reference."""
from __future__ import annotations

import math

import numpy as np

from gmlx.image.conditioning import Reference

from .family import REFERENCES


class Prepared(tuple):
    """The prepared references of a request, in order. ``crop`` is None:
    the output of an edit is never cropped."""

    crop = None


class BadReference(ValueError):
    """A reference image the rules refuse. ``index`` is its place in the
    request."""

    def __init__(self, message: str, index: int):
        super().__init__(message)
        self.index = index


def _rgb(image):
    """``image`` upright and in RGB: the EXIF orientation applied, as
    image loaders do before the reference reads a file, and an alpha
    channel dropped, not composited, as the reference's convert does."""
    from PIL import Image, ImageOps

    if isinstance(image, Reference):
        image = image.pixels
    if isinstance(image, np.ndarray):
        a = np.asarray(image)
        if a.dtype != np.uint8 or a.ndim != 3 or a.shape[2] not in (3, 4):
            raise ValueError("a reference array must be (H, W, 3) or "
                             f"(H, W, 4) uint8, not {a.dtype} {a.shape}")
        image = Image.fromarray(a, "RGBA" if a.shape[2] == 4 else "RGB")
    return ImageOps.exif_transpose(image).convert("RGB")


def _reference(image) -> Reference:
    px = np.asarray(image, np.uint8)
    alpha = np.full(px.shape[:2] + (1,), 255, np.uint8)
    return Reference(np.concatenate([px, alpha], axis=2))


def check_size(width: int, height: int) -> None:
    """Refuse a reference with a side under the least or an aspect ratio
    past the most, as cap_min_pixels does."""
    r = REFERENCES.values
    low, most = r["min_side"], r["max_aspect"]
    if width < low or height < low:
        raise ValueError(f"a reference of {width}x{height} pixels has a "
                         f"side under {low}")
    if width / height > most or height / width > most:
        raise ValueError(f"a reference of {width}x{height} pixels is wider "
                         f"or taller than {most} to 1")


def fitted_size(width: int, height: int, limit: int) -> tuple:
    """(width, height) of a reference under ``limit`` pixels, as
    cap_pixels sizes it."""
    if width * height <= limit:
        return width, height
    scale = math.sqrt(limit / (width * height))
    return int(width * scale), int(height * scale)


def prepare_one(image, limit: int) -> Reference:
    """One reference: checked, fitted under ``limit`` pixels by a Lanczos
    resize, and cropped to the grid, centered."""
    from PIL import Image

    image = _rgb(image)
    check_size(*image.size)
    fit = fitted_size(*image.size, limit)
    if fit != image.size:
        image = image.resize(fit, Image.Resampling.LANCZOS)
    g = REFERENCES.values["grid"]
    width, height = image.size
    w, h = width // g * g, height // g * g
    left, top = (width - w) // 2, (height - h) // 2
    return _reference(image.crop((left, top, left + w, top + h)))


def prepare(images) -> Prepared:
    """The references of a request by the rule of their count. An image
    the rules refuse raises BadReference."""
    images = list(images)
    r = REFERENCES.values
    if len(images) > r["max_refs"]:
        raise ValueError(f"at most {r['max_refs']} references, and the "
                         f"request has {len(images)}")
    limit = r["one_max_pixels"] if len(images) == 1 else r["multi_max_pixels"]
    out = []
    for i, image in enumerate(images):
        try:
            out.append(prepare_one(image, limit))
        except ValueError as e:
            raise BadReference(str(e), i) from None
    return Prepared(out)


def default_size(refs) -> tuple | None:
    """(width, height) of the first prepared reference, None without."""
    return (refs[0].width, refs[0].height) if refs else None
