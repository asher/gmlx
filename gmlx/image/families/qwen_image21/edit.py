# SPDX-License-Identifier: Apache-2.0 AND MIT
# SPDX-FileCopyrightText: Copyright 2026 Qwen-Image Team and The HuggingFace Team
# Ported from calculate_dimensions, the image preprocessing of __call__ and
# the image branch of _get_qwen_prompt_embeds in diffusers'
# src/diffusers/pipelines/qwenimage21/pipeline_qwenimage21.py at e0abab83
# and modified for gmlx (Apache-2.0; see licenses/diffusers-LICENSE).
"""Reference images of Qwen-Image 2.1: their resize, the prompt template
that holds their slots, and the encoder tap over text and images.

A reference is made RGBA and resized with Lanczos to the area the request
gives, at its own aspect ratio, each side a multiple of 32. The encoder
sees it with the alpha composited over white, and the VAE sees all four
channels. The pixels come from Pillow, whose RGBA resize and paste the
reference pipeline runs, so both see the bytes the reference would.

The template opens the user turn with one slot per reference. The pad
token of a slot becomes one token per 32 by 32 pixels of its reference,
which is the vision tower's output grid, and each of those rows of the
features stands for four latent tokens of the reference in the DiT.

The tap over text and images, with the tower at float32 precision, is
gmlx.image.vision_tap's; this module gives it the ids and the pixels of
the references.
"""

from __future__ import annotations

import hashlib
import math

import mlx.core as mx
import numpy as np

from gmlx.image import vision_tap
from gmlx.image.conditioning import Reference

from .family import PROMPT, REFERENCES

MAX_ASPECT = 200            # the image processor refuses a longer image


def dimensions(area: int, ratio: float) -> tuple[int, int]:
    """(width, height) of a reference of aspect ``ratio``, width over
    height, at ``area`` pixels: each side rounded to the side multiple."""
    m = REFERENCES.values["side_multiple"]
    width = math.sqrt(area * ratio)
    height = width / ratio
    return round(width / m) * m, round(height / m) * m


def _pil(image):
    from PIL import Image

    if isinstance(image, Image.Image):
        return image
    arr = np.asarray(image)
    if arr.dtype != np.uint8 or arr.ndim not in (2, 3):
        raise ValueError(
            "a reference image is a PIL image or a uint8 array (H, W), "
            f"(H, W, 3) or (H, W, 4), not {arr.dtype} {arr.shape}")
    return Image.fromarray(arr)


class SmallReference(ValueError):
    """A reference under the encoder's least area at the area asked for.
    ``side`` is the least side of a square area that takes it."""

    def __init__(self, message: str, side: int):
        super().__init__(message)
        self.side = side


def least_side(ratio: float) -> int:
    """The least side of a square area at which a reference of aspect
    ``ratio`` comes out at the encoder's least area or more."""
    r = REFERENCES.values
    side = math.isqrt(r["min_area"] - 1) + 1
    while True:
        width, height = dimensions(side * side, ratio)
        if (width * height >= r["min_area"]
                and min(width, height) >= r["slot_px"]):
            return side
        side += 1


def prepare(image, area: int) -> Reference:
    """One reference image at the size the family runs it at."""
    from PIL import Image

    r = REFERENCES.values
    img = _pil(image)
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    w, h = img.size
    if min(w, h) <= 0:
        raise ValueError("a reference image has no pixels")
    width, height = dimensions(int(area), w / h)
    px = r["slot_px"]
    if min(width, height) < px or max(width, height) / min(width,
                                                           height) > MAX_ASPECT:
        raise ValueError(
            f"a reference image of {w}x{h} pixels is too long for its "
            f"area: the encoder takes an aspect ratio up to {MAX_ASPECT}")
    if width * height < r["min_area"]:
        side = least_side(w / h)
        raise SmallReference(
            f"a reference of {width}x{height} pixels is under the "
            f"{r['min_area']} pixels the encoder takes; use a reference "
            f"area of {side} x {side} pixels or more for this aspect ratio",
            side)
    if (width, height) != (w, h):
        img = img.resize((width, height),
                         getattr(Image.Resampling, r["resample"].upper()))
    return Reference(np.asarray(img, np.uint8))


def over_white(ref: Reference) -> np.ndarray:
    """The image the encoder sees: (H, W, 3) uint8, the alpha composited
    over white."""
    from PIL import Image

    rgba = Image.fromarray(ref.pixels, "RGBA")
    white = Image.new("RGB", rgba.size,
                      tuple(REFERENCES.values["encoder_background"]))
    white.paste(rgba, mask=rgba.getchannel("A"))
    return np.asarray(white, np.uint8)


def slots(ref: Reference) -> int:
    """Image rows the encoder gives a reference."""
    px = REFERENCES.values["slot_px"]
    return (ref.height // px) * (ref.width // px)


def render(prompt: str, count: int) -> str:
    """The prompt in the template of ``count`` references."""
    p = PROMPT.values
    slot = p["image_slot_join"].join(
        p["image_slot"].format(n=i) for i in range(1, count + 1))
    return p["template_edit"].format(slots=slot,
                                     prompt=prompt or p["empty_prompt"])


def template_id() -> str:
    """sha256 of what turns a prompt and its references into rows beside
    the token ids: the template, the slot text and the dropped rows."""
    p = PROMPT.values
    text = "\x00".join((p["template_edit"], p["image_slot"],
                        p["image_slot_join"], f"drop={p['drop_tokens']}"))
    return hashlib.sha256(text.encode()).hexdigest()


def _pad_id(tokenizer) -> int:
    ids = tokenizer.encode(PROMPT.values["image_pad"],
                           add_special_tokens=False)
    if len(ids) != 1:
        raise ValueError("the tokenizer has no single image pad token")
    return int(ids[0])


def token_ids(handle, prompt: str, refs) -> list[int]:
    """Token ids of the prompt with the slots of ``refs``: the pad token
    of each slot repeated once per image row of its reference."""
    if not refs:
        raise ValueError("the template of references needs a reference")
    tok = handle.tokenizer
    pad = _pad_id(tok)
    ids = list(tok.encode(render(prompt, len(refs)),
                          add_special_tokens=False))
    if ids.count(pad) != len(refs):
        raise ValueError(
            f"the rendered prompt has {ids.count(pad)} image pad tokens "
            f"for {len(refs)} references; a prompt cannot hold "
            f"{PROMPT.values['image_pad']}")
    out, at = [], 0
    for i in ids:
        if i == pad:
            out.extend([pad] * slots(refs[at]))
            at += 1
        else:
            out.append(i)
    return out


def _pixels(handle, ref: Reference):
    """The tower's input for one reference and its grid (1, h, w) in
    patches. The reference is already at a size the processor keeps."""
    proc = getattr(handle.processor, "image_processor", None)
    if proc is None:
        raise ValueError("the encoder has no image processor")
    chw = np.transpose(over_white(ref), (2, 0, 1))
    out = proc([chw])
    grid = [int(v) for v in np.asarray(out["image_grid_thw"])[0]]
    patch = int(proc.patch_size)
    if grid != [1, ref.height // patch, ref.width // patch]:
        raise ValueError(
            f"the image processor resized a reference of {ref.width}x"
            f"{ref.height} pixels to a grid of {grid[2]}x{grid[1]} patches")
    return mx.array(np.asarray(out["pixel_values"], np.float32)), grid


def encode(handle, prompt: str, refs, *, deepstack: int | None = None):
    """(ids, features, image_slots) of the prompt and its references.
    ``features`` is (L, D) float32 without the rows of the system turn,
    and ``image_slots`` are its image rows, ascending. ``deepstack`` cuts
    the count of deepstack blocks added, for a test of a wrong encoder."""
    model = handle.model
    if model is None:
        raise RuntimeError(f"the {handle.spec.name} text encoder was "
                           "released")
    tower = getattr(model, "vision_tower", None)
    if tower is None:
        raise ValueError(
            "reference images need the encoder's vision tower: load the "
            "pipeline with the encoder's mmproj")
    spec = handle.spec
    ids = token_ids(handle, prompt, refs)
    pad = _pad_id(handle.tokenizer)
    features, rows = vision_tap.tap(
        handle, ids, (_pixels(handle, ref) for ref in refs), pad=pad,
        deepstack=deepstack)
    image_slots = tuple(int(r) - spec.drop for r in rows)
    if image_slots and image_slots[0] < 0:
        raise ValueError("an image slot is in the dropped rows")
    return ids, features, image_slots
