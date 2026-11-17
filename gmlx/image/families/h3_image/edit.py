"""The reference images of a MiniMax-H3-Image edit, by the rules of
ComfyUI's MiniMaxH3ImageConditioning (comfy_extras/nodes_minimax_h3.py).

image_1 is the picture to edit. Each reference, upright and in RGB with an
alpha channel dropped, has two copies: the VAE's, its short edge scaled
into 1024 to 2048 pixels, and the vision tower's, its short edge scaled to
1024, each side of both rounded to the 32 pixel grid. A copy at the size
of its image is the image's own bytes, and any other Pillow's Lanczos
resize of the 8-bit image, as ComfyUI's lanczos runs it. The VAE encodes
the VAE copy, and the DiT runs its latents as a reference frame. The
prompt reads each vision copy after its label, "Picture i: ", as a vision
block before the prompt text (``gmlx.image.h3.pictures``), and the image
processor keeps it at its size within its bounds. A request with
references and no size takes image_1's VAE size.

gmlx takes references at 1:3 to 3:1, the family's shapes, which keep a
vision copy within 3072 vision tokens; ComfyUI takes any.
"""
from __future__ import annotations

import hashlib

import numpy as np

from gmlx.image.conditioning import Reference

from .family import REFERENCES, TITLE

_R = REFERENCES.values
LABEL = _R["label"]
BOUNDS = tuple(_R["pixels"])
MAX_ASPECT = 3
# Qwen3-VL's vision block in text, its markers and one image pad, whose
# bytes bound the three ids before the pads expand.
VISION_BLOCK = "<|vision_start|><|image_pad|><|vision_end|>"


class BadReference(ValueError):
    """A reference image the rules refuse. ``index`` is its place in the
    request."""

    def __init__(self, message: str, index: int):
        super().__init__(message)
        self.index = index


def short_edge(width: int, height: int, low: int, high: int) -> tuple:
    """(width, height) of an image of ``width`` x ``height`` scaled at its
    shape to a short edge of ``low`` to ``high``, each side rounded to the
    grid and at least one grid step: ComfyUI's _size_short_edge."""
    g = _R["grid"]
    least = min(width, height)
    scale = min(max(least, low), high) / least
    return tuple(max(g, round(side * scale / g) * g)
                 for side in (width, height))


def _rgb(image):
    """``image`` upright and in RGB, as ComfyUI's image loader reads it:
    the EXIF orientation applied, and an alpha channel dropped, not
    composited."""
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


def _copy(image, size: tuple) -> np.ndarray:
    """(H, W, 3) uint8 of ``image`` at ``size``: its bytes at its own size,
    else its Lanczos resize."""
    from PIL import Image

    if image.size != size:
        image = image.resize(size, Image.Resampling.LANCZOS)
    return np.asarray(image, np.uint8)


def prepare_reference(image) -> Reference:
    """One reference: its VAE copy as the Reference's pixels, alpha 255,
    and its vision copy."""
    image = _rgb(image)
    w, h = image.size
    if max(w, h) > MAX_ASPECT * min(w, h):
        raise ValueError(
            f"a reference of {w}x{h} pixels is outside the shapes "
            f"{TITLE} takes, 1:{MAX_ASPECT} to {MAX_ASPECT}:1")
    low = _R["short_edge"]
    vae = _copy(image, short_edge(w, h, low, _R["vae_max_short_edge"]))
    alpha = np.full(vae.shape[:2] + (1,), 255, np.uint8)
    return Reference(np.concatenate([vae, alpha], axis=2),
                     vision=_copy(image, short_edge(w, h, low, low)))


def prepare(images) -> tuple:
    """The references of a request, in order. An image the rules refuse
    raises BadReference."""
    images = list(images)
    if len(images) > _R["max_refs"]:
        raise ValueError(f"at most {_R['max_refs']} references, and the "
                         f"request has {len(images)}")
    out = []
    for i, image in enumerate(images):
        try:
            out.append(prepare_reference(image))
        except ValueError as e:
            raise BadReference(str(e), i) from None
    return tuple(out)


def default_size(refs) -> tuple | None:
    """(width, height) of image_1's VAE copy, None without references."""
    return (refs[0].width, refs[0].height) if refs else None


def _vision(refs) -> list:
    out = []
    for ref in refs:
        if ref.vision is None:
            raise ValueError(f"a {TITLE} reference needs its vision copy: "
                             "prepare it with gmlx.image.api.references")
        out.append(ref.vision)
    return out


def token_ids(handle, prompt: str, refs) -> list[int]:
    """Token ids of the prompt with the vision block of each reference
    after its label, before the prompt text."""
    from gmlx.image.h3.pictures import picture_ids

    if not refs:
        raise ValueError("the template of references needs a reference")
    sizes = [(int(v.shape[1]), int(v.shape[0])) for v in _vision(refs)]
    ids, _ = picture_ids(handle.spec, handle.tokenizer, prompt, sizes, LABEL)
    return ids


def encode(handle, prompt: str, refs):
    """(ids, features, image_slots) of the prompt and its references.
    ``features`` is (L, D) float32, and ``image_slots`` are its vision
    rows, each block less its markers, ascending."""
    from gmlx.image.h3.pictures import encode_pictures, span_slots

    ids = token_ids(handle, prompt, refs)
    features, spans = encode_pictures(handle, prompt, _vision(refs), LABEL,
                                      BOUNDS)
    if features.shape[1] != len(ids):
        raise ValueError(f"the encoder gave {features.shape[1]} rows for "
                         f"{len(ids)} token ids")
    return ids, features[0], span_slots(spans)


def render(prompt: str, count: int) -> str:
    """The prompt with the label and vision markers of ``count``
    references, one image pad each, before their pads expand."""
    from gmlx.image import text_encoder as te
    from gmlx.image.h3.encoder import spec

    return "".join(LABEL.format(i=i) + VISION_BLOCK
                   for i in range(1, count + 1)) + te.render(spec(), prompt)


def template_id() -> str:
    """sha256 of what turns a prompt and its references into rows beside
    the token ids: the label, the image processor's bounds, and the vision
    tower's float32 pass."""
    text = "\x00".join((LABEL, f"pixels={BOUNDS[0]},{BOUNDS[1]}",
                        "tower=float32"))
    return hashlib.sha256(text.encode()).hexdigest()
