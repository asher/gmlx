"""MiniMax-H3 keyframes in the prompt: the ids, the vision spans, the
pixels the vision tower reads, and the encode.

A keyframe enters the encoder's prompt as a vision block before the
prompt text, and its rows of the features come from the vision tower.
The DiT runs each block with its markers at the video time and with the
video tag; ``spans`` are those blocks, (start, stop) rows of the prefix.

The picture functions take the label before each block and the bounds in
pixels of the image processor; the keyframe ones give them video's.

Importing this module loads no mlx.
"""

from __future__ import annotations

import numpy as np

from . import family as fam


def picture_ids(spec, tokenizer, prompt: str, sizes, label: str) -> tuple:
    """(ids, spans) of ``prompt`` with images of ``sizes`` (width,
    height) in the order given, each after ``label`` with its count in
    ``{i}``. A prompt with no images and no tokens is the spec's empty
    ids."""
    from gmlx.image import text_encoder as te

    ids: list[int] = []
    spans = []
    for i, (w, h) in enumerate(sizes, 1):
        if w % fam.VISION_PX or h % fam.VISION_PX:
            raise ValueError(f"a picture of {w}x{h} pixels is not on the "
                             f"vision tower's {fam.VISION_PX}-pixel grid")
        ids += te._encode(spec, tokenizer, label.format(i=i))
        start = len(ids)
        n = fam.vision_tokens(w, h)
        ids += [fam.VISION_START] + [fam.IMAGE_PAD] * n + [fam.VISION_END]
        spans.append((start, len(ids)))
    ids += te._encode(spec, tokenizer, te.render(spec, prompt))
    return (ids or list(spec.empty_ids)), tuple(spans)


def keyframe_ids(spec, tokenizer, prompt: str, sizes) -> tuple:
    """``picture_ids`` with video's label."""
    return picture_ids(spec, tokenizer, prompt, sizes, fam.PICTURE)


def token_tags(n: int, spans) -> np.ndarray:
    """ComfyUI's minimax_token_tags of a prefix of ``n`` rows: 1, and 0
    over each vision span."""
    tags = np.ones(n, np.int64)
    for start, stop in spans:
        tags[start:stop] = 0
    return tags


def spans_of(tags) -> tuple:
    """The vision spans of ComfyUI's minimax_token_tags: each run of 0."""
    t = np.asarray(tags).reshape(-1)
    if not np.isin(t, (0, 1)).all():
        raise ValueError("minimax_token_tags hold 0 and 1 alone")
    edges = np.flatnonzero(np.diff(np.concatenate([[1], t, [1]])))
    return tuple((int(a), int(b)) for a, b in zip(edges[::2], edges[1::2]))


def picture_pixels(handle, image: np.ndarray, bounds: tuple):
    """The tower's input for one picture, (H, W, 3) uint8 at the size the
    tower reads it, and its grid (1, h, w) in patches, by the encoder's
    image processor at ``bounds``, (least, most) pixels, and ComfyUI's
    normalization."""
    import mlx.core as mx

    proc = getattr(handle.processor, "image_processor", None)
    if proc is None:
        raise ValueError("the encoder has no image processor; load it with "
                         "its mmproj")
    for name in ("image_mean", "image_std"):
        if [float(v) for v in getattr(proc, name)] != [0.5] * 3:
            raise ValueError(f"the encoder's image processor has {name} "
                             f"{list(getattr(proc, name))}, not 0.5")
    lo, hi = bounds
    out = proc([np.transpose(image, (2, 0, 1))], min_pixels=lo,
               max_pixels=hi)
    grid = [int(v) for v in np.asarray(out["image_grid_thw"])[0]]
    patch = int(proc.patch_size)
    h, w = image.shape[:2]
    if grid != [1, h // patch, w // patch]:
        raise ValueError(f"the image processor resized a picture of {w}x{h} "
                         f"pixels to a grid of {grid[2]}x{grid[1]} patches")
    return mx.array(np.asarray(out["pixel_values"], np.float32)), grid


def pixels(handle, image: np.ndarray):
    """``picture_pixels`` at video's bounds, which keep a keyframe at its
    canvas size."""
    return picture_pixels(handle, image, fam.KEYFRAME_PIXELS)


def encode_pictures(handle, prompt: str, images, label: str,
                    bounds: tuple) -> tuple:
    """(features, spans): the encoder features (1, L, D) float32 of
    ``prompt`` with images, each (H, W, 3) uint8 at the size the tower
    reads it, after ``label`` at the image processor's ``bounds``, and
    their vision spans."""
    import mlx.core as mx

    from gmlx.image import vision_tap

    if handle.model is None:
        raise RuntimeError(f"the {handle.spec.name} text encoder was "
                           "released")
    if getattr(handle.model, "vision_tower", None) is None:
        raise ValueError("pictures need the encoder's vision tower; load "
                         "the encoder with the entry's mmproj")
    sizes = [(int(im.shape[1]), int(im.shape[0])) for im in images]
    ids, spans = picture_ids(handle.spec, handle.tokenizer, prompt, sizes,
                             label)
    feats, _ = vision_tap.tap(handle, ids,
                              (picture_pixels(handle, im, bounds)
                               for im in images),
                              pad=fam.IMAGE_PAD)
    out = feats[None].astype(mx.float32)
    mx.eval(out)
    return out, spans


def encode(handle, prompt: str, images) -> tuple:
    """``encode_pictures`` of keyframes at the canvas size, with video's
    label and bounds."""
    return encode_pictures(handle, prompt, images, fam.PICTURE,
                           fam.KEYFRAME_PIXELS)
