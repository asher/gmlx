"""MiniMax-H3 pictures in the prompt, shared by video's keyframes and
the image's references: the ids, the vision spans, the pixels the vision
tower reads, and the encode.

A picture enters the encoder's prompt as a vision block after its label
and before the prompt text, and its rows of the features come from the
vision tower. The DiT runs each block with its markers at the video tag;
``spans`` are those blocks, (start, stop) rows of the prefix. A caller
gives the label, with the picture's count in ``{i}``, and the bounds in
pixels of the image processor.

Importing this module loads no mlx.
"""

from __future__ import annotations

import numpy as np

from .encoder import (IMAGE_PAD, VISION_END, VISION_PX, VISION_START,
                      vision_tokens)


def picture_ids(spec, tokenizer, prompt: str, sizes, label: str) -> tuple:
    """(ids, spans) of ``prompt`` with images of ``sizes`` (width,
    height) in the order given, each after ``label`` with its count in
    ``{i}``. A prompt with no images and no tokens is the spec's empty
    ids."""
    from gmlx.image import text_encoder as te

    ids: list[int] = []
    spans = []
    for i, (w, h) in enumerate(sizes, 1):
        if w % VISION_PX or h % VISION_PX:
            raise ValueError(f"a picture of {w}x{h} pixels is not on the "
                             f"vision tower's {VISION_PX}-pixel grid")
        ids += te._encode(spec, tokenizer, label.format(i=i))
        start = len(ids)
        n = vision_tokens(w, h)
        ids += [VISION_START] + [IMAGE_PAD] * n + [VISION_END]
        spans.append((start, len(ids)))
    ids += te._encode(spec, tokenizer, te.render(spec, prompt))
    return (ids or list(spec.empty_ids)), tuple(spans)


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
                              pad=IMAGE_PAD)
    out = feats[None].astype(mx.float32)
    mx.eval(out)
    return out, spans


def span_slots(spans) -> tuple:
    """The image slots of vision spans: each span's rows less its start
    and end markers, ascending."""
    return tuple(r for a, b in spans for r in range(a + 1, b - 1))


def slot_spans(slots) -> tuple:
    """The vision spans of image slots: each run of consecutive slots with
    the marker row before it and after it."""
    out: list = []
    for s in (int(v) for v in slots):
        if out and s == out[-1][1] - 1:
            out[-1][1] = s + 2
        elif out and s < out[-1][1] - 1:
            raise ValueError(f"image slots {list(slots)} are not ascending")
        else:
            if s < 1:
                raise ValueError(f"image slot {s} has no marker row before "
                                 "it")
            out.append([s - 1, s + 2])
    return tuple((a, b) for a, b in out)
