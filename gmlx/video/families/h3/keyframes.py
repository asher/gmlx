"""MiniMax-H3 keyframes in the prompt: the ids, the vision spans, the
pixels the vision tower reads, and the encode.

A keyframe enters the encoder's prompt as a vision block before the
prompt text, and its rows of the features come from the vision tower.
The DiT runs each block with its markers at the video time and with the
video tag; ``spans`` are those blocks, (start, stop) rows of the prefix.

The picture functions live in ``gmlx.image.h3.pictures`` and take the
label before each block, the bounds in pixels of the image processor and
the noun their errors use; the keyframe ones give them video's, so a bad
keyframe's error says keyframe.

Importing this module loads no mlx.
"""

from __future__ import annotations

import numpy as np

from gmlx.image.h3.pictures import (encode_pictures, picture_ids,
                                    picture_pixels, spans_of, token_tags)

from . import family as fam

__all__ = ["encode", "encode_pictures", "keyframe_ids", "picture_ids",
           "picture_pixels", "pixels", "spans_of", "token_tags"]

# What a keyframe's errors call it.
NOUN = "keyframe"


def keyframe_ids(spec, tokenizer, prompt: str, sizes) -> tuple:
    """``picture_ids`` with video's label."""
    return picture_ids(spec, tokenizer, prompt, sizes, fam.PICTURE,
                       noun=NOUN)


def pixels(handle, image: np.ndarray):
    """``picture_pixels`` at video's bounds, which keep a keyframe at its
    canvas size."""
    return picture_pixels(handle, image, fam.KEYFRAME_PIXELS, noun=NOUN)


def encode(handle, prompt: str, images) -> tuple:
    """``encode_pictures`` of keyframes at the canvas size, with video's
    label and bounds."""
    return encode_pictures(handle, prompt, images, fam.PICTURE,
                           fam.KEYFRAME_PIXELS, noun=NOUN)
