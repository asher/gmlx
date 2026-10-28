"""The text encoder of a video family.

A video family conditions on a text model through the image stack's
``EncoderSpec`` and ``EncoderHandle``. Its spec carries what the
reference's tokenizer adds to the encoder's own: tokens at fixed ids and
the ids of an empty prompt. MiniMax-H3's tokenizer adds seven tokens
whose ids the stock Qwen3-VL GGUF holds as filler entries, so the GGUF
tokenizer cannot produce them. The handle splits them out of the prompt
and puts their ids between the encoded pieces, which leaves the GGUF
tokenizer as it is for chat.

Importing this module loads no mlx.
"""

from __future__ import annotations


def load_encoder(family, path: str, *, override: bool = False,
                 mmproj: str | None = None, log=print):
    """An owned ``EncoderHandle`` for the family's spec from the encoder
    GGUF at ``path``."""
    from gmlx.image import text_encoder as te

    if family.encoder is None:
        raise ValueError(f"video family {family.name!r} has no text encoder")
    return te.load_encoder(path, family.encoder(), override=override,
                           mmproj=mmproj, log=log, tag="video")
