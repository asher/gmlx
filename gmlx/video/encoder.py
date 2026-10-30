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

import os


def header_tokenizer(family, path: str):
    """The tokenizer the encoder's load builds, from the GGUF header
    alone, with no tensor reads. A file that is not the family's encoder
    is refused first, by its arch and shape."""
    from gmlx.image import text_encoder as te
    from gmlx.image.fingerprint import view_from_scans
    from gmlx.load import loadlog
    from gmlx.load.config_synth import GGUF_ARCH_TO_MODEL_TYPE
    from gmlx.load.headerscan import scan_gguf
    from gmlx.load.tokenizer import (finish_gguf_tokenizer,
                                     load_tokenizer_from_gguf)

    if family.encoder is None:
        raise ValueError(f"video family {family.name!r} has no text encoder")
    scan = scan_gguf(os.path.expanduser(path), include_tensors=False,
                     array_limit=1 << 62)
    # The identity digests read tensors; the load checks them.
    te.check_encoder_file(view_from_scans([scan]), family.encoder(),
                          override=True)
    arch = scan.kv["general.architecture"]
    with loadlog.quiet():
        tok = load_tokenizer_from_gguf(scan.kv, arch)
        finish_gguf_tokenizer(tok, GGUF_ARCH_TO_MODEL_TYPE.get(arch))
    return tok


def load_encoder(family, path: str, *, override: bool = False,
                 mmproj: str | None = None, log=print):
    """An owned ``EncoderHandle`` for the family's spec from the encoder
    GGUF at ``path``."""
    from gmlx.image import text_encoder as te

    if family.encoder is None:
        raise ValueError(f"video family {family.name!r} has no text encoder")
    return te.load_encoder(path, family.encoder(), override=override,
                           mmproj=mmproj, log=log, tag="video")
