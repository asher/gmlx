"""Load a video DiT GGUF through the image loader.

``load_dit`` identifies the file with the video rules and hands the result
to ``gmlx.image.loader.load_dit``, which builds the family's module tree,
installs a KQuantLinear per quantized matrix and fills every tensor. A
variant that runs video sparse attention, or a run with int8 attention
or int8 linears, is refused first where this install or Mac does not run
it. With int8 linears the family's block linears are converted to
``Int8Linear`` leaves once the load returns, before any adapter wraps
them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import mlx.core as mx
import mlx.nn as nn

from gmlx.image.fingerprint import (Identified, ImageModelError,
                                    view_from_scans)

from .fingerprint import VideoIdentified, VideoModelError, identify
from .registry import (attention_refusal, int8_fallback, linears_refusal,
                       vsa_refusal)

if TYPE_CHECKING:
    from gmlx.image.registry import Family


def identify_dit(path: str, *, family: str | None = None,
                 variant: str | None = None) -> VideoIdentified:
    """The family, variant and qkv layout of a DiT GGUF from its header,
    as ``variant`` when given and the weights allow it."""
    from gmlx.image.loader import _scan_shard
    from gmlx.load.preflight import find_split_shards

    try:
        view = view_from_scans([_scan_shard(s)
                                for s in find_split_shards(path)])
    except ValueError as e:
        raise VideoModelError(str(e)) from e
    return identify(view, family=family, variant=variant)


def load_dit(path: str, *, family: str | None = None,
             variant: str | None = None, act_dtype: mx.Dtype | None = None,
             attention: str = "dense", linears: str = "file",
             log=print) -> tuple[nn.Module, VideoIdentified]:
    """Load a video DiT GGUF to run with ``attention`` and ``linears``.
    Raises ``VideoModelError`` for a file gmlx video refuses, or an
    attention or linears this install or Mac does not run."""
    from gmlx.image.loader import load_dit as load_image_dit

    vid = identify_dit(path, family=family, variant=variant)
    fam = vid.family
    if vid.variant == "unknown":
        log(f"[video] warning: this {fam.title} GGUF matches no known "
            f"{fam.title} weights; it runs as {fam.unknown_variant}")
    if vid.layout == "unverified":
        log(f"[video] warning: this {fam.title} GGUF is not a known file, so "
            f"its qkv row order is unverified; it runs as [q; k; v]")
    elif vid.layout != "qkv":
        raise VideoModelError(
            f"this {fam.title} GGUF stores qkv as {vid.layout}, which gmlx "
            f"video does not run")
    if vid.config.get("vsa_sparsity"):
        why = vsa_refusal(fam, vid.variant)
        if why:
            raise VideoModelError(why)
    why = attention_refusal(fam, vid.variant, attention)
    if why:
        raise VideoModelError(why)
    why = linears_refusal(fam, vid.variant, linears)
    if why:
        raise VideoModelError(why)
    gate = {}
    if linears == "int8":
        from .memfit import int8_delta, int8_transient

        gate = {"gate_extra": int8_delta(fam, vid.view)
                + int8_transient(fam, vid.view),
                "gate_hint": f"Run with int8 {int8_fallback(fam, vid.variant)}"
                             ", or try a smaller DiT quant"}
    # The image loader reads only fields a video family also has.
    ident = Identified(family=cast("Family", fam), config=vid.config,
                       prefix=vid.prefix,
                       layout=vid.layout, variant=vid.variant,
                       identity_hash=vid.identity_hash, view=vid.view)
    try:
        model, _ = load_image_dit(path, act_dtype=act_dtype,
                                  identified=ident, log=log, kind="video",
                                  **gate)
    except ImageModelError as e:
        raise VideoModelError(str(e)) from e
    if attention != "dense":
        model.attention = attention
    if linears == "int8":
        try:
            n = model.convert_linears(fam.int8_linears)
        except ValueError as e:
            raise VideoModelError(str(e)) from e
        log(f"[video] {fam.title}: {n} linears converted to int8")
    return model, vid
