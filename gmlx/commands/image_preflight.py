"""Startup fit check for ``gmlx image``.

The memory plan of the request, from the headers of the DiT, the text
encoder and the VAE, before any weights load, with what the load holds
beside them. A request that cannot fit
with every lever applied is refused with the plan's terms, instead of
failing after a load of several GB. The last lever is the staged load,
which the command follows when the plan takes it.
GMLX_TOOL_PREFLIGHT=0 skips the check.
"""
from __future__ import annotations

import os


def enabled() -> bool:
    return os.environ.get("GMLX_TOOL_PREFLIGHT", "1") != "0"


def identify_dit(path: str, variant: str | None = None):
    """The fingerprint of the DiT GGUF at ``path``, from its header, under
    ``variant`` when one is given."""
    from gmlx.image.loader import identify_dit as identify

    return identify(path, variant=variant)


def load_transient(encoder: str | None, vae: str | None,
                   mmproj: str | None) -> int:
    """Bytes the load of the parts holds over what stays resident, from
    their headers."""
    from gmlx.image import memfit
    from gmlx.image.text_encoder import _header_view
    from gmlx.image.vae import read_header

    return memfit.load_transient(
        None if encoder is None else _header_view(encoder),
        None if vae is None else read_header(str(vae)),
        None if mmproj is None else _header_view(mmproj))


def dit_transient(dit: str, act) -> int:
    """Bytes the load of the DiT holds over what stays resident, from its
    header: the 16-bit float tensors it casts to ``act``."""
    from gmlx.image import memfit
    from gmlx.image.text_encoder import _header_view

    return memfit.dit_transient(_header_view(dit), act)


def cast_of(view, family: str) -> int:
    """File bytes of the 16-bit float tensors of a DiT header view that
    its load casts to the activation dtype of ``family`` on this chip, as
    the load gate counts them. 0 on a chip that runs the family in no
    dtype."""
    from gmlx.image import memfit
    from gmlx.image.dtypes import ImageDtypeError, image_activation

    try:
        act = image_activation(family).dit
    except ImageDtypeError:
        return 0
    return memfit.cast_bytes(view, act)


def dit_cast_bytes(dit: str, family: str | None = None) -> int:
    """``cast_of`` the DiT file at ``dit``, of ``family`` or the family
    its header names. 0 for a header that cannot be read."""
    from gmlx.image.text_encoder import _header_view

    try:
        if family is None:
            family = identify_dit(str(dit)).family.name
        return cast_of(_header_view(str(dit)), family)
    except Exception:                          # noqa: BLE001 - advisory
        return 0


def quant_cast_bytes(quant: str, size: int, family: str) -> int:
    """``cast_of`` a DiT not on disk, from its quant and file size: the
    whole file for a 16-bit float quant the family does not run in."""
    from types import SimpleNamespace

    q = quant.upper()
    view = SimpleNamespace(tensors={"weight": SimpleNamespace(
        codec=q, nbytes=int(size), shape=())})
    return cast_of(view, family) if q in ("F16", "BF16") else 0


def prompt_rows(family, prompt: str, refs: int = 0) -> int:
    """The most text rows the features of ``prompt`` hold on ``family``,
    from its bytes in the family's template, before a tokenizer loads. An
    edit whose encoder reads its ``refs`` reference images reads the
    prompt in the family's template for references, with the slot text of
    each, and is priced at least at the family's default rows, those of a
    plan that does not know the prompt. The plan prices the image rows of
    the references apart, from their sizes."""
    from gmlx.image import memfit
    from gmlx.image.text_encoder import (encoder_spec, rows_at_most,
                                         text_rows_at_most)

    spec = encoder_spec(family)
    edit = family.edit
    if refs and edit is not None and not edit.text_only:
        return max(text_rows_at_most(spec, edit.render(prompt, refs)),
                   memfit.PEAKS[family.name].text_rows)
    return rows_at_most(spec, prompt)


def prompt_terms(family, prompts, refs=()) -> dict:
    """The ``memfit.plan_memory`` arguments that price the texts the
    encoder reads, the positive first, beside the references of ``refs``,
    (height, width) in pixels: the caption and the text rows at the most
    rows of them by ``prompt_rows``, and on a family that caches a prefix
    the prefix rows of each conditioning by its own text. Empty with no
    texts, or on a family whose plan does not price the prompt."""
    from gmlx.image import memfit

    if not prompts or not memfit.prices_prompt(family.name):
        return {}
    rows = [prompt_rows(family, p, len(refs)) for p in prompts]
    terms = dict(caption_tokens=max(rows), text_tokens=max(rows))
    prefix = memfit.PEAKS[family.name].prefix
    if prefix is not None:
        terms["prefix_rows"] = [prefix.sized_rows(n, refs) for n in rows]
    return terms


def check(dit: str, encoder: str | None, vae: str | None, *, height: int,
          width: int, cfg: float | None, adapters=(),
          init_image: bool = False, keep_encoder: bool = False,
          mmproj: str | None = None, refs=(), variant: str | None = None,
          prompts=()):
    """The memory plan of one request, or None when the check is off or
    the variant has no defaults and no cfg is given. ``adapters`` is the
    byte count of each adapter, ``init_image`` says the request encodes
    an init image of its size, and ``keep_encoder`` that the encoder is
    not released in it. ``mmproj`` is the encoder's mmproj and ``refs``
    the (height, width) in pixels of each reference image of the request.
    ``prompts`` holds the texts the encoder reads, the positive first,
    priced by ``prompt_terms``; with none, or on a family whose plan does
    not price the prompt, the rows are the family's most.
    Raises ``ImageMemoryError`` when the request cannot fit."""
    if not enabled():
        return None
    from gmlx.image import memfit
    from gmlx.image.api import _header_bytes, sampler_defaults
    from gmlx.image.dtypes import image_activation
    from gmlx.image.text_encoder import _header_view

    ident = identify_dit(dit, variant=variant)
    fam = ident.family.name
    if cfg is None:
        row = sampler_defaults(fam, ident.variant)
        if row is None:
            return None
        cfg = row.cfg
    act = image_activation(fam)
    tower = None if mmproj is None else _header_view(mmproj)
    # The DiT loads after the encoder, in either order.
    cast = dit_transient(dit, act.dit)
    sizes = [(int(h), int(w)) for h, w in refs]
    return memfit.plan_memory(
        _header_bytes(dit, "dit"), _header_bytes(encoder, "encoder"),
        _header_bytes(vae, "vae"), tower,
        adapters=adapters, family=fam, height=height, width=width, cfg=cfg,
        dtype=act.dit, vae_dtype=act.vae, init_image=init_image,
        ref_sizes=sizes, **prompt_terms(ident.family, prompts, sizes),
        load_peak=max(load_transient(encoder, vae, mmproj), cast),
        # Each part alone, for the staged order.
        load_parts=(load_transient(encoder, None, mmproj),
                    max(load_transient(None, vae, None), cast)),
        encoder_owned=not keep_encoder)
