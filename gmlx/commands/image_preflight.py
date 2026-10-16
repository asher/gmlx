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
    ``prompts`` holds the texts of a request whose encoder reads them
    alone, in the family's template, and prices their rows at the most
    their bytes can give; without them the rows are the family's most.
    Raises ``ImageMemoryError`` when the request cannot fit."""
    if not enabled():
        return None
    from gmlx.image import memfit
    from gmlx.image.api import _header_bytes, sampler_defaults
    from gmlx.image.dtypes import image_activation
    from gmlx.image.text_encoder import (_header_view, encoder_spec,
                                         rows_at_most)

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
    caption = None
    if prompts:
        spec = encoder_spec(ident.family)
        caption = max(rows_at_most(spec, p) for p in prompts)
    return memfit.plan_memory(
        _header_bytes(dit, "dit"), _header_bytes(encoder, "encoder"),
        _header_bytes(vae, "vae"), tower,
        adapters=adapters, family=fam, height=height, width=width, cfg=cfg,
        dtype=act.dit, vae_dtype=act.vae, init_image=init_image,
        caption_tokens=caption, ref_sizes=[(int(h), int(w)) for h, w in refs],
        load_peak=max(load_transient(encoder, vae, mmproj), cast),
        # Each part alone, for the staged order.
        load_parts=(load_transient(encoder, None, mmproj),
                    max(load_transient(None, vae, None), cast)),
        encoder_owned=not keep_encoder)
