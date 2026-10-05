"""Startup fit check for ``gmlx image``.

The memory plan of the request, from the headers of the DiT, the text
encoder and the VAE, before any weights load. A request that cannot fit
with every lever applied is refused with the plan's terms, instead of
failing after a load of several GB. GMLX_TOOL_PREFLIGHT=0 skips it.
"""
from __future__ import annotations

import os


def enabled() -> bool:
    return os.environ.get("GMLX_TOOL_PREFLIGHT", "1") != "0"


def identify_dit(path: str):
    """The fingerprint of the DiT GGUF at ``path``, from its header."""
    from gmlx.image.loader import identify_dit as identify

    return identify(path)


def check(dit: str, encoder: str | None, vae: str | None, *, height: int,
          width: int, cfg: float | None):
    """The memory plan of one request, or None when the check is off or
    the variant has no defaults and no cfg is given. Raises
    ``ImageMemoryError`` when the request cannot fit."""
    if not enabled():
        return None
    from gmlx.image import memfit
    from gmlx.image.api import _header_bytes, sampler_defaults
    from gmlx.image.dtypes import image_activation

    ident = identify_dit(dit)
    fam = ident.family.name
    if cfg is None:
        row = sampler_defaults(fam, ident.variant)
        if row is None:
            return None
        cfg = row.cfg
    act = image_activation(fam)
    return memfit.plan_memory(
        _header_bytes(dit, "dit"), _header_bytes(encoder, "encoder"),
        _header_bytes(vae, "vae"), family=fam, height=height, width=width,
        cfg=cfg, dtype=act.dit, vae_dtype=act.vae)
