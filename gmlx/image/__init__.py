"""Image diffusion models from GGUF files.

Nothing under this package is imported by ``import gmlx``. Families live in
``gmlx.image.families``, one directory each, found by ``registry.families``.
The primitives are exported here and load ``gmlx.image.api`` on first use,
so importing a submodule such as ``gmlx.image.registry`` stays light.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .api import (SampleStopped, define_patch_table, denoise_step,
                      encode_references, encode_text, guide, load,
                      plan_request, references, sample, sampler_defaults,
                      vae_decode, vae_encode)

__all__ = [
    "SampleStopped",
    "define_patch_table",
    "denoise_step",
    "encode_references",
    "encode_text",
    "guide",
    "load",
    "plan_request",
    "references",
    "sample",
    "sampler_defaults",
    "vae_decode",
    "vae_encode",
]


def __getattr__(name: str):
    if name in __all__:
        from . import api

        return getattr(api, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
