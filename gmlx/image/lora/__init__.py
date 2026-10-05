"""LoRA adapters for image DiTs.

``load_adapter`` reads a community LoRA safetensors file into a pipeline,
``table_entries`` gives the entries of a patch table for adapters at their
strengths, and ``gmlx.image.define_patch_table`` names the table that
``denoise_step`` and ``sample`` then run under. ``register_pair`` serves a
caller that holds the tensors itself.
"""

from __future__ import annotations

from .install import ImageLoRALinear, pair_hash, register_pair
from .patch_table import LoadedAdapter, load_adapter, table_entries
from .reader import (AdapterError, AdapterFamilyError, UnmatchedKeysError,
                     UnsupportedAdapterError, read_lora, resolve)

__all__ = [
    "AdapterError",
    "AdapterFamilyError",
    "ImageLoRALinear",
    "LoadedAdapter",
    "UnmatchedKeysError",
    "UnsupportedAdapterError",
    "load_adapter",
    "pair_hash",
    "read_lora",
    "register_pair",
    "resolve",
    "table_entries",
]
