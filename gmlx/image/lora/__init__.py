"""LoRA adapters for image DiTs.

``load_adapter`` reads a community LoRA safetensors file into a pipeline,
``table_entries`` gives the entries of a patch table for adapters at their
strengths, and ``gmlx.image.define_patch_table`` names the table that
``denoise_step`` and ``sample`` then run under. ``register_pair`` and
``plans_from_patches`` serve callers that hold the tensors themselves, as
the ComfyUI node pack does. ``apply_text_adapters`` installs the text
encoder halves on an owned encoder.
"""

from __future__ import annotations

from .install import (EncoderLoRALinear, ImageLoRALinear, pair_hash,
                      register_pair)
from .patch_table import (LoadedAdapter, apply_text_adapters, load_adapter,
                          plans_from_patches, register_patch, table_entries,
                          text_state)
from .reader import (AdapterError, AdapterFamilyError, UnmatchedKeysError,
                     UnsupportedAdapterError, read_lora, resolve)

__all__ = [
    "AdapterError",
    "AdapterFamilyError",
    "EncoderLoRALinear",
    "ImageLoRALinear",
    "LoadedAdapter",
    "UnmatchedKeysError",
    "UnsupportedAdapterError",
    "apply_text_adapters",
    "load_adapter",
    "pair_hash",
    "plans_from_patches",
    "read_lora",
    "register_pair",
    "register_patch",
    "resolve",
    "table_entries",
    "text_state",
]
