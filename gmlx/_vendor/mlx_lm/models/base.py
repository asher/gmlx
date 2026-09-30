# Copyright © 2023-2024 Apple Inc.

import inspect
from dataclasses import dataclass
from typing import Any, Optional

import mlx.core as mx
from mlx.utils import tree_map

# gmlx: gmlx.cache.kv_cache and gmlx.cache.attention own the mask and SDPA
# helpers of this module, so their bodies are deleted here and imported by
# name. BaseModelArgs stays.
from gmlx.cache.attention import (
    create_attention_mask,
    create_ssm_mask,
    quantized_scaled_dot_product_attention,
    scaled_dot_product_attention,
)
from gmlx.cache.kv_cache import create_causal_mask


@dataclass
class BaseModelArgs:
    @classmethod
    def from_dict(cls, params):
        return cls(
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            }
        )
