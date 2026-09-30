# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Asher Feldman
# Portions copyright (c) 2025 Prince Canuma (mlx-vlm, MIT)
# Vendored from mlx-vlm 0.6.15, mlx_vlm/models/qwen3_5_moe/config.py.
"""mlx-vlm 0.6.15 qwen3.5 MoE ``TextConfig``, the text half only.

The multimodal ``ModelConfig`` and ``VisionConfig`` stay mlx-vlm's.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

from gmlx._vendor.mlx_vlm.models.base import BaseModelConfig


@dataclass
class TextConfig(BaseModelConfig):
    model_type: str
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    linear_num_value_heads: int
    linear_num_key_heads: int
    linear_key_head_dim: int
    linear_value_head_dim: int
    linear_conv_kernel_dim: int
    num_experts: int
    num_experts_per_tok: int
    shared_expert_intermediate_size: int
    moe_intermediate_size: int
    rms_norm_eps: float
    vocab_size: int
    num_key_value_heads: int
    max_position_embeddings: int
    eos_token_id: Optional[Union[int, List[int]]] = None
    tie_word_embeddings: bool = False
    attention_bias: bool = False
    head_dim: Optional[int] = None
    rope_parameters: Optional[Dict[str, Union[float, str, bool, List[int]]]] = field(
        default_factory=lambda: {
            "type": "default",
            "mrope_section": [11, 11, 10],
            "rope_theta": 100000,
            "partial_rotary_factor": 0.25,
        }
    )
    full_attention_interval: int = 4

    def __post_init__(self):
        if self.rope_parameters:
            # Normalize rope_parameters keys (accept both 'rope_type' and 'type')
            if (
                "type" not in self.rope_parameters
                and "rope_type" in self.rope_parameters
            ):
                self.rope_parameters["type"] = self.rope_parameters.pop("rope_type")

            required_keys = {
                "mrope_section",
                "type",
                "rope_theta",
                "partial_rotary_factor",
            }
            if not all(key in self.rope_parameters for key in required_keys):
                raise ValueError(f"rope_parameters must contain keys {required_keys}")
