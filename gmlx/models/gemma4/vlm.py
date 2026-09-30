# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Asher Feldman
"""The module ``build_vlm_model`` builds a gemma4 VLM from.

The owned container and language model, with the stock mlx-vlm configs,
vision tower and audio encoder, under the names mlx-vlm's model modules
export. ``update_module_configs`` finds the four config classes here.
"""

from gmlx._vendor.mlx_vlm.models.gemma4.audio import AudioEncoder
from gmlx._vendor.mlx_vlm.models.gemma4.config import (
    AudioConfig,
    ModelConfig,
    TextConfig,
    VisionConfig,
)
from gmlx._vendor.mlx_vlm.models.gemma4.vision import VisionModel

from .gemma4 import Model
from .language import LanguageModel

__all__ = [
    "AudioConfig",
    "AudioEncoder",
    "LanguageModel",
    "Model",
    "ModelConfig",
    "TextConfig",
    "VisionConfig",
    "VisionModel",
]
