# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Asher Feldman
"""The module a gemma4 VLM builds from on every gmlx route.

``build_vlm_model`` builds from it, and mlx-vlm's own loader does too once
``install_owned_vlm_resolver`` hands it back for the stock package. It
carries every public name of ``mlx_vlm.models.gemma4``: the owned
container and language model, and the stock configs, vision tower, audio
encoder and processors. ``update_module_configs`` finds the four config
classes here.
"""

from gmlx._vendor.mlx_vlm.models.gemma4.audio import AudioEncoder
from gmlx._vendor.mlx_vlm.models.gemma4.audio_feature_extractor import (
    Gemma4AudioFeatureExtractor,
)
from gmlx._vendor.mlx_vlm.models.gemma4.config import (
    AudioConfig,
    ModelConfig,
    TextConfig,
    VisionConfig,
)
from gmlx._vendor.mlx_vlm.models.gemma4.processing_gemma4 import (
    Gemma4ImageProcessor,
    Gemma4Processor,
    Gemma4VideoProcessor,
)
from gmlx._vendor.mlx_vlm.models.gemma4.vision import VisionModel

from .gemma4 import Model
from .language import LanguageModel

__all__ = [
    "AudioConfig",
    "AudioEncoder",
    "Gemma4AudioFeatureExtractor",
    "Gemma4ImageProcessor",
    "Gemma4Processor",
    "Gemma4VideoProcessor",
    "LanguageModel",
    "Model",
    "ModelConfig",
    "TextConfig",
    "VisionConfig",
    "VisionModel",
]
