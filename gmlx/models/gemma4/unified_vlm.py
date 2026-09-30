# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Asher Feldman
"""The module a gemma4_unified VLM builds from on every gmlx route.

``build_vlm_model`` builds from it, and mlx-vlm's own loader does too once
``install_owned_vlm_resolver`` hands it back for the stock package. It
carries every public name of ``mlx_vlm.models.gemma4_unified``: the owned
container and language model, and the stock configs, vision embedder and
processors. ``update_module_configs`` finds the four config classes here.
"""

from gmlx._vendor.mlx_vlm.models.gemma4_unified import (
    AudioConfig,
    Gemma4UnifiedAudioFeatureExtractor,
    Gemma4UnifiedImageProcessor,
    Gemma4UnifiedProcessor,
    Gemma4UnifiedVideoProcessor,
    ImageProcessor,
    ModelConfig,
    TextConfig,
    VisionConfig,
    VisionEmbedder,
    VisionModel,
)

from .language import LanguageModel
from .unified import Model

__all__ = [
    "AudioConfig",
    "Gemma4UnifiedAudioFeatureExtractor",
    "Gemma4UnifiedImageProcessor",
    "Gemma4UnifiedProcessor",
    "Gemma4UnifiedVideoProcessor",
    "ImageProcessor",
    "LanguageModel",
    "Model",
    "ModelConfig",
    "TextConfig",
    "VisionConfig",
    "VisionEmbedder",
    "VisionModel",
]
