# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Asher Feldman
"""SenseNova Loop with its vision tower (mmproj ``qwen3vl_merger``).

The tower is the Qwen3.5 one with deepstack off, so the container is the
copied qwen3_5 ``Model`` with the looped language model in place of the
Qwen3.5 one. The language model carries the speculative hooks, so an
mmproj load and the native MTP head compose.
"""

from __future__ import annotations

import mlx.nn as nn

from gmlx._vendor.mlx_vlm.models.qwen3_5.config import ModelConfig, VisionConfig
from gmlx._vendor.mlx_vlm.models.qwen3_5.qwen3_5 import Model as _Q35Model
from gmlx._vendor.mlx_vlm.models.qwen3_5.vision import VisionModel

from .language import SenseNovaLoopLanguageModel as LanguageModel
from .language import TextConfig

__all__ = ["Model", "ModelConfig", "TextConfig", "VisionConfig",
           "LanguageModel", "VisionModel"]


class Model(_Q35Model):
    def __init__(self, config: ModelConfig):
        nn.Module.__init__(self)
        self.config = config
        self.vision_tower = VisionModel(config.vision_config)
        self.language_model = LanguageModel(config.text_config, config)
