# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Asher Feldman
# Portions copyright (c) 2025 Apple Inc. (mlx-lm, MIT)
# Vendored from mlx-lm 0.31.3, mlx_lm/models/gemma4.py.
# Source sha256: 4671e4a63cb9849582abac566599a0a85370a46d410f4ad69d81a88788d00fd8
"""mlx-lm 0.31.3 Gemma 4 container, owned by gmlx.

All of ``mlx_lm/models/gemma4.py``. Edits to the upstream file: the
copyright line moves to the header, ``BaseModelArgs`` comes from mlx-lm by
absolute import, and the text model is ``gmlx.models.gemma4.text``.
"""

from dataclasses import dataclass
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten

from gmlx._vendor.mlx_lm.models.base import BaseModelArgs

from . import text as gemma4_text


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "gemma4"
    text_config: dict = None
    vocab_size: int = 262144

    def __post_init__(self):
        if self.text_config is None:
            self.text_config = {}
        self.text_config["vocab_size"] = self.vocab_size
        self.text_config["num_attention_heads"] = self.text_config.get(
            "num_attention_heads", 8
        )
        self.text_config["num_key_value_heads"] = self.text_config.get(
            "num_key_value_heads", 1
        )


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.language_model = gemma4_text.Model(
            gemma4_text.ModelArgs.from_dict(args.text_config)
        )

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        input_embeddings: Optional[mx.array] = None,
        per_layer_inputs: Optional[mx.array] = None,
    ):
        return self.language_model(
            inputs,
            cache=cache,
            input_embeddings=input_embeddings,
            per_layer_inputs=per_layer_inputs,
        )

    def sanitize(self, weights):
        new_weights = {}
        for k, v in weights.items():
            starts_w_model = k.startswith("model.")

            k = k.removeprefix("model.")
            if k.startswith(
                (
                    "vision_tower",
                    "multi_modal_projector",
                    "audio_tower",
                    "embed_audio",
                    "embed_vision",
                )
            ):
                continue

            if not starts_w_model:
                new_weights[k] = v
                continue

            if k.startswith("language_model"):
                k = k.replace("language_model.", "language_model.model.")

            new_weights[k] = v

        return self.language_model.sanitize(new_weights)

    @property
    def layers(self):
        return self.language_model.layers

    @property
    def quant_predicate(self):
        return self.language_model.quant_predicate

    def make_cache(self):
        return self.language_model.make_cache()
