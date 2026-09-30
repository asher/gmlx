"""The owned Gemma 4 text model on the mlx-lm text-model contract.

Every gmlx text build of Gemma 4 resolves its classes here through
``get_model_classes``: the GGUF loader, and through it chat, run, bench,
serve, train and distill, and the MLX checkpoint loads of distill and of
``vlm_text_only``. A gemma4 container config builds ``lm_container``, the
vendored mlx-lm container, around this model. ``Model`` is the owned
``LanguageModel`` called the way mlx-lm's gemma4_text ``Model`` is called:
``model(inputs, cache, input_embeddings, per_layer_inputs)`` returns the
logits, and the embeddings are scaled by ``embed_scale`` as mlx-lm scales
them. It holds the text model at ``model.*``, the parameter tree of
mlx-lm's class.

``ModelArgs.from_dict`` builds the stock mlx-vlm ``TextConfig``, which has
every field of mlx-lm's ``ModelArgs``. Where a config leaves out
``rope_parameters`` it takes mlx-lm's default, whose full-attention
partial rotary factor is 0.25 where mlx-vlm's is 1.0. It sets
``use_bidirectional_attention`` to None: the text route carries no media,
and mlx-lm's ``ModelArgs`` has no such field.
"""

from __future__ import annotations

import copy
from typing import Any, Optional

import mlx.core as mx
from gmlx._vendor.mlx_vlm.models.gemma4.config import TextConfig

from . import lm_sanitize
from .language import LanguageModel

# mlx-lm gemma4_text ModelArgs.__post_init__.
_MLX_LM_ROPE_PARAMETERS = {
    "full_attention": {
        "partial_rotary_factor": 0.25,
        "rope_theta": 1000000.0,
        "rope_type": "proportional",
    },
    "sliding_attention": {
        "partial_rotary_factor": 1.0,
        "rope_theta": 10000.0,
        "rope_type": "default",
    },
}


class ModelArgs:
    """The config bridge from a text config dict to the owned model's
    ``TextConfig``."""

    @classmethod
    def from_dict(cls, params) -> TextConfig:
        params = dict(params or {})
        if params.get("rope_parameters") is None:
            params["rope_parameters"] = copy.deepcopy(_MLX_LM_ROPE_PARAMETERS)
        params["use_bidirectional_attention"] = None
        return TextConfig.from_dict(params)


class Model(LanguageModel):
    """The owned Gemma 4 language model with mlx-lm's text call."""

    def __init__(self, args: TextConfig):
        if not getattr(args, "tie_word_embeddings", True):
            raise ValueError(
                "the owned Gemma 4 text model ties its output head to the "
                "token embedding; the config sets tie_word_embeddings false"
            )
        super().__init__(args)
        self.args = args
        self.tie_word_embeddings = True

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        input_embeddings: Optional[mx.array] = None,
        per_layer_inputs: Optional[mx.array] = None,
    ):
        if input_embeddings is not None:
            # mlx-lm recovers the ids of a PLE model by inverting the
            # embedding, which a KQuantEmbedding holding wire bytes cannot do.
            if (inputs is None and per_layer_inputs is None
                    and self.model.hidden_size_per_layer_input):
                raise ValueError(
                    "the Gemma 4 per-layer inputs need the token ids or "
                    "per_layer_inputs beside input_embeddings"
                )
            input_embeddings = input_embeddings * self.model.embed_scale
        return super().__call__(
            inputs,
            inputs_embeds=input_embeddings,
            cache=cache,
            per_layer_inputs=per_layer_inputs,
        ).logits

    def sanitize(self, weights):
        """mlx-lm's gemma4_text sanitize, then drop the k and v weights of
        KV-shared layers, which build none. Raw HF checkpoints carry them."""
        weights = lm_sanitize.sanitize(self, weights)
        return {
            k: v for k, v in weights.items()
            if not self._is_unused_shared_kv_weight(
                k if k.startswith("language_model.") else f"language_model.{k}")
        }


def get_model_classes(config: dict) -> tuple[Any, Any]:
    """The (Model, ModelArgs) pair a gmlx text build uses for ``config``:
    the owned classes for gemma4_text and the vendored container for
    gemma4, else mlx-lm's. ``mlx_lm.utils.load_model`` passes ``config`` by
    name. Every architecture builds through here, so the pair is typed
    Any, as mlx-lm's own resolver returns it."""
    model_type = config.get("model_type")
    if model_type == "gemma4_text":
        return Model, ModelArgs
    if model_type == "gemma4":
        from . import lm_container

        return lm_container.Model, lm_container.ModelArgs
    from gmlx._vendor.mlx_lm.utils import _get_classes

    return _get_classes(config)
