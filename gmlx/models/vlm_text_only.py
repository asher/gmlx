"""Text-only Model adapter vendored from mlx-vlm 0.6.4 models/text_only.py.

mlx-vlm 0.6.15 removed the wrapper; gmlx keeps it as the stable seam for
wrapping GGUF-synthesized text models in the vlm interface.

A forward called with ``logits_to_keep`` on one of the copy's mlx-lm models,
or on the owned Gemma 4 text model, runs the head on the last columns only:
the model's trunk (``model``) hands the head just those columns. A prefill
forward of many columns then costs one column of the vocabulary projection,
and the batch engine runs the last prompt chunk and the first token in that
one forward. A trunk whose ``__call__`` takes ``keep_last`` gets the count,
and skips work for the other columns where it can: in the Gemma 4 text
trunks, every layer after the last cache write runs on the kept columns
only.
"""

import contextlib
import functools
import inspect
import threading
from dataclasses import dataclass
from typing import Any, Dict, Optional

import mlx.core as mx
import mlx.nn as nn

from gmlx._vendor.mlx_vlm.models.base import InputEmbeddingsFeatures, LanguageModelOutput

_is_text_model_arch = True

# The copy's mlx-lm model code: each Model runs its trunk at ``model`` and
# then only per-column steps (the head, a logit softcap) on its output.
_COPY_MODELS = "gmlx._vendor.mlx_lm.models."
# The owned text models that run the same way.
_OWNED_MODELS = frozenset({"gmlx.models.gemma4.text"})


class _KeptColumns:
    """Stands in for a wrapped model's trunk during one forward on the
    thread that armed it, and returns the last ``keep`` columns of the
    trunk's output. Attribute reads reach the trunk, so the head still finds
    a tied embedding through it. Other threads get the whole output."""

    def __init__(self, trunk, keep: int):
        self.__dict__.update(
            _trunk=trunk, _keep=keep, _thread=threading.get_ident())

    def __getattr__(self, name):
        return getattr(self.__dict__["_trunk"], name)

    def __call__(self, *args, **kwargs):
        mine = threading.get_ident() == self._thread
        if mine and _takes_keep_last(self._trunk):
            kwargs["keep_last"] = self._keep
        out = self._trunk(*args, **kwargs)
        if (mine and isinstance(out, mx.array) and out.ndim == 3
                and out.shape[1] > self._keep):
            out = out[:, -self._keep:, :]
        return out


@functools.lru_cache(maxsize=None)
def _class_takes_keep_last(cls) -> bool:
    try:
        return "keep_last" in inspect.signature(cls.__call__).parameters
    except (TypeError, ValueError):
        return False


def _takes_keep_last(trunk) -> bool:
    """True when the trunk can skip work for columns the caller drops, as
    the Gemma 4 text trunk does after its last cache write."""
    return _class_takes_keep_last(type(trunk))


@dataclass
class AttributeConfig:
    """Attribute-style config wrapper."""

    values: Dict[str, Any]

    def __post_init__(self):
        for key, value in self.values.items():
            setattr(self, key, value)

    def to_dict(self):
        return dict(self.values)


class TextConfig(AttributeConfig):
    @classmethod
    def from_dict(cls, params):
        return cls(dict(params or {}))


class ModelConfig(AttributeConfig):
    @classmethod
    def from_dict(cls, params):
        params = dict(params or {})
        from gmlx.models.gemma4.text import get_model_classes

        model_class, model_args_class = get_model_classes(params)
        return cls(params, model_class, model_args_class)

    def __init__(self, values, model_class, model_args_class):
        self.model_class = model_class
        self.model_args_class = model_args_class
        super().__init__(values)
        self.text_config = TextConfig.from_dict(values.get("text_config", {}))

    def __post_init__(self):
        for key, value in self.values.items():
            if key in {
                "text_config",
                "vision_config",
                "audio_config",
                "projector_config",
                "perceiver_config",
            }:
                continue
            setattr(self, key, value)


class LanguageModel(nn.Module):
    """Adapter that makes an mlx-lm model return mlx-vlm language outputs."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self._model = model
        self.model_type = getattr(model, "model_type", None)

    @property
    def model(self):
        return getattr(self._model, "model", self._model)

    @property
    def supports_logits_to_keep(self) -> bool:
        """True when a forward can hand the head only the kept columns: the
        wrapped model is one of the copy's mlx-lm models or an owned text
        model in ``_OWNED_MODELS``, with its trunk at ``model``."""
        module = type(self._model).__module__
        return ((module.startswith(_COPY_MODELS) or module in _OWNED_MODELS)
                and isinstance(getattr(self._model, "model", None), nn.Module))

    @contextlib.contextmanager
    def _keeping(self, keep):
        # The instance dict, not setattr: an mlx Module's setattr of a
        # non-array value drops the child module from its tree. The trunk
        # stays a child, and the entry shadows it for this forward only.
        attrs = vars(self._model)
        if not keep or "model" in attrs or not self.supports_logits_to_keep:
            yield False
            return
        attrs["model"] = _KeptColumns(self._model.model, int(keep))
        try:
            yield True
        finally:
            attrs.pop("model", None)

    @property
    def layers(self):
        if hasattr(self._model, "layers"):
            return self._model.layers
        return []

    def make_cache(self):
        if hasattr(self._model, "make_cache"):
            return self._model.make_cache()

        from gmlx.cache.kv_cache import KVCache

        return [KVCache() for _ in self.layers]

    def shard(self, group: Optional[mx.distributed.Group] = None):
        if not hasattr(self._model, "shard"):
            raise ValueError("The underlying mlx-lm model does not support sharding")
        return self._model.shard(group)

    @staticmethod
    def _accepted_params(callable_obj) -> set[str]:
        try:
            return set(inspect.signature(callable_obj).parameters)
        except (TypeError, ValueError):
            return set()

    def _token_embedding(self):
        candidates = (
            (self.model, "embed_tokens"),
            (self.model, "wte"),
            (self.model, "embeddings"),
            (self.model, "tok_embeddings"),
            (self.model, "word_embeddings"),
            (self._model, "embed_tokens"),
            (self._model, "wte"),
            (self._model, "embeddings"),
        )
        for module, attr in candidates:
            if hasattr(module, attr):
                return getattr(module, attr)
        return None

    def input_embeds(self, input_ids: mx.array) -> mx.array:
        embedding = self._token_embedding()
        if embedding is None:
            raise ValueError(
                "The wrapped mlx-lm model does not expose token embeddings."
            )
        return embedding(input_ids)

    def _project_hidden_states(self, hidden_states):
        if hasattr(self._model, "lm_head"):
            return self._model.lm_head(hidden_states)

        embedding = self._token_embedding()
        if embedding is not None and hasattr(embedding, "as_linear"):
            return embedding.as_linear(hidden_states)

        return None

    def _call_inner_with_embeddings(self, inputs, cache=None, input_embeddings=None):
        inner = self.model
        if inner is self._model:
            return None

        trunk = inner._trunk if isinstance(inner, _KeptColumns) else inner
        params = self._accepted_params(trunk.__call__)
        if "input_embeddings" not in params:
            return None

        call_kwargs = {"input_embeddings": input_embeddings}
        if "cache" in params:
            call_kwargs["cache"] = cache

        hidden_states = inner(inputs, **call_kwargs)
        return self._project_hidden_states(hidden_states)

    def _call_model(self, inputs, cache=None, **kwargs):
        params = self._accepted_params(self._model.__call__)
        input_embeddings = kwargs.get("inputs_embeds", kwargs.get("input_embeddings"))

        call_kwargs = {}
        if "cache" in params:
            call_kwargs["cache"] = cache

        for source, target in (
            ("mask", "mask"),
            ("inputs_embeds", "input_embeddings"),
            ("input_embeddings", "input_embeddings"),
            ("per_layer_inputs", "per_layer_inputs"),
        ):
            if target in params and source in kwargs and kwargs[source] is not None:
                call_kwargs[target] = kwargs[source]

        if "input_embeddings" not in params and input_embeddings is not None:
            logits = self._call_inner_with_embeddings(
                inputs, cache=cache, input_embeddings=input_embeddings
            )
            if logits is not None:
                return logits

        return self._model(inputs, **call_kwargs)

    def __call__(self, inputs: mx.array, cache=None, **kwargs):
        keep = kwargs.pop("logits_to_keep", None)
        with self._keeping(keep) as kept:
            logits = self._call_model(inputs, cache=cache, **kwargs)
        # A model whose forward never ran its trunk keeps the contract too.
        if kept and logits.ndim == 3 and logits.shape[1] > int(keep):
            logits = logits[:, -int(keep):, :]
        return LanguageModelOutput(logits=logits)


class Model(nn.Module):
    """VLM-shaped wrapper around a text-only model."""

    _is_text_model = True

    def __init__(self, model_or_config, config: Optional[Dict[str, Any]] = None):
        super().__init__()
        if isinstance(model_or_config, ModelConfig):
            self.config = model_or_config
            model_args = self.config.model_args_class.from_dict(self.config.values)
            model = self.config.model_class(model_args)
        else:
            model = model_or_config
            self.config = AttributeConfig(dict(config or {}))
        self.language_model = LanguageModel(model)

    def sanitize(self, weights):
        model = self.language_model._model
        if hasattr(model, "sanitize"):
            return model.sanitize(weights)
        return weights

    def load_weights(self, weights, *args, **kwargs):
        return self.language_model._model.load_weights(weights, *args, **kwargs)

    def get_input_embeddings(
        self,
        input_ids: Optional[mx.array] = None,
        pixel_values: Optional[mx.array] = None,
        **kwargs,
    ):
        if pixel_values is not None:
            raise ValueError("Text-only models do not accept image inputs.")
        if (
            kwargs.get("input_features") is not None
            or kwargs.get("audio_values") is not None
        ):
            raise ValueError("Text-only models do not accept audio inputs.")
        if input_ids is None:
            raise ValueError("input_ids are required for text-only models.")
        return InputEmbeddingsFeatures(
            inputs_embeds=self.language_model.input_embeds(input_ids)
        )

    def __call__(
        self,
        input_ids: mx.array,
        pixel_values: Optional[mx.array] = None,
        attention_mask: Optional[mx.array] = None,
        **kwargs,
    ):
        if pixel_values is not None:
            raise ValueError("Text-only models do not accept image inputs.")
        if attention_mask is not None:
            kwargs.setdefault("mask", attention_mask)
        return self.language_model(input_ids, **kwargs)

    def make_cache(self):
        return self.language_model.make_cache()

    def shard(self, group: Optional[mx.distributed.Group] = None):
        return self.language_model.shard(group)

    @property
    def layers(self):
        return self.language_model.layers


TextOnlyModel = Model
