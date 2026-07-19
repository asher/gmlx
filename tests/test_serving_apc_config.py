"""The APC prompt cache reads ``self.model.config.image_token_id`` off the inner
``language_model``; a text GGUF's inner LM (``text_only.LanguageModel``) carries
``model_type`` but no ``.config``, so an enabled cache used to crash generation with
``'LanguageModel' object has no attribute 'config'``. ``_ensure_inner_config`` mirrors
the config onto the inner LM. These are the regression guards (CPU-only)."""
import types

import mlx.nn as nn

from gmlx.server_bridge_vlm import _AttrDict, _ensure_inner_config


def test_attaches_config_to_nn_module_inner_lm():
    inner = nn.Linear(2, 2)                     # an mlx Module, like the real inner LM
    inner.model_type = "qwen3"                  # has model_type, no config (the gap)
    model = types.SimpleNamespace(language_model=inner)
    cfg = _AttrDict({"model_type": "qwen3", "vocab_size": 1000})

    _ensure_inner_config(model, cfg)

    # APC's exact read must now resolve to None instead of raising
    assert getattr(inner.config, "image_token_id", None) is None
    assert getattr(inner.config, "image_token_index", None) is None
    assert inner.config.model_type == "qwen3"   # attribute-style still works
    assert inner.config.get("vocab_size") == 1000
    # stored as a plain attribute, NOT injected into the parameter tree
    assert "config" not in inner.parameters()


def test_noop_when_inner_already_has_config():
    existing = _AttrDict({"model_type": "gemma4"})
    inner = types.SimpleNamespace(config=existing)   # a VLM LM already has config
    model = types.SimpleNamespace(language_model=inner)

    _ensure_inner_config(model, _AttrDict({"model_type": "other"}))

    assert inner.config is existing                  # left untouched


def test_noop_without_language_model():
    model = types.SimpleNamespace(config={})          # no inner LM at all
    _ensure_inner_config(model, _AttrDict({}))         # must not raise


def test_defensive_on_slotted_inner():
    model = types.SimpleNamespace(language_model=object())   # can't take attributes
    _ensure_inner_config(model, _AttrDict({}))               # swallowed, no raise
