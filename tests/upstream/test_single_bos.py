"""A prompt that opens with BOS gets no second BOS from mlx-vlm.

mlx-vlm passes ``add_special_tokens=True`` to ``prepare_inputs`` for every
model type outside a few gemma families. A chat template that already opens
with BOS then reached the model with two. ``gmlx.upstream.single_bos`` wraps
``prepare_inputs`` so the run path (``generate.dispatch``), the batch path
(``generate.ar``) and serve (``ResponseGenerator._cpu_preprocess``) all
apply the text path's single-BOS rule.
"""

from __future__ import annotations

import importlib
import types

import pytest

pytest.importorskip("mlx_vlm")

import gmlx.upstream.single_bos as single_bos  # noqa: E402

BOS = "<|begin_of_text|>"


@pytest.fixture
def seen(monkeypatch):
    """Install the wrapper over a recording stand-in for the stock function."""
    utils = importlib.import_module("mlx_vlm.utils")
    calls = {}

    def stock(processor, *args, **kwargs):
        calls.update(kwargs)
        return {}

    monkeypatch.setattr(utils, "prepare_inputs", stock)
    for name in ("mlx_vlm.generate.dispatch", "mlx_vlm.generate.ar",
                 "mlx_vlm.server.generation"):
        monkeypatch.setattr(importlib.import_module(name), "prepare_inputs", stock)
    assert single_bos.install()
    assert single_bos.install()  # idempotent: one wrapper layer
    return calls


def _processor(chat_template="t"):
    tokenizer = types.SimpleNamespace(bos_token=BOS)
    return types.SimpleNamespace(tokenizer=tokenizer, chat_template=chat_template)


@pytest.mark.parametrize("prompts,asked,expect", [
    (BOS + "hi", True, False),             # template BOS: no second one
    ([BOS + "a", BOS + "b"], True, False),  # every row opens with BOS
    ([BOS + "a", "b"], True, True),         # one row without: keep specials
    ("hi", True, True),                     # no template BOS: tokenizer adds it
    (BOS + "hi", False, False),             # never turned on
])
def test_wrapper_drops_special_tokens_only_after_a_leading_bos(
        seen, prompts, asked, expect):
    utils = importlib.import_module("mlx_vlm.utils")
    utils.prepare_inputs(_processor(), prompts=prompts, add_special_tokens=asked)
    assert seen["add_special_tokens"] is expect


def test_every_imported_caller_holds_the_wrapper(seen):
    utils = importlib.import_module("mlx_vlm.utils")
    for name in ("mlx_vlm.generate.dispatch", "mlx_vlm.generate.ar",
                 "mlx_vlm.server.generation"):
        assert importlib.import_module(name).prepare_inputs is utils.prepare_inputs


@pytest.mark.parametrize("model_type", ["muse_glimmer", "llama", "diffusion_gemma"])
def test_serve_preprocess_keeps_one_bos(seen, model_type):
    generation = importlib.import_module("mlx_vlm.server.generation")
    config = types.SimpleNamespace(model_type=model_type, image_token_index=None)
    rg = types.SimpleNamespace(model=types.SimpleNamespace(config=config),
                               processor=_processor())
    generation.ResponseGenerator._cpu_preprocess(rg, BOS + "hi")
    assert seen["add_special_tokens"] is False


def test_bos_read_from_a_bare_tokenizer_processor():
    # DiffusionGemma serves with the tokenizer as its processor.
    assert single_bos.opens_with_bos(types.SimpleNamespace(bos_token=BOS), BOS)
    assert not single_bos.opens_with_bos(types.SimpleNamespace(bos_token=None), BOS)
