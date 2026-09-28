"""single_bos_fix: a rendered prompt that opens with BOS holds exactly one
BOS once tokenized, on the server and the generate path, with and without
an image.

The tokenizer is a GGUF-synthesized one whose encode adds BOS by default,
as the Gemma-4 and Llama-3 GGUFs ask with ``add_bos_token``. Stock mlx-vlm
chooses ``add_special_tokens`` by model type, and its ``process_inputs``
hands the flag only to a processor that names it as a parameter.
"""

from __future__ import annotations

import importlib
import inspect
import types

import pytest

pytest.importorskip("mlx_vlm")

import mlx.core as mx  # noqa: E402
from mlx_lm.tokenizer_utils import TokenizerWrapper  # noqa: E402
from PIL import Image  # noqa: E402
from tokenizers import pre_tokenizers  # noqa: E402

from gmlx.load.tokenizer import load_tokenizer_from_gguf  # noqa: E402
from gmlx.serve.bridge_vlm import _make_text_processor  # noqa: E402
from gmlx.upstream.single_bos_fix import install_single_bos_fix  # noqa: E402

_GENERATION = importlib.import_module("mlx_vlm.server.generation")
_DISPATCH = importlib.import_module("mlx_vlm.generate.dispatch")
_UTILS = importlib.import_module("mlx_vlm.utils")
_BOS = "<s>"
_MODEL_TYPES = ["gemma4", "gemma4_text", "diffusion_gemma", "llama", "qwen3_5"]


def _tokenizer():
    """Byte-level GGUF tokenizer with BOS id 0 and add_bos_token set."""
    tokens = ["<s>", "</s>", "<pad>"] + sorted(
        pre_tokenizers.ByteLevel.alphabet()) + ["He", "wo"]
    return load_tokenizer_from_gguf({
        "general.architecture": "qwen2",
        "tokenizer.ggml.model": "gpt2",
        "tokenizer.ggml.pre": "qwen2",
        "tokenizer.ggml.tokens": tokens,
        "tokenizer.ggml.merges": ["H e", "w o"],
        "tokenizer.ggml.token_type": [3, 3, 3] + [1] * (len(tokens) - 3),
        "tokenizer.ggml.bos_token_id": 0,
        "tokenizer.ggml.eos_token_id": 1,
        "tokenizer.ggml.padding_token_id": 2,
        "tokenizer.ggml.add_bos_token": True,
    }, "qwen2")


class _KwargsProcessor:
    """Takes ``add_special_tokens`` only through ``**kwargs`` and hands it
    to the tokenizer, as Gemma-4's processor does."""

    chat_template = "t"

    def __init__(self):
        self.tokenizer = _tokenizer()

    def __call__(self, images=None, text=None, **kwargs):
        kwargs.pop("return_tensors", None)
        enc = self.tokenizer([text] if isinstance(text, str) else text,
                             **kwargs)
        return {"input_ids": mx.array(enc["input_ids"]),
                "attention_mask": mx.array(enc["attention_mask"]),
                "pixel_values": mx.zeros((len(images), 3, 2, 2))}


class _NamedProcessor(_KwargsProcessor):
    """Names ``add_special_tokens`` as a parameter."""

    def __call__(self, images=None, text=None, add_special_tokens=True,
                 **kwargs):
        return super().__call__(images, text,
                                add_special_tokens=add_special_tokens,
                                **kwargs)


@pytest.fixture
def fixed(monkeypatch):
    for module in (_GENERATION, _DISPATCH):
        monkeypatch.setattr(module, "prepare_inputs", module.prepare_inputs)
    monkeypatch.setattr(_UTILS, "process_inputs", _UTILS.process_inputs)
    install_single_bos_fix()
    install_single_bos_fix()


def _model(model_type):
    return types.SimpleNamespace(config=types.SimpleNamespace(
        model_type=model_type, image_token_index=None))


def _serve(model_type, processor, prompt, images=None):
    rg = types.SimpleNamespace(model=_model(model_type), processor=processor)
    out = _GENERATION.ResponseGenerator._cpu_preprocess(rg, prompt, images)
    return out["input_ids"]


def _generate(model_type, processor, prompt, images=None):
    return _DISPATCH._prepare_generation_inputs(
        _model(model_type), processor, prompt, images, None, None, {})[0]


def _bos_ids(input_ids):
    row = [int(t) for t in input_ids.tolist()[0]]
    assert row[0] == 0, row
    return row.count(0)


_PATHS = {"serve": _serve, "generate": _generate}


@pytest.mark.parametrize("path", sorted(_PATHS))
@pytest.mark.parametrize("model_type", _MODEL_TYPES)
@pytest.mark.parametrize("prompt", [_BOS + "Hello", "Hello"])
def test_text_prompt_holds_one_bos(fixed, path, model_type, prompt):
    processor = _make_text_processor(TokenizerWrapper(_tokenizer()))
    ids = _PATHS[path](model_type, processor, prompt)
    assert _bos_ids(ids) == 1


@pytest.mark.parametrize("path", sorted(_PATHS))
@pytest.mark.parametrize("model_type", ["gemma4", "llama"])
@pytest.mark.parametrize("processor", [_KwargsProcessor, _NamedProcessor])
def test_image_prompt_holds_one_bos(fixed, path, model_type, processor):
    ids = _PATHS[path](model_type, processor(), _BOS + "Hello",
                       [Image.new("RGB", (4, 4))])
    assert _bos_ids(ids) == 1


def test_a_processor_without_the_flag_or_kwargs_is_called_as_before(fixed):
    seen = {}

    def process(text=None, images=None, padding=None, return_tensors=None):
        seen.update(text=text, padding=padding)
        return {"input_ids": mx.array([[0, 5]])}

    processor = types.SimpleNamespace(tokenizer=_tokenizer(), process=process)
    _UTILS.process_inputs(processor, _BOS + "Hello", images=[object()],
                          add_special_tokens=True)
    assert seen == {"text": _BOS + "Hello", "padding": True}


def test_gemma4_processor_takes_the_flag_only_through_kwargs():
    from mlx_vlm.models.gemma4.processing_gemma4 import Gemma4Processor

    params = inspect.signature(Gemma4Processor.__call__).parameters
    assert "add_special_tokens" not in params
    assert any(p.kind is p.VAR_KEYWORD for p in params.values())
