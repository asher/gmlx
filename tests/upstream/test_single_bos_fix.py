"""single_bos_fix: a rendered prompt that opens with BOS holds exactly one
BOS once tokenized, on the server and the generate path, with and without
an image, and the token count routes count the same prompt.

The tokenizer is a GGUF-synthesized one whose encode adds BOS by default,
as the Gemma-4 and Llama-3 GGUFs ask with ``add_bos_token``. Stock mlx-vlm
chooses ``add_special_tokens`` by model type, and its ``process_inputs``
hands the flag only to a processor that names it as a parameter. The fix
leaves a tokenizer that adds more than BOS, embeddings, reranking and a
processor outside the known forwarding types as stock.
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
from gmlx.upstream.single_bos_fix import (  # noqa: E402
    _adds_only_bos,
    _forwards_to_tokenizer,
    install_single_bos_fix,
)

_GENERATION = importlib.import_module("mlx_vlm.server.generation")
_DISPATCH = importlib.import_module("mlx_vlm.generate.dispatch")
_UTILS = importlib.import_module("mlx_vlm.utils")
_ANTHROPIC = importlib.import_module("mlx_vlm.server.anthropic")
_OPENAI = importlib.import_module("mlx_vlm.server.openai")
_EMBEDDINGS = importlib.import_module("mlx_vlm.server.embeddings")
_RERANKING = importlib.import_module("mlx_vlm.server.reranking")
_BOS = "<s>"
_MODEL_TYPES = ["gemma4", "gemma4_text", "diffusion_gemma", "llama", "qwen3_5"]
_GEMMA4 = "mlx_vlm.models.gemma4.processing_gemma4:Gemma4Processor"
_FORWARDING = [
    _GEMMA4,
    "mlx_vlm.models.gemma4_unified.processing_gemma4_unified"
    ":Gemma4UnifiedProcessor",
    "mlx_vlm.models.pixtral.processing_pixtral:PixtralProcessor",
]


def _class(path):
    module, name = path.split(":")
    return getattr(importlib.import_module(module), name)


def _tokenizer(eos=False):
    """Byte-level GGUF tokenizer with BOS id 0 and add_bos_token set, and
    with ``eos`` add_eos_token (EOS id 1) as well."""
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
        "tokenizer.ggml.add_eos_token": eos,
    }, "qwen2")


class _KwargsProcessor(_class(_GEMMA4)):
    """Gemma-4's processor type, which takes ``add_special_tokens`` only
    through ``**kwargs`` and hands it to the tokenizer. The call stands in
    for its image expansion."""

    chat_template = "t"

    def __init__(self, eos=False):  # noqa: D107  skips the stock setup
        self.tokenizer = _tokenizer(eos)

    def __call__(self, images=None, text=None, **kwargs):
        kwargs.pop("return_tensors", None)
        enc = self.tokenizer([text] if isinstance(text, str) else text,
                             **kwargs)
        return {"input_ids": mx.array(enc["input_ids"]),
                "attention_mask": mx.array(enc["attention_mask"]),
                "pixel_values": mx.zeros((len(images), 3, 2, 2))}


class _NamedProcessor:
    """Names ``add_special_tokens`` as a parameter."""

    chat_template = "t"

    def __init__(self, eos=False):
        self.tokenizer = _tokenizer(eos)

    def __call__(self, images=None, text=None, add_special_tokens=True,
                 **kwargs):
        return _KwargsProcessor.__call__(
            self, images, text, add_special_tokens=add_special_tokens,
            **kwargs)


class _StrictProcessor(_NamedProcessor):
    """Takes ``**kwargs`` and rejects a key it does not know, so passing it
    ``add_special_tokens`` raises."""

    def __call__(self, images=None, text=None, **kwargs):
        unknown = set(kwargs) - {"padding", "return_tensors", "padding_side"}
        if unknown:
            raise TypeError(f"unexpected keyword arguments {sorted(unknown)}")
        return _KwargsProcessor.__call__(self, images, text, **kwargs)


_BINDINGS = (_GENERATION, _DISPATCH, _ANTHROPIC, _OPENAI, _EMBEDDINGS,
             _RERANKING)


@pytest.fixture
def fixed(monkeypatch):
    for module in _BINDINGS:
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


def _row(input_ids):
    return [int(t) for t in input_ids.tolist()[0]]


def _bos_ids(input_ids):
    row = _row(input_ids)
    assert row[0] == 0, row
    return row.count(0)


def _stock(processor, prompt, **kwargs):
    """``prepare_inputs`` and ``process_inputs`` as mlx-vlm ships them."""
    prepare = _UTILS.prepare_inputs
    process = _UTILS.process_inputs
    while hasattr(process, "__wrapped__"):
        process = process.__wrapped__
    saved = _UTILS.process_inputs
    _UTILS.process_inputs = process
    try:
        return prepare(processor, prompts=prompt, **kwargs)
    finally:
        _UTILS.process_inputs = saved


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


@pytest.mark.parametrize("path", sorted(_PATHS))
@pytest.mark.parametrize("model_type", ["gemma4_text", "llama"])
def test_a_tokenizer_that_adds_eos_keeps_the_stock_encoding(fixed, path,
                                                            model_type):
    """Encoding without special tokens drops EOS too, so the fix stands
    down for a tokenizer that adds more than BOS."""
    tokenizer = _tokenizer(eos=True)
    processor = _make_text_processor(TokenizerWrapper(tokenizer))
    ids = _row(_PATHS[path](model_type, processor, _BOS + "Hello"))
    assert ids == tokenizer.encode(_BOS + "Hello")
    assert ids[0] == 0 and ids[-1] == 1


# Laguna's BOS is token 2, which is also its EOS. Its chat templates open
# with it.
_LAGUNA_BOS = "\u3008|EOS|\u3009"
_LAGUNA_TEMPLATE = (
    _LAGUNA_BOS + "{% for m in messages %}<{{ m.role }}>{{ m.content }}"
    "</{{ m.role }}>{% endfor %}{% if add_generation_prompt %}<assistant>"
    "{% endif %}")


def _laguna_tokenizer(adds):
    """A byte-level tokenizer with Laguna's BOS at id 2 that adds ``adds``
    to a sequence: "bos", "none", "bos-eos", or "unnamed-bos", a BOS its
    config does not name."""
    from transformers import PreTrainedTokenizerFast

    tokens = (["<unk>", "<pad>", _LAGUNA_BOS, "</assistant>"]
              + sorted(pre_tokenizers.ByteLevel.alphabet()) + ["He", "wo"])
    tokenizer = load_tokenizer_from_gguf({
        "general.architecture": "qwen2",
        "tokenizer.ggml.model": "gpt2",
        "tokenizer.ggml.pre": "qwen2",
        "tokenizer.ggml.tokens": tokens,
        "tokenizer.ggml.merges": ["H e", "w o"],
        "tokenizer.ggml.token_type": [3] * 4 + [1] * (len(tokens) - 4),
        "tokenizer.ggml.bos_token_id": 2,
        "tokenizer.ggml.eos_token_id": 2,
        "tokenizer.ggml.padding_token_id": 1,
        "tokenizer.ggml.add_bos_token": adds != "none",
        "tokenizer.ggml.add_eos_token": adds == "bos-eos",
    }, "qwen2")
    if adds == "unnamed-bos":
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer.backend_tokenizer,
            eos_token=_LAGUNA_BOS, pad_token="<pad>")
    return tokenizer


@pytest.mark.parametrize("path", sorted(_PATHS))
@pytest.mark.parametrize("adds", ["bos", "none", "bos-eos", "unnamed-bos"])
def test_a_laguna_chat_prompt_holds_the_template_bos_alone(fixed, path, adds):
    """mlx-vlm's model type rule leaves Laguna's markers to its chat
    template on the generate path; the server's rule does not list laguna.
    Both paths hold the template's BOS alone, whatever the tokenizer adds."""
    from mlx_vlm.models.laguna.processing_laguna import LagunaProcessor

    processor = LagunaProcessor(_laguna_tokenizer(adds),
                                chat_template=_LAGUNA_TEMPLATE)
    prompt = processor.apply_chat_template(
        [{"role": "user", "content": "Hello"}], tokenize=False,
        add_generation_prompt=True)
    assert prompt.startswith(_LAGUNA_BOS)
    ids = _row(_PATHS[path]("laguna", processor, prompt))
    assert ids[0] == 2 and ids.count(2) == 1


def test_a_laguna_prompt_without_bos_keeps_the_tokenizer_bos(fixed):
    from mlx_vlm.models.laguna.processing_laguna import LagunaProcessor

    processor = LagunaProcessor(_laguna_tokenizer("bos"),
                                chat_template=_LAGUNA_TEMPLATE)
    ids = _row(_serve("laguna", processor, "Hello"))
    assert ids[0] == 2 and ids.count(2) == 1


def _run(tokenizer, prompt, monkeypatch):
    """The ids ``gmlx run`` prefills for the rendered ``prompt``."""
    generation = importlib.import_module("gmlx.gen.generation")
    seen = []

    def stream(model, tok, ids, **kwargs):
        seen.append([int(t) for t in ids])
        return iter(())

    monkeypatch.setattr(importlib.import_module("mlx_lm.generate"),
                        "stream_generate", stream)
    monkeypatch.setattr("gmlx.stream.expert_streaming._resolve_prefill_step",
                        lambda model, step: (None, False))
    generation.generate(object(), tokenizer, prompt,
                        apply_chat_template=False, max_tokens=1)
    (ids,) = seen
    return ids


def _chat(tokenizer, prompt, monkeypatch):
    """The ids ``gmlx chat`` prefills for the rendered ``prompt`` on a turn
    that starts the cache."""
    chat = importlib.import_module("gmlx.tui.chat")
    return [int(t) for t in chat._encode_turn(tokenizer, prompt, True)]


@pytest.mark.parametrize(
    "kind", ["bos-tok", "bos-eos-tok", "unnamed-bos-tok", "unnamed-bos-eos-tok"])
@pytest.mark.parametrize("prompt", [_BOS + "Hello", "Hello"],
                         ids=["bos", "no-bos"])
def test_run_and_chat_encode_one_bos_and_no_eos(fixed, monkeypatch, kind,
                                                prompt):
    """``gmlx run`` and ``gmlx chat`` give a prompt one BOS and no trailing
    EOS, whatever the tokenizer adds and whether or not it names its BOS.
    A tokenizer that adds only a named BOS encodes as the server does. The
    server keeps the stock encoding of the others."""
    backend = _tokenizer("-eos" in kind)
    if kind.startswith("unnamed"):
        backend.bos_token = None
    tokenizer = TokenizerWrapper(backend)
    want = [0] + [int(t) for t in tokenizer.encode(
        prompt.removeprefix(_BOS), add_special_tokens=False)]
    assert _run(tokenizer, prompt, monkeypatch) == want
    assert _chat(tokenizer, prompt, monkeypatch) == want
    if kind == "bos-tok":
        serve = _serve("llama", _make_text_processor(tokenizer), prompt)
        assert _row(serve) == want


def test_only_a_tokenizer_that_adds_bos_alone_drops_it():
    assert _adds_only_bos(_tokenizer())
    assert not _adds_only_bos(_tokenizer(eos=True))
    assert not _adds_only_bos(types.SimpleNamespace(bos_token="<s>"))


@pytest.mark.parametrize("model_type", ["gemma4", "llama"])
def test_a_strict_kwargs_processor_is_called_as_stock(fixed, model_type):
    """A processor outside the known forwarding types never receives the
    flag, even when its call takes ``**kwargs``."""
    image = [Image.new("RGB", (4, 4))]
    ids = _serve(model_type, _StrictProcessor(), _BOS + "Hello", image)
    assert _row(ids) == _row(_stock(_StrictProcessor(), _BOS + "Hello",
                                    images=image)["input_ids"])


@pytest.mark.parametrize("module", [_EMBEDDINGS, _RERANKING],
                         ids=["embeddings", "reranking"])
def test_embeddings_and_reranking_tokenize_as_stock(fixed, module):
    image = [Image.new("RGB", (4, 4))]
    out = module.prepare_inputs(_KwargsProcessor(), images=image,
                                prompts=_BOS + "Hello")
    stock = _stock(_KwargsProcessor(), _BOS + "Hello", images=image)
    assert _row(out["input_ids"]) == _row(stock["input_ids"])
    assert _bos_ids(out["input_ids"]) == 2


_GEMMA4_PROMPT = _BOS + "<|turn>user\nHello<turn|>\n<|turn>model\n"


def _count(route, processor, config, prompt, monkeypatch, rg):
    """The ``input_tokens`` a count route returns for a one-message chat
    whose render is ``prompt``. The resolver fills the model cache as
    stock ``get_cached_model`` does."""
    from fastapi.testclient import TestClient

    app = importlib.import_module("mlx_vlm.server.app").app
    server_runtime = importlib.import_module("mlx_vlm.server.runtime")
    runtime = server_runtime.runtime
    module = _ANTHROPIC if route == "anthropic" else _OPENAI
    monkeypatch.setattr(runtime, "model_cache",
                        server_runtime.ModelCacheRegistry())

    def get_cached_model(*args, **kwargs):
        runtime.model_cache.set("text_generation", {
            "model": None, "processor": processor, "config": config})
        return None, processor, config

    monkeypatch.setattr(module, "get_cached_model", get_cached_model)
    monkeypatch.setattr(module, "apply_chat_template",
                        lambda *a, **k: prompt)
    monkeypatch.setattr(runtime, "response_generator", rg, raising=False)
    client = TestClient(app)
    messages = [{"role": "user", "content": "Hello"}]
    if route == "anthropic":
        r = client.post("/v1/messages/count_tokens", json={
            "model": "m", "max_tokens": 1, "messages": messages})
    else:
        r = client.post("/v1/responses/input_tokens", json={
            "model": "m", "input": messages})
    assert r.status_code == 200, r.text
    return r.json()["input_tokens"]


@pytest.mark.parametrize("route", ["anthropic", "openai"])
@pytest.mark.parametrize("model_type", ["gemma4", "gemma4_text", "qwen3_5"])
@pytest.mark.parametrize("prompt", [_GEMMA4_PROMPT, _GEMMA4_PROMPT[3:]],
                         ids=["bos", "no-bos"])
@pytest.mark.parametrize("eos", [False, True], ids=["bos-tok", "bos-eos-tok"])
@pytest.mark.parametrize("resident", [True, False],
                         ids=["generator", "fallback"])
def test_token_counts_match_the_prompt(fixed, monkeypatch, route, model_type,
                                       prompt, eos, resident):
    """The count routes tokenize the rendered prompt through the server's
    preprocess when a generator is resident, and through their own
    ``prepare_inputs`` otherwise. Both count what generation encodes: the
    Gemma-4 template owns its markers, and every other model adds the
    tokenizer's BOS and, with a tokenizer that adds one, its EOS."""
    tokenizer = _tokenizer(eos)
    tokenizer.chat_template = "t"
    processor = _make_text_processor(TokenizerWrapper(tokenizer))
    config = types.SimpleNamespace(model_type=model_type,
                                   image_token_index=None)
    rg = types.SimpleNamespace(model=_model(model_type), processor=processor)
    rg._cpu_preprocess = types.MethodType(
        _GENERATION.ResponseGenerator._cpu_preprocess, rg)
    count = _count(route, processor, config, prompt, monkeypatch,
                   rg if resident else None)
    assert count == len(_row(_serve(model_type, processor, prompt)))


@pytest.mark.parametrize("module", [_ANTHROPIC, _OPENAI],
                         ids=["anthropic", "openai"])
def test_an_image_prompt_count_matches_the_prompt(fixed, module):
    """The count fallback's call, as the routes make it."""
    image = [Image.new("RGB", (4, 4))]
    out = module.prepare_inputs(_KwargsProcessor(), images=image,
                                prompts=_GEMMA4_PROMPT,
                                image_token_index=None)
    assert _row(out["input_ids"]) == _row(
        _serve("gemma4", _KwargsProcessor(), _GEMMA4_PROMPT, image))


def test_a_processor_without_the_flag_or_kwargs_is_called_as_before(fixed):
    seen = {}

    def process(text=None, images=None, padding=None, return_tensors=None):
        seen.update(text=text, padding=padding)
        return {"input_ids": mx.array([[0, 5]])}

    processor = types.SimpleNamespace(tokenizer=_tokenizer(), process=process)
    _UTILS.process_inputs(processor, _BOS + "Hello", images=[object()],
                          add_special_tokens=True)
    assert seen == {"text": _BOS + "Hello", "padding": True}


@pytest.mark.parametrize("build", ["_synthesize_muse_glimmer_processor",
                                   "_synthesize_glm5next_processor"],
                         ids=["muse_glimmer", "glm5_next"])
def test_gmlx_processors_that_forward_kwargs_are_known(build):
    """The Muse Glimmer and GLM-5.3 processors hand **kwargs to their
    tokenizer, and the fix knows it."""
    vlm = importlib.import_module("gmlx.load.vlm")
    processor = getattr(vlm, build)(_tokenizer(), {
        "clip.vision.patch_size": 14, "clip.vision.spatial_merge_size": 2})
    assert _forwards_to_tokenizer(processor)
    rows = [_row(processor(text=[_BOS + "Hello"],
                           add_special_tokens=flag)["input_ids"])
            for flag in (True, False)]
    assert rows[0] == [0] + rows[1] and rows[1][0] == 0


@pytest.mark.parametrize("path", _FORWARDING)
def test_known_processors_take_the_flag_only_through_kwargs(path):
    cls = _class(path)
    params = inspect.signature(cls.__call__).parameters
    assert "add_special_tokens" not in params
    assert any(p.kind is p.VAR_KEYWORD for p in params.values())
    assert _forwards_to_tokenizer(object.__new__(cls))
    assert not _forwards_to_tokenizer(_StrictProcessor())
