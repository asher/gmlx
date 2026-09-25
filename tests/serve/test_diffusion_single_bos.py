"""DiffusionGemma prompts carry one BOS on the run and serve paths: the
GGUF template opens with BOS and the tokenizer adds one by default."""

from __future__ import annotations

import importlib
import types

import pytest

pytest.importorskip("mlx_vlm")

BOS = "<bos>"


class _Tok:
    """A tokenizer whose default encode prepends BOS (id 2)."""

    bos_token = BOS

    def encode(self, text, add_special_tokens=True):
        ids = [2 if w == BOS else 10 + len(w) for w in text.replace(BOS, f" {BOS} ").split()]
        return ([2] + ids) if add_special_tokens else ids


def test_run_path_encodes_a_rendered_prompt_with_one_bos(monkeypatch):
    import gmlx.gen.diffusion as diffusion

    # mlx_vlm.generate is also a function name on the package, so the
    # submodule is reached through importlib.
    upstream = importlib.import_module("mlx_vlm.generate.diffusion")

    seen = {}

    def fake_generate(model, processor, backend, input_ids, *args, **kwargs):
        seen["ids"] = input_ids.tolist()[0]
        return iter(())

    monkeypatch.setattr(diffusion, "_diffusion_io", lambda tok: (None, None, set()))
    monkeypatch.setattr(upstream, "stream_diffusion_generate", fake_generate)
    list(diffusion.stream(object(), _Tok(), BOS + " system hi"))
    assert seen["ids"].count(2) == 1 and seen["ids"][0] == 2


def _fake_rg(model_type):
    config = types.SimpleNamespace(model_type=model_type, image_token_index=None)
    return types.SimpleNamespace(
        model=types.SimpleNamespace(config=config),
        processor=types.SimpleNamespace(bos_token=BOS, chat_template="t"))


@pytest.mark.parametrize("model_type,prompt,special", [
    ("diffusion_gemma", BOS + "hi", False),
    ("diffusion_gemma", "hi", True),
    ("gemma4", BOS + "hi", False),
    ("llama", BOS + "hi", True),
])
def test_serve_path_adds_special_tokens_only_without_a_leading_bos(
        monkeypatch, model_type, prompt, special):
    generation = pytest.importorskip("mlx_vlm.server.generation")

    from gmlx.serve.patches.chat_behavior import install_diffusion_single_bos

    cls = generation.ResponseGenerator
    monkeypatch.setattr(cls, "_cpu_preprocess", cls._cpu_preprocess)
    seen = {}

    def fake_prepare(processor, **kwargs):
        seen.update(kwargs)
        return {}

    monkeypatch.setattr(generation, "prepare_inputs", fake_prepare)
    install_diffusion_single_bos()
    install_diffusion_single_bos()
    cls._cpu_preprocess(_fake_rg(model_type), prompt)
    assert seen["add_special_tokens"] is special
