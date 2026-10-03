"""DiffusionGemma prompts carry one BOS on the run path: the GGUF template
opens with BOS and the tokenizer adds one by default. The serve path is
covered by tests/upstream/test_single_bos.py."""

from __future__ import annotations

import importlib

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
