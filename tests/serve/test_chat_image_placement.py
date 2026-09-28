"""/v1/chat/completions renders each image in the message that sent it,
as /v1/responses does, so a turn's prompt starts with the previous turn's
prompt. Stock chat completions reduces each message to its text and the
template places every image in the last user message."""

from __future__ import annotations

import importlib
import types
from pathlib import Path

import pytest

pytest.importorskip("mlx_vlm")

from transformers.utils.chat_template_utils import render_jinja_template  # noqa: E402

from gmlx.serve.patches import _common as sp_common  # noqa: E402
from gmlx.serve.patches.chat_behavior import install_role_normalization  # noqa: E402
from gmlx.serve.patches.render import (  # noqa: E402
    install_chat_image_placement,
    install_faithful_history,
)

_OPENAI = importlib.import_module("mlx_vlm.server.openai")
_APP = importlib.import_module("mlx_vlm.server.app")
_RUNTIME = importlib.import_module("mlx_vlm.server.runtime").runtime
_FIXTURES = Path(__file__).parents[1] / "fixtures"

# model_type: template fixture, the placeholder one image renders as, and
# the text that closes a user message.
_MODELS = {
    "gemma4": ("gemma4_template.jinja", "<|image|>", "<turn|>\n"),
    "qwen3_5": ("qwen35_template.jinja",
                "<|vision_start|><|image_pad|><|vision_end|>",
                "<|im_end|>\n"),
}


class _Processor:
    """Renders the fixture template the way a tokenizer does."""

    def __init__(self, model_type):
        self.chat_template = (_FIXTURES / _MODELS[model_type][0]).read_text()

    def apply_chat_template(self, messages, tokenize=False,
                            add_generation_prompt=True, **kw):
        out = render_jinja_template(
            [messages], chat_template=self.chat_template,
            add_generation_prompt=add_generation_prompt, **kw)
        rendered = out[0] if isinstance(out, tuple) else out
        return rendered[0] if isinstance(rendered, list) else rendered


class _Tok:
    text, finish_reason, token_count = "ok", "stop", 1


class _RG:
    """Records the prompt and images the route hands to generation."""

    def __init__(self):
        self.calls = []

    def generate(self, prompt, images=None, audio=None, args=None,
                 videos=None):
        self.calls.append((prompt, list(images or ())))
        return types.SimpleNamespace(prompt_tokens=1, uid=1), iter([_Tok()])


@pytest.fixture
def route(monkeypatch):
    """The chat route with the gmlx render wraps, on a stub model."""
    routes = sp_common._snapshot_routes(_APP.app)
    for target in sp_common._render_target_modules():
        monkeypatch.setattr(target, "apply_chat_template",
                            target.apply_chat_template)
    monkeypatch.setattr(_OPENAI, "extract_text_from_content",
                        _OPENAI.extract_text_from_content)
    monkeypatch.setattr(_OPENAI, "_normalize_instruction_messages",
                        _OPENAI._normalize_instruction_messages)
    rg = _RG()
    monkeypatch.setattr(_RUNTIME, "response_generator", rg, raising=False)
    install_faithful_history()
    install_role_normalization()
    install_chat_image_placement()
    from fastapi.testclient import TestClient

    client = TestClient(_APP.app)

    def render(model_type, messages):
        monkeypatch.setattr(_OPENAI, "get_cached_model", lambda *a, **k: (
            None, _Processor(model_type), {"model_type": model_type}))
        r = client.post("/v1/chat/completions", json={
            "model": "m", "messages": messages, "max_tokens": 1})
        assert r.status_code == 200, r.text
        return rg.calls.pop()

    yield render
    sp_common._restore_routes(_APP.app, routes)


def _image(name):
    return {"type": "image_url", "image_url": {"url": f"http://img/{name}"}}


def _text(t):
    return {"type": "text", "text": t}


def _turns():
    turn1 = [{"role": "system", "content": "Be brief."},
             {"role": "user", "content": [
                 _text("Here is the first picture."), _image("a"),
                 _text("Describe it.")]}]
    turn2 = turn1 + [
        {"role": "assistant", "content": "A red car."},
        {"role": "user", "content": [
            _text("And a second one."), _image("b"),
            _text("How do they differ?")]}]
    return turn1, turn2


@pytest.mark.parametrize("model_type", sorted(_MODELS))
def test_each_image_renders_in_its_own_message(route, model_type):
    image = _MODELS[model_type][1]
    turn1, turn2 = _turns()
    prompt1, images1 = route(model_type, turn1)
    prompt2, images2 = route(model_type, turn2)
    assert images1 == ["http://img/a"]
    assert images2 == ["http://img/a", "http://img/b"]
    # Image A sits before the first question, image B before the second.
    first, second = prompt2.split("A red car.")
    assert first.count(image) == 1 and second.count(image) == 1
    assert first.index(image) < first.index("Describe it.")
    assert second.index(image) < second.index("How do they differ?")
    # Turn 2 renders turn 1 as it was, so its prompt starts with turn 1's
    # through the first question. Qwen3.5 closes an empty think block in
    # the generation prompt and not in history, so the check stops there.
    end = _MODELS[model_type][2]
    assert prompt2.startswith(prompt1[:prompt1.rindex(end) + len(end)])
    if model_type == "gemma4":
        assert prompt2.startswith(prompt1)


@pytest.mark.parametrize("model_type", sorted(_MODELS))
def test_images_in_one_message_keep_their_order(route, model_type):
    image = _MODELS[model_type][1]
    prompt, images = route(model_type, [{"role": "user", "content": [
        _image("a"), _text("then"), _image("b"), _text("compare them.")]}])
    assert images == ["http://img/a", "http://img/b"]
    assert prompt.count(image) == 2


@pytest.mark.parametrize("model_type", sorted(_MODELS))
def test_a_text_chat_renders_as_before(route, monkeypatch, model_type):
    """Text parts and plain strings render byte for byte as the stock
    route renders them."""
    messages = [{"role": "system", "content": "Be brief."},
                {"role": "user", "content": [_text("Hi"), _text("there.")]},
                {"role": "assistant", "content": "Hello."},
                {"role": "user", "content": "What is 2 + 2?"}]
    patched = route(model_type, messages)
    extract = _OPENAI.extract_text_from_content
    monkeypatch.setattr(_OPENAI, "extract_text_from_content",
                        getattr(extract, "__wrapped__", extract))
    stock = route(model_type, messages)
    assert patched == stock and patched[1] == []
