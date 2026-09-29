"""/v1/messages renders a user message's tool results before its text, in
block order, as the Messages API orders them. The stock conversion emits the
text first, between the assistant's tool call and the tool response, and the
Gemma-4 templates then drop the tool response."""

from __future__ import annotations

import importlib
import types
from pathlib import Path

import pytest

pytest.importorskip("mlx_vlm")

from transformers.utils.chat_template_utils import render_jinja_template  # noqa: E402

from gmlx.load.tokenizer import bundled_chat_template, finish_gguf_tokenizer  # noqa: E402
from gmlx.serve.patches import _common as sp_common  # noqa: E402
from gmlx.serve.patches.chat_behavior import install_role_normalization  # noqa: E402
from gmlx.serve.patches.messages_route import install_messages_route  # noqa: E402
from gmlx.serve.patches.render import (  # noqa: E402
    install_chat_image_placement,
    install_faithful_history,
)

_ANTHROPIC = importlib.import_module("mlx_vlm.server.anthropic")
_PROMPT_UTILS = importlib.import_module("mlx_vlm.prompt_utils")
_APP = importlib.import_module("mlx_vlm.server.app")
_RUNTIME = importlib.import_module("mlx_vlm.server.runtime").runtime
_FIXTURES = Path(__file__).parents[1] / "fixtures"

# Template: model type and chat template (a GGUF header's, or the one gmlx
# ships for the type).
_TEMPLATES = {
    "gemma4": ("gemma4", "gemma4_template.jinja"),
    "gemma4-12b": ("gemma4_unified", "gemma4_12b_template.jinja"),
    "qwen3.5": ("qwen3_5", "qwen35_template.jinja"),
    "qwen3.6": ("qwen3_5_moe", "qwen36_template.jinja"),
    "qwen3.8": ("qwen3_5", "qwen38_template.jinja"),
    "qwen3": ("qwen3", "qwen3_template.jinja"),
    "gpt-oss": ("gpt_oss", "gptoss_template.jinja"),
    "deepseek-v4": ("deepseek_v4", "deepseek_v4_template.jinja"),
    "deepseek-v4.1": ("deepseek_v41", None),
    "kimi-k2.5": ("kimi_k25", "kimi_k25_template.jinja"),
    "kimi-k3": ("kimi_k3", "kimi_k3_template.jinja"),
}


class _Processor:
    """Renders the template as the tokenizer gmlx builds for the type."""

    def __init__(self, name):
        model_type, fixture = _TEMPLATES[name]
        self.chat_template = (
            (_FIXTURES / fixture).read_text() if fixture
            else bundled_chat_template(model_type))
        finish_gguf_tokenizer(self, model_type)

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
    def __init__(self):
        self.calls = []

    def generate(self, prompt, images=None, audio=None, args=None,
                 videos=None):
        self.calls.append((prompt, list(images or ())))
        return types.SimpleNamespace(prompt_tokens=1, uid=1), iter([_Tok()])


@pytest.fixture
def route(monkeypatch):
    """/v1/messages with the gmlx render wraps, on a stub model."""
    routes = sp_common._snapshot_routes(_APP.app)
    for target in sp_common._render_target_modules():
        monkeypatch.setattr(target, "apply_chat_template",
                            target.apply_chat_template)
    monkeypatch.setattr(_PROMPT_UTILS, "get_chat_template",
                        _PROMPT_UTILS.get_chat_template)
    monkeypatch.setattr(_ANTHROPIC, "_anthropic_messages_to_internal",
                        _ANTHROPIC._anthropic_messages_to_internal)
    rg = _RG()
    monkeypatch.setattr(_RUNTIME, "response_generator", rg, raising=False)
    install_faithful_history()
    install_role_normalization()
    install_chat_image_placement()
    install_messages_route()
    from fastapi.testclient import TestClient

    client = TestClient(_APP.app)

    def render(name, messages, stock=False):
        processor = _Processor(name)
        monkeypatch.setattr(_ANTHROPIC, "get_cached_model", lambda *a, **k: (
            None, processor, {"model_type": _TEMPLATES[name][0]}))
        if stock:
            claim = _ANTHROPIC._anthropic_messages_to_internal
            monkeypatch.setattr(_ANTHROPIC, "_anthropic_messages_to_internal",
                                getattr(claim, "_gmlx_orig", claim))
        r = client.post("/v1/messages", json={
            "model": "m", "max_tokens": 1, "messages": messages,
            "tools": _TOOLS})
        assert r.status_code == 200, r.text
        return rg.calls.pop()

    yield render
    sp_common._restore_routes(_APP.app, routes)


_TOOLS = [{"name": "shot", "description": "Take a screenshot.",
           "input_schema": {"type": "object", "properties": {}}}]


def _text(t):
    return {"type": "text", "text": t}


def _result(i, text):
    return {"type": "tool_result", "tool_use_id": f"toolu_{i}",
            "content": [_text(text)]}


def _round(*blocks, calls=1):
    """The user asks, the assistant calls ``shot`` ``calls`` times, and the
    next user message holds ``blocks``."""
    uses = [{"type": "tool_use", "id": f"toolu_{i}", "name": "shot",
             "input": {}} for i in range(calls)]
    return [{"role": "user", "content": [_text("ASK-FIRST")]},
            {"role": "assistant", "content": [_text("Taking it.")] + uses},
            {"role": "user", "content": list(blocks)}]


# Conversation: the texts that must render in this order after the call.
_WITH_TEXT = {
    "a result and text": (
        _round(_result(0, "RESULT-ONE"), _text("TEXT-AFTER")),
        ("RESULT-ONE", "TEXT-AFTER")),
    "two results and text": (
        _round(_result(0, "RESULT-ONE"), _result(1, "RESULT-TWO"),
               _text("TEXT-AFTER"), calls=2),
        ("RESULT-ONE", "RESULT-TWO", "TEXT-AFTER")),
    "a later turn": (
        _round(_result(0, "RESULT-ONE"), _text("TEXT-AFTER")) + [
            {"role": "assistant", "content": [_text("ANSWER")]},
            {"role": "user", "content": [_text("NEXT-QUESTION")]}],
        ("RESULT-ONE", "TEXT-AFTER", "ANSWER", "NEXT-QUESTION")),
}


@pytest.mark.parametrize("case", sorted(_WITH_TEXT))
@pytest.mark.parametrize("name", sorted(_TEMPLATES))
def test_tool_results_render_before_the_text_beside_them(route, name, case):
    messages, order = _WITH_TEXT[case]
    prompt, _ = route(name, messages)
    pos = prompt.index("ASK-FIRST")
    for want in order:
        found = prompt.find(want, pos)
        assert found >= 0, f"{want!r} missing or out of order in {prompt!r}"
        pos = found + len(want)


_RESULTS_ONLY = {
    "one result": _round(_result(0, "RESULT-ONE")),
    "two results": _round(_result(0, "RESULT-ONE"), _result(1, "RESULT-TWO"),
                          calls=2),
}


@pytest.mark.parametrize("case", sorted(_RESULTS_ONLY))
@pytest.mark.parametrize("name", sorted(_TEMPLATES))
def test_a_message_of_tool_results_renders_as_before(route, name, case):
    messages = _RESULTS_ONLY[case]
    assert route(name, messages) == route(name, messages, stock=True)
