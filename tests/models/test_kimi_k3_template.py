#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Asher Feldman
"""Kimi-K3 chat-template render checks (fixture = the llama.cpp PR #26185
Kimi-K3.jinja, the template real GGUFs embed). Rendered through a jinja2
sandbox configured the way transformers renders chat templates, so a pass
here means tok.apply_chat_template works on a template-carrying GGUF."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

jinja2 = pytest.importorskip("jinja2")
from jinja2.sandbox import ImmutableSandboxedEnvironment  # noqa: E402

_TMPL = (Path(__file__).parents[1] / "fixtures"
         / "kimi_k3_template.jinja").read_text()

GEN_TAIL_THINK = '<|open|>message role="assistant"<|sep|><|open|>think<|sep|>'
GEN_TAIL_RESPONSE = ('<|open|>message role="assistant"<|sep|>'
                     '<|open|>response<|sep|>')


def _tojson(value, ensure_ascii=True, indent=None, separators=None,
            sort_keys=False):
    return json.dumps(value, ensure_ascii=ensure_ascii, indent=indent,
                      separators=separators, sort_keys=sort_keys)


def _raise(message):
    raise jinja2.exceptions.TemplateError(message)


def _render(messages, **kwargs):
    env = ImmutableSandboxedEnvironment(
        extensions=["jinja2.ext.loopcontrols"],
        trim_blocks=True, lstrip_blocks=True)
    env.filters["tojson"] = _tojson
    env.globals["raise_exception"] = _raise
    return env.from_string(_TMPL).render(messages=messages, **kwargs)


_CHAT = [
    {"role": "system", "content": "Be helpful."},
    {"role": "user", "content": "hi"},
]


def test_generation_prompt_pre_opens_think():
    out = _render(_CHAT, add_generation_prompt=True)
    assert out.endswith(GEN_TAIL_THINK)


def test_generation_prompt_thinking_off_pre_opens_response():
    out = _render(_CHAT, add_generation_prompt=True, thinking=False)
    assert out.endswith(GEN_TAIL_RESPONSE)


def test_multi_turn_history_sections_and_terminators():
    msgs = _CHAT + [
        {"role": "assistant", "content": "Hello!",
         "reasoning_content": "Simple greeting."},
        {"role": "user", "content": "thanks"},
    ]
    out = _render(msgs, add_generation_prompt=True)
    # Prior assistant turn renders think + response sections, closed, and a
    # turn terminator; the new turn pre-opens think.
    assert "<|open|>think<|sep|>Simple greeting.<|close|>think<|sep|>" in out
    assert "<|open|>response<|sep|>Hello!<|close|>response<|sep|>" in out
    assert out.count("<|end_of_msg|>") >= 3  # system, assistant, users
    assert out.endswith(GEN_TAIL_THINK)


def test_tool_declaration_call_and_result_round_trip():
    tools = [{"type": "function", "function": {
        "name": "get_weather",
        "description": "Weather lookup.",
        "parameters": {"type": "object", "properties": {
            "city": {"type": "string"}}, "required": ["city"]},
    }}]
    msgs = _CHAT + [
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "call_1", "type": "function", "function": {
                "name": "get_weather",
                "arguments": {"city": "Tokyo"}}}]},
        {"role": "tool", "tool_call_id": "call_1", "content": "22C, clear"},
    ]
    out = _render(msgs, tools=tools, add_generation_prompt=True)
    assert 'type="tool-declare"' in out
    assert '<|open|>call tool="get_weather" index="1"<|sep|>' in out
    assert '<|open|>argument key="city" type="string"<|sep|>Tokyo' in out
    assert '<|open|>message role="tool" tool="get_weather" index="1"<|sep|>' \
        in out
    assert "22C, clear" in out
    assert out.endswith(GEN_TAIL_THINK)


def test_thinking_effort_renders_and_validates():
    out = _render(_CHAT, add_generation_prompt=True, thinking_effort="low")
    assert "thinking_effort=low" in out
    with pytest.raises(jinja2.exceptions.TemplateError,
                       match="thinking_effort"):
        _render(_CHAT, add_generation_prompt=True, thinking_effort="medium")


# mlx-lm's TokenizerWrapper only detects thinking support from vocab token
# pairs (<think>/</think> etc). K3's think channel is XTML: "think" is plain
# text between structural tokens, so detection misses it and the wrapper
# injects enable_thinking=False into a thinking-default template. The loader
# helper renders the template's own default generation prompt and flips the
# wrapper's think markers when it opens the XTML think channel.

class _StubRaw:
    chat_template = _TMPL

    def apply_chat_template(self, messages, add_generation_prompt=False,
                            tokenize=False, **kwargs):
        return _render(messages,
                       add_generation_prompt=add_generation_prompt)

    def encode(self, text, add_special_tokens=False):
        return [len(text), 7]


class _StubWrapper:
    _think_start = None
    _think_end = None
    _think_start_tokens = None
    _think_end_tokens = None

    @property
    def has_thinking(self):
        return self._think_start is not None


def test_detect_xtml_thinking_flips_wrapper_default():
    from gmlx.load.loader import _detect_xtml_thinking

    w = _StubWrapper()
    _detect_xtml_thinking(w, _StubRaw(), lambda *_: None)
    assert w.has_thinking
    assert w._think_start == "<|open|>think<|sep|>"
    assert w._think_end == "<|close|>think<|sep|>"
    assert w._think_start_tokens == (len(w._think_start), 7)
    assert w._think_end_tokens == (len(w._think_end), 7)


def test_detect_xtml_thinking_ignores_non_xtml_templates():
    from gmlx.load.loader import _detect_xtml_thinking

    class _PlainRaw(_StubRaw):
        chat_template = "{{ messages }}"

        def apply_chat_template(self, messages, **kwargs):
            return "no think channel here"

    w = _StubWrapper()
    _detect_xtml_thinking(w, _PlainRaw(), lambda *_: None)
    assert not w.has_thinking


def test_detect_xtml_thinking_respects_existing_detection():
    from gmlx.load.loader import _detect_xtml_thinking

    class _Failing(_StubRaw):
        def apply_chat_template(self, messages, **kwargs):
            raise AssertionError("must not render when already detected")

    w = _StubWrapper()
    w._think_start = "<think>"
    _detect_xtml_thinking(w, _Failing(), lambda *_: None)
    assert w._think_start == "<think>"
    assert w._think_start_tokens is None
