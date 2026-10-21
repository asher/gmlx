# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Asher Feldman
"""Beam tool calls: the parser, the extractor hook and the stream filter.

Beam writes each call as its own Harmony message with XML parameters, and
the last call ends at the <|call|> stop token, which never reaches the text.
"""

from __future__ import annotations

import json

import pytest

from gmlx._vendor.mlx_vlm.tools.extractor import process_tool_calls
from gmlx.models.beam import tools
from gmlx.tui.reasoning import ReasoningFilter

_HEAD = "<|start|>assistant to={}<|channel|>analysis <|constrain|>xml<|message|>"
_TOOLS = [{"type": "function", "function": {"name": "get", "parameters": {
    "properties": {
        "n": {"type": "integer"},
        "s": {"type": "string"},
        "o": {"type": "object"},
        "u": {"anyOf": [{"type": "integer"}, {"type": "string"}]},
    }}}}]


def _param(name, value, close=True):
    return (f'<rfl:parameter name="{name}">\n{value}'
            + ("\n</rfl:parameter>" if close else "\n"))


def _calls(text, tools_=_TOOLS):
    res = process_tool_calls(text, tools, tools_)
    return [(c["function"]["name"], json.loads(c["function"]["arguments"]))
            for c in res.calls], res.remaining_text


def test_one_call_with_typed_values():
    text = (_HEAD.format("get") + _param("n", "3") + "\n" + _param("s", "007")
            + _param("o", '{"a": [1, 2]}') + _param("u", "x1"))
    calls, _ = _calls(text)
    assert calls == [("get", {"n": 3, "s": "007", "o": {"a": [1, 2]},
                              "u": "x1"})]
    calls, _ = _calls(_HEAD.format("get") + _param("u", "12"))
    assert calls == [("get", {"u": 12})]


def test_parallel_calls_after_reasoning():
    text = ("<|channel|>analysis<|message|>Two reads.<|end|>"
            + _HEAD.format("get") + _param("n", "1") + "<|end|>"
            + _HEAD.format("bash") + _param("command", "ls -la"))
    calls, remaining = _calls(text)
    assert calls == [("get", {"n": 1}), ("bash", {"command": "ls -la"})]
    assert remaining == "<|channel|>analysis<|message|>Two reads.<|end|>"


def test_multi_line_value_keeps_its_inner_newlines():
    value = "line one\n\n  line two\n"
    calls, _ = _calls(_HEAD.format("bash") + _param("command", value))
    assert calls == [("bash", {"command": value})]


def test_missing_close_tag_is_tolerated():
    text = (_HEAD.format("get") + _param("s", "first", close=False)
            + _param("n", "4", close=False))
    calls, _ = _calls(text)
    assert calls == [("get", {"s": "first", "n": 4})]


def test_call_at_offset_zero():
    """The generation prompt ends at <|start|>assistant, so the output can
    open with the recipient."""
    text = (" to=get<|channel|>analysis <|constrain|>xml<|message|>"
            + _param("n", "5"))
    calls, remaining = _calls(text)
    assert calls == [("get", {"n": 5})]
    assert remaining == ""


def test_truncated_header_is_no_call():
    calls, remaining = _calls(" to=get<|channel|>analysis <|constrain|>xml")
    assert calls == []
    assert remaining == " to=get<|channel|>analysis <|constrain|>xml"


def test_truncated_body_keeps_the_complete_parameters():
    text = (_HEAD.format("get") + _param("n", "6")
            + '<rfl:parameter name="s')
    calls, _ = _calls(text)
    assert calls == [("get", {"n": 6})]


@pytest.mark.parametrize("text", [
    "<|channel|>final<|message|>Hello\nworld",
    "Plain text with no markers.\n",
    "<|channel|>analysis<|message|>Think<|end|><|start|>assistant"
    "<|channel|>final<|message|>The answer is to=x.",
])
def test_plain_answer_text_yields_no_calls(text):
    """tool_call_start is empty, so the default slicing would turn any text
    into a call. The hook keeps the module off that path."""
    calls, remaining = _calls(text)
    assert calls == []
    assert remaining == text


def test_registration_routes_beam_templates_only():
    from pathlib import Path

    from gmlx._vendor.mlx_vlm.tools import registry

    tools.ensure_registered()
    fixtures = Path(__file__).parents[1] / "fixtures"
    beam = (fixtures / "beam_template.jinja").read_text()
    assert registry._infer_tool_parser(beam) == "beam"
    assert registry.load_tool_module("beam") is tools
    gptoss = (fixtures / "gptoss_template.jinja").read_text()
    assert registry._infer_tool_parser(gptoss) != "beam"
    assert tools.tool_call_start == "" and tools.tool_call_end == ""


def _stream(text, **kw):
    f = ReasoningFilter(**kw)
    spans = []
    for ch in text:
        spans += f.feed(ch)
    spans += f.flush()
    reason = "".join(t for t, m in spans if m == "reason")
    answer = "".join(t for t, m in spans if m == "answer")
    return reason, answer


def test_filter_hides_a_streamed_call_from_reasoning_and_answer():
    text = ("<|channel|>analysis<|message|>Need a listing.<|end|>"
            + _HEAD.format("bash") + _param("command", "ls -la"))
    assert _stream(text, start_in_header=True) == ("Need a listing.", "")


def test_filter_hides_a_call_at_offset_zero():
    text = (" to=bash<|channel|>analysis <|constrain|>xml<|message|>"
            + _param("command", "ls") + "<|end|>" + _HEAD.format("get")
            + _param("n", "1"))
    assert _stream(text, start_in_header=True) == ("", "")


def test_filter_shows_the_answer_after_a_call():
    text = (_HEAD.format("get") + _param("n", "1") + "<|end|>"
            + "<|start|>assistant<|channel|>final<|message|>Done.")
    assert _stream(text, start_in_header=True) == ("", "Done.")


def test_gpt_oss_tool_arguments_keep_their_routing():
    """gpt-oss constrains to JSON on its commentary channel. gmlx parses no
    gpt-oss calls, and the arguments stay out of the answer."""
    text = ("<|channel|>analysis<|message|>x<|end|><|start|>assistant"
            '<|channel|>commentary to=functions.f <|constrain|>json'
            '<|message|>{"a":1}<|call|>')
    reason, answer = _stream(text, start_in_header=True)
    assert answer == ""
    assert reason == 'x{"a":1}'
