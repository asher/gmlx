"""Inkling chat format: the reasoning effort, the block-header split and
the tool-call parser."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from transformers.utils.chat_template_utils import render_jinja_template

from gmlx.tui.reasoning import (
    ReasoningFilter,
    inkling_effort,
    map_thinking_controls,
    off_by_budget,
    split_harmony_reply,
)

TEMPLATE = (Path(__file__).parents[1] / "fixtures"
            / "inkling2_template.jinja").read_text()

THINK = "<|message_model|><|content_thinking|>"
TEXT = "<|message_model|><|content_text|>"
END = "<|end_message|>"
STOP = "<|content_model_end_sampling|>"
CALL = ('<|message_model|>get_weather<|content_invoke_tool_json|>'
        '{"name":"get_weather","args":{"city":"Paris","days":2}}' + END)


def _render(messages=None, **kwargs):
    out = render_jinja_template(
        [messages or [{"role": "user", "content": "Hi"}]],
        chat_template=TEMPLATE, add_generation_prompt=True, **kwargs)
    rendered = out[0] if isinstance(out, tuple) else out
    return rendered[0] if isinstance(rendered, list) else rendered


@pytest.mark.parametrize("value, want", [
    ("high", "high"), (" MAX ", "max"), ("minimal", "minimal"),
    ("off", "none"), ("disabled", "none"), ("false", "none"), ("0", "none"),
    ("none", "none"), (0.5, 0.5), ("0.25", 0.25), (0, 0.0), (0.99, 0.99),
])
def test_effort_values_the_template_takes(value, want):
    assert inkling_effort(value) == want


@pytest.mark.parametrize("value", ["extreme", "", 1.0, -0.1, "2", True])
def test_effort_values_the_template_refuses(value):
    with pytest.raises(ValueError):
        inkling_effort(value)


def test_the_canonical_efforts_render():
    for value in ("none", "minimal", "low", "medium", "high", "xhigh",
                  "max", 0.5):
        assert "Thinking effort level:" in _render(reasoning_effort=value)
    assert "Thinking effort level: 0<" in _render(reasoning_effort="none")


def test_thinking_off_is_effort_none_not_a_budget():
    assert not off_by_budget("off", TEMPLATE)
    out = map_thinking_controls({}, "off", None, TEMPLATE)
    assert out == {"reasoning_effort": "none"}
    # An explicit effort wins over the switch.
    out = map_thinking_controls({}, "off", "low", TEMPLATE)
    assert out == {"reasoning_effort": "low"}
    # On leaves the template default.
    assert map_thinking_controls({}, "on", None, TEMPLATE) == {}


def test_an_unknown_effort_is_dropped_with_a_warning():
    heard = []
    out = map_thinking_controls({}, None, "extreme", TEMPLATE,
                                warn=heard.append)
    assert out == {} and heard


def _stash(profile=None, req_kw=None, effort=None, switch=None,
           request_switch=False):
    """``_inkling_effort_kwargs`` on ``merged`` = the profile's kwargs.
    ``effort`` is the request's top-level effort, ``switch`` the resolved
    thinking switch."""
    from gmlx.serve.patches.chat_behavior import _inkling_effort_kwargs

    request = SimpleNamespace(chat_template_kwargs=req_kw)
    args = SimpleNamespace(reasoning_effort=effort)
    merged = {**(profile or {}), **(req_kw or {})}
    _inkling_effort_kwargs(args, request, merged, TEMPLATE, switch=switch,
                           request_switch=request_switch)
    return merged.get("reasoning_effort")


HIGH = {"reasoning_effort": "high"}
NONE = {"reasoning_effort": "none"}


@pytest.mark.parametrize("kw, want", [
    ({}, None),
    ({"effort": "OFF"}, "none"),
    ({"effort": "xhigh"}, "xhigh"),
    ({"switch": False, "request_switch": True}, "none"),
    # A server-wide switch with no effort anywhere.
    ({"switch": False}, "none"),
    # A request switch beats the profile's effort.
    ({"profile": HIGH, "switch": False, "request_switch": True}, "none"),
    ({"profile": NONE, "switch": True, "request_switch": True}, None),
    # A server-wide switch loses to the profile's own effort.
    ({"profile": HIGH, "switch": False}, "high"),
    # The request's effort beats the profile's effort.
    ({"profile": HIGH, "effort": "none"}, "none"),
    # The request's own template kwargs beat everything.
    ({"req_kw": {"enable_thinking": False}}, "none"),
    ({"profile": HIGH, "req_kw": {"enable_thinking": False}}, "none"),
    ({"req_kw": {"thinking": {"type": "disabled"}}, "profile": HIGH}, "none"),
    ({"req_kw": {"enable_thinking": "false"}}, "none"),
    ({"req_kw": {"enable_thinking": 0}}, "none"),
    ({"req_kw": {"enable_thinking": None}, "profile": HIGH}, "high"),
    # The profile's own template kwargs are the profile switch.
    ({"profile": {"enable_thinking": False}}, "none"),
    ({"req_kw": {"reasoning_effort": "low"}, "effort": "max",
      "switch": False, "request_switch": True}, "low"),
])
def test_serve_effort_precedence(kw, want):
    assert _stash(**kw) == want


def test_serve_refuses_an_effort_the_template_would_refuse():
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as err:
        _stash(effort="extreme")
    assert err.value.status_code == 400
    with pytest.raises(HTTPException):
        _stash(req_kw={"enable_thinking": "maybe"})


def _filtered(text, step=None):
    filt = ReasoningFilter()
    spans = []
    if step is None:
        spans += filt.feed(text)
    else:
        for i in range(0, len(text), step):
            spans += filt.feed(text[i:i + step])
    spans += filt.flush()
    reason = "".join(t for t, m in spans if m == "reason")
    answer = "".join(t for t, m in spans if m == "answer")
    return reason, answer


@pytest.mark.parametrize("step", [None, 1, 3])
def test_the_filter_splits_a_turn_by_block(step):
    text = (THINK + "Think first." + END + TEXT + "The answer." + END
            + CALL + STOP)
    assert _filtered(text, step) == ("Think first.", "The answer.")


def test_the_reply_split_returns_reasoning_and_content():
    text = THINK + "Plan." + END + TEXT + "Done." + END + STOP
    assert split_harmony_reply(text) == ("Plan.", "Done.")


def test_a_tool_call_header_is_never_display_text():
    reason, answer = _filtered(THINK + "Call it." + END + CALL + STOP)
    assert reason == "Call it."
    assert answer == ""


def test_the_tool_parser_reads_every_call():
    from gmlx._vendor.mlx_vlm.tools.extractor import process_tool_calls
    from gmlx.models.inkling2 import tools

    second = CALL.replace("Paris", "Rome")
    text = THINK + "Two calls." + END + CALL + second + STOP
    result = process_tool_calls(text, tools, None)
    assert [c["function"]["name"] for c in result.calls] == ["get_weather"] * 2
    args = [json.loads(c["function"]["arguments"]) for c in result.calls]
    assert args == [{"city": "Paris", "days": 2}, {"city": "Rome", "days": 2}]
    assert "get_weather" not in result.remaining_text


def test_the_header_names_the_tool():
    from gmlx.models.inkling2 import tools

    text = ('<|message_model|>f<|content_invoke_tool_json|>'
            '{"name":"g","args":{}}' + END)
    assert tools.parse_tool_call(text)["name"] == "f"


def test_the_tool_parser_round_trips_the_template_render():
    from gmlx.models.inkling2 import tools

    call = {"type": "function", "function": {
        "name": "lookup", "arguments": {"query": "a \"b\"", "n": [1, 2]}}}
    text = _render([{"role": "user", "content": "Hi"},
                    {"role": "assistant", "content": "",
                     "tool_calls": [call]}])
    (span,) = tools.find_tool_calls(text)
    parsed = tools.parse_tool_call(text[span[0]:span[1]])
    assert parsed == {"name": "lookup",
                      "arguments": {"query": 'a "b"', "n": [1, 2]}}


def test_the_template_routes_to_the_inkling_parser():
    from gmlx._vendor.mlx_vlm.tools import registry
    from gmlx.models.inkling2 import tools

    tools.ensure_registered()
    assert registry._infer_tool_parser(TEMPLATE) == "inkling"
    assert registry.load_tool_module("inkling") is tools
