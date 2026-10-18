"""K2 Horizon tool-call parser (``gmlx.models.k2_horizon.tools``). CPU-only,
no model load.

``parse_tool_call`` gets the inner text of an ``<ifm|tool_calls>`` section,
as mlx-vlm's ``process_tool_calls`` hands it over. The argument cases mirror
the K2 Horizon block of llama.cpp's ``tests/test-chat.cpp``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import gmlx.models.k2_horizon.tools as k2_tools
from gmlx.models.k2_horizon.tools import (
    install_message_normalizer,
    normalize_messages,
    parse_tool_call,
    tool_call_end,
    tool_call_start,
)

_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
_TEMPLATE = (_FIXTURES / "k2_horizon_template.jinja").read_text()


def _tool(name, properties, required=()):
    return {"type": "function", "function": {
        "name": name,
        "parameters": {"type": "object", "properties": properties,
                       "required": list(required)}}}


_SPECIAL = _tool("special_function", {"arg1": {"type": "integer"}}, ["arg1"])
_GET_TIME = _tool("get_time", {"city": {"type": "string"}}, ["city"])
_EMPTY = _tool("empty_args", {})
_2REQ = _tool("tool_2req_4opt", {
    "req1": {"type": "string"}, "req2": {"type": "integer"},
    "opt1": {"type": "string"}, "opt2": {"type": "integer"},
}, ["req1", "req2"])
_UNION = _tool("set_union", {
    "value": {"type": ["string", "object"]},
    "amount": {"type": ["string", "integer"]},
}, ["value", "amount"])


def _call(name, args=(), typed=False):
    body = f"<ifm|tool_call>{name}\n"
    for key, value, *rest in args:
        body += f"<ifm|arg_key>{key}</ifm|arg_key>\n"
        if typed and rest:
            body += f"<ifm|arg_type>{rest[0]}</ifm|arg_type>\n"
        body += f"<ifm|arg_value>{value}</ifm|arg_value>\n"
    return body + "</ifm|tool_call>"


def _inner(*calls):
    return "\n" + "\n".join(calls) + "\n"


def _registry():
    # A plain import: a registry move must fail here, not skip.
    from gmlx._vendor.mlx_vlm.tools import registry

    return registry


def _render(messages, tools=None, **kwargs):
    from transformers.utils.chat_template_utils import _compile_jinja_template

    return _compile_jinja_template(_TEMPLATE).render(
        messages=messages, tools=tools, add_generation_prompt=True,
        bos_token="", **kwargs)


def test_delimiters_are_the_wrapper_tags():
    # The streaming suppressor keys on tool_call_start, so it must be the
    # section tag, not the per-call tag it shares a prefix with.
    assert tool_call_start == "<ifm|tool_calls>"
    assert tool_call_end == "</ifm|tool_calls>"


def test_xml_call():
    out = parse_tool_call(_inner(_call("special_function", [("arg1", "1")])),
                          [_SPECIAL])
    assert out == [{"name": "special_function", "arguments": {"arg1": 1}}]


def test_xml_typed_call():
    text = _inner(_call("special_function", [("arg1", "1", "integer")],
                        typed=True))
    assert "<ifm|arg_type>" in text
    assert parse_tool_call(text, [_SPECIAL]) == [
        {"name": "special_function", "arguments": {"arg1": 1}}]


def test_json_call():
    text = _inner('<ifm|tool_call>{"name": "special_function", '
                  '"arguments": {"arg1": 1}}</ifm|tool_call>')
    assert parse_tool_call(text, [_SPECIAL]) == [
        {"name": "special_function", "arguments": {"arg1": 1}}]


def test_json_call_with_string_arguments():
    text = _inner('<ifm|tool_call>{"name": "special_function", '
                  '"arguments": "{\\"arg1\\": 1}"}</ifm|tool_call>')
    assert parse_tool_call(text, [_SPECIAL]) == [
        {"name": "special_function", "arguments": {"arg1": 1}}]


def test_zero_arg_call():
    assert parse_tool_call(_inner(_call("empty_args")), [_EMPTY]) == [
        {"name": "empty_args", "arguments": {}}]


def test_parallel_calls():
    text = _inner(_call("get_time", [("city", "Paris")]),
                  _call("get_time", [("city", "Rome")]))
    assert parse_tool_call(text, [_GET_TIME]) == [
        {"name": "get_time", "arguments": {"city": "Paris"}},
        {"name": "get_time", "arguments": {"city": "Rome"}},
    ]


def test_args_in_any_order_keep_schema_types():
    text = _inner(_call("tool_2req_4opt", [("req2", "7"), ("req1", "hello")]))
    assert parse_tool_call(text, [_2REQ]) == [
        {"name": "tool_2req_4opt", "arguments": {"req2": 7, "req1": "hello"}}]


def test_string_schema_value_stays_raw():
    text = _inner(_call("tool_2req_4opt", [("req1", "123"), ("req2", "1"),
                                           ("opt1", "true")]))
    assert parse_tool_call(text, [_2REQ])[0]["arguments"] == {
        "req1": "123", "req2": 1, "opt1": "true"}


@pytest.mark.parametrize(
    "value", ["true", "42", "null", "[]", '"quoted"', "{not valid json"])
def test_union_with_string_keeps_other_json_as_text(value):
    text = _inner(_call("set_union", [("value", value), ("amount", "42")]))
    assert parse_tool_call(text, [_UNION]) == [
        {"name": "set_union", "arguments": {"value": value, "amount": 42}}]


def test_union_with_string_takes_its_json_type():
    text = _inner(_call("set_union", [("value", '{"a": 1}'),
                                      ("amount", "2 dollars")]))
    assert parse_tool_call(text, [_UNION]) == [
        {"name": "set_union",
         "arguments": {"value": {"a": 1}, "amount": "2 dollars"}}]


def test_multiline_string_value_kept_exactly():
    text = _inner(_call("get_time", [("city", "San\nFrancisco\n")]))
    assert parse_tool_call(text, [_GET_TIME])[0]["arguments"] == {
        "city": "San\nFrancisco\n"}


def test_untyped_value_falls_back_to_text():
    text = _inner(_call("unknown_tool", [("a", "Paris"), ("b", "[1, 2]")]))
    assert parse_tool_call(text, None) == [
        {"name": "unknown_tool", "arguments": {"a": "Paris", "b": [1, 2]}}]


def test_freeform_text_is_not_a_call():
    out = parse_tool_call("just some text", None)
    assert out == {"name": "unknown", "arguments": {"raw": "just some text"}}


@pytest.mark.parametrize("fmt", ["xml", "xml_typed", "json"])
def test_template_render_round_trips(fmt):
    tools = [_tool("get_weather", {"city": {"type": "string"},
                                   "days": {"type": "integer"},
                                   "opts": {"type": "object"}}, ["city"])]
    calls = [{"city": "Paris\nFrance", "days": 3, "opts": {"a": [1, 2]}}, {}]
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "", "reasoning_content": "r",
         "tool_calls": [{"type": "function", "function": {
             "name": "get_weather", "arguments": a}} for a in calls]},
        {"role": "tool", "content": "sunny"},
    ]
    out = _render(messages, tools, tool_call_format=fmt)
    # The system turn spells the section tags in its instructions.
    turn = out.split("<|ifm|im_start|>assistant", 1)[1]
    inner = turn.split(tool_call_start, 1)[1].split(tool_call_end, 1)[0]
    assert parse_tool_call(inner, tools) == [
        {"name": "get_weather", "arguments": a} for a in calls]


# --- message normalizer -----------------------------------------------------


def test_normalizer_adds_empty_reasoning_to_bare_assistant_turns():
    msgs = [{"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"}]
    out = normalize_messages(msgs)
    assert out[1] == {"role": "assistant", "content": "hello",
                      "reasoning_content": ""}
    assert "reasoning_content" not in msgs[1]
    assert out[0] is msgs[0]


@pytest.mark.parametrize("field", ["think", "think_fast", "think_faster",
                                   "reasoning_content", "reasoning"])
def test_normalizer_keeps_turns_that_carry_a_reasoning_field(field):
    msgs = [{"role": "assistant", "content": "x", field: "r"}]
    assert normalize_messages(msgs) is msgs


def test_normalizer_lets_the_template_render_client_history():
    from jinja2.exceptions import TemplateError

    msgs = [{"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "user", "content": "again"}]
    with pytest.raises(TemplateError, match="thinking field"):
        _render(msgs)
    out = _render(normalize_messages(msgs))
    assert "<ifm|think>\n</ifm|think>hello" in out


def test_install_message_normalizer_wraps_once():
    class _Tok:
        def __init__(self):
            self.seen = []

        def apply_chat_template(self, conversation, *args, **kwargs):
            self.seen.append(conversation)
            return "OK"

    tok = _Tok()
    install_message_normalizer(tok)
    first = tok.apply_chat_template
    install_message_normalizer(tok)
    assert tok.apply_chat_template is first
    assert tok.apply_chat_template(
        [{"role": "assistant", "content": "x"}]) == "OK"
    assert tok.seen == [[{"role": "assistant", "content": "x",
                          "reasoning_content": ""}]]


# --- registry ---------------------------------------------------------------


def _stock_choice(registry, text):
    matches = [s for s in registry.SPECS
               if s.matches(text) and s.name != "k2_horizon"]
    return max(matches, key=lambda s: s.priority).name if matches else None


def test_stock_registry_would_misroute_the_template():
    # The template's XML tool listing carries both minicpm5 markers.
    assert _stock_choice(_registry(), _TEMPLATE) == "minicpm5"


def test_ensure_registered_infers_k2_horizon_from_template():
    registry = _registry()
    k2_tools.ensure_registered()
    assert registry._infer_tool_parser(_TEMPLATE) == "k2_horizon"


def test_k2_markers_claim_no_other_fixture_template():
    registry = _registry()
    k2_tools.ensure_registered()
    spec = next(s for s in registry.SPECS if s.name == "k2_horizon")
    others = [p for p in _FIXTURES.glob("*template*.jinja")
              if p.name != "k2_horizon_template.jinja"]
    assert others
    claimed = [p.name for p in others if spec.matches(p.read_text())]
    assert claimed == []


def test_ensure_registered_is_idempotent_and_loadable():
    registry = _registry()
    k2_tools.ensure_registered()
    before = registry.SPECS
    k2_tools.ensure_registered()
    assert registry.SPECS == before
    assert sum(s.name == "k2_horizon" for s in registry.SPECS) == 1
    assert registry.load_tool_module("k2_horizon") is k2_tools


def test_parser_output_is_json_serializable():
    out = parse_tool_call(_inner(_call("set_union", [("value", '{"a": 1}'),
                                                    ("amount", "3")])),
                          [_UNION])
    json.dumps(out)
