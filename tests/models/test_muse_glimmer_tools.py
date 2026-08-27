"""Muse Glimmer ATEM tool-call parser (``gmlx.muse_glimmer_tools``).
CPU-only, no model load.

Exercises ``parse_tool_call`` on wrapper-inner text as mlx-vlm's
``process_tool_calls`` hands it over (the caller strips the module's
``tool_call_start``/``tool_call_end``), the schema-driven value coercion, and
the ``ensure_registered`` graft into mlx-vlm's parser registry.
"""

from __future__ import annotations

import sys

import pytest

from gmlx import muse_glimmer_tools
from gmlx.muse_glimmer_tools import (
    parse_tool_call,
    tool_call_end,
    tool_call_start,
)

_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "days": {"type": "integer"},
                    "precise": {"type": "boolean"},
                    "ratio": {"type": "number"},
                    "opts": {"type": "object"},
                    "tags": {"type": "array"},
                },
            },
        },
    }
]


def _invoke(name, args=()):
    parts = [f'<atem:invoke name="{name}">']
    for k, v in args:
        parts.append(f'<atem:parameter name="{k}">{v}</atem:parameter>')
    parts.append("</atem:invoke>")
    return "".join(parts)


def test_wrapper_tags_are_the_outer_block():
    # The streaming suppressor keys on the wrapper, not the per-call tag, so a
    # partial tail is always a prefix of tool_call_start.
    assert tool_call_start == "<atem:function_calls>"
    assert tool_call_end == "</atem:function_calls>"


def test_single_call_parses_name_and_arguments():
    calls = parse_tool_call(_invoke("get_weather", [("city", "Paris")]), _TOOLS)
    assert calls == [{"name": "get_weather", "arguments": {"city": "Paris"}}]


def test_parallel_calls_parse_in_order():
    text = _invoke("get_weather", [("city", "Paris")]) + _invoke(
        "get_weather", [("city", "Rome")])
    calls = parse_tool_call(text, _TOOLS)
    assert [c["arguments"]["city"] for c in calls] == ["Paris", "Rome"]


def test_values_are_coerced_by_the_tool_schema():
    calls = parse_tool_call(_invoke("get_weather", [
        ("city", "Paris"),
        ("days", "3"),
        ("precise", "true"),
        ("ratio", "0.5"),
        ("opts", '{"a": 1}'),
        ("tags", '["x", "y"]'),
    ]), _TOOLS)
    args = calls[0]["arguments"]
    assert args["city"] == "Paris"
    assert args["days"] == 3 and isinstance(args["days"], int)
    assert args["precise"] is True
    assert args["ratio"] == pytest.approx(0.5)
    assert args["opts"] == {"a": 1}
    assert args["tags"] == ["x", "y"]


def test_unknown_parameter_stays_a_string():
    # No schema entry means no guess at the type.
    calls = parse_tool_call(
        _invoke("get_weather", [("mystery", "3")]), _TOOLS)
    assert calls[0]["arguments"]["mystery"] == "3"


def test_unschemad_tool_leaves_every_value_a_string():
    calls = parse_tool_call(_invoke("other", [("days", "3")]), _TOOLS)
    assert calls[0]["arguments"]["days"] == "3"


def test_no_tools_argument_is_tolerated():
    calls = parse_tool_call(_invoke("get_weather", [("city", "Paris")]))
    assert calls[0]["name"] == "get_weather"


def test_freeform_text_becomes_an_unknown_envelope():
    # Never guess a call out of prose - hand the raw text back instead.
    out = parse_tool_call("just some prose", _TOOLS)
    assert out == {"name": "unknown", "arguments": {"raw": "just some prose"}}


def test_multiline_parameter_value_is_preserved():
    body = _invoke("get_weather", [("city", "Paris\nFrance")])
    assert parse_tool_call(body, _TOOLS)[0]["arguments"]["city"] == "Paris\nFrance"


def test_ensure_registered_grafts_into_mlx_vlm():
    pytest.importorskip("mlx_vlm.tool_parsers")
    muse_glimmer_tools.ensure_registered()
    assert "mlx_vlm.tool_parsers.muse_glimmer" in sys.modules
    mod = sys.modules["mlx_vlm.tool_parsers.muse_glimmer"]
    assert hasattr(mod, "parse_tool_call")
    muse_glimmer_tools.ensure_registered()   # idempotent
