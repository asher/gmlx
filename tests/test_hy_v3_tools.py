"""Hy3 tool-call parser (``gmlx.hy_v3_tools``). CPU-only, no model load.

Exercises ``parse_tool_call`` on wrapper-inner text as mlx-vlm's
``process_tool_calls`` hands it over (the wrapper tags are the module's
``tool_call_start``/``tool_call_end`` and are stripped by the caller), and the
``ensure_registered`` graft into mlx-vlm's parser registry.
"""

from __future__ import annotations

import sys

import pytest

from gmlx import hy_v3_tools
from gmlx.hy_v3_tools import parse_tool_call, tool_call_end, tool_call_start

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
                    "units": {"type": "string"},
                },
            },
        },
    }
]


def _body(name, args=()):
    """One call body as the template renders it (name, sep, arg pairs)."""
    parts = [f"{name}<tool_sep:opensource>\n"]
    for k, v in args:
        parts.append(
            f"<arg_key:opensource>{k}</arg_key:opensource>\n"
            f"<arg_value:opensource>{v}</arg_value:opensource>\n"
        )
    return "".join(parts)


def _calls(*bodies):
    """Wrapper-inner text: each body inside its per-call tags."""
    return "\n".join(
        f"<tool_call:opensource>{b}</tool_call:opensource>" for b in bodies
    )


def _registry():
    return pytest.importorskip("mlx_vlm.tool_parsers")


def test_delimiters_are_the_wrapper_tags():
    # The streaming suppressor keys on tool_call_start; the inner tag would
    # let the wrapper open tag leak ('<tool_calls' diverges from '<tool_call:'
    # at the 's').
    assert tool_call_start == "<tool_calls:opensource>"
    assert tool_call_end == "</tool_calls:opensource>"


def test_parse_returns_list_of_calls():
    out = parse_tool_call(
        _calls(_body("get_weather", [("city", "Paris")]),
               _body("get_weather", [("city", "Tokyo")])),
        _TOOLS,
    )
    assert out == [
        {"name": "get_weather", "arguments": {"city": "Paris"}},
        {"name": "get_weather", "arguments": {"city": "Tokyo"}},
    ]


def test_parse_schema_string_args_stay_strings():
    # '123' is a legal city name; the declared type: string must survive.
    out = parse_tool_call(
        _calls(_body("get_weather", [("city", "123"), ("days", "3")])), _TOOLS
    )
    assert out == [{"name": "get_weather", "arguments": {"city": "123", "days": 3}}]


def test_parse_non_string_values_roundtrip_json():
    out = parse_tool_call(
        _calls(_body("get_weather", [("days", "[1, 2]"), ("units", '"metric"')])),
        _TOOLS,
    )
    # days is JSON-decoded; units is declared string so the tojson quoting is
    # kept literally (the template only tojson-serializes non-strings).
    assert out[0]["arguments"] == {"days": [1, 2], "units": '"metric"'}


def test_parse_unknown_tool_json_coerces_all_values():
    out = parse_tool_call(
        _calls(_body("other_fn", [("n", "7"), ("s", "plain")])), _TOOLS
    )
    assert out[0]["arguments"] == {"n": 7, "s": "plain"}  # unparseable stays text


def test_parse_bare_name_zero_arg_call():
    # No per-call tags: the text is treated as one bare call body.
    assert parse_tool_call("get_time\n", None) == {
        "name": "get_time",
        "arguments": {},
    }


def test_parse_freeform_text_falls_back_to_unknown():
    for text in ("not a tool call at all", "done!\nnext_step", "a\tb"):
        out = parse_tool_call(text, None)
        assert out["name"] == "unknown"
        assert out["arguments"] == {"raw": text.strip()}


def test_parse_multiline_value():
    out = parse_tool_call(
        _calls(_body("get_weather", [("city", "San\nFrancisco")])), _TOOLS
    )
    assert out[0]["arguments"] == {"city": "San\nFrancisco"}


# --- registry graft ----------------------------------------------------------


def test_ensure_registered_infers_hy_v3_from_template():
    registry = _registry()
    hy_v3_tools.ensure_registered()
    # The raw Hy3 Jinja builds its tags via .format(HYTK), so the template
    # contains the unformatted literal.
    template = "...{% set sep = '<tool_sep{}>'.format(HYTK) %}..."
    assert registry._infer_tool_parser(template) == "hy_v3"
    # A template with pre-expanded tags matches the suffixed spelling.
    assert registry._infer_tool_parser("...<tool_sep:opensource>...") == "hy_v3"
    # Stock templates are untouched by the prepend.
    assert registry._infer_tool_parser("...<arg_key>...") == "glm47"


def test_ensure_registered_makes_module_loadable():
    registry = _registry()
    hy_v3_tools.ensure_registered()
    mod = registry.load_tool_module("hy_v3")
    assert mod.parse_tool_call is parse_tool_call
    assert mod.tool_call_start == tool_call_start


def test_ensure_registered_idempotent():
    registry = _registry()
    hy_v3_tools.ensure_registered()
    before = list(registry._TEMPLATE_MARKERS)
    hy_v3_tools.ensure_registered()
    assert registry._TEMPLATE_MARKERS == before
    assert sys.modules["mlx_vlm.tool_parsers.hy_v3"] is not None
