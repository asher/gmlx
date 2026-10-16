"""Muse Spark chat surface: the chat template the GGUF ships, its
reasoning_effort levels, and the ATEM tool calls it frames. CPU-only, no
model load. The template renders through the same transformers Jinja path
that a GGUF tokenizer's apply_chat_template takes."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from jinja2.exceptions import TemplateError
from transformers.utils.chat_template_utils import render_jinja_template

from gmlx.gen import profiles

_TEMPLATE = (
    Path(__file__).parents[1] / "fixtures" / "muse_spark_template.jinja"
).read_text()
_HI = [{"role": "user", "content": "hi"}]
_WEATHER = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}, "days": {"type": "integer"}},
            },
        },
    }
]


def _render(messages=_HI, add_generation_prompt=True, **kw):
    out = render_jinja_template(
        [messages],
        chat_template=_TEMPLATE,
        add_generation_prompt=add_generation_prompt,
        bos_token="<|begin_of_text|>",
        current_date="2026-10-16",
        **kw,
    )
    rendered = out[0] if isinstance(out, tuple) else out
    return rendered[0] if isinstance(rendered, list) else rendered


# reasoning_effort maps to the "Reasoning strength" line of the system turn.
_STRENGTH = {"low": 8, "medium": 32, "high": 256, "xhigh": 512}


@pytest.mark.parametrize("effort", sorted(_STRENGTH))
def test_each_reasoning_effort_sets_its_strength(effort):
    prompt = _render(reasoning_effort=effort)
    assert f"Reasoning strength: {_STRENGTH[effort]}." in prompt
    assert prompt.count("Reasoning strength:") == 1


@pytest.mark.parametrize("value", [None, ""])
def test_reasoning_effort_defaults_to_xhigh(value):
    kw = {} if value is None else {"reasoning_effort": value}
    assert "Reasoning strength: 512." in _render(**kw)


@pytest.mark.parametrize("effort", ["minimal", "none", "max", "HIGH"])
def test_an_unknown_reasoning_effort_is_refused(effort):
    with pytest.raises(TemplateError, match="Unsupported reasoning_effort"):
        _render(reasoning_effort=effort)


def test_each_profile_intent_renders():
    fam = profiles.FAMILIES["muse-spark"]
    for name, delta in fam["intents"].items():
        level = name.split("-", 1)[1]
        prompt = _render(**delta["chat_template_kwargs"])
        assert f"Reasoning strength: {_STRENGTH[level]}." in prompt


def test_a_reasoning_strength_line_in_the_system_turn_wins():
    prompt = _render(
        [{"role": "system", "content": "Reasoning effort: 99."}, *_HI],
        reasoning_effort="low",
    )
    assert "Reasoning strength: 99." in prompt
    assert "Reasoning strength: 8." not in prompt


def test_the_generation_prompt_stops_mid_header():
    """The model writes its own recipient, so the serve and REPL splitters
    start inside a header and the budget does not start counting."""
    from gmlx.gen.thinking_budget import prompt_opens_thinking
    from gmlx.tui.reasoning import prompt_opens_header

    prompt = _render()
    assert prompt.endswith("<|eot|><|start|>assistant")
    assert prompt_opens_header(prompt)
    s = profiles.family_base("muse-spark")["sampling"]
    assert not prompt_opens_thinking(
        prompt, s["thinking_start_token"], s["thinking_end_token"]
    )


def test_a_split_reply_renders_back_as_the_model_wrote_it():
    """The split reply goes back as history. Reasoning renders to self,
    and content with a tool call renders as a commentary aside before it."""
    from gmlx.tui.reasoning import split_harmony_reply

    reasoning, content = split_harmony_reply(
        " to=self<|message|>Need the weather.<|eom|>"
        "<|start|>assistant to=commentary<|message|>Let me look.<|eom|>"
        "<|start|>assistant to=get_weather<|message|>",
        start_in_header=True,
    )
    history = [
        *_HI,
        {
            "role": "assistant",
            "reasoning_content": reasoning,
            "content": content,
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": {"city": "Paris"}},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "Sunny"},
    ]
    prompt = _render(history, tools=_WEATHER)
    assert (
        "<|start|>assistant to=self<|message|>Need the weather.<|eom|>"
        "<|start|>assistant to=commentary<|message|>Let me look.<|eom|>"
        "<|start|>assistant to=get_weather<|message|><atem:function_calls>" in prompt
    )
    assert '<atem:parameter name="city">Paris</atem:parameter>' in prompt
    assert "<|start|>tool get_weather<|message|>" in prompt


def test_string_tool_arguments_need_the_serve_decode():
    """The template refuses JSON-string arguments; serve decodes them for
    muse_spark before it renders."""
    from gmlx.serve.patches.render import _decode_tool_arguments

    msgs = [
        *_HI,
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"city": "Paris"}',
                    },
                }
            ],
        },
    ]
    with pytest.raises(TemplateError, match="dict"):
        _render(msgs, tools=_WEATHER)
    _decode_tool_arguments(msgs)
    assert '<atem:parameter name="city">Paris</atem:parameter>' in _render(
        msgs, tools=_WEATHER
    )


def test_the_template_routes_to_the_atem_parser():
    import gmlx.models.muse_glimmer.tools as atem_tools
    from gmlx._vendor.mlx_vlm.tools import registry

    atem_tools.ensure_registered()
    assert registry._infer_tool_parser(_TEMPLATE) == "muse_glimmer"


def test_parallel_calls_each_in_their_own_message_parse():
    """The template asks for one call per message. Each message carries its
    own wrapper block, so every call parses."""
    import gmlx.models.muse_glimmer.tools as atem_tools
    from gmlx._vendor.mlx_vlm.tools.extractor import process_tool_calls

    def call(city):
        return (
            '<atem:function_calls>\n<atem:invoke name="get_weather">\n'
            f'<atem:parameter name="city">{city}</atem:parameter>\n'
            '<atem:parameter name="days">3</atem:parameter>\n'
            "</atem:invoke>\n</atem:function_calls>"
        )

    text = (
        " to=commentary<|message|>Checking both.<|eom|>"
        "<|start|>assistant to=get_weather<|message|>"
        + call("Paris")
        + "<|eom|><|start|>assistant to=get_weather<|message|>"
        + call("Rome")
    )
    tc = process_tool_calls(text, atem_tools, _WEATHER)
    assert [json.loads(c["function"]["arguments"]) for c in tc.calls] == [
        {"city": "Paris", "days": 3},
        {"city": "Rome", "days": 3},
    ]
