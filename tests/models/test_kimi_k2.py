"""Moonshot Kimi-K2.5/K2.7 (llama.cpp arch 'deepseek2').

K2.x is DeepSeek-V3-shaped - MLA + fine-grained sigmoid-gated MoE - so it
converts to the shared 'deepseek2' arch and rides the stock deepseek_v3 model
class. What is Kimi-specific is the surrounding contract: Moonshot's sampling
card, the <|im_*|> chat format whose generation prompt leaves <think> open, and
the tool-call section format.
"""

from __future__ import annotations

import pytest

import gmlx.gen.profiles as profiles
from gmlx.gen.thinking_budget import prompt_opens_thinking

# The generation prompt the K2.x template renders for a single user turn.
KIMI_GEN_PROMPT = (
    "<|im_user|>user<|im_middle|>hello<|im_end|>"
    "<|im_assistant|>assistant<|im_middle|><think>"
)


def test_generation_prompt_leaves_thinking_open():
    """K2.x forces thinking by opening a bare '<think>' after the assistant
    header, so the stream starts *inside* reasoning. Without this the first
    reasoning tokens render as answer text."""
    assert prompt_opens_thinking(KIMI_GEN_PROMPT)
    # Once the model closes the block, the prompt no longer opens one.
    assert not prompt_opens_thinking(KIMI_GEN_PROMPT + "musing</think>answer")


def test_kimi_k2_sampling_family():
    assert profiles.detect_family("deepseek2", "Kimi-K2.7-Code") == "kimi-k2"
    assert profiles.family_base("kimi-k2")["sampling"]["temperature"] == 1.0


def test_tool_parser_resolves_from_the_template():
    """mlx-vlm's registry already ships a kimi_k2 parser; K2.x resolves onto it
    by template marker, so gmlx needs no parser of its own. This pins the
    resolution - a registry rename would silently drop tool calling."""
    tool_parsers = pytest.importorskip("mlx_vlm.tool_parsers")
    template = (
        "{%- if tools %}<|im_system|>tool_declare<|im_middle|>{{ tools }}"
        "<|im_end|>{% endif %}<|tool_calls_section_begin|>"
        "<|tool_call_begin|>{{ id }}<|tool_call_argument_begin|>{{ args }}"
        "<|tool_call_end|><|tool_calls_section_end|>"
    )
    assert tool_parsers._infer_tool_parser(template) == "kimi_k2"


def test_tool_call_section_parses():
    """Kimi ids carry the function name and a counter: functions.<name>:<n>."""
    kimi_k2 = pytest.importorskip("mlx_vlm.tool_parsers.kimi_k2")
    body = (
        "<|tool_call_begin|>functions.read_file:0<|tool_call_argument_begin|>"
        '{"path": "a.py", "n": 3}<|tool_call_end|>'
        "<|tool_call_begin|>functions.ls:1<|tool_call_argument_begin|>"
        "{}<|tool_call_end|>"
    )
    assert kimi_k2.tool_call_start == "<|tool_calls_section_begin|>"
    assert kimi_k2.parse_tool_call(body) == [
        {"id": "functions.read_file:0", "name": "read_file",
         "arguments": {"path": "a.py", "n": 3}},
        {"id": "functions.ls:1", "name": "ls", "arguments": {}},
    ]
