"""DeepSeek-V4.1 prompt format and DSML tool calls.

The oracle is the reference ``encoding/tests`` pairs from the model repo:
five JSON conversations and the exact prompt each must render to. CPU-only.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from transformers.utils.chat_template_utils import _compile_jinja_template

from gmlx.load.tokenizer import bundled_chat_template
from gmlx.models.deepseek_v41 import tools as ds41_tools

_CASES = Path(__file__).resolve().parents[1] / "fixtures" / "deepseek41_encoding"


@pytest.fixture(scope="module")
def template():
    return _compile_jinja_template(bundled_chat_template("deepseek_v41"))


def _case(idx: int):
    data = json.loads((_CASES / f"test_input_{idx}.json").read_text())
    if isinstance(data, list):
        data = {"messages": data}
    kwargs = {"enable_thinking": (data.get("thinking_mode") or "chat") == "thinking"}
    if data.get("reasoning_effort") is not None:
        kwargs["reasoning_effort"] = data["reasoning_effort"]
    return data, kwargs


@pytest.mark.parametrize("idx", [1, 2, 3, 4, 5])
def test_reference_cases_render_byte_exact(template, idx):
    data, kwargs = _case(idx)
    rendered = template.render(
        messages=ds41_tools.normalize_messages(data["messages"]),
        tools=data.get("tools"),
        add_generation_prompt=True,
        **kwargs,
    )
    assert rendered == (_CASES / f"test_output_{idx}.txt").read_text()


def test_thinking_default_is_on(template):
    out = template.render(messages=[{"role": "user", "content": "hi"}],
                          add_generation_prompt=True)
    assert out.endswith("<｜Assistant｜><think>")
    assert "Reasoning Effort: 75 " in out


@pytest.mark.parametrize("effort,budget", [("low", 50), ("high", 75),
                                           ("max", 100), (42, 42)])
def test_reasoning_effort_renders_a_numeric_budget(template, effort, budget):
    out = template.render(messages=[{"role": "user", "content": "hi"}],
                          add_generation_prompt=True, reasoning_effort=effort)
    assert out.startswith(
        f"<｜begin▁of▁sentence｜><｜System｜>Reasoning Effort: {budget} ")


def test_chat_mode_has_no_effort_line(template):
    out = template.render(messages=[{"role": "user", "content": "hi"}],
                          add_generation_prompt=True, enable_thinking=False)
    assert "Reasoning Effort" not in out
    assert out.endswith("<｜Assistant｜></think>")


def test_bad_reasoning_effort_is_rejected(template):
    import jinja2

    with pytest.raises(jinja2.exceptions.TemplateError):
        template.render(messages=[{"role": "user", "content": "hi"}],
                        reasoning_effort="turbo")


def test_mid_conversation_system_message_opens_an_assistant_turn(template):
    out = template.render(
        messages=[{"role": "user", "content": "hi"},
                  {"role": "system", "content": "be brief"}],
        add_generation_prompt=True, enable_thinking=False)
    assert out.endswith("<｜System｜>be brief<｜Assistant｜></think>")


def test_tool_results_merge_into_the_user_turn(template):
    out = template.render(
        messages=[
            {"role": "user", "content": "weather?"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"type": "function",
                 "function": {"name": "get", "arguments": {"city": "Beijing"}}}]},
            {"role": "tool", "tool_call_id": "c0", "content": "sunny"},
            {"role": "tool", "tool_call_id": "c1", "content": "22C"},
        ],
        add_generation_prompt=True, enable_thinking=False)
    assert "<｜User｜><tool_result>sunny</tool_result>\n\n<tool_result>22C</tool_result>" in out


def test_earlier_reasoning_is_dropped_without_tools(template):
    msgs = [{"role": "user", "content": "a"},
            {"role": "assistant", "content": "A", "reasoning_content": "secret"},
            {"role": "user", "content": "b"}]
    out = template.render(messages=msgs, add_generation_prompt=True)
    assert "secret" not in out


def test_reasoning_is_kept_when_tools_are_present(template):
    msgs = [{"role": "system", "content": "s"},
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "A", "reasoning_content": "kept"},
            {"role": "user", "content": "b"}]
    tools = [{"type": "function", "function": {"name": "t", "description": "d",
                                               "parameters": {}}}]
    out = template.render(messages=msgs, tools=tools, add_generation_prompt=True)
    assert "kept" in out


def test_namespaced_tool_name_round_trips(template):
    tools = [{"type": "function", "namespace": "web",
              "function": {"name": "search", "description": "d",
                           "parameters": {}}}]
    out = template.render(messages=[{"role": "system", "content": "s"},
                                    {"role": "user", "content": "q"}],
                          tools=tools, add_generation_prompt=True)
    assert '{"name": "web::search"' in out


def test_rendered_prompt_carries_exactly_one_bos(template):
    """The template emits the BOS itself and add_bos_token is false, so a
    tokenizer that also prepends one would double it."""
    out = template.render(messages=[{"role": "system", "content": "s"},
                                    {"role": "user", "content": "hi"}],
                          add_generation_prompt=True)
    bos = "<｜begin▁of▁sentence｜>"
    assert out.count(bos) == 1
    assert out.startswith(bos)


# --- tool parser -----------------------------------------------------------


def test_parser_reads_a_rendered_call_block(template):
    calls = [{"type": "function", "function": {
        "name": "get_weather",
        "arguments": {"location": "Beijing", "num": 3, "flags": ["a", "b"]}}}]
    out = template.render(
        messages=[{"role": "user", "content": "q"},
                  {"role": "assistant", "content": "", "tool_calls": calls}],
        add_generation_prompt=False, enable_thinking=False)
    body = out.split(ds41_tools.tool_call_start)[1].split(
        ds41_tools.tool_call_end)[0]
    assert ds41_tools.parse_tool_call(body) == [
        {"name": "get_weather",
         "arguments": {"location": "Beijing", "num": 3, "flags": ["a", "b"]}}]


def test_parser_keeps_the_namespace_in_the_name():
    text = (
        '<｜DSML｜ invoke name="web::search">\n'
        '<｜DSML｜ parameter name="q" string="true">cats</｜DSML｜ parameter>\n'
        '</｜DSML｜ invoke>'
    )
    assert ds41_tools.parse_tool_call(text) == [
        {"name": "web::search", "arguments": {"q": "cats"}}]


def test_parser_ignores_the_v4_unspaced_tags():
    text = ('<｜DSML｜invoke name="t">'
            '<｜DSML｜parameter name="a" string="true">1</｜DSML｜parameter>'
            '</｜DSML｜invoke>')
    assert ds41_tools.parse_tool_call(text)["name"] == "unknown"


def test_normalize_messages_parses_json_argument_strings():
    msgs = [{"role": "assistant", "tool_calls": [
        {"type": "function",
         "function": {"name": "t", "arguments": '{"a": 1}'}}]}]
    out = ds41_tools.normalize_messages(msgs)
    assert out[0]["tool_calls"][0]["function"]["arguments"] == {"a": 1}
    assert msgs[0]["tool_calls"][0]["function"]["arguments"] == '{"a": 1}'


def test_normalize_messages_leaves_unparseable_arguments_alone():
    msgs = [{"role": "assistant", "tool_calls": [
        {"type": "function", "function": {"name": "t", "arguments": "oops"}}]}]
    assert ds41_tools.normalize_messages(msgs) is msgs


def test_bundled_template_resolves_by_gguf_arch():
    """The header-only paths (--report-only, the MTP and VLM loads) address
    the template by architecture, not by synthesized model type."""
    from gmlx.load.tokenizer import bundled_chat_template_for_arch

    assert (bundled_chat_template_for_arch("deepseek41")
            == bundled_chat_template("deepseek_v41"))
    assert bundled_chat_template_for_arch("deepseek4") is None
    assert bundled_chat_template_for_arch(None) is None
