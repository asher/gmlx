"""The continuation of a final assistant message, as llama-server renders it:
the field parse, the mode, the handler class of each template and each
class's prefill."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gmlx.gen import continuation as cont

_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


@pytest.mark.parametrize("value,want", [
    (True, "auto"), (False, None), (None, None),
    ("content", "content"), ("reasoning_content", "reasoning_content"),
])
def test_parse_continue_reads_llama_values(value, want):
    assert cont.parse_continue(value) == want


@pytest.mark.parametrize("value", ["yes", 0, 1, [], {}])
def test_parse_continue_refuses_other_values(value):
    with pytest.raises(ValueError) as e:
        cont.parse_continue(value)
    assert str(e.value) == ('Invalid continue_final_message: expected a '
                            'boolean, "content" or "reasoning_content"')


def _msg(content=None, reasoning=None):
    m = {"role": "assistant", "content": content}
    if reasoning is not None:
        m["reasoning_content"] = reasoning
    return m


@pytest.mark.parametrize("requested,message,mode,content,reasoning", [
    ("auto", _msg("C"), "content", "C", ""),
    ("auto", _msg(None, "R"), "reasoning_content", "", "R"),
    ("auto", _msg("C", "R"), "content", "C", "R"),
    ("auto", _msg("", "R"), "reasoning_content", "", "R"),
    ("auto", _msg([], "R"), "reasoning_content", "", "R"),
    ("auto", _msg([{"type": "text", "text": ""}], "R"), "content", "", "R"),
    ("auto", _msg([{"type": "image_url", "image_url": {"url": "u"}}]),
     "content", "", ""),
    ("auto", _msg([{"type": "text", "text": " a "}, {"type": "text", "text": "b"}]),
     "content", " a \n\nb", ""),
    ("content", _msg(None, "R"), "content", "", "R"),
    ("reasoning_content", _msg("C", "R"), "reasoning_content", "C", "R"),
])
def test_resolve_picks_the_mode_from_the_raw_message(requested, message, mode,
                                                     content, reasoning):
    got = cont.resolve(requested, message, "qwen3-coder")
    assert (got.mode, got.content, got.reasoning) == (mode, content, reasoning)


@pytest.mark.parametrize("message", [_msg(None), _msg(""), _msg([]),
                                     _msg("", "")])
def test_an_empty_message_has_no_continuation(message):
    assert cont.resolve("auto", message) is None


def test_render_content_keeps_llama_delimiters():
    parts = [{"type": "text", "text": "a"}, {"type": "text", "text": ""},
             {"type": "text", "text": "b"}]
    assert cont.render_content(parts) == "a\n\n\n\nb"
    assert cont.render_content([{"type": "text", "text": ""},
                                {"type": "text", "text": "b"}]) == "b"


_HANDLERS = {
    "qwen35_template.jinja": "qwen3-coder",
    "qwen35_9b_template.jinja": "qwen3-coder",
    "qwen36_template.jinja": "qwen3-coder",
    "qwen38_template.jinja": "qwen3-coder",
    "minicpmv46_template.jinja": "qwen3-coder",
    "minicpmv47_template.jinja": "qwen3-coder",
    "nemotron_bolt_template.jinja": "qwen3-coder",
    "gemma4_template.jinja": "gemma4",
    "gemma4_12b_template.jinja": "gemma4",
    "gptoss_template.jinja": "gpt-oss",
    "beam_template.jinja": "harmony-xml",
    "kimi_k25_template.jinja": "kimi-k2",
    "kimi_k3_template.jinja": "kimi-k3",
    "k2_horizon_template.jinja": "k2-horizon",
    "minimax_m31_template.jinja": "minimax-m3",
    "minimax_m31_turns.jinja": "minimax-m3",
    "deepseek_v4_template.jinja": "deepseek-v3-2",
    "muse_template.jinja": "muse-glimmer",
    "muse_spark_template.jinja": "muse-spark",
    "mistral3_template.jinja": "ministral-3",
    "qwen3_template.jinja": "autoparser",
    "glm5_template.jinja": "autoparser",
    "flint_template.jinja": "autoparser",
    "ricursive_template.jinja": "autoparser",
    "minimax_m31_thinking.jinja": "autoparser",
    "mistral7b_v03_template.jinja": "autoparser",
    "granite_switch_template.jinja": "autoparser",
    "granite_switch_systemone.jinja": "autoparser",
    "smollm2_template.jinja": "autoparser",
}


def test_every_template_fixture_has_a_handler_row():
    assert {p.name for p in _FIXTURES.glob("*.jinja")} == set(_HANDLERS)


@pytest.mark.parametrize("name", sorted(_HANDLERS))
def test_handler_for_matches_llama(name):
    assert cont.handler_for((_FIXTURES / name).read_text()) == _HANDLERS[name]


def test_a_gemma_template_without_the_openai_line_is_legacy():
    src = (_FIXTURES / "gemma4_12b_template.jinja").read_text()
    assert "{#- OpenAI Chat Completions:" in src
    legacy = src.replace("{#- OpenAI Chat Completions:", "{#- Chat:")
    assert cont.handler_for(legacy) == "gemma4-legacy"


def test_the_handlers_match_the_ones_llama_server_logged():
    fixture = json.loads((_FIXTURES / "llama_continuation.json").read_text())
    for row in fixture["rows"]:
        src = (_FIXTURES / row["template"]).read_text()
        assert cont.handler_for(src) == row["handler"], row["id"]


def test_the_autoparser_list_is_the_fixture_templates():
    """The accepted autoparser templates are the ones llama-server rendered
    for the fixture, and no row of theirs shows a reasoning marker."""
    fixture = json.loads((_FIXTURES / "llama_continuation.json").read_text())
    names = {r["template"] for r in fixture["rows"] if r["handler"] == "autoparser"}
    assert {fixture["files"][n]["sha256"] for n in names} \
        == cont.AUTOPARSER_TEMPLATES
    for row in fixture["rows"]:
        if row["handler"] == "autoparser":
            assert "plan" not in row["prompt"], row["id"]


@pytest.mark.parametrize("name,ok", [
    ("smollm2_template.jinja", True),
    ("mistral7b_v03_template.jinja", False),
    ("granite_switch_template.jinja", False),
    ("granite_switch_systemone.jinja", False),
    ("qwen3_template.jinja", False),
    ("qwen35_9b_template.jinja", True),
    ("gemma4_12b_template.jinja", True),
    ("gptoss_template.jinja", False),
    ("kimi_k3_template.jinja", False),
])
def test_recipe_supported(name, ok):
    src = (_FIXTURES / name).read_text()
    assert cont.recipe_supported(cont.handler_for(src), src, []) is ok


def test_gemma4_legacy_is_refused_with_a_tool_message():
    assert cont.recipe_supported("gemma4-legacy", "", [{"role": "user"}])
    assert not cont.recipe_supported(
        "gemma4-legacy", "", [{"role": "user"}, {"role": "tool"}])


def _render(base, gen):
    return lambda g: base + (gen if g else "")


def _c(handler, mode, content="C", reasoning="R"):
    return cont.Continuation(mode=mode, content=content, reasoning=reasoning,
                             handler=handler)


@pytest.mark.parametrize("mode,template,tail", [
    ("content", "<think>", "<|im_start|>assistant\n<think>\nR\n</think>\n\nC"),
    ("reasoning_content", "<think>", "<|im_start|>assistant\n<think>\nR"),
    ("content", "", "<|im_start|>assistant\nC"),
    ("reasoning_content", "", "<|im_start|>assistant\n"),
])
def test_qwen3_coder_recipe(mode, template, tail):
    got = cont.continued_prompt(_c("qwen3-coder", mode), template,
                                _render("H", "<gen>"))
    assert got == "H" + tail


@pytest.mark.parametrize("base,mode,tail", [
    ("H<turn|>\n", "content", "<|turn>model\n<|channel>thought\nR<channel|>C"),
    ("H<turn|>\n", "reasoning_content", "<|turn>model\n<|channel>thought\nR"),
    ("H", "content", "<|channel>thought\nR<channel|>C"),
])
def test_gemma4_recipe(base, mode, tail):
    for handler in ("gemma4", "gemma4-legacy"):
        got = cont.continued_prompt(_c(handler, mode), "", _render(base, "<gen>"))
        assert got == base + tail


@pytest.mark.parametrize("mode,tail", [
    ("content", "<|im_start|>assistant\nC"),
    ("reasoning_content", "<|im_start|>assistant\n"),
])
def test_autoparser_recipe_appends_the_generation_prompt(mode, tail):
    got = cont.continued_prompt(_c("autoparser", mode), "",
                                _render("H", "<|im_start|>assistant\n"))
    assert got == "H" + tail


def test_generation_prompt_is_the_common_prefix_diff():
    assert cont.generation_prompt(
        lambda g: "abX" + ("Y" if g else "Z")) == "Y"


def test_a_render_that_returns_no_string_raises_type_error():
    for handler in ("qwen3-coder", "gemma4", "autoparser"):
        with pytest.raises(TypeError):
            cont.continued_prompt(_c(handler, "content"), "",
                                  lambda g: {"role": "assistant"})


def test_a_handler_without_a_recipe_raises_value_error():
    with pytest.raises(ValueError):
        cont.continued_prompt(_c("gpt-oss", "content"), "", _render("H", ""))
