"""system_content_fix: a text-only system or developer message reaches the
Gemma-4 chat template as a string.

mlx-vlm builds the messages of a Gemma-4 request as lists of text parts,
and the Gemma-4 template prints the leading system or developer message
with ``| trim``, which renders a list as its Python repr. The fix applies
to a processor the loader enabled for a Gemma-4 model type only: other
templates iterate the content of every message. Each render goes through
the stock ``apply_chat_template`` and a fixture template.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytest.importorskip("gmlx._vendor.mlx_vlm")

import gmlx._vendor.mlx_vlm.prompt_utils as pu  # noqa: E402
from transformers.utils.chat_template_utils import render_jinja_template  # noqa: E402

from gmlx.upstream.system_content_fix import (  # noqa: E402
    enable_system_content_fix,
    install_system_content_fix,
)

_FIXTURES = Path(__file__).parents[1] / "fixtures"
# The idefics3 / SmolVLM template: every message's content is iterated.
_IDEFICS3 = (
    "<|im_start|>{% for message in messages %}"
    "{{message['role'] | capitalize}}"
    "{% if message['content'][0]['type'] == 'image' %}{{':'}}"
    "{% else %}{{': '}}{% endif %}"
    "{% for line in message['content'] %}"
    "{% if line['type'] == 'text' %}{{line['text']}}"
    "{% elif line['type'] == 'image' %}{{ '<image>' }}{% endif %}"
    "{% endfor %}<end_of_utterance>\n{% endfor %}"
    "{% if add_generation_prompt %}{{ 'Assistant:' }}{% endif %}")
# DiffusionGemma's template uses the Gemma-4 turns.
_TEMPLATES = {"gemma4": (_FIXTURES / "gemma4_template.jinja").read_text(),
              "gemma4_unified": (_FIXTURES / "gemma4_template.jinja").read_text(),
              "diffusion_gemma": (_FIXTURES / "gemma4_template.jinja").read_text(),
              "qwen3_5": (_FIXTURES / "qwen35_template.jinja").read_text(),
              "idefics3": _IDEFICS3}
_GEMMA4 = ("<bos><|turn>system\nBe brief.<turn|>\n<|turn>user\nHi.<turn|>\n"
           "<|turn>model\nHello.<turn|>\n<|turn>user\nBye.<turn|>\n"
           "<|turn>model\n")
_EXPECTED = {
    "gemma4": _GEMMA4,
    "gemma4_unified": _GEMMA4,
    "diffusion_gemma": _GEMMA4,
    "qwen3_5": ("<|im_start|>system\nBe brief.<|im_end|>\n<|im_start|>user\n"
                "Hi.<|im_end|>\n<|im_start|>assistant\nHello.<|im_end|>\n"
                "<|im_start|>user\nBye.<|im_end|>\n<|im_start|>assistant\n"
                "<think>\n\n</think>\n\n"),
    "idefics3": ("<|im_start|>System: Be brief.<end_of_utterance>\n"
                 "User: Hi.<end_of_utterance>\n"
                 "Assistant: Hello.<end_of_utterance>\n"
                 "User: Bye.<end_of_utterance>\nAssistant:"),
}


class _Processor:
    """Renders a fixture template the way a tokenizer does, enabled for
    the fix the way the VLM loader enables the processor it builds."""

    def __init__(self, model_type):
        self.chat_template = _TEMPLATES[model_type]
        enable_system_content_fix(self, model_type)

    def apply_chat_template(self, messages, tokenize=False,
                            add_generation_prompt=True, **kw):
        out = render_jinja_template(
            [messages], chat_template=self.chat_template,
            add_generation_prompt=add_generation_prompt, bos_token="<bos>",
            **kw)
        rendered = out[0] if isinstance(out, tuple) else out
        return rendered[0] if isinstance(rendered, list) else rendered


@pytest.fixture(autouse=True)
def installed(monkeypatch):
    monkeypatch.setattr(pu, "get_chat_template", pu.get_chat_template)
    install_system_content_fix()
    install_system_content_fix()


def _conversation(role):
    return [{"role": role, "content": "Be brief."},
            {"role": "user", "content": "Hi."},
            {"role": "assistant", "content": "Hello."},
            {"role": "user", "content": "Bye."}]


@pytest.mark.parametrize("model_type", sorted(_TEMPLATES))
@pytest.mark.parametrize("role", ["system", "developer"])
def test_system_and_history_render_as_text(model_type, role):
    expected = _EXPECTED[model_type]
    if model_type == "idefics3":
        # The idefics3 template names each role as sent.
        expected = expected.replace("System:", f"{role.capitalize()}:")
    # enable_thinking is explicit: another test may have installed the
    # template-default wrap, which changes the qwen3.5 generation prompt.
    out = pu.apply_chat_template(_Processor(model_type),
                                 {"model_type": model_type},
                                 _conversation(role),
                                 add_generation_prompt=True,
                                 enable_thinking=False)
    assert out == expected


@pytest.mark.parametrize("model_type,enabled", [
    ("gemma4", True), ("gemma4_unified", True), ("diffusion_gemma", True),
    ("gemma4_text", False),
    ("idefics3", False), ("smolvlm", False), ("llava", False),
    ("qwen3_5", False), (None, False)])
def test_only_the_gemma4_model_types_enable_the_fix(model_type, enabled):
    class Stub:
        pass

    stub = Stub()
    assert enable_system_content_fix(stub, model_type) is enabled
    assert getattr(stub, "_gmlx_system_as_text", False) is enabled


def test_an_idefics3_template_reads_the_system_text_from_its_list():
    """Every template outside the Gemma-4 types iterates list content. A
    string there renders nothing: the loop walks its characters."""
    seen = []

    class Recording(_Processor):
        def apply_chat_template(self, messages, **kw):
            seen.append(messages[0]["content"])
            return super().apply_chat_template(messages, **kw)

    out = pu.apply_chat_template(Recording("idefics3"),
                                 {"model_type": "idefics3"},
                                 _conversation("system"),
                                 add_generation_prompt=True)
    assert "System: Be brief.<end_of_utterance>" in out
    assert isinstance(seen[0], list)


def _turn_text(rendered, role):
    return re.search(rf"<\|turn>{role}\n(.*?)<turn\|>", rendered,
                     re.S).group(1)


def test_system_text_parts_join_as_the_template_joins_user_parts():
    """The Gemma-4 template renders each text part of a user message with
    ``| trim`` and no separator. A system message's parts join the same
    way."""
    parts = [{"type": "text", "text": "Be brief. "},
             {"type": "text", "text": "\nAnswer in French."}]
    proc = _Processor("gemma4")
    as_user = proc.apply_chat_template(
        [{"role": "user", "content": parts}], add_generation_prompt=False)
    as_system = pu.get_chat_template(
        proc, [{"role": "system", "content": parts},
               {"role": "user", "content": "Hi."}], False)
    assert _turn_text(as_user, "user") == "Be brief.Answer in French."
    assert _turn_text(as_system, "system") == _turn_text(as_user, "user")


def test_only_text_only_system_and_developer_content_is_joined(monkeypatch):
    seen = []

    def render(processor, messages, add_generation_prompt, tokenize=False,
               **kwargs):
        seen.append(messages)
        return ""

    monkeypatch.setattr(pu, "get_chat_template", render)
    install_system_content_fix()
    parts = [{"type": "text", "text": "A. "}, {"type": "input_text", "text": "B."}]
    image = [{"type": "image"}, {"type": "text", "text": "C."}]
    messages = [
        {"role": "system", "content": parts, "name": "s"},
        {"role": "developer", "content": image},
        {"role": "user", "content": parts},
        {"role": "assistant", "content": parts},
    ]
    pu.get_chat_template(_Processor("gemma4"), messages, True)
    pu.get_chat_template(_Processor("idefics3"), messages, True)
    pu.get_chat_template(None, messages, True)
    assert seen == [[
        {"role": "system", "content": "A.B.", "name": "s"},
        {"role": "developer", "content": image},
        {"role": "user", "content": parts},
        {"role": "assistant", "content": parts},
    ], messages, messages]
