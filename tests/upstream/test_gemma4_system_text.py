"""A Gemma 4 system or developer message renders as the text model renders it.

mlx-vlm's message rebuild builds the messages of a Gemma 4 vision type as
lists of text parts, and Gemma4Processor runs that rebuild once more inside
its own render. The 12B template prints a list system message with a space
after each part, and the older template prints it as a Python repr. The
rebuild gives a system or developer message its text as a string, as it
gives an assistant message, so each render matches the ``gemma4_text``
render of the same messages.
"""

from __future__ import annotations

import importlib
import re
from pathlib import Path

import pytest

pytest.importorskip("gmlx._vendor.mlx_vlm")

import gmlx._vendor.mlx_vlm.prompt_utils as pu  # noqa: E402
from gmlx._vendor.mlx_vlm.models.gemma4.processing_gemma4 import Gemma4Processor  # noqa: E402
from transformers.utils.chat_template_utils import render_jinja_template  # noqa: E402

from gmlx.serve.patches._common import _render_target_modules  # noqa: E402
from gmlx.serve.patches.render import install_faithful_history  # noqa: E402

_FIXTURES = Path(__file__).parents[1] / "fixtures"
_TEMPLATES = {"12b": (_FIXTURES / "gemma4_12b_template.jinja").read_text(),
              "legacy": (_FIXTURES / "gemma4_template.jinja").read_text()}
_GEMMA4_TYPES = ("gemma4", "gemma4_unified", "diffusion_gemma")
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


class _Tokenizer:
    """Renders a template as an HF tokenizer does."""

    def __init__(self, template, bos="<bos>"):
        self.chat_template = template
        self._special = {"bos_token": bos} if bos else {}

    def apply_chat_template(self, messages, tokenize=False,
                            add_generation_prompt=True, **kw):
        out = render_jinja_template(
            [messages], chat_template=self.chat_template,
            add_generation_prompt=add_generation_prompt,
            **{**self._special, **kw})
        rendered = out[0] if isinstance(out, tuple) else out
        return rendered[0] if isinstance(rendered, list) else rendered


class _Processor:
    """Forwards to its tokenizer, as the GGUF text processor does."""

    def __init__(self, template, bos="<bos>"):
        self._inner = _Tokenizer(template, bos)

    def __getattr__(self, name):
        if name == "_inner":
            raise AttributeError(name)
        return getattr(self._inner, name)


class _Gemma4Processor:
    """Renders as Gemma4Processor does: the rebuild once more, then the
    tokenizer."""

    apply_chat_template = Gemma4Processor.apply_chat_template
    _extract_media_from_messages = Gemma4Processor._extract_media_from_messages
    _as_list = staticmethod(Gemma4Processor._as_list)

    def __init__(self, template, model_type):
        self.model_type = model_type
        self.tokenizer = _Tokenizer(template)
        self.chat_template = template


@pytest.fixture
def stock():
    return pu.apply_chat_template


@pytest.fixture
def faithful(monkeypatch):
    """The served render: the faithful-history wrap and its nested layer."""
    for target in [*_render_target_modules(), pu]:
        monkeypatch.setattr(target, "apply_chat_template",
                            target.apply_chat_template)
    monkeypatch.setattr(pu, "get_chat_template", pu.get_chat_template)
    install_faithful_history()
    return importlib.import_module(
        "gmlx._vendor.mlx_vlm.server.openai").apply_chat_template


def _conversation(role):
    return [{"role": role, "content": "Be brief."},
            {"role": "user", "content": "Hi."},
            {"role": "assistant", "content": "Hello."},
            {"role": "user", "content": "Bye."}]


def _render(render, processor, model_type, messages, **kw):
    return render(processor, {"model_type": model_type},
                  [dict(m) for m in messages], add_generation_prompt=True,
                  enable_thinking=False, **kw)


def _text_render(template, messages):
    return _render(pu.apply_chat_template, _Processor(template),
                   "gemma4_text", messages)


def _system_turn(rendered):
    return re.search(r"<\|turn>system\n(.*?)<turn\|>", rendered,
                     re.S).group(1)


@pytest.mark.parametrize("path", ["stock", "faithful"])
@pytest.mark.parametrize("role", ["system", "developer"])
@pytest.mark.parametrize("template", sorted(_TEMPLATES))
@pytest.mark.parametrize("model_type", _GEMMA4_TYPES)
def test_the_gemma4_processor_renders_the_text_models_prompt(
        request, path, role, template, model_type):
    render = request.getfixturevalue(path)
    messages = _conversation(role)
    got = _render(render, _Gemma4Processor(_TEMPLATES[template], model_type),
                  model_type, messages)
    want = _text_render(_TEMPLATES[template], messages)
    assert _system_turn(want) == "Be brief."
    assert got == want


@pytest.mark.parametrize("videos", [1, 2])
@pytest.mark.parametrize("role", ["system", "developer"])
@pytest.mark.parametrize("model_type", _GEMMA4_TYPES)
def test_a_video_request_renders_the_text_models_system_turn(
        faithful, model_type, role, videos):
    """Each video once added a space ahead of the system text."""
    messages = _conversation(role)
    got = _render(faithful, _Gemma4Processor(_TEMPLATES["12b"], model_type),
                  model_type, messages,
                  video=[f"clip{i}.mp4" for i in range(videos)])
    assert _system_turn(got) == "Be brief."


_CONVERSATION_TAIL = ("<|turn>user\nHi.<turn|>\n<|turn>model\nHello.<turn|>\n"
                      "<|turn>user\nBye.<turn|>\n<|turn>model\n")
# The render of _conversation("system") on the text processor before the
# change, which joined the system parts at the template call.
_PLAIN_BEFORE = {
    "12b": ("<bos><|turn>system\nBe brief.<turn|>\n" + _CONVERSATION_TAIL
            + "<|channel>thought\n<channel|>"),
    "legacy": "<bos><|turn>system\nBe brief.<turn|>\n" + _CONVERSATION_TAIL,
}


@pytest.mark.parametrize("path", ["stock", "faithful"])
@pytest.mark.parametrize("model_type", _GEMMA4_TYPES)
def test_the_text_processor_on_a_vision_type_renders_the_text_models_prompt(
        request, path, model_type):
    """The DiffusionGemma GGUF serves its vision type on the text
    processor, which runs the rebuild once. Its prompt is the one it got
    before the change."""
    render = request.getfixturevalue(path)
    messages = _conversation("system")
    for name, template in _TEMPLATES.items():
        got = _render(render, _Processor(template), model_type, messages)
        assert got == _text_render(template, messages) == _PLAIN_BEFORE[name]


@pytest.mark.parametrize("model_type", _GEMMA4_TYPES)
@pytest.mark.parametrize("role", ["system", "developer"])
def test_the_rebuild_gives_gemma4_system_text_as_a_string(model_type, role):
    built = pu.apply_chat_template(None, {"model_type": model_type},
                                   _conversation(role), return_messages=True)
    assert built[0] == {"role": role, "content": "Be brief."}
    assert built[1]["content"] == [
        {"type": "text", "text": "Hi.", "content": "Hi."}]


@pytest.mark.parametrize("model_type", ["qwen3_5", "pixtral", "gemma3n"])
def test_other_list_types_keep_list_system_content(model_type):
    built = pu.apply_chat_template(None, {"model_type": model_type},
                                   _conversation("system"),
                                   return_messages=True)
    assert isinstance(built[0]["content"], list)


@pytest.mark.parametrize("model_type,template,want", [
    ("qwen3_5", (_FIXTURES / "qwen35_template.jinja").read_text(),
     "<|im_start|>system\nBe brief.<|im_end|>\n<|im_start|>user\n"
     "Hi.<|im_end|>\n<|im_start|>assistant\nHello.<|im_end|>\n"
     "<|im_start|>user\nBye.<|im_end|>\n<|im_start|>assistant\n"
     "<think>\n\n</think>\n\n"),
    ("idefics3", _IDEFICS3,
     "<|im_start|>System: Be brief.<end_of_utterance>\n"
     "User: Hi.<end_of_utterance>\nAssistant: Hello.<end_of_utterance>\n"
     "User: Bye.<end_of_utterance>\nAssistant:"),
])
def test_other_templates_render_as_before(model_type, template, want):
    out = _render(pu.apply_chat_template, _Processor(template, bos=None),
                  model_type, _conversation("system"))
    assert out == want


_IMAGE = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}}
_CALL = [{"id": "c1", "type": "function",
          "function": {"name": "look", "arguments": "{}"}}]
_THOUGHT_OFF = "<|turn>model\n<|channel>thought\n<channel|>"
# (messages, render kwargs, the render before the change): requests with no
# system or developer message.
_NO_SYSTEM = {
    "images-in-place": ([{"role": "user", "content": [
        {"type": "text", "text": "Look at"}, _IMAGE,
        {"type": "text", "text": "and"}, _IMAGE]}], {"num_images": 2},
        "<bos><|turn>user\nLook at<|image|>and<|image|><turn|>\n"
        + _THOUGHT_OFF),
    "audio": ([{"role": "user", "content": [
        {"type": "text", "text": "Listen."},
        {"type": "input_audio", "input_audio": {"data": "AA==", "format": "wav"}}]}],
        {"num_audios": 1},
        "<bos><|turn>user\nListen.<|audio|><turn|>\n" + _THOUGHT_OFF),
    "video": (_conversation("user")[1:], {"video": ["clip.mp4"]},
              "<bos><|turn>user\n<|video|>Hi.<turn|>\n<|turn>model\n"
              "<|video|>Hello.<turn|>\n<|turn>user\n<|video|>Bye.<turn|>\n"
              + _THOUGHT_OFF),
    "tool-images": ([
        {"role": "user", "content": "Look."},
        {"role": "assistant", "content": "", "tool_calls": _CALL},
        {"role": "tool", "tool_call_id": "c1",
         "content": [{"type": "image"}, {"type": "text", "text": "Done."}]},
        {"role": "user", "content": "What is it?"}], {"num_images": 1},
        "<bos><|turn>user\nLook.<turn|>\n<|turn>model\n"
        "<|tool_call>call:look{}<tool_call|><|tool_response>response:look"
        "{value:<|\"|>Done.<|\"|>}<tool_response|><|image|>"
        "<|turn>user\nWhat is it?<turn|>\n" + _THOUGHT_OFF),
    "reasoning-tool-calls": ([
        {"role": "user", "content": "Weather?"},
        {"role": "assistant", "content": "", "reasoning_content": "Check.",
         "tool_calls": _CALL},
        {"role": "tool", "tool_call_id": "c1", "content": "Sunny."}], {},
        "<bos><|turn>user\nWeather?<turn|>\n<|turn>model\n"
        "<|channel>thought\nCheck.\n<channel|><|tool_call>call:look{}"
        "<tool_call|><|tool_response>response:look"
        "{value:<|\"|>Sunny.<|\"|>}<tool_response|>"),
}


@pytest.mark.parametrize("case", sorted(_NO_SYSTEM))
def test_a_render_with_no_system_message_does_not_change(faithful, case):
    messages, kw, before = _NO_SYSTEM[case]
    processor = _Gemma4Processor(_TEMPLATES["12b"], "gemma4_unified")
    assert _render(faithful, processor, "gemma4_unified", messages,
                   **kw) == before


@pytest.mark.integration
def test_the_gguf_vocab_encodes_no_space_before_the_system_turn_end(
        gguf_index, faithful):
    """On a Gemma 4 GGUF's own vocab and template, the vision render of a
    system message encodes to the text model's ids. Header only."""
    from helpers import require_arch
    from gmlx.distill.tokens import tokenizer_from_gguf

    tok = tokenizer_from_gguf(require_arch(gguf_index, "gemma4"))
    processor = _Gemma4Processor(tok.chat_template, "gemma4_unified")
    processor.tokenizer = tok
    messages = _conversation("system")
    got = tok.encode(_render(faithful, processor, "gemma4_unified", messages),
                     add_special_tokens=False)
    want = tok.encode(_render(pu.apply_chat_template, tok, "gemma4_text",
                              messages), add_special_tokens=False)
    end = tok.convert_tokens_to_ids("<turn|>")
    assert got[got.index(end) - 1] != tok.convert_tokens_to_ids("\u2581")
    assert got == want
