"""/v1/messages renders a user message's tool results before its text, in
block order, as the Messages API orders them. The stock conversion emits the
text first, between the assistant's tool call and the tool response, and the
Gemma-4 templates then drop the tool response. The Mistral3 template rejects
a user turn right after a tool turn, so there the text folds into the tool
results.

An image in a tool result renders once: in the tool turn on a template that
shows an image there, else in a user message right after the tool results,
as the Responses route places it. The stock render also counts it into the
last user message."""

from __future__ import annotations

import importlib
import types
from pathlib import Path

import pytest

pytest.importorskip("gmlx._vendor.mlx_vlm")

from jinja2.exceptions import UndefinedError  # noqa: E402
from gmlx._vendor.mlx_vlm.models.gemma4.processing_gemma4 import Gemma4Processor  # noqa: E402
from transformers.utils.chat_template_utils import render_jinja_template  # noqa: E402

from gmlx.load.tokenizer import bundled_chat_template, finish_gguf_tokenizer  # noqa: E402
from gmlx.serve.patches import _common as sp_common  # noqa: E402
from gmlx.serve.patches import render  # noqa: E402
from gmlx.serve.patches.chat_behavior import install_role_normalization  # noqa: E402
from gmlx.serve.patches.messages_route import install_messages_route  # noqa: E402
from gmlx.serve.patches.render import (  # noqa: E402
    install_chat_image_placement,
    install_faithful_history,
)
from gmlx.upstream.patching import under, wrapped_with  # noqa: E402

_ANTHROPIC = importlib.import_module("gmlx._vendor.mlx_vlm.server.anthropic")
_OPENAI = importlib.import_module("gmlx._vendor.mlx_vlm.server.openai")
_PROMPT_UTILS = importlib.import_module("gmlx._vendor.mlx_vlm.prompt_utils")
_APP = importlib.import_module("gmlx._vendor.mlx_vlm.server.app")
_RUNTIME = importlib.import_module("gmlx._vendor.mlx_vlm.server.runtime").runtime
_FIXTURES = Path(__file__).parents[1] / "fixtures"
_CLAIM = "_gmlx_messages_route"

# Template: model type and chat template (a GGUF header's, or the one gmlx
# ships for the type).
_TEMPLATES = {
    "gemma4": ("gemma4", "gemma4_template.jinja"),
    "gemma4-12b": ("gemma4_unified", "gemma4_12b_template.jinja"),
    "qwen3.5": ("qwen3_5", "qwen35_template.jinja"),
    "qwen3.6": ("qwen3_5_moe", "qwen36_template.jinja"),
    "qwen3.8": ("qwen3_5", "qwen38_template.jinja"),
    "qwen3": ("qwen3", "qwen3_template.jinja"),
    "gpt-oss": ("gpt_oss", "gptoss_template.jinja"),
    "deepseek-v4": ("deepseek_v4", "deepseek_v4_template.jinja"),
    "deepseek-v4.1": ("deepseek_v41", None),
    "kimi-k2.5": ("kimi_k25", "kimi_k25_template.jinja"),
    "kimi-k3": ("kimi_k3", "kimi_k3_template.jinja"),
    "mistral3": ("mistral3", "mistral3_template.jinja"),
    "muse-spark": ("muse_spark", "muse_spark_template.jinja"),
}
# Templates the Messages cases above do not take: Mistral 7B v0.3 takes
# nine-character call ids only.
_CHAT_TEMPLATES = {
    "mistral-7b": ("mistral", "mistral7b_v03_template.jinja"),
}


class _Processor:
    """Renders the template as the tokenizer gmlx builds for the type."""

    def __init__(self, model_type, fixture):
        base = model_type.removesuffix("_vl")
        self.chat_template = (
            (_FIXTURES / fixture).read_text() if fixture
            else bundled_chat_template(base))
        finish_gguf_tokenizer(self, base)

    def apply_chat_template(self, messages, tokenize=False,
                            add_generation_prompt=True, **kw):
        out = render_jinja_template(
            [messages], chat_template=self.chat_template,
            add_generation_prompt=add_generation_prompt, **kw)
        rendered = out[0] if isinstance(out, tuple) else out
        return rendered[0] if isinstance(rendered, list) else rendered


class _Gemma4Processor:
    """Renders as mlx-vlm's Gemma4Processor does: the stock rebuild once
    more, then the tokenizer."""

    apply_chat_template = Gemma4Processor.apply_chat_template
    _extract_media_from_messages = Gemma4Processor._extract_media_from_messages
    _as_list = staticmethod(Gemma4Processor._as_list)

    def __init__(self, model_type, fixture):
        self.model_type = model_type
        self.tokenizer = _Processor(model_type, fixture)
        self.chat_template = self.tokenizer.chat_template


class _Tok:
    text, finish_reason, token_count = "ok", "stop", 1


class _RG:
    def __init__(self):
        self.calls = []

    def generate(self, prompt, images=None, audio=None, args=None,
                 videos=None):
        self.calls.append((prompt, list(images or ())))
        return types.SimpleNamespace(prompt_tokens=1, uid=1), iter([_Tok()])


@pytest.fixture
def route(monkeypatch):
    """/v1/messages with the gmlx render wraps, on a stub model."""
    yield from _route(monkeypatch)


@pytest.fixture
def killed_route(monkeypatch):
    """/v1/messages installed under ``GMLX_FAITHFUL_HISTORY=0``."""
    monkeypatch.setenv("GMLX_FAITHFUL_HISTORY", "0")
    yield from _route(monkeypatch)


def _route(monkeypatch):
    routes = sp_common._snapshot_routes(_APP.app)
    for target in sp_common._render_target_modules():
        monkeypatch.setattr(target, "apply_chat_template",
                            target.apply_chat_template)
    for name in ("get_chat_template", "apply_chat_template"):
        monkeypatch.setattr(_PROMPT_UTILS, name, getattr(_PROMPT_UTILS, name))
    for name in ("extract_text_from_content",
                 "_normalize_instruction_messages"):
        monkeypatch.setattr(_OPENAI, name, getattr(_OPENAI, name))
    # The vendored table lacks the DeepSeek rows (the loader adds them).
    for row in ("deepseek_v4_vl", "deepseek_v41_vl"):
        monkeypatch.setitem(_PROMPT_UTILS.MODEL_CONFIG, row,
                            _PROMPT_UTILS.MessageFormat.LIST_WITH_IMAGE_FIRST)
    conversion = _ANTHROPIC._anthropic_messages_to_internal
    if wrapped_with(conversion, _CLAIM):
        conversion = under(conversion, _CLAIM)
    monkeypatch.setattr(_ANTHROPIC, "_anthropic_messages_to_internal",
                        conversion)
    rg = _RG()
    monkeypatch.setattr(_RUNTIME, "response_generator", rg, raising=False)
    install_faithful_history()
    install_role_normalization()
    install_chat_image_placement()
    install_messages_route()
    from fastapi.testclient import TestClient

    client = TestClient(_APP.app)

    def render(name, messages, stock=False, vision=False, status=200,
               path="/v1/messages", processor=None):
        model_type, fixture = (
            _VISION if vision else {**_TEMPLATES, **_CHAT_TEMPLATES})[name][:2]
        kind = (_Gemma4Processor if model_type.startswith("gemma4")
                else _Mistral7B if name == "mistral-7b" else _Processor)
        processor = processor or kind(model_type, fixture)
        if vision:
            processor.image_token = _VISION[name][2]
        for module in (_ANTHROPIC, _OPENAI):
            monkeypatch.setattr(module, "get_cached_model", lambda *a, **k: (
                None, processor, {"model_type": model_type}))
        if stock:
            monkeypatch.setattr(_ANTHROPIC, "_anthropic_messages_to_internal",
                                conversion)
        if path == "/v1/responses":
            body = {"model": "m", "max_output_tokens": 16, "input": messages}
        elif path == "/v1/chat/completions":
            body = {"model": "m", "max_tokens": 1, "messages": messages}
        else:
            body = {"model": "m", "max_tokens": 1, "messages": messages,
                    "tools": _TOOLS}
        r = client.post(path, json=body)
        assert r.status_code == status, r.text
        return rg.calls.pop() if status == 200 else r.text

    yield render
    sp_common._restore_routes(_APP.app, routes)


_TOOLS = [{"name": "shot", "description": "Take a screenshot.",
           "input_schema": {"type": "object", "properties": {}}}]


def _text(t):
    return {"type": "text", "text": t}


def _result(i, text):
    return {"type": "tool_result", "tool_use_id": f"toolu_{i}",
            "content": [_text(text)]}


def _round(*blocks, calls=1):
    """The user asks, the assistant calls ``shot`` ``calls`` times, and the
    next user message holds ``blocks``."""
    uses = [{"type": "tool_use", "id": f"toolu_{i}", "name": "shot",
             "input": {}} for i in range(calls)]
    return [{"role": "user", "content": [_text("ASK-FIRST")]},
            {"role": "assistant", "content": [_text("Taking it.")] + uses},
            {"role": "user", "content": list(blocks)}]


# Conversation: the texts that must render in this order after the call.
_WITH_TEXT = {
    "a result and text": (
        _round(_result(0, "RESULT-ONE"), _text("TEXT-AFTER")),
        ("RESULT-ONE", "TEXT-AFTER")),
    "two results and text": (
        _round(_result(0, "RESULT-ONE"), _result(1, "RESULT-TWO"),
               _text("TEXT-AFTER"), calls=2),
        ("RESULT-ONE", "RESULT-TWO", "TEXT-AFTER")),
    "a later turn": (
        _round(_result(0, "RESULT-ONE"), _text("TEXT-AFTER")) + [
            {"role": "assistant", "content": [_text("ANSWER")]},
            {"role": "user", "content": [_text("NEXT-QUESTION")]}],
        ("RESULT-ONE", "TEXT-AFTER", "ANSWER", "NEXT-QUESTION")),
}


@pytest.mark.parametrize("case", sorted(_WITH_TEXT))
@pytest.mark.parametrize("name", sorted(_TEMPLATES))
def test_tool_results_render_before_the_text_beside_them(route, name, case):
    messages, order = _WITH_TEXT[case]
    prompt, _ = route(name, messages)
    pos = prompt.index("ASK-FIRST")
    for want in order:
        found = prompt.find(want, pos)
        assert found >= 0, f"{want!r} missing or out of order in {prompt!r}"
        pos = found + len(want)


_RESULTS_ONLY = {
    "one result": _round(_result(0, "RESULT-ONE")),
    "two results": _round(_result(0, "RESULT-ONE"), _result(1, "RESULT-TWO"),
                          calls=2),
}


@pytest.mark.parametrize("case", sorted(_RESULTS_ONLY))
@pytest.mark.parametrize("name", sorted(_TEMPLATES))
def test_a_message_of_tool_results_renders_as_before(route, name, case):
    messages = _RESULTS_ONLY[case]
    assert route(name, messages) == route(name, messages, stock=True)


_QWEN = "<|image_pad|>"
_DEEPSEEK = "<\uff5cdeepseek_image\uff5c>"
# Template: model type, chat template, and the text one image renders as.
_VISION = {
    "qwen3.5": ("qwen3_5", "qwen35_template.jinja", _QWEN),
    "qwen3.6": ("qwen3_5_moe", "qwen36_template.jinja", _QWEN),
    "qwen3.8": ("qwen3_5", "qwen38_template.jinja", _QWEN),
    "kimi-k2.5": ("kimi_k25", "kimi_k25_template.jinja", "<|media_pad|>"),
    "gemma4-12b": ("gemma4_unified", "gemma4_12b_template.jinja",
                   "<|image|>"),
    "gemma4": ("gemma4", "gemma4_template.jinja", "<|image|>"),
    "deepseek-v4": ("deepseek_v4_vl", "deepseek_v4_template.jinja",
                    _DEEPSEEK),
    "deepseek-v4.1": ("deepseek_v41_vl", None, _DEEPSEEK),
    "mistral3": ("mistral3", "mistral3_template.jinja", "[IMG]"),
    "glm-5.3": ("glm5_next", "glm5_template.jinja", "<|begin_of_image|>"),
    "muse": ("muse_glimmer", "muse_template.jinja", "<|patch|>"),
    "muse-spark": ("muse_spark", "muse_spark_template.jinja", "<|patch|>"),
}
# The templates that show an image in a tool turn.
_IN_TOOL_TURN = ("gemma4-12b", "glm-5.3", "kimi-k2.5", "muse", "muse-spark",
                 "qwen3.5", "qwen3.6", "qwen3.8")


def test_the_tool_image_types_are_the_checked_ones():
    assert set(render._TOOL_IMAGES) == {_VISION[n][0] for n in _IN_TOOL_TURN}


def _image(name):
    return {"type": "image",
            "source": {"type": "url", "url": f"http://img/{name}"}}


def _shot(*blocks):
    return {"type": "tool_result", "tool_use_id": "toolu_0",
            "content": [_text("RESULT-ONE"), *blocks]}


def _in_order(prompt, *wants):
    pos = prompt.index("ASK-FIRST")
    for want in wants:
        found = prompt.find(want, pos)
        assert found >= 0, f"{want!r} missing or out of order in {prompt!r}"
        pos = found + len(want)


@pytest.mark.parametrize("name", sorted(_VISION))
def test_a_tool_result_image_renders_once_before_the_text(route, name):
    image = _VISION[name][2]
    prompt, images = route(name, _round(_shot(_image("s")),
                                        _text("TEXT-AFTER")), vision=True)
    assert images == ["http://img/s"]
    assert prompt.count(image) == 1
    _in_order(prompt, "RESULT-ONE", image, "TEXT-AFTER")


@pytest.mark.parametrize("name", _IN_TOOL_TURN)
def test_a_tool_result_image_alone_renders_in_its_tool_turn(route, name):
    image = _VISION[name][2]
    prompt, images = route(name, _round(_shot(_image("s"))), vision=True)
    assert images == ["http://img/s"]
    assert prompt.count(image) == 1
    _in_order(prompt, "RESULT-ONE", image)


def test_the_gemma4_12b_template_shows_a_tool_image_after_the_response(route):
    prompt, _ = route("gemma4-12b", _round(_shot(_image("s"))), vision=True)
    assert 'RESULT-ONE<|"|>}<tool_response|><|image|>' in prompt


@pytest.mark.parametrize("name", ("deepseek-v4", "deepseek-v4.1",
                                  "gemma4", "mistral3"))
def test_a_tool_turn_without_images_gets_its_text(route, name):
    """A template that shows no image in a tool turn gets the tool text as
    a string: DeepSeek-V4 raises on a list there, DeepSeek-V4.1 prints an
    unsupported block, and Mistral3 prints the list."""
    prompt, _ = route(name, _round(_shot(_image("s"))), vision=True)
    tail = prompt.split("ASK-FIRST")[-1]
    assert "'type'" not in tail and '"type"' not in tail
    assert "Unsupported" not in tail
    _in_order(prompt, "RESULT-ONE")


# Template: the turn a text-only template renders for a user message with
# no text.
_EMPTY_USER = {"qwen3": "<|im_start|>user\n<|im_end|>",
               "gpt-oss": "<|start|>user<|message|><|end|>"}


@pytest.mark.parametrize("after", (True, False), ids=("text", "no-text"))
@pytest.mark.parametrize("name", sorted(_EMPTY_USER))
def test_a_text_only_model_gets_a_tool_image_turn_as_its_text(route, name,
                                                              after):
    blocks = [_shot(_image("s"))] + ([_text("TEXT-AFTER")] if after else [])
    prompt, _ = route(name, _round(*blocks))
    assert _EMPTY_USER[name] not in prompt
    assert "'type'" not in prompt and '"type": "image"' not in prompt
    _in_order(prompt, "RESULT-ONE", *(["TEXT-AFTER"] if after else []))


def test_mistral3_reads_the_user_text_in_the_tool_results(route):
    """Mistral3's template rejects a user turn right after a tool turn, so
    the text and images beside the tool results fold into them."""
    prompt, _ = route("mistral3", _round(_result(0, "RESULT-ONE"),
                                         _text("TEXT-AFTER")))
    assert ("[TOOL_RESULTS]RESULT-ONE\n\nTEXT-AFTER[/TOOL_RESULTS]"
            in prompt)
    prompt, images = route("mistral3", _round(
        _shot(), _image("x"), _text("TEXT-AFTER")), vision=True)
    assert images == ["http://img/x"]
    assert ("[TOOL_RESULTS]RESULT-ONE\n\n[IMG]TEXT-AFTER[/TOOL_RESULTS]"
            in prompt)


@pytest.mark.parametrize("name", ("qwen3.5", "gemma4"))
def test_the_kill_switch_restores_the_stock_conversion(killed_route, name):
    messages = _round(_shot(_image("s")), _text("TEXT-AFTER")) + [
        {"role": "assistant", "content": [_text("ANSWER")]},
        {"role": "user", "content": [_image("x"), _text("Q2")]}]
    assert killed_route(name, messages, vision=True) == killed_route(
        name, messages, vision=True, stock=True)


class _Mistral7B(_Processor):
    """Mistral 7B v0.3: its template writes the BOS and EOS tokens."""

    def apply_chat_template(self, messages, tokenize=False,
                            add_generation_prompt=True, **kw):
        return super().apply_chat_template(
            messages, tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
            bos_token="<s>", eos_token="</s>", **kw)


def _chat_round(*, system=None):
    """A chat request: the user asks, the assistant calls ``shot``, the tool
    answers, and the user writes again right after it."""
    call = "abcdefghi"  # Mistral 7B takes nine-character call ids only.
    head = [{"role": "system", "content": system}] if system else []
    return head + [
        {"role": "user", "content": "ASK-FIRST"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": call, "type": "function",
             "function": {"name": "shot", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": call, "content": "RESULT-ONE"},
        {"role": "user", "content": "TEXT-AFTER"}]


def test_mistral3_reads_a_chat_user_turn_after_a_tool_in_the_results(route):
    prompt, _ = route("mistral3", _chat_round(), path="/v1/chat/completions")
    assert ("[TOOL_RESULTS]RESULT-ONE\n\nTEXT-AFTER[/TOOL_RESULTS]"
            in prompt)


def test_mistral3_reads_a_responses_user_item_after_a_tool_output(route):
    items = [
        {"role": "user", "content": [
            {"type": "input_text", "text": "ASK-FIRST"}]},
        {"type": "function_call", "call_id": "c1", "name": "shot",
         "arguments": "{}"},
        {"type": "function_call_output", "call_id": "c1",
         "output": "RESULT-ONE"},
        {"role": "user", "content": [
            {"type": "input_text", "text": "TEXT-AFTER"}]}]
    prompt, _ = route("mistral3", items, path="/v1/responses")
    assert ("[TOOL_RESULTS]RESULT-ONE\n\nTEXT-AFTER[/TOOL_RESULTS]"
            in prompt)


def test_a_fold_that_loses_the_system_text_is_rejected(route):
    """The Mistral 7B template writes the system text into the last message
    only when that is a user message. A fold that ends on the tool message
    would drop it, so the template's error stands."""
    prompt, _ = route("mistral-7b", _chat_round(),
                      path="/v1/chat/completions")
    assert '{"content": RESULT-ONE\n\nTEXT-AFTER, ' in prompt
    body = route("mistral-7b", _chat_round(system="SYSTEM-TEXT"),
                 path="/v1/chat/completions", status=500)
    assert "must alternate" in body


class _UndefinedAfterTool(_Processor):
    """Fails a user message right after a tool message with a TemplateError
    subclass, as a template bug does, rather than the template's own raise."""

    def apply_chat_template(self, messages, **kw):
        roles = [m["role"] for m in messages]
        if any(a == "tool" and b == "user" for a, b in zip(roles, roles[1:])):
            raise UndefinedError("UNDEFINED-AFTER-TOOL")
        return super().apply_chat_template(messages, **kw)


def test_a_template_bug_after_a_tool_is_not_folded(route):
    """Only the template's own raise_exception folds the user text into the
    tool results; any other template error stands."""
    body = route("mistral3", _chat_round(), path="/v1/chat/completions",
                 status=500, processor=_UndefinedAfterTool(*_TEMPLATES[
                     "mistral3"][:2]))
    assert "UNDEFINED-AFTER-TOOL" in body


def test_mistral3_still_rejects_two_user_turns(route):
    body = route("mistral3", [{"role": "user", "content": [_text("ONE")]},
                              {"role": "user", "content": [_text("TWO")]}],
                 status=500)
    assert "must alternate" in body


@pytest.mark.parametrize("name", ("gemma4", "gemma4-12b", "qwen3.5"))
def test_a_tool_image_and_a_later_image_keep_their_order(route, name):
    image = _VISION[name][2]
    messages = _round(_shot(_image("s")), _text("TEXT-AFTER")) + [
        {"role": "assistant", "content": [_text("ANSWER")]},
        {"role": "user", "content": [_text("NEXT-QUESTION"), _image("x")]}]
    prompt, images = route(name, messages, vision=True)
    assert images == ["http://img/s", "http://img/x"]
    assert prompt.count(image) == 2
    _in_order(prompt, "RESULT-ONE", image, "TEXT-AFTER", "ANSWER",
              "NEXT-QUESTION", image)


def _slots(prompt, image, anchors):
    """The anchor text each image slot of ``prompt`` follows."""
    out, pos = [], prompt.find(image)
    while pos >= 0:
        before = prompt[:pos]
        out.append(max(anchors, key=before.rfind))
        pos = prompt.find(image, pos + len(image))
    return out


# The templates that show no image in a tool turn: Gemma-4 through the
# processor's own rebuild, and two that take the tool text as a string.
_AFTER_TOOL_TURN = ("gemma4", "mistral3", "deepseek-v4")


@pytest.mark.parametrize("name", _AFTER_TOOL_TURN)
def test_each_image_fills_the_slot_of_its_turn(route, name):
    """A tool image the next message does not follow up, then an image in a
    later turn: each slot gets its own image."""
    messages = _round(_shot(_image("s"))) + [
        {"role": "assistant", "content": [_text("ANSWER")]},
        {"role": "user", "content": [_image("x"), _text("Q2")]},
        {"role": "assistant", "content": [_text("ANS2")]},
        {"role": "user", "content": [_text("Q3")]}]
    prompt, images = route(name, messages, vision=True)
    assert images == ["http://img/s", "http://img/x"]
    assert _slots(prompt, _VISION[name][2], (
        "ASK-FIRST", "RESULT-ONE", "ANSWER", "Q2", "ANS2", "Q3")) == [
        "RESULT-ONE", "ANSWER"]


@pytest.mark.parametrize("name", ("qwen3.5", "kimi-k2.5", "gemma4",
                                  "deepseek-v4"))
def test_an_image_beside_a_tool_result_and_no_text_renders(route, name):
    messages = [
        {"role": "user", "content": [_text("ASK-FIRST"), _image("a")]},
        {"role": "assistant", "content": [
            _text("Taking it."),
            {"type": "tool_use", "id": "toolu_0", "name": "shot",
             "input": {}}]},
        {"role": "user", "content": [_shot(_image("s")), _image("x")]}]
    prompt, images = route(name, messages, vision=True)
    assert images == ["http://img/a", "http://img/s", "http://img/x"]
    assert _slots(prompt, _VISION[name][2], ("ASK-FIRST", "RESULT-ONE")) == [
        "ASK-FIRST", "RESULT-ONE", "RESULT-ONE"]


@pytest.mark.parametrize("name", ("qwen3.5", "gemma4", "deepseek-v4"))
def test_an_image_ahead_of_a_tool_result_follows_its_image(route, name):
    """The tool result renders first, so its image comes first in the list
    though the message sends the other image ahead of it."""
    prompt, images = route(name, _round(
        _image("x"), _shot(_image("s")), _text("TEXT-AFTER")), vision=True)
    assert images == ["http://img/s", "http://img/x"]
    image = _VISION[name][2]
    _in_order(prompt, "RESULT-ONE", image, image, "TEXT-AFTER")


@pytest.mark.parametrize("name", _AFTER_TOOL_TURN)
def test_a_tool_image_stays_put_in_the_next_turn(route, name):
    image = _VISION[name][2]
    first = _round(_shot(_image("s")))
    prompt1, images1 = route(name, first, vision=True)
    prompt2, images2 = route(name, first + [
        {"role": "assistant", "content": [_text("ANSWER")]},
        {"role": "user", "content": [_text("NEXT-QUESTION")]}], vision=True)
    assert images1 == images2 == ["http://img/s"]
    assert prompt1.count(image) == prompt2.count(image) == 1
    _in_order(prompt1, "RESULT-ONE", image)
    assert prompt2.startswith(prompt1[:prompt1.index(image) + len(image)])


@pytest.mark.parametrize("name", ("gemma4", "gemma4-12b", "qwen3.5"))
def test_the_next_turn_starts_with_a_tool_image_turn(route, name):
    first = _round(_shot(_image("s")), _text("TEXT-AFTER"))
    prompt1, _ = route(name, first, vision=True)
    prompt2, _ = route(name, first + [
        {"role": "assistant", "content": [_text("ANSWER")]},
        {"role": "user", "content": [_text("NEXT-QUESTION")]}], vision=True)
    head = prompt1[:prompt1.index("TEXT-AFTER") + len("TEXT-AFTER")]
    assert prompt2.startswith(head)


def test_the_image_salt_keys_a_tool_image_where_it_renders(route):
    """The next turn keys the tool image at the start and with the hash
    this turn keyed it."""
    import numpy as np
    from gmlx._vendor.mlx_vlm.models.qwen3_vl.processing_qwen3_vl import (
        Qwen3VLImageProcessor,
    )
    from PIL import Image
    from types import SimpleNamespace

    from gmlx.cache.media_salt import MediaSalt, image_spans

    ip = Qwen3VLImageProcessor(patch_size=14, merge_size=1, min_pixels=0,
                               max_pixels=14 * 14 * 64)
    raw = ip([Image.new("RGB", (56, 28), (200, 0, 0))])
    cfg = SimpleNamespace(image_token_id=9)
    proc = SimpleNamespace(image_processor=ip)

    def ids(prompt):
        # One text id per character; each pad expands to the image's run.
        row, parts = [], prompt.split(_QWEN)
        assert len(parts) == 2, prompt
        row = [1] * len(parts[0]) + [9] * int(np.prod(raw["image_grid_thw"][0]))
        return np.array([row + [1] * len(parts[1])])

    first = _round(_shot(_image("s")), _text("TEXT-AFTER"))
    prompt1, _ = route("qwen3.5", first, vision=True)
    prompt2, _ = route("qwen3.5", first + [
        {"role": "assistant", "content": [_text("ANSWER")]},
        {"role": "user", "content": [_text("NEXT-QUESTION")]}], vision=True)
    spans1 = image_spans(ids(prompt1), raw, cfg, proc)
    spans2 = image_spans(ids(prompt2), raw, cfg, proc)
    assert spans1 is not None and spans2 == spans1
    end = prompt1.index("TEXT-AFTER")
    assert MediaSalt(5, spans2).at(end) == MediaSalt(5, spans1).at(end)
