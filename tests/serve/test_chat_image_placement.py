"""/v1/chat/completions renders each image in the message that sent it,
as /v1/responses does, so a turn's prompt starts with the previous turn's
prompt. Stock chat completions reduces each message to its text and the
template places every image in the last user message.

Within a message, the text and images render in the order the client sent
them. The stock rebuild joins a message's text into one part and puts
every image before it."""

from __future__ import annotations

import importlib
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("gmlx._vendor.mlx_vlm")

from gmlx._vendor.mlx_vlm.models.gemma4.processing_gemma4 import Gemma4Processor  # noqa: E402
from transformers.utils.chat_template_utils import render_jinja_template  # noqa: E402

from gmlx.cache.media_salt import MediaSalt, image_spans  # noqa: E402
from gmlx.serve.patches import _common as sp_common  # noqa: E402
from gmlx.serve.patches import render  # noqa: E402
from gmlx.serve.patches.chat_behavior import install_role_normalization  # noqa: E402
from gmlx.serve.patches.messages_route import install_messages_route  # noqa: E402
from gmlx.serve.patches.render import (  # noqa: E402
    install_chat_image_placement,
    install_faithful_history,
)

_OPENAI = importlib.import_module("gmlx._vendor.mlx_vlm.server.openai")
_ANTHROPIC = importlib.import_module("gmlx._vendor.mlx_vlm.server.anthropic")
_PROMPT_UTILS = importlib.import_module("gmlx._vendor.mlx_vlm.prompt_utils")
_APP = importlib.import_module("gmlx._vendor.mlx_vlm.server.app")
_RUNTIME = importlib.import_module("gmlx._vendor.mlx_vlm.server.runtime").runtime
_FIXTURES = Path(__file__).parents[1] / "fixtures"

# model_type: template fixture, the placeholder one image renders as, and
# the text that closes a user message.
_MODELS = {
    "gemma4": ("gemma4_template.jinja", "<|image|>", "<turn|>\n"),
    "qwen3_5": ("qwen35_template.jinja",
                "<|vision_start|><|image_pad|><|vision_end|>",
                "<|im_end|>\n"),
}


class _Processor:
    """Renders the fixture template the way a tokenizer does."""

    def __init__(self, model_type):
        self.chat_template = (_FIXTURES / _MODELS[model_type][0]).read_text()

    def apply_chat_template(self, messages, tokenize=False,
                            add_generation_prompt=True, **kw):
        out = render_jinja_template(
            [messages], chat_template=self.chat_template,
            add_generation_prompt=add_generation_prompt, **kw)
        rendered = out[0] if isinstance(out, tuple) else out
        return rendered[0] if isinstance(rendered, list) else rendered


class _Gemma4Processor:
    """Renders as mlx-vlm's Gemma4Processor does: its apply_chat_template
    runs the stock rebuild on the messages once more and renders the result
    with the tokenizer."""

    apply_chat_template = Gemma4Processor.apply_chat_template
    _extract_media_from_messages = Gemma4Processor._extract_media_from_messages
    _as_list = staticmethod(Gemma4Processor._as_list)

    def __init__(self, model_type):
        self.model_type = model_type
        self.tokenizer = _Processor("gemma4")
        self.chat_template = self.tokenizer.chat_template


class _Tok:
    text, finish_reason, token_count = "ok", "stop", 1


class _RG:
    """Records the prompt and images the route hands to generation."""

    def __init__(self):
        self.calls = []

    def generate(self, prompt, images=None, audio=None, args=None,
                 videos=None):
        self.calls.append((prompt, list(images or ())))
        return types.SimpleNamespace(prompt_tokens=1, uid=1), iter([_Tok()])


@pytest.fixture
def route(monkeypatch):
    """The chat route with the gmlx render wraps, on a stub model."""
    routes = sp_common._snapshot_routes(_APP.app)
    for target in sp_common._render_target_modules():
        monkeypatch.setattr(target, "apply_chat_template",
                            target.apply_chat_template)
    monkeypatch.setattr(_OPENAI, "extract_text_from_content",
                        _OPENAI.extract_text_from_content)
    monkeypatch.setattr(_OPENAI, "_normalize_instruction_messages",
                        _OPENAI._normalize_instruction_messages)
    for name in ("get_chat_template", "apply_chat_template"):
        monkeypatch.setattr(_PROMPT_UTILS, name, getattr(_PROMPT_UTILS, name))
    monkeypatch.setattr(_ANTHROPIC, "_anthropic_messages_to_internal",
                        _ANTHROPIC._anthropic_messages_to_internal)
    rg = _RG()
    monkeypatch.setattr(_RUNTIME, "response_generator", rg, raising=False)
    install_faithful_history()
    install_role_normalization()
    install_chat_image_placement()
    install_messages_route()
    from fastapi.testclient import TestClient

    client = TestClient(_APP.app)

    def render(model_type, messages, responses=False, processor=None,
               anthropic=False, tools=None):
        processor = processor or _Processor(model_type)
        cached = lambda *a, **k: (  # noqa: E731
            None, processor, {"model_type": model_type})
        monkeypatch.setattr(_OPENAI, "get_cached_model", cached)
        monkeypatch.setattr(_ANTHROPIC, "get_cached_model", cached)
        if anthropic:
            body = {"model": "m", "max_tokens": 1,
                    "messages": [m for m in messages if m["role"] != "system"]}
            system = [m["content"] for m in messages if m["role"] == "system"]
            if system:
                body["system"] = system[0]
            if tools:
                body["tools"] = tools
            r = client.post("/v1/messages", json=body)
        elif responses:
            r = client.post("/v1/responses", json={
                "model": "m", "input": messages, "max_output_tokens": 16})
        else:
            r = client.post("/v1/chat/completions", json={
                "model": "m", "messages": messages, "max_tokens": 1})
        assert r.status_code == 200, r.text
        return rg.calls.pop()

    yield render
    sp_common._restore_routes(_APP.app, routes)


def _image(name):
    return {"type": "image_url", "image_url": {"url": f"http://img/{name}"}}


def _text(t):
    return {"type": "text", "text": t}


def _turns():
    turn1 = [{"role": "system", "content": "Be brief."},
             {"role": "user", "content": [
                 _text("Here is the first picture."), _image("a"),
                 _text("Describe it.")]}]
    turn2 = turn1 + [
        {"role": "assistant", "content": "A red car."},
        {"role": "user", "content": [
            _text("And a second one."), _image("b"),
            _text("How do they differ?")]}]
    return turn1, turn2


@pytest.mark.parametrize("model_type", sorted(_MODELS))
def test_each_image_renders_in_its_own_message(route, model_type):
    image = _MODELS[model_type][1]
    turn1, turn2 = _turns()
    prompt1, images1 = route(model_type, turn1)
    prompt2, images2 = route(model_type, turn2)
    assert images1 == ["http://img/a"]
    assert images2 == ["http://img/a", "http://img/b"]
    # Image A sits before the first question, image B before the second.
    first, second = prompt2.split("A red car.")
    assert first.count(image) == 1 and second.count(image) == 1
    assert first.index(image) < first.index("Describe it.")
    assert second.index(image) < second.index("How do they differ?")
    # Turn 2 renders turn 1 as it was, so its prompt starts with turn 1's
    # through the first question. Qwen3.5 closes an empty think block in
    # the generation prompt and not in history, so the check stops there.
    end = _MODELS[model_type][2]
    assert prompt2.startswith(prompt1[:prompt1.rindex(end) + len(end)])
    if model_type == "gemma4":
        assert prompt2.startswith(prompt1)


@pytest.mark.parametrize("model_type", sorted(_MODELS))
def test_images_in_one_message_keep_their_order(route, model_type):
    image = _MODELS[model_type][1]
    prompt, images = route(model_type, [{"role": "user", "content": [
        _image("a"), _text("then"), _image("b"), _text("compare them.")]}])
    assert images == ["http://img/a", "http://img/b"]
    assert prompt.count(image) == 2


@pytest.mark.parametrize("model_type", sorted(_MODELS))
def test_a_text_chat_renders_as_before(route, monkeypatch, model_type):
    """Text parts and plain strings render byte for byte as the stock
    route renders them."""
    messages = [{"role": "system", "content": "Be brief."},
                {"role": "user", "content": [_text("Hi"), _text("there.")]},
                {"role": "assistant", "content": "Hello."},
                {"role": "user", "content": "What is 2 + 2?"}]
    patched = route(model_type, messages)
    extract = _OPENAI.extract_text_from_content
    monkeypatch.setattr(_OPENAI, "extract_text_from_content",
                        getattr(extract, "__wrapped__", extract))
    stock = route(model_type, messages)
    assert patched == stock and patched[1] == []


def _assert_in_order(prompt, marker, *contents):
    """Each text part and image of ``contents`` appears in ``prompt`` after
    the one before it, one marker per image."""
    items = [item for content in contents for item in content]
    assert prompt.count(marker) == sum(
        item["type"] == "image_url" for item in items)
    pos = 0
    for item in items:
        want = marker if item["type"] == "image_url" else item["text"]
        found = prompt.find(want, pos)
        assert found >= 0, f"{want!r} out of order in {prompt!r}"
        pos = found + len(want)


_LAYOUTS = {
    "text-image": [_text("The picture below is a photo."), _image("a")],
    "image-text": [_image("a"), _text("What is in this picture?")],
    "image-text-image": [_image("a"), _text("and then a drawing:"),
                         _image("b")],
    "text-image-text-image-text": [
        _text("The first picture is a photo."), _image("a"),
        _text("The second picture is a drawing."), _image("b"),
        _text("Which picture is the drawing?")],
}


@pytest.mark.parametrize("layout", sorted(_LAYOUTS))
@pytest.mark.parametrize("model_type", sorted(_MODELS))
def test_text_and_images_keep_the_client_order(route, model_type, layout):
    content = _LAYOUTS[layout]
    prompt, images = route(model_type, [{"role": "user", "content": content}])
    assert images == [item["image_url"]["url"] for item in content
                      if item["type"] == "image_url"]
    _assert_in_order(prompt, _MODELS[model_type][1], content)


@pytest.mark.parametrize("model_type", sorted(_MODELS))
def test_a_responses_message_keeps_the_client_order(route, model_type):
    content = _LAYOUTS["text-image-text-image-text"]
    sent = [{"type": "input_image", "image_url": item["image_url"]["url"]}
            if item["type"] == "image_url"
            else {"type": "input_text", "text": item["text"]}
            for item in content]
    prompt, images = route(model_type, [{"role": "user", "content": sent}],
                           responses=True)
    assert images == ["http://img/a", "http://img/b"]
    _assert_in_order(prompt, _MODELS[model_type][1], content)


@pytest.mark.parametrize("model_type", sorted(_MODELS))
def test_each_turn_keeps_its_text_and_image_order(route, model_type):
    turn1, turn2 = _turns()
    prompt1, _ = route(model_type, turn1)
    prompt2, images2 = route(model_type, turn2)
    assert images2 == ["http://img/a", "http://img/b"]
    _assert_in_order(prompt2, _MODELS[model_type][1], turn2[1]["content"],
                     turn2[3]["content"])
    end = _MODELS[model_type][2]
    assert prompt2.startswith(prompt1[:prompt1.rindex(end) + len(end)])


@pytest.mark.parametrize("model_type", ("gemma4", "gemma4_unified"))
def test_the_gemma4_processor_keeps_the_client_order(route, model_type):
    """Gemma4Processor.apply_chat_template, which renders the prompt, runs
    the stock rebuild once more before the tokenizer renders."""
    content = _LAYOUTS["text-image-text-image-text"]
    prompt, _ = route(model_type, [{"role": "user", "content": content}],
                      processor=_Gemma4Processor(model_type))
    _assert_in_order(prompt, "<|image|>", content)
    _, turn2 = _turns()
    prompt2, _ = route(model_type, turn2,
                       processor=_Gemma4Processor(model_type))
    _assert_in_order(prompt2, "<|image|>", turn2[1]["content"],
                     turn2[3]["content"])


# The model types whose chat template renders an image item where the
# message holds it and whose processor expands the image markers in prompt
# order.
_IN_PLACE = ("deepseek_v41_vl", "deepseek_v4_vl", "gemma4", "gemma4_unified",
             "glm5_next", "kimi_k25", "mistral3", "muse_glimmer", "muse_spark",
             "nemotron_h_omni", "pixtral", "qwen3_5", "qwen3_5_moe",
             "qwen3_omni_moe", "qwen3_vl", "qwen3_vl_moe", "qwen4_exp")
# The rows mlx-vlm's table lacks, which the loader adds on load.
_LOADER_ROWS = ("deepseek_v41_vl", "deepseek_v4_vl")


def test_the_in_place_types_are_the_checked_ones():
    assert set(getattr(render, "_IN_PLACE_IMAGES", ())) == set(_IN_PLACE)


def _rebuilt(monkeypatch, model_type, content, num_images):
    if model_type in _LOADER_ROWS:
        monkeypatch.setitem(_PROMPT_UTILS.MODEL_CONFIG, model_type,
                            _PROMPT_UTILS.MessageFormat.LIST_WITH_IMAGE_FIRST)
    for target in sp_common._render_target_modules():
        monkeypatch.setattr(target, "apply_chat_template",
                            _PROMPT_UTILS.apply_chat_template)
    for name in ("get_chat_template", "apply_chat_template"):
        monkeypatch.setattr(_PROMPT_UTILS, name, getattr(_PROMPT_UTILS, name))
    install_faithful_history()
    msgs = _OPENAI.apply_chat_template(
        None, {"model_type": model_type},
        [{"role": "user", "content": content}],
        num_images=num_images, return_messages=True)
    stock = _PROMPT_UTILS.apply_chat_template(
        None, {"model_type": model_type},
        [{"role": "user", "content": content}],
        num_images=num_images, return_messages=True)
    return msgs[0]["content"], stock[0]["content"]


def _kinds(content):
    return [(p["type"], p.get("text")) for p in content]


@pytest.mark.parametrize("model_type", _IN_PLACE)
def test_the_rebuilt_message_holds_the_client_order(monkeypatch, model_type):
    content = [_text("A"), {"type": "image"}, _text(" B"), _text("C "),
               {"type": "image"}, _text("D ")]
    got, _ = _rebuilt(monkeypatch, model_type, content, 2)
    # Text between two images joins as the stock extraction joins it, and
    # the message's ends are stripped as before.
    assert _kinds(got) == [("text", "A"), ("image", None), ("text", " B C "),
                           ("image", None), ("text", "D")]


@pytest.mark.parametrize("model_type", ("gemma4", "kimi_k25", "qwen3_5"))
def test_a_message_in_the_stock_order_rebuilds_as_before(monkeypatch,
                                                         model_type):
    image = {"type": "image"}
    # Images before the text (after it on Kimi), and images alone.
    ordered = ([_text(" A "), _text("B"), image, image]
               if model_type == "kimi_k25"
               else [image, image, _text(" A "), _text("B")])
    for content, n in ((ordered, 2), ([image], 1), ([image, image], 2)):
        got, stock = _rebuilt(monkeypatch, model_type, content, n)
        assert got == stock
    # An image the route counted without a marker in the message (a legacy
    # side channel image) leaves the placement to the stock rebuild.
    got, stock = _rebuilt(monkeypatch, model_type,
                          [_text("A"), image, _text("B")], 2)
    assert got == stock


def test_a_template_that_places_images_itself_stays_stock(monkeypatch):
    """The llava-hf template renders every image of a message ahead of its
    text whatever the content order."""
    content = [_text("A"), {"type": "image"}, _text("B")]
    got, stock = _rebuilt(monkeypatch, "llava", content, 1)
    assert got == stock


def test_the_image_salt_follows_the_images_in_place(route):
    """The prefix cache keys a prefix on the images that start inside it.
    With image B after the text that follows image A, a request that adds
    B keys its prefix through that text as the request that sent A alone,
    and each image run hashes its own image."""
    from gmlx._vendor.mlx_vlm.models.qwen3_vl.processing_qwen3_vl import (
        Qwen3VLImageProcessor,
    )
    from PIL import Image

    ip = Qwen3VLImageProcessor(patch_size=14, merge_size=1, min_pixels=0,
                               max_pixels=14 * 14 * 64)
    pics = [Image.new("RGB", (56, 28), (200, 0, 0)),
            Image.new("RGB", (28, 42), (0, 200, 0))]
    cfg = SimpleNamespace(image_token_id=9)
    proc = SimpleNamespace(image_processor=ip)
    pad = "<|image_pad|>"

    def ids(prompt, grids):
        # One text id per character; each pad expands to its image's run.
        row, parts = [], prompt.split(pad)
        for i, part in enumerate(parts):
            row += [1] * len(part)
            if i < len(grids):
                row += [9] * int(np.prod(grids[i]))
        return np.array([row])

    first = [_text("The first picture is a photo."), _image("a"),
             _text("The second picture is a drawing.")]
    both = first + [_image("b"), _text("Which one is the drawing?")]
    salts = []
    for content, n in ((first, 1), (both, 2)):
        prompt, _ = route("qwen3_5", [{"role": "user", "content": content}])
        raw = ip(pics[:n])
        spans = image_spans(ids(prompt, raw["image_grid_thw"]), raw, cfg,
                            proc)
        assert spans is not None and len(spans) == n
        salts.append((prompt, spans))
    (p1, spans1), (p2, spans2) = salts
    # Image A keys at the same start with the same hash in both requests,
    # and image B starts after the text that follows A.
    assert spans2[0] == spans1[0]
    start_b = spans2[1][0]
    text_b = p2.index("The second picture is a drawing.")
    assert p2.index(pad) < text_b < p2.index(pad, p2.index(pad) + 1)
    assert start_b > spans2[0][0] + len("The second picture is a drawing.")
    # Each span hashes its own image: B keys as it does alone.
    alone = ids(p1, ip(pics[1:])["image_grid_thw"])
    assert spans2[1][1] == image_spans(alone, ip(pics[1:]), cfg, proc)[0][1]
    # Through the text before B, both requests key alike.
    s1, s2 = MediaSalt(5, spans1), MediaSalt(5, spans2)
    assert s2.at(start_b) == s1.at(start_b)
    assert s2.at(start_b + 1) != s1.at(start_b + 1)


# /v1/messages: the stock conversion joins a user message's text into one
# string and passes its images in the request's image list, so the render
# places every image in the last user message, ahead of its text.


def _anthropic(messages):
    """``messages`` with each image part as an Anthropic image block."""
    def block(item):
        if item["type"] != "image_url":
            return item
        return {"type": "image",
                "source": {"type": "url", "url": item["image_url"]["url"]}}
    return [{**m, "content": [block(i) for i in m["content"]]}
            if isinstance(m["content"], list) else m for m in messages]


@pytest.mark.parametrize("layout", sorted(_LAYOUTS))
@pytest.mark.parametrize("model_type", sorted(_MODELS))
def test_a_messages_request_keeps_the_client_order(route, model_type, layout):
    content = _LAYOUTS[layout]
    prompt, images = route(
        model_type, _anthropic([{"role": "user", "content": content}]),
        anthropic=True)
    assert images == [item["image_url"]["url"] for item in content
                      if item["type"] == "image_url"]
    _assert_in_order(prompt, _MODELS[model_type][1], content)


@pytest.mark.parametrize("model_type", sorted(_MODELS))
def test_each_messages_image_renders_in_its_own_turn(route, model_type):
    image = _MODELS[model_type][1]
    turn1, turn2 = (_anthropic(t) for t in _turns())
    prompt1, images1 = route(model_type, turn1, anthropic=True)
    prompt2, images2 = route(model_type, turn2, anthropic=True)
    assert images1 == ["http://img/a"]
    assert images2 == ["http://img/a", "http://img/b"]
    first, second = prompt2.split("A red car.")
    assert first.count(image) == 1 and second.count(image) == 1
    _assert_in_order(prompt2, image, _turns()[1][1]["content"],
                     _turns()[1][3]["content"])
    end = _MODELS[model_type][2]
    assert prompt2.startswith(prompt1[:prompt1.rindex(end) + len(end)])
    if model_type == "gemma4":
        assert prompt2.startswith(prompt1)


@pytest.mark.parametrize("model_type", ("gemma4", "gemma4_unified"))
def test_the_gemma4_processor_keeps_a_messages_order(route, model_type):
    content = _LAYOUTS["text-image-text-image-text"]
    prompt, _ = route(model_type,
                      _anthropic([{"role": "user", "content": content}]),
                      processor=_Gemma4Processor(model_type), anthropic=True)
    _assert_in_order(prompt, "<|image|>", content)


def test_the_image_salt_follows_the_messages_turns(route):
    """Turn 2 keys image A at the start and with the hash turn 1 keyed it,
    so its prefix through turn 1's question matches turn 1's."""
    from gmlx._vendor.mlx_vlm.models.qwen3_vl.processing_qwen3_vl import (
        Qwen3VLImageProcessor,
    )
    from PIL import Image

    ip = Qwen3VLImageProcessor(patch_size=14, merge_size=1, min_pixels=0,
                               max_pixels=14 * 14 * 64)
    pics = [Image.new("RGB", (56, 28), (200, 0, 0)),
            Image.new("RGB", (28, 42), (0, 200, 0))]
    cfg = SimpleNamespace(image_token_id=9)
    proc = SimpleNamespace(image_processor=ip)
    pad = "<|image_pad|>"

    def ids(prompt, grids):
        row, parts = [], prompt.split(pad)
        for i, part in enumerate(parts):
            row += [1] * len(part)
            if i < len(grids):
                row += [9] * int(np.prod(grids[i]))
        return np.array([row])

    salts = []
    for turn, n in zip(_turns(), (1, 2)):
        prompt, _ = route("qwen3_5", _anthropic(turn), anthropic=True)
        raw = ip(pics[:n])
        spans = image_spans(ids(prompt, raw["image_grid_thw"]), raw, cfg,
                            proc)
        assert spans is not None and len(spans) == n
        salts.append(spans)
    spans1, spans2 = salts
    assert spans2[0] == spans1[0]
    s1, s2 = MediaSalt(5, spans1), MediaSalt(5, spans2)
    assert s2.at(spans2[1][0]) == s1.at(spans2[1][0])


_TOOL = [{"name": "shot", "description": "Take a screenshot.",
          "input_schema": {"type": "object", "properties": {}}}]


def _tool_turns(*results):
    """A tool round: the user asks, the assistant calls ``shot`` once per
    result, and the last user message holds ``results``."""
    calls = [{"type": "tool_use", "id": f"toolu_{i}", "name": "shot",
              "input": {}} for i in range(len(results))]
    return [{"role": "user", "content": [_text("Take a screenshot.")]},
            {"role": "assistant", "content": [_text("Taking it.")] + calls},
            {"role": "user", "content": list(results)}]


def _result(i, *content):
    return {"type": "tool_result", "tool_use_id": f"toolu_{i}",
            "content": list(content)}


_SHOT = {"type": "image", "source": {"type": "url", "url": "http://img/s"}}
_TOOL_ONLY = {
    "one result": _tool_turns(_result(0, _text("Saved."))),
    "two results": _tool_turns(_result(0, _text("First.")),
                               _result(1, _text("Second."))),
    "a result with an image": _tool_turns(_result(0, _text("Here."), _SHOT)),
}


@pytest.mark.parametrize("case", sorted(_TOOL_ONLY))
@pytest.mark.parametrize("model_type", sorted(_MODELS))
def test_messages_tool_rounds_render_as_before(route, monkeypatch,
                                               model_type, case):
    """A user message of tool results alone converts as the stock route
    converts it."""
    messages = _TOOL_ONLY[case]
    patched = route(model_type, messages, anthropic=True, tools=_TOOL)
    claim = _ANTHROPIC._anthropic_messages_to_internal
    monkeypatch.setattr(_ANTHROPIC, "_anthropic_messages_to_internal",
                        getattr(claim, "_gmlx_orig", claim))
    assert patched == route(model_type, messages, anthropic=True, tools=_TOOL)
