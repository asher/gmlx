"""/v1/chat/completions renders each video and each audio clip in the
message that sent it, at the place the client put it, for the model types
of ``render._IN_PLACE_VIDEOS`` and ``render._IN_PLACE_AUDIO``.

The stock rebuild puts every video of a request at the start of every
message, and every clip after the text of the last user message. Every
other model type keeps that render byte for byte.

The LFM3 renders run through the processor class gmlx builds for the
type, on a stand-in template. It is not the LFM3 chat template, which is
not committed. tests/models/test_lfm3_vision_live.py holds the shipped
template to the same placement."""

from __future__ import annotations

import base64
import importlib
import io
import types
import wave
from pathlib import Path

import pytest

pytest.importorskip("gmlx._vendor.mlx_vlm")

from gmlx._vendor.mlx_vlm.models.gemma4.processing_gemma4 import Gemma4Processor  # noqa: E402
from transformers.utils.chat_template_utils import render_jinja_template  # noqa: E402

from gmlx.serve.patches import _common as sp_common  # noqa: E402
from gmlx.serve.patches import render  # noqa: E402
from gmlx.serve.patches.chat_behavior import install_role_normalization  # noqa: E402
from gmlx.serve.patches.render import (  # noqa: E402
    install_chat_image_placement,
    install_faithful_history,
)

_OPENAI = importlib.import_module("gmlx._vendor.mlx_vlm.server.openai")
_PROMPT_UTILS = importlib.import_module("gmlx._vendor.mlx_vlm.prompt_utils")
_APP = importlib.import_module("gmlx._vendor.mlx_vlm.server.app")
_RUNTIME = importlib.import_module("gmlx._vendor.mlx_vlm.server.runtime").runtime
_FIXTURES = Path(__file__).parents[1] / "fixtures"

_STAND_IN = """{%- for message in messages -%}
<|im_start|>{{ message['role'] }}
{% if message['content'] is string -%}
{{ message['content'] }}
{%- else -%}
{%- for item in message['content'] or [] -%}
{%- if item['type'] in ('image', 'video', 'audio') -%}
<|vision_start|><|{{ item['type'] }}|><|vision_end|>
{%- elif item['type'] == 'text' -%}
{{ item['text'] }}
{%- endif -%}
{%- endfor -%}
{%- endif -%}
<|im_end|>
{% endfor -%}
{%- if add_generation_prompt -%}
<|im_start|>assistant
{% endif -%}
"""
_VIDEO = "<|vision_start|><|video|><|vision_end|>"
_AUDIO = "<|vision_start|><|audio|><|vision_end|>"
_IMAGE = "<|vision_start|><|image|><|vision_end|>"
_SPECIALS = ["<|video|>", "<|image|>", "<|vision_start|>", "<|vision_end|>",
             "<|audio|>"]
_META = {
    "clip.vision.projector_type": "lfm3vl",
    "clip.vision.patch_size": 16,
    "clip.vision.spatial_merge_size": 2,
    "clip.vision.image_min_pixels": 65536,
    "clip.vision.image_max_pixels": 2097152,
    "clip.vision.image_mean": [0.5, 0.5, 0.5],
    "clip.vision.image_std": [0.5, 0.5, 0.5],
    "clip.has_audio_encoder": True,
    "clip.audio.num_mel_bins": 128,
}


def _lfm3_processor(model_type):
    """The processor class gmlx builds for an LFM3 pair."""
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    from gmlx.load import vlm

    vocab = {"[UNK]": 0, **{t: 29 + i for i, t in enumerate(_SPECIALS)}}
    tok = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.add_special_tokens(_SPECIALS)
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="[UNK]",
                                   pad_token="[UNK]")
    fast.chat_template = _STAND_IN
    meta = dict(_META)
    if model_type == "lfm3_vl":
        del meta["clip.has_audio_encoder"]
    return vlm._synthesize_lfm3_processor(fast, meta, model_type)


class _Template:
    """Renders a fixture template the way a tokenizer does. It takes
    videos itself, so the route does not turn them into frames."""

    video_processor = object()

    def __init__(self, fixture):
        self.chat_template = (_FIXTURES / fixture).read_text()

    def process(self, text=None, images=None, videos=None):
        raise NotImplementedError

    def apply_chat_template(self, messages, tokenize=False,
                            add_generation_prompt=True, **kw):
        out = render_jinja_template(
            [messages], chat_template=self.chat_template,
            add_generation_prompt=add_generation_prompt, **kw)
        rendered = out[0] if isinstance(out, tuple) else out
        return rendered[0] if isinstance(rendered, list) else rendered


class _Gemma4:
    """Renders as mlx-vlm's Gemma4Processor does: its apply_chat_template
    runs the stock rebuild once more, with the videos again."""

    apply_chat_template = Gemma4Processor.apply_chat_template
    _extract_media_from_messages = Gemma4Processor._extract_media_from_messages
    _as_list = staticmethod(Gemma4Processor._as_list)
    video_processor = object()

    def __init__(self, model_type):
        self.model_type = model_type
        self.tokenizer = _Template("gemma4_template.jinja")
        self.chat_template = self.tokenizer.chat_template

    def process(self, text=None, images=None, videos=None):
        raise NotImplementedError


def _processor(model_type):
    if model_type in ("lfm3_omni", "lfm3_vl", "lfm3_audio"):
        return _lfm3_processor(model_type)
    if model_type == "gemma4":
        return _Gemma4(model_type)
    return _Template("qwen35_template.jinja")


class _Tok:
    text, finish_reason, token_count = "ok", "stop", 1


class _RG:
    """Records what the route hands to generation."""

    def __init__(self):
        self.calls = []

    def generate(self, prompt, images=None, audio=None, args=None,
                 videos=None):
        self.calls.append(types.SimpleNamespace(
            prompt=prompt, images=list(images or ()),
            audio=len(audio or ()), videos=list(videos or ())))
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
    rg = _RG()
    monkeypatch.setattr(_RUNTIME, "response_generator", rg, raising=False)
    install_faithful_history()
    install_role_normalization()
    install_chat_image_placement()
    from fastapi.testclient import TestClient

    client = TestClient(_APP.app)
    processors: dict = {}

    def chat(model_type, messages):
        processor = processors.setdefault(model_type, _processor(model_type))
        monkeypatch.setattr(
            _OPENAI, "get_cached_model",
            lambda *a, **k: (None, processor, {"model_type": model_type}))
        r = client.post("/v1/chat/completions", json={
            "model": "m", "messages": messages, "max_tokens": 1})
        assert r.status_code == 200, r.text
        return rg.calls.pop()

    yield chat
    sp_common._restore_routes(_APP.app, routes)


@pytest.fixture
def stock(route, monkeypatch):
    """The same route with the placement marks off: what a request with
    no image rendered before the marks existed."""
    def chat(model_type, messages):
        extract = _OPENAI.extract_text_from_content
        with monkeypatch.context() as patch:
            patch.setattr(_OPENAI, "extract_text_from_content",
                          getattr(extract, "__wrapped__", extract))
            return route(model_type, messages)
    return chat


def _video(name):
    return {"type": "video_url", "video_url": {"url": f"http://vid/{name}"}}


def _image(name):
    return {"type": "image_url", "image_url": {"url": f"http://img/{name}"}}


def _clip():
    """A 10 ms silent WAV as an ``input_audio`` part."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x00" * 160)
    return {"type": "input_audio", "input_audio": {
        "data": base64.b64encode(buffer.getvalue()).decode(),
        "format": "wav"}}


def _text(t):
    return {"type": "text", "text": t}


def _order(prompt, *wanted):
    """Assert the pieces appear once each as often as listed, in order."""
    pos = 0
    for piece in wanted:
        found = prompt.find(piece, pos)
        assert found >= 0, f"{piece!r} out of order in {prompt!r}"
        pos = found + len(piece)
    for marker in (_VIDEO, _AUDIO, _IMAGE):
        assert prompt.count(marker) == wanted.count(marker), prompt


_LFM3 = ("lfm3_omni", "lfm3_vl")
_HEARS = ("lfm3_omni", "lfm3_audio")


@pytest.mark.parametrize("model_type", _LFM3)
def test_a_video_stays_out_of_the_system_message(route, model_type):
    call = route(model_type, [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": [_video("a"), _text("What happens?")]}])
    _order(call.prompt, "Be brief.", "<|im_start|>user", _VIDEO,
           "What happens?")
    assert call.videos == ["http://vid/a"]


@pytest.mark.parametrize("model_type", _LFM3)
def test_a_second_turn_starts_with_the_first_turns_prompt(route, model_type):
    turn1 = [{"role": "system", "content": "Be brief."},
             {"role": "user", "content": [_video("a"),
                                          _text("What happens?")]}]
    turn2 = turn1 + [{"role": "assistant", "content": "A bar moves."},
                     {"role": "user", "content": "Which way?"}]
    first, second = route(model_type, turn1), route(model_type, turn2)
    _order(second.prompt, _VIDEO, "What happens?", "A bar moves.",
           "Which way?")
    assert second.prompt.startswith(first.prompt)
    assert second.videos == ["http://vid/a"]


@pytest.mark.parametrize("model_type", _LFM3)
def test_each_video_renders_in_its_own_turn(route, model_type):
    call = route(model_type, [
        {"role": "user", "content": [_video("a"), _text("First clip.")]},
        {"role": "assistant", "content": "Seen."},
        {"role": "user", "content": [_text("Second clip:"), _video("b")]}])
    _order(call.prompt, _VIDEO, "First clip.", "Seen.", "Second clip:",
           _VIDEO)
    assert call.videos == ["http://vid/a", "http://vid/b"]


@pytest.mark.parametrize("model_type", _LFM3)
def test_a_video_keeps_its_place_among_text_and_images(route, model_type):
    call = route(model_type, [{"role": "user", "content": [
        _text("AAA"), _video("a"), _text("BBB"), _image("x"), _text("CCC"),
        _video("b"), _image("y")]}])
    _order(call.prompt, "AAA", _VIDEO, "BBB", _IMAGE, "CCC", _VIDEO, _IMAGE)
    assert call.videos == ["http://vid/a", "http://vid/b"]
    assert call.images == ["http://img/x", "http://img/y"]


@pytest.mark.parametrize("model_type", _LFM3)
def test_a_video_first_message_renders_as_before(route, stock, model_type):
    """One message with its video first is the one layout the stock
    rebuild already rendered in the client's order."""
    messages = [{"role": "user", "content": [
        _video("a"), _text("What"), _text("happens?")]}]
    assert route(model_type, messages).prompt == \
        stock(model_type, messages).prompt


@pytest.mark.parametrize("model_type", _HEARS)
def test_a_clip_renders_where_its_message_holds_it(route, model_type):
    turn1 = [{"role": "system", "content": "Be brief."},
             {"role": "user", "content": [_clip(), _text("What is said?")]}]
    turn2 = turn1 + [
        {"role": "assistant", "content": "A forecast."},
        {"role": "user", "content": [_text("And in this one?"), _clip()]}]
    first, second = route(model_type, turn1), route(model_type, turn2)
    _order(first.prompt, "Be brief.", _AUDIO, "What is said?")
    _order(second.prompt, _AUDIO, "What is said?", "A forecast.",
           "And in this one?", _AUDIO)
    assert second.prompt.startswith(first.prompt)
    assert (first.audio, second.audio) == (1, 2)


@pytest.mark.parametrize("model_type", _HEARS)
def test_a_clip_alone_is_the_whole_user_turn(route, model_type):
    """The transcription form: a system prompt and a turn that holds
    only the clip."""
    call = route(model_type, [
        {"role": "system", "content": "Perform ASR in English."},
        {"role": "user", "content": [_clip()]}])
    assert f"<|im_start|>user\n{_AUDIO}<|im_end|>" in call.prompt


def test_clips_videos_and_images_share_one_message(route):
    call = route("lfm3_omni", [{"role": "user", "content": [
        _image("x"), _text("AAA"), _clip(), _text("BBB"), _video("a"),
        _clip(), _text("CCC")]}])
    _order(call.prompt, _IMAGE, "AAA", _AUDIO, "BBB", _VIDEO, _AUDIO, "CCC")
    assert (call.audio, call.videos) == (2, ["http://vid/a"])


def _conversations():
    return {
        "video-first": [{"role": "user", "content": [
            _video("a"), _text("What happens?")]}],
        "text-first": [{"role": "user", "content": [
            _text("Watch this."), _video("a"), _text("What happens?")]}],
        "two-turns": [
            {"role": "user", "content": [_video("a"), _text("First.")]},
            {"role": "assistant", "content": "Seen."},
            {"role": "user", "content": "And then?"}],
        "clip": [
            {"role": "user", "content": [_clip(), _text("What is said?")]},
            {"role": "assistant", "content": "A forecast."},
            {"role": "user", "content": [_text("Hear"), _text("this:"),
                                         _clip()]}],
    }


@pytest.mark.parametrize("model_type, name", [
    (model_type, name) for model_type in ("gemma4", "qwen3_5")
    for name in sorted(_conversations())
    # The Qwen3.5 template refuses an audio item.
    if (model_type, name) != ("qwen3_5", "clip")])
def test_other_model_types_render_as_before(route, stock, model_type, name):
    """A type outside the lists keeps the stock render byte for byte,
    though its messages carry the marks. Gemma 4 renders through its
    processor's nested rebuild."""
    assert model_type not in render._IN_PLACE_VIDEOS | render._IN_PLACE_AUDIO
    messages = _conversations()[name]
    placed, before = route(model_type, messages), stock(model_type, messages)
    assert placed.prompt == before.prompt
    assert (placed.videos, placed.audio) == (before.videos, before.audio)


def test_a_listed_type_falls_back_when_a_video_has_no_mark(route,
                                                           monkeypatch):
    """A video the marks do not account for keeps the stock rebuild, so
    the processor still gets one marker per video."""
    monkeypatch.setattr(render, "_video_reference", lambda item: None)
    call = route("lfm3_omni", [{"role": "user", "content": [
        _text("What happens?"), _video("a")]}])
    _order(call.prompt, _VIDEO, "What happens?")


def test_the_lists_hold_the_checked_types():
    """A type renders a video or a clip in place only after a check
    through the route on its real processor class."""
    assert render._IN_PLACE_VIDEOS == {"lfm3_omni", "lfm3_vl"}
    assert render._IN_PLACE_AUDIO == {"lfm3_omni", "lfm3_audio"}
    assert render._IN_PLACE_VIDEOS <= render._IN_PLACE_IMAGES
    # LFM3-Audio takes no image, so it has no image order to keep.
    assert render._IN_PLACE_AUDIO - {"lfm3_audio"} <= render._IN_PLACE_IMAGES


def test_a_responses_message_gets_no_video_mark():
    """Only chat completions content is marked. The Responses route reads
    no video part."""
    content = [_text("AAA"), _video("a"), _clip()]
    assert render._placed_content(render._PlacedImages(content)) == [
        {"type": "text", "text": "AAA"}]
    assert render._placed_content(render._PlacedMedia(content)) == [
        {"type": "text", "text": "AAA"}, {"type": "audio"}]
    assert render._placed_content(render._PlacedParts(content)) == [
        {"type": "text", "text": "AAA"}, {"type": "video"},
        {"type": render._SENT_AUDIO}]
