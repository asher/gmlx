"""The fields of POST /v1/embeddings (gmlx.serve.embeddings_request): the
input forms and their grouping as vLLM groups them, the content part types,
the chat template's rendering of a conversation, and the vLLM fields that
gmlx takes for one value only. No model and no media decode."""

from __future__ import annotations

import base64

import pytest

from gmlx.serve.embeddings_errors import EmbeddingsRequestError
from gmlx.serve.embeddings_request import MediaRef, items, parse, render

IMG = "data:image/png;base64,iVBORw0KGgo="
IMAGE = {"type": "image_url", "image_url": {"url": IMG}}
AUDIO = {"type": "input_audio", "input_audio": {"data": "UklGRg==", "format": "wav"}}


def _items(body):
    return items(body)[0]


def test_a_string_is_one_item():
    assert [i.parts for i in _items({"input": "hello"})] == [["hello"]]


def test_a_part_dict_is_one_item_like_its_one_part_list():
    one = _items({"input": IMAGE})
    listed = _items({"input": [[IMAGE]]})
    bare = _items({"input": [IMAGE]})
    assert len(one) == len(listed) == len(bare) == 1
    assert one[0].parts == listed[0].parts == bare[0].parts
    assert one[0].parts == [MediaRef("image", "image_url", ref=IMG)]


def test_a_flat_list_of_parts_is_one_item_per_part():
    got = _items({"input": [IMAGE, {"type": "text", "text": "a cat"}]})
    assert [i.parts for i in got] == [[MediaRef("image", "image_url", ref=IMG)],
                                      ["a cat"]]


def test_an_inner_list_is_one_interleaved_item():
    got = _items({"input": [["a photo of ", IMAGE, {"type": "text", "text": "!"}],
                            "plain"]})
    assert len(got) == 2
    assert got[0].parts[0] == "a photo of " and got[0].parts[2] == "!"
    assert isinstance(got[0].parts[1], MediaRef)
    assert got[1].parts == ["plain"]


def test_chat_messages_group_as_vllm_groups_them():
    convo = [{"role": "user", "content": "hi"}]
    found, chat = items({"input": convo})
    assert chat and len(found) == 1
    found, chat = items({"input": [convo, convo]})
    assert chat and len(found) == 2
    found, chat = items({"messages": convo})
    assert chat and len(found) == 1
    found, chat = items({"messages": [convo, convo, convo]})
    assert chat and len(found) == 3


@pytest.mark.parametrize("body, words", [
    ({"input": [{"role": "user", "content": "hi"}, "text"]}, "mixes chat messages"),
    ({"input": [[{"role": "user", "content": "hi"}, IMAGE]]}, "mixes chat messages"),
    ({"input": {"role": "user", "content": "hi"}}, "must be in a list"),
    ({"input": "x", "messages": [{"role": "user", "content": "y"}]}, "exactly one"),
    ({}, "exactly one"),
    ({"messages": "hi"}, "list of chat messages"),
    ({"input": [[1, 2, 3]]}, "token ids"),
    ({"input": [IMAGE, []]}, "empty list"),
    ({"input": [{"type": "image_file", "path": "/x"}]}, "'image_file' is not supported"),
    ({"input": [{"type": "image_url", "image_url": {"url": IMG}, "file_id": "f"}]},
     "file_id is not supported"),
    ({"input": [{"type": "audio", "url": "a", "path": "/b"}]}, "takes one of"),
    ({"input": [{"type": "image", "base64": "not base64!"}]}, "base64"),
    ({"input": [{"type": "input_audio", "input_audio": {}}]}, "input_audio.data"),
])
def test_bad_inputs_are_refused(body, words):
    with pytest.raises(EmbeddingsRequestError, match=words) as ei:
        items(body)
    assert ei.value.status_code == 400


_B64 = base64.b64encode(b"\x00\x01\x02").decode()


@pytest.mark.parametrize("part, want", [
    ({"type": "text", "text": "hi"}, "hi"),
    ({"type": "image_url", "image_url": IMG}, MediaRef("image", "image_url", ref=IMG)),
    ({"type": "input_image", "image_url": IMG},
     MediaRef("image", "input_image", ref=IMG)),
    ({"type": "image", "image": IMG}, MediaRef("image", "image.image", ref=IMG)),
    ({"type": "image", "url": "https://x/y.png"},
     MediaRef("image", "image.url", ref="https://x/y.png")),
    ({"type": "image", "path": "/m/a.png"},
     MediaRef("image", "image.path", ref="/m/a.png", path=True)),
    ({"type": "image", "base64": _B64},
     MediaRef("image", "image.base64", data=b"\x00\x01\x02")),
    ({"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                  "data": _B64}},
     MediaRef("image", "image.source.data", data=b"\x00\x01\x02")),
    ({"type": "input_audio", "input_audio": {"data": _B64, "format": "wav"}},
     MediaRef("audio", "input_audio.data", data=b"\x00\x01\x02")),
    ({"type": "input_audio", "input_audio": {"data": "data:audio/wav;base64,AA=="}},
     MediaRef("audio", "input_audio.data", ref="data:audio/wav;base64,AA==")),
    ({"type": "audio_url", "audio_url": {"url": "/m/a.wav"}},
     MediaRef("audio", "audio_url", ref="/m/a.wav")),
    ({"type": "audio", "audio": "/m/a.wav"},
     MediaRef("audio", "audio.audio", ref="/m/a.wav")),
    ({"type": "audio", "base64": _B64},
     MediaRef("audio", "audio.base64", data=b"\x00\x01\x02")),
    ({"type": "video_url", "video_url": {"url": "/m/v.mp4"}},
     MediaRef("video", "video_url", ref="/m/v.mp4")),
    ({"type": "input_video", "video_url": "/m/v.mp4"},
     MediaRef("video", "input_video", ref="/m/v.mp4")),
    ({"type": "video", "video": "/m/v.mp4"},
     MediaRef("video", "video.video", ref="/m/v.mp4")),
    ({"type": "video", "path": "/m/v.mp4"},
     MediaRef("video", "video.path", ref="/m/v.mp4", path=True)),
])
def test_each_part_type(part, want):
    assert _items({"input": [[part]]})[0].parts == [want]


def test_a_conversation_renders_as_the_chat_template_does():
    convo = [
        {"role": "user", "content": [IMAGE, {"type": "text", "text": "a dog"}]},
        {"role": "system", "content": "title: none | text: "},
        {"role": "assistant", "content": [{"type": "text", "text": " and <|audio|>"},
                                          AUDIO]},
    ]
    item = render(convo)
    # System text first; the user's image gets a placeholder where it sits;
    # the assistant's text names its own, so its audio adds none.
    assert item.parts[0] == "title: none | text: <|image|>a dog and <|audio|>"
    assert [m.kind for m in item.parts[1:]] == ["image", "audio"]
    assert item.chat


@pytest.mark.parametrize("field, value", [
    ("embed_dtype", "float16"), ("embed_dtype", "bfloat16"),
    ("endianness", "big"), ("use_activation", False),
    ("truncate_prompt_tokens", 128), ("truncate_prompt_tokens", 0),
    ("chat_template", "{{ x }}"), ("padding", "max_length"),
    ("truncation_side", "left"), ("media_io_kwargs", {"video": {"num_frames": 8}}),
    ("priority", 1),
])
def test_vllm_fields_gmlx_does_not_honor_are_refused(field, value):
    with pytest.raises(EmbeddingsRequestError, match=field) as ei:
        parse({"input": "x", field: value})
    assert ei.value.status_code == 400


@pytest.mark.parametrize("field, value", [
    ("embed_dtype", "float32"), ("endianness", "native"), ("endianness", "little"),
    ("use_activation", True), ("truncate_prompt_tokens", -1),
    ("padding", "do_not_pad"), ("truncation_side", "right"), ("priority", 0),
    ("request_id", "abc"), ("cache_salt", "s"), ("user", "u"),
    ("add_generation_prompt", True), ("continue_final_message", False),
    ("chat_template_kwargs", {"enable_thinking": False}),
])
def test_vllm_fields_gmlx_honors_or_ignores_pass(field, value):
    assert parse({"input": "x", field: value}).items[0].parts == ["x"]


def test_parse_reads_the_other_fields():
    r = parse({"input": ["a", IMAGE], "encoding_format": "BASE64",
               "dimensions": 256, "prompt_name": "query",
               "mm_processor_kwargs": {"max_soft_tokens": 560},
               "add_special_tokens": False})
    assert (r.encoding_format, r.dimensions, r.prompt_name) == ("base64", 256, "query")
    assert r.mm_processor_kwargs == {"max_soft_tokens": 560}
    assert r.add_special_tokens is False and r.has_media and not r.chat


@pytest.mark.parametrize("body, words", [
    ({"input": "x", "dimensions": "256"}, "'dimensions' must be an integer"),
    ({"input": "x", "dimensions": True}, "'dimensions' must be an integer"),
    ({"input": "x", "prompt_name": 3}, "prompt_name must be a string"),
    ({"input": "x", "mm_processor_kwargs": []}, "must be an object"),
    ({"input": "x", "add_special_tokens": "no"}, "true or false"),
    ({"input": "x", "encoding_format": "float16"}, "encoding_format"),
])
def test_bad_fields_are_refused(body, words):
    with pytest.raises(EmbeddingsRequestError, match=words):
        parse(body)
