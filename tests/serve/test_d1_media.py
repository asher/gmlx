#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Asher Feldman
"""The media form of a d1 decision, as llama-server reads it: where media
come from, their order, what is taken out of the state, and the limits.
Nothing decodes here."""

from __future__ import annotations

import pytest

from gmlx.serve import d1_media
from gmlx.systemone import d1

_IMG = "data:image/png;base64,AA=="
_VID = "data:video/mp4;base64,AA=="
_AUD = "data:audio/wav;base64,AA=="


def _q(**extra):
    return {"q": {"type": "noul", "instructions": "?", **extra}}


def test_media_order_is_files_then_images_then_message_parts():
    state = [{"role": "user", "content": [
        {"type": "text", "text": "look"},
        {"type": "image_url", "image_url": {"url": _IMG}},
        {"type": "input_audio", "input_audio": {"data": "UklGRg==", "format": "wav"}},
        {"type": "video_url", "video_url": _VID}]},
        {"role": "assistant", "content": "plain"}]
    got_state, refs, spoken = d1_media.collect(
        {"state": state, "files": [_AUD], "images": [_VID], "questions": _q()})
    assert [k for k, _ in refs] == ["audio", "video", "image", "audio", "video"]
    assert got_state == [{"role": "user", "content": [{"type": "text", "text": "look"}]},
                         {"role": "assistant", "content": "plain"}]
    assert spoken == {}


def test_a_wrapped_message_state_keeps_its_other_keys():
    state = {"messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": _IMG}]}], "meta": 1}
    got, refs, _ = d1_media.collect({"state": state, "questions": _q()})
    assert got == {"messages": [{"role": "user", "content": []}], "meta": 1}
    assert refs == [("image", _IMG)]


def test_a_spoken_question_is_collected_apart_from_the_state():
    body = {"state": None, "questions": {
        "s": {"type": "noul", "instructions": {"type": "input_audio",
                                               "input_audio": {"data": "UklGRg=="}}}}}
    _, refs, spoken = d1_media.collect(body)
    assert refs == [] and spoken == {"s": "UklGRg=="}
    assert d1.body_media(body)


@pytest.mark.parametrize("body,message", [
    ({"videos": [_VID]}, '"videos" is not supported'),
    ({"images": "x"}, '"images" must be an array'),
    ({"files": ["https://example.com/a.png"]}, "must be data URLs"),
    ({"files": ["data:text/plain;base64,AA=="]}, "not an image, audio or video"),
    ({"files": [_IMG] * 9}, "too many media files, the maximum is 8"),
    ({"state": [{"content": [{"type": "input_audio",
                              "input_audio": {"data": "http://x/a.wav"}}]}]},
     "audio must be base64 data"),
])
def test_media_llama_server_refuses_are_refused(body, message):
    with pytest.raises(d1_media.D1MediaError, match=message):
        d1_media.collect({"state": None, "questions": _q(), **body})


def test_markers_expand_in_order():
    items = [d1_media.Item("image", {}, "<I>"), d1_media.Item("audio", {}, "<A>")]
    text = f"x{d1_media.MARKER}y{d1_media.MARKER}z"
    assert d1_media.expand(text, items) == "x<I>y<A>z"
    with pytest.raises(RuntimeError, match="2 media markers for 1 items"):
        d1_media.expand(text, items[:1])


def test_a_prompt_with_media_counts_toward_the_cap():
    template = "{% for image in images %}{{ image }}{% endfor %}{{ instructions }}"
    fmt = d1.D1Format(template, {"type": "lfm3-d1-omni"}, [],
                      lambda t: list(range(len(t))) if len(t) > 3 else [1])
    (q,) = d1.parse_questions({"state": None, "questions": _q()}, omni=True,
                              max_questions=4)
    media = d1_media.Media(None, [d1_media.Item("image", {}, "I" * 50)])
    (p,) = d1_media.build_passes(fmt, [q], media, cap=100)
    assert len(p.ids) == 51 and p.items == media.items
    with pytest.raises(d1.D1RequestError, match="media included"):
        d1_media.build_passes(fmt, [q], media, cap=50)
