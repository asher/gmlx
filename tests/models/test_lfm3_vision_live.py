#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Asher Feldman
"""LFM3-Omni vision against the reference implementation's recording.

``integration``; needs ``KQUANT_TEST_GGUF_DIR`` to hold an LFM3-Omni text
GGUF and its ``lfm3vl`` projector, in any subfolder.
``scripts/lfm3_oracle/vision.py`` records the fixture from the vendor's
checkpoint in float32: an image and a clip, their patches, their projected
features, and the token ids of one prompt for each.

The float32 stream is held to a tight floor, which is what shows that
the port is the same function.

The tower runs its 16-bit weights in a 16-bit stream by default, as the
vendor's checkpoint does. The fixture also holds the features of the
vendor's tower in bfloat16. The 16-bit stream is held to that recording:
it may be at most ``STREAM_MARGIN`` times as far from the float32
features as the vendor's own 16-bit tower is. The margin was set before
the first comparison. A second test checks that the 16-bit and float32
streams give the same greedy answer tokens up to the first near tie of
the two best logits. On the image prompt they part at such a tie, where
the vendor's own 16-bit and float32 towers part too.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

import mlx.core as mx

pytestmark = pytest.mark.integration

_NEEDS_GPU = pytest.mark.skipif(
    bool(os.environ.get("KQUANT_FORCE_CPU")),
    reason="a real-weights tower forward is a GPU workload")

FIXTURE = Path(__file__).parent.parent / "fixtures/lfm3_oracle/vision.npz"

# The fixture stores features as float16, which bounds the float32 floor.
F32_REL_RMS = 1e-3
STREAM_MARGIN = 1.5
GREEDY_TOKENS = 32


@pytest.fixture(scope="module")
def ref():
    return np.load(FIXTURE)


@pytest.fixture(scope="module")
def pair(gguf_dir, gguf_index):
    from tests.lfm3_header import find_vision_pair

    found = find_vision_pair(gguf_dir, gguf_index)
    if found is None:
        pytest.skip("no LFM3 model with its lfm3vl projector under "
                    "KQUANT_TEST_GGUF_DIR")
    return found


@pytest.fixture(scope="module")
def loaded(pair):
    from gmlx.load.vlm import load_vlm_model

    return load_vlm_model(pair[0], pair[1])


def _distance(got, want) -> tuple[float, float]:
    """Relative RMS error and mean cosine over the feature rows."""
    got = np.asarray(got, dtype=np.float64)
    want = np.asarray(want, dtype=np.float64)
    cosine = (got * want).sum(-1) / (
        np.linalg.norm(got, axis=-1) * np.linalg.norm(want, axis=-1))
    return (float(np.linalg.norm(got - want) / np.linalg.norm(want)),
            float(cosine.mean()))


def _features(model, pixel_values, grid) -> np.ndarray:
    return np.asarray(model.vision_features(
        mx.array(pixel_values), grid).astype(mx.float32))


def test_the_load_fills_every_parameter(pair, capsys):
    """The load leaves no projector tensor, the speech generator's
    included."""
    from gmlx.load.headerscan import scan_gguf
    from gmlx.load.vlm import load_vlm_model

    model, config, _processor = load_vlm_model(pair[0], pair[1], verbose=True)
    log = capsys.readouterr().out
    assert "redundant" not in log, log
    scan = scan_gguf(pair[1], include_tensors=True)
    assert config["model_type"] == (
        "lfm3_omni" if scan.kv.get("clip.has_audio_encoder") else "lfm3_vl")
    assert " 0 skipped" in log, log
    assert len(model.vision_tower.layers) == \
        scan.kv["clip.vision.block_count"]
    # The head masks the speech markers and the media placeholders, as on
    # a load without the projector.
    from gmlx.models.lfm3.model import MEDIA_PLACEHOLDERS, MODE_MARKERS

    vocab = _processor.tokenizer.get_vocab()
    assert list(model.language_model.suppressed_tokens) == sorted(
        vocab[name] for name in MODE_MARKERS + MEDIA_PLACEHOLDERS)


def test_patches_are_the_reference_processors(loaded, ref):
    """Neither input is resized, so the patches are equal to rounding."""
    _model, _config, processor = loaded
    from PIL import Image

    image = processor.image_processor.preprocess(
        [Image.fromarray(ref["image"])])
    assert np.array_equal(image["image_grid_thw"], ref["image_grid"])
    np.testing.assert_allclose(image["pixel_values"],
                               ref["image_pixel_values"], atol=1e-6)
    clip = processor.video_processor.preprocess([ref["clip"]])
    assert np.array_equal(clip["video_grid_thw"], ref["clip_grid"])
    np.testing.assert_allclose(clip["pixel_values_videos"],
                               ref["clip_pixel_values"], atol=1e-6)


@_NEEDS_GPU
@pytest.mark.parametrize("what", ["image", "clip"])
def test_a_float32_stream_gives_the_reference_features(loaded, ref, what):
    model = loaded[0]
    tower = model.vision_tower
    tower.stream_dtype = mx.float32
    try:
        got = _features(model, ref[f"{what}_pixel_values"],
                        ref[f"{what}_grid"])
    finally:
        tower.stream_dtype = None
    rel, _cosine = _distance(got, ref[f"{what}_features"])
    assert rel <= F32_REL_RMS, rel


def _holds_the_16_bit_recording(got, ref, what) -> None:
    from tests.lfm3_header import stream_distance

    rel, miss = stream_distance(got, ref[f"{what}_features"])
    ref_rel, ref_miss = stream_distance(ref[f"{what}_features_bf16"],
                                        ref[f"{what}_features"])
    assert rel <= STREAM_MARGIN * ref_rel, (what, rel, ref_rel)
    assert miss <= STREAM_MARGIN * ref_miss, (what, miss, ref_miss)


@_NEEDS_GPU
@pytest.mark.parametrize("what", ["image", "clip"])
def test_the_default_stream_holds_the_reference_16_bit_tower(loaded, ref,
                                                             what):
    model = loaded[0]
    got = _features(model, ref[f"{what}_pixel_values"], ref[f"{what}_grid"])
    assert np.isfinite(got).all()
    _holds_the_16_bit_recording(got, ref, what)


@_NEEDS_GPU
@pytest.mark.parametrize("what", ["image", "clip"])
def test_the_16_bit_and_float32_streams_give_the_same_answer(loaded, ref,
                                                             what):
    """The greedy tokens after the recorded prompt, with the tower in
    each stream. The two may part only where the best two logits are a
    near tie."""
    from tests.lfm3_header import greedy_tokens, same_answer_up_to_a_tie

    model = loaded[0]
    key = "pixel_values" if what == "image" else "pixel_values_videos"
    grid = "image_grid_thw" if what == "image" else "video_grid_thw"
    inputs = {key: mx.array(ref[f"{what}_pixel_values"]),
              grid: ref[f"{what}_grid"]}
    ids = ref[f"{what}_prompt_ids"]
    default = greedy_tokens(model, ids, GREEDY_TOKENS, **inputs)
    model.vision_tower.stream_dtype = mx.float32
    try:
        exact = greedy_tokens(model, ids, GREEDY_TOKENS, **inputs)
    finally:
        model.vision_tower.stream_dtype = None
    assert same_answer_up_to_a_tie(default, exact), (default[0], exact[0])
    # The towers agree on the start of the answer.
    assert default[0][:8] == exact[0][:8]


def test_the_debug_switch_runs_the_towers_in_float32(pair, monkeypatch):
    from gmlx.load.vlm import load_vlm_model
    from gmlx.models.lfm3.vlm_model import TOWER_FP32_ENV

    monkeypatch.setenv(TOWER_FP32_ENV, "1")
    model, _config, _processor = load_vlm_model(*pair)
    assert model.vision_tower.stream_dtype == mx.float32
    if getattr(model, "audio_encoder", None) is not None:
        assert model.audio_encoder.stream_dtype == mx.float32


def test_prompts_are_the_reference_token_ids(loaded, ref):
    """The chat template and the expansion write the ids the reference
    template and processor write, for an image and for a clip."""
    from PIL import Image

    from gmlx._vendor.mlx_vlm.utils import VideoMetadata

    processor = loaded[2]
    tok = processor.tokenizer

    def prompt(kind, question):
        return tok.apply_chat_template(
            [{"role": "user", "content": [
                {"type": kind}, {"type": "text", "text": question}]}],
            add_generation_prompt=True, tokenize=False)

    out = processor(images=[Image.fromarray(ref["image"])],
                    text=[prompt("image", "Describe this image.")],
                    add_special_tokens=False)
    assert np.asarray(out["input_ids"]).reshape(-1).tolist() == \
        ref["image_prompt_ids"].tolist()

    meta = VideoMetadata(
        total_num_frames=int(ref["clip_total"]), fps=float(ref["clip_fps"]),
        frames_indices=ref["clip_indices"].tolist())
    out = processor(videos=[ref["clip"]],
                    text=[prompt("video", "What happens?")],
                    video_metadata=[meta], add_special_tokens=False)
    assert np.asarray(out["input_ids"]).reshape(-1).tolist() == \
        ref["clip_prompt_ids"].tolist()


_MARKS = r"AAA|BBB|CCC|DDD|EEE|<\|image\|>|<\|video\|>|<\|audio\|>"


def _marks(tok, messages, **kw) -> list[str]:
    import re

    return re.findall(_MARKS, tok.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False, **kw))


def test_the_shipped_template_is_what_the_serve_render_lists_say(loaded):
    """The serve render keeps lists by model type. Each says what the
    shipped template does: it renders an image, a video and an audio item
    where the message holds it, it renders an image inside a tool turn,
    and it takes a tool call's arguments as a mapping and not as a JSON
    string."""
    from gmlx.serve.patches import render

    tok = loaded[2].tokenizer
    model_type = loaded[1]["model_type"]
    assert model_type in render._IN_PLACE_IMAGES
    assert model_type in render._IN_PLACE_VIDEOS
    assert model_type in render._TOOL_IMAGES
    assert model_type in render._DICT_TOOL_ARGUMENTS

    def text(value):
        return {"type": "text", "text": value}

    user = {"role": "user", "content": [
        text("AAA"), {"type": "image"}, text("BBB"), {"type": "image"},
        text("CCC"), {"type": "video"}, text("DDD")]}
    assert _marks(tok, [user]) == [
        "AAA", "<|image|>", "BBB", "<|image|>", "CCC", "<|video|>", "DDD"]
    hears = loaded[1].get("audio_config") is not None
    assert (model_type in render._IN_PLACE_AUDIO) == hears
    if hears:
        heard = {"role": "user", "content": [
            text("AAA"), {"type": "audio"}, text("BBB"), {"type": "video"},
            {"type": "audio"}, text("EEE")]}
        assert _marks(tok, [heard, {"role": "assistant", "content": "CCC"},
                            {"role": "user", "content": "DDD"}]) == [
            "AAA", "<|audio|>", "BBB", "<|video|>", "<|audio|>", "EEE",
            "CCC", "DDD"]

    tools = [{"type": "function", "function": {
        "name": "shot", "description": "Take a screenshot.",
        "parameters": {"type": "object", "properties": {
            "window": {"type": "string"}}}}}]

    def turns(arguments):
        return [
            {"role": "user", "content": [text("AAA")]},
            {"role": "assistant", "content": "", "tool_calls": [{
                "id": "c1", "type": "function",
                "function": {"name": "shot", "arguments": arguments}}]},
            {"role": "tool", "tool_call_id": "c1", "content": [
                text("BBB"), {"type": "image"}, text("CCC")]}]

    assert _marks(tok, turns({"window": "main"}), tools=tools) == [
        "AAA", "BBB", "<|image|>", "CCC"]
    with pytest.raises(Exception, match="mapping"):
        _marks(tok, turns('{"window": "main"}'), tools=tools)


@_NEEDS_GPU
def test_a_float16_stream_does_not_overflow(pair, ref, monkeypatch):
    """M1 and M2 run float16, whose range ends at 65,504. The tower's
    activations stay inside it on real inputs, and the features hold the
    16-bit floor."""
    from tests.lfm3_header import load_as_f16_projector

    monkeypatch.setenv("GMLX_ACTIVATION_DTYPE", "float16")
    model, _config, _processor = load_as_f16_projector(*pair)
    for what in ("image", "clip"):
        features = model.vision_features(
            mx.array(ref[f"{what}_pixel_values"]), ref[f"{what}_grid"])
        assert features.dtype == mx.float16, what
        got = np.asarray(features.astype(mx.float32))
        assert np.isfinite(got).all(), what
        _holds_the_16_bit_recording(got, ref, what)
