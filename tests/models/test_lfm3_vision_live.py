#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Asher Feldman
"""LFM3-Omni vision against the reference implementation's recording.

``integration``; needs ``KQUANT_TEST_GGUF_DIR`` to hold an LFM3-Omni text
GGUF and its ``lfm3vl`` projector, in any subfolder.
``scripts/lfm3_oracle/vision.py`` records the fixture from the vendor's
checkpoint in float32: an image and a clip, their patches, their projected
features, and the token ids of one prompt for each.

The tower runs its 16-bit weights in a 16-bit stream by default, as the
vendor's checkpoint does. The features then differ from the float32
recording by a few percent, the same distance the vendor's own 16-bit
tower is from it. The float32 stream is held to a tight floor, which is
what shows that the port is the same function.
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
STREAM_REL_RMS = 0.06
STREAM_MEAN_COSINE = 0.998


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
    """The speech generator's tensors are the only ones the load
    leaves."""
    from gmlx.load.headerscan import scan_gguf
    from gmlx.load.vlm import load_vlm_model

    model, config, _processor = load_vlm_model(pair[0], pair[1], verbose=True)
    log = capsys.readouterr().out
    assert "redundant" not in log, log
    scan = scan_gguf(pair[1], include_tensors=True)
    assert config["model_type"] == (
        "lfm3_omni" if scan.kv.get("clip.has_audio_encoder") else "lfm3_vl")
    generator = sum(t.name.startswith("a.gen.") for t in scan.tensors)
    assert f" {generator} skipped" in log, log
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


@_NEEDS_GPU
@pytest.mark.parametrize("what", ["image", "clip"])
def test_the_default_stream_holds_the_16_bit_floor(loaded, ref, what):
    model = loaded[0]
    got = _features(model, ref[f"{what}_pixel_values"], ref[f"{what}_grid"])
    assert np.isfinite(got).all()
    rel, cosine = _distance(got, ref[f"{what}_features"])
    assert rel <= STREAM_REL_RMS and cosine >= STREAM_MEAN_COSINE, \
        (rel, cosine)


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


_MARKS = r"AAA|BBB|CCC|DDD|<\|image\|>|<\|video\|>"


def _marks(tok, messages, **kw) -> list[str]:
    import re

    return re.findall(_MARKS, tok.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False, **kw))


def test_the_shipped_template_is_what_the_serve_render_lists_say(loaded):
    """The serve render keeps three lists by model type. Each says what
    the shipped template does: it renders a media item where the message
    holds it, it renders an image inside a tool turn, and it takes a tool
    call's arguments as a mapping and not as a JSON string."""
    from gmlx.serve.patches import render

    tok = loaded[2].tokenizer
    model_type = loaded[1]["model_type"]
    assert model_type in render._IN_PLACE_IMAGES
    assert model_type in render._TOOL_IMAGES
    assert model_type in render._DICT_TOOL_ARGUMENTS

    def text(value):
        return {"type": "text", "text": value}

    user = {"role": "user", "content": [
        text("AAA"), {"type": "image"}, text("BBB"), {"type": "image"},
        text("CCC"), {"type": "video"}, text("DDD")]}
    assert _marks(tok, [user]) == [
        "AAA", "<|image|>", "BBB", "<|image|>", "CCC", "<|video|>", "DDD"]

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
        rel, cosine = _distance(got, ref[f"{what}_features"])
        assert rel <= STREAM_REL_RMS and cosine >= STREAM_MEAN_COSINE, \
            (what, rel, cosine)
