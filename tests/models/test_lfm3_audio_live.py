#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Asher Feldman
"""LFM3-Omni speech input against the reference implementation's
recording.

``integration``; needs ``KQUANT_TEST_GGUF_DIR`` to hold an LFM3-Omni text
GGUF and its projector, in any subfolder.
``scripts/lfm3_oracle/audio.py`` records the fixture from the vendor's
checkpoint in float32: two clips, their spectrograms, their projected
features alone and as one batch, and the token ids of a prompt holding
both.

The float32 stream is held to a tight floor, which is what shows that
the port is the same function.

The encoder runs its 16-bit weights in a 16-bit stream by default, as
the vendor's checkpoint does. The fixture also holds the batch's
features from the vendor's model in bfloat16. The 16-bit stream is held
to that recording: it may be at most ``STREAM_MARGIN`` times as far from
the float32 features as the vendor's own 16-bit model is. The margin was
set before the first comparison. A second test checks that the 16-bit
and float32 streams give the same greedy answer tokens up to the first
near tie of the two best logits.
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
    reason="a real-weights encoder forward is a GPU workload")

FIXTURE = Path(__file__).parent.parent / "fixtures/lfm3_oracle/audio.npz"

# The fixture stores spectrograms and features as float16.
MEL_MAX_ABS = 2e-3
F32_REL_RMS = 1e-3
STREAM_MARGIN = 1.5
GREEDY_TOKENS = 32


@pytest.fixture(scope="module")
def ref():
    return np.load(FIXTURE)


@pytest.fixture(scope="module")
def loaded(gguf_dir, gguf_index):
    from gmlx.load.headerscan import scan_gguf
    from gmlx.load.vlm import load_vlm_model
    from tests.lfm3_header import find_vision_pair

    found = find_vision_pair(gguf_dir, gguf_index)
    if found is None or not scan_gguf(
            found[1], include_tensors=False).kv.get("clip.has_audio_encoder"):
        pytest.skip("no LFM3-Omni model with its projector under "
                    "KQUANT_TEST_GGUF_DIR")
    return load_vlm_model(*found)


def _waves(ref) -> list:
    return [ref[f"clip{i}"].astype(np.float32) / 32768.0 for i in range(2)]


def _distance(got, want) -> tuple[float, float]:
    """Relative RMS error and mean cosine over the feature rows."""
    got = np.asarray(got, dtype=np.float64)
    want = np.asarray(want, dtype=np.float64)
    cosine = (got * want).sum(-1) / (
        np.linalg.norm(got, axis=-1) * np.linalg.norm(want, axis=-1))
    return (float(np.linalg.norm(got - want) / np.linalg.norm(want)),
            float(cosine.mean()))


def _features(model, pre) -> np.ndarray:
    return np.asarray(model.audio_features(
        mx.array(pre["input_features"]),
        pre["feature_attention_mask"]).astype(mx.float32))


def test_the_load_fills_the_encoder_and_the_speech_generator(
        gguf_dir, gguf_index, capsys):
    from gmlx.load.headerscan import scan_gguf
    from gmlx.load.vlm import load_vlm_model
    from tests.lfm3_header import find_vision_pair

    found = find_vision_pair(gguf_dir, gguf_index)
    if found is None:
        pytest.skip("no LFM3 model with its projector")
    scan = scan_gguf(found[1], include_tensors=True)
    if not scan.kv.get("clip.has_audio_encoder"):
        pytest.skip("the projector has no audio encoder")
    model, config, _processor = load_vlm_model(*found, verbose=True)
    log = capsys.readouterr().out
    assert "redundant" not in log, log
    assert " 0 skipped" in log, log
    if scan.kv.get("clip.has_gen_audio_encoder"):
        assert len(model.depth_decoder.layers) == \
            scan.kv["clip.gen.audio.block_count"]
    assert len(model.audio_encoder.layers) == \
        scan.kv["clip.audio.block_count"]
    assert config["audio_config"]["sliding_window"] == \
        scan.kv["clip.audio.window_size"]
    # The depthwise taps and the downsampler's convolutions stay float32.
    assert model.audio_encoder.layers[0].conv.kernel.dtype == mx.float32
    assert model.audio_encoder.downsample.conv_in.weight.dtype == mx.float32


def test_spectrograms_are_the_reference_extractors(loaded, ref):
    extractor = loaded[2].feature_extractor
    waves = _waves(ref)
    pre = extractor(waves)
    assert pre["input_features"].shape == (2, 128, 130)
    assert pre["feature_attention_mask"].sum(-1).tolist() == [130, 51]
    for i in range(2):
        want = ref[f"clip{i}_mel"].astype(np.float32)
        got = pre["input_features"][i, :, :want.shape[1]]
        assert np.abs(got - want).max() <= MEL_MAX_ABS
        assert extractor.tokens(len(waves[i])) == \
            ref[f"clip{i}_features"].shape[0]
    # The shorter clip is padded with zeros past its own frames.
    assert not pre["input_features"][1, :, 51:].any()


@_NEEDS_GPU
@pytest.mark.parametrize("what", ["clip0", "clip1", "batch"])
def test_a_float32_stream_gives_the_reference_features(loaded, ref, what):
    model, extractor = loaded[0], loaded[2].feature_extractor
    waves = _waves(ref)
    pre = extractor(waves if what == "batch" else [waves[int(what[-1])]])
    encoder = model.audio_encoder
    encoder.stream_dtype = mx.float32
    try:
        got = _features(model, pre)
    finally:
        encoder.stream_dtype = None
    rel, _cosine = _distance(got, ref[f"{what}_features"])
    assert rel <= F32_REL_RMS, rel


def _holds_the_16_bit_recording(got, ref) -> None:
    from tests.lfm3_header import stream_distance

    rel, miss = stream_distance(got, ref["batch_features"])
    ref_rel, ref_miss = stream_distance(ref["batch_features_bf16"],
                                        ref["batch_features"])
    assert rel <= STREAM_MARGIN * ref_rel, (rel, ref_rel)
    assert miss <= STREAM_MARGIN * ref_miss, (miss, ref_miss)


@_NEEDS_GPU
def test_the_default_stream_holds_the_reference_16_bit_model(loaded, ref):
    model, extractor = loaded[0], loaded[2].feature_extractor
    got = _features(model, extractor(_waves(ref)))
    assert np.isfinite(got).all()
    _holds_the_16_bit_recording(got, ref)


@_NEEDS_GPU
def test_the_16_bit_and_float32_streams_give_the_same_answer(loaded, ref):
    """The greedy tokens after the recorded two-clip prompt, with the
    encoder in each stream. The two may part only where the best two
    logits are a near tie."""
    from tests.lfm3_header import greedy_tokens, same_answer_up_to_a_tie

    model = loaded[0]
    pre = loaded[2].feature_extractor(_waves(ref))
    inputs = {"input_features": mx.array(pre["input_features"]),
              "feature_attention_mask": pre["feature_attention_mask"]}
    default = greedy_tokens(model, ref["prompt_ids"], GREEDY_TOKENS, **inputs)
    model.audio_encoder.stream_dtype = mx.float32
    try:
        exact = greedy_tokens(model, ref["prompt_ids"], GREEDY_TOKENS,
                              **inputs)
    finally:
        model.audio_encoder.stream_dtype = None
    assert same_answer_up_to_a_tie(default, exact), (default[0], exact[0])
    assert default[0][:8] == exact[0][:8]


@_NEEDS_GPU
def test_a_float16_stream_does_not_overflow(gguf_dir, gguf_index, ref,
                                            monkeypatch):
    """M1 and M2 run float16, whose range ends at 65,504. The encoder's
    activations stay inside it on real inputs, and the features hold the
    16-bit floor."""
    from tests.lfm3_header import find_vision_pair, load_as_f16_projector

    found = find_vision_pair(gguf_dir, gguf_index)
    if found is None:
        pytest.skip("no LFM3 model with its projector")
    monkeypatch.setenv("GMLX_ACTIVATION_DTYPE", "float16")
    model, _config, processor = load_as_f16_projector(*found)
    if getattr(model, "audio_encoder", None) is None:
        pytest.skip("the projector has no audio encoder")
    pre = processor.feature_extractor(_waves(ref))
    features = model.audio_features(
        mx.array(pre["input_features"]), pre["feature_attention_mask"])
    assert features.dtype == mx.float16
    got = np.asarray(features.astype(mx.float32))
    assert np.isfinite(got).all()
    _holds_the_16_bit_recording(got, ref)


def test_a_prompt_with_two_clips_is_the_reference_token_ids(loaded, ref):
    processor = loaded[2]
    text = processor.tokenizer.apply_chat_template(
        [{"role": "user", "content": [
            {"type": "audio"}, {"type": "text", "text": "What do you hear?"},
            {"type": "audio"}]}],
        add_generation_prompt=True, tokenize=False)
    out = processor(text=[text], audio=_waves(ref), add_special_tokens=False)
    assert np.asarray(out["input_ids"]).reshape(-1).tolist() == \
        ref["prompt_ids"].tolist()
    assert np.asarray(out["input_features"]).shape == (2, 128, 130)


SOUNDTRACK = Path(__file__).parent.parent / "fixtures/lfm3_oracle/soundtrack.npz"


def test_a_video_with_its_soundtrack_is_the_reference_token_ids(loaded):
    """``scripts/lfm3_oracle/soundtrack.py`` records the reference
    processor with ``use_audio_in_video``. The ids depend on the frame
    geometry and the clip lengths only, so the media are zeros."""
    from gmlx._vendor.mlx_vlm.utils import VideoMetadata

    rec = np.load(SOUNDTRACK)
    processor = loaded[2]
    frames = np.zeros(tuple(rec["shape"]), dtype=np.uint8)

    def encode(content, soundtrack, clips=None):
        text = processor.tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            add_generation_prompt=True, tokenize=False)
        meta = VideoMetadata(
            total_num_frames=int(rec["total"]), fps=float(rec["fps"]),
            frames_indices=rec["indices"].tolist())
        return processor(
            text=[text], videos=[frames], video_metadata=[meta],
            soundtracks=[np.zeros(soundtrack, dtype=np.float32)],
            audio=clips, add_special_tokens=False)

    question = {"type": "text", "text": "What happens?"}
    for name in ("full", "short"):
        out = encode([{"type": "video"}, question],
                     int(rec[f"{name}_samples"]))
        assert np.asarray(out["input_ids"]).reshape(-1).tolist() == \
            rec[f"{name}_ids"].tolist(), name

    before, track, after = (int(n) for n in rec["mixed_samples"])
    out = encode([{"type": "audio"}, {"type": "video"}, question,
                  {"type": "audio"}], track,
                 [np.zeros(before, dtype=np.float32),
                  np.zeros(after, dtype=np.float32)])
    assert np.asarray(out["input_ids"]).reshape(-1).tolist() == \
        rec["mixed_ids"].tolist()
    # The encoder gets the clips in prompt order, the soundtrack between.
    assert np.asarray(out["feature_attention_mask"]).sum(-1).tolist() == \
        rec["mixed_frames"].tolist()


# -- a model that hears and has no vision tower -------------------------------

@pytest.fixture(scope="module")
def audio_only_pair(gguf_dir, gguf_index):
    from tests.lfm3_header import find_audio_only_pair

    found = find_audio_only_pair(gguf_dir, gguf_index)
    if found is None:
        pytest.skip("no LFM3-Audio model with its projector under "
                    "KQUANT_TEST_GGUF_DIR")
    return found


@pytest.fixture(scope="module")
def audio_only(audio_only_pair):
    from gmlx.load.vlm import load_vlm_model

    return load_vlm_model(*audio_only_pair)


def test_an_audio_only_load_fills_every_parameter(audio_only_pair, capsys):
    from gmlx.load.headerscan import scan_gguf
    from gmlx.load.vlm import load_vlm_model

    kv = scan_gguf(audio_only_pair[1], include_tensors=False).kv
    model, config, _processor = load_vlm_model(*audio_only_pair, verbose=True)
    log = capsys.readouterr().out
    assert "redundant" not in log, log
    assert " 0 skipped" in log, log
    assert config["model_type"] == "lfm3_audio"
    assert getattr(model, "vision_tower", None) is None
    assert len(model.audio_encoder.layers) == kv["clip.audio.block_count"]
    assert len(model.depth_decoder.layers) == \
        kv["clip.gen.audio.block_count"]
    assert model.speech_capabilities() is not None


_MARKS = r"AAA|BBB|CCC|DDD|EEE|<\|image\|>|<\|video\|>|<\|audio\|>"


def _marks(tok, messages, **kw) -> list[str]:
    import re

    return re.findall(_MARKS, tok.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False, **kw))


def test_the_audio_only_template_is_what_the_serve_render_lists_say(
        audio_only):
    """The shipped template renders an audio item where the message holds
    it, and takes a tool call's arguments as a mapping and not as a JSON
    string."""
    from gmlx.serve.patches import render

    tok = audio_only[2].tokenizer
    model_type = audio_only[1]["model_type"]
    assert model_type in render._IN_PLACE_AUDIO
    assert model_type in render._DICT_TOOL_ARGUMENTS
    assert model_type not in render._IN_PLACE_VIDEOS
    assert model_type not in render._IN_PLACE_IMAGES

    def text(value):
        return {"type": "text", "text": value}

    heard = {"role": "user", "content": [
        text("AAA"), {"type": "audio"}, text("BBB"), {"type": "audio"},
        text("EEE")]}
    assert _marks(tok, [heard, {"role": "assistant", "content": "CCC"},
                        {"role": "user", "content": [
                            {"type": "audio"}, text("DDD")]}]) == [
        "AAA", "<|audio|>", "BBB", "<|audio|>", "EEE", "CCC", "<|audio|>",
        "DDD"]

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
            {"role": "tool", "tool_call_id": "c1", "content": [text("BBB")]}]

    assert _marks(tok, turns({"window": "main"}), tools=tools) == [
        "AAA", "BBB"]
    with pytest.raises(Exception, match="mapping"):
        _marks(tok, turns('{"window": "main"}'), tools=tools)


def test_an_audio_only_prompt_gives_each_clip_its_tokens(audio_only):
    """Two clips of 1 s and 2.5 s: each marker becomes its own clip's
    tokens, 12.5 a second, in the order of the message."""
    from gmlx.models.lfm3.audio import num_audio_tokens

    _model, config, processor = audio_only
    text = processor.tokenizer.apply_chat_template(
        [{"role": "user", "content": [
            {"type": "audio"}, {"type": "text", "text": "AAA"},
            {"type": "audio"}]}],
        add_generation_prompt=True, tokenize=False)
    clips = [np.zeros(16000, np.float32), np.zeros(40000, np.float32)]
    out = processor(text=[text], audio=clips, add_special_tokens=False)
    ids = np.asarray(out["input_ids"]).reshape(-1)
    runs = [len(run) for run in np.split(
        ids, np.flatnonzero(np.diff(ids == config["audio_token_id"])) + 1)
        if run[0] == config["audio_token_id"]]
    assert runs == [num_audio_tokens(16000), num_audio_tokens(40000)]
    assert runs[0] < runs[1]
    assert np.asarray(out["input_features"]).shape[0] == 2
