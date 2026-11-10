#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Asher Feldman
"""The LFM3-Omni speech generator against the reference implementation's
recording.

``integration``; needs ``KQUANT_TEST_GGUF_DIR`` to hold an LFM3-Omni text
GGUF and its projector, in any subfolder.
``scripts/lfm3_oracle/speech.py`` records the fixture from the vendor's
checkpoint in float32: the depth decoder's logits for two hidden states,
the embedding two frames feed back to the text model, and the codec's
audio for eight frames of codes. The hidden states and the codes are
seeded noise, so these check the functions and not speech.

The float32 stream is held to a tight floor, which is what shows that
the port is the same function.

Both decoders run their 16-bit weights in a 16-bit stream by default.
The fixture also holds the vendor's own results in bfloat16. The 16-bit
stream may be at most ``STREAM_MARGIN`` times as far from the float32
recording as the vendor's 16-bit model is. The feedback embedding is
stored in 16 bits and returned in 16 bits, so it has only that check.
The floors and the margin were set before the first comparison.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

import mlx.core as mx

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        bool(os.environ.get("KQUANT_FORCE_CPU")),
        reason="a real-weights decoder forward is a GPU workload"),
]

FIXTURE = Path(__file__).parent.parent / "fixtures/lfm3_oracle/speech.npz"

F32_REL_RMS = 1e-3
AUDIO_F32_SNR_DB = 60.0
STREAM_MARGIN = 1.5


@pytest.fixture(scope="module")
def ref():
    return np.load(FIXTURE)


def _pair(gguf_dir, gguf_index):
    from gmlx.load.headerscan import scan_gguf
    from tests.lfm3_header import find_vision_pair

    found = find_vision_pair(gguf_dir, gguf_index)
    if found is None or not scan_gguf(found[1], include_tensors=False).kv.get(
            "clip.has_gen_audio_encoder"):
        pytest.skip("no LFM3 model that speaks, with its projector, under "
                    "KQUANT_TEST_GGUF_DIR")
    return found


@pytest.fixture(scope="module")
def model(gguf_dir, gguf_index):
    from gmlx.load.vlm import load_vlm_model

    return load_vlm_model(*_pair(gguf_dir, gguf_index))[0]


@pytest.fixture
def stream(model):
    """Set both decoders' stream dtype for one test."""
    depth, codec = model.depth_decoder, model.codec_decoder

    def use(dtype):
        depth.stream_dtype = codec.stream_dtype = dtype

    yield use
    depth.stream_dtype = codec.stream_dtype = None


def _rel(got, want) -> float:
    got = np.asarray(got, dtype=np.float64)
    want = np.asarray(want, dtype=np.float64)
    return float(np.linalg.norm(got - want) / np.linalg.norm(want))


def _snr_db(got, want) -> float:
    return -20.0 * float(np.log10(_rel(got, want)))


def _passes(ref):
    return (("all", np.ones_like(ref["masked_some"])),
            ("some", ref["masked_some"]))


def _logits(depth, ref, mask, dtype) -> np.ndarray:
    return np.asarray(depth.logits(
        mx.array(ref["hidden"]).astype(dtype),
        mx.array(ref["codes"].astype(np.int32)), mx.array(mask)))


def test_the_generator_is_the_references_shape(model, ref):
    cfg = model.config.speech_config
    assert ref["codes"].shape[1] == cfg.num_codebooks
    assert ref["logits_all"].shape[-1] == cfg.codebook_size
    assert ref["hidden"].shape[1] == cfg.projection_dim
    assert cfg.schedule == [4, 4, 4]
    assert ref["audio"].shape == (
        ref["codec_codes"].shape[0] * cfg.frame_samples,)


def test_the_depth_decoder_is_the_reference_in_float32(model, ref, stream):
    stream(mx.float32)
    for name, mask in _passes(ref):
        got = _logits(model.depth_decoder, ref, mask, mx.float32)
        want = ref[f"logits_{name}"]
        assert _rel(got[mask], want[mask]) <= F32_REL_RMS, name
        assert (got.argmax(-1) == want.argmax(-1))[mask].all(), name


def test_the_depth_decoder_holds_the_references_16_bit_distance(
        model, ref, stream):
    depth = model.depth_decoder
    stream(depth.proj_in.weight.dtype)
    assert depth.proj_in.weight.dtype in (mx.bfloat16, mx.float16)
    for name, mask in _passes(ref):
        got = _logits(depth, ref, mask, depth.proj_in.weight.dtype)
        want = ref[f"logits_{name}"]
        vendor = _rel(ref[f"logits_{name}_bf16"][mask], want[mask])
        assert _rel(got[mask], want[mask]) <= STREAM_MARGIN * vendor, name


def test_the_feedback_embedding_holds_the_references_16_bit_distance(
        model, ref):
    got = model.depth_decoder.feedback_embedding(
        mx.array(ref["codes"].astype(np.int32)))
    assert got.dtype == model.depth_decoder.feedback.dtype
    vendor = _rel(ref["feedback_bf16"], ref["feedback"])
    assert _rel(got.astype(mx.float32), ref["feedback"]) <= \
        STREAM_MARGIN * vendor


def test_the_codec_is_the_reference_in_float32(model, ref, stream):
    stream(mx.float32)
    audio = model.codec_decoder.decode(ref["codec_codes"])
    assert audio.shape == ref["audio"].shape
    assert _snr_db(audio, ref["audio"]) >= AUDIO_F32_SNR_DB


def test_the_codec_holds_the_references_16_bit_distance(model, ref):
    audio = model.codec_decoder.decode(ref["codec_codes"])
    vendor = _rel(ref["audio_bf16"], ref["audio"])
    assert _rel(audio, ref["audio"]) <= STREAM_MARGIN * vendor


def _streamed(codec, codes) -> list:
    state = codec.new_stream()
    parts = [codec.decode(codes[i:i + 1], state, finalize=False)
             for i in range(codes.shape[0])]
    parts.append(codec.decode(codes[:0], state, finalize=True))
    return parts


def test_a_stream_of_frames_is_the_whole_clips_audio(model, ref, stream):
    codec = model.codec_decoder
    stream(mx.float32)
    parts = _streamed(codec, ref["codec_codes"])
    assert parts[0].shape == (1440,) and parts[1].shape == (1920,)
    assert _rel(np.concatenate(parts),
                codec.decode(ref["codec_codes"])) <= F32_REL_RMS
    # In 16 bits a frame at a time rounds differently from the whole
    # clip, so the stream is held to the recording, as the whole clip is.
    stream(None)
    audio = np.concatenate(_streamed(codec, ref["codec_codes"]))
    assert _rel(audio, ref["audio"]) <= STREAM_MARGIN * _rel(
        ref["audio_bf16"], ref["audio"])


def test_a_float16_stream_does_not_overflow(gguf_dir, gguf_index, ref,
                                            monkeypatch):
    """M1 and M2 run float16, whose range ends at 65,504. Both decoders
    stay inside it and hold the 16-bit distance."""
    from tests.lfm3_header import load_as_f16_projector

    monkeypatch.setenv("GMLX_ACTIVATION_DTYPE", "float16")
    f16 = load_as_f16_projector(*_pair(gguf_dir, gguf_index))[0]
    depth, codec = f16.depth_decoder, f16.codec_decoder
    assert depth.proj_in.weight.dtype == mx.float16
    assert codec.out_proj.weight.dtype == mx.float16
    for name, mask in _passes(ref):
        got = _logits(depth, ref, mask, mx.float16)
        assert np.isfinite(got).all(), name
        want = ref[f"logits_{name}"]
        vendor = _rel(ref[f"logits_{name}_bf16"][mask], want[mask])
        assert _rel(got[mask], want[mask]) <= STREAM_MARGIN * vendor, name
    polar = codec.polar(mx.array(ref["codec_codes"].astype(np.int32)))
    assert np.isfinite(np.asarray(polar)).all()
    audio = codec.decode(ref["codec_codes"])
    assert np.isfinite(audio).all()
    assert _rel(audio, ref["audio"]) <= STREAM_MARGIN * _rel(
        ref["audio_bf16"], ref["audio"])
