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

``scripts/lfm3_oracle/tts.py`` records a sentence the reference spoke:
the prompt, the frames it sampled, and the best first-pass code of each
codebook at every frame. The replay test feeds the same frames through
gmlx's text model and holds the depth decoder to the same codes. The
text GGUF's weights are not the reference's, so the basis is again the
reference's own 16-bit distance: the divergence may be at most
``REPLAY_KL_MARGIN`` times it, and the top-1 agreement at most
``REPLAY_TOP1_SLACK`` under it. A quantized text model has the looser
pair. Both pairs were set before the first comparison.

The last test speaks a sentence and has the same model transcribe it.
It is a check that the whole path works, not a quality floor.
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
TTS_FIXTURE = Path(__file__).parent.parent / "fixtures/lfm3_oracle/tts.npz"

F32_REL_RMS = 1e-3
AUDIO_F32_SNR_DB = 60.0
STREAM_MARGIN = 1.5
REPLAY_KL_MARGIN = {"16-bit": 1.5, "quantized": 3.0}
REPLAY_TOP1_SLACK = {"16-bit": 0.02, "quantized": 0.05}
# GGUF file types whose weights are float: F32, F16, BF16.
FLOAT_FILE_TYPES = (0, 1, 32)
ROUND_TRIP = "The children built a small boat and sailed it across the pond."
ROUND_TRIP_EDITS = 2


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
def loaded(gguf_dir, gguf_index):
    from gmlx.load.vlm import load_vlm_model

    pair = _pair(gguf_dir, gguf_index)
    return (*load_vlm_model(*pair), pair)


@pytest.fixture(scope="module")
def model(loaded):
    return loaded[0]


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


# -- a spoken sentence --------------------------------------------------------


def _log_softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = x - x.max(axis=-1, keepdims=True)
    return x - np.log(np.exp(x).sum(axis=-1, keepdims=True))


def test_a_replayed_sentence_picks_the_references_codes(loaded):
    from gmlx.load.headerscan import scan_gguf

    model, _config, _processor, pair = loaded
    tts = np.load(TTS_FIXTURE)
    file_type = scan_gguf(pair[0], include_tensors=False).kv.get(
        "general.file_type")
    kind = "16-bit" if file_type in FLOAT_FILE_TYPES else "quantized"
    depth, lm = model.depth_decoder, model.language_model
    ids = mx.array(tts["input_ids"][None].astype(np.int32))
    codes = mx.array(tts["codes"].astype(np.int32))
    frames = codes.shape[0]
    prompt = lm.model.embed_tokens(ids)
    spoken = depth.feedback_embedding(codes)[None].astype(prompt.dtype)
    rows = mx.concatenate(
        [ids, mx.zeros((1, frames), dtype=ids.dtype)], axis=1)
    hidden = lm.model(rows, None, mx.concatenate([prompt, spoken], axis=1))
    # The state after the prompt picks frame 0, and so on. The last
    # follows the last frame.
    hidden = hidden[0, ids.shape[1] - 1:]
    assert hidden.shape[0] == frames + 1
    blank = mx.zeros((frames + 1, depth.config.num_codebooks), dtype=mx.int32)
    logits = np.asarray(depth.logits(hidden, blank, blank == 0))

    agreement = float((logits.argmax(-1) == tts["top1"]).mean())
    assert agreement >= float(tts["top1_bf16"].mean()) \
        - REPLAY_TOP1_SLACK[kind], kind
    at = tts["logit_frames"]
    want = _log_softmax(tts["logits"])
    got = _log_softmax(logits[at])
    divergence = float((np.exp(want) * (want - got)).sum(-1).mean())
    assert divergence <= REPLAY_KL_MARGIN[kind] * float(
        tts["kl_bf16"][at].mean()), kind
    # The reference ended the audio after the last frame.
    end = depth.config.eos_code
    assert (tts["top1"][-1] == end).any()
    assert (logits[-1].argmax(-1) == end).any()


def _words(text: str) -> list:
    return "".join(c.lower() if c.isalnum() else " " for c in text).split()


def _edits(a: list, b: list) -> int:
    row = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        prev, row[0] = row[0], i
        for j, y in enumerate(b, 1):
            prev, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1,
                                       prev + (x != y))
    return row[-1]


def test_the_model_transcribes_the_sentence_it_spoke(loaded):
    from gmlx.models.lfm3 import speak
    from gmlx.serve.media_decode import ffmpeg_resample

    model, _config, processor, _pair_ = loaded
    tok = processor.tokenizer
    spoken = speak.generate_speech(
        model, processor, speak.tts_prompt(tok, ROUND_TRIP, "female"),
        max_tokens=300, key=mx.random.key(0))
    assert spoken.summary.finish_reason == "stop"
    assert spoken.text == ""
    cfg = model.config.speech_config
    assert spoken.audio.shape == (spoken.summary.frames * cfg.frame_samples,)
    # About 3 to 6 seconds of speech for 12 words.
    assert 30 <= spoken.summary.frames <= 90

    clip = np.asarray(ffmpeg_resample(spoken.audio, cfg.sample_rate, 16000),
                      dtype=np.float32)
    prompt = tok.apply_chat_template(
        [{"role": "system", "content": "Perform ASR in English."},
         {"role": "user", "content": [{"type": "audio"}]}],
        add_generation_prompt=True, tokenize=False)
    heard = speak.generate_speech(
        model, processor, prompt, audio=[clip], mode="text", max_tokens=64)
    assert heard.summary.frames == 0 and heard.audio.size == 0
    assert _edits(_words(ROUND_TRIP), _words(heard.text)) <= ROUND_TRIP_EDITS, \
        heard.text
