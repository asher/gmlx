#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Asher Feldman
"""The LFM3 speech encoder and projector against a float64 reference.

The reference follows llama.cpp's ``lfm3a`` graph in the layout the GGUF
stores: convolutions as ``[out, in / groups, kH, kW]`` over a
``[channels, mel, time]`` plane, one frame of a spectrogram at a time.
The encoder runs the MLX layout and whole clips. The two agree only if
the kernel layout, the channel-major flatten, the causal taps, the
attention window and the projector are all right.
"""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten, tree_unflatten

from gmlx.models.lfm3 import audio
from gmlx.models.lfm3.audio import (CONV, FULL, AudioConfig, AudioEncoder,
                                    AudioFeatureExtractor, AudioProjector)

CONFIG = dict(
    hidden_size=16, intermediate_size=24, num_hidden_layers=4,
    num_attention_heads=4, num_key_value_heads=2, conv_channels=4,
    num_mel_bins=16, sliding_window=4, layer_types=[CONV, FULL, CONV, FULL],
    projection_dim=24, rope_theta=100.0)


def _build(seed: int = 0):
    """An encoder and projector with random float32 weights, and the
    weights as float64 arrays by name."""
    cfg = AudioConfig(**CONFIG)
    encoder, projector = AudioEncoder(cfg), AudioProjector(cfg)
    rng = np.random.default_rng(seed)
    weights = {}
    for prefix, module in (("enc", encoder), ("proj", projector)):
        params = []
        for name, value in tree_flatten(module.parameters()):
            scale = 1.0 if value.ndim == 1 else 1.0 / math.sqrt(
                math.prod(value.shape[1:]))
            arr = rng.standard_normal(value.shape) * scale
            if name.endswith("norm.weight"):
                arr = 1.0 + 0.1 * arr
            arr = arr.astype(np.float32)
            params.append((name, mx.array(arr)))
            weights[f"{prefix}.{name}"] = arr.astype(np.float64)
        module.update(tree_unflatten(params))
    return cfg, encoder, projector, weights


def _rms(x, w, eps):
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps) * w


def _silu(x):
    return x / (1.0 + np.exp(-x))


def _conv2d(x, w, b):
    """``x`` is ``[channels, H, W]``. ``w`` is the MLX-layout kernel,
    read here as ``[out, in / groups, kH, kW]``: 3x3 with stride 2 and
    padding 1, or 1x1."""
    w = w.transpose(0, 3, 1, 2)
    k = w.shape[-1]
    stride, pad = (2, 1) if k == 3 else (1, 0)
    per_group = w.shape[1]
    groups = x.shape[0] // per_group
    per_out = w.shape[0] // groups
    x = np.pad(x, ((0, 0), (pad, pad), (pad, pad)))
    rows = (x.shape[1] - k) // stride + 1
    cols = (x.shape[2] - k) // stride + 1
    out = np.zeros((w.shape[0], rows, cols))
    for o in range(w.shape[0]):
        g = o // per_out
        for r in range(rows):
            for c in range(cols):
                patch = x[g * per_group:(g + 1) * per_group,
                          r * stride:r * stride + k,
                          c * stride:c * stride + k]
                out[o, r, c] = (patch * w[o]).sum()
    return out + b[:, None, None]


def _reference(cfg: AudioConfig, w: dict, mel: np.ndarray) -> np.ndarray:
    """``mel`` is ``[mel bins, frames]``. Returns the projected features,
    ``[tokens, projection_dim]``."""
    d = "enc.downsample."
    x = mel[None].astype(np.float64)
    x = _silu(_conv2d(x, w[d + "conv_in.weight"], w[d + "conv_in.bias"]))
    x = _conv2d(x, w[d + "conv_mid.weight"], w[d + "conv_mid.bias"])
    x = _silu(_conv2d(x, w[d + "lin_mid.weight"], w[d + "lin_mid.bias"]))
    x = _conv2d(x, w[d + "conv_end.weight"], w[d + "conv_end.bias"])
    x = _silu(_conv2d(x, w[d + "lin_end.weight"], w[d + "lin_end.bias"]))
    channels, bins, tokens = x.shape
    # One vector per token: channel c, mel bin m at c * bins + m.
    flat = np.stack([x[:, :, t].reshape(channels * bins)
                     for t in range(tokens)])
    h = flat @ w[d + "lin_out.weight"].T + w[d + "lin_out.bias"]

    n_heads, n_kv = cfg.num_attention_heads, cfg.num_key_value_heads
    head_dim = cfg.hidden_size // n_heads
    inv_freq = 1.0 / (cfg.rope_theta ** (
        np.arange(0, head_dim, 2, dtype=np.float64) / head_dim))

    def rope(v, pos):
        ang = np.concatenate([pos * inv_freq, pos * inv_freq])
        half = head_dim // 2
        turned = np.concatenate([-v[half:], v[:half]])
        return v * np.cos(ang) + turned * np.sin(ang)

    for i, kind in enumerate(cfg.layer_types):
        p = f"enc.layers.{i}."
        x = _rms(h, w[p + "operator_norm.weight"], cfg.norm_eps)
        if kind == CONV:
            gated = x @ w[p + "conv.in_proj.weight"].T
            b, c, v = np.split(gated, 3, axis=-1)
            bx = b * v
            taps = w[p + "conv.kernel"]
            width = taps.shape[1]
            mixed = np.zeros_like(bx)
            for t in range(tokens):
                for k in range(width):
                    src = t - (width - 1 - k)
                    if src >= 0:
                        mixed[t] += bx[src] * taps[:, k]
            h = h + (c * mixed) @ w[p + "conv.out_proj.weight"].T
        else:
            a = p + "self_attn."
            q = (x @ w[a + "q_proj.weight"].T).reshape(
                tokens, n_heads, head_dim)
            k = (x @ w[a + "k_proj.weight"].T).reshape(
                tokens, n_kv, head_dim)
            v = (x @ w[a + "v_proj.weight"].T).reshape(
                tokens, n_kv, head_dim)
            q = _rms(q, w[a + "q_layernorm.weight"], cfg.norm_eps)
            k = _rms(k, w[a + "k_layernorm.weight"], cfg.norm_eps)
            out = np.zeros((tokens, n_heads, head_dim))
            for t in range(tokens):
                first = max(0, t - cfg.sliding_window + 1)
                for head in range(n_heads):
                    kv = head // (n_heads // n_kv)
                    query = rope(q[t, head], t)
                    scores = np.array([
                        query @ rope(k[s, kv], s)
                        for s in range(first, t + 1)]) / math.sqrt(head_dim)
                    probs = np.exp(scores - scores.max())
                    probs /= probs.sum()
                    out[t, head] = probs @ v[first:t + 1, kv]
            h = h + out.reshape(tokens, -1) @ w[a + "out_proj.weight"].T
        x = _rms(h, w[p + "ffn_norm.weight"], cfg.norm_eps)
        f = p + "feed_forward."
        h = h + (_silu(x @ w[f + "gate_proj.weight"].T)
                 * (x @ w[f + "up_proj.weight"].T)) @ w[
                     f + "down_proj.weight"].T
    h = _rms(h, w["enc.embedding_norm.weight"], cfg.norm_eps)

    mean = h.mean(-1, keepdims=True)
    var = ((h - mean) ** 2).mean(-1, keepdims=True)
    h = ((h - mean) / np.sqrt(var + cfg.projector_layer_norm_eps)
         * w["proj.layer_norm.weight"] + w["proj.layer_norm.bias"])
    h = h @ w["proj.linear_1.weight"].T + w["proj.linear_1.bias"]
    h = 0.5 * h * (1.0 + np.vectorize(math.erf)(h / math.sqrt(2.0)))
    return h @ w["proj.linear_2.weight"].T + w["proj.linear_2.bias"]


def _run(encoder, projector, mel: np.ndarray) -> np.ndarray:
    return np.asarray(projector(encoder(mx.array(mel.astype(np.float32)))))


@pytest.mark.parametrize("frames", [8, 21, 1, 40])
def test_a_clip_matches_the_reference(frames):
    """Whole groups of 8 frames, a short last group, and one frame."""
    cfg, encoder, projector, w = _build()
    mel = np.random.default_rng(frames).standard_normal((16, frames))
    got = _run(encoder, projector, mel)
    assert got.shape == (-(frames // -8), 24)
    np.testing.assert_allclose(got, _reference(cfg, w, mel), rtol=2e-4,
                               atol=2e-5)


def test_a_position_reads_only_its_window():
    """A position attends to itself and the three before it, so a change
    reaches the outputs at its own position and the next three."""
    cfg = AudioConfig(**CONFIG)
    attn = audio.AudioAttention(cfg)
    mx.random.seed(2)
    attn.update(tree_unflatten([
        (k, mx.random.normal(v.shape) * 0.3 + (1.0 if "norm" in k else 0.0))
        for k, v in tree_flatten(attn.parameters())]))
    cos, sin = audio._rope_tables(12, 4, 100.0)
    x = mx.random.normal((1, 12, 16))
    moved = np.asarray(x).copy()
    moved[0, 5] += 1.0
    base = np.asarray(attn(x, cos, sin))[0]
    got = np.asarray(attn(mx.array(moved), cos, sin))[0]
    changed = np.abs(got - base).max(-1) > 1e-6
    assert changed.tolist() == [False] * 5 + [True] * 4 + [False] * 3


@pytest.mark.parametrize("stretch, q_block, layer_tokens", [
    (5, 5, 1000), (1000, 1000, 5), (1000, 1000, 1), (3, 2, 7)])
def test_passes_over_a_long_clip_keep_the_output(monkeypatch, stretch,
                                                 q_block, layer_tokens):
    """A clip is downsampled a stretch at a time, attends in query
    blocks, and a long one runs each layer and the projector a span at a
    time. The output is one pass's."""
    cfg, encoder, projector, w = _build()
    mel = np.random.default_rng(3).standard_normal((16, 93))
    monkeypatch.setattr(audio, "DOWNSAMPLE_TOKENS", stretch)
    monkeypatch.setattr(audio, "Q_BLOCK", q_block)
    monkeypatch.setattr(audio, "LAYER_TOKENS", layer_tokens)
    np.testing.assert_allclose(_run(encoder, projector, mel),
                               _reference(cfg, w, mel), rtol=2e-4, atol=2e-5)


def test_the_encoder_keeps_a_16_bit_stream():
    """16-bit weights give 16-bit features: the float32 convolutions,
    norms and rope do not widen the stream. ``stream_dtype`` overrides."""
    _cfg, encoder, projector, _w = _build()
    encoder.set_dtype(mx.bfloat16)
    projector.set_dtype(mx.bfloat16)
    mel = mx.zeros((16, 24))
    assert projector(encoder(mel)).dtype == mx.bfloat16
    encoder.stream_dtype = mx.float32
    assert encoder(mel).dtype == mx.float32


def test_the_encoder_refuses_another_spectrogram_shape():
    _cfg, encoder, _projector, _w = _build()
    with pytest.raises(ValueError, match=r"shape \(16, frames\)"):
        encoder(mx.zeros((1, 16, 24)))
    with pytest.raises(ValueError, match=r"shape \(16, frames\)"):
        encoder(mx.zeros((24, 16)))


# -- features -----------------------------------------------------------------

@pytest.mark.parametrize("samples, padded, frames, tokens", [
    (1, 400, 1, 1),
    (400, 400, 1, 1),
    (401, 560, 2, 1),
    (1680, 1680, 9, 2),
    (8333, 8400, 51, 7),
    (20931, 21040, 130, 17),
    (16000, 16080, 99, 13),
])
def test_frame_and_token_counts(samples, padded, frames, tokens):
    assert audio.pad_length(samples) == padded
    assert audio.num_frames(samples) == frames
    assert audio.num_audio_tokens(samples) == tokens
    assert audio.log_mel(np.zeros(samples, np.float32)).shape == (128, frames)


def test_the_spectrogram_is_the_reference_formula(monkeypatch):
    """Frame by frame in float64: a periodic Hann window, no centering,
    the Slaney bank, log10 floored at -12, then (x + 4) / 4."""
    rng = np.random.default_rng(4)
    wave = (rng.standard_normal(3000) * 0.1).astype(np.float32)
    padded = np.pad(wave.astype(np.float64), (0, 3120 - 3000))
    window = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(400) / 400)
    filters = audio._mel_filters(128).astype(np.float64)
    want = np.empty((128, 18))
    for f in range(18):
        spec = np.fft.rfft(padded[f * 160:f * 160 + 400] * window)
        want[:, f] = np.maximum(
            np.log10(filters @ (np.abs(spec) ** 2)), -12.0)
    want = (want + 4.0) / 4.0
    got = audio.log_mel(wave)
    assert got.dtype == np.float32
    np.testing.assert_allclose(got, want, atol=2e-4)
    # The frame matrix is built a stretch at a time on a long clip.
    monkeypatch.setattr(audio, "_STFT_FRAMES", 5)
    np.testing.assert_array_equal(audio.log_mel(wave), got)


def test_silence_sits_at_the_floor():
    assert np.all(audio.log_mel(np.zeros(1600, np.float32)) == -2.0)


def test_clips_are_padded_to_the_longest_with_their_own_mask():
    extractor = AudioFeatureExtractor(128)
    rng = np.random.default_rng(5)
    clips = [rng.standard_normal(n).astype(np.float32) for n in (1680, 720)]
    out = extractor(clips)
    assert out["input_features"].shape == (2, 128, 9)
    assert out["feature_attention_mask"].tolist() == [[1] * 9,
                                                      [1] * 3 + [0] * 6]
    np.testing.assert_array_equal(out["input_features"][1, :, :3],
                                  audio.log_mel(clips[1]))
    assert not out["input_features"][1, :, 3:].any()
    assert [extractor.tokens(len(c)) for c in clips] == [2, 1]
    assert extractor.sampling_rate == 16000


# -- processing ---------------------------------------------------------------

_SPECIALS = ["<|video|>", "<|image|>", "<|vision_start|>", "<|vision_end|>",
             "<|audio|>"]
_META = {
    "clip.vision.projector_type": "lfm3vl",
    "clip.vision.embedding_length": 32,
    "clip.vision.feed_forward_length": 48,
    "clip.vision.block_count": 1,
    "clip.vision.attention.head_count": 4,
    "clip.vision.attention.head_count_kv": 2,
    "clip.vision.patch_size": 16,
    "clip.vision.spatial_merge_size": 2,
    "clip.vision.projection_dim": 24,
    "clip.vision.image_min_pixels": 65536,
    "clip.vision.image_max_pixels": 2097152,
    "clip.vision.image_mean": [0.5, 0.5, 0.5],
    "clip.vision.image_std": [0.5, 0.5, 0.5],
    "clip.has_audio_encoder": True,
    "clip.audio.projector_type": "lfm3a",
    "clip.audio.embedding_length": 16,
    "clip.audio.feed_forward_length": 24,
    "clip.audio.block_count": 3,
    "clip.audio.attention.head_count": 4,
    "clip.audio.attention.head_count_kv": 2,
    "clip.audio.attention.layer_norm_epsilon": 1e-5,
    "clip.audio.num_mel_bins": 128,
    "clip.audio.window_size": 4,
    "clip.audio.projection_dim": 24,
}
_LLM_META = {"tokenizer.ggml.tokens": ["<unk>"] * 29 + _SPECIALS}


def _tok():
    """A word-level tokenizer with the media tokens at 29 to 33."""
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    vocab = {"[UNK]": 0, "hi": 1,
             **{t: 29 + i for i, t in enumerate(_SPECIALS)}}
    tok = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.add_special_tokens(_SPECIALS)
    return PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="[UNK]",
                                   pad_token="[UNK]")


def _processor(**meta):
    from gmlx.load import vlm

    return vlm._synthesize_lfm3_processor(_tok(), {**_META, **meta},
                                          "lfm3_omni")


def _clip(samples: int) -> np.ndarray:
    return np.zeros(samples, dtype=np.float32)


def test_a_clip_becomes_its_tokens_between_one_pair_of_markers():
    """The template wraps the placeholder in markers. The expansion keeps
    one pair. Clips expand in prompt order."""
    proc = _processor()
    out = proc(audio=[_clip(8333), _clip(1680)],
               text=["<|vision_start|><|audio|><|vision_end|> hi <|audio|>"])
    assert np.array(out["input_ids"])[0].tolist() == (
        [31] + [33] * 7 + [32, 1, 31] + [33] * 2 + [32])
    assert np.array(out["input_features"]).shape == (2, 128, 51)
    assert np.array(out["feature_attention_mask"]).sum(-1).tolist() == [51, 9]
    # One clip may be given bare.
    assert (np.array(proc(audio=_clip(1680), text=["<|audio|>"])[
        "input_ids"]) == 33).sum() == 2


@pytest.mark.parametrize("kwargs, message", [
    (dict(text=["hi"], audio=[_clip(400)]), "Found 0 <|audio|> tokens"),
    (dict(text=["<|audio|> hi"]), "Found 1 <|audio|> tokens"),
    (dict(text=["<|audio|>"], audio=[np.zeros((2, 400), np.float32)]),
     "mono clips"),
    (dict(text=["<|audio|>"], audio=[np.zeros(0, np.float32)]),
     "at least one sample"),
])
def test_the_processor_refuses_what_it_cannot_place(kwargs, message):
    with pytest.raises(ValueError, match=message.replace("|", r"\|")):
        _processor()(**kwargs)


def test_a_model_without_the_encoder_refuses_audio():
    proc = _processor(**{"clip.has_audio_encoder": False})
    assert proc.feature_extractor is None
    with pytest.raises(ValueError, match="no audio encoder"):
        proc(text=["<|audio|>"], audio=[_clip(400)])


# -- a video's soundtrack ------------------------------------------------------

_INDICES = (0, 15, 30, 45, 60, 75, 89)


def _video_meta():
    from gmlx._vendor.mlx_vlm.utils import VideoMetadata

    return VideoMetadata(total_num_frames=90, fps=30.0,
                         frames_indices=list(_INDICES))


def _frames():
    return np.zeros((7, 3, 96, 160), dtype=np.uint8)


def _runs(ids, keep=(29, 33)) -> list:
    """``(token, count)`` for each run of the video (29) and audio (33)
    tokens. Any other token ends a run."""
    out: list = []
    last = None
    for i in (int(i) for i in ids):
        if i in keep and i == last:
            out[-1][1] += 1
        elif i in keep:
            out.append([i, 1])
        last = i
    return [(token, count) for token, count in out]


def test_a_soundtrack_is_cut_at_each_groups_first_frame():
    """The groups of the clip start at 0, 1, 2 and 2.97 s, and a token
    covers 0.08 s."""
    from gmlx.models.lfm3.processing import soundtrack_bounds

    def bounds(tokens):
        return soundtrack_bounds(tokens, _INDICES, 30.0, 2, 4, 0.08)

    assert AudioFeatureExtractor.seconds_per_token == 0.08
    assert bounds(38) == [0, 12, 25, 37, 38]
    # A soundtrack that ends early leaves the later groups without sound.
    assert bounds(11) == [0, 11, 11, 11, 11]
    assert bounds(20) == [0, 12, 20, 20, 20]
    # One that runs past the last frame gives the rest to the last group.
    assert bounds(90) == [0, 12, 25, 37, 90]


def test_each_group_is_followed_by_its_piece_of_the_soundtrack():
    from gmlx.models.lfm3.processing import video_expansion

    group = "<|vision_start|><|video|><|video|><|vision_end|>"
    assert video_expansion(2, [0.25, 1.0], [0, 3, 4]) == (
        "<|vision_start|><0.2 seconds>" + group + "<|audio|>" * 3
        + "<1.0 seconds>" + group + "<|audio|><|vision_end|>")
    assert "<|audio|>" not in video_expansion(2, [0.25, 1.0])


def test_a_video_with_a_soundtrack_holds_its_audio_tokens():
    proc = _processor()
    out = proc(videos=[_frames()], video_metadata=[_video_meta()],
               soundtracks=[_clip(48000)], text=["<|video|> hi"])
    assert _runs(np.array(out["input_ids"])[0]) == [
        (29, 15), (33, 12), (29, 15), (33, 13), (29, 15), (33, 12),
        (29, 15), (33, 1)]
    assert np.array(out["input_features"]).shape == (1, 128, 299)


def test_the_encoder_gets_soundtracks_and_clips_in_prompt_order():
    proc = _processor()
    out = proc(videos=[_frames(), _frames()],
               video_metadata=[_video_meta(), _video_meta()],
               soundtracks=[None, _clip(14000)],
               audio=[_clip(8333), _clip(1680)],
               text=["<|audio|> hi <|video|> hi <|video|> hi <|audio|>"])
    assert _runs(np.array(out["input_ids"])[0]) == [
        (33, 7), (29, 15), (29, 15), (29, 15), (29, 15),
        (29, 15), (33, 11), (29, 15), (29, 15), (29, 15), (33, 2)]
    assert np.array(out["feature_attention_mask"]).sum(-1).tolist() == [
        51, 86, 9]


def test_a_video_without_sound_is_written_as_before():
    proc = _processor()
    kwargs = dict(videos=[_frames()], video_metadata=[_video_meta()],
                  text=["<|video|> hi"])
    plain = np.array(proc(**kwargs)["input_ids"]).tolist()
    for tracks in ([None], [np.zeros(0, np.float32)]):
        out = proc(soundtracks=tracks, **kwargs)
        assert np.array(out["input_ids"]).tolist() == plain
        assert "input_features" not in out


def test_the_processor_refuses_soundtracks_it_cannot_pair():
    proc = _processor()
    with pytest.raises(ValueError, match="2 soundtracks for 1 videos"):
        proc(videos=[_frames()], video_metadata=[_video_meta()],
             soundtracks=[_clip(400), _clip(400)], text=["<|video|>"])
    with pytest.raises(ValueError, match="mono clips"):
        proc(videos=[_frames()], video_metadata=[_video_meta()],
             soundtracks=[np.zeros((2, 400), np.float32)],
             text=["<|video|>"])
    silent = _processor(**{"clip.has_audio_encoder": False})
    with pytest.raises(ValueError, match="no audio encoder"):
        silent(videos=[_frames()], video_metadata=[_video_meta()],
               soundtracks=[_clip(400)], text=["<|video|>"])


def test_the_input_pipeline_keeps_every_clip_at_16_khz():
    """``prepare_inputs`` loads at the extractor's rate and keeps the
    clips after the first only for a processor that says it takes them."""
    from gmlx._vendor.mlx_vlm.utils import prepare_inputs

    out = prepare_inputs(
        _processor(), prompts="<|audio|> hi <|audio|>",
        audio=[_clip(8333), _clip(1680)])
    assert (np.array(out["input_ids"]) == 33).sum() == 7 + 2
    assert tuple(out["input_features"].shape) == (2, 128, 51)


# -- load ---------------------------------------------------------------------

def _mmproj_shapes(kinds=(CONV, FULL, CONV)):
    """An Omni mmproj's audio tensors in GGUF ne order, with the vision
    tensors the config synth needs and one speech generator tensor."""
    shapes = {
        "v.patch_embd.weight": [16, 16, 3, 32],
        "v.patch_embd.weight.1": [16, 16, 3, 32],
        "v.embedding_norm.weight": [32],
        "mm.input_norm.weight": [32],
        "mm.input_norm.bias": [32],
        "mm.0.weight": [128, 40],
        "mm.0.bias": [40],
        "mm.2.weight": [40, 24],
        "mm.2.bias": [24],
        "v.blk.0.ln1.weight": [32], "v.blk.0.ln2.weight": [32],
        "v.blk.0.ffn_gate.weight": [32, 48],
        "v.blk.0.ffn_up.weight": [32, 48],
        "v.blk.0.ffn_down.weight": [48, 32],
        "v.blk.0.conv2d_in_proj.weight": [32, 96],
        "v.blk.0.conv2d_spatial.weight": [3, 3, 1, 32],
        "v.blk.0.conv2d_out_proj.weight": [32, 32],
        "a.post_ln.weight": [16],
        "a.pre_encode.out.weight": [64, 16],
        "a.pre_encode.out.bias": [16],
        "mm.a.mlp.0.weight": [16], "mm.a.mlp.0.bias": [16],
        "mm.a.mlp.1.weight": [16, 24], "mm.a.mlp.1.bias": [24],
        "mm.a.mlp.3.weight": [24, 24], "mm.a.mlp.3.bias": [24],
        "a.gen.code.mask_embd": [1024],
    }
    for index, pointwise in (("0", 1), ("2", 1), ("3", 4), ("5", 1),
                             ("6", 4)):
        k = 1 if pointwise == 4 else 3
        shapes[f"a.conv1d.{index}.weight"] = [k, k, pointwise, 4]
        shapes[f"a.conv1d.{index}.bias"] = [1, 1, 4]
    for i, kind in enumerate(kinds):
        pre = f"a.blk.{i}."
        shapes.update({
            pre + "ln1.weight": [16], pre + "ln2.weight": [16],
            pre + "ffn_gate.weight": [16, 24],
            pre + "ffn_up.weight": [16, 24],
            pre + "ffn_down.weight": [24, 16],
        })
        if kind == CONV:
            shapes.update({
                pre + "conv_pw1.weight": [16, 48],
                pre + "conv_dw.weight": [3, 16],
                pre + "conv_pw2.weight": [16, 16],
            })
        else:
            shapes.update({
                pre + "attn_q.weight": [16, 16],
                pre + "attn_k.weight": [16, 8],
                pre + "attn_v.weight": [16, 8],
                pre + "attn_out.weight": [16, 16],
                pre + "attn_q_norm.weight": [4],
                pre + "attn_k_norm.weight": [4],
            })
    return shapes


def _synth(meta=None, llm=None, shapes=None, width=24):
    from gmlx.load import vlm

    return vlm._synthesize_lfm3_vlm_config(
        {"hidden_size": width}, meta or _META, llm or _LLM_META,
        shapes or _mmproj_shapes(), "lfm3_omni")


def test_config_synth_reads_the_encoder_from_the_header_and_tensors():
    cfg = _synth()
    assert cfg["audio_token_id"] == 33
    assert cfg["audio_config"] == {
        "model_type": "lfm3_audio_encoder", "hidden_size": 16,
        "intermediate_size": 24, "num_hidden_layers": 3,
        "num_attention_heads": 4, "num_key_value_heads": 2,
        "num_mel_bins": 128, "sliding_window": 4,
        "layer_types": [CONV, FULL, CONV], "projection_dim": 24,
        "norm_eps": 1e-5, "conv_L_cache": 3, "conv_channels": 4}
    silent = _synth(meta={**_META, "clip.has_audio_encoder": False})
    assert "audio_config" not in silent and "audio_token_id" not in silent


def test_config_synth_refuses_what_llama_cpp_refuses():
    from gmlx.load import vlm

    with pytest.raises(vlm.UnsupportedVLMError, match="only 'lfm3a'"):
        _synth(meta={**_META, "clip.audio.projector_type": "ultravox"})
    with pytest.raises(ValueError, match="attention window 0"):
        _synth(meta={**_META, "clip.audio.window_size": 0})
    broken = _mmproj_shapes()
    del broken["a.blk.1.attn_k_norm.weight"]
    with pytest.raises(ValueError, match="lfm3a layer 1 is neither"):
        _synth(shapes=broken)
    both = _mmproj_shapes()
    both.update({"a.blk.1.conv_pw1.weight": [16, 48],
                 "a.blk.1.conv_dw.weight": [3, 16],
                 "a.blk.1.conv_pw2.weight": [16, 16]})
    with pytest.raises(ValueError, match="lfm3a layer 1 is neither"):
        _synth(shapes=both)
    headless = _mmproj_shapes()
    del headless["a.post_ln.weight"]
    with pytest.raises(ValueError, match="missing tensor a.post_ln.weight"):
        _synth(shapes=headless)
    with pytest.raises(ValueError, match="audio projection width 40"):
        _synth(meta={**_META, "clip.audio.projection_dim": 40})
    with pytest.raises(ValueError, match=r"no <\|audio\|> token"):
        _synth(llm={"tokenizer.ggml.tokens": ["<unk>"] * 29 + _SPECIALS[:4]})


def test_remap_fills_every_encoder_and_projector_param():
    from gmlx.load import vlm

    shapes = _mmproj_shapes()
    arrays = {k: mx.zeros(tuple(reversed(s)), dtype=mx.float32)
              for k, s in shapes.items()}
    out, skipped, kq = vlm.remap_vision_arrays(arrays, "lfm3_omni",
                                               with_audio=True)
    assert skipped == ["a.gen.code.mask_embd"] and not kq
    config = AudioConfig.from_dict(_synth()["audio_config"])
    want = {f"audio_encoder.{k}": v.shape
            for k, v in tree_flatten(AudioEncoder(config).parameters())}
    want.update({f"audio_projector.{k}": v.shape for k, v in
                 tree_flatten(AudioProjector(config).parameters())})
    got = {k: v.shape for k, v in out.items() if k.startswith("audio_")}
    assert got == want
    # Without the encoder, its tensors are left with the generator's.
    _out, skipped, _kq = vlm.remap_vision_arrays(arrays, "lfm3_omni")
    assert sum(name.startswith(("a.", "mm.a.")) for name in skipped) == \
        sum(name.startswith(("a.", "mm.a.")) for name in shapes)


def test_remap_moves_the_kernels_to_the_mlx_layout():
    """A convolution arrives as [out, in / groups, kH, kW] and its bias
    as [out, 1, 1]. The depthwise taps keep their [channels, width]."""
    from gmlx.load import vlm

    shapes = _mmproj_shapes()
    arrays = {k: mx.zeros(tuple(reversed(s)), dtype=mx.float32)
              for k, s in shapes.items()}
    kernel = np.arange(4 * 9, dtype=np.float32).reshape(4, 1, 3, 3)
    pointwise = np.arange(16, dtype=np.float32).reshape(4, 4, 1, 1)
    taps = np.arange(48, dtype=np.float32).reshape(16, 3)
    arrays["a.conv1d.0.weight"] = mx.array(kernel)
    arrays["a.conv1d.3.weight"] = mx.array(pointwise)
    arrays["a.conv1d.0.bias"] = mx.arange(4).reshape(4, 1, 1)
    arrays["a.blk.0.conv_dw.weight"] = mx.array(taps)
    out, _skipped, _kq = vlm.remap_vision_arrays(arrays, "lfm3_omni",
                                                 with_audio=True)
    d = "audio_encoder.downsample."
    np.testing.assert_array_equal(np.asarray(out[d + "conv_in.weight"]),
                                  kernel.transpose(0, 2, 3, 1))
    np.testing.assert_array_equal(np.asarray(out[d + "lin_mid.weight"]),
                                  pointwise.transpose(0, 2, 3, 1))
    assert np.asarray(out[d + "conv_in.bias"]).tolist() == [0, 1, 2, 3]
    np.testing.assert_array_equal(
        np.asarray(out["audio_encoder.layers.0.conv.kernel"]), taps)


def test_the_float32_pins_cover_the_encoders_convolutions():
    from gmlx.load.loader import fp32_keep_for

    keep = fp32_keep_for("lfm3_omni")
    cfg = AudioConfig(**CONFIG)
    names = [f"audio_encoder.{k}" for k, v in
             tree_flatten(AudioEncoder(cfg).parameters())]
    pinned = {n for n in names if any(n.endswith(k) or k in n for k in keep)}
    assert pinned == {
        "audio_encoder.layers.0.conv.kernel",
        "audio_encoder.layers.2.conv.kernel",
        *(f"audio_encoder.downsample.{c}.{leaf}"
          for c in ("conv_in", "conv_mid", "lin_mid", "conv_end", "lin_end")
          for leaf in ("weight", "bias"))}


# -- the container ------------------------------------------------------------

def _container(with_audio: bool = True):
    from gmlx.models.lfm3.model import CONV as T_CONV
    from gmlx.models.lfm3.model import FULL as T_FULL
    from gmlx.models.lfm3.model import SLIDING, ModelArgs
    from gmlx.models.lfm3.vision import VisionConfig
    from gmlx.models.lfm3.vlm_model import Model, ModelConfig

    text = ModelArgs(
        model_type="lfm3", hidden_size=24, num_hidden_layers=3,
        intermediate_size=32, num_attention_heads=4, num_key_value_heads=2,
        head_dim=6, vocab_size=40, layer_types=[T_CONV, T_FULL, SLIDING],
        sliding_window=8)
    vision = VisionConfig(
        hidden_size=32, intermediate_size=48, num_hidden_layers=1,
        num_attention_heads=4, num_key_value_heads=2, patch_size=4,
        layer_types=["conv"], projector_hidden_size=40, projection_dim=24)
    config = ModelConfig(
        text_config=text, vision_config=vision,
        audio_config=dict(CONFIG) if with_audio else {},  # pyright: ignore[reportArgumentType]
        image_token_id=30, video_token_id=29,
        audio_token_id=33 if with_audio else None, vocab_size=40)
    model = Model(config)
    mx.random.seed(0)
    model.update(tree_unflatten([
        (k, mx.random.normal(v.shape) * 0.1)
        for k, v in tree_flatten(model.parameters())]))
    return model


def _spectrograms(frames=(21, 8)):
    rng = np.random.default_rng(9)
    features = np.zeros((len(frames), 16, max(frames)), dtype=np.float32)
    mask = np.zeros((len(frames), max(frames)), dtype=np.int32)
    for i, n in enumerate(frames):
        features[i, :, :n] = rng.standard_normal((16, n))
        mask[i, :n] = 1
    return features, mask


def test_features_land_on_the_audio_rows_in_clip_order():
    model = _container()
    features, mask = _spectrograms()
    ids = mx.array([[1, 33, 33, 33, 2, 33, 3]])
    got = model.get_input_embeddings(
        ids, input_features=mx.array(features),
        feature_attention_mask=mask).inputs_embeds[0]
    text = model.language_model.model.embed_tokens(ids)[0]
    for row in (0, 4, 6):
        assert mx.array_equal(got[row], text[row])
    first = model.audio_projector(
        model.audio_encoder(mx.array(features[0, :, :21])))
    second = model.audio_projector(
        model.audio_encoder(mx.array(features[1, :, :8])))
    assert first.shape == (3, 24) and second.shape == (1, 24)
    assert mx.allclose(got[1:4], first.astype(got.dtype))
    assert mx.allclose(got[5:6], second.astype(got.dtype))
    logits = model(ids, input_features=mx.array(features),
                   feature_attention_mask=mask).logits
    assert logits.shape == (1, 7, 40)

    with pytest.raises(ValueError, match="2 audio placeholder tokens but 4"):
        model.get_input_embeddings(
            mx.array([[1, 33, 33, 2]]), input_features=mx.array(features),
            feature_attention_mask=mask)
    with pytest.raises(ValueError, match="needs feature_attention_mask"):
        model.get_input_embeddings(ids, input_features=mx.array(features))


def test_a_clip_is_encoded_on_its_own_frames():
    """The padding a longer clip gives a shorter one does not reach its
    features."""
    model = _container()
    features, mask = _spectrograms()
    both = model.audio_features(mx.array(features), mask)
    alone = model.audio_features(mx.array(features[1:, :, :8]), mask[1:, :8])
    assert both.shape == (4, 24)
    assert mx.array_equal(both[3:], alone)


def test_a_model_without_the_encoder_has_no_audio_path():
    model = _container(with_audio=False)
    assert model.config.audio_config is None
    assert not hasattr(model, "audio_encoder")
    assert model.config.text_config.media_token_ids == []
    with pytest.raises(ValueError, match="no audio encoder"):
        model.audio_features(mx.zeros((1, 16, 8)), np.ones((1, 8), np.int32))


def test_the_serve_prefix_cache_counts_the_audio_token_as_media():
    """The serve engine runs the language model alone, so the container
    puts the audio id on the text config with the image and video ids."""
    import importlib
    from types import SimpleNamespace

    from gmlx.cache.apc_media import (CausalMedia, install_causal_serve_media,
                                      install_media_token_ids)

    install_media_token_ids()
    assert install_causal_serve_media()
    ar = importlib.import_module("gmlx._vendor.mlx_vlm.generate.ar")
    model = _container()
    assert model.config.text_config.media_token_ids == [33]
    ids = ar.BatchGenerator._apc_media_token_ids(
        SimpleNamespace(model=model.language_model))
    assert isinstance(ids, CausalMedia) and ids == {29, 30, 33}


# -- LFM3-Audio: the speech encoder alone --------------------------------------

_AUDIO_ONLY_META = {k: v for k, v in _META.items()
                    if not k.startswith("clip.vision.")}


def _audio_only_shapes():
    return {k: v for k, v in _mmproj_shapes().items()
            if k.startswith(("a.", "mm.a."))}


def _audio_only_synth(llm=None, meta=None):
    from gmlx.load import vlm

    return vlm._synthesize_lfm3_vlm_config(
        {"hidden_size": 24}, meta or _AUDIO_ONLY_META, llm or _LLM_META,
        _audio_only_shapes(), "lfm3_audio")


def test_a_projector_with_the_speech_encoder_alone_is_lfm3_audio():
    from gmlx.load import vlm

    assert vlm.resolve_vlm_model_type("lfm3", _AUDIO_ONLY_META) == "lfm3_audio"
    assert vlm.resolve_vlm_model_type("lfm3", _META) == "lfm3_omni"
    with pytest.raises(vlm.UnsupportedVLMError, match="'lfm3a' on LLM arch"):
        vlm.resolve_vlm_model_type("gemma4", _AUDIO_ONLY_META)


def test_the_audio_only_config_has_no_vision_side():
    cfg = _audio_only_synth()
    assert cfg["model_type"] == "lfm3_audio" and cfg["vision_config"] is None
    assert cfg["audio_config"] == _synth()["audio_config"]
    assert (cfg["audio_token_id"], cfg["image_token_id"],
            cfg["video_token_id"]) == (33, 30, 29)
    # A tokenizer without the image and video tokens leaves their ids out.
    bare = _audio_only_synth(llm={"tokenizer.ggml.tokens":
                                  ["<unk>"] * 33 + ["<|audio|>"]})
    assert (bare["audio_token_id"], bare["image_token_id"],
            bare["video_token_id"]) == (33, None, None)
    with pytest.raises(ValueError, match="no audio encoder"):
        _audio_only_synth(meta={**_AUDIO_ONLY_META,
                                "clip.has_audio_encoder": False})


def test_the_audio_only_remap_needs_no_vision_tensor():
    from gmlx.load import vlm

    arrays = {k: mx.zeros(tuple(reversed(s)), dtype=mx.float32)
              for k, s in _audio_only_shapes().items()}
    out, skipped, kq = vlm.remap_vision_arrays(arrays, "lfm3_audio",
                                               with_audio=True)
    assert skipped == ["a.gen.code.mask_embd"] and not kq
    config = AudioConfig.from_dict(_audio_only_synth()["audio_config"])
    want = {f"audio_encoder.{k}": v.shape
            for k, v in tree_flatten(AudioEncoder(config).parameters())}
    want.update({f"audio_projector.{k}": v.shape for k, v in
                 tree_flatten(AudioProjector(config).parameters())})
    assert {k: v.shape for k, v in out.items()} == want


def test_the_audio_only_type_has_its_load_rows():
    from gmlx._vendor.mlx_vlm import prompt_utils
    from gmlx.load.loader import fp32_keep_for
    from gmlx.models import owned

    keep = set(fp32_keep_for("lfm3_audio"))
    assert keep == set(fp32_keep_for("lfm3_omni")) - {".spatial_conv.weight"}
    assert owned.OWNED_VLM_MODELS["lfm3_audio"] == \
        owned.OWNED_VLM_MODELS["lfm3_omni"]
    assert prompt_utils.MODEL_CONFIG["lfm3_audio"] == \
        prompt_utils.MODEL_CONFIG["lfm3_omni"]


def _audio_only_processor():
    from gmlx.load import vlm

    return vlm._synthesize_lfm3_processor(_tok(), _AUDIO_ONLY_META,
                                          "lfm3_audio")


def test_the_audio_only_processor_writes_bare_audio_tokens():
    proc = _audio_only_processor()
    assert proc.image_processor is None and proc.video_processor is None
    out = proc(audio=[_clip(8333), _clip(1680)],
               text=["<|audio|> hi <|audio|>"])
    assert np.array(out["input_ids"])[0].tolist() == (
        [33] * 7 + [1] + [33] * 2)
    assert np.array(out["feature_attention_mask"]).sum(-1).tolist() == [51, 9]
    assert "input_features" in proc.model_input_names


def test_the_audio_only_processor_refuses_images_and_videos():
    from PIL import Image

    proc = _audio_only_processor()
    with pytest.raises(ValueError, match="no vision tower"):
        proc(images=[Image.new("RGB", (64, 64))], text=["<|image|>"])
    with pytest.raises(ValueError, match="no vision tower"):
        proc(videos=[_frames()], video_metadata=[_video_meta()],
             text=["<|video|>"])


def test_the_input_pipeline_runs_the_audio_only_processor():
    from gmlx._vendor.mlx_vlm.utils import prepare_inputs

    out = prepare_inputs(_audio_only_processor(), prompts="<|audio|> hi",
                         audio=[_clip(8333)])
    assert (np.array(out["input_ids"]) == 33).sum() == 7
    assert tuple(out["input_features"].shape) == (1, 128, 51)


def _audio_only_container():
    from gmlx.models.lfm3.model import CONV as T_CONV
    from gmlx.models.lfm3.model import FULL as T_FULL
    from gmlx.models.lfm3.model import SLIDING, ModelArgs
    from gmlx.models.lfm3.vlm_model import Model, ModelConfig

    text = ModelArgs(
        model_type="lfm3", hidden_size=24, num_hidden_layers=3,
        intermediate_size=32, num_attention_heads=4, num_key_value_heads=2,
        head_dim=6, vocab_size=40, layer_types=[T_CONV, T_FULL, SLIDING],
        sliding_window=8)
    config = ModelConfig(
        text_config=text, vision_config=None, model_type="lfm3_audio",
        audio_config=dict(CONFIG),  # pyright: ignore[reportArgumentType]
        image_token_id=30, video_token_id=29, audio_token_id=33,
        vocab_size=40)
    model = Model(config)
    mx.random.seed(0)
    model.update(tree_unflatten([
        (k, mx.random.normal(v.shape) * 0.1)
        for k, v in tree_flatten(model.parameters())]))
    return model


def test_the_audio_only_container_has_no_vision_tower():
    model = _audio_only_container()
    names = [k for k, _ in tree_flatten(model.parameters())]
    assert not any(k.startswith(("vision_tower", "multi_modal_projector"))
                   for k in names)
    assert model.config.vision_config is None
    features, mask = _spectrograms()
    ids = mx.array([[1, 33, 33, 33, 2, 33, 3]])
    logits = model(ids, input_features=mx.array(features),
                   feature_attention_mask=mask).logits
    assert logits.shape == (1, 7, 40)
    assert model.config.text_config.media_token_ids == [33]
    with pytest.raises(ValueError, match="no vision tower"):
        model.get_input_embeddings(
            mx.array([[1, 30, 2]]), mx.zeros((4, 48)),
            image_grid_thw=[(1, 2, 2)])
    with pytest.raises(ValueError, match="no vision tower"):
        model.vision_features(mx.zeros((4, 48)), [(1, 2, 2)])


def test_the_default_config_keeps_its_vision_tower():
    from gmlx.models.lfm3.vlm_model import ModelConfig

    assert ModelConfig().vision_config is not None
    assert ModelConfig(model_type="lfm3_audio").vision_config is None
