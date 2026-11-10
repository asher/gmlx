#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Asher Feldman
"""The LFM3 speech generator on random weights: the depth decoder's
passes and sampling against a loop written as the reference writes it,
the codec decoder's stream against its whole-clip decode, and the load
path's config and tensor names."""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten, tree_unflatten

from gmlx.models.lfm3.model import CONV, SLIDING
from gmlx.models.lfm3.speech import (CodecDecoder, DepthDecoder,
                                     SpeechConfig, hann_window, overlap_add)

BOOKS, CODES = 12, 37


def _config(**kw) -> SpeechConfig:
    base = dict(
        hidden_size=32, intermediate_size=48, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, projection_dim=24,
        codebook_size=CODES, codec_hidden_size=32,
        codec_intermediate_size=48, codec_num_attention_heads=4,
        codec_num_key_value_heads=2, codec_window=8, quantizer_dim=20,
        n_fft=64, hop_length=16, repetition_window=3)
    base.update(kw)
    return SpeechConfig(**base)


def _random(module):
    mx.random.seed(0)
    module.update(tree_unflatten([
        (k, mx.random.normal(v.shape) * 0.3)
        for k, v in tree_flatten(module.parameters())]))
    return module


@pytest.fixture(scope="module")
def depth():
    return _random(DepthDecoder(_config()))


@pytest.fixture(scope="module")
def codec():
    return _random(CodecDecoder(_config()))


def _hidden(rows=2, seed=1):
    return mx.array(np.random.default_rng(seed).standard_normal(
        (rows, 24)).astype(np.float32))


# -- depth decoder ------------------------------------------------------------


def test_a_pass_scores_every_codebook(depth):
    codes = mx.zeros((2, BOOKS), dtype=mx.int32)
    out = depth.logits(_hidden(), codes, mx.ones((2, BOOKS), dtype=mx.bool_))
    assert out.shape == (2, BOOKS, CODES) and out.dtype == mx.float32


def test_a_masked_code_is_not_read_and_a_known_one_reaches_the_rest(depth):
    rng = np.random.default_rng(2)
    codes = rng.integers(0, CODES - 1, size=(1, BOOKS)).astype(np.int32)
    masked = np.zeros((1, BOOKS), dtype=bool)
    masked[0, [1, 5, 9]] = True
    base = depth.logits(_hidden(1), mx.array(codes), mx.array(masked))
    other = codes.copy()
    other[0, 5] = (other[0, 5] + 7) % (CODES - 1)
    assert mx.array_equal(
        base, depth.logits(_hidden(1), mx.array(other), mx.array(masked)))
    # The attention is bidirectional: a known code moves earlier and
    # later codebooks.
    other = codes.copy()
    other[0, 6] = (other[0, 6] + 7) % (CODES - 1)
    moved = depth.logits(_hidden(1), mx.array(other), mx.array(masked))
    delta = np.abs(np.asarray(moved - base)).max(axis=-1)[0]
    assert delta[1] > 1e-4 and delta[9] > 1e-4


def test_the_feedback_embedding_is_the_sum_of_one_row_per_codebook(depth):
    codes = np.random.default_rng(3).integers(0, CODES, size=(2, BOOKS))
    got = np.asarray(depth.feedback_embedding(mx.array(codes)))
    table = np.asarray(depth.feedback)
    want = sum(table[codes[:, i] + i * CODES] for i in range(BOOKS))
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-5)


def _reference_frame(depth, hidden, order, seen=None):
    """A greedy frame as the reference loops it: each pass scores the
    codebooks still hidden, then reveals its share of the order."""
    cfg = depth.config
    codes = np.zeros((hidden.shape[0], BOOKS), dtype=np.int32)
    revealed = np.zeros_like(codes, dtype=bool)
    start = 0
    for count in cfg.schedule:
        pick = order[:, start:start + count]
        start += count
        scores = np.asarray(depth.logits(
            hidden, mx.array(codes), mx.array(~revealed)))
        for row in range(codes.shape[0]):
            for index in pick[row]:
                line = scores[row, index].copy()
                if seen is not None:
                    hit = seen[index]
                    line[hit] = np.where(
                        line[hit] > 0, line[hit] / cfg.repetition_penalty,
                        line[hit] * cfg.repetition_penalty)
                codes[row, index] = int(line.argmax())
        np.put_along_axis(revealed, pick, True, axis=1)
    return codes


def test_a_greedy_frame_is_the_references_loop(depth):
    rng = np.random.default_rng(4)
    order = np.stack([rng.permutation(BOOKS) for _ in range(2)])
    got = depth.sample_frame(_hidden(), mx.random.key(0), temperature=0.0,
                             order=mx.array(order))
    assert got.dtype == mx.int32
    assert np.asarray(got).tolist() == _reference_frame(
        depth, _hidden(), order).tolist()


def test_the_penalty_covers_the_last_frames_of_each_codebook(depth):
    rng = np.random.default_rng(5)
    history = rng.integers(0, CODES, size=(5, BOOKS))
    seen = np.asarray(depth.penalty_mask(history))
    assert seen.shape == (BOOKS, CODES)
    for book in range(BOOKS):
        # The window is 3 frames.
        assert set(np.flatnonzero(seen[book])) == set(history[-3:, book])
    assert depth.penalty_mask([]) is None and depth.penalty_mask(None) is None
    order = np.stack([rng.permutation(BOOKS) for _ in range(2)])
    got = depth.sample_frame(_hidden(), mx.random.key(0), mx.array(seen),
                             temperature=0.0, order=mx.array(order))
    assert np.asarray(got).tolist() == _reference_frame(
        depth, _hidden(), order, seen).tolist()


def test_no_penalty_means_no_mask():
    off = DepthDecoder(_config(repetition_penalty=1.0))
    assert off.penalty_mask(np.zeros((2, BOOKS), dtype=np.int64)) is None


def test_a_key_fixes_the_frame(depth):
    hidden = _hidden()
    first = depth.sample_frame(hidden, mx.random.key(11))
    assert mx.array_equal(first, depth.sample_frame(hidden, mx.random.key(11)))
    others = [depth.sample_frame(hidden, mx.random.key(k)) for k in range(4)]
    assert any(not mx.array_equal(first, o) for o in others)


def test_top_k_draws_only_the_best_codes(depth):
    hidden = _hidden(1)
    order = mx.arange(BOOKS)[None]
    # The first pass scores with every codebook masked.
    scores = np.asarray(depth.logits(
        hidden, mx.zeros((1, BOOKS), dtype=mx.int32),
        mx.ones((1, BOOKS), dtype=mx.bool_)))[0]
    best = np.argsort(scores, axis=-1)[:, -2:]
    for seed in range(8):
        got = np.asarray(depth.sample_frame(
            hidden, mx.random.key(seed), temperature=5.0, top_k=2,
            order=order))[0]
        assert all(got[i] in best[i] for i in range(4))


def test_a_schedule_must_cover_the_codebooks():
    short = DepthDecoder(_config(schedule=[4, 4]))
    with pytest.raises(ValueError, match="does not cover 12 codebooks"):
        short.sample_frame(_hidden(), mx.random.key(0))


def test_a_frame_builds_as_one_graph(depth):
    """mx.compile raises on any evaluation inside the function it
    traces, so a host read between the passes fails here."""
    hidden = _hidden()
    frame = mx.compile(lambda h, key: depth.sample_frame(h, key))
    first = frame(hidden, mx.random.key(1))
    assert mx.array_equal(first, depth.sample_frame(hidden, mx.random.key(1)))
    # The draws follow the key, so two frames do not repeat one draw.
    assert any(not mx.array_equal(first, frame(hidden, mx.random.key(k)))
               for k in range(2, 6))


# -- codec decoder ------------------------------------------------------------


def _frames(count, seed=0):
    return np.random.default_rng(seed).integers(
        0, CODES - 1, size=(count, BOOKS))


def test_a_frame_is_its_hop_times_the_frame_stack(codec):
    cfg = codec.config
    assert cfg.frame_samples == 96 and cfg.eos_code == CODES - 1
    audio = codec.decode(_frames(5))
    assert audio.shape == (5 * 96,) and audio.dtype == np.float32
    assert np.isfinite(audio).all() and np.abs(audio).max() > 0


@pytest.mark.parametrize("cuts", [(1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
                                  (1, 4, 5), (7,), ()])
def test_a_stream_decodes_to_the_whole_clips_audio(codec, cuts):
    frames = _frames(11)
    whole = codec.decode(frames)
    stream = codec.new_stream()
    parts, start = [], 0
    for cut in (*cuts, 11):
        parts.append(codec.decode(frames[start:cut], stream, finalize=False))
        start = cut
    parts.append(codec.decode(frames[:0], stream, finalize=True))
    np.testing.assert_allclose(np.concatenate(parts), whole, atol=2e-6)
    assert stream.finished


def test_the_first_chunk_holds_back_the_overlap(codec):
    """A frame is 96 samples here, and a decode gives 48 more that the
    next frame adds to. The first chunk drops the 24 samples of the
    window's lead-in and holds back those 48. Each later frame gives a
    whole 96, and the last gives 24 of the held samples too."""
    frames = _frames(3)
    stream = codec.new_stream()
    assert codec.decode(frames[:1], stream, finalize=False).shape == (72,)
    assert codec.decode(frames[1:2], stream, finalize=False).shape == (96,)
    assert codec.decode(frames[2:], stream, finalize=True).shape == (120,)


def test_the_shipped_first_chunk_is_60_ms():
    cfg = SpeechConfig()
    trim = (cfg.n_fft - cfg.hop_length) // 2
    assert cfg.frame_samples == 1920
    assert cfg.frame_samples - trim == 1440 and 2 * trim == 960


def test_the_codec_refuses_what_it_cannot_decode(codec):
    with pytest.raises(ValueError, match="the end code has no audio"):
        codec.decode(np.full((1, BOOKS), CODES - 1))
    stream = codec.new_stream()
    codec.decode(_frames(1), stream)
    with pytest.raises(ValueError, match="finalized"):
        codec.decode(_frames(1), stream)


def test_no_frames_decode_to_no_audio(codec):
    assert codec.decode(_frames(0)).shape == (0,)
    stream = codec.new_stream()
    assert codec.decode(_frames(0), stream, finalize=False).shape == (0,)
    assert not stream.started and not stream.finished
    assert codec.decode(_frames(0), stream, finalize=True).shape == (0,)
    assert stream.finished


def test_overlap_add_sums_frames_a_hop_apart():
    frames = np.random.default_rng(0).standard_normal(
        (5, 64)).astype(np.float32)
    want = np.zeros(4 * 16 + 64, dtype=np.float32)
    for i in range(5):
        want[i * 16:i * 16 + 64] += frames[i]
    np.testing.assert_allclose(overlap_add(frames, 16), want, atol=1e-6)
    with pytest.raises(ValueError, match="not a multiple"):
        overlap_add(frames, 15)


def test_the_window_is_the_periodic_hann():
    window = hann_window(8)
    assert window.dtype == np.float32 and window[0] == 0.0
    np.testing.assert_allclose(window[4], 1.0)
    np.testing.assert_allclose(window[1], window[7])


# -- load ---------------------------------------------------------------------

_META = {
    "clip.has_gen_audio_encoder": True,
    "clip.gen.audio.projector_type": "lfm3a_gen",
    "clip.gen.audio.embedding_length": 32,
    "clip.gen.audio.feed_forward_length": 48,
    "clip.gen.audio.block_count": 2,
    "clip.gen.audio.attention.head_count": 4,
    "clip.gen.audio.attention.head_count_kv": 2,
    "clip.gen.audio.attention.layer_norm_epsilon": 1e-5,
    "clip.gen.audio.projection_dim": 24,
    "clip.gen.audio.sampling.temp": 0.5,
    "clip.gen.audio.sampling.top_k": 16,
    "clip.gen.audio.sampling.penalty_repeat": 1.1,
    "clip.gen.audio.sampling.penalty_last_n": 9,
}
_CODEC_KINDS = (CONV, SLIDING, CONV)


def _shapes():
    """A speech generator's tensors in GGUF ne order."""
    shapes = {
        "a.gen.code.proj_in.weight": [24, BOOKS * 32],
        "a.gen.code.proj_in.bias": [BOOKS * 32],
        "a.gen.code.embd.weight": [32, CODES, BOOKS],
        "a.gen.code.head_norm.weight": [32, BOOKS],
        "a.gen.code.mask_embd": [32],
        "a.gen.code.out_embd.weight": [24, BOOKS * CODES],
        "a.gen.code.output_norm.weight": [32],
        "a.gen.wav.quant.first.codebook.weight": [4, CODES - 1],
        "a.gen.wav.quant.first.out_proj.weight": [48, 20],
        "a.gen.wav.quant.first.out_proj.bias": [20],
        "a.gen.wav.tfm.in_proj.weight": [20, 32],
        "a.gen.wav.tfm.in_proj.bias": [32],
        "a.gen.wav.tfm.out_proj.weight": [32, 66],
        "a.gen.wav.tfm.out_proj.bias": [66],
        "a.gen.wav.tfm.output_norm.weight": [32],
    }

    def block(pre, kind, sandwich):
        shapes.update({
            pre + "ln1.weight": [32], pre + "ln2.weight": [32],
            pre + "ffn_gate.weight": [32, 48],
            pre + "ffn_up.weight": [32, 48],
            pre + "ffn_down.weight": [48, 32]})
        if sandwich:
            shapes.update({pre + "attn_post_norm.weight": [32],
                           pre + "ffn_post_norm.weight": [32]})
        if kind == CONV:
            shapes.update({pre + "conv_pw1.weight": [32, 96],
                           pre + "conv_dw.weight": [7, 32],
                           pre + "conv_pw2.weight": [32, 32]})
        else:
            shapes.update({
                pre + "attn_q.weight": [32, 32],
                pre + "attn_k.weight": [32, 16],
                pre + "attn_v.weight": [32, 16],
                pre + "attn_out.weight": [32, 32],
                pre + "attn_gate.weight": [32, 4],
                pre + "attn_q_norm.weight": [8],
                pre + "attn_k_norm.weight": [8]})

    for i in range(2):
        block(f"a.gen.code.blk.{i}.", SLIDING, sandwich=False)
    for i, kind in enumerate(_CODEC_KINDS):
        block(f"a.gen.wav.tfm.blk.{i}.", kind, sandwich=True)
    return shapes


def _synth(meta=None, shapes=None, width=24):
    from gmlx.load import vlm

    return vlm._synthesize_lfm3_speech_config(
        {"hidden_size": width}, meta or _META, shapes or _shapes())


def test_config_synth_reads_the_generator_from_the_header_and_tensors():
    cfg = SpeechConfig.from_dict(_synth())
    assert (cfg.hidden_size, cfg.intermediate_size, cfg.num_hidden_layers,
            cfg.num_attention_heads, cfg.num_key_value_heads) == (
                32, 48, 2, 4, 2)
    assert (cfg.num_codebooks, cfg.codebook_size, cfg.eos_code) == (
        BOOKS, CODES, CODES - 1)
    assert (cfg.temperature, cfg.top_k, cfg.repetition_penalty,
            cfg.repetition_window) == (0.5, 16, 1.1, 9)
    assert cfg.codec_layer_types == list(_CODEC_KINDS)
    assert (cfg.codec_hidden_size, cfg.codec_intermediate_size,
            cfg.codec_num_attention_heads, cfg.codec_num_key_value_heads,
            cfg.codec_conv_L_cache, cfg.quantizer_dim, cfg.fsq_dim,
            cfg.n_fft) == (32, 48, 4, 2, 7, 20, 4, 64)


def test_config_synth_keeps_the_cards_sampling_without_the_keys():
    meta = {k: v for k, v in _META.items() if ".sampling." not in k}
    cfg = SpeechConfig.from_dict(_synth(meta=meta))
    assert (cfg.temperature, cfg.top_k, cfg.repetition_penalty,
            cfg.repetition_window) == (0.4, 0, 1.2, 64)


def test_config_synth_refuses_another_generator():
    from gmlx.load import vlm

    with pytest.raises(vlm.UnsupportedVLMError, match="only 'lfm3a_gen'"):
        _synth(meta={**_META, "clip.gen.audio.projector_type": "csm"})
    with pytest.raises(ValueError, match="speech projection width 24"):
        _synth(width=40)
    missing = _shapes()
    del missing["a.gen.code.out_embd.weight"]
    with pytest.raises(ValueError, match="missing tensor a.gen.code.out_embd"):
        _synth(shapes=missing)
    fewer = _shapes()
    fewer["a.gen.wav.quant.first.codebook.weight"] = [4, CODES - 2]
    with pytest.raises(ValueError, match="the codec has 35 codes"):
        _synth(shapes=fewer)
    both = _shapes()
    both["a.gen.wav.tfm.blk.0.attn_q.weight"] = [32, 32]
    with pytest.raises(ValueError, match="codec layer 0 is neither"):
        _synth(shapes=both)


def test_remap_fills_every_generator_param():
    from gmlx.load import vlm

    shapes = _shapes()
    arrays = {k: mx.zeros(tuple(reversed(s)), dtype=mx.float32)
              for k, s in shapes.items()}
    arrays["a.gen.flow.input_proj.weight"] = mx.zeros((2, 2))
    out, skipped, kq = vlm.remap_vision_arrays(
        arrays, "lfm3_audio", with_audio=True, mm_meta=_META)
    assert skipped == ["a.gen.flow.input_proj.weight"] and not kq
    config = SpeechConfig.from_dict(_synth())
    want = {f"depth_decoder.{k}": v.shape
            for k, v in tree_flatten(DepthDecoder(config).parameters())}
    want.update({f"codec_decoder.{k}": v.shape for k, v in
                 tree_flatten(CodecDecoder(config).parameters())})
    assert {k: v.shape for k, v in out.items()} == want
    # A projector that does not say it speaks keeps the tensors out.
    _out, skipped, _kq = vlm.remap_vision_arrays(
        arrays, "lfm3_audio", with_audio=True)
    assert len(skipped) == len(arrays)


def test_the_float32_pins_cover_the_code_table_and_the_head_norm():
    from gmlx.load.loader import fp32_keep_for

    config = SpeechConfig.from_dict(_synth())
    names = [f"depth_decoder.{k}" for k, _ in
             tree_flatten(DepthDecoder(config).parameters())]
    names += [f"codec_decoder.{k}" for k, _ in
              tree_flatten(CodecDecoder(config).parameters())]
    for model_type in ("lfm3_omni", "lfm3_audio"):
        keep = fp32_keep_for(model_type)
        pinned = {n for n in names if any(n.endswith(k) or k in n
                                          for k in keep)}
        assert pinned == {
            "depth_decoder.head_norm", "codec_decoder.codebook",
            "codec_decoder.layers.0.conv.kernel",
            "codec_decoder.layers.2.conv.kernel"}


def test_a_model_that_speaks_holds_both_decoders(monkeypatch):
    from gmlx.models.lfm3.model import CONV as T_CONV
    from gmlx.models.lfm3.model import FULL as T_FULL
    from gmlx.models.lfm3.model import ModelArgs
    from gmlx.models.lfm3.vlm_model import Model, ModelConfig

    def build(speech):
        text = ModelArgs(
            model_type="lfm3", hidden_size=24, num_hidden_layers=2,
            intermediate_size=32, num_attention_heads=4,
            num_key_value_heads=2, head_dim=6, vocab_size=40,
            layer_types=[T_CONV, T_FULL])
        return Model(ModelConfig(
            text_config=text, vision_config=None, model_type="lfm3_audio",
            speech_config=speech,  # pyright: ignore[reportArgumentType]
            audio_token_id=33, vocab_size=40))

    model = build(_synth())
    assert isinstance(model.depth_decoder, DepthDecoder)
    assert isinstance(model.codec_decoder, CodecDecoder)
    assert model.config.speech_config.codebook_size == CODES
    assert model.depth_decoder.stream_dtype is None
    silent = build({})
    assert silent.config.speech_config is None
    assert not hasattr(silent, "depth_decoder")
    # The debug switch covers the decoders too.
    from gmlx.models.lfm3.vlm_model import TOWER_FP32_ENV

    monkeypatch.setenv(TOWER_FP32_ENV, "1")
    wide = build(_synth())
    assert wide.depth_decoder.stream_dtype == mx.float32
    assert wide.codec_decoder.stream_dtype == mx.float32
