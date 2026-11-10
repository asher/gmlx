#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Asher Feldman
"""The LFM3 vision tower and projector against a float64 reference.

The reference follows llama.cpp's ``lfm3vl`` graph: patches in raster
order, one patch kernel per temporal slice, and the 2x2 join before the
projector. The tower takes patches grouped by merge block, as the
processor writes them. The two layouts agree only if the patch layout,
the position ids, the convolution's reordering and the projector's join
are all right.
"""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten, tree_unflatten

from gmlx.models.lfm3 import vision
from gmlx.models.lfm3.processing import patchify
from gmlx.models.lfm3.vision import (CONV, FULL, MultiModalProjector,
                                     VisionConfig, VisionModel)

PATCH = 4
CONFIG = dict(
    hidden_size=32, intermediate_size=48, num_hidden_layers=4,
    num_attention_heads=4, num_key_value_heads=2, patch_size=PATCH,
    layer_types=[CONV, FULL, CONV, FULL], projector_hidden_size=40,
    projection_dim=24, rope_theta=100.0)


def _build(seed: int = 0):
    """A tower and projector with random float32 weights, and the weights
    as float64 arrays by name."""
    cfg = VisionConfig(**CONFIG)
    tower, projector = VisionModel(cfg), MultiModalProjector(cfg)
    rng = np.random.default_rng(seed)
    weights = {}
    for prefix, module in (("tower", tower), ("proj", projector)):
        params = []
        for name, value in tree_flatten(module.parameters()):
            scale = 1.0 if value.ndim == 1 else 1.0 / math.sqrt(value.shape[-1])
            arr = rng.standard_normal(value.shape) * scale
            if name.endswith("norm.weight") or name.endswith("layernorm.weight"):
                arr = 1.0 + 0.1 * arr
            arr = arr.astype(np.float32)
            params.append((name, mx.array(arr)))
            weights[f"{prefix}.{name}"] = arr.astype(np.float64)
        module.update(tree_unflatten(params))
    return cfg, tower, projector, weights


def _rms(x, w, eps):
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps) * w


def _reference(cfg: VisionConfig, w: dict, frames: np.ndarray) -> np.ndarray:
    """``frames`` is ``[2, C, H, W]``: the two temporal slices of one
    grid. Returns the projected tokens, ``[gh/2 * gw/2, projection_dim]``,
    block by block in raster order."""
    p, d = cfg.patch_size, cfg.hidden_size
    _, channels, height, width = frames.shape
    gh, gw = height // p, width // p
    n = gh * gw
    heads, kv = cfg.num_attention_heads, cfg.num_key_value_heads
    hd = d // heads

    # One kernel per temporal slice, summed.
    kernels = w["tower.patchifier.weight"].reshape(d, channels, 2, p, p)
    x = np.zeros((gh, gw, d))
    for t in range(2):
        cut = frames[t].reshape(channels, gh, p, gw, p).transpose(1, 3, 0, 2, 4)
        x += np.einsum("yxcij,dcij->yxd", cut, kernels[:, :, t])
    x = _rms(x.reshape(n, d), w["tower.embedding_norm.weight"], cfg.norm_eps)

    # Axial rope in raster order: pairs (j, j + hd/2), the first quarter
    # by row, the next by column.
    spatial = hd // 2
    inv = 1.0 / cfg.rope_theta ** (np.arange(0, spatial, 2) / spatial)
    rows, cols = np.divmod(np.arange(n), gw)
    angle = np.concatenate([rows[:, None] * inv, cols[:, None] * inv], axis=-1)
    cos, sin = np.cos(angle), np.sin(angle)

    def rope(t):                      # [n, heads, hd]
        a, b = t[..., :hd // 2], t[..., hd // 2:]
        c, s = cos[:, None, :], sin[:, None, :]
        return np.concatenate([a * c - b * s, b * c + a * s], axis=-1)

    for i, kind in enumerate(cfg.layer_types):
        pre = f"tower.layers.{i}."
        h = _rms(x, w[pre + "operator_norm.weight"], cfg.norm_eps)
        if kind == CONV:
            proj = h @ w[pre + "conv.in_proj.weight"].T
            first, middle, third = np.split(proj, 3, axis=-1)
            gated = np.pad((first * third).reshape(gh, gw, d),
                           [(1, 1), (1, 1), (0, 0)])
            taps = w[pre + "conv.spatial_conv.weight"][..., 0]   # [d, 3, 3]
            windows = np.lib.stride_tricks.sliding_window_view(
                gated, (3, 3), axis=(0, 1))                      # [gh, gw, d, 3, 3]
            conv = np.einsum("yxdij,dij->yxd", windows, taps).reshape(n, d)
            op = (middle * conv) @ w[pre + "conv.out_proj.weight"].T
        else:
            a = pre + "self_attn."
            q = (h @ w[a + "q_proj.weight"].T).reshape(n, heads, hd)
            k = (h @ w[a + "k_proj.weight"].T).reshape(n, kv, hd)
            v = (h @ w[a + "v_proj.weight"].T).reshape(n, kv, hd)
            q = rope(_rms(q, w[a + "q_layernorm.weight"], cfg.norm_eps))
            k = rope(_rms(k, w[a + "k_layernorm.weight"], cfg.norm_eps))
            k = np.repeat(k, heads // kv, axis=1)
            v = np.repeat(v, heads // kv, axis=1)
            scores = np.einsum("qhd,khd->hqk", q, k) / math.sqrt(hd)
            scores = np.exp(scores - scores.max(-1, keepdims=True))
            scores /= scores.sum(-1, keepdims=True)
            out = np.einsum("hqk,khd->qhd", scores, v).reshape(n, d)
            op = out @ w[a + "out_proj.weight"].T
        x = x + op
        h = _rms(x, w[pre + "ffn_norm.weight"], cfg.norm_eps)
        f = pre + "feed_forward."
        gate = h @ w[f + "w1.weight"].T
        gate = gate / (1.0 + np.exp(-gate))
        x = x + (gate * (h @ w[f + "w3.weight"].T)) @ w[f + "w2.weight"].T

    mean = x.mean(-1, keepdims=True)
    var = ((x - mean) ** 2).mean(-1, keepdims=True)
    x = ((x - mean) / np.sqrt(var + cfg.projector_layer_norm_eps)
         * w["proj.layer_norm.weight"] + w["proj.layer_norm.bias"])
    # Join each 2x2 block: (0,0), (0,1), (1,0), (1,1).
    x = x.reshape(gh // 2, 2, gw // 2, 2, d).transpose(0, 2, 1, 3, 4)
    x = x.reshape(n // 4, 4 * d)
    x = x @ w["proj.linear_1.weight"].T + w["proj.linear_1.bias"]
    x = 0.5 * x * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))
    return x @ w["proj.linear_2.weight"].T + w["proj.linear_2.bias"]


def _run(tower, projector, frames: np.ndarray) -> np.ndarray:
    patches, grid = patchify(frames.astype(np.float32), PATCH, 2, 2)
    hidden = tower(mx.array(patches), grid)
    return np.asarray(projector(hidden).astype(mx.float32))


@pytest.mark.parametrize("height, width", [(16, 24), (24, 8), (8, 8)])
def test_a_still_image_matches_the_reference(height, width):
    cfg, tower, projector, w = _build()
    image = np.random.default_rng(1).standard_normal((1, 3, height, width))
    got = _run(tower, projector, image)
    want = _reference(cfg, w, np.repeat(image, 2, axis=0))
    assert got.shape == want.shape == (height * width // (PATCH * 2) ** 2, 24)
    np.testing.assert_allclose(got, want, rtol=2e-3, atol=2e-4)


def test_a_frame_pair_matches_the_reference():
    """The two frames of a temporal slice go to the two patch kernels."""
    cfg, tower, projector, w = _build()
    pair = np.random.default_rng(2).standard_normal((2, 3, 16, 16))
    got = _run(tower, projector, pair)
    np.testing.assert_allclose(got, _reference(cfg, w, pair),
                               rtol=2e-3, atol=2e-4)
    swapped = _reference(cfg, w, pair[::-1])
    assert np.abs(got - swapped).max() > 0.05


def test_the_slices_of_a_clip_do_not_see_each_other():
    """A clip of three pairs gives each pair the tokens it gets alone."""
    _cfg, tower, projector, _w = _build()
    clip = np.random.default_rng(3).standard_normal((6, 3, 8, 16))
    got = _run(tower, projector, clip)
    alone = np.concatenate(
        [_run(tower, projector, clip[i:i + 2]) for i in (0, 2, 4)], axis=0)
    np.testing.assert_allclose(got, alone, rtol=1e-4, atol=1e-5)


def test_an_odd_clip_repeats_its_last_frame():
    _cfg, tower, projector, _w = _build()
    clip = np.random.default_rng(4).standard_normal((3, 3, 8, 8))
    padded = np.concatenate([clip, clip[-1:]], axis=0)
    np.testing.assert_array_equal(_run(tower, projector, clip),
                                  _run(tower, projector, padded))


def test_query_blocks_and_layer_evaluation_keep_the_output(monkeypatch):
    """A large image attends in query blocks and evaluates each layer."""
    _cfg, tower, projector, _w = _build()
    image = np.random.default_rng(5).standard_normal((1, 3, 16, 24))
    want = _run(tower, projector, image)
    monkeypatch.setattr(vision, "Q_BLOCK", 5)
    monkeypatch.setattr(vision, "EVAL_PATCHES", 1)
    np.testing.assert_allclose(_run(tower, projector, image), want,
                               rtol=1e-4, atol=1e-5)


def test_the_tower_keeps_a_16_bit_stream():
    """16-bit weights give a 16-bit hidden state: the float32 norms, rope
    and convolution sum do not widen it."""
    _cfg, tower, _projector, _w = _build()
    tower.set_dtype(mx.bfloat16)
    patches, grid = patchify(
        np.zeros((1, 3, 8, 8), dtype=np.float32), PATCH, 2, 2)
    assert tower(mx.array(patches), grid).dtype == mx.bfloat16


# -- processing -------------------------------------------------------------

@pytest.mark.parametrize("size, want", [
    ((256, 256), (256, 256)),
    ((517, 803), (512, 800)),
    ((448, 640), (448, 640)),
    ((40, 60), (224, 320)),
    ((3000, 4000), (1248, 1664)),
    ((100, 1900), (96, 1888)),
    ((2160, 3840), (1056, 1920)),
    ((33, 33), (256, 256)),
])
def test_image_sizes_are_the_reference_processors(size, want):
    """Recorded from the reference ``smart_resize`` with Omni's bounds."""
    from gmlx.models.lfm3.processing import smart_resize

    assert smart_resize(*size, 32, 65536, 2097152) == want


@pytest.mark.parametrize("clip, want", [
    ((7, 96, 160), (96, 160)),
    ((4, 720, 1280), (704, 1280)),
    ((384, 1080, 1920), (192, 320)),
    ((120, 480, 640), (384, 512)),
    ((2, 20, 20), (64, 64)),
    ((768, 2160, 3840), (128, 224)),
])
def test_frame_sizes_are_the_reference_processors(clip, want):
    """The pixel budget counts the whole clip, so a long clip gets small
    frames."""
    from gmlx.models.lfm3.processing import (VIDEO_MAX_PIXELS,
                                             VIDEO_MIN_PIXELS,
                                             video_smart_resize)

    assert video_smart_resize(*clip, 2, 32, VIDEO_MIN_PIXELS,
                              VIDEO_MAX_PIXELS) == want


@pytest.mark.parametrize("total, fps, count, head, tail", [
    (90, 30.0, 6, [0, 18, 36, 53], [71, 89]),
    (3, 30.0, 3, [0, 1, 2], [1, 2]),
    (9000, 29.97, 384, [0, 23, 47, 70], [8976, 8999]),
    (60000, 30.0, 384, [0, 157, 313, 470], [59842, 59999]),
    (240, 24.0, 20, [0, 13, 25, 38], [226, 239]),
    (7, 2.0, 7, [0, 1, 2, 3], [5, 6]),
])
def test_frames_are_sampled_as_the_reference_samples(total, fps, count, head,
                                                     tail):
    """Two frames per second, at least 4, at most 384 on Omni, and never
    more than the clip has."""
    from gmlx.models.lfm3.processing import sample_frame_indices

    got = sample_frame_indices(total, fps, 2.0, 4, 384)
    assert len(got) == count
    assert got[:4].tolist() == head and got[-2:].tolist() == tail


def test_a_slice_is_stamped_at_the_mean_time_of_its_frames():
    from gmlx.models.lfm3.processing import frame_timestamps

    assert frame_timestamps([0, 15, 30, 45, 60, 75, 89], 30.0, 2) == \
        pytest.approx([0.25, 1.25, 2.25, 89 / 30])


def test_a_patch_is_flattened_channel_temporal_then_pixels():
    """Patches come out block by block. Each holds its channels, and each
    channel its two frames."""
    frames = np.zeros((2, 3, 8, 8), dtype=np.float32)
    for t in range(2):
        for c in range(3):
            # Value = frame, channel, row, column.
            frames[t, c] = (1000 * t + 100 * c
                            + 10 * np.arange(8)[:, None] + np.arange(8))
    flat, grid = patchify(frames, 2, 2, 2)
    assert grid == (1, 4, 4) and flat.shape == (16, 3 * 2 * 2 * 2)
    # The first block holds patches (0,0), (0,1), (1,0), (1,1).
    corners = [flat[i].reshape(3, 2, 2, 2)[0, 0, 0, 0] for i in range(6)]
    assert corners == [0, 2, 20, 22, 4, 6]
    one = flat[1].reshape(3, 2, 2, 2)
    assert one[2, 1, 1, 0] == 1000 + 200 + 10 + 2


_SPECIALS = ["<|video|>", "<|image|>", "<|vision_start|>", "<|vision_end|>"]
_META = {
    "clip.vision.projector_type": "lfm3vl",
    "clip.vision.embedding_length": 32,
    "clip.vision.feed_forward_length": 48,
    "clip.vision.block_count": 3,
    "clip.vision.attention.head_count": 4,
    "clip.vision.attention.head_count_kv": 2,
    "clip.vision.attention.layer_norm_epsilon": 1e-5,
    "clip.vision.patch_size": 16,
    "clip.vision.spatial_merge_size": 2,
    "clip.vision.projection_dim": 24,
    "clip.vision.image_min_pixels": 65536,
    "clip.vision.image_max_pixels": 2097152,
    "clip.vision.image_mean": [0.5, 0.5, 0.5],
    "clip.vision.image_std": [0.5, 0.5, 0.5],
}
_LLM_META = {"tokenizer.ggml.tokens": ["<unk>"] * 29 + ["<|video|>",
                                                       "<|image|>"]}


def _tok():
    """A word-level tokenizer with the media tokens at 29 to 32."""
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    vocab = {"[UNK]": 0, "hi": 1,
             **{t: 29 + i for i, t in enumerate(_SPECIALS)}}
    tok = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.add_special_tokens(_SPECIALS)
    return PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="[UNK]")


def _processor(model_type="lfm3_omni", **meta):
    from gmlx.load import vlm

    return vlm._synthesize_lfm3_processor(_tok(), {**_META, **meta},
                                          model_type)


def _clip_meta(indices=(0, 15, 30, 45, 60, 75, 89), fps=30.0):
    from gmlx._vendor.mlx_vlm.utils import VideoMetadata

    return VideoMetadata(total_num_frames=90, fps=fps,
                         frames_indices=list(indices))


def test_an_image_becomes_its_soft_tokens_between_one_pair_of_markers():
    """The template wraps the placeholder in markers. The expansion keeps
    one pair."""
    from PIL import Image

    proc = _processor()
    out = proc(images=[Image.new("RGB", (640, 448))],
               text=["<|vision_start|><|image|><|vision_end|> hi"])
    assert np.array(out["input_ids"])[0].tolist() == \
        [31] + [30] * 280 + [32, 1]
    assert np.array(out["pixel_values"]).shape == (28 * 40, 1536)
    assert np.array(out["image_grid_thw"]).tolist() == [[1, 28, 40]]


def test_a_video_becomes_one_stamped_group_per_frame_pair():
    proc = _processor()
    grid = np.array([4, 6, 10])
    group = "<|vision_start|>" + "<|video|>" * 15 + "<|vision_end|>"
    assert proc._video_text(grid, _clip_meta()) == (
        "<|vision_start|><0.2 seconds>" + group + "<1.2 seconds>" + group
        + "<2.2 seconds>" + group + "<3.0 seconds>" + group
        + "<|vision_end|>")
    out = proc(videos=[np.zeros((7, 3, 96, 160), dtype=np.uint8)],
               text=["<|vision_start|><|video|><|vision_end|> hi"],
               video_metadata=[_clip_meta()])
    assert np.array(out["pixel_values_videos"]).shape == (4 * 60, 1536)
    assert np.array(out["video_grid_thw"]).tolist() == [[4, 6, 10]]
    assert (np.array(out["input_ids"]) == 29).sum() == 60


def test_the_input_pipeline_hands_the_processor_the_frame_times():
    """``process_inputs`` passes an argument only when the processor's
    signature names it."""
    from gmlx._vendor.mlx_vlm.utils import process_inputs

    out = process_inputs(
        _processor(), prompts=["<|video|> hi"],
        videos=[np.zeros((4, 3, 64, 64), dtype=np.uint8)], fps=[2.0],
        padding=False,
        video_metadata=[_clip_meta(indices=(0, 1, 2, 3))])
    assert (np.array(out["input_ids"]) == 29).sum() == 2 * 4


def test_images_and_videos_expand_in_prompt_order():
    from PIL import Image

    proc = _processor()
    out = proc(images=[Image.new("RGB", (256, 256)),
                       Image.new("RGB", (512, 256))],
               videos=[np.zeros((4, 3, 64, 64), dtype=np.uint8)],
               text=["<|image|> hi <|video|> hi <|image|>"],
               video_metadata=[_clip_meta(indices=(0, 1, 2, 3))])
    ids = np.array(out["input_ids"])[0]
    runs = [int(i) for i in ids[np.r_[True, ids[1:] != ids[:-1]]]
            if i in (29, 30)]
    assert runs == [30, 29, 29, 30]
    assert (ids == 30).sum() == 64 + 128 and (ids == 29).sum() == 2 * 4


@pytest.mark.parametrize("kwargs, message", [
    (dict(text=["hi"], images=["x"]), "Found 0 <|image|> tokens"),
    (dict(text=["<|image|> hi"]), "Found 1 <|image|> tokens"),
    (dict(text=["<|video|>"], videos=[np.zeros((4, 3, 64, 64), np.uint8)]),
     "frame timestamps"),
    (dict(text=["hi"], audio=["clip.wav"]), "audio input is not supported"),
])
def test_the_processor_refuses_what_it_cannot_place(kwargs, message):
    with pytest.raises(ValueError, match=message.replace("|", r"\|")):
        _processor()(**kwargs)


def test_an_image_is_capped_at_4096_tokens_and_frames_by_model_type():
    """VL's projector allows 16.7 MP. An image stays under 4,096 tokens,
    as llama.cpp caps it."""
    from gmlx._vendor.mlx_vlm.utils import processor_video_sampling

    omni = _processor()
    vl = _processor("lfm3_vl", **{"clip.vision.image_max_pixels": 16777216})
    assert omni.image_processor.max_pixels == 2097152
    assert vl.image_processor.max_pixels == 4096 * 32 * 32
    assert vl.image_processor.target_size(4000, 6000) == (1664, 2496)
    for proc, frames in ((omni, 384), (vl, 768)):
        sampling = processor_video_sampling(proc)
        assert (sampling.fps, sampling.min_frames, sampling.max_frames) == \
            (2.0, 4, frames)
        assert proc.video_processor.sample_frames_in_loader


# -- load -------------------------------------------------------------------

def test_resolver():
    from gmlx.load import vlm

    meta = {"clip.vision.projector_type": "lfm3vl"}
    assert vlm.resolve_vlm_model_type("lfm3", meta) == "lfm3_vl"
    assert vlm.resolve_vlm_model_type(
        "lfm3", {**meta, "clip.has_audio_encoder": True}) == "lfm3_omni"
    with pytest.raises(vlm.UnsupportedVLMError, match="lfm3vl"):
        vlm.resolve_vlm_model_type("llama", meta)


def _mmproj_shapes(kinds=(CONV, FULL, CONV)):
    """An lfm3vl mmproj's tensors, in GGUF ne order, with a few of the
    audio tensors an Omni file also holds."""
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
        "a.post_ln.weight": [768],
        "mm.a.mlp.0.weight": [768],
        "a.gen.code.mask_embd": [1024],
    }
    for i, kind in enumerate(kinds):
        pre = f"v.blk.{i}."
        shapes.update({
            pre + "ln1.weight": [32], pre + "ln2.weight": [32],
            pre + "ffn_gate.weight": [32, 48],
            pre + "ffn_up.weight": [32, 48],
            pre + "ffn_down.weight": [48, 32],
        })
        if kind == CONV:
            shapes.update({
                pre + "conv2d_in_proj.weight": [32, 96],
                pre + "conv2d_spatial.weight": [3, 3, 1, 32],
                pre + "conv2d_out_proj.weight": [32, 32],
            })
        else:
            shapes.update({
                pre + "attn_q.weight": [32, 32],
                pre + "attn_k.weight": [32, 16],
                pre + "attn_v.weight": [32, 16],
                pre + "attn_out.weight": [32, 32],
                pre + "attn_q_norm.weight": [8],
                pre + "attn_k_norm.weight": [8],
            })
    return shapes


def test_config_synth_reads_the_layer_kinds_from_the_tensors():
    from gmlx.load import vlm

    cfg = vlm._synthesize_lfm3_vlm_config(
        {"hidden_size": 24}, _META, _LLM_META, _mmproj_shapes(), "lfm3_omni")
    vision = cfg["vision_config"]
    assert vision["layer_types"] == [CONV, FULL, CONV]
    assert (vision["conv_kernel_size"], vision["projector_hidden_size"]) == \
        (3, 40)
    assert (vision["num_attention_heads"],
            vision["num_key_value_heads"]) == (4, 2)
    assert (cfg["model_type"], cfg["video_token_id"],
            cfg["image_token_id"]) == ("lfm3_omni", 29, 30)


def test_config_synth_refuses_what_llama_cpp_refuses():
    from gmlx.load import vlm

    def synth(meta=_META, llm=_LLM_META, shapes=None, width=24):
        return vlm._synthesize_lfm3_vlm_config(
            {"hidden_size": width}, meta, llm, shapes or _mmproj_shapes(),
            "lfm3_vl")

    with pytest.raises(ValueError, match="spatial_merge_size 3"):
        synth(meta={**_META, "clip.vision.spatial_merge_size": 3})
    broken = _mmproj_shapes()
    del broken["v.blk.1.attn_out.weight"]
    with pytest.raises(ValueError, match="layer 1 is neither"):
        synth(shapes=broken)
    both = _mmproj_shapes()
    both["v.blk.1.conv2d_in_proj.weight"] = [32, 96]
    both["v.blk.1.conv2d_spatial.weight"] = [3, 3, 1, 32]
    both["v.blk.1.conv2d_out_proj.weight"] = [32, 32]
    with pytest.raises(ValueError, match="layer 1 is neither"):
        synth(shapes=both)
    with pytest.raises(ValueError, match="does not match the lfm3 hidden"):
        synth(width=2048)
    with pytest.raises(ValueError, match=r"no <\|image\|> token"):
        synth(llm={"tokenizer.ggml.tokens": ["<unk>"] * 29 + ["<|video|>"]})


def test_remap_fills_every_tower_and_projector_param():
    from gmlx.load import vlm

    shapes = _mmproj_shapes()
    arrays = {k: mx.zeros(tuple(reversed(s)), dtype=mx.float32)
              for k, s in shapes.items()}
    out, skipped, kq = vlm.remap_vision_arrays(arrays, "lfm3_omni")
    assert sorted(skipped) == ["a.gen.code.mask_embd", "a.post_ln.weight",
                               "mm.a.mlp.0.weight"] and not kq
    cfg = vlm._synthesize_lfm3_vlm_config(
        {"hidden_size": 24}, _META, _LLM_META, shapes, "lfm3_omni")
    vision = VisionConfig.from_dict(cfg["vision_config"])
    want = {f"vision_tower.{k}": v.shape
            for k, v in tree_flatten(VisionModel(vision).parameters())}
    want.update({f"multi_modal_projector.{k}": v.shape for k, v in
                 tree_flatten(MultiModalProjector(vision).parameters())})
    assert {k: v.shape for k, v in out.items()} == want


def test_remap_lays_the_kernels_out_as_the_processor_flattens_patches():
    """The first kernel reads the first frame of a pair. The depthwise
    taps move to the MLX layout."""
    from gmlx.load import vlm

    shapes = _mmproj_shapes()
    arrays = {k: mx.zeros(tuple(reversed(s)), dtype=mx.float32)
              for k, s in shapes.items()}
    first = np.arange(32 * 3 * 16 * 16, dtype=np.float32).reshape(32, 3, 16, 16)
    arrays["v.patch_embd.weight"] = mx.array(first)
    arrays["v.patch_embd.weight.1"] = mx.array(-first)
    taps = np.arange(32 * 9, dtype=np.float32).reshape(32, 1, 3, 3)
    arrays["v.blk.0.conv2d_spatial.weight"] = mx.array(taps)
    out, _skipped, _kq = vlm.remap_vision_arrays(arrays, "lfm3_vl")
    weight = np.asarray(out["vision_tower.patchifier.weight"]).reshape(
        32, 3, 2, 16, 16)
    np.testing.assert_array_equal(weight[:, :, 0], first)
    np.testing.assert_array_equal(weight[:, :, 1], -first)
    np.testing.assert_array_equal(
        np.asarray(out["vision_tower.layers.0.conv.spatial_conv.weight"]),
        taps.transpose(0, 2, 3, 1))
    del arrays["v.patch_embd.weight.1"]
    with pytest.raises(ValueError, match="one patch kernel per temporal"):
        vlm.remap_vision_arrays(arrays, "lfm3_vl")


def test_the_float32_pins_cover_both_convolutions():
    from gmlx.load.loader import fp32_keep_for

    for model_type in ("lfm3_omni", "lfm3_vl"):
        keep = fp32_keep_for(model_type)
        assert any("layers.0.conv.spatial_conv.weight".endswith(k)
                   for k in keep)
        assert any("language_model.model.layers.0.conv.kernel".endswith(k)
                   for k in keep)


# -- the container ----------------------------------------------------------

def _container():
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
        text_config=text, vision_config=VisionConfig(**CONFIG),
        image_token_id=30, video_token_id=29, vocab_size=40)
    model = Model(config)
    mx.random.seed(0)
    model.update(tree_unflatten([
        (k, mx.random.normal(v.shape) * 0.1)
        for k, v in tree_flatten(model.parameters())]))
    return model


def test_features_land_on_the_placeholder_rows_in_order():
    model = _container()
    rng = np.random.default_rng(6)
    image, _ = patchify(rng.standard_normal((1, 3, 8, 16)).astype(np.float32),
                        PATCH, 2, 2)
    clip, _ = patchify(rng.standard_normal((4, 3, 8, 8)).astype(np.float32),
                       PATCH, 2, 2)
    ids = mx.array([[1, 30, 30, 2, 29, 29, 3]])
    kwargs = dict(image_grid_thw=np.array([[1, 2, 4]]),
                  pixel_values_videos=mx.array(clip),
                  video_grid_thw=np.array([[2, 2, 2]]))
    got = model.get_input_embeddings(ids, mx.array(image),
                                     **kwargs).inputs_embeds[0]
    text = model.language_model.model.embed_tokens(ids)[0]
    for row in (0, 3, 6):
        assert mx.array_equal(got[row], text[row])
    want_image = model.vision_features(mx.array(image), [(1, 2, 4)])
    want_clip = model.vision_features(mx.array(clip), [(2, 2, 2)])
    assert mx.allclose(got[1:3], want_image.astype(got.dtype))
    assert mx.allclose(got[4:6], want_clip.astype(got.dtype))
    logits = model(ids, mx.array(image), **kwargs).logits
    assert logits.shape == (1, 7, 40)

    with pytest.raises(ValueError, match="1 image placeholder tokens but 2"):
        model.get_input_embeddings(mx.array([[1, 30, 2]]), mx.array(image),
                                   image_grid_thw=np.array([[1, 2, 4]]))
    with pytest.raises(ValueError, match="needs image_grid_thw"):
        model.get_input_embeddings(ids, mx.array(image))


def test_a_long_clip_runs_a_few_slices_at_a_time(monkeypatch):
    """The passes of a clip give the features one pass gives."""
    from gmlx.models.lfm3 import vlm_model

    model = _container()
    clip, grid = patchify(
        np.random.default_rng(8).standard_normal((10, 3, 8, 8)).astype(
            np.float32), PATCH, 2, 2)
    want = model.vision_features(mx.array(clip), [grid])
    calls = []
    tower = model.vision_tower
    monkeypatch.setattr(
        type(tower), "__call__",
        lambda self, patches, g, _orig=type(tower).__call__:
            calls.append(tuple(g)) or _orig(self, patches, g))
    monkeypatch.setattr(vlm_model, "CLIP_PATCHES", 8)
    got = model.vision_features(mx.array(clip), [grid])
    assert calls == [(2, 2, 2), (2, 2, 2), (1, 2, 2)]
    assert mx.allclose(got, want, rtol=1e-4, atol=1e-5)
    with pytest.raises(ValueError, match="grids hold 20 patches but 24"):
        model.vision_features(mx.array(np.concatenate([clip, clip[:4]])),
                              [grid])


def test_the_serve_prefix_cache_reads_lfm3_media_as_causal():
    """The serve engine runs the language model alone, so the container
    puts its media ids on the text config. LFM3 is causal, so the engine
    can restore a prefix and take the images of the suffix from the whole
    prompt's embeddings. Run and chat read the container and keep the
    stock rule."""
    import importlib
    from types import SimpleNamespace

    from gmlx._vendor.mlx_vlm import apc
    from gmlx.cache.apc_media import (CausalMedia, install_causal_serve_media,
                                      install_media_token_ids)

    install_media_token_ids()
    assert install_causal_serve_media()
    ar = importlib.import_module("gmlx._vendor.mlx_vlm.generate.ar")
    model = _container()
    engine = SimpleNamespace(model=model.language_model)
    ids = ar.BatchGenerator._apc_media_token_ids(engine)
    assert isinstance(ids, CausalMedia) and ids == {29, 30}
    assert apc.prefix_leaves_text_only_suffix([1, 30, 30, 2, 29], 2, ids)
    stock = apc.multimodal_token_ids_from_config(model.config)
    assert not isinstance(stock, CausalMedia)
    assert apc.media_safe_prefix_min([1, 30, 30, 2], stock) == 3
