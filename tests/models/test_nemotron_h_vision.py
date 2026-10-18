#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Asher Feldman
"""Nemotron-3.5-Bolt vision (mmproj ``nemotron_omni``) against a float64
reference of llama.cpp's nemotron-omni graph, on tiny random weights. No
GGUF and no download.

The reference follows the C++ step by step: the 16x16 conv patch embed on
the image, ggml's bilinear upscale of the position table, the registers
prepended after the position add, ``build_vit`` with exact GELU, the
ggml reshape/permute sequence of ``build_patch_merge_permute``, and the
RMSNorm / relu^2 projector. The other tests cover the image processor's
resize and token counts, the ``<img>`` wrapping, the resolver, the remap
and the config synth.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten

from gmlx.load import vlm
from gmlx.models.nemotron_h.vision import VisionConfig, VisionModel

_CFG = dict(hidden_size=32, intermediate_size=64, num_hidden_layers=2,
            num_attention_heads=4, patch_size=4, num_registers=3,
            position_grid=6, merge=2, projector_hidden_size=48,
            projection_dim=24, layer_norm_eps=1e-6, projector_norm_eps=1e-5)


def _tower(seed=0) -> VisionModel:
    tower = VisionModel(VisionConfig(**_CFG))
    rng = np.random.default_rng(seed)
    params = [(k, mx.array(rng.normal(0, 0.3, v.shape).astype(np.float32)))
              for k, v in tree_flatten(tower.parameters())]
    tower.update(tree_unflatten(params))
    mx.eval(tower.parameters())
    return tower


def _w(tower, path) -> np.ndarray:
    return np.array(dict(tree_flatten(tower.parameters()))[path],
                    dtype=np.float64)


def _ggml_bilinear(table: np.ndarray, side: int, gh: int, gw: int):
    """ggml_compute_forward_upscale_f32, GGML_SCALE_MODE_BILINEAR, no
    flags, on the ``[side*side, C]`` table (row y*side + x)."""
    src = table.reshape(side, side, -1)
    out = np.zeros((gh, gw, src.shape[-1]))
    sf0 = np.float32(gw) / np.float32(side)
    sf1 = np.float32(gh) / np.float32(side)
    for i1 in range(gh):
        y = (np.float32(i1) + np.float32(0.5)) / sf1 - np.float32(0.5)
        y0 = int(math.floor(y))
        y1 = min(max(y0 + 1, 0), side - 1)
        y0 = min(max(y0, 0), side - 1)
        dy = min(max(float(y) - y0, 0.0), 1.0)
        for i0 in range(gw):
            x = (np.float32(i0) + np.float32(0.5)) / sf0 - np.float32(0.5)
            x0 = int(math.floor(x))
            x1 = min(max(x0 + 1, 0), side - 1)
            x0 = min(max(x0, 0), side - 1)
            dx = min(max(float(x) - x0, 0.0), 1.0)
            out[i1, i0] = (src[y0, x0] * (1 - dx) * (1 - dy)
                           + src[y0, x1] * dx * (1 - dy)
                           + src[y1, x0] * (1 - dx) * dy
                           + src[y1, x1] * dx * dy)
    return out.reshape(gh * gw, -1)


def _ggml_patch_merge_permute(x: np.ndarray, gh: int, gw: int, m: int):
    """build_patch_merge_permute, step by step. numpy axes are ggml's ne
    reversed, so ggml permute(0, 2, 1, 3) is a swap of numpy axes 0, 1."""
    c = x.shape[-1]
    a = x.reshape(gh, gw, c)                          # ne (C, W, H)
    a = a.reshape(gh, gw // m, c * m)                 # ne (C*m, W/m, H)
    a = np.transpose(a, (1, 0, 2))                    # ne (C*m, H, W/m)
    a = a.reshape(gw // m, gh // m, c * m * m)        # ne (C*m*m, H/m, W/m)
    a = np.transpose(a, (1, 0, 2))                    # ne (C*m*m, W/m, H/m)
    return a.reshape(-1, c * m * m)


def _layer_norm(x, w, b, eps):
    mu = x.mean(-1, keepdims=True)
    var = ((x - mu) ** 2).mean(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * w + b


def _gelu_erf(x):
    return 0.5 * x * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))


def _ref_tower(tower, image: np.ndarray) -> np.ndarray:
    """image: [C, H, W] normalized pixels."""
    cfg = tower.config
    p, d = cfg.patch_size, cfg.hidden_size
    c, h, w = image.shape
    gh, gw = h // p, w // p
    kernel = _w(tower, "patch_embed.weight").reshape(d, c, p, p)
    x = np.zeros((gh, gw, d))
    for y in range(gh):
        for xx in range(gw):
            patch = image[:, y * p:(y + 1) * p, xx * p:(xx + 1) * p]
            x[y, xx] = np.tensordot(kernel, patch, axes=([1, 2, 3], [0, 1, 2]))
    x = x.reshape(gh * gw, d)
    x = x + _ggml_bilinear(_w(tower, "position_embedding"),
                           cfg.position_grid, gh, gw)
    x = np.concatenate([_w(tower, "registers"), x], axis=0)
    nh = cfg.num_attention_heads
    hd = d // nh
    for i in range(cfg.num_hidden_layers):
        b = f"blocks.{i}."

        def lin(name, v, b=b):
            return v @ _w(tower, b + name + ".weight").T + _w(
                tower, b + name + ".bias")

        hn = _layer_norm(x, _w(tower, b + "norm1.weight"),
                         _w(tower, b + "norm1.bias"), cfg.layer_norm_eps)
        q, k, v = (lin(f"attn.{n}_proj", hn).reshape(-1, nh, hd)
                   for n in "qkv")
        s = np.einsum("qhd,khd->hqk", q, k) / math.sqrt(hd)
        s = np.exp(s - s.max(-1, keepdims=True))
        s = s / s.sum(-1, keepdims=True)
        o = np.einsum("hqk,khd->qhd", s, v).reshape(-1, d)
        x = x + lin("attn.out_proj", o)
        hn = _layer_norm(x, _w(tower, b + "norm2.weight"),
                         _w(tower, b + "norm2.bias"), cfg.layer_norm_eps)
        x = x + lin("mlp.fc2", _gelu_erf(lin("mlp.fc1", hn)))
    x = x[cfg.num_registers:]
    x = _ggml_patch_merge_permute(x, gh, gw, cfg.merge)
    x = x / np.sqrt((x * x).mean(-1, keepdims=True)
                    + cfg.projector_norm_eps) * _w(tower, "projector.norm.weight")
    x = np.maximum(x @ _w(tower, "projector.fc1.weight").T, 0.0) ** 2
    return x @ _w(tower, "projector.fc2.weight").T


def _patchify(image: np.ndarray, p: int) -> np.ndarray:
    """The processor's patch layout: row-major patches, channel-major."""
    c, h, w = image.shape
    a = image.reshape(c, h // p, p, w // p, p).transpose(1, 3, 0, 2, 4)
    return a.reshape((h // p) * (w // p), c * p * p)


@pytest.mark.parametrize("gh,gw", [(4, 6), (6, 6), (8, 2)])
def test_tower_matches_llama_cpp_graph(gh, gw):
    tower = _tower(seed=gh * 10 + gw)
    p = _CFG["patch_size"]
    image = np.random.default_rng(1).normal(0, 1, (3, gh * p, gw * p))
    got = tower(mx.array(_patchify(image, p).astype(np.float32)), (gh, gw))
    ref = _ref_tower(tower, image)
    assert got.shape == ((gh // 2) * (gw // 2), _CFG["projection_dim"])
    np.testing.assert_allclose(np.array(got), ref, rtol=2e-3, atol=2e-3)


def test_pixel_shuffle_channel_order():
    # Channel c + C*dx + 2*C*dy of a merged row reads patch (2y+dy, 2x+dx).
    from gmlx.models.nemotron_h.vision import pixel_shuffle
    gh, gw, c = 4, 6, 3
    x = np.arange(gh * gw * c, dtype=np.float32).reshape(gh * gw, c)
    got = np.array(pixel_shuffle(mx.array(x), gh, gw, 2))
    np.testing.assert_array_equal(got, _ggml_patch_merge_permute(x, gh, gw, 2))
    for y in range(gh // 2):
        for xx in range(gw // 2):
            for dy in range(2):
                for dx in range(2):
                    patch = (2 * y + dy) * gw + 2 * xx + dx
                    np.testing.assert_array_equal(
                        got[y * (gw // 2) + xx, c * dx + 2 * c * dy:][:c],
                        x[patch])


def test_position_table_identity_at_native_grid():
    tower = _tower()
    from gmlx.models.nemotron_h.vision import resize_positions
    side = _CFG["position_grid"]
    got = resize_positions(tower.position_embedding, side, side, side)
    np.testing.assert_array_equal(np.array(got),
                                  np.array(tower.position_embedding))


def _processor():
    return vlm._NemotronOmniGgufImageProcessor(
        image_mean=[0.48145467, 0.4578275, 0.40821072],
        image_std=[0.26862955, 0.26130259, 0.27577711],
        patch_size=16, merge=2, min_pixels=262144, max_pixels=3407872)


@pytest.mark.parametrize("hw,target", [
    ((480, 640), (480, 640)),       # already aligned, inside the budget
    ((500, 700), (512, 704)),       # rounded to the nearest 32
    ((100, 100), (512, 512)),       # below the minimum: scaled up, ceiled
    ((1800, 2600), (1536, 2208)),   # above the maximum: scaled down, floored
    ((3000, 40), (4448, 64)),       # thin: rounded under the minimum, ceiled
])
def test_resize_follows_calc_size_preserved_ratio(hw, target):
    assert _processor().target_size(*hw) == target


def test_tokens_and_patches_of_one_image():
    from PIL import Image
    proc = _processor()
    out = proc([Image.new("RGB", (640, 480), (255, 0, 0))])
    assert out["image_grid_thw"].tolist() == [[1, 30, 40]]
    assert out["pixel_values"].shape == (30 * 40, 3 * 16 * 16)
    assert proc.soft_tokens(out["image_grid_thw"][0]) == 300
    # Pure red, CLIP-normalized, channel-major within each patch.
    red = (1.0 - 0.48145467) / 0.26862955
    np.testing.assert_allclose(out["pixel_values"][0, :256], red, rtol=1e-5)
    np.testing.assert_allclose(
        out["pixel_values"][0, 256:512], -0.4578275 / 0.26130259, rtol=1e-5)


_SPECIALS = ["<image>", "<img>", "</img>"]


def _Tok():
    """A word-level tokenizer with Bolt's image ids: <image> 18, <img> 19,
    </img> 20."""
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    vocab = {"[UNK]": 0, "hi": 1, **{t: 18 + i for i, t in enumerate(_SPECIALS)}}
    tok = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.add_special_tokens(_SPECIALS)
    return PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="[UNK]")


def _omni_meta():
    return {
        "clip.vision.image_mean": [0.48145467, 0.4578275, 0.40821072],
        "clip.vision.image_std": [0.26862955, 0.26130259, 0.27577711],
        "clip.vision.patch_size": 16,
        "clip.vision.projector.scale_factor": 2,
        "clip.vision.image_min_pixels": 262144,
        "clip.vision.image_max_pixels": 3407872,
        "clip.vision.embedding_length": 1280,
        "clip.vision.feed_forward_length": 5120,
        "clip.vision.block_count": 32,
        "clip.vision.attention.head_count": 16,
        "clip.vision.projection_dim": 4096,
        "clip.vision.attention.layer_norm_epsilon": 1e-6,
        "clip.projector_type": "nemotron_omni",
    }


def test_processor_wraps_each_image():
    proc = vlm._synthesize_nemotron_omni_processor(_Tok(), _omni_meta())
    grids = np.array([[1, 30, 40], [1, 16, 16]])
    text = proc._expand(["<image 1><image>A<image 2><image>B"], grids)[0]
    assert text == ("<image 1><img>" + "<image>" * 300 + "</img>A"
                    "<image 2><img>" + "<image>" * 64 + "</img>B")
    with pytest.raises(ValueError, match="more <image> placeholders"):
        proc._expand(["<image><image>"], grids[:1])


def test_processor_stream_brackets_the_image_rows():
    from PIL import Image
    proc = vlm._synthesize_nemotron_omni_processor(_Tok(), _omni_meta())
    out = proc(images=[Image.new("RGB", (512, 512))], text=["<image> hi"])
    ids = np.array(out["input_ids"])[0].tolist()
    assert ids == [19] + [18] * 256 + [20, 1]
    assert np.array(out["pixel_values"]).shape == (32 * 32, 768)


def test_processor_refuses_video():
    proc = vlm._synthesize_nemotron_omni_processor(_Tok(), _omni_meta())
    with pytest.raises(ValueError, match="video input is not supported"):
        proc(text="hi", videos=["clip.mp4"])


def test_resolver():
    meta = {"clip.projector_type": "nemotron_omni"}
    assert vlm.resolve_vlm_model_type("nemotron_h_moe", meta) == \
        "nemotron_h_omni"
    with pytest.raises(vlm.UnsupportedVLMError, match="nemotron_omni"):
        vlm.resolve_vlm_model_type("llama", meta)


def _bolt_mmproj_shapes(n_layers=2):
    """The Bolt mmproj's tensors, in GGUF ne order, with fewer blocks."""
    shapes = {
        "mm.model.mlp.0.weight": [5120],
        "mm.model.mlp.1.weight": [5120, 20480],
        "mm.model.mlp.3.weight": [20480, 4096],
        "v.class_embd": [1280, 10],
        "v.patch_embd.weight": [16, 16, 3, 1280],
        "v.patch_embd.weight.1": [16, 16, 6, 1280],
        "v.position_embd.weight": [1280, 16384],
    }
    for i in range(n_layers):
        for t in ("attn_q", "attn_k", "attn_v", "attn_out"):
            shapes[f"v.blk.{i}.{t}.weight"] = [1280, 1280]
            shapes[f"v.blk.{i}.{t}.bias"] = [1280]
        shapes[f"v.blk.{i}.ffn_up.weight"] = [1280, 5120]
        shapes[f"v.blk.{i}.ffn_up.bias"] = [5120]
        shapes[f"v.blk.{i}.ffn_down.weight"] = [5120, 1280]
        shapes[f"v.blk.{i}.ffn_down.bias"] = [1280]
        for t in ("ln1", "ln2"):
            shapes[f"v.blk.{i}.{t}.weight"] = [1280]
            shapes[f"v.blk.{i}.{t}.bias"] = [1280]
    return shapes


def test_config_synth_reads_tensor_dims():
    meta = {**_omni_meta(), "clip.vision.block_count": 2}
    llm_meta = {"tokenizer.ggml.tokens": ["<unk>"] * 18 + ["<image>"]}
    cfg = vlm._synthesize_nemotron_omni_vlm_config(
        {"hidden_size": 4096, "vocab_size": 131072}, meta, llm_meta,
        _bolt_mmproj_shapes())
    v = cfg["vision_config"]
    assert (v["num_registers"], v["position_grid"],
            v["projector_hidden_size"]) == (10, 128, 20480)
    assert cfg["image_token_id"] == 18
    with pytest.raises(ValueError, match="hidden size"):
        vlm._synthesize_nemotron_omni_vlm_config(
            {"hidden_size": 2688}, meta, llm_meta, _bolt_mmproj_shapes())


def test_remap_fills_every_tower_param():
    shapes = _bolt_mmproj_shapes()
    arrays = {k: mx.zeros(tuple(reversed(s)), dtype=mx.float32)
              for k, s in shapes.items()}
    out, skipped, kq = vlm.remap_vision_arrays(arrays, "nemotron_h_omni")
    assert skipped == ["v.patch_embd.weight.1"] and not kq
    meta = {**_omni_meta(), "clip.vision.block_count": 2}
    cfg = vlm._synthesize_nemotron_omni_vlm_config(
        {"hidden_size": 4096}, meta,
        {"tokenizer.ggml.tokens": ["<unk>"] * 18 + ["<image>"]}, shapes)
    tower = VisionModel(VisionConfig.from_dict(cfg["vision_config"]))
    params = dict(tree_flatten(tower.parameters()))
    assert {k[len("vision_tower."):] for k in out} == set(params)
    for k, v in out.items():
        assert v.shape == params[k[len("vision_tower."):]].shape, k


def _container():
    from gmlx.models.nemotron_h.vlm_model import Model, ModelConfig
    from gmlx.models.nemotron_h.model import ModelArgs

    text = ModelArgs(
        model_type="nemotron_h", vocab_size=40, hidden_size=24,
        intermediate_size=32, num_hidden_layers=2, max_position_embeddings=64,
        num_attention_heads=4, num_key_value_heads=2, attention_bias=False,
        mamba_num_heads=4, mamba_head_dim=6, mamba_proj_bias=False,
        ssm_state_size=4, conv_kernel=4, n_groups=2, mlp_bias=False,
        layer_norm_epsilon=1e-5, use_bias=False, use_conv_bias=True,
        hybrid_override_pattern=["M", "*"], head_dim=6)
    return Model(ModelConfig(text_config=text,
                             vision_config=VisionConfig(**_CFG),
                             image_token_id=18))


def test_container_scatters_features_on_image_rows():
    model = _container()
    p = _CFG["patch_size"]
    pixels = mx.zeros((4 * 6, 3 * p * p))
    feats = model.vision_tower(pixels, (4, 6))
    ids = mx.array([[19] + [18] * 6 + [20, 5]])
    emb = model.get_input_embeddings(
        ids, pixels, image_grid_thw=np.array([[1, 4, 6]])).inputs_embeds
    np.testing.assert_allclose(np.array(emb[0, 1:7]), np.array(feats),
                               rtol=1e-6)
    np.testing.assert_array_equal(
        np.array(emb[0, 7]),
        np.array(model.language_model.backbone.embeddings(mx.array([20]))[0]))
    logits = model(ids, pixels, image_grid_thw=np.array([[1, 4, 6]])).logits
    assert logits.shape == (1, 9, 40)
    with pytest.raises(ValueError, match="6 image placeholder"):
        model.get_input_embeddings(
            ids, mx.zeros((8 * 6, 3 * p * p)),
            image_grid_thw=np.array([[1, 8, 6]]))


def test_container_language_model_carries_the_spec_hooks():
    # A drafter runs on text turns through the container's language model;
    # image turns keep their features on the same class.
    from gmlx.load.mtp_target import (
        _MTP_TARGET_HOOKS_BY_TYPE,
        _spec_hook_key,
        _vlm_spec_language_model,
    )

    model = _container()
    lm = model.language_model
    for hook in _MTP_TARGET_HOOKS_BY_TYPE[_spec_hook_key("nemotron_h_omni")]:
        assert callable(getattr(lm, hook)), hook
    assert callable(lm.set_dflash_capture)
    assert lm.model.embed_tokens is lm.backbone.embeddings
    assert _vlm_spec_language_model("nemotron_h_omni") is None

    p = _CFG["patch_size"]
    pixels = mx.random.normal((4 * 6, 3 * p * p), key=mx.random.key(1))
    ids = mx.array([[19] + [18] * 6 + [20, 5]])
    grid = np.array([[1, 4, 6]])
    emb = model.get_input_embeddings(ids, pixels, image_grid_thw=grid)
    with_image = model(ids, pixels, image_grid_thw=grid).logits
    np.testing.assert_allclose(
        np.array(with_image),
        np.array(lm(ids, inputs_embeds=emb.inputs_embeds).logits), rtol=1e-6)
    assert not np.allclose(np.array(with_image[0, -1]),
                           np.array(lm(ids).logits[0, -1]))


def test_fp32_keep_row_covers_tower_and_router():
    from gmlx.load.loader import _FP32_KEEP_BY_MODEL_TYPE
    keep = _FP32_KEEP_BY_MODEL_TYPE["nemotron_h_omni"]
    for name in ("vision_tower.blocks.0.norm1.weight",
                 "vision_tower.position_embedding",
                 "language_model.backbone.layers.1.mixer.gate.weight"):
        assert any(s in name for s in keep), name
    assert set(_FP32_KEEP_BY_MODEL_TYPE["nemotron_h"]) <= set(keep)
