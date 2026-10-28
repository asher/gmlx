#!/usr/bin/env python3
"""MiniMax-M3.1-Flash image input (mmproj projector ``minimax_m3``).

The converter reorders the ViT q/k rows for llama.cpp's rope; the remap
undoes it for the HF-layout tower. The processor writes patches in
merge-window order and wraps each image in its markers; the container
scatters the merged tokens onto the image rows, and the text tower's
sparse attention reads them in chunked prefill as in one shot. A header
check runs on the real mmproj when it is on disk.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

import mlx.core as mx
from mlx.utils import tree_flatten

from gmlx.load import vlm
from gmlx._vendor.mlx_vlm.models.minimax_m3_vl.vision import (
    VisionModel,
    _apply_vision_rope,
)
from gmlx.models.minimax_m3_vl import Model, ModelConfig, VisionConfig

from tests.models.test_minimax_m3_msa import _M31, _args

MT = vlm.MINIMAX_M3_VISION
REAL_MMPROJ = os.path.expanduser(
    "~/llm/gguf/unsloth__MiniMax-M3.1-Flash-GGUF/mmproj-BF16.gguf")

# The real ViT's head layout (16 heads of 80) on a 2-head, 2-block tower.
_VCFG = dict(hidden_size=160, intermediate_size=64, num_attention_heads=2,
             num_hidden_layers=2, patch_size=14, layer_norm_eps=1e-5,
             rope_theta=10000.0,
             img_token_compression_config={"spatial_merge_size": 2,
                                           "temporal_patch_size": 2})
_IDS = dict(image_token_id=90, video_token_id=91, vision_start_token_id=92,
            vision_end_token_id=93)


def _converter_permute(t: np.ndarray, n_head: int) -> np.ndarray:
    """llama.cpp conversion/minimax.py ``_permute_vit_qk``, verbatim in
    numpy."""
    d_head = t.shape[0] // n_head
    axis_dim = 2 * ((2 * (d_head // 2) // 3) // 2)
    ah = axis_dim // 2
    half = 3 * ah
    perm = []
    perm += list(range(0, ah))
    perm += list(range(half, half + ah))
    perm += list(range(ah, 2 * ah))
    perm += list(range(half + ah, half + 2 * ah))
    perm += list(range(2 * ah, 3 * ah))
    perm += list(range(half + 2 * ah, half + 3 * ah))
    perm += list(range(2 * half, d_head))
    idx = np.array(perm)
    if t.ndim == 2:
        return t.reshape(n_head, d_head, t.shape[1])[:, idx, :].reshape(t.shape)
    return t.reshape(n_head, d_head)[:, idx].reshape(t.shape)


@pytest.mark.parametrize("shape", [(16 * 80, 1280), (16 * 80,)])
def test_qk_unpermute_inverts_the_converter(shape):
    w = np.random.default_rng(0).standard_normal(shape).astype(np.float32)
    gguf = _converter_permute(w, 16)
    assert not np.array_equal(gguf, w)
    back = vlm._minimax_m3_unpermute_qk(mx.array(gguf), 16)
    np.testing.assert_array_equal(np.array(back), w)


def _ggml_neox(x, pos, n_dims, offset, theta=10000.0):
    """ggml_rope_ext NEOX on dims [offset, offset + n_dims) of the last axis
    (ggml_rope_set_offset), one position per row."""
    out = x.copy()
    half = n_dims // 2
    inv = theta ** (-np.arange(half) * 2.0 / n_dims)
    ang = pos[:, None].astype(np.float64) * inv[None]
    cos, sin = np.cos(ang)[:, None], np.sin(ang)[:, None]
    a = x[..., offset:offset + half]
    b = x[..., offset + half:offset + n_dims]
    out[..., offset:offset + half] = a * cos - b * sin
    out[..., offset + half:offset + n_dims] = b * cos + a * sin
    return out


def test_hf_rope_on_unpermuted_rows_matches_llama_cpp_rope():
    # q in the GGUF row order, roped as clip_graph_minimax_m3::apply_rope
    # does (t unrotated, h then w with NEOX at offsets axd and 2*axd,
    # positions in its bh, bw, mh, mw loop order), must equal the vendored
    # HF rope on the rows the remap restores.
    gh, gw, n_head, d = 4, 6, 2, 80
    axd = 2 * ((2 * (d // 2) // 3) // 2)
    rng = np.random.default_rng(1)
    q_gguf = rng.standard_normal((gh * gw, n_head, d)).astype(np.float32)
    pos_h, pos_w = [], []
    for bh in range(gh // 2):
        for bw in range(gw // 2):
            for mh in range(2):
                for mw in range(2):
                    pos_h.append(bh * 2 + mh)
                    pos_w.append(bw * 2 + mw)
    ref = _ggml_neox(q_gguf, np.array(pos_h), axd, axd)
    ref = _ggml_neox(ref, np.array(pos_w), axd, 2 * axd)

    tower = VisionModel(VisionConfig.from_dict(
        {**_VCFG, "num_attention_heads": n_head, "hidden_size": n_head * d}))
    freqs = tower.vision_model._rotary_pos_emb(mx.array([[1, gh, gw]]))
    rope = (mx.cos(freqs), mx.sin(freqs))

    def unpermute(x):  # [N, H, d] -> HF rows, per token
        flat = mx.array(x.reshape(len(x), n_head * d).T)
        return vlm._minimax_m3_unpermute_qk(flat, n_head).T.reshape(
            len(x), n_head, d)

    got = _apply_vision_rope(unpermute(q_gguf)[None], rope)[0]
    np.testing.assert_allclose(np.array(got), np.array(unpermute(ref)),
                               rtol=1e-4, atol=1e-4)


def _mmproj_shapes(cfg=_VCFG, text_hidden=64, proj_hidden=48):
    """Tensor shapes of a tiny minimax_m3 mmproj, numpy order."""
    hid, ffn, p = cfg["hidden_size"], cfg["intermediate_size"], cfg["patch_size"]
    shapes = {
        "v.patch_embd.weight": (hid, 3, p, p),
        "v.patch_embd.weight.1": (hid, 3, p, p),
        "v.pre_ln.weight": (hid,), "v.pre_ln.bias": (hid,),
        "mm.1.weight": (proj_hidden, hid), "mm.1.bias": (proj_hidden,),
        "mm.2.weight": (text_hidden, proj_hidden), "mm.2.bias": (text_hidden,),
        "mm.merger.fc1.weight": (text_hidden, 4 * text_hidden),
        "mm.merger.fc1.bias": (text_hidden,),
        "mm.merger.fc2.weight": (text_hidden, text_hidden),
        "mm.merger.fc2.bias": (text_hidden,),
    }
    for i in range(cfg["num_hidden_layers"]):
        b = f"v.blk.{i}"
        for n in ("attn_q", "attn_k", "attn_v", "attn_out"):
            shapes[f"{b}.{n}.weight"] = (hid, hid)
            shapes[f"{b}.{n}.bias"] = (hid,)
        for n in ("ln1", "ln2"):
            shapes[f"{b}.{n}.weight"] = (hid,)
            shapes[f"{b}.{n}.bias"] = (hid,)
        shapes[f"{b}.ffn_up.weight"] = (ffn, hid)
        shapes[f"{b}.ffn_up.bias"] = (ffn,)
        shapes[f"{b}.ffn_down.weight"] = (hid, ffn)
        shapes[f"{b}.ffn_down.bias"] = (hid,)
    return shapes


def _mm_meta(cfg=_VCFG, projection_dim=64):
    return {
        "clip.projector_type": "minimax_m3",
        "clip.vision.embedding_length": cfg["hidden_size"],
        "clip.vision.feed_forward_length": cfg["intermediate_size"],
        "clip.vision.block_count": cfg["num_hidden_layers"],
        "clip.vision.attention.head_count": cfg["num_attention_heads"],
        "clip.vision.patch_size": cfg["patch_size"],
        "clip.vision.image_size": 672,
        "clip.vision.projection_dim": projection_dim,
        "clip.vision.spatial_merge_size": 2,
        "clip.vision.attention.layer_norm_epsilon": 1e-5,
        "clip.vision.image_mean": [0.48145467, 0.4578275, 0.40821072],
        "clip.vision.image_std": [0.26862955, 0.26130259, 0.27577711],
    }


def _llm_meta():
    toks = ["a"] * 100
    for tok, i in ((vlm.MINIMAX_M3_IMAGE_TOKEN, 90),
                   (vlm.MINIMAX_M3_VIDEO_TOKEN, 91),
                   (vlm.MINIMAX_M3_IMAGE_START, 92),
                   (vlm.MINIMAX_M3_IMAGE_END, 93)):
        toks[i] = tok
    return {"tokenizer.ggml.tokens": toks}


def _vision_params(model) -> dict:
    return {k: v.shape for k, v in tree_flatten(model.parameters())
            if not k.startswith("language_model.")}


def test_remap_fills_every_vision_param_and_stacks_patch_slices():
    shapes = _mmproj_shapes()
    rng = np.random.default_rng(2)
    arrays = {k: mx.array(rng.standard_normal(s).astype(np.float32))
              for k, s in shapes.items()}
    out, skipped, kq = vlm.remap_vision_arrays(arrays, MT, mm_meta=_mm_meta())
    assert skipped == [] and kq == {}
    cfg = vlm._synthesize_minimax_m3_vlm_config(
        {"hidden_size": 64, "vocab_size": 100}, _mm_meta(), _llm_meta(),
        shapes)
    assert cfg["projector_hidden_size"] == 48
    assert {k: cfg[k] for k in _IDS} == _IDS
    model = Model(ModelConfig(text_config=_args(**_M31),
                              vision_config=VisionConfig.from_dict(
                                  cfg["vision_config"]),
                              projector_hidden_size=48, **_IDS))
    params = _vision_params(model)
    assert set(out) == set(params)
    for k, v in out.items():
        assert v.shape == params[k], k
    patch = np.array(out[f"{vlm._MINIMAX_M3_VM}.embeddings.patch_embedding.weight"])
    np.testing.assert_array_equal(patch[:, :, 0],
                                  np.array(arrays["v.patch_embd.weight"]))
    np.testing.assert_array_equal(patch[:, :, 1],
                                  np.array(arrays["v.patch_embd.weight.1"]))
    # v is not reordered; q is.
    layer = f"{vlm._MINIMAX_M3_VM}.encoder.layers.0.self_attn"
    np.testing.assert_array_equal(np.array(out[f"{layer}.v_proj.weight"]),
                                  np.array(arrays["v.blk.0.attn_v.weight"]))
    np.testing.assert_array_equal(
        _converter_permute(np.array(out[f"{layer}.q_proj.weight"]), 2),
        np.array(arrays["v.blk.0.attn_q.weight"]))


def test_config_refuses_an_mmproj_of_another_width():
    with pytest.raises(ValueError, match="hidden size"):
        vlm._synthesize_minimax_m3_vlm_config(
            {"hidden_size": 32}, _mm_meta(), _llm_meta(), _mmproj_shapes())


_SPECIALS = [vlm.MINIMAX_M3_IMAGE_TOKEN, vlm.MINIMAX_M3_VIDEO_TOKEN,
             vlm.MINIMAX_M3_IMAGE_START, vlm.MINIMAX_M3_IMAGE_END]


def _tok():
    """A word-level tokenizer with M3.1's marker ids."""
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    vocab = {"[UNK]": 0, "hi": 1,
             **{t: i for t, i in zip(_SPECIALS, (90, 91, 92, 93))}}
    tok = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.add_special_tokens(_SPECIALS)
    return PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="[UNK]",
                                   pad_token="[UNK]")


@pytest.mark.parametrize("hw,target", [
    ((448, 448), (448, 448)),       # aligned, inside the budget
    ((488, 640), (476, 644)),       # test-1.jpeg, rounded to 28
    ((20, 20), (84, 84)),           # under 8 tokens: scaled up, ceiled
    ((2000, 3000), (532, 812)),     # over 576 tokens: scaled down, floored
])
def test_resize_follows_llama_cpp_token_limits(hw, target):
    ip = vlm._minimax_m3_image_processor(_mm_meta())
    assert ip.target_size(*hw) == target
    h, w = target
    assert 8 <= (h // 28) * (w // 28) <= 576


def test_patches_flatten_in_merge_window_order():
    # Each 14x14 patch of a 112x112 image is one grey level, so a patch row
    # names the grid cell it came from.
    from PIL import Image

    ip = vlm._minimax_m3_image_processor(
        {**_mm_meta(), "clip.vision.image_mean": [0.0] * 3,
         "clip.vision.image_std": [1.0] * 3})
    cell = np.arange(64, dtype=np.uint8).reshape(8, 8) * 3
    img = np.repeat(np.repeat(cell, 14, 0), 14, 1)
    out = ip([Image.fromarray(np.stack([img] * 3, -1))])
    assert out["image_grid_thw"].tolist() == [[1, 8, 8]]
    px = out["pixel_values"]
    assert px.shape == (64, 3 * 2 * 14 * 14)
    order = [(bh * 2 + mh) * 8 + bw * 2 + mw
             for bh in range(4) for bw in range(4)
             for mh in range(2) for mw in range(2)]
    np.testing.assert_allclose(px[:, 0] * 255.0, cell.reshape(-1)[order],
                               atol=1e-3)
    assert np.all(px == px[:, :1])  # flat patches, both temporal slices


def test_processor_wraps_each_image_in_its_markers():
    from PIL import Image

    proc = vlm._synthesize_minimax_m3_processor(_tok(), _mm_meta())
    out = proc(images=[Image.new("RGB", (448, 448))],
               text=[vlm.MINIMAX_M3_IMAGE_TOKEN + " hi"])
    ids = np.array(out["input_ids"])[0].tolist()
    assert ids == [92] + [90] * 256 + [93, 1]
    assert np.array(out["pixel_values"]).shape == (32 * 32, 1176)
    with pytest.raises(ValueError, match="video input is not supported"):
        proc(text="hi", videos=["clip.mp4"])


def _container(seed=0):
    mx.random.seed(seed)
    model = Model(ModelConfig(text_config=_args(**_M31),
                              vision_config=VisionConfig.from_dict(_VCFG),
                              projector_hidden_size=48, **_IDS))
    # The patch embed starts at zeros, which makes every token the same.
    pe = model.vision_tower.vision_model.embeddings.patch_embedding
    pe.weight = 0.05 * mx.random.normal(pe.weight.shape)
    return model


def test_container_scatters_merged_tokens_and_chunked_msa_matches():
    # 4x6 patches -> 6 merged tokens inside a 64-token prompt: 8 key blocks
    # of 8 against a top-2 selection, so sparse attention selects. The
    # chunk boundary falls inside the image.
    model = _container(seed=4)
    gh, gw = 4, 6
    pixels = mx.random.normal((gh * gw, 3 * 2 * 14 * 14), key=mx.random.key(5))
    grid = np.array([[1, gh, gw]])
    feats = model.image_features(pixels, grid)
    assert feats.shape == (6, 64)
    assert float(mx.abs(feats - feats[0]).max()) > 1e-2
    mx.random.seed(6)
    text = mx.random.randint(0, 80, (1, 64)).tolist()[0]
    ids = text[:20] + [92] + [90] * 6 + [93] + text[28:]
    ids = mx.array([ids])
    emb = model.get_input_embeddings(ids, pixels,
                                     image_grid_thw=grid).inputs_embeds
    np.testing.assert_allclose(np.array(emb[0, 21:27]), np.array(feats),
                               rtol=1e-5, atol=1e-5)
    np.testing.assert_array_equal(
        np.array(emb[0, 27]),
        np.array(model.language_model.model.embed_tokens(mx.array([93]))[0]))

    lm = model.language_model
    one = model(ids, pixels, cache=model.make_cache(),
                image_grid_thw=grid).logits
    cache = model.make_cache()
    two = mx.concatenate(
        [lm(ids[:, :24], inputs_embeds=emb[:, :24], cache=cache).logits,
         lm(ids[:, 24:], inputs_embeds=emb[:, 24:], cache=cache).logits],
        axis=1)
    assert mx.allclose(one, two, atol=1e-2, rtol=1e-2).item()
    # The image rows reach the logits, and logits_to_keep keeps the last.
    blank = lm(ids, inputs_embeds=model.get_input_embeddings(ids).inputs_embeds,
               cache=model.make_cache()).logits
    assert not mx.allclose(one[:, 21:], blank[:, 21:], atol=1e-3).item()
    last = model(ids, pixels, cache=model.make_cache(), image_grid_thw=grid,
                 logits_to_keep=1).logits
    assert last.shape == (1, 1, 100)
    # A one-row head runs another matmul kernel.
    assert mx.allclose(last[:, 0], one[:, -1], atol=1e-2, rtol=1e-2).item()
    with pytest.raises(ValueError, match="6 image placeholder"):
        model.get_input_embeddings(
            ids, mx.zeros((8 * 6, 1176)), image_grid_thw=np.array([[1, 8, 6]]))


def test_sanitize_runs_the_text_rules_on_the_language_model_only():
    model = _container()
    w = mx.zeros((4, 4), dtype=mx.float32)
    out = model.sanitize({
        "language_model.model.layers.0.self_attn.index_q_proj.weight": w,
        "language_model.mtp.0.weight": w,
        f"{vlm._MINIMAX_M3_VM}.pre_layrnorm.weight": w,
    })
    assert set(out) == {
        "language_model.model.layers.0.self_attn.index_q_proj.weight",
        f"{vlm._MINIMAX_M3_VM}.pre_layrnorm.weight"}
    # The indexer projection narrows as on the text path.
    text = model.language_model.sanitize(
        {"model.layers.0.self_attn.index_q_proj.weight": w})
    assert out["language_model.model.layers.0.self_attn.index_q_proj.weight"
               ].dtype == text["model.layers.0.self_attn.index_q_proj.weight"].dtype
    assert out[f"{vlm._MINIMAX_M3_VM}.pre_layrnorm.weight"].dtype == mx.float32


@pytest.mark.skipif(not os.path.exists(REAL_MMPROJ),
                    reason="MiniMax-M3.1-Flash mmproj not on disk")
def test_real_mmproj_header_maps_onto_the_tower():
    # Header only: shapes come from the tensor table, arrays stay lazy.
    from gmlx.load.headerscan import scan_gguf
    from gmlx._vendor.mlx_vlm.models.minimax_m3_vl.minimax_m3_vl import (
        MiniMaxProjector,
    )

    scan = scan_gguf(REAL_MMPROJ)
    kv = scan.kv
    assert vlm.resolve_vlm_model_type("minimax-m3", kv) == MT
    shapes = {t.name: tuple(reversed(t.shape)) for t in scan.tensors}
    toks = ["a"] * 200064
    for tok, i in ((vlm.MINIMAX_M3_IMAGE_TOKEN, 200025),
                   (vlm.MINIMAX_M3_VIDEO_TOKEN, 200026),
                   (vlm.MINIMAX_M3_IMAGE_START, 200029),
                   (vlm.MINIMAX_M3_IMAGE_END, 200030)):
        toks[i] = tok
    cfg = vlm._synthesize_minimax_m3_vlm_config(
        {"hidden_size": 6144, "vocab_size": 200064}, kv,
        {"tokenizer.ggml.tokens": toks}, shapes)
    arrays = {k: mx.zeros(s) for k, s in shapes.items()}
    out, skipped, _ = vlm.remap_vision_arrays(arrays, MT, mm_meta=kv)
    assert skipped == []
    vc = VisionConfig.from_dict(cfg["vision_config"])
    assert (vc.hidden_size, vc.num_attention_heads, vc.num_hidden_layers,
            vc.spatial_merge_size, vc.temporal_patch_size) == (1280, 16, 32, 2, 2)
    want = {f"vision_tower.{k}": v.shape
            for k, v in tree_flatten(VisionModel(vc).parameters())}
    for name, mod in (
            ("multi_modal_projector", MiniMaxProjector(
                1280, cfg["projector_hidden_size"], 6144, True)),
            ("patch_merge_mlp", MiniMaxProjector(4 * 6144, 6144, 6144, True))):
        want.update({f"{name}.{k}": v.shape
                     for k, v in tree_flatten(mod.parameters())})
    assert {k: tuple(v.shape) for k, v in out.items()} == \
        {k: tuple(v) for k, v in want.items()}
    ip = vlm._minimax_m3_image_processor(kv)
    assert ip.target_size(672, 672) == (672, 672)
    assert ip.target_size(1344, 1344) == (672, 672)


def test_the_tower_loads_in_float32():
    # The bfloat16 tower drifts far from llama.cpp on low-norm image tokens.
    from gmlx.load.loader import float_dtype_rules

    assert float_dtype_rules("minimax_m3_vision")["fp32_cast"] == ("vision_tower.",)
