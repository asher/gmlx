"""DeepSeek-V4-Flash-Vision-Exp loader plumbing: mmproj remap, VLM config
synth, the GGUF-only processor (block expansion + image_spans), and the
container build. CPU-only, synthetic metadata; the real-mmproj remap check
runs only when the local file is present."""

import os
import sys

import numpy as np
import mlx.core as mx
import pytest

sys.path.insert(0, os.path.dirname(__file__))

from gmlx.load.vlm import (
    DEEPSEEK4V_IMAGE_TOKEN,
    DEEPSEEK4V_IMAGE_TOKEN_ID,
    _synthesize_deepseek4v_processor,
    build_vlm_model,
    remap_vision_arrays,
    synthesize_vlm_config,
)
from gmlx.models.deepseek_v4 import image_block as ib

from test_config_synth import _DEEPSEEK4_SHAPES, _deepseek4_meta

MM_META = {
    "clip.has_vision_encoder": True,
    "clip.projector_type": "deepseek4v",
    "clip.vision.projection_dim": 64,
    "clip.vision.image_size": 672,
    "clip.vision.patch_size": 14,
    "clip.vision.embedding_length": 16,
    "clip.vision.feed_forward_length": 24,
    "clip.vision.block_count": 2,
    "clip.vision.attention.head_count": 2,
    "clip.vision.image_mean": [0.5],
    "clip.vision.image_std": [0.5],
    "clip.vision.attention.layer_norm_epsilon": 1e-6,
    "clip.use_silu": True,
    "clip.vision.projector.scale_factor": 3,
    "clip.vision.image_min_pixels": 147456,
}

MMPROJ = os.path.expanduser(
    "~/llm/gguf/unsloth__DeepSeek-V4-Flash-Vision-Exp-GGUF/mmproj-BF16.gguf")


def _fake_mmproj_arrays(depth=2, dim=16, ffn=24, out=64, patch=14):
    a = {
        "v.patch_embd.weight": mx.zeros((dim, 3, patch, patch)),
        "v.patch_embd.bias": mx.zeros((dim,)),
        "v.post_ln.weight": mx.ones((dim,)),
        "mm.1.weight": mx.zeros((out, dim * 9)),
        "mm.1.bias": mx.zeros((out,)),
        "mm.2.weight": mx.zeros((out, out)),
        "mm.2.bias": mx.zeros((out,)),
        "v.token_embd.img_start": mx.zeros((out,)),
        "v.token_embd.img_pad": mx.zeros((out,)),
        "v.token_embd.img_end": mx.zeros((out,)),
        "v.image_newline": mx.zeros((out,)),
    }
    for i in range(depth):
        for sub in ("attn_q", "attn_k", "attn_v", "attn_out"):
            a[f"v.blk.{i}.{sub}.weight"] = mx.zeros((dim, dim))
            a[f"v.blk.{i}.{sub}.bias"] = mx.zeros((dim,))
        a[f"v.blk.{i}.ffn_gate.weight"] = mx.zeros((ffn, dim))
        a[f"v.blk.{i}.ffn_up.weight"] = mx.zeros((ffn, dim))
        a[f"v.blk.{i}.ffn_down.weight"] = mx.zeros((dim, ffn))
        a[f"v.blk.{i}.ln1.weight"] = mx.ones((dim,))
        a[f"v.blk.{i}.ln2.weight"] = mx.ones((dim,))
    return a


def _config():
    from gmlx.models.deepseek_v4.vlm_model import ensure_registered
    ensure_registered()
    return synthesize_vlm_config(
        "deepseek_v4_vl", _deepseek4_meta(), _DEEPSEEK4_SHAPES, MM_META)


def test_remap_covers_the_whole_tower_and_nothing_else():
    arrays = _fake_mmproj_arrays()
    out, skipped, kq = remap_vision_arrays(arrays, "deepseek_v4_vl")
    assert skipped == [] and kq == {}
    assert out["vision_tower.patch_embed.proj.weight"].shape == (16, 3 * 14 * 14)
    assert "vision_tower.blocks.1.attn.q_proj.bias" in out
    assert "vision_tower.blocks.0.mlp.down_proj.weight" in out
    assert "vision_tower.norm.weight" in out
    assert out["vision_tower.aligner.w1.weight"].shape == (64, 144)
    for k in ("image_start", "image_pad", "image_newline", "image_end"):
        assert k in out
    # every remapped name is a parameter of the container
    cfg = _config()
    model, _ = build_vlm_model(cfg)
    import mlx.utils as mu
    names = {k for k, _ in mu.tree_flatten(model.parameters())}
    missing = sorted(set(out) - names)
    assert not missing, missing
    vision_params = sorted(k for k in names if not k.startswith("language_model."))
    assert sorted(out) == vision_params


def test_config_synth_fields():
    cfg = _config()
    assert cfg["model_type"] == "deepseek_v4_vl"
    assert cfg["text_config"]["model_type"] == "deepseek_v4"
    vc = cfg["vision_config"]
    assert (vc["depth"], vc["hidden_size"], vc["num_heads"]) == (2, 16, 2)
    assert vc["intermediate_size"] == 24 and vc["out_hidden_size"] == 64
    assert vc["downsample_ratio"] == 3 and vc["min_pixels"] == 147456
    assert vc["max_n_token"] == 384 and vc["max_wh_ratio"] == 8
    assert vc["rms_norm_eps"] == pytest.approx(1e-6)
    vocab = cfg["vocab_size"]
    assert vocab == cfg["text_config"]["vocab_size"]
    assert cfg["media_token_ids"] == [vocab + t for t in range(5)]
    assert cfg["text_config"]["media_token_ids"] == cfg["media_token_ids"]
    # no token list in the synthetic meta: the placeholder id is the pin
    assert cfg["image_token_id"] == DEEPSEEK4V_IMAGE_TOKEN_ID
    meta = dict(_deepseek4_meta())
    meta["tokenizer.ggml.tokens"] = ["a", "b", DEEPSEEK4V_IMAGE_TOKEN, "c"]
    cfg2 = synthesize_vlm_config(
        "deepseek_v4_vl", meta, _DEEPSEEK4_SHAPES, MM_META)
    assert cfg2["image_token_id"] == cfg2["image_token_index"] == 2


def test_container_build_carries_media_ids_and_sanitize():
    cfg = _config()
    model, _ = build_vlm_model(cfg)
    assert type(model).__module__ == "gmlx.models.deepseek_v4.vlm_model"
    assert model.config.model_type == "deepseek_v4_vl"
    lm = model.language_model
    assert lm.config.model_type == "deepseek_v4"
    assert lm.config.media_token_ids == cfg["media_token_ids"]
    assert lm.config.vision_router_bias is False   # fixture has no _vl bias
    # sanitize: vision keys pass through, text keys go through the tower's
    w = {"image_start": mx.zeros((4,)),
         "language_model.model.layers.0.ffn.gate.tid2eid": mx.zeros((3, 2))}
    out = model.sanitize(w)
    assert out["image_start"].shape == (4,)
    assert out["language_model.model.layers.0.ffn.gate.tid2eid"].dtype == mx.int32


def _tokenizer(auto_bos=False):
    from tokenizers import Tokenizer, models, pre_tokenizers, processors
    from transformers import PreTrainedTokenizerFast

    bos = "<\uff5cbegin\u2581of\u2581sentence\uff5c>"
    vocab = {"[UNK]": 0, "hi": 1, "there": 2, "what": 3, "is": 4,
             "this": 5, bos: 6, DEEPSEEK4V_IMAGE_TOKEN: 7}
    tok = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.add_special_tokens([bos, DEEPSEEK4V_IMAGE_TOKEN])
    if auto_bos:
        tok.post_processor = processors.TemplateProcessing(
            single=f"{bos} $A", special_tokens=[(bos, 6)])
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok, bos_token=bos, unk_token="[UNK]")
    return fast, bos


def _processor(auto_bos=False):
    tok, bos = _tokenizer(auto_bos)
    proc = _synthesize_deepseek4v_processor(
        tok, MM_META, {"vocab_size": 8, "image_token_id": 7})
    return proc, bos


def _image(w=300, h=200, seed=0):
    from PIL import Image
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 255, (h, w, 3), dtype=np.uint8))


def test_processor_expands_placeholder_into_block_with_spans():
    proc, bos = _processor()
    text = f"{bos}what is {DEEPSEEK4V_IMAGE_TOKEN} this"
    feat = proc(images=[_image()], text=text)
    ids = feat["input_ids"]
    assert ids.shape[0] == 1
    ids = [int(t) for t in ids[0].tolist()]
    meta = feat["image_meta"]
    assert meta.shape == (1, 4)
    n_vit_h, n_vit_w, n_llm_h, n_llm_w = (int(v) for v in meta[0].tolist())
    assert feat["pixel_values"].shape == (n_vit_h * n_vit_w, 3 * 14 * 14)
    # 300x200 is below min_pixels: upscaled, then the aligner grid follows
    assert n_llm_h == -(-n_vit_h // 3) and n_llm_w == -(-n_vit_w // 3)
    # ids: BOS what is <block> this, block at position 3
    types, _ = ib.build_image_block(n_llm_h, n_llm_w, 3)
    assert ids[:3] == [6, 3, 4]
    assert ids[3:3 + len(types)] == [8 + int(t) for t in types]
    assert ids[3 + len(types):] == [5]
    spans = feat["image_spans"]
    assert isinstance(spans, list) and spans == [[3, 3 + len(types)]]
    assert ids.count(6) == 1                       # exactly one BOS
    assert feat["attention_mask"].shape == (1, len(ids))
    assert ib.block_length(n_llm_h, n_llm_w, 3) <= 384


def test_processor_bos_guard_against_an_auto_bos_tokenizer():
    proc, bos = _processor(auto_bos=True)
    feat = proc(text=f"{bos}hi there")
    ids = [int(t) for t in feat["input_ids"][0].tolist()]
    assert ids == [6, 1, 2]
    # without the template's BOS the tokenizer adds its own, once
    feat = proc(text="hi there")
    assert [int(t) for t in feat["input_ids"][0].tolist()] == [6, 1, 2]


def test_processor_text_only_and_mismatches():
    proc, bos = _processor()
    feat = proc(text=f"{bos}hi there")
    assert "image_spans" not in feat and "pixel_values" not in feat
    assert [int(t) for t in feat["input_ids"][0].tolist()] == [6, 1, 2]
    with pytest.raises(ValueError, match="placeholders"):
        proc(images=[_image()],
             text=f"{bos}{DEEPSEEK4V_IMAGE_TOKEN} {DEEPSEEK4V_IMAGE_TOKEN}")
    with pytest.raises(ValueError, match="placeholders"):
        proc(images=[_image(), _image(seed=1)],
             text=f"{bos}{DEEPSEEK4V_IMAGE_TOKEN} hi")
    with pytest.raises(ValueError, match="single-row"):
        proc(images=[_image()],
             text=[f"{bos}{DEEPSEEK4V_IMAGE_TOKEN}", f"{bos}hi"])


def test_processor_two_images_lead_pads_follow_the_running_length():
    proc, bos = _processor()
    text = f"{bos}{DEEPSEEK4V_IMAGE_TOKEN} hi {DEEPSEEK4V_IMAGE_TOKEN}"
    feat = proc(images=[_image(), _image(w=200, h=300, seed=1)], text=text)
    ids = [int(t) for t in feat["input_ids"][0].tolist()]
    spans = feat["image_spans"]
    assert len(spans) == 2 and spans[0][1] + 1 == spans[1][0]
    for (s, e) in spans:
        assert ids[s + ib.lead_pads(s)] == 8 + ib.IMAGE_START
        assert ids[e - 1] == 8 + ib.IMAGE_END
        assert (s + ib.lead_pads(s)) % 4 == 3
    assert feat["image_meta"].shape == (2, 4)
    n = sum(int(m[0]) * int(m[1]) for m in feat["image_meta"].tolist())
    assert feat["pixel_values"].shape[0] == n


def test_image_processor_geometry_matches_reference_rules():
    proc, _ = _processor()
    ip = proc.image_processor
    # a wide source at the ratio cap stretches instead of padding
    best_w, best_h, n_vit_h, n_vit_w, n_llm_h, n_llm_w, stretch = ip.geometry(
        4000, 200)
    assert stretch and best_w == n_vit_w * 14 and best_h == n_vit_h * 14
    assert ib.block_length(n_llm_h, n_llm_w, 0) <= 384
    # a large square lands on the token budget without stretching
    *_, n_llm_h, n_llm_w, stretch = ip.geometry(2000, 2000)
    assert not stretch and ib.block_length(n_llm_h, n_llm_w, 0) <= 384
    # patch vectors are (C, ph, pw)-ordered: a constant-per-channel image
    # gives 196 equal values per channel run
    from PIL import Image
    img = Image.fromarray(np.stack([np.full((420, 420), v, np.uint8)
                                    for v in (0, 127, 255)], axis=-1))
    patches, meta = ip._one(img)
    assert meta == (30, 30, 10, 10)
    row = patches[0]
    assert np.allclose(row[:196], -1.0) and np.allclose(row[392:], 1.0)
    assert np.allclose(row[196:392], (127 / 255 - 0.5) / 0.5)


@pytest.mark.skipif(not os.path.exists(MMPROJ), reason="local mmproj absent")
def test_real_mmproj_remaps_onto_the_tower():
    from gmlx.load.wire import load_gguf_wire_bytes
    arrays, codecs, _arch, mm_meta, _shapes = load_gguf_wire_bytes(
        MMPROJ, zero_copy=True, expect_quant=False)
    out, skipped, kq = remap_vision_arrays(arrays, "deepseek_v4_vl")
    assert skipped == [] and kq == {}
    cfg = synthesize_vlm_config(
        "deepseek_v4_vl", _deepseek4_meta(), _DEEPSEEK4_SHAPES, mm_meta)
    vc = cfg["vision_config"]
    assert (vc["depth"], vc["hidden_size"], vc["num_heads"]) == (32, 1024, 16)
    assert vc["out_hidden_size"] == 4096 and vc["downsample_ratio"] == 3
    assert out["vision_tower.patch_embed.proj.weight"].shape == (1024, 588)
    assert out["vision_tower.aligner.w1.weight"].shape == (4096, 9216)
    assert out["image_newline"].shape == (4096,)
    assert len(out) == 427
