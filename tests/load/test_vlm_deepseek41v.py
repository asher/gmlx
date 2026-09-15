"""DeepSeek-V4.1-Flash-Vision loader plumbing: mmproj remap (fused wqkv and
w1 split), VLM config synth, the GGUF-only processor (block expansion), and
the container build. CPU-only, synthetic metadata; the real-mmproj remap
check runs only when the local file is present."""

import math
import os
import sys

import numpy as np
import mlx.core as mx
import pytest

sys.path.insert(0, os.path.dirname(__file__))

from gmlx.load.vlm import (
    DEEPSEEK41V_IMAGE_TOKEN,
    DEEPSEEK41V_IMAGE_TOKEN_ID,
    _synthesize_deepseek41v_processor,
    build_vlm_model,
    remap_vision_arrays,
    synthesize_vlm_config,
)
from gmlx.models.deepseek_v41 import image_block as ib

from test_config_synth import _DEEPSEEK41_SHAPES, _deepseek41_meta

ARCH = "deepseek4-vision"
MM_META = {
    "general.architecture": ARCH,
    f"{ARCH}.block_count": 2,
    f"{ARCH}.embedding_length": 16,
    f"{ARCH}.feed_forward_length": 24,
    f"{ARCH}.attention.head_count": 2,
    f"{ARCH}.attention.layer_norm_rms_epsilon": 1e-6,
    f"{ARCH}.patch_size": 14,
    f"{ARCH}.projection_length": 64,
    f"{ARCH}.downsample_ratio": 3,
    f"{ARCH}.image.max_tokens": 1024,
    f"{ARCH}.image.min_pixels": 295936,
    f"{ARCH}.image.max_width_height_ratio": 0,
    f"{ARCH}.image.token_id": 129264,
    f"{ARCH}.rope.freq_base": 10000.0,
}

MMPROJ = os.path.expanduser(
    "~/llm/gguf/antirez__deepseek-v4.1-flash-gguf/"
    "DeepSeek-V4.1-Flash-Vision.gguf")


def _fake_mmproj_arrays(depth=2, dim=16, ffn=24, out=64, patch=14):
    a = {
        "vision.patch_embed.proj.weight": mx.zeros((dim, 3 * patch * patch)),
        "vision.patch_embed.proj.bias": mx.zeros((dim,)),
        "vision.norm.weight": mx.ones((dim,)),
        "aligner.w1.weight": mx.zeros((out, dim * 9)),
        "aligner.w1.bias": mx.zeros((out,)),
        "aligner.w2.weight": mx.zeros((out, out)),
        "aligner.w2.bias": mx.zeros((out,)),
        "image_start": mx.zeros((out,)),
        "image_newline": mx.zeros((out,)),
        "image_end": mx.zeros((out,)),
    }
    for i in range(depth):
        # The reference fuses q/k/v and gate/up on the output dim.
        a[f"vision.blocks.{i}.attn.wqkv.weight"] = mx.zeros((3 * dim, dim))
        a[f"vision.blocks.{i}.attn.wqkv.bias"] = mx.zeros((3 * dim,))
        a[f"vision.blocks.{i}.attn.wo.weight"] = mx.zeros((dim, dim))
        a[f"vision.blocks.{i}.attn.wo.bias"] = mx.zeros((dim,))
        a[f"vision.blocks.{i}.mlp.w1.weight"] = mx.zeros((2 * ffn, dim))
        a[f"vision.blocks.{i}.mlp.w2.weight"] = mx.zeros((dim, ffn))
        a[f"vision.blocks.{i}.norm1.weight"] = mx.ones((dim,))
        a[f"vision.blocks.{i}.norm2.weight"] = mx.ones((dim,))
    return a


def _config():
    from gmlx.models.deepseek_v41.vlm_model import ensure_registered
    ensure_registered()
    return synthesize_vlm_config(
        "deepseek_v41_vl", _deepseek41_meta(), _DEEPSEEK41_SHAPES, MM_META)


def test_remap_covers_the_whole_tower_and_nothing_else():
    arrays = _fake_mmproj_arrays()
    out, skipped, kq = remap_vision_arrays(arrays, "deepseek_v41_vl")
    assert skipped == [] and kq == {}
    assert out["vision_tower.patch_embed.proj.weight"].shape == (16, 3 * 14 * 14)
    for proj in ("q_proj", "k_proj", "v_proj"):
        assert out[f"vision_tower.blocks.1.attn.{proj}.weight"].shape == (16, 16)
        assert out[f"vision_tower.blocks.1.attn.{proj}.bias"].shape == (16,)
    assert out["vision_tower.blocks.0.mlp.gate_proj.weight"].shape == (24, 16)
    assert out["vision_tower.blocks.0.mlp.up_proj.weight"].shape == (24, 16)
    assert out["vision_tower.blocks.0.mlp.down_proj.weight"].shape == (16, 24)
    assert out["vision_tower.norm.weight"].shape == (16,)
    assert out["vision_tower.aligner.w1.weight"].shape == (64, 144)
    for k in ("image_start", "image_newline", "image_end"):
        assert k in out
    assert "image_pad" not in out                  # V4.1 has no pad sentinel

    cfg = _config()
    model, _ = build_vlm_model(cfg)
    import mlx.utils as mu
    names = {k for k, _ in mu.tree_flatten(model.parameters())}
    missing = sorted(set(out) - names)
    assert not missing, missing
    vision_params = sorted(k for k in names if not k.startswith("language_model."))
    assert sorted(out) == vision_params


def test_fused_rows_split_in_reference_order():
    dim = 16
    w = mx.arange(3 * dim * dim, dtype=mx.float32).reshape(3 * dim, dim)
    arrays = _fake_mmproj_arrays()
    arrays["vision.blocks.0.attn.wqkv.weight"] = w
    out, _, _ = remap_vision_arrays(arrays, "deepseek_v41_vl")
    for i, proj in enumerate(("q_proj", "k_proj", "v_proj")):
        want = w[i * dim:(i + 1) * dim]
        assert mx.array_equal(out[f"vision_tower.blocks.0.attn.{proj}.weight"],
                              want)


def test_config_synth_fields():
    cfg = _config()
    assert cfg["model_type"] == "deepseek_v41_vl"
    assert cfg["text_config"]["model_type"] == "deepseek_v41"
    vc = cfg["vision_config"]
    assert (vc["depth"], vc["hidden_size"], vc["num_heads"]) == (2, 16, 2)
    assert vc["intermediate_size"] == 24 and vc["out_hidden_size"] == 64
    assert vc["downsample_ratio"] == 3 and vc["min_pixels"] == 295936
    assert vc["max_n_token"] == 1024
    # 0 is the converter's spelling of the reference's None: no cap.
    assert vc["max_wh_ratio"] is None
    assert vc["rms_norm_eps"] == pytest.approx(1e-6)
    vocab = cfg["vocab_size"]
    assert vocab == cfg["text_config"]["vocab_size"]
    assert cfg["media_token_ids"] == [vocab + t for t in range(4)]
    assert cfg["text_config"]["media_token_ids"] == cfg["media_token_ids"]
    assert cfg["image_token_id"] == cfg["image_token_index"] == 129264


def test_image_token_id_falls_back_to_the_vocab_then_the_pin():
    mm = dict(MM_META)
    del mm[f"{ARCH}.image.token_id"]
    cfg = synthesize_vlm_config(
        "deepseek_v41_vl", _deepseek41_meta(), _DEEPSEEK41_SHAPES, mm)
    assert cfg["image_token_id"] == DEEPSEEK41V_IMAGE_TOKEN_ID
    meta = dict(_deepseek41_meta())
    meta["tokenizer.ggml.tokens"] = ["a", "b", DEEPSEEK41V_IMAGE_TOKEN, "c"]
    cfg = synthesize_vlm_config(
        "deepseek_v41_vl", meta, _DEEPSEEK41_SHAPES, mm)
    assert cfg["image_token_id"] == cfg["image_token_index"] == 2


def test_container_build_carries_media_ids_and_sanitize():
    cfg = _config()
    model, _ = build_vlm_model(cfg)
    assert type(model).__module__ == "gmlx.models.deepseek_v41.vlm_model"
    assert model.config.model_type == "deepseek_v41_vl"
    lm = model.language_model
    assert lm.config.model_type == "deepseek_v41"
    assert lm.config.media_token_ids == cfg["media_token_ids"]
    # sanitize: vision keys pass through, text keys go through the tower's
    tc = cfg["text_config"]
    groups, rank = tc["o_groups"], tc["o_lora_rank"]
    w = {"image_start": mx.zeros((4,)),
         "language_model.model.layers.0.attn.wo_a.weight":
             mx.zeros((groups * rank, tc["hidden_size"]))}
    out = model.sanitize(w)
    assert out["image_start"].shape == (4,)
    assert out["language_model.model.layers.0.attn.wo_a.weight"].shape == (
        groups, rank, tc["hidden_size"])


def _tokenizer():
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    bos = "<\uff5cbegin\u2581of\u2581sentence\uff5c>"
    vocab = {"[UNK]": 0, "hi": 1, "there": 2, "what": 3, "is": 4,
             "this": 5, bos: 6, DEEPSEEK41V_IMAGE_TOKEN: 7}
    tok = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.add_special_tokens([bos, DEEPSEEK41V_IMAGE_TOKEN])
    return PreTrainedTokenizerFast(
        tokenizer_object=tok, bos_token=bos, unk_token="[UNK]"), bos


def _processor():
    tok, bos = _tokenizer()
    proc = _synthesize_deepseek41v_processor(
        tok, MM_META, {"vocab_size": 8, "image_token_id": 7,
                       "vision_config": {"patch_size": 14,
                                         "downsample_ratio": 3,
                                         "max_n_token": 1024,
                                         "min_pixels": 295936,
                                         "max_wh_ratio": None}})
    return proc, bos


def _image(w=300, h=200, seed=0):
    from PIL import Image
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 255, (h, w, 3), dtype=np.uint8))


def test_processor_expands_placeholder_into_a_reading_order_block():
    proc, bos = _processor()
    text = f"{bos}what is {DEEPSEEK41V_IMAGE_TOKEN} this"
    feat = proc(images=[_image()], text=text)
    ids = [int(t) for t in feat["input_ids"][0].tolist()]
    meta = feat["image_meta"]
    assert meta.shape == (1, 4)
    n_vit_h, n_vit_w, n_llm_h, n_llm_w = (int(v) for v in meta[0].tolist())
    assert feat["pixel_values"].shape == (n_vit_h * n_vit_w, 3 * 14 * 14)
    assert n_llm_h == -(-n_vit_h // 3) and n_llm_w == -(-n_vit_w // 3)
    types, perm = ib.build_image_block(n_llm_h, n_llm_w)
    assert ids[:3] == [6, 3, 4]
    assert ids[3:3 + len(types)] == [8 + int(t) for t in types]
    assert ids[3 + len(types):] == [5]
    assert list(perm) == list(range(n_llm_h * n_llm_w))   # reading order
    assert "image_spans" not in feat        # causal attention needs no spans
    assert ids.count(6) == 1                # exactly one BOS
    assert feat["attention_mask"].shape == (1, len(ids))
    assert len(types) == ib.block_length(n_llm_h, n_llm_w) <= 1024


def test_processor_text_only_and_mismatches():
    proc, bos = _processor()
    feat = proc(text=f"{bos}hi there")
    assert "pixel_values" not in feat
    assert [int(t) for t in feat["input_ids"][0].tolist()] == [6, 1, 2]
    with pytest.raises(ValueError, match="placeholders"):
        proc(images=[_image()],
             text=f"{bos}{DEEPSEEK41V_IMAGE_TOKEN} {DEEPSEEK41V_IMAGE_TOKEN}")
    with pytest.raises(ValueError, match="placeholders"):
        proc(images=[_image(), _image(seed=1)],
             text=f"{bos}{DEEPSEEK41V_IMAGE_TOKEN} hi")
    with pytest.raises(ValueError, match="single-row"):
        proc(images=[_image()],
             text=[f"{bos}{DEEPSEEK41V_IMAGE_TOKEN}", f"{bos}hi"])


def test_processor_two_images_stack_in_order():
    proc, bos = _processor()
    text = f"{bos}{DEEPSEEK41V_IMAGE_TOKEN} hi {DEEPSEEK41V_IMAGE_TOKEN}"
    feat = proc(images=[_image(), _image(w=200, h=300, seed=1)], text=text)
    ids = [int(t) for t in feat["input_ids"][0].tolist()]
    assert feat["image_meta"].shape == (2, 4)
    n = sum(int(m[0]) * int(m[1]) for m in feat["image_meta"].tolist())
    assert feat["pixel_values"].shape[0] == n
    starts = [i for i, t in enumerate(ids) if t == 8 + ib.IMAGE_START]
    ends = [i for i, t in enumerate(ids) if t == 8 + ib.IMAGE_END]
    assert len(starts) == len(ends) == 2 and starts[1] == ends[0] + 2


# --- geometry against a literal port of the reference ----------------------

def _ref_num_image_tokens(n_llm_h, n_llm_w):
    return n_llm_h * (n_llm_w + 1) + 2


def _ref_llm_grid(best_height, best_width, patch_size, downsample_ratio):
    return (math.ceil((best_height // patch_size) / downsample_ratio),
            math.ceil((best_width // patch_size) / downsample_ratio))


def _ref_solve_resize_ratio(height, width, patch_size, downsample_ratio,
                            max_n_token):
    r = height / width
    max_w_float = math.sqrt((max_n_token - 2) / r + 0.25) - 0.5
    max_h_float = max_w_float * r
    cell = patch_size * downsample_ratio
    if max_w_float < 1.0:
        return (max_n_token - 2) // 2 * cell, cell
    if max_h_float < 1.0:
        return cell, (max_n_token - 3) * cell
    beta = min(math.floor(max_w_float) * cell / width,
               math.floor(max_h_float) * cell / height)
    return (math.floor(height * beta / patch_size) * patch_size,
            math.floor(width * beta / patch_size) * patch_size)


def _ref_plan(width, height, *, patch_size, downsample_ratio, max_n_token,
              min_pixels, max_wh_ratio):
    p = patch_size
    if max_wh_ratio is not None and width > height * max_wh_ratio:
        width = height * max_wh_ratio
    if 0 < width * height < min_pixels:
        ratio = (min_pixels / (width * height)) ** 0.5
        width = int(width * ratio)
        height = int(height * ratio)
    best_width = math.ceil(width / p) * p
    best_height = math.ceil(height / p) * p
    n_llm_h, n_llm_w = _ref_llm_grid(best_height, best_width, p,
                                     downsample_ratio)
    if _ref_num_image_tokens(n_llm_h, n_llm_w) > max_n_token:
        best_height, best_width = _ref_solve_resize_ratio(
            height, width, p, downsample_ratio, max_n_token)
        n_llm_h, n_llm_w = _ref_llm_grid(best_height, best_width, p,
                                         downsample_ratio)
        assert _ref_num_image_tokens(n_llm_h, n_llm_w) <= max_n_token
    return n_llm_h, n_llm_w, best_height, best_width


_SIZES = [(w, h) for w in (64, 300, 640, 1024, 2000, 4000, 8000)
          for h in (48, 200, 480, 1024, 2000, 4000, 8000)]


@pytest.mark.parametrize("width,height", _SIZES)
def test_plan_image_grid_matches_the_reference(width, height):
    kw = dict(patch_size=14, downsample_ratio=3, max_n_token=1024,
              min_pixels=295936, max_wh_ratio=None)
    got = ib.plan_image_grid(width, height, **kw)
    assert got == _ref_plan(width, height, **kw)
    n_llm_h, n_llm_w = got[0], got[1]
    assert ib.num_image_tokens(n_llm_h, n_llm_w) <= 1024


def test_block_layout_matches_the_reference():
    types, perm = ib.build_image_block(2, 3)
    assert list(types) == [ib.IMAGE_START,
                           ib.IMAGE, ib.IMAGE, ib.IMAGE, ib.IMAGE_NEW_LINE,
                           ib.IMAGE, ib.IMAGE, ib.IMAGE, ib.IMAGE_NEW_LINE,
                           ib.IMAGE_END]
    assert list(perm) == list(range(6))
    assert len(types) == ib.num_image_tokens(2, 3) == 10


def test_pixels_normalize_with_the_reference_constants():
    proc, _ = _processor()
    ip = proc.image_processor
    from PIL import Image
    img = Image.fromarray(np.stack([np.full((420, 420), v, np.uint8)
                                    for v in (0, 127, 255)], axis=-1))
    patches, meta = ip._one(img)
    assert meta[0] == meta[1] and meta[2] == meta[3]   # square source
    row = patches[0]
    # (C, ph, pw) order: one 196-value run per channel, scaled (x/255-.5)/.5
    assert np.allclose(row[:196], -1.0) and np.allclose(row[392:], 1.0)
    assert np.allclose(row[196:392], (127 / 255 - 0.5) / 0.5)


@pytest.mark.skipif(not os.path.exists(MMPROJ), reason="local mmproj absent")
def test_real_mmproj_remaps_onto_the_tower():
    from gmlx.load.wire import load_gguf_wire_bytes
    arrays, _codecs, _arch, mm_meta, _shapes = load_gguf_wire_bytes(
        MMPROJ, zero_copy=True, expect_quant=False)
    out, skipped, kq = remap_vision_arrays(arrays, "deepseek_v41_vl")
    assert skipped == [] and kq == {}
    cfg = synthesize_vlm_config(
        "deepseek_v41_vl", _deepseek41_meta(), _DEEPSEEK41_SHAPES, mm_meta)
    vc = cfg["vision_config"]
    assert (vc["depth"], vc["hidden_size"], vc["num_heads"]) == (32, 1024, 16)
    assert vc["out_hidden_size"] == 5120 and vc["downsample_ratio"] == 3
    assert vc["max_n_token"] == 1024 and vc["max_wh_ratio"] is None
    assert out["vision_tower.patch_embed.proj.weight"].shape == (1024, 588)
    assert out["vision_tower.blocks.31.attn.q_proj.weight"].shape == (1024, 1024)
    assert out["vision_tower.blocks.31.mlp.up_proj.weight"].shape == (2816, 1024)
    assert out["vision_tower.aligner.w1.weight"].shape == (5120, 9216)
    assert out["image_newline"].shape == (5120,)
    assert len(out) == 426
