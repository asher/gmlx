"""Hy4-flash vision loader plumbing: the mmproj remap (three convolutions
as Linears), the VLM config synth, the GGUF-only processor (resize rule and
slot expansion), and the container build. CPU-only, synthetic metadata
with the released mmproj's tensor names."""

import os
import sys

import mlx.core as mx
import mlx.utils as mu
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))

from gmlx.load.vlm import (
    HY4FLASH_IMAGE_PAD_PREFIX,
    HY4FLASH_IMAGE_PAD_TOKEN_ID,
    HY4FLASH_VISION,
    UnsupportedVLMError,
    _HyV4FlashGgufImageProcessor,
    _synthesize_hy4flash_processor,
    build_vlm_model,
    remap_vision_arrays,
    resolve_vlm_model_type,
    synthesize_vlm_config,
)
from gmlx.models.hy_v4_flash import positions as rp

from test_config_synth import _HYF_SHAPES, _hy_v4_flash_meta

DEPTH, DIM, FFN, MERGE, OUT, PATCH = 2, 16, 24, 20, 32, 16

# The released mmproj's keys, at a small size.
MM_META = {
    "general.architecture": "clip",
    "general.type": "mmproj",
    "clip.has_vision_encoder": True,
    "clip.projector_type": "hy_v4_flash",
    "clip.use_gelu": True,
    "clip.vision.projection_dim": OUT,
    "clip.vision.image_size": 2048,
    "clip.vision.patch_size": PATCH,
    "clip.vision.embedding_length": DIM,
    "clip.vision.feed_forward_length": FFN,
    "clip.vision.block_count": DEPTH,
    "clip.vision.attention.head_count": 2,
    "clip.vision.attention.layer_norm_epsilon": 1e-5,
    "clip.vision.spatial_merge_size": 2,
    "clip.vision.image_min_pixels": 4096,
    "clip.vision.image_max_pixels": 4194304,
    "clip.vision.image_mean": [0.5, 0.5, 0.5],
    "clip.vision.image_std": [0.5, 0.5, 0.5],
}

# Every tensor name of the released file: nine at the top, sixteen a block.
_BLOCK = ("attn_q", "attn_k", "attn_v", "attn_out", "ln1", "ln2", "ffn_up",
          "ffn_down")


def _mmproj_arrays(seed=7):
    rng = np.random.default_rng(seed)

    def arr(*shape):
        return mx.array(rng.standard_normal(shape).astype(np.float32))

    a = {
        "v.patch_embd.weight": arr(DIM, 3, PATCH, PATCH),
        "v.patch_embd.bias": arr(DIM),
        "v.position_embd.weight": arr(128 * 128, DIM),
        "mm.pre_norm.weight": arr(DIM),
        "mm.0.weight": arr(MERGE, DIM, 2, 2),
        "mm.0.bias": arr(MERGE),
        "mm.2.weight": arr(OUT, MERGE, 1, 1),
        "mm.2.bias": arr(OUT),
        "v.image_newline": arr(OUT),
    }
    shapes = {"attn_q": (DIM, DIM), "attn_k": (DIM, DIM),
              "attn_v": (DIM, DIM), "attn_out": (DIM, DIM),
              "ln1": (DIM,), "ln2": (DIM,), "ffn_up": (FFN, DIM),
              "ffn_down": (DIM, FFN)}
    for i in range(DEPTH):
        for sub in _BLOCK:
            a[f"v.blk.{i}.{sub}.weight"] = arr(*shapes[sub])
            a[f"v.blk.{i}.{sub}.bias"] = arr(shapes[sub][0])
    return a


def _llm_meta():
    return dict(_hy_v4_flash_meta())


def _config(llm_meta=None):
    from gmlx.models.hy_v4_flash.vlm_model import ensure_registered

    ensure_registered()
    arrays = _mmproj_arrays()
    return synthesize_vlm_config(
        HY4FLASH_VISION, llm_meta or _llm_meta(), _HYF_SHAPES, MM_META,
        mm_shapes={k: tuple(v.shape) for k, v in arrays.items()})


def test_the_projector_resolves_onto_the_container():
    assert resolve_vlm_model_type("hy_v4_flash", MM_META) == HY4FLASH_VISION
    with pytest.raises(UnsupportedVLMError, match="hy_v4_flash.*'hyv4'"):
        resolve_vlm_model_type("hyv4", MM_META)


def test_config_synth_fields():
    cfg = _config()
    assert cfg["model_type"] == "hy_v4_flash_vl"
    assert cfg["text_config"]["model_type"] == "hy_v4_flash"
    vc = cfg["vision_config"]
    assert (vc["depth"], vc["hidden_size"], vc["num_heads"]) == (DEPTH, DIM, 2)
    assert vc["intermediate_size"] == FFN and vc["out_hidden_size"] == OUT
    assert vc["patch_size"] == PATCH and vc["spatial_merge_size"] == 2
    # The merge width is not a key: it is the merge bias's length.
    assert vc["merge_hidden_size"] == MERGE
    assert vc["position_grid"] == 128
    assert vc["layer_norm_eps"] == pytest.approx(1e-5)
    vocab = cfg["vocab_size"]
    assert vocab == cfg["text_config"]["vocab_size"]
    assert cfg["media_token_ids"] == [vocab + k for k in range(3)]


def test_the_image_pad_id_is_found_by_prefix_then_pinned():
    # The token text ends in a revision tag that differs between
    # conversions, so the id is found by what comes before it.
    assert _config()["image_token_id"] == HY4FLASH_IMAGE_PAD_TOKEN_ID
    meta = _llm_meta()
    meta["tokenizer.ggml.tokens"] = [
        "a", "b", HY4FLASH_IMAGE_PAD_PREFIX + "6124c78e\uff5c>", "c"]
    cfg = _config(meta)
    assert cfg["image_token_id"] == cfg["image_token_index"] == 2


def test_remap_covers_the_whole_tower_and_nothing_else():
    arrays = _mmproj_arrays()
    out, skipped, kq = remap_vision_arrays(arrays, HY4FLASH_VISION)
    assert skipped == [] and kq == {}

    model, _ = build_vlm_model(_config())
    params = dict(mu.tree_flatten(model.parameters()))
    vision = sorted(k for k in params if not k.startswith("language_model."))
    assert sorted(out) == vision
    for k in vision:
        assert tuple(out[k].shape) == tuple(params[k].shape), k

    # A convolution kernel [out, c, kh, kw] is the Linear's row as it lies.
    for src, dst in (
            ("v.patch_embd.weight", "vision_tower.patch_embed.proj.weight"),
            ("mm.0.weight", "vision_tower.projector.merge.weight"),
            ("mm.2.weight", "vision_tower.projector.out.weight")):
        want = arrays[src].reshape(arrays[src].shape[0], -1)
        assert mx.array_equal(out[dst], want), src


@pytest.fixture
def _on_cpu():
    # Exact fp32: a GPU matmul and a GPU convolution round apart.
    prev = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(prev)


def test_the_loaded_tower_is_the_file_s_convolutions(_on_cpu):
    """The patch embed and the merge, run as Linears on the remapped
    weights, give what the file's convolutions give on an image."""
    arrays = _mmproj_arrays()
    out, _, _ = remap_vision_arrays(arrays, HY4FLASH_VISION)
    model, _ = build_vlm_model(_config())
    tower = model.vision_tower
    tower.load_weights(
        [(k[len("vision_tower."):], v) for k, v in out.items()])

    ip = _HyV4FlashGgufImageProcessor(patch_size=PATCH)
    feat = ip([_image(96, 64)])
    n_h, n_w = (int(v) for v in feat["image_patch_grid"][0])
    assert (n_h, n_w) == (4, 6)
    patches = mx.array(feat["pixel_values"])

    # The image the patches were cut from, [1, H, W, C].
    pixels = patches.reshape(n_h, n_w, 3, PATCH, PATCH).transpose(
        0, 3, 1, 4, 2).reshape(1, n_h * PATCH, n_w * PATCH, 3)
    kernel = arrays["v.patch_embd.weight"].transpose(0, 2, 3, 1)
    want = mx.conv2d(pixels, kernel, stride=PATCH) + arrays["v.patch_embd.bias"]
    got = tower.patch_embed.proj(patches)
    assert np.allclose(np.array(got), np.array(want).reshape(n_h * n_w, DIM),
                       atol=1e-4)

    mx.random.seed(3)
    x = mx.random.normal((n_h * n_w, DIM))
    merged = mx.conv2d(
        tower.projector.pre_norm(x).reshape(1, n_h, n_w, DIM),
        arrays["mm.0.weight"].transpose(0, 2, 3, 1),
        stride=2) + arrays["mm.0.bias"]
    import mlx.nn as nn

    want = mx.conv2d(nn.gelu(merged),
                     arrays["mm.2.weight"].transpose(0, 2, 3, 1))
    want = want + arrays["mm.2.bias"]
    assert np.allclose(np.array(tower.projector(x, n_h, n_w)),
                       np.array(want[0]), atol=1e-4)


def test_container_build_and_sanitize_drop_the_draft_block():
    cfg = _config()
    model, _ = build_vlm_model(cfg)
    assert type(model).__module__ == "gmlx.models.hy_v4_flash.vlm_model"
    lm = model.language_model
    assert lm.config.model_type == "hy_v4_flash"
    assert lm.image_token_id == cfg["image_token_id"]
    n = cfg["text_config"]["num_hidden_layers"]
    w = {"vision_tower.image_newline": mx.zeros((OUT,)),
         "language_model.model.layers.0.self_attn.sinks": mx.zeros((2,)),
         f"language_model.model.layers.{n}.self_attn.sinks": mx.zeros((2,))}
    assert sorted(model.sanitize(w)) == [
        "language_model.model.layers.0.self_attn.sinks",
        "vision_tower.image_newline"]


def test_the_engram_table_streams_under_the_text_tower():
    import gmlx.stream.table_stream as ts

    model, _ = build_vlm_model(_config())
    tabs = ts.streamable_tables_for(model)
    assert [(t.param_path, t.gguf_name) for t, _ in tabs] == [
        ("language_model.model.engram_embed", "per_layer_token_embd.weight")]


# The reference's smart_resize on its own examples: input (w, h) -> the
# resized (w, h). 80 px rounds half to even (2.5 -> 2), an edge shorter
# than half a cell takes the minimum-pixels branch, and a large image is
# scaled into the pixel budget.
@pytest.mark.parametrize("size,resized", [
    ((64, 64), (64, 64)),
    ((640, 480), (640, 480)),
    ((1920, 1080), (1920, 1088)),
    ((2048, 2048), (2048, 2048)),
    ((4032, 3024), (2336, 1760)),
    ((8000, 400), (8000, 384)),
    ((80, 80), (64, 64)),
    ((200, 10), (288, 32)),
])
def test_the_resize_rule(size, resized):
    ip = _HyV4FlashGgufImageProcessor()
    h_bar, w_bar = ip.resized(*size)
    assert (w_bar, h_bar) == resized
    assert h_bar % 32 == 0 and w_bar % 32 == 0


def test_the_serve_estimate_bounds_the_largest_image():
    from gmlx.serve.estimate import _MEDIA_TOKENS_PER_IMAGE

    ip = _HyV4FlashGgufImageProcessor()
    # The tallest grid the pixel budget and the aspect cap allow: 5 cells
    # wide, 819 rows.
    h_bar, w_bar = ip.resized(160, 26208)
    assert (h_bar // 32, w_bar // 32) == (819, 5)
    most = 819 * 6
    for w in (33, 64, 128, 143, 160, 175, 209, 512, 2048, 4032, 9000):
        for h in (w, 3 * w, 40 * w, 150 * w, 164 * w, 200 * w):
            h_bar, w_bar = ip.resized(w, h)
            most = max(most, (h_bar // 32) * (w_bar // 32 + 1))
    assert most <= _MEDIA_TOKENS_PER_IMAGE[HY4FLASH_VISION]


def test_an_extreme_aspect_ratio_is_refused():
    with pytest.raises(ValueError, match="aspect ratio"):
        _HyV4FlashGgufImageProcessor().resized(8000, 30)


def _tokenizer():
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    bos = "<bos>"
    pad = HY4FLASH_IMAGE_PAD_PREFIX + "x\uff5c>"
    vocab = {"[UNK]": 0, "hi": 1, "there": 2, "what": 3, "is": 4,
             "this": 5, bos: 6, pad: 7, "<img>": 8, "</img>": 9}
    tok = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.add_special_tokens([bos, pad, "<img>", "</img>"])
    return PreTrainedTokenizerFast(
        tokenizer_object=tok, bos_token=bos, unk_token="[UNK]"), bos, pad


def _processor():
    tok, bos, pad = _tokenizer()
    proc = _synthesize_hy4flash_processor(
        tok, MM_META, {"vocab_size": 10, "image_token_id": 7})
    return proc, bos, pad


def _image(w=300, h=200, seed=0):
    from PIL import Image

    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 255, (h, w, 3), dtype=np.uint8))


def test_processor_reads_its_settings_from_the_mmproj():
    proc, _, _ = _processor()
    ip = proc.image_processor
    assert (ip.patch_size, ip.merge_size, ip.min_pixels, ip.max_pixels) == (
        16, 2, 4096, 4194304)
    assert ip.image_mean == [0.5] * 3 and ip.image_std == [0.5] * 3


def test_processor_expands_each_pad_into_its_image_s_slots():
    proc, bos, pad = _processor()
    text = f"{bos}what is <img>{pad}</img> this <img>{pad}</img>"
    # 300 x 200 -> 288 x 192 (9 x 6 cells); 64 x 96 -> 2 x 3 cells.
    feat = proc(images=[_image(300, 200), _image(64, 96, seed=1)], text=text)
    grids = [[int(v) for v in row] for row in
             feat["image_patch_grid"].tolist()]
    assert grids == [[12, 18], [6, 4]]
    assert feat["pixel_values"].shape == (12 * 18 + 6 * 4, 3 * 16 * 16)

    ids = [int(t) for t in feat["input_ids"][0].tolist()]
    first = [10 + k for k in rp.image_slot_kinds(6, 9)]
    second = [10 + k for k in rp.image_slot_kinds(3, 2)]
    assert ids == [6, 3, 4, 8] + first + [9, 5, 8] + second + [9]
    assert len(first) == 6 * 10 and len(second) == 3 * 3
    assert 7 not in ids
    # Pixels land in [-1, 1].
    pv = np.array(feat["pixel_values"])
    assert pv.min() >= -1.0 and pv.max() <= 1.0


def test_a_placeholder_count_that_differs_from_the_images_is_refused():
    proc, bos, pad = _processor()
    with pytest.raises(ValueError, match="placeholders"):
        proc(images=[_image()], text=f"{bos}{pad} {pad}")
    with pytest.raises(ValueError, match="placeholders"):
        proc(images=[_image(), _image()], text=f"{bos}{pad}")


def test_the_cache_salt_counts_an_image_s_whole_run():
    from gmlx.cache.media_salt import _expected_tokens, _patch_counts

    raw = {"image_patch_grid": np.array([[12, 18], [6, 4]])}
    assert _patch_counts(raw).tolist() == [12 * 18, 6 * 4]
    counts, whole_run = _expected_tokens(raw, 2, {}, None)
    assert counts == [6 * 10, 3 * 3] and whole_run
