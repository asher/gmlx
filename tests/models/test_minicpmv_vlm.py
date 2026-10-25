"""MiniCPM-V 4.6 / 4.7: the mmproj remap, config synth, the image processor,
the 4.7 canvas positions and the vision tower. CPU only, with no GGUF
and no weights.

Expectations come from the transformers reference at commit 674c0e3
(``minicpmv4_6`` image processor and processor, ``minicpmv4_7``
``get_rope_index``), not from this port. The canvas cases are the
reference's own output, recorded by ``scripts/minicpmv_canvas_fixtures.py``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from mlx.utils import tree_flatten  # noqa: E402

from gmlx.load import vlm as gvlm  # noqa: E402
from gmlx.load.vlm import (  # noqa: E402
    UnsupportedVLMError,
    _synthesize_minicpmv_vlm_config,
    remap_vision_arrays,
)
from gmlx.models.minicpmv.language import TextConfig  # noqa: E402
from gmlx.models.minicpmv.processing import (  # noqa: E402
    MiniCPMVImageProcessor,
)
from gmlx.models.minicpmv.rope import (  # noqa: E402
    CanvasTokens,
    canvas_rope_index,
    torch_linspace_round,
)
from gmlx.models.minicpmv.vision import (  # noqa: E402
    MLP,
    MergerBlock,
    VisionConfig,
    VitMerger,
    position_bucket_ids,
)
from gmlx.models.minicpmv.vlm_model import Model, ModelConfig  # noqa: E402

_FIXTURE = (Path(__file__).parents[1] / "fixtures"
            / "minicpmv_canvas_positions.json")

# Small ids for the tiny model; the real vocab's are in the fixture.
_IDS = dict(image_token_id=500, image_start_id=501, image_end_id=502,
            slice_start_id=503, slice_end_id=504, image_id_start_id=505,
            image_id_end_id=506, newline_id=198)
PAD, IMG, IMG_END, SLICE, SLICE_END = 500, 501, 502, 503, 504


def _text_dict(model_type="qwen3_5"):
    d = dict(
        model_type=model_type, hidden_size=64, intermediate_size=128,
        linear_num_value_heads=4, linear_num_key_heads=2,
        linear_key_head_dim=16, linear_value_head_dim=16,
        linear_conv_kernel_dim=4, num_hidden_layers=4, num_attention_heads=4,
        rms_norm_eps=1e-6, vocab_size=600, num_key_value_heads=2,
        max_position_embeddings=2048, head_dim=16, full_attention_interval=2,
        rope_parameters={"type": "default", "mrope_section": [2, 1, 1],
                         "rope_theta": 100000, "partial_rotary_factor": 0.5})
    if model_type == "qwen3_5_moe":
        d.update(num_experts=4, num_experts_per_tok=2,
                 shared_expert_intermediate_size=32, moe_intermediate_size=32)
    return d


def _vision(**over):
    kw = dict(hidden_size=32, intermediate_size=48, num_hidden_layers=3,
              num_attention_heads=2, insert_layer_id=1,
              window_intermediate_size=40, out_hidden_size=64,
              position_grid=8)
    kw.update(over)
    return VisionConfig(**kw)


def _model(mode="16x", canvas=True, sep=True, text="qwen3_5"):
    mx.random.seed(3)
    cfg = ModelConfig(
        text_config=TextConfig.from_dict(_text_dict(text)),
        vision_config=_vision(downsample_mode=mode),
        mrope_mode="canvas" if canvas else None, sep_embeds=sep,
        vocab_size=600, **_IDS)
    model = Model(cfg)
    if sep:
        model.sep_embeds = mx.random.normal((4, 64))
    mx.eval(model.parameters())
    return model



def _mmproj_arrays(vc: VisionConfig, *, vit_merger=True, sep=True):
    """mmproj tensors in numpy order, as the GGUF loader hands them over."""
    d, ff, out = vc.hidden_size, vc.intermediate_size, vc.out_hidden_size
    p = vc.patch_size
    z = np.zeros
    a = {"v.patch_embd.weight": z((d, 3, p, p)), "v.patch_embd.bias": z(d),
         "v.position_embd.weight": z((vc.position_grid ** 2, d)),
         "v.post_ln.weight": z(d), "v.post_ln.bias": z(d),
         "mm.input_norm.weight": z(4 * d), "mm.input_norm.bias": z(4 * d),
         "mm.up.weight": z((4 * d, 4 * d)), "mm.up.bias": z(4 * d),
         "mm.down.weight": z((out, 4 * d)), "mm.down.bias": z(out)}
    for i in range(vc.num_hidden_layers):
        for n, shape in (("attn_q", (d, d)), ("attn_k", (d, d)),
                         ("attn_v", (d, d)), ("attn_out", (d, d)),
                         ("ffn_up", (ff, d)), ("ffn_down", (d, ff))):
            a[f"v.blk.{i}.{n}.weight"] = z(shape)
            a[f"v.blk.{i}.{n}.bias"] = z(shape[0])
        for n in ("ln1", "ln2"):
            a[f"v.blk.{i}.{n}.weight"] = z(d)
            a[f"v.blk.{i}.{n}.bias"] = z(d)
    if vit_merger:
        wi = vc.window_intermediate_size
        for n, shape in (("attn_q", (d, d)), ("attn_k", (d, d)),
                         ("attn_v", (d, d)), ("attn_out", (d, d)),
                         ("ds_ffn_up", (wi, 4 * d)),
                         ("ds_ffn_down", (d, wi))):
            a[f"v.vit_merger.{n}.weight"] = z(shape)
            a[f"v.vit_merger.{n}.bias"] = z(shape[0])
        for n, w in (("ln1", d), ("ds_ln", 4 * d)):
            a[f"v.vit_merger.{n}.weight"] = z(w)
            a[f"v.vit_merger.{n}.bias"] = z(w)
    if sep:
        a["v.tok_embd_sep"] = z((4, out))
    return {k: mx.array(v, dtype=mx.float32) for k, v in a.items()}


@pytest.mark.parametrize("mode", ["16x", "4x"])
def test_every_mmproj_tensor_lands_on_a_real_parameter(mode):
    model = _model(mode)
    arrays = _mmproj_arrays(model.config.vision_config,
                            vit_merger=mode == "16x")
    out, skipped, _ = remap_vision_arrays(arrays, "minicpmv4_7")
    assert skipped == []
    params = dict(tree_flatten(model.parameters()))
    assert set(out) <= set(params)
    for k, v in out.items():
        assert tuple(v.shape) == tuple(params[k].shape), k
    tower = {k for k in params if not k.startswith("language_model.")}
    assert tower == set(out)


def test_patch_conv_is_moved_to_the_mlx_layout():
    w = mx.arange(2 * 3 * 2 * 2, dtype=mx.float32).reshape(2, 3, 2, 2)
    out, _, _ = remap_vision_arrays({"v.patch_embd.weight": w}, "minicpmv4_6")
    got = out["vision_tower.embeddings.patch_embedding.weight"]
    assert got.shape == (2, 2, 2, 3)
    # [out, C, kh, kw] -> [out, kh, kw, C]
    assert float(got[1, 0, 1, 2]) == float(w[1, 2, 0, 1])


def test_unknown_tensor_is_skipped_not_guessed():
    arrays = {"v.blk.0.mystery.weight": mx.zeros((2,)),
              "v.vit_merger.mystery.weight": mx.zeros((2,))}
    out, skipped, _ = remap_vision_arrays(arrays, "minicpmv4_7")
    assert out == {} and sorted(skipped) == sorted(arrays)



_MM_META = {
    "general.type": "mmproj",
    "clip.projector_type": "minicpmv4_7",
    "clip.vision.projection_dim": 2048,
    "clip.vision.image_size": 448,
    "clip.vision.patch_size": 14,
    "clip.vision.embedding_length": 1152,
    "clip.vision.feed_forward_length": 4304,
    "clip.vision.block_count": 27,
    "clip.vision.attention.head_count": 16,
    "clip.vision.image_mean": [0.5, 0.5, 0.5],
    "clip.vision.image_std": [0.5, 0.5, 0.5],
    "clip.vision.projector.scale_factor": 4,
    "clip.vision.max_slice_nums": 9,
    "clip.vision.wa_layer_indexes": [6],
    "clip.vision.attention.layer_norm_epsilon": 1e-6,
}
_MARKERS = ["<|image_pad|>", "<image>", "</image>", "<slice>", "</slice>",
            "<image_id>", "</image_id>", "<|video_pad|>"]


def _llm_meta(tokens=None, order="zyxt"):
    toks = tokens if tokens is not None else (
        ["x"] * 198 + ["\u010a"] + _MARKERS)
    meta = {"general.architecture": "qwen35moe",
            "tokenizer.ggml.tokens": toks}
    if order is not None:
        meta["qwen35moe.rope.section_order"] = order
    return meta


_SHAPES = {"v.position_embd.weight": [1152, 4900],
           "v.vit_merger.ds_ffn_up.weight": [4608, 17216]}


def _synth(mm=None, llm=None, model_type="minicpmv4_7", names=None):
    return _synthesize_minicpmv_vlm_config(
        {"model_type": "qwen3_5_moe", "vocab_size": 248320},
        mm or _MM_META, model_type, llm or _llm_meta(), _SHAPES,
        names if names is not None else {"v.tok_embd_sep"})


def test_config_reads_the_mmproj_metadata_and_shapes():
    cfg = _synth()
    vc = cfg["vision_config"]
    assert (vc["hidden_size"], vc["num_hidden_layers"], vc["out_hidden_size"],
            vc["position_grid"], vc["insert_layer_id"],
            vc["window_intermediate_size"], vc["downsample_mode"]) == (
        1152, 27, 2048, 70, 6, 17216, "16x")
    assert cfg["mrope_mode"] == "canvas" and cfg["sep_embeds"] is True
    assert cfg["text_config"]["model_type"] == "qwen3_5_moe"


def test_marker_ids_come_from_the_vocab_with_the_byte_level_newline():
    cfg = _synth()
    assert cfg["newline_id"] == 198
    assert [cfg[k] for k in ("image_token_id", "image_start_id",
                             "image_end_id", "slice_start_id", "slice_end_id",
                             "image_id_start_id", "image_id_end_id",
                             "video_token_id")] == list(range(199, 207))


@pytest.mark.parametrize("missing", ["\u010a", "<slice>", "</image_id>"])
def test_a_missing_marker_is_refused(missing):
    toks = [t for t in ["x"] * 198 + ["\u010a"] + _MARKERS if t != missing]
    with pytest.raises(UnsupportedVLMError, match="marker"):
        _synth(llm=_llm_meta(toks))


def test_a_section_order_other_than_zyxt_is_refused():
    with pytest.raises(UnsupportedVLMError, match="section_order"):
        _synth(llm=_llm_meta(order="tyxz"))
    assert _synth(llm=_llm_meta(order=None))["image_token_id"] == 199


def test_a_4x_mmproj_builds_no_window_merger():
    mm = {**_MM_META, "clip.vision.projector.scale_factor": 2}
    vc = VisionConfig.from_dict(_synth(mm=mm)["vision_config"])
    assert vc.downsample_mode == "4x" and not vc.use_vit_merger
    assert vc.merge_factor == 2
    assert not hasattr(_model("4x").vision_tower, "vit_merger")


def test_46_positions_stay_sequential_and_4_6_has_no_slice_cap_key():
    mm = {k: v for k, v in _MM_META.items()
          if k != "clip.vision.max_slice_nums"}
    mm["clip.projector_type"] = "minicpmv4_6"
    assert "mrope_mode" not in _synth(mm=mm, model_type="minicpmv4_6")
    assert _processor(mm).image_processor.max_slice_nums == 9


@pytest.mark.parametrize("text", ["qwen3_5", "qwen3_5_moe"])
def test_text_config_follows_the_text_model_type(text):
    tc = TextConfig.from_dict(_text_dict(text))
    assert tc.model_type == text
    assert getattr(tc, "num_experts", 0) == (4 if text == "qwen3_5_moe" else 0)
    with pytest.raises(ValueError, match="not supported"):
        TextConfig.from_dict({**_text_dict(), "model_type": "llama"})



def _image(w, h, value=128):
    from PIL import Image

    return Image.fromarray(np.full((h, w, 3), value, dtype=np.uint8))


@pytest.mark.parametrize("w,h,grid", [
    (448, 448, None),       # one crop's worth: no slices
    (300, 200, None),
    (449, 448, [1, 2]),     # just over and a hair wide: two slices
    (600, 600, [2, 1]),     # a square ties 1x2 and 2x1; more rows win
    (1600, 600, [1, 4]),    # five crops' worth; 1x4 fits the 8:3 shape best
    (600, 1600, [4, 1]),
    (2000, 2000, [3, 3]),   # capped at max_slice_nums
])
def test_slice_grid_follows_the_reference_rule(w, h, grid):
    assert MiniCPMVImageProcessor().get_sliced_grid((h, w)) == grid


def test_resize_rounds_half_to_even():
    # 478 x 420: the width lands on 8.5 alignment units, which Python's
    # round takes to 8 (448 px), not 9 (504 px).
    ip = MiniCPMVImageProcessor()
    assert ip.find_best_resize((420, 478)) == (392, 448)
    assert ip.find_best_resize((200, 300), allow_upscale=True) == (392, 560)
    assert ip.find_best_resize((200, 300)) == (224, 280)


def test_crops_are_packed_patches_with_their_grids():
    ip = MiniCPMVImageProcessor()
    pre = ip.preprocess([_image(1600, 600), _image(300, 200)])
    sizes = pre["target_sizes"].tolist()
    assert pre["grids"] == [[1, 4], None]
    assert pre["crops_per_image"] == [5, 1]
    assert sum(h * w for h, w in sizes) == pre["pixel_values"].shape[0]
    assert pre["pixel_values"].shape[1] == 14 * 14 * 3
    assert all(h % 4 == 0 and w % 4 == 0 for h, w in sizes)
    # A flat mid-grey image normalizes to (128/255 - 0.5) / 0.5.
    np.testing.assert_allclose(pre["pixel_values"][:5],
                               (128 / 255 - 0.5) / 0.5, atol=1e-6)


def test_patch_rows_are_row_major_in_kh_kw_c_order():
    from PIL import Image

    ip = MiniCPMVImageProcessor(image_mean=[0, 0, 0], image_std=[1, 1, 1])
    arr = np.zeros((28, 28, 3), dtype=np.uint8)
    arr[0, 14, 1] = 255           # patch (0, 1), row 0, col 0, channel 1
    arr[15, 2, 2] = 255           # patch (1, 0), row 1, col 2, channel 2
    patches, size = ip._patches(Image.fromarray(arr))
    assert size == [2, 2]
    assert patches[1, 1] == 1.0
    assert patches[2, (1 * 14 + 2) * 3 + 2] == 1.0
    assert patches.sum() == 2.0


def _processor(mm=None):
    from tokenizers import pre_tokenizers

    from gmlx.load.tokenizer import load_tokenizer_from_gguf

    specials = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"] + _MARKERS
    tokens = specials + sorted(pre_tokenizers.ByteLevel.alphabet()) + ["He"]
    tok = load_tokenizer_from_gguf({
        "general.architecture": "qwen35",
        "tokenizer.ggml.model": "gpt2",
        "tokenizer.ggml.pre": "qwen2",
        "tokenizer.ggml.tokens": tokens,
        "tokenizer.ggml.merges": ["H e"],
        "tokenizer.ggml.token_type": [3] * len(specials)
        + [1] * (len(tokens) - len(specials)),
        "tokenizer.ggml.eos_token_id": 2,
        "tokenizer.ggml.padding_token_id": 0,
    }, "qwen35")
    return gvlm._synthesize_vlm_processor(
        "minicpmv4_7", tok, mm or _MM_META)


def _tid(proc, s):
    return proc.tokenizer.convert_tokens_to_ids(s)


def test_each_placeholder_expands_to_its_image_block():
    proc = _processor()
    out = proc(images=[_image(1600, 600)], text="<|image_pad|>He")
    ids = np.asarray(out["input_ids"]).reshape(-1).tolist()
    pad, nl = _tid(proc, "<|image_pad|>"), _tid(proc, "\u010a")
    ts = np.asarray(out["target_sizes"]).tolist()
    want = [_tid(proc, "<image_id>"), _tid(proc, "0"),
            _tid(proc, "</image_id>"), _tid(proc, "<image>")]
    want += [pad] * (ts[0][0] * ts[0][1] // 16) + [_tid(proc, "</image>")]
    for h, w in ts[1:]:
        want += ([_tid(proc, "<slice>")] + [pad] * (h * w // 16)
                 + [_tid(proc, "</slice>")])
    assert ids[:len(want)] == want
    assert nl not in ids          # one row of slices, no row break
    assert np.asarray(out["pixel_values"]).shape[0] == sum(
        h * w for h, w in ts)


def test_slice_rows_are_joined_by_newlines():
    proc = _processor()
    out = proc(images=[_image(600, 1600)], text="<|image_pad|>")
    ids = np.asarray(out["input_ids"]).reshape(-1).tolist()
    assert ids.count(_tid(proc, "\u010a")) == 3       # four rows


def test_image_ids_count_per_sample():
    proc = _processor()
    out = proc(images=[_image(64, 64), _image(64, 64), _image(64, 64)],
               text=["<|image_pad|>He<|image_pad|>", "<|image_pad|>"],
               padding=True)
    rows = np.asarray(out["input_ids"]).tolist()
    start = _tid(proc, "<image_id>")

    def numbers(row):
        return [row[i + 1] for i, t in enumerate(row) if t == start]

    assert numbers(rows[0]) == [_tid(proc, "0"), _tid(proc, "1")]
    assert numbers(rows[1]) == [_tid(proc, "0")]


def test_image_count_must_match_the_placeholders():
    proc = _processor()
    with pytest.raises(ValueError, match="placeholder"):
        proc(images=[_image(64, 64)], text="He")
    with pytest.raises(ValueError, match="placeholder"):
        proc(images=None, text="<|image_pad|>")


def test_video_is_refused():
    with pytest.raises(ValueError, match="video"):
        _processor()(text="He", videos=[np.zeros((2, 8, 8, 3))])



_REAL = CanvasTokens(248056, 248078, 248079, 248088, 248089, 198)
_CASES = json.loads(_FIXTURE.read_text())["cases"]


@pytest.mark.parametrize("case", _CASES, ids=[c["name"] for c in _CASES])
def test_canvas_positions_match_the_reference(case):
    pad = case["pad"]
    ids = np.array([[0] * pad + case["input_ids"]])
    mask = np.array([[0] * pad + [1] * len(case["input_ids"])])
    merge = 2 if case["downsample_mode"] == "4x" else 4
    pos, deltas = canvas_rope_index(
        ids, case["target_sizes"] or None, _REAL, merge=merge,
        attention_mask=mask if pad else None)
    assert pos[:, 0].tolist() == case["position_ids"]
    # The reference counts deltas against the kept tokens; the qwen3.5
    # decode adds them to the padded row length.
    assert int(deltas[0, 0]) == case["rope_deltas"] - pad


def test_fixture_covers_the_float32_rounding_pairs():
    names = {c["name"] for c in _CASES}
    assert {"linspace_48_23", "linspace_90_39", "linspace_96_47"} <= names
    # torch.linspace(0, c - 1, g).round() on the CPU, not exact rounding.
    assert int(torch_linspace_round(47, 23)[11]) == 23
    assert int(torch_linspace_round(89, 39)[19]) == 45
    assert int(torch_linspace_round(95, 47)[23]) == 47
    # Each sits on an exact half that ties-to-even takes the other way.
    assert [round(47 * 11 / 22), round(89 * 19 / 38),
            round(95 * 23 / 46)] == [24, 44, 48]


def test_two_rows_take_their_crops_in_batch_order():
    a = next(c for c in _CASES if c["name"] == "grid_2x2")
    b = next(c for c in _CASES if c["name"] == "overview_only")
    n = max(len(a["input_ids"]), len(b["input_ids"]))
    rows = [[0] * (n - len(c["input_ids"])) + c["input_ids"] for c in (a, b)]
    mask = [[0] * (n - len(c["input_ids"])) + [1] * len(c["input_ids"])
            for c in (a, b)]
    pos, deltas = canvas_rope_index(
        np.array(rows), a["target_sizes"] + b["target_sizes"], _REAL,
        merge=4, attention_mask=np.array(mask))
    for r, c in enumerate((a, b)):
        keep = np.array(mask[r], dtype=bool)
        assert pos[:, r, keep].tolist() == c["position_ids"]
        assert int(deltas[r, 0]) == c["rope_deltas"] - int((~keep).sum())


def test_more_crops_than_sizes_is_loud():
    a = next(c for c in _CASES if c["name"] == "grid_2x2")
    with pytest.raises(ValueError, match="target_sizes"):
        canvas_rope_index(np.array([a["input_ids"]]), a["target_sizes"][:2],
                          _REAL, merge=4)


def test_text_only_input_counts_sequentially():
    model = _model()
    pos, deltas = model.language_model.get_rope_index(
        mx.array([[1, 2, 3, 4]]))
    assert np.asarray(pos).tolist() == [[[0, 1, 2, 3]]] * 3
    assert np.asarray(deltas).tolist() == [[0]]


def test_image_tokens_without_crop_sizes_are_refused():
    model = _model()
    with pytest.raises(ValueError, match="target_sizes"):
        model.language_model.get_rope_index(mx.array([[1, IMG, PAD, IMG_END]]))
    # 4.6 lays image tokens out sequentially and needs no sizes.
    seq = _model(canvas=False)
    pos, _ = seq.language_model.get_rope_index(
        mx.array([[1, IMG, PAD, IMG_END]]))
    assert np.asarray(pos)[0, 0].tolist() == [0, 1, 2, 3]



def test_position_buckets_are_the_bucketize_grid():
    # torch.bucketize(i / n, arange(1 / 70, 1, 1 / 70), right=True) on a
    # 28-patch side: bucket floor(70 i / 28) = floor(2.5 i).
    ids = position_bucket_ids(28, 28, 70).reshape(28, 28)
    assert ids[0].tolist() == [math.floor(2.5 * i) for i in range(28)]
    assert ids[:, 0].tolist() == [70 * math.floor(2.5 * i) for i in range(28)]


def _gelu_kind(act):
    x = mx.array([1.5])
    erf = float(nn_gelu_erf(x)[0])
    return "erf" if abs(float(act(x)[0]) - erf) < 1e-7 else "tanh"


def nn_gelu_erf(x):
    return 0.5 * x * (1 + mx.erf(x / math.sqrt(2)))


def test_gelu_kinds_match_the_reference():
    vc = _vision()
    assert _gelu_kind(MLP(8, 8).act) == "tanh"            # SigLIP
    assert _gelu_kind(VitMerger(vc).act) == "tanh"        # gelu_pytorch_tanh
    assert _gelu_kind(MergerBlock(8, 8).act) == "erf"     # nn.GELU()


def test_window_merger_folds_each_2x2_window_in_row_major_order():
    mx.random.seed(0)
    vc = _vision()
    vm = VitMerger(vc)
    mx.eval(vm.parameters())
    h, w, d = 4, 6, vc.hidden_size
    x = mx.random.normal((1, h * w, d))
    got = vm(x, h, w)[0]
    normed = vm.layer_norm1(x)[0]
    for r in range(h // 2):
        for c in range(w // 2):
            idx = [(2 * r + i) * w + 2 * c + j for i in (0, 1) for j in (0, 1)]
            win = x[0][mx.array(idx)]
            win = win + vm.self_attn(normed[mx.array(idx)][None])[0]
            want = vm.linear_2(vm.act(vm.linear_1(
                vm.pre_norm(win.reshape(1, -1)))))[0] + win.mean(axis=0)
            np.testing.assert_allclose(np.asarray(got[r * (w // 2) + c]),
                                       np.asarray(want), atol=1e-5)


def test_tower_returns_one_token_per_4x4_patches_in_crop_order():
    model = _model()
    sizes = [[8, 12], [8, 8], [8, 12]]
    n = sum(h * w for h, w in sizes)
    pv = mx.random.normal((n, 14 * 14 * 3))
    out = model.vision_tower(pv, sizes)
    assert out.shape == (6 + 4 + 6, 64)
    # Crops of one size run as one batch; each crop is encoded alone.
    alone = model.vision_tower(pv[96 + 64:], [[8, 12]])
    np.testing.assert_allclose(np.asarray(out[10:]), np.asarray(alone),
                               atol=1e-4)



def _image_prompt():
    """One 2x1-slice image between text, with a newline after it and an
    image marker typed as text outside the block."""
    ov = [IMG] + [PAD] * 4 + [IMG_END]
    sl = [SLICE] + [PAD] * 4 + [SLICE_END]
    return [7, IMG_END] + ov + sl + [198] + sl + [198, 9, SLICE_END]


def test_separator_rows_touch_only_the_image_block():
    model = _model()
    ids = np.array([_image_prompt()])
    flat, rows = model._separator_rows(ids)
    seq = ids[0].tolist()
    assert [seq[i] for i in flat] == [IMG_END, SLICE, SLICE_END, 198, SLICE,
                                      SLICE_END]
    assert rows == [0, 1, 2, 3, 1, 2]
    assert 1 not in flat and len(seq) - 1 not in flat


def _forward_inputs(model):
    ids = mx.array([_image_prompt()])
    sizes = mx.array([[8, 8], [8, 8], [8, 8]])
    pv = mx.random.normal((3 * 64, 14 * 14 * 3))
    return ids, pv, sizes


def test_end_to_end_forward_with_an_image():
    model = _model()
    ids, pv, sizes = _forward_inputs(model)
    f = model.get_input_embeddings(ids, pv, target_sizes=sizes)
    assert f.position_ids.shape == (3, 1, ids.shape[1])
    assert int(f.rope_deltas[0, 0]) < 0
    sep = np.asarray(model.sep_embeds)
    np.testing.assert_allclose(np.asarray(f.inputs_embeds[0, 7]), sep[0])
    out = model(ids, pv, cache=model.make_cache(), target_sizes=sizes)
    logits = getattr(out, "logits", out)
    assert logits.shape == (1, ids.shape[1], 600)


def test_text_only_forward_returns_sequential_positions():
    model = _model()
    f = model.get_input_embeddings(mx.array([[1, 2, 3]]))
    assert f.position_ids.shape == (3, 1, 3)
    assert np.asarray(f.rope_deltas).tolist() == [[0]]
    out = model(mx.array([[1, 2, 3]]), cache=model.make_cache())
    assert getattr(out, "logits", out).shape == (1, 3, 600)


def test_feature_count_mismatch_is_loud():
    model = _model()
    ids, pv, sizes = _forward_inputs(model)
    with pytest.raises(ValueError, match="do not match"):
        model.get_input_embeddings(ids, pv[:128], target_sizes=sizes[:2])


@pytest.mark.parametrize("text", ["qwen3_5", "qwen3_5_moe"])
def test_language_model_is_the_owned_spec_class(text):
    from gmlx.load.mtp_target import _MTP_TARGET_HOOKS
    from gmlx.models.qwen35.owned import is_owned_language_model

    lm = _model(text=text).language_model
    assert is_owned_language_model(lm)
    assert all(hasattr(lm, h) for h in _MTP_TARGET_HOOKS)
    assert hasattr(lm, "set_dflash_capture")
