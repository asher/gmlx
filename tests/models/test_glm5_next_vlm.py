"""GLM-5.3-Flash vision: preprocessing geometry, the mmproj remap, the rope
position order and the merger's consecutive-4 grouping. CPU-only - no GGUF,
no weights, no image decode.

Every case here is a silent-failure surface: a wrong align direction, a
transposed conv or an off-by-one grid produces a model that runs and
describes the wrong picture. Expectations are derived from llama.cpp PR
27754 (``mtmd_image_preprocessor_glm5next`` in ``tools/mtmd/mtmd-image.cpp``
and ``clip_graph_glm4v``), not read back off this port.
"""

from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from mlx.utils import tree_flatten  # noqa: E402

from gmlx.load.vlm import (  # noqa: E402
    _Glm5NextGgufImageProcessor,
    _synthesize_glm5next_vlm_config,
    remap_vision_arrays,
    resolve_vlm_model_type,
)
from gmlx.models.glm5_next.vision import (  # noqa: E402
    Glm5NextPatchMerger,
    Glm5NextVisionModel,
    VisionConfig,
)

from test_glm5_next import _tiny_args  # noqa: E402


def _proc():
    return _Glm5NextGgufImageProcessor(
        image_mean=[0.48145466, 0.4578275, 0.40821073],
        image_std=[0.26862954, 0.26130258, 0.27577711])


def _tiny_vision(out_hidden_size=24):
    return VisionConfig(
        depth=2, hidden_size=32, num_heads=4, intermediate_size=48,
        patch_size=2, temporal_patch_size=2, spatial_merge_size=2,
        out_hidden_size=out_hidden_size, proj_intermediate_size=40)


# --- smart resize (canvas search) --------------------------------------------


@pytest.mark.parametrize("h,w,expect", [
    (448, 448, (448, 448)),      # aligned and in budget: untouched
    (450, 300, (476, 308)),      # align UP, never down
    (100, 100, (112, 112)),      # aligned area == min budget exactly
    (28, 28, (112, 112)),        # below min: sqrt-scale up to 16 tokens
    (1, 1, (112, 112)),          # degenerate input still yields a canvas
    (10, 2000, (28, 2016)),      # thin strip: clamp only the short side
    (5000, 3000, (3220, 1932)),  # over max: binary-search content height
    (3000, 5000, (1932, 3220)),  # transposed strip searches the other axis
    (10000, 2000, (5600, 1120)),   # lands exactly on the 8000-token cap
    (100000, 100, (74648, 84)),    # extreme ratio: width floors to one run
])
def test_smart_resize_matches_the_llama_cpp_search(h, w, expect):
    assert _proc()._smart_resize(h, w) == expect


def test_the_token_cap_case_is_exact():
    p = _proc()
    ch, cw = p._smart_resize(10000, 2000)
    assert (ch // 28) * (cw // 28) == 8000


@pytest.mark.parametrize("h,w", [
    (448, 448), (450, 300), (37, 91), (28, 28), (1, 1), (10, 2000),
    (5000, 3000), (10000, 2000), (100000, 100), (12345, 6789), (3, 4096),
])
def test_canvas_is_aligned_and_inside_the_token_budget(h, w):
    p = _proc()
    ch, cw = p._smart_resize(h, w)
    assert ch % 28 == 0 and cw % 28 == 0
    assert p.min_pixels <= ch * cw <= p.max_pixels


# --- content geometry (resize vs pad) ----------------------------------------


@pytest.mark.parametrize("h,w,canvas,content", [
    # >= min budget: never upscaled to fill; the canvas pads instead.
    (300, 500, (308, 504), (300, 500)),
    (120, 120, (140, 140), (120, 120)),
    (200, 800, (224, 812), (200, 800)),
    # below min budget: content upscales to fill the canvas.
    (100, 100, (112, 112), (112, 112)),
    # over max budget: content shrinks with its ratio.
    (10000, 2000, (5600, 1120), (5600, 1120)),
])
def test_geometry_matches_the_reference(h, w, canvas, content):
    assert _proc()._geometry(h, w) == (canvas, content)


@pytest.mark.parametrize("h,w", [
    (300, 500), (100, 100), (10000, 2000), (7, 7), (2000, 10), (999, 1001),
])
def test_content_always_fits_its_canvas(h, w):
    (ch, cw), (content_h, content_w) = _proc()._geometry(h, w)
    assert 1 <= content_h <= ch and 1 <= content_w <= cw


# --- patchify -----------------------------------------------------------------


def test_patchify_shapes_and_grid():
    from PIL import Image
    img = Image.new("RGB", (500, 300))     # PIL size is (W, H)
    p = _proc()
    out = p(img)
    assert out["image_grid_thw"].tolist() == [[1, 22, 36]]
    assert out["pixel_values"].shape == (22 * 36, 3 * 2 * 14 * 14)
    assert p.soft_tokens(out["image_grid_thw"][0]) == 11 * 18


def test_patchify_is_the_qwen2vl_block_shuffle():
    """Row k of pixel_values must be the (c, t, y, x)-flattened 14x14 patch at
    the k-th position of the h//2,2,w//2,2 shuffle - the order the tower's
    rope ids and the merger's stride-2 conv both assume."""
    from PIL import Image
    rng = np.random.default_rng(0)
    # 112x140 is aligned, above the 16-token minimum and under the max, so
    # canvas == content == input: no resample perturbs the pixels.
    raw = rng.integers(0, 256, size=(112, 140, 3), dtype=np.uint8)
    img = Image.fromarray(raw)
    p = _proc()
    out = p(img)
    gh, gw = 8, 10
    assert out["image_grid_thw"].tolist() == [[1, gh, gw]]

    mean = np.array(p.image_mean, dtype=np.float32)
    std = np.array(p.image_std, dtype=np.float32)
    arr = ((raw.astype(np.float32) / 255.0 - mean) / std).transpose(2, 0, 1)

    shuffled = [(a * 2 + i, b * 2 + j)
                for a in range(gh // 2) for b in range(gw // 2)
                for i in range(2) for j in range(2)]
    for row, (ph, pw) in enumerate(shuffled):
        patch = arr[:, ph * 14:(ph + 1) * 14, pw * 14:(pw + 1) * 14]
        ref = np.repeat(patch[:, None], 2, axis=1).reshape(-1)
        np.testing.assert_allclose(out["pixel_values"][row], ref, atol=1e-6)


def test_padding_is_black_and_top_left():
    from PIL import Image
    img = Image.new("RGB", (500, 300), (255, 255, 255))  # canvas 308x504 pads
    p = _proc()
    out = p(img)["pixel_values"].reshape(22 * 36, 3, 2, 14, 14)
    mean = np.array(p.image_mean, dtype=np.float32)
    std = np.array(p.image_std, dtype=np.float32)
    black = np.broadcast_to((-mean / std).reshape(3, 1, 1, 1), (3, 2, 14, 14))
    white = np.broadcast_to(((1.0 - mean) / std).reshape(3, 1, 1, 1),
                            (3, 2, 14, 14))

    def row(ph, pw, gw=36):
        return ((ph // 2) * (gw // 2) + pw // 2) * 4 + (ph % 2) * 2 + pw % 2

    np.testing.assert_allclose(out[row(0, 0)], white, atol=1e-5)
    # Bottom row of patches: image rows 300..307 are pad.
    bottom = out[row(21, 0)]                    # [3, 2, 14, 14] = (c, t, y, x)
    np.testing.assert_allclose(bottom[:, :, 6:], black[:, :, 6:], atol=1e-5)
    np.testing.assert_allclose(bottom[:, :, :6], white[:, :, :6], atol=1e-5)


# --- mmproj tensor remap ------------------------------------------------------

_BLK_LEAVES = (
    "attn_qkv.weight", "attn_qkv.bias", "attn_out.weight", "attn_out.bias",
    "attn_q_norm.weight", "attn_k_norm.weight", "ln1.weight", "ln2.weight",
    "ffn_gate.weight", "ffn_gate.bias", "ffn_up.weight", "ffn_up.bias",
    "ffn_down.weight", "ffn_down.bias",
)
_TOP = (
    "v.patch_embd.weight", "v.patch_embd.weight.1", "v.patch_embd.bias",
    "v.post_ln.weight",
    "mm.patch_merger.weight", "mm.patch_merger.bias", "mm.model.fc.weight",
    "mm.post_norm.weight", "mm.post_norm.bias",
    "mm.gate.weight", "mm.up.weight", "mm.down.weight",
)


def _mmproj_arrays(n_layers=2):
    """The full glm5next mmproj tensor name set (from the shipped GGUF), with
    real shapes only where a remap transform depends on them."""
    arrays = {n: mx.zeros((2, 2)) for n in _TOP}
    arrays["v.patch_embd.weight"] = mx.zeros((32, 3, 2, 2))
    arrays["v.patch_embd.weight.1"] = mx.zeros((32, 3, 2, 2))
    arrays["v.patch_embd.bias"] = mx.zeros((32,))
    arrays["mm.patch_merger.weight"] = mx.zeros((24, 32, 2, 2))
    for i in range(n_layers):
        for leaf in _BLK_LEAVES:
            arrays[f"v.blk.{i}.{leaf}"] = mx.zeros((2, 2))
    return arrays


def test_every_mmproj_tensor_is_claimed():
    arrays = _mmproj_arrays()
    out, skipped, kq = remap_vision_arrays(arrays, "glm5_next")
    assert skipped == [] and kq == {}
    # The two temporal conv slices fuse into one Conv3d weight.
    assert len(out) == len(arrays) - 1


def test_patch_conv_fuses_to_conv3d_layout():
    out, _, _ = remap_vision_arrays(_mmproj_arrays(), "glm5_next")
    # [out, in, kH, kW] x2 -> MLX Conv3d [out, t, kH, kW, in].
    assert out["vision_tower.patch_embed.proj.weight"].shape == (32, 2, 2, 2, 3)


def test_patch_conv_temporal_slices_keep_their_order():
    arrays = _mmproj_arrays()
    arrays["v.patch_embd.weight"] = mx.zeros((32, 3, 2, 2))
    arrays["v.patch_embd.weight.1"] = mx.ones((32, 3, 2, 2))
    out, _, _ = remap_vision_arrays(arrays, "glm5_next")
    w = out["vision_tower.patch_embed.proj.weight"]
    assert float(w[:, 0].max()) == 0.0 and float(w[:, 1].min()) == 1.0


def test_patch_merger_conv_is_transposed_to_mlx_layout():
    out, _, _ = remap_vision_arrays(_mmproj_arrays(), "glm5_next")
    # [out, in, kH, kW] -> MLX Conv2d [out, kH, kW, in].
    assert out["vision_tower.merger.patch_merger.weight"].shape == (24, 2, 2, 32)


def test_block_and_projector_tensors_land_on_the_vendored_paths():
    out, _, _ = remap_vision_arrays(_mmproj_arrays(), "glm5_next")
    assert "vision_tower.blocks.0.attn.qkv.weight" in out
    assert "vision_tower.blocks.1.attn.q_norm.weight" in out
    assert "vision_tower.blocks.0.norm2.weight" in out
    assert "vision_tower.blocks.1.mlp.down_proj.bias" in out
    assert "vision_tower.post_layernorm.weight" in out
    assert "vision_tower.merger.fc.weight" in out
    assert "vision_tower.merger.mlp.up_proj.weight" in out


def test_unknown_tensor_is_skipped_not_guessed():
    arrays = _mmproj_arrays()
    arrays["v.blk.0.mystery.weight"] = mx.zeros((2, 2))
    arrays["a.conv_out.weight"] = mx.zeros((2, 2))
    _, skipped, _ = remap_vision_arrays(arrays, "glm5_next")
    assert sorted(skipped) == ["a.conv_out.weight", "v.blk.0.mystery.weight"]


def test_every_remapped_name_is_a_real_parameter_path():
    """Closure against the model tree: a typo in either map loads-and-drops
    silently under strict=False, so every target must resolve to an actual
    parameter of the container."""
    from gmlx.models.glm5_next.vlm_model import Model, ModelConfig
    cfg = ModelConfig(text_config=_tiny_args(), vision_config=_tiny_vision(),
                      image_token_id=7, vocab_size=_tiny_args().vocab_size)
    paths = {name for name, _ in tree_flatten(Model(cfg).parameters())}
    out, _, _ = remap_vision_arrays(_mmproj_arrays(), "glm5_next")
    missing = sorted(set(out) - paths)
    assert missing == []


# --- config synthesis ---------------------------------------------------------

_MM_META = {
    "clip.vision.block_count": 24,
    "clip.vision.embedding_length": 1024,
    "clip.vision.feed_forward_length": 4096,
    "clip.vision.attention.head_count": 16,
    "clip.vision.patch_size": 14,
    "clip.vision.projection_dim": 4096,
    "clip.vision.spatial_merge_size": 2,
    "clip.vision.attention.layer_norm_epsilon": 1e-5,
    "clip.vision.swiglu_limit": 10.0,
}


def test_resolve_picks_the_vendored_container():
    assert resolve_vlm_model_type(
        "glm5next", {"clip.projector_type": "glm5next"}) == "glm5_next"


def test_config_synth_reads_the_mmproj_metadata():
    cfg = _synthesize_glm5next_vlm_config(
        {"vocab_size": 154880}, _MM_META, {},
        mm_shapes={"mm.up.weight": (4096, 10240)})
    vc = VisionConfig.from_dict(cfg["vision_config"])
    assert (vc.depth, vc.hidden_size, vc.num_heads) == (24, 1024, 16)
    assert (vc.intermediate_size, vc.patch_size) == (4096, 14)
    assert (vc.temporal_patch_size, vc.spatial_merge_size) == (2, 2)
    assert (vc.out_hidden_size, vc.proj_intermediate_size) == (4096, 10240)
    assert vc.rms_norm_eps == pytest.approx(1e-5)
    assert vc.swiglu_limit == 10.0
    assert cfg["model_type"] == "glm5_next"
    assert cfg["vocab_size"] == 154880


def test_image_token_id_comes_from_the_vocab_with_a_pinned_fallback():
    tokens = ["a"] * 5 + ["<|image|>"]
    cfg = _synthesize_glm5next_vlm_config(
        {}, _MM_META, {"tokenizer.ggml.tokens": tokens})
    assert cfg["image_token_id"] == cfg["image_token_index"] == 5
    cfg = _synthesize_glm5next_vlm_config({}, _MM_META, {})
    assert cfg["image_token_id"] == 154854


# --- vision tower -------------------------------------------------------------


def test_rope_positions_follow_the_block_shuffle():
    model = Glm5NextVisionModel(_tiny_vision())
    freqs = model._rot_pos_emb([(1, 4, 4)])
    dim = (32 // 4) // 4          # head_dim/4 frequencies per axis
    assert freqs.shape == (16, 2 * dim)
    # inv_freq[0] == 1.0, so column 0 is the h position and column ``dim``
    # the w position, in patch order.
    h = [0, 0, 1, 1, 0, 0, 1, 1, 2, 2, 3, 3, 2, 2, 3, 3]
    w = [0, 1, 0, 1, 2, 3, 2, 3, 0, 1, 0, 1, 2, 3, 2, 3]
    assert freqs[:, 0].tolist() == h
    assert freqs[:, dim].tolist() == w


def test_tower_output_is_one_feature_per_merged_cell():
    cfg = _tiny_vision()
    model = Glm5NextVisionModel(cfg)
    mx.eval(model.parameters())
    patches = mx.random.normal((24, 3 * 2 * 2 * 2))   # grid 4x6
    out = model(patches, [(1, 4, 6)])
    assert out.shape == (6, cfg.out_hidden_size)


def test_merger_conv_matches_the_explicit_block_matmul():
    """The stride-2 conv over each consecutive-4 token block must equal
    sum_{i,j} x[4k + 2i + j] @ W[:, i, j, :]^T + b - i.e. the [out, kH, kW, in]
    layout with the 2x2 block in row-major token order."""
    cfg = _tiny_vision()
    merger = Glm5NextPatchMerger(cfg)
    mx.eval(merger.parameters())
    x = mx.random.normal((8, cfg.hidden_size))
    got = merger.patch_merger(x.reshape(2, 2, 2, cfg.hidden_size))
    w, b = merger.patch_merger.weight, merger.patch_merger.bias
    for k in range(2):
        ref = b
        for i in range(2):
            for j in range(2):
                ref = ref + x[4 * k + 2 * i + j] @ w[:, i, j, :].T
        # M5 f32 GEMM runs at TF32 precision by default (~5e-4 here); a
        # wrong weight layout is O(1).
        assert float(mx.abs(got[k, 0, 0] - ref).max()) < 5e-3


def test_merger_rows_depend_only_on_their_own_block():
    cfg = _tiny_vision()
    merger = Glm5NextPatchMerger(cfg)
    mx.eval(merger.parameters())
    x = mx.random.normal((8, cfg.hidden_size))
    base = merger(x)
    bumped = mx.concatenate([x[:5], x[5:6] + 1.0, x[6:]], axis=0)
    out = merger(bumped)
    assert float(mx.abs(out[0] - base[0]).max()) < 1e-6
    assert float(mx.abs(out[1] - base[1]).max()) > 1e-4


# --- container splice ---------------------------------------------------------


def _tiny_model():
    from gmlx.models.glm5_next.vlm_model import Model, ModelConfig
    targs = _tiny_args()
    # Tower features land in the text residual stream, so out_hidden must be
    # the text hidden size (4096 == 4096 in the real model).
    cfg = ModelConfig(text_config=targs,
                      vision_config=_tiny_vision(targs.hidden_size),
                      image_token_id=7, vocab_size=targs.vocab_size)
    mx.random.seed(0)
    model = Model(cfg)
    mx.eval(model.parameters())
    return model


def test_features_splice_at_the_placeholder_positions_in_order():
    model = _tiny_model()
    ids = mx.array([[3, 9, 7, 7, 7, 7, 11, 5]])
    embeds = model.language_model.model.embed_tokens(ids)
    features = mx.arange(4)[:, None] + mx.zeros((4, embeds.shape[-1]))
    merged = model.merge_input_ids_with_image_features(
        7, features, embeds, ids)
    for pos, rank in ((2, 0), (3, 1), (4, 2), (5, 3)):
        assert float(mx.abs(merged[0, pos] - rank).max()) == 0.0
    for pos in (0, 1, 6, 7):
        assert float(mx.abs(merged[0, pos] - embeds[0, pos]).max()) == 0.0


def test_placeholder_count_mismatch_is_loud():
    model = _tiny_model()
    ids = mx.array([[3, 7, 7, 5]])
    embeds = model.language_model.model.embed_tokens(ids)
    with pytest.raises(ValueError, match="placeholder"):
        model.merge_input_ids_with_image_features(
            7, mx.zeros((3, embeds.shape[-1])), embeds, ids)


def test_end_to_end_forward_with_an_image():
    model = _tiny_model()
    patches = mx.random.normal((16, 3 * 2 * 2 * 2))    # grid 4x4 -> 4 tokens
    grid = mx.array([[1, 4, 4]])
    ids = mx.array([[3, 9, 7, 7, 7, 7, 11, 5]])
    out = model(ids, pixel_values=patches, image_grid_thw=grid,
                cache=model.make_cache())
    assert out.logits.shape == (1, 8, model.config.text_config.vocab_size)


def test_text_only_forward_needs_no_image_kwargs():
    model = _tiny_model()
    ids = mx.array([[3, 9, 11, 5]])
    out = model(ids, cache=model.make_cache())
    assert out.logits.shape == (1, 4, model.config.text_config.vocab_size)


def test_language_model_carries_the_mtp_spec_hooks():
    # mtp_load's VLM branch probes these on model.language_model and refuses
    # text-only MTP if any is missing; the load then fails outright.
    lm = _tiny_model().language_model
    for hook in ("speculative_logits_from_hidden",
                 "speculative_argmax_from_hidden",
                 "speculative_verify_hidden", "rollback_speculative_cache",
                 "chunked_prefill_policy"):
        assert callable(getattr(lm, hook)), hook


def test_return_hidden_yields_the_pre_norm_trunk_hidden():
    """The owned MTP engine's prefill retains hidden_states for the drafter's
    hnorm, which consumes the PRE-final-norm collapsed hidden; logits must
    still come from the normed stream."""
    model = _tiny_model()
    lm = model.language_model
    ids = mx.array([[3, 9, 11, 5]])
    out = lm(ids, cache=lm.make_cache())
    spec = lm(ids, cache=lm.make_cache(), return_hidden=True)
    assert float(mx.abs(spec.logits - out.logits).max()) < 1e-5
    (raw,) = spec.hidden_states
    relogits = lm.speculative_logits_from_hidden(raw)
    assert float(mx.abs(relogits - out.logits).max()) < 1e-5
    # raw is pre-norm: applying the final norm must change it.
    assert float(mx.abs(lm.model.norm(raw) - raw).max()) > 1e-4
