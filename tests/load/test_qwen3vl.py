"""Qwen3-VL as a text model (GGUF arch qwen3vl) and as a vision model.

CPU-size tests on synthetic headers and tiny random models. The text row
loads the qwen3 class, which has one rotary axis; the checkpoint was trained
with three. The first group shows the two agree when all three axes carry
the same position, which is what a text-only prompt does. The second group
covers the VLM config, the vision remap with the deepstack mergers, and the
language tower of the VLM container against the text class.
"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx.utils import tree_flatten

import gmlx.load.vlm as gvlm
from gmlx._vendor.mlx_lm.models.rope_utils import initialize_rope
from gmlx._vendor.mlx_lm.utils import _get_classes
from gmlx._vendor.mlx_vlm.models.qwen3_vl.language import (
    Qwen3VLRotaryEmbedding,
    apply_multimodal_rotary_pos_emb,
)
from gmlx.cache.attention import create_attention_mask
from gmlx.cache.kv_cache import make_prompt_cache
from gmlx.load.config_synth import synthesize_config
from test_config_synth import _QWEN3VL_SHAPES, _qwen3vl_meta

# The Qwen3-VL-8B text header: head width, rope base, axis widths, context.
HEAD_DIM = 128
BASE = 5000000.0
SECTIONS = [24, 20, 20]
CONTEXT = 262144
HALF = HEAD_DIM // 2
L = 8


def _text_rope(base=BASE, traditional=False):
    rope = initialize_rope(HEAD_DIM, base=base, traditional=traditional,
                           scaling_config=None,
                           max_position_embeddings=CONTEXT)
    assert isinstance(rope, nn.RoPE)
    return rope


def _vlm_rope():
    return Qwen3VLRotaryEmbedding(
        HEAD_DIM, max_position_embeddings=CONTEXT, base=BASE,
        rope_scaling={"type": "default", "mrope_section": SECTIONS})


def _unit():
    """First half 1, second half 0: each rotated pair reads (cos, sin) of
    its angle, so a difference between two outputs is an angle difference."""
    x = np.zeros((1, 1, L, HEAD_DIM), dtype=np.float32)
    x[..., :HALF] = 1.0
    return mx.array(x)


def _positions(offset):
    pos2 = mx.arange(offset, offset + L).reshape(1, L)
    return pos2, mx.tile(pos2[None], (3, 1, 1))


def _reference(offset):
    """The rotation of ``_unit`` in float64, half-split pairs, base BASE."""
    freqs = BASE ** (-np.arange(HALF, dtype=np.float64) / HALF)
    ang = np.arange(offset, offset + L, dtype=np.float64)[:, None] * freqs
    return np.concatenate([np.cos(ang), np.sin(ang)], axis=-1)


def _vlm_rotations(rot, x, offset):
    """``x`` rotated by the copied class every way its attention can: the
    cos and sin tables with 2-D and 3-D position ids, and the fused kernel
    where the device has one."""
    out = {}
    for name, pos in zip(("pos2d", "pos3d"), _positions(offset)):
        cos, sin = rot(x, pos)
        out[f"tables_{name}"] = apply_multimodal_rotary_pos_emb(
            x, x, cos, sin)[0]
        if rot.fused_apply:
            out[f"fused_{name}"] = rot.apply_rotary(
                x, x, pos, unsqueeze_dim=1)[0]
    return out


def test_every_frequency_pair_reads_one_of_the_three_axes():
    # With one position on all three axes, the axis a pair reads does not
    # change its angle, so the three-axis rope is a one-axis rope.
    rot = _vlm_rope()
    selector = rot.position_selector.tolist()
    assert len(selector) == HALF
    assert set(selector) == {0, 1, 2}
    # The table is the one-axis table of this base: base ** (-i / half).
    table = np.array(rot.inv_freq).astype(np.float64)
    want = BASE ** (-np.arange(HALF, dtype=np.float64) / HALF)
    assert (np.abs(table - want) / want).max() < 1e-6


def test_three_axis_rope_with_one_position_is_the_text_rope():
    # The first positions, where fp32 resolves an angle to under 1e-6.
    x = _unit()
    rope, rot = _text_rope(), _vlm_rope()
    text = rope(x, offset=0)
    ref = _reference(0)
    assert np.abs(np.array(text)[0, 0] - ref).max() < 1e-6
    for name, got in _vlm_rotations(rot, x, 0).items():
        assert np.abs(np.array(got)[0, 0] - ref).max() < 1e-6, name
        assert float(mx.abs(got - text).max()) < 1e-6, name


@pytest.mark.parametrize("offset", [1000, 4096, 16384, 65536, CONTEXT - L])
def test_the_two_ropes_agree_to_the_fp32_angle_at_depth(offset):
    # At depth an fp32 angle is only as fine as the spacing of fp32 at the
    # position. Both classes stay within that of the float64 rotation, and
    # so of each other: the gap is rounding, not another base or pairing.
    x = _unit()
    rope, rot = _text_rope(), _vlm_rope()
    bound = 2 * float(np.spacing(np.float32(offset + L - 1))) + 1e-6
    ref = _reference(offset)
    text = rope(x, offset=offset)
    assert np.abs(np.array(text)[0, 0] - ref).max() < bound
    for name, got in _vlm_rotations(rot, x, offset).items():
        assert np.abs(np.array(got)[0, 0] - ref).max() < bound, name
        assert float(mx.abs(got - text).max()) < bound, name


def test_random_inputs_rotate_the_same_through_both_classes():
    mx.random.seed(0)
    q = mx.random.normal((1, 4, L, HEAD_DIM))
    k = mx.random.normal((1, 2, L, HEAD_DIM))
    rope, rot = _text_rope(), _vlm_rope()
    for pos in _positions(0):
        cos, sin = rot(k, pos)
        vq, vk = apply_multimodal_rotary_pos_emb(q, k, cos, sin)
        assert float(mx.abs(vq - rope(q)).max()) < 1e-5
        assert float(mx.abs(vk - rope(k)).max()) < 1e-5


def test_another_base_or_pairing_does_not_pass_the_same_check():
    # The check above is not loose: the qwen3 class default base (1e6) and
    # the even/odd pairing both miss by orders of magnitude at once.
    x = _unit()
    got = _vlm_rotations(_vlm_rope(), x, 0)["tables_pos3d"]
    assert float(mx.abs(_text_rope(base=1000000.0)(x) - got).max()) > 0.1
    assert float(mx.abs(_text_rope(traditional=True)(x) - got).max()) > 0.1


# The VLM: a tiny Qwen3-VL pair built from synthetic headers.

_TOKENS = ["t"] * 28 + ["<|vision_start|>", "<|vision_end|>",
                         "<|image_pad|>", "<|video_pad|>"]
VISION_START, VISION_END, IMAGE_PAD, VIDEO_PAD = 28, 29, 30, 31

# A 3-block tower whose blocks 1 and 2 feed deepstack mergers.
_MM_META = {
    "clip.projector_type": "qwen3vl_merger",
    "clip.vision.patch_size": 16,
    "clip.vision.image_size": 64,
    "clip.vision.block_count": 3,
    "clip.vision.embedding_length": 32,
    "clip.vision.feed_forward_length": 64,
    "clip.vision.projection_dim": 64,
    "clip.vision.attention.head_count": 2,
    "clip.vision.spatial_merge_size": 2,
    "clip.vision.is_deepstack_layers": [False, True, True],
}


def _llm_meta():
    m = _qwen3vl_meta()
    m["tokenizer.ggml.tokens"] = list(_TOKENS)
    m["qwen3vl.n_deepstack_layers"] = 2
    return m


def _mm_arrays():
    """The tensors of a Qwen3-VL mmproj, by their GGUF names, at the sizes
    of ``_MM_META``."""
    mx.random.seed(1)

    def r(*shape):
        return mx.random.normal(shape) * 0.05

    width, ffn, merged, out = 32, 64, 128, 64
    arrays = {
        "v.patch_embd.weight": r(width, 3, 16, 16),
        "v.patch_embd.weight.1": r(width, 3, 16, 16),
        "v.patch_embd.bias": r(width),
        "v.position_embd.weight": r(16, width),
        "v.post_ln.weight": r(width), "v.post_ln.bias": r(width),
        "mm.0.weight": r(merged, merged), "mm.0.bias": r(merged),
        "mm.2.weight": r(out, merged), "mm.2.bias": r(out),
    }
    for b in range(3):
        arrays.update({
            f"v.blk.{b}.attn_qkv.weight": r(3 * width, width),
            f"v.blk.{b}.attn_qkv.bias": r(3 * width),
            f"v.blk.{b}.attn_out.weight": r(width, width),
            f"v.blk.{b}.attn_out.bias": r(width),
            f"v.blk.{b}.ffn_up.weight": r(ffn, width),
            f"v.blk.{b}.ffn_up.bias": r(ffn),
            f"v.blk.{b}.ffn_down.weight": r(width, ffn),
            f"v.blk.{b}.ffn_down.bias": r(width),
            f"v.blk.{b}.ln1.weight": r(width), f"v.blk.{b}.ln1.bias": r(width),
            f"v.blk.{b}.ln2.weight": r(width), f"v.blk.{b}.ln2.bias": r(width),
        })
    for b in (1, 2):
        arrays.update({
            f"v.deepstack.{b}.norm.weight": r(merged),
            f"v.deepstack.{b}.norm.bias": r(merged),
            f"v.deepstack.{b}.fc1.weight": r(merged, merged),
            f"v.deepstack.{b}.fc1.bias": r(merged),
            f"v.deepstack.{b}.fc2.weight": r(out, merged),
            f"v.deepstack.{b}.fc2.bias": r(out),
        })
    return arrays


def _vlm_config():
    return gvlm.synthesize_vlm_config(
        "qwen3_vl", _llm_meta(), _QWEN3VL_SHAPES, _MM_META)


def _tiny_vlm():
    """The tiny pair as a model: the vision half from the remapped mmproj
    tensors, the text half random."""
    model, _ = gvlm.build_vlm_model(_vlm_config())
    vision, skipped, _ = gvlm.remap_vision_arrays(
        _mm_arrays(), "qwen3_vl", mm_meta=_MM_META)
    assert skipped == []
    mx.random.seed(2)
    weights = dict(vision)
    for k, v in tree_flatten(model.language_model.parameters()):
        weights[f"language_model.{k}"] = mx.random.normal(v.shape) * 0.05
    model.load_weights(list(weights.items()), strict=True)
    mx.eval(model.parameters())
    return model


def _text_model(vlm):
    """The text row's class with the language tower's weights."""
    config = synthesize_config(_llm_meta(), tensor_shapes=_QWEN3VL_SHAPES)
    Model, ModelArgs = _get_classes(config)
    model = Model(ModelArgs.from_dict(config))
    model.load_weights(
        list(tree_flatten(vlm.language_model.parameters())), strict=True)
    mx.eval(model.parameters())
    return model


def test_vlm_config_reads_both_headers():
    cfg = _vlm_config()
    assert cfg["model_type"] == "qwen3_vl"
    text = cfg["text_config"]
    assert text["model_type"] == "qwen3_vl"
    assert text["rope_theta"] == 5000000.0
    # the first three of the file's four axis widths.
    assert text["rope_scaling"] == {"type": "default",
                                    "mrope_section": [4, 2, 2]}
    vision = cfg["vision_config"]
    assert vision["model_type"] == "qwen3_vl"
    assert vision["depth"] == 3
    assert vision["out_hidden_size"] == text["hidden_size"] == 64
    assert vision["num_position_embeddings"] == 16
    # the blocks the mmproj flags, in order.
    assert vision["deepstack_visual_indexes"] == [1, 2]
    # the marker ids of this vocab, not the class defaults.
    assert cfg["image_token_id"] == IMAGE_PAD
    assert cfg["video_token_id"] == VIDEO_PAD
    assert cfg["vision_start_token_id"] == VISION_START
    assert cfg["vision_end_token_id"] == VISION_END
    assert cfg["vocab_size"] == len(_TOKENS)


def test_qwen36_vision_config_keeps_deepstack_off():
    # The Qwen3.6 containers drop the tower's deepstack features, so their
    # config builds no merger whatever the mmproj flags.
    for model_type in ("qwen3_5", "qwen3_5_moe"):
        cfg = gvlm._synthesize_qwen35_vision_config(_MM_META, model_type)
        assert cfg["deepstack_visual_indexes"] == []


def test_vlm_config_refuses_a_text_header_without_axis_widths():
    meta = _llm_meta()
    del meta["qwen3vl.rope.dimension_sections"]
    with pytest.raises(ValueError, match="qwen3vl.rope.dimension_sections"):
        gvlm.synthesize_vlm_config(
            "qwen3_vl", meta, _QWEN3VL_SHAPES, _MM_META)


def test_vlm_config_refuses_axis_widths_that_miss_the_head_width():
    meta = _llm_meta()
    meta["qwen3vl.rope.dimension_sections"] = [24, 20, 20, 0]
    with pytest.raises(ValueError, match="half the head width"):
        gvlm.synthesize_vlm_config(
            "qwen3_vl", meta, _QWEN3VL_SHAPES, _MM_META)


def test_vlm_config_refuses_files_that_disagree_on_deepstack():
    meta = _llm_meta()
    meta["qwen3vl.n_deepstack_layers"] = 3
    with pytest.raises(gvlm.UnsupportedVLMError, match="not a pair"):
        gvlm.synthesize_vlm_config(
            "qwen3_vl", meta, _QWEN3VL_SHAPES, _MM_META)


def test_vision_remap_fills_the_tower_and_its_deepstack_mergers():
    arrays = _mm_arrays()
    out, skipped, kqmeta = gvlm.remap_vision_arrays(
        arrays, "qwen3_vl", mm_meta=_MM_META)
    # every mmproj tensor is consumed (the two patch slices become one).
    assert skipped == []
    assert kqmeta == {}
    assert len(out) == len(arrays) - 1
    # block 1 is the first merger of the list, block 2 the second.
    for j, block in enumerate((1, 2)):
        for src, dst in (("norm", "norm"), ("fc1", "linear_fc1"),
                         ("fc2", "linear_fc2")):
            for leaf in ("weight", "bias"):
                got = out[f"vision_tower.deepstack_merger_list.{j}.{dst}.{leaf}"]
                assert got is arrays[f"v.deepstack.{block}.{src}.{leaf}"]
    # and the remapped names are the tower's parameters, no more, no fewer.
    model, _ = gvlm.build_vlm_model(_vlm_config())
    params = dict(tree_flatten(model.vision_tower.parameters()))
    assert {f"vision_tower.{k}" for k in params} == set(out)
    for k, v in params.items():
        assert out[f"vision_tower.{k}"].shape == v.shape, k


def test_vision_remap_refuses_a_merger_of_an_unflagged_block():
    arrays = _mm_arrays()
    arrays["v.deepstack.0.norm.weight"] = mx.zeros((128,))
    with pytest.raises(ValueError, match="v.deepstack.0.norm.weight"):
        gvlm.remap_vision_arrays(arrays, "qwen3_vl", mm_meta=_MM_META)


def test_qwen36_vision_remap_leaves_deepstack_tensors_unused():
    # The Qwen3.6 remap names them as skipped; it maps nothing onto mergers
    # the Qwen3.6 tower does not build.
    arrays = _mm_arrays()
    out, skipped, _ = gvlm.remap_vision_arrays(
        arrays, "qwen3_5", mm_meta=_MM_META)
    assert sorted(skipped) == sorted(
        k for k in arrays if k.startswith("v.deepstack."))
    assert not any("deepstack" in k for k in out)


def test_text_class_and_language_tower_give_the_same_logits():
    vlm = _tiny_vlm()
    text = _text_model(vlm)
    ids = mx.array([[1, 5, 9, 2, 7, 3, 11, 4, 6, 8]])
    want = text(ids)
    got = vlm(ids).logits
    assert float(mx.abs(got - want).max()) < 1e-5
    # and through a cache, prefill then one decode step.
    nxt = mx.array([[12]])
    c_text = make_prompt_cache(text)
    c_vlm = make_prompt_cache(vlm.language_model)
    assert float(mx.abs(vlm.language_model(ids, cache=c_vlm).logits
                        - text(ids, cache=c_text)).max()) < 1e-5
    assert float(mx.abs(vlm.language_model(nxt, cache=c_vlm).logits
                        - text(nxt, cache=c_text)).max()) < 1e-5


def test_text_class_and_language_tower_give_the_same_last_hidden_state():
    # The hidden state after the last layer, before the final norm. The
    # container has a call for it; the text class has none, so its layers
    # are run here in order with the mask its own forward builds.
    vlm = _tiny_vlm()
    text = _text_model(vlm)
    ids = mx.array([[1, 5, 9, 2, 7, 3, 11, 4, 6, 8]])
    n_layers = len(vlm.language_model.model.layers)
    got = vlm.hidden_state_at_layer(ids, n_layers)

    h = text.model.embed_tokens(ids)
    mask = create_attention_mask(h, None)
    for layer in text.model.layers:
        h = layer(h, mask, None)
    assert got.shape == h.shape == (1, 10, 64)
    assert float(mx.abs(got - h).max()) < 1e-5
    # the final norm is what separates it from the model's own output.
    assert float(mx.abs(text.model.norm(h) - text.model(ids)).max()) < 1e-6
    assert float(mx.abs(got - text.model(ids)).max()) > 1e-3


def _image_turn(grid_h=4, grid_w=4):
    """One image of ``grid_h`` x ``grid_w`` patches between two text spans."""
    n_soft = (grid_h // 2) * (grid_w // 2)
    ids = mx.array([[1, VISION_START] + [IMAGE_PAD] * n_soft
                    + [VISION_END, 5, 6]])
    mx.random.seed(3)
    pixels = mx.random.normal((grid_h * grid_w, 3 * 2 * 16 * 16))
    grid = mx.array([[1, grid_h, grid_w]])
    return ids, pixels, grid, n_soft


def test_image_positions_follow_the_grid():
    # An image token takes the position of the block start on the time
    # axis, plus its row on the height axis and its column on the width
    # axis; the text after the block resumes past the larger grid side.
    vlm = _tiny_vlm()
    ids, _, grid, n_soft = _image_turn(grid_h=4, grid_w=6)
    assert n_soft == 6
    pos, _ = vlm.language_model.get_rope_index(ids, image_grid_thw=grid)
    t, h, w = (pos[i, 0].tolist() for i in range(3))
    assert t[:2] == h[:2] == w[:2] == [0, 1]
    assert t[2:8] == [2] * 6
    assert h[2:8] == [2, 2, 2, 3, 3, 3]
    assert w[2:8] == [2, 3, 4, 2, 3, 4]
    assert t[8:] == h[8:] == w[8:] == [5, 6, 7]


def test_deepstack_feature_j_joins_after_decoder_layer_j():
    vlm = _tiny_vlm()
    ids, pixels, grid, n_soft = _image_turn()
    feats = vlm.get_input_embeddings(ids, pixels, image_grid_thw=grid)
    deep = feats.deepstack_visual_embeds
    # one feature per flagged block, one row per image token, text width.
    assert len(deep) == 2
    assert all(d.shape == (n_soft, 64) for d in deep)

    lm = vlm.language_model.model
    h = feats.inputs_embeds
    mask = create_attention_mask(h, None)
    image = slice(2, 2 + n_soft)
    for j, layer in enumerate(lm.layers):
        h = layer(h, mask, None, feats.position_ids, None)
        if j < len(deep):
            h = mx.concatenate(
                [h[:, :image.start], h[:, image] + deep[j][None],
                 h[:, image.stop:]], axis=1)
        got = vlm.hidden_state_at_layer(
            ids, j + 1, pixel_values=pixels, image_grid_thw=grid)
        assert float(mx.abs(got - h).max()) < 1e-5, j
    # the features are not a no-op: without them the state differs.
    plain = lm(ids, inputs_embeds=feats.inputs_embeds,
               position_ids=feats.position_ids, apply_final_norm=False)
    assert float(mx.abs(plain - h).max()) > 1e-3


def test_deepstack_features_come_from_the_flagged_vision_blocks():
    vlm = _tiny_vlm()
    _, pixels, grid, _ = _image_turn()
    tower = vlm.vision_tower
    _, deep = tower(pixels, grid)

    h = tower.patch_embed(pixels) + tower.fast_pos_embed_interpolate(grid)
    rot = tower.rot_pos_emb(grid).reshape(h.shape[0], -1)
    bounds = mx.array([0, h.shape[0]])
    want = []
    for b, block in enumerate(tower.blocks):
        h = block(h, cu_seqlens=bounds, rotary_pos_emb=rot)
        if b in (1, 2):
            want.append(tower.deepstack_merger_list[len(want)](h))
    assert len(deep) == len(want) == 2
    for got, ref in zip(deep, want):
        assert float(mx.abs(got - ref).max()) < 1e-5
