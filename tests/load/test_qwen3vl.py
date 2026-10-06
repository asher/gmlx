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


# The three-axis rotary position where the axes differ.

def _axis_of_pair():
    """The axis each frequency pair reads, written out here and not taken
    from the class: time, then height at pairs 1, 4, 7, ... and width at
    pairs 2, 5, 8, ..., for as many pairs as each of the two is wide; time
    keeps the rest, which with [24, 20, 20] is pairs 60 to 63."""
    axis = np.zeros(HALF, dtype=np.int64)
    axis[1:3 * SECTIONS[1]:3] = 1
    axis[2:3 * SECTIONS[2]:3] = 2
    return axis


# One row per axis. No two axes agree at any of the L places, and the
# values reach what a 4096 token image inside a long prompt can give.
_POS_3D = np.array([
    [5, 5, 5, 700, 700, 3000, 1, 0],
    [0, 17, 300, 2, 1900, 44, 3999, 9],
    [9, 4000, 60, 1, 250, 3, 11, 2600],
])


def _angles(axis, pos=_POS_3D):
    freqs = BASE ** (-np.arange(HALF, dtype=np.float64) / HALF)
    return pos[axis].T.astype(np.float64) * freqs


def _rotation(ang):
    return np.concatenate([np.cos(ang), np.sin(ang)], axis=-1)


def _pair_bound(ang):
    """What fp32 can hold of each angle's cos and sin: the position times
    the frequency rounds to a few spacings of fp32 at the angle."""
    one = 8 * np.spacing(np.abs(ang).astype(np.float32)).astype(np.float64)
    return np.concatenate([one, one], axis=-1) + 1e-6


def test_three_axis_rope_at_unequal_positions_is_the_float64_rotation():
    assert np.bincount(_axis_of_pair()).tolist() == SECTIONS
    assert (_POS_3D[0] != _POS_3D[1]).all()
    assert (_POS_3D[0] != _POS_3D[2]).all()
    assert (_POS_3D[1] != _POS_3D[2]).all()
    ang = _angles(_axis_of_pair())
    ref, bound = _rotation(ang), _pair_bound(ang)
    x, rot = _unit(), _vlm_rope()
    pos = mx.array(_POS_3D).reshape(3, 1, L)
    cos, sin = rot(x, pos)
    got = {"tables": apply_multimodal_rotary_pos_emb(x, x, cos, sin)[0]}
    if rot.fused_apply:
        got["fused"] = rot.apply_rotary(x, x, pos, unsqueeze_dim=1)[0]
    for name, out in got.items():
        gap = np.abs(np.array(out)[0, 0].astype(np.float64) - ref)
        assert (gap <= bound).all(), (name, float((gap / bound).max()))


def _chunked_axis():
    return np.repeat(np.arange(3), SECTIONS)


def _interleaved_axis(sections):
    axis = np.zeros(HALF, dtype=np.int64)
    axis[1:3 * sections[1]:3] = 1
    axis[2:3 * sections[2]:3] = 2
    return axis


@pytest.mark.parametrize("name", [
    "chunked", "height_width_swapped", "time_height_swapped",
    "sections_20_24_20", "sections_20_20_24"])
def test_another_axis_layout_does_not_pass_the_unequal_position_check(name):
    # The same bound refuses each layout a wrong table or a wrong section
    # list would give, by a wide margin, so the check above can tell them.
    right = _axis_of_pair()
    wrong = {
        "chunked": _chunked_axis(),
        "height_width_swapped": np.array([0, 2, 1])[right],
        "time_height_swapped": np.array([1, 0, 2])[right],
        "sections_20_24_20": _interleaved_axis([20, 24, 20]),
        "sections_20_20_24": _interleaved_axis([20, 20, 24]),
    }[name]
    ang = _angles(right)
    gap = np.abs(_rotation(_angles(wrong)) - _rotation(ang))
    assert float((gap / _pair_bound(ang)).max()) > 100


# The processor the two headers give.

_MARKERS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>",
            "<|vision_start|>", "<|vision_end|>", "<|image_pad|>",
            "<|video_pad|>"]


def _flat_image(width, height, value=0):
    from PIL import Image

    return Image.fromarray(np.full((height, width, 3), value, dtype=np.uint8))


def _processor(model_type):
    from tokenizers import pre_tokenizers

    from gmlx.load.tokenizer import load_tokenizer_from_gguf

    tokens = _MARKERS + sorted(pre_tokenizers.ByteLevel.alphabet()) + ["He"]
    tokenizer = load_tokenizer_from_gguf({
        "general.architecture": "qwen3vl",
        "tokenizer.ggml.model": "gpt2",
        "tokenizer.ggml.pre": "qwen2",
        "tokenizer.ggml.tokens": tokens,
        "tokenizer.ggml.merges": ["H e"],
        "tokenizer.ggml.token_type": (
            [3] * len(_MARKERS) + [1] * (len(tokens) - len(_MARKERS))),
        "tokenizer.ggml.eos_token_id": 2,
        "tokenizer.ggml.padding_token_id": 0,
    }, "qwen3vl")
    # The values the Qwen3-VL-8B mmproj header carries.
    mm_meta = dict(_MM_META)
    mm_meta["clip.vision.image_mean"] = [0.5, 0.5, 0.5]
    mm_meta["clip.vision.image_std"] = [0.5, 0.5, 0.5]
    return gvlm._synthesize_vlm_processor(model_type, tokenizer, mm_meta)


def test_processor_constants_are_the_official_ones():
    # Qwen3-VL's preprocessor_config.json: patch 16, merge 2, temporal patch
    # 2, rescale 1 / 255, mean and std 0.5 on every channel.
    ip = _processor("qwen3_vl").image_processor
    assert (ip.patch_size, ip.merge_size, ip.temporal_patch_size) == (16, 2, 2)
    assert ip.do_rescale and ip.do_normalize
    assert ip.rescale_factor == pytest.approx(0.00392156862745098, rel=1e-12)
    assert list(ip.image_mean) == [0.5, 0.5, 0.5]
    assert list(ip.image_std) == [0.5, 0.5, 0.5]


def test_processor_maps_a_byte_to_twice_its_fraction_minus_one():
    # The arithmetic those constants give, on pixels: 0 goes to -1, 255 to
    # 1, and v to 2 * v / 255 - 1. A doubled rescale or the CLIP mean and
    # std both miss it.
    ip = _processor("qwen3_vl").image_processor
    for value in (0, 51, 128, 255):
        out = ip([_flat_image(256, 256, value)])
        assert out["image_grid_thw"].tolist() == [[1, 16, 16]]
        want = 2.0 * value / 255.0 - 1.0
        assert np.abs(out["pixel_values"] - want).max() < 1e-6, value


@pytest.mark.parametrize("width,height,tokens", [
    (64, 64, 64), (128, 128, 64), (160, 120, 70), (224, 224, 64),
    (256, 256, 64), (2560, 1920, 4015)])
def test_qwen3vl_image_token_counts(width, height, tokens):
    # The official counts at the low end, where the lower bound resizes up,
    # and the upper bound's count for an image past it.
    ip = _processor("qwen3_vl").image_processor
    assert ip.num_image_tokens(height, width) == tokens
    grid = ip([_flat_image(width, height)])["image_grid_thw"]
    assert int(grid.prod()) // 4 == tokens


def test_qwen3vl_pixel_bounds_are_64_and_4096_image_tokens():
    ip = _processor("qwen3_vl").image_processor
    assert ip.min_pixels == 64 * 32 * 32 == 65536
    assert ip.max_pixels == 4096 * 32 * 32 == 4194304
    # the largest image the bound lets through whole.
    assert ip.num_image_tokens(2048, 2048) == 4096


def test_qwen36_keeps_its_pixel_bounds():
    for model_type in ("qwen3_5", "qwen3_5_moe"):
        ip = _processor(model_type).image_processor
        assert ip.min_pixels == 8 * 32 * 32
        assert ip.max_pixels == 4096 * 32 * 32
        assert ip.num_image_tokens(64, 64) == 9


# The merger and the refusals from their other side.

def test_merger_activation_is_the_exact_gelu():
    import math

    tower = _tiny_vlm().vision_tower
    grid = np.linspace(-6.0, 6.0, 241)
    exact = np.array([0.5 * v * (1.0 + math.erf(v / math.sqrt(2.0)))
                      for v in grid])
    tanh = np.array([0.5 * v * (1.0 + math.tanh(
        math.sqrt(2.0 / math.pi) * (v + 0.044715 * v ** 3))) for v in grid])
    # the two forms are apart by more than a hundred times the bound.
    assert np.abs(tanh - exact).max() > 1e-4
    mergers = [tower.merger] + list(tower.deepstack_merger_list)
    assert len(mergers) == 3
    for merger in mergers:
        got = np.array(merger.act_fn(mx.array(grid.astype(np.float32))))
        assert np.abs(got - exact).max() < 1e-6


def test_vlm_config_refuses_axis_widths_under_the_head_width():
    meta = _llm_meta()
    widths = list(meta["qwen3vl.rope.dimension_sections"])
    widths[0] -= 1
    meta["qwen3vl.rope.dimension_sections"] = widths
    with pytest.raises(ValueError, match="half the head width"):
        gvlm.synthesize_vlm_config(
            "qwen3_vl", meta, _QWEN3VL_SHAPES, _MM_META)


def test_vlm_config_refuses_a_text_file_with_fewer_deepstack_layers():
    meta = _llm_meta()
    meta["qwen3vl.n_deepstack_layers"] = 1
    with pytest.raises(gvlm.UnsupportedVLMError, match="not a pair"):
        gvlm.synthesize_vlm_config(
            "qwen3_vl", meta, _QWEN3VL_SHAPES, _MM_META)


def test_vlm_config_refuses_a_pair_that_names_no_deepstack_blocks():
    # Neither header has a deepstack key: the load would build no merger
    # and drop the features without a word.
    meta = _llm_meta()
    del meta["qwen3vl.n_deepstack_layers"]
    mm_meta = dict(_MM_META)
    del mm_meta["clip.vision.is_deepstack_layers"]
    with pytest.raises(gvlm.UnsupportedVLMError,
                       match="names no deepstack blocks"):
        gvlm.synthesize_vlm_config(
            "qwen3_vl", meta, _QWEN3VL_SHAPES, mm_meta)
    # One key is enough: the mmproj's flags alone give the blocks.
    cfg = gvlm.synthesize_vlm_config(
        "qwen3_vl", meta, _QWEN3VL_SHAPES, _MM_META)
    assert cfg["vision_config"]["deepstack_visual_indexes"] == [1, 2]


# Cached image features.

def _cache_entry(vlm, pixels, grid):
    """The entry format, built here by hand: the main features, then the
    deepstack features in block order, one array."""
    main, deep = vlm.vision_tower(pixels.astype(mx.float32), grid)
    entry = mx.stack([main] + list(deep))
    mx.eval(entry)
    return entry


def test_encode_images_gives_the_cache_entry():
    vlm = _tiny_vlm()
    _, pixels, grid, n_soft = _image_turn()
    entry = vlm.encode_images(pixels, image_grid_thw=grid)
    # the generate path evaluates an entry this way before it stores it.
    mx.eval(*entry)
    assert entry.shape == (3, n_soft, 64)
    assert float(mx.abs(entry - _cache_entry(vlm, pixels, grid)).max()) < 1e-6


def test_classes_with_their_own_join_do_not_inherit_encode_images():
    # The generate path caches through encode_images where a class has it.
    # The Qwen3.6 and qwen4_exp containers subclass this one and join the
    # main features alone, so the entry of this class is not theirs.
    from gmlx._vendor.mlx_vlm.models.qwen3_5.qwen3_5 import Model as Qwen35
    from gmlx._vendor.mlx_vlm.models.qwen3_5_moe.qwen3_5_moe import (
        Model as Qwen35Moe,
    )
    from gmlx._vendor.mlx_vlm.models.qwen3_vl.qwen3_vl import Model as Qwen3VL
    from gmlx.models.qwen4_exp.vlm_model import Model as Qwen4Exp

    assert hasattr(Qwen3VL, "encode_images")
    assert hasattr(_tiny_vlm(), "encode_images")
    for cls in (Qwen35, Qwen35Moe, Qwen4Exp):
        assert issubclass(cls, Qwen3VL)
        assert not hasattr(cls, "encode_images"), cls
        assert not hasattr(cls, "encode_image"), cls


def test_cached_image_features_keep_the_deepstack_features():
    vlm = _tiny_vlm()
    ids, pixels, grid, _ = _image_turn()
    want = vlm.get_input_embeddings(ids, pixels, image_grid_thw=grid)
    got = vlm.get_input_embeddings(
        ids, pixels, image_grid_thw=grid,
        cached_image_features=_cache_entry(vlm, pixels, grid))
    # the main features land where the tower's would.
    assert float(mx.abs(got.inputs_embeds - want.inputs_embeds).max()) < 1e-6
    assert want.deepstack_visual_embeds is not None
    assert got.deepstack_visual_embeds is not None
    assert float(mx.abs(mx.array(got.deepstack_visual_embeds)
                        - mx.array(want.deepstack_visual_embeds)).max()) < 1e-6


def test_cached_image_features_give_the_same_hidden_state():
    vlm = _tiny_vlm()
    ids, pixels, grid, _ = _image_turn()
    n_layers = len(vlm.language_model.model.layers)
    want = vlm.hidden_state_at_layer(
        ids, n_layers, pixel_values=pixels, image_grid_thw=grid)
    got = vlm.hidden_state_at_layer(
        ids, n_layers, pixel_values=pixels, image_grid_thw=grid,
        cached_image_features=_cache_entry(vlm, pixels, grid))
    assert float(mx.abs(got - want).max()) < 1e-5


def test_cached_image_features_give_the_same_logits():
    vlm = _tiny_vlm()
    ids, pixels, grid, _ = _image_turn()
    want = vlm(ids, pixel_values=pixels, image_grid_thw=grid).logits
    got = vlm(ids, pixel_values=pixels, image_grid_thw=grid,
              cached_image_features=_cache_entry(vlm, pixels, grid)).logits
    assert float(mx.abs(got - want).max()) < 1e-5


def test_cached_image_features_without_pixels_are_not_ignored():
    # A caller that kept the features and dropped the pixels gets the image
    # embeddings and the image positions, not the text-only ones.
    vlm = _tiny_vlm()
    ids, pixels, grid, _ = _image_turn()
    want = vlm.get_input_embeddings(ids, pixels, image_grid_thw=grid)
    got = vlm.get_input_embeddings(
        ids, None, image_grid_thw=grid,
        cached_image_features=_cache_entry(vlm, pixels, grid))
    assert float(mx.abs(got.inputs_embeds - want.inputs_embeds).max()) < 1e-6
    assert got.position_ids.tolist() == want.position_ids.tolist()
    assert float(mx.abs(mx.array(got.deepstack_visual_embeds)
                        - mx.array(want.deepstack_visual_embeds)).max()) < 1e-6


def test_main_features_alone_are_refused_as_a_cache_entry():
    vlm = _tiny_vlm()
    ids, pixels, grid, _ = _image_turn()
    main, _ = vlm.vision_tower(pixels.astype(mx.float32), grid)
    for features in (main, main[None], [main]):
        with pytest.raises(ValueError, match="deepstack features"):
            vlm.get_input_embeddings(
                ids, pixels, image_grid_thw=grid,
                cached_image_features=features)


def test_a_cache_entry_of_another_image_is_refused():
    vlm = _tiny_vlm()
    ids, _, grid, n_soft = _image_turn()
    _, pixels, other_grid, other_soft = _image_turn(grid_h=4, grid_w=6)
    assert other_soft != n_soft
    with pytest.raises(ValueError, match="image tokens"):
        vlm.get_input_embeddings(
            ids, None, image_grid_thw=grid,
            cached_image_features=_cache_entry(vlm, pixels, other_grid))


class _CountedTower:
    """The tower behind a call counter."""

    def __init__(self, tower):
        self.tower = tower
        self.calls = 0

    def __call__(self, *args, **kwargs):
        self.calls += 1
        return self.tower(*args, **kwargs)


def test_a_vision_cache_is_filled_once_and_read_by_key():
    # The serve path passes its cache and a key, and the class fills it.
    from gmlx._vendor.mlx_vlm.vision_cache import VisionFeatureCache

    vlm = _tiny_vlm()
    ids, pixels, grid, n_soft = _image_turn()
    want = vlm.get_input_embeddings(ids, pixels, image_grid_thw=grid)
    counted = _CountedTower(vlm.vision_tower)
    vlm.vision_tower = counted
    cache = VisionFeatureCache()
    first = vlm.get_input_embeddings(
        ids, pixels, image_grid_thw=grid, vision_cache=cache, _image_key="a")
    assert counted.calls == 1 and len(cache) == 1
    assert cache.get("a").shape == (3, n_soft, 64)
    again = vlm.get_input_embeddings(
        ids, pixels, image_grid_thw=grid, vision_cache=cache, _image_key="a")
    assert counted.calls == 1
    for got in (first, again):
        assert float(mx.abs(got.inputs_embeds - want.inputs_embeds).max()) < 1e-6
        assert float(mx.abs(mx.array(got.deepstack_visual_embeds)
                            - mx.array(want.deepstack_visual_embeds)).max()) < 1e-6
    # another key is another image.
    vlm.get_input_embeddings(
        ids, pixels, image_grid_thw=grid, vision_cache=cache, _image_key="b")
    assert counted.calls == 2 and len(cache) == 2


# The float dtypes of a loaded pair, read from two files.

_FILE_LAYOUTS = ["bf16", "f16", "all_bf16", "all_f16"]


def _mint_tiny_qwen3vl_pair(text_path, mmproj_path, layout):
    """A loadable Qwen3-VL pair. The text file is float32. ``layout`` is how
    the mmproj stores its floats: ``bf16`` and ``f16`` as llama.cpp writes
    those files (the matrices of the blocks and the mergers in the 16-bit
    type, with the patch slices too in F16; the position table and every
    1-D tensor float32), ``all_bf16`` and ``all_f16`` with every tensor in
    the 16-bit type. Returns the mmproj tensors as the float32 values the
    file holds."""
    from gguf import GGUFWriter
    from gguf.constants import GGMLQuantizationType
    from tokenizers import pre_tokenizers

    tokens = _MARKERS + sorted(pre_tokenizers.ByteLevel.alphabet()) + ["He"]
    hid, layers, heads, kv, ffn, hd = 64, 2, 4, 2, 128, 16
    rng = np.random.default_rng(5)

    def values(name, shape):
        if name.endswith(".bias"):
            a = 0.1 * rng.standard_normal(shape)
        elif len(shape) == 1:
            a = 1.0 + 0.1 * rng.standard_normal(shape)
        else:
            a = rng.standard_normal(shape) / np.sqrt(np.prod(shape[1:]))
        return np.ascontiguousarray(a, dtype=np.float32)

    w = GGUFWriter(text_path, "qwen3vl")
    w.add_uint32("qwen3vl.embedding_length", hid)
    w.add_uint32("qwen3vl.block_count", layers)
    w.add_uint32("qwen3vl.attention.head_count", heads)
    w.add_uint32("qwen3vl.attention.head_count_kv", kv)
    w.add_uint32("qwen3vl.feed_forward_length", ffn)
    w.add_uint32("qwen3vl.context_length", 1024)
    w.add_float32("qwen3vl.attention.layer_norm_rms_epsilon", 1e-6)
    w.add_uint32("qwen3vl.attention.key_length", hd)
    w.add_float32("qwen3vl.rope.freq_base", 5000000.0)
    w.add_array("qwen3vl.rope.dimension_sections", [4, 2, 2, 0])
    w.add_uint32("qwen3vl.n_deepstack_layers", 2)
    w.add_string("tokenizer.ggml.model", "gpt2")
    w.add_string("tokenizer.ggml.pre", "qwen2")
    w.add_array("tokenizer.ggml.tokens", tokens)
    w.add_array("tokenizer.ggml.merges", ["H e"])
    w.add_array("tokenizer.ggml.token_type",
                [3] * len(_MARKERS) + [1] * (len(tokens) - len(_MARKERS)))
    w.add_uint32("tokenizer.ggml.eos_token_id", 2)
    w.add_uint32("tokenizer.ggml.padding_token_id", 0)
    text = {"token_embd.weight": (len(tokens), hid),
            "output_norm.weight": (hid,),
            "output.weight": (len(tokens), hid)}
    for i in range(layers):
        text.update({
            f"blk.{i}.attn_norm.weight": (hid,),
            f"blk.{i}.ffn_norm.weight": (hid,),
            f"blk.{i}.attn_q.weight": (heads * hd, hid),
            f"blk.{i}.attn_k.weight": (kv * hd, hid),
            f"blk.{i}.attn_v.weight": (kv * hd, hid),
            f"blk.{i}.attn_output.weight": (hid, heads * hd),
            f"blk.{i}.attn_q_norm.weight": (hd,),
            f"blk.{i}.attn_k_norm.weight": (hd,),
            f"blk.{i}.ffn_gate.weight": (ffn, hid),
            f"blk.{i}.ffn_up.weight": (ffn, hid),
            f"blk.{i}.ffn_down.weight": (hid, ffn),
        })
    for name, shape in text.items():
        w.add_tensor(name, values(name, shape))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()

    w = GGUFWriter(mmproj_path, "clip")
    w.add_bool("clip.has_vision_encoder", True)
    w.add_string("clip.projector_type", "qwen3vl_merger")
    for key in ("patch_size", "image_size", "block_count", "embedding_length",
                "feed_forward_length", "projection_dim",
                "attention.head_count", "spatial_merge_size"):
        w.add_uint32(f"clip.vision.{key}", _MM_META[f"clip.vision.{key}"])
    w.add_float32("clip.vision.attention.layer_norm_epsilon", 1e-6)
    w.add_array("clip.vision.is_deepstack_layers",
                _MM_META["clip.vision.is_deepstack_layers"])
    w.add_array("clip.vision.image_mean", [0.5, 0.5, 0.5])
    w.add_array("clip.vision.image_std", [0.5, 0.5, 0.5])
    stored = {}
    for name, shape in ((k, tuple(v.shape)) for k, v in _mm_arrays().items()):
        a = values(name, shape)
        matrix = len(shape) == 2 and "position_embd" not in name
        patch = "patch_embd.weight" in name
        codec = {"bf16": "BF16" if matrix else "F32",
                 "f16": "F16" if matrix or patch else "F32",
                 "all_bf16": "BF16", "all_f16": "F16"}[layout]
        if codec == "BF16":
            bits = (a.view(np.uint32) >> 16).astype(np.uint16)
            w.add_tensor(name, bits, raw_shape=a.shape,
                         raw_dtype=GGMLQuantizationType.BF16)
            a = (bits.astype(np.uint32) << 16).view(np.float32)
        elif codec == "F16":
            half = a.astype(np.float16)
            w.add_tensor(name, half)
            a = half.astype(np.float32)
        else:
            w.add_tensor(name, a)
        stored[name] = a
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return stored


def _load_tiny_pair(tmp_path, monkeypatch, layout):
    from test_loader_api import _keep_vlm_render_bindings

    _keep_vlm_render_bindings(monkeypatch)
    text, mmproj = (str(tmp_path / n) for n in ("t.gguf", "mmproj.gguf"))
    stored = _mint_tiny_qwen3vl_pair(text, mmproj, layout)
    model, _, _ = gvlm.load_vlm_model(text, mmproj)
    return model, stored


def _float32_tower(model, stored):
    """A second tower that holds the file's values as float32 arrays."""
    from gmlx._vendor.mlx_vlm.models.qwen3_vl.vision import VisionModel

    vision, skipped, _ = gvlm.remap_vision_arrays(
        {k: mx.array(v) for k, v in stored.items()}, "qwen3_vl",
        mm_meta=_MM_META)
    assert skipped == []
    tower = VisionModel(model.config.vision_config)
    tower.load_weights([(k.removeprefix("vision_tower."), v)
                        for k, v in vision.items()], strict=True)
    tower.eval()
    mx.eval(tower.parameters())
    return tower


def _tower_inputs():
    # 4 x 6 patches: a grid that is not the position table's own 4 x 4, so
    # the table is interpolated.
    mx.random.seed(3)
    return mx.random.normal((24, 3 * 2 * 16 * 16)), mx.array([[1, 4, 6]])


def _rel_l2(got, want):
    got, want = got.astype(mx.float32), want.astype(mx.float32)
    return float(mx.linalg.norm(got - want) / mx.linalg.norm(want))


@pytest.mark.parametrize("layout", _FILE_LAYOUTS)
def test_the_tower_of_a_16_bit_mmproj_computes_in_float32(
        tmp_path, monkeypatch, layout):
    # The tower's residual stream needs float32. A file that stores the
    # tower in a 16-bit float loads it as float32 arrays with the file's
    # values, and its features are those of a float32 tower.
    model, stored = _load_tiny_pair(tmp_path, monkeypatch, layout)
    tower = model.vision_tower
    params = dict(tree_flatten(tower.parameters()))
    assert {str(v.dtype) for v in params.values()} == {"mlx.core.float32"}
    ref = _float32_tower(model, stored)
    for name, want in tree_flatten(ref.parameters()):
        assert mx.array_equal(params[name], want), name

    pixels, grid = _tower_inputs()
    main, deep = tower(pixels, grid)
    want, want_deep = ref(pixels, grid)
    assert main.dtype == mx.float32
    assert [d.dtype for d in deep] == [mx.float32, mx.float32]
    assert _rel_l2(main, want) < 1e-6
    for got, r in zip(deep, want_deep):
        assert _rel_l2(got, r) < 1e-6


def test_image_features_join_in_the_language_model_dtype(
        tmp_path, monkeypatch):
    # The float32 features are cast where they join the token embeddings,
    # so the decoder stays in its own dtype. Float32 deepstack features
    # would promote every layer after the first to float32.
    from gmlx.load.dtypes import activation_dtype

    model, _ = _load_tiny_pair(tmp_path, monkeypatch, "bf16")
    lm_dtype = model.language_model.model.embed_tokens.weight.dtype
    assert lm_dtype == activation_dtype()
    if lm_dtype == mx.float32:
        pytest.skip("the activation dtype is float32: nothing to tell apart")
    pixels, grid = _tower_inputs()
    image_pad, n_soft = _MARKERS.index("<|image_pad|>"), 6
    ids = mx.array([[1, _MARKERS.index("<|vision_start|>")]
                    + [image_pad] * n_soft
                    + [_MARKERS.index("<|vision_end|>"), 10, 11]])
    assert model.config.image_token_index == image_pad

    main, deep = model.vision_tower(pixels, grid)
    assert main.dtype == mx.float32
    feats = model.get_input_embeddings(ids, pixels, image_grid_thw=grid)
    assert feats.inputs_embeds.dtype == lm_dtype
    assert mx.array_equal(feats.inputs_embeds[0, 2:2 + n_soft],
                          main.astype(lm_dtype))
    assert len(feats.deepstack_visual_embeds) == 2
    for got, d in zip(feats.deepstack_visual_embeds, deep):
        assert got.dtype == lm_dtype
        assert mx.array_equal(got, d.astype(lm_dtype))
    assert model.encode_images(pixels, image_grid_thw=grid).dtype == lm_dtype
    logits = model(ids, pixels, image_grid_thw=grid).logits
    assert logits.dtype == lm_dtype
    assert logits.shape == (1, ids.shape[1], 264)
