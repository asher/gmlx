"""qwen4exp VLM wrapper: config synth, model build, mrope parity."""

import mlx.core as mx
import mlx.nn as nn

from gmlx import vlm as gvlm
from gmlx.config_synth import synthesize_config
from gmlx.qwen4_exp_model import (
    Model as TextModel,
    ModelArgs,
    ensure_registered,
)
from tests.test_config_synth import _QWEN4EXP_SHAPES, _qwen4exp_meta

_MM_META = {
    "clip.projector_type": "qwen3vl_merger",
    "clip.vision.patch_size": 16,
    "clip.vision.image_size": 768,
    "clip.vision.block_count": 2,
    "clip.vision.embedding_length": 32,
    "clip.vision.feed_forward_length": 64,
    "clip.vision.projection_dim": 64,
    "clip.vision.attention.head_count": 2,
    "clip.vision.spatial_merge_size": 2,
}


def test_resolve_and_config_synth():
    assert gvlm.resolve_vlm_model_type("qwen4exp", _MM_META) == "qwen4_exp"
    cfg = gvlm.synthesize_vlm_config(
        "qwen4_exp", _qwen4exp_meta(True, True), _QWEN4EXP_SHAPES, _MM_META)
    assert cfg["model_type"] == "qwen4_exp"
    assert cfg["vision_config"]["spatial_merge_size"] == 2
    assert cfg["vision_config"]["deepstack_visual_indexes"] == []


def _tiny_vlm():
    ensure_registered()
    from gmlx import qwen4_exp_vlm_model as vm

    vm.ensure_registered()
    text = dict(synthesize_config(_qwen4exp_meta(True, True), _QWEN4EXP_SHAPES))
    cfg = {
        "model_type": "qwen4_exp",
        "text_config": text,
        "vision_config": {
            "model_type": "qwen3_5",
            "depth": 1, "hidden_size": 32, "intermediate_size": 64,
            "out_hidden_size": text["hidden_size"], "num_heads": 2,
            "patch_size": 16, "spatial_merge_size": 2,
            "temporal_patch_size": 2, "num_position_embeddings": 4,
            "in_channels": 3, "image_size": 64,
            "deepstack_visual_indexes": [],
        },
        "vocab_size": text["vocab_size"],
        "image_token_id": 20, "video_token_id": 21,
        "vision_start_token_id": 18, "vision_end_token_id": 19,
    }
    model, _ = gvlm.build_vlm_model(cfg)
    mx.random.seed(0)
    flat = {}
    for k, v in nn.utils.tree_flatten(model.parameters()):
        flat[k] = (mx.random.normal(v.shape) * 0.05).astype(v.dtype)
    model.load_weights(list(flat.items()), strict=False)
    mx.eval(model.parameters())
    return model


def test_build_and_text_forward_matches_text_model():
    """Text-only through the VLM wrapper == the vendored text Model."""
    model = _tiny_vlm()
    text_args = ModelArgs.from_dict(
        dict(synthesize_config(_qwen4exp_meta(True, True), _QWEN4EXP_SHAPES)))
    ref = TextModel(text_args)
    weights = dict(nn.utils.tree_flatten(model.language_model.parameters()))
    ref.load_weights([(k, v) for k, v in weights.items()], strict=True)

    ids = mx.random.randint(0, text_args.vocab_size, (1, 10))
    cache = model.make_cache()
    out = model.language_model(ids, cache=cache).logits
    ref_out = ref(ids, cache=ref.make_cache())
    assert mx.abs(out - ref_out).max() < 1e-5


def test_mrope_positions_equal_streams_match_text_path():
    """position_ids with equal t/h/w streams reproduce the offset path
    through prefill + decode (cache-position block rope included)."""
    model = _tiny_vlm()
    lm = model.language_model
    vocab = lm.args.vocab_size
    ids = mx.random.randint(0, vocab, (1, 11))
    nxt = mx.random.randint(0, vocab, (1, 1))

    c_text = lm.make_cache()
    a = lm(ids, cache=c_text).logits
    a2 = lm(nxt, cache=c_text).logits

    c_pos = lm.make_cache()
    pos = mx.broadcast_to(mx.arange(11, dtype=mx.int32)[None, None], (3, 1, 11))
    b = lm(ids, cache=c_pos, position_ids=pos).logits
    lm._rope_deltas = mx.zeros((1, 1), dtype=mx.int32)
    b2 = lm(nxt, cache=c_pos).logits
    assert mx.abs(a - b).max() < 2e-4
    assert mx.abs(a2 - b2).max() < 2e-4
    lm._rope_deltas = None


def test_get_rope_index_borrowed_semantics():
    model = _tiny_vlm()
    lm = model.language_model
    # 4 image patches -> merge 2 -> 1 soft token at position 2.
    ids = mx.array([[1, 18, 20, 19, 5, 6]])
    grid = mx.array([[1, 2, 2]])
    pos, deltas = lm.get_rope_index(ids, image_grid_thw=grid)
    assert pos.shape == (3, 1, 6)
    assert deltas.shape == (1, 1)
    # streams differ inside the image, equal outside
    p = pos[:, 0].tolist()
    assert p[0][:2] == p[1][:2] == p[2][:2]


def test_registered_module_and_seam_row():
    import sys

    from gmlx import qwen4_exp_vlm_model as vm
    from gmlx.upstream_seams import VENDORED_MLX_VLM_MODULES

    vm.ensure_registered()
    assert "mlx_vlm.models.qwen4_exp" in sys.modules
    assert VENDORED_MLX_VLM_MODULES["gmlx.qwen4_exp_vlm_model"] == \
        "mlx_vlm.models.qwen4_exp"
