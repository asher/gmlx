"""Spec-target seam: the VLM path builds the same hook-bearing language
model the text MTP path uses. qwen3.5 swaps in the owned classes,
constructed from the VLM's real ModelConfig so mrope on image turns stays
correct. A plain qwen3.5 VLM build already holds
gmlx.models.qwen35.language's text model, which the copied package's
language path is an alias of. gemma4 builds the owned container, which
already holds the owned language model. No GGUFs: tiny synthetic configs.
"""

import mlx.core as mx
import mlx.nn as nn
import pytest

pytest.importorskip("gmlx._vendor.mlx_vlm.models.qwen3_5.language")

from gmlx._vendor.mlx_vlm.models.qwen3_5 import language as _CopyPathL
from gmlx._vendor.mlx_vlm.models.qwen3_5.config import ModelConfig as Q35ModelConfig
from gmlx._vendor.mlx_vlm.models.qwen3_5.config import VisionConfig as Q35VisionConfig
from gmlx._vendor.mlx_vlm.models.qwen3_5.qwen3_5 import Model as Q35Model

import gmlx.models.gemma4.language as gemma4_lang
import gmlx.models.qwen35.owned as qwen35_owned
from gmlx.load.mtp_target import _spec_hook_key, _vlm_spec_language_model
from gmlx.load.vlm import _swap_spec_language_model, build_vlm_model
from gmlx._vendor.mlx_vlm.models.qwen3_5.config import TextConfig as Q35TextConfig
from gmlx.models.qwen35.language import LanguageModel as Q35LanguageModel


def _q35_text_cfg():
    return Q35TextConfig(
        model_type="qwen3_5",
        hidden_size=64,
        intermediate_size=128,
        linear_num_value_heads=4,
        linear_num_key_heads=2,
        linear_key_head_dim=32,
        linear_value_head_dim=32,
        linear_conv_kernel_dim=4,
        num_hidden_layers=4,
        num_attention_heads=4,
        rms_norm_eps=1e-6,
        vocab_size=128,
        num_key_value_heads=2,
        max_position_embeddings=2048,
        tie_word_embeddings=True,
        head_dim=32,
        rope_parameters={
            "type": "default",
            "mrope_section": [2, 1, 1],
            "rope_theta": 100000,
            "partial_rotary_factor": 0.25,
        },
        full_attention_interval=4,
    )


def _q35_vlm():
    """The tree load_vlm_model hands the spec swap: the copied container,
    which builds the text model of gmlx.models.qwen35.language."""
    cfg = Q35ModelConfig(
        text_config=_q35_text_cfg(),
        vision_config=Q35VisionConfig(
            depth=1, hidden_size=32, intermediate_size=64, out_hidden_size=64,
            num_heads=2, patch_size=14, num_position_embeddings=16,
            deepstack_visual_indexes=[]),
        model_type="qwen3_5",
        vocab_size=128,
        image_token_id=124,
        video_token_id=125,
        vision_start_token_id=126,
        vision_end_token_id=127,
    )
    return Q35Model(cfg)


def _log(*_a, **_k):
    pass


def test_a_plain_qwen35_build_holds_the_gmlx_text_model():
    """The copied container builds it with no swap after the build, from
    the arguments the swap passed."""
    import gmlx.load.vlm as vlm
    import gmlx.models.qwen35.language as owned_lang

    assert _CopyPathL is owned_lang
    assert not hasattr(vlm, "_swap_qwen35_base_language_model")
    model = _q35_vlm()
    assert type(model.language_model) is Q35LanguageModel
    assert type(model.language_model).__module__ == "gmlx.models.qwen35.language"
    assert model.language_model.config is model.config
    assert model.language_model.args is model.config.text_config


@pytest.mark.parametrize("model_type,path", [
    ("qwen3_5", "gmlx.models.qwen35.language"),
    ("qwen3_5_moe", "gmlx.models.qwen35.moe_language")])
def test_the_loader_build_of_a_qwen35_vlm_holds_the_gmlx_text_model(
        model_type, path):
    import dataclasses
    import importlib

    text = dataclasses.asdict(_q35_text_cfg())
    text["model_type"] = model_type
    if model_type == "qwen3_5_moe":
        text.update(num_experts=4, num_experts_per_tok=2,
                    moe_intermediate_size=64, shared_expert_intermediate_size=64)
    raw = {"model_type": model_type, "text_config": text,
           "vision_config": {"depth": 1, "hidden_size": 32,
                             "intermediate_size": 64, "out_hidden_size": 64,
                             "num_heads": 2, "patch_size": 14,
                             "num_position_embeddings": 16,
                             "deepstack_visual_indexes": []},
           "vocab_size": 128, "image_token_id": 124, "video_token_id": 125,
           "vision_start_token_id": 126, "vision_end_token_id": 127}
    model, _ = build_vlm_model(raw)
    lm_cls = importlib.import_module(path).LanguageModel
    assert type(model.language_model) is lm_cls
    assert lm_cls.__module__ == path


def test_qwen35_swap_yields_owned_class_with_identical_tree():
    model = _q35_vlm()
    assert type(model.language_model) is Q35LanguageModel
    before = {k for k, _ in nn.utils.tree_flatten(
        model.language_model.parameters())}
    _swap_spec_language_model(model, "qwen3_5", log=_log)
    assert qwen35_owned.is_owned_language_model(model)
    after = {k for k, _ in nn.utils.tree_flatten(
        model.language_model.parameters())}
    assert before == after


def test_swapped_lm_keeps_the_real_vision_config():
    """The swap must construct from the VLM's real ModelConfig: mrope
    get_rope_index reads vision_config.spatial_merge_size and the image
    token ids, which the text path's empty default would break."""
    model = _q35_vlm()
    _swap_spec_language_model(model, "qwen3_5", log=_log)
    lm = model.language_model
    assert lm.config is model.config
    ids = mx.array([[1, 126, 124, 127, 5, 6]])
    grid = mx.array([[1, 2, 2]])
    pos, deltas = lm.get_rope_index(ids, image_grid_thw=grid)
    assert pos.shape == (3, 1, 6)
    assert deltas.shape == (1, 1)


def test_qwen35_moe_row_resolves_the_owned_moe_class():
    row = _vlm_spec_language_model("qwen3_5_moe")
    assert row is not None
    cls, _build = row
    assert cls is qwen35_owned.language_model_class("qwen3_5_moe")


def _g4_container_config(model_type):
    text = {"model_type": ("gemma4_unified_text" if model_type == "gemma4_unified"
                           else "gemma4_text"),
            "hidden_size": 64, "num_hidden_layers": 6,
            "intermediate_size": 128, "num_attention_heads": 4,
            "head_dim": 16, "global_head_dim": 32, "vocab_size": 128,
            "vocab_size_per_layer_input": 128, "num_key_value_heads": 2,
            "num_kv_shared_layers": 2, "hidden_size_per_layer_input": 0,
            "sliding_window": 32, "sliding_window_pattern": 3}
    return {"model_type": model_type, "text_config": text,
            "vision_config": {}}


@pytest.mark.parametrize("model_type", ["gemma4", "gemma4_unified"])
def test_a_gemma4_vlm_builds_the_owned_container(model_type):
    """The owned container holds the owned language model, which carries
    the spec hooks, so the spec swap has no row and leaves it in place.
    Its parameter tree is the stock container's, and every config the
    stock module resolves resolves to the same class."""
    import importlib

    from mlx.utils import tree_flatten
    from gmlx._vendor.mlx_vlm.utils import get_model_and_args, update_module_configs

    import gmlx.models.gemma4.gemma4 as owned_g4
    import gmlx.models.gemma4.unified as owned_uni

    raw = _g4_container_config(model_type)
    model, _ = build_vlm_model(dict(raw))
    owned_cls = owned_uni.Model if model_type == "gemma4_unified" else owned_g4.Model
    assert type(model) is owned_cls
    assert type(model.language_model) is gemma4_lang.LanguageModel
    assert not hasattr(model, "no_chunked_prefill")
    assert _vlm_spec_language_model(model_type) is None
    lm = model.language_model
    _swap_spec_language_model(model, model_type, log=_log)
    assert model.language_model is lm

    stock_module, _mt = get_model_and_args(dict(raw))
    cfg = update_module_configs(
        stock_module.ModelConfig.from_dict(raw), stock_module, raw,
        ["text", "vision", "projector", "audio"])
    stock = stock_module.Model(cfg)
    assert type(model.config) is type(cfg)
    for name in ("text_config", "vision_config", "audio_config"):
        assert type(getattr(model.config, name)) is type(getattr(cfg, name)), name
    owned_tree = {k: v.shape for k, v in tree_flatten(model.parameters())}
    stock_tree = {k: v.shape for k, v in tree_flatten(stock.parameters())}
    assert owned_tree == stock_tree
    assert importlib.import_module(type(model).__module__) is not stock_module


def test_no_row_for_vendored_and_unwired_model_types():
    # Vendored archs carry their hook mixins in their own vlm_model, the
    # owned gemma4 containers hold the owned language model, and unwired
    # archs fail loud at the per-arch hook check downstream.
    for mt in ("muse_glimmer", "glm5_next", "qwen4_exp", "deepseek_v4_vl",
               "gemma4", "gemma4_unified", "pixtral", "llava",
               "qwen3_omni_moe", "kimi_k25"):
        assert _vlm_spec_language_model(mt) is None, mt


def test_spec_hook_key_aliases():
    assert _spec_hook_key("gemma4") == "gemma4_text"
    assert _spec_hook_key("gemma4_unified") == "gemma4_text"
    for mt in ("qwen3_5", "qwen3_5_moe", "muse_glimmer", "glm5_next",
               "qwen4_exp"):
        assert _spec_hook_key(mt) == mt
    # the Vision-Exp container carries the deepseek_v4 text hooks
    assert _spec_hook_key("deepseek_v4_vl") == "deepseek_v4"
