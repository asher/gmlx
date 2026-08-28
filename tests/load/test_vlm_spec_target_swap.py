"""Spec-target seam: the VLM path builds the same hook-bearing language
model the text MTP path uses (owned qwen3.5/gemma4 classes), constructed
from the VLM's real ModelConfig so mrope on image turns stays correct.
No GGUFs: tiny synthetic configs, real mlx-vlm wrapper classes.
"""

from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import pytest

pytest.importorskip("mlx_vlm.models.qwen3_5.language")

from mlx_vlm.models.gemma4 import language as _G
from mlx_vlm.models.gemma4.config import TextConfig as G4TextConfig
from mlx_vlm.models.qwen3_5.config import ModelConfig as Q35ModelConfig
from mlx_vlm.models.qwen3_5.config import TextConfig as Q35TextConfig
from mlx_vlm.models.qwen3_5.config import VisionConfig as Q35VisionConfig
from mlx_vlm.models.qwen3_5.language import LanguageModel as Q35LanguageModel
from mlx_vlm.models.qwen3_5.qwen3_5 import Model as Q35Model

import gmlx.models.gemma4.owned as gemma4_owned
import gmlx.models.qwen35.owned as qwen35_owned
from gmlx.load.loader import _spec_hook_key, _vlm_spec_language_model
from gmlx.load.vlm import _swap_spec_language_model


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


def _g4_cfg():
    return G4TextConfig(
        model_type="gemma4_text",
        hidden_size=64,
        num_hidden_layers=6,
        intermediate_size=128,
        num_attention_heads=4,
        head_dim=16,
        global_head_dim=32,
        rms_norm_eps=1e-6,
        vocab_size=128,
        vocab_size_per_layer_input=128,
        num_key_value_heads=2,
        num_kv_shared_layers=2,
        hidden_size_per_layer_input=0,
        sliding_window=32,
        sliding_window_pattern=3,
        tie_word_embeddings=True,
    )


def _log(*_a, **_k):
    pass


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


def test_swap_is_a_no_op_when_owned_disabled(monkeypatch):
    monkeypatch.setenv("GMLX_QWEN_OWNED", "0")
    model = _q35_vlm()
    lm_before = model.language_model
    _swap_spec_language_model(model, "qwen3_5", log=_log)
    assert model.language_model is lm_before
    assert not qwen35_owned.is_owned_language_model(model)


def test_qwen35_moe_row_resolves_the_owned_moe_class():
    row = _vlm_spec_language_model("qwen3_5_moe")
    assert row is not None
    cls, _build = row
    assert cls is qwen35_owned.language_model_class("qwen3_5_moe")


def test_gemma4_swap_and_no_chunked_prefill_copy():
    """gemma4 and gemma4_unified share one row (same stock LanguageModel
    class); gemma4_unified stamps no_chunked_prefill at construction and
    the swap must carry it."""
    for mt in ("gemma4", "gemma4_unified"):
        stock = _G.LanguageModel(_g4_cfg())
        stock.no_chunked_prefill = True
        model = SimpleNamespace(
            language_model=stock,
            config=SimpleNamespace(text_config=_g4_cfg()),
        )
        _swap_spec_language_model(model, mt, log=_log)
        assert isinstance(model.language_model,
                          gemma4_owned.OwnedGemma4LanguageModel), mt
        assert model.language_model.no_chunked_prefill is True, mt


def test_no_row_for_vendored_and_unwired_model_types():
    # Vendored archs carry their hook mixins in their own vlm_model;
    # unwired archs fail loud at the per-arch hook check downstream.
    for mt in ("muse_glimmer", "glm5_next", "qwen4_exp",
               "pixtral", "llava", "qwen3_omni_moe", "kimi_k25"):
        assert _vlm_spec_language_model(mt) is None, mt


def test_spec_hook_key_aliases():
    assert _spec_hook_key("gemma4") == "gemma4_text"
    assert _spec_hook_key("gemma4_unified") == "gemma4_text"
    for mt in ("qwen3_5", "qwen3_5_moe", "muse_glimmer", "glm5_next",
               "qwen4_exp"):
        assert _spec_hook_key(mt) == mt
