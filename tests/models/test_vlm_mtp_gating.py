"""Install gating for qwen MTP targets on the vlm MTP path.

Since the spec-target seam, ``load_vlm_mtp_model``'s target is built on
the owned classes by default; a ``GMLX_QWEN_OWNED=0`` fallback stays
stock. The install sites therefore gate on the built tree
(``qwen35_owned.is_owned_language_model``), never on the config's
model_type. A stock tree takes the full patched regime (the vlm
regime); an owned tree arms ``prepare_gdn``, which raises on a tree
with no owned GatedDeltaNet.
"""

from types import SimpleNamespace

import mlx.core as mx
import pytest

pytest.importorskip("mlx_vlm.models.qwen3_5.language")

from mlx_vlm.models.qwen3_5 import language as _L
from mlx_vlm.models.qwen3_5.config import TextConfig as Q35TextConfig
from mlx_vlm.models.qwen3_5.language import LanguageModel as Q35LanguageModel

import gmlx.upstream.gdn_patches as gp
import gmlx.models.qwen35.attn as qwen35_attn
import gmlx.models.qwen35.gdn as qwen35_gdn
import gmlx.models.qwen35.owned as qwen35_owned
import gmlx.models.qwen35.verify_fold as qwen35_verify_fold
from gmlx.spec.mtp_load import _install_stock_qwen35_verify_patches
from gmlx.spec.ragged_decode import install_unified_ragged_plan


def _cfg():
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


def _top():
    return SimpleNamespace(
        vision_config=SimpleNamespace(spatial_merge_size=2),
        image_token_id=124,
        video_token_id=125,
        vision_start_token_id=126,
    )


def _stock():
    lm = Q35LanguageModel(_cfg(), _top())
    mx.eval(lm.parameters())
    return lm


def _owned():
    lm = qwen35_owned.language_model_class("qwen3_5")(_cfg(), _top())
    mx.eval(lm.parameters())
    return lm


@pytest.fixture
def _patch_state():
    """Save/reset/restore every global + install latch the stock patch
    set touches, per the test_envflags convention: restoring the
    symbols without the latches would turn the next install into a
    silent no-op."""
    saved = {
        "tvl": _L._target_verify_linear,
        "tvlpa": _L._target_verify_left_padded_attention,
        "sdpa": _L.scaled_dot_product_attention,
        "ragged": _L._qwen3_5_ragged_decode_attention,
        "gdn_call": _L.Qwen3_5GatedDeltaNet.__call__,
        "bf16_flag": gp._BF16_VERIFY_LINEAR_PATCHED,
        "bsdpa_flag": gp._BATCHED_VERIFY_SDPA_PATCHED,
        "fold_flag": qwen35_verify_fold._installed,
        "fv_installed": gp._FUSED_VERIFY_PATCH.installed,
        "fv_stock": gp._FUSED_VERIFY_PATCH.stock,
    }
    gp._BF16_VERIFY_LINEAR_PATCHED = False
    gp._BATCHED_VERIFY_SDPA_PATCHED = False
    qwen35_verify_fold._installed = False
    gp._FUSED_VERIFY_PATCH.installed = False
    yield
    _L._target_verify_linear = saved["tvl"]
    _L._target_verify_left_padded_attention = saved["tvlpa"]
    _L.scaled_dot_product_attention = saved["sdpa"]
    _L._qwen3_5_ragged_decode_attention = saved["ragged"]
    _L.Qwen3_5GatedDeltaNet.__call__ = saved["gdn_call"]
    gp._BF16_VERIFY_LINEAR_PATCHED = saved["bf16_flag"]
    gp._BATCHED_VERIFY_SDPA_PATCHED = saved["bsdpa_flag"]
    qwen35_verify_fold._installed = saved["fold_flag"]
    gp._FUSED_VERIFY_PATCH.installed = saved["fv_installed"]
    gp._FUSED_VERIFY_PATCH.stock = saved["fv_stock"]


def test_is_owned_language_model_reads_the_tree():
    assert not qwen35_owned.is_owned_language_model(_stock())
    assert qwen35_owned.is_owned_language_model(_owned())
    # And through a VLM-shaped wrapper (language_model attribute).
    wrapper = SimpleNamespace(language_model=_stock())
    assert not qwen35_owned.is_owned_language_model(wrapper)


def test_stock_tree_gets_full_patch_regime(_patch_state):
    stock = _stock()
    assert not qwen35_owned.is_owned_language_model(stock)
    _install_stock_qwen35_verify_patches(stock)

    # Every global a stock forward reads is now the gmlx implementation.
    assert _L._target_verify_linear.__module__.startswith("gmlx")
    assert _L._target_verify_left_padded_attention.__module__.startswith(
        "gmlx"
    )
    assert _L.scaled_dot_product_attention.__module__ == (
        "gmlx.models.qwen35.verify_fold"
    )
    assert (
        _L._qwen3_5_ragged_decode_attention
        is qwen35_attn.ragged_decode_attention
    )
    assert _L.Qwen3_5GatedDeltaNet.__call__.__module__ == "gmlx.upstream.gdn_patches"


def test_unified_ragged_plan_reinstalls_after_restore(_patch_state):
    """The installer keys on the installed object's identity, not a
    latch, so a restore-then-reinstall cycle works without latch
    surgery (and no attribute is stamped onto the owned function)."""
    install_unified_ragged_plan()
    assert (
        _L._qwen3_5_ragged_decode_attention
        is qwen35_attn.ragged_decode_attention
    )
    assert not hasattr(
        qwen35_attn.ragged_decode_attention, "_kq_gguf_unified_ragged_plan"
    )
    stock_fn = _L._qwen3_5_ragged_decode_attention
    _L._qwen3_5_ragged_decode_attention = lambda *a, **k: None
    install_unified_ragged_plan()
    assert _L._qwen3_5_ragged_decode_attention is stock_fn


def test_prepare_gdn_raises_on_stock_tree():
    with pytest.raises(RuntimeError, match="armed 0 layers"):
        qwen35_gdn.prepare_gdn(_stock())


def test_prepare_gdn_arms_owned_tree():
    owned = _owned()
    # 4 layers at full_attention_interval=4 -> 3 gated-delta layers.
    assert qwen35_gdn.prepare_gdn(owned) == 3
