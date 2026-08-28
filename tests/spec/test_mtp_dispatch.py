"""Shared drafter dispatch and arming for the text and VLM MTP loaders:
one kind-dispatched assistant loader, one companion auto-resolver, one
post-install arming helper - so every drafter shape and every patch
regime routes identically on both paths.
"""

from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import pytest

pytest.importorskip("mlx_vlm.models.qwen3_5.language")

import gmlx.load.arch_table as arch_table
import gmlx.models.qwen35.owned as qwen35_owned
import gmlx.spec.mtp_load as mtp_load
from gmlx.load.loader import (
    _MTP_TARGET_HOOKS,
    _MTP_TARGET_HOOKS_BY_TYPE,
    _spec_hook_key,
)
from gmlx.spec.mtp_load import (
    _arm_spec_target,
    _load_assistant_drafter,
    _resolve_companion_drafter,
)


# --- assistant dispatch ------------------------------------------------------


def _stub_loaders(monkeypatch, arches):
    calls = []
    monkeypatch.setattr(mtp_load, "_drafter_header_arch",
                        lambda p: arches.get(p))
    for name in ("_load_deepseek4_mtp_drafter", "_load_qwen4exp_mtp_drafter",
                 "_load_nemotron_mtp_drafter", "_load_dflash_drafter"):
        monkeypatch.setattr(
            mtp_load, name,
            lambda path, model, cfg, *, zero_copy, log, _n=name:
                calls.append((_n, path)) or _n)
    monkeypatch.setattr(
        mtp_load, "_load_gemma4_assistant_drafter",
        lambda path, model, *, zero_copy, log:
            calls.append(("_load_gemma4_assistant_drafter", path))
            or "_load_gemma4_assistant_drafter")
    return calls


@pytest.mark.parametrize("model_type,drafter_arch,want", [
    ("deepseek_v4", "deepseek4-dspark", "_load_deepseek4_mtp_drafter"),
    ("qwen4_exp", "qwen4exp-mtp", "_load_qwen4exp_mtp_drafter"),
    ("nemotron_h", "nemotron_h_moe", "_load_nemotron_mtp_drafter"),
    ("qwen3_5", "dflash", "_load_dflash_drafter"),
    ("muse_glimmer", "dflash", "_load_dflash_drafter"),
    ("gemma4_text", "gemma4_assistant", "_load_gemma4_assistant_drafter"),
])
def test_assistant_dispatch_routes_by_kind(monkeypatch, model_type,
                                           drafter_arch, want):
    """Both paths hand this helper their text-config dict; the routing must
    be a function of (model_type, drafter header) only. In particular a
    qwen4exp-mtp companion must never fall into the gemma4 loader (the old
    VLM branch's binary dflash-or-gemma4 dispatch did exactly that)."""
    calls = _stub_loaders(monkeypatch, {"d.gguf": drafter_arch})
    got = _load_assistant_drafter(
        "d.gguf", model=None, text_config_dict={"model_type": model_type},
        zero_copy=True, log=lambda *a: None)
    assert got == want
    assert calls == [(want, "d.gguf")]


# --- companion auto-resolution -----------------------------------------------


def test_companion_auto_domain_matches_the_table():
    assert set(arch_table.MTP_COMPANION_AUTO_MODEL_TYPES) == {
        "deepseek_v4", "qwen4_exp", "muse_glimmer"}
    # Every auto family has a drafter-arch row to search with.
    for mt in arch_table.MTP_COMPANION_AUTO_MODEL_TYPES:
        assert arch_table.drafter_arches(mt)


@pytest.mark.parametrize("model_type", ["deepseek_v4", "qwen4_exp",
                                        "muse_glimmer"])
def test_companion_resolver_promotes_and_raises(monkeypatch, model_type):
    import gmlx.load.discovery as discovery

    found = {"path": "/x/companion.gguf"}
    monkeypatch.setattr(discovery, "find_mtp_companion",
                        lambda gguf, arches: found["path"])
    assert _resolve_companion_drafter(
        model_type, "/x/t.gguf", log=lambda *a: None) == "/x/companion.gguf"

    found["path"] = None
    with pytest.raises(ValueError, match="companion"):
        _resolve_companion_drafter(model_type, "/x/t.gguf",
                                   log=lambda *a: None)


def test_companion_resolver_skips_native_head_families(monkeypatch):
    """qwen3_5 has a dflash row in MTP_DRAFTER_ARCHES but must NOT
    auto-promote: its native in-GGUF head wins over a sidecar, so a DFlash2
    companion stays explicit --draft-gguf."""
    import gmlx.load.discovery as discovery

    monkeypatch.setattr(discovery, "find_mtp_companion",
                        lambda gguf, arches: "/x/sneaky-dflash2.gguf")
    for mt in ("qwen3_5", "qwen3_5_moe", "gemma4_text", "glm5_next",
               "hy_v3", None):
        assert _resolve_companion_drafter(
            mt, "/x/t.gguf", log=lambda *a: None) is None, mt


# --- per-arch hook rows ------------------------------------------------------


def test_vlm_hook_rows_resolve_per_arch():
    """The VLM loader's fail-loud check must use the same per-arch rows as
    _build_mtp_target, via the alias map for the gemma4 VLM types."""
    assert (_MTP_TARGET_HOOKS_BY_TYPE.get(_spec_hook_key("gemma4"))
            == _MTP_TARGET_HOOKS_BY_TYPE["gemma4_text"])
    assert (_MTP_TARGET_HOOKS_BY_TYPE.get(_spec_hook_key("gemma4_unified"))
            == _MTP_TARGET_HOOKS_BY_TYPE["gemma4_text"])
    for mt in ("muse_glimmer", "glm5_next", "qwen4_exp", "deepseek_v4"):
        assert _spec_hook_key(mt) in _MTP_TARGET_HOOKS_BY_TYPE
    # qwen3_5 takes the full default row.
    assert _MTP_TARGET_HOOKS_BY_TYPE.get(
        _spec_hook_key("qwen3_5"), _MTP_TARGET_HOOKS) is _MTP_TARGET_HOOKS


# --- arming ------------------------------------------------------------------


def _tiny_q35(owned: bool):
    from mlx_vlm.models.qwen3_5.config import TextConfig

    cfg = TextConfig(
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
    top = SimpleNamespace(vision_config=SimpleNamespace(spatial_merge_size=2),
                          image_token_id=124, video_token_id=125,
                          vision_start_token_id=126)
    if owned:
        lm = qwen35_owned.language_model_class("qwen3_5")(cfg, top)
    else:
        from mlx_vlm.models.qwen3_5.language import LanguageModel

        lm = LanguageModel(cfg, top)
    mx.eval(lm.parameters())
    return SimpleNamespace(language_model=lm)


def test_arming_owned_tree_arms_gdn_not_stock_patches(monkeypatch):
    calls = []
    monkeypatch.setattr(mtp_load, "prepare_gdn",
                        lambda m: calls.append("prepare_gdn") or 3)
    monkeypatch.setattr(mtp_load, "_install_stock_qwen35_verify_patches",
                        lambda m: calls.append("stock_patches"))
    monkeypatch.setattr(mtp_load, "_patch_dense_head_verify",
                        lambda m: calls.append("dense_head"))
    model = _tiny_q35(owned=True)
    _arm_spec_target(model, {"model_type": "qwen3_5"},
                     stock_vlm_regime=True, log=lambda *a: None)
    assert calls == ["prepare_gdn", "dense_head"]


def test_arming_stock_tree_regime_discriminator(monkeypatch):
    """VLM stock fallback gets the full stock patch set; the text stock
    fallback stays bare (its debugging contract). Dense-head verify runs on
    both."""
    for regime, want in ((True, ["stock_patches", "dense_head"]),
                         (False, ["dense_head"])):
        calls = []
        monkeypatch.setattr(mtp_load, "prepare_gdn",
                            lambda m: calls.append("prepare_gdn") or 3)
        monkeypatch.setattr(mtp_load, "_install_stock_qwen35_verify_patches",
                            lambda m: calls.append("stock_patches"))
        monkeypatch.setattr(mtp_load, "_patch_dense_head_verify",
                            lambda m: calls.append("dense_head"))
        model = _tiny_q35(owned=False)
        _arm_spec_target(model, {"model_type": "qwen3_5"},
                         stock_vlm_regime=regime, log=lambda *a: None)
        assert calls == want, regime


def test_arming_gemma4_is_a_no_op(monkeypatch):
    for name in ("prepare_gdn", "_install_stock_qwen35_verify_patches",
                 "_patch_dense_head_verify"):
        monkeypatch.setattr(mtp_load, name,
                            lambda m: pytest.fail("gemma4 must arm nothing"))
    model = SimpleNamespace(language_model=SimpleNamespace())
    for mt in ("gemma4", "gemma4_text", "gemma4_unified"):
        _arm_spec_target(model, {"model_type": mt},
                         stock_vlm_regime=True, log=lambda *a: None)


def test_qwen4exp_ba_cat_is_idempotent():
    """prepare_runtime may run in both load_vlm_model and _arm_spec_target;
    the second pass must not re-cat the b/a weights."""
    from gmlx.load.config_synth import synthesize_config
    from gmlx.models.qwen4_exp.model import ModelArgs, Model, prepare_runtime
    from test_config_synth import _QWEN4EXP_SHAPES, _qwen4exp_meta

    args = ModelArgs.from_dict(
        dict(synthesize_config(_qwen4exp_meta(True, True), _QWEN4EXP_SHAPES)))
    model = Model(args)
    mx.eval(model.parameters())
    first = prepare_runtime(model)
    weights_before = dict(nn.utils.tree_flatten(model.parameters()))
    second = prepare_runtime(model)
    assert second["gdn_ba_cat"] == 0
    assert second["gdn_fused"] == first["gdn_fused"]
    for k, v in nn.utils.tree_flatten(model.parameters()):
        assert mx.array_equal(v, weights_before[k]), k
