#!/usr/bin/env python3
"""The ``dflash`` GGUF container holds two unrelated drafters: DeepSeek-V4's
DSpark (the unsloth release, normalized into the gmlx-native namespace) and
Muse Glimmer's, which keeps its own Qwen3-shaped remap. Both are covered here,
along with the tensor-presence test that tells them apart.
Name/metadata logic only - synthetic arrays, no GGUF files, no model load."""
from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")

from gmlx.spec.mtp_load import (  # noqa: E402
    dflash_container,
    normalize_dflash_arrays,
    remap_deepseek4_dspark_arrays,
    remap_muse_glimmer_dflash_arrays,
)

# Tensor skeleton of the unsloth dspark-DeepSeek-V4-Flash-0731-Q8_0.gguf:
# per-stage leaves match the native sidecar; drafter-level names go through
# llama.cpp's root map (fc / enc.output_norm / output_norm / markov_w1 /
# markov_w2 / conf_proj / output_hc_*).
_STAGE_LEAVES = (
    "attn_kv.weight", "attn_kv_a_norm.weight", "attn_norm.weight",
    "attn_output_a.weight", "attn_output_b.weight", "attn_q_a.weight",
    "attn_q_a_norm.weight", "attn_q_b.weight", "attn_sinks.weight",
    "exp_probs_b.bias", "ffn_down_exps.weight", "ffn_down_shexp.weight",
    "ffn_gate_exps.weight", "ffn_gate_inp.weight", "ffn_gate_shexp.weight",
    "ffn_norm.weight", "ffn_up_exps.weight", "ffn_up_shexp.weight",
    "hc_attn_base.weight", "hc_attn_fn.weight", "hc_attn_scale.weight",
    "hc_ffn_base.weight", "hc_ffn_fn.weight", "hc_ffn_scale.weight",
)
_ROOT_LEAVES = (
    "fc.weight", "enc.output_norm.weight", "output_norm.weight",
    "markov_w1.weight", "markov_w2.weight", "conf_proj.weight",
    "output_hc_base.weight", "output_hc_fn.weight", "output_hc_scale.weight",
)
_META = {
    "general.architecture": "dflash",
    "dflash.block_size": 5,
    "dflash.target_layers": [41, 42, 43],
    "tokenizer.ggml.mask_token_id": 128799,
}


def _dflash_fixture(n_stages=3):
    arrays = {}
    for k in range(n_stages):
        for leaf in _STAGE_LEAVES:
            arrays[f"blk.{k}.{leaf}"] = mx.zeros((2, 2))
    for leaf in _ROOT_LEAVES:
        arrays[leaf] = mx.zeros((2, 2))
    arrays["markov_w1.weight"] = mx.zeros((129280, 256))
    kquant = {"blk.0.attn_kv.weight": "q8_0", "fc.weight": "q8_0"}
    arrays["blk.0.attn_kv.scales"] = mx.zeros((1,))
    arrays["fc.scales"] = mx.zeros((1,))
    return arrays, kquant, dict(_META)


def test_stage_tensors_land_under_mtp():
    arrays, kquant, meta = _dflash_fixture()
    n_arrays, n_kquant, _ = normalize_dflash_arrays(arrays, kquant, meta)
    assert "mtp.0.attn_kv.weight" in n_arrays
    assert "mtp.2.ffn_gate_exps.weight" in n_arrays
    assert n_kquant["mtp.0.attn_kv.weight"] == "q8_0"
    assert "mtp.0.attn_kv.scales" in n_arrays        # quant siblings ride along
    assert not any(n.startswith("blk.") for n in n_arrays)


def test_root_tensors_get_dspark_names():
    arrays, kquant, meta = _dflash_fixture()
    n_arrays, n_kquant, _ = normalize_dflash_arrays(arrays, kquant, meta)
    assert "mtp.0.main_proj.weight" in n_arrays      # fc
    assert "mtp.0.main_proj.scales" in n_arrays
    assert n_kquant["mtp.0.main_proj.weight"] == "q8_0"
    assert "mtp.0.main_norm.weight" in n_arrays      # enc.output_norm
    assert "mtp.2.norm.weight" in n_arrays           # output_norm -> final stage
    assert "mtp.2.markov_head.markov_w1.weight" in n_arrays
    assert "mtp.2.confidence_head.proj.weight" in n_arrays
    assert "mtp.2.hc_head_fn.weight" in n_arrays     # output_hc_fn


def test_meta_translation_unshifts_target_layers():
    """llama.cpp writes capture layers +1 (its layer 0 is the embedding) and
    the noise token as the tokenizer mask token; both must be undone."""
    arrays, kquant, meta = _dflash_fixture()
    _, _, n_meta = normalize_dflash_arrays(arrays, kquant, meta)
    assert n_meta["dspark.target_layer_ids"] == [40, 41, 42]
    assert n_meta["dspark.noise_token_id"] == 128799
    assert n_meta["dspark.block_size"] == 5
    assert n_meta["dspark.markov_rank"] == 256       # from markov_w1's shape


def test_unknown_tensor_is_hard_error():
    arrays, kquant, meta = _dflash_fixture()
    arrays["mystery.weight"] = mx.zeros((2, 2))
    with pytest.raises(RuntimeError, match="unknown tensor"):
        normalize_dflash_arrays(arrays, kquant, meta)


def test_normalized_set_survives_dspark_remap():
    """End to end through the closed-set dspark remap: every translated name
    must be known and every drafter param must be produced."""
    arrays, kquant, meta = _dflash_fixture()
    n_arrays, n_kquant, _ = normalize_dflash_arrays(arrays, kquant, meta)
    # attn_output_a is reshaped (o_groups x o_lora_rank x -1) by the remap
    for k in range(3):
        n_arrays[f"mtp.{k}.attn_output_a.weight"] = mx.zeros((8 * 1024, 4))
    hf, hf_kq, stats = remap_deepseek4_dspark_arrays(
        n_arrays, n_kquant, n_stages=3, o_groups=8, o_lora_rank=1024
    )
    for key in ("main_proj.weight", "main_norm.weight", "norm.weight",
                "markov_w1.weight", "markov_w2.weight",
                "confidence_proj.weight", "hc_head.fn",
                "stages.0.block.attn.wkv.weight",
                "stages.2.block.ffn.switch_mlp.gate_proj.weight"):
        assert key in hf, key
    assert stats["mapped"] == len(
        [n for n in n_arrays if not n.endswith(".scales")]
    )


# --- Muse Glimmer's dflash container -----------------------------------------

# 5 Qwen3-shaped layers plus three roots; no markov/confidence/hc head and no
# attn_q_a, which is what separates it from DSpark.
_MUSE_BLK_LEAVES = (
    "attn_norm.weight", "attn_q.weight", "attn_k.weight", "attn_v.weight",
    "attn_output.weight", "attn_q_norm.weight", "attn_k_norm.weight",
    "ffn_norm.weight", "ffn_gate.weight", "ffn_up.weight", "ffn_down.weight",
)
_MUSE_ROOT_LEAVES = ("fc.weight", "enc.output_norm.weight", "output_norm.weight")


def _muse_fixture(n_layers=5):
    arrays = {}
    for i in range(n_layers):
        for leaf in _MUSE_BLK_LEAVES:
            arrays[f"blk.{i}.{leaf}"] = mx.zeros((2, 2))
    for leaf in _MUSE_ROOT_LEAVES:
        arrays[leaf] = mx.zeros((2, 2))
    kquant = {"blk.0.attn_q.weight": "q4_k", "fc.weight": "q4_k"}
    arrays["blk.0.attn_q.scales"] = mx.zeros((1,))
    arrays["fc.scales"] = mx.zeros((1,))
    return arrays, kquant


def test_container_tells_the_two_drafters_apart():
    muse, _ = _muse_fixture()
    dspark, _, _ = _dflash_fixture()
    assert dflash_container(muse) == "muse_glimmer"
    assert dflash_container(dspark) == "dspark"


def test_container_rejects_an_unrecognized_dflash():
    with pytest.raises(Exception):
        dflash_container({"blk.0.mystery.weight": mx.zeros((2, 2))})


def test_muse_remap_produces_every_drafter_param():
    arrays, kquant = _muse_fixture()
    hf, hf_kq, stats = remap_muse_glimmer_dflash_arrays(arrays, kquant)
    for key in ("fc.weight", "hidden_norm.weight", "norm.weight",
                "layers.0.input_layernorm.weight",
                "layers.0.self_attn.q_proj.weight",
                "layers.0.self_attn.k_proj.weight",
                "layers.0.self_attn.v_proj.weight",
                "layers.0.self_attn.o_proj.weight",
                "layers.0.self_attn.q_norm.weight",
                "layers.0.self_attn.k_norm.weight",
                "layers.0.post_attention_layernorm.weight",
                "layers.4.mlp.gate_proj.weight",
                "layers.4.mlp.up_proj.weight",
                "layers.4.mlp.down_proj.weight"):
        assert key in hf, key
    assert hf_kq["layers.0.self_attn.q_proj.weight"] == "q4_k"
    assert hf_kq["fc.weight"] == "q4_k"
    assert not any(n.startswith("blk.") for n in hf)
    assert stats["mapped"] == len(
        [n for n in arrays if not n.endswith(".scales")])


def test_muse_remap_keeps_the_two_norms_distinct():
    """``enc.output_norm`` is the post-fc encoder norm and ``output_norm`` the
    drafter's final norm; swapping them silently corrupts the borrowed head."""
    arrays, kquant = _muse_fixture()
    arrays["enc.output_norm.weight"] = mx.full((2, 2), 3.0)
    arrays["output_norm.weight"] = mx.full((2, 2), 7.0)
    hf, _, _ = remap_muse_glimmer_dflash_arrays(arrays, kquant)
    assert float(hf["hidden_norm.weight"][0, 0]) == 3.0
    assert float(hf["norm.weight"][0, 0]) == 7.0


def test_muse_remap_unknown_tensor_is_hard_error():
    arrays, kquant = _muse_fixture()
    arrays["blk.0.mystery.weight"] = mx.zeros((2, 2))
    with pytest.raises(Exception):
        remap_muse_glimmer_dflash_arrays(arrays, kquant)


def test_dspark_path_is_unchanged_by_the_split():
    """The muse container must not perturb DSpark: its fixture still lands on
    the dspark namespace with the same mapped count."""
    arrays, kquant, meta = _dflash_fixture()
    n_arrays, n_kquant, n_meta = normalize_dflash_arrays(arrays, kquant, meta)
    assert "mtp.0.main_proj.weight" in n_arrays
    assert n_meta["dspark.target_layer_ids"] == [40, 41, 42]
    assert not any(n.startswith("blk.") for n in n_arrays)
