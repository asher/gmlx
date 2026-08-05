#!/usr/bin/env python3
"""dflash -> deepseek4-dspark normalization: llama.cpp's container for the
DSpark drafter (the unsloth release) translated to the gmlx-native namespace.
Name/metadata logic only - synthetic arrays, no GGUF files, no model load."""
from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")

from gmlx.mtp_load import (  # noqa: E402
    normalize_dflash_arrays,
    remap_deepseek4_dspark_arrays,
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
