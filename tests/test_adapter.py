#!/usr/bin/env python3
"""GGUF LoRA adapter loader (P1): the pure ``build_adapter_plan`` core - a/b
pairing, rank/scale, GGUF->module name remap, and loud failures on malformed or
mismatched adapters. CPU-only (fake KV dict + mx arrays, no GGUF file, no model)."""
from __future__ import annotations

import mlx.core as mx
import pytest

from gmlx import adapter  # noqa: E402

# in=128, out=64, rank=8 -> shared dim 8 is unambiguous
IN, OUT, R = 128, 64, 8


def _meta(arch="qwen3", alpha=16.0, atype="lora"):
    return {"general.architecture": arch, "adapter.type": atype,
            "adapter.lora.alpha": alpha}


def _pair(base):
    # a: in->r, b: r->out (orientation is pinned later in P2; rank is the shared dim)
    return {f"{base}.lora_a": mx.zeros((R, IN)),
            f"{base}.lora_b": mx.zeros((OUT, R))}


def test_build_plan_qwen3_dense_maps_to_module_paths():
    arrays = {**_pair("blk.0.attn_q.weight"), **_pair("blk.1.ffn_gate.weight")}
    plan = adapter.build_adapter_plan(_meta(), arrays)
    assert plan.alpha == 16.0 and plan.arch == "qwen3"
    assert set(plan.modules) == {
        "model.layers.0.self_attn.q_proj", "model.layers.1.mlp.gate_proj"}
    m = plan.modules["model.layers.0.self_attn.q_proj"]
    assert m.transform == "passthrough"          # qwen3 dense: no qk_permute


def test_scale_is_alpha_over_rank():
    plan = adapter.build_adapter_plan(_meta(alpha=32.0), _pair("blk.0.attn_v.weight"))
    m = plan.modules["model.layers.0.self_attn.v_proj"]
    assert m.rank == R
    assert m.scale == pytest.approx(32.0 / R)


def test_rank_is_the_shared_dim():
    plan = adapter.build_adapter_plan(_meta(), _pair("blk.0.attn_output.weight"))
    assert plan.modules["model.layers.0.self_attn.o_proj"].rank == R


def test_llama_attn_q_carries_qk_permute_transform():
    # llama-family q/k are stored permuted; P1 records the hint, P2 applies it.
    plan = adapter.build_adapter_plan(_meta(arch="llama"), _pair("blk.0.attn_q.weight"))
    assert plan.modules["model.layers.0.self_attn.q_proj"].transform == "qk_permute"


def test_missing_pair_half_raises():
    arrays = {"blk.0.attn_q.weight.lora_a": mx.zeros((R, IN))}   # no .lora_b
    with pytest.raises(ValueError, match="missing its .lora_b"):
        adapter.build_adapter_plan(_meta(), arrays)


def test_unmappable_target_raises_not_silently_dropped():
    arrays = _pair("blk.0.some_unknown_tensor.weight")
    with pytest.raises(ValueError, match="doesn't map to a module"):
        adapter.build_adapter_plan(_meta(), arrays)


def test_non_lora_adapter_type_raises():
    with pytest.raises(ValueError, match="expected 'lora'"):
        adapter.build_adapter_plan(_meta(atype="control_vector"),
                                   _pair("blk.0.attn_q.weight"))


def test_missing_alpha_raises():
    meta = _meta()
    del meta["adapter.lora.alpha"]
    with pytest.raises(ValueError, match="no adapter.lora.alpha"):
        adapter.build_adapter_plan(meta, _pair("blk.0.attn_q.weight"))


def test_arch_mismatch_raises():
    with pytest.raises(ValueError, match="different base model"):
        adapter.build_adapter_plan(_meta(arch="llama"),
                                   _pair("blk.0.attn_q.weight"), base_arch="qwen3")


def test_base_arch_drives_remap_when_adapter_arch_absent():
    meta = _meta()
    del meta["general.architecture"]
    plan = adapter.build_adapter_plan(meta, _pair("blk.0.attn_q.weight"),
                                      base_arch="qwen3")
    assert "model.layers.0.self_attn.q_proj" in plan.modules


def test_no_pairs_raises():
    with pytest.raises(ValueError, match="no lora_a/lora_b"):
        adapter.build_adapter_plan(_meta(), {"general.foo": mx.zeros((2,))})


def test_rank_inference_no_shared_dim_raises():
    arrays = {"blk.0.attn_q.weight.lora_a": mx.zeros((7, IN)),
              "blk.0.attn_q.weight.lora_b": mx.zeros((OUT, 9))}   # 7 vs 9: no shared
    with pytest.raises(ValueError, match="share no dimension"):
        adapter.build_adapter_plan(_meta(), arrays)
