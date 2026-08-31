#!/usr/bin/env python3
"""Nemotron-3.5-Lightning (nemotron_h_moe) load-path units: trunk/NextN layer
split in config synthesis, trunk overflow strip, the closed MTP-block remap,
and sidecar classification in discovery."""

from __future__ import annotations

import numpy as np
import pytest

import mlx.core as mx

from gmlx.load.config_synth import synthesize_config

ARCH = "nemotron_h_moe"


def _lightning_meta(block_count=5, nextn=1):
    """Tiny Lightning-shaped KV: per-layer arrays span block_count INCLUDING
    the trailing MTP block (which reads as attention: kv>0)."""
    kv = [0, 4, 0, 0, 4][:block_count]
    ff = [0, 0, 128, 128, 128][:block_count]
    return {
        "general.architecture": ARCH,
        f"{ARCH}.embedding_length": 64,
        f"{ARCH}.block_count": block_count,
        f"{ARCH}.nextn_predict_layers": nextn,
        f"{ARCH}.attention.head_count": 4,
        f"{ARCH}.attention.head_count_kv": kv,
        f"{ARCH}.feed_forward_length": ff,
        f"{ARCH}.context_length": 1024,
        f"{ARCH}.attention.layer_norm_rms_epsilon": 1e-6,
        f"{ARCH}.attention.key_length": 16,
        f"{ARCH}.ssm.inner_size": 128,
        f"{ARCH}.ssm.time_step_rank": 4,
        f"{ARCH}.ssm.state_size": 16,
        f"{ARCH}.ssm.conv_kernel": 4,
        f"{ARCH}.ssm.group_count": 2,
        f"{ARCH}.expert_count": 8,
        f"{ARCH}.expert_used_count": 2,
        f"{ARCH}.expert_feed_forward_length": 128,
        f"{ARCH}.expert_shared_count": 1,
        f"{ARCH}.expert_shared_feed_forward_length": 128,
        "tokenizer.ggml.tokens": ["t"] * 32,
    }


def test_pattern_excludes_nextn_block():
    """The hybrid pattern and num_hidden_layers cover the trunk only; the
    trailing MTP block (kv>0, would read as '*') must not become a trunk
    layer."""
    config = synthesize_config(_lightning_meta(), tensor_shapes={})
    assert config["num_hidden_layers"] == 4
    assert config["mtp_num_hidden_layers"] == 1
    assert len(config["hybrid_override_pattern"]) == 4
    assert config["hybrid_override_pattern"] == ["M", "*", "E", "E"]


def test_pattern_no_nextn_unchanged():
    config = synthesize_config(_lightning_meta(nextn=0), tensor_shapes={})
    assert config["num_hidden_layers"] == 5
    assert len(config["hybrid_override_pattern"]) == 5


def test_strip_nextn_trunk_overflow():
    from gmlx.load.loader import strip_nextn_trunk_overflow

    meta = {f"{ARCH}.nextn_predict_layers": 1, f"{ARCH}.block_count": 5}
    w = {
        "backbone.layers.3.mixer.in_proj.weight": 1,
        "backbone.layers.4.mixer.in_proj.weight": 2,   # MTP block, drop
        "model.layers.4.post_attention_layernorm.weight": 3,  # canonical-map
        "lm_head.weight": 4,
    }
    kq = {"backbone.layers.4.mixer.in_proj.weight": "q4_k"}
    dropped = strip_nextn_trunk_overflow(w, kq, meta, ARCH)
    assert dropped == 2
    assert set(w) == {"backbone.layers.3.mixer.in_proj.weight", "lm_head.weight"}
    assert not kq


def test_strip_noop_other_arch():
    from gmlx.load.loader import strip_nextn_trunk_overflow

    w = {"backbone.layers.4.mixer.in_proj.weight": 1}
    assert strip_nextn_trunk_overflow(w, {}, {}, "qwen3") == 0
    assert w


# ---------------------------------------------------------------------------
# MTP-block remap (closed set)
# ---------------------------------------------------------------------------

N_HEAD, N_KV, HEAD_DIM, H = 2, 1, 8, 16


def _mtp_arrays(blk=4):
    p = f"blk.{blk}."
    a = {
        p + "nextn.eh_proj.weight": mx.zeros((H, 2 * H)),
        p + "nextn.enorm.weight": mx.zeros((H,)),
        p + "nextn.hnorm.weight": mx.zeros((H,)),
        p + "nextn.shared_head_norm.weight": mx.zeros((H,)),
        p + "attn_norm.weight": mx.zeros((H,)),
        p + "attn_q.weight": mx.arange(N_HEAD * HEAD_DIM * H,
                                       dtype=mx.float32).reshape(
                                           N_HEAD * HEAD_DIM, H),
        p + "attn_k.weight": mx.arange(N_KV * HEAD_DIM * H,
                                       dtype=mx.float32).reshape(
                                           N_KV * HEAD_DIM, H),
        p + "attn_v.weight": mx.zeros((N_KV * HEAD_DIM, H)),
        p + "attn_output.weight": mx.zeros((H, N_HEAD * HEAD_DIM)),
        p + "post_attention_norm.weight": mx.zeros((H,)),
        p + "ffn_gate_inp.weight": mx.zeros((8, H)),
        p + "ffn_up_exps.weight": mx.zeros((8, 32, H)),
        p + "ffn_down_exps.weight": mx.zeros((8, H, 32)),
        p + "ffn_up_shexp.weight": mx.zeros((32, H)),
        p + "ffn_down_shexp.weight": mx.zeros((H, 32)),
        p + "exp_probs_b.bias": mx.zeros((8,)),
        # companion-GGUF globals: shared from the target, skipped here
        "token_embd.weight": mx.zeros((32, H)),
        "output.weight": mx.zeros((32, H)),
        "output_norm.weight": mx.zeros((H,)),
    }
    return a


def test_remap_mtp_closed_set():
    from gmlx.load.transforms import qk_permute_wire
    from gmlx.models.nemotron_h.mtp import remap_nemotron_mtp_arrays

    arrays = _mtp_arrays()
    w, kq, stats = remap_nemotron_mtp_arrays(
        arrays, {}, first_mtp_block=4, n_head=N_HEAD, n_head_kv=N_KV)
    assert stats["mapped"] == 16
    assert stats["skipped"] == 3
    assert stats["qk_permute_applied"] == 2
    assert w["fc.weight"].shape == (H, 2 * H)
    assert "layers.0.mlp.gate.e_score_correction_bias" in w
    assert "token_embd.weight" not in w and "output.weight" not in w
    assert mx.array_equal(
        w["layers.0.self_attn.q_proj.weight"],
        qk_permute_wire(arrays["blk.4.attn_q.weight"], N_HEAD))
    assert mx.array_equal(
        w["layers.0.self_attn.k_proj.weight"],
        qk_permute_wire(arrays["blk.4.attn_k.weight"], N_KV))
    assert not kq


def test_remap_mtp_unknown_tensor_raises():
    from gmlx.models.nemotron_h.mtp import remap_nemotron_mtp_arrays

    arrays = _mtp_arrays()
    arrays["blk.4.mystery.weight"] = mx.zeros((2, 2))
    with pytest.raises(RuntimeError, match="unknown tensor"):
        remap_nemotron_mtp_arrays(
            arrays, {}, first_mtp_block=4, n_head=N_HEAD, n_head_kv=N_KV)


def test_remap_mtp_kquant_scales_carry():
    from gmlx.models.nemotron_h.mtp import remap_nemotron_mtp_arrays

    arrays = _mtp_arrays()
    arrays["blk.4.ffn_up_shexp.scales"] = mx.zeros((32, 2))
    kq_meta = {"blk.4.ffn_up_shexp.weight": "q5_0"}
    w, kq, _ = remap_nemotron_mtp_arrays(
        arrays, kq_meta, first_mtp_block=4, n_head=N_HEAD, n_head_kv=N_KV)
    assert kq["layers.0.mlp.shared_experts.up_proj.weight"] == "q5_0"
    assert "layers.0.mlp.shared_experts.up_proj.scales" in w


# ---------------------------------------------------------------------------
# discovery: mtp-*.gguf sidecar classification
# ---------------------------------------------------------------------------

def _mint_gguf(path, *, nextn, tensor_names):
    from gguf import GGUFWriter

    w = GGUFWriter(str(path), ARCH)
    w.add_string("general.name", "lightning-fixture")
    w.add_uint32(f"{ARCH}.block_count", 5)
    w.add_uint32(f"{ARCH}.embedding_length", 64)
    if nextn:
        w.add_uint32(f"{ARCH}.nextn_predict_layers", nextn)
    for name in tensor_names:
        w.add_tensor(name, np.zeros((4, 4), dtype=np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return str(path)


def test_probe_mtp_only_sidecar(tmp_path):
    from gmlx.load.discovery import _probe_mtp_only
    from gmlx.load.headerscan import scan_gguf

    sidecar = _mint_gguf(
        tmp_path / "mtp-x.gguf", nextn=1,
        tensor_names=["token_embd.weight", "blk.4.nextn.eh_proj.weight",
                      "blk.4.attn_q.weight"])
    kv = scan_gguf(sidecar, include_tensors=False).kv
    assert _probe_mtp_only(sidecar, kv) is True


def test_probe_mtp_only_full_model(tmp_path):
    from gmlx.load.discovery import _probe_mtp_only
    from gmlx.load.headerscan import scan_gguf

    full = _mint_gguf(
        tmp_path / "model.gguf", nextn=1,
        tensor_names=["token_embd.weight", "blk.0.attn_q.weight",
                      "blk.4.nextn.eh_proj.weight"])
    kv = scan_gguf(full, include_tensors=False).kv
    assert _probe_mtp_only(full, kv) is False

    plain = _mint_gguf(
        tmp_path / "plain.gguf", nextn=0,
        tensor_names=["token_embd.weight", "blk.0.attn_q.weight"])
    kv = scan_gguf(plain, include_tensors=False).kv
    assert _probe_mtp_only(plain, kv) is False
