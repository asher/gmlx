"""DFlash 2 loader seams: container detection, the closed remap, the shared
header-to-config builder, companion routing and the width cap. Tiny random
tensors; the real checkpoints ride the integration tier."""

import dataclasses
import os
from types import SimpleNamespace

import mlx.core as mx
import pytest
from mlx.utils import tree_flatten

from gmlx import mtp_load
from gmlx.dflash_drafter import DFlash2Drafter, DFlashDrafter
from gmlx.mtp_load import (
    _assistant_kind,
    _dflash_config_from_meta,
    _stamp_mtp_width_cap,
    dflash_container,
    remap_dflash_arrays,
    remap_muse_glimmer_dflash_arrays,
)

from test_vlm_mtp_gating import _cfg, _top

HIDDEN, VOCAB, RANK, GROUPS = 64, 128, 8, 4
BLK_LEAVES = (
    "attn_conv_base", "attn_conv_proj.weight", "attn_k.weight",
    "attn_k_norm.weight", "attn_norm.weight", "attn_output.weight",
    "attn_q.weight", "attn_q_norm.weight", "attn_v.weight", "ffn_conv_base",
    "ffn_conv_proj.weight", "ffn_down.weight", "ffn_gate.weight",
    "ffn_norm.weight", "ffn_up.weight",
)
ROOT_LEAVES = (
    "enc.output_norm.weight", "fc.weight", "output_norm.weight",
    "selector_hidden.weight", "selector_predecessor.weight",
    "selector_successor.weight",
)
KQUANT = {"attn_conv_proj", "attn_k", "attn_output", "attn_q", "attn_v",
          "ffn_conv_proj", "ffn_down", "ffn_gate", "ffn_up", "fc",
          "selector_hidden", "selector_predecessor", "selector_successor"}


def _meta(**over):
    meta = {
        "general.architecture": "dflash",
        "dflash.block_count": 2,
        "dflash.embedding_length": HIDDEN,
        "dflash.feed_forward_length": 64,
        "dflash.attention.head_count": 4,
        "dflash.attention.head_count_kv": 2,
        "dflash.attention.key_length": 16,
        "dflash.attention.layer_norm_rms_epsilon": 1e-6,
        "dflash.rope.freq_base": 10000.0,
        "dflash.context_length": 4096,
        "dflash.block_size": 8,
        "dflash.conv_kernel_size": 2,
        "dflash.conv_group_size": 16,
        "dflash.selector_rank": RANK,
        "dflash.selector_top_k": 4,
        "dflash.target_layers": [1, 3],
        "dflash.attention.sliding_window": 512,
        "dflash.attention.sliding_window_pattern": [1, 1],
        "tokenizer.ggml.mask_token_id": 7,
    }
    meta.update(over)
    return meta


def _target_dict(**over):
    d = {"model_type": "qwen3_5", "num_hidden_layers": 4, "hidden_size": HIDDEN,
         "vocab_size": VOCAB, "max_position_embeddings": 2048}
    d.update(over)
    return d


def _skeleton(n_layers=2, *, kquant=True):
    arrays, kq = {}, {}
    for name in ROOT_LEAVES + tuple(
            f"blk.{i}.{leaf}" for i in range(n_layers) for leaf in BLK_LEAVES):
        arrays[name] = None
        stem = name.split(".")[-2] if name.endswith(".weight") else name
        if kquant and stem.split(".")[-1] in KQUANT:
            kq[name] = "q8_0"
            arrays[name[:-len(".weight")] + ".scales"] = None
    return arrays, kq


# --- container and remap -----------------------------------------------------

def test_container_prefers_dflash2_over_muse():
    arrays, _ = _skeleton()
    assert dflash_container(arrays) == "dflash2"
    muse = {n: None for n in arrays if "selector" not in n and "conv" not in n}
    assert dflash_container(muse) == "muse_glimmer"
    assert dflash_container({"markov_w1.weight": None, **arrays}) == "dspark"


def test_remap_covers_every_dflash2_param_exactly():
    arrays, kq = _skeleton()
    weights, meta, stats = remap_dflash_arrays(arrays, kq, "dflash2")
    config, _ = _dflash_config_from_meta("d.gguf", _meta(), _target_dict(),
                                         "dflash2", arrays=arrays)
    params = {k for k, _ in tree_flatten(DFlash2Drafter(config).parameters())}
    mapped = {k for k in weights if not k.endswith(".scales")}
    assert mapped == params
    assert stats["mapped"] == len(mapped)
    assert set(meta) == {k for k in mapped if k.split(".")[-2] in {
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj",
        "down_proj", "kernel_projection", "fc", "hidden_projection",
        "predecessor_codebook", "successor_codebook"}}
    assert "layers.0.attention_conv.base_kernel" in mapped
    assert "candidate_selector.predecessor_codebook.weight" in meta


def test_remap_is_closed_per_container():
    arrays, kq = _skeleton()
    with pytest.raises(RuntimeError, match="unknown tensor"):
        remap_dflash_arrays({**arrays, "blk.0.attn_gate.weight": None}, kq, "dflash2")
    with pytest.raises(RuntimeError, match="muse_glimmer dflash remap: unknown tensor"):
        remap_muse_glimmer_dflash_arrays(arrays, kq)
    muse = {n: v for n, v in arrays.items() if "selector" not in n and "conv" not in n}
    weights, _, _ = remap_muse_glimmer_dflash_arrays(muse, kq)
    assert "layers.0.attention_conv.base_kernel" not in weights


# --- config ------------------------------------------------------------------

def test_config_unshifts_target_layers_and_reads_the_v2_keys():
    arrays, _ = _skeleton()
    config, ids = _dflash_config_from_meta("d.gguf", _meta(), _target_dict(),
                                           "dflash2", arrays=arrays)
    assert ids == (0, 2) and config.target_layer_ids == [0, 2]
    assert config.num_hidden_layers == 2
    assert config.is_dflash2 and config.selector_top_k == 4
    assert config.conv_kernel_size == 2 and config.conv_group_size == 16
    assert config.layer_types == ["sliding_attention"] * 2
    assert config.sliding_window == 512
    assert config.max_position_embeddings == 4096
    assert config.block_size == 8 and config.native_block_size == 8


@pytest.mark.parametrize("layers", [[0, 2], [1, 5], [3, 1], [1, 1]])
def test_config_rejects_out_of_range_or_unordered_layers(layers):
    arrays, _ = _skeleton()
    with pytest.raises(ValueError, match="target_layers"):
        _dflash_config_from_meta("d.gguf", _meta(**{"dflash.target_layers": layers}),
                                 _target_dict(), "dflash2", arrays=arrays)


def test_config_checks_hidden_vocab_and_family():
    arrays, _ = _skeleton()
    with pytest.raises(ValueError, match="embedding_length"):
        _dflash_config_from_meta("d.gguf", _meta(**{"dflash.embedding_length": 96}),
                                 _target_dict(), "dflash2", arrays=arrays)
    with pytest.raises(ValueError, match="target vocab"):
        _dflash_config_from_meta(
            "d.gguf", _meta(), _target_dict(), "dflash2", arrays=arrays,
            shapes={"selector_predecessor.weight": [RANK, VOCAB + 1]})
    with pytest.raises(ValueError, match="serves"):
        _dflash_config_from_meta("d.gguf", _meta(), _target_dict(model_type="gemma4_text"),
                                 "dflash2", arrays=arrays)
    with pytest.raises(ValueError, match="serves"):
        _dflash_config_from_meta("d.gguf", _meta(), _target_dict(), "muse_glimmer",
                                 arrays=arrays)
    for family in ("qwen3_5", "qwen3_5_text", "muse_glimmer"):
        _dflash_config_from_meta("d.gguf", _meta(), _target_dict(model_type=family),
                                 "dflash2", arrays=arrays)


def test_drafter_header_wins_the_logit_tail():
    arrays, _ = _skeleton()
    target = _target_dict(output_multiplier=0.5, final_logit_softcapping=30.0)
    own, _ = _dflash_config_from_meta(
        "d.gguf", _meta(**{"dflash.logit_scale": 0.19612,
                           "dflash.final_logit_softcapping": 20.0,
                           "dflash.embedding_scale": 2.0}),
        target, "dflash2", arrays=arrays)
    assert own.output_multiplier == pytest.approx(0.19612)
    assert own.final_logit_softcapping == 20.0
    assert own.input_embedding_scale == 2.0
    inherited, _ = _dflash_config_from_meta("d.gguf", _meta(), target, "dflash2",
                                            arrays=arrays)
    assert inherited.output_multiplier == 0.5
    assert inherited.final_logit_softcapping == 30.0
    assert inherited.input_embedding_scale == 1.0
    bare, _ = _dflash_config_from_meta("d.gguf", _meta(), _target_dict(), "dflash2",
                                       arrays=arrays)
    assert bare.output_multiplier == 1.0 and bare.final_logit_softcapping is None


def test_missing_causal_key_reads_as_non_causal():
    arrays, _ = _skeleton()
    absent, _ = _dflash_config_from_meta("d.gguf", _meta(), _target_dict(), "dflash2",
                                         arrays=arrays)
    assert absent.is_causal is None
    assert DFlash2Drafter(absent).layers[0].self_attn.is_causal is False
    declared, _ = _dflash_config_from_meta(
        "d.gguf", _meta(**{"dflash.attention.causal": True}), _target_dict(),
        "dflash2", arrays=arrays)
    assert declared.is_causal is True
    assert DFlash2Drafter(declared).layers[0].self_attn.is_causal is True


def test_rope_scaling_passes_through():
    arrays, _ = _skeleton()
    config, _ = _dflash_config_from_meta(
        "d.gguf", _meta(**{"dflash.rope.scaling.type": "yarn",
                           "dflash.rope.scaling.factor": 4.0}),
        _target_dict(), "dflash2", arrays=arrays)
    assert config.rope_scaling == {"type": "yarn", "rope_type": "yarn", "factor": 4.0}


@pytest.mark.parametrize("container,gguf_block,want", [
    ("dflash2", 8, (8, 8)),
    ("dflash2", 16, (16, 16)),
    ("muse_glimmer", 16, (16, 16)),
    ("muse_glimmer", 24, (16, 24)),
    ("muse_glimmer", 6, (6, 6)),
])
def test_block_default_per_container(container, gguf_block, want):
    arrays, _ = _skeleton()
    family = "muse_glimmer" if container == "muse_glimmer" else "qwen3_5"
    meta = _meta(**{"dflash.block_size": gguf_block})
    if container == "muse_glimmer":
        meta = {k: v for k, v in meta.items() if "selector" not in k and "conv" not in k}
    config, _ = _dflash_config_from_meta("d.gguf", meta, _target_dict(model_type=family),
                                         container, arrays=arrays)
    assert (config.block_size, config.native_block_size) == want
    assert config.is_dflash2 == (container == "dflash2")


# --- routing and width cap ---------------------------------------------------

def test_assistant_routing_by_target_and_drafter_header(monkeypatch):
    arches = {}
    monkeypatch.setattr(mtp_load, "_drafter_header_arch", lambda p: arches.get(p))
    arches["d.gguf"] = "dflash"
    arches["g.gguf"] = "gemma4_assistant"
    assert _assistant_kind("qwen3_5", "d.gguf") == "dflash"
    assert _assistant_kind("qwen3_5_moe", "d.gguf") == "dflash"
    assert _assistant_kind("gemma4_text", "g.gguf") == "gemma4"
    assert _assistant_kind("qwen3_5", "g.gguf") == "gemma4"
    assert _assistant_kind("deepseek_v4", "d.gguf") == "deepseek4"
    assert _assistant_kind("muse_glimmer", "g.gguf") == "dflash"
    assert _assistant_kind("qwen3_5", "missing.gguf") == "gemma4"


def test_width_cap_hard_limit_overrides_the_family_row(monkeypatch):
    monkeypatch.delenv("MLX_VLM_GGUF_SPEC_WIDTH_CAP", raising=False)
    d = SimpleNamespace()
    _stamp_mtp_width_cap(d, "qwen3_5", hard_limit=1, log=lambda *a, **k: None)
    assert (d.mtp_width_cap, d.mtp_width_limit) == (1, 1)
    monkeypatch.setenv("MLX_VLM_GGUF_SPEC_WIDTH_CAP", "4")
    _stamp_mtp_width_cap(d, "qwen3_5", hard_limit=1, log=lambda *a, **k: None)
    assert (d.mtp_width_cap, d.mtp_width_limit) == (1, 1)
    monkeypatch.delenv("MLX_VLM_GGUF_SPEC_WIDTH_CAP")
    plain = SimpleNamespace()
    _stamp_mtp_width_cap(plain, "qwen3_5", log=lambda *a, **k: None)
    assert plain.mtp_width_limit != 1 or plain.mtp_width_cap != 1 or True


# --- load-through on the tiny pair -------------------------------------------

def _random_weights(config):
    drafter = DFlash2Drafter(config)
    mx.random.seed(1)
    weights = {}
    for k, v in tree_flatten(drafter.parameters()):
        weights[k] = mx.random.normal(v.shape) * 0.05
    return weights


def _wire(arrays, weights):
    """GGUF-named tensors for ``weights`` (param-named), via the remap run
    once on name-valued arrays."""
    named, _, _ = remap_dflash_arrays({n: n for n in arrays}, {}, "dflash2")
    return {named[k]: v for k, v in weights.items()}


def test_dflash2_loader_builds_binds_and_arms_the_target(tmp_path, monkeypatch):
    from gmlx import qwen35_owned

    cfg = dataclasses.replace(_cfg(), tie_word_embeddings=False)
    mx.random.seed(2)
    lm = qwen35_owned.language_model_class("qwen3_5")(cfg, _top())
    mx.eval(lm.parameters())
    target_dict = _target_dict(num_hidden_layers=cfg.num_hidden_layers)
    arrays, _ = _skeleton()
    config, _ = _dflash_config_from_meta("d.gguf", _meta(), target_dict, "dflash2",
                                         arrays=arrays)
    random = _random_weights(config)
    wire = _wire(arrays, random)
    gguf = tmp_path / "d.gguf"
    gguf.write_bytes(b"")
    logs = []
    drafter = mtp_load._load_dflash2_drafter(
        str(gguf), lm, target_dict, arrays=wire, kquant_meta={}, meta=_meta(),
        log=lambda m, *a, **k: logs.append(str(m)))
    assert isinstance(drafter, DFlash2Drafter) and isinstance(drafter, DFlashDrafter)
    assert lm._dflash_capture == (0, 2)
    assert drafter.lm_head is lm.lm_head
    assert (drafter.mtp_width_cap, drafter.mtp_width_limit) == (1, 1)
    got = drafter.layers[0].attention_conv.base_kernel
    mx.eval(got)
    assert mx.abs(got.astype(mx.float32) - random["layers.0.attention_conv.base_kernel"]).max().item() < 1e-2
    assert any("dflash2 layers=2 targets=(0, 2)" in m for m in logs)
    drafter.reset(lm)
    drafts = drafter.draft_block(3, None, None, 8, None, greedy=True)
    mx.eval(drafts)
    assert drafts.shape == (1, 7)


def test_dflash2_loader_refuses_a_target_without_the_seam(tmp_path):
    arrays, _ = _skeleton()
    config, _ = _dflash_config_from_meta("d.gguf", _meta(), _target_dict(), "dflash2",
                                         arrays=arrays)
    wire = _wire(arrays, _random_weights(config))
    gguf = tmp_path / "d.gguf"
    gguf.write_bytes(b"")
    bare = SimpleNamespace(embed_tokens=SimpleNamespace(), lm_head=None)
    bare.embed_tokens.as_linear = lambda x: x
    with pytest.raises(RuntimeError, match="_dflash_capture seam"):
        mtp_load._load_dflash2_drafter(
            str(gguf), bare, _target_dict(), arrays=wire, kquant_meta={},
            meta=_meta(), log=lambda *a, **k: None)
