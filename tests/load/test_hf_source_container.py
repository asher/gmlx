"""An hf_source container config loads the text model a GGUF load builds.

A multimodal HF config nests its text model under text_config. The loader
flattens it to the model type the GGUF synth gives, so the class resolver
and every load step keyed on model_type see the text model. The tiny
configs check the flat config against the text model args the mlx_lm
container builds. The tier 2 case loads Gemma-4 E2B with and without its
HF config and compares logits.
"""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from test_loader_api import _mint_tiny_llama

_FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"

_GEMMA4_TEXT = dict(
    model_type="gemma4_text", hidden_size=64, num_hidden_layers=2,
    intermediate_size=128, num_attention_heads=2, head_dim=16,
    global_head_dim=32, num_key_value_heads=1, num_kv_shared_layers=0,
    hidden_size_per_layer_input=8, vocab_size=128,
    vocab_size_per_layer_input=128, sliding_window=16,
    layer_types=["sliding_attention", "full_attention"],
    tie_word_embeddings=True, use_bidirectional_attention="vision")
_GEMMA3_TEXT = dict(
    model_type="gemma3_text", hidden_size=64, num_hidden_layers=2,
    intermediate_size=128, head_dim=16, sliding_window=16)
_QWEN3_5_TEXT = dict(
    hidden_size=64, num_hidden_layers=4, intermediate_size=128,
    num_attention_heads=4, num_key_value_heads=2, head_dim=16,
    linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=16,
    linear_value_head_dim=16, linear_conv_kernel_dim=4,
    full_attention_interval=4, vocab_size=128, rms_norm_eps=1e-6,
    max_position_embeddings=1024,
    rope_parameters={"rope_theta": 1e6, "partial_rotary_factor": 0.25,
                     "rope_type": "default"})
_QWEN3_MOE_TEXT = dict(
    model_type="qwen3_vl_moe_text", hidden_size=64, num_hidden_layers=2,
    intermediate_size=128, num_attention_heads=4, num_key_value_heads=2,
    head_dim=16, num_experts=4, num_experts_per_tok=2, decoder_sparse_step=1,
    mlp_only_layers=[], moe_intermediate_size=32, rms_norm_eps=1e-6,
    vocab_size=128, rope_theta=1e6, max_position_embeddings=1024,
    norm_topk_prob=True, tie_word_embeddings=False,
    rope_scaling={"rope_type": "default", "mrope_section": [2, 3, 3],
                  "mrope_interleaved": True})
_MINISTRAL3_TEXT = dict(
    model_type="ministral3", hidden_size=64, num_hidden_layers=2,
    intermediate_size=128, num_attention_heads=4, num_key_value_heads=2,
    head_dim=16, vocab_size=128, rms_norm_eps=1e-5,
    max_position_embeddings=1024,
    rope_parameters={"rope_theta": 1e6, "rope_type": "yarn", "factor": 16.0,
                     "original_max_position_embeddings": 16384,
                     "beta_fast": 32.0, "beta_slow": 1.0,
                     "mscale": 1.0, "mscale_all_dim": 1.0,
                     "llama_4_scaling_beta": 0.1})

# (container config, model_type the flat config names, mlx_lm container
# type that builds the reference text args)
_CONTAINERS = {
    "gemma4": ({"model_type": "gemma4", "vocab_size": 128,
                "text_config": _GEMMA4_TEXT, "vision_config": {}},
               "gemma4_text", "gemma4"),
    "gemma4_unified": ({"model_type": "gemma4_unified", "vocab_size": 128,
                        "text_config": dict(_GEMMA4_TEXT,
                                            model_type="gemma4_unified_text"),
                        "vision_config": {}},
                       "gemma4_text", "gemma4"),
    "gemma3": ({"model_type": "gemma3", "text_config": _GEMMA3_TEXT},
               "gemma3_text", "gemma3"),
    "mistral3": ({"model_type": "mistral3", "text_config": _MINISTRAL3_TEXT},
                 "ministral3", "mistral3"),
    "qwen3_5": ({"model_type": "qwen3_5", "tie_word_embeddings": False,
                 "text_config": dict(_QWEN3_5_TEXT, model_type="qwen3_5_text")},
                "qwen3_5", "qwen3_5"),
    "qwen3_5_moe": ({"model_type": "qwen3_5_moe", "tie_word_embeddings": False,
                     "text_config": dict(
                         _QWEN3_5_TEXT, model_type="qwen3_5_moe_text",
                         num_experts=4, num_experts_per_tok=2,
                         moe_intermediate_size=32,
                         shared_expert_intermediate_size=32)},
                    "qwen3_5_moe", "qwen3_5_moe"),
    "qwen3_vl_moe": ({"model_type": "qwen3_vl_moe", "tie_word_embeddings": False,
                      "text_config": _QWEN3_MOE_TEXT},
                     "qwen3_moe", "qwen3_vl_moe"),
}


def _write(tmp_path: Path, cfg: dict) -> str:
    d = tmp_path / "src"
    d.mkdir()
    (d / "config.json").write_text(json.dumps(cfg))
    return str(d)


def _text_args(model) -> dict:
    args = vars(getattr(model, "language_model", model).args)
    return {k: v for k, v in args.items() if k != "model_type"}


def _container_text_args(cfg: dict, container_type: str) -> dict:
    """The text model args the mlx_lm container class builds from ``cfg``."""
    from mlx_lm.utils import _get_classes

    cfg = json.loads(json.dumps(dict(cfg, model_type=container_type)))
    Model, ModelArgs = _get_classes(cfg)
    return _text_args(Model(ModelArgs.from_dict(cfg)))


@pytest.fixture
def resolved(monkeypatch) -> list[str]:
    """The model types the mlx_lm class resolver is asked for."""
    import mlx_lm.utils

    seen: list[str] = []
    real = mlx_lm.utils._get_classes

    def spy(config):
        seen.append(config["model_type"])
        return real(config)

    monkeypatch.setattr(mlx_lm.utils, "_get_classes", spy)
    return seen


@pytest.mark.parametrize("name", sorted(_CONTAINERS))
def test_a_container_builds_the_text_model_its_container_builds(name, tmp_path):
    from gmlx.load.loader import _load_config_from_source, build_model

    container, model_type, container_type = _CONTAINERS[name]
    ref = _container_text_args(container, container_type)
    cfg = _load_config_from_source(_write(tmp_path, container))
    assert cfg["model_type"] == model_type
    assert "text_config" not in cfg
    model, _ = build_model(cfg)
    got = _text_args(model)
    if "vocab_size" not in cfg:
        # A load takes it from the GGUF's token_embd rows.
        del got["vocab_size"], ref["vocab_size"]
    assert got == ref


@pytest.mark.parametrize("name", ["gemma4", "gemma4_unified"])
def test_the_resolver_sees_gemma4_text(name, tmp_path, resolved):
    from gmlx.load.loader import _load_config_from_source, build_model

    cfg = _load_config_from_source(_write(tmp_path, _CONTAINERS[name][0]))
    model, _ = build_model(cfg)
    assert resolved == ["gemma4_text"]
    assert type(model).__module__ == "mlx_lm.models.gemma4_text"


def test_gemma4_leaves_out_the_image_attention_setting(tmp_path):
    # A load without hf_source has no key, and no image tokens to apply it to.
    from gmlx.load.loader import _load_config_from_source

    cfg = _load_config_from_source(_write(tmp_path, _CONTAINERS["gemma4"][0]))
    assert cfg["model_type"] == "gemma4_text"
    assert "use_bidirectional_attention" not in cfg


def test_gemma3_takes_the_container_heads_and_the_gguf_vocab(tmp_path):
    # The container default vocab_size, 262208, counts 64 rows a llama.cpp
    # GGUF does not have. The load takes the token_embd rows instead.
    from gmlx.load.loader import _load_config_from_source, _vocab_from_token_embd

    cfg = _load_config_from_source(_write(tmp_path, _CONTAINERS["gemma3"][0]))
    assert "vocab_size" not in cfg
    assert (cfg["num_attention_heads"], cfg["num_key_value_heads"]) == (8, 4)
    cfg = _vocab_from_token_embd(cfg, {"token_embd.weight": [64, 262144]})
    assert cfg["vocab_size"] == 262144


def test_mistral3_unties_by_default_and_builds_llama_for_a_mistral_text(tmp_path):
    from gmlx.load.loader import _load_config_from_source

    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    cfg = _load_config_from_source(
        _write(tmp_path / "a", _CONTAINERS["mistral3"][0]))
    assert cfg["tie_word_embeddings"] is False
    small = _load_config_from_source(_write(
        tmp_path / "b",
        {"model_type": "mistral3", "text_config": {"model_type": "mistral"}}))
    assert small["model_type"] == "llama"


def test_qwen3_5_exposes_the_linear_heads_to_the_load(tmp_path):
    # The tiled V-head fixup reads the head counts from the top level.
    from gmlx.load.loader import _load_config_from_source
    from gmlx.upstream.gdn_patches import _needs_tiled_v_patch

    for name in ("qwen3_5", "qwen3_5_moe"):
        d = tmp_path / name
        d.mkdir()
        cfg = _load_config_from_source(_write(d, _CONTAINERS[name][0]))
        assert _needs_tiled_v_patch(cfg)


def test_other_configs_pass_through(tmp_path):
    from gmlx.load.loader import _load_config_from_source

    gemma3n = {"model_type": "gemma3n", "text_config": {"hidden_size": 64}}
    for name, cfg in (("flat", _GEMMA4_TEXT), ("gemma3n", gemma3n)):
        (tmp_path / name).mkdir()
        assert _load_config_from_source(_write(tmp_path / name, cfg)) == cfg


@pytest.fixture(scope="module")
def tiny_gguf(tmp_path_factory) -> str:
    p = tmp_path_factory.mktemp("container") / "tiny-llama.gguf"
    _mint_tiny_llama(str(p))
    return str(p)


_G3_HID, _G3_LAYERS, _G3_HEADS, _G3_KV, _G3_FFN, _G3_HD = 64, 2, 4, 2, 128, 16
# The image token fields of a Gemma-3 4B to 27B container config.
_GEMMA3_IMAGE_FIELDS = {"image_token_index": 262144, "boi_token_index": 255999,
                        "eoi_token_index": 256000, "mm_tokens_per_image": 256}


def _mint_tiny_gemma3(path: str) -> None:
    """A loadable gemma3 GGUF with one token_embd row per token, as
    llama.cpp writes Gemma-3: 262144 rows where the HF config counts
    262208."""
    from gguf import GGUFWriter
    from test_loader_api import _MERGES, _TOKENS

    w = GGUFWriter(path, "gemma3")
    for key, value in (("embedding_length", _G3_HID),
                       ("block_count", _G3_LAYERS),
                       ("attention.head_count", _G3_HEADS),
                       ("attention.head_count_kv", _G3_KV),
                       ("feed_forward_length", _G3_FFN),
                       ("context_length", 1024),
                       ("attention.key_length", _G3_HD),
                       ("attention.value_length", _G3_HD),
                       ("attention.sliding_window", 16),
                       ("attention.sliding_window_pattern", _G3_LAYERS)):
        w.add_uint32(f"gemma3.{key}", value)
    w.add_float32("gemma3.attention.layer_norm_rms_epsilon", 1e-6)
    w.add_float32("gemma3.rope.freq_base", 1e6)
    w.add_string("tokenizer.ggml.model", "gpt2")
    w.add_string("tokenizer.ggml.pre", "llama-bpe")
    w.add_array("tokenizer.ggml.tokens", _TOKENS)
    w.add_array("tokenizer.ggml.merges", _MERGES)
    w.add_array("tokenizer.ggml.token_type",
                [3, 3, 3] + [1] * (len(_TOKENS) - 3))
    w.add_uint32("tokenizer.ggml.bos_token_id", 0)
    w.add_uint32("tokenizer.ggml.eos_token_id", 1)
    rng = np.random.default_rng(0)

    def t(name, *shape):
        w.add_tensor(name, rng.standard_normal(shape).astype(np.float32) * 0.02)

    t("token_embd.weight", len(_TOKENS), _G3_HID)
    t("output_norm.weight", _G3_HID)
    for i in range(_G3_LAYERS):
        for norm in ("attn_norm", "post_attention_norm", "ffn_norm",
                     "post_ffw_norm"):
            t(f"blk.{i}.{norm}.weight", _G3_HID)
        t(f"blk.{i}.attn_q_norm.weight", _G3_HD)
        t(f"blk.{i}.attn_k_norm.weight", _G3_HD)
        t(f"blk.{i}.attn_q.weight", _G3_HEADS * _G3_HD, _G3_HID)
        t(f"blk.{i}.attn_k.weight", _G3_KV * _G3_HD, _G3_HID)
        t(f"blk.{i}.attn_v.weight", _G3_KV * _G3_HD, _G3_HID)
        t(f"blk.{i}.attn_output.weight", _G3_HID, _G3_HEADS * _G3_HD)
        t(f"blk.{i}.ffn_gate.weight", _G3_FFN, _G3_HID)
        t(f"blk.{i}.ffn_up.weight", _G3_FFN, _G3_HID)
        t(f"blk.{i}.ffn_down.weight", _G3_HID, _G3_FFN)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


def test_a_gemma3_container_takes_the_vocab_from_the_gguf(tmp_path):
    # The container's text config names no vocab_size, and the family
    # default counts 64 rows the GGUF does not have. The load takes the
    # token_embd rows, as a load without hf_source does, and the two
    # configs agree on the image token fields.
    import gmlx
    from gmlx.cache.media_salt import image_token_ids
    from gmlx.commands.tool_preflight import _synth_config
    from test_loader_api import _TOKENS

    gguf = str(tmp_path / "tiny-gemma3.gguf")
    _mint_tiny_gemma3(gguf)
    text = {k: v for k, v in _synth_config(gguf).items() if k != "vocab_size"}
    src = _write(tmp_path, {"model_type": "gemma3", "text_config": text,
                            **_GEMMA3_IMAGE_FIELDS})
    ids = mx.array([[0, 5, 6, 7]])
    ref, ref_cfg, _ = gmlx.load_model(gguf)
    model, cfg, _ = gmlx.load_model(gguf, hf_source=src)
    assert cfg["vocab_size"] == ref_cfg["vocab_size"] == len(_TOKENS)
    for key in (*_GEMMA3_IMAGE_FIELDS, "image_token_id"):
        assert cfg.get(key) == ref_cfg.get(key), key
    assert image_token_ids(cfg) == image_token_ids(ref_cfg)
    assert mx.array_equal(model(ids), ref(ids))


def test_a_mistral3_container_loads_a_llama_gguf(tiny_gguf, tmp_path):
    import gmlx
    from gmlx.commands.tool_preflight import _synth_config

    text = dict(_synth_config(tiny_gguf), model_type="mistral")
    src = _write(tmp_path, {"model_type": "mistral3", "text_config": text,
                            "vision_config": {}})
    ids = mx.array([[1, 2, 3, 4]])
    model, config, _ = gmlx.load_model(tiny_gguf, hf_source=src)
    assert config["model_type"] == "llama"
    ref, _, _ = gmlx.load_model(tiny_gguf)
    assert mx.array_equal(model(ids), ref(ids))


# Tier 2: Gemma-4 E2B from its GGUF, with and without its HF config.


def _e2b(gguf_index) -> str:
    from gmlx.load.headerscan import scan_gguf

    for path in gguf_index.get("gemma4", []):
        kv = scan_gguf(path, include_tensors=False).kv
        if (kv.get("gemma4.embedding_length") == 1536
                and kv.get("gemma4.block_count") == 35):
            return path
    pytest.skip("no Gemma-4 E2B GGUF under KQUANT_TEST_GGUF_DIR")


def test_gemma4_e2b_with_its_hf_config_matches_the_gguf_load(
        gguf_index, tmp_path, monkeypatch):
    import gmlx
    import gmlx.load.loader as loader

    gguf = _e2b(gguf_index)
    src = tmp_path / "src"
    src.mkdir()
    (src / "config.json").write_text(
        (_FIXTURES / "gemma4_e2b_it_config.json").read_text())
    ids = mx.array([[2, 818, 5279, 529, 7001, 563]])

    ref, ref_cfg, _ = gmlx.load_model(gguf)
    want = np.array(ref(ids).astype(mx.float32))
    del ref
    model, cfg, _ = gmlx.load_model(gguf, hf_source=str(src))
    got = np.array(model(ids).astype(mx.float32))
    del model
    assert cfg["model_type"] == ref_cfg["model_type"] == "gemma4_text"
    assert "use_bidirectional_attention" not in cfg
    assert (got.argmax(-1) == want.argmax(-1)).all()
    np.testing.assert_allclose(got, want, atol=1e-3, rtol=0)

    # The container route itself stops with the names it could not fill.
    monkeypatch.setattr(loader, "_text_config_from_container", lambda c: c)
    with pytest.raises(loader.UnfilledParametersError) as ei:
        gmlx.load_model(gguf, hf_source=str(src))
    assert "language_model.model.embed_tokens.weight" in str(ei.value)
