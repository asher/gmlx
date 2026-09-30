"""The sentence encoders in gmlx.models.embedders.

Tiny random-weight BERT, XLM-RoBERTa, ModernBERT and bidirectional Gemma3
models from transformers are saved as MLX checkpoints and loaded through the
gmlx checkpoint loader. On a ragged batch, the last hidden states match
transformers at every real token, and the sentence vectors match the
sentence-transformers pooling, dense layers and normalization of those
states. Everything runs on the CPU stream in float32.

The other tests cover the checkpoint rules (pooling configs, sanitize,
quantized layers, refused model types) and a guard: no gmlx entry point
imports mlx_embeddings or a stock mlx-vlm or mlx-lm, and pyproject.toml
does not list mlx-embeddings.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
import tomllib
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx.utils import tree_flatten

import gmlx.load.dtypes as act_dtypes
from gmlx.models.embedders import base, checkpoint
from gmlx.models.embedders import bert as bert_mod
from gmlx.models.embedders import gemma_embedding as ge
from gmlx.models.embedders import modernbert as mb_mod
from gmlx.models.embedders import qwen3 as q3_mod

_REPO = Path(__file__).resolve().parents[2]
_LENGTHS = [14, 9, 3]          # the ragged batch: longest first, then shorter rows
_VOCAB = 64


@pytest.fixture
def cpu():
    prev = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        yield
    finally:
        mx.set_default_device(prev)


def _ragged(pad_id: int, lo: int = 4, seed: int = 0):
    """Right-padded ``(ids, mask)`` for :data:`_LENGTHS`, real ids in
    ``[lo, _VOCAB)`` so no real token equals the pad id."""
    rng = np.random.default_rng(seed)
    width = max(_LENGTHS)
    ids = np.full((len(_LENGTHS), width), pad_id, dtype=np.int64)
    mask = np.zeros((len(_LENGTHS), width), dtype=np.int64)
    for i, n in enumerate(_LENGTHS):
        ids[i, :n] = rng.integers(lo, _VOCAB, n)
        mask[i, :n] = 1
    return ids, mask


def _write(path: Path, config: dict, weights: dict, pooling: str | None = None):
    """An MLX checkpoint folder: config.json, model.safetensors and, when
    ``pooling`` is given, a sentence-transformers Pooling config."""
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text(json.dumps(config))
    mx.save_safetensors(str(path / "model.safetensors"),
                        {k: mx.array(v) if isinstance(v, np.ndarray) else v
                         for k, v in weights.items()})
    if pooling is not None:
        (path / "modules.json").write_text(json.dumps([
            {"idx": 0, "name": "0", "path": "",
             "type": "sentence_transformers.models.Transformer"},
            {"idx": 1, "name": "1", "path": "1_Pooling",
             "type": "sentence_transformers.models.Pooling"}]))
        (path / "1_Pooling").mkdir()
        (path / "1_Pooling" / "config.json").write_text(json.dumps({
            "pooling_mode_cls_token": pooling == "cls",
            "pooling_mode_mean_tokens": pooling == "mean",
            "pooling_mode_lasttoken": pooling == "last"}))
    return path


# transformers parity

def _perturb(torch, model, seed: int = 0):
    """Random weights far from the init, so every block moves the output."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in model.named_parameters():
            noise = torch.randn(p.shape, generator=g)
            if p.ndim == 1 and ("norm" in name.lower() or "LayerNorm" in name):
                p.copy_(1.0 + 0.2 * noise if name.endswith("weight") else 0.1 * noise)
            else:
                p.copy_(0.2 * noise)


def _state(model, prefix: str = "") -> dict:
    return {prefix + k: v.detach().float().numpy()
            for k, v in model.state_dict().items()}


def _st_vectors(hidden, mask, pooling, dense=()):
    """The sentence-transformers Pooling, Dense and Normalize modules."""
    h = hidden.astype(np.float64)
    if pooling == "cls":
        x = h[:, 0]
    elif pooling == "last":
        last = [np.flatnonzero(row).max() for row in mask]
        x = h[np.arange(len(h)), last]
    else:
        m = mask[..., None].astype(np.float64)
        x = (h * m).sum(1) / np.maximum(m.sum(1), 1e-9)
    for w in dense:
        x = x @ w.T.astype(np.float64)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def _check(model, hf, ids, mask, pooling, dense=()):
    import torch

    with torch.no_grad():
        ref = hf(input_ids=torch.from_numpy(ids),
                 attention_mask=torch.from_numpy(mask)).last_hidden_state.numpy()
    out = model(mx.array(ids), attention_mask=mx.array(mask))
    got = np.asarray(out.last_hidden_state)
    real = mask.astype(bool)
    scale = np.abs(ref[real]).max()
    np.testing.assert_allclose(got[real], ref[real], rtol=0, atol=2e-5 * scale)
    want = _st_vectors(ref, mask, pooling, dense)
    np.testing.assert_allclose(np.asarray(out.text_embeds), want, atol=2e-5)
    # The shortest row alone, with no mask, gives the same vector as in the batch.
    real_ids = ids[-1][mask[-1].astype(bool)]
    alone = model(mx.array(real_ids[None])).text_embeds
    np.testing.assert_allclose(np.asarray(alone)[0], np.asarray(out.text_embeds)[-1],
                               atol=2e-5)


@pytest.mark.parametrize("kind", ["bert", "xlm-roberta"])
def test_bert_family_matches_transformers(tmp_path, cpu, kind):
    torch = pytest.importorskip("torch")
    tf = pytest.importorskip("transformers")
    common = dict(vocab_size=_VOCAB, hidden_size=32, num_hidden_layers=2,
                  num_attention_heads=4, intermediate_size=48, hidden_act="gelu")
    if kind == "bert":
        cfg = tf.BertConfig(max_position_embeddings=32, type_vocab_size=2,
                            pad_token_id=0, **common)
        hf = tf.BertModel(cfg, add_pooling_layer=True)
        pad = 0
    else:
        cfg = tf.XLMRobertaConfig(max_position_embeddings=32, type_vocab_size=1,
                                  pad_token_id=1, bos_token_id=0,
                                  eos_token_id=2, **common)
        hf = tf.XLMRobertaModel(cfg, add_pooling_layer=True)
        pad = 1
    hf.eval()
    _perturb(torch, hf)
    config = cfg.to_dict()
    assert config["model_type"] == kind
    path = _write(tmp_path / kind, config, _state(hf), pooling="cls")
    model = checkpoint.build(str(path))
    assert isinstance(model, bert_mod.Model) and model.args.pooling == "cls"
    ids, mask = _ragged(pad)
    _check(model, hf, ids, mask, "cls")


def test_modernbert_matches_transformers(tmp_path, cpu):
    torch = pytest.importorskip("torch")
    tf = pytest.importorskip("transformers")
    # local_attention 4 lets each local layer see 2 tokens either side, so the
    # 14-token row exercises the window; layers 0 and 3 are global.
    cfg = tf.ModernBertConfig(
        vocab_size=_VOCAB, hidden_size=32, intermediate_size=48,
        num_hidden_layers=4, num_attention_heads=4, local_attention=4,
        pad_token_id=0, bos_token_id=1, eos_token_id=2, cls_token_id=1,
        sep_token_id=2)
    hf = tf.ModernBertModel(cfg).eval()
    _perturb(torch, hf)
    config = cfg.to_dict()
    assert config["layer_types"] == ["full_attention", "sliding_attention",
                                     "sliding_attention", "full_attention"]
    path = _write(tmp_path / "mb", config, _state(hf))
    model = checkpoint.build(str(path))
    assert isinstance(model, mb_mod.Model) and model.args.pooling == "mean"
    assert "attn_norm" not in model.model.layers[0]
    assert "attn_norm" in model.model.layers[1]
    ids, mask = _ragged(0, lo=3)
    _check(model, hf, ids, mask, "mean")


def test_embedding_gemma_matches_transformers(tmp_path, cpu):
    torch = pytest.importorskip("torch")
    tf = pytest.importorskip("transformers")
    # sliding_window 8 lets a sliding layer see 4 tokens either side, so the
    # 14-token row exercises the window.
    cfg = tf.Gemma3TextConfig(
        vocab_size=_VOCAB, hidden_size=32, intermediate_size=48,
        num_hidden_layers=4, num_attention_heads=2, num_key_value_heads=1,
        head_dim=16, query_pre_attn_scalar=16, sliding_window=8,
        layer_types=["sliding_attention", "full_attention"] * 2,
        use_bidirectional_attention=True, max_position_embeddings=64,
        pad_token_id=0, bos_token_id=2, eos_token_id=1)
    hf = tf.Gemma3TextModel(cfg).eval()
    _perturb(torch, hf)
    config = dict(cfg.to_dict(), sliding_window=8)   # the checkpoint's value
    rng = np.random.default_rng(1)
    dense = [(0.2 * rng.standard_normal((128, 32))).astype(np.float32),
             (0.2 * rng.standard_normal((32, 128))).astype(np.float32)]
    weights = dict(_state(hf, "model."), **{f"dense.{i}.weight": w
                                            for i, w in enumerate(dense)})
    path = _write(tmp_path / "eg", config, weights)
    model = checkpoint.build(str(path))
    assert isinstance(model, ge.Model)
    assert model.args.sliding_window_pattern == 2
    assert [layer.self_attn.is_sliding for layer in model.layers] == [
        True, False, True, False]
    ids, mask = _ragged(0, lo=3)
    _check(model, hf, ids, mask, "mean", dense)


@pytest.mark.parametrize("side", ["right", "left"])
def test_qwen3_embedding_matches_transformers(tmp_path, cpu, side):
    torch = pytest.importorskip("torch")
    tf = pytest.importorskip("transformers")
    cfg = tf.Qwen3Config(
        vocab_size=_VOCAB, hidden_size=32, intermediate_size=48,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        head_dim=8, max_position_embeddings=64, tie_word_embeddings=True)
    hf = tf.Qwen3Model(cfg).eval()
    _perturb(torch, hf)
    config = cfg.to_dict()
    assert config["model_type"] == "qwen3"
    # A bare Qwen3Model tree, as the upstream checkpoint stores it.
    path = _write(tmp_path / "q3", config, _state(hf), pooling="last")
    model = checkpoint.build(str(path))
    assert isinstance(model, q3_mod.Model) and model.args.pooling == "last"
    ids, mask = _ragged(0, lo=3)
    if side == "left":
        ids = np.stack([np.roll(r, len(r) - int(m.sum()))
                        for r, m in zip(ids, mask)])
        mask = np.stack([np.roll(m, len(m) - int(m.sum())) for m in mask])
    _check(model, hf, ids, mask, "last")


# checkpoint rules

_TINY_GEMMA = dict(model_type="gemma3_text", vocab_size=_VOCAB, hidden_size=64,
                   intermediate_size=96, num_hidden_layers=2,
                   num_attention_heads=2, num_key_value_heads=1, head_dim=32,
                   query_pre_attn_scalar=32, sliding_window=8,
                   sliding_window_pattern=2, use_bidirectional_attention=True)
_TINY_QWEN3 = dict(model_type="qwen3", vocab_size=_VOCAB, hidden_size=64,
                   intermediate_size=96, num_hidden_layers=2,
                   num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                   rms_norm_eps=1e-6, max_position_embeddings=64,
                   rope_theta=1e4)


def _tiny_bert_config(model_type="bert", **kw):
    return {"model_type": model_type, "vocab_size": _VOCAB, "hidden_size": 64,
            "num_hidden_layers": 2, "num_attention_heads": 4,
            "intermediate_size": 128, "max_position_embeddings": 32,
            "type_vocab_size": 1, "layer_norm_eps": 1e-12, "pad_token_id": 0,
            **kw}


def _params(model) -> dict:
    return dict(tree_flatten(model.parameters()))


def test_pooling_mode_reads_the_sentence_transformers_config(tmp_path):
    p = _write(tmp_path / "cls", _tiny_bert_config(), {}, pooling="cls")
    assert checkpoint.pooling_mode(str(p), "bert") == "cls"
    p = _write(tmp_path / "mean", _tiny_bert_config("xlm-roberta"), {},
               pooling="mean")
    assert checkpoint.pooling_mode(str(p), "xlm-roberta") == "mean"


def test_pooling_mode_defaults_by_model_type(tmp_path, capsys):
    # The mlx-community conversions list 1_Pooling in modules.json without
    # shipping it, so the default applies: the mean, as sentence-transformers
    # takes for a model without a Pooling config, but the last token for
    # Qwen3-Embedding.
    p = tmp_path / "nopool"
    p.mkdir()
    (p / "modules.json").write_text(json.dumps([
        {"idx": 1, "path": "1_Pooling",
         "type": "sentence_transformers.models.Pooling"}]))
    for model_type, mode in (("xlm-roberta", "mean"), ("xlm_roberta", "mean"),
                             ("bert", "mean"), ("modernbert", "mean"),
                             ("gemma3_text", "mean"), ("qwen3", "last")):
        assert checkpoint.pooling_mode(str(p), model_type) == mode
        # One line says what was inferred.
        err = capsys.readouterr().err.strip().splitlines()
        assert len(err) == 1 and "no sentence-transformers Pooling config" in err[0]
        assert f"the default for {checkpoint.FAMILIES[model_type]}" in err[0]
    # A Pooling config is read, not inferred, and logs nothing.
    q = _write(tmp_path / "cfg", _tiny_bert_config("xlm-roberta"), {},
               pooling="mean")
    assert checkpoint.pooling_mode(str(q), "xlm-roberta") == "mean"
    assert capsys.readouterr().err == ""


@pytest.mark.parametrize("config,card,family", [
    ({"_name_or_path": "BAAI/bge-small-en-v1.5"}, "", "BGE"),
    ({"_name_or_path": "/tmp/models/bge-large-en"}, "", "BGE"),
    ({}, "---\nlicense: mit\nbase_model: BAAI/bge-m3\n---\n# card\n", "BGE-M3"),
    ({}, "---\nbase_model:\n- 'Snowflake/snowflake-arctic-embed-m-v1.5'\n---\n",
     "Arctic-Embed"),
    ({}, "---\ntags:\n- mlx\nmodel-index:\n- name: snowflake-arctic-embed-l-v2.0\n"
         "  results: []\n---\n", "Arctic-Embed"),
    ({"_name_or_path": "BAAI/bge-reranker-base"},
     "---\nbase_model: intfloat/e5-base\n---\n", None),
    ({}, "# no front matter\nbase_model: BAAI/bge-m3\n", None),
])
def test_pooling_mode_takes_the_first_token_for_known_families(
        tmp_path, capsys, config, card, family):
    # Without a Pooling config, a checkpoint that names a first-token family
    # in config.json or its model card pools the first token.
    p = tmp_path / "ckpt"
    p.mkdir()
    (p / "config.json").write_text(json.dumps(dict(config, model_type="bert")))
    (p / "README.md").write_text(card)
    for model_type in ("bert", "xlm-roberta"):
        mode = checkpoint.pooling_mode(str(p), model_type)
        err = capsys.readouterr().err
        if family is None:
            assert mode == "mean" and "the default for" in err
        else:
            assert mode == "cls" and f"as the upstream {family} config does" in err
    # A Pooling config beats the family.
    (p / "modules.json").write_text(json.dumps([
        {"idx": 1, "path": "1_Pooling",
         "type": "sentence_transformers.models.Pooling"}]))
    (p / "1_Pooling").mkdir()
    (p / "1_Pooling" / "config.json").write_text(json.dumps(
        {"pooling_mode_mean_tokens": True}))
    assert checkpoint.pooling_mode(str(p), "bert") == "mean"


_FIRST_TOKEN_REPOS = [
    *(f"BAAI/bge-{size}-{lang}{version}" for size in ("small", "base", "large")
      for lang in ("en", "zh") for version in ("", "-v1.5")),
    "BAAI/bge-large-zh-noinstruct",
    "mixedbread-ai/mxbai-embed-large-v1",
]


@pytest.mark.parametrize("repo", _FIRST_TOKEN_REPOS)
def test_every_bge_v1_and_v15_model_and_mxbai_large_pool_the_first_token(
        tmp_path, capsys, repo):
    # The upstream Pooling configs of the BGE v1 and v1.5 models, English and
    # Chinese, and of mxbai-embed-large-v1 take the first token. The repo
    # id is the folder name, or the config's _name_or_path.
    p = tmp_path / repo.replace("/", "__")
    p.mkdir()
    (p / "config.json").write_text(json.dumps({"model_type": "bert"}))
    q = tmp_path / "ckpt"
    q.mkdir()
    (q / "config.json").write_text(json.dumps({"model_type": "bert",
                                               "_name_or_path": repo}))
    assert checkpoint.pooling_mode(str(p), "bert") == "cls"
    assert checkpoint.pooling_mode(str(q), "bert") == "cls"
    capsys.readouterr()
    # Names that only share a prefix are no family.
    for other in ("BAAI/bge-en-icl", "BAAI/bge-reranker-large",
                  "mixedbread-ai/mxbai-embed-large-v10",
                  "BAAI/bge-small-english"):
        assert checkpoint._first_token_family([other]) is None, other


@pytest.mark.parametrize("where,family", [
    ("mlx-community__bge-m3-mlx-8bit", "BGE-M3"),
    ("hub/models--mlx-community--bge-m3-mlx-8bit/snapshots/0a1b2c3d", "BGE-M3"),
    ("mlx-community__snowflake-arctic-embed-l-v2.0-4bit", "Arctic-Embed"),
    ("bge-m3", "BGE-M3"),
    ("bge-small-en-v1.5_mlx", "BGE"),
    ("hub/models--mlx-community--bge-m3x/snapshots/0a1b2c3d", None),
    ("mlx-community__my-encoder", None),
])
def test_pooling_mode_takes_the_family_from_the_repo_or_folder_name(
        tmp_path, capsys, where, family):
    # A snapshot without its model card, as a download that keeps only the
    # weights, configs and tokenizer files leaves it, and a config.json whose
    # _name_or_path is null, as in the mlx-community BGE-M3 and Arctic-Embed
    # conversions. The repo or folder name still gives the family.
    p = tmp_path / where
    p.mkdir(parents=True)
    (p / "config.json").write_text(json.dumps(
        {"model_type": "xlm-roberta", "_name_or_path": None}))
    _modules_json(p, "Pooling", "Normalize")      # listed, not shipped
    mode = checkpoint.pooling_mode(str(p), "xlm-roberta")
    err = capsys.readouterr().err
    if family is None:
        assert mode == "mean" and "the default for XLM-RoBERTa" in err
    else:
        assert mode == "cls" and f"as the upstream {family} config does" in err


def test_pooling_mode_refuses_other_modes(tmp_path):
    p = _write(tmp_path / "max", _tiny_bert_config(), {}, pooling="max")
    cfg = json.loads((p / "1_Pooling" / "config.json").read_text())
    cfg["pooling_mode_max_tokens"] = True
    (p / "1_Pooling" / "config.json").write_text(json.dumps(cfg))
    with pytest.raises(checkpoint.UnsupportedEncoderError, match="max_tokens"):
        checkpoint.pooling_mode(str(p), "bert")


@pytest.mark.parametrize("model_type", ["lfm2", "siglip", "colqwen2_5",
                                        "qwen3_vl", "colidefics3"])
def test_unsupported_model_type_names_the_supported_ones(tmp_path, model_type):
    p = _write(tmp_path / "q", {"model_type": model_type}, {"x": np.zeros(1)})
    with pytest.raises(checkpoint.UnsupportedEncoderError) as ei:
        checkpoint.build(str(p))
    msg = ei.value.reason
    assert repr(model_type) in msg and str(p) not in msg
    assert msg.startswith(f"model_type {model_type!r} is not a supported "
                          "embeddings model. Supported: EmbeddingGemma, BERT, "
                          "XLM-RoBERTa, ModernBERT and Qwen3-Embedding.")
    # A Qwen model type points at the Qwen3-Embedding paths.
    assert ("qwen3-embed-0.6b" in msg) == model_type.startswith("qwen")
    assert set(checkpoint.FAMILIES) == set(checkpoint.MODEL_TYPES)


def _modules_json(path: Path, *kinds: str):
    (path / "modules.json").write_text(json.dumps([
        {"idx": i, "name": str(i), "path": "" if k == "Transformer" else f"{i}_{k}",
         "type": f"sentence_transformers.models.{k}"}
        for i, k in enumerate(("Transformer", *kinds))]))


def test_gemma3_causal_lm_is_not_an_encoder(tmp_path, cpu):
    # A Gemma 3 causal LM shares model_type gemma3_text with EmbeddingGemma.
    # Without use_bidirectional_attention or the sentence-transformers Dense
    # modules, the loader refuses it and says why.
    lm = {k: v for k, v in _TINY_GEMMA.items() if k != "use_bidirectional_attention"}
    weights = _params(ge.Model(ge.ModelArgs.from_dict(lm)))
    for name, config in (("absent", lm),
                         ("false", dict(lm, use_bidirectional_attention=False))):
        p = _write(tmp_path / name, config, weights)
        with pytest.raises(checkpoint.UnsupportedEncoderError) as ei:
            checkpoint.build(str(p))
        assert "Gemma 3 causal LM, not EmbeddingGemma" in ei.value.reason
        assert str(p) not in ei.value.reason
    # Either mark makes it an encoder.
    p = _write(tmp_path / "flag", _TINY_GEMMA, weights)
    assert isinstance(checkpoint.build(str(p)), ge.Model)
    p = _write(tmp_path / "dense", lm, weights)
    _modules_json(p, "Pooling", "Dense", "Dense", "Normalize")
    assert isinstance(checkpoint.build(str(p)), ge.Model)


def test_qwen3_causal_lm_is_not_an_encoder(tmp_path, cpu):
    # A Qwen3 causal LM shares model_type qwen3 with Qwen3-Embedding, which
    # ships the sentence-transformers Pooling module.
    weights = _params(q3_mod.Model(q3_mod.ModelArgs.from_dict(_TINY_QWEN3)))
    p = _write(tmp_path / "lm", _TINY_QWEN3, weights)
    with pytest.raises(checkpoint.UnsupportedEncoderError) as ei:
        checkpoint.build(str(p))
    assert "Qwen3 causal LM, not Qwen3-Embedding" in ei.value.reason
    assert "qwen3-embed-0.6b" in ei.value.reason
    _modules_json(p, "Pooling", "Normalize")
    assert isinstance(checkpoint.build(str(p)), q3_mod.Model)


def test_bert_sanitize_drops_prefixes_and_heads():
    w = {"roberta.embeddings.word_embeddings.weight": 1,
         "roberta.embeddings.position_ids": 2,
         "pooler.dense.weight": 3, "pooler.dense.bias": 4,
         "lm_head.dense.weight": 5, "cls.predictions.bias": 6,
         "bert.encoder.layer.0.output.dense.weight": 7}
    assert bert_mod.Model.sanitize(w) == {
        "embeddings.word_embeddings.weight": 1,
        "encoder.layer.0.output.dense.weight": 7}


def test_modernbert_sanitize_homes_the_tree_and_drops_the_head():
    w = {"embeddings.tok_embeddings.weight": 1, "model.final_norm.weight": 2,
         "head.dense.weight": 3, "decoder.bias": 4}
    assert mb_mod.Model.sanitize(w) == {
        "model.embeddings.tok_embeddings.weight": 1,
        "model.final_norm.weight": 2}


@pytest.mark.parametrize("stored", ["float", "uint32", "uint8"])
def test_embedding_gemma_scale_is_the_rounded_root_of_the_width(cpu, stored):
    # sqrt(768) is 27.71, which bfloat16 rounds to 27.75. The scale follows
    # the activation dtype for float weights, a uint32 quantized table and a
    # uint8 GGUF table alike; the weight dtype would truncate it to 27.
    import mlx_kquant as kq

    from gmlx.load.modules import install_kquant_modules

    width = 768
    enc = ge.Encoder(ge.ModelArgs.from_dict(dict(
        _TINY_GEMMA, hidden_size=width, num_hidden_layers=0)))
    table = mx.random.normal((_VOCAB, width), key=mx.random.key(0))
    enc.embed_tokens.weight = table.astype(mx.bfloat16)
    if stored == "uint32":
        nn.quantize(enc, group_size=64, bits=8)
    elif stored == "uint8":
        wq, scales = kq.quantize(table, "q8_0")
        install_kquant_modules(enc, {"embed_tokens.weight": "q8_0"},
                               act_dtype=mx.bfloat16)
        enc.embed_tokens.weight = wq.reshape(_VOCAB, -1)
        enc.embed_tokens.scales = scales
    assert enc.embed_tokens.weight.dtype == {
        "float": mx.bfloat16, "uint32": mx.uint32, "uint8": mx.uint8}[stored]
    ids = mx.array([[1, 5, 9]])
    raw = enc.embed_tokens(ids)
    assert raw.dtype == mx.bfloat16
    scale = mx.array(width**0.5, mx.bfloat16)
    assert scale.item() == 27.75
    assert mx.array_equal(enc.embed(ids), raw * scale)
    assert not mx.array_equal(enc.embed(ids), raw * 27)


def test_gemma_layer_types_must_be_periodic():
    base_args = dict(model_type="gemma3_text", hidden_size=32,
                     num_hidden_layers=4, intermediate_size=48)
    a = ge.ModelArgs.from_dict(dict(base_args, layer_types=[
        "sliding_attention"] * 3 + ["full_attention"]))
    assert a.sliding_window_pattern == 4
    with pytest.raises(ValueError, match="layer_types"):
        ge.ModelArgs.from_dict(dict(base_args, layer_types=[
            "full_attention", "sliding_attention", "sliding_attention",
            "full_attention"]))
    a = ge.ModelArgs.from_dict(dict(base_args, rope_parameters={
        "full_attention": {"rope_theta": 5.0},
        "sliding_attention": {"rope_theta": 7.0}}))
    assert (a.rope_theta, a.rope_local_base_freq) == (5.0, 7.0)


def test_window_and_padding_masks(cpu):
    assert base.window_mask(3, 2) is None                 # the window covers it
    w = np.asarray(base.window_mask(5, 1))
    assert w.tolist() == [[abs(q - k) <= 1 for k in range(5)] for q in range(5)]
    assert base.padding_mask(None) is None
    pad = np.asarray(base.padding_mask(mx.array([[1, 1, 0]])))
    assert pad.shape == (1, 1, 1, 3) and pad.ravel().tolist() == [True, True, False]
    both = np.asarray(base.combine_masks(mx.array(pad), None, mx.array(w[:3, :3])))
    assert both.shape == (1, 1, 3, 3)
    assert both[0, 0].tolist() == [[True, True, False], [True, True, False],
                                   [False, True, False]]


@pytest.fixture(params=["cpu", "default"])
def device(request):
    prev = mx.default_device()
    if request.param == "cpu":
        mx.set_default_device(mx.cpu)
    try:
        yield
    finally:
        mx.set_default_device(prev)


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16, mx.float32])
def test_additive_mask_fills_the_dtype_minimum(dtype):
    assert base.additive_mask(None, dtype) is None
    add = base.additive_mask(mx.array([[True, False], [False, True]]), dtype)
    assert add.dtype == dtype
    lo = float(np.asarray(mx.array(mx.finfo(dtype).min, dtype).astype(mx.float32)))
    assert np.asarray(add.astype(mx.float32)).tolist() == [[0, lo], [lo, 0]]


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16, mx.float32])
@pytest.mark.parametrize("qscale", [1.0, -30.0])
def test_additive_mask_keeps_a_row_with_no_allowed_key_finite(device, dtype,
                                                              qscale):
    # Row 1 allows no key, as a padding query outside every real token's
    # window does. With negative scores, a row of float16 minimums would
    # overflow to -inf on the CPU and give NaN.
    allowed = mx.array([[True, False], [False, False]])
    add = base.additive_mask(allowed, dtype)
    q = mx.full((1, 1, 2, 4), qscale, dtype=dtype)
    k = mx.ones((1, 1, 2, 4), dtype=dtype)
    v = mx.array(np.arange(8, dtype=np.float32).reshape(1, 1, 2, 4)).astype(dtype)
    out = mx.fast.scaled_dot_product_attention(q, k, v, scale=1.0, mask=add)
    out = np.asarray(out.astype(mx.float32))
    assert np.isfinite(out).all()
    np.testing.assert_allclose(out[0, 0, 0], [0, 1, 2, 3], atol=1e-3)


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
def test_additive_mask_outweighs_a_far_negative_score(device, dtype):
    # The one allowed key scores -40000, and the masked key 0. The fill must
    # sink the masked key below it, which -1e4 does not.
    add = base.additive_mask(mx.array([[True, False]]), dtype)
    q = mx.full((1, 1, 1, 4), -100.0, dtype=dtype)
    k = mx.array([[[[100.0] * 4, [0.0] * 4]]], dtype=dtype)
    v = mx.array([[[[1.0] * 4, [-1.0] * 4]]], dtype=dtype)
    out = mx.fast.scaled_dot_product_attention(q, k, v, scale=1.0, mask=add)
    np.testing.assert_allclose(np.asarray(out.astype(mx.float32)).ravel(),
                               [1.0] * 4)


def test_pool_mean_ignores_padding(cpu):
    h = mx.array([[[1.0, 2.0], [3.0, 4.0], [100.0, 100.0]]])
    m = mx.array([[1, 1, 0]])
    assert np.asarray(base.pool(h, m, "mean")).tolist() == [[2.0, 3.0]]
    assert np.asarray(base.pool(h, m, "cls")).tolist() == [[1.0, 2.0]]
    np.testing.assert_allclose(np.asarray(base.pool(h, None, "mean")),
                               [[104.0 / 3, 106.0 / 3]], rtol=1e-6)


def test_pool_last_takes_the_last_real_token_on_either_side(cpu):
    h = mx.array([[[1.0], [2.0], [3.0]], [[4.0], [5.0], [6.0]]])
    right = mx.array([[1, 1, 0], [1, 1, 1]])
    left = mx.array([[0, 1, 1], [0, 0, 1]])
    assert np.asarray(base.pool(h, right, "last")).tolist() == [[2.0], [6.0]]
    assert np.asarray(base.pool(h, left, "last")).tolist() == [[3.0], [6.0]]
    assert np.asarray(base.pool(h, None, "last")).tolist() == [[3.0], [6.0]]


def _quant_predicate(overrides):
    def pred(path, module):
        if not isinstance(module, (nn.Linear, nn.Embedding)):
            return False
        return overrides.get(path, True)
    return pred


def test_quantized_checkpoint_loads_the_same_layers(tmp_path, cpu):
    # A per-layer entry overrides the global bits; the position embeddings
    # stay unquantized because the weights carry no scales for them.
    config = _tiny_bert_config("xlm-roberta")
    quant = {"group_size": 32, "bits": 8,
             "encoder.layer.1.intermediate.dense": {"group_size": 32, "bits": 4}}
    ref = bert_mod.Model(bert_mod.ModelArgs.from_dict(dict(config, pooling="mean")))
    overrides = {"embeddings.position_embeddings": False,
                 "encoder.layer.1.intermediate.dense": quant[
                     "encoder.layer.1.intermediate.dense"]}
    nn.quantize(ref, group_size=32, bits=8,
                class_predicate=_quant_predicate(overrides))
    mx.eval(ref.parameters())
    path = _write(tmp_path / "q", dict(config, quantization=quant), _params(ref))
    model = checkpoint.build(str(path))
    layer1 = model.encoder.layer[1]
    assert isinstance(layer1.intermediate.dense, nn.QuantizedLinear)
    assert layer1.intermediate.dense.bits == 4
    assert model.encoder.layer[0].intermediate.dense.bits == 8
    assert isinstance(model.embeddings.word_embeddings, nn.QuantizedEmbedding)
    assert type(model.embeddings.position_embeddings) is nn.Embedding
    ids, mask = _ragged(0)
    assert model.args.pooling == "mean"
    a = ref(mx.array(ids), attention_mask=mx.array(mask)).text_embeds
    b = model(mx.array(ids), attention_mask=mx.array(mask)).text_embeds
    assert np.array_equal(np.asarray(a), np.asarray(b))


@pytest.mark.parametrize("stored", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize("policy,gemma_dtype", [
    ((17, None), mx.bfloat16),        # auto on an M5 picks bfloat16
    ((13, None), mx.float32),         # auto on an M1 picks float16
    ((17, "float16"), mx.float32),    # a float16 request
    ((13, "bfloat16"), mx.bfloat16),  # a bfloat16 request on an M1
    ((17, "float32"), mx.float32),
])
def test_embedding_gemma_never_runs_float16(tmp_path, cpu, monkeypatch,
                                             stored, policy, gemma_dtype):
    # EmbeddingGemma casts its 16-bit parameters, quantization scales among
    # them, to bfloat16, or to float32 where the chip policy picks float16.
    # Other model types keep the checkpoint's dtype.
    gen, value = policy
    monkeypatch.setattr(act_dtypes, "gpu_arch_gen", lambda: gen)
    # The rule on a GPU. The CPU device has its own rule, tested below.
    monkeypatch.setattr(checkpoint, "cpu_default_device", lambda: False)
    if value is None:
        monkeypatch.delenv(act_dtypes.ENV_VAR, raising=False)
    else:
        monkeypatch.setenv(act_dtypes.ENV_VAR, value)
    for name, mod, config in (("gemma", ge, _TINY_GEMMA),
                              ("bert", bert_mod, _tiny_bert_config())):
        model = mod.Model(mod.ModelArgs.from_dict(config))
        nn.quantize(model, group_size=32, bits=8)
        model.set_dtype(stored)
        mx.eval(model.parameters())
        path = _write(tmp_path / name, dict(
            config, quantization={"group_size": 32, "bits": 8}), _params(model))
        built = checkpoint.build(str(path))
        dtypes = {v.dtype for _, v in tree_flatten(built.parameters())
                  if mx.issubdtype(v.dtype, mx.floating)}
        want = gemma_dtype if name == "gemma" else stored
        assert dtypes == {want}, (name, dtypes)
        ids, mask = _ragged(0, lo=4)
        out = built(mx.array(ids), attention_mask=mx.array(mask))
        assert out.last_hidden_state.dtype == want


def _dequantized(weights: dict, group_size: int, bits: int) -> dict:
    """The float32 weights a quantized checkpoint stores."""
    out = {}
    for k, v in weights.items():
        if k.endswith((".scales", ".biases")):
            continue
        prefix = k.removesuffix(".weight")
        if f"{prefix}.scales" in weights:
            v = mx.dequantize(v, weights[f"{prefix}.scales"].astype(mx.float32),
                              weights[f"{prefix}.biases"].astype(mx.float32),
                              group_size=group_size, bits=bits)
        out[k] = v.astype(mx.float32)
    return out


@pytest.mark.parametrize("model_type,stored", [
    ("gemma3_text", mx.float16), ("gemma3_text", mx.bfloat16),
    ("bert", mx.bfloat16)])
def test_quantized_encoder_runs_float32_on_the_cpu(tmp_path, cpu, monkeypatch,
                                                   model_type, stored):
    # The stock CPU quantized matmul accumulates in the activation dtype,
    # which in bfloat16 is far off a float32 run. On the CPU, a quantized
    # encoder that would run bfloat16 runs float32, cast straight from the
    # stored dtype, and its vectors match the float32 model of the weights it
    # stores.
    monkeypatch.setattr(act_dtypes, "gpu_arch_gen", lambda: 17)
    monkeypatch.delenv(act_dtypes.ENV_VAR, raising=False)
    mod, config = ((ge, _TINY_GEMMA) if model_type == "gemma3_text"
                   else (bert_mod, _tiny_bert_config()))
    model = mod.Model(mod.ModelArgs.from_dict(config))
    nn.quantize(model, group_size=32, bits=8)
    model.set_dtype(stored)
    mx.eval(model.parameters())
    weights = _params(model)
    path = _write(tmp_path / "q", dict(
        config, quantization={"group_size": 32, "bits": 8}), weights)
    built = checkpoint.build(str(path))
    assert {v.dtype for _, v in tree_flatten(built.parameters())
            if mx.issubdtype(v.dtype, mx.floating)} == {mx.float32}
    ref = mod.Model(mod.ModelArgs.from_dict(dict(config, pooling="mean")))
    ref.load_weights(list(_dequantized(weights, 32, 8).items()), strict=True)
    ids, mask = _ragged(0, lo=4)
    a = built(mx.array(ids), attention_mask=mx.array(mask)).text_embeds
    b = ref(mx.array(ids), attention_mask=mx.array(mask)).text_embeds
    np.testing.assert_allclose(np.asarray(a), np.asarray(b), atol=1e-5)
    # Off the CPU they stay bfloat16.
    monkeypatch.setattr(checkpoint, "cpu_default_device", lambda: False)
    built = checkpoint.build(str(path))
    assert {v.dtype for _, v in tree_flatten(built.parameters())
            if mx.issubdtype(v.dtype, mx.floating)} == {mx.bfloat16}


def test_checkpoint_must_hold_every_weight(tmp_path, cpu):
    config = _tiny_bert_config()
    w = _params(bert_mod.Model(bert_mod.ModelArgs.from_dict(config)))
    del w["encoder.layer.1.output.LayerNorm.bias"]
    path = _write(tmp_path / "short", config, w)
    with pytest.raises(ValueError, match="LayerNorm.bias"):
        checkpoint.build(str(path))


# the guard: nothing reaches mlx_embeddings

def _tokenizer(path: Path, pad_id: int = 0):
    from tokenizers import Tokenizer, models, pre_tokenizers, processors
    from transformers import PreTrainedTokenizerFast

    words = "alpha beta gamma delta epsilon zeta eta theta".split()
    specials = ["[PAD]", "[UNK]", "[CLS]", "[SEP]"]
    if pad_id:
        specials[0], specials[pad_id] = specials[pad_id], specials[0]
    vocab = {t: i for i, t in enumerate(specials + words)}
    tk = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tk.pre_tokenizer = pre_tokenizers.Whitespace()
    tk.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]",
        special_tokens=[("[CLS]", vocab["[CLS]"]), ("[SEP]", vocab["[SEP]"])])
    PreTrainedTokenizerFast(
        tokenizer_object=tk, unk_token="[UNK]", pad_token="[PAD]",
        cls_token="[CLS]", sep_token="[SEP]",
        model_max_length=32).save_pretrained(str(path))


def _tiny_checkpoints(root: Path) -> dict:
    """One small random checkpoint per safetensors encoder kind, with a
    tokenizer, and a quantized one."""
    kinds = {
        "bert": (bert_mod, _tiny_bert_config(), 0),
        "xlm-roberta": (bert_mod, _tiny_bert_config(
            "xlm-roberta", max_position_embeddings=40, pad_token_id=1), 1),
        "modernbert": (mb_mod, dict(
            model_type="modernbert", vocab_size=_VOCAB, hidden_size=64,
            intermediate_size=96, num_hidden_layers=3, num_attention_heads=4,
            local_attention=4), 0),
        "gemma3_text": (ge, _TINY_GEMMA, 0),
        "qwen3": (q3_mod, _TINY_QWEN3, 0),
    }
    out = {}
    for name, (mod, config, pad) in kinds.items():
        model = mod.Model(mod.ModelArgs.from_dict(config))
        if name == "bert":
            nn.quantize(model, group_size=32, bits=8)
            config = dict(config, quantization={"group_size": 32, "bits": 8})
        mx.eval(model.parameters())
        path = _write(root / name, config, _params(model),
                      pooling="last" if name == "qwen3" else None)
        _tokenizer(path, pad)
        out[name] = str(path)
    return out


_GUARD = textwrap.dedent("""
    import importlib, json, sys

    attempts = []
    stock = ("mlx_embeddings", "mlx_vlm", "mlx_lm")

    class Block:
        def find_spec(self, name, path=None, target=None):
            if name.split(".")[0] in stock:
                attempts.append(name)
                raise ImportError("blocked import of " + name)
            return None

    sys.meta_path.insert(0, Block())

    import mlx.core as mx
    mx.set_default_device(mx.cpu)

    import gmlx
    import gmlx.commands.cli
    import gmlx.serve.server
    import gmlx.serve.embeddings as emb
    import gmlx.serve.patches as sp
    from gmlx.load import loader
    from gmlx.load.config_synth import synthesize_config
    from fastapi.testclient import TestClient

    dirs = json.loads(sys.argv[1])
    app = importlib.import_module("gmlx._vendor.mlx_vlm.server.app").app
    widths = {}
    for kind, path in dirs.items():
        sp.install_embeddings_route(path)
        r = TestClient(app).post("/v1/embeddings", json={
            "input": ["alpha beta gamma", "delta"]})
        assert r.status_code == 200, (kind, r.status_code, r.text)
        widths[kind] = len(r.json()["data"][0]["embedding"])

    # The GGUF encoder kind: the loader's gemma_embedding build.
    arch = "gemma-embedding"
    meta = {"general.architecture": arch, f"{arch}.context_length": 64,
            f"{arch}.embedding_length": 64, f"{arch}.feed_forward_length": 96,
            f"{arch}.block_count": 2, f"{arch}.attention.head_count": 2,
            f"{arch}.attention.head_count_kv": 1,
            f"{arch}.attention.key_length": 32,
            f"{arch}.attention.value_length": 32,
            f"{arch}.attention.layer_norm_rms_epsilon": 1e-6,
            f"{arch}.attention.sliding_window": 8,
            f"{arch}.rope.freq_base": 1e6, "tokenizer.ggml.tokens": ["a"] * 64}
    model, _ = loader.build_model(synthesize_config(meta, tensor_shapes={}))
    out = model(mx.array([[1, 2, 3]]))
    widths["gemma-embedding"] = int(out.text_embeds.shape[-1])

    print(json.dumps({"attempts": attempts, "widths": widths,
                      "loaded": sorted(m for m in sys.modules
                                       if m.split(".")[0] in stock)}))
""")


def test_no_gmlx_entry_point_imports_a_stock_package(tmp_path):
    dirs = _tiny_checkpoints(tmp_path)
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(
        [str(_REPO), os.environ.get("PYTHONPATH", "")]),
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    r = subprocess.run([sys.executable, "-c", _GUARD, json.dumps(dirs)],
                       capture_output=True, text=True, env=env, timeout=600,
                       cwd=str(tmp_path))
    assert r.returncode == 0, r.stderr[-4000:]
    got = json.loads(r.stdout.strip().splitlines()[-1])
    assert got["attempts"] == [] and got["loaded"] == []
    assert got["widths"] == {"bert": 64, "xlm-roberta": 64, "modernbert": 64,
                             "gemma3_text": 64, "qwen3": 64,
                             "gemma-embedding": 64}


def test_pyproject_does_not_list_mlx_embeddings():
    project = tomllib.loads((_REPO / "pyproject.toml").read_text())["project"]
    deps = list(project.get("dependencies", []))
    for extra in project.get("optional-dependencies", {}).values():
        deps += extra
    assert not [d for d in deps if "mlx-embeddings" in d or "mlx_embeddings" in d]
    # The back-compat extra stays, empty.
    assert project["optional-dependencies"]["embeddings"] == []
