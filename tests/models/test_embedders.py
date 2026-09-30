"""The sentence encoders in gmlx.models.embedders.

Tiny random-weight BERT, XLM-RoBERTa, ModernBERT and bidirectional Gemma3
models from transformers are saved as MLX checkpoints and loaded through the
gmlx checkpoint loader. On a ragged batch, the last hidden states match
transformers at every real token, and the sentence vectors match the
sentence-transformers pooling, dense layers and normalization of those
states. Everything runs on the CPU stream in float32.

The other tests cover the checkpoint rules: pooling configs, sanitize,
quantized layers and refused model types.
"""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx.utils import tree_flatten

from gmlx.models.embedders import base, checkpoint
from gmlx.models.embedders import bert as bert_mod
from gmlx.models.embedders import gemma_embedding as ge
from gmlx.models.embedders import modernbert as mb_mod

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
            "pooling_mode_mean_tokens": pooling == "mean"}))
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
    n = _LENGTHS[-1]
    alone = model(mx.array(ids[-1:, :n])).text_embeds
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


# checkpoint rules

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


def test_pooling_mode_defaults_by_model_type(tmp_path):
    # The mlx-community conversions list 1_Pooling in modules.json without
    # shipping it, so the default applies.
    p = tmp_path / "nopool"
    p.mkdir()
    (p / "modules.json").write_text(json.dumps([
        {"idx": 1, "path": "1_Pooling",
         "type": "sentence_transformers.models.Pooling"}]))
    assert checkpoint.pooling_mode(str(p), "xlm-roberta") == "cls"
    assert checkpoint.pooling_mode(str(p), "bert") == "mean"
    assert checkpoint.pooling_mode(str(p), "modernbert") == "mean"
    assert checkpoint.pooling_mode(str(p), "gemma3_text") == "mean"


def test_pooling_mode_refuses_other_modes(tmp_path):
    p = _write(tmp_path / "max", _tiny_bert_config(), {}, pooling="max")
    cfg = json.loads((p / "1_Pooling" / "config.json").read_text())
    cfg["pooling_mode_max_tokens"] = True
    (p / "1_Pooling" / "config.json").write_text(json.dumps(cfg))
    with pytest.raises(checkpoint.UnsupportedEncoderError, match="max_tokens"):
        checkpoint.pooling_mode(str(p), "bert")


def test_unsupported_model_type_is_refused(tmp_path):
    p = _write(tmp_path / "q", {"model_type": "qwen3"}, {"x": np.zeros(1)})
    with pytest.raises(checkpoint.UnsupportedEncoderError) as ei:
        checkpoint.build(str(p))
    msg = str(ei.value)
    assert "qwen3" in msg and "modernbert" in msg and "xlm-roberta" in msg


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


def test_pool_mean_ignores_padding(cpu):
    h = mx.array([[[1.0, 2.0], [3.0, 4.0], [100.0, 100.0]]])
    m = mx.array([[1, 1, 0]])
    assert np.asarray(base.pool(h, m, "mean")).tolist() == [[2.0, 3.0]]
    assert np.asarray(base.pool(h, m, "cls")).tolist() == [[1.0, 2.0]]
    np.testing.assert_allclose(np.asarray(base.pool(h, None, "mean")),
                               [[104.0 / 3, 106.0 / 3]], rtol=1e-6)


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
    ref = bert_mod.Model(bert_mod.ModelArgs.from_dict(dict(config, pooling="cls")))
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
    assert model.args.pooling == "cls"
    a = ref(mx.array(ids), attention_mask=mx.array(mask)).text_embeds
    b = model(mx.array(ids), attention_mask=mx.array(mask)).text_embeds
    assert np.array_equal(np.asarray(a), np.asarray(b))


def test_checkpoint_must_hold_every_weight(tmp_path, cpu):
    config = _tiny_bert_config()
    w = _params(bert_mod.Model(bert_mod.ModelArgs.from_dict(config)))
    del w["encoder.layer.1.output.LayerNorm.bias"]
    path = _write(tmp_path / "short", config, w)
    with pytest.raises(ValueError, match="LayerNorm.bias"):
        checkpoint.build(str(path))
