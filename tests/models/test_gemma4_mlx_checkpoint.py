"""Gemma 4 MLX checkpoints load the owned model on every MLX-checkpoint route.

distill's ``load_mlx_student``, ``vlm_text_only.ModelConfig`` and the
mlx-lm server's loader resolve their classes through the copy's
``_get_classes``, where the gemma4_text path is an alias of
``gmlx.models.gemma4.text``: a flat gemma4_text config builds the owned
text model, and a gemma4 container config builds the copy's mlx-lm
container around it. The tiny float32 checkpoints are written to a temp
dir, flat and container, float and affine quantized with a per-path
router entry. The reference is the copied mlx-lm ``load_model`` on the
same directory with mlx-lm's own gemma4 classes, from the upstream files
of tests/pristine.py, run in a child. Two raw HF directories, named as the Google checkpoints name
their tensors, carry the k and v weights of KV-shared layers and tower
keys, which the copy's ``load`` refuses, and load in full. A LoRA
adapter saved the way mlx-lm saves one loads on both sides. A child
process locks what a load on these routes leaves behind: an
unconfigured attention dispatch, and the owned module at the copy's
gemma4_text path.

K-quant checkpoints, in the layout mlx-kquant's converter writes, build
the owned model: flat MoE, flat dense and container, beside a llama one that
builds the copy's class. Their reference is the same build with mlx-lm's
own classes in the same child, and the activation dtype is pinned to
bfloat16 on both sides. That reference shares gmlx's K-quant build, so a
second one does not: beside each K-quant checkpoint the fixture writes
the float32 checkpoint of the same weights, each wire tensor through
``mlx_kquant.dequantize``, and the copied mlx-lm ``load_model`` reads it
with mlx-lm's own classes. The K-quant logits stay near its float32
logits, and each K-quant leaf of the loaded model gives the float32
product of its dequantized weight.

On the mlx-vlm server, a path that is not a GGUF loads with the copied
mlx-vlm loader, whose gemma4 and gemma4_unified packages hold the owned
language model. Those tiny checkpoints are written from the upstream
mlx-vlm classes, and the loaded container's text logits match the
upstream container's.
"""

from __future__ import annotations

import importlib
import json
import os
import re
import shutil
import subprocess
import sys
import textwrap
import types
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx.utils import tree_flatten, tree_unflatten

from gmlx._vendor.mlx_lm.models import gemma4 as lm_container

from gmlx.load import vlm
from gmlx.models.gemma4 import language, text
from pristine import pristine

_TEXT = dict(
    model_type="gemma4_text", hidden_size=64, num_hidden_layers=4,
    intermediate_size=128, num_attention_heads=4, head_dim=16,
    global_head_dim=32, rms_norm_eps=1e-6, vocab_size=128,
    vocab_size_per_layer_input=128, num_key_value_heads=2,
    num_kv_shared_layers=0, hidden_size_per_layer_input=0,
    sliding_window=8, sliding_window_pattern=2, tie_word_embeddings=True,
)
_MOE = dict(_TEXT, enable_moe_block=True, num_experts=4, top_k_experts=2,
            moe_intermediate_size=32, attention_k_eq_v=True,
            num_global_key_value_heads=1)
# E2B shape: per-layer inputs and KV-shared layers.
_PLE = dict(_TEXT, num_kv_shared_layers=2, hidden_size_per_layer_input=8)
_QUANT = {"group_size": 32, "bits": 4}
_ROUTER_BITS = {"group_size": 64, "bits": 8}
IDS = [2, 17, 42, 99, 7, 63, 5, 28, 11, 90, 4, 8, 9, 10]
_TOWER_KEYS = ("model.vision_tower.encoder.weight",
               "model.audio_tower.encoder.weight",
               "model.embed_vision.embedding_projection.weight",
               "model.embed_audio.embedding_projection.weight")
# The per-layer HF tensor names of google/gemma-4-E2B-it and
# google/gemma-4-26B-A4B-it, layer index N, under model.language_model.
_E2B_LAYER = {
    "input_layernorm.weight", "layer_scalar", "mlp.down_proj.weight",
    "mlp.gate_proj.weight", "mlp.up_proj.weight",
    "per_layer_input_gate.weight", "per_layer_projection.weight",
    "post_attention_layernorm.weight", "post_feedforward_layernorm.weight",
    "post_per_layer_input_norm.weight", "pre_feedforward_layernorm.weight",
    "self_attn.k_norm.weight", "self_attn.k_proj.weight",
    "self_attn.o_proj.weight", "self_attn.q_norm.weight",
    "self_attn.q_proj.weight", "self_attn.v_proj.weight"}
_A4B_LAYER = {
    "experts.down_proj", "experts.gate_up_proj", "input_layernorm.weight",
    "layer_scalar", "mlp.down_proj.weight", "mlp.gate_proj.weight",
    "mlp.up_proj.weight", "post_attention_layernorm.weight",
    "post_feedforward_layernorm.weight", "post_feedforward_layernorm_1.weight",
    "post_feedforward_layernorm_2.weight", "pre_feedforward_layernorm.weight",
    "pre_feedforward_layernorm_2.weight", "router.per_expert_scale",
    "router.proj.weight", "router.scale", "self_attn.k_norm.weight",
    "self_attn.k_proj.weight", "self_attn.o_proj.weight",
    "self_attn.q_norm.weight", "self_attn.q_proj.weight",
    "self_attn.v_proj.weight"}


def _tokenizer(d: Path) -> None:
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    vocab = {"<pad>": 0, "<eos>": 1, "<bos>": 2}
    vocab.update({f"w{i}": i for i in range(3, _TEXT["vocab_size"])})
    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<pad>"))
    tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    PreTrainedTokenizerFast(tokenizer_object=tok, eos_token="<eos>",
                            bos_token="<bos>", pad_token="<pad>").save_pretrained(d)


def _stock_classes(config: dict):
    """mlx-lm's own classes for ``config``: the upstream files for the
    gemma4 types, whose paths in the copy hold the owned model."""
    from gmlx._vendor.mlx_lm.utils import _get_classes

    if config.get("model_type") in ("gemma4_text", "gemma4"):
        module = pristine("gmlx._vendor.mlx_lm.models." + config["model_type"])
        return module.Model, module.ModelArgs
    return _get_classes(config)


def _stock_model(config: dict):
    """mlx-lm's class for ``config``, with random float32 weights."""
    Model, ModelArgs = _stock_classes(json.loads(json.dumps(config)))
    mx.random.seed(3)
    model = Model(ModelArgs.from_dict(json.loads(json.dumps(config))))
    # Norm scales away from one, so a dropped norm weight shows.
    model.update(tree_unflatten([
        (k, v + 0.1 * mx.random.normal(v.shape)) if "norm" in k or "scale" in k
        else (k, v) for k, v in tree_flatten(model.parameters())]))
    mx.eval(model.parameters())
    return model


def _save(d: Path, config: dict, weights: dict) -> Path:
    d.mkdir(parents=True)
    mx.save_safetensors(str(d / "model.safetensors"), weights)
    (d / "config.json").write_text(json.dumps(config))
    _tokenizer(d)
    return d


def _checkpoint(d: Path, container: bool, quantized: bool) -> Path:
    text_cfg = dict(_MOE, eos_token_id=1)
    config = ({"model_type": "gemma4", "text_config": text_cfg,
               "vocab_size": _TEXT["vocab_size"], "eos_token_id": 1}
              if container else text_cfg)
    model = _stock_model(config)
    prefix = "language_model." if container else ""
    if quantized:
        router = f"{prefix}model.layers.0.router.proj"
        config = dict(config, quantization={**_QUANT, router: _ROUTER_BITS})

        def pick(path, m):
            if path == router:
                return _ROUTER_BITS
            return hasattr(m, "to_quantized") and m.weight.shape[-1] % 32 == 0

        nn.quantize(model, **_QUANT, class_predicate=pick)
        mx.eval(model.parameters())
    return _save(d, config, dict(tree_flatten(model.parameters())))


# A llama K-quant checkpoint, for the route's other architectures.
_LLAMA = dict(model_type="llama", hidden_size=64, num_hidden_layers=2,
              intermediate_size=128, num_attention_heads=4,
              num_key_value_heads=2, rms_norm_eps=1e-6, vocab_size=128,
              tie_word_embeddings=False, eos_token_id=1)


def _kq_config(name: str) -> dict:
    """The config of K-quant checkpoint ``name``. The embedding emits
    bfloat16, and the owned softcap returns float32 logits where upstream
    rounds them to bfloat16 (test_gemma4_owned_softcap.py), so the
    Gemma 4 softcap is off here."""
    if name == "llama_kq":
        return dict(_LLAMA)
    text_cfg = dict(_TEXT if name == "flat_kq_dense" else _MOE,
                    eos_token_id=1, final_logit_softcapping=None)
    if name == "container_kq":
        return {"model_type": "gemma4", "text_config": text_cfg,
                "vocab_size": _TEXT["vocab_size"], "eos_token_id": 1}
    return text_cfg


_KQ_NAMES = ("flat_kq", "flat_kq_dense", "container_kq", "llama_kq")


def _deq(name: str) -> str:
    """The name of the dequantized float32 checkpoint of K-quant ``name``."""
    return name.replace("_kq", "") + "_deq"


def _kq_checkpoint(d: Path, config: dict, wrong: bool = False) -> tuple[Path, ...]:
    """A checkpoint in the layout mlx-kquant's converter writes: q8_0
    wire bytes and the one-byte scales placeholder at every embedding,
    linear and switch leaf whose rows the codec takes, and the map of
    them in the config. The converter itself walks stock mlx-lm classes.

    Beside it, the float32 checkpoint of the same weights: each wire
    tensor through ``mlx_kquant.dequantize`` here, no quantization block.
    With ``wrong``, a third one that holds one attention projection with
    its rows reversed."""
    import mlx_kquant as kq

    from gmlx._vendor.mlx_lm.models.switch_layers import SwitchLinear

    model = _stock_model(config)
    weights = dict(tree_flatten(model.parameters()))
    floats = dict(weights)
    per_tensor = {}
    for path, m in model.named_modules():
        if (isinstance(m, (nn.Embedding, nn.Linear, SwitchLinear))
                and m.weight.shape[-1] % 32 == 0):
            wire, scales = kq.quantize(m.weight, "q8_0")
            weights[f"{path}.weight"] = wire
            weights[f"{path}.scales"] = mx.zeros((1,), dtype=mx.uint8)
            floats[f"{path}.weight"] = kq.dequantize(wire, scales, "q8_0",
                                                     mx.float32)
            per_tensor[path] = "q8_0"
    mx.eval(weights, floats)
    block = {"mode": "kquant", "per_tensor": per_tensor}
    out = [_save(d, dict(config, quantization=block, quantization_config=block),
                 weights),
           _save(d.with_name(_deq(d.name)), config, floats)]
    if wrong:
        key = sorted(k for k in floats if k.endswith("self_attn.q_proj.weight"))[0]
        assert key.removesuffix(".weight") in per_tensor
        out.append(_save(d.with_name(_deq(d.name) + "_wrong"), config,
                         dict(floats, **{key: floats[key][::-1]})))
    return tuple(out)


def _hf_names(weights: dict, moe: bool) -> dict:
    """mlx-lm container names to the Google checkpoint names."""
    out = {}
    for k, v in weights.items():
        k = "model.language_model." + k.removeprefix("language_model.model.")
        if moe and ".experts.switch_glu." in k:
            base = k.split(".experts.")[0] + ".experts"
            if k.endswith("down_proj.weight"):
                out[f"{base}.down_proj"] = v
            elif k.endswith("gate_proj.weight"):
                up = weights[k.replace("model.language_model.",
                                       "language_model.model.")
                             .replace("gate_proj", "up_proj")]
                out[f"{base}.gate_up_proj"] = mx.concatenate([v, up], axis=-2)
            continue
        out[k] = v
    return out


def _raw_hf(d: Path, text_cfg: dict, moe: bool) -> tuple[Path, Path]:
    """A raw HF directory and the same weights under mlx-lm names."""
    config = {"model_type": "gemma4", "vocab_size": _TEXT["vocab_size"],
              "eos_token_id": 1,
              "text_config": dict(text_cfg, use_bidirectional_attention="vision"),
              "vision_config": {"hidden_size": 16}}
    mlx_cfg = {"model_type": "gemma4", "vocab_size": _TEXT["vocab_size"],
               "eos_token_id": 1, "text_config": text_cfg}
    weights = dict(tree_flatten(_stock_model(mlx_cfg).parameters()))
    hf = _hf_names(weights, moe)
    first_shared = text_cfg["num_hidden_layers"] - text_cfg["num_kv_shared_layers"]
    rng = np.random.default_rng(5)
    for layer in range(first_shared, text_cfg["num_hidden_layers"]):
        head = text_cfg["head_dim"]
        for leaf, shape in (("k_proj.weight", (2 * head, 64)),
                            ("v_proj.weight", (2 * head, 64)),
                            ("k_norm.weight", (head,))):
            hf[f"model.language_model.layers.{layer}.self_attn.{leaf}"] = (
                mx.array(rng.standard_normal(shape).astype(np.float32)))
    for key in _TOWER_KEYS:
        hf[key] = mx.zeros((4, 4))
    raw = _save(d / "hf", config, hf)
    return raw, _save(d / "mlx", mlx_cfg, weights)


@pytest.fixture(scope="module")
def dirs(tmp_path_factory) -> dict:
    root = tmp_path_factory.mktemp("g4ckpt")
    out = {}
    for container in (False, True):
        for quantized in (False, True):
            name = (("container" if container else "flat")
                    + ("_q4" if quantized else ""))
            out[name] = _checkpoint(root / name, container, quantized)
    for name in _KQ_NAMES:
        made = _kq_checkpoint(root / name, _kq_config(name),
                              wrong=name == "flat_kq")
        out[name], out[_deq(name)] = made[:2]
        if made[2:]:
            out[_deq(name) + "_wrong"] = made[2]
    out["hf_ple"], out["hf_ple_mlx"] = _raw_hf(root / "hf_ple", _PLE, False)
    out["hf_moe"], out["hf_moe_mlx"] = _raw_hf(root / "hf_moe", _MOE, True)
    return out


@pytest.fixture(scope="module")
def adapter(tmp_path_factory, dirs) -> Path:
    """A LoRA adapter for the flat checkpoint, made with mlx-lm's
    linear_to_lora_layers on mlx-lm's class and saved as mlx-lm saves one."""
    from gmlx._vendor.mlx_lm.tuner.utils import linear_to_lora_layers
    from gmlx._vendor.mlx_lm.utils import load_model

    d = tmp_path_factory.mktemp("adapter")
    model, _ = load_model(dirs["flat"])
    model.freeze()
    params = {"rank": 4, "scale": 10.0, "dropout": 0.0,
              "keys": ["self_attn.q_proj", "self_attn.v_proj", "mlp.down_proj"]}
    linear_to_lora_layers(model, 2, params)
    mx.random.seed(9)
    trained = {k: 0.05 * mx.random.normal(v.shape)
               for k, v in tree_flatten(model.trainable_parameters())}
    mx.save_safetensors(str(d / "adapters.safetensors"), trained)
    (d / "adapter_config.json").write_text(json.dumps(
        {"fine_tune_type": "lora", "num_layers": 2, "lora_parameters": params}))
    return d


_REFERENCE = textwrap.dedent("""
    import json, sys
    import mlx.core as mx
    import numpy as np
    mx.set_default_device(mx.cpu)
    from pathlib import Path
    from gmlx._vendor.mlx_lm.utils import (
        _get_classes, load_adapters, load_config, load_model)
    from gmlx.load import mlx_checkpoint
    jobs, ids, out = json.loads(sys.argv[1]), json.loads(sys.argv[2]), sys.argv[3]
    sys.path.insert(0, sys.argv[4])
    from pristine import pristine

    def stock_classes(config):
        if config.get("model_type") in ("gemma4_text", "gemma4"):
            module = pristine("gmlx._vendor.mlx_lm.models." + config["model_type"])
            return module.Model, module.ModelArgs
        return _get_classes(config)

    got, classes = {}, set()
    for name, (path, adapter) in jobs.items():
        if "_kq" in name:
            config = load_config(Path(path))
            model = mlx_checkpoint._load_kquant(
                Path(path), config,
                mlx_checkpoint._kquant_per_tensor(config, Path(path)), False,
                stock_classes)
        else:
            model = load_model(Path(path), False,
                               get_model_classes=stock_classes)[0]
        if adapter is not None:
            model = load_adapters(model, adapter)
            model.eval()
        classes.add(type(model).__module__)
        got[name] = np.array(model(mx.array([ids])).astype(mx.float32))
    np.savez(out, **got)
    print(json.dumps(sorted(classes)))
""")


@pytest.fixture(scope="module")
def reference(dirs, adapter, tmp_path_factory) -> dict:
    """The logits of the copy's classes: the copied mlx-lm load for each
    mlx-lm-named checkpoint, the dequantized ones among them, and the
    K-quant build with the copy's class resolver for each K-quant one,
    from a child."""
    jobs = {name: [str(dirs[name]), None]
            for name in ("flat", "flat_q4", "container", "container_q4",
                         *_KQ_NAMES, *map(_deq, _KQ_NAMES),
                         _deq("flat_kq") + "_wrong")}
    jobs["adapter"] = [str(dirs["flat"]), str(adapter)]
    jobs["flat_kq_adapter"] = [str(dirs["flat_kq"]), str(adapter)]
    out = tmp_path_factory.mktemp("ref") / "ref.npz"
    env = dict(os.environ, GMLX_ACTIVATION_DTYPE="bfloat16")
    r = subprocess.run([sys.executable, "-c", _REFERENCE, json.dumps(jobs),
                        json.dumps(IDS), str(out),
                        str(Path(__file__).resolve().parents[1])],
                       capture_output=True, text=True, env=env, timeout=600)
    assert r.returncode == 0, r.stderr[-4000:]
    assert json.loads(r.stdout.strip().splitlines()[-1]) == [
        "gmlx._vendor.mlx_lm.models.llama",
        "pristine.mlx_lm.models.gemma4",
        "pristine.mlx_lm.models.gemma4_text"]
    return dict(np.load(out))


def _logits(model) -> np.ndarray:
    with mx.stream(mx.cpu):
        return np.array(model(mx.array([IDS])).astype(mx.float32))


def _owned_classes(model) -> bool:
    if isinstance(model, lm_container.Model):
        return isinstance(model.language_model, text.Model)
    return isinstance(model, text.Model)


@pytest.mark.parametrize("name", ["flat", "flat_q4", "container", "container_q4"])
def test_the_student_route_builds_the_owned_model(name, dirs, reference):
    from gmlx.distill.student import load_mlx_student

    model, cfg, tok = load_mlx_student(str(dirs[name]))
    assert _owned_classes(model)
    assert cfg["model_type"] == ("gemma4" if "container" in name else "gemma4_text")
    assert tok.eos_token_id == 1
    got, want = _logits(model), reference[name]
    assert got.shape == want.shape
    assert np.abs(got - want).max() <= 1e-5


@pytest.fixture
def bf16_activations(monkeypatch):
    """The activation dtype of the K-quant reference."""
    monkeypatch.setenv("GMLX_ACTIVATION_DTYPE", "bfloat16")


@pytest.mark.parametrize("name", ["flat_kq", "flat_kq_dense", "container_kq"])
def test_a_kquant_checkpoint_builds_the_owned_model(name, dirs, reference,
                                                    bf16_activations):
    """The copy's load_model cannot read a K-quant checkpoint."""
    from mlx_kquant.nn import KQuantEmbedding, KQuantLinear, KQuantSwitchLinear

    from gmlx.distill.student import load_mlx_student

    model, cfg, tok = load_mlx_student(str(dirs[name]))
    assert _owned_classes(model)
    kinds = {type(m) for _, m in model.named_modules()}
    assert {KQuantEmbedding, KQuantLinear} <= kinds
    assert (KQuantSwitchLinear in kinds) == (name != "flat_kq_dense")
    assert cfg["quantization"]["mode"] == "kquant"
    assert tok.eos_token_id == 1
    got, want = _logits(model), reference[name]
    assert got.shape == want.shape
    assert np.abs(got - want).max() <= 1e-5


def test_a_kquant_checkpoint_of_another_architecture_builds_mlx_lms_class(
        dirs, reference, bf16_activations):
    from mlx_kquant.nn import KQuantLinear

    from gmlx.load import mlx_checkpoint

    model, tok = mlx_checkpoint.load(str(dirs["llama_kq"]))
    assert type(model).__module__ == "gmlx._vendor.mlx_lm.models.llama"
    assert isinstance(model.layers[0].self_attn.q_proj, KQuantLinear)
    assert tok.eos_token_id == 1
    assert np.abs(_logits(model) - reference["llama_kq"]).max() <= 1e-5


def test_a_kquant_checkpoint_with_a_model_file_builds_that_files_class(
        dirs, reference, tmp_path, bf16_activations):
    """The copy's load_model reads group_size and bits from the
    quantization block, and a K-quant block has neither."""
    from mlx_kquant.nn import KQuantEmbedding, KQuantLinear

    from gmlx.load import mlx_checkpoint

    d = tmp_path / "custom"
    shutil.copytree(dirs["llama_kq"], d)
    config = json.loads((d / "config.json").read_text())
    (d / "config.json").write_text(json.dumps(dict(config, model_file="custom.py")))
    (d / "custom.py").write_text(textwrap.dedent("""
        from gmlx._vendor.mlx_lm.models.llama import Model as Llama, ModelArgs


        class Model(Llama):
            pass
    """))
    model, tok, cfg = mlx_checkpoint.load(str(d), return_config=True)
    assert (type(model).__module__, type(model).__name__) == ("custom_model", "Model")
    assert isinstance(model.model.embed_tokens, KQuantEmbedding)
    assert isinstance(model.layers[0].self_attn.q_proj, KQuantLinear)
    assert cfg["model_file"] == "custom.py"
    assert tok.eos_token_id == 1
    assert np.abs(_logits(model) - reference["llama_kq"]).max() <= 1e-5


def test_a_kquant_model_file_that_is_no_module_names_the_file(dirs, tmp_path):
    from gmlx.load import mlx_checkpoint

    d = tmp_path / "nofile"
    d.mkdir()
    config = json.loads((dirs["llama_kq"] / "config.json").read_text())
    (d / "config.json").write_text(json.dumps(dict(config, model_file="custom.txt")))
    with pytest.raises(ValueError, match=re.escape(
            f"{d / 'custom.txt'}: not a Python module")):
        mlx_checkpoint.load(str(d))


# K-quant logits against the float32 logits of mlx-lm's class on the
# dequantized weights. The K-quant matmul takes and returns bfloat16 where
# the float model runs float32, so each leaf is off by a fraction of a
# percent (the leaf test below) and the layers of a toy model grow that:
# measured 0.026 for llama_kq, 0.195 for the dense Gemma 4 and 0.418 for
# the MoE ones, where it can turn a routing choice, on logits that reach
# 2.0, 3.8 and 4.4. Each tolerance is about twice its measurement.
_DEQ_ATOL = {"llama_kq": 0.06, "flat_kq_dense": 0.4, "flat_kq": 0.9,
             "container_kq": 0.9}
# One q_proj with its rows reversed moves the float logits of the MoE
# checkpoint by 4.3.
_DEQ_WRONG = 3.0


@pytest.mark.parametrize("name", _KQ_NAMES)
def test_a_kquant_checkpoint_gives_the_logits_of_its_dequantized_weights(
        name, dirs, reference, bf16_activations):
    """The reference never enters gmlx's K-quant route: the weights are
    dequantized in the fixture and mlx-lm's own class runs them."""
    from gmlx.load import mlx_checkpoint

    model, _ = mlx_checkpoint.load(str(dirs[name]))
    got, want = _logits(model), reference[_deq(name)]
    assert got.shape == want.shape
    gap = float(np.abs(got - want).max())
    assert gap <= _DEQ_ATOL[name], gap


def test_the_dequantized_reference_sees_one_wrong_projection(reference):
    """The logit tolerance has the power to see a single leaf with wrong
    weights: one q_proj with its rows reversed."""
    name = _deq("flat_kq")
    gap = float(np.abs(reference[name + "_wrong"] - reference[name]).max())
    assert gap > _DEQ_WRONG > 3 * _DEQ_ATOL["flat_kq"], gap


# A K-quant leaf against the float32 product of its dequantized weight, as
# a fraction of the largest output: measured at most 0.011 on a linear,
# 0.008 on an expert stack and 0.003 on an embedding lookup, and 1.34 on
# the projection with reversed rows.
_LEAF_RTOL = 0.02


def _leaf_gaps(model, twin: dict, per_tensor: dict) -> dict:
    """``{leaf: gap}`` of each K-quant leaf of ``model`` on one random
    input against plain float32 arithmetic on ``twin``, the weights of the
    dequantized checkpoint. An embedding gives its lookup and, under
    ``.as_linear``, its product."""
    from mlx_kquant.nn import KQuantEmbedding, KQuantLinear, KQuantSwitchLinear

    leaves = dict(model.named_modules())
    rng = np.random.default_rng(0)
    pairs = {}
    with mx.stream(mx.cpu):
        for path in per_tensor:
            m, w = leaves[path], twin[f"{path}.weight"]
            bias = twin.get(f"{path}.bias")
            x = mx.array(rng.standard_normal((1, 5, w.shape[-1])).astype(np.float32))
            if isinstance(m, KQuantEmbedding):
                ids = mx.array([IDS])
                pairs[path] = m(ids), w[ids]
                pairs[f"{path}.as_linear"] = m.as_linear(x), x @ w.T
            elif isinstance(m, KQuantSwitchLinear):
                x = mx.expand_dims(x, (-3, -2))
                picks = mx.array(rng.integers(0, w.shape[0], size=(1, 5, 2)))
                want = mx.gather_mm(x, w.swapaxes(-1, -2), rhs_indices=picks)
                if bias is not None:
                    want = want + mx.expand_dims(bias[picks], -2)
                pairs[path] = m(x, picks), want
            else:
                assert isinstance(m, KQuantLinear), (path, type(m))
                want = x @ w.T
                pairs[path] = m(x), want if bias is None else want + bias
        out = {}
        for path, (got, want) in pairs.items():
            assert got.shape == want.shape, path
            got, want = (np.array(a.astype(mx.float32)) for a in (got, want))
            out[path] = float(np.abs(got - want).max() / np.abs(want).max())
    return out


def _kq_leaves(dirs, name: str, twin: str):
    from gmlx.load import mlx_checkpoint

    model, _ = mlx_checkpoint.load(str(dirs[name]))
    config = json.loads((dirs[name] / "config.json").read_text())
    per_tensor = config["quantization"]["per_tensor"]
    weights = mx.load(str(dirs[twin] / "model.safetensors"))
    return per_tensor, weights, _leaf_gaps(model, weights, per_tensor)


@pytest.mark.parametrize("name", _KQ_NAMES)
def test_each_kquant_leaf_gives_the_product_of_its_dequantized_weight(
        name, dirs, bf16_activations):
    """Every path of the checkpoint's map holds its own wire bytes under
    its own codec in the loaded model, whichever class built it."""
    per_tensor, _, gaps = _kq_leaves(dirs, name, _deq(name))
    assert set(per_tensor) <= set(gaps)
    worst = max(gaps, key=gaps.__getitem__)
    assert gaps[worst] <= _LEAF_RTOL, (worst, gaps[worst])


def test_a_leaf_with_wrong_weights_is_far_from_its_product(dirs,
                                                           bf16_activations):
    """The leaf tolerance sees the one projection whose rows the wrong
    checkpoint holds reversed, and only that one."""
    _, weights, gaps = _kq_leaves(dirs, "flat_kq", _deq("flat_kq") + "_wrong")
    wrong = sorted(k for k in weights
                   if k.endswith("self_attn.q_proj.weight"))[0]
    over = {path for path, gap in gaps.items() if gap > _LEAF_RTOL}
    assert over == {wrong.removesuffix(".weight")}
    assert gaps[wrong.removesuffix(".weight")] > 25 * _LEAF_RTOL


def test_an_adapter_loads_on_a_kquant_checkpoint(dirs, adapter, reference,
                                                 bf16_activations):
    """The copy's LoRA setup wraps a K-quant layer through
    gmlx.tune.kquant_lora."""
    from gmlx._vendor.mlx_lm.tuner.lora import LoRALinear

    from gmlx.distill.student import load_mlx_student

    model, _, _ = load_mlx_student(str(dirs["flat_kq"]), adapter_path=str(adapter))
    assert _owned_classes(model)
    wrapped = sum(isinstance(m, LoRALinear) for _, m in model.named_modules())
    assert wrapped == len([k for k in mx.load(str(adapter / "adapters.safetensors"))
                           if k.endswith(".lora_a")])
    got, want = _logits(model), reference["flat_kq_adapter"]
    assert np.abs(got - want).max() <= 1e-5
    assert np.abs(want - reference["flat_kq"]).max() > 1e-3


def test_a_kquant_config_with_no_map_names_the_checkpoint(dirs, tmp_path):
    from gmlx.load import mlx_checkpoint

    d = tmp_path / "nomap"
    d.mkdir()
    config = json.loads((dirs["flat_kq"] / "config.json").read_text())
    config["quantization"] = config["quantization_config"] = {"mode": "kquant"}
    (d / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match=re.escape(f"{d}: kquant config has no 'per_tensor' map")):
        mlx_checkpoint.load(str(d))


def _switch_twin(switch):
    class Model(nn.Module):
        def __init__(self, args):
            super().__init__()
            self.experts = switch(64, 32, 4, bias=False)

    return Model


class _TwinArgs:
    @staticmethod
    def from_dict(config):
        return config


def test_a_kquant_build_swaps_the_switch_classes_of_mlx_lm_and_mlx_vlm(tmp_path):
    """Models built from mlx-vlm's code hold its copy of SwitchLinear, a
    distinct type, and a K-quant expert stack must attach to it too."""
    import mlx_kquant as kq
    from mlx_kquant.nn import KQuantSwitchLinear
    from gmlx._vendor.mlx_lm.models.switch_layers import SwitchLinear
    from gmlx._vendor.mlx_vlm.models.switch_layers import SwitchLinear as VlmSwitchLinear

    from gmlx.load import mlx_checkpoint

    src = _switch_twin(SwitchLinear)(None)
    mx.random.seed(4)
    wire, _ = kq.quantize(mx.random.normal((4, 32, 64)), "q8_0")
    src.experts = KQuantSwitchLinear(4, 32, 64, False, "q8_0")
    src.experts.weight = wire
    mx.save_safetensors(str(tmp_path / "model.safetensors"),
                        dict(tree_flatten(src.parameters())))
    built = {}
    for cls in (SwitchLinear, VlmSwitchLinear):
        built[cls] = mlx_checkpoint._load_kquant(
            tmp_path, {}, {"experts": "q8_0"}, False,
            lambda config, cls=cls: (_switch_twin(cls), _TwinArgs))
        assert isinstance(built[cls].experts, KQuantSwitchLinear)
    assert mx.array_equal(built[SwitchLinear].experts.weight,
                          built[VlmSwitchLinear].experts.weight)


@pytest.mark.parametrize("over", [True, False], ids=["over", "under"])
def test_the_cache_pass_sizes_an_mlx_teacher_before_reading_it(over, dirs,
                                                               reference,
                                                               monkeypatch,
                                                               bf16_activations):
    """An MLX checkpoint teacher has no experts to stream, so the cache
    pass loads it lazy, sizes it against the wired budget and refuses one
    over it before its weights are read."""
    from gmlx.distill import student as _student
    from gmlx.distill import teacher as _teacher

    loads, sized = [], []
    real_load, real_size = _student.load_mlx_student, _teacher.teacher_over_budget

    def load(path, adapter_path=None, lazy=False):
        loads.append(lazy)
        return real_load(path, adapter_path=adapter_path, lazy=lazy)

    def size(model, budget=None):
        _, total, _ = real_size(model, budget=1)
        sized.append(total)
        return over, total, total - 1 if over else total

    monkeypatch.setattr(_student, "load_mlx_student", load)
    monkeypatch.setattr(_teacher, "teacher_over_budget", size)
    opts = types.SimpleNamespace(teacher=str(dirs["flat_kq"]), hf_source=None,
                                 stream_experts=False)
    if over:
        with pytest.raises(ValueError, match="against a wired budget"):
            _teacher.load_teacher(opts)
    else:
        model, config, arch, streaming, offloaded = _teacher.load_teacher(opts)
        assert _owned_classes(model)
        assert (arch, streaming, offloaded) == ("gemma4_text", False, 0)
        assert np.abs(_logits(model) - reference["flat_kq"]).max() <= 1e-5
    assert loads == [True] and len(sized) == 1 and sized[0] > 0


def test_the_student_route_resolves_a_repo_id(dirs, reference, monkeypatch):
    """A name that is no local path is a Hugging Face repo id, fetched with
    the files mlx-lm fetches while the offline window is shut."""
    import huggingface_hub

    from gmlx.distill.student import load_mlx_student
    from gmlx.serve import hf_cache

    calls = []

    def snapshot_download(repo_id, **kw):
        calls.append((repo_id, kw, hf_cache._OFFLINE_LOCK._is_owned()))
        return str(dirs["container_q4"])

    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot_download)
    model, cfg, tok = load_mlx_student("example-org/tiny-gemma-4")
    [(repo_id, kw, locked)] = calls
    assert (repo_id, locked) == ("example-org/tiny-gemma-4", True)
    assert kw["revision"] is None
    assert {"*.json", "model*.safetensors", "tokenizer.model"} <= set(
        kw["allow_patterns"])
    assert _owned_classes(model)
    assert cfg["model_type"] == "gemma4"
    assert tok.eos_token_id == 1
    assert np.abs(_logits(model) - reference["container_q4"]).max() <= 1e-5


def test_a_local_directory_is_not_fetched(dirs, monkeypatch):
    import huggingface_hub

    from gmlx.load import mlx_checkpoint

    def snapshot_download(*a, **k):
        raise AssertionError("fetched a local directory")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot_download)
    assert mlx_checkpoint.resolve(str(dirs["flat"])) == dirs["flat"]


@pytest.mark.parametrize("name", ["flat", "flat_q4", "container", "container_q4"])
def test_the_text_only_route_builds_the_owned_model(name, dirs, reference):
    from gmlx._vendor.mlx_lm.utils import load_config

    from gmlx.models.vlm_text_only import ModelConfig, TextOnlyModel

    path = dirs[name]
    config = load_config(path)
    model = TextOnlyModel(ModelConfig.from_dict(config))
    inner = model.language_model._model
    assert _owned_classes(inner)
    weights = model.sanitize(mx.load(str(path / "model.safetensors")))
    if (quant := config.get("quantization")) is not None:
        # mlx-lm's load_model predicate: the per-path entry, else any
        # module whose scales the checkpoint holds.
        nn.quantize(inner, group_size=quant["group_size"], bits=quant["bits"],
                    class_predicate=lambda p, m: quant.get(p) or (
                        hasattr(m, "to_quantized") and f"{p}.scales" in weights))
    model.load_weights(list(weights.items()), strict=True)
    assert np.abs(_logits(inner) - reference[name]).max() <= 1e-5


@pytest.mark.parametrize("name,layer_keys", [("hf_ple", _E2B_LAYER),
                                             ("hf_moe", _A4B_LAYER)])
def test_a_raw_hf_directory_loads_in_full(name, layer_keys, dirs):
    """Every parameter loads, the tower keys and the KV-shared k and v
    weights go, and the logits equal the owned load of the same weights
    under mlx-lm names."""
    from gmlx.distill.student import load_mlx_student

    raw = mx.load(str(dirs[name] / "model.safetensors"))
    names = {k.split(".", 4)[-1] for k in raw if ".layers." in k}
    assert names == layer_keys
    model, _, _ = load_mlx_student(str(dirs[name]))
    assert _owned_classes(model)
    kept = model.sanitize(dict(raw))
    assert not any(k.startswith(("vision_tower", "audio_tower", "embed_vision",
                                 "embed_audio")) for k in kept)
    assert set(kept) == {k for k, _ in tree_flatten(model.parameters())}
    ref, _, _ = load_mlx_student(str(dirs[f"{name}_mlx"]))
    assert np.array_equal(_logits(model), _logits(ref))


def test_an_adapter_loads_on_the_owned_model(dirs, adapter, reference):
    from safetensors import safe_open

    from gmlx.distill.student import load_mlx_student

    model, _, _ = load_mlx_student(str(dirs["flat"]), adapter_path=str(adapter))
    assert isinstance(model, text.Model)
    params = {k for k, _ in tree_flatten(model.parameters())}
    with safe_open(str(adapter / "adapters.safetensors"), "numpy") as f:
        keys = set(f.keys())
    assert keys and keys <= params
    got, want = _logits(model), reference["adapter"]
    assert np.abs(got - want).max() <= 1e-5
    assert np.abs(want - reference["flat"]).max() > 1e-3


_ROUTES = textwrap.dedent("""
    import json, sys
    import mlx.core as mx
    mx.set_default_device(mx.cpu)
    from gmlx._vendor.mlx_lm.utils import load_config
    from gmlx.distill.student import load_mlx_student
    from gmlx.models.vlm_text_only import ModelConfig, TextOnlyModel
    from gmlx.cache import attention
    from gmlx.models.gemma4 import batched_sdpa, dispatch
    from gmlx.upstream import cascade_sdpa, sparse_sdpa

    paths = json.loads(sys.argv[1])
    for path in paths:
        TextOnlyModel(ModelConfig.from_dict(load_config(__import__("pathlib").Path(path))))
    model = load_mlx_student(paths[0])[0]
    base, calls = dispatch._base, []
    dispatch._base = lambda *a: calls.append(1) or base(*a)
    before = (sparse_sdpa.claims(), cascade_sdpa.claims(), batched_sdpa.claims())
    for ids in ([[2, 17, 42, 99, 7, 63]], [[5]]):
        mx.eval(model(mx.array(ids)))
    print(json.dumps({
        "copy": {m: sys.modules[m].__name__
                 for m in ("gmlx._vendor.mlx_lm.models.gemma4_text",
                           "gmlx._vendor.mlx_lm.models.gemma4")
                 if m in sys.modules},
        "gates": dispatch.gates(),
        "base_is_owned": base is attention.scaled_dot_product_attention,
        "base_calls": len(calls),
        "claims_moved": [a != b for a, b in zip(before, (
            sparse_sdpa.claims(), cascade_sdpa.claims(), batched_sdpa.claims()))],
    }))
""")


@pytest.fixture(scope="module")
def routes(dirs) -> dict:
    paths = [str(dirs["flat"]), str(dirs["container_q4"]), str(dirs["hf_ple"])]
    r = subprocess.run([sys.executable, "-c", _ROUTES, json.dumps(paths)],
                       capture_output=True, text=True, env=dict(os.environ),
                       timeout=600)
    assert r.returncode == 0, r.stderr[-4000:]
    return json.loads(r.stdout.strip().splitlines()[-1])


def test_the_mlx_checkpoint_routes_find_the_owned_module_at_the_copied_path(routes):
    assert routes["copy"] == {
        "gmlx._vendor.mlx_lm.models.gemma4_text": "gmlx.models.gemma4.text",
        "gmlx._vendor.mlx_lm.models.gemma4": "gmlx._vendor.mlx_lm.models.gemma4"}


def test_an_mlx_checkpoint_load_leaves_the_dispatch_unconfigured(routes):
    """Every attention call goes to the owned base, and no claim moves."""
    assert routes["gates"] == {"sparse": False, "cascade": False,
                               "batched": False}
    assert routes["base_is_owned"]
    assert routes["base_calls"] == 2 * _MOE["num_hidden_layers"]
    assert routes["claims_moved"] == [False, False, False]


@pytest.mark.parametrize("name", ["flat", "flat_q4", "container", "container_q4",
                                  "flat_kq", "container_kq"])
def test_the_mlx_lm_server_route_builds_the_owned_model(name, dirs, reference,
                                                        monkeypatch, bf16_activations):
    """A path that is not a GGUF falls through the bridge to the copy's
    ``ModelProvider._load``, which loads with the server module's ``load``.
    A K-quant checkpoint stays on the sequential path, as a GGUF does."""
    from gmlx._vendor.mlx_lm import server

    from gmlx.serve import bridge_lm

    monkeypatch.setattr(server.ModelProvider, bridge_lm._BRIDGE_FLAG, False,
                        raising=False)
    monkeypatch.setattr(server.ModelProvider, "_load",
                        server.ModelProvider._load)
    monkeypatch.setattr(server, "load", server.load)
    bridge_lm.install_gguf_bridge()
    provider = types.SimpleNamespace(
        is_distributed=False, _tokenizer_config={},
        cli_args=types.SimpleNamespace(use_default_chat_template=False))
    server.ModelProvider._load(provider, str(dirs[name]))
    assert _owned_classes(provider.model)
    assert provider.is_batchable is ("_kq" not in name)
    assert provider.tokenizer.eos_token_id == 1
    assert np.abs(_logits(provider.model) - reference[name]).max() <= 1e-5


_VISION = {
    "gemma4": dict(hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                   num_attention_heads=2, num_key_value_heads=2, head_dim=16,
                   global_head_dim=16, position_embedding_size=64,
                   default_output_length=4),
    "gemma4_unified": dict(patch_size=4, model_patch_size=4, mm_embed_dim=32,
                           mm_posemb_size=16, num_soft_tokens=4,
                           output_proj_dims=32),
}


def _vlm_checkpoint(d: Path, model_type: str) -> tuple[Path, np.ndarray]:
    """A tiny float32 mlx-vlm checkpoint written from the upstream
    container of ``model_type``, and that container's text logits."""
    from gmlx._vendor.mlx_vlm.utils import update_module_configs

    text_type = ("gemma4_unified_text" if model_type == "gemma4_unified"
                 else "gemma4_text")
    config = {"model_type": model_type, "vocab_size": _TEXT["vocab_size"],
              "eos_token_id": 1, "text_config": dict(_MOE, model_type=text_type),
              "vision_config": _VISION[model_type], "audio_config": None}
    stock = importlib.import_module(f"gmlx._vendor.mlx_vlm.models.{model_type}")
    cfg = update_module_configs(
        stock.ModelConfig.from_dict(json.loads(json.dumps(config))), stock,
        json.loads(json.dumps(config)), ["text", "vision", "projector", "audio"])
    mx.random.seed(3)
    model = _upstream_container(model_type)(cfg)
    assert type(model.language_model) is not language.LanguageModel
    model.update(tree_unflatten([
        (k, v + 0.1 * mx.random.normal(v.shape)) if "norm" in k or "scale" in k
        else (k, v) for k, v in tree_flatten(model.parameters())]))
    mx.eval(model.parameters())
    return (_save(d, config, dict(tree_flatten(model.parameters()))),
            _vlm_logits(model))


def _upstream_container(model_type: str):
    return pristine("gmlx._vendor.mlx_vlm.models." + (
        "gemma4_unified.gemma4_unified" if model_type == "gemma4_unified"
        else "gemma4.gemma4")).Model


def _built_container(model_type: str):
    """The container a gmlx route builds: the copy's for gemma4, and the
    owned one, which the copy's path is an alias of, for gemma4_unified."""
    from gmlx._vendor.mlx_vlm.models.gemma4 import gemma4 as copy_g4

    from gmlx.models.gemma4 import unified

    return unified.Model if model_type == "gemma4_unified" else copy_g4.Model


def _vlm_logits(model) -> np.ndarray:
    with mx.stream(mx.cpu):
        out = model.language_model(mx.array([IDS])).logits
        return np.array(out.astype(mx.float32))


@pytest.mark.parametrize("model_type", ["gemma4", "gemma4_unified"])
def test_the_copied_package_holds_the_owned_classes(model_type):
    """mlx-vlm's loader resolves the copied package, with no resolver of
    gmlx's over it, and finds the owned language model and the container a
    gmlx route builds there."""
    from gmlx._vendor.mlx_vlm import utils

    assert not hasattr(vlm, "install_owned_vlm_resolver")
    assert not hasattr(utils.get_model_and_args, "_gmlx_orig")
    module, resolved = utils.get_model_and_args({"model_type": model_type})
    assert resolved == model_type
    assert module.__name__ == f"gmlx._vendor.mlx_vlm.models.{model_type}"
    assert module.LanguageModel is language.LanguageModel
    assert module.Model is _built_container(model_type)
    assert module.Model is not _upstream_container(model_type)


def test_a_checkpoint_model_file_keeps_its_module(tmp_path):
    from gmlx._vendor.mlx_vlm import utils

    (tmp_path / "custom.py").write_text("MARK = 7\n")
    module, model_type = utils.get_model_and_args(
        {"model_type": "gemma4", "model_file": "custom.py"}, tmp_path)
    assert (module.MARK, model_type) == (7, "custom")


@pytest.mark.parametrize("model_type", ["gemma4", "gemma4_unified"])
def test_the_mlx_vlm_server_route_builds_the_owned_container(model_type,
                                                             tmp_path,
                                                             monkeypatch):
    """A path that is not a GGUF falls through the bridge to mlx-vlm's
    ``load_model_resources``. The tiny directory has no processor, which
    is beside the point here."""
    from gmlx._vendor.mlx_vlm import utils

    from gmlx.serve import bridge_vlm

    path, want = _vlm_checkpoint(tmp_path / model_type, model_type)
    gen = importlib.import_module("gmlx._vendor.mlx_vlm.server.generation")
    monkeypatch.setattr(gen, bridge_vlm._BRIDGE_FLAG, False, raising=False)
    monkeypatch.setattr(gen, "load_model_resources", gen.load_model_resources)
    monkeypatch.setattr(utils, "get_model_and_args", utils.get_model_and_args)
    monkeypatch.setattr(utils, "load_processor",
                        lambda *a, **k: types.SimpleNamespace())
    monkeypatch.setattr(utils, "load_image_processor", lambda *a, **k: None)
    bridge_vlm.install_gguf_server_bridge()
    model, _processor, config = gen.load_model_resources(str(path), None)
    assert type(model) is _built_container(model_type)
    assert type(model.language_model) is language.LanguageModel
    assert config.model_type == model_type
    got = _vlm_logits(model)
    assert got.shape == want.shape
    assert np.abs(got - want).max() <= 1e-5
