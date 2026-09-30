"""Gemma 4 MLX checkpoints load the owned model on every MLX-checkpoint route.

distill's ``load_mlx_student``, ``vlm_text_only.ModelConfig`` and the
mlx-lm server's loader resolve their classes through
``gmlx.models.gemma4.text.get_model_classes``: a flat gemma4_text config
builds the owned text model, and a gemma4 container config builds the
vendored mlx-lm container around it. The tiny float32 checkpoints are
written to a temp dir, flat and container, float and affine quantized
with a per-path router entry. The reference is ``mlx_lm.load`` on the
same directory, run in a child that imports no gmlx. Two raw HF
directories, named as the Google checkpoints name their tensors, carry
the k and v weights of KV-shared layers and tower keys, which
``mlx_lm.load`` refuses, and load in full. A LoRA adapter saved the way
mlx-lm saves one loads on both sides. Two child processes lock what a
load on these routes leaves behind: an unconfigured attention dispatch,
and no stock mlx-lm gemma4 module imported.

On the mlx-vlm server, a path that is not a GGUF loads with mlx-vlm's
loader, whose resolver hands back the owned module of a gemma4 or
gemma4_unified checkpoint. Those tiny checkpoints are written from the
stock mlx-vlm classes, and the owned container's text logits match the
stock container's.
"""

from __future__ import annotations

import importlib
import json
import os
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

from gmlx.load import vlm
from gmlx.models.gemma4 import language, lm_container, text

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


def _stock_model(config: dict):
    """mlx-lm's class for ``config``, with random float32 weights."""
    from gmlx._vendor.mlx_lm.utils import _get_classes

    Model, ModelArgs = _get_classes(json.loads(json.dumps(config)))
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
    from mlx_lm import load
    jobs, ids, out = json.loads(sys.argv[1]), json.loads(sys.argv[2]), sys.argv[3]
    got = {}
    for name, (path, adapter) in jobs.items():
        model = load(path, adapter_path=adapter)[0]
        got[name] = np.array(model(mx.array([ids])).astype(mx.float32))
    np.savez(out, **got)
    print(json.dumps(sorted(m for m in sys.modules if m.startswith("gmlx"))))
""")


@pytest.fixture(scope="module")
def reference(dirs, adapter, tmp_path_factory) -> dict:
    """mlx_lm.load logits for each mlx-lm-named checkpoint, from a child
    that imports no gmlx."""
    jobs = {name: [str(dirs[name]), None]
            for name in ("flat", "flat_q4", "container", "container_q4")}
    jobs["adapter"] = [str(dirs["flat"]), str(adapter)]
    out = tmp_path_factory.mktemp("ref") / "ref.npz"
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    r = subprocess.run([sys.executable, "-c", _REFERENCE, json.dumps(jobs),
                        json.dumps(IDS), str(out)],
                       capture_output=True, text=True, env=env, timeout=600,
                       cwd=str(out.parent))
    assert r.returncode == 0, r.stderr[-4000:]
    assert json.loads(r.stdout.strip().splitlines()[-1]) == []
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
    from mlx_lm.utils import load_config
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
        "stock": sorted(m for m in ("mlx_lm.models.gemma4_text", "mlx_lm.models.gemma4")
                        if m in sys.modules),
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


def test_the_mlx_checkpoint_routes_import_no_stock_gemma4_text(routes):
    assert routes["stock"] == []


def test_an_mlx_checkpoint_load_leaves_the_dispatch_unconfigured(routes):
    """Every attention call goes to the owned base, and no claim moves."""
    assert routes["gates"] == {"sparse": False, "cascade": False,
                               "batched": False}
    assert routes["base_is_owned"]
    assert routes["base_calls"] == 2 * _MOE["num_hidden_layers"]
    assert routes["claims_moved"] == [False, False, False]


@pytest.mark.parametrize("name", ["flat", "flat_q4", "container", "container_q4"])
def test_the_mlx_lm_server_route_builds_the_owned_model(name, dirs, reference,
                                                        monkeypatch):
    """A path that is not a GGUF falls through the bridge to the stock
    ``ModelProvider._load``, which loads with the server module's ``load``."""
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
    """A tiny float32 mlx-vlm checkpoint written from the stock container
    of ``model_type``, and that container's text logits."""
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
    model = stock.Model(cfg)
    model.update(tree_unflatten([
        (k, v + 0.1 * mx.random.normal(v.shape)) if "norm" in k or "scale" in k
        else (k, v) for k, v in tree_flatten(model.parameters())]))
    mx.eval(model.parameters())
    return (_save(d, config, dict(tree_flatten(model.parameters()))),
            _vlm_logits(model))


def _vlm_logits(model) -> np.ndarray:
    with mx.stream(mx.cpu):
        out = model.language_model(mx.array([IDS])).logits
        return np.array(out.astype(mx.float32))


@pytest.mark.parametrize("model_type", ["gemma4", "gemma4_unified"])
def test_the_owned_module_carries_the_stock_package_names(model_type,
                                                          monkeypatch):
    """mlx-vlm's loader and processor lookup find every name of the stock
    package on the owned module, the same object but for the container
    and the language model."""
    from gmlx._vendor.mlx_vlm import utils

    stock = importlib.import_module(f"gmlx._vendor.mlx_vlm.models.{model_type}")
    owned = importlib.import_module(vlm._OWNED_VLM_MODULES[model_type])
    names = {n for n, v in vars(stock).items()
             if not n.startswith("_") and not isinstance(v, types.ModuleType)}
    assert set(owned.__all__) == names
    assert {n for n in names if getattr(owned, n) is not getattr(stock, n)} == {
        "LanguageModel", "Model"}
    assert owned.LanguageModel is language.LanguageModel

    monkeypatch.setattr(utils, "get_model_and_args", utils.get_model_and_args)
    before = utils.get_model_and_args
    vlm.install_owned_vlm_resolver()
    vlm.install_owned_vlm_resolver()
    assert utils.get_model_and_args._gmlx_orig is before
    assert utils.get_model_and_args({"model_type": model_type}) == (
        owned, model_type)
    other, _ = utils.get_model_and_args({"model_type": "gemma3"})
    assert other.__name__ == "gmlx._vendor.mlx_vlm.models.gemma3"


def test_a_checkpoint_model_file_keeps_its_module(tmp_path, monkeypatch):
    from gmlx._vendor.mlx_vlm import utils

    (tmp_path / "custom.py").write_text("MARK = 7\n")
    monkeypatch.setattr(utils, "get_model_and_args", utils.get_model_and_args)
    vlm.install_owned_vlm_resolver()
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
    owned = importlib.import_module(vlm._OWNED_VLM_MODULES[model_type])
    assert type(model) is owned.Model
    assert type(model.language_model) is language.LanguageModel
    assert config.model_type == model_type
    got = _vlm_logits(model)
    assert got.shape == want.shape
    assert np.abs(got - want).max() <= 1e-5
