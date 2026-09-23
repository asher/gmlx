"""gmlx.tune.lora: the student setup on a tiny mlx-lm model, the
multiplier rules, the GGUF export filter, and the two argument helpers
``gmlx train`` resolves before it loads anything."""
from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import pytest

from mlx_lm.models import llama
from mlx_lm.models.llama import ModelArgs
from mlx_lm.tuner.lora import LoRALinear
from mlx.utils import tree_flatten

from gmlx.tune.lora import (
    LORA_KEYS,
    lora_modules_to_gguf,
    lora_scale,
    prepare_lora_student,
    probe_writable,
    resolve_model_arg,
)


def _model(layers=2):
    mx.random.seed(0)
    args = ModelArgs(model_type="llama", hidden_size=32, num_hidden_layers=layers,
                     intermediate_size=64, num_attention_heads=4,
                     num_key_value_heads=2, rms_norm_eps=1e-5, vocab_size=64)
    return llama.Model(args)


def test_prepare_adapts_every_layer_on_the_given_keys_and_freezes_the_base():
    model = _model(layers=3)
    n = prepare_lora_student(model, rank=2, scale=1.0, keys=LORA_KEYS)
    assert n == 3 * len(LORA_KEYS)
    names = [k for k, _ in tree_flatten(model.trainable_parameters())]
    assert names and all(k.endswith((".lora_a", ".lora_b")) for k in names)
    assert len(names) == 2 * n


def test_prepare_num_layers_and_default_keys():
    model = _model(layers=3)
    n = prepare_lora_student(model, rank=2, scale=1.0, num_layers=1)
    adapted = [k for k, m in model.named_modules() if isinstance(m, LoRALinear)]
    assert len(adapted) == n > 0
    assert all(".layers.2." in k for k in adapted)


def test_lora_scale_rules():
    assert lora_scale(16, scale=2.0) == 2.0
    assert lora_scale(16, alpha=32.0) == 2.0
    with pytest.raises(ValueError):
        lora_scale(16)
    with pytest.raises(ValueError):
        lora_scale(16, scale=1.0, alpha=1.0)


def test_export_filter_uses_dotted_suffixes():
    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.down_proj = LoRALinear.from_base(nn.Linear(4, 4), r=2, scale=1.0)

    class Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = Block()
            self.switch_mlp = Block()

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = [Layer()]

    model = Model()
    model.freeze()
    model.apply_to_modules(
        lambda _k, m: m.unfreeze(keys=["lora_a", "lora_b"], recurse=False)
        if isinstance(m, LoRALinear) else None)
    paths = [p for p, _a, _b in lora_modules_to_gguf(model)]
    assert paths == ["layers.0.mlp.down_proj", "layers.0.switch_mlp.down_proj"]
    dense = [p for p, _a, _b in lora_modules_to_gguf(model, keys=("mlp.down_proj",))]
    assert dense == ["layers.0.mlp.down_proj"]
    a = dict((p, a) for p, a, _b in lora_modules_to_gguf(model))["layers.0.mlp.down_proj"]
    assert a.shape == (2, 4)      # GGUF orientation: (rank, in)


def test_resolve_model_arg_paths_and_config_ids(tmp_path):
    base = tmp_path / "base.gguf"
    base.write_text("x")
    assert resolve_model_arg(str(base)) == (str(base), None, None)
    assert resolve_model_arg("Some-Model.gguf")[0] == "Some-Model.gguf"
    cfg = tmp_path / "config.yaml"
    cfg.write_text(f"models:\n  mybase:\n    path: {base}\n")
    path, note, err = resolve_model_arg("mybase", str(cfg))
    assert (path, err) == (str(base), None)
    assert "mybase" in note and str(base) in note
    path, note, err = resolve_model_arg("unknown-id", str(cfg))
    assert err is None and path == "unknown-id"


def test_probe_writable(tmp_path):
    assert probe_writable(str(tmp_path / "sub" / "adapter.gguf")) is None
    assert (tmp_path / "sub").is_dir()
    blocker = tmp_path / "file"
    blocker.write_text("x")
    assert probe_writable(str(blocker / "adapter.gguf")) is not None



def test_prepare_leaves_expert_stacks_out_of_the_adapter():
    """mlx-lm wraps a switch (expert) layer in LoRA factors that are never
    trained or exported here; the wrapper comes off again so the fused
    expert paths stay and no gather runs for nothing."""
    from mlx_lm.models.switch_layers import SwitchLinear
    from mlx_lm.tuner.lora import LoRASwitchLinear

    model = _model(layers=2)
    model.layers[0].mlp.gate_proj = SwitchLinear(32, 64, 2)
    n = prepare_lora_student(model, rank=2, scale=1.0, keys=["mlp.gate_proj", "self_attn.q_proj"])
    assert n == 3
    assert isinstance(model.layers[0].mlp.gate_proj, SwitchLinear)
    assert isinstance(model.layers[1].mlp.gate_proj, LoRALinear)
    assert not any(isinstance(m, LoRASwitchLinear) for _k, m in model.named_modules())
    names = [k for k, _ in tree_flatten(model.trainable_parameters())]
    assert len(names) == 2 * n and not any("layers.0.mlp" in k for k in names)
