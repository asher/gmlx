"""LoRA on merged shared-input tensors and on SenseNova Loop: adapters keyed
the way llama.cpp keys them (a looped file's visit-indexed blocks, the
merged ``attn_qkv_gate`` and ``ssm_beta_alpha``), and training on the
physical layers through the looped forward."""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

import mlx.core as mx

sys.path.insert(0, os.path.dirname(__file__))
import qwen35_tiny as T  # noqa: E402

ARCH = "sensenova_loop"
IDS = [[5, 9, 30, 41, 7, 66, 12, 3, 90, 100, 15, 22]]
RANK, ALPHA = 4, 8.0
# Physical 5 (after the loop) sits at file block 9; physical 1 and 2 are
# looped GDN layers with merged tensors; physical 3 loops with attention.
AFTER = T.LAYERS - 1
AFTER_BLOCK = T._file_block(ARCH, AFTER)
MERGED = {
    "blk.1.attn_qkv_gate.weight": ("blk.1.attn_qkv.weight", "blk.1.attn_gate.weight"),
    "blk.2.ssm_beta_alpha.weight": ("blk.2.ssm_beta.weight", "blk.2.ssm_alpha.weight"),
    "blk.3.attn_q.weight": ("blk.3.attn_q.weight",),
    "blk.0.ffn_up.weight": ("blk.0.ffn_up.weight",),
}
PAIRS = {
    ARCH: {f"blk.{AFTER_BLOCK}.ffn_down.weight": (f"blk.{AFTER}.ffn_down.weight",), **MERGED},
    "qwen35": MERGED,
}


@pytest.fixture(autouse=True)
def _f32(monkeypatch):
    monkeypatch.setenv("GMLX_ACTIVATION_DTYPE", "float32")


def _load(path):
    import gmlx

    return gmlx.load_model(str(path), verbose=False)[:2]


def _logits(model):
    with mx.stream(mx.cpu):
        return np.array(model(mx.array(IDS)).astype(mx.float32))


def _write_adapter(path, pairs: dict, arch: str = ARCH) -> None:
    import gguf

    w = gguf.GGUFWriter(str(path), arch)
    w.add_type(gguf.GGUFType.ADAPTER)
    w.add_string(gguf.Keys.Adapter.TYPE, "lora")
    w.add_float32(gguf.Keys.Adapter.LORA_ALPHA, ALPHA)
    for name, (a, b) in pairs.items():
        w.add_tensor(name + ".lora_a", a)
        w.add_tensor(name + ".lora_b", b)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


@pytest.mark.parametrize("arch", [ARCH, "qwen35"])
def test_an_adapter_on_the_file_names_gives_the_merged_weights(tmp_path, arch):
    from gmlx.load.adapter import apply_gguf_adapter

    v = T.values(arch)
    rng = np.random.default_rng(3)
    merged = dict(v)
    pairs = {}
    for name, parts in PAIRS[arch].items():
        rows = [v[p].shape[0] for p in parts]
        a = (rng.standard_normal((RANK, v[parts[0]].shape[1])) * 0.1).astype(np.float32)
        b = (rng.standard_normal((sum(rows), RANK)) * 0.1).astype(np.float32)
        pairs[name] = (a, b)
        delta = (ALPHA / RANK) * (b @ a)
        for p, d in zip(parts, np.split(delta, np.cumsum(rows)[:-1])):
            merged[p] = v[p] + d
    T.write(tmp_path / "base.gguf", arch, v, fused=True)
    T.write(tmp_path / "merged.gguf", arch, merged, fused=True)
    _write_adapter(tmp_path / "lora.gguf", pairs, arch)

    model, config = _load(tmp_path / "base.gguf")
    base = _logits(model)
    n = apply_gguf_adapter(model, config, str(tmp_path / "lora.gguf"), base_arch=arch)
    assert n == sum(len(p) for p in PAIRS[arch].values())
    want = _logits(_load(tmp_path / "merged.gguf")[0])
    got = _logits(model)
    assert np.abs(want - base).max() > 1e-2
    np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-4)


def test_an_adapter_on_a_repeated_visit_is_refused(tmp_path):
    from gmlx.load.adapter import apply_gguf_adapter

    v = T.values(ARCH)
    T.write(tmp_path / "base.gguf", ARCH, v, fused=True)
    a = np.zeros((RANK, T.FFN), np.float32)
    b = np.zeros((T.H, RANK), np.float32)
    _write_adapter(tmp_path / "lora.gguf", {f"blk.{T.END}.ffn_down.weight": (a, b)})
    model, config = _load(tmp_path / "base.gguf")
    with pytest.raises(ValueError, match="repeated visit"):
        apply_gguf_adapter(model, config, str(tmp_path / "lora.gguf"), base_arch=ARCH)


def _adapted(path):
    from gmlx._vendor.mlx_lm.tuner.lora import LoRALinear
    from gmlx.tune.lora import prepare_lora_student

    model, config = _load(path)
    n = prepare_lora_student(model, rank=RANK, scale=ALPHA / RANK)
    loras = [m for _p, m in model.named_modules() if isinstance(m, LoRALinear)]
    assert n == len(loras)
    assert not any(isinstance(m.linear, LoRALinear) for m in loras)
    return model, config, loras


def test_training_adapts_each_physical_layer_once(tmp_path):
    from mlx.utils import tree_flatten

    T.write(tmp_path / "base.gguf", ARCH, T.values(ARCH), fused=True)
    model, _config, _loras = _adapted(tmp_path / "base.gguf")
    layers = {k.split(".")[2] for k, _ in tree_flatten(model.trainable_parameters())}
    assert layers == {str(i) for i in range(T.LAYERS)}


def test_a_trained_adapter_loads_back_from_the_file_names(tmp_path):
    from gmlx.load.adapter import apply_gguf_adapter, base_tensor_names
    from gmlx.load.headerscan import scan_gguf
    from gmlx.tune.lora import save_trained_adapter

    path = tmp_path / "base.gguf"
    T.write(path, ARCH, T.values(ARCH), fused=True)
    model, config, loras = _adapted(path)
    rng = np.random.default_rng(4)
    for m in loras:
        m.lora_b = mx.array(rng.standard_normal(m.lora_b.shape).astype(np.float32) * 0.05)
    model.eval()
    trained = _logits(model)

    out = tmp_path / "lora.gguf"
    n = save_trained_adapter(model, config, base_arch=ARCH, out_path=str(out),
                             scale=ALPHA / RANK, base_names=base_tensor_names(str(path)))
    assert n == len(loras)
    names = {t.name for t in scan_gguf(str(out)).tensors}
    assert f"blk.{AFTER_BLOCK}.ffn_down.weight.lora_a" in names
    assert "blk.1.attn_qkv_gate.weight.lora_b" in names
    assert "blk.2.ssm_beta_alpha.weight.lora_b" in names
    assert not any(t.startswith(f"blk.{AFTER}.") for t in names)

    fresh, fresh_config = _load(path)
    apply_gguf_adapter(fresh, fresh_config, str(out), base_arch=ARCH)
    np.testing.assert_allclose(_logits(fresh), trained, rtol=1e-4, atol=1e-4)


def test_a_checkpointed_training_step_reaches_the_first_layer(tmp_path):
    import mlx.nn as nn
    from mlx.utils import tree_flatten

    from gmlx.tune.checkpoint import checkpoint_layers

    T.write(tmp_path / "base.gguf", ARCH, T.values(ARCH), fused=True)
    model, _config, loras = _adapted(tmp_path / "base.gguf")
    for m in loras:
        m.lora_b = mx.ones_like(m.lora_b) * 0.01
    model.train()
    assert checkpoint_layers(model) >= 1
    ids = mx.array(IDS)

    def loss(params):
        model.update(params)
        logits = model(ids[:, :-1])
        return nn.losses.cross_entropy(logits, ids[:, 1:]).mean()

    _value, grads = mx.value_and_grad(loss)(model.trainable_parameters())
    flat = dict(tree_flatten(grads))
    first = [g for k, g in flat.items() if k.startswith("model.layers.0.")]
    assert first and all(bool(mx.isfinite(g).all()) for g in first)
    assert any(float(mx.abs(g).max()) > 0 for g in first)
