#!/usr/bin/env python3
"""LoRA train -> GGUF save fidelity (train T2): the extraction + orientation
transpose + q/k forward-permute + scale chain must reconstruct an mlx-lm
``LoRALinear``'s exact delta after a save -> ``load_lora_adapter`` round-trip.
CPU-only - hand-built LoRALinear layers, no training, no kernels."""
from __future__ import annotations

import mlx.core as mx
import pytest

pytest.importorskip("gguf")
import mlx.nn as nn  # noqa: E402

from gmlx import adapter, train  # noqa: E402
from gmlx.transforms import qk_permute_wire  # noqa: E402

LoRALinear = pytest.importorskip("mlx_lm.tuner.lora").LoRALinear

N_HEAD, N_HEAD_KV, HEAD_DIM = 4, 2, 8
IN, R, S = 16, 4, 2.0
Q_OUT, K_OUT = N_HEAD * HEAD_DIM, N_HEAD_KV * HEAD_DIM   # 32, 16
CONFIG = {"num_attention_heads": N_HEAD, "num_key_value_heads": N_HEAD_KV,
          "num_hidden_layers": 1}


def _lora(out, *, seed):
    mx.random.seed(seed)
    ll = LoRALinear(input_dims=IN, output_dims=out, r=R, scale=S)
    ll.lora_a = mx.random.normal((IN, R))   # lora_b inits to zero -> a real delta
    ll.lora_b = mx.random.normal((R, out))
    return ll


class _Attn(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = _lora(Q_OUT, seed=1)
        self.k_proj = _lora(K_OUT, seed=2)


class _Mlp(nn.Module):
    def __init__(self):
        super().__init__()
        self.down_proj = _lora(IN, seed=3)


class _Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = _Attn()
        self.mlp = _Mlp()


class _Inner(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = [_Layer()]


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _Inner()


def _delta(ll, x):
    """The LoRALinear's adapter contribution alone (forward minus the base)."""
    return ll(x) - ll.linear(x)


def test_trained_lora_roundtrips_to_gguf_delta(tmp_path):
    model = _Model()
    model.freeze()
    model.apply_to_modules(
        lambda _k, m: m.unfreeze(keys=["lora_a", "lora_b"], recurse=False)
        if isinstance(m, LoRALinear) else None)

    out = str(tmp_path / "trained.gguf")
    n = train.save_trained_adapter(model, CONFIG, base_arch="llama",
                                   out_path=out, rank=R, scale=S)
    assert n == 3

    plan = adapter.load_lora_adapter(out)
    assert plan.alpha == pytest.approx(S * R)
    assert set(plan.modules) == {
        "model.layers.0.self_attn.q_proj",
        "model.layers.0.self_attn.k_proj",
        "model.layers.0.mlp.down_proj",
    }

    layer = model.model.layers[0]
    cases = {
        "model.layers.0.self_attn.q_proj": (layer.self_attn.q_proj, N_HEAD),
        "model.layers.0.self_attn.k_proj": (layer.self_attn.k_proj, N_HEAD_KV),
        "model.layers.0.mlp.down_proj": (layer.mlp.down_proj, None),
    }
    x = mx.random.normal((3, IN))
    for path, (ll, nh) in cases.items():
        lm = plan.modules[path]
        assert lm.scale == pytest.approx(S)           # alpha/rank == trained scale
        b = qk_permute_wire(lm.b, nh) if lm.transform == "qk_permute" else lm.b
        recon = lm.scale * (x @ lm.a.T) @ b.T         # the loader+install forward
        assert mx.allclose(recon, _delta(ll, x), atol=1e-5)


def test_no_lora_layers_raises(tmp_path):
    class Plain(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(IN, IN)

    with pytest.raises(ValueError, match="no trained LoRA layers"):
        train.save_trained_adapter(Plain(), CONFIG, base_arch="llama",
                                   out_path=str(tmp_path / "x.gguf"),
                                   rank=R, scale=S)
