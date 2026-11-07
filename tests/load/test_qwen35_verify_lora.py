"""The owned qwen3.5 MoE block calls its expert leaves directly at verify
widths. An expert adapter must reach that path: a verify block of three
tokens matches the same tokens decoded one at a time with the adapter on,
and equals the unadapted verify output with every slot off."""
from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from gmlx import lora_rows
import gmlx.load.adapter as adapter
import gmlx.load.modules as modules
import gmlx.models.qwen35.layers as qwen35_layers

D, INTER, E, K, T = 256, 256, 8, 2, 3


@pytest.fixture(autouse=True)
def _static():
    lora_rows.configure("static", 1)
    yield
    lora_rows.configure("static", 1)


class _Shell(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.mlp = block


def _shell(seed=0):
    import mlx_kquant as kq
    from mlx_kquant.nn import KQuantSwitchLinear

    block_cls, _ = qwen35_layers.moe_layer_classes()
    mx.random.seed(seed)
    rng = np.random.default_rng(seed)
    block = block_cls(SimpleNamespace(
        hidden_size=D, moe_intermediate_size=INTER,
        shared_expert_intermediate_size=INTER, num_experts=E,
        num_experts_per_tok=K))
    for name, (o, i) in (("gate_proj", (INTER, D)), ("up_proj", (INTER, D)),
                         ("down_proj", (D, INTER))):
        leaf = KQuantSwitchLinear(E, o, i, False, "q8_0")
        w = mx.array((rng.standard_normal((E, o, i)) * 0.05).astype(np.float32))
        leaf.weight, leaf.scales = kq.quantize(w, "q8_0")
        setattr(block.switch_mlp, name, leaf)
    block.eval()
    shell = _Shell(block)
    mx.eval(shell.parameters())
    return shell


def _install(shell, r=4):
    rng = np.random.default_rng(1)
    path = "mlp.switch_mlp.down_proj"
    a = mx.array((rng.standard_normal((E, r, INTER)) * 0.5).astype(np.float32))
    b = mx.array((rng.standard_normal((E, D, r)) * 0.5).astype(np.float32))
    plan = adapter.LoraAdapter(alpha=float(r), arch="qwen35moe", modules={
        path: adapter.LoraModule(module_path=path, a=a, b=b, rank=r, scale=1.0,
                                 experts=True)})
    assert modules.install_lora_adapter(shell, plan) == 1


def _run(reg, fn):
    with lora_rows.bind(reg), lora_rows.published([1]):
        y = fn()
        mx.eval(y)
    return y


def _rel(a, b):
    a, b = np.array(a.astype(mx.float32)), np.array(b.astype(mx.float32))
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def _x():
    rng = np.random.default_rng(2)
    return mx.array(rng.standard_normal((1, T, D)).astype(np.float32)).astype(mx.bfloat16)


def test_verify_block_applies_the_expert_adapter():
    shell = _shell()
    x = _x()
    lora_rows.configure("rows", 1)
    reg = lora_rows.UidRegistry()
    reg.register(1, (1.0,))
    base = _run(reg, lambda: shell.mlp(x, target_verify=True))
    _install(shell)
    got = _run(reg, lambda: shell.mlp(x, target_verify=True))
    steps = mx.concatenate(
        [_run(reg, lambda i=i: shell.mlp(x[:, i:i + 1], target_verify=False))
         for i in range(T)], axis=1)
    assert _rel(base, steps) > 0.2
    assert _rel(got, steps) < 0.03


def test_verify_block_with_every_slot_off_equals_the_unadapted_block():
    shell = _shell()
    x = _x()
    lora_rows.configure("rows", 1)
    reg = lora_rows.UidRegistry()
    reg.register(1, (0.0,))
    base = _run(reg, lambda: shell.mlp(x, target_verify=True))
    _install(shell)
    got = _run(reg, lambda: shell.mlp(x, target_verify=True))
    assert mx.array_equal(got, base)
