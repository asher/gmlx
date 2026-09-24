"""An adapter on a projection that a fused MoE decode path reads in place of
calling it: the qwen3-next fused block takes the stock forward, the hy_v3
shared-expert fold drops its stamp and leaves the shared expert to the
caller, and a router concat built before the install is rebuilt. Each
fused result must match the stock forward with the same adapter. Real kq
kernels on the GPU."""
from __future__ import annotations

import os
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from gmlx import lora_rows
import gmlx.load.adapter as adapter
import gmlx.load.modules as modules

if os.environ.get("KQUANT_FORCE_CPU") or not mx.metal.is_available():
    pytest.skip("the fused MoE kernels need a real GPU and a GPU default device",
                allow_module_level=True)

D, INTER, E, K = 256, 256, 4, 2


@pytest.fixture(autouse=True)
def _static():
    lora_rows.configure("static", 1)
    yield
    lora_rows.configure("static", 1)


class _Shell(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.mlp = block


def _quantized(leaf, shape, rng):
    import mlx_kquant as kq

    w = mx.array((rng.standard_normal(shape) * 0.05).astype(np.float32))
    leaf.weight, leaf.scales = kq.quantize(w, "q8_0")
    return leaf


def _quantize_glu(glu, rng):
    from mlx_kquant.nn import KQuantSwitchLinear

    for name, (o, i) in (("gate_proj", (INTER, D)), ("up_proj", (INTER, D)),
                         ("down_proj", (D, INTER))):
        setattr(glu, name, _quantized(
            KQuantSwitchLinear(E, o, i, False, "q8_0"), (E, o, i), rng))


def _quantize_mlp(mlp, rng):
    from mlx_kquant.nn import KQuantLinear

    for name, (o, i) in (("gate_proj", (INTER, D)), ("up_proj", (INTER, D)),
                         ("down_proj", (D, INTER))):
        setattr(mlp, name, _quantized(KQuantLinear(i, o, False, "q8_0"), (o, i), rng))


def _qwen3next_shell(seed=0):
    from mlx_lm.models.qwen3_next import Qwen3NextSparseMoeBlock

    mx.random.seed(seed)
    rng = np.random.default_rng(seed)
    block = Qwen3NextSparseMoeBlock(SimpleNamespace(
        hidden_size=D, moe_intermediate_size=INTER,
        shared_expert_intermediate_size=INTER, norm_topk_prob=True,
        num_experts=E, num_experts_per_tok=K))
    _quantize_glu(block.switch_mlp, rng)
    _quantize_mlp(block.shared_expert, rng)
    block.eval()
    shell = _Shell(block)
    modules.install_fused_moe_glu(shell)
    if type(shell.mlp).__name__ != "_FusedKQuantMoeBlock":
        pytest.skip("the fused MoE block is not installable in this build")
    mx.eval(shell.parameters())
    return shell


def _hyv3_shell(seed=0):
    from gmlx.models.hy_v3.model import MoE

    mx.random.seed(seed)
    rng = np.random.default_rng(seed)
    block = MoE(SimpleNamespace(
        hidden_size=D, expert_hidden_dim=INTER, num_experts=E,
        num_experts_per_tok=K, num_shared_experts=1, route_norm=True,
        router_scaling_factor=1.5, enable_moe_fp32_combine=False))
    _quantize_glu(block.switch_mlp, rng)
    _quantize_mlp(block.shared_mlp, rng)
    block.eval()
    shell = _Shell(block)
    modules.install_fused_moe_glu(shell)
    if modules.install_hyv3_shexp_fold(shell) != 1:
        pytest.skip("the shared-expert fold is not installable in this build")
    mx.eval(shell.parameters())
    return shell


def _install(shell, paths, seed=1, r=4):
    """A dense LoRA pair per path, large enough to move the block output."""
    rng = np.random.default_rng(seed)
    mods = {}
    for path in paths:
        leaf = dict(shell.named_modules())[path]
        out, d_in = ((leaf.weight.shape[0], D) if hasattr(leaf, "kquant_type")
                     else leaf.weight.shape)
        a = mx.array((rng.standard_normal((r, d_in)) * 0.5).astype(np.float32))
        b = mx.array((rng.standard_normal((out, r)) * 0.5).astype(np.float32))
        mods[path] = adapter.LoraModule(module_path=path, a=a, b=b, rank=r, scale=1.0)
    plan = adapter.LoraAdapter(alpha=float(r), arch="qwen35moe", modules=mods)
    assert modules.install_lora_adapter(shell, plan) == len(paths)


def _x(seed=2):
    return mx.array(np.random.default_rng(seed).standard_normal((1, 1, D)).astype(np.float32)).astype(mx.bfloat16)


def _rel(a, b):
    a = np.array(a.astype(mx.float32))
    b = np.array(b.astype(mx.float32))
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def _stock(shell, x, monkeypatch):
    with monkeypatch.context() as m:
        m.setattr(modules, "_FUSED_MOE_ENABLED", False)
        y = shell.mlp(x)
        mx.eval(y)
    return y


def test_adapted_shared_expert_reaches_the_fused_block_output(monkeypatch):
    shell = _qwen3next_shell()
    x = _x()
    base = shell.mlp(x)
    _install(shell, ["mlp.shared_expert.gate_proj", "mlp.shared_expert.down_proj"])
    y = shell.mlp(x)
    mx.eval(base, y)
    ref = _stock(shell, x, monkeypatch)
    assert _rel(base, ref) > 0.2
    assert _rel(y, ref) < 0.03


def test_router_adapter_installed_after_a_decode_reaches_the_routing(monkeypatch):
    shell = _qwen3next_shell()
    x = _x()
    base = shell.mlp(x)
    mx.eval(base)
    if not isinstance(vars(shell.mlp).get("_kq_router_cat"), mx.array):
        pytest.skip("this build does not fuse the router")
    _install(shell, ["mlp.gate", "mlp.shared_expert_gate"])
    y = shell.mlp(x)
    mx.eval(y)
    with monkeypatch.context() as m:
        m.setattr(modules, "_FUSED_MOE_BLOCK_ENABLED", False)
        ref = shell.mlp(x)
        mx.eval(ref)
    assert _rel(base, ref) > 0.2
    assert _rel(y, ref) < 0.03


def test_a_trained_student_decodes_through_the_fused_block(monkeypatch):
    from mlx_lm.tuner.lora import LoRALinear

    from gmlx.tune.lora import prepare_lora_student

    shell = _qwen3next_shell()

    class _Model(nn.Module):
        def __init__(self, layer):
            super().__init__()
            self.layers = [layer]

    model = _Model(shell)
    assert prepare_lora_student(model, rank=4, scale=1.0) > 0
    adapted = [m for _, m in model.named_modules() if isinstance(m, LoRALinear)]
    assert any(m is shell.mlp.shared_expert.gate_proj for m in adapted)
    rng = np.random.default_rng(3)
    for m in adapted:
        m.lora_b = mx.array((rng.standard_normal(m.lora_b.shape) * 0.5).astype(np.float32))
    model.eval()
    x = _x()
    y = shell.mlp(x)
    mx.eval(y)
    assert _rel(y, _stock(shell, x, monkeypatch)) < 0.03


def test_adapted_shared_expert_drops_the_fold_and_reaches_the_output(monkeypatch):
    shell = _hyv3_shell()
    x = _x()
    base = shell.mlp(x)
    mx.eval(base)
    _install(shell, ["mlp.shared_mlp.gate_proj", "mlp.shared_mlp.down_proj"])
    y = shell.mlp(x)
    mx.eval(y)
    assert shell.mlp.switch_mlp._kq_shexp_mod is None
    ref = _stock(shell, x, monkeypatch)
    assert _rel(base, ref) > 0.2
    assert _rel(y, ref) < 0.03
