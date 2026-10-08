"""A qwen3-next-shaped MoE block whose shared expert cannot ride the expert
gathers: the fused block class fuses the router alone. The result must
match the block's own forward, with the same experts picked. Real kq
kernels on the GPU."""
from __future__ import annotations

import os
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

import gmlx.load.modules as modules

if os.environ.get("KQUANT_FORCE_CPU") or not mx.metal.is_available():
    pytest.skip("the fused MoE kernels need a real GPU and a GPU default device",
                allow_module_level=True)

D, INTER, E, K = 256, 256, 64, 6


class _Shell(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.mlp = block


def _block(seed=0):
    """A qwen4exp block: fp32 router, q8_0 expert stacks, bf16 shared
    expert (not quantized, so it cannot be folded into the gathers)."""
    import mlx_kquant as kq
    from mlx_kquant.nn import KQuantSwitchLinear

    from gmlx.models.qwen4_exp.model import SparseMoeBlock

    mx.random.seed(seed)
    rng = np.random.default_rng(seed)
    block = SparseMoeBlock(SimpleNamespace(
        hidden_size=D, moe_intermediate_size=INTER,
        shared_expert_intermediate_size=INTER, norm_topk_prob=True,
        num_experts=E, num_experts_per_tok=K))
    for name, (o, i) in (("gate_proj", (INTER, D)), ("up_proj", (INTER, D)),
                         ("down_proj", (D, INTER))):
        leaf = KQuantSwitchLinear(E, o, i, False, "q8_0")
        w = mx.array((rng.standard_normal((E, o, i)) * 0.05).astype(np.float32))
        leaf.weight, leaf.scales = kq.quantize(w, "q8_0")
        setattr(block.switch_mlp, name, leaf)
    block.shared_expert.set_dtype(mx.bfloat16)
    block.shared_expert_gate.set_dtype(mx.bfloat16)
    block.eval()
    shell = _Shell(block)
    stock = type(block)
    modules.install_fused_moe_glu(shell)
    mx.eval(shell.parameters())
    return shell.mlp, stock


def _x(t, seed=1):
    mx.random.seed(seed)
    return (mx.random.normal((1, t, D)) * 0.5).astype(mx.bfloat16)


@pytest.fixture
def calls(monkeypatch):
    """Counts kq.moe_router_topk calls and keeps the experts it picked."""
    import mlx_kquant as kq

    seen = []
    real = kq.moe_router_topk

    def spy(*a, **kw):
        out = real(*a, **kw)
        seen.append(out[0])
        return out

    monkeypatch.setattr(kq, "moe_router_topk", spy)
    return seen


def _stock_picks(block, x):
    gates = mx.softmax(block.gate(x), axis=-1, precise=True)
    return np.sort(np.array(mx.argpartition(gates, kth=-K, axis=-1)[..., -K:]), -1)


def test_install_puts_the_block_in_router_only_mode():
    block, stock = _block()
    assert type(block).__name__ == "_FusedKQuantMoeBlock"
    assert block._kq_shexp_fold is False
    assert stock.__name__ == "SparseMoeBlock"


@pytest.mark.parametrize("t", [1, 4, 10])
def test_router_only_matches_the_stock_forward(calls, t):
    block, stock = _block()
    x = _x(t)
    got = block(x)
    want = stock.__call__(block, x)
    mx.eval(got, want)
    assert len(calls) == 1
    assert got.dtype == x.dtype and got.shape == x.shape
    picked = np.sort(np.array(calls[0]).reshape(1, t, K), -1)
    assert np.array_equal(picked, _stock_picks(block, x))
    g = np.array(got.astype(mx.float32))
    w = np.array(want.astype(mx.float32))
    # one bf16 rounding of the mix and of the shared-gate product
    assert np.abs(g - w).max() <= 2e-2 * np.abs(w).max()


def test_wide_call_takes_the_stock_forward(calls):
    block, stock = _block()
    x = _x(11)  # 11 * K >= 64: the prefill width
    got = block(x)
    want = stock.__call__(block, x)
    mx.eval(got, want)
    assert calls == []
    assert mx.array_equal(got, want).item()


def test_router_env_off_takes_the_stock_forward(calls, monkeypatch):
    block, stock = _block()
    monkeypatch.setattr(modules, "_FUSED_MOE_ROUTER_ENABLED", False)
    x = _x(1)
    got = block(x)
    want = stock.__call__(block, x)
    mx.eval(got, want)
    assert calls == []
    assert mx.array_equal(got, want).item()


def test_control_fallback_keeps_the_activation_dtype():
    """The expert-controls forward on a block with an fp32 router returns
    the activation dtype."""
    from gmlx.stream.moe_experts import qwen3_next_moe_forward

    block, _ = _block()
    assert block.gate.weight.dtype == mx.float32
    x = _x(12)
    assert qwen3_next_moe_forward(block, x).dtype == x.dtype
