"""A qwen3-next-shaped MoE block from a low-bit file: the shared expert's
gate, up and down tensors each carry their own codec, none of them the
experts'. The fused block folds the shared expert into the gathers. On a
kq build with no slot-parallel folded down gather it fuses the router
alone at widths 2 and 3. Both must match the block's own forward. Real kq
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

kq = pytest.importorskip("mlx_kquant")
if not hasattr(kq, "shexp_glu_combo_has_kernel"):
    pytest.skip("this mlx-kquant build has no split-codec shared-expert kernels",
                allow_module_level=True)

D, INTER, E, K = 256, 256, 64, 6


class _Shell(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.mlp = block


def _block(sgate="iq4_xs", sup="iq3_s", sdown="iq4_nl", seed=0):
    """iq2_s expert gate and up stacks with a q2_0 down stack, and a
    shared expert in three other codecs."""
    from mlx_kquant.nn import KQuantLinear, KQuantSwitchLinear

    from gmlx.models.qwen4_exp.model import SparseMoeBlock

    mx.random.seed(seed)
    rng = np.random.default_rng(seed)
    block = SparseMoeBlock(SimpleNamespace(
        hidden_size=D, moe_intermediate_size=INTER,
        shared_expert_intermediate_size=INTER, norm_topk_prob=True,
        num_experts=E, num_experts_per_tok=K))
    for name, (o, i), codec, scodec in (
            ("gate_proj", (INTER, D), "iq2_s", sgate),
            ("up_proj", (INTER, D), "iq2_s", sup),
            ("down_proj", (D, INTER), "q2_0", sdown)):
        leaf = KQuantSwitchLinear(E, o, i, False, codec)
        w = mx.array((rng.standard_normal((E, o, i)) * 0.05).astype(np.float32))
        leaf.weight, leaf.scales = kq.quantize(w, codec)
        setattr(block.switch_mlp, name, leaf)
        dense = KQuantLinear(i, o, False, scodec)
        w = mx.array((rng.standard_normal((o, i)) * 0.05).astype(np.float32))
        dense.weight, dense.scales = kq.quantize(w, scodec)
        setattr(block.shared_expert, name, dense)
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


def _rel(a, b):
    a, b = a.astype(mx.float32), b.astype(mx.float32)
    return (mx.abs(a - b).max() / mx.abs(b).max()).item()


def _calls(monkeypatch, name):
    seen = []
    real = getattr(kq, name)

    def spy(*a, **kw):
        seen.append(kw)
        return real(*a, **kw)

    monkeypatch.setattr(kq, name, spy)
    return seen


_HAS_MIX_SP = hasattr(kq, "shexp_mix_slot_parallel")
_FOLD_T = [1, 2, 3, 4, 6] if _HAS_MIX_SP else [1, 4, 6]


def _no_mix_sp(monkeypatch):
    """A kq build with no slot-parallel folded down gather."""
    monkeypatch.delattr(kq, "shexp_mix_slot_parallel", raising=False)


def test_block_folds_with_a_skip_at_widths_two_and_three(monkeypatch):
    _no_mix_sp(monkeypatch)
    block, stock = _block()
    assert type(block).__name__ == "_FusedKQuantMoeBlock"
    assert getattr(block, "_kq_shexp_fold", True) is True
    assert block._kq_fold_skip_t == (2, 3)


@pytest.mark.skipif(not _HAS_MIX_SP, reason="needs kq.shexp_mix_slot_parallel")
def test_block_folds_at_every_width_on_a_slot_parallel_down_gather():
    assert kq.shexp_mix_slot_parallel("q2_0", "iq4_nl")
    block, stock = _block()
    assert getattr(block, "_kq_shexp_fold", True) is True
    assert not hasattr(block, "_kq_fold_skip_t")


@pytest.mark.parametrize("t", _FOLD_T)
def test_fold_matches_the_stock_forward(t, monkeypatch):
    block, stock = _block()
    x = _x(t)
    want = stock.__call__(block, x)
    glu = _calls(monkeypatch, "moe_glu_gather_shexp_kq")
    mix = _calls(monkeypatch, "gather_qmv_mix_kq")
    got = block(x)
    mx.eval(got, want)
    assert got.dtype == x.dtype and got.shape == x.shape
    assert glu == [{"act": "silu", "shexp_kquant_type": "iq4_xs",
                    "shexp_up_kquant_type": "iq3_s"}]
    assert mix == [{"shexp_kquant_type": "iq4_nl"}]
    assert _rel(got, want) < 2e-2


@pytest.mark.parametrize("t", [2, 3])
def test_widths_two_and_three_fuse_the_router_alone(t, monkeypatch):
    _no_mix_sp(monkeypatch)
    block, stock = _block()
    x = _x(t)
    want = stock.__call__(block, x)
    glu = _calls(monkeypatch, "moe_glu_gather_shexp_kq")
    router = _calls(monkeypatch, "moe_router_topk")
    got = block(x)
    mx.eval(got, want)
    assert glu == [] and len(router) == 1
    assert got.dtype == x.dtype
    assert _rel(got, want) < 2e-2


def test_one_codec_for_gate_and_up_folds_at_every_width(monkeypatch):
    """A shared gate and up in one upcast codec ride the gather that
    shares experts between rows, so no width is skipped."""
    block, stock = _block(sgate="q6_k", sup="q6_k", sdown="q8_0")
    assert not hasattr(block, "_kq_fold_skip_t")
    x = _x(2)
    want = stock.__call__(block, x)
    glu = _calls(monkeypatch, "moe_glu_gather_shexp_kq")
    got = block(x)
    mx.eval(got, want)
    assert glu == [{"act": "silu", "shexp_kquant_type": "q6_k"}]
    assert _rel(got, want) < 2e-2


def test_a_pair_with_no_kernel_fuses_the_router_alone():
    block, stock = _block(sgate="iq4_xs", sup="iq4_nl", sdown="iq4_nl")
    assert type(block).__name__ == "_FusedKQuantMoeBlock"
    assert block._kq_shexp_fold is False
    x = _x(1)
    assert _rel(block(x), stock.__call__(block, x)) < 2e-2
