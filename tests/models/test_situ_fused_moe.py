"""situ experts on the fused MoE gathers.

SituActivation (Kimi K3, Inkling 2) maps to the kq act situ when the
installed mlx-kquant has it. GMLX_FUSED_MOE_SITU=0 keeps the stock path.
The output tests run the real kernels on q4_k experts in an Inkling 2 MoE
block."""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

import mlx_kquant as kq
from helpers import _real_apple_gpu

import gmlx.load.modules as modules
from gmlx.load.modules import install_fused_moe_glu
from gmlx.models.inkling2 import model as ink

D, INTER, E = 512, 256, 8


class _Shell(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.mlp = block


def _block(codec="q4_k", weights=False, beta=4.0, lb=25.0):
    from mlx_kquant.nn import KQuantSwitchLinear

    args = ink.ModelArgs(
        model_type="inkling2", hidden_size=D, num_hidden_layers=1,
        num_attention_heads=4, num_key_value_heads=2, head_dim=128,
        vocab_size=64, unpadded_vocab_size=64, layer_types=["full"],
        num_experts=E, num_experts_per_tok=2, moe_intermediate_size=INTER,
        expert_weights_scale=8.0, situ_beta=beta, situ_linear_beta=lb)
    blk = ink.MoE(args)
    rng = np.random.default_rng(5)
    for name, (o, i) in (("gate_proj", (INTER, D)), ("up_proj", (INTER, D)),
                         ("down_proj", (D, INTER))):
        m = KQuantSwitchLinear(E, o, i, False, codec)
        if weights:
            w = rng.standard_normal((E, o, i)).astype(np.float32)
            # gate rows wide enough that tanh(g / beta) bends
            w *= (6.0 if name == "gate_proj" else 1.0) / np.sqrt(i)
            q = kq.quantize(mx.array(w).reshape(-1, i), codec)[0]
            m.weight = q.reshape(E, o, q.shape[-1])
        setattr(blk.switch_mlp, name, m)
    if weights:
        blk.gate.weight = mx.array(
            rng.standard_normal((E, D)).astype(np.float32) / np.sqrt(D))
    blk.eval()
    return blk


def _caps():
    return modules._FusedMoeCaps()


needs_situ = pytest.mark.skipif(
    not (_caps().has_base and _caps().has_situ),
    reason="installed mlx-kquant has no situ act")
gpu = pytest.mark.skipif(not _real_apple_gpu(), reason="GPU kernels")


def test_kq_without_the_act_keeps_the_stack_unfused(monkeypatch):
    base = modules._FusedMoeCaps

    class _Caps(base):
        def __init__(self):
            super().__init__()
            self.has_situ = False  # an older mlx-kquant

    monkeypatch.setattr(modules, "_FusedMoeCaps", _Caps)
    blk = _block()
    assert install_fused_moe_glu(_Shell(blk)) == 0
    assert getattr(blk.switch_mlp, "_kq_glu_act", None) is None


@needs_situ
def test_situ_maps_with_its_constants(monkeypatch):
    monkeypatch.delenv("GMLX_FUSED_MOE_SITU", raising=False)
    blk = _block(lb=-1.0)
    assert install_fused_moe_glu(_Shell(blk)) == 1
    sw = blk.switch_mlp
    assert sw._kq_glu_act == "situ"
    assert sw._kq_glu_limit == 4.0
    assert sw._kq_glu_alpha == 0.0  # an unbounded up branch


@needs_situ
def test_lever_off_keeps_the_stock_path(monkeypatch):
    monkeypatch.setenv("GMLX_FUSED_MOE_SITU", "0")
    blk = _block()
    assert install_fused_moe_glu(_Shell(blk)) == 0


def _x(t, seed=9):
    x = np.random.default_rng(seed).standard_normal((1, t, D))
    return mx.array(x.astype(np.float32)).astype(mx.bfloat16)


def _rel(got, want):
    return float(mx.abs(got - want).max() / mx.abs(want).max())


@needs_situ
@gpu
@pytest.mark.parametrize("t", [1, 4])
@pytest.mark.parametrize("lb", [25.0, 0.5, 0.0])
def test_fused_situ_matches_unfused(monkeypatch, t, lb):
    monkeypatch.delenv("GMLX_FUSED_MOE_SITU", raising=False)
    blk = _block(weights=True, lb=lb)
    x = _x(t)
    want = blk(x).astype(mx.float32)
    assert install_fused_moe_glu(_Shell(blk)) == 1
    seen = []
    real = kq.moe_glu_gather_kq

    def spy(*a, **k):
        seen.append(k)
        return real(*a, **k)

    monkeypatch.setattr(kq, "moe_glu_gather_kq", spy)
    got = blk(x).astype(mx.float32)
    assert [k["act"] for k in seen] == ["situ"]
    assert seen[0]["limit"] == 4.0 and seen[0]["alpha"] == lb
    assert _rel(got, want) < 2e-2
