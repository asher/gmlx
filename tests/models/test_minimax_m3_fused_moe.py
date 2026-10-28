#!/usr/bin/env python3
"""minimax-m3 fused MoE: SwiGLUOAI maps to the kq act swiglu_oai behind a
caps check, and the ungated shared expert folds into the same gathers. Both
are on by default; GMLX_FUSED_MOE_SWIGLU_OAI=0 and GMLX_FUSED_MOE_SHEXP_OAI=0
turn them off. The block
adds the shared expert once on every path. The output tests run the real
kernels on q4_k experts with a q5_k shared expert."""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

import mlx_kquant as kq
from helpers import _real_apple_gpu

import gmlx.load.modules as modules
from gmlx.load.modules import install_fused_moe_glu, install_hyv3_shexp_fold
from gmlx.models import minimax_m3 as m3

D, INTER = 256, 256


class _Shell(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.mlp = block


def _args(limit=1.0):
    return m3.ModelArgs(
        model_type="minimax_m3", hidden_size=D, intermediate_size=INTER,
        dense_intermediate_size=INTER, shared_intermediate_size=INTER,
        num_attention_heads=4, num_key_value_heads=2, num_hidden_layers=2,
        num_local_experts=8, num_experts_per_tok=2, rms_norm_eps=1e-6,
        rope_theta=5e6, rotary_dim=32, vocab_size=64, head_dim=64,
        mlp_layer_types=["dense", "sparse"], swiglu_limit=limit,
        routed_scaling_factor=2.0)


def _block(codec="q4_k", shexp_codec="q5_k", weights=False, limit=1.0):
    """An M3 MoE block on KQuant projections: zeros by default (the
    eligibility reads shapes and codecs only), or random weights quantized
    to the codecs. limit 1.0 so the clamps engage at unit-scale inputs."""
    from mlx_kquant.nn import KQuantLinear, KQuantSwitchLinear

    args = _args(limit)
    blk = m3.MiniMaxM3SparseMoeBlock(args)
    rng = np.random.default_rng(5)

    def wire(shape, c):
        w = rng.standard_normal(shape).astype(np.float32) / np.sqrt(shape[-1])
        q = kq.quantize(mx.array(w).reshape(-1, shape[-1]), c)[0]
        return q.reshape(*shape[:-1], q.shape[-1])

    e = args.num_local_experts
    for name, (o, i) in (("gate_proj", (INTER, D)), ("up_proj", (INTER, D)),
                         ("down_proj", (D, INTER))):
        m = KQuantSwitchLinear(e, o, i, False, codec)
        if weights:
            m.weight = wire((e, o, i), codec)
        setattr(blk.switch_mlp, name, m)
    for name, (o, i) in (("gate_proj", (INTER, D)), ("up_proj", (INTER, D)),
                         ("down_proj", (D, INTER))):
        m = KQuantLinear(i, o, False, shexp_codec)
        if weights:
            m.weight = wire((o, i), shexp_codec)
        setattr(blk.shared_experts, name, m)
    if weights:
        blk.gate.weight = mx.array(
            rng.standard_normal((e, D)).astype(np.float32) / np.sqrt(D))
    blk.eval()
    return blk


def _installed(blk):
    shell = _Shell(blk)
    return install_fused_moe_glu(shell), install_hyv3_shexp_fold(shell)


def _caps():
    return modules._FusedMoeCaps()


needs_oai = pytest.mark.skipif(
    not (_caps().has_base and _caps().has_swiglu_oai),
    reason="installed mlx-kquant has no swiglu_oai")
needs_shexp_oai = pytest.mark.skipif(
    not (_caps().has_base and _caps().has_shexp_oai),
    reason="installed mlx-kquant has no swiglu_oai shared-expert fold")
gpu = pytest.mark.skipif(not _real_apple_gpu(), reason="GPU kernels")


@pytest.fixture
def levers(monkeypatch):
    def set_(oai="1", shexp="0"):
        monkeypatch.setenv("GMLX_FUSED_MOE_SWIGLU_OAI", oai)
        monkeypatch.setenv("GMLX_FUSED_MOE_SHEXP_OAI", shexp)
    return set_


# install: act kind and stamping


@needs_shexp_oai
def test_unset_levers_fuse_the_stack_and_fold_the_shared_expert(monkeypatch):
    monkeypatch.delenv("GMLX_FUSED_MOE_SWIGLU_OAI", raising=False)
    monkeypatch.delenv("GMLX_FUSED_MOE_SHEXP_OAI", raising=False)
    blk = _block()
    assert _installed(blk) == (1, 1)
    assert blk.switch_mlp._kq_glu_act == "swiglu_oai"
    assert blk.switch_mlp._kq_shexp_mod is blk.shared_experts


@needs_oai
def test_lever_off_keeps_the_stack_unfused(levers):
    levers(oai="0")
    blk = _block()
    assert _installed(blk) == (0, 0)
    assert getattr(blk.switch_mlp, "_kq_glu_act", None) is None


@needs_oai
def test_lever_on_maps_swiglu_oai_with_its_constants(levers):
    levers(oai="1")
    blk = _block(limit=1.0)
    assert _installed(blk) == (1, 0)
    sw = blk.switch_mlp
    assert sw._kq_glu_act == "swiglu_oai"
    assert sw._kq_glu_alpha == pytest.approx(1.702)
    assert sw._kq_glu_limit == 1.0


@needs_oai
def test_kq_without_the_act_keeps_the_stack_unfused(levers, monkeypatch):
    levers(oai="1", shexp="1")
    base = modules._FusedMoeCaps

    class _Caps(base):
        def __init__(self):
            super().__init__()
            self.has_swiglu_oai = False  # an older mlx-kquant
            self.has_shexp_oai = False

    monkeypatch.setattr(modules, "_FusedMoeCaps", _Caps)
    blk = _block()
    assert _installed(blk) == (0, 0)


@needs_shexp_oai
def test_fold_stamps_only_with_its_lever_and_matching_constants(levers):
    levers(oai="1", shexp="0")
    blk = _block()
    assert _installed(blk) == (1, 0)
    levers(oai="1", shexp="1")
    blk = _block()
    assert _installed(blk) == (1, 1)
    assert blk.switch_mlp._kq_shexp_mod is blk.shared_experts
    blk = _block()
    blk.shared_experts.limit = 7.0  # the kernel runs every slot alike
    assert _installed(blk) == (1, 0)


# real kernels


def _x(t, seed=9):
    x = np.random.default_rng(seed).standard_normal((1, t, D)).astype(np.float32)
    return mx.array(x).astype(mx.bfloat16)


def _rel(got, want):
    return float(mx.abs(got - want).max() / mx.abs(want).max())


@needs_oai
@gpu
@pytest.mark.parametrize("t", [1, 3])
def test_fused_routed_experts_match_unfused(levers, monkeypatch, t):
    levers(oai="1", shexp="0")
    blk = _block(weights=True)
    x = _x(t)
    want = blk(x).astype(mx.float32)
    assert _installed(blk) == (1, 0)
    seen = []
    real = kq.moe_glu_gather_kq

    def spy(*a, **k):
        seen.append(k)
        return real(*a, **k)

    monkeypatch.setattr(kq, "moe_glu_gather_kq", spy)
    got = blk(x).astype(mx.float32)
    assert [k["act"] for k in seen] == ["swiglu_oai"]
    assert seen[0]["limit"] == 1.0 and seen[0]["alpha"] == pytest.approx(1.702)
    assert _rel(got, want) < 2e-2


@needs_shexp_oai
@gpu
@pytest.mark.parametrize("t", [1, 3])
def test_folded_shared_expert_is_added_once(levers, monkeypatch, t):
    """With the fold the gather's last slot is the shared expert and the
    block returns the mixed result as is: the shared MLP never runs, and a
    second add would put the output a whole shared expert off."""
    levers(oai="1", shexp="1")
    blk = _block(weights=True)
    x = _x(t)
    want = blk(x).astype(mx.float32)
    shared = blk.shared_experts(x).astype(mx.float32)
    assert _installed(blk) == (1, 1)
    assert blk.switch_mlp._kq_shexp_mod is blk.shared_experts
    seen, mlp_calls = [], []
    real = kq.moe_glu_gather_shexp_kq

    def spy(*a, **k):
        h = real(*a, **k)
        seen.append((h.shape, k))
        return h

    monkeypatch.setattr(kq, "moe_glu_gather_shexp_kq", spy)
    orig = m3.MiniMaxM3MLP.__call__
    monkeypatch.setattr(
        m3.MiniMaxM3MLP, "__call__",
        lambda self, v: mlp_calls.append(1) or orig(self, v))
    got = blk(x).astype(mx.float32)
    assert len(seen) == 1
    shape, kw = seen[0]
    assert shape == (t, 3, INTER)  # two routed slots and the shared one
    assert kw["act"] == "swiglu_oai" and kw["shexp_kquant_type"] == "q5_k"
    assert kw["limit"] == 1.0 and kw["alpha"] == pytest.approx(1.702)
    assert mlp_calls == []
    assert _rel(got, want) < 2e-2
    assert _rel(got + shared, want) > 0.2


@needs_shexp_oai
@gpu
def test_folded_shared_expert_is_added_once_at_prefill_width(levers, monkeypatch):
    """At 64 or more index rows the sorted prefill mixes the routed slots
    and adds the stamped shared expert itself; the block must not add it
    again."""
    levers(oai="1", shexp="1")
    blk = _block(weights=True)
    x = _x(40)  # 80 index rows at top-2
    want = blk(x).astype(mx.float32)
    shared = blk.shared_experts(x).astype(mx.float32)
    assert _installed(blk) == (1, 1)
    sorted_calls = []
    cls = type(blk.switch_mlp)
    real = cls._kq_mix_sorted
    monkeypatch.setattr(
        cls, "_kq_mix_sorted",
        lambda self, *a: sorted_calls.append(1) or real(self, *a))
    got = blk(x).astype(mx.float32)
    assert sorted_calls
    assert _rel(got, want) < 2e-2
    assert _rel(got + shared, want) > 0.2
