#!/usr/bin/env python3
"""deepseek-v4 shared-expert fold: install_hyv3_shexp_fold stamps eligible
V4 MoE blocks whose LimitedSwiGLU shared expert matches the stack's
silu_limit, the fused decode branch hands the kernels the clamp, and the
V4 and V4.1 blocks add the shared expert exactly once on every path.
CPU-safe: the shexp kernels are monkeypatched fakes."""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

import gmlx.load.modules as modules
from gmlx.load.modules import install_fused_moe_glu, install_hyv3_shexp_fold
from gmlx.models.deepseek_v4 import model as v4
from gmlx.models.deepseek_v41 import model as v41

D, INTER = 256, 256


class _Shell(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.mlp = block


def _args(experts=4, top_k=2, limit=10.0):
    return v4.ModelArgs(
        hidden_size=D,
        moe_intermediate_size=INTER,
        n_routed_experts=experts,
        num_experts_per_tok=top_k,
        n_shared_experts=1,
        num_hidden_layers=4,
        vocab_size=64,
        swiglu_limit=limit,
    )


def _kq_moe(codec="iq2_xxs", shexp_codec="q8_0", experts=4, top_k=2,
            limit=10.0, cls=v4.DeepseekV4MoE):
    """A V4 MoE block at fused-kernel geometry (K/N % 256) on a routed
    layer (index past the hash layers), projections swapped for
    wire-shaped KQuant placeholders (zeros: eligibility reads shapes and
    codecs only, and the kernel calls are faked)."""
    from mlx_kquant.nn import KQuantLinear, KQuantSwitchLinear

    block = cls(_args(experts, top_k, limit), 3)
    for name, (o, i) in (("gate_proj", (INTER, D)), ("up_proj", (INTER, D)),
                         ("down_proj", (D, INTER))):
        setattr(block.switch_mlp, name,
                KQuantSwitchLinear(experts, o, i, False, codec))
        setattr(block.shared_experts, name,
                KQuantLinear(i, o, False, shexp_codec))
    block.eval()
    return block


def _installed(blk):
    shell = _Shell(blk)
    n_glu = install_fused_moe_glu(shell)
    n_fold = install_hyv3_shexp_fold(shell)
    return n_glu, n_fold


def _has_shexp_limit():
    return modules._FusedMoeCaps().has_shexp_limit


needs_fold = pytest.mark.skipif(
    not _has_shexp_limit(),
    reason="installed mlx-kquant has no silu_limit shared-expert fold")


# install: stamping + eligibility


@needs_fold
def test_install_stamps_shared_experts():
    blk = _kq_moe()
    n_glu, n_fold = _installed(blk)
    assert n_glu == 1
    assert blk.switch_mlp._kq_glu_act == "silu_limit"
    assert blk.switch_mlp._kq_glu_limit == 10.0
    assert n_fold == 1
    assert blk.switch_mlp._kq_shexp_mod is blk.shared_experts


def test_zero_limit_is_plain_silu_fold():
    blk = _kq_moe(limit=0.0)
    n_glu, n_fold = _installed(blk)
    assert n_glu == 1
    assert blk.switch_mlp._kq_glu_act == "silu"
    assert n_fold == 1


@needs_fold
@pytest.mark.parametrize("mut", ["limit", "codec", "caps"])
def test_ineligible_not_stamped(mut, monkeypatch):
    if mut == "limit":
        blk = _kq_moe()
        blk.shared_experts.swiglu_limit = 7.0  # kernel clamps every slot alike
    elif mut == "codec":
        blk = _kq_moe(shexp_codec="q4_0")  # not the codec nor an upcast
    else:
        blk = _kq_moe()
        base = modules._FusedMoeCaps

        class _Caps(base):
            def __init__(self):
                super().__init__()
                self.has_shexp_limit = False  # an older mlx-kquant

        monkeypatch.setattr(modules, "_FusedMoeCaps", _Caps)
    _, n_fold = _installed(blk)
    assert n_fold == 0
    assert getattr(blk.switch_mlp, "_kq_shexp_mod", None) is None


# fused decode branch: kernels get the clamp, mixed return used as-is


@needs_fold
def test_fused_branch_passes_limit(monkeypatch):
    import mlx_kquant as kq

    blk = _kq_moe()
    _installed(blk)
    implicit = modules._FusedMoeCaps().mix_implicit
    monkeypatch.setattr(modules, "_kq_fused_device_ok", lambda *m: True)

    seen = {}

    def fake_glu(x, gw, uw, sgw, suw, ktype, idx, **kw):
        seen["glu"] = {"ktype": ktype, "kw": kw}
        return mx.zeros((x.shape[0], idx.shape[1] + 1, INTER), x.dtype)

    def fake_mix(h, dw, sdw, ktype, idx, sc, **kw):
        seen["mix"] = {"ktype": ktype, "kw": kw,
                       "sc": np.array(sc.astype(mx.float32))}
        return mx.full((h.shape[0], D), 7.0, h.dtype)

    monkeypatch.setattr(kq, "moe_glu_gather_shexp_kq", fake_glu)
    monkeypatch.setattr(kq, "gather_qmv_mix_kq", fake_mix)

    mx.random.seed(3)
    x = mx.random.normal((1, 1, D)).astype(mx.bfloat16)
    y = blk(x, None)
    mx.eval(y)

    assert y.shape == (1, 1, D) and y.dtype == mx.bfloat16
    # mixed return used as-is: a second shared add would break 7
    assert np.allclose(np.array(y.astype(mx.float32)), 7.0)
    assert seen["glu"]["ktype"] == "iq2_xxs"
    assert seen["glu"]["kw"].get("act") == "silu_limit"
    assert seen["glu"]["kw"].get("limit") == 10.0
    assert seen["glu"]["kw"].get("shexp_kquant_type") == "q8_0"
    assert seen["mix"]["kw"].get("shexp_kquant_type") == "q8_0"
    sc = seen["mix"]["sc"]
    if implicit:
        assert sc.shape == (1, 2)
    else:
        assert sc.shape == (1, 3) and sc[0, -1] == 1.0


@needs_fold
def test_stamped_fallback_single_shexp_add(monkeypatch):
    blk = _kq_moe()
    _installed(blk)

    calls = []
    orig = v4.DeepseekV4MLP.__call__
    monkeypatch.setattr(
        v4.DeepseekV4MLP, "__call__",
        lambda self, x: calls.append(1) or orig(self, x))

    # idx.size 64 fails the fused branch's < 64 gate: stock fallback
    mx.random.seed(4)
    x = mx.random.normal((1, 32, D)).astype(mx.bfloat16)
    y = blk(x, None)
    mx.eval(y)
    assert y.shape == (1, 32, D)
    assert len(calls) == 1


# MoE return-shape contract (stub GLU: no kernels at all)


class _StubGLU(nn.Module):
    _kq_mix_scores = True

    def __init__(self, mixed):
        super().__init__()
        self._mixed = mixed
        self.saw_scores = []

    def __call__(self, x, inds, scores=None):
        self.saw_scores.append(scores is not None)
        if self._mixed:
            return mx.zeros(x.shape, x.dtype)
        k = inds.shape[-1]
        return mx.zeros((*x.shape[:-1], k, x.shape[-1]), x.dtype)


def _stub_moe(mixed, stamp, cls):
    blk = _kq_moe(cls=cls)
    stub = _StubGLU(mixed)
    if stamp:
        object.__setattr__(stub, "_kq_shexp_mod", blk.shared_experts)
    blk.switch_mlp = stub
    return blk


@pytest.mark.parametrize("cls", [v4.DeepseekV4MoE, v41.DeepseekV41MoE])
@pytest.mark.parametrize("mixed,stamp,n_shexp", [
    (True, True, 0),    # mixed + stamped: kernel consumed the shexp
    (True, False, 1),   # mixed, no fold: block adds it
    (False, True, 1),   # stamped fallback: unmixed, block mixes + adds
    (False, False, 1),  # stock shape
])
def test_moe_contract_adds_shexp_once(monkeypatch, cls, mixed, stamp, n_shexp):
    blk = _stub_moe(mixed, stamp, cls)
    calls = []
    orig = v4.DeepseekV4MLP.__call__
    monkeypatch.setattr(
        v4.DeepseekV4MLP, "__call__",
        lambda self, x: calls.append(1) or orig(self, x))

    x = mx.random.normal((1, 4, D)).astype(mx.bfloat16)
    y = blk(x, None)
    mx.eval(y)
    assert y.shape == (1, 4, D) and y.dtype == mx.bfloat16
    assert len(calls) == n_shexp
    assert blk.switch_mlp.saw_scores == [True]


@pytest.mark.parametrize("mixed,stamp,n_shexp", [
    (True, True, 0),
    (False, True, 1),
])
def test_v41_profiled_path_keeps_contract(monkeypatch, mixed, stamp, n_shexp):
    """The level-2 profile path of the V4.1 block runs the same two helpers
    between its marks."""
    monkeypatch.setattr(v41, "_SUB_PROFILE", True)
    log = []
    monkeypatch.setattr(v41, "_PROF_LOG", log)
    blk = _stub_moe(mixed, stamp, v41.DeepseekV41MoE)
    calls = []
    orig = v4.DeepseekV4MLP.__call__
    monkeypatch.setattr(
        v4.DeepseekV4MLP, "__call__",
        lambda self, x: calls.append(1) or orig(self, x))

    x = mx.random.normal((1, 1, D)).astype(mx.bfloat16)
    y = blk(x, None)
    mx.eval(y)
    assert y.shape == (1, 1, D)
    assert len(calls) == n_shexp
    assert [k for k, *_ in log] == ["f.route", "f.exp", "f.shexp"]
