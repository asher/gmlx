#!/usr/bin/env python3
"""hy_v3 shared-expert fold: install_hyv3_shexp_fold stamps eligible MoE
blocks' fused SwitchGLUs, the fused decode branch rides the shexp gathers
with a ones-column mix weight, and the MoE return-shape contract adds the
shared expert exactly once on every path. CPU-safe: the shexp kernels are
monkeypatched fakes; kernel numerics live in mlx-kquant's own suite."""

from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

import mlx_kquant as _kq

from gmlx.modules import (
    install_fused_moe_glu,
    install_hyv3_router_fuse,
    install_hyv3_shexp_fold,
)

_ROUTER_OK = '"sigmoid"' in (getattr(_kq.moe_router_topk, "__doc__", "") or "")
router_cap = pytest.mark.skipif(
    not _ROUTER_OK, reason="installed mlx_kquant lacks sigmoid router scoring")


class _Shell(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.mlp = block


def _kq_moe(codec="iq4_xs", shexp_codec="q8_0", d=256, inter=256,
            experts=4, top_k=2, n_shared=1):
    """Real hy_v3 MoE at fused-kernel geometry (K/N % 256), projections
    swapped for wire-shaped KQuant placeholders (zeros: eligibility only
    reads shapes + codecs, and the kernel calls are faked)."""
    from mlx_kquant.nn import KQuantLinear, KQuantSwitchLinear

    from gmlx.hy_v3_model import MoE

    args = SimpleNamespace(
        hidden_size=d,
        expert_hidden_dim=inter,
        num_experts=experts,
        num_experts_per_tok=top_k,
        num_shared_experts=n_shared,
        route_norm=True,
        router_scaling_factor=1.5,
        enable_moe_fp32_combine=False,
    )
    block = MoE(args)
    for name, (o, i) in (("gate_proj", (inter, d)), ("up_proj", (inter, d)),
                         ("down_proj", (d, inter))):
        setattr(block.switch_mlp, name,
                KQuantSwitchLinear(experts, o, i, False, codec))
    if n_shared:
        si = inter * n_shared
        for name, (o, i) in (("gate_proj", (si, d)), ("up_proj", (si, d)),
                             ("down_proj", (d, si))):
            setattr(block.shared_mlp, name,
                    KQuantLinear(i, o, False, shexp_codec))
    block.eval()
    return block


def _installed(blk):
    shell = _Shell(blk)
    n_glu = install_fused_moe_glu(shell)
    n_fold = install_hyv3_shexp_fold(shell)
    return n_glu, n_fold


# install: stamping + eligibility


def test_install_stamps_shexp_module():
    blk = _kq_moe()
    n_glu, n_fold = _installed(blk)
    assert n_glu == 1  # regime 2 took the SwitchGLU first
    assert type(blk.switch_mlp).__name__ == "_FusedKQuantSwitchGLU"
    assert n_fold == 1
    assert blk.switch_mlp._kq_shexp_mod is blk.shared_mlp


@pytest.mark.parametrize("mut", ["none", "wide", "codec"])
def test_ineligible_not_stamped(mut):
    if mut == "none":
        blk = _kq_moe(n_shared=0)  # no shared expert at all
    elif mut == "wide":
        blk = _kq_moe(n_shared=2)  # shexp inter 512: not shape-matched
    else:
        blk = _kq_moe(shexp_codec="q4_0")  # not the codec nor an upcast
    _, n_fold = _installed(blk)
    assert n_fold == 0
    assert getattr(blk.switch_mlp, "_kq_shexp_mod", None) is None


def test_block_env_disables_fold(monkeypatch):
    blk = _kq_moe()
    shell = _Shell(blk)
    install_fused_moe_glu(shell)
    monkeypatch.setenv("GMLX_FUSED_MOE_BLOCK", "0")
    assert install_hyv3_shexp_fold(shell) == 0


# fused decode branch: kernels + mix-weight layout + no double add


def test_fused_branch_rides_shexp_kernels(monkeypatch):
    import mlx_kquant as kq

    from gmlx import modules

    blk = _kq_moe()
    _installed(blk)
    monkeypatch.setattr(modules, "_kq_fused_device_ok", lambda *m: True)

    seen = {}

    def fake_glu(x, gw, uw, sgw, suw, ktype, idx, **kw):
        seen["glu"] = {"ktype": ktype, "kw": kw, "sgw": tuple(sgw.shape)}
        return mx.zeros((x.shape[0], idx.shape[1] + 1, 256), mx.float32)

    def fake_mix(h, dw, sdw, ktype, idx, sc, **kw):
        seen["mix"] = {"ktype": ktype, "kw": kw, "sc": np.array(
            sc.astype(mx.float32)), "sdw": tuple(sdw.shape)}
        return mx.full((h.shape[0], 256), 7.0, mx.float32)

    monkeypatch.setattr(kq, "moe_glu_gather_shexp_kq", fake_glu)
    monkeypatch.setattr(kq, "gather_qmv_mix_kq", fake_mix)

    mx.random.seed(3)
    x = mx.random.normal((1, 1, 256)).astype(mx.bfloat16)
    y = blk(x)
    mx.eval(y)

    assert y.shape == (1, 1, 256) and y.dtype == mx.bfloat16
    # mixed return used as-is: a second python-side shared add would break 7
    assert np.allclose(np.array(y.astype(mx.float32)), 7.0)
    assert seen["glu"]["ktype"] == "iq4_xs"
    assert seen["glu"]["kw"].get("act") == "silu"
    assert seen["glu"]["kw"].get("shexp_kquant_type") == "q8_0"
    assert seen["glu"]["sgw"] == tuple(blk.shared_mlp.gate_proj.weight.shape)
    assert seen["mix"]["kw"].get("shexp_kquant_type") == "q8_0"
    assert seen["mix"]["sdw"] == tuple(blk.shared_mlp.down_proj.weight.shape)
    sc = seen["mix"]["sc"]
    assert sc.shape == (1, 3)  # top_k routed slots + shexp slot
    assert sc[0, -1] == 1.0  # constant shexp mix weight rides last
    # routed slots: renormed scores * routed_scaling_factor (1.5)
    assert abs(sc[0, :2].sum() - 1.5) < 2e-2


def test_stamped_fallback_single_shexp_add(monkeypatch):
    from gmlx.hy_v3_model import MLP

    blk = _kq_moe()
    _installed(blk)

    calls = []
    orig = MLP.__call__
    monkeypatch.setattr(
        MLP, "__call__", lambda self, x: calls.append(1) or orig(self, x))

    # idx.size 64 fails the fused branch's < 64 gate: stock fallback
    mx.random.seed(4)
    x = mx.random.normal((1, 32, 256)).astype(mx.bfloat16)
    y = blk(x)
    mx.eval(y)
    assert y.shape == (1, 32, 256)
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


def _stub_moe(mixed, stamp):
    blk = _kq_moe()
    stub = _StubGLU(mixed)
    if stamp:
        object.__setattr__(stub, "_kq_shexp_mod", blk.shared_mlp)
    blk.switch_mlp = stub
    return blk


@pytest.mark.parametrize("mixed,stamp,n_shexp", [
    (True, True, 0),    # mixed + stamped: kernel consumed the shexp
    (True, False, 1),   # mixed, no fold: block adds it
    (False, True, 1),   # stamped fallback: unmixed, block mixes + adds
    (False, False, 1),  # stock shape
])
def test_moe_contract_adds_shexp_once(monkeypatch, mixed, stamp, n_shexp):
    from gmlx.hy_v3_model import MLP

    blk = _stub_moe(mixed, stamp)
    calls = []
    orig = MLP.__call__
    monkeypatch.setattr(
        MLP, "__call__", lambda self, x: calls.append(1) or orig(self, x))

    x = mx.random.normal((1, 4, 256)).astype(mx.bfloat16)
    y = blk(x)
    mx.eval(y)
    assert y.shape == (1, 4, 256) and y.dtype == mx.bfloat16
    assert len(calls) == n_shexp
    assert blk.switch_mlp.saw_scores == [True]  # ds4-style scores passing


def test_mix_env_off_keeps_scores_out(monkeypatch):
    from gmlx import hy_v3_model

    monkeypatch.setattr(hy_v3_model, "_MOE_MIX_SCORES", False)
    blk = _stub_moe(mixed=False, stamp=False)
    x = mx.random.normal((1, 4, 256)).astype(mx.bfloat16)
    y = blk(x)
    mx.eval(y)
    assert y.shape == (1, 4, 256)
    assert blk.switch_mlp.saw_scores == [False]  # called without scores


# sigmoid router fuse


@router_cap
def test_router_fuse_installs():
    blk = _kq_moe()
    shell = _Shell(blk)
    assert install_hyv3_router_fuse(shell) == 1
    assert type(blk.router).__name__ == "_FusedHyV3Router"
    assert install_hyv3_router_fuse(shell) == 0  # swapped class is not MoEGate


@router_cap
def test_fused_router_rides_kernel(monkeypatch):
    from gmlx import modules

    blk = _kq_moe()
    install_hyv3_router_fuse(_Shell(blk))
    monkeypatch.setattr(modules, "_kq_fused_device_ok", lambda *m: True)

    seen = {}

    def fake_router(logits, top_k, norm, **kw):
        seen["logits"] = tuple(logits.shape)
        seen["top_k"] = top_k
        seen["norm"] = norm
        seen["kw"] = kw
        t = logits.shape[0]
        return (mx.zeros((t, top_k), mx.uint32),
                mx.ones((t, top_k), mx.float32))

    monkeypatch.setattr(_kq, "moe_router_topk", fake_router)

    x = mx.random.normal((1, 1, 256)).astype(mx.bfloat16)
    inds, sc = blk.router(x)
    mx.eval(inds, sc)
    assert inds.shape == (1, 1, 2) and sc.shape == (1, 1, 2)
    assert seen["logits"] == (1, 4)  # [t, E]
    assert seen["top_k"] == 2 and seen["norm"] is True
    assert seen["kw"]["scoring"] == "sigmoid"
    assert seen["kw"]["shared_gate"] is False
    assert seen["kw"]["scale"] == 1.5
    assert seen["kw"]["bias"].dtype == mx.float32

    # top_k = 1 must not renorm (stock skips it: single score stays raw)
    blk.router.top_k = 1
    blk.router(x)
    assert seen["norm"] is False and seen["top_k"] == 1


@router_cap
def test_fused_router_prefill_falls_back(monkeypatch):
    blk = _kq_moe()
    install_hyv3_router_fuse(_Shell(blk))
    monkeypatch.setattr(
        _kq, "moe_router_topk",
        lambda *a, **k: pytest.fail("prefill must keep the stock epilogue"))

    x = mx.random.normal((1, 64, 256)).astype(mx.bfloat16)
    inds, sc = blk.router(x)
    mx.eval(inds, sc)
    assert inds.shape == (1, 64, 2) and sc.shape == (1, 64, 2)


@router_cap
def test_fused_router_matches_expert_select():
    if mx.default_device() == mx.cpu:
        pytest.skip("fused router kernel is Metal-only")
    from gmlx.hy_v3_model import expert_select

    mx.random.seed(11)
    blk = _kq_moe(experts=16, top_k=4)
    blk.router.expert_bias = mx.random.normal((16,)) * 0.1
    install_hyv3_router_fuse(_Shell(blk))

    x = mx.random.normal((2, 1, 256)).astype(mx.bfloat16)
    inds, sc = blk.router(x)
    ri, rs = expert_select(
        blk.router.gate(x), blk.router.expert_bias, 4, 1.5, True)
    mx.eval(inds, sc, ri, rs)

    got = np.array(inds).reshape(2, 4)
    want = np.array(ri).reshape(2, 4)
    gsc = np.array(sc.astype(mx.float32)).reshape(2, 4)
    wsc = np.array(rs.astype(mx.float32)).reshape(2, 4)
    for t in range(2):
        assert set(got[t]) == set(want[t])  # slot order may differ
        gd = dict(zip(got[t].tolist(), gsc[t].tolist()))
        wd = dict(zip(want[t].tolist(), wsc[t].tolist()))
        for e in gd:
            assert abs(gd[e] - wd[e]) < 1e-5
