#!/usr/bin/env python3
"""deepseek-v4 shared-expert fold: install_dsv4_shexp_fold stamps eligible
MoE blocks' fused SwitchGLUs (opt-in), the fused branch rides the shexp
gathers with the LimitedSwiGLU epilogue and a ones-column mix weight, and
DeepseekV4MoE adds the shared expert exactly once on every path. CPU-safe:
the shexp kernels are monkeypatched fakes; kernel numerics live in
mlx-kquant's own suite."""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from gmlx.modules import install_dsv4_shexp_fold, install_fused_moe_glu

D = 256
INTER = 256
EXPERTS = 4
TOP_K = 2
LIMIT = 10.0


class _Shell(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.mlp = block


def _args(limit=LIMIT, n_shared=1):
    from gmlx.deepseek_v4_model import ModelArgs

    return ModelArgs(
        hidden_size=D,
        moe_intermediate_size=INTER,
        n_routed_experts=EXPERTS,
        num_experts_per_tok=TOP_K,
        n_shared_experts=n_shared,
        num_hash_layers=0,
        swiglu_limit=limit,
        vocab_size=64,
    )


def _kq_moe(codec="iq2_xxs", shexp_codec="q8_0", limit=LIMIT, n_shared=1,
            shexp_limit=None):
    """Real DeepseekV4MoE at fused-kernel geometry (K/N % 256) with
    wire-shaped KQuant placeholders (zeros: eligibility reads shapes and
    codecs only, and the kernel calls are faked)."""
    from mlx_kquant.nn import KQuantLinear, KQuantSwitchLinear

    from gmlx.deepseek_v4_model import DeepseekV4MoE

    block = DeepseekV4MoE(_args(limit=limit, n_shared=n_shared), 4)
    for name, (o, i) in (("gate_proj", (INTER, D)), ("up_proj", (INTER, D)),
                         ("down_proj", (D, INTER))):
        setattr(block.switch_mlp, name,
                KQuantSwitchLinear(EXPERTS, o, i, False, codec))
    si = INTER * n_shared
    for name, (o, i) in (("gate_proj", (si, D)), ("up_proj", (si, D)),
                         ("down_proj", (D, si))):
        setattr(block.shared_experts, name,
                KQuantLinear(i, o, False, shexp_codec))
    if shexp_limit is not None:
        block.shared_experts.swiglu_limit = shexp_limit
    block.eval()
    return block


def _installed(blk, monkeypatch, on=True):
    monkeypatch.setenv("GMLX_SHEXP_FOLD", "1" if on else "0")
    shell = _Shell(blk)
    n_glu = install_fused_moe_glu(shell)
    n_fold = install_dsv4_shexp_fold(shell)
    return n_glu, n_fold


# install: opt-in gate + stamping + eligibility


def test_install_stamps_shexp_module(monkeypatch):
    blk = _kq_moe()
    n_glu, n_fold = _installed(blk, monkeypatch)
    assert n_glu == 1  # regime 2 took the SwitchGLU first
    assert type(blk.switch_mlp).__name__ == "_FusedKQuantSwitchGLU"
    assert n_fold == 1
    assert blk.switch_mlp._kq_shexp_mod is blk.shared_experts


def test_fold_is_opt_in(monkeypatch):
    """Unset env must not stamp: the fold measured slower than the stock
    three projections on V4-Flash, so it ships off."""
    monkeypatch.delenv("GMLX_SHEXP_FOLD", raising=False)
    blk = _kq_moe()
    shell = _Shell(blk)
    install_fused_moe_glu(shell)
    assert install_dsv4_shexp_fold(shell) == 0
    assert getattr(blk.switch_mlp, "_kq_shexp_mod", None) is None


@pytest.mark.parametrize("mut", ["wide", "codec", "limit"])
def test_ineligible_not_stamped(monkeypatch, mut):
    if mut == "wide":
        blk = _kq_moe(n_shared=2)  # shexp inter 512: not shape-matched
    elif mut == "codec":
        blk = _kq_moe(shexp_codec="q4_0")  # not the codec nor an upcast
    else:
        # one epilogue serves both slot kinds, so a shared expert that
        # clamps at a different bound cannot ride along
        blk = _kq_moe(shexp_limit=4.0)
    _, n_fold = _installed(blk, monkeypatch)
    assert n_fold == 0
    assert getattr(blk.switch_mlp, "_kq_shexp_mod", None) is None


def test_missing_kernel_capability_blocks_fold(monkeypatch):
    """Older mlx-kquant builds have no limit kwarg on the shexp op; the
    sniff must keep V4 unfolded rather than dispatch a plain-silu rider."""
    from gmlx import modules

    blk = _kq_moe()
    monkeypatch.setattr(modules._FusedMoeCaps, "__init__",
                        _caps_without_limit(modules._FusedMoeCaps.__init__))
    _, n_fold = _installed(blk, monkeypatch)
    assert n_fold == 0


def _caps_without_limit(orig):
    def patched(self):
        orig(self)
        self.shexp_limit_ok = False
    return patched


def test_block_env_disables_fold(monkeypatch):
    blk = _kq_moe()
    monkeypatch.setenv("GMLX_SHEXP_FOLD", "1")
    shell = _Shell(blk)
    install_fused_moe_glu(shell)
    monkeypatch.setenv("GMLX_FUSED_MOE_BLOCK", "0")
    assert install_dsv4_shexp_fold(shell) == 0


# fused decode branch: kernels + epilogue args + mix-weight layout


def test_fused_branch_rides_shexp_kernels(monkeypatch):
    import mlx_kquant as kq

    from gmlx import modules

    blk = _kq_moe()
    _installed(blk, monkeypatch)
    monkeypatch.setattr(modules, "_kq_fused_device_ok", lambda *m: True)

    seen = {}

    # Both fakes return x.dtype, as the real kernels do. DeepseekV4MoE has
    # no closing astype (hy_v3's does), so the block's output dtype IS the
    # kernel's: a kernel that returned f32 would silently widen the
    # residual stream downstream.
    def fake_glu(x, gw, uw, sgw, suw, ktype, idx, **kw):
        seen["glu"] = {"ktype": ktype, "kw": kw, "sgw": tuple(sgw.shape)}
        return mx.zeros((x.shape[0], idx.shape[1] + 1, D), x.dtype)

    def fake_mix(h, dw, sdw, ktype, idx, sc, **kw):
        seen["mix"] = {"ktype": ktype, "kw": kw, "sc": np.array(
            sc.astype(mx.float32)), "sdw": tuple(sdw.shape)}
        return mx.full((h.shape[0], D), 7.0, h.dtype)

    monkeypatch.setattr(kq, "moe_glu_gather_shexp_kq", fake_glu)
    monkeypatch.setattr(kq, "gather_qmv_mix_kq", fake_mix)

    mx.random.seed(3)
    x = mx.random.normal((1, 1, D)).astype(mx.bfloat16)
    y = blk(x, None)
    mx.eval(y)

    assert y.shape == (1, 1, D) and y.dtype == mx.bfloat16
    # mixed return used as-is: a second python-side shared add would break 7
    assert np.allclose(np.array(y.astype(mx.float32)), 7.0)
    assert seen["glu"]["ktype"] == "iq2_xxs"
    # V4's LimitedSwiGLU rides both slot kinds with the same clamp
    assert seen["glu"]["kw"].get("act") == "silu_limit"
    assert seen["glu"]["kw"].get("limit") == LIMIT
    assert seen["glu"]["kw"].get("shexp_kquant_type") == "q8_0"
    assert seen["glu"]["sgw"] == tuple(
        blk.shared_experts.gate_proj.weight.shape)
    assert seen["mix"]["kw"].get("shexp_kquant_type") == "q8_0"
    assert seen["mix"]["sdw"] == tuple(
        blk.shared_experts.down_proj.weight.shape)
    sc = seen["mix"]["sc"]
    assert sc.shape == (1, TOP_K + 1)
    assert sc[0, -1] == 1.0  # constant shexp mix weight rides last


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
        object.__setattr__(stub, "_kq_shexp_mod", blk.shared_experts)
    blk.switch_mlp = stub
    return blk


@pytest.mark.parametrize("mixed,stamp,n_shexp", [
    (True, True, 0),    # mixed + stamped: kernel consumed the shexp
    (True, False, 1),   # mixed, no fold: block adds it
    (False, True, 1),   # stamped fallback: unmixed, block mixes + adds
    (False, False, 1),  # stock shape
])
def test_moe_contract_adds_shexp_once(monkeypatch, mixed, stamp, n_shexp):
    from gmlx.deepseek_v4_model import DeepseekV4MLP

    blk = _stub_moe(mixed, stamp)
    calls = []
    orig = DeepseekV4MLP.__call__
    monkeypatch.setattr(
        DeepseekV4MLP, "__call__",
        lambda self, x: calls.append(1) or orig(self, x))

    x = mx.random.normal((1, 4, D)).astype(mx.bfloat16)
    y = blk(x, None)
    mx.eval(y)
    assert y.shape == (1, 4, D) and y.dtype == mx.bfloat16
    assert len(calls) == n_shexp
    assert blk.switch_mlp.saw_scores == [True]


def test_mix_env_off_keeps_scores_out(monkeypatch):
    from gmlx import deepseek_v4_model

    monkeypatch.setattr(deepseek_v4_model, "_MOE_MIX_SCORES", False)
    blk = _stub_moe(mixed=False, stamp=False)
    x = mx.random.normal((1, 4, D)).astype(mx.bfloat16)
    y = blk(x, None)
    mx.eval(y)
    assert y.shape == (1, 4, D)
    assert blk.switch_mlp.saw_scores == [False]  # called without scores
