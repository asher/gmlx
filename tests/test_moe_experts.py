#!/usr/bin/env python3
"""Adaptive MoE fan-out (``--moe-expert-mass`` / ``--moe-expert-probe``): the
mass filter must keep shapes and the weight sum, drop only the lowest-weight
experts, and duplicate the top-1 id into dropped slots; the installers must
hook exactly the offloaded blocks; the per-arch copied forwards must match
stock when the controls are unset. Pure CPU, tiny synthetic blocks."""

from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from gmlx.loader import install_expert_streaming, install_moe_experts_override
from gmlx.moe_experts import (
    _INLINE_SWAPS,
    ExpertProbe,
    _block_class,
    _mass_filter,
    install_moe_expert_mass,
    install_moe_expert_probe,
)

from test_offload import _kquant_glu


def test_mass_filter_numerics():
    weights = mx.array([[[0.5, 0.05, 0.3, 0.15], [0.25, 0.25, 0.25, 0.25]]])
    inds = mx.array([[[7, 1, 4, 2], [3, 5, 0, 6]]], dtype=mx.uint32)

    e, w = _mass_filter(inds, weights, 1.0)
    assert e.shape == inds.shape and w.shape == weights.shape
    mx.eval(e, w)
    # p=1.0 keeps everything: same id set, same total, weight-sorted order
    assert sorted(np.array(e)[0, 0].tolist()) == [1, 2, 4, 7]
    assert np.allclose(np.array(w).sum(-1), np.array(weights).sum(-1))
    assert np.array(w)[0, 0].tolist() == sorted(np.array(weights)[0, 0], reverse=True)

    # p=0.7 on [.5,.3,.15,.05]: prefix mass [0,.5,.8,.95] -> keep 2, renorm
    # to the original sum, dropped slots duplicate the top-1 id at weight 0
    e, w = _mass_filter(inds, weights, 0.7)
    mx.eval(e, w)
    e0, w0 = np.array(e)[0, 0], np.array(w)[0, 0]
    assert e0.tolist() == [7, 4, 7, 7]
    assert np.allclose(w0, [0.625, 0.375, 0.0, 0.0])
    # flat weights at p=0.7: three experts needed (.25, .5, .75 crossings)
    e1, w1 = np.array(e)[0, 1], np.array(w)[0, 1]
    assert (w1 > 0).sum() == 3 and w1[3] == 0.0
    assert np.allclose(w1.sum(), 1.0)
    assert len(set(e1.tolist())) == 3  # dropped slot duplicates a kept id

    # tiny p always keeps the top expert
    e, w = _mass_filter(inds, weights, 0.01)
    mx.eval(e, w)
    assert np.array(e)[0, 0].tolist() == [7, 7, 7, 7]
    assert np.allclose(np.array(w)[0, 0], [1.0, 0.0, 0.0, 0.0])


def test_probe_counts_and_report(capsys):
    rng = np.random.default_rng(11)
    probe = ExpertProbe()
    k, g = 8, len(probe.grid)
    ref_hist = np.zeros((g, k + 1), dtype=np.int64)
    tokens = 0
    for _ in range(97):  # crosses the 64-record flush threshold
        wnp = rng.random((1, 3, k), dtype=np.float32)
        probe.record(0, mx.array(wnp))
        w = np.sort(wnp.reshape(-1, k), -1)[:, ::-1]
        total = w.sum(-1, keepdims=True) + 1e-20
        prefix = np.cumsum(w, -1, dtype=np.float32) - w
        for gi, p in enumerate(probe.grid):
            counts = (prefix < np.float32(p) * total).sum(-1)
            ref_hist[gi] += np.bincount(counts, minlength=k + 1)
        tokens += w.shape[0]

    probe.report()
    out = capsys.readouterr().out
    assert "prefill fan-out probe" in out and f"{tokens} token-layer" in out
    assert "decode fan-out probe" not in out  # no single-token records yet
    prefill = probe._buckets["prefill"]
    assert prefill.tokens == tokens
    assert np.array_equal(prefill.hist, ref_hist)
    probe.report()  # idempotent
    assert capsys.readouterr().out == ""


def test_probe_splits_decode_from_prefill(capsys):
    """Single-token records land in the decode bucket, multi-token in the
    prefill bucket, and the report prints each phase's own table (a long
    prompt must not skew the decode distribution the table sizes P against)."""
    rng = np.random.default_rng(12)
    probe = ExpertProbe()
    k = 4
    for _ in range(3):
        probe.record(0, mx.array(rng.random((1, 1, k), dtype=np.float32)))
    probe.record(0, mx.array(rng.random((1, 7, k), dtype=np.float32)))
    probe._flush()
    assert probe._buckets["decode"].tokens == 3
    assert probe._buckets["prefill"].tokens == 7
    probe.report()
    out = capsys.readouterr().out
    assert "decode fan-out probe over 3 token-layer" in out
    assert "prefill fan-out probe over 7 token-layer" in out
    assert out.index("decode fan-out") < out.index("prefill fan-out")


class _TupleGate(nn.Module):
    """DeepSeek-style router: returns (inds, weights) and owns top_k."""

    def __init__(self, n_experts=4, dim=32):
        super().__init__()
        self.top_k = 2
        self.weight = mx.random.normal((n_experts, dim))

    def __call__(self, x):
        gates = mx.softmax(x @ self.weight.T, axis=-1, precise=True)
        k = self.top_k
        inds = mx.argpartition(-gates, kth=k - 1, axis=-1)[..., :k]
        weights = mx.take_along_axis(gates, inds, axis=-1)
        return inds, weights


class _Block(nn.Module):
    def __init__(self, glu, gate):
        super().__init__()
        self.num_experts_per_tok = 2
        self.gate = gate
        self.switch_mlp = glu

    def __call__(self, x):
        inds, weights = self.gate(x)
        y = self.switch_mlp(x, inds)
        return (y * weights[..., None]).sum(axis=-2)


def _shell(block):
    class _Shell:
        pass

    class _Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = block

    model = _Shell()
    model.layers = [_Layer()]
    model.parameters = lambda: {"glu": block.switch_mlp.parameters()}
    return model


def test_gate_wrap_targets_offloaded(monkeypatch):
    mx.random.seed(4)
    block = _Block(_kquant_glu(), _TupleGate())
    model = _shell(block)

    assert install_moe_expert_mass(model, 0.5) == 0  # nothing offloaded yet

    monkeypatch.setattr(
        mx, "device_info", lambda: {"max_recommended_working_set_size": 1024}
    )
    install_expert_streaming(model)

    with pytest.raises(ValueError):
        install_moe_expert_mass(model, 0.0)
    with pytest.raises(ValueError):
        install_moe_expert_mass(model, 1.5)

    assert install_moe_expert_mass(model, 1e-6) == 1
    assert type(block.gate).__name__ == "_TupleGate_ExpertCtl"
    assert block.gate._kq_li == 0

    x = mx.random.normal((1, 5, 32))
    inds, weights = block.gate(x)
    mx.eval(inds, weights)
    assert inds.shape == (1, 5, 2)
    for t in range(5):
        assert len(set(np.array(inds)[0, t].tolist())) == 1  # collapsed to top-1
    assert np.allclose(np.array(weights)[..., 1], 0.0)
    mx.eval(block(x))  # end-to-end through the offloaded gather


def test_linear_gate_block_is_unsupported(monkeypatch, capsys):
    class _LinearGateBlock(nn.Module):
        def __init__(self, glu):
            super().__init__()
            self.gate = nn.Linear(32, 4, bias=False)
            self.switch_mlp = glu

    block = _LinearGateBlock(_kquant_glu())
    model = _shell(block)
    monkeypatch.setattr(
        mx, "device_info", lambda: {"max_recommended_working_set_size": 1024}
    )
    install_expert_streaming(model)
    assert install_moe_expert_mass(model, 0.9) == 0
    assert "_LinearGateBlock" in capsys.readouterr().out
    assert isinstance(block.gate, nn.Linear)  # never wrapped


def test_probe_install_and_gate_recording(monkeypatch):
    mx.random.seed(5)
    block = _Block(_kquant_glu(), _TupleGate())
    model = _shell(block)
    monkeypatch.setattr(
        mx, "device_info", lambda: {"max_recommended_working_set_size": 1024}
    )
    install_expert_streaming(model)
    assert install_moe_expert_probe(model) == 1

    probe = block.gate._kq_expert_probe
    x = mx.random.normal((1, 5, 32))
    inds, weights = block.gate(x)
    mx.eval(inds, weights)
    # lossless: full trained fan-out survives
    assert inds.shape == (1, 5, 2)
    assert (np.array(weights) > 0).all()
    probe._flush()
    assert probe._buckets["prefill"].tokens == 5


class _SwitchRecorder(nn.Module):
    """Wraps a SwitchGLU to capture the indices each call receives."""

    def __init__(self, glu):
        super().__init__()
        self.glu = glu
        self.seen = []

    def __call__(self, x, inds, *args):
        self.seen.append(np.array(inds))
        return self.glu(x, inds, *args)


def _arch_fixtures():
    from mlx_lm.models.minimax import MiniMaxSparseMoeBlock
    from mlx_lm.models.qwen3_moe import Qwen3MoeSparseMoeBlock
    from mlx_lm.models.qwen3_next import Qwen3NextSparseMoeBlock

    from gmlx.minimax_m3_model import MiniMaxM3SparseMoeBlock

    common = dict(
        hidden_size=16,
        moe_intermediate_size=32,
        intermediate_size=32,
        num_experts=8,
        num_local_experts=8,
        num_experts_per_tok=4,
        norm_topk_prob=True,
        shared_expert_intermediate_size=32,
        routed_scaling_factor=1.5,
        swiglu_alpha=1.702,
        swiglu_limit=7.0,
        shared_intermediate_size=32,
    )
    args = SimpleNamespace(**common)
    return [
        Qwen3MoeSparseMoeBlock(args),
        Qwen3NextSparseMoeBlock(args),
        MiniMaxSparseMoeBlock(args),
        MiniMaxM3SparseMoeBlock(args),
    ]


def test_inline_swaps_match_stock():
    mx.random.seed(6)
    x = mx.random.normal((1, 5, 16))
    for block in _arch_fixtures():
        mx.eval(block.parameters())
        name = type(block).__name__
        ref = np.array(block(x))

        block.__class__ = _block_class(type(block), _INLINE_SWAPS[name])
        assert type(block).__name__ == name + "_ExpertCtl"
        # controls unset: the copied forward is numerically identical
        assert np.array_equal(np.array(block(x)), ref), name

        rec = _SwitchRecorder(block.switch_mlp)
        block.switch_mlp = rec
        object.__setattr__(block, "_kq_expert_mass", 1.0)
        assert np.allclose(np.array(block(x)), ref, atol=1e-5), name

        object.__setattr__(block, "_kq_expert_mass", 1e-6)
        out = np.array(block(x))
        assert out.shape == ref.shape, name
        low = rec.seen[-1]
        assert low.shape[-1] == 4, name
        for t in range(low.shape[1]):
            assert len(set(low[0, t].tolist())) == 1, name  # collapsed to top-1


def test_hunyuan_seam_gated_by_attrs():
    from gmlx.loader import _patch_hunyuan_norm_topk

    args = SimpleNamespace(
        hidden_size=16,
        intermediate_size=32,
        use_mixed_mlp_moe=False,
        num_shared_expert=1,
        num_experts=8,
        moe_topk=4,
        moe_intermediate_size=None,
    )
    from mlx_lm.models.hunyuan import MoeBlock

    class _Holder(nn.Module):
        def __init__(self):
            super().__init__()
            self.blk = MoeBlock(args)

    holder = _Holder()
    mx.eval(holder.parameters())
    _patch_hunyuan_norm_topk(holder)
    block = holder.blk
    assert type(block).__name__ == "_NormTopKMoE"

    mx.random.seed(7)
    x = mx.random.normal((1, 5, 16))
    ref = np.array(block(x))

    rec = _SwitchRecorder(block.switch_mlp)
    block.switch_mlp = rec
    object.__setattr__(block, "_kq_expert_mass", 1.0)
    assert np.allclose(np.array(block(x)), ref, atol=1e-5)

    object.__setattr__(block, "_kq_expert_mass", 1e-6)
    mx.eval(block(x))
    low = rec.seen[-1]
    for t in range(low.shape[1]):
        assert len(set(low[0, t].tolist())) == 1


def test_fused_kquant_block_is_hooked(monkeypatch):
    """A block the loader fused at load (_FusedKQuantMoeBlock) has the hook
    baked into its forward; the installer only needs to set the attrs."""

    class _FusedKQuantMoeBlock(nn.Module):
        def __init__(self, glu):
            super().__init__()
            self.gate = nn.Linear(32, 4, bias=False)
            self.top_k = 2
            self.switch_mlp = glu

    block = _FusedKQuantMoeBlock(_kquant_glu())
    model = _shell(block)
    monkeypatch.setattr(
        mx, "device_info", lambda: {"max_recommended_working_set_size": 1024}
    )
    install_expert_streaming(model)
    assert install_moe_expert_mass(model, 0.9) == 1
    assert type(block).__name__ == "_FusedKQuantMoeBlock"  # no class swap
    assert block._kq_expert_mass == 0.9
    assert block._kq_li == 0


def test_mass_composes_with_fixed_k(monkeypatch):
    mx.random.seed(8)
    block = _Block(_kquant_glu(), _TupleGate())
    model = _shell(block)
    monkeypatch.setattr(
        mx, "device_info", lambda: {"max_recommended_working_set_size": 1024}
    )
    install_expert_streaming(model)
    install_moe_experts_override(model, 1)  # fixed cap first, like _apply_placement
    assert install_moe_expert_mass(model, 0.9) == 1

    x = mx.random.normal((1, 5, 32))
    inds, weights = block.gate(x)
    mx.eval(inds, weights)
    assert inds.shape == (1, 5, 1)  # mass filter ran within the lowered k
    assert (np.array(weights) > 0).all()  # sum preserved, single survivor


def _hy_v3_moe():
    """Real hy_v3 MoE block (its gate submodule sits at ``router``), the
    SwitchGLU swapped for the kquant fixture the offload installer wraps."""
    from gmlx.hy_v3_model import MoE

    args = SimpleNamespace(
        hidden_size=32,
        expert_hidden_dim=64,
        num_experts=4,
        num_experts_per_tok=2,
        num_shared_experts=0,
        route_norm=True,
        router_scaling_factor=1.5,
        enable_moe_fp32_combine=False,
    )
    block = MoE(args)
    block.switch_mlp = _kquant_glu()
    mx.eval(block.parameters())
    return block


def test_hy_v3_router_attr_is_hooked(monkeypatch):
    """hy_v3 names its DeepSeek-shaped gate ``router``: the mass installer
    must find it there, and the fixed-k override must reach the live
    ``router.top_k`` (the block-level num_experts_per_tok is dead in the
    hy_v3 forward)."""
    mx.random.seed(9)
    block = _hy_v3_moe()
    model = _shell(block)
    monkeypatch.setattr(
        mx, "device_info", lambda: {"max_recommended_working_set_size": 1024}
    )
    install_expert_streaming(model)

    assert install_moe_expert_mass(model, 1e-6) == 1
    assert type(block.router).__name__ == "MoEGate_ExpertCtl"
    assert block.router._kq_li == 0

    x = mx.random.normal((1, 5, 32))
    inds, weights = block.router(x)
    mx.eval(inds, weights)
    assert inds.shape == (1, 5, 2)
    for t in range(5):
        assert len(set(np.array(inds)[0, t].tolist())) == 1  # collapsed to top-1
    assert np.allclose(np.array(weights)[..., 1], 0.0)
    mx.eval(block(x))  # end-to-end through the offloaded gather

    install_moe_experts_override(model, 1)
    assert block.router.top_k == 1  # live attr, not the dead block-level one
