#!/usr/bin/env python3
"""Adaptive MoE fan-out (``--moe-expert-mass`` / ``--moe-expert-probe``): the
mass filter must keep shapes and the weight sum, drop only the lowest-weight
experts, and duplicate the top-1 id into dropped slots; the installers must
hook exactly the offloaded blocks; the per-arch copied forwards must match
stock when the controls are unset. Pure CPU, tiny synthetic blocks."""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from gmlx.stream.expert_streaming import install_expert_streaming, install_moe_experts_override
from gmlx.stream.moe_experts import (
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
    from gmlx._vendor.mlx_lm.models.minimax import MiniMaxSparseMoeBlock
    from gmlx._vendor.mlx_lm.models.qwen3_moe import Qwen3MoeSparseMoeBlock
    from gmlx._vendor.mlx_lm.models.qwen3_next import Qwen3NextSparseMoeBlock

    from gmlx.models.minimax_m3 import MiniMaxM3SparseMoeBlock

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


def test_m3_block_hands_scores_to_scores_sink():
    """The m3 block passes its routing weights to a module advertising
    _kq_scores_sink (the streaming offload wrapper around a stock base:
    m3's SwiGLUOAI stack never gets the fused mix-seam class); an
    unmixed return keeps the block's python-side sum, so the output
    matches the stock forward."""
    mx.random.seed(13)
    block = _arch_fixtures()[-1]
    mx.eval(block.parameters())
    x = mx.random.normal((1, 3, 16))
    ref = np.array(block(x))

    seen = []
    inner = block.switch_mlp

    class _Sink(nn.Module):
        _kq_scores_sink = True

        def __call__(self, x, inds, scores=None):
            seen.append(None if scores is None else np.array(scores))
            return inner(x, inds)

    block.switch_mlp = _Sink()
    out = np.array(block(x))
    assert seen and seen[-1] is not None
    assert seen[-1].shape == (1, 3, 4)
    # normalized top-k weights scaled by routed_scaling_factor
    assert np.allclose(seen[-1].sum(-1), 1.5, atol=1e-3)
    assert np.allclose(out, ref, atol=1e-6)


def test_hunyuan_seam_gated_by_attrs():
    from gmlx.load.loader import _patch_hunyuan_norm_topk

    args = SimpleNamespace(
        hidden_size=16,
        intermediate_size=32,
        use_mixed_mlp_moe=False,
        num_shared_expert=1,
        num_experts=8,
        moe_topk=4,
        moe_intermediate_size=None,
    )
    from gmlx._vendor.mlx_lm.models.hunyuan import MoeBlock

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
    from gmlx.models.hy_v3.model import MoE

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


# miss-shed / layer-shed installers (--moe-miss-shed / --moe-layer-shed)
def test_miss_shed_install_targets_decode_feeder_layers(monkeypatch, capsys):
    from gmlx.stream.moe_experts import install_moe_miss_shed

    block = _Block(_kquant_glu(), _TupleGate())
    model = _shell(block)
    monkeypatch.setattr(
        mx, "device_info", lambda: {"max_recommended_working_set_size": 1024}
    )
    install_expert_streaming(model)

    for bad in (0.0, 1.5):
        with pytest.raises(ValueError):
            install_moe_miss_shed(model, bad)

    # no decode feeder attached => announced no-effect
    assert install_moe_miss_shed(model, 0.9) == 0
    assert "no effect" in capsys.readouterr().out

    object.__setattr__(block.switch_mlp, "_kq_decode_feeder", object())
    assert install_moe_miss_shed(model, 0.9) == 1
    assert block.switch_mlp._kq_miss_shed == 0.9
    assert not hasattr(block.gate, "_kq_miss_shed")
    # Confirmation banners are verbose-only.
    assert "miss-shed 0.9" not in capsys.readouterr().out
    import gmlx.load.loadlog as loadlog
    monkeypatch.setattr(loadlog, "_LAST_VERBOSE", True)
    object.__setattr__(block.switch_mlp, "_kq_miss_shed", None)
    assert install_moe_miss_shed(model, 0.9) == 1
    assert "miss-shed 0.9" in capsys.readouterr().out


def test_layer_shed_install_targets_streamed_glus(monkeypatch, capsys):
    from gmlx.stream.moe_experts import install_moe_layer_shed

    block = _Block(_kquant_glu(), _TupleGate())
    block.shared_expert = nn.Linear(32, 32)
    model = _shell(block)

    for bad in (0.0, 1.0):
        with pytest.raises(ValueError):
            install_moe_layer_shed(model, bad)

    assert install_moe_layer_shed(model, 0.1) == 0  # nothing streamed yet
    assert "no effect" in capsys.readouterr().out

    monkeypatch.setattr(
        mx, "device_info", lambda: {"max_recommended_working_set_size": 1024}
    )
    install_expert_streaming(model)
    assert install_moe_layer_shed(model, 0.1) == 1
    assert block.switch_mlp._kq_layer_shed == 0.1
    # routers/gates never shed: the hook lives on the wrapped container only
    assert not hasattr(block.gate, "_kq_layer_shed")
    # Confirmation banners are verbose-only.
    assert "layer-shed 0.1" not in capsys.readouterr().out
    import gmlx.load.loadlog as loadlog
    monkeypatch.setattr(loadlog, "_LAST_VERBOSE", True)
    assert install_moe_layer_shed(model, 0.1) == 1
    assert "layer-shed 0.1" in capsys.readouterr().out


class _CoveringFeeder:
    """A decode feeder that covers the layer and serves each expert from
    its own slot, recording what it stages."""

    def __init__(self):
        self.stage_calls = []
        self._layer_shed_n = 0

    def forward_start(self, li):
        pass

    def repay_at(self, li, ids, ring_ok=False):
        return False

    def ensure_wired(self):
        pass

    def covers(self, li):
        return True

    def slot_count(self, li):
        return 1 << 16

    def shed_misses(self, li, ids, scores, keep_mass):
        return None

    def stage(self, li, ids):
        self.stage_calls.append(np.array(ids).reshape(-1).tolist())
        return ids.astype(np.uint32)

    def wedged_at(self, li):
        return False

    @contextmanager
    def swapped(self, li):
        yield


def _block_out(block, x, rows=None):
    """The block's output, with the expert container's per-expert rows
    multiplied by ``rows(indices, scores)`` when given."""
    c = block.switch_mlp
    real = type(c).__call__
    if rows is not None:
        def masked(self, x_, inds, *a, **k):
            return real(self, x_, inds, *a, **k) * rows(inds, a)[..., None]

        type(c).__call__ = masked
    try:
        out = block(x)
        mx.eval(out)
        return np.array(out)
    finally:
        type(c).__call__ = real


def test_a_shed_layer_runs_what_its_block_keeps(monkeypatch, capsys):
    """One decode token through real blocks, the layer shed. Qwen3-Next
    keeps its shared expert only. Muse Spark has none and hands the
    experts its routing scores, so it runs its top expert alone. Qwen3 MoE
    has none and keeps its scores, so the lever leaves it alone."""
    from gmlx._vendor.mlx_lm.models.qwen3_moe import Qwen3MoeSparseMoeBlock
    from gmlx._vendor.mlx_lm.models.qwen3_next import Qwen3NextSparseMoeBlock

    from gmlx.stream import expert_streaming
    from gmlx.stream.moe_experts import install_moe_layer_shed

    monkeypatch.setattr(
        mx, "device_info", lambda: {"max_recommended_working_set_size": 1024}
    )
    monkeypatch.setattr(expert_streaming, "_UNSHED_NOTED", False)
    args = SimpleNamespace(
        hidden_size=32, moe_intermediate_size=64, intermediate_size=64,
        num_experts=4, num_experts_per_tok=2, norm_topk_prob=True,
        shared_expert_intermediate_size=64)

    def top_slot(inds, a):
        return (mx.arange(inds.shape[-1])
                == mx.argmax(a[0], axis=-1)[..., None]).astype(mx.float32)

    for name, block in (("qwen3_next", Qwen3NextSparseMoeBlock(args)),
                        ("muse_spark", _spark_block()),
                        ("qwen3_moe", Qwen3MoeSparseMoeBlock(args))):
        mx.random.seed(5)
        if name != "muse_spark":
            block.switch_mlp = _kquant_glu()
        mx.eval(block.parameters())
        install_expert_streaming(_shell(block))
        df = _CoveringFeeder()
        object.__setattr__(block.switch_mlp, "_kq_decode_feeder", df)
        x = mx.random.normal((1, 1, 48 if name == "muse_spark" else 32))
        full = _block_out(block, x)
        if name == "muse_spark":
            want = _block_out(block, x, top_slot)
        elif name == "qwen3_next":
            want = _block_out(block, x, lambda inds, a: mx.zeros(inds.shape))
        else:
            want = full
        hooked = install_moe_layer_shed(_shell(block), 0.999999)
        assert hooked == (name != "qwen3_moe"), name
        df.stage_calls.clear()
        shed = _block_out(block, x)
        object.__setattr__(block.switch_mlp, "_kq_layer_shed", None)
        assert np.allclose(shed, want, atol=1e-5), name
        if name == "qwen3_moe":
            assert df._layer_shed_n == 0
            assert "no effect" in capsys.readouterr().out
        else:
            assert df._layer_shed_n == 1
            assert not np.allclose(shed, full, atol=1e-3), name
            staged = [len(c) for c in df.stage_calls]
            assert staged == ([1] if name == "muse_spark" else []), name


def test_layer_shed_sees_a_dense_or_shared_mlp_beside_the_experts(
        monkeypatch):
    """Real decoder layers. Gemma 4 runs a dense MLP beside its experts and
    Granite MoE Hybrid a shared MLP, both outside the experts' own block:
    a shed layer keeps that path. Muse Spark and Qwen3 MoE layers have only
    attention beside the MoE block: a shed Muse Spark layer keeps its top
    expert, and a Qwen3 MoE block, which keeps its scores, is not
    hooked."""
    import dataclasses
    import json
    from pathlib import Path

    from gmlx._vendor.mlx_lm.models import granitemoehybrid as granite
    from gmlx._vendor.mlx_lm.models import qwen3_moe
    from gmlx.models.gemma4.language import DecoderLayer as Gemma4Layer
    from gmlx.models.gemma4.language import TextConfig
    from gmlx.models.muse_spark import model as spark
    from gmlx.stream.moe_experts import install_moe_layer_shed

    monkeypatch.setattr(
        mx, "device_info", lambda: {"max_recommended_working_set_size": 1024}
    )
    fixture = (Path(__file__).parents[1] / "fixtures"
               / "gemma4_26b_a4b_it_config.json")
    tc = json.loads(fixture.read_text())["text_config"]
    tc.update(hidden_size=32, intermediate_size=64, moe_intermediate_size=64,
              num_experts=4, top_k_experts=2, num_attention_heads=2,
              num_key_value_heads=1, num_global_key_value_heads=1,
              head_dim=16, global_head_dim=16, num_hidden_layers=1,
              num_kv_shared_layers=0, layer_types=["sliding_attention"],
              vocab_size=64, hidden_size_per_layer_input=0)
    fields = {f.name for f in dataclasses.fields(TextConfig)}
    gemma = Gemma4Layer(
        TextConfig(**{k: v for k, v in tc.items() if k in fields}), 0)
    gran = granite.GraniteMoeHybridLayer(granite.ModelArgs(
        model_type="granitemoehybrid", vocab_size=64, hidden_size=32,
        intermediate_size=64, num_hidden_layers=1,
        max_position_embeddings=128, num_attention_heads=2,
        num_key_value_heads=1, attention_bias=False,
        embedding_multiplier=1.0, attention_multiplier=1.0,
        logits_scaling=1.0, residual_multiplier=1.0,
        layer_types=["attention"], rms_norm_eps=1e-5, rope_theta=1e4,
        num_local_experts=4, num_experts_per_tok=2,
        shared_intermediate_size=64), "attention")
    muse = spark.DecoderLayer(spark.ModelArgs(
        model_type="muse_spark", hidden_size=48, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        vocab_size=32, layer_types=["sliding_attention"], num_experts=4,
        num_experts_per_tok=2, moe_intermediate_size=64,
        moe_latent_size=32), True)
    qwen = qwen3_moe.Qwen3MoeDecoderLayer(qwen3_moe.ModelArgs(
        model_type="qwen3_moe", hidden_size=32, num_hidden_layers=1,
        intermediate_size=64, num_attention_heads=2, num_experts=4,
        num_experts_per_tok=2, decoder_sparse_step=1, mlp_only_layers=[],
        moe_intermediate_size=64, rms_norm_eps=1e-5, vocab_size=64,
        num_key_value_heads=1, head_dim=16, rope_theta=1e4,
        tie_word_embeddings=False, max_position_embeddings=128,
        norm_topk_prob=True), 0)
    for layer, owner, attr, top1 in (
            (gemma, gemma.experts, "switch_glu", False),
            (gran, gran.block_sparse_moe, "switch_mlp", False),
            (muse, muse.mlp, "switch_mlp", True),
            (qwen, qwen.mlp, "switch_mlp", True)):
        glu = _kquant_glu()
        setattr(owner, attr, glu)

        class _Shell:
            pass

        model = _Shell()
        model.layers = [layer]
        model.parameters = lambda glu=glu: {"glu": glu.parameters()}
        install_expert_streaming(model)
        glu = getattr(owner, attr)
        if layer is qwen:
            assert install_moe_layer_shed(model, 0.1) == 0
            assert not hasattr(glu, "_kq_layer_shed")
            continue
        assert install_moe_layer_shed(model, 0.1) == 1, type(layer)
        assert glu._kq_shed_keeps_top1 is top1, type(layer)


def _k3_block():
    from gmlx.models.kimi_k3 import KimiK3MoE

    args = SimpleNamespace(
        hidden_size=32,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=64,
        routed_expert_hidden_size=None,
        has_routed_norm=False,
        rms_norm_eps=1e-5,
        num_shared_experts=0,
        situ_beta=1.0,
        situ_linear_beta=0.0,
        moe_renormalize=True,
        routed_scaling_factor=1.0,
    )
    block = KimiK3MoE(args)
    block.switch_mlp = _kquant_glu()
    return block


def test_kimi_k3_block_hooked_via_callback(monkeypatch):
    """KimiK3MoE routes with an inline plain-Linear top-k, so _install
    targets the block itself and its forward calls back into
    _apply_expert_controls at the selection seam."""
    mx.random.seed(7)
    block = _k3_block()
    model = _shell(block)
    monkeypatch.setattr(
        mx, "device_info", lambda: {"max_recommended_working_set_size": 1024}
    )
    x = mx.random.normal((1, 5, 32))
    mx.eval(block.parameters())
    ref = np.array(block(x))

    install_expert_streaming(model)
    base = np.array(block(x))
    # Streamed experts run on the CPU stream; vs the GPU dense reference the
    # q8_0 kernels differ by one fp16 ulp, so the cross-device check is loose.
    assert np.allclose(base, ref, atol=1e-2)
    assert install_moe_expert_probe(model) == 1
    probe = block._kq_expert_probe
    out = np.array(block(x))
    assert out.shape == ref.shape
    assert np.allclose(out, base, atol=1e-5)  # probe is lossless
    probe._flush()
    assert probe._buckets["prefill"].tokens == 5

    assert install_moe_expert_mass(model, 1e-6) == 1
    rec = _SwitchRecorder(block.switch_mlp)
    block.switch_mlp = rec
    out = np.array(block(x))
    assert out.shape == ref.shape
    low = rec.seen[-1]
    assert low.shape == (1, 5, 2)
    for t in range(low.shape[1]):
        assert len(set(low[0, t].tolist())) == 1  # collapsed to top-1


def test_valid_expert_ids_replaces_out_of_range_rows(capsys):
    from gmlx.stream.expert_streaming import _valid_expert_ids

    mod = SimpleNamespace(gate_proj=SimpleNamespace(num_experts=8), _kq_li=3)
    ok = mx.array(np.array([[1, 5, 7], [0, 2, 4]], dtype=np.uint32))
    ids, out = _valid_expert_ids(mod, ok)
    assert out is ok
    assert ids.tolist() == [[1, 5, 7], [0, 2, 4]]

    bad = mx.array(np.array([[1, 5, 7], [0xFFFFFFFF] * 3], dtype=np.uint32))
    ids, out = _valid_expert_ids(mod, bad)
    assert ids.tolist() == [[1, 5, 7], [0, 1, 2]]
    assert out is not bad and out.dtype == mx.uint32
    assert np.array(out).tolist() == ids.tolist()
    assert "layer 3: router found no expert for 1 token(s)" in capsys.readouterr().err


def _spark_block():
    from gmlx.models.muse_spark.model import ModelArgs, MuseSparkMoE

    args = ModelArgs(
        model_type="muse_spark",
        hidden_size=48,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        vocab_size=32,
        layer_types=["sliding_attention"],
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=64,
        moe_latent_size=32,
    )
    block = MuseSparkMoE(args)
    block.switch_mlp = _kquant_glu()
    return block


def test_muse_spark_block_hooked_via_callback(monkeypatch):
    """MuseSparkMoE routes with an inline plain-Linear top-k, so _install
    targets the block itself and its forward calls back into
    _apply_expert_controls after selection. The top-k override reads the
    int on the block."""
    from gmlx.stream.moe_experts import _hook_target

    mx.random.seed(8)
    block = _spark_block()
    assert _hook_target(block) is block
    model = _shell(block)
    monkeypatch.setattr(
        mx, "device_info", lambda: {"max_recommended_working_set_size": 1024}
    )
    x = mx.random.normal((1, 5, 48))
    mx.eval(block.parameters())
    ref = np.array(block(x))

    install_expert_streaming(model)
    base = np.array(block(x))
    # Streamed experts run on the CPU stream and differ from the GPU
    # reference by about one bf16 ulp. The per-expert norm brings outputs to
    # unit scale, so the bound is relative.
    assert np.abs(base - ref).max() <= 3e-2 * np.abs(ref).max()
    assert install_moe_expert_probe(model) == 1
    probe = block._kq_expert_probe
    out = np.array(block(x))
    assert np.allclose(out, base, atol=1e-5)  # probe is lossless
    probe._flush()
    assert probe._buckets["prefill"].tokens == 5

    assert install_moe_expert_mass(model, 1e-6) == 1
    rec = _SwitchRecorder(block.switch_mlp)
    block.switch_mlp = rec
    out = np.array(block(x))
    assert out.shape == ref.shape
    low = rec.seen[-1]
    assert low.shape == (1, 5, 2)
    for t in range(low.shape[1]):
        assert len(set(low[0, t].tolist())) == 1  # collapsed to top-1


def test_muse_spark_top_k_override_hits_the_block(monkeypatch):
    mx.random.seed(9)
    block = _spark_block()
    model = _shell(block)
    monkeypatch.setattr(
        mx, "device_info", lambda: {"max_recommended_working_set_size": 1024}
    )
    install_expert_streaming(model)
    assert install_moe_experts_override(model, 1) == 1
    assert block.num_experts_per_tok == 1
    rec = _SwitchRecorder(block.switch_mlp)
    block.switch_mlp = rec
    mx.eval(block(mx.random.normal((1, 3, 48))))
    assert rec.seen[-1].shape == (1, 3, 1)
