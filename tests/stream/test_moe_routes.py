"""Route record and replay through the expert-controls seam (CPU fake
blocks): replaying a forward's own routes reproduces it bit for bit, a
swapped id changes the output, chunk offsets line up, and the installers
refuse what they cannot replay."""

from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from gmlx.stream.moe_routes import (
    RouteRecorder,
    RouteReplay,
    _gate_weights_fn,
    clear_moe_route_controls,
    install_moe_route_record,
    install_moe_route_replay,
    moe_layers,
)
from test_moe_experts import _arch_fixtures, _Block, _k3_block, _shell, _TupleGate
from test_offload import _kquant_glu


def _gptoss_block():
    from mlx_lm.models.gpt_oss import MLPBlock

    cfg = SimpleNamespace(
        hidden_size=16, intermediate_size=32, num_local_experts=8,
        num_experts_per_tok=4,
    )
    return MLPBlock(cfg)


def _record(model, x):
    rec = install_moe_route_record(model)
    ref = np.array(model.layers[0].mlp(x))
    routes = rec.take()
    clear_moe_route_controls(model)
    return ref, routes, rec.layers


def _swap_one(routes, pos, n_experts):
    """Return a copy with position ``pos`` of layer 0 row 0 routed to ids
    outside its recorded set."""
    out = routes.copy()
    k = out.shape[-1]
    live = set(out[0, 0, pos].tolist())
    fresh = [e for e in range(n_experts) if e not in live][:k]
    assert len(fresh) == k
    out[0, 0, pos] = fresh
    return out


def _check_block(block, x, n_experts):
    model = _shell(block)
    mx.eval(block.parameters())
    ref, routes, layers = _record(model, x)
    assert layers == [0]
    assert routes.shape == (1, 1, x.shape[1], routes.shape[-1])
    name = type(block).__name__

    # replaying the live routes reproduces the forward bit for bit
    assert install_moe_route_replay(model, RouteReplay(routes, layers)) == 1
    assert np.array_equal(np.array(block(x)), ref), name
    clear_moe_route_controls(model)

    # a swapped id changes that position only
    swapped = _swap_one(routes, 2, n_experts)
    install_moe_route_replay(model, RouteReplay(swapped, layers))
    out = np.array(block(x))
    assert not np.array_equal(out[0, 2], ref[0, 2]), name
    assert np.array_equal(np.delete(out, 2, axis=1), np.delete(ref, 2, axis=1)), name
    clear_moe_route_controls(model)

    # chunk offsets line up
    replay = RouteReplay(routes, layers)
    install_moe_route_replay(model, replay)
    a = np.array(block(x[:, :2]))
    replay.advance(2)
    b = np.array(block(x[:, 2:]))
    assert np.array_equal(np.concatenate([a, b], axis=1), ref), name
    clear_moe_route_controls(model)

    # controls cleared: the swapped forward is the stock forward
    assert np.array_equal(np.array(block(x)), ref), name


def test_inline_blocks_replay():
    mx.random.seed(11)
    x = mx.random.normal((1, 5, 16))
    for block in _arch_fixtures():
        _check_block(block, x, 8)


def test_gptoss_block_replay():
    mx.random.seed(12)
    x = mx.random.normal((1, 5, 16))
    block = _gptoss_block()
    mx.eval(block.parameters())
    stock = np.array(block(x))
    _check_block(block, x, 8)
    assert type(block).__name__ == "MLPBlock_ExpertCtl"
    assert np.array_equal(np.array(block(x)), stock)


class _WeightedGate(_TupleGate):
    def _kq_route_weights(self, x, inds):
        gates = mx.softmax(x @ self.weight.T, axis=-1, precise=True)
        return mx.take_along_axis(gates, inds, axis=-1)


def test_gate_block_replay_via_adapter_method():
    mx.random.seed(13)
    x = mx.random.normal((1, 5, 32))
    block = _Block(_kquant_glu(), _WeightedGate())
    _check_block(block, x, 4)
    assert type(block.gate).__name__ == "_WeightedGate_ExpertCtl"


def test_gate_without_adapter_records_but_refuses_replay():
    mx.random.seed(14)
    x = mx.random.normal((1, 5, 32))
    block = _Block(_kquant_glu(), _TupleGate())
    model = _shell(block)
    mx.eval(block.parameters())
    ref, routes, layers = _record(model, x)
    assert routes.shape == (1, 1, 5, 2)
    with pytest.raises(ValueError, match="unsupported on MoE block"):
        install_moe_route_replay(model, RouteReplay(routes, layers))
    assert np.array_equal(np.array(block(x)), ref)


def test_mlx_lm_deepseek_gate_adapter_matches_select():
    from mlx_lm.models.deepseek_v3 import MoEGate

    for n_group, topk_group, norm in ((1, 1, True), (2, 1, False), (4, 2, True)):
        cfg = SimpleNamespace(
            num_experts_per_tok=2, norm_topk_prob=norm, n_routed_experts=8,
            routed_scaling_factor=2.5, n_group=n_group, topk_group=topk_group,
            hidden_size=16, topk_method="noaux_tc",
        )
        gate = MoEGate(cfg)
        mx.random.seed(15)
        gate.weight = mx.random.normal((8, 16))
        gate.e_score_correction_bias = mx.random.normal((8,))
        x = mx.random.normal((2, 3, 16))
        inds, weights = gate(x)
        fn = _gate_weights_fn(gate)
        assert fn is not None
        mx.eval(inds, weights)
        assert np.array_equal(np.array(fn(gate, x, inds)), np.array(weights)), (
            n_group, topk_group, norm)


def test_kimi_k3_replay():
    mx.random.seed(16)
    x = mx.random.normal((1, 5, 32))
    _check_block(_k3_block(), x, 4)


def test_hunyuan_replay():
    from mlx_lm.models.hunyuan import MoeBlock

    from gmlx.load.loader import _patch_hunyuan_norm_topk

    args = SimpleNamespace(
        hidden_size=16, intermediate_size=32, use_mixed_mlp_moe=False,
        num_shared_expert=1, num_experts=8, moe_topk=4,
        moe_intermediate_size=None,
    )

    class _Holder(nn.Module):
        def __init__(self):
            super().__init__()
            self.blk = MoeBlock(args)

    holder = _Holder()
    mx.eval(holder.parameters())
    _patch_hunyuan_norm_topk(holder)
    mx.random.seed(17)
    x = mx.random.normal((1, 5, 16))
    _check_block(holder.blk, x, 8)


def test_replay_refuses_unsupported_block(capsys):
    class _LinearGateBlock(nn.Module):
        def __init__(self, glu):
            super().__init__()
            self.gate = nn.Linear(32, 4, bias=False)
            self.switch_mlp = glu

        def __call__(self, x):
            g = self.gate(x)
            inds = mx.argpartition(-g, kth=1, axis=-1)[..., :2]
            w = mx.take_along_axis(g, inds, axis=-1)
            return (self.switch_mlp(x, inds) * w[..., None]).sum(axis=-2)

    model = _shell(_LinearGateBlock(_kquant_glu()))
    rec = install_moe_route_record(model)
    assert rec.layers == []
    assert "skipped unsupported block(s): _LinearGateBlock" in capsys.readouterr().out
    with pytest.raises(ValueError, match="layer 0 _LinearGateBlock"):
        install_moe_route_replay(model, RouteReplay(np.zeros((1, 5, 2)), [0]))


def test_replay_shape_checks():
    mx.random.seed(18)
    x = mx.random.normal((1, 5, 16))
    block = _arch_fixtures()[0]
    model = _shell(block)
    mx.eval(block.parameters())
    _, routes, layers = _record(model, x)

    with pytest.raises(ValueError, match="do not match the model's MoE layers"):
        install_moe_route_replay(model, RouteReplay(routes, [3]))
    with pytest.raises(ValueError, match="carry 1 layers, the layer list 2"):
        RouteReplay(routes, [0, 1])

    replay = RouteReplay(routes, layers, offset=3)
    install_moe_route_replay(model, replay)
    with pytest.raises(ValueError, match="runs past the recorded length"):
        mx.eval(block(x))
    replay.offset = 0
    with pytest.raises(ValueError, match="recorded 1 rows"):
        mx.eval(block(mx.concatenate([x, x], axis=0)))
    with pytest.raises(ValueError, match="recorded k=4"):
        replay.ids_for(0, mx.zeros((1, 5, 3), dtype=mx.uint32))
    clear_moe_route_controls(model)


def test_batched_rows_and_two_layers():
    """Routes stack in layer order and keep the batch axis: a two-row
    recording replays on the same two rows, and a model with two MoE layers
    records one entry per layer."""
    mx.random.seed(19)
    blocks = _arch_fixtures()[:2]

    class _Layer(nn.Module):
        def __init__(self, blk):
            super().__init__()
            self.mlp = blk

    class _Shell:
        pass

    model = _Shell()
    model.layers = [_Layer(b) for b in blocks]
    for b in blocks:
        mx.eval(b.parameters())
    assert moe_layers(model) == [0, 1]

    x = mx.random.normal((2, 5, 16))
    rec = install_moe_route_record(model)
    refs = [np.array(b(x)) for b in blocks]
    routes = rec.take()
    assert rec.layers == [0, 1]
    assert routes.shape == (2, 2, 5, 4)
    clear_moe_route_controls(model)

    assert install_moe_route_replay(model, RouteReplay(routes, [0, 1])) == 2
    for b, ref in zip(blocks, refs):
        assert np.array_equal(np.array(b(x)), ref)
    clear_moe_route_controls(model)


def test_recorder_take_resets_and_joins_chunks():
    rec = RouteRecorder()
    rec.layers = [4]
    rec.record(4, mx.array(np.arange(8).reshape(1, 2, 4)))
    rec.record(4, mx.array(np.arange(8, 20).reshape(1, 3, 4)))
    out = rec.take()
    assert out.shape == (1, 1, 5, 4)
    assert out.dtype == np.int32
    assert out[0, 0, 4].tolist() == [16, 17, 18, 19]
    assert rec.take().shape == (0, 1, 0, 1)
