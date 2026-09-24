"""Route record and replay through the expert-controls seam (CPU fake
blocks): replaying a forward's own routes reproduces it bit for bit, a
swapped id changes the output, chunk offsets line up, and the installers
refuse what they cannot replay."""

from __future__ import annotations

import importlib
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


# mlx-lm group_expert_select gates beyond deepseek_v3 and glm4_moe:
# module -> the MoE block class that owns the gate.
_SIGMOID_GATE_BLOCKS = {
    "deepseek_v32": "DeepseekV32MoE",
    "exaone_moe": "MoE",
    "glm4_moe_lite": "Glm4MoeLiteMoE",
    "mimo_v2_flash": "MoE",
}


def _sigmoid_gate_cfg(n_group=1, topk_group=1, norm=True, k=4):
    return SimpleNamespace(
        num_experts_per_tok=k, norm_topk_prob=norm, n_routed_experts=8,
        num_experts=8, routed_scaling_factor=2.5, n_group=n_group,
        topk_group=topk_group, hidden_size=16, moe_intermediate_size=32,
        n_shared_experts=None, num_shared_experts=0, topk_method="noaux_tc",
    )


def _randomize_gate(gate):
    # MoEGate starts at zeros, which ties every score
    gate.weight = mx.random.normal(gate.weight.shape)
    gate.e_score_correction_bias = mx.random.normal(gate.e_score_correction_bias.shape)


def _dsv32_fp32_router(module, monkeypatch):
    from gmlx.upstream.dsv32_patches import _patch_dsv32_moe_gate_fp32

    monkeypatch.setenv("GMLX_DSV32_GATE_FP32", "1")
    _patch_dsv32_moe_gate_fp32(module)


@pytest.mark.parametrize("arch", sorted(_SIGMOID_GATE_BLOCKS))
def test_mlx_lm_sigmoid_gate_adapters_match_select(arch):
    """Each adapter gives the gate's own weights at the gate's own ids, bit
    for bit, across group masks, renormalization, k=1 and both dtypes."""
    MoEGate = importlib.import_module(f"mlx_lm.models.{arch}").MoEGate
    for dtype in (mx.float32, mx.bfloat16):
        for n_group, topk_group, norm, k in ((1, 1, True, 2), (2, 1, False, 2), (4, 2, True, 2),
                                             (1, 1, True, 1)):
            gate = MoEGate(_sigmoid_gate_cfg(n_group, topk_group, norm, k))
            mx.random.seed(20)
            _randomize_gate(gate)
            gate.weight = gate.weight.astype(dtype)
            x = mx.random.normal((2, 3, 16)).astype(dtype)
            inds, weights = gate(x)
            fn = _gate_weights_fn(gate)
            assert fn is not None, arch
            mx.eval(inds, weights)
            assert np.array_equal(np.array(fn(gate, x, inds)), np.array(weights)), (
                arch, dtype, n_group, topk_group, norm, k)


def test_dsv32_gate_adapter_follows_the_fp32_router_patch(monkeypatch):
    """glm-dsa GGUFs load deepseek_v32 with the fp32 router patch on: the
    adapter runs the router matmul in fp32 on a flagged gate and in the
    model dtype on an unflagged one, matching the gate either way."""
    from mlx_lm.models.deepseek_v32 import MoEGate

    class _Holder(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate = MoEGate(_sigmoid_gate_cfg())

    holder = _Holder()
    mx.random.seed(21)
    _randomize_gate(holder.gate)
    holder.set_dtype(mx.bfloat16)
    _dsv32_fp32_router(holder, monkeypatch)
    gate = holder.gate
    assert gate._dsv32_gate_fp32
    x = mx.random.normal((2, 5, 16)).astype(mx.bfloat16)
    fn = _gate_weights_fn(gate)
    at = {}
    for flag in (True, False):
        gate._dsv32_gate_fp32 = flag
        inds, weights = gate(x)
        mx.eval(inds, weights)
        assert np.array_equal(np.array(fn(gate, x, inds)), np.array(weights)), flag
        at[flag] = np.array(fn(gate, x, mx.zeros_like(inds) + mx.arange(4, dtype=inds.dtype)))
    # the two branches differ on bf16, so both were exercised
    assert not np.array_equal(at[True], at[False])


@pytest.mark.parametrize("arch,fp32", [(a, False) for a in sorted(_SIGMOID_GATE_BLOCKS)]
                         + [("deepseek_v32", True)])
def test_mlx_lm_sigmoid_gate_blocks_replay(arch, fp32, monkeypatch):
    """Replaying a block's own recorded routes reproduces its unhooked
    forward bit for bit, in float32 and (GPU only, the CPU gather_mm is
    float32 only) in bfloat16, and a swapped id moves that position only."""
    mod = importlib.import_module(f"mlx_lm.models.{arch}")
    mx.random.seed(22)
    x = mx.random.normal((1, 5, 16))
    block = getattr(mod, _SIGMOID_GATE_BLOCKS[arch])(_sigmoid_gate_cfg())
    _randomize_gate(block.gate)
    if fp32:
        _dsv32_fp32_router(block, monkeypatch)
        assert block.gate._dsv32_gate_fp32
    mx.eval(block.parameters())
    stock = np.array(block(x))
    _check_block(block, x, 8)
    assert np.array_equal(np.array(block(x)), stock), arch

    if mx.default_device() != mx.gpu:
        return
    block.set_dtype(mx.bfloat16)
    xb = x.astype(mx.bfloat16)
    stock = np.array(block(xb).astype(mx.float32))
    model = _shell(block)
    rec = install_moe_route_record(model)
    mx.eval(block(xb))
    routes = rec.take()
    clear_moe_route_controls(model)
    install_moe_route_replay(model, RouteReplay(routes, rec.layers))
    assert np.array_equal(np.array(block(xb).astype(mx.float32)), stock), arch
    clear_moe_route_controls(model)


def _deepseek_v2_block():
    from mlx_lm.models.deepseek_v2 import DeepseekV2MoE

    cfg = SimpleNamespace(
        num_experts_per_tok=2, n_routed_experts=8, routed_scaling_factor=1.0,
        topk_method="greedy", n_group=1, topk_group=1, hidden_size=16,
        moe_intermediate_size=32, n_shared_experts=None,
    )
    return DeepseekV2MoE(cfg)


def test_replay_unsupported_names_a_gate_without_an_adapter():
    """deepseek_v2's softmax gate has no weights adapter: the query names it
    by module and class without swapping it, before and after recording."""
    from gmlx.stream.moe_routes import replay_unsupported

    block = _deepseek_v2_block()
    model = _shell(block)
    want = [(0, "mlx_lm.models.deepseek_v2.MoEGate")]
    assert replay_unsupported(model) == want
    assert type(block.gate).__name__ == "MoEGate"
    rec = install_moe_route_record(model)
    assert rec.layers == [0]
    clear_moe_route_controls(model)
    assert replay_unsupported(model) == want
    with pytest.raises(ValueError, match="layer 0 MoEGate_ExpertCtl"):
        install_moe_route_replay(model, RouteReplay(np.zeros((1, 5, 2)), [0]))


def test_replay_unsupported_agrees_with_the_installer():
    """The query lists a block exactly when install_moe_route_replay
    refuses it, over every block shape the seam knows."""
    from gmlx.stream.moe_routes import replay_unsupported

    blocks = list(_arch_fixtures()) + [
        _gptoss_block(), _k3_block(), _deepseek_v2_block(),
        _Block(_kquant_glu(), _WeightedGate()), _Block(_kquant_glu(), _TupleGate()),
    ]
    for arch, cls in _SIGMOID_GATE_BLOCKS.items():
        blocks.append(getattr(importlib.import_module(f"mlx_lm.models.{arch}"), cls)(_sigmoid_gate_cfg()))
    seen = set()
    for block in blocks:
        model = _shell(block)
        listed = replay_unsupported(model)
        try:
            install_moe_route_replay(model, RouteReplay(np.zeros((1, 5, 2)), [0]))
            refused = False
        except ValueError as e:
            assert "unsupported on MoE block" in str(e)
            refused = True
        clear_moe_route_controls(model)
        assert bool(listed) == refused, type(block).__name__
        seen.add(refused)
    assert seen == {True, False}


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

    from gmlx.stream.moe_routes import replay_unsupported

    model = _shell(_LinearGateBlock(_kquant_glu()))
    assert replay_unsupported(model) == [(0, "_LinearGateBlock")]
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


def test_fused_block_records_rows_apart(monkeypatch):
    """The loader's fused MoE block routes over flat (tokens, k) ids; the
    seam has to see the batch shape or a two-row chunk records as one row
    and the teacher pass refuses the routes."""
    import gmlx.load.modules as lm

    d, E, k, B, L = 16, 8, 4, 2, 3

    class _Base(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate = nn.Linear(d, E, bias=False)
            self.shared_expert_gate = nn.Linear(d, 1, bias=False)
            self.top_k = k
            self.norm_topk_prob = True
            proj = lambda: SimpleNamespace(weight=mx.zeros((1,)), kquant_type="q8_0")  # noqa: E731
            self.switch_mlp = SimpleNamespace(gate_proj=proj(), up_proj=proj(), down_proj=proj())
            self.shared_expert = SimpleNamespace(gate_proj=proj(), up_proj=proj(), down_proj=proj())

    kq = SimpleNamespace(
        moe_glu_gather_shexp_kq=lambda xf, *a, **kw: mx.zeros((xf.shape[0], k + 1, 8)),
        gather_qmv_mix_kq=lambda h, *a, **kw: mx.zeros((h.shape[0], d), dtype=mx.bfloat16),
    )
    cls = lm._make_fused_block(_Base, SimpleNamespace(kq=kq))
    block = cls()
    block.eval()  # the fused path serves inference only
    mx.eval(block.parameters())
    monkeypatch.setattr(lm, "_kq_fused_device_ok", lambda *mods: True)
    model = _shell(block)
    rec = install_moe_route_record(model)
    x = mx.random.normal((B, L, d)).astype(mx.bfloat16)
    y = block(x)
    mx.eval(y)
    assert y.shape == (B, L, d)
    routes = rec.take()
    clear_moe_route_controls(model)
    assert routes.shape == (1, B, L, k)


def test_recorded_ids_do_not_hold_the_selection_buffers():
    """The ids a router hands the seam are a view of its [..., E] sort
    buffer. The recorder keeps a copy tied into the forward that consumes
    the ids, so the buffers go when each layer's forward is evaluated
    instead of staying alive for every layer until take."""
    from gmlx.stream.moe_experts import _apply_expert_controls

    B, T, E, k, n_layers = 1, 8192, 384, 8, 3
    rec = RouteRecorder()
    rec.layers = list(range(n_layers))
    mx.eval(mx.zeros((1,)))
    mx.synchronize()
    mx.clear_cache()
    base = mx.get_active_memory()
    for li in range(n_layers):
        g = mx.random.normal((B, T, E))
        inds = mx.argpartition(g, kth=-k, axis=-1)[..., -k:]
        w = mx.take_along_axis(g, inds, axis=-1)
        inds, w = _apply_expert_controls(SimpleNamespace(_kq_li=li, _kq_route_record=rec), inds, w)
        mx.eval((w * inds.astype(mx.float32)).sum())
        del g, inds, w
    # A GPU command buffer releases its inputs in its completion handler,
    # after mx.eval returns, so the last layer's buffers count as active
    # until the stream is synchronized.
    mx.synchronize()
    mx.clear_cache()
    held = mx.get_active_memory() - base
    kept = n_layers * B * T * k * 4
    assert held < kept + (1 << 20), f"{held / 1e6:.1f} MB held for {kept / 1e6:.1f} MB of ids"
    out = rec.take()
    assert out.shape == (n_layers, B, T, k) and int(out.max()) < E


def test_replay_refuses_routes_past_the_expert_count():
    routes = np.zeros((1, 1, 5, 4), dtype=np.int32)
    routes[0, 0, 2] = [0, 1, 2, 9]
    assert RouteReplay(routes, [0], n_experts=10).rows == 1
    with pytest.raises(ValueError, match="routes carry expert id 9, the model has 8 experts"):
        RouteReplay(routes, [0], n_experts=8)
