"""An adapter on a projection that a fused MoE decode path reads in place of
calling it: the qwen3-next fused block and the gemma4 fused experts call
their adapted members in a forward where one of the adapter's slots is on
and stay fused otherwise, the hy_v3 shared-expert fold drops its stamp and
leaves the shared expert to the caller, and a router concat built before
the install is rebuilt. Each fused result must match the stock forward
with the same adapter. Real kq kernels on the GPU."""
from __future__ import annotations

import os
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from gmlx import lora_rows
import gmlx.load.adapter as adapter
import gmlx.load.modules as modules

if os.environ.get("KQUANT_FORCE_CPU") or not mx.metal.is_available():
    pytest.skip("the fused MoE kernels need a real GPU and a GPU default device",
                allow_module_level=True)

D, INTER, E, K = 256, 256, 4, 2


@pytest.fixture(autouse=True)
def _static():
    lora_rows.configure("static", 1)
    yield
    lora_rows.configure("static", 1)


class _Shell(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.mlp = block


def _quantized(leaf, shape, rng):
    import mlx_kquant as kq

    w = mx.array((rng.standard_normal(shape) * 0.05).astype(np.float32))
    leaf.weight, leaf.scales = kq.quantize(w, "q8_0")
    return leaf


def _quantize_glu(glu, rng):
    from mlx_kquant.nn import KQuantSwitchLinear

    for name, (o, i) in (("gate_proj", (INTER, D)), ("up_proj", (INTER, D)),
                         ("down_proj", (D, INTER))):
        setattr(glu, name, _quantized(
            KQuantSwitchLinear(E, o, i, False, "q8_0"), (E, o, i), rng))


def _quantize_mlp(mlp, rng):
    from mlx_kquant.nn import KQuantLinear

    for name, (o, i) in (("gate_proj", (INTER, D)), ("up_proj", (INTER, D)),
                         ("down_proj", (D, INTER))):
        setattr(mlp, name, _quantized(KQuantLinear(i, o, False, "q8_0"), (o, i), rng))


def _qwen3next_block(seed):
    from gmlx._vendor.mlx_lm.models.qwen3_next import Qwen3NextSparseMoeBlock

    mx.random.seed(seed)
    rng = np.random.default_rng(seed)
    block = Qwen3NextSparseMoeBlock(SimpleNamespace(
        hidden_size=D, moe_intermediate_size=INTER,
        shared_expert_intermediate_size=INTER, norm_topk_prob=True,
        num_experts=E, num_experts_per_tok=K))
    _quantize_glu(block.switch_mlp, rng)
    _quantize_mlp(block.shared_expert, rng)
    block.eval()
    return block


def _qwen3next_shell(seed=0):
    shell = _Shell(_qwen3next_block(seed))
    modules.install_fused_moe_glu(shell)
    if type(shell.mlp).__name__ != "_FusedKQuantMoeBlock":
        pytest.skip("the fused MoE block is not installable in this build")
    mx.eval(shell.parameters())
    return shell


class _Two(nn.Module):
    def __init__(self, b0, b1):
        super().__init__()
        self.layers = [b0, b1]

    def __call__(self, x):
        return self.layers[1](self.layers[0](x))


def _qwen3next_shell2(seed=0):
    """Two fused blocks in sequence under one parent."""
    shell = _Two(_qwen3next_block(seed), _qwen3next_block(seed + 10))
    modules.install_fused_moe_glu(shell)
    if any(type(b).__name__ != "_FusedKQuantMoeBlock" for b in shell.layers):
        pytest.skip("the fused MoE block is not installable in this build")
    mx.eval(shell.parameters())
    return shell


def _hyv3_shell(seed=0):
    from gmlx.models.hy_v3.model import MoE

    mx.random.seed(seed)
    rng = np.random.default_rng(seed)
    block = MoE(SimpleNamespace(
        hidden_size=D, expert_hidden_dim=INTER, num_experts=E,
        num_experts_per_tok=K, num_shared_experts=1, route_norm=True,
        router_scaling_factor=1.5, enable_moe_fp32_combine=False))
    _quantize_glu(block.switch_mlp, rng)
    _quantize_mlp(block.shared_mlp, rng)
    block.eval()
    shell = _Shell(block)
    modules.install_fused_moe_glu(shell)
    if modules.install_hyv3_shexp_fold(shell) != 1:
        pytest.skip("the shared-expert fold is not installable in this build")
    mx.eval(shell.parameters())
    return shell


def _install(shell, paths, seed=1, r=4, slot=0, inter=INTER):
    """A LoRA pair per path, large enough to move the block output: a dense
    pair on a linear, an expert pair on an expert stack's down_proj."""
    rng = np.random.default_rng(seed)
    mods = {}
    for path in paths:
        leaf = dict(shell.named_modules())[path]
        d_in = inter if path.endswith("down_proj") else D
        if leaf.weight.ndim == 3:
            e, out = leaf.weight.shape[:2]
            a = mx.array((rng.standard_normal((e, r, d_in)) * 0.5).astype(np.float32))
            b = mx.array((rng.standard_normal((e, out, r)) * 0.5).astype(np.float32))
            mods[path] = adapter.LoraModule(module_path=path, a=a, b=b, rank=r,
                                            scale=1.0, experts=True)
            continue
        out = leaf.weight.shape[0]
        if not hasattr(leaf, "kquant_type"):
            d_in = leaf.weight.shape[1]
        a = mx.array((rng.standard_normal((r, d_in)) * 0.5).astype(np.float32))
        b = mx.array((rng.standard_normal((out, r)) * 0.5).astype(np.float32))
        mods[path] = adapter.LoraModule(module_path=path, a=a, b=b, rank=r, scale=1.0)
    plan = adapter.LoraAdapter(alpha=float(r), arch="qwen35moe", modules=mods)
    assert modules.install_lora_adapter(shell, plan, slot=slot) == len(paths)


def _x(seed=2):
    return mx.array(np.random.default_rng(seed).standard_normal((1, 1, D)).astype(np.float32)).astype(mx.bfloat16)


def _rel(a, b):
    a = np.array(a.astype(mx.float32))
    b = np.array(b.astype(mx.float32))
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def _stock(shell, x, monkeypatch):
    with monkeypatch.context() as m:
        m.setattr(modules, "_FUSED_MOE_ENABLED", False)
        y = shell.mlp(x)
        mx.eval(y)
    return y


def test_adapted_shared_expert_reaches_the_fused_block_output(monkeypatch):
    shell = _qwen3next_shell()
    x = _x()
    base = shell.mlp(x)
    _install(shell, ["mlp.shared_expert.gate_proj", "mlp.shared_expert.down_proj"])
    y = shell.mlp(x)
    mx.eval(base, y)
    ref = _stock(shell, x, monkeypatch)
    assert _rel(base, ref) > 0.2
    assert _rel(y, ref) < 0.03


def test_router_adapter_installed_after_a_decode_reaches_the_routing(monkeypatch):
    shell = _qwen3next_shell()
    x = _x()
    base = shell.mlp(x)
    mx.eval(base)
    if not isinstance(vars(shell.mlp).get("_kq_router_cat"), mx.array):
        pytest.skip("this build does not fuse the router")
    _install(shell, ["mlp.gate", "mlp.shared_expert_gate"])
    y = shell.mlp(x)
    mx.eval(y)
    with monkeypatch.context() as m:
        m.setattr(modules, "_FUSED_MOE_BLOCK_ENABLED", False)
        ref = shell.mlp(x)
        mx.eval(ref)
    assert _rel(base, ref) > 0.2
    assert _rel(y, ref) < 0.03


def test_a_trained_student_decodes_through_the_fused_block(monkeypatch):
    from gmlx._vendor.mlx_lm.tuner.lora import LoRALinear

    from gmlx.tune.lora import prepare_lora_student

    shell = _qwen3next_shell()

    class _Model(nn.Module):
        def __init__(self, layer):
            super().__init__()
            self.layers = [layer]

    model = _Model(shell)
    assert prepare_lora_student(model, rank=4, scale=1.0) > 0
    adapted = [m for _, m in model.named_modules() if isinstance(m, LoRALinear)]
    assert any(m is shell.mlp.shared_expert.gate_proj for m in adapted)
    rng = np.random.default_rng(3)
    for m in adapted:
        m.lora_b = mx.array((rng.standard_normal(m.lora_b.shape) * 0.5).astype(np.float32))
    model.eval()
    x = _x()
    y = shell.mlp(x)
    mx.eval(y)
    assert _rel(y, _stock(shell, x, monkeypatch)) < 0.03


def test_adapted_shared_expert_drops_the_fold_and_reaches_the_output(monkeypatch):
    shell = _hyv3_shell()
    x = _x()
    base = shell.mlp(x)
    mx.eval(base)
    _install(shell, ["mlp.shared_mlp.gate_proj", "mlp.shared_mlp.down_proj"])
    y = shell.mlp(x)
    mx.eval(y)
    assert shell.mlp.switch_mlp._kq_shexp_mod is None
    ref = _stock(shell, x, monkeypatch)
    assert _rel(base, ref) > 0.2
    assert _rel(y, ref) < 0.03


# Expert adapters on fused blocks: the block calls its members in a forward
# where one of its adapter's slots is on, and folds otherwise.

EXPERT_PATHS = ["mlp.switch_mlp.down_proj", "mlp.shared_expert.down_proj"]


@pytest.fixture
def kq_calls(monkeypatch):
    """Counts the fused kernels by name and keeps each fold call's gate
    weight, so a test can tell which block folded."""
    import mlx_kquant as kq

    seen = {"fold": [], "router": 0, "mix_ns": 0}

    def wrap(name, record):
        real = getattr(kq, name)

        def spy(*a, **kw):
            record(a)
            return real(*a, **kw)

        monkeypatch.setattr(kq, name, spy)

    wrap("moe_glu_gather_shexp_kq", lambda a: seen["fold"].append(a[1]))
    wrap("moe_router_topk", lambda a: seen.__setitem__("router", seen["router"] + 1))
    if hasattr(kq, "gather_qmv_mix_ns_kq"):
        wrap("gather_qmv_mix_ns_kq",
             lambda a: seen.__setitem__("mix_ns", seen["mix_ns"] + 1))
    return seen


def _rows(n, **uids):
    """Rows mode with ``n`` slots and a registry holding each uid's row."""
    lora_rows.configure("rows", n)
    reg = lora_rows.UidRegistry()
    for uid, row in uids.items():
        reg.register(int(uid[1:]), row)
    return reg


def _run(reg, uids, fn):
    with lora_rows.bind(reg), lora_rows.published(uids):
        y = fn()
        mx.eval(y)
    return y


def _reset(seen):
    seen["fold"].clear()
    seen["router"] = 0
    seen["mix_ns"] = 0


def test_expert_adapter_installs_on_the_fused_block(monkeypatch):
    shell = _qwen3next_shell()
    x = _x()
    base = shell.mlp(x)
    mx.eval(base)
    _install(shell, ["mlp.switch_mlp.down_proj"])
    y = shell.mlp(x)
    mx.eval(y)
    ref = _stock(shell, x, monkeypatch)
    assert _rel(base, ref) > 0.2
    assert _rel(y, ref) < 0.03


def test_a_forward_with_every_slot_off_folds(kq_calls):
    shell = _qwen3next_shell()
    x = _x()
    reg = _rows(1, u1=(0.0,))
    pre = _run(reg, [1], lambda: shell.mlp(x))
    _install(shell, EXPERT_PATHS)
    _reset(kq_calls)
    y = _run(reg, [1], lambda: shell.mlp(x))
    assert len(kq_calls["fold"]) == 1
    assert mx.array_equal(y, pre)


def test_a_forward_with_a_slot_on_calls_the_members(kq_calls, monkeypatch):
    shell = _qwen3next_shell()
    x = _x()
    reg = _rows(1, u1=(1.0,))
    _install(shell, EXPERT_PATHS)
    _reset(kq_calls)
    y = _run(reg, [1], lambda: shell.mlp(x))
    assert kq_calls["fold"] == []
    routed = kq_calls["router"]
    ref = _run(reg, [1], lambda: _stock(shell, x, monkeypatch))
    pre = _run(_rows(1, u1=(0.0,)), [1], lambda: _stock(shell, x, monkeypatch))
    assert _rel(pre, ref) > 0.2
    assert _rel(y, ref) < 0.03
    if isinstance(vars(shell.mlp).get("_kq_router_cat"), mx.array):
        assert routed == 1


def test_each_row_of_a_mixed_forward_matches_its_own_reference(monkeypatch):
    shell = _qwen3next_shell()
    x = mx.concatenate([_x(2), _x(3)], axis=0)
    reg = _rows(1, u1=(1.0,), u0=(0.0,))
    _install(shell, EXPERT_PATHS)
    y = _run(reg, [1, 0], lambda: shell.mlp(x))
    for i, uid in enumerate((1, 0)):
        ref = _run(reg, [uid], lambda: _stock(shell, x[i:i + 1], monkeypatch))
        assert _rel(y[i:i + 1], ref) < 0.03


def test_base_rows_stay_fused_across_adapted_blocks(kq_calls):
    shell = _qwen3next_shell2()
    x = _x()
    reg = _rows(2, u1=(0.0, 0.0), u2=(0.0, 1.0))
    pre = _run(reg, [1], lambda: shell(x))
    _install(shell, ["layers.0.switch_mlp.down_proj",
                     "layers.0.shared_expert.down_proj"], slot=0)
    _install(shell, ["layers.1.switch_mlp.down_proj",
                     "layers.1.shared_expert.down_proj"], seed=3, slot=1)
    assert shell.layers[0]._kq_lora_slots == (0,)
    assert shell.layers[1]._kq_lora_slots == (1,)
    _reset(kq_calls)
    y = _run(reg, [1], lambda: shell(x))
    assert len(kq_calls["fold"]) == 2
    assert mx.array_equal(y, pre)
    _reset(kq_calls)
    _run(reg, [2], lambda: shell(x))
    gate0 = shell.layers[0].switch_mlp.gate_proj.weight
    assert len(kq_calls["fold"]) == 1 and kq_calls["fold"][0] is gate0


def test_a_stamped_leaf_without_slots_on_record_calls_the_members(kq_calls):
    shell = _qwen3next_shell()
    x = _x()
    reg = _rows(1, u1=(0.0,))
    _install(shell, ["mlp.switch_mlp.down_proj"])
    del vars(shell.mlp)["_kq_lora_slots"]
    _reset(kq_calls)
    _run(reg, [1], lambda: shell.mlp(x))
    assert kq_calls["fold"] == []


def test_a_table_only_offload_wrap_keeps_the_adapted_block(kq_calls, monkeypatch):
    """A table-only plan whose wire holds keeps the experts resident and
    wraps the adapted SwitchGLU in the offload class: the block still calls
    its members with the slot on and folds with it off."""
    import gmlx.stream.expert_streaming as es
    import gmlx.stream.table_stream as ts

    shell = _qwen3next_shell()

    class _Trunk(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = [shell]

    x = _x()
    reg = _rows(1, u1=(1.0,), u0=(0.0,))
    pre = _run(reg, [0], lambda: shell.mlp(x))
    _install(shell, EXPERT_PATHS)
    on = _run(reg, [1], lambda: shell.mlp(x))
    monkeypatch.setattr(es, "stream_budget_bytes", lambda: 1 << 40)
    monkeypatch.setattr(ts, "file_table_bytes", lambda m: 0)
    monkeypatch.setattr(ts, "table_bytes", lambda m: 0)
    monkeypatch.setattr(ts, "tables_for_memory", lambda p: False)
    monkeypatch.setattr(ts, "streamed_table_array_ids", lambda m: ())
    monkeypatch.setattr(ts, "install_table_streaming", lambda m: (0, ["t"]))
    monkeypatch.setattr(ts, "stream_plan", lambda *a, **k: SimpleNamespace(
        tables=True, experts=False, table_kept=False))
    monkeypatch.setattr(es, "_wire_resident_experts", lambda *a, **k: True)
    monkeypatch.setattr(es, "credit_streamed", lambda *a, **k: 0.0)
    monkeypatch.setattr(es, "end_build_credit", lambda *a, **k: None)
    es.install_expert_streaming(_Trunk())
    names = [c.__name__ for c in type(shell.mlp.switch_mlp).__mro__]
    assert names[0].endswith("_CPUOffload") and "_LoRASwitchGLU" in names
    _reset(kq_calls)
    assert mx.array_equal(_run(reg, [1], lambda: shell.mlp(x)), on)
    assert kq_calls["fold"] == []
    off = _run(reg, [0], lambda: shell.mlp(x))
    assert len(kq_calls["fold"]) == 1
    assert mx.array_equal(off, pre)


def test_an_arena_swap_maps_each_slot_to_its_owner_expert():
    """Under expert streaming the fused switch gathers from an arena whose
    rows are slots, here 6 slots for 4 experts. Each slot gives its owner
    expert's output with that expert's delta, and a dead slot gives zero."""
    from gmlx.stream.feeder_common import swapped_weights

    shell = _qwen3next_shell()
    _install(shell, ["mlp.switch_mlp.down_proj"])
    glu = shell.mlp.switch_mlp
    owner = np.array([2, -1, 0, 3, 1, -1], dtype=np.int32)
    live = owner >= 0

    def arena(w):
        rows = mx.take(w, mx.array(np.where(live, owner, 0).astype(np.uint32)), axis=0)
        keep = mx.array(live).reshape((-1,) + (1,) * (w.ndim - 1))
        return mx.where(keep, rows, mx.zeros_like(rows))

    views = {k: arena(getattr(glu, f"{k}_proj").weight) for k in ("gate", "up", "down")}
    x = mx.concatenate([_x(2), _x(3)], axis=0)
    slots = np.array([[[0, 3]], [[4, 2]]], dtype=np.uint32)
    experts = mx.array(owner[slots].astype(np.uint32))
    on, off = _rows(1, u1=(1.0,)), _rows(1, u0=(0.0,))
    with swapped_weights({k: (glu,) for k in views}, views, slot_owner=lambda: owner):
        got = _run(on, [1, 1], lambda: glu(x, mx.array(slots)))
        dead = _run(on, [1], lambda: glu(x[:1], mx.array([[[1, 5]]], dtype=mx.uint32)))
    ref = _run(on, [1, 1], lambda: glu(x, experts))
    base = _run(off, [0, 0], lambda: glu(x, experts))
    assert _rel(base, ref) > 0.2
    assert _rel(got, ref) < 0.03
    assert mx.array_equal(dead, mx.zeros_like(dead))


QE, QK = 64, 6


def _qwen4exp_block(quant_shared, seed=0):
    """A qwen4exp block: fp32 router, q8_0 expert stacks, and a bf16 shared
    expert (router-only) or a q8_0 one (fold)."""
    from gmlx.models.qwen4_exp.model import SparseMoeBlock

    mx.random.seed(seed)
    rng = np.random.default_rng(seed)
    block = SparseMoeBlock(SimpleNamespace(
        hidden_size=D, moe_intermediate_size=INTER,
        shared_expert_intermediate_size=INTER, norm_topk_prob=True,
        num_experts=QE, num_experts_per_tok=QK))
    from mlx_kquant.nn import KQuantSwitchLinear

    for name, (o, i) in (("gate_proj", (INTER, D)), ("up_proj", (INTER, D)),
                         ("down_proj", (D, INTER))):
        setattr(block.switch_mlp, name, _quantized(
            KQuantSwitchLinear(QE, o, i, False, "q8_0"), (QE, o, i), rng))
    if quant_shared:
        _quantize_mlp(block.shared_expert, rng)
    else:
        block.shared_expert.set_dtype(mx.bfloat16)
    block.shared_expert_gate.set_dtype(mx.bfloat16)
    block.eval()
    shell = _Shell(block)
    modules.install_fused_moe_glu(shell)
    if type(shell.mlp).__name__ != "_FusedKQuantMoeBlock":
        pytest.skip("the fused MoE block is not installable in this build")
    assert getattr(shell.mlp, "_kq_shexp_fold", True) is quant_shared
    mx.eval(shell.parameters())
    return shell


@pytest.mark.parametrize("quant_shared", [False, True], ids=["router_only", "fold"])
def test_qwen4exp_expert_adapter_matches_the_stock_forward(
        quant_shared, kq_calls, monkeypatch):
    shell = _qwen4exp_block(quant_shared)
    mx.random.seed(4)
    x = mx.random.normal((1, 2, D)) * 0.5
    on = _rows(1, u1=(1.0,), u0=(0.0,))
    pre = _run(on, [0], lambda: shell.mlp(x))
    _install(shell, EXPERT_PATHS)
    _reset(kq_calls)
    off = _run(on, [0], lambda: shell.mlp(x))
    assert len(kq_calls["fold"]) == (1 if quant_shared else 0)
    assert mx.array_equal(off, pre)
    y = _run(on, [1], lambda: shell.mlp(x))
    ref = _run(on, [1], lambda: _stock(shell, x, monkeypatch))
    assert _rel(pre, ref) > 0.2
    g, w = np.array(y.astype(mx.float32)), np.array(ref.astype(mx.float32))
    assert np.abs(g - w).max() <= 2e-2 * np.abs(w).max()


GE, GK = 8, 2


def _gemma_experts(seed=0):
    from gmlx.models.gemma4.language import Experts

    rng = np.random.default_rng(seed)
    ex = Experts(SimpleNamespace(hidden_size=D, moe_intermediate_size=INTER,
                                 num_experts=GE))
    from mlx_kquant.nn import KQuantSwitchLinear

    for name, (o, i) in (("gate_proj", (INTER, D)), ("up_proj", (INTER, D)),
                         ("down_proj", (D, INTER))):
        setattr(ex.switch_glu, name, _quantized(
            KQuantSwitchLinear(GE, o, i, False, "q8_0"), (GE, o, i), rng))
    ex.eval()

    class Shell(nn.Module):
        def __init__(self):
            super().__init__()
            self.experts = ex

    shell = Shell()
    modules.install_fused_moe_glu(shell)
    if type(shell.experts).__name__ != "_FusedKQuantGemmaExperts":
        pytest.skip("the fused gemma experts are not installable in this build")
    mx.eval(shell.parameters())
    return shell


def test_gemma_experts_call_the_adapted_glu_only_with_a_slot_on(kq_calls, monkeypatch):
    shell = _gemma_experts()
    rng = np.random.default_rng(7)
    x = _x()
    inds = mx.array(rng.integers(0, GE, (1, 1, GK)).astype(np.uint32))
    w = mx.array(rng.random((1, 1, GK)).astype(np.float32)).astype(mx.bfloat16)
    reg = _rows(1, u1=(1.0,), u0=(0.0,))
    pre = _run(reg, [0], lambda: shell.experts(x, inds, w))
    _install(shell, ["experts.switch_glu.down_proj"])
    assert shell.experts._kq_lora_slots == (0,)
    _reset(kq_calls)
    off = _run(reg, [0], lambda: shell.experts(x, inds, w))
    assert kq_calls["mix_ns"] == 1
    assert mx.array_equal(off, pre)
    _reset(kq_calls)
    y = _run(reg, [1], lambda: shell.experts(x, inds, w))
    assert kq_calls["mix_ns"] == 0
    # The stock leaf with the plain-op delta, not the fused epilogue.
    ref = _run(reg, [1], lambda: _stock_experts(shell, x, inds, w, monkeypatch))
    assert _rel(pre, ref) > 0.2
    assert _rel(y, ref) < 0.03


def _stock_experts(shell, x, inds, w, monkeypatch):
    with monkeypatch.context() as m:
        m.setattr(modules, "_FUSED_MOE_ENABLED", False)
        y = shell.experts(x, inds, w)
        mx.eval(y)
    return y
