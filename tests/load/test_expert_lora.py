#!/usr/bin/env python3
"""Expert-stack LoRA (P1): the 3-D pair loader, ``expert_delta`` against an
independent per-expert ``scale * B_e @ A_e`` reference in every layout the
SwitchGLU forward takes (fused ``[t, k, I]``, stock unsorted, stock sorted
with a non-uniform row vector), and the loud install gates. CPU-only: float
SwitchLinear stacks for the stock layouts, faked kq kernels for the fused
one."""
from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from mlx_lm.models.switch_layers import SwitchGLU

from gmlx import lora_rows
import gmlx.load.adapter as adapter
import gmlx.load.modules as modules
from gmlx.load.remap import RemapDecision, parse_gguf_name


@pytest.fixture(autouse=True)
def _static():
    lora_rows.configure("static", 1)
    yield
    lora_rows.configure("static", 1)


class _Holder(nn.Module):
    def __init__(self, glu):
        super().__init__()
        self.switch_mlp = glu


def _glu(d=8, inter=16, experts=4, seed=0):
    mx.random.seed(seed)
    glu = SwitchGLU(d, inter, experts)
    for name in ("gate_proj", "up_proj", "down_proj"):
        leaf = getattr(glu, name)
        leaf.weight = mx.random.normal(leaf.weight.shape) * 0.3
    glu.eval()
    return glu


def _lm(a, b, scale, path="switch_mlp.down_proj"):
    return adapter.LoraModule(module_path=path, a=a, b=b, rank=a.shape[1],
                              scale=scale, experts=True)


def _ref_delta(glu, x, idx, a, b, scale, s_rows=None):
    """Independent reference: per token and routed expert, h = act(up, gate)
    on that expert, delta = scale * (h @ a_e.T) @ b_e.T, times the row's
    scale. Returns [B, L, k, out]."""
    B, L, k = idx.shape
    xe = mx.expand_dims(x, (-2, -3))
    up = glu.up_proj(xe, idx)
    gate = glu.gate_proj(xe, idx)
    h = glu.activation(up, gate)                      # [B, L, k, 1, I]
    out = np.zeros((B, L, k, b.shape[1]), np.float32)
    hn, an, bn, idn = (np.array(h.astype(mx.float32)), np.array(a), np.array(b),
                       np.array(idx))
    for bi in range(B):
        for li in range(L):
            for ki in range(k):
                e = int(idn[bi, li, ki])
                z = hn[bi, li, ki, 0] @ an[e].T
                out[bi, li, ki] = scale * (z @ bn[e].T)
                if s_rows is not None:
                    out[bi, li, ki] *= s_rows[bi]
    return mx.array(out)


def _install(model, a, b, scale, path="switch_mlp.down_proj"):
    plan = adapter.LoraAdapter(alpha=scale * a.shape[1], arch="qwen3moe",
                               modules={path: _lm(a, b, scale, path)})
    return modules.install_lora_adapter(model, plan)


@pytest.mark.parametrize("B,L,k", [(1, 1, 2), (2, 3, 2), (2, 8, 4)])
def test_stock_layouts_match_per_expert_reference(B, L, k):
    # (2, 8, 4) has indices.size == 64: the sorted stock path.
    glu = _glu()
    ref_glu = _glu()
    model = _Holder(glu)
    E, d, inter, r = 4, 8, 16, 2
    mx.random.seed(1)
    a = mx.random.normal((E, r, inter))
    b = mx.random.normal((E, d, r))
    assert _install(model, a, b, 0.5) == 1
    assert type(model.switch_mlp).__name__ == "_LoRASwitchGLU"
    assert isinstance(model.switch_mlp, SwitchGLU)
    x = mx.random.normal((B, L, d))
    idx = mx.random.randint(0, E, (B, L, k)).astype(mx.uint32)
    y = model.switch_mlp(x, idx)
    ref = ref_glu(x, idx) + _ref_delta(ref_glu, x, idx, a, b, 0.5)
    assert y.shape == ref.shape
    # The sorted layout's leaf calls take the GEMM path, TF32 by default on
    # M5, while the reference h comes from the exact per-row GEMV path.
    tol = 2e-2 if B * L * k >= 64 else 1e-4
    assert mx.allclose(y, ref, rtol=tol, atol=tol), float(mx.abs(y - ref).max())


def test_sorted_path_with_nonuniform_rows():
    lora_rows.configure("rows", 1)
    glu, ref_glu = _glu(), _glu()
    model = _Holder(glu)
    E, d, inter, r = 4, 8, 16, 2
    a = mx.random.normal((E, r, inter))
    b = mx.random.normal((E, d, r))
    _install(model, a, b, 0.5)
    B, L, k = 4, 4, 4                                  # 64 -> sorted
    x = mx.random.normal((B, L, d))
    idx = mx.random.randint(0, E, (B, L, k)).astype(mx.uint32)
    s = [0.0, 1.0, 0.5, 2.0]
    lora_rows.set_rows(s)
    y = model.switch_mlp(x, idx)
    lora_rows.clear_rows()
    base = ref_glu(x, idx)                      # same sorted path as the wrap
    ref = base + _ref_delta(ref_glu, x, idx, a, b, 0.5, s)
    assert mx.allclose(y, ref, rtol=2e-2, atol=2e-2), float(mx.abs(y - ref).max())
    # the scale-0 row is bit-exact the base computed on the same path
    assert mx.array_equal(y[0], base[0])


def test_rows_mode_unpublished_raises_in_expert_path():
    lora_rows.configure("rows", 1)
    model = _Holder(_glu())
    _install(model, mx.zeros((4, 2, 16)), mx.zeros((4, 8, 2)), 0.5)
    with pytest.raises(lora_rows.LoraRowsError):
        model.switch_mlp(mx.zeros((1, 1, 8)), mx.zeros((1, 1, 2), mx.uint32))


def test_scores_mix_python_side_on_stock_branch():
    glu, ref_glu = _glu(), _glu()
    model = _Holder(glu)
    E, d, inter, r = 4, 8, 16, 2
    a = mx.random.normal((E, r, inter))
    b = mx.random.normal((E, d, r))
    _install(model, a, b, 0.5)
    x = mx.random.normal((2, 1, d))
    idx = mx.random.randint(0, E, (2, 1, 2)).astype(mx.uint32)
    sc = mx.random.uniform(shape=(2, 1, 2))
    y = model.switch_mlp(x, idx, sc)
    unmixed = ref_glu(x, idx) + _ref_delta(ref_glu, x, idx, a, b, 0.5)
    ref = (unmixed * sc[..., None]).sum(-2)
    assert y.shape == (2, 1, d)
    assert mx.allclose(y, ref, atol=1e-4)


# fused layout: faked kernels, the delta rides the materialised h

def _fused_model(monkeypatch, d=256, inter=256, experts=4):
    import mlx_kquant as kq
    from mlx_kquant.nn import KQuantSwitchLinear

    glu = SwitchGLU(d, inter, experts)
    for name, (o, i) in (("gate_proj", (inter, d)), ("up_proj", (inter, d)),
                         ("down_proj", (d, inter))):
        setattr(glu, name, KQuantSwitchLinear(experts, o, i, False, "q8_0"))
    glu.eval()
    model = _Holder(glu)
    if modules.install_fused_moe_glu(model) != 1:
        pytest.skip("fused MoE glu not installable in this build")
    monkeypatch.setattr(modules, "_kq_fused_device_ok", lambda *m: True)
    # These tests fake the base kernels and check the plain-op delta path;
    # the in-op epilogue path is covered by test_lora_epilogue_wiring.py.
    monkeypatch.setattr(modules, "_KQ_LORA_EPILOGUE", False)
    return model, kq


def test_fused_branch_adds_mixed_delta(monkeypatch):
    model, kq = _fused_model(monkeypatch)
    glu = model.switch_mlp
    if not glu._kq_mix_scores:
        pytest.skip("build without gather_qmv_mix_ns_kq")
    E, d, inter, r = 4, 256, 256, 1
    mx.random.seed(2)
    a = mx.random.normal((E, r, inter)).astype(mx.float16)
    b = mx.random.normal((E, d, r)).astype(mx.float16)
    _install(model, a, b, 1.0)
    assert type(model.switch_mlp).__name__ == "_LoRASwitchGLU"
    assert hasattr(model.switch_mlp, "_fused_ok")
    h_val = mx.random.normal((2, 2, inter)).astype(mx.bfloat16)
    monkeypatch.setattr(kq, "moe_glu_gather_kq",
                        lambda x, gw, uw, kt, idx, **kw: h_val)
    monkeypatch.setattr(kq, "gather_qmv_mix_ns_kq",
                        lambda h, dw, kt, idx, sc: mx.full((2, d), 7.0, mx.bfloat16))
    x = mx.zeros((2, 1, d), mx.bfloat16)
    idx = mx.array([[[0, 3]], [[2, 1]]], dtype=mx.uint32)
    sc = mx.array([[[0.25, 0.75]], [[0.5, 0.5]]], dtype=mx.bfloat16)
    y = model.switch_mlp(x, idx, sc)
    assert y.shape == (2, 1, d)
    hn, an, bn = (np.array(h_val.astype(mx.float32)), np.array(a.astype(mx.float32)),
                  np.array(b.astype(mx.float32)))
    ref = np.full((2, d), 7.0, np.float32)
    scn = np.array(sc.astype(mx.float32))
    for t in range(2):
        for j in range(2):
            e = int(np.array(idx)[t, 0, j])
            ref[t] += scn[t, 0, j] * ((hn[t, j] @ an[e].T) @ bn[e].T)
    assert np.allclose(np.array(y.astype(mx.float32))[:, 0], ref, rtol=2e-2, atol=0.2)


def test_fused_branch_scores_none_unmixed(monkeypatch):
    model, kq = _fused_model(monkeypatch)
    E, d, inter = 4, 256, 256
    a = mx.zeros((E, 1, inter), mx.float16)
    b = mx.zeros((E, d, 1), mx.float16)
    _install(model, a, b, 1.0)
    monkeypatch.setattr(kq, "moe_glu_gather_kq",
                        lambda x, gw, uw, kt, idx, **kw: mx.zeros((1, 2, inter), mx.bfloat16))
    monkeypatch.setattr(kq, "gather_qmv_kq",
                        lambda h, dw, kt, idx, **kw: mx.full((1, 2, d), 3.0, mx.bfloat16))
    y = model.switch_mlp(mx.zeros((1, 1, d), mx.bfloat16), mx.zeros((1, 1, 2), mx.uint32))
    assert y.shape == (1, 1, 2, d)
    assert np.allclose(np.array(y.astype(mx.float32)), 3.0)


# loader

def test_loader_builds_expert_pair_with_positional_rank():
    dec = parse_gguf_name("qwen3moe", "blk.0.ffn_down_exps.weight")
    if dec.kind != RemapDecision.KIND_MAP:
        pytest.skip("qwen3moe expert remap unavailable")
    E, r, inn, out = 6, 1, 8, 4
    arrays = {"blk.0.ffn_down_exps.weight.lora_a": mx.zeros((E, r, inn)),
              "blk.0.ffn_down_exps.weight.lora_b": mx.zeros((E, out, r))}
    plan = adapter.build_adapter_plan(
        {"adapter.type": "lora", "adapter.lora.alpha": 1.0,
         "general.architecture": "qwen3moe"}, arrays)
    (lm,) = plan.modules.values()
    assert lm.experts and lm.rank == 1 and lm.scale == 1.0   # not min({1, 6}) by luck
    assert lm.module_path.endswith("switch_mlp.down_proj")


def test_loader_rejects_malformed_expert_pair():
    arrays = {"blk.0.ffn_down_exps.weight.lora_a": mx.zeros((6, 2, 8)),
              "blk.0.ffn_down_exps.weight.lora_b": mx.zeros((6, 4, 3))}
    with pytest.raises(ValueError, match="expert lora pair"):
        adapter.build_adapter_plan(
            {"adapter.type": "lora", "adapter.lora.alpha": 1.0,
             "general.architecture": "qwen3moe"}, arrays)


# install gates

def test_gate_up_expert_target_raises():
    model = _Holder(_glu())
    with pytest.raises(NotImplementedError, match="down_proj"):
        _install(model, mx.zeros((4, 1, 8)), mx.zeros((4, 16, 1)), 1.0,
                 path="switch_mlp.up_proj")


def test_offload_class_in_mro_raises():
    glu = _glu()
    sub = type("SwitchGLU_CPUOffload", (type(glu),), {})
    glu.__class__ = sub
    model = _Holder(glu)
    with pytest.raises(NotImplementedError, match="_CPUOffload"):
        _install(model, mx.zeros((4, 1, 16)), mx.zeros((4, 8, 1)), 1.0)


def test_shexp_stamp_raises():
    glu = _glu()
    object.__setattr__(glu, "_kq_shexp_mod", object())
    model = _Holder(glu)
    with pytest.raises(NotImplementedError, match="shared-expert"):
        _install(model, mx.zeros((4, 1, 16)), mx.zeros((4, 8, 1)), 1.0)


def test_missing_expert_target_raises():
    model = _Holder(_glu())
    with pytest.raises(ValueError, match="no matching module"):
        _install(model, mx.zeros((4, 1, 16)), mx.zeros((4, 8, 1)), 1.0,
                 path="other.down_proj")


def test_mixed_dense_and_expert_targets_both_install():
    class _Blk(nn.Module):
        def __init__(self):
            super().__init__()
            self.switch_mlp = _glu()
            self.o_proj = nn.Linear(8, 8, bias=False)
    model = _Blk()
    plan = adapter.LoraAdapter(alpha=1.0, arch="qwen3moe", modules={
        "switch_mlp.down_proj": _lm(mx.zeros((4, 1, 16)), mx.zeros((4, 8, 1)), 1.0),
        "o_proj": adapter.LoraModule(module_path="o_proj", a=mx.zeros((1, 8)),
                                     b=mx.zeros((8, 1)), rank=1, scale=1.0),
    })
    assert modules.install_lora_adapter(model, plan) == 2
    assert type(model.o_proj).__name__ == "LoRAKQuantLinear"
    assert type(model.switch_mlp).__name__ == "_LoRASwitchGLU"


# streaming: slot ids under the feeder swap map back through the owner table

def test_slot_ids_map_through_owner_table_and_mask_dead_slots():
    import numpy as np
    from gmlx.stream.feeder_common import swapped_weights

    glu, ref_glu = _glu(), _glu()
    model = _Holder(glu)
    E, d, inter, r = 4, 8, 16, 2
    a = mx.random.normal((E, r, inter))
    b = mx.random.normal((E, d, r))
    _install(model, a, b, 0.5)
    lo = glu.down_proj._kq_lora
    # arena of 6 slots: slot s holds expert owner[s]; slots 1 and 4 are
    # empty / zeroed and never carry a live expert
    owner = np.array([2, -1, 0, 3, -3, 1], dtype=np.int32)
    # slot views: the stack reordered to slot order (zeros in dead slots)
    def slot_stack(w):
        out = mx.zeros((len(owner),) + w.shape[1:], w.dtype)
        for s_, e in enumerate(owner):
            if e >= 0:
                out[s_] = w[e]
        return out
    views = {k: slot_stack(getattr(glu, f"{k}_proj").weight)
             for k in ("gate", "up", "down")}
    entry = {k: (glu,) for k in views}
    x = mx.random.normal((2, 3, d))
    slots = mx.array([[[0, 2], [3, 5], [2, 0]], [[5, 3], [0, 2], [3, 5]]],
                     dtype=mx.uint32)
    with swapped_weights(entry, views, slot_owner=lambda: owner):
        assert lo.owner is not None
        y = model.switch_mlp(x, slots)
    assert lo.owner is None                      # restored on exit
    eidx = mx.array(owner[np.array(slots)], dtype=mx.uint32)
    ref = ref_glu(x, eidx) + _ref_delta(ref_glu, x, eidx, a, b, 0.5)
    assert mx.allclose(y, ref, atol=1e-4), float(mx.abs(y - ref).max())
    # a dead slot in the routing: its expert output is zero and so is its delta
    slots_dead = mx.array([[[1, 4]]], dtype=mx.uint32)
    with swapped_weights(entry, views, slot_owner=lambda: owner):
        y0 = model.switch_mlp(x[:1, :1], slots_dead)
    assert mx.array_equal(y0, mx.zeros_like(y0))


def test_prefill_ring_swap_keeps_expert_ids():
    from gmlx.stream.feeder_common import swapped_weights

    glu = _glu()
    model = _Holder(glu)
    _install(model, mx.random.normal((4, 2, 16)), mx.random.normal((4, 8, 2)), 0.5)
    lo = glu.down_proj._kq_lora
    views = {k: getattr(glu, f"{k}_proj").weight for k in ("gate", "up", "down")}
    with swapped_weights({k: (glu,) for k in views}, views):
        assert lo.owner is None


def test_decode_feeder_swap_hands_the_live_owner_table():
    import numpy as np
    import gmlx.stream.decode_feeder as decode_feeder

    glu = _glu()
    model = _Holder(glu)
    _install(model, mx.random.normal((4, 2, 16)), mx.random.normal((4, 8, 2)), 0.5)
    lo = glu.down_proj._kq_lora

    class _Fake:
        _layers = {7: {"down": (glu, None, None, 0, (4, 8, 16))}}
        _views = {(7, "down"): glu.down_proj.weight}
        _owner = {7: np.array([1, 0, -1, 2], dtype=np.int32)}
    fake = _Fake()
    with decode_feeder.DecodeFeeder.swapped(fake, 7):
        assert lo.owner is not None
        fake._owner[7] = np.array([3, 3, 3, 3], dtype=np.int32)   # a shed restaged
        assert list(lo.owner()) == [3, 3, 3, 3]                   # read live
    assert lo.owner is None
