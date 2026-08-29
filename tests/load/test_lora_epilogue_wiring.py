#!/usr/bin/env python3
"""The in-op LoRA epilogue wiring: with a kq build that carries
``HAS_LORA_EPILOGUE``, the dense wrapper hands its tables and the published
row factor to ``KQuantLinear(x, lora=...)`` and the expert container hands
``lora_a/lora_b`` (plus the arena owner table and the per-row factor) to the
fused down exit instead of computing the delta with plain ops. CPU-safe:
the kq gathers are monkeypatched fakes and the device check is stubbed."""
from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from mlx_lm.models.switch_layers import SwitchGLU

from gmlx import lora_rows
import gmlx.load.modules as modules
from gmlx.load.modules import ExpertLoRA

D, INTER, E, K = 64, 96, 4, 2


class _Holder(nn.Module):
    def __init__(self, glu):
        super().__init__()
        self.experts = glu


def _fused_lora_container():
    from mlx_kquant.nn import KQuantSwitchLinear

    glu = SwitchGLU(D, INTER, E)
    for name, (o, i) in (("gate_proj", (INTER, D)), ("up_proj", (INTER, D)),
                         ("down_proj", (D, INTER))):
        setattr(glu, name, KQuantSwitchLinear(E, o, i, False, "q8_0"))
    glu.eval()
    model = _Holder(glu)
    if modules.install_fused_moe_glu(model) != 1:
        pytest.skip("fused MoE glu not installable in this build")
    glu = model.experts
    lo = ExpertLoRA(mx.ones((E, INTER, 3), dtype=mx.float32) * 0.01,
                    mx.ones((E, 3, D), dtype=mx.float32) * 0.01, 2.0)
    lo.slot = 0
    object.__setattr__(glu.down_proj, "_kq_lora", lo)
    cls = modules._make_lora_switch_glu(type(glu))
    object.__setattr__(glu, "__class__", cls)
    return glu, lo


@pytest.fixture
def epilogue_on(monkeypatch):
    monkeypatch.setattr(modules, "_KQ_LORA_EPILOGUE", True)
    monkeypatch.setattr(modules, "_lora_on_cpu", lambda: False)
    monkeypatch.setattr(modules, "_kq_fused_device_ok", lambda *m: True)
    import mlx_kquant as kq
    seen = {}

    def glu_fake(x, gw, uw, ktype, idx, **kw):
        return mx.full((idx.shape[0], idx.shape[1], INTER), 1.0, mx.bfloat16)

    def mix_ns(h, dw, ktype, idx, sc, **kw):
        seen["mix_kw"] = kw
        seen["mix_idx"] = idx
        return mx.full((h.shape[0], D), 7.0, mx.bfloat16)

    def plain(h, dw, ktype, idx, **kw):
        seen["plain_kw"] = kw
        return mx.full((h.shape[0], h.shape[1], D), 3.0, mx.bfloat16)

    monkeypatch.setattr(kq, "moe_glu_gather_kq", glu_fake)
    monkeypatch.setattr(kq, "gather_qmv_mix_ns_kq", mix_ns)
    monkeypatch.setattr(kq, "gather_qmv_kq", plain)
    return seen


def test_mix_exit_gets_tables_no_table_no_rows_in_static_mode(epilogue_on):
    glu, lo = _fused_lora_container()
    if not glu._kq_mix_scores:
        pytest.skip("build without gather_qmv_mix_ns_kq")
    lora_rows.configure("static")
    x = mx.zeros((2, 1, D), mx.bfloat16)
    idx = mx.array(np.array([[[0, 1]], [[2, 3]]], dtype=np.uint32))
    sc = mx.full((2, 1, K), 0.5, mx.bfloat16)
    y = glu(x, idx, sc)
    kw = epilogue_on["mix_kw"]
    assert set(kw) == {"lora_a", "lora_b"}
    assert kw["lora_a"].dtype == mx.bfloat16 and kw["lora_a"].shape == (E, INTER, 3)
    assert kw["lora_b"].shape == (E, 3, D)
    # scale folded into b
    assert float(kw["lora_b"][0, 0, 0].astype(mx.float32)) == pytest.approx(0.02, rel=1e-2)
    assert y.shape == (2, 1, D)                  # mixed in the kernel
    assert np.allclose(np.array(y.astype(mx.float32)), 7.0)


def test_owner_table_and_rows_ride_as_operands(epilogue_on):
    glu, lo = _fused_lora_container()
    if not glu._kq_mix_scores:
        pytest.skip("build without gather_qmv_mix_ns_kq")
    lo.owner = lambda: np.array([3, -1, 0, 2, 1, -1])
    lora_rows.configure("rows")
    x = mx.zeros((2, 1, D), mx.bfloat16)
    idx = mx.array(np.array([[[0, 1]], [[2, 5]]], dtype=np.uint32))   # slots
    sc = mx.full((2, 1, K), 0.5, mx.bfloat16)
    lora_rows.set_rows([1.0, 0.0])
    try:
        glu(x, idx, sc)
    finally:
        lora_rows.set_rows(None)
        lora_rows.configure("static")
    kw = epilogue_on["mix_kw"]
    assert set(kw) == {"lora_a", "lora_b", "lora_table", "lora_rows"}
    assert kw["lora_table"].dtype == mx.int32
    assert np.array(kw["lora_table"]).tolist() == [3, -1, 0, 2, 1, -1]
    assert kw["lora_rows"].dtype == mx.float32
    assert np.array(kw["lora_rows"]).tolist() == [[1.0, 1.0], [0.0, 0.0]]
    # The kernel gets the raw slot ids; the table remaps inside it.
    assert np.array(epilogue_on["mix_idx"]).tolist() == [[0, 1], [2, 5]]


def test_unmixed_exit_and_rank_bound_fallback(epilogue_on, monkeypatch):
    glu, lo = _fused_lora_container()
    lora_rows.configure("static")
    x = mx.zeros((2, 1, D), mx.bfloat16)
    idx = mx.array(np.array([[[0, 1]], [[2, 3]]], dtype=np.uint32))
    y = glu(x, idx)                                  # no scores: plain exit
    assert set(epilogue_on["plain_kw"]) == {"lora_a", "lora_b"}
    assert y.shape == (2, 1, K, D)
    assert np.allclose(np.array(y.astype(mx.float32)), 3.0)
    # rank x k past the kernel bound: the plain-op delta path takes over
    # (fake exits see no lora kwargs; the delta is added python-side).
    epilogue_on.clear()
    lo.a_t = mx.zeros((E, INTER, 300), dtype=mx.float32)
    lo.b_t = mx.zeros((E, 300, D), dtype=mx.float32)
    lo._tables = {}
    monkeypatch.setattr(modules, "expert_delta",
                        lambda lo, h, ids, s=None: mx.zeros((h.shape[0], 1, D), h.dtype))
    y = glu(x, idx)
    assert epilogue_on["plain_kw"] == {}
    assert y.shape == (2, 1, K, D)


def test_dense_wrapper_hands_tables_and_rows_to_base(monkeypatch):
    from mlx_kquant.nn import KQuantLinear

    base = KQuantLinear(D, D, False, "q8_0")
    calls = {}

    def fake_call(self, x, lora=None):
        calls["lora"] = lora
        return mx.zeros(x.shape[:-1] + (D,), x.dtype)

    monkeypatch.setattr(KQuantLinear, "__call__", fake_call)
    monkeypatch.setattr(modules, "_KQ_LORA_EPILOGUE", True)
    a = mx.ones((3, D), dtype=mx.float32)     # lora_a (r, in) as stored
    b = mx.ones((D, 3), dtype=mx.float32)     # lora_b (out, r)
    wrap = modules.LoRAKQuantLinear(base, a, b, scale=0.5, slot=0)
    lora_rows.configure("rows")
    lora_rows.set_rows([2.0, 0.0])
    try:
        wrap(mx.zeros((2, 4, D), mx.float16))
    finally:
        lora_rows.set_rows(None)
        lora_rows.configure("static")
    a_t, b_t, rows = calls["lora"]
    assert a_t.shape == (D, 3) and b_t.shape == (3, D)
    assert a_t.dtype == mx.float16 and b_t.dtype == mx.float16
    assert rows.dtype == mx.float32
    assert np.array(rows).tolist() == [2.0] * 4 + [0.0] * 4
    # f32 activations keep the plain-op path (the epilogue is f16/bf16 only)
    calls.clear()
    monkeypatch.setattr(KQuantLinear, "__call__",
                        lambda self, x: mx.zeros(x.shape[:-1] + (D,), x.dtype))
    y = wrap(mx.zeros((1, D), mx.float32))
    assert "lora" not in calls and y.shape == (1, D)


def test_row_factors_are_dense_for_one_published_row():
    # mx.repeat of a single row scale evaluates to a stride-0 view; the kq
    # epilogue reads lora_rows densely, so the cached factor is wrapped in
    # mx.contiguous (layout is not observable from Python; this covers the
    # value path, the kq suite covers strided operands).
    from gmlx import lora_rows
    lora_rows.set_rows([0.5])
    try:
        for arr in (lora_rows.dense_rows(0, 16), lora_rows.flat_rows(0, 4, 6)):
            mx.eval(arr)
            assert np.array_equal(np.array(arr), np.full(arr.shape, 0.5, np.float32))
    finally:
        lora_rows.set_rows(None)
