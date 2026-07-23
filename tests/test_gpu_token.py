#!/usr/bin/env python3
"""GPU-autonomous token (``gmlx.gpu_token`` + the wrapper's route_shed
branch): slot-table snapshots, the boundary step's fence ordering
(popularity -> flush -> prestage -> snapshot), and numeric transparency of
the no-eval decode path. Requires an mlx_kquant build with ``route_shed``
(the gpu-dispatch branch); skipped wholesale on older installs."""

from __future__ import annotations

import time
from contextlib import contextmanager

import numpy as np
import pytest

import mlx.core as mx
from mlx_lm.models.switch_layers import SwitchGLU

from gmlx.gpu_token import GpuTokenState, route_shed_op
from gmlx.loader import install_expert_streaming

from test_decode_feeder import _make_feeder

pytestmark = pytest.mark.skipif(
    route_shed_op() is None,
    reason="installed mlx_kquant has no route_shed (gpu-dispatch branch)",
)


def _wait_pending(feeder, li, timeout=5.0):
    t0 = time.monotonic()
    while feeder._pending.get(li):
        pend = feeder._pending[li]
        if all(f.done() for _, futs, _ in pend.values() for f in futs):
            return
        assert time.monotonic() - t0 < timeout, "prestage read never landed"
        time.sleep(0.005)


def test_boundary_prestages_misses_and_fences_tables(monkeypatch, tmp_path):
    """A recorded miss earns popularity, prestages at the boundary, and
    enters the table only after its read published - never mid-flight."""
    feeder, _ = _make_feeder(monkeypatch, tmp_path, slots_per_layer=2)
    feeder.stage(0, np.array([[0, 1]], dtype=np.uint32))
    gt = GpuTokenState(feeder, keep_mass=0.9)

    tbl = gt.table(0)
    assert np.array_equal(
        np.array(tbl) >= 0, feeder._slot_of[0] >= 0)

    idx = mx.array([[0, 2]], dtype=mx.uint32)
    sc = mx.array([[0.7, 0.3]], dtype=mx.float32)
    slots, mix, m_ids, m_sc = gt._route_shed(idx, sc, tbl)
    mx.eval(slots, mix, m_ids, m_sc)
    assert np.array(m_ids).reshape(-1)[0] == 2  # non-resident 2 shed

    lookups0, calls0 = feeder._lookups, feeder._calls
    gt.record(0, idx, sc, m_ids, m_sc)
    gt.boundary()

    # Popularity and ledger credit for the whole routed set.
    assert feeder._counts[0][0] > 0 and feeder._counts[0][2] > 0
    assert feeder._lookups == lookups0 + 2
    assert feeder._calls == calls0 + 1
    assert feeder._shed_n == 1
    assert gt.tokens == 1 and gt.miss_n == 1
    # keep_mass 0.9 allows 10% mass; 0.3 dropped = over budget.
    assert gt.over_budget_layers == 1

    # The boundary snapshot must NOT contain expert 2 (its read is still
    # in flight / unpublished) and must not map the eviction victim.
    tbl2 = np.array(gt.table(0))
    assert tbl2[2] < 0

    # Once the read lands, the NEXT boundary publishes and the snapshot
    # picks it up.
    _wait_pending(feeder, 0)
    gt.record(0, idx, sc, m_ids, m_sc)  # any record to drive the boundary
    gt.boundary()
    assert np.array(gt.table(0))[2] >= 0


def test_on_layer_entry_detects_token_wrap(monkeypatch, tmp_path):
    feeder, _ = _make_feeder(monkeypatch, tmp_path)
    feeder.stage(0, np.array([[0, 1]], dtype=np.uint32))
    gt = GpuTokenState(feeder)
    idx = mx.array([[0, 1]], dtype=mx.uint32)
    sc = mx.array([[0.5, 0.5]], dtype=mx.float32)
    outs = gt._route_shed(idx, sc, gt.table(0))
    mx.eval(*outs)

    gt.on_layer_entry(0)
    gt.record(0, idx, sc, outs[2], outs[3])
    gt.on_layer_entry(1)  # same token, higher layer: no boundary
    assert gt.tokens == 0
    gt.record(1, idx, sc, outs[2], outs[3])
    gt.on_layer_entry(0)  # wrap: previous token flushed
    assert gt.tokens == 1
    assert not gt._records


def _autonomous_glu(monkeypatch, fake_df):
    mx.random.seed(11)
    glu = SwitchGLU(16, 32, 4)
    mx.eval(glu.parameters())

    class _Holder:
        pass

    layer = _Holder()
    layer.modules = lambda: [glu]
    model = _Holder()
    model.layers = [layer]
    model.parameters = lambda: {"glu": glu.parameters()}
    monkeypatch.setattr(
        mx, "device_info", lambda: {"max_recommended_working_set_size": 1024}
    )
    install_expert_streaming(model)
    object.__setattr__(glu, "_kq_decode_feeder", fake_df)
    object.__setattr__(glu, "_kq_li", 5)
    object.__setattr__(glu, "_kq_gpu_token", GpuTokenState(fake_df))
    return glu


class _FakeDF:
    """Identity-arena stand-in exposing exactly what the autonomous path
    touches: residency state for snapshots, the boundary bookkeeping
    fields, and recording prestage/flush calls."""

    def __init__(self, n_experts=4, absent=()):
        slot_of = np.arange(n_experts, dtype=np.int32)
        for e in absent:
            slot_of[e] = -1
        self._slot_of = {5: slot_of}
        self._counts = {5: np.zeros(n_experts, dtype=np.float64)}
        self._calls = 0
        self._lookups = 0
        self._hits = 0
        self._layer_lookups = {5: 0}
        self._layer_hits = {5: 0}
        self._shed_n = 0
        self._shed_mass = 0.0
        self._shed_tokens = 0
        self.prestage_calls = []
        self.flush_calls = []
        self.swap_calls = []
        self.stage_calls = []

    def covers(self, li):
        return True

    def wedged_at(self, li):
        return False

    @contextmanager
    def swapped(self, li):
        self.swap_calls.append(li)
        yield

    def stage(self, li, ids):
        self.stage_calls.append((li, ids.copy()))
        return ids.astype(np.uint32)

    def _flush_pending(self, li):
        self.flush_calls.append(li)

    def prestage(self, li, ids):
        self.prestage_calls.append((li, ids.copy()))


def test_wrapper_autonomous_all_resident_is_transparent(monkeypatch):
    """All routed experts resident: the no-eval path must reproduce the
    scores-mixed reference exactly, without calling stage()."""
    df = _FakeDF()
    glu = _autonomous_glu(monkeypatch, df)
    x1 = mx.random.normal((1, 1, 16))
    i1 = mx.array([[[1, 3]]], dtype=mx.uint32)
    sc = mx.array([[[0.6, 0.4]]], dtype=mx.float32)
    ref = (mx.array(glu.__class__.__bases__[0].__call__(glu, x1, i1))
           * sc[..., None]).sum(axis=-2)
    mx.eval(ref)

    out = glu(x1, i1, sc)
    mx.eval(out)
    assert mx.allclose(ref, out, atol=1e-5, rtol=1e-5)
    assert df.stage_calls == []  # no host staging on this path
    assert df.swap_calls == [5]
    gt = glu._kq_gpu_token
    assert gt.layer_calls == 1 and gt.tokens == 0  # boundary not yet


def test_wrapper_autonomous_miss_sheds_and_prestages(monkeypatch):
    """A non-resident routed expert is shed (survivors renormalized to the
    token's full mass) and prestaged at the next token boundary."""
    df = _FakeDF(absent=(3,))
    glu = _autonomous_glu(monkeypatch, df)
    x1 = mx.random.normal((1, 1, 16))
    i1 = mx.array([[[1, 3]]], dtype=mx.uint32)
    sc = mx.array([[[0.6, 0.4]]], dtype=mx.float32)
    # Survivor keeps full mass: expert 1 at weight 1.0.
    ref = mx.array(glu.__class__.__bases__[0].__call__(glu, x1, i1))[
        ..., 0, :]
    mx.eval(ref)

    out = glu(x1, i1, sc)
    mx.eval(out)
    assert mx.allclose(ref, out, atol=1e-5, rtol=1e-5)

    # Second token: entry at the same layer wraps -> boundary consumes.
    out2 = glu(x1, i1, sc)
    mx.eval(out2)
    gt = glu._kq_gpu_token
    assert gt.tokens == 1 and gt.miss_n == 1
    assert df.flush_calls == [5]
    assert len(df.prestage_calls) == 1
    li, ids = df.prestage_calls[0]
    assert li == 5 and list(ids.reshape(-1)) == [3]
    assert df._counts[5][1] > 0 and df._counts[5][3] > 0
    assert df._shed_n == 1
