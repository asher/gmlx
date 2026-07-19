#!/usr/bin/env python3
"""Quantized-operand indexer scoring (indexer_q) integration wiring.

The prefill indexer arm can score on the QAT codes+scales directly
(int8 tensor-op MMA in mlx-kquant) instead of the round-tripped fp16
operands. Losslessness rests on a precondition -- pool rows are FP4-grid
fixed points -- certified once per process and guarded by permanent
disarm. These tests pin the gmlx wiring: selection equality vs the fp16
kernel arm, the grid certificate, off-grid disarm, and the kill switch.

Selection equality is asserted as (a) bit-equal score tensors between the
two arms -- the lossless contract -- and (b) equal selected-score
multisets. Raw index sets are NOT compared: dsa_topk_indices breaks exact
score ties nondeterministically across allocations (same artifact family
as the parity-test logit-tie gotcha), and synthetic on-grid data ties
freely at the k boundary.
Metal-only; skipped when the ops are absent."""

from __future__ import annotations

import os

import numpy as np
import mlx.core as mx
import mlx.nn as nn
import pytest

import gmlx.deepseek_v4_model as md

kq = pytest.importorskip("mlx_kquant")

_SYMS = md._DSA_SYMS["indexer_q"]
pytestmark = [
    pytest.mark.skipif(
        not all(hasattr(kq, s) for s in _SYMS),
        reason="mlx-kquant lacks indexer_q ops",
    ),
    pytest.mark.skipif(
        # Metal present is not enough: KQUANT_FORCE_CPU pins the default
        # device to CPU, where the indexer_q ops have no implementation.
        not mx.metal.is_available() or bool(os.environ.get("KQUANT_FORCE_CPU")),
        reason="requires Metal (indexer_q ops are GPU-only)",
    ),
]

H, D, HID, L, P = 64, 128, 256, 64, 4096


@pytest.fixture(autouse=True)
def _fresh_dsa_state(monkeypatch):
    monkeypatch.setattr(
        md, "_dsa_state", {k: None for k in md._dsa_state}
    )
    monkeypatch.setattr(md, "_pool_grid_certified", False)
    monkeypatch.delenv("GMLX_DSA_INDEXER_Q", raising=False)


def _indexer_stub():
    idx = md.Indexer.__new__(md.Indexer)
    idx.n_heads = H
    idx.head_dim = D
    idx.index_topk = 512
    idx.scale = D ** -0.5
    idx.weights_proj = nn.Linear(HID, H, bias=False)
    idx.weights_proj.weight = idx.weights_proj.weight.astype(mx.float16)
    return idx


def _topk_score_spy(monkeypatch):
    """Record the score tensor each _kernel_topk arm hands to topk."""
    import mlx_kquant as kq

    seen = []
    orig = kq.dsa_topk_indices

    def spy(scores, k, **kw):
        mx.eval(scores)
        seen.append(np.array(scores.astype(mx.float32)))
        return orig(scores, k, **kw)

    monkeypatch.setattr(kq, "dsa_topk_indices", spy)
    return seen


def _selection_equivalent(scores, sel_a, sel_b):
    """Same scores + same selected-score multiset per row == equivalent
    selection up to exact ties."""
    s = scores[0, 0]
    a = np.sort(np.take_along_axis(s, np.array(sel_a)[0].astype(int), -1), -1)
    b = np.sort(np.take_along_axis(s, np.array(sel_b)[0].astype(int), -1), -1)
    return bool((a == b).all())


def _operands(seed=7, on_grid=True):
    mx.random.seed(seed)
    x = mx.random.normal((1, L, HID)).astype(mx.float16)
    q_raw = mx.random.normal((1, H, L, D)).astype(mx.float16)
    q = md._indexer_qat_roundtrip(q_raw)
    q_quant = kq.dsa_indexer_qat_quant(q_raw)
    pooled = mx.random.normal((1, P, D)).astype(mx.float16)
    if on_grid:
        pooled = md._fp4_e2m1_roundtrip(pooled)
    mx.eval(x, q, pooled, *q_quant)
    return x, q, q_quant, pooled


def test_quant_arm_matches_fp16_arm(monkeypatch):
    idx = _indexer_stub()
    x, q, q_quant, pooled = _operands()
    seen = _topk_score_spy(monkeypatch)
    got_q = idx._kernel_topk(x, q, pooled, None, 512, 1 << 20,
                             q_quant=q_quant)
    assert got_q is not None
    assert md._dsa_state["indexer_q"] is True
    assert md._pool_grid_certified is True

    monkeypatch.setitem(md._dsa_state, "indexer_q", False)
    got_f = idx._kernel_topk(x, q, pooled, None, 512, 1 << 20,
                             q_quant=None)
    assert got_f is not None
    s_quant, s_f16 = seen
    assert (s_quant == s_f16).all()  # the lossless contract, bit-level
    assert _selection_equivalent(s_f16, got_q, got_f)


def test_offgrid_pool_disarms_and_falls_back(monkeypatch, capsys):
    idx = _indexer_stub()
    x, q, q_quant, pooled = _operands(on_grid=False)
    seen = _topk_score_spy(monkeypatch)
    got = idx._kernel_topk(x, q, pooled, None, 512, 1 << 20,
                           q_quant=q_quant)
    assert got is not None  # fp16 kernel arm answered
    assert md._dsa_state["indexer_q"] is False
    assert md._pool_grid_certified is False
    assert "indexer_q" in capsys.readouterr().err

    got_f = idx._kernel_topk(x, q, pooled, None, 512, 1 << 20,
                             q_quant=None)
    s_a, s_b = seen
    assert (s_a == s_b).all()
    assert _selection_equivalent(s_a, got, got_f)


def test_grid_certificate_memoizes_success_only(monkeypatch):
    rows = md._fp4_e2m1_roundtrip(
        mx.random.normal((1, 512, D)).astype(mx.float16)
    )
    assert md._pool_rows_on_grid(rows) is True
    assert md._pool_grid_certified is True

    monkeypatch.setattr(md, "_pool_grid_certified", False)
    off = mx.random.normal((1, 512, D)).astype(mx.float16)
    assert md._pool_rows_on_grid(off) is False
    assert md._pool_grid_certified is False


def test_kill_switch_disables_probe(monkeypatch):
    monkeypatch.setenv("GMLX_DSA_INDEXER_Q", "0")
    assert md._dsa_probe("indexer_q") is False


def test_unaligned_width_pads_quant_operands(monkeypatch):
    idx = _indexer_stub()
    mx.random.seed(11)
    l_odd = 96  # not a 64 multiple -> pad_l = 32
    x = mx.random.normal((1, l_odd, HID)).astype(mx.float16)
    q_raw = mx.random.normal((1, H, l_odd, D)).astype(mx.float16)
    q = md._indexer_qat_roundtrip(q_raw)
    q_quant = kq.dsa_indexer_qat_quant(q_raw)
    pooled = md._fp4_e2m1_roundtrip(
        mx.random.normal((1, P - 32, D)).astype(mx.float16)
    )  # unaligned P too
    seen = _topk_score_spy(monkeypatch)
    got_q = idx._kernel_topk(x, q, pooled, None, 512, 1 << 20,
                             q_quant=q_quant)
    assert got_q is not None and md._dsa_state["indexer_q"] is True
    monkeypatch.setitem(md._dsa_state, "indexer_q", False)
    got_f = idx._kernel_topk(x, q, pooled, None, 512, 1 << 20,
                             q_quant=None)
    s_quant, s_f16 = seen
    assert s_quant.shape == s_f16.shape == (1, 1, l_odd, P - 32)
    assert (s_quant == s_f16).all()
    assert _selection_equivalent(s_f16, got_q, got_f)


def test_warm_compiles_all_armed_groups():
    n = md.warm_kernel_pipelines()
    # qat + scores/topk chain + decode + quant chain
    assert n >= 3
    assert md._dsa_state["indexer"] is True
    assert md._dsa_state["indexer_q"] is not False
    assert md._pool_grid_certified is False  # warm never certifies


def test_warm_kill_switch(monkeypatch):
    monkeypatch.setenv("GMLX_DSA_WARM", "0")
    assert md.warm_kernel_pipelines() == 0
    assert md._dsa_state["indexer"] is None  # not even probed


def test_warm_swallows_failures_without_disarming(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("warm failure")

    monkeypatch.setattr(kq, "dsa_indexer_scores", boom)
    monkeypatch.setattr(kq, "dsa_indexer_scores_q", boom)
    n = md.warm_kernel_pipelines()  # must not raise
    assert md._dsa_state["indexer"] is True
    assert md._dsa_state["indexer_q"] is True
    assert n >= 1  # qat and decode still warmed
