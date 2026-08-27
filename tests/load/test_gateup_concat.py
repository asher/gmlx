#!/usr/bin/env python3
"""Prefill gate+up concat parity. install_fused_moe_glu stamps _kq_gate_up
(one KQuantSwitchLinear over the concatenated [gate; up] wire bytes) and the
sorted-prefill branch routes through it; the output must be bit-exact vs the
stock two-gather path, because the concat only changes which rows one gather
touches, never any row's accumulation."""
from __future__ import annotations

import numpy as np
import mlx.core as mx
import mlx.nn as nn
import pytest

from gguf import quants
from gguf.constants import GGMLQuantizationType as GT
from mlx_lm.models.switch_layers import SwitchGLU

import gmlx.load.modules as modules

DIM, HID, E = 256, 256, 8


class _Holder(nn.Module):
    def __init__(self, glu):
        super().__init__()
        self.experts = glu


def _q8_wire(rng, rows, cols):
    wf = rng.standard_normal((rows, cols), dtype=np.float32)
    return quants.quantize(wf, GT.Q8_0).astype(np.uint8)


def _write_wire(rng, model):
    for proj, cols in (("gate_proj", DIM), ("up_proj", DIM),
                       ("down_proj", HID)):
        leaf = getattr(model.experts, proj)
        wire = np.stack(
            [_q8_wire(rng, leaf.weight.shape[1], cols) for _ in range(E)], 0)
        leaf.weight = mx.array(wire)


def _build(rng, weights_after_install=False):
    import mlx_kquant as kq
    if not hasattr(kq, "moe_glu_gather_kq"):
        pytest.skip("mlx_kquant without fused MoE kernels")
    model = _Holder(SwitchGLU(DIM, HID, E))
    model.eval()  # training=True would skip the concat path entirely
    meta = {f"experts.{p}.weight": "q8_0"
            for p in ("gate_proj", "up_proj", "down_proj")}
    modules.install_kquant_modules(model, meta)
    if weights_after_install:
        # loader order: module install first, weight update afterwards
        assert modules.install_fused_moe_glu(model) == 1
        _write_wire(rng, model)
    else:
        _write_wire(rng, model)
        assert modules.install_fused_moe_glu(model) == 1
    return model


def _prefill_inputs(rng, tokens, k=4):
    x = mx.array(rng.standard_normal((1, tokens, DIM)).astype(np.float16))
    inds = mx.array(rng.integers(0, E, (1, tokens, k)).astype(np.uint32))
    return x, inds


@pytest.mark.parametrize("tokens", [16, 150])
@pytest.mark.parametrize("weights_after_install", [False, True])
def test_concat_prefill_matches_stock(tokens, weights_after_install,
                                      monkeypatch):
    rng = np.random.default_rng(7)
    model = _build(rng, weights_after_install=weights_after_install)
    # install only stamps the pending flag; the concat is built from the
    # live wire bytes on the first sorted-prefill call (the loader writes
    # weights after module install, so an eager build would freeze the
    # ctor placeholders)
    assert getattr(model.experts, "_kq_gate_up", None) is None
    assert model.experts._kq_gate_up_pending
    x, inds = _prefill_inputs(rng, tokens)  # tokens*4 >= 64: sorted prefill
    y_concat = model.experts(x, inds)
    mx.eval(y_concat)
    gu = model.experts._kq_gate_up
    assert gu is not None and not model.experts._kq_gate_up_pending
    assert gu.weight.shape == (E, 2 * HID, model.experts.gate_proj.weight.shape[2])
    monkeypatch.setattr(modules, "_GATEUP_CONCAT_ENABLED", False)
    y_stock = model.experts(x, inds)
    mx.eval(y_stock)
    assert y_concat.dtype == y_stock.dtype
    assert np.array_equal(np.array(y_concat), np.array(y_stock)), (
        np.abs(np.array(y_concat, np.float32)
               - np.array(y_stock, np.float32)).max())


def test_concat_skipped_when_disabled(monkeypatch):
    monkeypatch.setattr(modules, "_GATEUP_CONCAT_ENABLED", False)
    model = _build(np.random.default_rng(7))
    assert getattr(model.experts, "_kq_gate_up", None) is None
    assert not getattr(model.experts, "_kq_gate_up_pending", False)


def test_decode_width_ignores_concat():
    rng = np.random.default_rng(7)
    model = _build(rng)
    x, inds = _prefill_inputs(rng, 4)  # 16 routed rows < 64: decode widths
    y = model.experts(x, inds)
    mx.eval(y)
    assert y.shape == (1, 4, 4, DIM)
