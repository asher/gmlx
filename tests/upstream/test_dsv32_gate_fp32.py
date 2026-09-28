"""DeepSeek-V3.2 MoE router selects experts from float32 logits, as HF does
(F.linear on float32 hidden and weight). Stock mlx-lm MoEGate rounds the
256 router logits to the model dtype before the grouped top-k. The
weights are bfloat16 by design."""
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm.models import deepseek_v32 as ds

from gmlx.upstream.dsv32_patches import _patch_dsv32_moe_gate_fp32

_CFG = SimpleNamespace(num_experts_per_tok=8, norm_topk_prob=True,
                       n_routed_experts=256, routed_scaling_factor=2.5,
                       n_group=8, topk_group=4, hidden_size=512,
                       topk_method="noaux_tc")


class _Holder(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = ds.MoEGate(_CFG)


def _holder():
    mx.random.seed(0)
    h = _Holder()
    h.gate.weight = (mx.random.normal((256, 512)) * 0.02).astype(mx.bfloat16)
    h.gate.e_score_correction_bias = mx.random.uniform(shape=(256,)) * 1e-3
    return h


def _sets(inds):
    return [set(r) for r in np.array(inds).reshape(-1, 8).tolist()]


def test_router_selects_from_float32_logits(monkeypatch):
    monkeypatch.setenv("GMLX_DSV32_GATE_FP32", "1")
    h = _holder()
    mx.random.seed(1)
    x = mx.random.normal((1, 256, 512)).astype(mx.bfloat16)
    g = h.gate
    ref_inds, ref_scores = ds.group_expert_select(
        x.astype(mx.float32) @ g.weight.T.astype(mx.float32),
        g.e_score_correction_bias, g.top_k, g.n_group, g.topk_group,
        g.routed_scaling_factor, g.norm_topk_prob)
    stock_inds, _ = g(x)
    assert _sets(stock_inds) != _sets(ref_inds)  # the defect is reachable
    _patch_dsv32_moe_gate_fp32(h)
    inds, scores = g(x)
    assert _sets(inds) == _sets(ref_inds)
    order = mx.argsort(inds, axis=-1)
    ref_order = mx.argsort(ref_inds, axis=-1)
    np.testing.assert_allclose(
        np.array(mx.take_along_axis(scores, order, axis=-1)),
        np.array(mx.take_along_axis(ref_scores, ref_order, axis=-1)),
        atol=1e-6)
