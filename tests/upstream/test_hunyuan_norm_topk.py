"""Hunyuan MoE renormalizes its top-k router weights (norm_topk_prob), as
the HF reference and llama.cpp's hunyuan-moe graph do. Stock mlx-lm
hunyuan.MoeBlock weights the experts by the raw softmax. The block runs
in float32."""
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm.models.hunyuan import MoeBlock

from gmlx.load.loader import _patch_hunyuan_norm_topk

_ARGS = SimpleNamespace(hidden_size=16, intermediate_size=32,
                        use_mixed_mlp_moe=False, num_shared_expert=1,
                        num_experts=8, moe_topk=4, moe_intermediate_size=None)


class _Holder(nn.Module):
    def __init__(self):
        super().__init__()
        self.blk = MoeBlock(_ARGS)


def _reference(block, x):
    gates = mx.softmax(block.gate(x).astype(mx.float32), axis=-1)
    inds = mx.argsort(-gates, axis=-1)[..., :block.top_k]
    w = mx.take_along_axis(gates, inds, axis=-1)
    w = w / w.sum(axis=-1, keepdims=True)
    return (block.switch_mlp(x, inds) * w[..., None]).sum(axis=-2)


def test_routed_weights_are_renormalized():
    mx.random.seed(3)
    holder = _Holder()
    mx.eval(holder.parameters())
    x = mx.random.normal((1, 6, 16))
    _patch_hunyuan_norm_topk(holder)
    got = np.array(holder.blk(x))
    ref = np.array(_reference(holder.blk, x))
    assert np.allclose(got, ref, atol=1e-5), np.abs(got - ref).max()
