"""The gpt-oss fused QKV decode wire: an adapter installed after load wraps
a projection in LoRAKQuantLinear, and the lazy wire build must hand the
module back to the stock path instead of reading the wire it no longer
has."""
from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

import gmlx.upstream.qkv_fuse as qf


class _PlainAttentionBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(8, 8, bias=True)
        self.k_proj = nn.Linear(8, 4, bias=True)
        self.v_proj = nn.Linear(8, 4, bias=True)
        self.num_attention_heads = 2
        self.num_key_value_heads = 1
        self.head_dim = 4


def test_adapted_projection_turns_the_fused_wire_off():
    from gmlx.load.modules import LoRAKQuantLinear

    m = _PlainAttentionBlock()
    m.__class__ = qf._make_fused(_PlainAttentionBlock)
    object.__setattr__(m, "_kq_wqkv", None)
    object.__setattr__(m, "_kq_bqkv", None)
    object.__setattr__(m, "_kq_qkv_off", False)
    m.q_proj = LoRAKQuantLinear(m.q_proj, mx.zeros((2, 8)), mx.zeros((8, 2)), 1.0)
    assert m._kq_build_fused() is None
    assert m._kq_qkv_off
