"""A quantized mmproj loads its tower matmuls as kquant modules.

The remap keeps the wire bytes of each quantized weight that its family maps
unchanged, and dequantizes a weight it transforms. In a tower only an
``nn.Linear`` takes wire bytes; every other module reads its weight as a
float array.
"""

import mlx.core as mx
import mlx.nn as nn
import mlx_kquant as kq
import numpy as np
from mlx.utils import tree_unflatten

import gmlx.load.vlm as vlm
from gmlx._vendor.mlx_vlm.models.gemma4.audio import AudioAttention
from gmlx._vendor.mlx_vlm.models.gemma4.config import AudioConfig
from gmlx.load.modules import install_kquant_modules


def _q8(shape, seed):
    w = mx.random.normal(shape, key=mx.random.key(seed))
    wire, scales = kq.quantize(w, "q8_0")
    deq = kq.dequantize(wire, scales, "q8_0", mx.float32).reshape(shape)
    return wire, scales, deq


def test_remap_packs_unchanged_weights_and_dequantizes_transformed_ones():
    q_wire, q_scales, _ = _q8((64, 64), 1)
    # A patch conv quantizes along its last axis like any tensor; gemma4
    # reshapes it to a Linear, so its bytes cannot stand in.
    p_wire, p_scales, p_deq = _q8((8, 3, 32, 32), 2)
    arrays = {"v.blk.0.attn_q.weight": q_wire, "v.blk.0.attn_q.scales": q_scales,
              "v.patch_embd.weight": p_wire, "v.patch_embd.scales": p_scales}
    codecs = {"v.blk.0.attn_q.weight": "q8_0", "v.patch_embd.weight": "q8_0"}

    out, _, meta = vlm.remap_vision_arrays(arrays, "gemma4", mm_codecs=codecs)

    q = "vision_tower.encoder.layers.0.self_attn.q_proj.linear.weight"
    p = "vision_tower.patch_embedder.input_proj.weight"
    assert meta == {q: "q8_0"}
    assert out[q] is q_wire
    assert out[q[: -len(".weight")] + ".scales"] is q_scales
    assert p[: -len(".weight")] + ".scales" not in out
    assert out[p].dtype == mx.float16
    want = np.array(vlm._patch_conv_to_linear(p_deq))
    np.testing.assert_allclose(np.array(out[p].astype(mx.float32)), want,
                               rtol=1e-3, atol=1e-3)


class _Tower(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(64, 16, bias=False)
        self.table = nn.Embedding(16, 64)


def test_only_a_linear_keeps_its_wire_bytes():
    p_wire, p_scales, _ = _q8((16, 64), 3)
    t_wire, t_scales, t_deq = _q8((16, 64), 4)
    weights = {"proj.weight": p_wire, "proj.scales": p_scales,
               "table.weight": t_wire, "table.scales": t_scales}
    meta = {"proj.weight": "q8_0", "table.weight": "q8_0"}

    n = vlm.unpack_unswappable_tower_weights(_Tower(), weights, meta)

    assert n == 1
    assert meta == {"proj.weight": "q8_0"}
    assert weights["proj.weight"] is p_wire and "proj.scales" in weights
    assert "table.scales" not in weights
    assert weights["table.weight"].dtype == mx.float16
    np.testing.assert_allclose(
        np.array(weights["table.weight"].astype(mx.float32)), np.array(t_deq),
        rtol=1e-3, atol=1e-3)


def test_gemma4_audio_relative_positions_run_on_kquant_weights():
    # AudioAttention holds relative_k_proj again at _rel_pos.pos_proj, and
    # the position embedding casts its input to that module's weight dtype.
    cfg = AudioConfig(hidden_size=64, num_attention_heads=2,
                      attention_chunk_size=4, attention_context_left=5,
                      attention_context_right=0)
    wire, scales, deq = _q8((64, 64), 5)
    ref = AudioAttention(cfg)
    ref.relative_k_proj.weight = deq
    packed = AudioAttention(cfg)
    install_kquant_modules(packed, {"relative_k_proj.weight": "q8_0"})
    packed.update(tree_unflatten([("relative_k_proj.weight", wire),
                                  ("relative_k_proj.scales", scales)]))

    assert packed._rel_pos.pos_proj is packed.relative_k_proj
    b, u, w, n, h = 1, 2, 4, 2, 32
    c = packed.context_size
    q = mx.random.normal((b, u, w, n, h), key=mx.random.key(6))
    k = mx.random.normal((b, u, c, n, h), key=mx.random.key(7))
    want = ref._rel_pos(q, k)
    got = packed._rel_pos(q, k)
    rel = float(mx.linalg.norm(got - want) / mx.linalg.norm(want))
    assert rel < 1e-2
