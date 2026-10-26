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


def _q8(shape, seed, scale=1.0):
    w = mx.random.normal(shape, key=mx.random.key(seed)) * scale
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


def _minicpm_tower_arrays():
    """A Q8_0 MiniCPM-V mmproj for a 64-wide two-block tower: every matmul
    and the position table quantized, norms, biases and the patch conv in
    float, as the mixed-quant uploads ship them."""
    d, ff, wi, out = 64, 128, 64, 64
    shapes = {"v.patch_embd.weight": (d, 3, 14, 14), "v.patch_embd.bias": (d,),
              "v.position_embd.weight": (64, d), "v.post_ln.weight": (d,),
              "v.post_ln.bias": (d,), "mm.input_norm.weight": (4 * d,),
              "mm.input_norm.bias": (4 * d,), "mm.up.weight": (4 * d, 4 * d),
              "mm.up.bias": (4 * d,), "mm.down.weight": (out, 4 * d),
              "mm.down.bias": (out,)}
    for i in range(2):
        for n, s in (("attn_q", (d, d)), ("attn_k", (d, d)),
                     ("attn_v", (d, d)), ("attn_out", (d, d)),
                     ("ffn_up", (ff, d)), ("ffn_down", (d, ff))):
            shapes[f"v.blk.{i}.{n}.weight"] = s
            shapes[f"v.blk.{i}.{n}.bias"] = (s[0],)
        for n in ("ln1", "ln2"):
            shapes[f"v.blk.{i}.{n}.weight"] = (d,)
            shapes[f"v.blk.{i}.{n}.bias"] = (d,)
    for n, s in (("attn_q", (d, d)), ("attn_k", (d, d)), ("attn_v", (d, d)),
                 ("attn_out", (d, d)), ("ds_ffn_up", (wi, 4 * d)),
                 ("ds_ffn_down", (d, wi))):
        shapes[f"v.vit_merger.{n}.weight"] = s
        shapes[f"v.vit_merger.{n}.bias"] = (s[0],)
    for n, w in (("ln1", d), ("ds_ln", 4 * d)):
        shapes[f"v.vit_merger.{n}.weight"] = (w,)
        shapes[f"v.vit_merger.{n}.bias"] = (w,)
    wire, floats, codecs = {}, {}, {}
    for seed, (name, shape) in enumerate(sorted(shapes.items())):
        if len(shape) == 2:
            w, s, deq = _q8(shape, seed, scale=0.05)
            wire[name], wire[name[:-len(".weight")] + ".scales"] = w, s
            floats[name], codecs[name] = deq, "q8_0"
        else:
            arr = mx.random.normal(shape, key=mx.random.key(seed)) * 0.05
            if name.endswith(("ln1.weight", "ln2.weight", "ln.weight",
                              "norm.weight", "post_ln.weight")):
                arr = arr + 1.0
            wire[name] = floats[name] = arr
    return wire, floats, codecs


def test_a_q8_0_minicpm_v_mmproj_runs_on_kquant_modules():
    from gmlx.models.minicpmv.vision import VisionConfig, VisionModel

    cfg = VisionConfig(hidden_size=64, intermediate_size=128,
                       num_hidden_layers=2, num_attention_heads=2,
                       insert_layer_id=0, window_intermediate_size=64,
                       out_hidden_size=64, position_grid=8)
    wire, floats, codecs = _minicpm_tower_arrays()
    prefix = "vision_tower."

    def tower(arrays, mm_codecs):
        out, skipped, meta = vlm.remap_vision_arrays(
            arrays, "minicpmv4_7", mm_codecs=mm_codecs)
        assert skipped == []
        model = nn.Module()
        model.vision_tower = VisionModel(cfg)
        vlm.unpack_unswappable_tower_weights(model, out, meta)
        install_kquant_modules(model, meta)
        model.update(tree_unflatten(list(out.items())))
        return model.vision_tower, meta

    packed, meta = tower(wire, codecs)
    ref, _ = tower(floats, {})
    assert f"{prefix}encoder.layers.0.self_attn.q_proj.weight" in meta
    assert f"{prefix}embeddings.position_embedding.weight" not in meta
    pv = mx.random.normal((2 * 64, 14 * 14 * 3), key=mx.random.key(9))
    sizes = [[8, 8], [8, 8]]
    want, got = ref(pv, sizes), packed(pv, sizes)
    rel = float(mx.linalg.norm(got - want) / mx.linalg.norm(want))
    # kquant computes float32 activations in bfloat16, about 0.3% per
    # linear. Over this tower that is 0.5% on the GPU and 1.0% on the CPU;
    # a wrong remap or layout is tens of percent.
    assert rel < 2e-2
