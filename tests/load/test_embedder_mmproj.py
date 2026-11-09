"""EmbeddingGemma 2's media towers from an mmproj GGUF.

A tiny Gemma 4 mmproj is written from random towers, as F32, as F16 with
F32 norms (llama.cpp's F16 layout) and with Q8_0 matmuls, then loaded by
``load_towers``. The towers keep 16-bit weights at rest: with float32
activations an F16 file stays F16, and a Q8_0 file keeps its matmuls as
kquant modules.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx.utils import tree_flatten

import gmlx.load.vlm as vlm
from gmlx.load import embedder_mmproj as em
from gmlx.models.embedders.gemma_embedding2 import Towers

_TEXT = 64


def _meta(**over) -> dict:
    meta = {
        "general.architecture": "clip",
        "clip.has_vision_encoder": True,
        "clip.vision.projector_type": "gemma4v",
        "clip.vision.projection_dim": _TEXT,
        "clip.vision.image_size": 224,
        "clip.vision.patch_size": 16,
        "clip.vision.embedding_length": 32,
        "clip.vision.feed_forward_length": 64,
        "clip.vision.block_count": 1,
        "clip.vision.attention.head_count": 2,
        "clip.vision.attention.layer_norm_epsilon": 1e-6,
        "clip.has_audio_encoder": True,
        "clip.audio.projector_type": "gemma4a",
        "clip.audio.projection_dim": _TEXT,
        "clip.audio.embedding_length": 32,
        "clip.audio.feed_forward_length": 128,
        "clip.audio.block_count": 1,
        "clip.audio.attention.head_count": 2,
        "clip.audio.num_mel_bins": 128,
    }
    meta.update(over)
    return meta


def _gguf_names() -> list[str]:
    names = ["v.patch_embd.weight", "v.position_embd.weight",
             "mm.input_projection.weight", "a.input_projection.weight",
             "a.pre_encode.out.weight", "a.pre_encode.out.bias",
             "mm.a.input_projection.weight"]
    for c in (0, 1):
        names += [f"a.conv1d.{c}.weight", f"a.conv1d.{c}.norm.weight"]
    for sub in ("attn_q", "attn_k", "attn_v", "attn_out", "ffn_gate", "ffn_up",
                "ffn_down", "ln1", "ln2", "attn_post_norm", "ffn_post_norm",
                "attn_q_norm", "attn_k_norm"):
        names.append(f"v.blk.0.{sub}.weight")
    for sub in vlm._GEMMA4A_CLIP_LINEAR:
        names.append(f"a.blk.0.{sub}.weight")
        names += [f"a.blk.0.{sub}.{s}" for s in vlm._GEMMA4V_CLIP_SCALARS]
    for sub in [*vlm._GEMMA4A_NORM, "per_dim_scale", "attn_k_rel", "conv_dw"]:
        names.append(f"a.blk.0.{sub}.weight")
    return names


def _to_gguf(name: str, params: dict) -> np.ndarray:
    """The GGUF tensor ``name`` that the remap maps onto ``params``."""
    if name.startswith(("a.", "mm.a.")):
        target, transform = vlm._gemma4_audio_name(name)
    else:
        target, is_patch = vlm._gemma4_vision_name(name)
        transform = "patch" if is_patch else None
    v = np.array(params[target].astype(mx.float32))
    if transform == "patch":            # [out, kh*kw*C] -> [out, C, kh, kw]
        return v.reshape(v.shape[0], 16, 16, 3).transpose(0, 3, 1, 2)
    if transform == "conv2d":           # [out, kh, kw, in] -> [out, in, kh, kw]
        return v.transpose(0, 3, 1, 2)
    if transform == "conv_dw":          # [C, K, 1] -> [C, K]
        return v[..., 0]
    if transform == "scalar":
        return v.reshape(1)
    return v


def _random_towers(meta) -> Towers:
    towers = Towers(_TEXT, vlm.gemma4_vision_config(meta),
                    vlm.gemma4_audio_config(meta))
    rng = np.random.default_rng(0)
    weights = []
    for name, p in tree_flatten(towers.parameters()):
        if p.ndim == 0:   # clip bounds
            w = np.float32(3.0 if "max" in name else -3.0)
        elif p.ndim == 1:
            w = 1.0 + 0.2 * rng.standard_normal(p.shape)
        else:
            w = rng.standard_normal(p.shape) / np.sqrt(np.prod(p.shape[1:]))
        weights.append((name, mx.array(np.asarray(w, np.float32))))
    towers.load_weights(weights)
    return towers


def _write(path, meta, params, kind: str) -> str:
    from gguf import GGMLQuantizationType as GT
    from gguf import GGUFWriter, quants

    w = GGUFWriter(str(path), "clip")
    for k, v in meta.items():
        if k == "general.architecture":
            continue
        if isinstance(v, bool):
            w.add_bool(k, v)
        elif isinstance(v, int):
            w.add_uint32(k, v)
        elif isinstance(v, float):
            w.add_float32(k, v)
        else:
            w.add_string(k, v)
    for name in _gguf_names():
        arr = _to_gguf(name, params).astype(np.float32)
        linear = arr.ndim == 2 and arr.shape[-1] % 32 == 0 and "position" not in name
        if kind == "q8_0" and linear:
            w.add_tensor(name, quants.quantize(arr, GT.Q8_0), raw_dtype=GT.Q8_0)
        elif kind == "f16" and arr.ndim >= 2:
            w.add_tensor(name, arr.astype(np.float16))
        else:
            w.add_tensor(name, arr)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return str(path)


@pytest.fixture(scope="module")
def reference():
    meta = _meta()
    towers = _random_towers(meta)
    return meta, towers, dict(tree_flatten(towers.parameters()))


def _cast(meta, params, dtype) -> Towers:
    """The reference towers with every weight cast to ``dtype``."""
    towers = Towers(_TEXT, vlm.gemma4_vision_config(meta),
                    vlm.gemma4_audio_config(meta))
    towers.load_weights([(k, v.astype(dtype)) for k, v in params.items()])
    return towers


def _run(towers):
    rng = np.random.default_rng(1)
    px = mx.array(rng.uniform(0, 1, (2, 3, 96, 96)).astype(np.float32))
    feat = mx.array(rng.standard_normal((1, 60, 128)).astype(np.float32))
    valid = mx.ones((1, 60), dtype=mx.bool_)
    img = towers.encode_images(px).astype(mx.float32)
    aud = towers.encode_audio(feat, valid).astype(mx.float32)
    return np.array(img), np.array(aud)


def _rel(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def test_f32_mmproj_loads_in_the_activation_dtype(tmp_path, reference):
    meta, towers, params = reference
    path = _write(tmp_path / "f32.gguf", meta, params, "f32")
    got = em.load_towers(path, text_hidden=_TEXT, act_dtype=mx.bfloat16,
                         log=lambda *a: None)
    dtypes = {v.dtype for _, v in tree_flatten(got.parameters())}
    assert dtypes == {mx.bfloat16}
    for g, w in zip(_run(got), _run(_cast(meta, params, mx.bfloat16))):
        assert _rel(g, w) < 1e-3


def test_f16_mmproj_stays_16_bit_under_float32_activations(tmp_path, reference):
    meta, towers, params = reference
    path = _write(tmp_path / "f16.gguf", meta, params, "f16")
    got = em.load_towers(path, text_hidden=_TEXT, act_dtype=mx.float32,
                         log=lambda *a: None)
    flat = dict(tree_flatten(got.parameters()))
    q = "vision_tower.encoder.layers.0.self_attn.q_proj.linear.weight"
    norm = "vision_tower.encoder.layers.0.input_layernorm.weight"
    assert flat[q].dtype == mx.float16
    assert flat[norm].dtype == mx.float32
    # The patch projection sets the tower's activation dtype.
    proj = "vision_tower.patch_embedder.input_proj.weight"
    assert flat[proj].dtype == mx.float32
    img, aud = _run(got)
    want_img, want_aud = _run(towers)
    assert _rel(img, want_img) < 5e-3 and _rel(aud, want_aud) < 5e-3


def test_q8_0_mmproj_keeps_its_matmuls_quantized(tmp_path, reference):
    from mlx_kquant.nn import KQuantLinear

    meta, towers, params = reference
    path = _write(tmp_path / "q8.gguf", meta, params, "q8_0")
    got = em.load_towers(path, text_hidden=_TEXT, act_dtype=mx.bfloat16,
                         log=lambda *a: None)
    kq = [n for n, m in got.named_modules() if isinstance(m, KQuantLinear)]
    assert "vision_tower.encoder.layers.0.self_attn.q_proj.linear" in kq
    assert "embed_audio.embedding_projection" in kq
    assert not any(isinstance(m, nn.Linear) and type(m) is nn.Linear
                   and n.endswith("q_proj.linear") for n, m in got.named_modules())
    for g, w in zip(_run(got), _run(_cast(meta, params, mx.bfloat16))):
        assert _rel(g, w) < 3e-2


@pytest.mark.parametrize("over, words", [
    ({"clip.vision.projection_dim": 32}, "projection_dim is 32"),
    ({"clip.audio.projector_type": "qwen2a"}, "audio projector is 'qwen2a'"),
    ({"general.architecture": "llama"}, "not an mmproj"),
    ({"clip.has_vision_encoder": False, "clip.has_audio_encoder": False},
     "no vision or audio encoder"),
])
def test_a_projector_for_another_model_is_refused(over, words):
    assert words in (em.mmproj_mismatch(_meta(**over), _TEXT) or "")


def test_the_real_header_passes():
    assert em.mmproj_mismatch(_meta(), _TEXT) is None


def test_transformed_weights_dequantize_to_the_chip_dtype():
    import mlx_kquant as kq

    w = mx.random.normal((8, 3, 32, 32), key=mx.random.key(2))
    wire, scales = kq.quantize(w, "q8_0")
    out, _, meta = vlm.remap_vision_arrays(
        {"v.patch_embd.weight": wire, "v.patch_embd.scales": scales}, "gemma4",
        mm_codecs={"v.patch_embd.weight": "q8_0"}, dequant_dtype=mx.bfloat16)
    assert meta == {}
    assert out["vision_tower.patch_embedder.input_proj.weight"].dtype == mx.bfloat16
