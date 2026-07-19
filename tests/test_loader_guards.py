#!/usr/bin/env python3
"""Loader guard helpers: `_own` must survive bf16 wire tensors (numpy has no
bf16 buffer format) and `_RemapDict` must refuse two source tensors remapping
to the same target name (a silent overwrite would serve a subtly wrong model)."""

from __future__ import annotations

import mlx.core as mx
import pytest

from gmlx.loader import _own, _RemapDict


def test_own_bf16_copies_via_f32():
    src = mx.array([1.0, -2.5, 3.25], dtype=mx.bfloat16)
    out = _own(src)
    assert out.dtype == mx.float32
    assert mx.array_equal(out, src.astype(mx.float32))


def test_own_f32_roundtrip():
    src = mx.arange(8).astype(mx.float32)
    out = _own(src)
    assert out.dtype == mx.float32
    assert mx.array_equal(out, src)


def test_remap_dict_rejects_duplicate_target():
    sink = _RemapDict()
    sink["model.layers.0.self_attn.q_proj.weight"] = 1
    with pytest.raises(ValueError, match="remap collision"):
        sink["model.layers.0.self_attn.q_proj.weight"] = 2


def test_remap_result_allows_native_fp_repack():
    # gpt-oss regression: the native-fp repack replaces each mxfp4 .weight and
    # .scales entry in place after remap. The anti-clobber guard must not
    # outlive remap population, or every mxfp4/nvfp4 model fails to load with
    # a phantom "remap collision".
    import numpy as np

    from gmlx.loader import remap_arrays
    from gmlx.native_fp import repack_native_fp_weights

    raw = mx.array(np.zeros((4, 2 * 17), dtype=np.uint8))  # 2 mxfp4 blocks/row
    arrays = {
        "blk.0.ffn_up_exps.weight": raw,
        "blk.0.ffn_up_exps.scales": mx.zeros((1,), dtype=mx.uint8),
    }
    codecs = {"blk.0.ffn_up_exps.weight": "mxfp4"}
    hf, hf_meta, _stats = remap_arrays(arrays, codecs, "gpt-oss")
    assert type(hf) is dict
    assert repack_native_fp_weights(hf, hf_meta) == 1
    key = "model.layers.0.mlp.experts.up_proj.weight"
    assert hf[key].dtype == mx.uint32
    assert hf["model.layers.0.mlp.experts.up_proj.scales"].dtype == mx.uint8
