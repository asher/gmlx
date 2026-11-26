"""llama.cpp's fused shared-input tensors (attn_qkv_gate, ssm_beta_alpha,
hc_*_down_inject) load as the split tensors they merge."""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

import mlx.core as mx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))
import qwen35_tiny as T  # noqa: E402

from gmlx.load.gguf_layout import split_shared_input  # noqa: E402

IDS = [[5, 9, 30, 41, 7, 66, 12, 3, 90, 100, 15, 22]]


def _logits(path):
    import gmlx

    model = gmlx.load_model(str(path), verbose=False)[0]
    return np.array(model(mx.array(IDS)).astype(mx.float32))


@pytest.mark.parametrize("q8", [(), ("attn_qkv.weight", "attn_gate.weight",
                                     "attn_qkv_gate.weight")],
                         ids=["f32", "q8_0"])
def test_a_fused_qwen35_file_gives_the_split_file_logits(tmp_path, q8):
    v = T.values("qwen35")
    T.write(tmp_path / "split.gguf", "qwen35", v, q8=q8)
    T.write(tmp_path / "fused.gguf", "qwen35", v, fused=True, q8=q8)
    split = _logits(tmp_path / "split.gguf")
    fused = _logits(tmp_path / "fused.gguf")
    assert split.shape == (1, len(IDS[0]), len(T.tokens()))
    np.testing.assert_array_equal(split, fused)


def test_the_flash_next_inject_rows_split_off_with_their_codec():
    # qwen4exp: [hc_attn_down | hc_attn_inject], inject rows = streams.
    streams, rank, width = 4, 8, 32
    merged = mx.arange((rank + streams) * width, dtype=mx.uint8).reshape(
        rank + streams, width)
    name = "blk.3.hc_attn_down_inject.weight"
    arrays = {name: merged, "blk.3.hc_attn_down_inject.scales": mx.zeros((1,))}
    codec = {name: "q8_0"}
    shapes = {name: [width, rank + streams]}
    meta = {"qwen4exp.hyper_connection.count": streams}

    assert split_shared_input(arrays, codec, meta, shapes, "qwen4exp") == 1
    down = arrays["blk.3.hc_attn_down.weight"]
    inject = arrays["blk.3.hc_attn_inject.weight"]
    assert mx.array_equal(down, merged[:rank])
    assert mx.array_equal(inject, merged[rank:])
    assert codec == {"blk.3.hc_attn_down.weight": "q8_0",
                     "blk.3.hc_attn_inject.weight": "q8_0"}
    assert shapes == {"blk.3.hc_attn_down.weight": [width, rank],
                      "blk.3.hc_attn_inject.weight": [width, streams]}
    assert name not in arrays and "blk.3.hc_attn_down_inject.scales" not in arrays
