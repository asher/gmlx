"""A native-head MTP load must fail before any heavy work, naming the real
cause, when the GGUF's header advertises a head the file does not carry: a
hand-stripped quant keeps ``nextn_predict_layers`` while the nextn block is
gone, and a header/tensor block-index mismatch yields an empty remap the
same way. Both native-head families store the head as
``blk.{num_hidden_layers}.nextn.*``."""

from __future__ import annotations

import pytest

from gmlx.spec.mtp_load import require_native_head_tensors


def _config(model_type: str, layers: int = 64) -> dict:
    return {
        "model_type": model_type,
        "num_hidden_layers": layers,
        "mtp_num_hidden_layers": 1,
    }


def test_headless_quant_raises_with_the_real_cause():
    arrays = {"blk.0.attn_q.weight": None, "output.weight": None}
    with pytest.raises(ValueError, match="no blk.64.nextn"):
        require_native_head_tensors(arrays, _config("qwen3_5"))


def test_block_index_mismatch_raises():
    # nextn tensors exist but at the wrong block: the remap would silently
    # yield an empty drafter, so the gate must treat this as absent too.
    arrays = {"blk.48.nextn.eh_proj.weight": None}
    with pytest.raises(ValueError, match="blk.64.nextn"):
        require_native_head_tensors(arrays, _config("qwen3_5"))


def test_hy3_block_marker():
    arrays = {"blk.0.attn_q.weight": None}
    with pytest.raises(ValueError, match="blk.80.nextn"):
        require_native_head_tensors(arrays, _config("hy_v3", layers=80))


def test_present_head_passes():
    arrays = {"blk.64.nextn.eh_proj.weight": None}
    require_native_head_tensors(arrays, _config("qwen3_5"))
    hy3 = {"blk.80.nextn.enorm.weight": None}
    require_native_head_tensors(hy3, _config("hy_v3", layers=80))
