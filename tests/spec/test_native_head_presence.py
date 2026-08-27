"""A native-head MTP load must fail with the real cause when the GGUF's
header advertises a head the file does not carry (a hand-stripped quant
keeps ``nextn_predict_layers`` while the nextn tensors are gone)."""

from __future__ import annotations

import pytest

from gmlx.spec.mtp_load import _load_mtp_drafter


def _config(model_type: str) -> dict:
    return {
        "model_type": model_type,
        "num_hidden_layers": 64,
        "mtp_num_hidden_layers": 1,
    }


def test_headless_qwen_names_the_stripped_head():
    arrays = {"blk.0.attn_q.weight": None, "output.weight": None}
    with pytest.raises(ValueError, match="no nextn tensors"):
        _load_mtp_drafter(arrays, {}, "qwen35", _config("qwen3_5"), target=None)


def test_headless_hy3_names_the_missing_block():
    arrays = {"blk.0.attn_q.weight": None}
    with pytest.raises(ValueError, match="blk.64"):
        _load_mtp_drafter(arrays, {}, "hy3", _config("hy_v3"), target=None)


def test_present_head_passes_the_gate():
    # The gate must not fire when the nextn block exists; the load then
    # proceeds past it (and fails later here only because the arrays are
    # fakes, which is fine - the gate's ValueError has a distinct message).
    arrays = {"blk.64.nextn.eh_proj.weight": None}
    try:
        _load_mtp_drafter(arrays, {}, "qwen35", _config("qwen3_5"), target=None)
    except ValueError as e:
        assert "no nextn tensors" not in str(e)
    except Exception:
        pass
