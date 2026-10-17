"""Prompt groups with a prefix cache, for image models.

A multi-row prompt batch cuts every row at the lowest row checkpoint
column, and left padding can place that column inside another row's
image. The prefill loops move such a boundary to the image edge and drop
the checkpoint (``gmlx.gen.media_spans``), so gemma-4 under bidirectional
image attention keeps batched groups like the causal models.
"""

import importlib

import pytest
from gmlx._vendor.mlx_vlm.apc import APCManager

import gmlx.load.vlm as gvlm
import gmlx.spec.engine as se

from test_apc_media import _model
from test_vlm_deepseek4v import _config as _dsv4_config
from test_vlm_spec_target_swap import _q35_vlm


def _group_width(monkeypatch, lm, manager) -> int:
    """The prompt group width the generator forms at a decode width of 16
    with the checkpoint tier off."""
    ar = importlib.import_module("gmlx._vendor.mlx_vlm.generate.ar")

    class _ProbeBG:
        def __init__(self, model, processor, **kwargs):
            self.prefill_batch_size = kwargs.get("prefill_batch_size", 8)
            self.completion_batch_size = kwargs.get("completion_batch_size")

    monkeypatch.setattr(ar, "BatchGenerator", _ProbeBG)
    monkeypatch.setattr(se, "_SPEC_APC_CKPT_DISABLED", True)
    se._install_apc_manager_stash()
    bg = ar.BatchGenerator(lm, None, apc_manager=manager,
                           prefill_batch_size=8, completion_batch_size=16)
    return bg.prefill_batch_size


def _language_model(model_type, text):
    if model_type == "qwen3_5":
        return _q35_vlm().language_model
    if model_type == "deepseek_v4_vl":
        return gvlm.build_vlm_model(_dsv4_config())[0].language_model
    return _model(model_type, text).language_model


@pytest.mark.parametrize("model_type,text,width", [
    ("gemma4", {"use_bidirectional_attention": "vision"}, 8),
    ("gemma4_unified", {}, 8),
    ("gemma4", {}, 8),
    ("gemma4_unified", {"use_bidirectional_attention": None}, 8),
    ("glm5_next", None, 8),
    ("muse_glimmer", None, 8),
    ("muse_spark", None, 8),
    ("qwen3_5", None, 8),
    # DSv4 VL attends bidirectionally inside an image block, and its
    # pooled caches get groups of one from the apc_pooling gate
    # (model_has_pools). The probe generator skips that gate, so it shows 8.
    ("deepseek_v4_vl", None, 8),
])
def test_prefix_cache_groups_follow_the_decode_width(
        model_type, text, width, monkeypatch):
    lm = _language_model(model_type, text)
    manager = APCManager(num_blocks=64, block_size=16)
    assert _group_width(monkeypatch, lm, manager) == width
    assert _group_width(monkeypatch, lm, None) == 8
