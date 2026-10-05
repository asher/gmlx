"""Real-weights coverage of the ``load_vlm_mtp_model`` build seam.

The multimodal MTP target is built on the owned classes, exactly like
``load_mtp_model``. The fixture restores the process-global patch state
after its one full load.

``integration`` + ``slow``; needs ``KQUANT_TEST_GGUF_DIR`` to contain a
qwen3.5/3.6 native-MTP model with a sibling mmproj GGUF that discovery
pairs to it. Every assert here is an engagement proof: class census,
patched-symbol identity or owned-forward counters, then a short greedy
forward.
"""

from __future__ import annotations

import os

import pytest

pytest.importorskip("gmlx._vendor.mlx_vlm")

pytestmark = pytest.mark.integration

_NEEDS_GPU = pytest.mark.skipif(
    bool(os.environ.get("KQUANT_FORCE_CPU")),
    reason="real-weights VLM MTP load is a GPU workload")


def _qwen_mtp_pair(gguf_dir):
    from gmlx.config import DiscoverSpec
    from gmlx.load.discovery import scan_dirs
    from gmlx.load.headerscan import scan_gguf

    spec = DiscoverSpec(
        dir=None, recursive=True, pair_mmproj=True, speculative="auto"
    )
    paired = [m for m in scan_dirs([spec], [str(gguf_dir)]) if m.mmproj]
    for m in paired:
        kv = scan_gguf(m.path, include_tensors=False).kv
        if str(kv.get("general.architecture", "")).startswith("qwen3"):
            return m
    pytest.skip(
        "no qwen3.x model with a paired sibling mmproj under "
        "KQUANT_TEST_GGUF_DIR"
    )


@pytest.fixture(scope="module")
def vlm_mtp_load(gguf_dir):
    """(model, drafter, tokenizer) loaded through load_vlm_mtp_model, with
    the process-global patch state restored at teardown so later tests in
    the same run see clean globals."""
    from gmlx.models.qwen35 import language as _L

    import gmlx.upstream.gdn_patches as gp
    import gmlx.models.qwen35.verify_fold as qwen35_verify_fold
    from gmlx.spec.mtp_load import load_vlm_mtp_model

    pair = _qwen_mtp_pair(gguf_dir)
    saved = {
        "tvl": _L._target_verify_linear,
        "tvlpa": _L._target_verify_left_padded_attention,
        "sdpa": _L.scaled_dot_product_attention,
        "ragged": _L._qwen3_5_ragged_decode_attention,
        "gdn_call": _L.Qwen3_5GatedDeltaNet.__call__,
        "bf16_flag": gp._BF16_VERIFY_LINEAR_PATCHED,
        "bsdpa_flag": gp._BATCHED_VERIFY_SDPA_PATCHED,
        "fold_flag": qwen35_verify_fold._installed,
        "fv_installed": gp._FUSED_VERIFY_PATCH.installed,
        "fv_stock": gp._FUSED_VERIFY_PATCH.stock,
    }
    try:
        model, drafter, _config, tokenizer, _processor = load_vlm_mtp_model(
            pair.path, pair.mmproj
        )
    except ValueError as e:
        if "no native MTP head" in str(e):
            pytest.skip(f"paired qwen model has no native MTP head: {e}")
        raise
    yield model, drafter, tokenizer
    _L._target_verify_linear = saved["tvl"]
    _L._target_verify_left_padded_attention = saved["tvlpa"]
    _L.scaled_dot_product_attention = saved["sdpa"]
    _L._qwen3_5_ragged_decode_attention = saved["ragged"]
    _L.Qwen3_5GatedDeltaNet.__call__ = saved["gdn_call"]
    gp._BF16_VERIFY_LINEAR_PATCHED = saved["bf16_flag"]
    gp._BATCHED_VERIFY_SDPA_PATCHED = saved["bsdpa_flag"]
    qwen35_verify_fold._installed = saved["fold_flag"]
    gp._FUSED_VERIFY_PATCH.installed = saved["fv_installed"]
    gp._FUSED_VERIFY_PATCH.stock = saved["fv_stock"]


@_NEEDS_GPU
def test_vlm_mtp_target_is_the_owned_tree_armed(vlm_mtp_load):
    import importlib

    from gmlx._vendor.mlx_lm.models import gated_delta as gd

    import gmlx.models.qwen35.owned as qwen35_owned

    model, drafter, _tok = vlm_mtp_load
    lm = model.language_model

    # Each gated delta layer holds the tiled order of its GGUF, and both
    # scan modules keep their own functions.
    vgd = importlib.import_module("gmlx.models.qwen35.gated_delta")
    layers = [layer.linear_attn for layer in lm.model.layers if layer.is_linear]
    assert layers and all(layer.gdn_kv_tiled for layer in layers)
    assert lm.args.kv_head_layout == "tiled"
    for module in (vgd, gd):
        for name in ("gated_delta_ops", "gated_delta_kernel"):
            assert getattr(module, name).__module__ == module.__name__

    # Same classes as the text MTP target, fused GDN armed in-tree.
    assert qwen35_owned.is_owned_language_model(model)
    armed = [
        m for m in lm.modules()
        if type(m).__name__ == "OwnedQwen3_5GatedDeltaNet"
        and getattr(m, "_gdn_owned_fused", None) is not None
    ]
    assert armed, "prepare_gdn armed no owned gated-delta layers"
    assert drafter is not None


@_NEEDS_GPU
def test_vlm_mtp_target_short_greedy_forward(vlm_mtp_load):
    import mlx.core as mx

    import gmlx.models.qwen35.owned as qwen35_owned

    model, _drafter, tokenizer = vlm_mtp_load
    lm = model.language_model
    calls_before = qwen35_owned.owned_call_count()
    ids = mx.array([tokenizer.encode("The capital of France is")])
    cache = lm.make_cache()
    logits = lm(ids, cache=cache).logits
    toks = []
    for _ in range(4):
        nxt = mx.argmax(logits[:, -1, :], axis=-1)
        mx.eval(nxt)
        toks.append(int(nxt.item()))
        logits = lm(nxt[:, None], cache=cache).logits
    text = tokenizer.decode(toks)
    assert len(toks) == 4 and text.strip(), (toks, text)
    # Engagement proof: the owned forward actually ran.
    assert qwen35_owned.owned_call_count() > calls_before
