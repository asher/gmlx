"""Real-weights coverage of the ``load_vlm_mtp_model`` build seam.

This is the third way gmlx reaches mlx-vlm forward code, next to
``load_mtp_model`` (owned classes at construction) and plain
``load_vlm_model``: the multimodal MTP target is built stock by
mlx_vlm.utils, never sees the owned-class selector, and must come out
with the full verify patch regime engaged, tiled-V included.

``integration`` + ``slow``; needs ``KQUANT_TEST_GGUF_DIR`` to contain a
qwen3.5/3.6 native-MTP model with a sibling mmproj GGUF that discovery
pairs to it. Every assert here is an engagement proof: class census,
patched-symbol identity, then a short greedy forward.
"""

from __future__ import annotations

import os

import pytest

pytest.importorskip("mlx_vlm")

pytestmark = [pytest.mark.integration, pytest.mark.slow]

_NEEDS_GPU = pytest.mark.skipif(
    bool(os.environ.get("KQUANT_FORCE_CPU")),
    reason="real-weights VLM MTP load is a GPU workload")


def _qwen_mtp_pair(gguf_dir):
    from gmlx.config import DiscoverSpec
    from gmlx.discovery import scan_dirs
    from gmlx.headerscan import scan_gguf

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
    """(model, drafter) loaded through load_vlm_mtp_model, with the
    process-global patch state restored at module teardown so later
    tests in the same run see clean globals."""
    from mlx_vlm.models.qwen3_5 import language as _L

    from gmlx import gdn_patches as gp
    from gmlx import qwen35_verify_fold
    from gmlx.mtp_load import load_vlm_mtp_model

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
def test_vlm_mtp_target_is_stock_with_full_patch_regime(vlm_mtp_load):
    import importlib

    from mlx_lm.models import gated_delta as gd
    from mlx_vlm.models.qwen3_5 import language as _L

    from gmlx import qwen35_attn, qwen35_owned

    model, drafter, _tok = vlm_mtp_load
    lm = model.language_model

    # Built stock: no owned class anywhere in the tree.
    assert not qwen35_owned.is_owned_language_model(model)
    owned_leak = {
        type(m).__name__
        for m in lm.modules()
        if type(m).__name__.startswith("Owned")
    }
    assert not owned_leak, f"owned classes on the stock build path: {owned_leak}"

    # tiled-V correctness rebind reached the vlm gated_delta module.
    vgd = importlib.import_module("mlx_vlm.models.qwen3_5.gated_delta")
    assert vgd.gated_delta_ops is gd.gated_delta_ops
    assert vgd._gated_delta_with_states_ops.__module__.startswith("gmlx")
    assert vgd._gated_delta_with_states_kernel is None

    # Full verify patch regime engaged on the module globals the stock
    # forward reads.
    assert _L._target_verify_linear.__module__.startswith("gmlx")
    assert _L._target_verify_left_padded_attention.__module__.startswith(
        "gmlx"
    )
    assert _L.scaled_dot_product_attention.__module__ == (
        "gmlx.qwen35_verify_fold"
    )
    assert (
        _L._qwen3_5_ragged_decode_attention
        is qwen35_attn.ragged_decode_attention
    )
    assert _L.Qwen3_5GatedDeltaNet.__call__.__module__ == "gmlx.gdn_patches"

    assert drafter is not None


@_NEEDS_GPU
def test_vlm_mtp_target_short_greedy_forward(vlm_mtp_load):
    import mlx.core as mx

    model, _drafter, tokenizer = vlm_mtp_load
    lm = model.language_model
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
