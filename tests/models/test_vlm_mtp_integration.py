"""Real-weights coverage of the ``load_vlm_mtp_model`` build seam.

Since the spec-target seam, the multimodal MTP target is built on the
owned classes by default, exactly like ``load_mtp_model``; the
``GMLX_QWEN_OWNED=0`` fallback keeps the stock tree with the full verify
patch regime (unlike the text path's bare-stock fallback). The fixture
parametrizes over both arms so one save/restore block covers them, at
one full load per arm.

``integration`` + ``slow``; needs ``KQUANT_TEST_GGUF_DIR`` to contain a
qwen3.5/3.6 native-MTP model with a sibling mmproj GGUF that discovery
pairs to it. Every assert here is an engagement proof: class census,
patched-symbol identity or owned-forward counters, then a short greedy
forward.
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


@pytest.fixture(scope="module", params=["1", "0"],
                ids=["owned", "stock-fallback"])
def vlm_mtp_load(request, gguf_dir):
    """(model, drafter, owned_arm) loaded through load_vlm_mtp_model under
    GMLX_QWEN_OWNED={1,0}, with the process-global patch state restored at
    teardown so later tests in the same run see clean globals."""
    from mlx_vlm.models.qwen3_5 import language as _L

    import gmlx.upstream.gdn_patches as gp
    import gmlx.models.qwen35.verify_fold as qwen35_verify_fold
    from gmlx.spec.mtp_load import load_vlm_mtp_model

    pair = _qwen_mtp_pair(gguf_dir)
    saved_env = os.environ.get("GMLX_QWEN_OWNED")
    os.environ["GMLX_QWEN_OWNED"] = request.param
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
    finally:
        if saved_env is None:
            os.environ.pop("GMLX_QWEN_OWNED", None)
        else:
            os.environ["GMLX_QWEN_OWNED"] = saved_env
    yield model, drafter, tokenizer, request.param == "1"
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
def test_vlm_mtp_target_regime_matches_the_arm(vlm_mtp_load):
    import importlib

    from mlx_lm.models import gated_delta as gd
    from mlx_vlm.models.qwen3_5 import language as _L

    import gmlx.models.qwen35.attn as qwen35_attn
    import gmlx.models.qwen35.owned as qwen35_owned

    model, drafter, _tok, owned_arm = vlm_mtp_load
    lm = model.language_model

    # mlx-vlm tiled-V rebind reached in both arms: since the plain-load fix
    # (vendored 0.6.15 gated_delta escapes the mlx-lm patch) every qwen3_5
    # VLM load rebinds the vendored module, owned or stock. Not an identity
    # check against gd: a later load can re-close the mlx-lm patch, leaving
    # vgd on an older (equally tiled) instance.
    vgd = importlib.import_module("mlx_vlm.models.qwen3_5.gated_delta")
    assert "_tiled_gated_delta_ops" in vgd.gated_delta_ops.__qualname__
    assert "_tiled_gated_delta_ops" in gd.gated_delta_ops.__qualname__

    if owned_arm:
        # Owned by default: same classes as the text MTP target, fused GDN
        # armed in-tree, no stock verify patch needed on the vlm module.
        assert qwen35_owned.is_owned_language_model(model)
        armed = [
            m for m in lm.modules()
            if type(m).__name__ == "OwnedQwen3_5GatedDeltaNet"
            and getattr(m, "_gdn_owned_fused", None) is not None
        ]
        assert armed, "prepare_gdn armed no owned gated-delta layers"
    else:
        # GMLX_QWEN_OWNED=0 fallback: stock tree, full stock verify patch
        # regime on the module globals the stock forward reads (the vlm
        # regime, unlike the bare text fallback).
        assert not qwen35_owned.is_owned_language_model(model)
        owned_leak = {
            type(m).__name__
            for m in lm.modules()
            if type(m).__name__.startswith("Owned")
        }
        assert not owned_leak, f"owned classes on the stock path: {owned_leak}"
        assert vgd._gated_delta_with_states_ops.__module__.startswith("gmlx")
        assert vgd._gated_delta_with_states_kernel is None
        assert _L._target_verify_linear.__module__.startswith("gmlx")
        assert _L._target_verify_left_padded_attention.__module__.startswith(
            "gmlx"
        )
        assert _L.scaled_dot_product_attention.__module__ == (
            "gmlx.models.qwen35.verify_fold"
        )
        assert (
            _L._qwen3_5_ragged_decode_attention
            is qwen35_attn.ragged_decode_attention
        )
        assert _L.Qwen3_5GatedDeltaNet.__call__.__module__ == (
            "gmlx.upstream.gdn_patches"
        )

    assert drafter is not None


@_NEEDS_GPU
def test_vlm_mtp_target_short_greedy_forward(vlm_mtp_load):
    import mlx.core as mx

    import gmlx.models.qwen35.owned as qwen35_owned

    model, _drafter, tokenizer, owned_arm = vlm_mtp_load
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
    # Engagement proof: the owned forward actually ran on the owned arm.
    if owned_arm:
        assert qwen35_owned.owned_call_count() > calls_before
