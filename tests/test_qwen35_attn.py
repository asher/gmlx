"""Owned Qwen3_5Attention: rebind wiring + route identity vs the patched path.

The oracle for every numerics test is the CURRENT PATCHED path (verify
fold + batched-verify masked SDPA + unified ragged plan, installed in
production order), not raw stock: the patched path carries the certified
production numerics. Both arms use the stock LanguageModel class so the
comparison isolates the attention treatment. Attention head_dim is 64 so
the ragged decode kernels engage at toy scale (their allowlist starts at
64); the f32 parametrization keeps them declined so the per-pad-group
loop is exercised too. Arms are built in eval mode (training-mode toys
route upstream dispatchers differently from production).
"""

import os
from types import SimpleNamespace

import mlx.core as mx
import pytest

pytest.importorskip("mlx_vlm.models.qwen3_5.language")

from mlx_vlm.models.cache import ArraysCache, BatchKVCache
from mlx_vlm.models.qwen3_5.config import TextConfig as Q35TextConfig
from mlx_vlm.models.qwen3_5.language import LanguageModel as Q35LanguageModel
from mlx_vlm.models.qwen3_5.language import Qwen3_5GatedDeltaNet
from mlx_vlm.models import base as _B
from mlx_vlm.models.qwen3_5 import language as _L

from gmlx import gdn_patches, qwen35_attn, qwen35_gdn, qwen35_verify_fold
from gmlx import ragged_decode

ATOL = 2e-2
_NEEDS_GPU = pytest.mark.skipif(
    bool(os.environ.get("KQUANT_FORCE_CPU")),
    reason="qwen3_5 GDN forward is Metal-only")

PROMPT = (3, 17, 42, 99, 7, 63, 5, 28)


def _cfg():
    return Q35TextConfig(
        model_type="qwen3_5",
        hidden_size=64,
        intermediate_size=128,
        linear_num_value_heads=4,
        linear_num_key_heads=2,
        linear_key_head_dim=32,
        linear_value_head_dim=32,
        linear_conv_kernel_dim=4,
        num_hidden_layers=4,
        num_attention_heads=4,
        rms_norm_eps=1e-6,
        vocab_size=128,
        num_key_value_heads=2,
        max_position_embeddings=16384,
        tie_word_embeddings=True,
        head_dim=64,
        rope_parameters={
            "type": "default",
            "mrope_section": [2, 1, 1],
            "rope_theta": 100000,
            "partial_rotary_factor": 0.25,
        },
        full_attention_interval=4,
    )


def _top():
    return SimpleNamespace(
        vision_config=SimpleNamespace(spatial_merge_size=2),
        image_token_id=124,
        video_token_id=125,
        vision_start_token_id=126,
    )


def _lm(dtype=None):
    mx.random.seed(11)
    lm = Q35LanguageModel(_cfg(), _top())
    if dtype is not None:
        lm.set_dtype(dtype)
    # Eval mode mirrors production loads (training-mode toys take
    # upstream ops fallbacks production never runs).
    lm.eval()
    mx.eval(lm.parameters())
    return lm


def _attns(lm):
    return [
        layer.self_attn for layer in lm.model.layers if not layer.is_linear
    ]


def _batch_caches(lm, pads):
    return [
        ArraysCache(size=2, left_padding=list(pads))
        if layer.is_linear
        else BatchKVCache(list(pads))
        for layer in lm.layers
    ]


def _close(a, b, atol):
    return (
        mx.abs(a.astype(mx.float32) - b.astype(mx.float32)).max().item() < atol
    )


def _greedy_chain(lm, ids, cache, steps):
    toks = []
    logits = lm(ids, cache=cache).logits
    for _ in range(steps):
        nxt = mx.argmax(logits[:, -1, :], axis=-1)
        toks.append(nxt)
        logits = lm(nxt[:, None], cache=cache).logits
    out = mx.stack(toks, axis=1)
    # Evaluate before the other arm runs (aliasing trap: an unevaluated
    # graph reads through later in-place cache mutations).
    mx.eval(out, logits)
    return out, logits


def _arms(dtype=None):
    """(patched-stock lm, owned-rebound lm) with identical weights.

    The patched arm installs the full production patch set in production
    install order (verify fold at load, batched-verify masked SDPA after,
    unified ragged plan process-global); the owned arm goes through the
    rebinds. Caller must use the _restore_patches fixture.
    """
    patched = _lm(dtype)
    gdn_patches._patch_gated_delta_tiled_v()
    gdn_patches._patch_mlxvlm_gated_delta_tiled_v()
    gdn_patches._patch_gated_delta_fused_verify(patched)
    qwen35_verify_fold.install_qwen35_verify_fold()
    gdn_patches._patch_batched_verify_sdpa()
    ragged_decode.install_unified_ragged_plan()
    owned = _lm(dtype)
    qwen35_gdn.rebind_gdn(owned)
    qwen35_attn.rebind_attn(owned)
    return patched, owned


@pytest.fixture
def _restore_patches(monkeypatch):
    # Restore every module global the arm installers touch AND the
    # install-once latches (restoring the symbol without the latch turns
    # the next install into a silent no-op).
    saved_gdn_call = Qwen3_5GatedDeltaNet.__call__
    saved_fv_installed = gdn_patches._FUSED_VERIFY_PATCH.installed
    saved_fv_stock = gdn_patches._FUSED_VERIFY_PATCH.stock
    saved_lpa = _L._target_verify_left_padded_attention
    saved_sdpa = _L.scaled_dot_product_attention
    saved_ragged = _L._qwen3_5_ragged_decode_attention
    saved_fold_installed = qwen35_verify_fold._installed
    saved_bvs = gdn_patches._BATCHED_VERIFY_SDPA_PATCHED
    saved_bvs_stock = gdn_patches._STOCK_VERIFY_LEFT_PADDED
    gdn_patches._FUSED_VERIFY_PATCH.installed = False
    qwen35_verify_fold._installed = False
    gdn_patches._BATCHED_VERIFY_SDPA_PATCHED = False
    monkeypatch.delenv("GMLX_FUSED_GDN", raising=False)
    yield
    Qwen3_5GatedDeltaNet.__call__ = saved_gdn_call
    gdn_patches._FUSED_VERIFY_PATCH.installed = saved_fv_installed
    gdn_patches._FUSED_VERIFY_PATCH.stock = saved_fv_stock
    _L._target_verify_left_padded_attention = saved_lpa
    _L.scaled_dot_product_attention = saved_sdpa
    _L._qwen3_5_ragged_decode_attention = saved_ragged
    qwen35_verify_fold._installed = saved_fold_installed
    gdn_patches._BATCHED_VERIFY_SDPA_PATCHED = saved_bvs
    gdn_patches._STOCK_VERIFY_LEFT_PADDED = saved_bvs_stock


# ---------------------------------------------------------------------------
# wiring + copy fidelity
# ---------------------------------------------------------------------------


def test_rebind_engagement():
    lm = _lm()
    n = qwen35_attn.rebind_attn(lm)
    attns = _attns(lm)
    assert n == len(attns) == 1
    for a in attns:
        assert type(a) is qwen35_attn.OwnedQwen3_5Attention
    assert qwen35_attn.rebind_attn(lm) == 0


def test_copies_match_upstream_source():
    """Every in-tree ragged/plan/cached helper is a literal copy of the
    upstream original (and the kernel sources are string-identical), so
    upstream drift fails here instead of as diverging kernels."""
    import inspect

    def norm(fn):
        return [
            line.rstrip()
            for line in inspect.getsource(fn).splitlines()
            if line.strip()
        ]

    for name in (
        "_qwen3_5_device_arch_suffix",
        "_qwen3_5_sdpa_vector_blocks",
        "_qwen3_5_sdpa_vector_plan",
        "_qwen3_5_ragged_sdpa_one_pass_kernel",
        "_qwen3_5_ragged_sdpa_two_pass_1_kernel",
        "_qwen3_5_ragged_sdpa_two_pass_2_kernel",
        "_qwen3_5_cached_i32_array",
        "_qwen3_5_cached_sdpa_scalars",
        "apply_multimodal_rotary_pos_emb",
    ):
        assert norm(getattr(qwen35_attn, name)) == norm(getattr(_L, name)), (
            f"owned copy of {name} drifted from the upstream original"
        )
    for name in (
        "_QWEN3_5_RAGGED_SDPA_ONE_PASS_SOURCE",
        "_QWEN3_5_RAGGED_SDPA_TWO_PASS_1_SOURCE",
        "_QWEN3_5_RAGGED_SDPA_TWO_PASS_2_SOURCE",
    ):
        assert getattr(qwen35_attn, name) == getattr(_L, name), (
            f"owned kernel source {name} drifted from the upstream original"
        )


# ---------------------------------------------------------------------------
# identity vs the patched path
# ---------------------------------------------------------------------------


def _identity_prefill_and_decode(patched, owned, pads, steps=6):
    ids = mx.array([list(PROMPT)] * len(pads))
    toks_p, log_p = _greedy_chain(
        patched, ids, _batch_caches(patched, pads), steps
    )
    toks_o, log_o = _greedy_chain(owned, ids, _batch_caches(owned, pads), steps)
    assert mx.array_equal(toks_p, toks_o).item()
    assert _close(log_p, log_o, ATOL)


def _count_ragged(monkeypatch):
    """Count owned-arm ragged dispatch calls and kernel engagements.

    The owned resolver looks the dispatch up as a module global at call
    time, so wrapping it counts only the owned arm; the patched arm's
    module rebind bound the original function object at install.
    """
    counts = {"calls": 0, "engaged": 0}
    inner = qwen35_attn.ragged_decode_attention

    def counting(queries, keys, values, pads, scale):
        counts["calls"] += 1
        out = inner(queries, keys, values, pads, scale)
        if out is not None:
            counts["engaged"] += 1
        return out

    monkeypatch.setattr(qwen35_attn, "ragged_decode_attention", counting)
    return counts


@_NEEDS_GPU
@pytest.mark.parametrize(
    "dtype", (None, mx.bfloat16), ids=("f32-grouploop", "bf16-kernels")
)
def test_prefill_decode_identity(_restore_patches, monkeypatch, dtype):
    patched, owned = _arms(dtype)
    counts = _count_ragged(monkeypatch)
    _identity_prefill_and_decode(patched, owned, [0])
    _identity_prefill_and_decode(patched, owned, [0, 0])
    _identity_prefill_and_decode(patched, owned, [2, 0])
    # Engagement proof: bf16 at head_dim 64 runs the in-tree kernels;
    # f32 is declined by the dtype gate and takes the group loop.
    assert counts["calls"] > 0
    if dtype is mx.bfloat16:
        assert counts["engaged"] > 0
    else:
        assert counts["engaged"] == 0


@_NEEDS_GPU
def test_verify_sink_identity(_restore_patches):
    patched, owned = _arms()
    for pads in ([0], [2, 0]):
        ids = mx.array([list(PROMPT)] * len(pads))
        block = mx.array([[5, 9, 21]] * len(pads))

        outs = []
        for lm in (patched, owned):
            cache = _batch_caches(lm, pads)
            lm(ids, cache=cache)
            out = lm(
                block,
                cache=cache,
                capture_layer_ids=[],
                return_hidden=True,
                return_shared_kv=True,
            )
            mx.eval(out.logits)
            outs.append(out)
        out_p, out_o = outs

        assert _close(out_p.logits, out_o.logits, ATOL)
        hs_p = out_p.hidden_states or []
        hs_o = out_o.hidden_states or []
        assert len(hs_p) == len(hs_o)
        for a, b in zip(hs_p, hs_o):
            assert _close(a, b, ATOL)


@_NEEDS_GPU
def test_deep_batched_fold_identity(_restore_patches, monkeypatch):
    """B=2 continuation blocks on a deep cache cross the verify-fold
    depth gate (keys >= 4096) so the per-row causal fold route runs;
    the trailing greedy steps run the ragged decode kernels at depth."""
    patched, owned = _arms(mx.bfloat16)
    pads = [50, 0]
    depth = 4200
    mx.random.seed(23)
    ids = mx.random.randint(1, 120, (2, depth))
    block = mx.array([[5, 9, 21], [8, 4, 33]])
    mx.eval(ids)

    # Owned-arm engagement: the fold splits the continuation block into
    # per-row "causal" base-sdpa calls (the patched chain early-bound
    # its own original, so wrapping the base module counts owned only).
    causal_rows = {"n": 0}
    base_inner = _B.scaled_dot_product_attention

    def counting(queries, keys, values, cache, scale, mask, sinks=None):
        if (
            isinstance(mask, str)
            and mask == "causal"
            and queries.shape[0] == 1
            and queries.shape[2] > 1
        ):
            causal_rows["n"] += 1
        return base_inner(
            queries, keys, values, cache=cache, scale=scale, mask=mask,
            sinks=sinks,
        )

    monkeypatch.setattr(_B, "scaled_dot_product_attention", counting)

    outs = []
    for lm in (patched, owned):
        cache = _batch_caches(lm, pads)
        lm(ids, cache=cache)
        logits = lm(block, cache=cache).logits
        mx.eval(logits)
        step = mx.argmax(logits[:, -1, :], axis=-1)[:, None]
        toks, last = _greedy_chain(lm, step, cache, 4)
        outs.append((logits, toks, last))
    (log_p, toks_p, last_p), (log_o, toks_o, last_o) = outs

    assert causal_rows["n"] >= 2
    assert _close(log_p, log_o, ATOL)
    assert mx.array_equal(toks_p, toks_o).item()
    assert _close(last_p, last_o, ATOL)


@_NEEDS_GPU
def test_ragged_bucket_straddle_identity(_restore_patches, monkeypatch):
    """Rows whose effective lengths straddle a plan bucket take the
    unified-plan route in both arms. Effective lengths ~900 and ~4200
    straddle every arch's bucket table (1024 on s/d chips, 4096 on the
    GQA rule), asserted against the in-tree plan below."""
    patched, owned = _arms(mx.bfloat16)
    pads = [3300, 0]
    depth = 4200
    mx.random.seed(29)
    ids = mx.random.randint(1, 120, (2, depth))
    mx.eval(ids)

    plan_a = qwen35_attn._qwen3_5_sdpa_vector_plan(depth - pads[0], 4, 2)
    plan_b = qwen35_attn._qwen3_5_sdpa_vector_plan(depth, 4, 2)
    assert plan_a != plan_b, "premise: rows must straddle a plan bucket"

    counts = _count_ragged(monkeypatch)
    toks_p, log_p = _greedy_chain(
        patched, ids, _batch_caches(patched, pads), 6
    )
    toks_o, log_o = _greedy_chain(owned, ids, _batch_caches(owned, pads), 6)
    assert counts["engaged"] > 0
    assert mx.array_equal(toks_p, toks_o).item()
    assert _close(log_p, log_o, ATOL)
