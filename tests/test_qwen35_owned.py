"""Owned qwen3.5 model-level forward: identity vs stock + wiring.

The owned forward must be a drop-in for the stock mlx-vlm
``Qwen3_5Model.__call__`` on every live route: plain B=1, single-row
batch cache (where the stock class takes its extract/recurse/merge
shortcut and the owned class does not), batched left-padded prefill
(per-row on both arms), and the verify-shaped sink path. Identity is
greedy-token equality plus a logits bound; routes that legitimately
differ (shortcut removed) get the contract-pin tolerance, routes that
run the same ops get a tight one.
"""

import os
from types import SimpleNamespace

import mlx.core as mx
import pytest

pytest.importorskip("mlx_vlm.models.qwen3_5.language")

from mlx_vlm.models.cache import ArraysCache, BatchKVCache
from mlx_vlm.models.qwen3_5.config import TextConfig as Q35TextConfig
from mlx_vlm.models.qwen3_5.language import LanguageModel as Q35LanguageModel

from gmlx import qwen35_owned
from gmlx.loader import _mtp_target_classes

ATOL = 2e-3  # differing-route bound (shortcut removed / kernel path)
TIGHT_ATOL = 1e-5  # same-ops bound

# The qwen3_5 GDN forward dispatches Metal-only kernels.
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
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_conv_kernel_dim=4,
        num_hidden_layers=4,
        num_attention_heads=4,
        rms_norm_eps=1e-6,
        vocab_size=128,
        num_key_value_heads=2,
        max_position_embeddings=2048,
        tie_word_embeddings=True,
        head_dim=32,
        rope_parameters={
            "type": "default",
            "mrope_section": [2, 1, 1],
            "rope_theta": 100000,
            "partial_rotary_factor": 0.25,
        },
        full_attention_interval=4,
    )


def _top():
    # get_rope_index dereferences vision config + multimodal token ids on
    # every fresh text forward; text-only construction needs the stubs.
    return SimpleNamespace(
        vision_config=SimpleNamespace(spatial_merge_size=2),
        image_token_id=124,
        video_token_id=125,
        vision_start_token_id=126,
    )


def _pair():
    """Stock and owned LanguageModel with identical weights."""
    mx.random.seed(11)
    stock = Q35LanguageModel(_cfg(), _top())
    mx.eval(stock.parameters())
    mx.random.seed(11)
    owned = qwen35_owned.OwnedQwen3_5LanguageModel(_cfg(), _top())
    mx.eval(owned.parameters())
    same = mx.array_equal(
        stock.model.embed_tokens.weight, owned.model.embed_tokens.weight
    ).item()
    assert same, "seeded construction diverged; weight sync broken"
    return stock, owned


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
    return mx.stack(toks, axis=1), logits


# ---------------------------------------------------------------------------
# wiring
# ---------------------------------------------------------------------------


def test_loader_selects_owned_by_default(monkeypatch):
    monkeypatch.delenv("GMLX_QWEN_OWNED", raising=False)
    cls, _build = _mtp_target_classes("qwen3_5")
    assert cls is qwen35_owned.OwnedQwen3_5LanguageModel


def test_loader_env_reverts_to_stock(monkeypatch):
    monkeypatch.setenv("GMLX_QWEN_OWNED", "0")
    cls, _build = _mtp_target_classes("qwen3_5")
    assert cls is Q35LanguageModel


def test_owned_moe_class_shape():
    cls = qwen35_owned.language_model_class("qwen3_5_moe")
    from mlx_vlm.models.qwen3_5_moe.language import (
        LanguageModel as MoeLanguageModel,
    )

    assert issubclass(cls, MoeLanguageModel)
    with pytest.raises(ValueError):
        qwen35_owned.language_model_class("gemma4_text")


def test_owned_inherits_mtp_hooks():
    for hook in (
        "speculative_verify_hidden",
        "speculative_verify_logits",
        "speculative_logits_from_hidden",
        "speculative_argmax_from_hidden",
        "rollback_speculative_cache",
        "chunked_prefill_policy",
    ):
        assert hasattr(qwen35_owned.OwnedQwen3_5LanguageModel, hook)


# ---------------------------------------------------------------------------
# identity vs stock
# ---------------------------------------------------------------------------


@_NEEDS_GPU
def test_b1_plain_cache_identity():
    stock, owned = _pair()
    ids = mx.array([list(PROMPT)])
    before = qwen35_owned.owned_call_count()
    toks_s, logits_s = _greedy_chain(stock, ids, stock.make_cache(), 6)
    mid = qwen35_owned.owned_call_count()
    toks_o, logits_o = _greedy_chain(owned, ids, owned.make_cache(), 6)
    after = qwen35_owned.owned_call_count()

    assert mid == before, "stock arm engaged the owned forward"
    assert after > mid, "owned arm did not engage the owned forward"
    assert mx.array_equal(toks_s, toks_o).item()
    assert _close(logits_s, logits_o, TIGHT_ATOL)


@_NEEDS_GPU
def test_b1_single_row_batch_cache_identity():
    # Stock takes the extract/recurse/merge shortcut here; owned takes the
    # direct batched path. Same tokens, logits within the route bound.
    stock, owned = _pair()
    ids = mx.array([list(PROMPT)])
    toks_s, logits_s = _greedy_chain(stock, ids, _batch_caches(stock, [0]), 5)
    toks_o, logits_o = _greedy_chain(owned, ids, _batch_caches(owned, [0]), 5)
    assert mx.array_equal(toks_s, toks_o).item()
    assert _close(logits_s, logits_o, ATOL)


@_NEEDS_GPU
def test_batched_padded_prefill_identity():
    stock, owned = _pair()
    pads = [2, 0, 1]
    rows = [
        [0, 0, 3, 17, 42, 99, 7, 63],
        [3, 17, 42, 99, 7, 63, 5, 28],
        [0, 3, 17, 42, 99, 7, 63, 5],
    ]
    ids = mx.array(rows)
    toks_s, logits_s = _greedy_chain(stock, ids, _batch_caches(stock, pads), 4)
    toks_o, logits_o = _greedy_chain(owned, ids, _batch_caches(owned, pads), 4)
    assert mx.array_equal(toks_s, toks_o).item()
    assert _close(logits_s, logits_o, ATOL)


@_NEEDS_GPU
def test_fully_padded_row_is_structural():
    # A row whose padding consumes the entire chunk recurses at S=0. The
    # stock class needs gmlx's guard patch for this; the owned forward
    # handles it structurally. Install the guard so the stock arm can be
    # compared at all.
    from gmlx.gdn_patches import _patch_qwen35_empty_sequence_guard

    _patch_qwen35_empty_sequence_guard()
    stock, owned = _pair()
    S = 4
    pads = [S, 0]
    rows = [[0] * S, [3, 17, 42, 99]]
    ids = mx.array(rows)
    out_s = stock(ids, cache=_batch_caches(stock, pads)).logits
    out_o = owned(ids, cache=_batch_caches(owned, pads)).logits
    assert out_s.shape == out_o.shape
    assert _close(out_s[1:], out_o[1:], ATOL)


@_NEEDS_GPU
def test_s0_direct_call_needs_no_guard():
    _stock, owned = _pair()
    empty = mx.zeros((1, 0, 64), dtype=owned.model.embed_tokens.weight.dtype)
    out = owned.model(mx.zeros((1, 0), dtype=mx.int32), inputs_embeds=empty)
    assert out.shape == (1, 0, 64)


@_NEEDS_GPU
def test_verify_shaped_sink_path_identity():
    stock, owned = _pair()
    pads = [0, 0]
    ids = mx.array([list(PROMPT), list(PROMPT[::-1])])
    cache_s = _batch_caches(stock, pads)
    cache_o = _batch_caches(owned, pads)
    stock(ids, cache=cache_s)
    owned(ids, cache=cache_o)

    block = mx.array([[5, 28, 3], [17, 42, 99]])
    out_s = stock(
        block,
        cache=cache_s,
        capture_layer_ids=[],
        return_hidden=True,
        return_shared_kv=True,
    )
    out_o = owned(
        block,
        cache=cache_o,
        capture_layer_ids=[],
        return_hidden=True,
        return_shared_kv=True,
    )
    assert _close(out_s.hidden_states[-1], out_o.hidden_states[-1], ATOL)
    assert _close(out_s.logits, out_o.logits, ATOL)
    assert (out_o.gdn_states is None) == (out_s.gdn_states is None)


@_NEEDS_GPU
def test_hidden_capture_layers_match():
    stock, owned = _pair()
    ids = mx.array([list(PROMPT)])
    out_s = stock(ids, cache=stock.make_cache(), capture_layer_ids=[1, 3])
    out_o = owned(ids, cache=owned.make_cache(), capture_layer_ids=[1, 3])
    assert len(out_s.hidden_states) == len(out_o.hidden_states)
    for a, b in zip(out_s.hidden_states, out_o.hidden_states):
        assert _close(a, b, ATOL)
