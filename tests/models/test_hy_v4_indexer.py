"""HY4 DSA lightning indexer: rope placement, scoring, and the shared chain.

Two things here differ from mlx-lm's deepseek_v32 indexer and are the reason
this file exists:

  1. Rope covers the last ``qk_rope_head_dim`` dims of each indexer head, not
     the first. The reverse convention loads clean and selects the wrong keys.
  2. Only the layers marked "full" own indexer weights. A "shared" layer
     reuses the most recent preceding full layer's selection, so the chain has
     to be threaded through the decoder stack.

Below ``index_topk`` cached keys the selection is the identity and the indexer
returns None (attention stays dense).
"""

from __future__ import annotations

import numpy as np
import pytest

import mlx.core as mx

from gmlx.models.hy_v4.model import (
    HyV4Indexer,
    HyV4Model,
    ModelArgs,
)

DIM = 16
IH = 2         # indexer heads
IHD = 8        # indexer head dim
PE = 4         # rope dims (the trailing half of IHD)
QLORA = 16


def _args(**over):
    base = dict(
        model_type="hy_v4", vocab_size=32, hidden_size=DIM, intermediate_size=32,
        moe_intermediate_size=16, num_hidden_layers=4, num_attention_heads=4,
        q_lora_rank=QLORA, kv_lora_rank=12, qk_nope_head_dim=8,
        qk_rope_head_dim=PE, v_head_dim=8, n_routed_experts=4,
        num_experts_per_tok=2, n_shared_experts=1, first_k_dense_replace=1,
        rms_norm_eps=1e-6, hc_mult=4, hc_eps=1e-6, hc_magnitude=2.0,
        index_n_heads=IH, index_head_dim=IHD, index_topk=4,
        index_is_full=[1, 1, 0, 1],
    )
    base.update(over)
    return ModelArgs(**base)


def _indexer(seed=0, **over):
    mx.random.seed(seed)
    idx = HyV4Indexer(_args(**over))
    mx.eval(idx.parameters())
    return idx


# --- rope placement ----------------------------------------------------------


def test_rope_lands_on_the_last_dims_not_the_first():
    idx = _indexer()
    src = np.random.default_rng(1).normal(
        scale=0.5, size=(1, 1, 3, IHD)).astype("float32")
    x = mx.array(src)
    nope = IHD - PE

    at_zero = np.array(idx._rope_tail(x, 0))
    shifted = np.array(idx._rope_tail(x, 5))

    # The leading nope dims never move, at any offset.
    assert np.allclose(at_zero[..., :nope], src[..., :nope], atol=1e-6)
    assert np.allclose(shifted[..., :nope], src[..., :nope], atol=1e-6)
    # Position 0 is the identity; later positions rotate the trailing dims.
    assert np.allclose(at_zero[:, :, 0, nope:], src[:, :, 0, nope:], atol=1e-6)
    assert not np.allclose(at_zero[:, :, 1:, nope:], src[:, :, 1:, nope:])
    # A cache offset shifts every position, the first one included.
    assert not np.allclose(shifted[..., nope:], at_zero[..., nope:])


def test_rope_head_dim_split_is_derived_not_assumed():
    idx = _indexer()
    assert idx.rope_head_dim == PE
    assert idx.nope_head_dim == IHD - PE


# --- scoring -----------------------------------------------------------------


def _ref_top_k(idx, x, qr, mask=None, offset=0):
    """relu(q . k) per head, weighted by the per-token head projection."""
    B, L, _ = x.shape
    q = idx.wq_b(qr).reshape(B, L, IH, IHD).swapaxes(1, 2)
    k = idx.k_norm(idx.wk(x)).reshape(B, 1, L, IHD)
    q = np.array(idx._rope_tail(q, offset), dtype=np.float64)
    k = np.array(idx._rope_tail(k, offset), dtype=np.float64)
    scores = np.maximum(q @ k.swapaxes(-1, -2), 0.0)
    w = np.array(idx.weights_proj(x), dtype=np.float64) * idx.weight_scale
    scores = (scores * w.swapaxes(-1, -2)[..., None]).sum(axis=1, keepdims=True)
    if mask is not None:
        scores = np.where(np.array(mask, dtype=bool), scores, -np.inf)
    return scores


def _assert_top_k_agrees(got, scores, k):
    """The selected score multiset must match the reference's.

    relu clamps many scores to exactly 0, so which of the tied keys a
    partition returns is arbitrary and the index sets legitimately differ.
    The scores they carry do not.
    """
    got = np.array(got)
    ordered = np.sort(scores, axis=-1)[..., -k:]
    for b in range(got.shape[0]):
        for i in range(got.shape[2]):
            picked = np.sort(scores[b, 0, i][got[b, 0, i]])
            assert np.allclose(picked, ordered[b, 0, i], atol=1e-9), i


def test_scoring_selects_the_reference_top_k():
    idx = _indexer(seed=3)
    L = 12
    rng = np.random.default_rng(4)
    x = mx.array(rng.normal(scale=0.6, size=(1, L, DIM)).astype("float32"))
    qr = mx.array(rng.normal(scale=0.6, size=(1, L, QLORA)).astype("float32"))

    got = idx(x, qr, mask=None)
    mx.eval(got)
    _assert_top_k_agrees(got, _ref_top_k(idx, x, qr), idx.index_topk)


def test_relu_sits_between_the_dot_and_the_head_weighting():
    # A negative head weight over a negative dot would flip the sign and
    # promote a key the reference clamps to zero. Force the situation with
    # large queries, then check the selection still tracks the clamped math.
    idx = _indexer(seed=5)
    L = 10
    rng = np.random.default_rng(6)
    x = mx.array(rng.normal(scale=0.6, size=(1, L, DIM)).astype("float32"))
    qr = mx.array(rng.normal(scale=2.0, size=(1, L, QLORA)).astype("float32"))
    scores = _ref_top_k(idx, x, qr)
    _assert_top_k_agrees(idx(x, qr, mask=None), scores, idx.index_topk)

    # Without the relu the same weights give a different ranking, so this is
    # a real discriminator rather than a tautology.
    q = idx.wq_b(qr).reshape(1, L, IH, IHD).swapaxes(1, 2)
    k = idx.k_norm(idx.wk(x)).reshape(1, 1, L, IHD)
    q = np.array(idx._rope_tail(q, 0), dtype=np.float64)
    k = np.array(idx._rope_tail(k, 0), dtype=np.float64)
    w = np.array(idx.weights_proj(x), dtype=np.float64) * idx.weight_scale
    unclamped = ((q @ k.swapaxes(-1, -2)) * w.swapaxes(-1, -2)[..., None]
                 ).sum(axis=1, keepdims=True)
    assert not np.allclose(np.argsort(-unclamped, axis=-1),
                           np.argsort(-scores, axis=-1))


def test_mask_excludes_future_keys():
    idx = _indexer(seed=7)
    L = 12
    rng = np.random.default_rng(8)
    x = mx.array(rng.normal(scale=0.6, size=(1, L, DIM)).astype("float32"))
    qr = mx.array(rng.normal(scale=0.6, size=(1, L, QLORA)).astype("float32"))
    causal = (np.arange(L)[None, :] <= np.arange(L)[:, None])[None, None]
    got = np.array(idx(x, qr, mask=mx.array(causal)))
    # Row i must never select a key > i once it has more than topk to pick.
    for i in range(idx.index_topk, L):
        assert got[0, 0, i].max() <= i


# --- the dense floor ---------------------------------------------------------


@pytest.mark.parametrize("L", [1, 3, 4])
def test_no_selection_below_top_k_keys(L):
    # At or below index_topk keys the selection would be the identity, so the
    # indexer returns None and attention stays dense.
    idx = _indexer(seed=9)
    rng = np.random.default_rng(10)
    x = mx.array(rng.normal(size=(1, L, DIM)).astype("float32"))
    qr = mx.array(rng.normal(size=(1, L, QLORA)).astype("float32"))
    assert idx(x, qr, mask=None) is None


def test_selection_appears_once_past_top_k_keys():
    idx = _indexer(seed=11)
    L = 5
    rng = np.random.default_rng(12)
    x = mx.array(rng.normal(size=(1, L, DIM)).astype("float32"))
    qr = mx.array(rng.normal(size=(1, L, QLORA)).astype("float32"))
    out = idx(x, qr, mask=None)
    assert out is not None
    assert out.shape == (1, 1, L, idx.index_topk)


# --- the shared chain --------------------------------------------------------


def test_only_full_layers_own_indexer_weights():
    model = HyV4Model(_args())
    owns = [layer.self_attn.indexer is not None for layer in model.layers]
    assert owns == [True, True, False, True]
    assert [layer.self_attn.is_full for layer in model.layers] == owns


def test_shared_layer_reuses_the_preceding_selection(monkeypatch):
    # Layer 2 is shared: it must consume the top_k layer 1 produced, and hand
    # it on unchanged. A broken chain silently makes shared layers dense.
    from gmlx.models.hy_v4.model import HyV4Attention

    model = HyV4Model(_args())
    mx.eval(model.parameters())
    order = {id(layer.self_attn): i for i, layer in enumerate(model.layers)}
    seen: dict[int, object] = {}
    real = HyV4Attention.__call__

    def wrap(self, x, mask=None, cache=None, top_k=None):
        seen[order[id(self)]] = top_k
        return real(self, x, mask=mask, cache=cache, top_k=top_k)

    monkeypatch.setattr(HyV4Attention, "__call__", wrap)

    L = 8
    out = model(mx.array(np.arange(L).reshape(1, L)), cache=model_cache(model))
    mx.eval(out)

    assert seen[0] is None                     # nothing precedes layer 0
    assert seen[1] is not None                 # layer 0's own selection
    assert seen[2] is not None                 # shared: got layer 1's pick
    assert seen[3] is not None
    assert seen[2].shape == (1, 1, L, 4)


def model_cache(model):
    """One entry per layer, in the shape Model.make_cache builds: a full
    layer carries the indexer slot, a shared layer carries the latent
    alone."""
    from mlx_lm.models.cache import CacheList, KVCache

    from gmlx.models.hy_v4.model import HyV4KVCache

    return [CacheList(HyV4KVCache(), KVCache())
            if layer.self_attn.indexer is not None else HyV4KVCache()
            for layer in model.layers]


def test_all_shared_after_the_first_full_layer_is_legal():
    # is_full = [1, 0, 0, 0]: one selection drives the whole stack.
    model = HyV4Model(_args(index_is_full=[1, 0, 0, 0]))
    mx.eval(model.parameters())
    owns = [layer.self_attn.indexer is not None for layer in model.layers]
    assert owns == [True, False, False, False]
    out = model(mx.array(np.arange(8).reshape(1, 8)), cache=model_cache(model))
    mx.eval(out)
    assert bool(mx.all(mx.isfinite(out)))
