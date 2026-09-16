"""Prefill query banding on DeepSeek-V4.1: the window layers, the sparse
layers and the indexer score each query block against the keys its window
reaches (plus its own top-k rows), and match the full-length pass."""
from __future__ import annotations

import importlib.util
import pathlib

import mlx.core as mx
from mlx_lm.models.base import create_causal_mask

from gmlx.models.deepseek_v4 import model as v4

_SIBLING = pathlib.Path(__file__).with_name("test_deepseek_v41_model.py")


def _model_module():
    spec = importlib.util.spec_from_file_location("_v41_model_tests_band", _SIBLING)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_query_bands_cover_the_window():
    bands = list(v4._query_bands(23, 30, 8, 5))
    assert [b[:2] for b in bands] == [(0, 5), (5, 10), (10, 15), (15, 20), (20, 23)]
    for qs, qe, ks, ke in bands:
        assert ks == max(0, qs + 7 - 7) and ke == qe + 7
    assert list(v4._query_bands(4, 4, 8, 16)) == [(0, 4, 0, 4)]
    assert list(v4._query_bands(3, 3, 8, 2)) == [(0, 2, 0, 2), (2, 3, 0, 3)]


def _window_case(B=2, H=4, L=23, koff=7, D=16, window=8, seed=0):
    mx.random.seed(seed)
    q = mx.random.normal((B, H, L, D))
    kv = mx.random.normal((B, 1, L + koff, D))
    mask = create_causal_mask(L, koff, window_size=window)
    sinks = mx.random.normal((H,))
    return q, kv, mask, sinks


def test_banded_window_attention_matches_dense():
    q, kv, mask, sinks = _window_case()
    full = mx.fast.scaled_dot_product_attention(
        q, kv, kv, scale=0.25, mask=mask, sinks=sinks
    )
    for block in (1, 5, 8, 23):
        got = v4._banded_window_attention(q, kv, mask, 0.25, sinks, 8, block)
        assert got.shape == full.shape
        assert mx.allclose(got, full, atol=1e-5, rtol=1e-5), block


def test_banded_sparse_attention_matches_full():
    q, kv, mask, sinks = _window_case()
    B, H, L, D = q.shape
    P, N = 12, 4
    pooled = mx.random.normal((B, P, D))
    topk = mx.random.randint(0, P, (B, L, N))
    pmask = v4._cacheless_pool_mask(P, L, 7, 2)
    smask = mx.take_along_axis(pmask[None], topk, axis=2)[:, None]
    for pm in (None, smask):
        full = v4._sparse_pooled_attention(
            q, kv, pooled, topk, mask, pm, 0.25, sinks
        )
        for block in (1, 5, 8):
            got = v4._sparse_pooled_attention_banded(
                q, kv, pooled, topk, mask, pm, 0.25, sinks, 8, block
            )
            assert got.shape == full.shape
            assert mx.allclose(got, full, atol=1e-5, rtol=1e-5), (block, pm is None)


def test_model_prefill_banded_matches_full(monkeypatch):
    """A 14-token prompt in blocks of 4 (window 4, so every block reaches
    3 rows of the block before it) gives the full pass's logits, on the
    plain indexer and on the two-level candidate indexer; the decode step
    after it agrees too."""
    tm = _model_module()
    prompt = mx.array([[5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31]])
    for over in (
        {},
        dict(candidate_source_layer=4, candidate_topk_blocks=1,
             candidate_block_size=2),
    ):
        model = tm._randomized(tm.Model(tm._args(**over)))
        outs = {}
        for block in (0, 4):
            monkeypatch.setattr(v4, "_PREFILL_BLOCK", block)
            cache = model.make_cache()
            out = model(prompt, cache=cache)
            step = model(mx.array([[33]]), cache=cache)
            mx.eval(out, step)
            outs[block] = (out, step)
        for a, b in zip(outs[0], outs[4]):
            assert mx.allclose(a, b, atol=1e-4, rtol=1e-4), over
