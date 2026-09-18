"""deepseek_v41 pooled-row mask as a rule: equal to the cache mask form."""

import mlx.core as mx
import numpy as np

import gmlx.models.deepseek_v4.model as dsv4
import gmlx.models.deepseek_v41.model as dsv41
from gmlx.models.deepseek_v4.cache import PoolingCache


def _cache(plen, ratio):
    c = PoolingCache(ratio)
    c._plen = plen
    return c


def test_rule_matches_cache_and_cacheless_masks():
    plen, L, q0, ratio = 300, 40, 1000, 4
    rule = dsv41._pool_mask(_cache(plen, ratio), plen, L, q0, ratio)
    assert isinstance(rule, dsv41._PoolMask)
    want = np.array(_cache(plen, ratio).make_mask(L, q0))
    assert np.array_equal(np.array(rule.full()), want)
    assert np.array_equal(
        np.array(dsv4._cacheless_pool_mask(plen, L, q0, ratio)), want)
    rows = np.concatenate([np.array(rule.rows(qs, min(L, qs + 16)))[0]
                           for qs in range(0, L, 16)])
    assert np.array_equal(rows, want)


def test_sparse_equals_the_gathered_mask():
    plen, L, q0, ratio = 300, 40, 1000, 1
    rule = dsv41._PoolMask(plen, L, q0, ratio)
    topk = mx.random.randint(0, plen, (2, L, 8))
    want = mx.take_along_axis(mx.broadcast_to(rule.full()[None], (2, L, plen)),
                              topk, axis=2)
    assert np.array_equal(np.array(rule.sparse(topk)), np.array(want))


def test_no_mask_when_every_row_is_visible():
    assert dsv41._pool_mask(_cache(0, 1), 0, 8, 100, 1) is None
    assert dsv41._pool_mask(_cache(50, 1), 50, 1, 100, 1) is None
    assert dsv41._pool_mask(None, 0, 8, 100, 1) is None


def test_array_wrapper_over_a_batched_mask():
    m = mx.random.randint(0, 2, (2, 6, 20)).astype(mx.bool_)
    w = dsv41._ArrayPoolMask(m)
    assert np.array_equal(np.array(w.rows(2, 5)), np.array(m[:, 2:5]))
    assert w.full() is m
    topk = mx.random.randint(0, 20, (2, 6, 4))
    assert np.array_equal(np.array(w.sparse(topk)),
                          np.array(mx.take_along_axis(m, topk, axis=2)))
