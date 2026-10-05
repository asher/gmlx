"""APC block harvest on an affine (``--kv-bits``) cache.

The cache a batched prefill commits under the affine scheme is mlx-vlm's
``BatchQuantizedKVCache``, whose keys are a tuple. The lone path holds a
``QuantizedKVCache``. The installed harvest must store the same blocks as
mlx-vlm's stock harvest for both, so that affine keeps partial prefix reuse.
"""
from __future__ import annotations

import importlib

import pytest

pytest.importorskip("mlx_vlm")

import mlx.core as mx  # noqa: E402

from gmlx.serve.patches.apc import install_apc_lone_harvest  # noqa: E402

B, H, D, T, BLOCK = 2, 2, 64, 64, 16
LEFT_PAD = [0, 16]
N_LAYERS = 4


class _Layers:
    """A full-attention stack without make_cache, so _make_cache builds one
    batch cache per layer."""
    layers = [object()] * N_LAYERS


def _kv(batch, seed):
    mx.random.seed(seed)
    k = mx.random.normal((batch, H, T, D)).astype(mx.float16)
    v = mx.random.normal((batch, H, T, D)).astype(mx.float16)
    return k, v


def _affine_batch_cache():
    ar = importlib.import_module("mlx_vlm.generate.ar")
    caches = ar._make_cache(_Layers(), LEFT_PAD, kv_bits=8, kv_group_size=64,
                            kv_quant_scheme="uniform")
    for i, c in enumerate(caches):
        c.update_and_fetch(*_kv(B, i))
    mx.eval([c.state for c in caches])
    return caches


def _affine_lone_cache():
    cache_mod = importlib.import_module("mlx_vlm.models.cache")
    caches = [cache_mod.QuantizedKVCache(group_size=64, bits=8)
              for _ in range(N_LAYERS)]
    for i, c in enumerate(caches):
        c.update_and_fetch(*_kv(1, i))
    mx.eval([c.state for c in caches])
    return caches


def _manager():
    apc = importlib.import_module("mlx_vlm.apc")
    return apc.APCManager(num_blocks=64, block_size=BLOCK)


@pytest.fixture
def harvests(monkeypatch):
    """(stock, installed) harvest functions; monkeypatch restores the
    module after the test."""
    apc = importlib.import_module("mlx_vlm.apc")
    stock = apc.harvest_blocks_from_batch_cache
    assert stock.__module__ == "mlx_vlm.apc"
    monkeypatch.setattr(apc, "harvest_blocks_from_batch_cache", stock)
    monkeypatch.setattr(apc, "_kq_lone_harvest", False, raising=False)
    install_apc_lone_harvest()
    installed = apc.harvest_blocks_from_batch_cache
    assert installed is not stock
    return stock, installed


def _assert_same_blocks(got, want):
    assert len(got) == len(want) > 0
    for g, w in zip(got, want):
        assert g.block_hash == w.block_hash
        assert g.token_ids == w.token_ids
        assert len(g.keys) == len(w.keys) == N_LAYERS
        for gk, wk, gv, wv in zip(g.keys, w.keys, g.values, w.values):
            assert gk.shape == wk.shape == (1, H, BLOCK, D)
            assert gk.dtype == wk.dtype
            assert mx.array_equal(gk, wk).item()
            assert mx.array_equal(gv, wv).item()


@pytest.mark.parametrize("row", [0, 1])
def test_batched_affine_harvest_matches_stock(harvests, row):
    stock, installed = harvests
    caches = _affine_batch_cache()
    # mlx-vlm keeps the last layer fp16, so the stack mixes both kinds.
    assert sum(isinstance(c.keys, tuple) for c in caches) == N_LAYERS - 1
    ids = list(range(T - LEFT_PAD[row]))
    want = stock(_manager(), caches, ids, batch_idx=row)
    got = installed(_manager(), caches, ids, batch_idx=row)
    assert len(got) == len(ids) // BLOCK
    _assert_same_blocks(got, want)


@pytest.mark.parametrize("row", [0, 1])
def test_commit_after_batched_affine_prefill_stores_blocks(harvests, row):
    """ar.py commits through the module global after a batched prefill and
    logs 'APC harvest failed during batched prefill' on any exception."""
    stock, _ = harvests
    apc = importlib.import_module("mlx_vlm.apc")
    caches = _affine_batch_cache()
    ids = list(range(T - LEFT_PAD[row]))
    want = stock(_manager(), caches, ids, batch_idx=row)
    got = apc.commit_prefix_blocks(_manager(), caches, ids, batch_idx=row)
    _assert_same_blocks(got, want)


def test_lone_affine_harvest_matches_stock(harvests):
    stock, installed = harvests
    caches = _affine_lone_cache()
    assert all(isinstance(c.keys, tuple) for c in caches)
    ids = list(range(T))
    want = stock(_manager(), caches, ids)
    got = installed(_manager(), caches, ids)
    assert len(got) == T // BLOCK
    _assert_same_blocks(got, want)
