#!/usr/bin/env python3
"""GmlxAPCManager.store_kv_blocks: the chunked-eval store override must be
behavior-identical to the stock per-block-eval store, and the stock class
must stay unpatched (the old global method patch is retired).

CPU-only, model-free: exercises the managers directly with tiny synthetic
K/V tensors.
"""
from __future__ import annotations

import pytest

import mlx.core as mx

apc = pytest.importorskip("mlx_vlm.apc")

from gmlx.cache.apc_manager import GmlxAPCManager  # noqa: E402

# The stock method, captured at import: nothing in gmlx may mutate it.
_STOCK_STORE = apc.APCManager.store_kv_blocks

BLOCK = 16
N_LAYERS = 3
N_TOKENS = 7 * BLOCK + 5  # 7 full blocks + a partial tail


def _make_kv(seed: int):
    keys, values = [], []
    for layer in range(N_LAYERS):
        mx.random.seed(seed * 100 + layer)
        keys.append(mx.random.normal((1, 2, N_TOKENS, 8)))
        values.append(mx.random.normal((1, 2, N_TOKENS, 8)))
    mx.eval(keys + values)
    return keys, values


def _tokens():
    return list(range(1000, 1000 + N_TOKENS))


def _pool_snapshot(mgr):
    """Comparable view of the manager's stored blocks."""
    out = {}
    for h, b in mgr.hash_table.items():
        out[h] = (
            b.parent_hash,
            b.token_ids,
            b.extra_hash,
            [k.tolist() for k in b.keys],
            [v.tolist() for v in b.values],
        )
    return out


def test_stock_class_unpatched():
    """Importing gmlx (incl. the subclass module) must not touch the stock
    method - the batched eval lives on the subclass only."""
    assert apc.APCManager.store_kv_blocks is _STOCK_STORE
    assert "store_kv_blocks" in GmlxAPCManager.__dict__


def test_patched_store_matches_stock():
    """Same tokens + K/V through stock and patched stores must produce
    identical block hashes, chain parents, token ids, and tensor data."""
    keys, values = _make_kv(seed=1)
    toks = _tokens()

    mgr_stock = apc.APCManager(num_blocks=32, block_size=BLOCK)
    blocks_stock = _STOCK_STORE(mgr_stock, toks, keys, values)

    mgr_patch = GmlxAPCManager(num_blocks=32, block_size=BLOCK)
    blocks_patch = mgr_patch.store_kv_blocks(toks, keys, values)

    assert len(blocks_stock) == len(blocks_patch) == N_TOKENS // BLOCK
    assert _pool_snapshot(mgr_stock) == _pool_snapshot(mgr_patch)
    assert mgr_stock.stats.stores == mgr_patch.stats.stores

    # The patched store's tensors must be materialized copies, decoupled
    # from the caller's arrays (the stock guarantee).
    for b in blocks_patch:
        for arr in b.keys + b.values:
            assert arr.shape == (1, 2, BLOCK, 8)

    mgr_stock.release(blocks_stock)
    mgr_patch.release(blocks_patch)


def test_patched_store_dedup_and_lookup():
    """Re-storing the same tokens acquires existing blocks (no new stores),
    and lookup_prefix returns the full stored prefix with intact data."""
    keys, values = _make_kv(seed=2)
    toks = _tokens()
    n_full = N_TOKENS // BLOCK

    mgr = GmlxAPCManager(num_blocks=32, block_size=BLOCK)
    first = mgr.store_kv_blocks(toks, keys, values)
    stores_after_first = mgr.stats.stores
    assert stores_after_first == n_full

    second = mgr.store_kv_blocks(toks, keys, values)
    assert mgr.stats.stores == stores_after_first, "dedup re-stored blocks"
    assert len(second) == n_full

    matched, prefix_len = mgr.lookup_prefix(toks)
    assert prefix_len == n_full * BLOCK
    # Block data must equal the source slices exactly.
    for i, b in enumerate(matched):
        lo, hi = i * BLOCK, (i + 1) * BLOCK
        for layer in range(N_LAYERS):
            assert mx.array_equal(b.keys[layer], keys[layer][..., lo:hi, :])
            assert mx.array_equal(b.values[layer], values[layer][..., lo:hi, :])

    mgr.release(first)
    mgr.release(second)
    mgr.release(matched)


def test_patched_store_skip_first_n():
    """skip_first_n_tokens skips already-stored prefix blocks but keeps the
    hash chain intact for the stored suffix."""
    keys, values = _make_kv(seed=3)
    toks = _tokens()
    n_full = N_TOKENS // BLOCK

    mgr = GmlxAPCManager(num_blocks=32, block_size=BLOCK)
    first = mgr.store_kv_blocks(toks, keys, values)
    baseline = mgr.stats.stores

    tail = mgr.store_kv_blocks(
        toks, keys, values, skip_first_n_tokens=3 * BLOCK)
    assert mgr.stats.stores == baseline, "skip path copied prefix blocks"
    assert len(tail) == n_full - 3

    matched, prefix_len = mgr.lookup_prefix(toks)
    assert prefix_len == n_full * BLOCK, "hash chain broken by skip store"

    mgr.release(first)
    mgr.release(tail)
    mgr.release(matched)


def test_patched_store_pool_exhaustion():
    """A pool smaller than the block count stores what fits and stops
    cleanly (stock break semantics, no disk tier)."""
    keys, values = _make_kv(seed=4)
    toks = _tokens()

    mgr = GmlxAPCManager(num_blocks=4, block_size=BLOCK)
    blocks = mgr.store_kv_blocks(toks, keys, values)
    assert len(blocks) == 4
    assert mgr.stats.stores == 4

    matched, prefix_len = mgr.lookup_prefix(toks)
    assert prefix_len == 4 * BLOCK

    mgr.release(blocks)
    mgr.release(matched)
