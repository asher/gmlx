"""kvarn arms in prefix_cache snapshots and cache_snapshot clones: tag
round trip, no-alias pins, clone-twin equality, ckpt-tier decline."""

from __future__ import annotations

import numpy as np
import pytest

import mlx.core as mx

from gmlx.cache.kvarn_cache import KVarNKVCache
from gmlx.cache.prefix_cache import (
    _KVARN_TAG,
    _eval_snapshot,
    _restore_entry,
    _snapshot_entry,
)

_NEEDS_GPU = pytest.mark.skipif(
    mx.default_device() != mx.gpu,
    reason="kvarn kernels are Metal-only; needs the GPU device",
)

H = 2
D = 128


def _tokens(n, seed=0):
    rng = np.random.default_rng(seed)
    k = mx.array(rng.standard_normal((1, H, n, D)).astype(np.float16))
    v = mx.array(rng.standard_normal((1, H, n, D)).astype(np.float16))
    return k, v


def _filled(n, seed=0):
    c = KVarNKVCache(tail_tokens=256)
    c.update_and_fetch(*_tokens(n, seed))
    return c


def _assert_same_content(a, b):
    assert (a.offset, a.n_sealed, a.tail_len) == (b.offset, b.n_sealed, b.tail_len)
    for x, y in zip(a.materialize(), b.materialize(), strict=True):
        assert np.array_equal(np.array(x), np.array(y))


@_NEEDS_GPU
def test_prefix_snapshot_round_trip():
    live = _filled(300)
    snap = _snapshot_entry(live)
    assert snap[0] == _KVARN_TAG and len(snap) == 3
    _eval_snapshot([snap])
    ref_mat = [np.array(m) for m in live.materialize()]

    # Live cache keeps decoding after the snapshot was taken.
    live.update_and_fetch(*_tokens(40, seed=7))

    fresh = KVarNKVCache(tail_tokens=256)
    _restore_entry(fresh, snap)
    assert fresh.offset == 300
    for got, want in zip(fresh.materialize(), ref_mat, strict=True):
        assert np.array_equal(np.array(got), want)


@_NEEDS_GPU
def test_prefix_restore_does_not_alias_stored_entry():
    live = _filled(300)
    snap = _snapshot_entry(live)
    _eval_snapshot([snap])
    a = KVarNKVCache(tail_tokens=256)
    _restore_entry(a, snap)
    a.update_and_fetch(*_tokens(64, seed=8))  # mutates a's buffers in place
    b = KVarNKVCache(tail_tokens=256)
    _restore_entry(b, snap)
    assert b.offset == 300
    _assert_same_content(b, _filled(300))


@_NEEDS_GPU
def test_clone_lm_twin_arm():
    from gmlx.cache.snapshot import _clone_lm_twin

    live = _filled(300)
    targets = []
    twin = _clone_lm_twin(live, targets)
    assert type(twin) is KVarNKVCache
    assert targets
    mx.eval(targets)
    live.update_and_fetch(*_tokens(50, seed=9))
    _assert_same_content(twin, _filled(300))


@_NEEDS_GPU
def test_ckpt_layout_tags_filled_kvarn():
    from gmlx.cache.snapshot import ckpt_layout

    from mlx_lm.models.cache import RotatingKVCache

    assert ckpt_layout([_filled(140), RotatingKVCache(max_size=64)]) == \
        ["kvarn:6:6:256", "rot:64:0"]
