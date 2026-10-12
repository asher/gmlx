"""QSAKVCache keeps each stream as a base and a decode tail. The owned
KVCache surface reads it as one stream: the prompt cache clone, the
capacity reserve, the memory profile, the speculative transaction and the
prefix snapshot each see every row, with rows still in the tail. The append
never goes through the one-buffer append of the base class."""

from __future__ import annotations

import threading

import mlx.core as mx
import numpy as np

from gmlx._vendor.mlx_vlm import apc_adapters
from gmlx._vendor.mlx_vlm.speculative import cache_state
from gmlx.cache import kv_cache as owned
from gmlx.cache import prefix_cache
from gmlx.models.qwen4_exp.model import QSAKVCache

H, D, DI = 2, 16, 8


def _rows(n, seed):
    mx.random.seed(seed)
    return (mx.random.normal((1, H, n, D)), mx.random.normal((1, H, n, D)),
            mx.random.normal((1, n, DI)))


def _filled(prefill=12, steps=3):
    """A cache after one prefill call and ``steps`` decode steps, and the
    three streams as NumPy arrays."""
    c = QSAKVCache(4)
    ref = [np.asarray(a) for a in _rows(prefill, 0)]
    assert c.append_qsa(*_rows(prefill, 0)) is False
    for i in range(steps):
        _push(c, ref, 1, 100 + i)
    assert c._tl == steps
    return c, ref


def _push(c, ref, n, seed):
    """Append ``n`` rows the way a decode or verify step does: they stay
    in the tail."""
    k, v, ik = _rows(n, seed)
    assert c.append_qsa(k, v, ik) is True
    for i, (a, axis) in enumerate(((k, 2), (v, 2), (ik, 1))):
        ref[i] = np.concatenate([ref[i], np.asarray(a)], axis=axis)


def _same(c, ref, n=None):
    n = ref[0].shape[2] if n is None else n
    k, v, ik = c.state
    assert c.offset == n
    assert np.array_equal(np.asarray(k), ref[0][:, :, :n])
    assert np.array_equal(np.asarray(v), ref[1][:, :, :n])
    assert np.array_equal(np.asarray(ik), ref[2][:, :n])


def test_the_append_never_calls_the_one_buffer_append(monkeypatch):
    """KVCache.update_and_fetch writes a row into the one K and V buffer.
    A step that is dispatched while the step before it is pending makes
    that write a copy of the whole buffer."""
    def one_buffer_append(self, keys, values):
        raise AssertionError("the one-buffer append ran")

    assert QSAKVCache.__mro__[1] is owned.KVCache
    monkeypatch.setattr(owned.KVCache, "update_and_fetch", one_buffer_append)
    c = QSAKVCache(4)
    ref = [np.asarray(a) for a in _rows(12, 0)]
    c.update_and_fetch_qsa(*_rows(12, 0))
    for seed in (1, 2, 3):
        k, v, ik = _rows(1, seed)
        got = c.update_and_fetch_qsa(k, v, ik)
        for i, (a, axis) in enumerate(((k, 2), (v, 2), (ik, 1))):
            ref[i] = np.concatenate([ref[i], np.asarray(a)], axis=axis)
            assert np.array_equal(np.asarray(got[i]), ref[i])
    base = c._kb
    _push(c, ref, 1, 7)
    assert c._kb is base and c._tl == 1
    _same(c, ref)


def test_reserve_grows_the_base_and_keeps_rows_of_the_tail():
    c, ref = _filled()
    targets = c.prefix_cache_reserve(600)
    assert targets and c._kb.shape[2] >= 600 and c._vb.shape[2] >= 600
    assert c._tl == 0
    _same(c, ref)
    _push(c, ref, 1, 8)
    assert c._tl == 1
    _same(c, ref)


def _clone(c):
    targets = []
    out = apc_adapters.clone_cache_entry(
        c, min_capacity_tokens=600, eval_targets=targets)
    mx.eval(targets)
    return out


def test_a_prompt_cache_clone_holds_every_stream_and_shares_no_buffer():
    c, ref = _filled()
    out = _clone(c)
    assert type(out) is QSAKVCache and out.ratio == 4
    assert out._kb.shape[2] >= 600
    kept = [a.copy() for a in ref]
    _same(out, kept)
    _push(c, ref, 1, 9)
    _same(out, kept)
    _push(out, kept, 2, 10)
    _same(c, ref)
    _same(out, kept)


def test_a_prompt_cache_clone_can_be_read_from_another_thread():
    """The disk writer of the prompt cache reads an entry from its own
    thread, which cannot evaluate an array of the thread that made it."""
    made, errors = {}, []

    def build():
        c, ref = _filled()
        made["out"], made["ref"] = _clone(c), ref

    def read():
        try:
            out = made["out"]
            mx.eval(out.keys, out.values, out.ik)
            made["state"] = [np.asarray(a) for a in out.state]
        except Exception as e:  # noqa: BLE001
            errors.append(repr(e))

    for fn in (build, read):
        t = threading.Thread(target=fn)
        t.start()
        t.join()
    assert errors == []
    for got, want in zip(made["state"], made["ref"]):
        assert np.array_equal(got, want)


def test_the_memory_profile_declines_and_the_byte_count_covers_the_tail():
    c, ref = _filled()
    held = (c._kb, c._vb, c._ib, c._kt, c._vt, c._it)
    assert all(a is not None for a in held)
    assert c.nbytes == sum(a.nbytes for a in held)
    assert owned.cache_nbytes(c) == c.nbytes
    # K and V alone are not the whole cache, so no per-token profile.
    assert c.memory_profile(c.offset) is None
    _same(c, ref)


def test_a_speculative_block_rolls_back_inside_the_tail():
    c, ref = _filled()
    n = c.offset
    tx = cache_state.start_speculative_cache([c], 4)
    _push(c, ref, 4, 11)
    assert c._tl == 7
    cache_state.rollback_speculative_cache([c], tx, 1, 4)
    assert c.offset == n + 2 and c._tl == 5
    _same(c, ref, n + 2)
    ref = [ref[0][:, :, :n + 2], ref[1][:, :, :n + 2], ref[2][:, :n + 2]]
    tx = cache_state.start_speculative_cache([c], 4)
    _push(c, list(ref), 4, 12)
    tx.abort()
    _same(c, ref)
    _push(c, ref, 1, 13)
    _same(c, ref)


def test_a_prefix_snapshot_round_trips_rows_of_the_tail():
    c, ref = _filled()
    snap = prefix_cache._snapshot_entry(c)
    assert snap[0] == prefix_cache._STATE_TAG
    mx.eval(snap[1])
    kept = [a.copy() for a in ref]
    _push(c, ref, 1, 14)
    fresh = QSAKVCache(2)
    prefix_cache._restore_entry(fresh, snap)
    assert fresh.ratio == 4
    _same(fresh, kept)
    _push(fresh, kept, 1, 15)
    _same(fresh, kept)
    _same(c, ref)
