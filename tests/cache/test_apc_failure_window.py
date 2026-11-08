"""The APC stores of a command buffer failure's window are taken back.

A failure reports at a wait that can come a tick after the failed work, so
the tick guard evicts what the reporting tick and the tick before it
stored, and every store of the prompt batch it drops. What a store reused
was written before the window and stays.
"""

import threading
from pathlib import Path

import mlx.core as mx

from gmlx._vendor.mlx_vlm import apc as _apc
from gmlx.cache.apc_manager import GmlxAPCManager

BS = 4


def _kv(n):
    return [mx.ones((1, 2, n, 4))], [mx.ones((1, 2, n, 4))]


def _store(m, ids, *, release=True):
    blocks = m.store_kv_blocks(list(ids), *_kv(len(ids)))
    hashes = [b.block_hash for b in blocks]
    if release:
        m.release(blocks)
    return blocks, hashes


def _held(m, hashes):
    return [m.hash_table.get(h) is not None for h in hashes]


def test_the_window_and_the_dropped_batch_go_and_older_stores_stay():
    m = GmlxAPCManager(num_blocks=64, block_size=BS)
    batch = object()
    m.kq_mark_tick(3, batch)
    with m.kq_scope(batch):
        _, ckpt = _store(m, range(100, 108))        # the batch's checkpoint
    m.kq_mark_tick(4, batch)
    _, old = _store(m, range(0, 8))                  # T-2
    m.kq_mark_tick(5, batch)
    _, prev = _store(m, range(200, 208))             # T-1
    m.kq_mark_tick(6, batch)
    _, now = _store(m, range(0, 16))                 # T, reusing T-2's blocks
    assert now[:2] == old

    assert m.kq_evict(5, owner=batch) == 2 + 2 + 2
    assert _held(m, old) == [True, True]
    assert _held(m, prev + now[2:] + ckpt) == [False] * 6
    # An evict after it finds nothing left to take back.
    assert m.kq_evict(5, owner=batch) == 0


def test_records_older_than_the_window_are_pruned_unless_the_batch_lives():
    m = GmlxAPCManager(num_blocks=64, block_size=BS)
    batch = object()
    m.kq_mark_tick(1, batch)
    with m.kq_scope(batch):
        _, ckpt = _store(m, range(100, 104))
    _, early = _store(m, range(0, 4))
    m.kq_mark_tick(5, batch)
    m.kq_mark_tick(6, None)                          # the batch is done
    assert m.kq_evict(0) == 0
    assert _held(m, early + ckpt) == [True, True]


def test_a_held_block_is_unhashed_now_and_freed_by_its_last_release():
    m = GmlxAPCManager(num_blocks=64, block_size=BS)
    m.kq_mark_tick(1)
    held, hs = _store(m, range(0, 8), release=False)
    m.kq_mark_tick(2)
    idle, idle_hs = _store(m, range(50, 54))
    assert m.kq_evict(1) == 3
    assert _held(m, hs + idle_hs) == [False] * 3
    assert idle[0].block_hash is None                # freed at once
    assert [b.block_hash for b in held] == hs        # a row still reads it
    m.release(held)
    assert [b.block_hash for b in held] == [None, None]
    assert all(b.ref_cnt == 0 for b in held)
    # The freed blocks go back to the pool for the next store.
    _, again = _store(m, range(0, 8))
    assert _held(m, again) == [True, True]


def _disk_manager(tmp_path):
    disk = _apc.DiskBlockStore(Path(tmp_path))
    return GmlxAPCManager(num_blocks=64, block_size=BS, disk=disk), disk


def _shards(tmp_path):
    return sorted(Path(tmp_path).rglob("*.safetensors"))


def test_a_written_shard_of_the_window_is_deleted(tmp_path):
    m, disk = _disk_manager(tmp_path)
    try:
        m.kq_mark_tick(1)
        _store(m, range(0, 8))
        disk.flush()
        assert _shards(tmp_path) and disk.num_blocks_indexed == 2
        m.kq_evict(1)
        assert _shards(tmp_path) == []
        assert disk.num_blocks_indexed == 0
        assert disk.disk_bytes == 0
    finally:
        disk.close()


def test_a_queued_disk_write_is_skipped_and_one_in_progress_is_deleted(
        tmp_path):
    m, disk = _disk_manager(tmp_path)
    started, gate = threading.Event(), threading.Event()
    calls = []
    write = disk._write_layer_major_snapshot

    def held_write(*a, **kw):
        calls.append(a)
        started.set()
        gate.wait()
        return write(*a, **kw)

    disk._write_layer_major_snapshot = held_write
    try:
        m.kq_mark_tick(1)
        _store(m, range(0, 8))                       # the writer holds this
        assert started.wait(30)
        m.kq_mark_tick(2)
        _store(m, range(300, 308))                   # queued behind it
        m.kq_evict(1)
        gate.set()
        disk.flush()
        assert len(calls) == 1                       # the queued one never ran
        assert _shards(tmp_path) == []
        assert disk.num_blocks_indexed == 0
        assert disk.disk_bytes == 0

        # A drop mark ends with its write: the same blocks store again.
        m.kq_mark_tick(3)
        _store(m, range(300, 308))
        disk.flush()
        assert len(_shards(tmp_path)) == 1
        assert disk.num_blocks_indexed == 2
    finally:
        gate.set()
        disk.close()
