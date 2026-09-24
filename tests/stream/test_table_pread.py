"""File-backed gather for a table past the device buffer ceiling."""
from __future__ import annotations

import gc
import threading

import mlx.core as mx
import numpy as np
import pytest

from gmlx.stream.table_pread import (
    install_deferred_tables,
    max_buffer_bytes,
    oversize_tables,
)
import gmlx.stream.table_stream as ts

gguf = pytest.importorskip("gguf")

ROWS, ROW_BYTES = 1000, 33


@pytest.fixture
def table_gguf(tmp_path):
    """A deepseek41 GGUF whose one engram table is raw I8 rows."""
    rng = np.random.default_rng(7)
    raw = rng.integers(0, 256, size=(ROWS, ROW_BYTES), dtype=np.uint8)
    # E4M3 / E8M0 NaN codes are a decode error; keep them out of the fixture.
    raw[raw & 127 == 127] = 1
    raw[:, 32][raw[:, 32] == 255] = 127
    path = str(tmp_path / "t.gguf")
    w = gguf.GGUFWriter(path, "deepseek41")
    w.add_tensor("blk.1.engram_embd.weight", raw.view(np.int8),
                 raw_dtype=gguf.GGMLQuantizationType.I8)
    w.add_tensor("token_embd.weight", np.zeros((4, 8), np.float32),
                 raw_dtype=gguf.GGMLQuantizationType.F32)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return path, raw


def _sources(path, monkeypatch, cap="1000"):
    monkeypatch.setenv("GMLX_TABLE_MAX_BUFFER", cap)
    return oversize_tables([path], "deepseek41")


def test_max_buffer_env_override(monkeypatch):
    monkeypatch.setenv("GMLX_TABLE_MAX_BUFFER", "123")
    assert max_buffer_bytes() == 123


def test_only_tables_past_the_ceiling(table_gguf, monkeypatch):
    path, _raw = table_gguf
    src = _sources(path, monkeypatch)
    assert list(src) == ["blk.1.engram_embd.weight"]
    s = src["blk.1.engram_embd.weight"]
    assert (s.rows, s.row_bytes, s.nbytes) == (ROWS, ROW_BYTES, ROWS * ROW_BYTES)

    # Under the ceiling nothing defers; an arch with no table registry never
    # scans at all.
    assert _sources(path, monkeypatch, cap="100000") == {}
    assert oversize_tables([path], "llama") == {}


def test_read_rows_matches_the_file(table_gguf, monkeypatch):
    path, raw = table_gguf
    s = _sources(path, monkeypatch)["blk.1.engram_embd.weight"]
    try:
        ids = np.array([0, ROWS - 1, 17, 17, 4])
        got = s.read(ids)
        assert got.dtype == np.uint8 and got.shape == (5, ROW_BYTES)
        assert np.array_equal(got, raw[ids])
        # A whole-table read exercises every page boundary.
        assert np.array_equal(s.read(np.arange(ROWS)), raw)
        with pytest.raises(IndexError):
            s.read(np.array([ROWS]))
    finally:
        s.close()


def test_read_is_worker_count_independent(table_gguf, monkeypatch):
    path, raw = table_gguf
    monkeypatch.setenv("GMLX_TABLE_PREAD_WORKERS", "1")
    s = _sources(path, monkeypatch)["blk.1.engram_embd.weight"]
    try:
        ids = np.arange(0, ROWS, 7)
        assert np.array_equal(s.read(ids), raw[ids])
    finally:
        s.close()


def _engram_table():
    from gmlx.models.deepseek_v41.model import EngramTable

    return EngramTable(ROWS, 32, f"e4m3_e8m0_32_row{ROW_BYTES}")


def test_install_swaps_the_table_onto_the_file(table_gguf, monkeypatch):
    path, raw = table_gguf
    src = _sources(path, monkeypatch)
    table = _engram_table()
    want = table.decode_gathered(mx.array(raw[[3, 900, 3]]))

    model = _FakeModel(table)
    assert install_deferred_tables(model, src) == ["blk.1.engram_embd.weight"]
    try:
        assert getattr(table, "weight", None) is None, "placeholder still held"
        # No array, so no budget seam counts it; the file size lives on
        # the source.
        assert ts.table_bytes(model) == 0
        assert ts.streamed_table_bytes(model) == 0
        assert table._kq_table_source.nbytes == ROWS * ROW_BYTES
        assert table._kq_table_streamed
        got = table(mx.array([[3, 900, 3]]))
        assert got.shape == (1, 3, 32)
        assert mx.array_equal(got, want.reshape(1, 3, 32))
        repr(table)  # the placeholder is gone; _extra_repr must not read it
        assert install_deferred_tables(model, src) == []  # idempotent
    finally:
        table._kq_table_source.close()


def test_prefetch_joins_the_same_ids_and_rereads_others(table_gguf, monkeypatch):
    path, raw = table_gguf
    src = _sources(path, monkeypatch)
    table = _engram_table()
    model = _FakeModel(table)
    install_deferred_tables(model, src)
    s = table._kq_table_source
    try:
        ids = mx.array([[3, 900, 3]])
        table.prefetch(ids)
        assert table._kq_ahead is not None
        got = table(ids)
        assert table._kq_ahead is None
        assert s.gathers == 1, "the call must join the read, not repeat it"
        want = table.decode_gathered(mx.array(raw[[3, 900, 3]])).reshape(1, 3, 32)
        assert mx.array_equal(got, want)
        # Other ids than the prefetched ones: the call reads for itself.
        table.prefetch(ids)
        other = mx.array([[5, 6]])
        got = table(other)
        assert s.gathers == 3
        assert mx.array_equal(
            got, table.decode_gathered(mx.array(raw[[5, 6]])).reshape(1, 2, 32))
        assert table(other).shape == (1, 2, 32)  # no prefetch pending
    finally:
        s.close()


def test_install_refuses_a_source_with_no_module(table_gguf, monkeypatch):
    path, _raw = table_gguf
    src = _sources(path, monkeypatch)
    with pytest.raises(ValueError, match="no module to attach"):
        install_deferred_tables(_FakeModel(None), src)
    assert install_deferred_tables(_FakeModel(None), {}) == []


class _Layer:
    def __init__(self, engram):
        self.engram = engram


class _Inner:
    def __init__(self, table):
        self.layers = [_Layer(None), _Layer(_Engram(table))] if table else []


class _Engram:
    def __init__(self, table):
        self.embed = table


class _FakeModel:
    """The shape ``streamable_tables_for`` walks: model.layers[i].engram."""
    model_type = "deepseek_v41"

    def __init__(self, table):
        self.model = _Inner(table)


def test_a_dropped_module_closes_its_source(table_gguf, monkeypatch):
    """The fd and the reader pool must not outlive the table."""
    path, _raw = table_gguf
    src = _sources(path, monkeypatch)["blk.1.engram_embd.weight"]
    model = _FakeModel(_engram_table())
    install_deferred_tables(model, {"blk.1.engram_embd.weight": src})
    src.read(np.array([0]))               # open the fd and the pool
    assert src._fd is not None and src._pool is not None

    del model
    gc.collect()   # a loaded model sits in reference cycles; see installs.py
    assert src._fd is None and src._pool is None


def test_a_kquant_table_past_the_ceiling_is_refused(table_gguf, monkeypatch):
    """Skipped tensors bring no scales, so there is nothing to dequantize
    against."""
    path, _raw = table_gguf
    src = _sources(path, monkeypatch)["blk.1.engram_embd.weight"]
    table = _engram_table()
    table.kquant_type = "q2_k"
    model = _FakeModel(table)
    install_deferred_tables(model, {"blk.1.engram_embd.weight": src})
    try:
        with pytest.raises(NotImplementedError, match="q2_k table past"):
            table(mx.array([[1]]))
    finally:
        src.close()


def test_gather_stats_totals_the_row_reads(table_gguf, monkeypatch):
    from gmlx.stream.table_pread import gather_stats

    path, _raw = table_gguf
    src = _sources(path, monkeypatch)["blk.1.engram_embd.weight"]
    model = _FakeModel(_engram_table())
    install_deferred_tables(model, {"blk.1.engram_embd.weight": src})
    try:
        assert gather_stats(model) == {"tables": 1, "gathers": 0, "rows": 0,
                                       "seconds": 0.0}
        model.model.layers[1].engram.embed(mx.array([[3, 900, 3]]))
        got = gather_stats(model)
        # Duplicate ids read once.
        assert (got["tables"], got["gathers"], got["rows"]) == (1, 1, 2)
        assert got["seconds"] > 0
    finally:
        src.close()


def test_close_returns_while_a_read_ahead_is_queued(table_gguf, monkeypatch):
    """A read-ahead takes the source lock when it starts. A close that
    waits for it while holding that lock never returns."""
    path, raw = table_gguf
    s = _sources(path, monkeypatch)["blk.1.engram_embd.weight"]
    s.read(np.array([0]))                # open the fd and the pools
    gate = threading.Event()
    s._ahead.submit(gate.wait)           # holds the one read-ahead thread
    try:
        queued = s.read_ahead(np.array([1, 2]))
        closer = threading.Thread(target=s.close, daemon=True)
        closer.start()
        closer.join(0.5)
    finally:
        gate.set()
    closer.join(5)
    assert not closer.is_alive(), "close deadlocked against the read-ahead"
    assert queued.cancelled() or queued.exception() is not None
    assert s._fd is None and s._pool is None and s._ahead is None


def test_a_read_after_close_raises(table_gguf, monkeypatch):
    path, _raw = table_gguf
    s = _sources(path, monkeypatch)["blk.1.engram_embd.weight"]
    s.read(np.array([0]))
    s.close()
    s.close()                            # idempotent
    with pytest.raises(RuntimeError, match="is closed"):
        s.read(np.array([0]))
    with pytest.raises(RuntimeError, match="is closed"):
        s.read_ahead(np.array([0]))
    assert s._fd is None


def test_close_leaves_the_fd_to_the_read_in_flight(table_gguf, monkeypatch):
    """A read in flight keeps its fd until it ends, so a close never hands
    its preads a closed or reused descriptor."""
    import gmlx.stream.table_pread as tp

    path, raw = table_gguf
    s = _sources(path, monkeypatch)["blk.1.engram_embd.weight"]
    started, gate = threading.Event(), threading.Event()
    real = tp.read_range_aligned

    def slow(fd, mv, off, size):
        started.set()
        gate.wait()
        real(fd, mv, off, size)

    monkeypatch.setattr(tp, "read_range_aligned", slow)
    fut = s.read_ahead(np.array([7]))
    try:
        assert started.wait(5)
        s.close(wait=False)
        assert s._fd is not None, "fd closed under a read in flight"
    finally:
        gate.set()
    assert np.array_equal(fut.result(timeout=5), raw[[7]])
    assert s._fd is None
