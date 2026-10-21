"""File-backed gather for a table past the device buffer ceiling."""
from __future__ import annotations

import gc
import threading
from concurrent.futures import CancelledError
from concurrent.futures import wait as futures_wait

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
    if closer.is_alive():
        # Free the read-ahead worker, or the interpreter hangs at exit
        # joining it.
        s._lock.release()
        closer.join(5)
        pytest.fail("close deadlocked against the read-ahead")
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


def test_a_refused_submit_waits_for_the_reads_already_queued(
        table_gguf, monkeypatch):
    """A close between the row-pool submits of one read refuses the rest.
    The read raises, and the fd stays open until the fills already
    submitted end, so no pread runs on a closed or reused fd."""
    import os

    import gmlx.stream.table_pread as tp

    monkeypatch.setenv("GMLX_TABLE_PREAD_WORKERS", "4")
    path, _raw = table_gguf
    s = _sources(path, monkeypatch)["blk.1.engram_embd.weight"]
    fd, pool, _ = s._ready()
    s._release()
    closed, late = threading.Event(), []
    running, close_called = threading.Semaphore(0), threading.Event()
    real_read, real_close = tp.read_range_aligned, os.close

    def read(rfd, mv, off, size):
        if rfd == fd and not close_called.is_set():
            # A fill holds here until the close: its reads straddle it.
            running.release()
            close_called.wait(30)
        if rfd == fd and closed.is_set():
            late.append(off)
        return real_read(rfd, mv, off, size)

    def close(cfd):
        if cfd == fd:
            closed.set()
        return real_close(cfd)

    monkeypatch.setattr(tp, "read_range_aligned", read)
    monkeypatch.setattr(tp.os, "close", close)
    real_submit, submitted = pool.submit, []

    def submit(fn, *args):
        if len(submitted) == 2:
            for _ in range(2):           # the first two fills are running
                assert running.acquire(timeout=30)
            closer = threading.Thread(target=s.close, kwargs={"wait": False})
            closer.start()
            closer.join()
            close_called.set()
        submitted.append(real_submit(fn, *args))
        return submitted[-1]

    monkeypatch.setattr(pool, "submit", submit)
    with pytest.raises(RuntimeError):
        s.read(np.arange(40))
    futures_wait(submitted, timeout=30)
    assert len(submitted) == 2
    assert closed.is_set() and s._fd is None
    assert late == [], f"{len(late)} preads ran on the closed fd"


def test_the_finalizer_closes_without_joining_the_readers(
        table_gguf, monkeypatch):
    """A dropped module closes its source from whatever thread collects
    it, a reader thread included, which must not join itself or the
    read-ahead that waits on it."""
    import types
    import weakref

    import gmlx.stream.table_pread as tp

    monkeypatch.setenv("GMLX_TABLE_PREAD_WORKERS", "4")
    path, raw = table_gguf
    src = _sources(path, monkeypatch)["blk.1.engram_embd.weight"]
    finalizers = []

    def finalize(obj, fn, *args):
        finalizers.append(weakref.finalize(obj, fn, *args))
        return finalizers[-1]

    monkeypatch.setattr(tp, "weakref", types.SimpleNamespace(finalize=finalize))
    model = _FakeModel(_engram_table())
    install_deferred_tables(model, {"blk.1.engram_embd.weight": src})
    (fin,) = finalizers
    # A close that joins would deadlock the reader below. Fail first.
    assert fin.peek()[2] == (False,), "the finalizer's close joins the readers"
    real, errors = tp.read_range_aligned, []

    def read(fd, mv, off, size):
        if threading.current_thread().name.startswith("gmlx-table_") \
                and fin.alive:
            try:
                fin()                    # as a collection on this thread
            except Exception as e:       # noqa: BLE001
                errors.append(e)
        return real(fd, mv, off, size)

    _, pool, _ = src._ready()
    src._release()
    real_submit, fills = pool.submit, []

    def submit(fn, *args):
        fills.append(real_submit(fn, *args))
        return fills[-1]

    monkeypatch.setattr(pool, "submit", submit)
    monkeypatch.setattr(tp, "read_range_aligned", read)
    fut = src.read_ahead(np.arange(40))
    done, _ = futures_wait([fut], timeout=5)
    if not done:
        # A read-ahead left waiting on the fills the close cancelled would
        # hang interpreter exit, which joins its worker. Wake it first.
        for f in fills:
            if f.cancelled():
                f.set_running_or_notify_cancel()
        pytest.fail("the read-ahead never ended after the finalizer's close")
    assert errors == []
    try:
        assert np.array_equal(fut.result(), raw[:40])
    except (CancelledError, RuntimeError):
        pass              # the close cancelled a queued fill or refused one
    assert src._fd is None
    del model


def test_a_close_that_cancels_queued_fills_ends_the_read(
        table_gguf, monkeypatch):
    """A close cancels the row fills no worker has started. The read they
    belong to raises rather than wait on them forever."""
    monkeypatch.setenv("GMLX_TABLE_PREAD_WORKERS", "4")
    path, _raw = table_gguf
    s = _sources(path, monkeypatch)["blk.1.engram_embd.weight"]
    _, pool, _ = s._ready()
    s._release()
    gate, busy = threading.Event(), threading.Semaphore(0)

    def hold():
        busy.release()
        gate.wait()

    for _ in range(4):                   # every row worker is busy
        pool.submit(hold)
    for _ in range(4):
        assert busy.acquire(timeout=5)
    real_submit, submitted, queued = pool.submit, [], threading.Event()

    def submit(fn, *args):
        submitted.append(real_submit(fn, *args))
        if len(submitted) == 4:
            queued.set()
        return submitted[-1]

    monkeypatch.setattr(pool, "submit", submit)
    raised = []

    def read():
        try:
            s.read(np.arange(40))
        except BaseException as e:       # noqa: BLE001
            raised.append(e)

    reader = threading.Thread(target=read, daemon=True)
    try:
        reader.start()
        assert queued.wait(5)
        s.close(wait=False)
    finally:
        gate.set()
    reader.join(5)
    assert not reader.is_alive(), "the read waits on a cancelled fill"
    assert all(f.cancelled() for f in submitted)
    assert len(raised) == 1 and isinstance(raised[0], CancelledError)
    assert s._fd is None


# The PLE-keyed conversion stores one shared table in a GGUF quant (Q5_K
# on the 1017 UD-IQ2_XXS file).
Q5K_ROWS, Q5K_DIMS, Q5K_BYTES = 40, 256, 176


@pytest.fixture
def q5k_table_gguf(tmp_path):
    rng = np.random.default_rng(11)
    raw = rng.integers(0, 256, size=(Q5K_ROWS, Q5K_BYTES), dtype=np.uint8)
    # Each block opens with its f16 scale and min; keep them finite.
    dd = rng.uniform(0.01, 0.1, size=(Q5K_ROWS, 2)).astype(np.float16)
    raw[:, :4] = dd.view(np.uint8).reshape(Q5K_ROWS, 4)
    path = str(tmp_path / "q5k.gguf")
    w = gguf.GGUFWriter(path, "deepseek41")
    w.add_tensor("per_layer_token_embd.weight", raw,
                 raw_dtype=gguf.GGMLQuantizationType.Q5_K)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return path, raw


class _SharedInner:
    def __init__(self, table):
        self.engram_embed = table
        self.layers = []


class _SharedModel:
    model_type = "deepseek_v41"

    def __init__(self, table):
        self.model = _SharedInner(table)


def test_a_kquant_table_past_the_ceiling_decodes_from_the_file(
        q5k_table_gguf, monkeypatch):
    from gguf.quants import dequantize as gguf_dequantize
    from mlx_kquant.nn import KQuantEmbedding

    from gmlx.load.dtypes import activation_dtype
    from gmlx.models.deepseek_v41.model import EngramTable

    path, raw = q5k_table_gguf
    src = _sources(path, monkeypatch)
    s = src["per_layer_token_embd.weight"]
    assert (s.codec, s.dims, s.row_bytes) == ("q5_k", Q5K_DIMS, Q5K_BYTES)

    table = EngramTable(Q5K_ROWS, Q5K_DIMS)
    model = _SharedModel(table)
    assert install_deferred_tables(model, src) == ["per_layer_token_embd.weight"]
    try:
        ids = [[3, 39, 3], [0, 17, 8]]
        got = table(mx.array(ids))
        # The same bytes held in an array, through the resident gather.
        resident = KQuantEmbedding(Q5K_ROWS, Q5K_DIMS, "q5_k",
                                   out_dtype=activation_dtype("deepseek_v41"))
        resident.weight = mx.array(raw)
        want = resident(mx.array(ids))
        assert got.shape == (2, 3, Q5K_DIMS) and got.dtype == want.dtype
        assert mx.array_equal(got, want)
        # And against gguf-py's decode of the format.
        ref = gguf_dequantize(raw, gguf.GGMLQuantizationType.Q5_K)[np.array(ids)]
        assert np.allclose(np.array(got.astype(mx.float32)), ref,
                           rtol=1e-2, atol=1e-3)
    finally:
        s.close()


def test_a_table_codec_the_build_lacks_is_refused(q5k_table_gguf, monkeypatch):
    import mlx_kquant as kq

    from gmlx.models.deepseek_v41.model import EngramTable

    path, _raw = q5k_table_gguf
    src = _sources(path, monkeypatch)
    monkeypatch.setattr(kq, "codecs", lambda: ["q8_0"])
    try:
        with pytest.raises(RuntimeError, match="q5_k codec"):
            install_deferred_tables(
                _SharedModel(EngramTable(Q5K_ROWS, Q5K_DIMS)), src)
    finally:
        src["per_layer_token_embd.weight"].close()


# The larger unsloth quants store the shared table as MXFP8 on GGML type
# id 43: per 32 values, one E8M0 scale byte, then 32 E4M3 bytes.
MX_ROWS, MX_DIMS = 24, 256


def _mxfp8_reference(raw: np.ndarray) -> np.ndarray:
    """A numpy port of the format's reference row decode."""
    code = np.arange(256)
    exp, man = (code >> 3) & 15, code & 7
    e4m3 = np.where(exp > 0, (8 + man) * 2.0 ** (exp - 10), man * 2.0 ** -9)
    e4m3 = np.where(code & 128, -e4m3, e4m3)
    e4m3[(code & 127) == 127] = np.nan
    blocks = raw.reshape(raw.shape[0], -1, 33)
    scale = 2.0 ** (blocks[..., :1].astype(np.float64) - 127)
    scale[blocks[..., :1] == 255] = np.nan
    return (e4m3[blocks[..., 1:]] * scale).reshape(raw.shape[0], -1)


@pytest.fixture
def mxfp8_table_gguf(tmp_path):
    from helpers import mint_type43_gguf

    rng = np.random.default_rng(13)
    raw = rng.integers(0, 256, size=(MX_ROWS, MX_DIMS // 32, 33),
                       dtype=np.uint8)
    # Every E4M3 code, both NaN codes among them, and scales across the
    # range, up to the NaN code. Scales at the bottom of the range give
    # subnormal values, which the GPU flushes to zero.
    raw[:8, :, 1:] = np.arange(256, dtype=np.uint8).reshape(8, 1, 32)
    raw[:, :, 0] = rng.integers(100, 140, size=(MX_ROWS, MX_DIMS // 32))
    raw[8, :, 0] = [16, 64, 126, 127, 128, 200, 254, 255]
    raw = raw.reshape(MX_ROWS, -1)
    path = str(tmp_path / "mxfp8.gguf")
    mint_type43_gguf(path, [("per_layer_token_embd.weight",
                             (MX_DIMS, MX_ROWS), "MXFP8", raw.tobytes())])
    return path, raw


def test_an_mxfp8_table_past_the_ceiling_decodes_from_the_file(
        mxfp8_table_gguf, monkeypatch):
    from gmlx.models.deepseek_v41.model import EngramTable

    path, raw = mxfp8_table_gguf
    src = _sources(path, monkeypatch)
    s = src["per_layer_token_embd.weight"]
    assert (s.codec, s.dims, s.row_bytes) == ("mxfp8", MX_DIMS, 264)

    # Read without a kquant mxfp8 codec.
    import mlx_kquant as kq
    monkeypatch.setattr(kq, "codecs", lambda: ["q8_0"])
    table = EngramTable(MX_ROWS, MX_DIMS)
    install_deferred_tables(_SharedModel(table), src)
    try:
        ids = np.array([[0, 7, 8, 23, 8]])
        got = np.array(table(mx.array(ids)).astype(mx.float32))[0]
        want = _mxfp8_reference(raw)[ids[0]]
        # The engram path rounds rows to bfloat16, as the reference does.
        want = np.array(mx.array(want.astype(np.float32))
                        .astype(mx.bfloat16).astype(mx.float32))
        assert np.array_equal(np.isnan(got), np.isnan(want))
        assert np.array_equal(np.nan_to_num(got), np.nan_to_num(want))
    finally:
        s.close()
