"""File-backed row gather for a lookup table no MLX array can hold.

A tensor larger than the device max buffer length has no MTLBuffer, so
neither a zero-copy view nor a copy of it exists: ``kq.load_gguf`` fails
on the whole file. DeepSeek-V4.1's fp8 engram tables are 101 GB each and
pass that ceiling on a 128 GB box (the same tables are 32 GB at Q2_K and
stay under it). The loader leaves such a tensor out of the wire load
through ``kq.load_gguf(skip=...)`` and the table module reads its rows
from the GGUF instead, which is what the reference runtime does too.

The rows a step needs are tiny next to the table: 24 rows of 264 bytes
per token per engram table. Reads bypass the page cache and round out to
page boundaries, per the ``decode_feeder`` rules, since a 264-byte row
is never page-aligned and an unaligned F_NOCACHE read can wedge in the
kernel. The cost is one host sync per gather, to read the row ids.

``GMLX_TABLE_MAX_BUFFER`` overrides the ceiling (bytes), for tests.
``GMLX_TABLE_PREAD_WORKERS`` sets the reader thread count.
"""
from __future__ import annotations

import fcntl
import os
import threading
import time
import weakref
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import mlx.core as mx
import numpy as np

from .feeder_common import read_range_aligned


def max_buffer_bytes() -> int | None:
    """The largest tensor this device can hold in one array, or None when
    the limit is unknown (no Metal device: a CPU-only build)."""
    env = os.environ.get("GMLX_TABLE_MAX_BUFFER", "")
    if env:
        return int(env)
    try:
        return int(mx.device_info()["max_buffer_length"])
    except Exception:
        return None


@dataclass
class TableSource:
    """One table's rows, addressed in the GGUF file."""
    path: str
    offset: int          # absolute byte offset of row 0
    rows: int
    row_bytes: int
    _fd: int | None = field(default=None, repr=False)
    _pool: ThreadPoolExecutor | None = field(default=None, repr=False)
    _ahead: ThreadPoolExecutor | None = field(default=None, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _size: int = field(default=0, repr=False)
    _workers: int = field(default=0, repr=False)
    # Gather cost, so a run can price the tier against its own wall.
    gathers: int = field(default=0, repr=False)
    gathered_rows: int = field(default=0, repr=False)
    gather_seconds: float = field(default=0.0, repr=False)

    @property
    def nbytes(self) -> int:
        return self.rows * self.row_bytes

    def _ready(self) -> tuple[int, ThreadPoolExecutor]:
        with self._lock:
            if self._fd is None:
                fd = os.open(self.path, os.O_RDONLY)
                if hasattr(fcntl, "F_NOCACHE"):
                    fcntl.fcntl(fd, fcntl.F_NOCACHE, 1)
                self._fd = fd
                self._size = os.fstat(fd).st_size
            if self._pool is None:
                self._workers = max(
                    1, int(os.environ.get("GMLX_TABLE_PREAD_WORKERS", "32")))
                self._pool = ThreadPoolExecutor(
                    self._workers, thread_name_prefix="gmlx-table")
                # One thread runs a whole read() ahead of the forward; it
                # must not sit in the row pool, whose workers it joins.
                self._ahead = ThreadPoolExecutor(
                    1, thread_name_prefix="gmlx-table-ahead")
            return self._fd, self._pool

    def read_ahead(self, ids: np.ndarray):
        """``read(ids)`` on its own thread; a Future of the rows."""
        self._ready()
        return self._ahead.submit(self.read, ids)

    def read(self, ids: np.ndarray) -> np.ndarray:
        """Rows ``ids`` as raw bytes, shape ``[len(ids), row_bytes]``."""
        fd, pool = self._ready()
        t0 = time.perf_counter()
        uniq, inv = np.unique(ids, return_inverse=True)
        if uniq.size and (uniq[0] < 0 or uniq[-1] >= self.rows):
            raise IndexError(
                f"table row id out of range: [{uniq[0]}, {uniq[-1]}] "
                f"vs {self.rows} rows in {os.path.basename(self.path)}")
        out = np.empty((uniq.size, self.row_bytes), dtype=np.uint8)
        rb, base, size = self.row_bytes, self.offset, self._size
        mv = memoryview(out.reshape(-1))

        def fill(lo: int, hi: int) -> None:
            for j in range(lo, hi):
                read_range_aligned(
                    fd, mv[j * rb:(j + 1) * rb],
                    base + int(uniq[j]) * rb, size)

        span = max(1, -(-uniq.size // self._workers))
        cuts = [(lo, min(lo + span, uniq.size))
                for lo in range(0, uniq.size, span)]
        if len(cuts) > 1:
            for f in [pool.submit(fill, lo, hi) for lo, hi in cuts]:
                f.result()
        elif cuts:
            fill(*cuts[0])
        self.gathers += 1
        self.gathered_rows += int(uniq.size)
        self.gather_seconds += time.perf_counter() - t0
        return out[inv]

    def close(self) -> None:
        with self._lock:
            if self._ahead is not None:
                self._ahead.shutdown(wait=True)
                self._ahead = None
            if self._pool is not None:
                self._pool.shutdown(wait=True)
                self._pool = None
            if self._fd is not None:
                os.close(self._fd)
                self._fd = None


def oversize_tables(shards: list[str], arch: str | None,
                    cap: int | None = None) -> dict[str, TableSource]:
    """Streamable tables of ``arch`` in ``shards`` that pass the device
    ceiling, as wire name -> source. A pure function of the files: the
    wire load calls it for the skip list, the model install for the
    sources."""
    from .table_stream import streamable_table_names

    pattern = streamable_table_names(arch)
    if pattern is None:
        return {}
    cap = max_buffer_bytes() if cap is None else cap
    if cap is None:
        return {}
    from gmlx.load.headerscan import scan_gguf

    out: dict[str, TableSource] = {}
    for shard in shards:
        h = scan_gguf(shard, array_limit=0)
        for t in h.tensors:
            if t.nbytes <= cap or not pattern.fullmatch(t.name):
                continue
            rows = int(np.prod(t.shape[1:])) if len(t.shape) > 1 else 1
            out[t.name] = TableSource(
                path=shard, offset=h.data_offset + t.offset, rows=rows,
                row_bytes=t.nbytes // rows)
    return out


_PREAD_CACHE: dict[type, type] = {}


def _pread_class(cls):
    """Per-instance ``__class__`` swap target: the row gather reads the
    GGUF instead of indexing a ``weight`` array, which this table has
    none of. Counts as streamed, since nothing of it is ever wired."""
    sub = _PREAD_CACHE.get(cls)
    if sub is not None:
        return sub

    class _TablePread(cls):
        _kq_table_streamed = True

        def prefetch(self, x) -> None:
            """Start the gather for ``x`` now; the ``__call__`` with the
            same ids joins it instead of reading. The ids depend only on
            the input tokens, so a forward can issue every table's read
            before its first layer runs."""
            src = self._kq_table_source
            mx.eval(x)
            ids = np.asarray(x, dtype=np.int64).reshape(-1)
            self._kq_ahead = (ids, src.read_ahead(ids))

        def __call__(self, x):
            src = self._kq_table_source
            mx.eval(x)
            ids = np.asarray(x, dtype=np.int64).reshape(-1)
            ahead = getattr(self, "_kq_ahead", None)
            if ahead is not None:
                self._kq_ahead = None
                if ahead[0].shape == ids.shape and np.array_equal(ahead[0], ids):
                    rows = ahead[1].result()
                else:
                    ahead[1].result()  # a stale read still owns the pool
                    rows = src.read(ids)
            else:
                rows = src.read(ids)
            out = mx.array(rows).reshape(*x.shape, src.row_bytes)
            if hasattr(self, "kquant_type"):
                # A skipped tensor brings no <name>.scales, so the module
                # holds only its own placeholder. Fail rather than
                # dequantize against it.
                raise NotImplementedError(
                    f"{self.kquant_type} table past the device buffer "
                    "ceiling: the file-backed gather handles raw-byte "
                    "rows only")
            decode = getattr(self, "decode_gathered", None)
            return out if decode is None else decode(out)

        def as_linear(self, x):
            raise RuntimeError(
                "as_linear on a file-backed table would read the whole "
                "table per call; such a table cannot back a tied lm_head")

        def _extra_repr(self):
            src = self._kq_table_source
            return f"{src.rows}, file-backed rows of {src.row_bytes} B"

    _TablePread.__name__ = cls.__name__ + "_TablePread"
    _PREAD_CACHE[cls] = _TablePread
    return _TablePread


def gather_stats(model) -> dict:
    """Row-gather totals over every file-backed table on ``model``. The
    tables run one after another in a forward, so the walls add up."""
    from .table_stream import streamable_tables_for

    out = {"tables": 0, "gathers": 0, "rows": 0, "seconds": 0.0}
    for _, mod in streamable_tables_for(model):
        src = getattr(mod, "_kq_table_source", None)
        if src is None:
            continue
        out["tables"] += 1
        out["gathers"] += src.gathers
        out["rows"] += src.gathered_rows
        out["seconds"] += src.gather_seconds
    return out


def install_deferred_tables(model, sources: dict[str, TableSource]) -> list[str]:
    """Attach file sources to the tables the wire load skipped, and drop
    their placeholder ``weight``. Returns the wire names installed this
    call; a source with no module on ``model`` raises, because the table
    would otherwise answer from an uninitialized array."""
    if not sources:
        return []
    from .table_stream import streamable_tables_for

    done: list[str] = []
    already: set[str] = set()
    for tier, mod in streamable_tables_for(model):
        src = sources.get(tier.gguf_name)
        if src is None:
            continue
        if getattr(mod, "_kq_table_source", None) is not None:
            already.add(tier.gguf_name)
            continue
        w = getattr(mod, "weight", None)
        if w is not None and int(w.shape[-1]) != src.row_bytes:
            raise ValueError(
                f"{tier.gguf_name}: model row width {w.shape[-1]} != file "
                f"{src.row_bytes}")
        mod._kq_table_source = src
        if w is not None:
            del mod["weight"]
        mod.__class__ = _pread_class(mod.__class__)
        # The fd and the reader pool outlive every reference but the
        # module's; a server that releases models would leak both.
        weakref.finalize(mod, src.close)
        done.append(tier.gguf_name)
    missed = sorted(set(sources) - set(done) - already)
    if missed:
        raise ValueError(
            f"{len(missed)} oversize tables have no module to attach to: "
            f"{missed}")
    return done
