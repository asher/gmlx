"""File-backed row gather for a lookup table no MLX array can hold.

A tensor larger than the device max buffer length has no MTLBuffer, so
neither a zero-copy view nor a copy of it exists: ``kq.load_gguf`` fails
on the whole file. DeepSeek-V4.1's fp8 engram tables are 101 GB each and
pass that ceiling on a 128 GB box (the same tables are 32 GB at Q2_K and
stay under it). The loader leaves such a tensor out of the wire load
through ``kq.load_gguf(skip=...)`` and the table module reads its rows
from the GGUF instead, which is what the reference runtime does too.

The rows a step needs are tiny next to the table: 24 rows of 264 bytes
per token per engram table. Rows stored in a GGUF quant decode through
``kq.dequantize``, float rows by their bytes' dtype. MXFP8 rows (one E8M0 scale byte, then 32 E4M3 values,
per block) decode through the engram codec's lookup tables, so reading
them needs no kquant codec. Reads bypass the page cache and round out to
page boundaries, per the ``decode_feeder`` rules, since a 264-byte row
is never page-aligned and an unaligned F_NOCACHE read can wedge in the
kernel. The cost is one host sync per gather, to read the row ids.

An MXFP8 table is read the same way at any size when mlx-kquant has no
mxfp8 codec to hold it as an array.

``GMLX_TABLE_MAX_BUFFER`` overrides the ceiling (bytes), for tests.
``GMLX_TABLE_PREAD_WORKERS`` sets the reader thread count.
"""
from __future__ import annotations

import fcntl
import os
import threading
import time
import weakref
from concurrent.futures import CancelledError, ThreadPoolExecutor
from dataclasses import dataclass, field

import mlx.core as mx
import numpy as np

from gmlx.device import metal_value

from .feeder_common import read_range_aligned


def max_buffer_bytes() -> int | None:
    """The largest tensor the Metal device can hold in one array, whatever
    the default device is, or None when the limit is unknown (no Metal
    device: a CPU-only build)."""
    env = os.environ.get("GMLX_TABLE_MAX_BUFFER", "")
    if env:
        return int(env)
    try:
        return int(metal_value("max_buffer_length"))
    except Exception:
        return None


@dataclass
class TableSource:
    """One table's rows, addressed in the GGUF file."""
    path: str
    offset: int          # absolute byte offset of row 0
    rows: int
    row_bytes: int
    # The kquant codec of the rows, or None for raw bytes the table module
    # decodes itself.
    codec: str | None = None
    dims: int = 0        # values per row, when codec is set
    out_dtype: mx.Dtype = mx.bfloat16
    _fd: int | None = field(default=None, repr=False)
    _pool: ThreadPoolExecutor | None = field(default=None, repr=False)
    _ahead: ThreadPoolExecutor | None = field(default=None, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _size: int = field(default=0, repr=False)
    _workers: int = field(default=0, repr=False)
    # Reads between _ready() and _release(). The last one out closes the
    # fd of a closed source, so no pread reads a closed or reused fd.
    _users: int = field(default=0, repr=False)
    _closed: bool = field(default=False, repr=False)
    # Gather cost, so a run can price the tier against its own wall.
    gathers: int = field(default=0, repr=False)
    gathered_rows: int = field(default=0, repr=False)
    gather_seconds: float = field(default=0.0, repr=False)

    @property
    def nbytes(self) -> int:
        return self.rows * self.row_bytes

    def _ready(self) -> tuple[int, ThreadPoolExecutor, ThreadPoolExecutor]:
        """The fd and the two pools, opened on first use. Every call is
        paired with ``_release``."""
        with self._lock:
            if self._closed:
                raise RuntimeError(
                    f"table source {os.path.basename(self.path)} is closed")
            if self._fd is None:
                fd = os.open(self.path, os.O_RDONLY)
                if hasattr(fcntl, "F_NOCACHE"):
                    fcntl.fcntl(fd, fcntl.F_NOCACHE, 1)
                self._fd = fd
                self._size = os.fstat(fd).st_size
            if self._pool is None or self._ahead is None:
                self._workers = max(
                    1, int(os.environ.get("GMLX_TABLE_PREAD_WORKERS", "32")))
                self._pool = ThreadPoolExecutor(
                    self._workers, thread_name_prefix="gmlx-table")
                # One thread runs a whole read() ahead of the forward; it
                # must not sit in the row pool, whose workers it joins.
                self._ahead = ThreadPoolExecutor(
                    1, thread_name_prefix="gmlx-table-ahead")
            self._users += 1
            return self._fd, self._pool, self._ahead

    def _release(self) -> None:
        with self._lock:
            self._users -= 1
            fd = None
            if self._closed and not self._users:
                fd, self._fd = self._fd, None
        if fd is not None:
            os.close(fd)

    def read_ahead(self, ids: np.ndarray):
        """``read(ids)`` on its own thread; a Future of the rows."""
        _, _, ahead = self._ready()
        try:
            return ahead.submit(self.read, ids)
        finally:
            self._release()

    def read(self, ids: np.ndarray) -> np.ndarray:
        """Rows ``ids`` as raw bytes, shape ``[len(ids), row_bytes]``."""
        fd, pool, _ = self._ready()
        try:
            return self._read(fd, pool, ids)
        finally:
            self._release()

    def _read(self, fd: int, pool: ThreadPoolExecutor,
              ids: np.ndarray) -> np.ndarray:
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
            futs = []
            try:
                for lo, hi in cuts:
                    futs.append(pool.submit(fill, lo, hi))
            finally:
                # A close can refuse the next submit. The fd stays open
                # until the reads already queued end. wait() never
                # returns on a fill the close cancelled, so each fill
                # is joined through its own future.
                for f in futs:
                    try:
                        f.exception()
                    except CancelledError:
                        pass
            for f in futs:
                f.result()
        elif cuts:
            fill(*cuts[0])
        self.gathers += 1
        self.gathered_rows += int(uniq.size)
        self.gather_seconds += time.perf_counter() - t0
        return out[inv]

    def close(self, wait: bool = True) -> None:
        """Stop the pools and close the fd. Later reads raise. A queued
        read-ahead is cancelled. The fd closes when the last read in
        flight ends. ``wait=False`` is the finalizer's form: a collection
        that runs on a reader thread cannot join that thread."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            pools = (self._ahead, self._pool)
            self._ahead = self._pool = None
            fd = None
            if not self._users:
                fd, self._fd = self._fd, None
        # Outside the lock: a running read-ahead takes it in _ready().
        for pool in pools:
            if pool is not None:
                pool.shutdown(wait=wait, cancel_futures=True)
        if fd is not None:
            os.close(fd)


def _table_sources(shards: list[str], arch: str | None,
                   cap: int | None) -> tuple[dict[str, TableSource], set]:
    """Every streamable table of ``arch`` in ``shards`` as wire name ->
    source, and the names no array can hold: past the device ceiling,
    or MXFP8 when mlx-kquant cannot hold it."""
    from .table_stream import streamable_table_names

    pattern = streamable_table_names(arch)
    if pattern is None:
        return {}, set()
    cap = max_buffer_bytes() if cap is None else cap
    from gmlx.load.headerscan import scan_gguf

    out: dict[str, TableSource] = {}
    held: set = set()
    for shard in shards:
        h = scan_gguf(shard, array_limit=0)
        for t in h.tensors:
            if not pattern.fullmatch(t.name):
                continue
            rows = int(np.prod(t.shape[1:])) if len(t.shape) > 1 else 1
            codec = _row_codec(t.type_name)
            out[t.name] = TableSource(
                path=shard, offset=h.data_offset + t.offset, rows=rows,
                row_bytes=t.nbytes // rows, codec=codec,
                dims=int(t.shape[0]) if codec else 0)
            too_big = cap is not None and t.nbytes > cap
            if too_big or (t.type_name == "MXFP8" and not _kq_has("mxfp8")):
                held.add(t.name)
    return out, held


def oversize_tables(shards: list[str], arch: str | None,
                    cap: int | None = None) -> dict[str, TableSource]:
    """Streamable tables of ``arch`` in ``shards`` that no array can hold,
    as wire name -> source: those past the device ceiling, and MXFP8
    ones when mlx-kquant cannot hold them."""
    sources, held = _table_sources(shards, arch, cap)
    return {n: s for n, s in sources.items() if n in held}


def deferred_tables(shards: list[str], arch: str | None,
                    cap: int | None = None, *, budget: int | None = None,
                    latched: bool | None = None,
                    env=None) -> dict[str, TableSource]:
    """Streamable tables of ``arch`` in ``shards`` that a load reads from
    the file, as wire name -> source: the oversize ones, and, on an arch
    that reads its tables ahead (``table_stream.tables_read_ahead``), all
    of them when they alone bring the text tensors under the stream
    budget (``table_stream.tables_bring_under``), so that an over-budget
    model loads with its experts resident. Under a stream flag
    (``latched``, read from the latch when None) the install's ladder
    streams the tables instead. ``budget`` defaults to the device stream
    budget. A function of the files, the build and the budget: the wire
    load calls it for the skip list, the model install for the
    sources."""
    from .table_stream import latched as table_latch
    from .table_stream import tables_bring_under, tables_read_ahead

    sources, held = _table_sources(shards, arch, cap)
    if not sources:
        return {}
    if latched is None:
        latched = table_latch()
    if (not latched and len(held) < len(sources)
            and tables_read_ahead(arch)):
        if budget is None:
            from gmlx.device import stream_budget_bytes

            budget = stream_budget_bytes()
        from gmlx.load.discovery import text_tensor_bytes

        text = text_tensor_bytes(shards[0]) if shards else None
        tbytes = sum(s.nbytes for s in sources.values())
        if text and tables_bring_under(text, tbytes, budget, env):
            return sources
    return {n: s for n, s in sources.items() if n in held}


def oversize_names(path: str, cap: int | None = None) -> list[str]:
    """Every tensor in ``path`` past the device ceiling, whatever its role.
    No array can hold one, so a load of the file must skip it."""
    cap = max_buffer_bytes() if cap is None else cap
    if cap is None:
        return []
    from gmlx.load.headerscan import scan_gguf

    return [t.name for t in scan_gguf(path, array_limit=0).tensors
            if t.nbytes > cap]


def _row_codec(type_name: str) -> str | None:
    """The codec of a table stored as GGML type ``type_name``: None for raw
    bytes (I8 / U8, decoded by the table module), else the lowercase type
    name, which is the kquant codec name."""
    name = type_name.lower()
    return None if name in ("i8", "u8") else name


def _check_codec(name: str, src: TableSource) -> None:
    if src.codec is None or src.codec == "mxfp8" or src.codec in _FLOAT_ROWS:
        return
    if not _kq_has(src.codec):
        raise RuntimeError(
            f"{name}: reading {src.codec} table rows from the file needs "
            f"an mlx-kquant build with a {src.codec} codec")


def _kq_has(codec: str) -> bool:
    import mlx_kquant as kq

    return codec in set(kq.codecs())


_FLOAT_ROWS = {"f32": mx.float32, "f16": mx.float16, "bf16": mx.bfloat16}

_MXFP8_LUTS: tuple | None = None


def _decode_mxfp8(raw: mx.array, dims: int) -> mx.array:
    global _MXFP8_LUTS
    if _MXFP8_LUTS is None:
        from gmlx.models.deepseek_v41.engram_codec import tables

        _MXFP8_LUTS = tuple(mx.array(t) for t in tables())
    values, scales = _MXFP8_LUTS
    # Metal flushes subnormals, so a block at the bottom of the scale range
    # (codes 0 and 1, values under 2^-126) decodes to zero on the GPU.
    blocks = raw.reshape(*raw.shape[:-1], dims // 32, 33).astype(mx.int32)
    out = values[blocks[..., 1:]] * scales[blocks[..., :1]]
    return out.reshape(*raw.shape[:-1], dims)


def _decode_rows(raw: mx.array, src: TableSource) -> mx.array:
    """``[..., row_bytes]`` uint8 rows to ``[..., dims]`` values. K-quant
    scales live in the row bytes; the scales input is the same 1-byte
    placeholder ``KQuantEmbedding`` passes."""
    if src.codec == "mxfp8":
        out = _decode_mxfp8(raw, src.dims)
    elif src.codec in _FLOAT_ROWS:
        out = raw.view(_FLOAT_ROWS[src.codec])
    else:
        import mlx_kquant as kq

        out = kq.dequantize(raw.reshape(-1, src.row_bytes),
                            mx.zeros((1,), dtype=mx.uint8), src.codec)
    return out.reshape(*raw.shape[:-1], src.dims).astype(src.out_dtype)


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
            if src.codec is not None:
                return _decode_rows(out, src)
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
            kind = f" {src.codec}" if src.codec else ""
            return f"{src.rows}, file-backed{kind} rows of {src.row_bytes} B"

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
        want = src.dims if src.codec else src.row_bytes
        if w is not None and int(w.shape[-1]) != want:
            raise ValueError(
                f"{tier.gguf_name}: model row width {w.shape[-1]} != file "
                f"{want}")
        _check_codec(tier.gguf_name, src)
        if src.codec is not None:
            from gmlx.load.dtypes import activation_dtype

            src.out_dtype = activation_dtype(getattr(model, "model_type", None))
        mod._kq_table_source = src
        if w is not None:
            del mod["weight"]
        mod.__class__ = _pread_class(mod.__class__)
        # The fd and the reader pool outlive every reference but the
        # module's; a server that releases models would leak both.
        weakref.finalize(mod, src.close, False)
        done.append(tier.gguf_name)
    missed = sorted(set(sources) - set(done) - already)
    if missed:
        raise ValueError(
            f"{len(missed)} oversize tables have no module to attach to: "
            f"{missed}")
    return done
