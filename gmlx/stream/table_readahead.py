"""Read a streamed lookup table's pages ahead of the prefill.

A table its load maps outside MLX's tracker, such as Qwen3.8-Flash-Next's
per-layer embedding table, is never wired: the gather on the table stream
reads its rows through page faults, one at a time, and a cold page stalls
the chunk for a disk read. The rows a token reads are a hash of its token
history, known before its chunk runs. Just before each prefill chunk's
forward (``read_ahead``, from the stock prompt step and the MTP prefill
body) the read-ahead hashes the chunk's rows on the host, asks the kernel
which of their pages are resident (``mincore``, one call per page) and
reads the rest into the page cache on worker threads (``pread``), in the
order the tokens read them, while the chunk's own gather runs. The rows
of the two chunks after it follow, so later chunks find their pages read.

It reads the prompts of a server with a prompt cache, whose prompt
batches keep each whole prompt in their cache metadata.
``GMLX_TABLE_READAHEAD=0`` turns it off and
``GMLX_TABLE_READAHEAD_WORKERS`` sets the reader count (16). The model's
streaming owner holds it (``installs.STREAM_ATTRS``), so a release closes
it with the other helpers.
"""
from __future__ import annotations

import ctypes
import logging
import os
import queue
import threading
import time
import weakref

import numpy as np

from gmlx.envflags import env_bool, env_int

_log = logging.getLogger(__name__)

# On the streaming owner and on the table module it reads for.
_ATTR = "_kq_table_readahead"
_UPTO = "_gmlx_table_readahead_upto"
_PROT_READ = 0x1
_MAP_SHARED = 0x0001


def readahead_enabled() -> bool:
    return env_bool("GMLX_TABLE_READAHEAD", True)


def _workers() -> int:
    return max(1, env_int("GMLX_TABLE_READAHEAD_WORKERS", 16))


class TableReadAhead:
    """Reads the pages of one table's rows into the page cache.

    ``hasher`` maps ``[B, context_len + T]`` token ids, history first, to
    their ``[B, T, heads]`` rows (``PLEHostHash``). ``submit`` takes spans
    ``(tokens, start, end)``: the rows of ``tokens[start:end]``, with the
    tokens before ``start`` as their history. One coordinator thread
    hashes each submission and maps its rows to pages; the workers check
    and read them, page ``k`` of a submission on worker ``k % workers``.
    """

    def __init__(self, path: str, offset: int, rows: int, row_bytes: int,
                 hasher, workers: int | None = None):
        from .feeder_common import DaemonPool

        self.path = path
        self.offset = int(offset)
        self.rows = int(rows)
        self.row_bytes = int(row_bytes)
        self.hasher = hasher
        self.page = os.sysconf("SC_PAGE_SIZE")
        # An own handle: argtypes set here stay off the shared one.
        libc = ctypes.CDLL(None, use_errno=True)
        libc.mmap.restype = ctypes.c_void_p
        libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int,
                              ctypes.c_int, ctypes.c_int, ctypes.c_int64]
        libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t,
                                 ctypes.c_void_p]
        self._libc = libc
        fd = os.open(path, os.O_RDONLY)
        size = os.fstat(fd).st_size
        addr = libc.mmap(None, size, _PROT_READ, _MAP_SHARED, fd, 0)
        if not addr or addr == ctypes.c_void_p(-1).value:
            os.close(fd)
            raise OSError(ctypes.get_errno(), f"mmap of {path} failed")
        self._fd, self._size, self._addr = fd, size, addr
        self.workers = int(workers or _workers())
        self._pool = DaemonPool(self.workers, name="gmlx-table-ahead")
        self._jobs: queue.SimpleQueue = queue.SimpleQueue()
        self._closed = False
        # Held across a submit's queueing, the hand-off to the pool and
        # close's shutdown of it, so no job is queued behind the stop and
        # no read is handed to a pool that is shut down.
        self._lock = threading.Lock()
        self._idle = threading.Condition()
        self._pending = 0
        self.stats = {"submits": 0, "tokens": 0, "pages": 0, "cold": 0,
                      "seconds": 0.0}
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="gmlx-table-ahead-hash")
        self._thread.start()

    def submit(self, spans) -> None:
        """Queue ``spans``, each ``(tokens, start, end)``. Returns at
        once."""
        if not spans:
            return
        with self._lock:
            # Under close's lock, so a job queued is ahead of its stop.
            if self._closed:
                return
            with self._idle:
                self._pending += 1
            self._jobs.put(list(spans))

    def wait(self, timeout: float | None = None) -> bool:
        """Block until every submission so far is read. False on timeout."""
        with self._idle:
            return self._idle.wait_for(lambda: self._pending == 0, timeout)

    def pages_of(self, rows: np.ndarray) -> np.ndarray:
        """File pages ``rows`` read, unique, in first-read order."""
        start = (self.offset
                 + rows.reshape(-1).astype(np.int64) * self.row_bytes)
        both = np.stack([start // self.page,
                         (start + self.row_bytes - 1) // self.page],
                        axis=1).reshape(-1)
        _, first = np.unique(both, return_index=True)
        return both[np.sort(first)]

    def _rows_of(self, spans) -> np.ndarray:
        h = self.hasher
        ctx, eos = h.context_len, h.eos_token_id
        out = []
        for tokens, start, end in spans:
            lo = start - ctx
            if lo >= 0:
                hist = np.asarray(tokens[lo:end], dtype=np.int64)
            else:
                hist = np.concatenate(
                    [np.full(-lo, eos, np.int64),
                     np.asarray(tokens[:end], dtype=np.int64)])
            rows = h(hist[None])[0]
            out.append(rows[(rows >= 0) & (rows < self.rows)])
        return np.concatenate(out) if out else np.zeros(0, np.int64)

    def _run(self) -> None:
        while True:
            spans = self._jobs.get()
            if spans is None:
                return
            t0 = time.perf_counter()
            try:
                if self._closed:
                    continue    # dropped; finally still counts it done
                pages = self.pages_of(self._rows_of(spans))
                n = len(pages)
                with self._lock:
                    # A closed read-ahead drops the job: its pool is shut
                    # down, and work handed to it would never run.
                    if self._closed:
                        continue
                    futs = [self._pool.submit(self._read,
                                              pages[k::self.workers])
                            for k in range(min(self.workers, n))]
                cold = 0
                for f in futs:
                    try:
                        cold += f.result()
                    except Exception:
                        _log.debug("table read-ahead read failed",
                                   exc_info=True)
                self.stats["submits"] += 1
                self.stats["tokens"] += sum(e - s for _, s, e in spans)
                self.stats["pages"] += n
                self.stats["cold"] += cold
                self.stats["seconds"] += time.perf_counter() - t0
            except Exception:
                _log.debug("table read-ahead failed", exc_info=True)
            finally:
                with self._idle:
                    self._pending -= 1
                    self._idle.notify_all()

    def _resident(self, off: int, vec) -> bool:
        return (self._libc.mincore(self._addr + off, self.page,
                                   ctypes.byref(vec)) == 0
                and bool(vec.value & 1))

    def _pread(self, off: int, buf) -> None:
        os.preadv(self._fd, [buf], off)

    def _read(self, pages) -> int:
        """Read the cold ones of ``pages``. Returns how many were cold."""
        vec = ctypes.c_ubyte()
        buf = bytearray(self.page)
        cold = 0
        try:
            for p in pages:
                if self._closed:
                    break
                off = int(p) * self.page
                if self._resident(off, vec):
                    continue
                self._pread(off, buf)
                cold += 1
        except OSError:
            pass
        return cold

    def close(self) -> None:
        """Stop the threads, unmap, close. Idempotent. The coordinator
        drops what is still queued and exits. Close waits for no worker:
        one still on a page may then call ``mincore`` on the unmapped
        range or ``pread`` a closed descriptor, or one reused since, both
        reads whose errors ``_read`` catches."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._pool.shutdown(wait=False, cancel_futures=True)
        self._jobs.put(None)
        self._thread.join(timeout=1.0)
        self._libc.munmap(self._addr, self._size)
        os.close(self._fd)


def readahead_of(model):
    """The read-ahead attached to ``model`` or a module along its wrapper
    chain (``installs.wrapper_chain``), or None."""
    from .installs import wrapper_chain

    for m in wrapper_chain(model):
        ra = getattr(m, _ATTR, None)
        if ra is not None:
            return ra
    return None


def plan_spans(batch, n: int) -> list:
    """What a prompt batch reads from the chunk of ``n`` columns it is
    about to run, as submissions for ``submit``: the chunk's own tokens
    first, then two steps past it. Each row whose prompt its cache
    metadata holds plans from its next token, or from where its last plan
    ended. A row's next token is its prompt length less the real columns
    still to run: the last ones of a left-padded row, the rest of its
    suffix on a right-padded one. A prefix the prompt cache restored is
    never read."""
    meta = getattr(batch, "_apc_meta", None) or ()
    if not meta:
        return []
    embeds = getattr(batch, "_inputs_embeds", None)
    ids = embeds if embeds is not None else batch._input_ids
    left = int(ids.shape[1])
    done = int(batch._processed_prompt_columns)
    step = max(int(n), int(getattr(batch, "prefill_step_size", 0) or 0))
    right = getattr(batch, "_right_pad_per_row", None) is not None
    suffix = list(getattr(batch, "_suffix_lens", None) or ())
    upto = batch.__dict__.setdefault(_UPTO, {})
    now, ahead = [], []
    for i, m in enumerate(meta):
        full = m.get("full_input_ids") if isinstance(m, dict) else None
        if full is None or i >= len(suffix):
            continue
        s = int(suffix[i])
        rest = max(0, s - done) if right else min(left, s)
        nxt = len(full) - rest
        start = max(upto.get(i, nxt), nxt)
        mid = min(len(full), nxt + int(n))
        end = min(len(full), mid + 2 * step)
        if mid > start:
            now.append((full, start, mid))
        if end > max(start, mid):
            ahead.append((full, max(start, mid), end))
        upto[i] = max(start, end)
    return [spans for spans in (now, ahead) if spans]


def read_ahead(batch, n: int) -> None:
    """Queue what a prompt batch reads in the chunk of ``n`` columns it is
    about to run, and in the two steps after it, on its model's read-ahead.
    Called just before the chunk's forward, whose table gather reads the
    chunk's rows. Nothing when the model has none."""
    ra = readahead_of(getattr(batch, "model", None))
    if ra is None:
        return
    try:
        for spans in plan_spans(batch, n):
            ra.submit(spans)
    except Exception:
        _log.debug("table read-ahead plan failed", exc_info=True)


def attach_table_readahead(model, gguf_path: str | None) -> bool:
    """Attach a read-ahead to the streamed table of ``model`` its load
    mapped untracked, whose model hashes its rows on the host
    (``table_row_hashes``), and hang it on ``model``, the streaming
    owner. Returns whether it attached one. Idempotent."""
    if (not readahead_enabled() or not gguf_path
            or getattr(model, _ATTR, None) is not None):
        return False
    from gmlx.load import loadlog
    from gmlx.load.preflight import find_split_shards

    from .table_pread import _table_sources
    from .table_stream import (STREAMABLE_TABLES, _table_hosts,
                               streamable_tables_for)

    for host in _table_hosts(model):
        hashes = getattr(host, "table_row_hashes", None)
        spec = STREAMABLE_TABLES.get(getattr(host, "model_type", None))
        if not callable(hashes) or spec is None:
            continue
        tables = [(tier, mod) for tier, mod in streamable_tables_for(host)
                  if getattr(mod, "_kq_table_untracked", False)
                  and getattr(mod, "_kq_table_streamed", False)]
        if not tables:
            continue
        hashers = hashes()
        shards = find_split_shards(gguf_path) or [gguf_path]
        sources, _ = _table_sources(shards, spec.arch, None)
        for tier, mod in tables:
            hasher = hashers.get(tier.param_path)
            src = sources.get(tier.gguf_name)
            if hasher is None or src is None:
                continue
            try:
                ra = TableReadAhead(src.path, src.offset, src.rows,
                                    src.row_bytes, hasher)
            except OSError as exc:
                loadlog.warn(f"[stream] {spec.label} not read ahead: {exc}")
                return False
            weakref.finalize(mod, ra.close)
            object.__setattr__(mod, _ATTR, ra)
            object.__setattr__(model, _ATTR, ra)
            loadlog.info(f"[stream] {spec.label} read ahead of each "
                         f"prefill chunk ({ra.workers} readers)")
            return True
    return False
