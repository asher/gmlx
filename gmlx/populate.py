"""Background page-cache populate for in-budget GGUF loads.

On a cold cache the first prefill demand-faults the weight mmap in fault
order, interleaved with GPU work; streaming the file in with sequential
preads instead lets the rest of load and the first request proceed against
a warming cache. This module is a dependency leaf (no mlx import at module
scope) so the kickoff can run before the heavy loader import wall.

Measured shape of the problem (29 GB A3B, M3 Max, cold cache):
- the stream takes ~4.1 s at 7 GB/s; any foreground work needing disk
  meanwhile (the CLI's import wall, thousands of small reads) is starved
  at the device (1.1 s -> 5.2 s measured; not GIL, not queue depth), so a
  fresh CLI cold run is bandwidth-serialized around ~9 s no matter how
  the phases are ordered - populate is not started before the loader
  imports (measured wash-to-harmful).
- in a warm process (the served case: imports paid, model cold), starting
  the stream at preflight and *waiting* for it before prefill beats racing
  prefill's scattered wiring faults against it by ~12% (7.8 vs 8.9 s to
  first token): the race collapses SSD throughput for both. Readers hand
  out chunks from one file-order frontier, and GGUF tensor order ~=
  forward-pass order, so the wait releases once the un-read tail is small
  enough to finish streaming under prefill's compute.

Models over the wired budget stream their experts from disk and must never
be populated: reading the whole file evicts its own head and fights the
decode arena for RAM. ``maybe_populate_for_load`` applies the same
0.9x-working-set rule the loader's residency warm uses, on file bytes
(>= param bytes, so strictly more conservative).

``GMLX_RESIDENCY_WARM=0`` disables populate along with the GPU touch;
``GMLX_POPULATE_EARLY=0`` restores the old phase-7 kickoff point;
``GMLX_POPULATE_WAIT=0`` restores the prefill/stream race.
"""

from __future__ import annotations

import os
import threading
import time

from .envflags import env_bool, env_float

_POPULATE_CHUNK = 32 << 20
_POPULATE_WORKERS = 8
_POPULATE_SKIP_RESIDENT = 0.95


class _Stream:
    """One file's populate: a shared file-order chunk frontier."""

    def __init__(self, path: str, size: int):
        self.path = path
        self.size = size
        self.next = 0        # first unclaimed byte (frontier)
        self.done = 0        # bytes actually read
        self.threads: list[threading.Thread] = []


_streams: dict[str, _Stream] = {}
_lock = threading.Lock()


def resident_fraction(path: str, samples: int = 512) -> float | None:
    """Sampled page-cache residency of a file via mincore (macOS reports
    UBC residency for an untouched PROT_READ mapping). None on any failure.
    Costs ~1 ms: one mmap + `samples` strided mincore calls, no page touches.
    """
    import ctypes
    import ctypes.util
    import mmap as _mmap

    try:
        libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
        libc.mmap.restype = ctypes.c_void_p
        libc.mmap.argtypes = [
            ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int64,
        ]
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return None
    try:
        size = os.fstat(fd).st_size
        if size == 0:
            return None
        pg = _mmap.PAGESIZE
        npages = (size + pg - 1) // pg
        prot_read, map_shared = 1, 1
        addr = libc.mmap(None, size, prot_read, map_shared, fd, 0)
        if addr in (None, ctypes.c_void_p(-1).value):
            return None
        try:
            step = max(1, npages // samples)
            vec = ctypes.create_string_buffer(1)
            hit = tot = 0
            for p in range(0, npages, step):
                if libc.mincore(
                    ctypes.c_void_p(addr + p * pg), ctypes.c_size_t(pg), vec
                ) != 0:
                    return None
                hit += vec.raw[0] & 1
                tot += 1
            return hit / tot
        finally:
            libc.munmap(ctypes.c_void_p(addr), ctypes.c_size_t(size))
    except Exception:
        return None
    finally:
        os.close(fd)


def _reader(st: _Stream) -> None:
    try:
        fd = os.open(st.path, os.O_RDONLY)
    except OSError:
        return
    try:
        buf = bytearray(_POPULATE_CHUNK)
        mv = memoryview(buf)
        while True:
            with _lock:
                off = st.next
                if off >= st.size:
                    return
                st.next = off + _POPULATE_CHUNK
            n = min(_POPULATE_CHUNK, st.size - off)
            r = os.preadv(fd, [mv[:n]], off)
            if r <= 0:
                return
            with _lock:
                st.done += r
    except OSError:
        pass
    finally:
        os.close(fd)


def start_populate(paths: list[str], log=print) -> None:
    """Stream GGUF shard(s) into the page cache on daemon reader threads.

    Readers claim 32 MB runs off one file-order frontier per shard, so
    coverage always grows front-to-back (~forward-pass order). A live
    stream is deduped (the phase-7 residency-warm re-kick is a no-op while
    the preflight kick is in flight); a finished one is retriable, so a
    later load in the same process re-streams a file whose pages were
    evicted in between (served: TTL unload, neighbor loads evict, then a
    re-request). Already-resident shards are skipped (a warm re-read is a
    pointless pass of buffer copies; this also covers a just-finished
    stream). Guaranteed reads, unlike F_RDADVISE; daemon threads own the
    fds. Registry insert and thread start share one lock hold so a
    concurrent kick or wait never sees a stream without live readers.
    """
    started = 0
    with _lock:
        for path in paths:
            key = os.path.realpath(path)
            st = _streams.get(key)
            if st is not None and any(t.is_alive() for t in st.threads):
                continue
            if (resident_fraction(path) or 0.0) >= _POPULATE_SKIP_RESIDENT:
                continue
            try:
                st = _Stream(path, os.path.getsize(path))
            except OSError:
                continue
            st.threads = [
                threading.Thread(target=_reader, args=(st,),
                                 name="gmlx-populate", daemon=True)
                for _ in range(_POPULATE_WORKERS)
            ]
            _streams[key] = st
            for t in st.threads:
                t.start()
            started += 1
    if started:
        log(f"[load_weights] page-cache populate started ({started} file(s), "
            f"{_POPULATE_WORKERS} readers)")


def wait_for(paths: list[str], log=print) -> None:
    """Block until in-flight populates of ``paths`` are nearly done.

    The first prefill's GPU-wiring faults are scattered reads; racing them
    against the stream collapses SSD throughput for both (measured: cold
    TTFT lands at the fully-sequential floor, as if there were no overlap).
    The file must be read once either way, so waiting is never slower. The
    release happens when the un-read tail drops below a fraction of the
    file (default 0.25): coverage grows in file order = forward order, so
    prefill starting at layer 0 trails the frontier and the tail finishes
    streaming under prefill's compute. No-op when nothing is in flight
    (warm loads, streaming models). GMLX_POPULATE_WAIT=0 disables,
    GMLX_POPULATE_WAIT_TAIL overrides the release fraction (0 = wait
    for every byte).
    """
    if not env_bool("GMLX_POPULATE_WAIT", True):
        return
    tail = env_float("GMLX_POPULATE_WAIT_TAIL", 0.25)
    with _lock:
        pending = [
            st for p in paths
            if (st := _streams.get(os.path.realpath(p))) is not None
        ]
    t0 = time.perf_counter()
    waited = False
    # A reader wedged on a dying disk/network volume would spin this loop
    # forever. Bail (log, don't raise: the barrier is perf-only and the daemon
    # readers keep streaming) once the unread tail makes no forward progress for
    # this long - not on a fixed wall-clock, so a merely-slow large load on slow
    # media is never aborted. <=0 disables the timeout (wait indefinitely).
    stall_s = env_float("GMLX_POPULATE_STALL_S", 60.0)
    for st in pending:
        last_remaining = None
        last_progress = time.perf_counter()
        while any(t.is_alive() for t in st.threads):
            with _lock:
                remaining = st.size - st.done
            if remaining <= tail * st.size:
                break
            now = time.perf_counter()
            if last_remaining is None or remaining < last_remaining:
                last_remaining, last_progress = remaining, now
            elif stall_s > 0 and now - last_progress > stall_s:
                log(f"[load_weights] populate stalled ({remaining / 1e9:.1f}GB "
                    f"unread, no progress for {stall_s:.0f}s); continuing")
                break
            waited = True
            time.sleep(0.05)
    if waited:
        log(f"[load_weights] waited {time.perf_counter() - t0:.1f}s for "
            f"page-cache populate")


def maybe_populate_for_load(paths: list[str], log=print) -> None:
    """Kick off populate for an in-budget load; no-op for streaming models."""
    if os.environ.get("GMLX_RESIDENCY_WARM", "") == "0":
        return
    if not env_bool("GMLX_POPULATE_EARLY", True):
        return
    try:
        total = sum(os.path.getsize(p) for p in paths)
        import mlx.core as mx

        budget = int(0.9 * mx.device_info()["max_recommended_working_set_size"])
    except Exception:
        return
    if not paths or total > budget:
        return
    start_populate(paths, log=log)
