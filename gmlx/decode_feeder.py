"""Decode feeder: GPU-resident expert arena for over-RAM MoE decode.

Streaming decode today runs the routed experts on the CPU stream against
mmap-backed weights: every miss demand-faults from disk and even a fully
cached expert is read at CPU-gemv bandwidth, several times slower than the
GPU reads the same bytes. This module keeps the *popular subset* of each
layer's expert stacks in per-layer wired arenas (``mlx_kquant.arena_alloc``:
page-aligned host memory, zero-copy visible to Metal) and runs decode-sized
expert calls on the GPU stream from the arena, remapping the router's expert
ids to arena slot ids on the host. Misses are pread straight from the GGUF
into an evicted slot - one trip, at SSD queue depth, no page-cache copy.

Synchronization is host-side only, and it is inherited rather than added:
the offload wrapper already evaluates the router (``mx.eval(indices)``) for
every decode-sized call, and layer L's router depends on layer L-1's expert
output, which depends on everything before it. So at the moment layer L is
being staged for token t, the most recent gather that referenced layer L's
arena - token t-1's - has already completed. Overwriting *any* slot of layer
L's arena at that point is safe, which is why there is no staging ring here:
eviction, adoption and staging are the same operation. The replacement
policy (least-popular first, never a slot the current call routes to) is the
popularity-based residency manager from the feeder design; the arena starts
empty and self-organizes toward the workload's hot set.

Enable with ``GMLX_FEEDER_DECODE=1`` (``--stream-experts`` models only - the
every-token layers must be on the GPU). Arena size defaults to what the wired
budget leaves after the non-expert weights and a KV reserve;
``GMLX_DECODE_ARENA_GB`` overrides. The
arena also answers system memory pressure arriving after load by shrinking
(and later regrowing) itself - see the pressure constants below. Miss reads
are joined with a timeout: a read wedged in the kernel is contained (slot
quarantined, expert dropped) rather than waited on - see the wedge
constants below.
"""

from __future__ import annotations

import fcntl
import mmap
import os
import queue
import threading
import time
from concurrent.futures import Future, wait as futures_wait
from contextlib import contextmanager
from functools import lru_cache

import numpy as np

from . import loadlog
from .envflags import env_int
from .feeder_common import ATTRS, KINDS, read_range, swapped_weights, verify_zero_copy

# Miss-pull parallelism: per layer, up to top_k experts x 3 stacks of
# MB-scale slices; enough in-flight preads for SSD sequential bandwidth.
_READ_WORKERS = 12

# Lookahead prestage (see ``prestage``): predictions from the lookahead
# hook (gmlx.lookahead) pre-read the next MoE layer's likely misses while
# the current layer computes. Capped at GMLX_DECODE_LOOKAHEAD_K ranked predictions
# per call (the ranking head is far more reliable than its tail - wasted
# reads scale with the cap and the ~25% mispredict rate) on a small
# dedicated pool with its own bounce buffers, so speculation can never
# starve demand misses of workers or buffers.
_LA_WORKERS = 6
_LA_K = 6

# Aligned miss reads. The kernel treats an F_NOCACHE read whose file
# offset is not page-aligned as misaligned and services the WHOLE request
# through the page-cache advisory-read machinery (page in, copy out, toss)
# instead of direct I/O - and that path's uninterruptible busy-page wait
# has been observed (rarely, on a file that is simultaneously mmapped and
# GPU-wired, under memory pressure) to never wake, wedging the read thread
# for good. Expert strides are never page multiples, so every raw miss
# read is misaligned. Rounding each read outward to page boundaries into a
# per-worker bounce buffer takes the true direct-I/O path; the memcpy into
# the slot is noise next to the SSD read. GMLX_DECODE_ALIGNED_READS=0
# restores raw preads.
_PAGE = mmap.PAGESIZE

# Timed miss-read join, the backstop behind aligned reads. A read that
# outlives GMLX_DECODE_READ_TIMEOUT seconds (default below, 0 disables the
# timer; a healthy NVMe miss read is single-digit ms) is treated as wedged
# in the kernel: its slot is quarantined, its expert is dropped from
# routing (see _quarantine), and decode continues. Re-reading the range
# from ANY path - pread, mmap fault, advisory - would wedge the caller too,
# so the drop is permanent for the process. After _MAX_WEDGES the file's
# cache state is presumed poisoned and miss staging stops entirely.
_READ_TIMEOUT_S = 5.0
_MAX_WEDGES = 3

# Popularity decay: halve every N stage calls (a single global counter across all
# covered layers, not per layer), so residency tracks the recent window instead of
# fossilizing the first prompt's routing.
_DECAY_EVERY = 4096

# Deliberately absent: a background seeder that pre-fills empty slots. The
# SSD is saturated by demand misses during exactly the window seeding would
# help, and a demand miss is a perfectly targeted read while a seed is a
# popularity guess - measured net-negative (seeding cost more decode
# throughput than its hit-rate gain returned). Organic fill is the seeder.

# System memory pressure: the arena is wired, so the kernel can never
# reclaim it - pressure arriving after load (another model, a build) must
# be answered by the feeder itself. Every _PRESSURE_POLL_EVERY stage calls
# the kernel's pressure level is read; warning steps the arena target down
# by _PRESSURE_STEP_FRAC of its sized capacity (critical takes two steps),
# floored at 1 - _PRESSURE_MAX_STEPS * _PRESSURE_STEP_FRAC of it, with
# _PRESSURE_COOLDOWN_POLLS between steps so one bad stretch doesn't drain
# the arena. Each layer re-allocates toward the target at its own stage()
# call - the same eval fence that lets stage() overwrite slots lets it
# replace the whole buffer - keeping its most popular residents, so a
# shrink costs the cold tail, never the hot set. _REGROW_NORMAL_POLLS of
# sustained normal pressure plus measured reclaimable-RAM headroom regrow
# one step at a time. GMLX_DECODE_PRESSURE=0 disables.
_PRESSURE_POLL_EVERY = 64
_PRESSURE_STEP_FRAC = 0.25
_PRESSURE_MAX_STEPS = 3
_PRESSURE_COOLDOWN_POLLS = 8
_REGROW_NORMAL_POLLS = 256


@lru_cache(maxsize=1)
def _libc():
    import ctypes

    try:
        return ctypes.CDLL(None, use_errno=True)
    except OSError:
        return None


def _pressure_level() -> int:
    """``kern.memorystatus_vm_pressure_level``: 1 normal, 2 warning,
    4 critical. Reads as normal wherever the sysctl is missing."""
    import ctypes

    libc = _libc()
    if libc is None:
        return 1
    try:
        val = ctypes.c_int(0)
        sz = ctypes.c_size_t(ctypes.sizeof(val))
        rc = libc.sysctlbyname(
            b"kern.memorystatus_vm_pressure_level",
            ctypes.byref(val), ctypes.byref(sz), None, 0,
        )
    except AttributeError:
        return 1
    return int(val.value) if rc == 0 else 1


class _DaemonReadPool:
    """ThreadPoolExecutor stand-in whose workers are daemon threads. The
    stdlib pool's workers are joined at interpreter exit, so one read
    wedged in the kernel would hang process shutdown; daemon workers die
    with the process instead. ``on_start`` runs once in each worker thread
    (the lookahead pool drops its disk-I/O priority there)."""

    def __init__(self, n: int, on_start=None):
        self._q: queue.Queue = queue.Queue()
        self._on_start = on_start
        self._threads = []
        for i in range(n):
            t = threading.Thread(
                target=self._run, daemon=True, name=f"gmlx-decode-read-{i}")
            t.start()
            self._threads.append(t)

    def _run(self) -> None:
        if self._on_start is not None:
            try:
                self._on_start()
            except Exception:
                pass
        while True:
            item = self._q.get()
            if item is None:
                return
            fut, fn, args = item
            if not fut.set_running_or_notify_cancel():
                continue
            try:
                fut.set_result(fn(*args))
            except BaseException as exc:
                fut.set_exception(exc)

    def submit(self, fn, *args) -> Future:
        fut: Future = Future()
        self._q.put((fut, fn, args))
        return fut

    def grow(self) -> None:
        """Replace a worker lost to a wedged read."""
        t = threading.Thread(
            target=self._run, daemon=True,
            name=f"gmlx-decode-read-{len(self._threads)}")
        t.start()
        self._threads.append(t)

    def shutdown(self, wait: bool = True) -> None:
        for _ in self._threads:
            self._q.put(None)
        if wait:
            for t in self._threads:
                t.join()


def _iopol_utility() -> None:
    """Drop the calling thread's disk-I/O priority (Darwin UTILITY tier),
    so the kernel schedules speculative lookahead reads behind demand
    misses. Colibri gets the same effect structurally from its single
    QD-1 pilot thread."""
    import ctypes

    libc = ctypes.CDLL(None, use_errno=True)
    # setiopolicy_np(IOPOL_TYPE_DISK=0, IOPOL_SCOPE_THREAD=1, IOPOL_UTILITY=4)
    libc.setiopolicy_np(0, 1, 4)


class DecodeFeeder:
    """Per-layer wired expert arenas with popularity-driven replacement."""

    def __init__(
        self,
        offsets: dict[int, list[tuple[str, int, int, int, str]]],
        modules: dict[int, list],
        arena_bytes: int,
    ):
        import mlx_kquant as kq

        # Construction happens inside the load session; capture its verbosity
        # for the end-of-run stat prints in close() (the session is gone then).
        self._stats_verbose = loadlog.is_verbose()
        # li -> kind -> (module, path, file_off, stride, shape)
        self._layers: dict[int, dict] = {}
        per_expert: dict[int, int] = {}  # li -> bytes per expert, all kinds
        n_experts: dict[int, int] = {}
        for li, ranges in offsets.items():
            kinds = {r[4] for r in ranges}
            if kinds != set(KINDS) or len(ranges) != len(KINDS):
                continue
            entry = {}
            total = 0
            n_exp0 = None
            for path, off, nbytes, n_exp, kind in ranges:
                if not n_exp or (n_exp0 is not None and n_exp != n_exp0):
                    break
                n_exp0 = n_exp
                mod = None
                for m in modules.get(li, ()):
                    proj = getattr(m, ATTRS[kind], None)
                    w = getattr(proj, "weight", None)
                    if (
                        w is not None
                        and w.nbytes == nbytes
                        and w.ndim == 3
                        and w.shape[0] == n_exp
                        and getattr(proj, "bias", None) is None
                    ):
                        mod = m
                        break
                if mod is None:
                    break
                stride = nbytes // n_exp
                entry[kind] = (mod, path, off, stride, tuple(w.shape))
                total += stride
            if len(entry) == len(KINDS):
                self._layers[li] = entry
                per_expert[li] = total
                n_experts[li] = n_exp0
        if not self._layers:
            raise RuntimeError("no layers with matching gate/up/down stacks")

        # Even per-layer budget split; ``slots`` experts resident per layer.
        per_layer_budget = arena_bytes // len(self._layers)
        self._slots: dict[int, int] = {}
        for li in list(self._layers):
            s = min(per_layer_budget // per_expert[li], n_experts[li])
            if s < 1:
                del self._layers[li]
                continue
            self._slots[li] = int(s)
        if not self._layers:
            raise RuntimeError(
                f"arena budget ({arena_bytes / 1e9:.1f} GB) fits no experts")

        # Miss reads bypass the page cache (F_NOCACHE): each byte is read
        # exactly once into the arena, and letting those reads populate the
        # cache makes the kernel page the arena's cold slots out to swap to
        # hold them - a hit on a swapped slot then faults from swap, which
        # is strictly worse than reading the GGUF. GMLX_DECODE_NOCACHE=0
        # restores cached reads (page-cache-assisted misses, for boxes with
        # RAM to spare).
        nocache = os.environ.get("GMLX_DECODE_NOCACHE", "1") != "0"
        self._fds: dict[str, int] = {}
        try:
            for entry in self._layers.values():
                for _, path, *_ in entry.values():
                    if path not in self._fds:
                        fd = os.open(path, os.O_RDONLY)
                        self._fds[path] = fd
                        if nocache and hasattr(fcntl, "F_NOCACHE"):
                            fcntl.fcntl(fd, fcntl.F_NOCACHE, 1)
            self._verify_zero_copy()
        except BaseException:
            self.close()
            raise

        # kq.arena_alloc keeps (array, writable memoryview) pairs alive
        # together; ``views`` reshapes the array to the gather's expectation.
        self._arena: dict[tuple[int, str], tuple] = {}
        self._views: dict[tuple[int, str], object] = {}
        self.arena_bytes = 0
        for li, entry in self._layers.items():
            s = self._slots[li]
            for kind, (_, _, _, stride, shape) in entry.items():
                a = kq.arena_alloc([s * stride])
                self._arena[(li, kind)] = a
                self._views[(li, kind)] = a[0].reshape((s,) + shape[1:])
                self.arena_bytes += s * stride
        self._locked: dict[tuple[int, str], tuple[int, int]] = {}
        self.locked_bytes = 0
        self._mlock_arena()

        # Aligned-read plumbing (see _PAGE above): per-worker bounce
        # buffers big enough for the largest expert plus the alignment
        # slack on both ends.
        self._aligned = os.environ.get("GMLX_DECODE_ALIGNED_READS", "1") != "0"
        self._sizes = {p: os.fstat(fd).st_size for p, fd in self._fds.items()}
        if self._aligned:
            max_stride = max(
                stride for entry in self._layers.values()
                for (_, _, _, stride, _) in entry.values())
            self._bounce_bytes = max_stride + 2 * _PAGE
            self._bounce: queue.Queue = queue.Queue()
            for _ in range(_READ_WORKERS):
                self._bounce.put(kq.arena_alloc([self._bounce_bytes]))

        # Residency state per layer. slot_of[e] is e's arena slot or -1;
        # owner[s] is slot s's expert or -1 (empty); counts is the decayed
        # routing popularity that picks eviction victims.
        self._slot_of = {
            li: np.full(n_experts[li], -1, dtype=np.int32) for li in self._layers
        }
        self._owner = {
            li: np.full(self._slots[li], -1, dtype=np.int32) for li in self._layers
        }
        self._counts = {
            li: np.zeros(n_experts[li], dtype=np.float64) for li in self._layers
        }
        self._calls = 0
        self._hits = 0
        self._lookups = 0
        self._layer_hits = {li: 0 for li in self._layers}
        self._layer_lookups = {li: 0 for li in self._layers}
        self._read_pool = _DaemonReadPool(_READ_WORKERS)

        # Lookahead prestage state (constants above; pool and bounce
        # buffers are lazy - most runs never prestage). _pending maps
        # li -> {expert: (slot, futures, submit time)}; a pending slot's
        # owner is -4 (reserved: invisible to the empty scan, the victim
        # scan, and the resize keep-set) until its read completes and the
        # main thread publishes it. Workers only ever write slot bytes;
        # all residency metadata stays main-thread.
        self._pending: dict[int, dict[int, tuple[int, list, float]]] = {}
        self._la_pool: _DaemonReadPool | None = None
        self._la_bounce: queue.Queue | None = None
        self._la_k = env_int("GMLX_DECODE_LOOKAHEAD_K", _LA_K)
        self._la_cancel = (
            os.environ.get("GMLX_DECODE_LOOKAHEAD_CANCEL", "1") != "0")
        self._la_submitted = 0
        self._la_adopted = 0
        self._la_waited = 0
        self._la_wasted = 0
        self._la_cancelled = 0

        # Stall accounting: how much wall time stage() spends blocked on
        # demand miss reads vs settling speculative reads. The demand
        # share is the ceiling any prefetch scheme can reclaim.
        self._t_demand = 0.0
        self._t_settle = 0.0
        self._t_start = time.monotonic()

        # GMLX_DECODE_FEEDER_VERIFY=1: sample-compare arena slots against
        # their file bytes at every publish and every routed use, and
        # re-check the previous call's routed slots on the next call (a
        # late overwrite during the gather window shows up there). Debug
        # instrument for corruption hunts; costs a few thousand small
        # preads per token.
        self._verify = os.environ.get("GMLX_DECODE_FEEDER_VERIFY", "0") == "1"
        self._verify_prev: dict[int, list[tuple[int, int]]] = {}

        # Wedge containment state (constants above). Slot owner sentinels:
        # -1 empty, -2 quarantined (never reused - the zombie read may
        # still write into it), -3 the layer's zero slot (dead experts map
        # here; zero bytes dequant to zero weights in every codec, so the
        # gather contributes exactly nothing for them).
        self._read_timeout = float(
            os.environ.get("GMLX_DECODE_READ_TIMEOUT", str(_READ_TIMEOUT_S)))
        self._dead: dict[int, np.ndarray] = {}
        self._zslot: dict[int, int] = {}
        self._wedged_layers: set[int] = set()
        self._wedges = 0
        self._staging_disabled = False

        # Pressure adaptation state (constants above). ``_slots`` tracks the
        # live per-layer size; ``_orig_slots`` is the sized capacity targets
        # are computed against.
        self._per_expert = {li: per_expert[li] for li in self._layers}
        self._orig_slots = dict(self._slots)
        self._pressure_on = os.environ.get("GMLX_DECODE_PRESSURE", "1") != "0"
        self._pressure_steps = 0
        self._pressure_polls = 0
        self._last_step_poll = -_PRESSURE_COOLDOWN_POLLS
        self._normal_polls = 0

    def _mlock_arena(self) -> None:
        """Wire the arena. The residency policy only works if the slots
        actually stay in RAM: unlocked, the kernel pages the cold tail out
        to swap under exactly the pressure a larger-than-RAM model creates,
        and an arena hit that faults from swap is slower than the GGUF read
        it was supposed to save. Locks buffer by buffer and stops at the
        first refusal (wire limit), leaving the rest kernel-managed;
        GMLX_DECODE_ARENA_MLOCK=0 disables. Partial wiring is announced:
        an arena that silently runs half-unwired shows up only as
        inexplicably slow decode, which is not a debuggable symptom."""
        for key in self._arena:
            if not self._mlock_buf(key):
                if os.environ.get("GMLX_DECODE_ARENA_MLOCK", "1") != "0":
                    print(
                        f"[stream] decode feeder: arena wiring stopped at "
                        f"{self.locked_bytes / 1e9:.1f} of "
                        f"{self.arena_bytes / 1e9:.1f} GB (mlock refused); "
                        "the unwired tail is kernel-managed and may page "
                        "under pressure")
                break

    def _mlock_buf(self, key: tuple[int, str]) -> bool:
        if os.environ.get("GMLX_DECODE_ARENA_MLOCK", "1") == "0":
            return False
        import ctypes

        libc = _libc()
        if libc is None:
            return False
        mv = self._arena[key][1]
        n = len(mv)
        try:
            addr = ctypes.addressof(ctypes.c_char.from_buffer(mv))
        except (TypeError, ValueError):
            return False
        if libc.mlock(ctypes.c_void_p(addr), ctypes.c_size_t(n)) != 0:
            return False
        self._locked[key] = (addr, n)
        self.locked_bytes += n
        return True

    def _munlock_buf(self, key: tuple[int, str]) -> None:
        entry = self._locked.pop(key, None)
        if entry is None:
            return
        import ctypes

        libc = _libc()
        if libc is not None:
            addr, n = entry
            libc.munlock(ctypes.c_void_p(addr), ctypes.c_size_t(n))
        self.locked_bytes -= entry[1]

    def _verify_zero_copy(self) -> None:
        li = min(self._layers)
        verify_zero_copy(
            li, ((k, mod, path, off)
                 for k, (mod, path, off, _, _) in self._layers[li].items()),
            self._fds)

    def covers(self, li: int) -> bool:
        return li in self._layers

    # staging

    def _read_expert(self, li: int, kind: str, e: int, slot: int,
                     bounce: queue.Queue | None = None) -> None:
        _, path, off, stride, _ = self._layers[li][kind]
        mv = self._arena[(li, kind)][1]
        dest = mv[slot * stride:(slot + 1) * stride]
        off_e = off + e * stride
        if not self._aligned:
            read_range(self._fds[path], dest, off_e)
            return
        a = off_e & ~(_PAGE - 1)
        b = min(
            (off_e + stride + _PAGE - 1) & ~(_PAGE - 1),
            self._sizes[path])
        pool = bounce if bounce is not None else self._bounce
        buf = pool.get()
        try:
            bmv = buf[1][: b - a]
            read_range(self._fds[path], bmv, a)
            dest[:] = bmv[off_e - a: off_e - a + stride]
        finally:
            pool.put(buf)

    def _verify_slot(self, li: int, e: int, s: int, when: str) -> None:
        """Compare head/mid/tail samples of slot ``s`` against expert
        ``e``'s file bytes; any mismatch is corruption - dump enough state
        to attribute it and raise."""
        for kind, (_, path, off, stride, _) in self._layers[li].items():
            mv = self._arena[(li, kind)][1]
            base = s * stride
            fbase = off + e * stride
            n = min(4096, stride)
            for o in {0, max(0, stride // 2 - n // 2), stride - n}:
                got = bytes(mv[base + o: base + o + n])
                want = os.pread(self._fds[path], n, fbase + o)
                if got != want:
                    pend = {
                        pe: ps for pe, (ps, _, _) in
                        self._pending.get(li, {}).items()}
                    raise RuntimeError(
                        f"[verify] arena bytes wrong at {when}: layer {li} "
                        f"expert {e} slot {s} kind {kind} sample_off {o} "
                        f"owner={int(self._owner[li][s])} pending={pend}")

    def stage(self, li: int, ids: np.ndarray) -> np.ndarray | None:
        """Map router expert ids to arena slots, pulling misses from the GGUF
        into evicted slots first. Returns the slot array (``ids``' shape,
        uint32), or None when the call cannot be served from the arena:
        more distinct experts than slots, staging disabled after repeated
        wedges, or a wedge the layer had no spare slot to contain. On None
        the caller falls back to the CPU path - with ids rewritten through
        ``redirect_dead`` when ``has_dead`` says the layer lost experts.

        Caller contract: ``mx.eval`` of the call's ``indices`` has run, so no
        in-flight gather references this layer's arena (see module docstring).
        """
        if self._pressure_on and self._calls % _PRESSURE_POLL_EVERY == 0:
            self._poll_pressure()
        uniq = np.unique(ids.reshape(-1))
        if self._pending.get(li):
            # Serve-time barrier: every speculative read for this layer
            # lands (or is quarantined) before any residency decision or
            # gather references it. From here down the layer is exactly a
            # demand-only layer.
            self._settle_pending(li, uniq)
            if li not in self._layers:
                return None  # a settle-time wedge took the layer out
        target = self._target_slots(li)
        if (
            target != self._slots[li]
            and li not in self._wedged_layers
            and not self._pending.get(li)
        ):
            # A wedged layer never resizes: reallocating would free the
            # buffer the zombie read may still write into. (The pending
            # guard is belt-and-braces: settle just drained the layer.)
            self._resize_layer(li, target)
        slot_of = self._slot_of[li]
        counts = self._counts[li]
        if self._verify:
            owner_v = self._owner[li]
            for pe, ps in self._verify_prev.get(li, ()):
                if ps < len(owner_v) and owner_v[ps] == pe:
                    self._verify_slot(li, pe, ps, "prev-routed")
        counts[uniq] += 1.0
        self._calls += 1
        if self._calls % _DECAY_EVERY == 0:
            for c in self._counts.values():
                c *= 0.5
        missing = uniq[slot_of[uniq] < 0]
        self._lookups += len(uniq)
        self._hits += len(uniq) - len(missing)
        self._layer_lookups[li] += len(uniq)
        self._layer_hits[li] += len(uniq) - len(missing)
        if len(missing):
            if self._staging_disabled:
                return None
            owner = self._owner[li]
            empty = np.flatnonzero(owner == -1)
            victims = list(empty[: len(missing)])
            need = len(missing) - len(victims)
            current = np.zeros(len(owner), dtype=bool)
            routed = slot_of[uniq]
            current[routed[routed >= 0]] = True
            if need > 0:
                # Evict the least-popular residents, never one the current
                # call routes to.
                cand = np.flatnonzero((owner >= 0) & ~current)
                if len(cand) < need:
                    return None
                order = np.argsort(counts[owner[cand]], kind="stable")
                victims.extend(cand[order[:need]])
            staged = []
            futs = []
            for e, s in zip(missing, victims):
                staged.append((int(e), int(s), int(owner[s])))
                futs.append([
                    self._read_pool.submit(
                        self._read_expert, li, kind, int(e), int(s))
                    for kind in KINDS
                ])
            timeout = self._read_timeout if self._read_timeout > 0 else None
            t0 = time.monotonic()
            futures_wait([f for fs in futs for f in fs], timeout=timeout)
            self._t_demand += time.monotonic() - t0
            err = None
            lost = False
            for (e, s, old), fs in zip(staged, futs):
                pend = sum(not f.done() for f in fs)
                if pend:
                    # Read outlived the timeout: wedged in the kernel.
                    lost |= not self._quarantine(li, e, s, old, current, pend)
                    continue
                exc = None
                for f in fs:
                    exc = exc or f.exception()
                if exc is not None:
                    # The victim slot may hold partial bytes: evict its old
                    # owner and leave the miss non-resident so a later call
                    # re-reads instead of serving a poisoned slot.
                    if old >= 0:
                        slot_of[old] = -1
                    owner[s] = -1
                    err = err or exc
                    continue
                if old >= 0:
                    slot_of[old] = -1
                owner[s] = e
                slot_of[e] = s
                if self._verify:
                    self._verify_slot(li, e, s, "publish-demand")
            if err is not None:
                raise err
            if lost:
                # A dead expert could not be zero-mapped: the layer is out
                # of service (covers() is now False). The caller must route
                # its fallback through redirect_dead-rewritten ids.
                return None
        if self._verify:
            owner_v = self._owner[li]
            # Zero-slot remaps (owner -3) hold zeros on purpose: verify
            # only slots genuinely owned by the routed expert.
            live = [
                (int(e), int(slot_of[e])) for e in uniq
                if slot_of[e] >= 0 and owner_v[slot_of[e]] == e]
            pend_slots = {
                ps for ps, _, _ in self._pending.get(li, {}).values()}
            bad = pend_slots.intersection(s for _, s in live)
            if bad:
                raise RuntimeError(
                    f"[verify] pending slot in routed set: layer {li} "
                    f"slots {sorted(bad)}")
            for e, s in live:
                self._verify_slot(li, e, s, "routed")
            self._verify_prev[li] = live
        return slot_of[ids].astype(np.uint32)

    # lookahead prestage

    def _la_state(self) -> _DaemonReadPool:
        if self._la_pool is None:
            import mlx_kquant as kq

            n = env_int("GMLX_DECODE_LOOKAHEAD_WORKERS", _LA_WORKERS)
            on_start = (
                _iopol_utility
                if os.environ.get("GMLX_DECODE_LOOKAHEAD_IOPOL", "1") != "0"
                else None)
            self._la_pool = _DaemonReadPool(n, on_start=on_start)
            if self._aligned:
                self._la_bounce = queue.Queue()
                for _ in range(n):
                    self._la_bounce.put(kq.arena_alloc([self._bounce_bytes]))
        return self._la_pool

    def prestage(self, li: int, pred_ids: np.ndarray) -> None:
        """Asynchronously pre-read predicted experts into layer ``li``'s
        arena. Safe under the same fence as ``stage`` (the caller's router
        eval for the *previous* MoE layer of the same token transitively
        fenced every gather that could reference this arena). Slots are
        reserved (-4) at submission and published only when their read has
        completed, so a concurrent lookup can never map an expert to a
        slot holding partial bytes. Predictions never touch popularity."""
        if (
            li not in self._layers
            or self._staging_disabled
            or li in self._wedged_layers
        ):
            return
        self._flush_pending(li)
        if li in self._wedged_layers or li not in self._layers:
            return  # the flush itself may have wedged the layer
        pending = self._pending.setdefault(li, {})
        slot_of = self._slot_of[li]
        owner = self._owner[li]
        counts = self._counts[li]
        dead = self._dead.get(li)
        # Only the top GMLX_DECODE_LOOKAHEAD_K ranks per row are considered (the
        # ranking head is reliable, the tail is not); residents among them
        # are simply already-good news, not license to dig deeper.
        rows = pred_ids.reshape(-1, pred_ids.shape[-1])[:, : self._la_k]
        picked: list[int] = []
        seen: set[int] = set()
        protected = np.zeros(len(owner), dtype=bool)
        for e in rows.T.reshape(-1):  # rank-major: every row's head first
            e = int(e)
            if e in seen:
                continue
            seen.add(e)
            if slot_of[e] >= 0:
                # A predicted expert that is already resident is the best
                # slot in the arena: shield it from this call's own victim
                # scan (a lower-ranked prediction must never evict it).
                protected[slot_of[e]] = True
                continue
            if e in pending or (dead is not None and dead[e]):
                continue
            picked.append(e)
        if not picked:
            return
        pool = self._la_state()
        empty = list(np.flatnonzero(owner == -1))
        for e in picked:
            if empty:
                s = int(empty.pop())
                old = -1
            else:
                # Evict only a resident no more popular than the
                # prediction: a wrong guess then costs a cold slot, never
                # a hot one - and never a slot this call's own higher
                # ranks predicted.
                cand = np.flatnonzero((owner >= 0) & ~protected)
                if not len(cand):
                    break
                v = int(np.argmin(counts[owner[cand]]))
                if counts[owner[cand]][v] > counts[e]:
                    continue
                s = int(cand[v])
                old = int(owner[s])
            if old >= 0:
                slot_of[old] = -1
            owner[s] = -4
            futs = [
                pool.submit(self._read_expert, li, kind, e, s, self._la_bounce)
                for kind in KINDS
            ]
            pending[e] = (s, futs, time.monotonic())
            self._la_submitted += 1

    def _flush_pending(self, li: int) -> None:
        """Publish completed prestage reads; quarantine ones that outlived
        the read timeout (same wedge semantics as demand misses - the file
        range is poisoned whether or not the prediction was right)."""
        pending = self._pending.get(li)
        if not pending:
            return
        owner = self._owner[li]
        now = time.monotonic()
        for e in list(pending):
            s, futs, t0 = pending[e]
            if all(f.done() for f in futs):
                del pending[e]
                cancelled = [f.cancelled() for f in futs]
                if any(cancelled):
                    # Fully cancelled: no disk work happened. A mixed entry
                    # read some kinds before the cancel landed; its partial
                    # bytes are discarded with the slot either way.
                    owner[s] = -1
                    if all(cancelled):
                        self._la_cancelled += 1
                    else:
                        self._la_wasted += 1
                    continue
                exc = next(
                    (f.exception() for f in futs if f.exception()), None)
                if exc is not None:
                    owner[s] = -1  # partial bytes: slot back to empty
                    self._la_wasted += 1
                else:
                    owner[s] = e
                    self._slot_of[li][e] = s
                    if self._verify:
                        self._verify_slot(li, e, s, "publish-flush")
            elif (
                self._read_timeout > 0
                and now - t0 > self._read_timeout
            ):
                del pending[e]
                pend = sum(not f.done() for f in futs)
                self._quarantine(
                    li, e, s, -1,
                    np.zeros(len(owner), dtype=bool), pend, pool="la")

    def _settle_pending(self, li: int, uniq: np.ndarray) -> None:
        """Serve-time barrier, first step of ``stage``: join EVERY pending
        prestage for the layer - routed or not - then publish successes and
        quarantine timeouts. Waiting on the wasted tail too is what makes
        the invariant airtight: after settle, no speculative write can
        overlap the gather this call builds, and the demand path below runs
        against fully-published residency (colibri's pilot holds the same
        per-layer barrier). Routed entries drop out of the miss set instead
        of being read twice; the wasted tail was submitted a full compute
        window ago and is almost always already done."""
        pending = self._pending[li]
        routed = {int(e) for e in uniq}
        done_first = {
            e for e in pending if all(f.done() for f in pending[e][1])}
        if self._la_cancel:
            # A pending prediction the router did not route to is provably
            # useless: cancel whatever has not started instead of spending
            # disk bandwidth on it. (Cancellation outcome depends on read
            # timing, so residency is no longer a pure function of the
            # routing sequence - GMLX_DECODE_LOOKAHEAD_CANCEL=0 restores
            # that property for A/Bs. Bytes stay safe either way: a
            # cancelled entry's slot goes back to empty, never published.)
            for e, (_, futs, _) in pending.items():
                if e not in routed:
                    for f in futs:
                        f.cancel()
        timeout = self._read_timeout if self._read_timeout > 0 else None
        t0 = time.monotonic()
        futures_wait(
            [f for _, futs, _ in pending.values() for f in futs],
            timeout=timeout)
        self._t_settle += time.monotonic() - t0
        owner = self._owner[li]
        slot_of = self._slot_of[li]
        for e, (s, futs, _) in list(pending.items()):
            del pending[e]
            pend = sum(not f.done() for f in futs)
            if pend:
                current = np.zeros(len(owner), dtype=bool)
                rs = slot_of[uniq]
                current[rs[rs >= 0]] = True
                if not self._quarantine(
                        li, e, s, -1, current, pend, pool="la"):
                    return  # layer out of service; caller checks covers
                continue
            cancelled = [f.cancelled() for f in futs]
            if any(cancelled):
                owner[s] = -1  # bytes, if any kind ran, discarded with the slot
                if all(cancelled):
                    self._la_cancelled += 1
                else:
                    self._la_wasted += 1
                continue
            exc = next(
                (f.exception() for f in futs if f.exception()), None)
            if exc is not None:
                owner[s] = -1  # falls into missing: demand path re-reads
                self._la_wasted += 1
                continue
            owner[s] = e
            slot_of[e] = s
            if e in routed:
                if e in done_first:
                    self._la_adopted += 1
                else:
                    self._la_waited += 1
            if self._verify:
                self._verify_slot(li, e, s, "publish-settle")

    # wedge containment

    def _quarantine(self, li: int, e: int, s: int, old: int,
                    current: np.ndarray, pend: int,
                    pool: str = "demand") -> bool:
        """Contain a read wedged in the kernel: quarantine its slot (the
        zombie read may complete into it at any time, so it is never
        reused and its buffer never freed), drop expert ``e`` from routing
        for good, and remap it to the layer's zero slot so arena gathers
        contribute exactly nothing for it - the mass filter's drop
        semantic. Returns False when the layer had no slot left to zero:
        the layer is then removed from service entirely."""
        owner = self._owner[li]
        slot_of = self._slot_of[li]
        if old >= 0:
            slot_of[old] = -1
        owner[s] = -2
        dead = self._dead.setdefault(
            li, np.zeros(len(slot_of), dtype=bool))
        dead[e] = True
        self._wedged_layers.add(li)
        self._wedges += 1
        # A wedged worker never returns: replace the thread and the bounce
        # buffer pinned in its blocked frame (in whichever pool the read
        # ran - demand misses or lookahead prestage).
        rpool = self._la_pool if pool == "la" else self._read_pool
        rbounce = self._la_bounce if pool == "la" else getattr(
            self, "_bounce", None)
        for _ in range(pend):
            if rpool is not None:
                rpool.grow()
            if self._aligned and rbounce is not None:
                import mlx_kquant as kq

                rbounce.put(kq.arena_alloc([self._bounce_bytes]))
        print(
            f"[stream] decode feeder: layer {li} expert {e} read wedged "
            f"in the kernel (>{self._read_timeout:.0f}s); slot quarantined,"
            " expert dropped from routing")
        if self._wedges >= _MAX_WEDGES and not self._staging_disabled:
            self._staging_disabled = True
            print(
                "[stream] decode feeder: repeated wedged reads - miss "
                "staging disabled, arena serves residents only")
        zs = self._zero_slot(li, current)
        if zs < 0:
            del self._layers[li]
            return False
        slot_of[e] = zs
        return True

    def _zero_slot(self, li: int, current: np.ndarray) -> int:
        """The layer's all-zero slot, created on first need from an empty
        slot (else by evicting the least-popular resident the current call
        does not route to). -1 when no slot qualifies."""
        zs = self._zslot.get(li)
        if zs is not None:
            return zs
        owner = self._owner[li]
        empty = np.flatnonzero(owner == -1)
        if len(empty):
            s = int(empty[0])
        else:
            cand = np.flatnonzero((owner >= 0) & ~current)
            if not len(cand):
                return -1
            counts = self._counts[li]
            s = int(cand[np.argmin(counts[owner[cand]])])
            self._slot_of[li][owner[s]] = -1
        for kind, (_, _, _, stride, _) in self._layers[li].items():
            mv = self._arena[(li, kind)][1]
            np.frombuffer(mv, dtype=np.uint8)[
                s * stride:(s + 1) * stride] = 0
        owner[s] = -3
        self._zslot[li] = s
        return s

    def has_dead(self, li: int) -> bool:
        return li in self._dead

    def wedged_at(self, li: int) -> bool:
        return li in self._wedged_layers

    def redirect_dead(self, li: int, ids: np.ndarray) -> np.ndarray:
        """Rewrite router ids so no non-arena consumer (mmap gather,
        advisory prefetch, prefill staging) touches a dead expert's file
        range - the wedged page poisons it until reboot. Each dead id
        becomes its row's first surviving id: the gate weight for the slot
        is unchanged, a bounded distortion on a rare fallback path against
        a guaranteed hang."""
        dead = self._dead.get(li)
        if dead is None:
            return ids
        flat = ids.reshape(-1, ids.shape[-1])
        bad = dead[flat]
        if not bad.any():
            return ids
        alive = np.flatnonzero(~dead)
        fallback = int(alive[0]) if len(alive) else 0
        out = flat.copy()
        for r in np.flatnonzero(bad.any(axis=1)):
            live = out[r][~bad[r]]
            out[r][bad[r]] = live[0] if len(live) else fallback
        return out.reshape(ids.shape)

    # memory pressure

    def _target_slots(self, li: int) -> int:
        if not self._pressure_steps:
            return self._orig_slots[li]
        frac = 1.0 - self._pressure_steps * _PRESSURE_STEP_FRAC
        return max(1, int(self._orig_slots[li] * frac))

    def _arena_bytes_at(self, steps: int) -> int:
        frac = max(0.0, 1.0 - steps * _PRESSURE_STEP_FRAC)
        return sum(
            max(1, int(self._orig_slots[li] * frac)) * self._per_expert[li]
            for li in self._layers
        )

    def _poll_pressure(self) -> None:
        level = _pressure_level()
        self._pressure_polls += 1
        if level >= 2:
            self._normal_polls = 0
            if (
                self._pressure_steps < _PRESSURE_MAX_STEPS
                and self._pressure_polls - self._last_step_poll
                >= _PRESSURE_COOLDOWN_POLLS
            ):
                step = 2 if level >= 4 else 1
                self._pressure_steps = min(
                    self._pressure_steps + step, _PRESSURE_MAX_STEPS)
                self._last_step_poll = self._pressure_polls
                print(
                    f"[stream] memory pressure (level {level}): decode "
                    f"arena shrinking toward "
                    f"{self._arena_bytes_at(self._pressure_steps) / 1e9:.1f} GB"
                )
                self._clear_mlx_cache()
            return
        self._normal_polls += 1
        if (
            self._pressure_steps
            and self._normal_polls >= _REGROW_NORMAL_POLLS
            and self._regrow_headroom_ok()
        ):
            self._pressure_steps -= 1
            self._last_step_poll = self._pressure_polls
            self._normal_polls = 0
            print(
                f"[stream] memory pressure cleared: decode arena regrowing "
                f"toward {self._arena_bytes_at(self._pressure_steps) / 1e9:.1f} GB"
            )

    def _regrow_headroom_ok(self) -> bool:
        """Only regrow into RAM nobody has to swap for (free + speculative
        + purgeable; not inactive, which includes other processes' anon
        memory): pressure subsiding means the system recovered, not that
        the memory is ours to take back."""
        need = self._arena_bytes_at(self._pressure_steps - 1) - self.arena_bytes
        try:
            from .loader import _available_ram_bytes, _ram_floor_bytes

            avail = _available_ram_bytes(include_inactive=False)
        except Exception:
            return True
        if avail is None:
            return True
        ram = None
        try:
            import mlx.core as mx

            ram = int(mx.device_info()["memory_size"])
        except Exception:
            pass
        return avail >= need + _ram_floor_bytes(ram or avail)

    def _clear_mlx_cache(self) -> None:
        """The arena's own buffers bypass MLX's allocator, but its cache of
        freed GPU buffers is also memory the system could use right now."""
        try:
            import mlx.core as mx

            (getattr(mx, "clear_cache", None) or mx.metal.clear_cache)()
        except Exception:
            pass

    def _resize_layer(self, li: int, new_s: int) -> None:
        """Reallocate a layer's arena at ``new_s`` slots, carrying over its
        most popular residents. In-place release (munlock + madvise) is not
        an option: gathers reference the layer's whole Metal buffer every
        token, so its pages stay GPU-resident whatever madvise says - only
        dropping the buffer returns memory. Reallocating one layer at its
        own stage() call bounds the transient old+new overlap to a single
        layer's stacks, and the caller's eval fence guarantees no in-flight
        gather references this layer."""
        import mlx_kquant as kq

        old_s = self._slots[li]
        entry = self._layers[li]
        owner = self._owner[li]
        counts = self._counts[li]
        slot_of = self._slot_of[li]
        residents = np.flatnonzero(owner >= 0)
        keep = residents
        if len(residents) > new_s:
            order = np.argsort(-counts[owner[residents]], kind="stable")
            keep = residents[order[:new_s]]
        new_owner = np.full(new_s, -1, dtype=np.int32)
        slot_of[:] = -1
        for ns, os_ in enumerate(keep):
            e = owner[os_]
            new_owner[ns] = e
            slot_of[e] = ns
        for kind, (_, _, _, stride, shape) in entry.items():
            key = (li, kind)
            a = kq.arena_alloc([new_s * stride])
            mv_new, mv_old = a[1], self._arena[key][1]
            for ns, os_ in enumerate(keep):
                mv_new[ns * stride:(ns + 1) * stride] = \
                    mv_old[os_ * stride:(os_ + 1) * stride]
            self._munlock_buf(key)
            self._arena[key] = a
            self._views[key] = a[0].reshape((new_s,) + shape[1:])
            self.arena_bytes += (new_s - old_s) * stride
            self._mlock_buf(key)
        self._owner[li] = new_owner
        self._slots[li] = new_s

    @contextmanager
    def swapped(self, li: int):
        """Swap the layer's expert weights to the arena views for the call."""
        entry = self._layers[li]
        views = {kind: self._views[(li, kind)] for kind in entry}
        with swapped_weights(entry, views):
            yield

    def close(self) -> None:
        if getattr(self, "_closed", False):
            return
        self._closed = True
        if getattr(self, "_lookups", 0):
            print(
                f"[stream] decode feeder arena hit rate: "
                f"{self._hits}/{self._lookups} "
                f"({100 * self._hits / self._lookups:.1f}%), "
                f"avg {self._lookups / self._calls:.1f} unique "
                "experts/token-layer"
            )
            # The cross-layer spread is what decides whether a
            # popularity-weighted per-layer budget split would beat the
            # even split: a flat spread says no.
            if getattr(self, "_stats_verbose", False):
                rates = {
                    li: self._layer_hits[li] / n
                    for li, n in self._layer_lookups.items() if n
                }
                if len(rates) > 1:
                    vals = sorted(rates.values())
                    cold = sorted(rates, key=rates.get)[:3]
                    print(
                        "[stream] decode feeder per-layer hit rate: "
                        f"median {100 * vals[len(vals) // 2]:.1f}% / "
                        f"max {100 * vals[-1]:.1f}%; coldest: "
                        + ", ".join(
                            f"L{li} {100 * rates[li]:.1f}%" for li in cold))
            self._lookups = 0
        if getattr(self, "_la_submitted", 0):
            adopted = self._la_adopted + self._la_waited
            print(
                f"[stream] lookahead prestage: {self._la_submitted} "
                f"predicted expert reads, {adopted} adopted by routing "
                f"({self._la_adopted} already done, {self._la_waited} "
                f"joined), {self._la_cancelled} cancelled unstarted, "
                f"{self._la_wasted} discarded (failed or part-cancelled)"
            )
        if getattr(self, "_calls", 0) and getattr(self, "_stats_verbose", False):
            wall = time.monotonic() - self._t_start
            print(
                f"[stream] decode feeder stalls: demand reads "
                f"{self._t_demand:.1f}s, prestage settle "
                f"{self._t_settle:.1f}s, over {wall:.0f}s wall"
            )
        wedges = getattr(self, "_wedges", 0)
        if wedges:
            print(
                f"[stream] decode feeder: {wedges} read(s) wedged in the "
                "kernel this run; the affected experts were dropped")
        for pool in (getattr(self, "_read_pool", None),
                     getattr(self, "_la_pool", None)):
            if pool is not None:
                # A wedged worker never returns; joining it would hang exit.
                pool.shutdown(wait=wedges == 0)
        locked, self._locked = getattr(self, "_locked", {}), {}
        self.locked_bytes = 0
        if locked:
            import ctypes

            libc = _libc()
            if libc is not None:
                for addr, n in locked.values():
                    libc.munlock(ctypes.c_void_p(addr), ctypes.c_size_t(n))
        fds, self._fds = self._fds, {}
        for fd in fds.values():
            try:
                os.close(fd)
            except OSError:
                pass

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def _register_exit_close(feeder) -> None:
    """Close (and print stats for) a still-live feeder at interpreter exit.
    Held through a weakref: a strong reference would pin every unloaded
    feeder's arena for the life of a long-running server."""
    import atexit
    import weakref

    wref = weakref.ref(feeder)

    def _close_at_exit():
        f = wref()
        if f is not None:
            f.close()

    atexit.register(_close_at_exit)


def maybe_make_decode_feeder(
    offsets, modules, arena_bytes: int
) -> DecodeFeeder | None:
    """A DecodeFeeder over the coverable layers, or None with a printed
    reason (opt-in feature: silence would read as 'enabled')."""
    try:
        import mlx.core as mx
        import mlx_kquant as kq

        if not hasattr(kq, "arena_alloc"):
            raise RuntimeError(
                "mlx-kquant lacks arena_alloc (needs the feeder-loop build)")
        if not mx.metal.is_available():
            raise RuntimeError("Metal unavailable")
        if "gpu" not in str(mx.default_device()).lower():
            raise RuntimeError(
                "default device is not GPU (the every-token layers must "
                "be on GPU)")
        if arena_bytes < (1 << 30):
            raise RuntimeError(
                f"arena budget too small ({arena_bytes / 1e9:.1f} GB)")
        feeder = DecodeFeeder(offsets, modules, arena_bytes)
        # CLI runs never tear the feeder down explicitly and __del__ is
        # not reliable at interpreter exit, so the hit-rate/prestage stats
        # lines would silently vanish; close() is idempotent, so the
        # server's explicit unload close makes this a no-op.
        _register_exit_close(feeder)
        return feeder
    except Exception as e:
        print(f"[stream] decode feeder unavailable ({e}); "
              "decode stays on the CPU page-cache path")
        return None
