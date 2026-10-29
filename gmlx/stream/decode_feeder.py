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

On by default for ``--stream-experts`` models (``--no-decode-feeder`` or
``GMLX_FEEDER_DECODE=0`` disables; the every-token layers must be on the
GPU). Arena size defaults to what the memory ceiling leaves after the
every-token weights, the priced KV room and the host floor; the arena
either keeps the prefill ring's room apart or holds it and lends it back
at each prefill (see ``ring_share`` below). ``GMLX_DECODE_ARENA_GB``
overrides. The
arena also answers system memory pressure arriving after load by shrinking
(and later regrowing) itself - see the pressure constants below. Miss reads
are joined with a timeout: a read wedged in the kernel is contained (slot
quarantined, expert dropped) rather than waited on - see the wedge
constants below.
"""

from __future__ import annotations

import fcntl
import math
import mmap
import os
import queue
import threading
import time
import weakref
from concurrent.futures import (
    Future, ThreadPoolExecutor, wait as futures_wait)
from contextlib import contextmanager
from functools import lru_cache, partial

import numpy as np

from . import keepwarm
import gmlx.load.loadlog as loadlog
from gmlx.envflags import env_bool, env_choice, env_float, env_int
from .feeder_common import (
    ATTRS,
    KINDS,
    DaemonPool,
    lock_pages,
    copy_experts,
    read_range,
    swapped_weights,
    unlock_pages,
    unshare_on_fork,
    verify_zero_copy,
)

# Miss-pull parallelism: per layer, up to top_k experts x 3 stacks of
# MB-scale slices; enough in-flight preads for SSD sequential bandwidth.
# GMLX_DECODE_READ_WORKERS overrides (a high-fan-out model at a mid hit
# rate can queue more than 12 slices in one demand wave).
_READ_WORKERS = 12

# Lookahead prestage (see ``prestage``): predictions from the lookahead
# hook (gmlx.stream.lookahead) pre-read the next MoE layer's likely misses while
# the current layer computes. Capped at GMLX_DECODE_LOOKAHEAD_K ranked predictions
# per call (the ranking head is far more reliable than its tail - wasted
# reads scale with the cap and the ~25% mispredict rate) on a small
# dedicated pool with its own bounce buffers, so speculation can never
# starve demand misses of workers or buffers.
_LA_WORKERS = 6
_LA_K = 6

# Prestage eviction when the arena is full (GMLX_DECODE_PRESTAGE_EVICT).
# "guard" displaces only a resident no more popular than the prediction,
# which stops prefetch in steady state: every resident was routed once.
# "popular" evicts the least-popular resident, already rank-gated by
# GMLX_DECODE_LOOKAHEAD_MIN_P. Routing is untouched either way.
_EVICT_POLICIES = ("auto", "guard", "popular")
_EVICT_DEFAULT = "auto"

# Barrier width (GMLX_DECODE_SETTLE). "routed" joins only the entries this
# call needs; leaving the rest outstanding is safe because a pending slot's
# owner is -4, invisible to the empty scan, victim scan and resize keep-set,
# so it never reaches the returned slot map. "all" joins every pending
# prestage: free at a few reads per token, 28s of a 123s run under
# "popular".
_SETTLE_POLICIES = ("auto", "all", "routed")
_SETTLE_DEFAULT = "auto"

# Fast-disk recipe (GMLX_DECODE_FAST_DISK=auto|on|off). Both policies above
# and the lookahead pool's disk priority depend on whether speculative
# reads steal bandwidth from demand misses. With the arena seeded from the
# prefill ring, a drive at _FAST_DISK_GBPS or more has bandwidth to spare
# (docs/internals/streaming-measurements.md has the figures). "auto" probes at
# load, takes the fast recipe above _FAST_DISK_GBPS, and demotes for good
# after _FAST_DISK_DEMOTE_WINDOWS saturated duty windows. Pool threads keep
# their start priority: Darwin sets disk policy per thread at thread start.
_FAST_DISK_GBPS = 5.0
_FAST_DISK_PROBE_READS = 32
_FAST_DISK_PROBE_DEPTH = 4
_FAST_DISK_PROBE_TIMEOUT_S = 5.0
_FAST_DISK_RECHECK = 4096
_FAST_DISK_DUTY = 0.9
_FAST_DISK_DEMOTE_WINDOWS = 2

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
# fossilizing the first prompt's routing. N is in stage calls, so the half-life in
# tokens is N / covered-MoE-layers - 4096 is ~98 tokens on a 42-layer model and
# ~128 on a 32-layer one. GMLX_DECODE_DECAY_EVERY overrides.
_DECAY_EVERY = 4096

# No background seeder fills empty slots from the drive: a seed read is a
# guess that spends bandwidth a demand read wanted, and on a fast drive it
# gains nothing measurable. Prestage claims the empty slots first, on router
# evidence rather than a popularity guess.
#
# Prefill seeding is different: every expert of a layer passes through the
# prefill ring anyway, so the prompt's most-routed experts are copied from
# the ring slot into empty arena slots (memory bandwidth only) while the
# ring moves on. Without it decode starts cold and misses every expert of
# its first tokens. Each prefill chunk's routing counts enter the
# popularity ledger at the weight of _SEED_COUNT_TOKENS decode tokens.
# GMLX_DECODE_SEED=0 disables.
_SEED_COUNT_TOKENS = 32
_SEED_COPY_WORKERS = 4


# A prefill ring pass copies the experts the arena holds out of their slots
# instead of reading them from the drive (arena_sources). Arena writes wait
# for those copies. GMLX_RING_FROM_ARENA=0 reads every expert from the drive.
# A copy takes milliseconds: a claim held past _SOURCE_JOIN_S belongs to a
# stage queued behind a wedged read, of a pass that ended, whose slot the
# next pass stages again.
_SOURCE_JOIN_S = 60.0


class ArenaSources:
    """A layer's arena residents for one prefill ring stage: expert
    ``experts[i]`` is in slot ``slots[i]`` of ``bufs[kind]``, a
    (arena pair, stride) tuple. The arena writes none of its slots until
    ``release``."""

    __slots__ = ("experts", "slots", "bufs", "_done")

    def __init__(self, experts, slots, bufs, done):
        self.experts = experts
        self.slots = slots
        self.bufs = bufs
        self._done = done

    def release(self) -> None:
        # A stage blocked on a wedged read keeps this object for good, so
        # it lets go of the arena and the feeder.
        done, self._done = self._done, None
        self.bufs = None
        if done is not None:
            done()


def _sync_gpu() -> None:
    """Wait for every queued GPU command."""
    try:
        import mlx.core as mx

        mx.synchronize()
    except Exception:  # noqa: S110 - no Metal device: nothing in flight
        pass


def _residency_apply(kq, ops: list) -> None:
    for old, new in ops:
        if old is not None:
            kq.residency_erase(old)
        if new is not None:
            kq.residency_insert(new)
    kq.residency_commit()


def _removed_bytes(ops: list) -> int:
    return sum(int(old.nbytes) for old, _ in ops if old is not None)


# Buffers a failed residency worker could not erase. They stay alive until
# exit: freeing a buffer the set still holds is the worse fault.
_PARKED: list = []


def _erase_or_park(kq, olds: list) -> None:
    try:
        _residency_apply(kq, [(old, None) for old in olds])
    except BaseException as e:
        _PARKED.extend(olds)
        try:
            print(f"[stream] residency erase failed again ({e!r}), keeping "
                  f"{len(olds)} buffers until exit")
        except BaseException:  # noqa: S110
            pass


def _wire_grown_ref(ref, key, a) -> None:
    # A queued job must not keep its feeder alive.
    feeder = ref()
    if feeder is not None:
        feeder._wire_grown(key, a)


class _ResidencyWorker:
    """Applies Metal residency set changes and commits them on its own
    thread, so a commit does not stall decode. A queued change holds the
    buffer it removes until the commit after it runs. ``held`` counts
    those bytes. Jobs queued with a change (wiring its added buffers) run
    just before it, and one that raises is skipped. If a residency call
    raises, the worker erases the buffers it was to remove, runs every job
    it has not run, and the calling thread commits every later change."""

    def __init__(self):
        self._cv = threading.Condition()
        self._ops: list = []
        self._jobs: list = []
        self._busy = False
        self._stop = False
        self._failed = False
        self.held = 0
        self._stats = env_bool("GMLX_DECODE_PHASE_STATS", False)
        self._thread = threading.Thread(
            target=self._run, name="gmlx-residency", daemon=True)
        self._thread.start()

    def put(self, ops: list, jobs: list = ()) -> bool:
        """Queue (removed, added) array pairs, either side None, and the
        callables to run before them. False once the worker has failed or
        stopped: the caller applies and runs them itself."""
        with self._cv:
            if self._failed or self._stop:
                return False
            self._ops.extend(ops)
            self._jobs.extend(jobs)
            self.held += _removed_bytes(ops)
            self._cv.notify()
        return True

    def drain(self) -> None:
        """Wait until every queued change is committed, or a failed worker
        has erased the buffers it removed."""
        if threading.current_thread() is self._thread:
            return  # it would wait for itself
        with self._cv:
            self._cv.wait_for(
                lambda: not (self._ops or self._jobs or self._busy))

    def stop(self, wait: bool = True) -> None:
        """End the thread once the queue is empty."""
        if wait:
            self.drain()
        with self._cv:
            self._stop = True
            self._cv.notify()

    def _run(self) -> None:
        import mlx_kquant as kq

        ops: list = []
        jobs: list = []
        try:
            while True:
                with self._cv:
                    self._cv.wait_for(
                        lambda: self._ops or self._jobs or self._stop)
                    if not (self._ops or self._jobs):
                        return
                    ops, self._ops = self._ops, []
                    jobs, self._jobs = self._jobs, []
                    self._busy = True
                t0 = time.monotonic()
                while jobs:
                    job = jobs.pop(0)
                    try:
                        job()
                    except Exception:  # noqa: S110 - a job never fails a commit
                        pass
                    del job  # it holds a buffer a shrink may replace
                t1 = time.monotonic()
                if ops:
                    _residency_apply(kq, ops)
                dt, dw = time.monotonic() - t1, t1 - t0
                n, gone = len(ops), _removed_bytes(ops)
                ops = []    # the removed buffers go only after their commit
                with self._cv:
                    self.held -= gone
                    self._busy = False
                    self._cv.notify_all()
                if self._stats:
                    try:
                        print(f"[phase] residency worker: {n} changes "
                              f"committed in {dt:.2f}s, wiring "
                              f"{dw:.2f}s", flush=True)
                    except Exception:  # noqa: S110 - a stats line never fails a commit
                        pass
        except BaseException as e:
            # The set's state is unknown. Erase every buffer a change
            # removes before it can be freed, here, since a closing feeder
            # leaves no other thread to do it. The arena holds the added
            # ones, and a later swap erases them inline.
            with self._cv:
                self._failed = True
                olds = [old for old, _ in ops + self._ops if old is not None]
                jobs = jobs + self._jobs
                ops, self._ops, self._jobs = [], [], []
            try:
                try:
                    print(f"[stream] residency worker failed ({e!r}), "
                          "committing on the decode thread from here on")
                except BaseException:  # noqa: S110
                    pass
                _erase_or_park(kq, olds)
                del olds
                for job in jobs:  # wiring touches no residency set
                    try:
                        job()
                    except BaseException:  # noqa: S110
                        pass
                del jobs
            finally:
                with self._cv:
                    self.held = 0
                    self._busy = False
                    self._cv.notify_all()


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

# Longest hold, in forwards, of a ring lend that wide calls keep after the
# ring came back in the forward one of them repaid in (see
# DecodeFeeder.repay_at).
_HOLD_FORWARDS = 16


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


def _iopol_utility() -> None:
    """Drop the calling thread's disk-I/O priority (Darwin UTILITY tier),
    so the kernel schedules speculative lookahead reads behind demand
    misses. Colibri gets the same effect structurally from its single
    QD-1 pilot thread."""
    import ctypes

    libc = ctypes.CDLL(None, use_errno=True)
    # setiopolicy_np(IOPOL_TYPE_DISK=0, IOPOL_SCOPE_THREAD=1, IOPOL_UTILITY=4)
    libc.setiopolicy_np(0, 1, 4)


def _refuse_expert_scales(proj, path: str) -> None:
    # The arena binds experts by slot, so a scale per expert would be read
    # at the slot id, by the kernels or by the wrapper alike.
    from gmlx.load.out_scale import weight_scale

    s = weight_scale(proj)
    if s is not None and s.size > 1:
        raise NotImplementedError(
            f"{path}: expert streaming does not support per-expert weight "
            f"scales yet")


class DecodeFeeder:
    """Per-layer wired expert arenas with popularity-driven replacement."""

    def __init__(
        self,
        offsets: dict[int, list[tuple[str, int, int, int, str]]],
        modules: dict[int, list],
        arena_bytes: int,
        stats_verbose: bool | None = None,
        lend_bytes: int = 0,
        fast_disk: str | None = None,
        ring_share: int = 0,
    ):
        import mlx_kquant as kq

        # None = follow loadlog; the CLI passes -v (built after its session).
        self._stats_verbose = (
            loadlog.is_verbose() if stats_verbose is None else stats_verbose)
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
                        _refuse_expert_scales(proj, path)
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

        # Sized capacity. ``_slots`` tracks the live per-layer size: a
        # governor shed or a ring lend (``lend_bytes``, see lend_for_ring)
        # takes it below capacity, and regrow returns it.
        self._orig_slots = dict(self._slots)
        self._per_expert = {li: per_expert[li] for li in self._layers}
        self.nominal_bytes = sum(
            self._orig_slots[li] * self._per_expert[li] for li in self._layers)
        self._pressure_steps = 0
        self._lend_frac = 1.0
        # Ring time-share: ``ring_share`` bytes of the arena are the prefill
        # ring's room. Each prefill pass that builds the ring takes them
        # back by shrinking whole layers to one slot (``_lent``), which
        # copies one expert per layer and frees the memory before the ring
        # allocates. The next decode drops the ring and clears ``_lent``,
        # and those layers regrow at their own stage(). The first prefill
        # starts lent, so the ring and the arena never sum past the budget.
        # A ring quarantined with its slots keeps the room for good
        # (``_ring_dead``): every layer then gives up its share of it.
        self._ring_share = int(ring_share)
        self._lent: set[int] = set()
        self._ring_lends = 0
        self._ring_dead = False
        # When the ring comes back in the forward a wide call repaid in,
        # the wide calls after it keep the new lend for a hold: one
        # forward, doubling each time it recurs up to _HOLD_FORWARDS. A
        # batch that overflows a later layer then does not release and
        # rebuild the ring every step, and one that overflows only now
        # and then is not held long.
        self._wide_repaid = False
        self._repaying_wide = False
        self._hold_left = 0
        self._hold_next = 1
        if self._ring_share:
            self._lent = self._pick_lent(self._ring_share)
            for li in self._lent:
                self._slots[li] = 1
        if lend_bytes > 0 and self.nominal_bytes:
            # The time-share layers have freed their bytes already.
            rest = self.nominal_bytes - self._ring_share
            self._lend_frac = max(0.0, 1.0 - lend_bytes / max(rest, 1))
            self._slots = {li: self._target_slots(li) for li in self._layers}
        # Priced KV room outside the arena (set by the loader); regrow keeps
        # the governor's headroom above half of it. The host floor and the
        # host overhead the arena was sized to leave in RAM, and the part
        # of them the ceiling leg charged (past the RAM outside the
        # ceiling): a second model load must leave them too.
        self._room_bytes = 0
        self._floor_bytes = 0
        self._host_bytes = 0
        self._outside_bytes = 0

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
                unshare_on_fork(a[1])
                self._arena[(li, kind)] = a
                self._views[(li, kind)] = a[0].reshape((s,) + shape[1:])
                self.arena_bytes += s * stride
        self._locked: dict[tuple[int, str], tuple[int, int]] = {}
        self.locked_bytes = 0
        # _locked, locked_bytes and the swap of an arena buffer: the
        # residency worker wires grown buffers (_wire_grown).
        self._lock_mu = threading.RLock()
        # The one-time wiring pass waits for ensure_wired (the wrapper's
        # first decode call), after the prefill ring is released. Until
        # then a layer wires at its first stage() call, before its slots
        # fill: untouched arena pages have no physical cost, and filled
        # pages left unwired are what the kernel swaps under pressure and
        # the wiring pass then drags back. The loader points _release_ring
        # at the prefill feeder's slot release so the handoff happens in
        # one place.
        self._mlock_deferred = True
        self._release_ring = None

        # Aligned-read plumbing (see _PAGE above): per-worker bounce
        # buffers big enough for the largest expert plus the alignment
        # slack on both ends.
        self._aligned = os.environ.get("GMLX_DECODE_ALIGNED_READS", "1") != "0"
        self._sizes = {p: os.fstat(fd).st_size for p, fd in self._fds.items()}
        read_workers = env_int("GMLX_DECODE_READ_WORKERS", _READ_WORKERS)
        if self._aligned:
            max_stride = max(
                stride for entry in self._layers.values()
                for (_, _, _, stride, _) in entry.values())
            self._bounce_bytes = max_stride + 2 * _PAGE
            self._bounce: queue.Queue = queue.Queue()
            for _ in range(read_workers):
                self._bounce.put(self._new_bounce())

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
        self._decay_every = max(
            1, env_int("GMLX_DECODE_DECAY_EVERY", _DECAY_EVERY))
        self._calls = 0
        self._hits = 0
        self._lookups = 0
        self._layer_hits = {li: 0 for li in self._layers}
        self._layer_lookups = {li: 0 for li in self._layers}
        self._read_pool = DaemonPool(read_workers)

        # Lookahead prestage state (constants above; pool and bounce
        # buffers are lazy - most runs never prestage). _pending maps
        # li -> {expert: (slot, futures, submit time)}; a pending slot's
        # owner is -4 (reserved: invisible to the empty scan, the victim
        # scan, and the resize keep-set) until its read completes and the
        # main thread publishes it. Workers only ever write slot bytes;
        # all residency metadata stays main-thread.
        self._pending: dict[int, dict[int, tuple[int, list, float]]] = {}
        # Prefill seeds (seed_from_ring): li -> ((expert, slot) rows, copy
        # future). Seeded slots hold -4 until _flush_seeds publishes them.
        self._seeds: dict[int, tuple[np.ndarray, Future]] = {}
        self._seed_on = env_bool("GMLX_DECODE_SEED", True)
        self._seed_pool: ThreadPoolExecutor | None = None
        self._seed_copy_pool: ThreadPoolExecutor | None = None
        self._seeded = 0
        self._seed_bytes = 0
        self._t_seed = 0.0
        # Prefill ring stages copying out of the arena (arena_sources).
        self._ring_from_arena = env_bool("GMLX_RING_FROM_ARENA", True)
        self._source_reads = 0
        self._source_cv = threading.Condition()
        self._la_pool: DaemonPool | None = None
        self._la_bounce: queue.Queue | None = None
        self._la_k = env_int("GMLX_DECODE_LOOKAHEAD_K", _LA_K)
        self._la_cancel = (
            os.environ.get("GMLX_DECODE_LOOKAHEAD_CANCEL", "1") != "0")
        self._fast_disk = self._resolve_fast_disk(fast_disk)
        self._evict = env_choice(
            "GMLX_DECODE_PRESTAGE_EVICT", _EVICT_DEFAULT, _EVICT_POLICIES)
        if self._evict == "auto":
            self._evict = "popular" if self._fast_disk else "guard"
        self._settle = env_choice(
            "GMLX_DECODE_SETTLE", _SETTLE_DEFAULT, _SETTLE_POLICIES)
        if self._settle == "auto":
            self._settle = "routed" if self._fast_disk else "all"
        self._duty_saturated = 0
        self._duty_marks = (0, 0.0)
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
        # Diagnostic ledger of every byte this feeder submits to a read
        # worker (GIL-serialized increments; close enough for attribution).
        self._bytes_read = 0

        # GMLX_LOG_ROUTED=path: append every decode stage() call's routed
        # expert ids to an npz (li per call + concatenated ids + row
        # offsets). Offline replay of arena policies against real routing
        # traces (speculative-verify byte economics, arena sizing).
        self._routed_log_path = os.environ.get("GMLX_LOG_ROUTED")
        self._routed_log: list | None = (
            [] if self._routed_log_path else None)

        # Split-rule decisions of the wrapper's token split (see
        # note_split_rule).
        self._rule_split = 0
        self._rule_ring = 0
        self._rule_ahead = 0

        # Lossy-lever accounting (shed_misses + the wrapper's layer shed).
        self._shed_n = 0
        self._shed_mass = 0.0
        self._shed_tokens = 0
        self._layer_shed_n = 0

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

        # Pressure adaptation state (constants above).
        self._pressure_on = os.environ.get("GMLX_DECODE_PRESSURE", "1") != "0"
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
        return self._lock_current(key, self._arena[key])

    def _lock_current(self, key: tuple[int, str], a) -> bool:
        """Lock ``a``'s pages and record them under ``key`` if ``a`` is
        still the layer's buffer. The lock runs outside ``_lock_mu``. A
        lock that lands on a replaced or already recorded buffer is undone,
        since page locks nest, so ``_locked`` names each current buffer once.
        False when ``a`` is not current or the kernel refuses."""
        with self._lock_mu:
            if getattr(self, "_closed", False) or self._arena.get(key) is not a:
                return False
            if key in self._locked:
                return True
        entry = lock_pages(a[1])
        if entry is None:
            return False
        with self._lock_mu:
            current = (not getattr(self, "_closed", False)
                       and self._arena.get(key) is a)
            first = current and key not in self._locked
            if first:
                self._locked[key] = entry
                self.locked_bytes += entry[1]
        if not first:
            unlock_pages(entry)
        return current

    def _munlock_buf(self, key: tuple[int, str]) -> None:
        with self._lock_mu:
            entry = self._locked.pop(key, None)
            if entry is None:
                return
            unlock_pages(entry)
            self.locked_bytes -= entry[1]

    def _wire_grown(self, key: tuple[int, str], a) -> None:
        """Wire a grown buffer, on the residency worker."""
        if os.environ.get("GMLX_DECODE_ARENA_MLOCK", "1") != "0":
            self._lock_current(key, a)

    def _verify_zero_copy(self) -> None:
        li = min(self._layers)
        verify_zero_copy(
            li, ((k, mod, path, off)
                 for k, (mod, path, off, _, _) in self._layers[li].items()),
            self._fds)

    def covers(self, li: int) -> bool:
        return li in self._layers

    def note_split_rule(self, split: bool, ahead: bool = False) -> None:
        """Count one split-rule decision for the -v summary. ``ahead``:
        the ring had already started the layer's read."""
        if split:
            self._rule_split += 1
        else:
            self._rule_ring += 1
            self._rule_ahead += int(ahead)

    def can_stage_smaller(self, li: int) -> bool:
        """True when a stage() refusal was a slot-count overflow that a
        smaller (token-split) call could clear, as opposed to a wedge or
        staging being disabled outright."""
        # A lent layer holds one slot: split pieces would bottom out at
        # one-token page-cache gathers. It takes the ring or one fallback.
        return (
            not self._staging_disabled
            and li in self._layers
            and li not in self._wedged_layers
            and li not in self._lent
        )

    # staging

    def _read_expert(self, li: int, kind: str, e: int, slot: int,
                     bounce: queue.Queue | None = None) -> None:
        _, path, off, stride, _ = self._layers[li][kind]
        mv = self._arena[(li, kind)][1]
        dest = mv[slot * stride:(slot + 1) * stride]
        off_e = off + e * stride
        n = stride
        if not self._aligned:
            self._bytes_read += n
            read_range(self._fds[path], dest, off_e)
            return
        a = off_e & ~(_PAGE - 1)
        b = min(
            (off_e + n + _PAGE - 1) & ~(_PAGE - 1),
            self._sizes[path])
        self._bytes_read += b - a
        # The read lands in a bounce buffer, not in the Metal-shared slot:
        # a pread straight into the slot contends with the gathers running
        # against the same buffer and costs more GPU time than it saves in
        # read wait.
        pool = bounce if bounce is not None else self._bounce
        buf = pool.get()
        try:
            bmv = buf[1][: b - a]
            read_range(self._fds[path], bmv, a)
            src = bmv[off_e - a: off_e - a + n]
            # numpy releases the GIL for the copy; a memoryview
            # assignment holds it for the whole expert.
            np.copyto(np.frombuffer(dest, dtype=np.uint8),
                      np.frombuffer(src, dtype=np.uint8))
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

    def _residency_worker(self) -> _ResidencyWorker | None:
        """The thread that commits the arena's residency set changes, or
        None to commit on the calling thread (GMLX_RESIDENCY_ASYNC=0)."""
        rw = getattr(self, "_rworker", None)
        if rw is None and env_bool("GMLX_RESIDENCY_ASYNC", True):
            rw = self._rworker = _ResidencyWorker()
        return rw

    def _residency_drain(self) -> None:
        """Commit every queued residency change before one made here."""
        rw = getattr(self, "_rworker", None)
        if rw is not None:
            rw.drain()

    def ensure_wired(self) -> None:
        """First decode call: release the prefill ring and wire the arena
        (one-time fault pass; a few seconds on a tens-of-GB arena). Called
        by the offload wrapper on one-token calls and, through repay_at,
        on wider calls at a forward's first layer, never inside a ring
        pass.

        Later decode calls repay a ring lend: the ring is dropped again and
        the arena's slot targets return to full size (layers regrow lazily
        at their own stage() calls, each re-wiring as it resizes)."""
        # A wide repay marks its forward: a ring rebuild later in it
        # starts a hold (see repay_at).
        if self._repaying_wide:
            self._repaying_wide = False
            self._wide_repaid = True
        if self._mlock_deferred:
            self._mlock_deferred = False
            # Ring buffers land in MLX's freed-buffer cache; _drop_ring
            # flushes it, so the arena wires into pages the OS has back.
            released = self._drop_ring()
            for li in list(self._seeds):
                self._flush_seeds(li)
            if self._seeded:
                print(
                    f"[stream] decode arena seeded from the prefill ring: "
                    f"{self._seeded} experts ({self._seed_bytes / 1e9:.1f} "
                    f"GB), copies {self._t_seed:.1f}s off the prefill path")
            t0 = time.monotonic()
            self._mlock_arena()
            t1 = time.monotonic()
            self._repay_lend(released)
            t2 = t3 = t1
            queued = False
            if env_bool("GMLX_GPU_RESIDENT", True):
                import mlx_kquant as kq

                if getattr(kq, "residency_insert", None):
                    self._gpu_resident = True
                    adds = [(None, a[0]) for a in self._arena.values()]
                    rw = self._residency_worker()
                    queued = rw is not None and rw.put(adds)
                    if not queued:
                        for _, a in adds:
                            kq.residency_insert(a)
                        t2 = time.monotonic()
                        kq.residency_commit()
                    t3 = time.monotonic()
            if env_bool("GMLX_DECODE_PHASE_STATS", False):
                res = ("residency queued to the worker" if queued else
                       f"residency insert {t2 - t1:.2f}s, "
                       f"commit {t3 - t2:.2f}s")
                print(f"[phase] first decode: arena wiring {t1 - t0:.2f}s, "
                      f"{res}", flush=True)
            return
        if self._owes_lend():
            self._repay_lend(self._drop_ring())

    def _owes_lend(self) -> bool:
        return not self._ring_dead and (self._lend_frac < 1.0
                                        or bool(self._lent))

    def _drop_ring(self) -> bool:
        """Release the prefill ring. False when the ring keeps its slots
        (quarantined: a staging read may still write them)."""
        if self._release_ring is None:
            return True
        gone = self._release_ring() is not False
        self._clear_mlx_cache()
        return gone

    def _repay_lend(self, released: bool) -> None:
        """The ring is gone: lent layers regrow to capacity at their own
        stage() calls. A ring that kept its slots still holds their room
        for good and nothing lends again. Every layer gives up its share
        of that room instead of the lent ones holding one slot each, so
        none is left out of the arena."""
        if released:
            self._lend_frac = 1.0
            self._lent.clear()
            return
        if self._ring_share and self.nominal_bytes:
            self._lend_frac = min(self._lend_frac, max(
                0.0, 1.0 - self._ring_share / self.nominal_bytes))
        self._lent.clear()
        self._ring_dead = True
        self._ring_share = 0
        # Before any lent layer regrows into the room.
        self._shrink_now()

    def at_target(self, li: int) -> bool:
        """Whether layer ``li`` holds its target slot count. A layer that
        still owes a resize (a repaid lend, a pressure step) must take
        stage() once, the only place it resizes."""
        return self._slots.get(li) == self._target_slots(li)

    def forward_start(self, li: int) -> None:
        """A multi-token call at ``li``, outside a token split. At the
        forward's first covered layer it ends the forward before, so a
        ring rebuild from here on belongs to this one. Every lend that
        follows no wide repay resets the hold to one forward."""
        if self._layers and li == min(self._layers):
            self._wide_repaid = False

    def repay_at(self, li: int, ids: np.ndarray,
                 ring_ok: bool = False) -> bool:
        """Whether a multi-token call routed to ``ids`` at ``li`` runs
        ensure_wired: the first decode's wiring, or the repay of a ring
        lend. Only at a forward's first covered layer: a call the arena
        serves there is a decode step (batched or speculative) or a short
        prompt, and a later layer of a ring pass must not drop the ring
        under the layers after it. Batched and speculative decode never
        send a one-token call, so without this they would never wire.

        ``ring_ok``: the call is wide enough to take a live ring. It keeps
        the lend when its experts overflow this layer once repaid, as it
        would take the ring here anyway, and while a hold from an earlier
        rebuild lasts. The caller has called forward_start."""
        if not self._layers or li != min(self._layers):
            return False
        if self._mlock_deferred:
            return True  # the first decode wires, at any width
        if not self._owes_lend():
            return False
        if not ring_ok:
            return True
        if self._hold_left:
            self._hold_left -= 1
            return False
        if (np.unique(ids.reshape(-1)).size
                > self._target_slots(li, lend=False)):
            return False
        self._repaying_wide = True
        return True

    def _pick_lent(self, nbytes: int) -> set[int]:
        """Layers whose capacity, shrunk to one slot, frees ``nbytes``.
        Layers still at one slot go first: a lend the last decode did not
        refill costs nothing to keep. Then the coldest, by decode hit
        rate, so the lend costs the fewest hits; layers with no lookups
        yet go in layer order. Counting capacity rather than live slots
        keeps the arena at its nominal size less one share, however often
        the ring returns before the layers refill."""
        hits = getattr(self, "_layer_hits", {})
        looks = getattr(self, "_layer_lookups", {})
        wedged = getattr(self, "_wedged_layers", set())

        def order(li):
            n = looks.get(li, 0)
            return (self._slots[li] > 1,
                    hits.get(li, 0) / n if n else 1.0, li)

        picked: set[int] = set()
        freed = 0
        for li in sorted((li for li in self._layers if li not in wedged),
                         key=order):
            if freed >= nbytes:
                break
            picked.add(li)
            freed += max(0, self._orig_slots[li] - 1) * self._per_expert[li]
        return picked

    def _lend_ring_share(self) -> int:
        """Shrink the time-share layers to one slot now. Their in-flight
        prestage reads join first (a resize must not free a buffer a read
        still writes into), after the queued gathers: a read that wedges
        in the join zeroes its slot. Returns the bytes freed."""
        _sync_gpu()
        self._lent = self._pick_lent(self._ring_share)
        empty = np.empty(0, dtype=np.int64)
        for li in self._lent:
            if self._pending.get(li):
                self._settle_pending(li, empty)
        freed = self._shrink_now()
        self._ring_lends += 1
        return freed

    def lend_for_ring(self, nbytes: int) -> None:
        """Shrink the wired arena by ~``nbytes`` so the prefill ring can
        re-allocate without breaching the wired cap. The prefill feeder
        calls this before rebuilding its slots on a post-decode prefill
        pass (chat follow-up turns). An arena that holds the ring's room
        (``ring_share``) frees it first, by emptying its coldest layers.
        Past that, the budget keeps the ring's room out of the arena, so
        the proportional lend below is a no-op while the kernel still has
        that room; it fires only on a box whose free RAM is gone.

        Shrinking is eager (every layer now): the ring path never calls
        stage(), so the lazy per-layer resize would not fire during the
        prefill that needs the memory. Each layer keeps its most popular
        residents, so the decode resume after this turn starts warm.
        ensure_wired repays the lend on the next decode call."""
        # A rebuild in the forward a wide call repaid in starts a hold,
        # twice as long as the last each time it recurs.
        if self._wide_repaid:
            self._wide_repaid = False
            self._hold_left = self._hold_next
            self._hold_next = min(2 * self._hold_next, _HOLD_FORWARDS)
        else:
            self._hold_left = 0
            self._hold_next = 1
        if self._ring_share and not self._lent:
            freed = self._lend_ring_share()
            if self._ring_lends == 1:
                loadlog.info(
                    f"[stream] decode arena lends {len(self._lent)} layers "
                    f"({freed / 1e9:.1f} GB) to the prefill ring at each "
                    "prefill; they refill at the next decode "
                    "(GMLX_RING_TIMESHARE=0 keeps the ring's room apart)")
            nbytes = max(0, nbytes - freed)
            if not nbytes:
                return
        if self._mlock_deferred:
            return  # arena never wired; ring + cold arena already coexist
        if not self.arena_bytes:
            return
        if self._kernel_has_room(nbytes):
            return
        frac = max(0.0, 1.0 - nbytes / self.arena_bytes) * self._lend_frac
        if frac >= self._lend_frac:
            return
        self._lend_frac = frac
        freed = self._shrink_now()
        print(
            f"[stream] decode arena lends {freed / 1e9:.1f} GB to the "
            f"prefill ring (kept {self.arena_bytes / 1e9:.1f} GB); "
            "restored at next decode"
        )

    @staticmethod
    def _kernel_has_room(nbytes: int) -> bool:
        """Whether the memory not wired covers ``nbytes`` above the
        kernel floor. Unknown counts as no room."""
        try:
            from gmlx.stream.budget import (
                kernel_floor_bytes,
                unwired_ram_bytes,
            )

            avail = unwired_ram_bytes()
            if avail is None:
                return False
            return avail - kernel_floor_bytes() >= nbytes
        except Exception:
            return False

    def _shrink_now(self) -> int:
        """Resize every layer down to its current target now. Eager,
        unlike the lazy per-layer resize at stage(): the caller needs the
        bytes before its next allocation. Returns the bytes freed."""
        _sync_gpu()  # no in-flight gather may reference a layer
        freed = 0
        for li in list(self._layers):
            self._flush_seeds(li)
            if li in self._wedged_layers or self._pending.get(li):
                continue  # same no-resize contract as stage()
            target = self._target_slots(li)
            if target < self._slots[li]:
                freed += (self._slots[li] - target) * self._per_expert[li]
                self._resize_layer(li, target)
        return freed

    # governor protocol (gmlx.serve.governor.register_cache)

    def governor_bytes(self) -> int:
        """Bytes the governor may reclaim: the arena above the pressure
        ladder's floor."""
        return max(
            0, self.arena_bytes - self._arena_bytes_at(_PRESSURE_MAX_STEPS))

    def governor_evict(self, fraction: float) -> int:
        """Shrink the arena by ``fraction`` of what is left above the
        ladder's floor, at least one step, keeping each layer's most
        popular residents. Returns the bytes freed. Regrow is the pressure
        ladder's, gated on the governor's headroom (see
        _regrow_headroom_ok), so a shrink the governor asked for does not
        bounce back into the pressure that caused it."""
        if fraction <= 0 or not self._layers:
            return 0
        remaining = _PRESSURE_MAX_STEPS - self._pressure_steps
        if remaining <= 0:
            return 0
        take = max(1, min(remaining, int(math.ceil(fraction * remaining))))
        self._pressure_steps += take
        self._last_step_poll = self._pressure_polls
        self._normal_polls = 0
        freed = self._shrink_now()
        print(
            f"[stream] decode arena shrinks {freed / 1e9:.1f} GB for the "
            f"memory governor (kept {self.arena_bytes / 1e9:.1f} GB); "
            "regrows when headroom returns"
        )
        return freed

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
        keepwarm.touch()
        self._join_source_reads()
        if ((self._pressure_on or self._pressure_steps)
                and self._calls % _PRESSURE_POLL_EVERY == 0):
            self._poll_pressure()
        uniq = np.unique(ids.reshape(-1))
        if self._routed_log is not None and ids.size <= 64:
            # Decode-shaped calls only (a prefill-tail batch would skew the
            # per-token trace).
            self._routed_log.append((li, uniq.astype(np.uint16)))
        if li in self._seeds:
            self._flush_seeds(li)
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
            # A call the grown layer cannot hold is refused either way.
            and not (self._slots[li] < target < uniq.size)
            and li not in self._wedged_layers
            and not self._pending.get(li)
        ):
            # A wedged layer never resizes: reallocating would free the
            # buffer the zombie read may still write into. (The pending
            # guard is belt-and-braces: settle just drained the layer.)
            self._resize_layer(li, target)
        if self._mlock_deferred and self._lend_frac >= 1.0:
            for kind in self._layers[li]:
                self._mlock_buf((li, kind))
        slot_of = self._slot_of[li]
        counts = self._counts[li]
        if self._verify:
            owner_v = self._owner[li]
            for pe, ps in self._verify_prev.get(li, ()):
                if ps < len(owner_v) and owner_v[ps] == pe:
                    self._verify_slot(li, pe, ps, "prev-routed")
        counts[uniq] += 1.0
        self._calls += 1
        if self._calls % self._decay_every == 0:
            for c in self._counts.values():
                c *= 0.5
        if self._fast_disk and self._calls % _FAST_DISK_RECHECK == 0:
            self._recheck_fast_disk()
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

    def _probe_read_rate(self) -> float:
        """Bytes per second the drive delivers at this feeder's own access
        shape: page-aligned F_NOCACHE reads of one expert stride each,
        scattered across the shard set at a demand wave's depth. Rate is the
        25th-percentile read's own rate times the depth, not total bytes over
        total time: F_NOCACHE still serves from pages an earlier run left
        resident, and those reads swamp an aggregate (28.3 GB/s measured on a
        14 GB/s drive). Measured against a deliberately warmed cache, p25
        holds to 12.8-14.9 GB/s up to half the sample cached, where the
        aggregate reads 26 and the median 48. 0.0 on any failure or on a
        probe that outlives _FAST_DISK_PROBE_TIMEOUT_S, which reads as "not a
        fast disk" and keeps the conservative recipe. Daemon workers and that
        timeout are what keep a wedge-prone path from hanging the load here,
        before any of the read-timeout machinery is armed."""
        try:
            picks = []
            rng = np.random.default_rng(0)
            entries = [(li, e) for li, e in self._layers.items()]
            for i in range(_FAST_DISK_PROBE_READS):
                li, entry = entries[i % len(entries)]
                _, path, off, stride, shape = entry[KINDS[i % len(KINDS)]]
                e = int(rng.integers(0, max(1, shape[0])))
                picks.append((path, off + e * stride, stride))
            if not picks:
                return 0.0

            tls = threading.local()
            # Sized here rather than from the bounce pool: the probe runs
            # under GMLX_DECODE_ALIGNED_READS=0 too, where there is none.
            nbuf = max(s for _, _, s in picks) + 2 * _PAGE

            def one(job):
                path, off, stride = job
                buf = getattr(tls, "buf", None)
                if buf is None:
                    buf = tls.buf = mmap.mmap(-1, nbuf)
                a = off & ~(_PAGE - 1)
                b = min((off + stride + _PAGE - 1) & ~(_PAGE - 1),
                        self._sizes[path])
                t = time.monotonic()
                read_range(self._fds[path], memoryview(buf)[: b - a], a)
                return (b - a), time.monotonic() - t

            pool = DaemonPool(_FAST_DISK_PROBE_DEPTH)
            try:
                futs = [pool.submit(one, job) for job in picks]
                _, pending = futures_wait(
                    futs, timeout=_FAST_DISK_PROBE_TIMEOUT_S)
                if pending:
                    return 0.0
                rates = [n / dt for n, dt in (f.result() for f in futs)
                         if dt > 0]
            finally:
                pool.shutdown(wait=False)
            if not rates:
                return 0.0
            return float(
                np.percentile(rates, 25)) * _FAST_DISK_PROBE_DEPTH
        except Exception:
            return 0.0

    def _resolve_fast_disk(self, mode: str | None = None) -> bool:
        """Whether this drive has bandwidth to spare for speculation (see
        _FAST_DISK_GBPS). ``mode`` is the model's own setting, and None
        reads GMLX_DECODE_FAST_DISK."""
        if mode is None:
            mode = env_choice(
                "GMLX_DECODE_FAST_DISK", "auto", ("auto", "on", "off"))
        self._probe_bps = 0.0
        if mode != "auto":
            return mode == "on"
        self._probe_bps = self._probe_read_rate()
        thr = env_float("GMLX_DECODE_FAST_DISK_GBPS", _FAST_DISK_GBPS) * 1e9
        return self._probe_bps >= thr

    def _recheck_fast_disk(self) -> None:
        """Demote the fast recipe if the drive turns out to be saturated
        after all. Duty is bytes this feeder actually read over the window,
        against what the probe said the drive can deliver in that wall time.
        One-way: a drive that saturated once is not worth re-probing mid-run,
        and flapping the policies churns residency."""
        calls, (mark_bytes, mark_t) = self._calls, self._duty_marks
        now = time.monotonic()
        got, self._duty_marks = self._bytes_read, (self._bytes_read, now)
        if not mark_t or self._probe_bps <= 0:
            return
        window = now - mark_t
        if window <= 0:
            return
        duty = (got - mark_bytes) / (window * self._probe_bps)
        if duty < _FAST_DISK_DUTY:
            self._duty_saturated = 0
            return
        self._duty_saturated += 1
        if self._duty_saturated < _FAST_DISK_DEMOTE_WINDOWS:
            return
        self._fast_disk = False
        self._evict, self._settle = "guard", "all"
        print(
            f"[stream] decode feeder: drive saturated ({duty:.0%} of "
            f"{self._probe_bps / 1e9:.1f} GB/s over {calls} stage calls) - "
            "prestage back to guess-grade eviction and the wide barrier"
        )

    def _new_bounce(self):
        """A read worker's bounce buffer, kept out of a forked child like
        the arena."""
        import mlx_kquant as kq

        buf = kq.arena_alloc([self._bounce_bytes])
        unshare_on_fork(buf[1])
        return buf

    def _la_state(self) -> DaemonPool:
        if self._la_pool is None:
            n = env_int("GMLX_DECODE_LOOKAHEAD_WORKERS", _LA_WORKERS)
            # Utility priority keeps speculation behind demand misses,
            # which is only worth its latency where the two compete.
            default = "0" if self._fast_disk else "1"
            on_start = (
                _iopol_utility
                if os.environ.get(
                    "GMLX_DECODE_LOOKAHEAD_IOPOL", default) != "0"
                else None)
            self._la_pool = DaemonPool(n, on_start=on_start)
            if self._aligned:
                self._la_bounce = queue.Queue()
                for _ in range(n):
                    self._la_bounce.put(self._new_bounce())
        return self._la_pool

    def shed_misses(self, li: int, ids: np.ndarray, scores: np.ndarray,
                    keep_mass: float) -> np.ndarray | None:
        """Lossy: bool keep-mask over ``ids`` dropping routed experts that
        are neither arena-resident nor prestage-inflight, lowest score
        first, capped at ``1 - keep_mass`` of the token's score mass.
        Returns None when nothing sheds. A shed expert is never staged, so
        it also earns no popularity credit (self-reinforcing by design:
        the arena's long tail stays cold)."""
        res = self._shed_mask(li, ids, scores, keep_mass)
        if res is None:
            return None
        keep, shed_mass, total = res
        self._shed_n += int((~keep).sum())
        self._shed_mass += shed_mass / max(total, 1e-20)
        self._shed_tokens += 1
        return keep

    def _shed_mask(self, li: int, ids: np.ndarray, scores: np.ndarray,
                   keep_mass: float):
        """Policy core shared by ``shed_misses`` (which books the stats)
        and keeper-mode ``prestage`` (which asks what would shed on
        arrival, stats-free). Returns ``(keep, shed_mass, total)`` or
        None when nothing sheds."""
        slot_of = self._slot_of.get(li)
        if slot_of is None:
            return None
        miss = slot_of[ids] < 0
        pend = self._pending.get(li)
        if pend and miss.any():
            miss &= np.array([e not in pend for e in ids])
        if not miss.any():
            return None
        total = float(scores.sum())
        budget = total * max(0.0, 1.0 - keep_mass)
        keep = np.ones(ids.size, dtype=bool)
        shed_mass = 0.0
        shed_n = 0
        for j in np.argsort(scores, kind="stable"):
            if not miss[j] or keep.sum() <= 1:
                continue
            s = float(scores[j])
            if shed_mass + s > budget:
                break
            keep[j] = False
            shed_mass += s
            shed_n += 1
        if not shed_n:
            return None
        return keep, shed_mass, total

    def prestage(self, li: int, pred_ids: np.ndarray, *,
                 demand: bool = False,
                 protect: np.ndarray | None = None,
                 keep_mass: float | None = None,
                 pred_scores: np.ndarray | None = None) -> None:
        """Asynchronously pre-read predicted experts into layer ``li``'s
        arena. Safe under the same fence as ``stage`` (the caller's router
        eval for the *previous* MoE layer of the same token transitively
        fenced every gather that could reference this arena). Slots are
        reserved (-4) at submission and published only when their read has
        completed, so a concurrent lookup can never map an expert to a
        slot holding partial bytes. Predictions never touch popularity.

        ``demand=True`` is the gpu-autonomous boundary mode: the ids are
        certain next-token demand (misses the kernel shed), not guesses,
        so the guess-grade eviction guard (never displace a more popular
        resident) is dropped and stage()'s policy applies: evict the
        least-popular resident outright. ``protect`` lists expert ids
        whose slots must survive this call (the token's routed set -
        routing locality makes them next token's likely demand).

        ``keep_mass`` + ``pred_scores`` (--moe-prestage keepers) filter a
        decode-shaped prediction through the active miss-shed policy
        first: predictions the shed would drop on arrival are not read at
        all, and the keepers - the reads the demand path would otherwise
        do synchronously next layer - stage demand-grade (full width, no
        rank cap, least-popular eviction)."""
        if (
            li not in self._layers
            or self._staging_disabled
            or li in self._wedged_layers
        ):
            return
        self._join_source_reads()
        self._flush_pending(li)
        if li in self._wedged_layers or li not in self._layers:
            return  # the flush itself may have wedged the layer
        pending = self._pending.setdefault(li, {})
        slot_of = self._slot_of[li]
        owner = self._owner[li]
        counts = self._counts[li]
        dead = self._dead.get(li)
        keeper = False
        if keep_mass is not None and pred_scores is not None:
            flat = pred_ids.reshape(-1, pred_ids.shape[-1])
            if flat.shape[0] == 1:  # decode-shaped predictions only
                keeper = True
                res = self._shed_mask(
                    li, flat[0],
                    np.asarray(pred_scores, dtype=np.float32).reshape(-1),
                    keep_mass)
                if res is not None:
                    kept = flat[0][res[0]]
                    if not kept.size:
                        return
                    pred_ids = kept.reshape(1, -1)
        hard = demand or keeper
        # Only the top GMLX_DECODE_LOOKAHEAD_K ranks per row are considered (the
        # ranking head is reliable, the tail is not); residents among them
        # are simply already-good news, not license to dig deeper.
        if hard:
            rows = pred_ids.reshape(-1, pred_ids.shape[-1])
        else:
            rows = pred_ids.reshape(-1, pred_ids.shape[-1])[:, : self._la_k]
        picked: list[int] = []
        seen: set[int] = set()
        protected = np.zeros(len(owner), dtype=bool)
        if protect is not None:
            ps = slot_of[protect.reshape(-1)]
            protected[ps[ps >= 0]] = True
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
                if (
                    not hard
                    and self._evict == "guard"
                    and counts[owner[cand]][v] > counts[e]
                ):
                    # Guess-grade guard: a wrong prediction may only cost
                    # a cold slot, never a hot one. Demand and keeper
                    # modes evict least-popular outright, like stage(), and
                    # so does the "popular" policy (see _EVICT_POLICIES).
                    continue
                s = int(cand[v])
                old = int(owner[s])
            if old >= 0:
                slot_of[old] = -1
            owner[s] = -4
            futs = [
                pool.submit(self._read_expert, li, kind, e, s,
                            self._la_bounce)
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

    # prefill ring copies out of the arena

    def arena_sources(
        self, li: int, geometry: dict[str, tuple[str, int, int]],
    ) -> ArenaSources | None:
        """The layer's resident experts for a prefill ring stage to copy
        instead of reading, or None. ``geometry`` maps each kind to the
        ring's (path, offset, nbytes) for the layer, and a layer whose
        stacks differ from the arena's gives none. Called on the thread
        that writes the arena, so the residents cannot change before the
        writers' next join."""
        if (
            not self._ring_from_arena
            or li not in self._layers
            or li in self._wedged_layers
        ):
            return None
        entry = self._layers[li]
        if set(geometry) != set(entry):
            return None
        for kind, (path, off, nbytes) in geometry.items():
            _, a_path, a_off, stride, shape = entry[kind]
            if (path, off, nbytes) != (a_path, a_off, stride * shape[0]):
                return None
        # A later chunk of a cold prompt copies what an earlier one seeded.
        item = self._seeds.get(li)
        if item is not None and item[1].done():
            self._flush_seeds(li)
        owner = self._owner[li]
        slots = np.flatnonzero(owner >= 0)
        if not len(slots):
            return None
        bufs = {kind: (self._arena[(li, kind)], entry[kind][3])
                for kind in entry}
        with self._source_cv:
            self._source_reads += 1
        return ArenaSources(owner[slots].astype(np.int64),
                            slots.astype(np.int64), bufs,
                            self._end_source_read)

    def _end_source_read(self) -> None:
        with self._source_cv:
            self._source_reads -= 1
            if self._source_reads <= 0:
                self._source_reads = 0
                self._source_cv.notify_all()

    def _join_source_reads(self) -> None:
        """Wait for the ring copies out of the arena. Every arena write
        (stage, prestage, resize, the zero slot) joins first."""
        if not self._source_reads:
            return
        with self._source_cv:
            if self._source_cv.wait_for(
                    lambda: not self._source_reads, timeout=_SOURCE_JOIN_S):
                return
            self._source_reads = 0
            self._ring_from_arena = False
        print(
            f"[stream] decode feeder: a prefill ring stage held arena slots "
            f"past {_SOURCE_JOIN_S:.0f}s; the ring reads every expert from "
            "the drive from here on")

    # prefill seeding

    def seed_from_ring(
        self,
        li: int,
        counts: np.ndarray,
        n_tokens: int,
        ring_mvs: dict[str, memoryview],
        present: np.ndarray | None = None,
    ) -> list[Future]:
        """Copy the layer's most-routed experts from its prefill ring slot
        into empty arena slots, off-thread. ``counts`` is the pass's
        routing count per expert, ``ring_mvs`` the slot bytes per kind
        (expert e at e * stride), ``present`` a mask of the experts the
        slot holds (None: all). Returns the copy futures the ring joins
        before it rewrites the slot. Slots publish at the layer's next
        stage() or at ensure_wired, on the main thread."""
        if not self._seed_on or li not in self._layers or li in self._seeds:
            return []
        owner = self._owner[li]
        slot_of = self._slot_of[li]
        if n_tokens > 0:
            self._counts[li] += counts * (_SEED_COUNT_TOKENS / n_tokens)
        if self._routed_log is not None:
            # The pass's routing multiset, keyed past the layer range, so
            # an offline replay can seed the same arena.
            self._routed_log.append((1000 + li, np.repeat(
                np.arange(len(counts)), counts.astype(np.int64)
            ).astype(np.uint16)))
        empty = np.flatnonzero(owner == -1)
        cand = np.flatnonzero((counts > 0) & (slot_of < 0))
        if present is not None:
            cand = cand[present[cand]]
        n = min(len(empty), len(cand))
        if n <= 0:
            return []
        order = np.argsort(-counts[cand], kind="stable")[:n]
        pairs = np.stack([cand[order], empty[:n]], axis=1).astype(np.int64)
        owner[pairs[:, 1]] = -4
        if self._seed_pool is None:
            self._seed_pool = ThreadPoolExecutor(
                max_workers=2, thread_name_prefix="gmlx-seed")
            self._seed_copy_pool = ThreadPoolExecutor(
                max_workers=_SEED_COPY_WORKERS,
                thread_name_prefix="gmlx-seed-copy")
        fut = self._seed_pool.submit(self._seed_layer, li, pairs, ring_mvs)
        self._seeds[li] = (pairs, fut)
        return [fut]

    def _seed_layer(self, li: int, pairs: np.ndarray, ring_mvs) -> None:
        t0 = time.monotonic()
        entry = self._layers[li]
        jobs = []
        for kind, (_, _, _, stride, _) in entry.items():
            dst = np.frombuffer(self._arena[(li, kind)][1], dtype=np.uint8)
            src = np.frombuffer(ring_mvs[kind], dtype=np.uint8)
            for chunk in np.array_split(pairs, _SEED_COPY_WORKERS):
                if len(chunk):
                    jobs.append(self._seed_copy_pool.submit(
                        copy_experts, dst, src, chunk, stride))
        for j in jobs:
            j.result()
        # Wire the layer now: its pages are resident from the copy, so
        # the first-decode wiring pass has nothing left to fault in.
        if self._lend_frac >= 1.0:
            for kind in entry:
                self._mlock_buf((li, kind))
        self._seed_bytes += len(pairs) * self._per_expert[li]
        self._t_seed += time.monotonic() - t0

    def _flush_seeds(self, li: int) -> None:
        item = self._seeds.pop(li, None)
        if item is None:
            return
        pairs, fut = item
        owner = self._owner[li]
        try:
            fut.result()
        except Exception as e:
            owner[pairs[:, 1]] = -1
            print(
                f"[stream] decode feeder: prefill seed of layer {li} failed "
                f"({e}); its slots stay empty")
            return
        owner[pairs[:, 1]] = pairs[:, 0]
        self._slot_of[li][pairs[:, 0]] = pairs[:, 1]
        self._seeded += len(pairs)
        if self._verify:
            for e, s in pairs:
                self._verify_slot(li, int(e), int(s), "seed")

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
        # A resize reallocates the buffers the outstanding writes target, and
        # stage() skips it while anything is pending - so a layer the
        # pressure ladder wants resized takes the wide barrier and drains.
        narrow = (
            self._settle == "routed"
            and self._target_slots(li) == self._slots[li]
        )
        futures_wait(
            [f for e, (_, futs, _) in pending.items() for f in futs
             if not narrow or e in routed],
            timeout=timeout)
        self._t_settle += time.monotonic() - t0
        owner = self._owner[li]
        slot_of = self._slot_of[li]
        for e, (s, futs, _) in list(pending.items()):
            if narrow and e not in routed and not all(f.done() for f in futs):
                # Still in flight and nothing here reads it: leave it
                # pending, to publish at whichever later call finds it done.
                continue
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
                rbounce.put(self._new_bounce())
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
        self._join_source_reads()
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

    def _target_slots(self, li: int, lend: bool = True) -> int:
        """Slots layer ``li`` should hold now; ``lend=False`` leaves out
        any ring lend."""
        if lend and li in self._lent:
            return 1
        frac = 1.0 - self._pressure_steps * _PRESSURE_STEP_FRAC
        if lend:
            frac *= self._lend_frac
        if frac >= 1.0:
            return self._orig_slots[li]
        return max(1, int(self._orig_slots[li] * frac))

    def _arena_bytes_at(self, steps: int) -> int:
        """Arena bytes at ``steps`` pressure steps, with the ring lend now
        in force: the targets _target_slots gives at that step."""
        frac = max(0.0, 1.0 - steps * _PRESSURE_STEP_FRAC) * self._lend_frac
        return sum(
            (1 if li in self._lent
             else max(1, int(self._orig_slots[li] * frac)))
            * self._per_expert[li]
            for li in self._layers
        )

    def _poll_pressure(self) -> None:
        # With pressure polling off only a governor shrink reaches here,
        # and it regrows on the same clock; the kernel level is not read.
        level = _pressure_level() if self._pressure_on else 0
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
        """Regrow only into memory not wired, the governor's own floor
        measure: other apps' memory counts, since the kernel compresses
        or swaps it. The arena's expert reads fill the file-backed
        queue, so a free-pages-only test never passes on a streaming
        model and a shrink would be permanent. The step must leave the
        loader's floor and the governor's kernel floor behind it, so a
        regrow cannot trip the floor that shrank the arena."""
        need = self._arena_bytes_at(self._pressure_steps - 1) - self.arena_bytes
        try:
            from gmlx.stream.budget import _ram_floor_bytes
            from gmlx.stream.budget import kernel_floor_bytes, unwired_ram_bytes

            avail = unwired_ram_bytes()
        except Exception:
            return True
        if avail is None:
            return True
        ram = None
        try:
            from gmlx.device import metal_value

            ram = int(metal_value("memory_size"))
        except Exception:  # noqa: S110 - no device info; the floor prices from avail
            pass
        if avail < need + _ram_floor_bytes(ram or avail) + kernel_floor_bytes():
            return False
        return self._governor_room_ok(need)

    def _governor_room_ok(self, need: int) -> bool:
        """Regrow only while the governor keeps half the KV room after
        it; MLX counts the arena, so a regrow into headroom the batch is
        about to use would be shed straight back."""
        try:
            from gmlx.stream.budget import governor_headroom_bytes

            head = governor_headroom_bytes()
        except Exception:
            return True
        if head is None:
            return True
        return head - need >= 0.5 * self._room_bytes

    def _clear_mlx_cache(self) -> None:
        """The arena's own buffers bypass MLX's allocator, but its cache of
        freed GPU buffers is also memory the system could use right now."""
        try:
            import mlx.core as mx

            (getattr(mx, "clear_cache", None) or mx.metal.clear_cache)()
        except Exception:  # noqa: S110 - cache release is advisory
            pass

    def _resize_layer(self, li: int, new_s: int) -> None:
        """Reallocate a layer's arena at ``new_s`` slots, carrying over its
        most popular residents. In-place release (munlock + madvise) is not
        an option: gathers reference the layer's whole Metal buffer every
        token, so its pages stay GPU-resident whatever madvise says - only
        dropping the buffer returns memory. Reallocating one layer at its
        own stage() call bounds the transient old+new overlap to a single
        layer's stacks, plus the old stacks a grow leaves to the residency
        worker, at most one grown layer's. The caller's eval fence
        guarantees no in-flight gather references this layer."""
        import mlx_kquant as kq

        self._join_source_reads()
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
        resident = getattr(self, "_gpu_resident", False)
        wire: list = []
        row = sum(v[3] for v in entry.values())
        rw = getattr(self, "_rworker", None)
        if resident and rw is not None and rw.held + old_s * row > new_s * row:
            rw.drain()  # at most one grown layer of old buffers waits
        swaps = []
        for kind, (_, _, _, stride, shape) in entry.items():
            key = (li, kind)
            a = kq.arena_alloc([new_s * stride])
            unshare_on_fork(a[1])
            mv_new, mv_old = a[1], self._arena[key][1]
            for ns, os_ in enumerate(keep):
                mv_new[ns * stride:(ns + 1) * stride] = \
                    mv_old[os_ * stride:(os_ + 1) * stride]
            with self._lock_mu:
                self._munlock_buf(key)
                if resident:
                    # a freed buffer must not stay in the residency set
                    swaps.append((self._arena[key][0], a[0]))
                self._arena[key] = a
            self._views[key] = a[0].reshape((new_s,) + shape[1:])
            self.arena_bytes += (new_s - old_s) * stride
            if not self._mlock_deferred:
                wire.append(key)
        # A grown layer wires and commits on the worker. A shrink frees its
        # bytes now, after every queued change: the check above drained
        # them.
        rw = self._residency_worker() if swaps and new_s > old_s else None
        jobs = ([partial(_wire_grown_ref, weakref.ref(self), k,
                         self._arena[k]) for k in wire]
                if rw is not None else [])
        if rw is None or not rw.put(swaps, jobs):
            for key in wire:
                self._mlock_buf(key)
            if swaps:
                for old, new in swaps:
                    kq.residency_erase(old)
                    kq.residency_insert(new)
                kq.residency_commit()
        del swaps, jobs
        self._owner[li] = new_owner
        self._slots[li] = new_s
        # Flush MLX's cache of freed GPU buffers: on a box sized to the
        # wired cap, that idle anon memory is what the kernel starts
        # compressing.
        self._clear_mlx_cache()

    @contextmanager
    def swapped(self, li: int):
        """Swap the layer's expert weights to the arena views for the call."""
        entry = self._layers[li]
        views = {kind: self._views[(li, kind)] for kind in entry}
        # The owner table is read at delta time, not here: a shed inside
        # the swap restages slots, and the delta must follow what ran.
        with swapped_weights(entry, views,
                             slot_owner=lambda: self._owner[li]):
            yield

    def close(self, wait: bool = True, report: bool = True) -> None:
        """``wait=False`` is the finalizer's form: a join from ``__del__``
        deadlocks when the collection runs inside a thread's bootstrap,
        which holds ``threading._shutdown_locks_lock`` that the join of a
        finished thread takes again. Nothing in flight references the
        feeder once it is collectable, so the pools drain on their own.
        ``report=False`` skips the stats lines, which from a finalizer
        would land in whatever the process prints then."""
        if getattr(self, "_closed", False):
            return
        self._closed = True
        for pool in (getattr(self, "_seed_pool", None),
                     getattr(self, "_seed_copy_pool", None)):
            if pool is not None:
                pool.shutdown(wait=wait)
        rw = getattr(self, "_rworker", None)
        if getattr(self, "_gpu_resident", False):
            import mlx_kquant as kq

            if getattr(kq, "residency_erase", None):
                # Queued behind the worker's changes, in their order.
                drops = [(a[0], None) for a in self._arena.values()]
                if rw is None or not rw.put(drops):
                    for a, _ in drops:
                        kq.residency_erase(a)
                    kq.residency_commit()
            self._gpu_resident = False
        if rw is not None:
            rw.stop(wait)
        if getattr(self, "_routed_log", None):
            lis = np.array([li for li, _ in self._routed_log], np.uint16)
            rows = [ids for _, ids in self._routed_log]
            offs = np.zeros(len(rows) + 1, np.int64)
            np.cumsum([len(r) for r in rows], out=offs[1:])
            np.savez(
                self._routed_log_path, li=lis, offsets=offs,
                ids=np.concatenate(rows) if rows else
                np.zeros(0, np.uint16))
            print(
                f"[stream] routed-id trace: {len(rows)} stage calls -> "
                f"{self._routed_log_path}")
            self._routed_log = None
        if not report or not getattr(self, "_stats_verbose", True):
            self._print_wedges(wait=wait)
            return
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
        if getattr(self, "_calls", 0):
            wall = time.monotonic() - self._t_start
            print(
                f"[stream] decode feeder stalls: demand reads "
                f"{self._t_demand:.1f}s, prestage settle "
                f"{self._t_settle:.1f}s, over {wall:.0f}s wall"
            )
            print(
                f"[stream] decode feeder bytes read: "
                f"{getattr(self, '_bytes_read', 0) / 1e9:.1f} GB")
        if getattr(self, "_rule_split", 0) or getattr(self, "_rule_ring", 0):
            print(
                f"[stream] arena split rule: {self._rule_split} layer calls "
                f"split, {self._rule_ring} took the prefill ring "
                f"({self._rule_ahead} already staged ahead)")
        if getattr(self, "_seeded", 0):
            print(
                f"[stream] decode feeder seeded {self._seeded} experts "
                f"({self._seed_bytes / 1e9:.1f} GB) from the prefill ring, "
                f"copies {self._t_seed:.1f}s")
        if getattr(self, "_ring_lends", 0):
            print(
                f"[stream] decode arena lent layers to the prefill ring at "
                f"{self._ring_lends} prefills")
        if getattr(self, "_shed_tokens", 0):
            print(
                f"[stream] miss-shed: {self._shed_n} experts shed over "
                f"{self._shed_tokens} token-layers (avg "
                f"{100 * self._shed_mass / self._shed_tokens:.1f}% of "
                "routed mass where shed)")
        if getattr(self, "_layer_shed_n", 0):
            print(
                f"[stream] layer-shed: {self._layer_shed_n} token-layer "
                "routed paths shed")
        self._print_wedges(wait=wait)

    def _print_wedges(self, wait: bool = True) -> None:
        wedges = getattr(self, "_wedges", 0)
        if wedges:
            print(
                f"[stream] decode feeder: {wedges} read(s) wedged in the "
                "kernel this run; the affected experts were dropped")
        for pool in (getattr(self, "_read_pool", None),
                     getattr(self, "_la_pool", None)):
            if pool is not None:
                # A wedged worker never returns; joining it would hang exit.
                # The finalizer passes wait=False: a collection that runs
                # on a read worker cannot join its own thread.
                pool.shutdown(wait=wait and wedges == 0)
        with getattr(self, "_lock_mu", threading.RLock()):
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
            self.close(wait=False, report=False)
        except BaseException:  # noqa: BLE001, S110 - incl. ^C during interpreter exit
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
        if f is None:
            return
        try:
            f.close()
        except KeyboardInterrupt:
            # ^C while already exiting: skip the rest of the cleanup
            # (munlock/stat printing) - the OS reclaims it all anyway.
            pass

    atexit.register(_close_at_exit)


def maybe_make_decode_feeder(
    offsets, modules, arena_bytes: int, stats_verbose: bool | None = None,
    lend_bytes: int = 0, fast_disk: str | None = None, ring_share: int = 0,
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
        feeder = DecodeFeeder(
            offsets, modules, arena_bytes, stats_verbose, lend_bytes=lend_bytes,
            fast_disk=fast_disk, ring_share=ring_share)
        # CLI runs never tear the feeder down explicitly and __del__ is
        # not reliable at interpreter exit, so the hit-rate/prestage stats
        # lines would silently vanish; close() is idempotent, so the
        # server's explicit unload close makes this a no-op.
        _register_exit_close(feeder)
        return feeder
    except (RuntimeError, OSError, ImportError, MemoryError) as e:
        print(f"[stream] decode feeder unavailable ({e}); "
              "decode stays on the CPU page-cache path")
        return None
