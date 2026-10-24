"""Kernel-side memory truth for the governor (macOS host_statistics64).

MLX's counters describe the process: active bytes, and a buffer cache
the allocator reuses so it reads as free from inside the process. The
kernel sees that cache as wired. Metal allocations never swap and never
enter the compressor, so a server that grows past the box's free pages
livelocks macOS into a watchdog panic while every MLX-side number still
reads healthy. This module reads the counters that actually predict
the panic: the memory not wired (free + pageable anonymous + file-backed
pages), and threads waiting on free pages while the kernel reclaims
nothing. ``reclaimable_bytes`` (free + purgeable + file-backed) remains
for the readers that must not count other apps' memory.

Darwin only; ``reclaimable_bytes`` returns None elsewhere or when the
struct layout fails its one-time self-check against sysctl.
"""

from __future__ import annotations

import ctypes
import logging
import os
import subprocess
import sys
import time

_log = logging.getLogger(__name__)

_PAGE = os.sysconf("SC_PAGESIZE") if hasattr(os, "sysconf") else 16384
_GB = 1 << 30


class _VMStat64(ctypes.Structure):
    _fields_ = [
        ("free_count", ctypes.c_uint32),
        ("active_count", ctypes.c_uint32),
        ("inactive_count", ctypes.c_uint32),
        ("wire_count", ctypes.c_uint32),
        ("zero_fill_count", ctypes.c_uint64),
        ("reactivations", ctypes.c_uint64),
        ("pageins", ctypes.c_uint64),
        ("pageouts", ctypes.c_uint64),
        ("faults", ctypes.c_uint64),
        ("cow_faults", ctypes.c_uint64),
        ("lookups", ctypes.c_uint64),
        ("hits", ctypes.c_uint64),
        ("purges", ctypes.c_uint64),
        ("purgeable_count", ctypes.c_uint32),
        ("speculative_count", ctypes.c_uint32),
        ("decompressions", ctypes.c_uint64),
        ("compressions", ctypes.c_uint64),
        ("swapins", ctypes.c_uint64),
        ("swapouts", ctypes.c_uint64),
        ("compressor_page_count", ctypes.c_uint32),
        ("throttled_count", ctypes.c_uint32),
        ("external_page_count", ctypes.c_uint32),
        ("internal_page_count", ctypes.c_uint32),
        ("total_uncompressed", ctypes.c_uint64),
    ]


_HOST_VM_INFO64 = 4
_libc = None
_host = None
_disabled: str | None = None if sys.platform == "darwin" else "not darwin"


def _bind():
    global _libc, _host
    if _libc is None:
        _libc = ctypes.CDLL(None, use_errno=True)
        _host = _libc.mach_host_self()


def snapshot() -> dict | None:
    """Raw byte counters, or None when unavailable/disabled."""
    if _disabled is not None:
        return None
    try:
        _bind()
        st = _VMStat64()
        count = ctypes.c_uint(ctypes.sizeof(_VMStat64) // 4)
        rc = _libc.host_statistics64(_host, _HOST_VM_INFO64,
                                     ctypes.byref(st), ctypes.byref(count))
        if rc != 0:
            return None
    except Exception:
        return None
    return {
        "free": st.free_count * _PAGE,
        "inactive": st.inactive_count * _PAGE,
        "purgeable": st.purgeable_count * _PAGE,
        "speculative": st.speculative_count * _PAGE,
        "filebacked": st.external_page_count * _PAGE,
        "wired": st.wire_count * _PAGE,
        "compressor": st.compressor_page_count * _PAGE,
    }


def reclaimable_bytes() -> int | None:
    """free + purgeable + file-backed, the sum ``live_reclaimable_bytes``
    reads. Mach's free and file-backed counts both hold the speculative
    pages, so free gives them up and they count once. File-backed pages
    are included on purpose: a model load legitimately drains free
    memory into the page cache (clean pages, reclaimable), while at a
    real freeze file-backed has collapsed too."""
    s = snapshot()
    if s is None:
        return None
    return s["free"] - s["speculative"] + s["purgeable"] + s["filebacked"]


# The sysctl page counts of live_reclaimable_bytes. vm.page_free_count
# leaves out the speculative pages, which Mach's free_count includes, and
# vm.page_pageable_external_count holds them, as file-backed read-ahead.
_LIVE_COUNTS = ("vm.page_free_count", "vm.page_purgeable_count",
                "vm.page_pageable_external_count")


def _sysctl_count(name: str) -> int | None:
    """One integer sysctl, read in process, or None when it fails."""
    try:
        _bind()
        value = ctypes.c_uint64(0)
        size = ctypes.c_size_t(ctypes.sizeof(value))
        rc = _libc.sysctlbyname(name.encode(), ctypes.byref(value),
                                ctypes.byref(size), None, ctypes.c_size_t(0))
    except Exception:
        return None
    if rc != 0:
        return None
    if size.value == 4:
        return value.value & 0xFFFFFFFF
    return value.value if size.value == 8 else None


def live_reclaimable_bytes() -> int | None:
    """free + purgeable + file-backed pages, from the sysctl counts at
    the moment of the call. host_statistics64 refreshes its counters
    about once a second, and a warm wire of tens of GB commits within one
    refresh, so a reader that must see a wire's own pages reads this.
    The speculative pages count once, in the file-backed count. None off
    Darwin or when a read fails."""
    if sys.platform != "darwin":
        return None
    pages = 0
    for name in _LIVE_COUNTS:
        n = _sysctl_count(name)
        if n is None:
            return None
        pages += n
    return pages * _PAGE


# The sysctl page counts of unwired_bytes: every page a wire can take,
# by dropping file pages or compressing other apps' memory.
_UNWIRED_COUNTS = ("vm.page_free_count", "vm.page_pageable_internal_count",
                   "vm.page_pageable_external_count")


def unwired_bytes() -> int | None:
    """Pages neither wired nor holding compressed data, read live from the
    sysctl counts: free, and the pageable anonymous and file-backed pages
    the kernel can compress or drop. A wire that stays under them leaves
    the kernel pages to reclaim. Firmware carve-outs (hw.memsize less
    hw.memsize_usable) and a few kernel pages fall outside all three
    counts, so this reads up to about 2 GB under RAM less the wired and
    compressor pages. None off Darwin or when a read fails."""
    if sys.platform != "darwin":
        return None
    pages = 0
    for name in _UNWIRED_COUNTS:
        n = _sysctl_count(name)
        if n is None:
            return None
        pages += n
    return pages * _PAGE


def free_page_waiters() -> int | None:
    """Threads waiting on the kernel for free pages
    (vm.page_free_wanted), live. Above zero, they wait on reclaim. None
    off Darwin or when the read fails."""
    if sys.platform != "darwin":
        return None
    return _sysctl_count("vm.page_free_wanted")


# Cumulative live counts that move while the kernel reclaims anonymous
# memory: bytes fed to the compressor, and compressor segments swapped out.
_RECLAIM_COUNTS = ("vm.compressor_input_bytes",
                   "vm.compressor.swapper.swapouts_total")


def reclaim_marks() -> tuple | None:
    """The reclaim counts, live. A later reading that differs shows the
    kernel compressed or swapped memory in between. None off Darwin or
    when a read fails."""
    if sys.platform != "darwin":
        return None
    marks = tuple(_sysctl_count(n) for n in _RECLAIM_COUNTS)
    return None if None in marks else marks


class StarveWatch:
    """Seconds threads have waited on the kernel for free pages while the
    kernel shows no reclaim: nothing compressed or swapped out, and no
    fewer threads waiting than at any reading of the streak. Waiters
    while it compresses other apps' memory are the kernel at work, and
    our own readers can be among them, so only a stall counts. 0 while
    no thread waits or the counts cannot be read."""

    def __init__(self):
        self.since = None
        self.marks = None
        self.low = None

    def reset(self) -> None:
        self.since = self.marks = self.low = None

    def __call__(self) -> float:
        waiters = free_page_waiters()
        if not waiters:
            self.reset()
            return 0.0
        now = time.monotonic()
        marks = reclaim_marks()
        if (self.since is None or marks != self.marks
                or (self.low is not None and waiters < self.low)):
            self.since = now
        self.marks = marks
        self.low = waiters if self.low is None else min(self.low, waiters)
        return now - self.since


def available_bytes() -> int | None:
    """free + inactive, what mlx-vlm's ``_free_ram_bytes`` parses out of
    ``vm_stat``, read in process. A fork beside a Metal-mapped buffer
    copies the buffer before the exec, so the serve process never shells
    out for this."""
    s = snapshot()
    if s is None:
        return None
    return s["free"] + s["inactive"]


def wired_bytes() -> int | None:
    s = snapshot()
    return None if s is None else s["wired"]


def selfcheck() -> str | None:
    """Cross-check the ctypes layout against sysctl once. Returns None
    when the sampler is trustworthy, else the reason it is disabled.
    Mach free_count includes speculative pages; vm.page_free_count does
    not. Free moves GBs/s on a busy box and the two reads are not
    simultaneous, so a few fresh pairs are tried."""
    global _disabled
    if _disabled is not None:
        return _disabled
    import time

    for _ in range(5):
        s = snapshot()
        if s is None:
            _disabled = "host_statistics64 unavailable"
            return _disabled
        try:
            # close_fds=False with an absolute path is posix_spawn: no
            # fork, so nothing Metal-mapped is copied.
            ref = int(subprocess.run(
                ["/usr/sbin/sysctl", "-n", "vm.page_free_count"],
                capture_output=True, text=True, timeout=5,
                close_fds=False).stdout) * _PAGE
        except Exception as e:
            _disabled = f"sysctl cross-check failed: {e}"
            return _disabled
        free_b = s["free"] - s["speculative"]
        if abs(free_b - ref) <= max(2 * _GB, 0.1 * ref):
            return None
        time.sleep(0.2)
    _disabled = (f"vm_statistics64 layout mismatch: ctypes free "
                 f"{free_b / _GB:.1f} GB vs sysctl {ref / _GB:.1f} GB")
    return _disabled
