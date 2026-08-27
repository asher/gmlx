"""Kernel-side memory truth for the governor (macOS host_statistics64).

MLX's counters describe the process: active bytes, and a buffer cache
the allocator reuses so it reads as free from inside the process. The
kernel sees that cache as wired. Metal allocations never swap and never
enter the compressor, so a server that grows past the box's free pages
livelocks macOS into a watchdog panic while every MLX-side number still
reads healthy. This module reads the counters that actually predict
the panic: free + purgeable + speculative + file-backed pages, the
same reclaimable sum the lab's external wired-watchdog uses.

Darwin only; ``reclaimable_bytes`` returns None elsewhere or when the
struct layout fails its one-time self-check against sysctl.
"""

from __future__ import annotations

import ctypes
import logging
import os
import subprocess
import sys

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
        "purgeable": st.purgeable_count * _PAGE,
        "speculative": st.speculative_count * _PAGE,
        "filebacked": st.external_page_count * _PAGE,
        "wired": st.wire_count * _PAGE,
        "compressor": st.compressor_page_count * _PAGE,
    }


def reclaimable_bytes() -> int | None:
    """free + purgeable + speculative + file-backed. File-backed pages
    are included on purpose: a model load legitimately drains free
    memory into the page cache (clean pages, reclaimable), while at a
    real freeze file-backed has collapsed too."""
    s = snapshot()
    if s is None:
        return None
    return s["free"] + s["purgeable"] + s["speculative"] + s["filebacked"]


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
            ref = int(subprocess.run(
                ["sysctl", "-n", "vm.page_free_count"],
                capture_output=True, text=True, timeout=5).stdout) * _PAGE
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
