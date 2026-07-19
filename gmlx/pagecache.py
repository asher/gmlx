"""Exit-time page-cache release for larger-than-RAM models (Darwin).

A model larger than RAM cycles its whole footprint through the unified
buffer cache on every decode pass, so the cached subset alive at process
exit is a random slice of the file: useless as a warm set for the next
load, but expensive for whoever faults next - the kernel reclaims those
pages one at a time inside the fault path, a compute-side decode tax with
no I/O-counter signature (measured 0.94 -> 1.44 tok/s on GLM-5.2 after a
sweep). ``msync(MS_INVALIDATE)`` over the shard files returns the cached
pages to the free list in one call per file. Models that fit in RAM are
left alone: their remnant is exactly what makes the next load warm.

``GMLX_RELEASE_PAGECACHE=0`` opts out.
"""

from __future__ import annotations

import atexit
import ctypes
import os
import sys

import numpy as np

from . import loadlog

_MS_INVALIDATE = 0x0002
_PROT_READ = 0x1
_MAP_SHARED = 0x0001

_libc = None


def _get_libc():
    global _libc
    if _libc is None:
        libc = ctypes.CDLL(None, use_errno=True)
        libc.mmap.restype = ctypes.c_void_p
        libc.mmap.argtypes = [
            ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int,
            ctypes.c_int, ctypes.c_int, ctypes.c_int64,
        ]
        libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        libc.msync.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
        libc.mincore.argtypes = [
            ctypes.c_void_p, ctypes.c_size_t, ctypes.c_char_p,
        ]
        _libc = libc
    return _libc


def _page_size() -> int:
    return os.sysconf("SC_PAGE_SIZE")


def _physical_ram_bytes() -> int | None:
    try:
        return os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (ValueError, OSError):
        return None


def _mapped(path):
    """(addr, size, fd) for a whole-file read-only mapping, or None."""
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return None
    size = os.fstat(fd).st_size
    if not size:
        os.close(fd)
        return None
    libc = _get_libc()
    addr = libc.mmap(None, size, _PROT_READ, _MAP_SHARED, fd, 0)
    if not addr or addr == ctypes.c_void_p(-1).value:
        os.close(fd)
        return None
    return addr, size, fd


def resident_file_bytes(path: str) -> int:
    """Bytes of ``path`` currently in the unified buffer cache."""
    if sys.platform != "darwin":
        return 0
    m = _mapped(path)
    if m is None:
        return 0
    addr, size, fd = m
    libc = _get_libc()
    page = _page_size()
    n = (size + page - 1) // page
    vec = ctypes.create_string_buffer(n)
    resident = 0
    if libc.mincore(ctypes.c_void_p(addr), size, vec) == 0:
        resident = int(
            (np.frombuffer(vec, dtype=np.uint8, count=n) & 1).sum()
        ) * page
    libc.munmap(ctypes.c_void_p(addr), size)
    os.close(fd)
    return resident


def release_file_cache(paths, log=None) -> int:
    """Evict each file's cached pages from the UBC via msync(MS_INVALIDATE).
    Returns the bytes that were resident beforehand and actually swept."""
    if sys.platform != "darwin":
        return 0
    libc = _get_libc()
    released = 0
    for path in paths:
        resident = resident_file_bytes(path)
        if not resident:
            continue
        m = _mapped(path)
        if m is None:
            continue
        addr, size, fd = m
        rc = libc.msync(ctypes.c_void_p(addr), size, _MS_INVALIDATE)
        libc.munmap(ctypes.c_void_p(addr), size)
        os.close(fd)
        if rc == 0:
            released += resident
    if log is not None and released:
        log(
            f"[pagecache] released {released / 1e9:.1f} GB of streaming-model "
            f"cache back to the free list (GMLX_RELEASE_PAGECACHE=0 disables)"
        )
    return released


_groups: list[list[str]] = []
_hook_installed = False
_log_release = False  # sweep chatter is --verbose only; captured at register


def _exit_sweep() -> None:
    paths: list[str] = []
    for group in _groups:
        paths.extend(p for p in group if p not in paths)
    del _groups[:]
    release_file_cache(paths, log=print if _log_release else None)


def register_streaming_release(paths) -> None:
    """Register shard paths for a page-cache sweep, if they exceed RAM.

    Called per model load; the >RAM gate keeps fully-resident models warm
    for their next load while larger-than-RAM ones sweep their remnant -
    at process exit, or earlier via ``release_streaming_for`` when a
    long-lived server unloads the model.
    """
    global _hook_installed, _log_release
    # Registration happens inside the load session; the sweep runs long
    # after it ends, so capture the session's verbosity now.
    _log_release = _log_release or loadlog.is_verbose()
    if sys.platform != "darwin":
        return
    if os.environ.get("GMLX_RELEASE_PAGECACHE", "1") == "0":
        return
    ram = _physical_ram_bytes()
    if ram is None:
        return
    try:
        total = sum(os.path.getsize(p) for p in paths)
    except OSError:
        return
    if total <= ram:
        return
    group = [str(p) for p in paths]
    if group not in _groups:
        _groups.append(group)
    if not _hook_installed:
        atexit.register(_exit_sweep)
        _hook_installed = True


def release_streaming_for(path) -> int:
    """Sweep the registered shard group containing ``path`` right now - the
    server unload seam, where waiting for process exit would leave the
    remnant taxing the next model's load. Returns the bytes released; 0
    when ``path`` was never registered (fits in RAM, opted out, or not
    loaded by this process)."""
    target = os.path.abspath(str(path))
    for group in list(_groups):
        if any(os.path.abspath(p) == target for p in group):
            _groups.remove(group)
            return release_file_cache(
                group, log=print if _log_release else None
            )
    return 0
