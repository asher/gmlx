"""Plumbing shared by the prefill and decode expert feeders (feeder.py /
decode_feeder.py): the expert-stack naming, the short-read-safe pread loop,
the zero-copy verification, and the weight-swap protocol. The two feeders'
staging designs are deliberately different (two-slot ring vs popularity
arena); only the pieces below are common."""

from __future__ import annotations

import mmap
import os
from contextlib import contextmanager

import numpy as np

KINDS = ("gate", "up", "down")
ATTRS = {k: f"{k}_proj" for k in KINDS}


def read_range(fd, mv, off: int) -> None:
    """pread ``len(mv)`` bytes at ``off`` into ``mv``, retrying short reads."""
    done, n = 0, len(mv)
    while done < n:
        r = os.preadv(fd, [mv[done:]], off + done)
        if r <= 0:
            raise OSError(f"short read at offset {off + done}")
        done += r


def read_range_aligned(fd, mv, off: int, file_size: int) -> None:
    """``read_range`` with every pread page-aligned. The kernel services
    an F_NOCACHE read at an unaligned offset through the page cache, and
    that path can wedge under pressure (see decode_feeder). The aligned
    middle lands in ``mv`` directly; the head and tail partial pages go
    through a scratch page."""
    n = len(mv)
    if n == 0:
        return
    page = mmap.PAGESIZE
    end = off + n
    a = off & ~(page - 1)
    b = min((end + page - 1) & ~(page - 1), file_size)
    if a == off and b == end:
        read_range(fd, mv, off)
        return
    head = off - a
    a_in = a + page if head else a
    if a_in >= end:
        buf = bytearray(b - a)
        read_range(fd, memoryview(buf), a)
        mv[:] = buf[head:head + n]
        return
    if head:
        buf = bytearray(page)
        read_range(fd, memoryview(buf), a)
        mv[:a_in - off] = buf[head:]
    b_in = end & ~(page - 1)
    if b_in > a_in:
        read_range(fd, mv[a_in - off:b_in - off], a_in)
    if b_in < end:
        buf = bytearray(b - b_in)
        read_range(fd, memoryview(buf), b_in)
        mv[b_in - off:] = buf[:end - b_in]


def lock_pages(mv) -> tuple[int, int] | None:
    """mlock the pages behind ``mv``. The (address, length) entry for
    ``unlock_pages``, or None when the platform or the kernel refuses."""
    import ctypes

    try:
        libc = ctypes.CDLL(None, use_errno=True)
        addr = ctypes.addressof(ctypes.c_char.from_buffer(mv))
    except (OSError, TypeError, ValueError):
        return None
    n = len(mv)
    if libc.mlock(ctypes.c_void_p(addr), ctypes.c_size_t(n)) != 0:
        return None
    return addr, n


def unshare_on_fork(mv) -> bool:
    """minherit(VM_INHERIT_NONE) on the whole pages inside ``mv``: a
    forked child does not map them, so a spawn copies nothing. Without
    it a fork copies every Metal-mapped buffer before the exec, the
    arena included. Only pages fully inside the buffer: a page shared with other data would vanish from
    the child too, and it dies before the exec. False when the platform
    refuses or no whole page fits."""
    import ctypes

    try:
        libc = ctypes.CDLL(None, use_errno=True)
        addr = ctypes.addressof(ctypes.c_char.from_buffer(mv))
        minherit = libc.minherit
    except (OSError, TypeError, ValueError, AttributeError):
        return False
    start, end = inner_pages(addr, len(mv))
    if end <= start:
        return False
    return minherit(ctypes.c_void_p(start), ctypes.c_size_t(end - start),
                    _VM_INHERIT_NONE) == 0


def inner_pages(addr: int, n: int) -> tuple[int, int]:
    """The page-aligned range fully inside ``[addr, addr + n)``."""
    page = mmap.PAGESIZE
    start = (addr + page - 1) & ~(page - 1)
    end = (addr + n) & ~(page - 1)
    return start, end


_VM_INHERIT_NONE = 2


def unlock_pages(entry: tuple[int, int]) -> None:
    import ctypes

    try:
        libc = ctypes.CDLL(None, use_errno=True)
    except OSError:
        return
    addr, n = entry
    libc.munlock(ctypes.c_void_p(addr), ctypes.c_size_t(n))


def verify_zero_copy(li: int, entries, fds: dict[str, int]) -> None:
    """The swap trick assumes each module weight is a zero-copy view of its
    file range: staged file bytes must be exactly what the compute would have
    read through the mmap. ``entries`` yields ``(kind, module, path, off)``
    for one layer; a head sample of each stack is compared byte-for-byte, so
    any loader-side transform disables the feeder rather than corrupting
    compute."""
    import mlx.core as mx

    for kind, mod, path, off in entries:
        w = getattr(mod, ATTRS[kind]).weight
        # CPU stream: a GPU slice of the file-backed stack would make the
        # driver page the referenced range in - the cost the feeders exist
        # to avoid. Flattening the whole stack would overflow the int32
        # shape dims on a >2 GiB stack, so slice off just enough leading
        # experts to cover the head sample before flattening.
        per = 1
        for d in w.shape[1:]:
            per *= d
        lead = min(w.shape[0], -(-4096 // max(per, 1)))
        with mx.stream(mx.cpu):
            head = bytes(np.array(w[:lead].reshape(-1)[:4096]))
        if os.pread(fds[path], len(head), off) != head:
            raise RuntimeError(
                f"layer {li} {kind} stack is not a zero-copy view of its "
                "GGUF range (loader transformed the bytes)"
            )


def slot_itemsize(nbytes_max: int, last_dims) -> int:
    """Arena granule for a slot: 1 (plain uint8) whenever the slot fits
    int32 shape dims; past that, the widest unsigned itemsize that divides
    every layer's wire row length (so the byte view can land back on each
    geometry) and brings the element count back under int32. Returns 0
    when no granule works (caller refuses)."""
    if nbytes_max <= 2**31 - 1:
        return 1
    for w in (8, 4, 2):
        if any(d % w for d in last_dims):
            continue
        if -(-nbytes_max // w) <= 2**31 - 1:
            return w
    return 0


def slot_view(arr, nbytes: int, shape):
    """First ``nbytes`` of a flat arena array as a zero-copy uint8 view of
    ``shape``. Wide arenas (uint16/32/64, from ``slot_itemsize``) slice at
    their granule and view back - still buffer-sharing."""
    import mlx.core as mx

    w = arr.itemsize
    if w == 1:
        return arr[:nbytes].reshape(shape)
    wide = arr[: nbytes // w].reshape(tuple(shape[:-1]) + (shape[-1] // w,))
    return mx.view(wide, mx.uint8)


def _expert_loras(proj):
    """Every expert LoRA stamped on a projection: the first adapter and any
    further slots (one resident base serving several adapted ids)."""
    lo = getattr(proj, "_kq_lora", None)
    if lo is None:
        return ()
    return (lo,) + tuple(getattr(proj, "_kq_lora_extra", None) or ())


@contextmanager
def swapped_weights(entry: dict, views: dict, slot_owner=None):
    """Swap each module's expert weight to ``views[kind]`` for the call body,
    restoring the originals on exit. ``entry`` is a layer's
    ``{kind: (module, ...)}`` mapping.

    ``slot_owner`` (a callable returning the slot -> expert id table of the
    swapped views, negative for empty / zeroed slots) rides along on a
    projection carrying a live expert LoRA (``_kq_lora``): under the swap
    the container receives slot ids, and the delta maps them back to the
    expert whose weights the slot holds. ``None`` (the prefill ring stages
    the whole stack in expert order) leaves the ids as they are."""
    saved = []
    try:
        for kind, (mod, *_) in entry.items():
            proj = getattr(mod, ATTRS[kind])
            saved.append((proj, proj.weight))
            proj.weight = views[kind]
            for lo in _expert_loras(proj):
                lo.owner = slot_owner
        yield
    finally:
        for proj, w in saved:
            proj.weight = w
            for lo in _expert_loras(proj):
                lo.owner = None
