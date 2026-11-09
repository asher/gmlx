"""Plumbing shared by the prefill and decode expert feeders (feeder.py /
decode_feeder.py): the expert-stack naming, the short-read-safe pread loop,
the zero-copy verification, and the weight-swap protocol. The two feeders'
staging designs are deliberately different (two-slot ring vs popularity
arena); only the pieces below are common."""

from __future__ import annotations

import mmap
import os
import queue
import threading
from concurrent.futures import Future
from contextlib import contextmanager
from typing import NamedTuple

import numpy as np

KINDS = ("gate", "up", "down")
ATTRS = {k: f"{k}_proj" for k in KINDS}


class FeederRange(NamedTuple):
    """One kind's expert bytes in the file: expert ``e`` is ``length``
    bytes at ``off + e * step``. ``half`` is 0 (gate) or 1 (up) for a cut
    of a fused gate_up tensor at ``tensor_off``, else None."""
    path: str
    off: int
    step: int
    length: int
    n_exp: int
    kind: str
    tensor_off: int
    half: int | None


def feeder_ranges(ranges) -> list[FeederRange] | None:
    """A layer's raw ``(path, off, nbytes, n_exp, kind)`` ranges as one
    gate, up and down range each, or None when the feeders cannot cover
    the layer. A fused gate_up range splits at each expert's row midpoint,
    as the loader splits the tensor."""
    return ranges_or_reason(ranges)[0]


def ranges_or_reason(ranges) -> tuple[list[FeederRange] | None, str | None]:
    """``feeder_ranges`` and, when it is None, why, for the feeders'
    refusal."""
    out = []
    for path, off, nbytes, n_exp, kind in ranges:
        if not n_exp:
            return None, "an expert stack with no per-expert layout"
        step = nbytes // n_exp
        if kind == "gate_up":
            if step % 2:
                return None, "a fused gate_up stack of odd size"
            half = step // 2
            out.append(FeederRange(path, off, step, half, n_exp, "gate",
                                   off, 0))
            out.append(FeederRange(path, off + half, step, half, n_exp,
                                   "up", off, 1))
        else:
            out.append(FeederRange(path, off, step, step, n_exp, kind,
                                   off, None))
    if sorted(r.kind for r in out) != sorted(KINDS):
        kinds = "/".join(sorted(r[4] for r in ranges))
        return None, f"expert stacks {kinds}, not gate/up/down"
    if len({r.n_exp for r in out}) != 1:
        return None, "expert stacks with different expert counts"
    return out, None


def no_layers_error(skipped: dict[str, int]) -> RuntimeError:
    """The feeders' refusal when no layer qualifies, with the reasons
    counted per layer."""
    if not skipped:
        return RuntimeError("no MoE expert layers")
    why = "; ".join(f"{reason} ({n} layer{'' if n == 1 else 's'})"
                    for reason, n in sorted(skipped.items(),
                                            key=lambda kv: -kv[1]))
    return RuntimeError(f"no layers the feeder can stream: {why}")


def kind_bytes(ranges) -> dict[str, int] | None:
    """Stack bytes per kind of a layer the feeders would cover, for
    sizing, or None. Needs no expert count: a fused gate_up range is half
    gate and half up."""
    out: dict[str, int] = {}
    for _, _, nbytes, _, kind in ranges:
        parts = ((("gate", nbytes // 2), ("up", nbytes // 2))
                 if kind == "gate_up" else ((kind, nbytes),))
        for k, nb in parts:
            if k in out:
                return None
            out[k] = nb
    return out if sorted(out) == sorted(KINDS) else None


def read_range(fd, mv, off: int) -> None:
    """pread ``len(mv)`` bytes at ``off`` into ``mv``, retrying short reads."""
    done, n = 0, len(mv)
    while done < n:
        r = os.preadv(fd, [mv[done:]], off + done)
        if r <= 0:
            raise OSError(f"short read at offset {off + done}")
        done += r


def copy_experts(dst, src, pairs, stride: int) -> None:
    """Copy expert ``src`` slot ``e`` to ``dst`` slot ``s`` for each
    (e, s) row of ``pairs``, ``stride`` bytes each."""
    for e, s in pairs:
        np.copyto(dst[s * stride:(s + 1) * stride],
                  src[e * stride:(e + 1) * stride])


# Size of a bounce buffer, the largest read through one. The copy out of
# it costs far less than the read.
BOUNCE_BYTES = 16 << 20
_bounce = threading.local()


def _bounce_buffer(nbytes: int) -> memoryview:
    """This thread's page-aligned bounce buffer, ``nbytes`` long."""
    buf = getattr(_bounce, "buf", None)
    if buf is None or len(buf) < nbytes:
        buf = _bounce.buf = mmap.mmap(-1, max(nbytes, mmap.PAGESIZE))
    return memoryview(buf)[:nbytes]


def read_range_aligned(fd, mv, off: int, file_size: int,
                       bounce: queue.Queue | None = None) -> None:
    """``read_range`` with every pread on the kernel's direct path: the
    file offset and the destination address page-aligned. An F_NOCACHE
    read that misses either is served through the page cache. A prefill
    ring pass then fills the cache at drive speed and the kernel
    compresses other memory to make room, and that path can wedge under
    pressure (see decode_feeder). The whole pages land in ``mv`` directly
    when ``mv`` and ``off`` share a page phase; the rest goes through a
    page-aligned bounce buffer: one taken for the call from ``bounce``, a
    queue of ``kq.arena_alloc`` (array, memoryview) pairs, or else this
    thread's own. A ring slot is page-aligned and a GGUF stack seldom
    starts on a page boundary, so in practice the ring bounces every
    byte."""
    mv = memoryview(mv).cast("B")
    n = len(mv)
    if n == 0:
        return
    end = off + n
    if end > file_size:
        raise OSError(
            f"read of {n} bytes at offset {off} runs past the end of the "
            f"file ({file_size} bytes)")
    page = mmap.PAGESIZE
    a = off & ~(page - 1)
    b = min((end + page - 1) & ~(page - 1), file_size)
    dst = np.frombuffer(mv, dtype=np.uint8)
    pieces = [(a, b)]
    if (dst.ctypes.data - off) % page == 0:
        d0 = (off + page - 1) & ~(page - 1)
        d1 = end & ~(page - 1)
        if d1 > d0:
            read_range(fd, mv[d0 - off:d1 - off], d0)
            pieces = [(p0, p1) for p0, p1 in ((a, d0), (d1, b)) if p1 > p0]
    if not pieces:
        return
    held = bounce.get() if bounce is not None else None
    try:
        step = len(held[1]) if held is not None else BOUNCE_BYTES
        for p0, p1 in pieces:
            for pos in range(p0, p1, step):
                stop = min(pos + step, p1)
                buf = (held[1][:stop - pos] if held is not None
                       else _bounce_buffer(stop - pos))
                read_range(fd, buf, pos)
                lo, hi = max(pos, off), min(stop, end)
                np.copyto(dst[lo - off:hi - off],
                          np.frombuffer(buf, dtype=np.uint8)[lo - pos:hi - pos])
    finally:
        if held is not None:
            bounce.put(held)


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
    read through the mmap. ``entries`` yields ``(kind, module, path, off,
    length)`` for one layer, ``length`` the bytes per expert; a head sample
    of each stack's first expert is compared byte-for-byte, so
    any loader-side transform disables the feeder rather than corrupting
    compute."""
    import mlx.core as mx

    for kind, mod, path, off, length in entries:
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
            head = bytes(np.array(w[:lead].reshape(-1)[:min(4096, length)]))
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
    marked = []
    try:
        for kind, (mod, *_) in entry.items():
            proj = getattr(mod, ATTRS[kind])
            saved.append((proj, proj.weight))
            proj.weight = views[kind]
            for lo in _expert_loras(proj):
                lo.owner = slot_owner
            if not getattr(mod, "_kq_weights_swapped", False):
                # The sorted-prefill gate+up concat (modules.py) is a copy
                # of the resident wire bytes in expert order; under a swap
                # the bound weights are a slot's bytes and the ids are slot
                # ids, so the block must gather from what is bound.
                object.__setattr__(mod, "_kq_weights_swapped", True)
                marked.append(mod)
        yield
    finally:
        for proj, w in saved:
            proj.weight = w
            for lo in _expert_loras(proj):
                lo.owner = None
        for mod in marked:
            object.__setattr__(mod, "_kq_weights_swapped", False)


class DaemonPool:
    """ThreadPoolExecutor stand-in whose workers are daemon threads. The
    stdlib pool's workers are joined at interpreter exit, so one read
    wedged in the kernel would hang process shutdown. Daemon workers die
    with the process instead. ``on_start`` runs once in each worker thread
    (the lookahead pool drops its disk-I/O priority there)."""

    def __init__(self, n: int, on_start=None, name: str = "gmlx-decode-read"):
        self._q: queue.Queue = queue.Queue()
        self._on_start = on_start
        self._name = name
        self._threads = []
        for i in range(n):
            t = threading.Thread(
                target=self._run, daemon=True, name=f"{name}-{i}")
            t.start()
            self._threads.append(t)

    def _run(self) -> None:
        if self._on_start is not None:
            try:
                self._on_start()
            except Exception:  # noqa: S110 - start hook is advisory
                pass
        while True:
            item = self._q.get()
            if item is None:
                return
            fut, fn, args = item
            if fut.set_running_or_notify_cancel():
                try:
                    fut.set_result(fn(*args))
                except BaseException as exc:
                    fut.set_exception(exc)
            # A worker blocked in get() is a live frame: locals kept
            # from the last item (the feeder's bound method, the
            # future) would pin the feeder and its modules past unload.
            del item, fut, fn, args

    def submit(self, fn, *args) -> Future:
        fut: Future = Future()
        self._q.put((fut, fn, args))
        return fut

    def grow(self) -> None:
        """Replace a worker lost to a wedged read."""
        t = threading.Thread(
            target=self._run, daemon=True,
            name=f"{self._name}-{len(self._threads)}")
        t.start()
        self._threads.append(t)

    def shutdown(self, wait: bool = True, cancel_futures: bool = False) -> None:
        if cancel_futures:
            # Work no worker has taken yet is dropped. Its future is
            # notified as a worker would, or wait() never returns on it.
            while True:
                try:
                    item = self._q.get_nowait()
                except queue.Empty:
                    break
                if item is not None:
                    item[0].cancel()
                    item[0].set_running_or_notify_cancel()
        for _ in self._threads:
            self._q.put(None)
        if wait:
            for t in self._threads:
                t.join()
