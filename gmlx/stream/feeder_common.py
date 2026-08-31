"""Plumbing shared by the prefill and decode expert feeders (feeder.py /
decode_feeder.py): the expert-stack naming, the short-read-safe pread loop,
the zero-copy verification, and the weight-swap protocol. The two feeders'
staging designs are deliberately different (two-slot ring vs popularity
arena); only the pieces below are common."""

from __future__ import annotations

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
