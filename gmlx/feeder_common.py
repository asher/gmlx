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
        # to avoid.
        with mx.stream(mx.cpu):
            head = bytes(np.array(w.reshape(-1)[:4096]))
        if os.pread(fds[path], len(head), off) != head:
            raise RuntimeError(
                f"layer {li} {kind} stack is not a zero-copy view of its "
                "GGUF range (loader transformed the bytes)"
            )


@contextmanager
def swapped_weights(entry: dict, views: dict):
    """Swap each module's expert weight to ``views[kind]`` for the call body,
    restoring the originals on exit. ``entry`` is a layer's
    ``{kind: (module, ...)}`` mapping."""
    saved = []
    try:
        for kind, (mod, *_) in entry.items():
            proj = getattr(mod, ATTRS[kind])
            saved.append((proj, proj.weight))
            proj.weight = views[kind]
        yield
    finally:
        for proj, w in saved:
            proj.weight = w
