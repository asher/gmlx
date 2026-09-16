"""Drop the streamed expert stacks' Metal buffers once the feeders own them.

Every no-copy view of the GGUF is a Metal buffer, and the driver's work
before each command buffer grows with the process's GPU-visible total
once that total passes physical RAM: 50-130 ms per large command buffer
on an M3 Max, untouched buffers included. A streamed model sits past RAM
by construction - the stacks alone are over the wired budget, and the
arena and the ring come on top. Both feeders read the stacks from the
file, so after install the arrays only hold the mapping.

``unmap_stacks`` swaps each covered stack for a one-row placeholder that
keeps the shape and the byte count the feeders and the budget read. The
offload wrapper maps a layer back through ``kq.load_gguf`` for the one
call that has no feeder path (arena overflow, a wedged read): a mapping,
not a read, and dropped again with the call's graph.

``GMLX_STREAM_UNMAP_STACKS=0`` keeps the views.
"""
from __future__ import annotations

import math
from contextlib import contextmanager

import mlx.core as mx

from .feeder_common import ATTRS, swapped_weights

STAMP = "_kq_stacks_unmapped"   # module attr: {kind: (path, name, shape)}


def placeholder(w):
    """A view with ``w``'s shape, dtype and ``nbytes`` over one row."""
    p = mx.broadcast_to(
        mx.zeros((1,) + tuple(w.shape[1:]), dtype=w.dtype), w.shape)
    mx.eval(p)
    return p


def _names_by_offset(path: str) -> dict[int, str]:
    from gmlx.load.headerscan import scan_gguf

    h = scan_gguf(path, array_limit=0)
    return {h.data_offset + t.offset: t.name for t in h.tensors}


def unmap_stacks(layers: dict) -> tuple[int, int]:
    """Replace the stacks of ``layers`` (a feeder's ``{li: {kind: (module,
    path, off, nbytes)}}``) with placeholders. Returns the layer and byte
    counts unmapped."""
    names: dict[str, dict[int, str]] = {}
    n_layers = 0
    nbytes = 0
    for li, entry in layers.items():
        stamps: dict[int, tuple] = {}
        for kind, (mod, path, off, nb) in entry.items():
            by_off = names.get(path)
            if by_off is None:
                by_off = names[path] = _names_by_offset(path)
            name = by_off.get(off)
            if name is None:
                raise RuntimeError(
                    f"layer {li} {kind} stack: no tensor at offset {off} "
                    f"of {path}")
            proj = getattr(mod, ATTRS[kind])
            w = proj.weight
            files = stamps.setdefault(id(mod), (mod, {}))[1]
            files[kind] = (path, name, tuple(w.shape))
            proj.weight = placeholder(w)
            nbytes += nb
        for mod, files in stamps.values():
            object.__setattr__(mod, STAMP, files)
        n_layers += 1
    return n_layers, nbytes


def _remap(files: dict) -> dict:
    import mlx_kquant as kq

    by_path: dict[str, dict] = {}
    views = {}
    for kind, (path, name, shape) in files.items():
        arrays = by_path.get(path)
        if arrays is None:
            arrays = by_path[path] = kq.load_gguf(path, True)[0]
        a = arrays[name]
        if tuple(a.shape) != shape:
            if a.size != math.prod(shape):
                raise RuntimeError(
                    f"{name}: file shape {a.shape} vs module {shape}")
            a = a.reshape(shape)
        views[kind] = a
    return views


@contextmanager
def stacks_remapped(mod):
    """The module's stacks mapped back from the file for the call body;
    a no-op on a module whose stacks were never unmapped."""
    files = getattr(mod, STAMP, None)
    if files is None:
        yield
        return
    entry = {kind: (mod,) for kind in files}
    with swapped_weights(entry, _remap(files)):
        yield
