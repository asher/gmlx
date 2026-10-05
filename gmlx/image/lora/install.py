"""``ImageLoRALinear``, the leaf swap, and the pair cache.

An image adapter is applied live beside the base weight, which stays
K-quant wire bytes. ``ImageLoRALinear`` wraps one linear leaf of a DiT. On
each call it asks the pipeline's ``PatchState`` for the factors of its
module path under the active patch table. With none it calls the base
leaf and returns that result. With factors it adds
``(x @ A^T) @ B^T`` to the base output on stock ops, where strength and
``alpha / rank`` are folded into B.

The class is separate from the text adapter wrap in ``gmlx.load.modules``
and reads nothing from ``gmlx.lora_rows``. That channel is global to the
process and belongs to text rows, and an image model shares a process with
text models.

Pairs live in the pipeline's ``adapters`` cache under the sha256 of their
content, so a file read twice or two files that share a pair hold it once.
The leaf swap runs when the first patch table with entries is defined, so
a pipeline that never uses an adapter keeps its plain leaves.
"""

from __future__ import annotations

import hashlib
import json
import struct
import weakref
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from . import plan
from .reader import AdapterError

# How a table is applied: one block-diagonal pair over the whole weight.
_PLACEMENT = "block"


def placement() -> str:
    return _PLACEMENT


@dataclass(frozen=True)
class Pair:
    """One cached pair: ``down`` (rank, in), ``up`` (out, rank), alpha."""
    down: mx.array
    up: mx.array
    alpha: float | None

    @property
    def nbytes(self) -> int:
        return int(self.down.nbytes) + int(self.up.nbytes)


def _dtype_name(arr: mx.array) -> str:
    return str(arr.dtype).rsplit(".", 1)[-1]


def _raw(arr: mx.array) -> np.ndarray:
    if arr.dtype == mx.bfloat16:
        arr = arr.view(mx.uint16)
    return np.ascontiguousarray(np.array(arr))


def pair_hash(down: mx.array, up: mx.array, alpha: float | None) -> str:
    """sha256 of a pair's content: a JSON line of each part's dtype and
    shape and the alpha as a float64 hex string, then the little-endian
    bytes of ``down`` and of ``up``."""
    head = {"v": 1,
            "down": [_dtype_name(down), list(down.shape)],
            "up": [_dtype_name(up), list(up.shape)],
            "alpha": None if alpha is None
            else struct.pack("<d", float(alpha)).hex()}
    h = hashlib.sha256(json.dumps(head, sort_keys=True,
                                  separators=(",", ":")).encode())
    h.update(_raw(down).data)
    h.update(_raw(up).data)
    return h.hexdigest()


def register_pair(pipe, down, up, alpha: float | None = None) -> str:
    """Keep a pair in the pipeline's cache and return its content hash,
    which names it in patch table entries. A pair already held is not
    stored again."""
    down, up = mx.array(down), mx.array(up)
    if down.ndim != 2 or up.ndim != 2 or up.shape[1] != down.shape[0]:
        raise AdapterError(
            f"a LoRA pair is down (rank, in) and up (out, rank), not "
            f"{tuple(down.shape)} and {tuple(up.shape)}")
    for a in (down, up):
        if not mx.issubdtype(a.dtype, mx.floating):
            raise AdapterError(f"a LoRA pair is floating point, not "
                               f"{_dtype_name(a)}")
    if alpha is not None:
        alpha = float(alpha)
    mx.eval(down, up)
    key = pair_hash(down, up, alpha)
    if pipe.adapters.get(key) is None:
        pair = Pair(down, up, alpha)
        if pair.nbytes > pipe.adapters.max_bytes:
            raise AdapterError(
                f"a LoRA pair of {pair.nbytes} bytes is over the adapter "
                f"cache of {pipe.adapters.max_bytes} bytes")
        pipe.adapters.put(key, pair, pair.nbytes)
    return key


# Module walking.

def linear_dims(mod) -> tuple[int, int] | None:
    """(out, in) of a linear module, for a float or a K-quant weight. A
    wrapped leaf answers for its base."""
    if isinstance(mod, ImageLoRALinear):
        mod = mod.base
    w = getattr(mod, "weight", None)
    if not isinstance(w, mx.array) or w.ndim != 2:
        return None
    codec = getattr(mod, "kquant_type", None)
    if codec is None:
        return int(w.shape[0]), int(w.shape[1])
    from mlx_kquant.codec_geometry import in_features

    return int(w.shape[0]), in_features(codec, int(w.shape[1]))


def modules(dit: nn.Module) -> dict:
    """Module path to module, as a patch table addresses them. A wrapped
    leaf appears once, under its own path."""
    found = dict(dit.named_modules())
    inner = {f"{p}.base" for p, m in found.items()
             if isinstance(m, ImageLoRALinear)}
    return {p: m for p, m in found.items() if p not in inner}


def linear_leaves(dit: nn.Module) -> dict:
    """Module path to base leaf, for every linear leaf."""
    out = {}
    for path, m in modules(dit).items():
        if linear_dims(m) is not None:
            out[path] = m.base if isinstance(m, ImageLoRALinear) else m
    return out


def _set(root, path: str, new) -> None:
    parts = path.split(".")
    obj = root
    for part in parts[:-1]:
        obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
    last = parts[-1]
    if last.isdigit():
        obj[int(last)] = new
    else:
        setattr(obj, last, new)


# Factors of one module under one table, by placement.

def _transposed(lm, dtype) -> tuple[mx.array, mx.array]:
    """A^T (in, rank) and B^T (rank, out) in ``dtype``."""
    return (mx.contiguous(lm.a.T).astype(dtype),
            mx.contiguous(lm.b.T).astype(dtype))


def _add(y: mx.array, z: mx.array, b_t: mx.array) -> mx.array:
    if y.dtype == b_t.dtype:
        return mx.addmm(y, z, b_t)
    return y + (z @ b_t).astype(y.dtype)


class _Block:
    """One pair over the whole weight."""

    def __init__(self, a_t: mx.array, b_t: mx.array):
        self.a_t, self.b_t = a_t, b_t

    def arrays(self) -> list:
        return [self.a_t, self.b_t]

    def __call__(self, base, x: mx.array) -> mx.array:
        return _add(base(x), x @ self.a_t, self.b_t)


def _factors(path: str, parts: list, dims: tuple, dtype, how: str, base):
    return _Block(*_transposed(plan.block_diagonal(path, parts, dims),
                               dtype))


def check_table(pipe, table) -> None:
    """Refuse a table that names a pair the pipeline does not hold, or one
    whose shape does not fit its module and band."""
    leaves = linear_leaves(pipe.dit)
    for path, band, patches in table["entries"]:
        dims = linear_dims(leaves[path])
        for key, _ in patches:
            pair = pipe.adapters.get(key)
            if pair is None:
                raise AdapterError(
                    f"patch table names adapter {key[:16]}, which this "
                    f"pipeline does not hold. Register the pair first")
            why = plan.fit(tuple(pair.up.shape), tuple(pair.down.shape),
                           band, dims)
            if why is not None:
                where = path if band is None else f"{path} band {band}"
                raise AdapterError(f"adapter {key[:16]} does not fit "
                                   f"{where}: {why}")


class BuiltTable:
    """Module path to factors for one table at one activation dtype and
    placement. A module's factors are built on its first call at that
    dtype, from the cached pairs alone, so no weight of the model is read
    and a dtype that few leaves run in costs only their factors."""

    def __init__(self, parts: dict, leaves: dict, dtype, how: str):
        self._parts = parts            # path -> [Part]
        self._leaves = leaves          # path -> base leaf
        self._dtype, self._how = dtype, how
        self._built: dict = {}
        self.nbytes = 0

    def __contains__(self, path: str) -> bool:
        return path in self._parts

    def get(self, path: str):
        f = self._built.get(path)
        if f is None:
            parts = self._parts.get(path)
            if parts is None:
                return None
            leaf = self._leaves[path]
            f = _factors(path, parts, linear_dims(leaf), self._dtype,
                         self._how, leaf)
            arrays = f.arrays()
            mx.eval(arrays)
            self.nbytes += sum(int(a.nbytes) for a in arrays)
            self._built[path] = f
        return f


def build_table(pipe, tid: str, dtype, how: str) -> BuiltTable:
    """The built table of ``tid``. Raises when a pair it names is no
    longer in the pipeline's cache."""
    table = pipe.patch_tables[tid]
    leaves = linear_leaves(pipe.dit)
    by_module: dict = {}
    for path, band, patches in table["entries"]:
        for key, strength in patches:
            if strength == 0.0:
                continue
            pair = pipe.adapters.get(key)
            if pair is None:
                raise AdapterError(
                    f"adapter {key[:16]} of patch table {tid[:16]} is no "
                    f"longer held. Register the pair again")
            by_module.setdefault(path, []).append(
                plan.Part(pair.down, pair.up, pair.alpha, band, strength))
    return BuiltTable(by_module, {p: leaves[p] for p in by_module}, dtype,
                      how)


class PatchState:
    """The patch table a DiT forward runs under. ``active`` is a table id
    with entries, or None for the base model. Built tables are cached in
    the pipeline per table id, dtype and placement, and the ones of the
    active table are also held here."""

    def __init__(self, pipe):
        self._pipe = weakref.ref(pipe)
        self.active: str | None = None
        self._cur: dict = {}

    def factors(self, path: str, dtype):
        tid = self.active
        if tid is None:
            return None
        key = (tid, dtype, _PLACEMENT)
        hit = self._cur.get(key)
        if hit is None:
            if any(k[0] != tid for k in self._cur):
                self._cur.clear()
            hit = self._cur[key] = self._table(key)
        return hit.get(path)

    def _table(self, key: tuple) -> BuiltTable:
        pipe = self._pipe()
        if pipe is None:
            raise RuntimeError("the image pipeline of this DiT is gone")
        hit = pipe.tables.get(key)
        if hit is None:
            hit = build_table(pipe, *key)
            pipe.tables.put(key, hit, hit.nbytes)
        return hit

    def account(self) -> None:
        """Record the bytes the active tables have built so far."""
        pipe = self._pipe()
        if pipe is None:
            return
        for key, table in self._cur.items():
            pipe.tables.put(key, table, table.nbytes)


class ImageLoRALinear(nn.Module):
    """A linear leaf of an image DiT with the active patch table's delta
    for its module path applied live."""

    _path: str
    _state: PatchState

    def __init__(self, base: nn.Module, path: str, state: PatchState):
        super().__init__()
        self.base = base
        object.__setattr__(self, "_path", path)
        object.__setattr__(self, "_state", state)
        self.freeze()

    @property
    def weight(self) -> mx.array:
        return self.base.weight

    def __call__(self, x: mx.array) -> mx.array:
        f = self._state.factors(self._path, x.dtype)
        if f is None:
            return self.base(x)
        return f(self.base, x)

    def _extra_repr(self) -> str:
        return f"path={self._path}, base={type(self.base).__name__}"


def ensure_wrapped(pipe) -> None:
    """Wrap every linear leaf of the DiT, once."""
    if pipe.lora is not None:
        return
    state = PatchState(pipe)
    for path, leaf in linear_leaves(pipe.dit).items():
        _set(pipe.dit, path, ImageLoRALinear(leaf, path, state))
    pipe.lora = state


def activate(pipe, tid: str):
    """Set the table the next DiT calls run under and return the previous
    one, for ``restore``."""
    state = pipe.lora
    if state is None:
        return None
    prev = state.active
    state.active = tid if pipe.patch_tables[tid]["entries"] else None
    return prev


def restore(pipe, prev) -> None:
    if pipe.lora is not None:
        pipe.lora.account()
        pipe.lora.active = prev
