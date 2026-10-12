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
text models. ``EncoderLoRALinear`` is the same for the text halves on the
pipeline's own encoder.

Pairs live in the pipeline's ``adapters`` cache under the sha256 of their
content, so a file read twice or two files that share a pair hold it once.
The leaf swap runs when the first patch table with entries is defined, so
a pipeline that never uses an adapter keeps its plain leaves.

GMLX_IMAGE_LORA_PLACEMENT picks how a table with output bands is applied,
for measurement. ``block`` is the block-diagonal pair over the whole
weight. ``slice`` adds each band's product into its output slice.
``split`` runs the base as one product per band.

With the ``lora`` route of ``gmlx.image.ops.kq_routes`` in force, the
block placement of a K-quant leaf is built in the form of mlx-kquant's
low-rank op: the base product and the delta are one call, the output
bands of the table go in as the op's band table, and the strength of
each part is an input array beside the pair. A table that differs from
another in its strengths alone so shares that table's pair. A call the
route does not take runs the block-diagonal pair on stock ops.
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

from gmlx import mlx_guards
from gmlx.envflags import env_choice
from gmlx.image import ops
from gmlx.image.ops import kq_routes

from . import plan
from .reader import AdapterError

PLACEMENTS = ("block", "slice", "split")
_PLACEMENT = env_choice("GMLX_IMAGE_LORA_PLACEMENT", "block", PLACEMENTS)
# The form of the block placement while the low-rank route is in force.
KERNEL = "kernel"


def placement() -> str:
    return _PLACEMENT


def form() -> str:
    """The form a table is built in now: the placement, or ``KERNEL`` for
    the block placement while the low-rank route is in force. A built
    table is kept under it."""
    if _PLACEMENT == "block" and "lora" in kq_routes.active():
        return KERNEL
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
        return mlx_guards.addmm(y, z, b_t)
    return y + (z @ b_t).astype(y.dtype)


class _Block:
    """One pair over the whole weight."""

    def __init__(self, a_t: mx.array, b_t: mx.array):
        self.a_t, self.b_t = a_t, b_t

    def arrays(self) -> list:
        return [self.a_t, self.b_t]

    def __call__(self, base, x: mx.array) -> mx.array:
        return _add(base(x), x @ self.a_t, self.b_t)


class _Slice:
    """One pair per output band, added into the band's slice."""

    def __init__(self, spans: list, out: int):
        self.spans, self.out = spans, out      # (start, stop, a_t, b_t)

    def arrays(self) -> list:
        return [a for s in self.spans for a in s[2:]]

    def __call__(self, base, x: mx.array) -> mx.array:
        y = base(x)
        parts, pos = [], 0
        for start, stop, a_t, b_t in self.spans:
            if start > pos:
                parts.append(y[..., pos:start])
            parts.append(_add(y[..., start:stop], x @ a_t, b_t))
            pos = stop
        if pos < self.out:
            parts.append(y[..., pos:])
        return parts[0] if len(parts) == 1 else mx.concatenate(parts, -1)


def _band_fn(base, start: int, stop: int):
    """The base product over output rows ``start`` to ``stop`` alone and
    the weight copy it holds, or None for a base this cannot split."""
    from mlx_kquant.nn import KQuantLinear

    bias = base.bias[start:stop] if "bias" in base else None
    w = base.weight[start:stop]
    if isinstance(base, KQuantLinear):
        import mlx_kquant as kq

        w = mx.contiguous(w)
        mx.eval(w)
        scales, codec = base.scales, base.kquant_type

        def kq_rows(x):
            y = kq.quantized_matmul(x, w, scales, codec, transpose=True)
            return y if bias is None else y + bias

        return kq_rows, w
    if type(base) is not nn.Linear:
        return None
    w_t = mx.contiguous(w.T)
    mx.eval(w_t)

    def rows(x):
        y = x @ w_t
        return y if bias is None else y + bias

    return rows, w_t


class _Split:
    """The base as one product per band, each with its own pair. A band
    with no pair has None factors."""

    def __init__(self, spans: list, out: int, fns: list, held: list):
        self.spans, self.out, self.fns, self.held = spans, out, fns, held

    def arrays(self) -> list:
        return [a for s in self.spans for a in s[2:]
                if a is not None] + self.held

    def __call__(self, base, x: mx.array) -> mx.array:
        parts = []
        for fn, span in zip(self.fns, self.spans):
            y = fn(x)
            if span[2] is not None:
                y = _add(y, x @ span[2], span[3])
            parts.append(y)
        return parts[0] if len(parts) == 1 else mx.concatenate(parts, -1)


class _Kernel:
    """The block placement of a K-quant leaf in the form of mlx-kquant's
    low-rank op: A^T and B^T with ``alpha / rank`` in B, one strength per
    rank, and the band table or None. ``pair`` is shared by the tables of
    the same parts at other strengths. ``plain`` builds the factors of
    the stock path from the pair's arrays, on the first call the route
    does not take, such as one of few rows."""

    def __init__(self, pair: tuple, strength: mx.array, plain,
                 kept=lambda: False):
        self.a_t, self.b_t, self.bands = pair
        self.strength = strength
        self._plain, self._stock = plain, None
        self._kept = kept

    def arrays(self) -> list:
        """The arrays this table holds that are not counted elsewhere:
        the pair too once the cache that kept it has shed it."""
        stock = [] if self._stock is None else self._stock.arrays()
        pair = [] if self._kept() else [self.a_t, self.b_t]
        return [self.strength, *pair,
                *(a for a in stock if a is not self.a_t and a is not self.b_t)]

    def stock(self):
        if self._stock is None:
            self._stock = self._plain(self.a_t, self.b_t)
            mx.eval(self._stock.arrays())
        return self._stock

    def __call__(self, base, x: mx.array) -> mx.array:
        leaf = kq_routes.lora_leaf(base, x)
        if leaf is None:
            return self.stock()(base, x)
        return ops.lora_fused(x, leaf, self.a_t, self.b_t, self.strength,
                              self.bands)


def kernel_pair(path: str, parts: list, dims: tuple, dtype) -> tuple:
    """A^T, B^T and the band table of the parts for mlx-kquant's low-rank
    op, with no strength in them."""
    bands = plan.band_table(parts, dims)
    return (*_transposed(plan.unit_pair(path, parts, dims), dtype), bands)


def _strengths(parts: list) -> mx.array:
    """One strength per rank of ``kernel_pair``, in float32."""
    return mx.array(np.concatenate(
        [np.full(int(p.down.shape[0]), p.strength, np.float32)
         for p in parts]))


def _factors(path: str, parts: list, dims: tuple, dtype, how: str, base,
             pair=None):
    if how == KERNEL:
        how = "block"
        ranks = sum(int(p.down.shape[0]) for p in parts)
        bands = plan.band_table(parts, dims)
        if pair is not None and kq_routes.lora_static(base, dtype, ranks,
                                                      bands):
            def plain(a_t: mx.array, b_t: mx.array):
                # The block-diagonal pair of the stock path: its A is the
                # pair's, and so is its B when every strength is 1.0, the
                # factor it then holds. Any other strength is folded into
                # a B of its own.
                if any(p.strength != 1.0 for p in parts):
                    b = plan.block_diagonal(path, parts, dims).b
                    b_t = mx.contiguous(b.T).astype(dtype)
                return _Block(a_t, b_t)
            got, kept = pair()
            return _Kernel(got, _strengths(parts), plain, kept)
    if how != "block":
        bands = plan.banded(path, parts, dims)
        if bands is not None and not (len(bands) == 1 and bands[0][:2]
                                      == (0, dims[0])):
            spans = [(s, e, *_transposed(lm, dtype)) for s, e, lm in bands]
            if how == "slice":
                return _Slice(spans, dims[0])
            full, pos = [], 0
            for span in spans:
                if span[0] > pos:
                    full.append((pos, span[0], None, None))
                full.append(span)
                pos = span[1]
            if pos < dims[0]:
                full.append((pos, dims[0], None, None))
            fns = [_band_fn(base, s[0], s[1]) for s in full]
            if all(fn is not None for fn in fns):
                return _Split(full, dims[0], [fn for fn, _ in fns],
                              [w for _, w in fns])
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

    def __init__(self, parts: dict, leaves: dict, dtype, how: str,
                 pairs=None):
        self._parts = parts            # path -> [Part]
        self._leaves = leaves          # path -> base leaf
        self._dtype, self._how = dtype, how
        self._pairs = pairs            # (path, parts, dims) -> kernel pair
        self._built: dict = {}

    def __contains__(self, path: str) -> bool:
        return path in self._parts

    @property
    def nbytes(self) -> int:
        """Bytes of the factors built so far."""
        return sum(int(a.nbytes) for f in self._built.values()
                   for a in f.arrays())

    def get(self, path: str):
        f = self._built.get(path)
        if f is None:
            parts = self._parts.get(path)
            if parts is None:
                return None
            leaf = self._leaves[path]
            dims = linear_dims(leaf)
            pair = None if self._pairs is None else \
                (lambda: self._pairs(path, parts, dims))
            f = _factors(path, parts, dims, self._dtype, self._how, leaf,
                         pair)
            mx.eval(f.arrays())
            self._built[path] = f
        return f


def build_table(pipe, tid: str, dtype, how: str) -> BuiltTable:
    """The built table of ``tid``. Raises when a pair it names is no
    longer in the pipeline's cache."""
    table = pipe.patch_tables[tid]
    leaves = linear_leaves(pipe.dit)
    by_module: dict = {}
    named: dict = {}
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
            named.setdefault(path, []).append((key, band))
    pairs = None
    if how == KERNEL:
        owner = weakref.ref(pipe)

        def pairs(path: str, parts: list, dims: tuple) -> tuple:
            # The pair of a module is named by its parts without their
            # strengths, so every table of those parts shares it. It is
            # counted in the cache while the cache keeps it, and in each
            # table that holds it after that.
            key = (KERNEL, path, tuple(named[path]), dtype)
            held = owner()
            hit = None if held is None else held.tables.get(key)
            if hit is None:
                hit = kernel_pair(path, parts, dims, dtype)
                mx.eval(hit[0], hit[1])
                if held is not None:
                    held.tables.put(key, hit, int(hit[0].nbytes)
                                    + int(hit[1].nbytes))

            def kept() -> bool:
                p = owner()
                return p is not None and p.tables.peek(key) is hit

            return hit, kept
    return BuiltTable(by_module, {p: leaves[p] for p in by_module}, dtype,
                      how, pairs)


class PatchState:
    """The patch table a DiT forward runs under. ``active`` is a table id
    with entries, or None for the base model. Built tables are cached in
    the pipeline per table id, dtype and form, and the ones of the
    active table are also held here. A DiT has one state, which every
    pipeline over it shares: ``activate`` names the pipeline whose table
    runs, and a table id names its content, so pipelines that define
    equal tables share the built one."""

    def __init__(self, pipe):
        self._pipe = weakref.ref(pipe)
        self.active: str | None = None
        self._cur: dict = {}

    def factors(self, path: str, dtype):
        tid = self.active
        if tid is None:
            return None
        key = (tid, dtype, form())
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


class EncoderLoRALinear(nn.Module):
    """A linear layer of an image pipeline's own text encoder with the
    summed delta of its text halves applied live on stock ops. ``lm``
    holds A (rank, in) and B (out, rank) with every strength and
    ``alpha / rank`` folded into B. Like ``ImageLoRALinear`` it reads
    nothing from ``gmlx.lora_rows``: the rows of a text batch in the same
    process say nothing about an image prompt."""

    def __init__(self, base: nn.Module, lm):
        super().__init__()
        self.base = base
        object.__setattr__(self, "_lm", lm)
        object.__setattr__(self, "_tables", {})
        self.freeze()

    @property
    def weight(self) -> mx.array:
        return self.base.weight

    def __call__(self, x: mx.array) -> mx.array:
        t = self._tables.get(x.dtype)
        if t is None:
            t = self._tables[x.dtype] = _transposed(self._lm, x.dtype)
            mx.eval(t)
        return _add(self.base(x), x @ t[0], t[1])

    def _extra_repr(self) -> str:
        return f"rank={self._lm.rank}, base={type(self.base).__name__}"


def _state_of(dit: nn.Module):
    """The patch state the wrapped leaves of ``dit`` read, or None when
    no leaf is wrapped."""
    for _, m in dit.named_modules():
        if isinstance(m, ImageLoRALinear):
            return m._state
    return None


def ensure_wrapped(pipe) -> None:
    """Wrap every linear leaf of the DiT, once. A second pipeline over a
    DiT that is already wrapped takes the state its leaves read, so each
    pipeline's table reaches them."""
    if pipe.lora is not None:
        return
    state = _state_of(pipe.dit)
    if state is None:
        state = PatchState(pipe)
        for path, leaf in linear_leaves(pipe.dit).items():
            _set(pipe.dit, path, ImageLoRALinear(leaf, path, state))
    pipe.lora = state


def drop_built(pipe) -> None:
    """Drop the built tables of the pipeline, which hold the leaves of
    its DiT, so that a DiT the pipeline lets go is freed."""
    pipe.tables.drop_where(lambda v: isinstance(v, BuiltTable))
    if pipe.lora is not None:
        pipe.lora._cur.clear()


def rewrap(pipe) -> None:
    """Wrap the linear leaves of a DiT loaded again into the pipeline with
    the patch state the pipeline holds, so its tables reach them."""
    state = pipe.lora
    if state is None:
        return
    for path, leaf in linear_leaves(pipe.dit).items():
        _set(pipe.dit, path, ImageLoRALinear(leaf, path, state))
    state._cur.clear()


def activate(pipe, tid: str):
    """Set the pipeline and the table the next DiT calls run under and
    return the previous ones, for ``restore``."""
    state = pipe.lora
    if state is None:
        return None
    prev = (state._pipe, state.active)
    state._pipe = weakref.ref(pipe)
    state.active = tid if pipe.patch_tables[tid]["entries"] else None
    return prev


def restore(pipe, prev) -> None:
    state = pipe.lora
    if state is None or prev is None:
        return
    state.account()
    state._pipe, state.active = prev
