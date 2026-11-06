"""``ImageLoRALinear``, the leaf swap, and the pair cache.

An image adapter is applied live beside the base weight, which stays
K-quant wire bytes. ``ImageLoRALinear`` wraps one linear leaf of a DiT. On
each call it asks the pipeline's ``PatchState`` for the factors of its
module path under the active patch table. With none it calls the base
leaf and returns that result. With factors it adds
``(x @ A^T) @ B^T`` to the base output on stock ops, where strength and
``alpha / rank`` are folded into B. A module with one pair over its whole
weight in the activation dtype applies them to ``x @ A^T`` instead and
reads the cached pair as it is, so its table holds no copy.

The class is separate from the text adapter wrap in ``gmlx.load.modules``
and reads nothing from ``gmlx.lora_rows``. That channel is global to the
process and belongs to text rows, and an image model shares a process with
text models. ``EncoderLoRALinear`` is the same for the text halves on the
pipeline's own encoder.

Pairs live in the pipeline's ``adapters`` cache under the sha256 of their
down and up, so a file read twice or two files that share a pair hold it
once, whatever alpha each gives it. A pair's alpha is a patch table's:
an entry gives it beside the strength.
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

import fnmatch
import hashlib
import json
import math
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
    """One cached pair: ``down`` (rank, in) and ``up`` (out, rank)."""
    down: mx.array
    up: mx.array

    @property
    def nbytes(self) -> int:
        return int(self.down.nbytes) + int(self.up.nbytes)


def _dtype_name(arr: mx.array) -> str:
    return str(arr.dtype).rsplit(".", 1)[-1]


def _raw(arr: mx.array) -> np.ndarray:
    if arr.dtype == mx.bfloat16:
        arr = arr.view(mx.uint16)
    return np.ascontiguousarray(np.array(arr))


def pair_hash(down: mx.array, up: mx.array) -> str:
    """sha256 of a pair's content: a JSON line of each part's dtype and
    shape, then the little-endian bytes of ``down`` and of ``up``."""
    return pair_key((_dtype_name(down), down.shape, _raw(down).data),
                    (_dtype_name(up), up.shape, _raw(up).data))


def pair_key(down: tuple, up: tuple) -> str:
    """``pair_hash`` of the parts as (dtype name, shape, little-endian
    bytes), for a pair not made into arrays yet."""
    head = {"v": 2, "down": [down[0], [int(d) for d in down[1]]],
            "up": [up[0], [int(d) for d in up[1]]]}
    h = hashlib.sha256(json.dumps(head, sort_keys=True,
                                  separators=(",", ":")).encode())
    h.update(down[2])
    h.update(up[2])
    return h.hexdigest()


class PairRoomError(AdapterError):
    """A pair the pipeline's pair cache cannot hold beside the pairs it
    keeps."""


def register_pair(pipe, down, up, *, key: str | None = None,
                  hold=()) -> str:
    """Keep a pair in the pipeline's cache and return its content hash,
    which names it in patch table entries beside its alpha. A pair
    already held is not stored again. An MLX array is kept as it is,
    since a copy would hold the pair twice while the reader still holds
    it. The caller must not change it afterwards. ``key`` is the pair's
    ``pair_hash`` when the caller has it already. ``hold`` names the
    pairs the caller put before this one for the same table or file,
    which the room for this one never sheds."""
    down, up = (a if isinstance(a, mx.array) else mx.array(a)
                for a in (down, up))
    if down.ndim != 2 or up.ndim != 2 or up.shape[1] != down.shape[0]:
        raise AdapterError(
            f"a LoRA pair is down (rank, in) and up (out, rank), not "
            f"{tuple(down.shape)} and {tuple(up.shape)}")
    for a in (down, up):
        if not mx.issubdtype(a.dtype, mx.floating):
            raise AdapterError(f"a LoRA pair is floating point, not "
                               f"{_dtype_name(a)}")
    mx.eval(down, up)
    if key is None:
        key = pair_hash(down, up)
    if pipe.adapters.get(key) is None:
        pair = Pair(down, up)
        if pair.nbytes > pipe.adapters.max_bytes:
            raise PairRoomError(
                f"a LoRA pair of {pair.nbytes} bytes is over the adapter "
                f"cache of {pipe.adapters.max_bytes} bytes")
        if not make_room(pipe, pipe.adapters, pair.nbytes, hold):
            raise PairRoomError(
                f"a LoRA pair of {pair.nbytes} bytes does not fit the "
                f"adapter cache of {pipe.adapters.max_bytes} bytes beside "
                f"the pairs it keeps")
        pipe.adapters.put(key, pair, pair.nbytes, shed=False)
    return key


# Static ownership. A pair's bytes are counted in the pair cache from its
# put, and a built table counts the arrays it made alone, so no array is
# counted under two keys. An entry that something held depends on is
# never shed: the entry's own pairs (the pair cache's keep set), the pairs
# a held built table names, and the kernel pairs it is built over.
# make_room sheds the dependents first.

def _tables(pipe) -> list:
    """The built tables the pipeline holds, the least recently used
    first: those of its tables cache, then any the patch state holds
    beside them."""
    out, seen = [], set()
    for k in pipe.tables.keys(lambda k: k[0] != KERNEL):
        t = pipe.tables.peek(k)
        if isinstance(t, BuiltTable):
            out.append(t)
            seen.add(id(t))
    out += [t for t in _current(pipe) if id(t) not in seen]
    return out


def _last_steps(pipe) -> dict:
    """When each built table of the pipeline's tables cache last stepped,
    by the table's id, on the clock the uses of the pair cache share: a
    step puts the tables it ran into the cache."""
    return {id(pipe.tables.peek(k)): pipe.tables.used(k)
            for k in pipe.tables.keys(lambda k: k[0] != KERNEL)}


def _current(pipe) -> list:
    """The built tables the patch state holds for this pipeline: those of
    the table it ran last, or runs now."""
    state = pipe.lora
    if state is None or state._held() is not pipe:
        return []
    return list(state._cur.values())


def _running(pipe) -> set:
    """The ids of the built tables of a forward in flight on this
    pipeline."""
    state = pipe.lora
    if state is None or state.active is None or state._pipe() is not pipe \
            or state._held() is not pipe:
        return set()
    return {id(t) for k, t in state._cur.items() if k[0] == state.active}


# A table that goes releases the pins it held, so every function that
# lets one go sheds the tables cache of the pipeline that held it, to its
# bound: _drop_tables and _release_cur.

def _drop_tables(pipe, match, shed: bool = True) -> None:
    """Drop the built tables ``match`` takes, from the tables cache and,
    when it holds this pipeline's tables, from the patch state. Then shed
    the tables cache to its bound, as the kernel pairs the tables pinned
    there may be past it; ``shed`` is off for the victims of make_room's
    tables loop, which sheds itself. The caller keeps out any table a
    forward runs."""
    pipe.tables.drop_where(lambda t: isinstance(t, BuiltTable) and match(t))
    state = pipe.lora
    if state is not None and state._held() is pipe:
        for k in [k for k, t in state._cur.items() if match(t)]:
            del state._cur[k]
    if shed:
        make_room(pipe, pipe.tables, 0)


def _release_cur(state) -> None:
    """Let go of the built tables the patch state holds, and shed the
    tables cache of the pipeline that held them to its bound."""
    holder = state._held()
    state._cur.clear()
    if holder is not None:
        make_room(holder, holder.tables, 0)


def make_room(pipe, cache, nbytes: int, hold=()) -> bool:
    """Shed entries of ``cache``, the pipeline's pair cache or its tables
    cache, until ``nbytes`` more fit, the least recently used first, and
    return whether they fit. No entry goes that something held depends
    on; its dependents go first.

    In the pair cache the pairs of the keep set and of ``hold`` stay, and
    so do the pairs of a held built table until the table goes. The
    unpinned pairs and the held tables that pin a pair go in one order of
    use, a pair's last get against a table's last step, and a table a
    forward runs never goes. A put whose bytes cannot fit beside the pairs
    that stay sheds nothing; a shed to the bound, of 0 bytes, sheds all
    it can.

    In the tables cache the tables the patch state holds stay, with the
    kernel pairs of any held table: the unpinned kernel pairs go first,
    then the least recently used other table. When the bytes cannot fit,
    everything else goes, and the caller keeps its entry past the
    bound."""
    def fits() -> bool:
        return cache.bytes + nbytes <= cache.max_bytes

    if fits():
        return True
    running = _running(pipe)
    if cache is pipe.adapters:
        base = set(cache.kept)
        base.update(hold)
        for t in _tables(pipe):
            if id(t) in running:
                base |= t.keys
        free = sum(cache.nbytes(k) for k in cache.keys(lambda k: k not in base))
        if nbytes and cache.bytes - free + nbytes > cache.max_bytes:
            return False
        while True:
            held = _tables(pipe)
            pinned = set(base)
            for t in held:
                pinned |= t.keys
            # The table that stepped least recently of those that can go,
            # after the unpinned pairs got before its step.
            steps = _last_steps(pipe)
            victim = min((t for t in held if id(t) not in running
                          and not t.keys <= base),
                         key=lambda t: steps.get(id(t), math.inf),
                         default=None)
            before = None if victim is None else steps.get(id(victim))
            if cache.shed_for(nbytes, pinned, before):
                return True
            if victim is None:
                return False
            _drop_tables(pipe, lambda t: t is victim)
    mine = {id(t) for t in _current(pipe)}
    while True:
        held = _tables(pipe)
        pinned = set(cache.keys(lambda k: k[0] != KERNEL))
        for t in held:
            pinned |= t.kernels
        if cache.shed_for(nbytes, pinned):
            return True
        victim = next((t for t in held if id(t) not in mine), None)
        if victim is None:
            return False
        _drop_tables(pipe, lambda t: t is victim, shed=False)


def shed_to_bounds(pipe) -> None:
    """Shed both caches of the pipeline to their bounds, after a bound
    was lowered."""
    make_room(pipe, pipe.adapters, 0)
    make_room(pipe, pipe.tables, 0)


def _keep_table(pipe, key, value, nbytes: int) -> None:
    """Count ``value`` in the tables cache at ``key``, with room made for
    it. A table the patch state holds, or a kernel pair it is built over,
    is kept past the bound when nothing else is left to shed."""
    nbytes = int(nbytes)
    grown = nbytes - (pipe.tables.nbytes(key)
                      if pipe.tables.peek(key) is value else 0)
    make_room(pipe, pipe.tables, grown)
    pipe.tables.put(key, value, nbytes, shed=False)


def _is_default(pipe, tid) -> bool:
    """Whether ``tid`` is the pipeline's own default table."""
    return tid is not None and tid == getattr(pipe, "default_table", None)


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


def _child(obj, part: str):
    """The child ``part`` of a module, list or dict of modules, as
    ``named_modules`` names it."""
    if isinstance(obj, (list, tuple)):
        return obj[int(part)]
    if isinstance(obj, dict) and not isinstance(obj, nn.Module):
        return obj[part]
    return getattr(obj, part)


def _set(root, path: str, new) -> None:
    parts = path.split(".")
    obj = root
    for part in parts[:-1]:
        obj = _child(obj, part)
    last = parts[-1]
    if isinstance(obj, list):
        obj[int(last)] = new
    elif isinstance(obj, dict) and not isinstance(obj, nn.Module):
        obj[last] = new
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


class _Own:
    """One pair over the whole weight, in the activation dtype or under
    float32 activations, on the cached pair's own arrays: ``x @ A^T``
    times the part's strength and ``alpha / rank`` in float32, then
    through B. The table holds no copy of the pair, which the pair cache
    counts and keeps while the table is held."""

    def __init__(self, part):
        self.down, self.up = part.down, part.up
        self.scale = part.strength * plan.pair_scale(
            part.alpha, int(part.down.shape[0]))

    def arrays(self) -> list:
        return []

    def __call__(self, base, x: mx.array) -> mx.array:
        z = x @ self.down.T
        z = (z.astype(mx.float32) * self.scale).astype(z.dtype)
        return _add(base(x), z, self.up.T)


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


def _kq_linear(base) -> bool:
    from mlx_kquant.nn import KQuantLinear

    return isinstance(base, KQuantLinear)


def _band_fn(base, start: int, stop: int):
    """The base product over output rows ``start`` to ``stop`` alone and
    the weight copy it holds, or None for a base this cannot split."""
    bias = base.bias[start:stop] if "bias" in base else None
    w = base.weight[start:stop]
    if _kq_linear(base):
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

    def __init__(self, pair: tuple, strength: mx.array, plain):
        self.a_t, self.b_t, self.bands = pair
        self.strength = strength
        self._plain, self._stock = plain, None

    def arrays(self) -> list:
        """The arrays this table made: the pair is the tables cache's,
        under its own key, kept while the table is held."""
        stock = [] if self._stock is None else self._stock.arrays()
        return [self.strength,
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


def _runs_as_kept(pair, dtype) -> bool:
    """Whether a factor in ``pair`` runs as the cache keeps it under
    ``dtype`` activations: in that dtype, or under float32, where MLX
    promotes it exactly."""
    return pair == dtype or dtype == mx.float32


def _band_spans(parts: list, dims: tuple) -> list | None:
    """``plan.banded`` of the parts by their shapes alone: (start, stop,
    ranks) per output band, or None where it gives None."""
    out = dims[0]
    groups: dict = {}
    for p in parts:
        if p.band is not None and p.band[0] != 0:
            return None
        span = (0, out) if p.band is None else (p.band[1],
                                                p.band[1] + p.band[2])
        groups[span] = groups.get(span, 0) + int(p.down.shape[0])
    spans = sorted(groups)
    if any(a[1] > b[0] for a, b in zip(spans, spans[1:])):
        return None
    return [(s, e, groups[(s, e)]) for s, e in spans]


def _splittable(base) -> bool:
    """Whether ``_band_fn`` can split ``base``."""
    return _kq_linear(base) or type(base) is nn.Linear


def _form(parts: list, dims: tuple, dtype, how: str, base,
          has_pair: bool) -> tuple:
    """The form ``_factors`` builds a module in at ``dtype``, with the
    figures its bytes take: ("kernel", ranks), ("slice", spans),
    ("split", spans), ("own",) or ("block", ranks). The build and the
    dry run read this one plan."""
    ranks = sum(int(p.down.shape[0]) for p in parts)
    if how == KERNEL:
        how = "block"
        bands = plan.band_table(parts, dims)
        if has_pair and kq_routes.lora_static(base, dtype, ranks, bands):
            return ("kernel", ranks)
    if how != "block":
        spans = _band_spans(parts, dims)
        if spans is not None and not (len(spans) == 1 and spans[0][:2]
                                      == (0, dims[0])):
            if how == "slice":
                return ("slice", spans)
            if _splittable(base):
                return ("split", spans)
    one = parts[0]
    if (len(parts) == 1 and one.band is None
            and _runs_as_kept(one.down.dtype, dtype)
            and _runs_as_kept(one.up.dtype, dtype)):
        return ("own",)
    return ("block", ranks)


def _factors(path: str, parts: list, dims: tuple, dtype, how: str, base,
             pair=None):
    form = _form(parts, dims, dtype, how, base, pair is not None)
    if form[0] == "kernel":
        def plain(a_t: mx.array, b_t: mx.array):
            # The block-diagonal pair of the stock path: its A is the
            # pair's, and so is its B when every strength is 1.0, the
            # factor it then holds. Any other strength is folded into a B
            # of its own.
            if any(p.strength != 1.0 for p in parts):
                b = plan.block_diagonal(path, parts, dims).b
                b_t = mx.contiguous(b.T).astype(dtype)
            return _Block(a_t, b_t)
        return _Kernel(pair(), _strengths(parts), plain)
    if form[0] in ("slice", "split"):
        spans = [(s, e, *_transposed(lm, dtype))
                 for s, e, lm in plan.banded(path, parts, dims)]
        if form[0] == "slice":
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
        return _Split(full, dims[0], [fn for fn, _ in fns],
                      [w for _, w in fns])
    if form[0] == "own":
        return _Own(parts[0])
    return _Block(*_transposed(plan.block_diagonal(path, parts, dims),
                               dtype))


def _factor_bytes(parts: list, dims: tuple, dtype, how: str, base,
                  has_pair: bool = True) -> tuple:
    """(table bytes, kernel pair bytes) ``_factors`` builds for a module
    at ``dtype``, from the parts' shapes: what the table's arrays hold
    once the module ran, its stock B included, and the kernel pair the
    tables cache keeps under its own key."""
    a = dtype.size
    out, inp = dims
    form = _form(parts, dims, dtype, how, base, has_pair)
    if form[0] == "kernel":
        r = form[1]
        stock = (out * r * a if any(p.strength != 1.0 for p in parts)
                 else 0)
        return 4 * r + stock, a * r * (inp + out)
    if form[0] in ("slice", "split"):
        got = sum(a * r * (inp + e - s) for s, e, r in form[1])
        if form[0] == "split":
            got += int(base.weight.nbytes)
        return got, 0
    if form[0] == "own":
        return 0, 0
    return a * form[1] * (inp + out), 0


def check_table(pipe, table) -> None:
    """Refuse a table that names a pair the pipeline does not hold, or one
    whose shape does not fit its module and band."""
    leaves = linear_leaves(pipe.dit)
    for path, band, patches in table["entries"]:
        dims = linear_dims(leaves[path])
        for key, _strength, _alpha in patches:
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
                 pairs=None, keys=frozenset(), kernels=None):
        self._parts = parts            # path -> [Part]
        self._leaves = leaves          # path -> base leaf
        self._dtype, self._how = dtype, how
        self._pairs = pairs            # (path, parts, dims) -> kernel pair
        self.keys = keys               # the pair keys it is built over
        self.kernels = set() if kernels is None else kernels
        self._built: dict = {}

    def __contains__(self, path: str) -> bool:
        return path in self._parts

    @property
    def nbytes(self) -> int:
        """Bytes of the factors built so far: the arrays the table made,
        never a pair's."""
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
    owner = weakref.ref(pipe)
    for path, band, patches in table["entries"]:
        for key, strength, alpha in patches:
            if strength == 0.0:
                continue
            pair = pipe.adapters.get(key)
            if pair is None:
                raise AdapterError(
                    f"adapter {key[:16]} of patch table {tid[:16]} is no "
                    f"longer held. Register the pair again")
            by_module.setdefault(path, []).append(
                plan.Part(pair.down, pair.up, alpha, band, strength))
            named.setdefault(path, []).append((key, band, alpha))
    kernels: set = set()
    pairs = None
    if how == KERNEL:
        def pairs(path: str, parts: list, dims: tuple) -> tuple:
            # The pair of a module is named by its parts without their
            # strengths, so every table of those parts shares it. Its B
            # holds each part's alpha / rank, so the alphas name it too.
            # The tables cache counts it under its own key and keeps it
            # while a held table is built over it.
            key = (KERNEL, path, tuple(named[path]), dtype)
            kernels.add(key)
            held = owner()
            hit = None if held is None else held.tables.get(key)
            if hit is None:
                hit = kernel_pair(path, parts, dims, dtype)
                mx.eval(hit[0], hit[1])
                if held is not None:
                    _keep_table(held, key, hit, int(hit[0].nbytes)
                                + int(hit[1].nbytes))
            return hit
    keys = frozenset(k for row in named.values() for k, _b, _a in row)
    return BuiltTable(by_module, {p: leaves[p] for p in by_module}, dtype,
                      how, pairs, keys, kernels)


@dataclass(frozen=True)
class _Like:
    """The shape and dtype of an array not made yet."""
    shape: tuple
    dtype: mx.Dtype

    @property
    def nbytes(self) -> int:
        return int(np.prod(self.shape)) * self.dtype.size


@dataclass(frozen=True)
class _Sized:
    """A part by the figures its factors' bytes take."""
    down: mx.array | _Like
    up: mx.array | _Like
    band: tuple | None
    strength: float


def f32_leaves(pipe) -> frozenset:
    """The paths of the DiT's linear leaves that its family names in
    ``lora_f32``: the leaves that see float32 rows under any activation
    dtype."""
    pats = getattr(getattr(pipe, "family", None), "lora_f32", ())
    if not pats:
        return frozenset()
    return frozenset(p for p in linear_leaves(pipe.dit)
                     if any(fnmatch.fnmatchcase(p, s) for s in pats))


def dry_run(pipe, tid: str) -> dict:
    """``table_figures`` of table ``tid`` at the DiT's activation dtype,
    with its family's float32 leaves. On the image worker, with the DiT
    loaded."""
    return table_figures(pipe, tid, pipe.dit.act_dtype, f32_leaves(pipe))


def table_figures(pipe, tid: str, act, f32=frozenset(),
                  shapes=None) -> dict:
    """The dry run of a forward on table ``tid`` at activation dtype
    ``act``: {key: (bytes once built, bytes held now)} for each tables
    cache key it holds, its built table at each dtype its modules run at
    and the kernel pairs, and for each pair the pair cache does not hold.
    A module runs at ``act``, or float32 for a leaf named in ``f32``, the
    family's leaves that see float32 rows. ``shapes`` gives (down shape,
    up shape, dtype) of each pair not held."""
    table = pipe.patch_tables[tid]
    leaves = linear_leaves(pipe.dit)
    how = form()
    out: dict = {}
    by_module: dict = {}
    named: dict = {}
    for path, band, patches in table["entries"]:
        for key, strength, alpha in patches:
            if strength == 0.0:
                continue
            pair = pipe.adapters.peek(key)
            if pair is None and key not in (shapes or {}):
                raise AdapterError(
                    f"adapter {key[:16]} of patch table {tid[:16]} is no "
                    f"longer held. Register the pair again")
            if pair is None:
                down, up, dtype = shapes[key]
                d, u = _Like(tuple(down), dtype), _Like(tuple(up), dtype)
                out[key] = (d.nbytes + u.nbytes, 0)
            else:
                d, u = pair.down, pair.up
            by_module.setdefault(path, []).append(
                _Sized(d, u, band, strength))
            named.setdefault(path, []).append((key, band, alpha))
    for path, parts in by_module.items():
        leaf = leaves[path]
        dims = linear_dims(leaf)
        if dims is None:            # check_table refuses such a leaf
            continue
        dtype = mx.float32 if (act == mx.float32 or path in f32) else act
        tb, pb = _factor_bytes(parts, dims, dtype, how, leaf, how == KERNEL)
        key = (tid, dtype, how)
        fig, held = out.get(key, (0, pipe.tables.nbytes(key)))
        out[key] = (fig + tb, held)
        if pb:
            key = (KERNEL, path, tuple(named[path]), dtype)
            out[key] = (pb, pipe.tables.nbytes(key))
    return out


class PatchState:
    """The patch table a DiT forward runs under. ``active`` is a table id
    with entries, or None for the base model. Built tables are cached in
    the pipeline per table id, dtype and form, and the ones of the
    active table are also held here. A DiT has one state, which every
    pipeline over it shares: ``activate`` names the pipeline whose table
    runs. The tables held here are of one pipeline, built over its pairs
    and counted in its tables cache, so a forward of another pipeline
    starts from that pipeline's own."""

    def __init__(self, pipe):
        self._pipe = weakref.ref(pipe)
        self.active: str | None = None
        self._cur: dict = {}
        # The pipeline whose tables cache holds the tables of ``_cur``.
        self._held = self._pipe

    def factors(self, path: str, dtype):
        hit = self._at(dtype)
        return None if hit is None else hit.get(path)

    def names(self, path: str, dtype) -> bool:
        """Whether the active table has a delta for ``path``, with no
        factors built: a probe of a route that may not run the leaf at
        ``dtype``."""
        hit = self._at(dtype)
        return hit is not None and path in hit

    def _at(self, dtype) -> BuiltTable | None:
        tid = self.active
        if tid is None:
            return None
        key = (tid, dtype, form())
        hit = self._cur.get(key)
        if hit is None or self._held() is not self._pipe():
            hit = self._table(key)
        return hit

    def _table(self, key: tuple) -> BuiltTable:
        """The built table at ``key`` of the pipeline that runs, held here
        from now on. The tables held for another table id or pipeline go
        first."""
        pipe = self._pipe()
        if pipe is None:
            raise RuntimeError("the image pipeline of this DiT is gone")
        if self._held() is not pipe or any(k[0] != key[0] for k in self._cur):
            _release_cur(self)
            self._held = self._pipe
        hit = pipe.tables.get(key)
        if hit is None:
            hit = build_table(pipe, *key)
            _keep_table(pipe, key, hit, hit.nbytes)
        self._cur[key] = hit
        return hit

    def account(self) -> None:
        """Record the bytes the tables of the forward that ends have built
        so far, and that they stepped. A forward that ran none of this
        pipeline's tables records nothing."""
        pipe = self._held()
        if pipe is None or pipe is not self._pipe():
            return
        for key, table in list(self._cur.items()):
            if key[0] == self.active:
                _keep_table(pipe, key, table, table.nbytes)


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
    its DiT, so that a DiT the pipeline lets go is freed. The kernel
    pairs they pinned past the tables bound go too. Tables the patch
    state holds for another pipeline over the DiT stay with it."""
    _drop_tables(pipe, lambda t: True)


def rewrap(pipe) -> None:
    """Wrap the linear leaves of a DiT loaded again into the pipeline with
    the patch state the pipeline holds, so its tables reach them."""
    state = pipe.lora
    if state is None:
        return
    for path, leaf in linear_leaves(pipe.dit).items():
        _set(pipe.dit, path, ImageLoRALinear(leaf, path, state))
    drop_built(pipe)


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
