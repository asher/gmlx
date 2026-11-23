"""Expert store for the fused MoE blocks (GMLX_MOE_EXPERT_STORE).

A full store is the default for the model types in _DEFAULT_FULL (Flash-Next)
and off for every other model, whose expert weights stay the GGUF's own
bytes, mapped zero copy. GMLX_MOE_EXPERT_STORE=0 keeps the mapped bytes for
every model. With a store, the routed experts of each fused
block are copied once into the layout the kq store kernels read
(``mlx_kquant.moe_store_pack``: every block field in a plane of its own,
the 64 output columns of a tile side by side), and the block's gathers run
on that copy (``moe_store_glu``, ``moe_store_down``). The copy holds the
same bytes as the GGUF blocks, so a full store is as large as the experts.

    GMLX_MOE_EXPERT_STORE=full   every expert of every block that has a
        store codec. The blocks then serve every width from the store and
        drop their mapped expert tensors, so the experts are held once.
    GMLX_MOE_EXPERT_STORE=down   the down stacks only. The gate and up
        stacks stay the GGUF's mapped bytes and run the in-place kernels
        with their bits; the down stack of each block is held once, in the
        store. On M5 Max the store down is the faster one from 3 rows and
        the store GLU the slower one below 4.
    GMLX_MOE_EXPERT_STORE=<GiB>  a bounded store: that many GiB of slots,
        the same share of experts in every block, the hot experts first
        (GMLX_MOE_EXPERT_STORE_HOT, an npz with ``counts`` [blocks, experts];
        without it the lowest ids). The mapped tensors stay, and an expert
        with no slot is read from them in place by the same kernel.

A bounded store serves the widths of GMLX_MOE_EXPERT_STORE_ROWS (default
2 to 16 rows, the verify widths), and its GLU only from
GMLX_MOE_EXPERT_STORE_GLU_ROWS rows up (default 8: below that the in-place
GLU is the faster one on M5 Max); other widths keep the in-place kernels.
GMLX_MOE_EXPERT_STORE_BLOCKS=N builds the stores of the first N blocks only.
A block whose codecs have no store format, a block with a LoRA slot on,
output scales, streamed experts or a shared expert that leaves the fold at
some width keeps the in-place kernels. A load inside ``off`` (training, a
streamed teacher, an adapter on the experts or shared expert of a block),
one that streams its experts, runs on the CPU device or is sharded builds
no store, nor does a drafter whose loader does not give its block in the
file.

A full or down store is built at load, one block at a time, before anything reads
or wires the mapped stacks: each stack is read from the file past the page
cache into a staging buffer and packed from there, so neither the file
mapping nor the page cache holds a second copy, and the planes join the
Metal residency set with the model's other weights. A bounded store is
built at the first forward of any block.
"""

from __future__ import annotations

import contextlib
import contextvars
import os
import re
import sys
import threading
import time
import weakref
from typing import Any, Callable, cast

import mlx.core as mx
import mlx_kquant as kq
import numpy as np

from gmlx import mlx_guards
from gmlx.envflags import env_bool, env_int

from . import loadlog

# mlx-kquant's store ops (moe_store_*), from 0.5.14 on and probed at run
# time: one loosely typed name, so the module checks against releases
# without them.
_kq_store: Any = kq
ABSENT = 0x80000000
_BLOCKS: list = []
_DONE: list = []
_GONE: set = set()
_BUILT = [False]
# A harness turns a bounded store off and on in one process (the compiled
# verify bodies are dropped by the caller on a change).
USE = [True]
STATS: dict = {}


# The model types, with their GGUF archs, that get a full store when
# GMLX_MOE_EXPERT_STORE is not set: those its kernels were tuned on.
_DEFAULT_FULL = {"qwen4_exp": "qwen4exp"}


def model_type_of(model):
    """The model type of a text model, of the one a wrapper holds, or of
    the target whose layers an MTP head carries (its ``config.text``)."""
    text = getattr(getattr(model, "config", None), "text", None)
    for m in (model, getattr(model, "language_model", None), text):
        t = getattr(m, "model_type", None)
        if isinstance(t, str):
            return t
    return None


_OFF: contextvars.ContextVar = contextvars.ContextVar(
    "gmlx_expert_store_off", default=None)


@contextlib.contextmanager
def off(reason: str):
    """No store for the loads inside: their blocks keep the mapped expert
    stacks, for a mode the store cannot serve (training, streamed
    experts). ``reason`` says which."""
    tok = _OFF.set(reason)
    try:
        yield
    finally:
        _OFF.reset(tok)


# An adapter tensor on an MoE block's experts or shared expert.
_ON_STORE_BLOCK = re.compile(
    r"^blk\.\d+\.ffn_(?:gate|up|down|gate_up)_(?:exps|shexp)\.")


def adapter_scope(adapter_paths):
    """``off`` for a load that an adapter of ``adapter_paths`` (one path
    or several) attaches to an MoE block's experts or shared expert: the
    install gives that block's store up, so a build first would be load
    time for nothing. Read from the adapter headers; a null context when
    no adapter takes such a target."""
    from .headerscan import scan_gguf

    paths = ((adapter_paths,) if isinstance(adapter_paths, str)
             else tuple(adapter_paths or ()))
    for path in paths:
        try:
            names = [t.name for t in scan_gguf(path).tensors]
        except Exception:
            continue                       # the load reports the file
        if any(_ON_STORE_BLOCK.match(n) for n in names):
            return off("an adapter on its experts")
    return contextlib.nullcontext()


_DRAFTER: contextvars.ContextVar = contextvars.ContextVar(
    "gmlx_expert_store_drafter", default=False)


@contextlib.contextmanager
def drafter_load():
    """The loads inside are speculative drafters. A drafter's module path
    does not give its block in the file, so only one whose loader sets
    ``_kq_store_blk_base`` to that block gets a full or down store."""
    tok = _DRAFTER.set(True)
    try:
        yield
    finally:
        _DRAFTER.reset(tok)


def store_mode(model_type=None):
    """None when off, ("full", 0), ("down", 0) or ("cap", bytes). With the
    variable unset, full for a model type in _DEFAULT_FULL, else off.
    Inside ``off``, None."""
    if _OFF.get() is not None:
        return None
    raw = os.environ.get("GMLX_MOE_EXPERT_STORE")
    if raw is None:
        return ("full", 0) if model_type in _DEFAULT_FULL else None
    v = raw.strip().lower()
    if v in ("", "0", "off"):
        return None
    if v in ("1", "full"):
        return ("full", 0)
    if v == "down":
        return ("down", 0)
    try:
        return ("cap", int(float(v.rstrip("g")) * 2**30))
    except ValueError:
        raise ValueError(
            "GMLX_MOE_EXPERT_STORE takes full, down or a size in GiB, got "
            f"{v!r}") from None


def _rows():
    v = os.environ.get("GMLX_MOE_EXPERT_STORE_ROWS", "2,16")
    lo, hi = (int(x) for x in v.split(","))
    glo = env_int("GMLX_MOE_EXPERT_STORE_GLU_ROWS", 8)
    return lo, hi, max(lo, glo)


def _wide_rows() -> int:
    """Rows of a forward from which a full store runs its wide form (the
    segment GEMM on the planes). 0 keeps the tile kernels at every width."""
    return env_int("GMLX_MOE_EXPERT_STORE_WIDE_ROWS", 160)


class ExpertStore:
    """One block's store: the slot table, the planes of gate, up and down,
    and the arrays the kernels take as the in-place expert stacks. ``wide``
    is the row count from which a full store runs the wide form (0: never),
    with the codecs and shapes that form takes in ``seg``. ``kinds``
    names the stacks the store holds in place of the mapped tensors: all
    three, or the down stack alone (then ``g`` and ``u`` are empty and the
    GLU runs in place)."""

    __slots__ = ("slots", "g", "u", "d", "gw", "uw", "dw", "lo", "hi",
                 "glo", "nbytes", "held", "wide", "seg", "kinds")

    def serves_wide(self, t: int) -> bool:
        return 0 < self.wide <= t

    def serves(self, t: int) -> bool:
        return self.lo <= t <= self.hi

    def serves_glu(self, t: int) -> bool:
        return self.glo <= t <= self.hi


def register(block, model_type=None) -> None:
    """Note a fused block of a model of ``model_type`` at install time; its
    store is built later."""
    if store_mode(model_type) is not None and hasattr(kq, "moe_store_pack"):
        _BLOCKS.append(weakref.ref(block))
        # A bounded store of this load is built at its first forward too.
        _BUILT[0] = False


def _projs(block):
    fold = getattr(block, "_kq_fold_projs", None)
    return fold() if fold is not None else None


def _eligible(block, kind: str = "full") -> bool:
    if block.__dict__.get("_kq_lora_slots") is not None:
        return False
    if kind != "cap" and (not getattr(block, "_kq_shexp_fold", True)
                          or getattr(block, "_kq_fold_skip_t", ())):
        # Where the fold is off the block runs its members on the mapped
        # stacks, which a full or down store lets go.
        return False
    projs = _projs(block)
    if projs is None:
        return False
    wg, wu, wd, sg, su, sd = projs
    if wu.kquant_type != wg.kquant_type or su.kquant_type != sg.kquant_type:
        return False
    for p in projs:
        if getattr(p, "out_scale", None) is not None:
            return False
    if wg.weight.ndim != 3 or wd.weight.ndim != 3 or sg.weight.ndim != 2:
        return False
    if kind == "down":
        # The wide forward reads the down planes through the segment GEMM,
        # and the leaf that does so stands in for the stock class only.
        from mlx_kquant.nn import KQuantSwitchLinear

        seg_ok = getattr(kq, "moe_store_seg_supported", None)
        return (type(wd) is KQuantSwitchLinear and "bias" not in wd
                and seg_ok is not None
                and _kq_store.moe_store_supported(wd.kquant_type, sd.kquant_type)
                and seg_ok(wd.kquant_type, int(wg.weight.shape[1])))
    return (_kq_store.moe_store_supported(wg.kquant_type, sg.kquant_type)
            and _kq_store.moe_store_supported(wd.kquant_type, sd.kquant_type))


_DOWN_LEAF: list = [None]


def _down_leaf_class():
    """The class a down store gives the block's down leaf: every caller of
    the members' sorted path (the prefill widths) gets the segment GEMM on
    the store planes, the bits of the in-place segment GEMM. The narrow
    widths never call the leaf (the fused block runs kq.moe_store_down)."""
    cls = _DOWN_LEAF[0]
    if cls is not None:
        return cls
    from mlx_kquant.nn import KQuantSwitchLinear

    class _StoreDownSwitchLinear(KQuantSwitchLinear):
        def __call__(self, x, indices, sorted_indices=False, x_rows=None):
            st = self.__dict__.get("_kq_down_store")
            if st is None:
                return super().__call__(
                    x, indices, sorted_indices=sorted_indices, x_rows=x_rows)
            if (not sorted_indices or indices.ndim != 1 or x_rows is not None
                    or x.size != indices.size * x.shape[-1]
                    or mx.default_device() != mx.gpu):
                raise RuntimeError(
                    "GMLX_MOE_EXPERT_STORE=down holds this block's down "
                    "experts in the store only; this forward needs the "
                    "mapped stack: load without the store")
            d_codec, n_exp, d_out = st.seg
            xf = x.reshape(indices.size, x.shape[-1])
            if xf.dtype == mx.float32:
                xf = xf.astype(mx.bfloat16)
            idx = indices
            if idx.dtype != mx.uint32:
                idx = idx.astype(mx.uint32)
            tr = kq.seg_tile_rows(idx.size, n_exp, xf.shape[-1], d_codec)
            maps = kq.expert_tile_map(idx, n_exp, tile_rows=tr)
            y = _kq_store.moe_store_seg(xf, st.d, d_codec, *maps, n_out=d_out,
                                 tile_rows=tr)
            return y[:, None, :]

    _StoreDownSwitchLinear.__name__ = "_StoreDownSwitchLinear"
    _DOWN_LEAF[0] = _StoreDownSwitchLinear
    return _StoreDownSwitchLinear


def _pack_block(b, ids, slots, kind, lo, hi, glo, src=None):
    """Build one block's store from its expert stacks, or from ``src``,
    the same three stacks' bytes held elsewhere. In full mode the stacks
    are then let go: the module keeps a placeholder with the stack's shape
    and byte count (so sizes read as before) and the kernels take a
    one-expert stub, which no slot ever reads. ``kind`` "down" stores the
    down stack alone (``src`` is then that one stack): gate and up stay as
    they are."""
    wg, wu, wd = _projs(b)[:3]
    st = ExpertStore()
    st.slots = mx.array(slots)
    ids_a = mx.array(ids)
    k_glu = wd.weight.shape[1]
    k_down = wg.weight.shape[1]
    st.gw, st.uw, st.dw = wg.weight, wu.weight, wd.weight
    if kind == "down":
        old = (wd.weight,)
        return _pack_down(b, st, ids_a, k_down, src), old
    st.kinds = ("gate", "up", "down")
    sg, su, sd = src or (wg.weight, wu.weight, wd.weight)
    st.g = _kq_store.moe_store_pack(sg, wg.kquant_type, k_glu, ids_a)
    st.u = _kq_store.moe_store_pack(su, wu.kquant_type, k_glu, ids_a)
    st.d = _kq_store.moe_store_pack(sd, wd.kquant_type, k_down, ids_a)
    st.lo, st.hi, st.glo = lo, hi, glo
    # The wide form reads slots as expert ids, so it needs every expert
    # stored in id order: a full store.
    st.wide, st.seg = 0, None
    seg_ok = getattr(kq, "moe_store_seg_supported", None)
    if (kind == "full" and seg_ok is not None
            and seg_ok(wg.kquant_type, k_glu)
            and seg_ok(wd.kquant_type, k_down)):
        st.wide = _wide_rows()
        st.seg = (wg.kquant_type, wd.kquant_type, int(wg.weight.shape[0]),
                  int(wg.weight.shape[1]), int(wd.weight.shape[1]))
    old = ()
    holders: list = []
    if kind == "full":
        from gmlx.stream.stack_unmap import placeholder

        old = (wg.weight, wu.weight, wd.weight)
        st.gw, st.uw, st.dw = (
            mx.zeros((1,) + tuple(w.shape[1:]), dtype=w.dtype) for w in old)
        holders = [placeholder(w) for w in old]
    mx.eval(st.slots, st.g, st.u, st.d, st.gw, st.uw, st.dw)
    st.nbytes = sum(p.nbytes for ps in (st.g, st.u, st.d) for p in ps)
    st.held = len(ids)
    if kind == "full":
        wg.weight, wu.weight, wd.weight = holders
        object.__setattr__(b, "_kq_store_only", True)
        if st.wide > 0:
            object.__setattr__(b.switch_mlp, "_kq_wide_store", st)
    object.__setattr__(b, "_kq_expert_store", st)
    return st, old


def _pack_down(b, st, ids_a, k_down, src):
    """The down store of one block: the down planes, a placeholder where
    the mapped down stack stood, and the down leaf on the class that reads
    the planes at the members' sorted widths. The store serves every width
    of the down and none of the GLU."""
    from gmlx.stream.stack_unmap import placeholder

    wd = _projs(b)[2]
    old = wd.weight
    st.kinds = ("down",)
    st.g, st.u = (), ()
    st.d = _kq_store.moe_store_pack(src[0] if src else old, wd.kquant_type, k_down,
                             ids_a)
    st.lo, st.hi, st.glo = 1, 1 << 30, 1 << 31
    st.wide = 0
    st.seg = (wd.kquant_type, int(old.shape[0]), int(old.shape[1]))
    st.dw = mx.zeros((1,) + tuple(old.shape[1:]), dtype=old.dtype)
    holder = placeholder(old)
    mx.eval(st.slots, st.d, st.dw)
    st.nbytes = sum(p.nbytes for p in st.d)
    st.held = int(ids_a.size)
    wd.weight = holder
    wd.__class__ = _down_leaf_class()
    object.__setattr__(wd, "_kq_down_store", st)
    object.__setattr__(b, "_kq_store_only", True)
    object.__setattr__(b, "_kq_store_down", True)
    object.__setattr__(b, "_kq_expert_store", st)
    return st


def _stream_bound(model) -> bool:
    """True when the load is over the wired budget, so its experts stream
    from the file and are never held: no store for such a model."""
    from mlx.utils import tree_flatten

    from gmlx.device import metal_value

    try:
        budget = int(0.9 * metal_value("max_recommended_working_set_size"))
    except Exception:
        return False
    flat = cast(list[tuple[str, mx.array]], tree_flatten(model.parameters()))
    return sum(a.nbytes for _, a in flat) > budget


# msync(2) flag on Darwin.
_MS_INVALIDATE = 0x2
_READ_CHUNK = 32 << 20
_READERS = 8
_CHECK = 4096


class _FileStacks:
    """The expert stacks of a load's GGUF shards, read from the file past
    the page cache (F_NOCACHE, the feeders' aligned reads) into staging
    buffers the pack kernel takes in place of the mapped tensors."""

    def __init__(self, paths):
        from gmlx.stream.prefetch import expert_offset_map

        self.ranges = {
            li: {kind: (path, off, nbytes)
                 for path, off, nbytes, _, kind in rs}
            for li, rs in expert_offset_map(paths[0]).items()}
        self.fds: dict = {}
        self.stage: dict = {}
        self.pool = None
        self.read_bytes = 0
        self.peak_stage = 0
        self._libc = None

    def _fd(self, path):
        import fcntl

        ent = self.fds.get(path)
        if ent is None:
            fd = os.open(path, os.O_RDONLY)
            if hasattr(fcntl, "F_NOCACHE"):
                fcntl.fcntl(fd, fcntl.F_NOCACHE, 1)
            ent = self.fds[path] = (fd, os.fstat(fd).st_size)
        return ent

    def read(self, li, ws, kinds=("gate", "up", "down")):
        """The stacks ``kinds`` of layer ``li`` (gate, up and down, or the
        down alone) as arrays shaped
        like the mapped ``ws``, or None when the file's ranges do not match
        them (the caller then packs from the mapping). The arrays are
        staging buffers reused by the next call: evaluate what reads them
        first."""
        from concurrent.futures import ThreadPoolExecutor

        from gmlx.stream.feeder_common import read_range_aligned

        rs = self.ranges.get(li)
        if rs is None or sorted(rs) != ["down", "gate", "up"]:
            return None
        todo = []
        for kind, w in zip(kinds, ws, strict=True):
            path, off, nbytes = rs[kind]
            if (nbytes != w.nbytes or w.dtype != mx.uint8
                    or nbytes > 0x7fffffff):
                return None
            todo.append((path, off, nbytes, w))
        # One buffer per stack size in use: gate and up share a size, so
        # each stack of the block has its own.
        sizes = [t[2] for t in todo]
        for n in [n for n in self.stage if n not in sizes]:
            del self.stage[n]
        if self.pool is None:
            self.pool = ThreadPoolExecutor(
                _READERS, thread_name_prefix="gmlx-store-read")
        out = []
        taken: dict = {}
        jobs = []
        for path, off, nbytes, w in todo:
            bufs = self.stage.setdefault(nbytes, [])
            i = taken.get(nbytes, 0)
            taken[nbytes] = i + 1
            if i >= len(bufs):
                bufs.append(kq.arena_alloc([nbytes]))
            arr, mv = bufs[i]
            fd, size = self._fd(path)
            for a in range(0, nbytes, _READ_CHUNK):
                e = min(a + _READ_CHUNK, nbytes)
                jobs.append(self.pool.submit(
                    read_range_aligned, fd, mv[a:e], off + a, size))
            out.append((arr, mv, w))
        for j in jobs:
            j.result()
        self.peak_stage = max(
            self.peak_stage,
            sum(n * len(b) for n, b in self.stage.items()))
        res = []
        for arr, mv, w in out:
            # The file range against the mapped tensor it stands for: the
            # first and the last page of each, read on the host.
            try:
                ref = memoryview(w).cast("B")
            except (TypeError, ValueError, BufferError):
                return None
            if (bytes(ref[:_CHECK]) != bytes(mv[:_CHECK])
                    or bytes(ref[-_CHECK:]) != bytes(mv[-_CHECK:])):
                return None
            self.read_bytes += w.nbytes
            res.append(arr.reshape(w.shape))
        return tuple(res)

    def drop_cached(self, li, kinds=("gate", "up", "down")) -> int:
        """Drop the page cache's copy of layer ``li``'s stacks ``kinds``, the whole
        pages inside each range (``msync`` with MS_INVALIDATE through a
        mapping of the range). The store holds these bytes now, and the
        kernel does not give the cache up fast enough on its own: with free
        memory gone it compresses the planes just built instead. The bytes
        of the ranges dropped."""
        import ctypes
        import mmap

        libc = self._libc
        if libc is None:
            libc = self._libc = ctypes.CDLL(None, use_errno=True)
            libc.mmap.restype = ctypes.c_void_p
            libc.mmap.argtypes = [
                ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int,
                ctypes.c_int, ctypes.c_longlong]
            libc.msync.argtypes = [
                ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
            libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        page = mmap.PAGESIZE
        done = 0
        for kind, (path, off, nbytes) in (self.ranges.get(li) or {}).items():
            if kind not in kinds:
                continue
            a = (off + page - 1) & ~(page - 1)
            b = (off + nbytes) & ~(page - 1)
            if b <= a:
                continue
            fd, _ = self._fd(path)
            addr = libc.mmap(None, b - a, mmap.PROT_READ, mmap.MAP_SHARED,
                             fd, a)
            if addr in (None, ctypes.c_void_p(-1).value):
                continue
            try:
                if libc.msync(addr, b - a, _MS_INVALIDATE) == 0:
                    done += b - a
            finally:
                libc.munmap(addr, b - a)
        return done

    def close(self):
        if self.pool is not None:
            self.pool.shutdown(wait=True)
        self.stage.clear()
        for fd, _ in self.fds.values():
            os.close(fd)
        self.fds.clear()


_PACE_STALL_S = 0.5
_PACE_MARGIN = 256 << 20


def _free_bytes():
    from gmlx.serve.kernel_vm import _sysctl_count

    n = _sysctl_count("vm.page_free_count")
    return None if n is None else n * os.sysconf("SC_PAGESIZE")


def _wait_for_free(need: int) -> float:
    """Hold the build until free memory covers the next block. The build
    takes memory far faster than the kernel drops file cache for it, and
    with free memory gone the kernel compresses anonymous pages instead
    (measured: 6 GiB in a second at 36 blocks, with 64 GiB of cache
    left). Waits while free memory is short and still rising; a stall of
    ``_PACE_STALL_S`` ends the wait, so a build is slowed, never refused.
    The seconds waited."""
    t0 = time.perf_counter()
    best = _free_bytes()
    if best is None:
        return 0.0
    last = t0
    while best < need + _PACE_MARGIN:
        time.sleep(0.005)
        free = _free_bytes()
        now = time.perf_counter()
        if free is None:
            break
        if free > best + (32 << 20):
            best, last = free, now
        elif now - last > _PACE_STALL_S:
            break
    return time.perf_counter() - t0


def _layer_index(name: str):
    m = re.search(r"(?:^|\.)layers\.(\d+)\.", name + ".")
    return int(m.group(1)) if m else None


def _blk(model, li):
    """The GGUF block number of ``model``'s layer ``li``. An MTP drafter
    sets ``_kq_store_blk_base``: its layer 0 is block num_hidden_layers of
    a NextN sidecar, and None for a file with no block names (no store,
    see ``drafter_load``)."""
    base = getattr(model, "_kq_store_blk_base", 0)
    return None if li is None or base is None else li + base


def _off_reason(paths=None, blocks=(), model=None) -> str:
    """Why a load of ``paths`` builds no full or down store, or empty."""
    from gmlx.device import cpu_default_device
    from gmlx.stream.table_stream import experts_stream_for_memory, latched

    if latched():
        return "the load streams its experts"
    if paths and experts_stream_for_memory(paths[0]):
        return "the check before the load streams its experts"
    if cpu_default_device():
        return "the default device is the CPU"
    if any(getattr(b, "sharding_group", None) is not None for b in blocks):
        return "the model is sharded"
    if (model is not None and _DRAFTER.get()
            and not isinstance(getattr(model, "_kq_store_blk_base", None), int)):
        # Its module path would read, and a release map back, the
        # target's blocks of a shared file.
        return "the drafter's blocks in its file are not known"
    return ""


def stored_file_ranges(paths) -> dict:
    """The byte ranges, by real shard path, of the expert stacks a full
    or down store will hold in place of the mapped tensors: the page-cache
    populate leaves them out, since the store reads them past the cache.
    Empty with the lever off or without store kernels."""
    if not paths or not hasattr(kq, "moe_store_pack"):
        return {}
    try:
        from gmlx.load.headerscan import scan_gguf
        from gmlx.stream.prefetch import _EXPS_RE

        mode = store_mode()
        if os.environ.get("GMLX_MOE_EXPERT_STORE") is None:
            arch = scan_gguf(paths[0], include_tensors=False,
                             array_limit=0).kv.get("general.architecture")
            mode = store_mode(next((t for t, a in _DEFAULT_FULL.items()
                                    if a == arch), None))
        if mode is None or mode[0] not in ("full", "down") or _off_reason(paths):
            return {}
        n_blocks = env_int("GMLX_MOE_EXPERT_STORE_BLOCKS", 0)
        layers: dict = {}
        for path in paths:
            scan = scan_gguf(path, array_limit=0)
            for t in scan.tensors:
                m = _EXPS_RE.fullmatch(t.name)
                if m:
                    layers.setdefault(int(m.group(1)), []).append(
                        (path, scan.data_offset + t.offset, t.nbytes,
                         m.group(2), t.type_name.lower()))
        out: dict = {}
        n = 0
        for li in sorted(layers):
            rs = layers[li]
            if sorted(r[3] for r in rs) != ["down", "gate", "up"]:
                continue
            if mode[0] == "down":
                rs = [r for r in rs if r[3] == "down"]
            if not all(_kq_store.moe_store_supported(r[4], "") for r in rs):
                continue
            # As build_at_load counts the lever: the blocks it stores.
            if n_blocks > 0 and n >= n_blocks:
                break
            n += 1
            for path, off, nbytes, _, _ in rs:
                out.setdefault(os.path.realpath(path), []).append(
                    (off, off + nbytes))
        return {k: sorted(v) for k, v in out.items()}
    except Exception:
        return {}


# The thread of the last load's kernel warm (tests join it).
_WARM: list = [None]


def _act_dtypes(model) -> list:
    """The half float dtypes ``model``'s parameters hold: the dtype its
    blocks hand the store kernels (a float32 residual is cast to
    bfloat16)."""
    from mlx.utils import tree_flatten

    flat = cast(list[tuple[str, mx.array]], tree_flatten(model.parameters()))
    seen = {p.dtype for _, p in flat if p.dtype in (mx.bfloat16, mx.float16)}
    seen.add(mx.bfloat16)
    return sorted(seen, key=str)


def warm_kernels(blocks, dtypes, *, log=None, kind="full"):
    """Build, on a thread beside the store build, the compute pipelines the
    store kernels of ``blocks`` reach (the narrow tile kernels with their
    plan and combine, and the wide form), one set a codec signature, so a
    first request does not wait on pipeline builds. Evaluates nothing.
    GMLX_MOE_EXPERT_STORE_WARM=0 turns it off. Returns the thread."""
    warm = getattr(kq, "moe_store_warm", None)
    if warm is None or not env_bool("GMLX_MOE_EXPERT_STORE_WARM", True):
        return None
    sigs = []
    for b in blocks:
        projs = _projs(b)
        if projs is None:
            continue
        wg, _, wd, sg, _, sd = projs
        # A down store has no GLU kernels: empty names skip them.
        sig = (wg.kquant_type if kind == "full" else "",
               sg.kquant_type if kind == "full" else "", wd.kquant_type,
               sd.kquant_type, int(wd.weight.shape[1]),
               int(wg.weight.shape[1]))
        if sig not in sigs:
            sigs.append(sig)
    if not sigs:
        return None
    say = log or print

    def run():
        t0 = time.perf_counter()
        n = 0
        try:
            for g, gs, d, ds, k_glu, k_down in sigs:
                n += warm(g, gs, d, ds, list(dtypes), k_glu=k_glu,
                          k_down=k_down)
        except Exception as e:
            say(f"[expert-store] kernel warm stopped ({e}); kernels build "
                "at their first use")
            return
        dt = time.perf_counter() - t0
        STATS.update(warm_pipelines=n, warm_seconds=dt)
        say(f"[expert-store] {n} kernel pipelines of {len(sigs)} codec "
            f"sets ready in {dt:.1f} s, built beside the load")

    th = threading.Thread(target=run, name="expert-store-warm", daemon=True)
    th.start()
    _WARM[0] = th
    return th


def build_at_load(model, *, log=None, drop=(), paths=None) -> int:
    """GMLX_MOE_EXPERT_STORE=full: build the stores of ``model``'s fused
    blocks right after its weights are assigned, before anything touches or
    wires them. One block at a time: its three stacks are read from the
    file (``paths``, the load's shards) past the page cache into staging
    buffers and packed into the store's planes, then the mapped arrays,
    never read, are let go (``drop``: the loader's dicts that still name
    them), so the experts are held once and the mapped stacks never join
    the resident set. Without ``paths``, or for a stack the file ranges do
    not match, the pack reads the mapping. Returns the blocks built."""
    # Ids of a past load's arrays may belong to live arrays by now.
    _GONE.clear()
    mode = store_mode(model_type_of(model))
    if mode is None or mode[0] not in ("full", "down") or not _BLOCKS:
        return 0
    kind = mode[0]
    kinds = ("gate", "up", "down") if kind == "full" else ("down",)
    # Printed like the [stream] lines: the line says how the experts are held.
    say = log or print
    mine = {id(m) for _, m in model.named_modules()}
    todo = [b for b in (r() for r in _BLOCKS)
            if b is not None and id(b) in mine
            and b.__dict__.get("_kq_expert_store") is None]
    if not todo:
        return 0
    # Layer order, whatever order the blocks were installed in: the BLOCKS
    # lever and the populate's skipped ranges both count from layer 0.
    names = {id(m): name for name, m in model.named_modules()}
    todo.sort(key=lambda b: (_layer_index(names.get(id(b), "")) is None,
                             _layer_index(names.get(id(b), "")) or 0))
    why = _off_reason(paths, todo, model)
    if why:
        say(f"[expert-store] off for this load: {why}")
        _forget(todo)
        return 0
    if _stream_bound(model):
        say("[expert-store] off for this load: the model is over the wired "
            "budget and streams its experts from the file")
        _forget(todo)
        return 0
    n_blocks = env_int("GMLX_MOE_EXPERT_STORE_BLOCKS", 0)
    warm_kernels([b for b in todo if _eligible(b, kind)], _act_dtypes(model),
                 log=say, kind=kind)
    t0 = time.perf_counter()
    held = n = skipped = mapped = 0
    waited = 0.0
    pace = env_bool("GMLX_MOE_EXPERT_STORE_PACE", True)
    uncache = (sys.platform == "darwin"
               and env_bool("GMLX_MOE_EXPERT_STORE_DROP_CACHE", True))
    dropped = 0
    files = None
    if paths and env_bool("GMLX_MOE_EXPERT_STORE_FILE_READ", True):
        try:
            files = _FileStacks(list(paths))
        except Exception as e:
            say(f"[expert-store] file ranges unreadable ({e}); the store "
                "is packed from the mapping")
    try:
        for b in todo:
            if not _eligible(b, kind):
                skipped += 1
                continue
            if n_blocks > 0 and n >= n_blocks:
                break
            ws = tuple(p.weight for p in _projs(b)[:3])
            if kind == "down":
                ws = ws[2:]
            if pace:
                waited += _wait_for_free(sum(w.nbytes for w in ws))
            n_exp = ws[0].shape[0]
            ids = np.arange(n_exp, dtype=np.uint32)
            src = None
            li = _blk(model, _layer_index(names.get(id(b), "")))
            if files is not None and li is not None:
                src = files.read(li, ws, kinds)
            if src is None:
                mapped += 1
            del ws
            st, old = _pack_block(b, ids, ids.copy(), kind, 1, 1 << 30, 1,
                                  src=src)
            object.__setattr__(b, "_kq_store_src", (
                tuple(paths) if paths else None, li, weakref.ref(model)))
            mx.synchronize()
            del src
            _GONE.update(id(a) for a in old)
            for d in drop:
                drop_mapped(d)
            del old
            if files is not None and li is not None and uncache:
                dropped += files.drop_cached(li, kinds)
            held += st.nbytes
            n += 1
    finally:
        stage = 0
        if files is not None:
            stage = files.peak_stage
            files.close()
    _forget(todo)
    dt = time.perf_counter() - t0
    STATS.update(blocks=n, of=len(todo), bytes=held, expert_bytes=held,
                 share=1.0, seconds=dt, mode=kind, skipped=skipped,
                 mapped=mapped, stage_bytes=stage, paced_seconds=waited,
                 uncached_bytes=dropped)
    say(f"[expert-store] {kind}: {n} of {len(todo)} blocks"
        + (" (down stacks; gate and up stay mapped), " if kind == "down"
           else ", ")
        + f"{held / 2**30:.2f} GiB, built in {dt:.1f} s"
        + (f", {n - mapped} read from the file past the page cache "
           f"(staging {stage / 2**30:.2f} GiB)" if n > mapped else "")
        + (f", {mapped} packed from the mapping" if mapped else "")
        + (f", {waited:.1f} s of it waiting for free memory"
           if waited >= 0.05 else "")
        + (f"; {dropped / 2**30:.2f} GiB of their file ranges dropped "
           f"from the page cache" if dropped else "")
        + (f"; {skipped} blocks keep their mapped experts (a codec with "
           f"no store format, a LoRA slot or a fold that turns off)"
           if skipped else ""))
    return n


def held(model) -> tuple[int, set]:
    """Bytes of ``model``'s expert stacks that a full store holds in place
    of the mapped tensors, and the ids of the placeholders standing in the
    module tree for them. A resident wire leaves both out: the store is the
    one copy and a placeholder has nothing to wire."""
    nbytes = 0
    ids: set = set()
    if not _DONE:
        return nbytes, ids
    for _, m in model.named_modules():
        if not m.__dict__.get("_kq_store_only"):
            continue
        st = m.__dict__.get("_kq_expert_store")
        kinds = getattr(st, "kinds", ("gate", "up", "down"))
        for kind, p in zip(("gate", "up", "down"), _member_projs(m),
                           strict=True):
            if kind in kinds:
                nbytes += p.weight.nbytes
                ids.add(id(p.weight))
    return nbytes, ids


def release(model, reason: str, *, log=None) -> int:
    """Give up the full store of ``model``'s blocks: see ``release_blocks``."""
    if not _DONE:
        return 0
    blocks = [m for _, m in model.named_modules()
              if m.__dict__.get("_kq_store_only")]
    return release_blocks(blocks, reason, log=log)


def release_blocks(blocks, reason: str, *, log=None) -> int:
    """Put the mapped expert stacks back on store-only ``blocks`` and drop
    their stores, for a mode the store cannot serve (streamed experts, an
    adapter on an expert stack): the mode works as with the lever off. The
    stacks are mapped again from the load's shards, zero copy, each shard
    once per call, and join the load's untracked weight bytes; the
    recorded verify graphs that hold the planes go now. Raises when the
    store was built without the shard paths. Returns the blocks
    released."""
    blocks = [b for b in blocks if b.__dict__.get("_kq_store_only")]
    if not blocks:
        return 0
    from gmlx.load.headerscan import scan_gguf

    want: dict = {}
    for b in blocks:
        paths, li, _ = b.__dict__.get("_kq_store_src") or (None, None, None)
        if not paths or li is None:
            raise RuntimeError(
                f"GMLX_MOE_EXPERT_STORE=full holds this model's experts in "
                f"the store only and {reason}; load without the store")
        st = b.__dict__.get("_kq_expert_store")
        for kind in getattr(st, "kinds", ("gate", "up", "down")):
            want[f"blk.{li}.ffn_{kind}_exps.weight"] = (b, kind, paths)
    got: dict = {}
    for path in {p for _, _, ps in want.values() for p in ps}:
        names = [t.name for t in scan_gguf(path, array_limit=0).tensors]
        mine = [n for n in names if n in want]
        if not mine:
            continue
        res = kq.load_gguf(path, True, [n for n in names if n not in want])
        got.update(res[0] if isinstance(res, tuple) else res)
    missing = [n for n in want if n not in got]
    if missing:
        raise RuntimeError(
            f"GMLX_MOE_EXPERT_STORE=full: {reason}, and the file has no "
            f"tensor {missing[0]} to map the experts back from")
    gone: set = set()
    models: dict = {}
    for name, (b, kind, _) in want.items():
        proj = getattr(b.switch_mlp, f"{kind}_proj")
        if got[name].nbytes != proj.weight.nbytes:
            raise RuntimeError(
                f"GMLX_MOE_EXPERT_STORE=full: {reason}, and {name} does not "
                "match the stack the store was built from")
    for name, (b, kind, _) in want.items():
        proj = getattr(b.switch_mlp, f"{kind}_proj")
        proj.weight = got[name].reshape(proj.weight.shape)
    mapped: dict = {}
    for b in blocks:
        st = b.__dict__.pop("_kq_expert_store", None)
        b.__dict__.pop("_kq_store_only", None)
        b.switch_mlp.__dict__.pop("_kq_wide_store", None)
        if b.__dict__.pop("_kq_store_down", None):
            from mlx_kquant.nn import KQuantSwitchLinear

            leaf = b.switch_mlp.down_proj
            leaf.__dict__.pop("_kq_down_store", None)
            leaf.__class__ = KQuantSwitchLinear
        _, _, mref = b.__dict__.pop("_kq_store_src")
        m = mref() if mref is not None else None
        if m is not None:
            models[id(m)] = m
            kinds = getattr(st, "kinds", ("gate", "up", "down"))
            mapped[id(m)] = mapped.get(id(m), 0) + sum(
                p.weight.nbytes for k, p in zip(
                    ("gate", "up", "down"), _member_projs(b), strict=True)
                if k in kinds)
        if st is not None:
            for ps in (st.g, st.u, st.d):
                for a in ps:
                    gone.add(id(a))
                    if hasattr(kq, "residency_erase"):
                        kq.residency_erase(a)
    if gone and hasattr(kq, "residency_commit"):
        kq.residency_commit()
    from gmlx.gen.prefill_decay import add_untracked_weights

    ids = {id(b) for b in blocks}
    for m in models.values():
        held_list = m.__dict__.get("_kq_resident_arrays")
        if held_list:
            held_list[:] = [a for a in held_list if id(a) not in gone]
        # The load counted the store as tracked memory; the stacks mapped
        # again are views MLX does not count.
        add_untracked_weights(mapped[id(m)],
                              getattr(m, "_kq_weights_key", None))
        _drop_verify_records(m, ids)
    _DONE[:] = [r for r in _DONE if r() is not None and id(r()) not in ids]
    mx.clear_cache()
    (log or print)(
        f"[expert-store] released on {len(blocks)} blocks: {reason}; their "
        "experts are read in place from the file")
    return len(blocks)


def _drop_verify_records(model, block_ids) -> None:
    """Release now the recorded verify graphs (qwen4_exp) of the layers
    that hold the blocks ``block_ids``: they keep the planes as
    constants, and their next use would drop them only then."""
    q4 = sys.modules.get("gmlx.models.qwen4_exp.model")
    if q4 is None:
        return
    layers = [m for _, m in model.named_modules()
              if "_verify_fused" in vars(m)
              and any(id(s) in block_ids for s in m.modules())]
    q4.drop_verify_records(layers)


def planes(model) -> list:
    """The store planes and slot tables of ``model``'s blocks: a resident
    wire inserts them with the model's weights."""
    out: list = []
    if not _DONE:
        return out
    for _, m in model.named_modules():
        st = m.__dict__.get("_kq_expert_store")
        if st is None or not m.__dict__.get("_kq_store_only"):
            continue
        for ps in (st.g, st.u, st.d):
            out.extend(ps)
    return out


def _member_projs(block):
    sw = block.switch_mlp
    return (sw.gate_proj, sw.up_proj, sw.down_proj)


def drop_mapped(arrays: dict, last: bool = False) -> None:
    """Take the mapped stacks a full store replaced out of a loader's dict
    that still names them, so their views of the file are freed. ``last``:
    no other dict names them, so the record of them is dropped too."""
    for k in [k for k, v in arrays.items() if id(v) in _GONE]:
        del arrays[k]
    if last:
        _GONE.clear()


def _forget(blocks) -> None:
    gone = {id(b) for b in blocks}
    _BLOCKS[:] = [r for r in _BLOCKS if r() is not None and id(r()) not in gone]
    _DONE.extend(weakref.ref(b) for b in blocks)


def _build_all() -> None:
    """The bounded store, built at the first forward of any block: the
    mapped stacks stay, so it has nothing to do before the load's wire."""
    _BUILT[0] = True
    mode = store_mode()
    blocks = [b for b in (r() for r in _BLOCKS) if b is not None]
    # A load after the first build adds its blocks; the built keep theirs.
    todo = [(i, b) for i, b in enumerate(blocks) if _eligible(b, "cap")
            and b.__dict__.get("_kq_expert_store") is None]
    if mode is None or not todo:
        return
    kind, cap = mode
    if kind in ("full", "down"):
        # A loader with no build_at_load call: the stacks are wired by now
        # and a store would be a second copy. The blocks stay in place.
        loadlog.warn(f"[expert-store] {kind}: this load path builds no store; "
                     f"{len(todo)} blocks keep their mapped experts")
        return
    total = 0
    for _, b in todo:
        wg, wu, wd = _projs(b)[:3]
        total += wg.weight.nbytes + wu.weight.nbytes + wd.weight.nbytes
    share = min(1.0, cap / total)
    hot = None
    hot_path = os.environ.get("GMLX_MOE_EXPERT_STORE_HOT", "")
    if hot_path and share < 1.0:
        hot = np.load(hot_path)["counts"]
    lo, hi, glo = _rows()
    n_blocks = env_int("GMLX_MOE_EXPERT_STORE_BLOCKS", 0)
    if n_blocks > 0:
        todo = todo[:n_blocks]
    t0 = time.perf_counter()
    held = 0
    for i, b in todo:
        n_exp = _projs(b)[0].weight.shape[0]
        n = n_exp if share >= 1.0 else max(1, int(n_exp * share))
        if n >= n_exp:
            ids = np.arange(n_exp, dtype=np.uint32)
        elif hot is not None and i < len(hot):
            ids = np.sort(np.argsort(-hot[i], kind="stable")[:n]).astype(
                np.uint32)
        else:
            ids = np.arange(n, dtype=np.uint32)
        slots = np.full((n_exp,), ABSENT, np.uint32)
        slots[ids] = np.arange(len(ids), dtype=np.uint32)
        st, _ = _pack_block(b, ids, slots, "cap", lo, hi, glo)
        held += st.nbytes
    mx.synchronize()
    dt = time.perf_counter() - t0
    STATS.update(blocks=len(todo), of=len(blocks), bytes=held,
                 expert_bytes=total, share=share, seconds=dt, mode=kind)
    loadlog.info(
        f"[expert-store] {kind}: {len(todo)} of {len(blocks)} blocks, "
        f"{held / 2**30:.2f} GiB of {total / 2**30:.2f} GiB expert bytes, "
        f"built in {dt:.1f} s")


def lookup(block, t: int):
    """The block's store when it serves a forward of ``t`` rows."""
    if not _BLOCKS and not _DONE:
        return None
    if _BLOCKS and not _BUILT[0]:
        _build_all()
    st = block.__dict__.get("_kq_expert_store")
    if st is None or not st.serves(t):
        return None
    if not USE[0] and not block.__dict__.get("_kq_store_only"):
        return None
    return st


def wide(block, t: int) -> bool:
    """True when a full store runs a forward of ``t`` rows on its wide
    form: the block's members' sorted path, with the segment GEMM reading
    the planes (modules._kq_prefill_store)."""
    st = block.__dict__.get("_kq_expert_store")
    return (st is not None and bool(block.__dict__.get("_kq_store_only"))
            and st.serves_wide(t))


# Gate codecs whose gate, up and SwiGLU run as one kq kernel on the wide
# form. iq2_s measured slower that way and keeps two calls.
_WIDE_PAIR_CODECS = ("iq3_xxs", "iq4_xs")


def _wide_pair_on(glu, g_codec: str) -> bool:
    """True when the wide form runs gate, up and the activation as one
    kernel: a paired codec, the plain SwiGLU (whose bits the kernel has),
    and GMLX_MOE_WIDE_PAIR not 0."""
    if g_codec not in _WIDE_PAIR_CODECS or not hasattr(kq, "moe_store_seg_pair"):
        return False
    from gmlx._vendor.mlx_lm.models.switch_layers import SwiGLU

    return (type(glu.activation) is SwiGLU
            and os.environ.get("GMLX_MOE_WIDE_PAIR", "1") != "0")


def _wide_sorted(glu, x, indices):
    """The sorted wide forward up to the down projection: ``(y, order)``
    with y ``[pairs, N]`` in expert order and order the argsort of the
    flat expert ids."""
    st = glu.__dict__["_kq_wide_store"]
    g_codec, d_codec, n_exp, inter, d_out = st.seg
    k = indices.shape[-1]
    flat = indices.flatten()
    order = mx.argsort(flat)
    idx = flat[order]
    if idx.dtype != mx.uint32:
        idx = idx.astype(mx.uint32)
    d_in = x.shape[-1]
    xf = x.reshape(-1, d_in)
    if xf.dtype == mx.float32:
        # The in-place segment GEMM takes a float32 input as bfloat16.
        xf = xf.astype(mx.bfloat16)
    rows = idx.size
    tr = kq.seg_tile_rows(rows, n_exp, d_in, g_codec)
    maps = kq.expert_tile_map(idx, n_exp, tile_rows=tr)
    xr = mlx_guards.floor_divide(order, k)
    if _wide_pair_on(glu, g_codec):
        (h,) = _kq_store.moe_store_seg_pair(xf, st.g, st.u, g_codec, *maps,
                                     n_out=inter, tile_rows=tr, x_rows=xr,
                                     fused=True)
    else:
        x_gate = _kq_store.moe_store_seg(xf, st.g, g_codec, *maps, n_out=inter,
                                  tile_rows=tr, x_rows=xr)
        x_up = _kq_store.moe_store_seg(xf, st.u, g_codec, *maps, n_out=inter,
                                tile_rows=tr, x_rows=xr)
        h = glu.activation(x_up, x_gate)
    trd = kq.seg_tile_rows(rows, n_exp, inter, d_codec)
    if trd != tr:
        maps = kq.expert_tile_map(idx, n_exp, tile_rows=trd)
    y = _kq_store.moe_store_seg(h, st.d, d_codec, *maps, n_out=d_out, tile_rows=trd)
    return y, order


def wide_glu(glu, x, indices):
    """A SwitchGLU's sorted wide forward on its block's full store:
    mlx-lm SwitchGLU.__call__ at sorting widths with each projection the
    segment GEMM of kq.gather_qmm_seg reading the store planes, so the
    values are those of the in-place sorted path where that path takes the
    segment GEMM. Unmixed ``[..., k, N]``. None when the block has no wide
    store."""
    if glu.__dict__.get("_kq_wide_store") is None:
        return None
    from gmlx._vendor.mlx_lm.models.switch_layers import _scatter_unsort

    y, order = _wide_sorted(glu, x, indices)
    y = _scatter_unsort(y[:, None, :], mx.argsort(order), indices.shape)
    return y.squeeze(-2)


# out[t, d] = sum over j of y[inv[t, j], d] * s[t, j]: the unsort, the score
# product and the sum over the routes of the op chain in one pass. The
# products and the sum run in the activation type, the sum in the order of
# MLX's column reduce (min(8, K) partial sums over the routes j = i, i + 8,
# then the partial sums added in sequence), so the bits are the op chain's.
_WIDE_COMBINE_SRC = """
    const uint d = thread_position_in_grid.x;
    const uint t = thread_position_in_grid.y;
    const uint D = y_shape[1];
    constexpr uint n = K < 8 ? K : 8;
    InT lanes[n];
    for (uint i = 0; i < n; ++i) lanes[i] = InT(0);
    for (uint j = 0; j < K; ++j) {
      const size_t pair = size_t(t) * K + j;
      const InT prod = y[size_t(inv[pair]) * D + d] * s[pair];
      lanes[j % n] = prod + lanes[j % n];
    }
    InT total = lanes[0];
    for (uint i = 1; i < n; ++i) total = lanes[i] + total;
    out[size_t(t) * D + d] = total;
"""
_wide_combine_kernel: Callable[..., list[mx.array]] | None = None


def _wide_combine_on() -> bool:
    return os.environ.get("GMLX_MOE_WIDE_COMBINE", "1") != "0"


def wide_combine(y, order, scores):
    """``[t, N]``: the rows of y ``[t * k, N]`` (sorted pairs, ``order`` the
    argsort that sorted them) mixed by scores ``[t, k]`` of y's dtype."""
    global _wide_combine_kernel
    if _wide_combine_kernel is None:
        _wide_combine_kernel = cast(Callable[..., list[mx.array]], mx.fast.metal_kernel(
            name="moe_wide_combine", input_names=["y", "inv", "s"],
            output_names=["out"], source=_WIDE_COMBINE_SRC))
    t, k = scores.shape
    pairs = t * k
    inv = mx.put_along_axis(
        mx.zeros((pairs,), dtype=mx.uint32), order.astype(mx.uint32),
        mx.arange(pairs, dtype=mx.uint32), axis=0)
    d_out = y.shape[-1]
    return _wide_combine_kernel(
        inputs=[y, inv, scores],
        template=[("InT", y.dtype), ("K", k)],
        grid=(d_out, t, 1), threadgroup=(64, 1, 1),
        output_shapes=[(t, d_out)], output_dtypes=[y.dtype])[0]


def wide_glu_mixed(glu, x, indices, scores):
    """The wide forward with the score mix in its last kernel: the bits of
    ``(wide_glu(glu, x, indices) * scores[..., None]).sum(-2)``. None when
    this call is not the wide form's (the caller then takes the member's
    own call and mixes). GMLX_MOE_WIDE_COMBINE=0 turns it off."""
    st = glu.__dict__.get("_kq_wide_store")
    k = indices.shape[-1]
    t = indices.size // k
    if (st is None or not _wide_combine_on() or not st.serves_wide(t)
            or getattr(glu, "_kq_arena_rows", False)
            or scores.dtype not in (mx.bfloat16, mx.float16)
            or x.dtype != scores.dtype
            or mx.default_device() != mx.gpu):
        return None
    fused_ok = getattr(glu, "_fused_ok", None)
    if fused_ok is not None and fused_ok(x, indices):
        return None
    y, order = _wide_sorted(glu, x, indices)
    if y.dtype != scores.dtype:
        return None
    out = wide_combine(y, order, scores.reshape(t, k))
    return out.reshape(*indices.shape[:-1], y.shape[-1])


def store_only(block) -> bool:
    """True once a full store has replaced the block's mapped tensors."""
    return bool(block.__dict__.get("_kq_store_only"))
