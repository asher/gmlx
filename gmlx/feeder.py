"""Feeder prefill: staged expert streaming for over-RAM MoE models.

The page-cache prefill path (prefetch.py) reads every expert byte twice on a
box that is at memory capacity by definition: once populating the cache and
once faulting it back into the compute, with the kernel reclaiming pages as
fast as they arrive. This module streams each layer's expert stacks from the
GGUF *directly* into a two-slot ring of GPU-visible staging buffers
(``mlx_kquant.arena_alloc``) and runs the expert GEMM on the GPU stream from
the slot, so the bytes make one trip and the page cache is never involved.

Synchronization is host-side only. Streaming prefill already materializes
the graph layer by layer (``mx.eval`` on each expert call's input), and that
eval is a completion proof: when the wrapper reaches layer L, layer L-1's
expert kernels have finished, so L-1's slot is free to overwrite. The
protocol per call is then simply: kick staging of the next uncovered layer
into the freed slot, block until this layer's slot is staged, swap the
module's expert weights to the slot views for the duration of the call. The
GPU-encoded event pair (kq.event_signal/event_wait) is not needed here; it
becomes necessary only for a decode feeder, which has no per-layer eval.

Enable with ``GMLX_FEEDER_PREFILL=1``. Requires an mlx-kquant with
``arena_alloc`` and layers whose gate/up/down expert stacks are zero-copy
views of the GGUF (verified byte-for-byte at init; any loader transform
disables the feeder rather than corrupting compute).
"""

from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

from .feeder_common import ATTRS, KINDS, read_range, swapped_weights, verify_zero_copy

# Per-range read granularity: large enough for sequential SSD clustering,
# small enough that one layer's three stacks spread across the pool.
_READ_CHUNK = 32 << 20
_READ_WORKERS = 12

# A staging wait longer than this means the SSD or the staging thread died;
# raising beats silently demand-faulting garbage.
_STAGE_TIMEOUT_S = 300.0


class PrefillFeeder:
    """Two-slot staged expert streaming; covered layers alternate slots."""

    def __init__(
        self,
        offsets: dict[int, list[tuple[str, int, int, int, str]]],
        modules: dict[int, list],
    ):
        import mlx_kquant as kq

        self._layers: dict[int, dict] = {}  # li -> {kind: (module, path, off, nbytes)}
        max_bytes: dict[str, int] = {}
        for li, ranges in offsets.items():
            kinds = {r[4] for r in ranges}
            if kinds != set(KINDS) or len(ranges) != len(KINDS):
                continue
            entry = {}
            for path, off, nbytes, _, kind in ranges:
                mod = None
                for m in modules.get(li, ()):
                    proj = getattr(m, ATTRS[kind], None)
                    w = getattr(proj, "weight", None)
                    if w is not None and w.nbytes == nbytes and w.ndim == 3:
                        mod = m
                        break
                if mod is None:
                    break
                entry[kind] = (mod, path, off, nbytes)
            if len(entry) == len(KINDS):
                self._layers[li] = entry
                for kind, (_, _, _, nbytes) in entry.items():
                    max_bytes[kind] = max(max_bytes.get(kind, 0), nbytes)
        if not self._layers:
            raise RuntimeError("no layers with matching gate/up/down stacks")

        self._fds: dict[str, int] = {}
        try:
            for entry in self._layers.values():
                for _, path, _, _ in entry.values():
                    if path not in self._fds:
                        self._fds[path] = os.open(path, os.O_RDONLY)
            self._verify_zero_copy()
        except BaseException:
            self.close()
            raise

        # Two flat slots per kind, sized for the largest layer; a layer uses
        # its assigned slot through a zero-copy slice+reshape view
        # of its own geometry (mixed-codec quants - e.g. Q5_K_M's q6_k down
        # stacks on some layers - make per-kind shapes non-uniform).
        self._slots = [
            {k: kq.arena_alloc([n]) for k, n in max_bytes.items()} for _ in (0, 1)
        ]
        self.slot_bytes = sum(a.nbytes for a, _ in self._slots[0].values())
        self._views: dict[tuple[int, int], dict] = {}  # (li, parity) -> kind -> view

        # Ring slot by position in the ordered covered set, not absolute
        # layer parity: coverage gaps (e.g. interval-2 MoE layers) would
        # otherwise map consecutive covered layers to the same slot.
        self._slot_of = {li: i % 2 for i, li in enumerate(sorted(self._layers))}

        self._stage_pool = ThreadPoolExecutor(max_workers=1)
        self._read_pool = ThreadPoolExecutor(max_workers=_READ_WORKERS)
        self._ready: dict[int, threading.Event] = {}
        self._last_li: int | None = None
        self._error: BaseException | None = None

    def _verify_zero_copy(self) -> None:
        li = min(self._layers)
        verify_zero_copy(
            li, ((k, mod, path, off)
                 for k, (mod, path, off, _) in self._layers[li].items()),
            self._fds)

    def covers(self, li: int) -> bool:
        return li in self._layers

    # staging

    def _stage(self, li: int) -> None:
        try:
            slot = self._slots[self._slot_of[li]]
            futs = []
            for kind, (_, path, off, nbytes) in self._layers[li].items():
                fd = self._fds[path]
                mv = slot[kind][1]
                for start in range(0, nbytes, _READ_CHUNK):
                    end = min(start + _READ_CHUNK, nbytes)
                    futs.append(self._read_pool.submit(
                        read_range, fd, mv[start:end], off + start))
            for f in futs:
                f.result()
        except BaseException as e:  # surfaced on the caller's next wait
            self._error = e
        finally:
            self._ready[li].set()

    def _kick(self, li: int) -> None:
        if li in self._layers and li not in self._ready:
            self._ready[li] = threading.Event()
            self._stage_pool.submit(self._stage, li)

    # the per-call protocol

    def _drain_on_new_pass(self, li: int) -> None:
        if self._last_li is None or li <= self._last_li:
            # New prefill pass (next chunk or new request). In-flight staging
            # from the old pass targets the same slots; drain before reusing.
            for ev in self._ready.values():
                ev.wait(_STAGE_TIMEOUT_S)
            self._ready.clear()
            self._error = None
        self._last_li = li

    @contextmanager
    def _swapped(self, li: int):
        """Swap the layer's expert weights to its slot views for the
        duration of the call."""
        entry = self._layers[li]
        views = self._views.get((li, self._slot_of[li]))
        if views is None:
            slot = self._slots[self._slot_of[li]]
            views = {}
            for kind, (mod, _, _, nbytes) in entry.items():
                shape = getattr(mod, ATTRS[kind]).weight.shape
                views[kind] = slot[kind][0][:nbytes].reshape(shape)
            self._views[(li, self._slot_of[li])] = views
        with swapped_weights(entry, views):
            yield

    @contextmanager
    def prefill_call(self, module, li: int):
        """Caller contract: ``mx.eval`` of this call's input has run (so the
        previous covered layer's compute is finished and its slot is free),
        and the expert call happens inside the ``with`` body."""
        self._drain_on_new_pass(li)
        self._kick(li)
        nxt = min((t for t in self._layers if t > li), default=None)
        if nxt is not None:
            self._kick(nxt)
        if not self._ready[li].wait(_STAGE_TIMEOUT_S):
            raise RuntimeError(f"[feeder] staging layer {li} timed out")
        if self._error is not None:
            raise RuntimeError(f"[feeder] staging failed: {self._error}")
        with self._swapped(li):
            yield

    @contextmanager
    def prefill_partial_call(self, module, li: int, ids):
        """Router-aware partial staging for short-chunk passes: stage only
        the routed experts' slices into the parity slot, at their original
        expert indices (the gather reads nothing else from the slot, so the
        rest may hold a previous pass's bytes), and synchronously - a short
        chunk's per-layer compute is far too small to hide whole-layer
        staging behind, so the pipelined ring buys nothing and over-reads.

        Caller contract: ``mx.eval`` of the call's ``indices`` has run (which
        both proves the previous layer's slot free and materialized ``ids``).
        """
        self._drain_on_new_pass(li)
        entry = self._layers[li]
        slot = self._slots[self._slot_of[li]]
        futs = []
        for kind, (mod, path, off, nbytes) in entry.items():
            n_exp = getattr(mod, ATTRS[kind]).weight.shape[0]
            stride = nbytes // n_exp
            mv = slot[kind][1]
            fd = self._fds[path]
            for e in ids:
                if 0 <= e < n_exp:
                    futs.append(self._read_pool.submit(
                        read_range, fd,
                        mv[e * stride:(e + 1) * stride], off + e * stride))
        for f in futs:
            f.result()
        with self._swapped(li):
            yield

    def close(self) -> None:
        pool = getattr(self, "_stage_pool", None)
        if pool is not None:
            pool.shutdown(wait=True)
        pool = getattr(self, "_read_pool", None)
        if pool is not None:
            pool.shutdown(wait=True)
        fds, self._fds = self._fds, {}
        for fd in fds.values():
            try:
                os.close(fd)
            except OSError:
                pass

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass  # GC-time cleanup must never raise


def maybe_make_prefill_feeder(offsets, modules) -> PrefillFeeder | None:
    """A PrefillFeeder over the coverable layers, or None with a printed
    reason (opt-in feature: silence would read as 'enabled')."""
    try:
        import mlx.core as mx
        import mlx_kquant as kq

        if not hasattr(kq, "arena_alloc"):
            raise RuntimeError(
                "mlx-kquant lacks arena_alloc (needs the feeder-loop build)")
        if not mx.metal.is_available():
            raise RuntimeError("Metal unavailable")
        return PrefillFeeder(offsets, modules)
    except Exception as e:
        print(f"[stream] feeder prefill unavailable ({e}); "
              "falling back to page-cache prefetch")
        return None
