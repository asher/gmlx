"""Sequential expert prefetch for streaming (over-wired-budget) models.

A model beyond the GPU wired budget streams its routed-expert weights from
disk through the page cache (see ``install_expert_streaming``). At prefill, a
chunk of more than a few tokens routes through essentially every expert, so
each layer needs ~its whole expert stack - a perfectly predictable read.
Demand faulting serves those reads cluster by cluster at random-read
bandwidth; populating the page cache ahead of execution streams them at
sequential bandwidth instead. Two population modes, paced a fixed window of
layers ahead of execution either way: explicit chunked preads on a worker
pool (the default - reads cannot be dropped), or ``F_RDADVISE`` advisories
(``GMLX_STREAM_PREFETCH_MODE=advise`` - zero-copy, but the kernel is free to
defer or drop advice under exactly the memory pressure a streaming model
creates).

Two constraints shape the design:

- **Advice must be paced at execution time.** The whole expert set is larger
  than RAM, so firing every layer's advisory up front (or at graph-build
  time, which a lazy framework reaches long before execution) would evict
  its own earlier reads. The offload wrapper therefore materializes the
  graph layer by layer during prefill (``mx.eval`` on the expert call's
  input) and advances the advisory window a fixed ``depth`` ahead.
- **Population advice works through any file descriptor.** ``F_RDADVISE``
  fills the shared page cache, so this module opens its own descriptors and
  never touches the loader's mmap.

Disable with ``GMLX_STREAM_PREFETCH=0``.
"""

from __future__ import annotations

import fcntl
import os
import re
import struct
import threading
from concurrent.futures import ThreadPoolExecutor

# Routed-expert stacks only: everything else (incl. shared experts,
# *_shexp) is wired and never streams.
_EXPS_RE = re.compile(r"blk\.(\d+)\.ffn_(gate|up|down|gate_up)_exps\.weight")

F_RDADVISE = 44  # macOS fcntl: asynchronous advisory read into the page cache


def expert_offset_map(
    gguf_path: str,
) -> dict[int, list[tuple[str, int, int, int, str]]]:
    """``layer -> [(shard_path, file_offset, n_bytes, n_experts, kind), ...]``
    for every routed-expert tensor, from the GGUF headers of all shards.
    ``n_experts`` is the stack's slowest dim (ggml ne order puts it last), so
    expert ``e``'s wire bytes are the contiguous ``n_bytes // n_experts``
    slice at index ``e``; 0 when the tensor is not a recognizable 3-dim
    stack (callers must then skip per-expert slicing). ``kind`` is the
    projection name from the tensor: "gate", "up", "down" or "gate_up"."""
    from .headerscan import scan_gguf
    from .preflight import find_split_shards

    out: dict[int, list[tuple[str, int, int, int, str]]] = {}
    for path in find_split_shards(gguf_path):
        scan = scan_gguf(path, array_limit=0)
        for t in scan.tensors:
            m = _EXPS_RE.fullmatch(t.name)
            if m:
                n_exp = int(t.shape[-1]) if len(t.shape) == 3 else 0
                if n_exp and t.nbytes % n_exp:
                    n_exp = 0
                out.setdefault(int(m.group(1)), []).append(
                    (path, scan.data_offset + t.offset, t.nbytes, n_exp,
                     m.group(2))
                )
    return out


class ExpertPrefetcher:
    """Paced advisory window over per-layer expert byte ranges.

    ``on_layer(li)`` (called by the offload wrapper as layer ``li`` starts
    executing) advances the window to ``li + depth``: each not-yet-advised
    layer in ``[li, li + depth]`` gets one ``F_RDADVISE`` per expert tensor,
    issued from a single worker thread (the call itself costs tens of ms per
    GB-scale range). A backward jump in ``li`` marks a new prefill pass
    (next chunk, or a new request) and resets the window; re-advising
    already-resident ranges is cheap.
    """

    # ``radvisory.ra_count`` is a signed int32: a >=2 GB expert stack must be
    # advised in chunks or struct.pack raises and the range silently degrades
    # to demand faulting - exactly on the largest tensors this exists for.
    _ADVISE_CHUNK = 1 << 30

    # Decode pull parallelism: per layer, top-k experts x 3 stacks of
    # MB-scale slices; enough in-flight preads to keep the SSD at sequential
    # bandwidth where the gemv's serial demand faulting cannot.
    _DECODE_WORKERS = 12

    # Speculative reads run on their own smaller pool so a burst of
    # speculation never queues ahead of the current layer's blocking pull
    # (ThreadPoolExecutor has no priorities; separate pools are the priority).
    _SPEC_WORKERS = 8

    def __init__(
        self,
        offsets: dict[int, list[tuple[str, int, int, int, str]]],
        depth: int = 2,
    ):
        self.offsets = offsets
        self.depth = depth
        self.enabled = True
        self._advised: set[int] = set()
        self._last_li: int | None = None
        self._fds: dict[str, int] = {}
        try:
            for ranges in offsets.values():
                for path, *_ in ranges:
                    if path not in self._fds:
                        self._fds[path] = os.open(path, os.O_RDONLY)
        except OSError:
            for fd in self._fds.values():   # no half-open leak on EMFILE/ENOENT
                os.close(fd)
            raise
        self._pool = ThreadPoolExecutor(max_workers=1)
        self._read_pool: ThreadPoolExecutor | None = None
        self._tls = threading.local()
        # Decode speculation state: per layer, the expert ids the router chose
        # on the most recent token (the prediction for the next one), and the
        # in-flight speculative reads keyed by expert id.
        self._spec_pool: ThreadPoolExecutor | None = None
        self._spec: dict[int, dict[int, list]] = {}
        self._last_ids: dict[int, list[int]] = {}
        # Off by default: prediction accuracy on large fine-grained expert
        # pools is low enough that the wasted reads steal SSD bandwidth from
        # the blocking pulls and evict useful page-cache entries (the box is
        # at memory capacity by definition here). GMLX_DECODE_SPEC_STATS=1
        # prints the measured reuse rate; enable speculation only when it is
        # comfortably above half.
        self._spec_depth = max(
            0, int(os.environ.get("GMLX_DECODE_SPEC_DEPTH", "0") or 0)
        )
        self._decode_layers = sorted(offsets)
        self._layer_pos = {li: i for i, li in enumerate(self._decode_layers)}

    def _advise(self, li: int) -> None:
        for path, off, nbytes, *_ in self.offsets.get(li, ()):
            # struct radvisory { off_t ra_offset; int ra_count; } (+4 pad)
            while nbytes > 0:
                take = min(nbytes, self._ADVISE_CHUNK)
                try:
                    fcntl.fcntl(
                        self._fds[path], F_RDADVISE,
                        struct.pack("=qi4x", off, take)
                    )
                except OSError:
                    break  # advisory only
                off += take
                nbytes -= take

    # Explicit-read prefill chunk size: large enough for sequential SSD
    # clustering, small enough that the pool interleaves layers.
    _PULL_CHUNK = 16 << 20

    def _pull(self, li: int) -> None:
        """Queue explicit chunked reads of layer ``li``'s whole expert
        ranges on the read pool (non-blocking population, prefill window).
        ``F_RDADVISE`` is advisory - the kernel may defer or drop it under
        pressure; a pread cannot be dropped, at the cost of burning pool
        threads and a page-cache copy."""
        if self._read_pool is None:
            self._read_pool = ThreadPoolExecutor(
                max_workers=self._DECODE_WORKERS)
        for path, off, nbytes, *_ in self.offsets.get(li, ()):
            while nbytes > 0:
                take = min(nbytes, self._PULL_CHUNK)
                self._read_pool.submit(self._read_slice, path, off, take)
                off += take
                nbytes -= take

    def on_layer(self, li: int) -> None:
        if not self.enabled:
            return
        if self._last_li is None or li < self._last_li:
            self._advised.clear()  # new prefill pass
        self._last_li = li
        pull = os.environ.get("GMLX_STREAM_PREFETCH_MODE", "pread") != "advise"
        for t in range(li, li + self.depth + 1):
            if t in self.offsets and t not in self._advised:
                self._advised.add(t)
                if pull:
                    self._pull(t)
                else:
                    self._pool.submit(self._advise, t)

    def _read_slice(self, path: str, off: int, nbytes: int) -> None:
        """Populate the page cache for one byte range with a blocking read
        into a per-thread scratch buffer (``preadv``: no per-call allocation).
        The data itself is discarded - the following compute reads the same
        pages through the loader's mmap."""
        buf = getattr(self._tls, "buf", None)
        if buf is None or len(buf) < nbytes:
            buf = bytearray(nbytes)
            self._tls.buf = buf
        fd = self._fds[path]
        view = memoryview(buf)
        done = 0
        while done < nbytes:
            try:
                n = os.preadv(fd, [view[done:nbytes]], off + done)
            except OSError:
                return  # populate-only: compute demand-faults the remainder
            if n <= 0:
                return
            done += n

    def _submit_reads(
        self, li: int, ids, pool: ThreadPoolExecutor
    ) -> dict[int, list]:
        """Queue slice reads of experts ``ids`` in layer ``li`` on ``pool``;
        returns ``id -> futures`` so callers can block per expert. Ranges
        whose expert geometry was unreadable (``n_experts == 0``) are skipped
        rather than pulled whole."""
        futs: dict[int, list] = {}
        for path, off, nbytes, n_exp, _ in self.offsets.get(li, ()):
            if not n_exp:
                continue
            stride = nbytes // n_exp
            for e in ids:
                if 0 <= e < n_exp:
                    futs.setdefault(e, []).append(pool.submit(
                        self._read_slice, path, off + e * stride, stride))
        return futs

    def on_decode(self, li: int, expert_ids) -> None:
        """Blocking qd-parallel pull of layer ``li``'s selected experts into
        the page cache, called by the offload wrapper between the router and
        the expert compute of a decode-sized call. Decode's alternative is
        demand faulting from inside the gemv - serial 16 KB clusters at
        random-read bandwidth; explicit parallel slice reads run at SSD
        sequential bandwidth, and already-resident slices cost only a
        page-cache copy.

        When ``GMLX_DECODE_SPEC_DEPTH`` is set above 0, also speculatively
        queues the next layers' reads using each layer's previous-token
        routing as the prediction, overlapping them with the expert math that
        executes between this call and the next one. Whether that pays
        depends entirely on the model's token-to-token routing reuse (see
        ``_spec_depth``); mispredicted reads are pure loss."""
        ranges = self.offsets.get(li)
        if not ranges:
            return
        if self._read_pool is None:
            self._read_pool = ThreadPoolExecutor(
                max_workers=self._DECODE_WORKERS)
        spec = self._spec.pop(li, {})
        fresh = self._submit_reads(
            li, [e for e in expert_ids if e not in spec], self._read_pool)
        for e in expert_ids:
            for f in spec.get(e, ()):
                f.result()
        # Mispredicted speculative reads are left to finish on their own:
        # waiting on them buys nothing and their bytes may serve a later token.
        for fl in fresh.values():
            for f in fl:
                f.result()
        prev = self._last_ids.get(li)
        if prev is not None and os.environ.get("GMLX_DECODE_SPEC_STATS"):
            self._spec_hits = getattr(self, "_spec_hits", 0) + len(
                set(prev) & set(expert_ids))
            self._spec_total = getattr(self, "_spec_total", 0) + len(
                set(expert_ids))
        self._last_ids[li] = list(expert_ids)
        self._speculate(li)

    def _speculate(self, li: int) -> None:
        pos = self._layer_pos.get(li)
        if pos is None or self._spec_depth <= 0 or not self._last_ids:
            return
        if self._spec_pool is None:
            self._spec_pool = ThreadPoolExecutor(
                max_workers=self._SPEC_WORKERS)
        n = len(self._decode_layers)
        for k in range(1, min(self._spec_depth, n - 1) + 1):
            # Wrapping past the last layer predicts the next token's first
            # layers - the head start there also spans lm_head + sampling.
            nxt = self._decode_layers[(pos + k) % n]
            if nxt in self._spec:
                continue
            pred = self._last_ids.get(nxt)
            if pred:
                self._spec[nxt] = self._submit_reads(
                    nxt, pred, self._spec_pool)

    def close(self) -> None:
        """Idempotent: teardown and GC finalization may both land here."""
        total = getattr(self, "_spec_total", 0)
        if total:
            print(
                f"[stream] decode expert reuse (prev-token prediction): "
                f"{getattr(self, '_spec_hits', 0)}/{total} "
                f"({100 * getattr(self, '_spec_hits', 0) / total:.1f}%)"
            )
            self._spec_total = 0
        self._pool.shutdown(wait=False)
        if self._read_pool is not None:
            self._read_pool.shutdown(wait=False)
            self._read_pool = None
        if self._spec_pool is not None:
            self._spec_pool.shutdown(wait=False)
            self._spec_pool = None
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
            pass


def maybe_make_prefetcher(
    gguf_path: str | None, depth: int = 2
) -> ExpertPrefetcher | None:
    """Build a prefetcher for a streaming model, or None (no path given,
    disabled via ``GMLX_STREAM_PREFETCH=0``, no expert tensors, or no gguf-py)."""
    if gguf_path is None or os.environ.get("GMLX_STREAM_PREFETCH") == "0":
        return None
    try:
        offsets = expert_offset_map(gguf_path)
    except Exception as e:  # gguf-py absent or header unreadable
        print(f"[stream] expert prefetch unavailable ({e}); prefill will demand-fault")
        return None
    if not offsets:
        return None
    try:
        return ExpertPrefetcher(offsets, depth=depth)
    except OSError as e:   # e.g. EMFILE - degrade, don't abort the model load
        print(f"[stream] expert prefetch unavailable ({e}); prefill will demand-fault")
        return None
