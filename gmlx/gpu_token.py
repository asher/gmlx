"""GPU-autonomous token state for streamed decode (gpu-dispatch Tier 2).

The offload wrapper's decode path today evaluates the graph once per MoE
layer so the host can map routed expert ids to arena slots and demand-read
misses (see decode_feeder.py). Under mlx_kquant.route_shed the remap and the
miss decision run on the GPU instead, so the whole token builds lazily and
flushes once at the logits. This module owns everything the host still does,
all of it at token boundaries:

- per-layer slot-table snapshots (expert id -> arena slot, int32, -1 =
  non-resident) handed to route_shed as graph inputs;
- the per-token record of each layer's routing and route_shed miss outputs
  (lazy arrays, kept alive so the token flush materializes them);
- the boundary step: popularity credit for last token's routing, publish
  of completed prestage reads, prestage submission for the misses, and
  ONLY THEN fresh table snapshots.

The boundary ordering is the safety fence. prestage() evicts a victim by
clearing its slot_of entry at submission time, so a table snapshot taken
after all submissions cannot map any expert to a slot a background read is
writing. Mid-token the feeder state is frozen (no stage() calls on this
path), so the snapshot stays valid for the whole graph. Misses beyond the
keep-mass budget cannot be demand-read mid-graph; they shed and are only
counted (the over-budget rate is a pre-registered health metric).

Enable with GMLX_GPU_AUTONOMOUS=1 (decode-sized calls on covered layers,
scores required). Prototype status: lookahead and the phase-stat per-layer
buckets do not apply on this path.
"""

from __future__ import annotations

import numpy as np

import mlx.core as mx

from .decode_feeder import _DECAY_EVERY
from .envflags import env_bool

_ENV = "GMLX_GPU_AUTONOMOUS"


def autonomous_enabled() -> bool:
    return env_bool(_ENV, False)


def route_shed_op():
    """The kernel op, or None when the installed mlx_kquant predates it."""
    try:
        import mlx_kquant as kq
    except ImportError:  # pragma: no cover - kquant is a hard dep in practice
        return None
    return getattr(kq, "route_shed", None)


def register_exit_stats(gt: GpuTokenState) -> None:
    """Print the autonomous ledger at exit. Weakref, like the feeder's own
    exit hook: a strong reference would pin an unloaded model's state."""
    import atexit
    import weakref

    wref = weakref.ref(gt)

    def _stats_at_exit():
        g = wref()
        if g is None:
            return
        # A partial final token never hit a boundary; fold it in.
        if g._records:
            g.boundary()
        line = g.close_stats()
        if line:
            print(line)

    atexit.register(_stats_at_exit)


class GpuTokenState:
    """Per-model autonomous-token bookkeeping around one decode feeder."""

    def __init__(self, feeder, keep_mass: float | None = None):
        self._dfr = feeder
        self._route_shed = route_shed_op()
        # Over-budget accounting threshold: the same P the python miss-shed
        # would have used. None = pure-lossless intent (any dropped mass is
        # over-budget).
        self._keep_mass = keep_mass
        self._tables: dict[int, mx.array] = {}
        # (li, indices, scores, miss_ids, miss_scores) lazy per-layer records
        # of the in-flight token, consumed at the next boundary.
        self._records: list[
            tuple[int, mx.array, mx.array, mx.array, mx.array]
        ] = []
        self._last_li = -1
        # Stats (reported at feeder close via close_stats()).
        self.tokens = 0
        self.layer_calls = 0
        self.miss_n = 0
        self.over_budget_layers = 0

    # ---- in-token (graph building) ----

    def on_layer_entry(self, li: int, keep_mass: float | None = None) -> None:
        """Token boundary detection: layer indices increase within a token,
        so a wrap (li <= last seen) means the previous token flushed.
        ``keep_mass`` mirrors the module's --moe-miss-shed P for the
        over-budget ledger (install_moe_miss_shed runs after load)."""
        if keep_mass is not None:
            self._keep_mass = keep_mass
        if self._records and li <= self._last_li:
            self.boundary()
        self._last_li = li

    def table(self, li: int) -> mx.array:
        tbl = self._tables.get(li)
        if tbl is None:
            tbl = self._snapshot(li)
            self._tables[li] = tbl
        return tbl

    def record(
        self,
        li: int,
        indices: mx.array,
        scores: mx.array,
        miss_ids: mx.array,
        miss_scores: mx.array,
        y: mx.array,
    ) -> None:
        self._records.append((li, indices, scores, miss_ids, miss_scores, y))
        self.layer_calls += 1

    # ---- token boundary (host) ----

    def boundary(self) -> None:
        """Consume the flushed token's records; refresh tables last."""
        dfr = self._dfr
        records, self._records = self._records, []
        self.tokens += 1
        # JOIN before any arena mutation. The decode loop pipelines: the
        # next token's graph is being built while the previous one may
        # still be executing, and np-reading the miss arrays only forces
        # the routers, not the gathers that read arena slots. Evaluating
        # every recorded layer output forces those gathers to completion,
        # so the prestage evictions below can never overwrite a slot a
        # live gather is reading.
        mx.eval(*[r[5] for r in records])
        budget = (
            (1.0 - self._keep_mass) if self._keep_mass is not None else 0.0
        )
        for li, indices, scores, miss_ids, miss_scores, _y in records:
            counts = dfr._counts.get(li)
            if counts is None:
                continue
            # The token flush materialized all of these (siblings of the
            # slots the gathers consumed); np.array is a copy, not a sync.
            routed = np.unique(np.array(indices).reshape(-1))
            mids = np.array(miss_ids).reshape(-1)
            missed = np.unique(mids[mids >= 0])
            # Popularity at stage()'s rate, misses included: shed experts
            # staying cold would blind the arena to routing drift. (The
            # python miss-shed deliberately starves them; here prestage
            # below re-warms them anyway, so credit keeps the two ledgers
            # comparable.)
            counts[routed] += 1.0
            dfr._calls += 1
            if dfr._calls % _DECAY_EVERY == 0:
                for c in dfr._counts.values():
                    c *= 0.5
            dfr._lookups += len(routed)
            dfr._hits += len(routed) - len(missed)
            dfr._layer_lookups[li] += len(routed)
            dfr._layer_hits[li] += len(routed) - len(missed)
            n_miss = len(missed)
            self.miss_n += n_miss
            if n_miss:
                msc = np.array(miss_scores).reshape(-1)
                dropped = float(msc[mids >= 0].sum())
                total = float(np.array(scores).sum())
                if dropped > budget * max(total, 1e-20):
                    self.over_budget_layers += 1
                dfr._shed_n += n_miss
                dfr._shed_mass += dropped / max(total, 1e-20)
                dfr._shed_tokens += 1
            # Publish reads submitted at the previous boundary, then submit
            # for this token's misses. One id per row: prestage keeps only
            # the top GMLX_DECODE_LOOKAHEAD_K ranks per row, and unlike
            # lookahead guesses these are certain-demand misses.
            dfr._flush_pending(li)
            if n_miss:
                dfr.prestage(li, missed.reshape(-1, 1))
        # All mutations done: snapshot every table the next graph will use.
        for li in list(self._tables):
            self._tables[li] = self._snapshot(li)

    def close_stats(self) -> str | None:
        if not self.tokens:
            return None
        return (
            f"[stream] gpu-autonomous: {self.tokens} tokens, "
            f"{self.layer_calls} layer calls, {self.miss_n} misses shed, "
            f"{self.over_budget_layers} over-budget layer calls "
            f"({100.0 * self.over_budget_layers / max(self.layer_calls, 1):.2f}%)"
        )

    # ---- internals ----

    def _snapshot(self, li: int) -> mx.array:
        return mx.array(self._dfr._slot_of[li].astype(np.int32, copy=True))
