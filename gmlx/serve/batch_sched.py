"""Decode-priority prefill pacing for the batched serve path.

Stock ``BatchGenerator._next`` runs one decode step then, unconditionally,
one prefill chunk (DEFAULT_PREFILL_STEP_SIZE=2048 tokens) per tick. At depth
a chunk costs orders of magnitude more GPU time than a decode step, so a
deep prompt head-of-line-blocks the live decode batch: streams decode ~1
token per chunk across a 25-48s prefill (measured d50k: 80-84% of decode
wall is stall; d14k: ~56%).

This seam paces the prefill arm by GPU-time share instead: a prefill chunk
is admitted only once decode has accumulated ``ratio x last_chunk_time``
since the previous chunk. ratio=1.0 => ~50/50 GPU split while both are
live; higher favors decode. Prefill runs at full speed whenever no decode
batch is active, so single-stream TTFT is untouched.

The wrapper also feeds prefill_decay's concurrency term (prefill ticks):
live decode width is noted every tick, and each admitted chunk's observed
wall cost is handed over so the next chunk can be sized to the
GMLX_PREFILL_TICK_MS stall budget. Pacing bounds the average GPU share;
the tick term bounds the per-chunk stall quantum. The two compose: pacing
self-tunes off the same observed chunk cost whatever chunk policy is in
force.

Installation is the same late-bound monkeypatch pattern as `spec_engine`:
wrap ``BatchGenerator._next``; on a paced tick, stash ``_prompt_batch`` +
``_unprocessed_sequences`` so the stock body sees no prefill work (decode
step still runs first), then restore. Stock scheduling order, admission,
APC arms are untouched on non-paced ticks. Chunk cost is observed via the
``_prompt_time_counter`` delta, so pacing self-tunes with depth, model,
and chunk-size policy (prefill_decay etc.) with no per-model constants.

Ratio resolution: `gmlx serve --decode-prefill-ratio` / config
`server.decode_prefill_ratio` both export GMLX_DECODE_PREFILL_RATIO; the
wrapper reads the env per tick (install order never matters, and a live
server can be re-paced for an A/B). The default is ``auto``: the
per-tick ratio comes from ``auto_ratio``, which selects the paced ratio
or 0 from measured signals. A numeric value pins a static split
(0 disables: stock 1:1); every other mechanism below (owed-decode
arithmetic, stash-and-restore, pressure notes) is identical under both
spellings. Very large static ratios starve admission at depth: each
pending chunk waits ratio x chunk_time of decode -- TTFT of queued
requests stretches accordingly.
"""

from __future__ import annotations

import logging
import os
import time

from .batch_rows import batch_rows

_log = logging.getLogger(__name__)

_INSTALLED_FLAG = "_kq_gguf_decode_priority_sched"


_ratio_memo: tuple[str, tuple[str, float | None]] = ("", ("static", 1.0))


def _ratio() -> tuple[str, float | None]:
    """Parse GMLX_DECODE_PREFILL_RATIO into ``(mode, value)``:
    ``("static", r)`` or ``("auto", None)``. Memoized on the raw string
    (in auto mode only the parse is memoized, never a resolved ratio);
    warns once per bad value rather than silently defaulting."""
    global _ratio_memo
    raw = os.environ.get("GMLX_DECODE_PREFILL_RATIO", "auto")
    if raw == _ratio_memo[0]:
        return _ratio_memo[1]
    if raw.strip().lower() == "auto":
        val: tuple[str, float | None] = ("auto", None)
    else:
        try:
            val = ("static", float(raw))
        except ValueError:
            _log.warning(
                "GMLX_DECODE_PREFILL_RATIO=%r is not a number or 'auto'; "
                "using auto", raw)
            val = ("auto", None)
    _ratio_memo = (raw, val)
    return val


def install_decode_priority_sched() -> None:
    """Pace prefill chunks against live decode GPU time. Idempotent.

    Installs unconditionally; the wrapper re-reads the ratio per tick and
    passes straight through at ratio <= 0 (stock scheduling)."""
    from mlx_vlm.generate import ar as _ar

    from . import prefill_decay as _pd

    if getattr(_ar.BatchGenerator._next, _INSTALLED_FLAG, False):
        return
    _orig_next = _ar.BatchGenerator._next

    def _observed_next(self, **kwargs):
        # Run the stock body and observe any prefill chunk it admitted via
        # the prompt-time counter delta. Feeds both consumers of the
        # observation: the pacer's owed-decode arithmetic and
        # prefill_decay's tick term (which sizes the NEXT chunk to the
        # stall budget).
        before = self._prompt_time_counter
        out = _orig_next(self, **kwargs)
        spent = self._prompt_time_counter - before
        if spent > 0.0:
            self._kq_last_chunk_time = spent
            self._kq_decode_since_chunk = 0.0
            _pd.note_chunk_cost(spent)
        return out

    def _tick(self, ratio, **kwargs):
        # One scheduler tick at a resolved ratio. Pace only when decode
        # AND prefill work are both live; otherwise stock behavior
        # (prefill at full speed, untouched TTFT).
        if ratio <= 0:
            # Ratio 0 is documented as stock: clear the tick term's
            # decode-pressure reading (0 clears, per the hook contract)
            # so chunk sizing goes stock too instead of riding a stale
            # pressure value.
            _pd.note_decode_pressure(0)
            return _observed_next(self, **kwargs)
        decode_rows = batch_rows(self)
        _pd.note_decode_pressure(decode_rows)
        prefill_live = self._prompt_batch is not None or bool(
            self._unprocessed_sequences
        )
        decode_live = decode_rows > 0
        if not (prefill_live and decode_live):
            return _observed_next(self, **kwargs)

        last_chunk = getattr(self, "_kq_last_chunk_time", 0.0)
        decode_owed = ratio * last_chunk
        decode_spent = getattr(self, "_kq_decode_since_chunk", 0.0)
        if decode_spent < decode_owed:
            # Paced tick: hide the prefill arms so the stock body runs the
            # decode step and falls through to a bare return. The admission
            # arm must be hidden too or it would start a SECOND prompt batch.
            stash_prompt = self._prompt_batch
            stash_pending = self._unprocessed_sequences
            self._prompt_batch = None
            self._unprocessed_sequences = []
            tic = time.perf_counter()
            try:
                out = _orig_next(self, **kwargs)
            finally:
                # insert() may have appended to (or rebound) the temp list
                # from a handler thread mid-call; merge, never clobber.
                arrived = self._unprocessed_sequences
                self._prompt_batch = stash_prompt
                self._unprocessed_sequences = stash_pending
                if arrived:
                    stash_pending.extend(arrived)
            self._kq_decode_since_chunk = decode_spent + (
                time.perf_counter() - tic
            )
            return out

        # Chunk-eligible tick: let the stock body run its prefill arm and
        # observe the chunk cost.
        return _observed_next(self, **kwargs)

    def _paced_next(self, **kwargs):
        mode, value = _ratio()
        if mode != "auto":
            return _tick(self, value, **kwargs)
        from . import auto_ratio

        ratio = auto_ratio.resolve(self)
        rows = batch_rows(self)
        before = self._prompt_time_counter
        tic = time.perf_counter()
        try:
            return _tick(self, ratio, **kwargs)
        finally:
            # The bracket rides syncs the tick already performs (chunk
            # mx.eval / decode blocking on the previous step); auto's
            # observe skips the poisoned first bracket after a chunk.
            auto_ratio.observe(
                self, time.perf_counter() - tic,
                self._prompt_time_counter - before, rows)

    setattr(_paced_next, _INSTALLED_FLAG, True)
    _ar.BatchGenerator._next = _paced_next
    mode, value = _ratio()
    _log.info(
        "decode-priority prefill pacing installed (%s)",
        "auto" if mode == "auto" else f"ratio={value:.2f}")
