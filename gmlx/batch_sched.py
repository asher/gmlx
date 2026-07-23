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
server can be re-paced for an A/B). 0 disables (stock 1:1). Very large
ratios starve admission at depth: each pending chunk waits ratio x
chunk_time of decode -- TTFT of queued requests stretches accordingly.
"""

from __future__ import annotations

import logging
import os
import time

_log = logging.getLogger(__name__)

_INSTALLED_FLAG = "_kq_gguf_decode_priority_sched"


_ratio_memo = ("", 1.0)


def _ratio() -> float:
    """Parse GMLX_DECODE_PREFILL_RATIO, memoized on the raw string; warn
    once per bad value rather than silently defaulting."""
    global _ratio_memo
    raw = os.environ.get("GMLX_DECODE_PREFILL_RATIO", "1.0")
    if raw == _ratio_memo[0]:
        return _ratio_memo[1]
    try:
        val = float(raw)
    except ValueError:
        _log.warning(
            "GMLX_DECODE_PREFILL_RATIO=%r is not a number; using 1.0", raw)
        val = 1.0
    _ratio_memo = (raw, val)
    return val


def install_decode_priority_sched() -> None:
    """Pace prefill chunks against live decode GPU time. Idempotent.

    Installs unconditionally; the wrapper re-reads the ratio per tick and
    passes straight through at ratio <= 0 (stock scheduling)."""
    from mlx_vlm.generate import ar as _ar

    if getattr(_ar.BatchGenerator._next, _INSTALLED_FLAG, False):
        return
    _orig_next = _ar.BatchGenerator._next

    def _paced_next(self, **kwargs):
        # Pace only when decode AND prefill work are both live; otherwise
        # stock behavior (prefill at full speed, untouched TTFT).
        ratio = _ratio()
        prefill_live = self._prompt_batch is not None or bool(
            self._unprocessed_sequences
        )
        decode_live = len(self._generation_batch) > 0
        if ratio <= 0 or not (prefill_live and decode_live):
            return _orig_next(self, **kwargs)

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
        # observe the chunk cost via the prompt-time counter delta.
        before = self._prompt_time_counter
        out = _orig_next(self, **kwargs)
        spent = self._prompt_time_counter - before
        if spent > 0.0:
            self._kq_last_chunk_time = spent
            self._kq_decode_since_chunk = 0.0
        return out

    setattr(_paced_next, _INSTALLED_FLAG, True)
    _ar.BatchGenerator._next = _paced_next
    _log.info(
        "decode-priority prefill pacing installed (ratio=%.2f)", _ratio()
    )
