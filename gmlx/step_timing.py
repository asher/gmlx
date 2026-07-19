"""Env-gated per-step timing for the served batch decode loop.

``GMLX_STEP_LOG=<path>`` wraps mlx-vlm's ``GenerationBatch._step``,
``GenerationBatch.next`` and ``BatchGenerator._next`` with perf_counter
stamps and appends one CSV row per call:

    kind,idx,dur_ms,gap_ms,batch

- ``step`` rows: the double-buffered submit+eval itself; ``gap_ms`` is the
  time since the previous step row returned (everything the serving stack
  does between steps: response building, scheduler bookkeeping, asyncio
  hand-offs, SSE writes).
- ``gnext`` rows: GenerationBatch.next (adds per-token response building).
- ``bnext`` rows: BatchGenerator._next (adds scheduler + cache housekeeping).

Nesting means per-token cost decomposes as: step dur (GPU-bound), gnext -
step (response build), bnext - gnext (scheduler), step gap - (bnext - step)
(above the batch generator). No-op unless the env var is set.
"""

import atexit
import os
import sys
import time

_PATH = None
_ROWS = []
_LAST_STEP_EXIT = [0.0]
_IDX = {"step": 0, "gnext": 0, "bnext": 0}


def _flush():
    global _PATH
    if not (_PATH and _ROWS):
        return
    try:
        header = not os.path.exists(_PATH) or os.path.getsize(_PATH) == 0
        with open(_PATH, "a") as f:
            if header:
                f.write("kind,idx,dur_ms,gap_ms,batch\n")
            f.writelines(_ROWS)
    except OSError as e:
        # A bad GMLX_STEP_LOG path must not crash the decode loop.
        print(f"[step-timing] cannot write {_PATH}: {e} - disabled",
              file=sys.stderr)
        _PATH = None
    _ROWS.clear()


def _row(kind, dur_ms, gap_ms, batch):
    if _PATH is None:                 # writes disabled after a flush failure; stop
        return                        # accumulating rows (the wrappers stay installed)
    _ROWS.append(f"{kind},{_IDX[kind]},{dur_ms:.3f},{gap_ms:.3f},{batch}\n")
    _IDX[kind] += 1
    if len(_ROWS) >= 400:
        _flush()


def install_step_timing() -> None:
    """Wrap the three decode-loop layers; idempotent; no-op without the env."""
    global _PATH
    _PATH = os.environ.get("GMLX_STEP_LOG")
    if not _PATH:
        return
    from mlx_vlm.generate import ar

    if getattr(ar.GenerationBatch._step, "_kq_step_timing", False):
        return

    orig_step = ar.GenerationBatch._step
    orig_gnext = ar.GenerationBatch.next
    orig_bnext = ar.BatchGenerator._next

    def _step(self):
        t0 = time.perf_counter()
        gap = (t0 - _LAST_STEP_EXIT[0]) * 1e3 if _LAST_STEP_EXIT[0] else 0.0
        out = orig_step(self)
        t1 = time.perf_counter()
        _LAST_STEP_EXIT[0] = t1
        _row("step", (t1 - t0) * 1e3, gap, len(self.uids))
        return out

    def _gnext(self):
        t0 = time.perf_counter()
        out = orig_gnext(self)
        _row("gnext", (time.perf_counter() - t0) * 1e3, 0.0, len(self.uids))
        return out

    def _bnext(self, **kwargs):
        t0 = time.perf_counter()
        out = orig_bnext(self, **kwargs)
        _row("bnext", (time.perf_counter() - t0) * 1e3, 0.0, -1)
        return out

    _step._kq_step_timing = True
    ar.GenerationBatch._step = _step
    ar.GenerationBatch.next = _gnext
    ar.BatchGenerator._next = _bnext
    atexit.register(_flush)
