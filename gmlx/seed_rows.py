"""Per-request seed under batching.

Eval harnesses and agent frameworks send ``seed`` per request. The batch
engine builds one sampler when the generator spins up, so before this,
only the first request's seed took effect and it colored every row.
Now each request's seed rides with its row: the shared sampler keeps a
uid-to-seed registry, and every keyed draw derives that row's key from
its own seed. Rows without a seed keep the stock derivation byte for
byte, so seeded and unseeded rows coexist in one batch.

Honest semantics (also in docs/server-config.md): seed guarantees a
deterministic sampling stream for that request. It does not guarantee
bitwise-identical output across runs with different batch composition,
because batched matmul reduction order shifts logits at float
tolerance. Same composition (for example solo replay) reproduces
exactly. With speculation, drafts are greedy and the target draws for a
seeded B=1 request come from the same per-request key stream, so a
same-setting replay matches; replays across different speculation
settings do not.

Wiring: the request's seed is stashed when the engine builds the
request's per-row hooks (the last per-request step before insert on the
single GPU thread) and bound to the uid insert returns. The decode step
and the speculative round both publish their row uids on the sampler
around each draw.
"""

from __future__ import annotations

import logging

_log = logging.getLogger(__name__)

_INSTALLED_FLAG = "_kq_gguf_seed_rows"
_MAX_SEEDS = 1024

# Single-slot handoff between the per-request argument hook and the
# insert that follows it on the GPU thread.
_PENDING: list = []


def register_row_seed(sampler, uid, seed) -> None:
    seeds = getattr(sampler, "_kq_row_seeds", None)
    if seeds is None:
        return
    seeds[uid] = int(seed)
    while len(seeds) > _MAX_SEEDS:
        seeds.pop(next(iter(seeds)))


def install_per_request_seed() -> None:
    """Bind each request's seed to its batch row. Idempotent."""
    from mlx_vlm.generate import ar as _ar
    from mlx_vlm.server.generation import ResponseGenerator

    if getattr(_ar.BatchGenerator.insert, _INSTALLED_FLAG, False):
        return

    _orig_criteria = ResponseGenerator._make_thinking_budget_criteria

    def _criteria_with_seed(self, args, input_ids):
        _PENDING.clear()
        seed = getattr(args, "seed", None)
        if seed is not None and getattr(args, "temperature", 1.0) != 0:
            _PENDING.append(int(seed))
        return _orig_criteria(self, args, input_ids)

    _orig_insert = _ar.BatchGenerator.insert

    def _insert_with_seed(self, *args, **kwargs):
        seed = _PENDING.pop() if _PENDING else None
        uids = _orig_insert(self, *args, **kwargs)
        if seed is not None and getattr(self, "sampler", None) is not None:
            for uid in uids:
                register_row_seed(self.sampler, uid, seed)
        return uids

    _orig_step = _ar.GenerationBatch._step

    def _step_with_rows(self):
        sampler = self.sampler
        if getattr(sampler, "_kq_row_seeds", None):
            sampler._kq_rows = list(self.uids)
            try:
                return _orig_step(self)
            finally:
                sampler._kq_rows = None
        return _orig_step(self)

    _orig_generate = _ar.PromptProcessingBatch.generate

    def _generate_with_rows(self, sampler, *args, **kwargs):
        if getattr(sampler, "_kq_row_seeds", None):
            sampler._kq_rows = list(self.uids)
            try:
                return _orig_generate(self, sampler, *args, **kwargs)
            finally:
                sampler._kq_rows = None
        return _orig_generate(self, sampler, *args, **kwargs)

    _orig_next = _ar.SpeculativeGenerationBatch.next

    def _next_with_rows(self):
        sampler = getattr(self, "sampler", None)
        if getattr(sampler, "_kq_row_seeds", None):
            sampler._kq_rows = list(self._all_uids)
            try:
                return _orig_next(self)
            finally:
                sampler._kq_rows = None
        return _orig_next(self)

    _criteria_with_seed.__dict__[_INSTALLED_FLAG] = True
    _insert_with_seed.__dict__[_INSTALLED_FLAG] = True
    _step_with_rows.__dict__[_INSTALLED_FLAG] = True
    _generate_with_rows.__dict__[_INSTALLED_FLAG] = True
    _next_with_rows.__dict__[_INSTALLED_FLAG] = True
    ResponseGenerator._make_thinking_budget_criteria = _criteria_with_seed
    _ar.BatchGenerator.insert = _insert_with_seed
    _ar.GenerationBatch._step = _step_with_rows
    _ar.PromptProcessingBatch.generate = _generate_with_rows
    _ar.SpeculativeGenerationBatch.next = _next_with_rows
    _log.info("per-request seed installed")
