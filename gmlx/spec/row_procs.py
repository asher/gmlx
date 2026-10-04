"""A request's logits processors on the owned MTP rounds.

Plain decode runs a row's processors (logit bias, the repetition,
presence and frequency penalties, a ``response_format`` grammar, XTC) on
each step's logits, with the prompt plus the generated tokens as context.
A verify forward scores several positions at once, so ``SpecRowProcessors``
runs the same chain, in the same order, at every position: position j gets
the committed context plus the drafts ahead of it. The walk then samples
the target from the processed logits, so every emitted token has the
distribution plain decode gives it, whatever the drafter proposed.

Processors that the stock ``ThinkingAwareLogitsProcessor`` wraps stay off
until the thinking end marker and then see only the tokens from the marker
on, as in plain decode.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import mlx.core as mx
import numpy as np

from .grammar import SpecGrammar, _fresh_matcher, apply_masks


class UnsupportedProcessor(ValueError):
    pass


@dataclass
class _Gate:
    """Thinking state of one wrapped processor. ``start`` indexes the row
    context where the processor's own context begins, once active."""

    end_id: int
    start: int | None


@dataclass
class _Step:
    fn: Callable | None          # None: the grammar step
    gate: _Gate | None


@dataclass
class RoundPlan:
    """One round's host-side inputs for ``SpecRowProcessors.apply``."""

    masks: np.ndarray | None     # packed grammar masks
    starts: list                 # per step: per-position context start or None
    seq: mx.array | None         # row context plus the drafts
    n_context: int


def supported(processors) -> bool:
    """Whether every processor can run per verify position."""
    from mlx_vlm.structured import (
        LLGuidanceLogitsProcessor,
        ThinkingAwareLogitsProcessor,
    )

    for proc in processors or ():
        if isinstance(proc, ThinkingAwareLogitsProcessor):
            proc = proc.processor
        if isinstance(proc, LLGuidanceLogitsProcessor):
            continue
        if not callable(proc) or hasattr(proc, "process_last_token"):
            return False
    return True


class SpecRowProcessors:
    """One row's processor chain, driven by verified rounds."""

    def __init__(self, steps: list[_Step], context, grammar=None):
        self.steps = steps
        self.context = [int(t) for t in context]
        self.grammar = grammar
        self.started = False
        self._context_arr = None

    @classmethod
    def from_processors(cls, processors, context) -> SpecRowProcessors | None:
        """The chain in a row's stock processors, or None when it is empty.
        ``context`` is the row's prompt tokens."""
        from mlx_vlm.structured import (
            LLGuidanceLogitsProcessor,
            ThinkingAwareLogitsProcessor,
        )

        steps: list[_Step] = []
        grammar = None
        for proc in processors or ():
            gate, inner = None, proc
            if isinstance(proc, ThinkingAwareLogitsProcessor):
                gate = _Gate(int(proc.thinking_end_token_id),
                             len(context) if proc._active else None)
                inner = proc.processor
            if isinstance(inner, LLGuidanceLogitsProcessor):
                if grammar is not None:
                    raise UnsupportedProcessor(
                        "more than one response_format grammar")
                grammar = SpecGrammar(
                    _fresh_matcher(inner),
                    vocab_size=inner.llg_tokenizer.vocab_size,
                    active=gate.start is not None if gate else True,
                    end_id=gate.end_id if gate else None)
                steps.append(_Step(None, None))
            elif callable(inner) and not hasattr(inner, "process_last_token"):
                steps.append(_Step(inner, gate))
            else:
                raise UnsupportedProcessor(
                    f"logits processor {type(inner).__name__} cannot run "
                    "on speculative rounds")
        return cls(steps, context, grammar) if steps else None

    def start(self, token: int) -> None:
        """Commit the first sampled token. A rebuilt round loop restarts
        from a token this row already holds, so later calls do nothing."""
        if not self.started:
            self.commit([token])

    def commit(self, tokens) -> None:
        """Advance past verified tokens."""
        self.started = True
        tokens = [int(t) for t in tokens]
        if not tokens:
            return
        n = len(self.context)
        self.context.extend(tokens)
        self._context_arr = None
        for step in self.steps:
            gate = step.gate
            if gate is not None and gate.start is None and gate.end_id in tokens:
                gate.start = n + tokens.index(gate.end_id)
        if self.grammar is not None:
            self.grammar.commit(tokens)

    def plan(self, drafts) -> RoundPlan | None:
        """Host-side inputs for a round that verifies ``drafts``, or None
        when no position needs processing."""
        drafts = [int(d) for d in drafts]
        n_ctx, n_pos = len(self.context), len(drafts) + 1
        masks = self.grammar.masks(drafts) if self.grammar is not None else None
        starts: list = []
        any_fn = False
        for step in self.steps:
            if step.fn is None:
                starts.append(None)
                continue
            gate = step.gate
            start = 0 if gate is None else gate.start
            rows = []
            for j in range(n_pos):
                rows.append(start)
                if (start is None and j < len(drafts)
                        and drafts[j] == gate.end_id):
                    start = n_ctx + j
            any_fn = any_fn or any(r is not None for r in rows)
            starts.append(rows)
        if masks is None and not any_fn:
            return None
        seq = None
        if any_fn:
            if self._context_arr is None:
                self._context_arr = mx.array(self.context, dtype=mx.int32)
            seq = self._context_arr
            if drafts:
                seq = mx.concatenate(
                    [seq, mx.array(drafts, dtype=mx.int32)])
        return RoundPlan(masks=masks, starts=starts, seq=seq, n_context=n_ctx)

    def apply(self, logits: mx.array, plan: RoundPlan) -> mx.array:
        """Run the chain on [n_pos, V] logits, one row per position."""
        x = logits
        seq, n = plan.seq, plan.n_context
        for step, rows in zip(self.steps, plan.starts):
            if step.fn is None:
                if plan.masks is not None:
                    x = apply_masks(x, plan.masks)
                continue
            if all(s is None for s in rows):
                continue
            assert seq is not None
            x = mx.concatenate(
                [x[j:j + 1] if s is None else step.fn(seq[s:n + j], x[j:j + 1])
                 for j, s in enumerate(rows)], axis=0)
        return x

    def processor(self, drafts) -> Callable[[mx.array], mx.array] | None:
        """``apply`` bound to this round's plan, or None."""
        plan = self.plan(drafts)
        if plan is None:
            return None
        return lambda logits: self.apply(logits, plan)


def batch_processors(rows, active_idx, drafts_rows):
    """Per-row round processors for a batch round, or None when no row
    needs any."""
    out = []
    for j, orig in enumerate(active_idx):
        rp = rows[orig] if orig < len(rows) else None
        out.append(rp.processor(drafts_rows[j]) if rp is not None else None)
    return out if any(p is not None for p in out) else None
