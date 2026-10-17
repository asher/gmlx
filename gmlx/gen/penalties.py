"""Presence and frequency penalties over the generated tokens.

The OpenAI ``presence_penalty`` subtracts the penalty once from every token
the response holds, and ``frequency_penalty`` subtracts it once per
occurrence. vLLM and SGLang count every token generated so far and never
the prompt, and model cards tune their values for that. The copied mlx-lm
and mlx-vlm samplers return :class:`OutputPenalty` for both.

A processor takes the decode loop's ``(tokens, logits)``: ``tokens`` is
the row's history, which ends with the last sampled token. The history at
the first call holds only the prompt, or its tail, since that call scores
the first generated token. Later calls add the tokens generated since.

``context_size`` None or 0 counts the whole output, in a count per vocab
entry that each call updates with the tokens added since the last call.
A positive ``context_size`` counts only the last N generated tokens.

A speculative verify scores each position with the drafts ahead of it, so
the history can grow by several tokens and then shrink. A call whose
history is no longer than the last one's is a rollback: the history before
its last token is unchanged, and the tokens counted after that point are
taken back out.
"""

from __future__ import annotations

import mlx.core as mx


def _bump(counts, tokens, delta, vocab):
    if counts is None:
        counts = mx.zeros((vocab,), dtype=mx.int32)
    return counts.at[tokens].add(delta)


class OutputPenalty:
    """A presence penalty, or a frequency penalty with ``frequency=True``,
    over one row's generated tokens. ``clone`` gives a fresh processor for
    another row."""

    def __init__(self, penalty: float, context_size: int | None = None, *,
                 frequency: bool = False):
        if context_size is not None and context_size < 0:
            raise ValueError(
                f"context_size must be None or at least 0, got {context_size}")
        self.penalty = penalty
        self.context_size = context_size or None
        self.frequency = frequency
        self.reset()

    def reset(self) -> None:
        self._base = None       # history length at the first call
        self._seen = 0          # history length at the last call
        self._hist = None       # the last call's history
        self._counts = None     # [V] int32 counts of history[base:seen]

    def clone(self) -> OutputPenalty:
        return OutputPenalty(self.penalty, self.context_size,
                             frequency=self.frequency)

    def __call__(self, tokens, logits: mx.array) -> mx.array:
        if not isinstance(tokens, mx.array):
            tokens = mx.array(tokens, dtype=mx.int32)
        tokens = tokens.reshape(-1)
        n = tokens.shape[0]
        if self._base is None:
            self._base = n
        if self.context_size is not None:
            return self._window(tokens, n, logits)

        lo = max(self._base, self._seen if n > self._seen else n - 1)
        counts, vocab = self._counts, logits.shape[-1]
        if lo < self._seen:
            counts = _bump(counts, self._hist[lo:self._seen], -1, vocab)
        if lo < n:
            counts = _bump(counts, tokens[lo:n], 1, vocab)
        self._counts, self._seen, self._hist = counts, n, tokens
        if counts is None:
            return logits
        hits = counts if self.frequency else counts > 0
        return logits - hits.astype(logits.dtype) * self.penalty

    def _window(self, tokens, n, logits):
        lo = max(self._base, n - self.context_size)
        if lo >= n:
            return logits
        recent = tokens[lo:n]
        if self.frequency:
            return logits.at[:, recent].subtract(self.penalty)
        logits[:, recent] -= self.penalty
        return logits
