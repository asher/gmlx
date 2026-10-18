"""Presence and frequency penalties over the generated tokens.

The OpenAI ``presence_penalty`` subtracts the penalty once from every token
the response holds, and ``frequency_penalty`` subtracts it once per
occurrence. vLLM and SGLang count every token generated so far and never
the prompt, and model cards tune their values for that. The copied mlx-lm
and mlx-vlm samplers return :class:`OutputPenalty` for both.

A processor takes the decode loop's ``(tokens, logits)``: ``tokens`` is
the row's history, which ends with the last sampled token. ``context_size``
None or 0 counts the whole output in a count per vocab entry. A positive
``context_size`` counts only the last N generated tokens.

A caller that knows what it committed says so. ``anchor(n)`` marks the
history before index n as accounted for, and ``commit(ids)`` counts the
ids generated after it. A call then counts the history past the anchor for
that call only, as the speculative verify rounds need for their drafts.
``resumed(ids)`` gives a fresh processor for a row replayed as its prompt
plus ``ids``.

A caller that only passes a history gets a rule from history lengths. The
history at the first call is context, since that call scores the first
generated token. A longer history adds the tokens past the last one. A
history no longer than the last one is a rollback of drafts: the history
before its last token is unchanged, and the tokens counted after that
point are taken back out.

Counts are indexed by token id and sized to the widest logits seen, so a
drafter with a narrower vocab than its target shares the processor.
"""

from __future__ import annotations

import mlx.core as mx


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
        self._anchor = None     # history length accounted for, when told
        self._base = None       # otherwise the history length at the first call
        self._seen = 0          # and the history length at the last call
        self._hist = None       # and the last call's history
        self._counts = None     # [width] int32 counts of the generated tokens
        self._width = 0
        self._tail = []         # window: last generated ids outside the history

    def clone(self) -> OutputPenalty:
        return OutputPenalty(self.penalty, self.context_size,
                             frequency=self.frequency)

    def resumed(self, generated) -> OutputPenalty:
        """A fresh processor for a row replayed as its prompt followed by
        ``generated``. Its first call scores the token after the replay,
        so that history is all context, and ``generated`` is counted."""
        out = self.clone()
        out._take([int(t) for t in generated])
        return out

    def anchor(self, n: int) -> None:
        """The history before index ``n`` is accounted for. Calls count the
        history past it for that call only, until ``commit``."""
        self._anchor = n

    def commit(self, ids) -> None:
        """Count ``ids``, generated right after the anchor, and move the
        anchor past them."""
        if self._anchor is None:
            raise RuntimeError("commit before anchor")
        ids = [int(t) for t in ids]
        self._take(ids)
        self._anchor += len(ids)

    def _take(self, ids) -> None:
        if not ids:
            return
        if self.context_size is not None:
            self._tail = (self._tail + ids)[-self.context_size:]
            return
        self._fit(max(ids) + 1)
        self._counts = self._counts.at[mx.array(ids, dtype=mx.int32)].add(1)

    def _fit(self, width: int) -> None:
        if width <= self._width:
            return
        grow = mx.zeros((width - self._width,), dtype=mx.int32)
        self._counts = (grow if self._counts is None
                        else mx.concatenate([self._counts, grow]))
        self._width = width

    def __call__(self, tokens, logits: mx.array) -> mx.array:
        if not isinstance(tokens, mx.array):
            tokens = mx.array(tokens, dtype=mx.int32)
        tokens = tokens.reshape(-1)
        n = tokens.shape[0]
        self._fit(logits.shape[-1])
        if self._anchor is not None:
            start = self._anchor
        else:
            if self._base is None:
                self._base = n
            start = self._base
        if self.context_size is not None:
            return self._apply(logits, self._window(tokens, n, start))
        if self._anchor is None:
            self._roll(tokens, n)
            return self._apply(logits, self._counts)
        counts = self._counts
        if n > start:
            counts = counts.at[tokens[start:n]].add(1)
        return self._apply(logits, counts)

    def _roll(self, tokens, n) -> None:
        lo = max(self._base, self._seen if n > self._seen else n - 1)
        counts = self._counts
        if lo < self._seen:
            counts = counts.at[self._hist[lo:self._seen]].add(-1)
        if lo < n:
            counts = counts.at[tokens[lo:n]].add(1)
        self._counts, self._seen, self._hist = counts, n, tokens

    def _window(self, tokens, n, start):
        size = self.context_size
        t = max(n - start, 0)
        parts = []
        if t < size and self._tail:
            parts.append(mx.array(self._tail[-(size - t):], dtype=mx.int32))
        if t:
            parts.append(tokens[max(start, n - size):n].astype(mx.int32))
        if not parts:
            return None
        recent = parts[0] if len(parts) == 1 else mx.concatenate(parts)
        return mx.zeros((self._width,), dtype=mx.int32).at[recent].add(1)

    def _apply(self, logits, counts):
        if counts is None:
            return logits
        counts = counts[:logits.shape[-1]]
        hits = counts if self.frequency else counts > 0
        return logits - hits.astype(logits.dtype) * self.penalty
