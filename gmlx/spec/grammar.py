"""Grammar-constrained speculative rounds (``response_format`` on MTP models).

A verify forward scores every draft position at once, so each position
needs the grammar mask for the text before it: the committed tokens plus
the drafts ahead of it. ``SpecGrammar.masks`` walks the matcher through the
drafts, records one mask per position, and rolls the matcher back. A draft
the grammar refuses ends the walk: the target cannot sample that draft
under its own mask, so the round rejects it and never reads the later
positions. The walk then takes the target sample from the masked logits,
so the output distribution is the one constrained plain decode gives.
"""

from __future__ import annotations

import logging

import mlx.core as mx
import numpy as np

_log = logging.getLogger(__name__)


class SpecGrammar:
    """One request's llguidance matcher, driven by verified rounds.

    ``active`` is False while the model thinks: the grammar constrains the
    answer, so it starts on the token after ``end_id`` (the stock
    ``ThinkingAwareLogitsProcessor`` contract).
    """

    def __init__(self, matcher, *, vocab_size: int, active: bool = True,
                 end_id: int | None = None):
        self.matcher = matcher
        self.words = (int(vocab_size) + 31) // 32
        self.active = bool(active)
        self.end_id = end_id
        self.started = False

    @classmethod
    def from_processors(cls, processors) -> SpecGrammar | None:
        """The grammar in a row's stock logits processors, or None."""
        from mlx_vlm.structured import (
            LLGuidanceLogitsProcessor,
            ThinkingAwareLogitsProcessor,
        )

        for proc in processors or ():
            active, end_id, inner = True, None, proc
            if isinstance(proc, ThinkingAwareLogitsProcessor):
                active = bool(proc._active)
                end_id = int(proc.thinking_end_token_id)
                inner = proc.processor
            if isinstance(inner, LLGuidanceLogitsProcessor):
                return cls(_fresh_matcher(inner),
                           vocab_size=inner.llg_tokenizer.vocab_size,
                           active=active, end_id=end_id)
        return None

    def start(self, token: int) -> None:
        """Commit the first sampled token. A rebuilt round loop restarts
        from a token this grammar already holds, so later calls do nothing."""
        if not self.started:
            self.commit([token])

    def commit(self, tokens) -> None:
        """Advance past verified tokens."""
        self.started = True
        m = self.matcher
        for t in tokens:
            t = int(t)
            if not self.active:
                self.active = t == self.end_id
                continue
            if m.is_stopped():
                # The grammar is complete. The row stops on this token.
                return
            if not m.consume_token(t):
                raise ValueError(
                    f"LLGuidance matcher error: {m.get_error()}")

    def masks(self, drafts) -> np.ndarray | None:
        """Packed masks for positions 0..len(drafts), int32 [n + 1, words].

        Row j allows the tokens that may follow the committed text plus
        ``drafts[:j]``. Rows past a refused draft, or past an end of
        sequence draft, allow every token, because the walk never reads
        them.
        None when no position is constrained (thinking, with no end
        marker among the drafts).
        """
        drafts = [int(d) for d in drafts]
        active = self.active
        if not active and self.end_id not in drafts:
            return None
        import llguidance.numpy as llg_np

        m = self.matcher
        out = np.full((len(drafts) + 1, self.words), -1, dtype=np.int32)
        consumed = 0
        try:
            for j in range(len(drafts) + 1):
                if active:
                    llg_np.fill_next_token_bitmask(m, out, j)
                if j == len(drafts):
                    break
                d = drafts[j]
                if not active:
                    active = d == self.end_id
                    continue
                if not _allowed(out[j], d) or m.is_stopped():
                    break
                if not m.consume_token(d):
                    break
                consumed += 1
        finally:
            if consumed:
                m.rollback(consumed)
        return out


def _fresh_matcher(proc):
    """A matcher at the start of the grammar. The stock processor sets one
    up when it masks the first token and consumes nothing until the next
    call, so that one is reused when present."""
    matchers = getattr(proc, "ll_matchers", None)
    if matchers and len(matchers) == 1:
        return matchers[0]
    from llguidance import LLMatcher

    m = LLMatcher(proc.llg_tokenizer, proc.grammar)
    err = m.get_error()
    if err:
        raise ValueError(f"LLGuidance matcher error: {err}")
    return m


def _allowed(row: np.ndarray, token: int) -> bool:
    word = token >> 5
    return word < row.shape[0] and bool((int(row[word]) >> (token & 31)) & 1)


def apply_masks(logits: mx.array, masks: np.ndarray) -> mx.array:
    """Set every token a packed mask refuses to -inf.

    ``logits`` is [..., V] with one row per mask row (any leading shape
    that flattens to ``masks.shape[0]`` rows). Vocab entries past the
    mask (a padded output head) are refused.
    """
    shape = logits.shape
    vocab = shape[-1]
    flat = logits.reshape(-1, vocab)
    packed = mx.array(np.ascontiguousarray(masks).reshape(flat.shape[0], -1)
                      .view(np.uint32))
    shifts = mx.arange(32, dtype=mx.uint32)
    bits = ((packed[:, :, None] >> shifts) & 1).reshape(flat.shape[0], -1)
    width = bits.shape[1]
    if width >= vocab:
        bits = bits[:, :vocab]
    else:
        bits = mx.concatenate(
            [bits, mx.zeros((flat.shape[0], vocab - width), dtype=bits.dtype)],
            axis=1)
    neg = mx.array(-float("inf"), dtype=flat.dtype)
    return mx.where(bits.astype(mx.bool_), flat, neg).reshape(shape)


def batch_masks(grammars, active_idx, drafts_rows, n_pos: int):
    """Stacked masks for a batch round, int32 [B, n_pos, words], or None
    when no row is constrained. Rows without a grammar allow every token."""
    rows = []
    words = 0
    for j, orig in enumerate(active_idx):
        g = grammars[orig] if orig < len(grammars) else None
        m = g.masks(drafts_rows[j]) if g is not None else None
        rows.append(m)
        if m is not None:
            words = m.shape[1]
    if not words:
        return None
    out = np.full((len(rows), n_pos, words), -1, dtype=np.int32)
    for j, m in enumerate(rows):
        if m is not None:
            out[j] = m
    return out
