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
    answer, so it starts on the token after ``gate`` opens (an
    ``AnswerGate``, the stock ``ThinkingAwareLogitsProcessor`` contract).
    ``maybe`` (a start marker's ids) is that wrapper's maybe state: the
    first position also allows the first start token, which, when the
    grammar refuses it, opens a block whose rest is forced.
    """

    def __init__(self, matcher, *, vocab_size: int, active: bool = True,
                 gate=None, maybe=None):
        self.matcher = matcher
        self.words = (int(vocab_size) + 31) // 32
        self.active = bool(active)
        self.gate = gate
        self.maybe = tuple(int(t) for t in maybe) if maybe else None
        self.force: list[int] = []
        self.started = False

    def _accepts(self, token: int) -> bool:
        import llguidance.numpy as llg_np

        row = np.full((1, self.words), -1, dtype=np.int32)
        llg_np.fill_next_token_bitmask(self.matcher, row, 0)
        return _allowed(row[0], token)

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
            if self.maybe is not None:
                start, self.maybe = self.maybe, None
                if t == start[0] and not self._accepts(t):
                    self.active = False
                    self.force = list(start[1:])
                    if self.gate is not None:
                        self.gate.feed(t)
                    continue
            if self.force:
                if t == self.force[0]:
                    self.force.pop(0)
                else:
                    self.force = []
                if self.gate is not None:
                    self.gate.feed(t)
                continue
            if not self.active:
                self.active = self.gate is not None and self.gate.feed(t)
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
        if self.maybe is not None or self.force:
            return self._maybe_masks(drafts)
        active = self.active
        hit = None if active or self.gate is None else self.gate.find(drafts)
        if not active and hit is None:
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
                    active = j == hit
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


    def _maybe_masks(self, drafts) -> np.ndarray:
        """``masks`` in the maybe state, or while a start marker's rest is
        forced. Row j is the grammar's mask, the forced token alone, or
        every token while a block is open."""
        import llguidance.numpy as llg_np

        m = self.matcher
        n = len(drafts) + 1
        out = np.full((n, self.words), -1, dtype=np.int32)
        gate = self.gate.restart(self.gate.tail) if self.gate is not None else None
        maybe, force = self.maybe, list(self.force)
        active = self.active and not force
        consumed = 0
        try:
            for j in range(n):
                grammar_row = None
                if force:
                    out[j] = 0
                    _allow(out[j], force[0])
                elif active:
                    llg_np.fill_next_token_bitmask(m, out, j)
                    if maybe is not None:
                        grammar_row = out[j].copy()
                        _allow(out[j], maybe[0])
                if j == n - 1:
                    break
                d = drafts[j]
                if force:
                    if d != force.pop(0):
                        break
                    if gate is not None:
                        gate.feed(d)
                    continue
                if not active:
                    active = gate is not None and gate.feed(d)
                    continue
                if maybe is not None:
                    start, maybe = maybe, None
                    if not _allowed(grammar_row, d):
                        if d != start[0]:
                            break
                        active, force = False, list(start[1:])
                        if gate is not None:
                            gate.feed(d)
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


def _allow(row: np.ndarray, token: int) -> None:
    words = row.view(np.uint32)
    words[token >> 5] |= np.uint32(1 << (token & 31))


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
