"""Context copy: speculative drafts taken from the prompt.

When the newest tokens of a generation also occur in the prompt, the prompt
tokens that follow that occurrence are a cheap draft. A round that copies
them verifies the copied run in place of the drafter's block, with the same
acceptance rule: an exact match at temperature 0, and under the stochastic
walk a copied token is a proposal with all its mass on one token. The
drafter still ingests the committed positions, so it drafts the next round
from full context.

The prompt index (``PromptIndex``) hashes every ``KEY``-token run of the
prompt in one vectorized pass, built while the prefill runs. A lookup takes
the generation's last ``KEY`` tokens and prefers the candidate whose
agreement with the generation runs furthest back, then the latest one.

A copy can run longer than the drafter's block. A verify wider than a
drafter round runs other kernels, whose rounding can pick a different
token where the target's top two candidates nearly tie, so the output can
differ from a run without copies from there on. ``GMLX_CONTEXT_COPY_MAX``
at the drafter's draft length keeps it token-identical. A copy is never
wider than the target can roll back (``spec_rollback_width``); where that
leaves less than ``SHORTEST_USEFUL`` tokens, the request does not copy.

Copy length adapts per request. At temperature 0, a copy round that is
accepted whole doubles the next copy, up to the longest, and a copy round
can add at least twice the tokens of an average drafter round. When
sampling, a copied token survives with the target's probability of it, so
whole accepts come in shorter runs: the length grows by half a start
length per whole accept, stops where the chance that a copied token
survives, from the request's own copy rounds, falls under ``KEEP``, and a
copy runs only when the tokens it is expected to add reach
``SAMPLED_MARGIN`` times a drafter round's.

A round that keeps under a quarter of its copy, or that yields fewer tokens
per second than the request's drafter rounds, is a misfire: copying pauses
for a number of tokens that doubles with each misfire in a row, and the
length starts over. The rate test and the floors matter for block
drafters, whose rounds verify a nearly accepted block for one drafter pass:
a wide copy verify that keeps a few tokens costs more than it saves.
"""

from __future__ import annotations

import math
import os
from collections import deque
from collections.abc import Sequence

import numpy as np

KEY = 6             # tokens that must match to propose a copy
REACH = 16          # tokens compared when ranking candidates
START = 8           # first copy length
LONGEST = 24        # default longest copy
# Upper bound of GMLX_CONTEXT_COPY_MAX. A target's rollback width
# (copy_limit) can bind first.
CEILING = 31
SHORTEST_USEFUL = 4  # copies shorter than this are not worth a round
CANDIDATES = 32     # latest candidate positions ranked per lookup
PAUSE = 16          # tokens without copies after a first misfire
PAUSE_MAX = 1024
RATE_WEIGHT = 0.25  # weight of the newest drafter round in its averages
# Sampling only.
KEEP = 0.3          # least survival chance worth a copied position
PRIOR_KEPT = 7.0    # prior of the per-token copy accept rate, 7 in 8
PRIOR_LOST = 1.0
DECAY = 0.9         # weight of older copy rounds in the accept rate
SAMPLED_MARGIN = 1.5

_MUL = np.uint64(0x9E3779B97F4A7C15)


def context_copy_enabled(drafter) -> bool:
    """GMLX_CONTEXT_COPY ``1`` or ``0`` sets it for every model; unset, the
    model's ``context_copy`` config key decides (stamped on the drafter at
    load)."""
    raw = os.environ.get("GMLX_CONTEXT_COPY", "").strip().lower()
    if raw in ("1", "on", "true", "yes"):
        return True
    if raw in ("0", "off", "false", "no"):
        return False
    return bool(getattr(drafter, "context_copy", False))


def longest_copy() -> int:
    """GMLX_CONTEXT_COPY_MAX, clamped to 2..CEILING."""
    try:
        n = int(os.environ.get("GMLX_CONTEXT_COPY_MAX", LONGEST))
    except ValueError:
        n = LONGEST
    return max(2, min(n, CEILING))


def target_copy_limit(lm) -> int | None:
    """Longest copy the target's rollback can undo after a partial accept:
    ``spec_rollback_width`` less the bonus position, or None for a target
    that undoes any width."""
    width = getattr(lm, "spec_rollback_width", None)
    return None if width is None else int(width) - 1


def copy_limit(lm) -> int | None:
    """The longest copy for this target, or None when the target cannot
    roll back a useful copy."""
    cap = longest_copy()
    rollback = target_copy_limit(lm)
    if rollback is not None:
        cap = min(cap, rollback)
    return cap if cap >= SHORTEST_USEFUL else None


def _hash_rows(ids: np.ndarray, n_rows: int) -> np.ndarray:
    h = np.zeros(n_rows, dtype=np.uint64)
    for j in range(KEY):
        h = h * _MUL + ids[j:j + n_rows].astype(np.uint64)
    return h


class PromptIndex:
    """Hashes of every ``KEY``-token run of a prompt that a token follows,
    built in one vectorized pass."""

    def __init__(self, prompt):
        self.ids = np.asarray(prompt, dtype=np.int64).reshape(-1)
        n_rows = max(len(self.ids) - KEY, 0)
        self._hash = _hash_rows(self.ids, n_rows)

    def __len__(self) -> int:
        return len(self.ids)

    def candidates(self, key: Sequence[int]) -> np.ndarray:
        """End positions of the prompt's runs equal to ``key``, latest
        last, at most ``CANDIDATES``."""
        if not len(self._hash):
            return np.zeros(0, dtype=np.int64)
        k = np.asarray(key, dtype=np.int64)
        rows = np.flatnonzero(self._hash == _hash_rows(k, 1)[0])
        if not len(rows):
            return rows
        rows = rows[-CANDIDATES:]
        same = (self.ids[rows[:, None] + np.arange(KEY)] == k).all(axis=1)
        return rows[same] + KEY


class PromptCopy:
    """Per-request prompt lookup state. ``extend`` takes every emitted
    token, ``propose`` returns a copy or an empty list, ``record`` takes the
    outcome of each copy round and ``drafted`` that of each drafter
    round."""

    def __init__(self, prompt, *, longest: int | None = None,
                 sampled: bool = False):
        self._index = (prompt if isinstance(prompt, PromptIndex)
                       else PromptIndex(prompt))
        ids = self._index.ids
        self._tail: deque[int] = deque(ids[-REACH:].tolist(), maxlen=REACH)
        self.longest = longest if longest is not None else longest_copy()
        self.sampled = sampled
        self.length = min(START, self.longest)
        self._wait = 0
        self._pause = PAUSE
        self._draft_rate: float | None = None
        self._draft_yield = 0.0
        self._kept = 0.0
        self._lost = 0.0
        self.rounds = 0
        self.copied = 0
        self.accepted = 0

    def extend(self, tokens: Sequence[int]) -> None:
        for t in tokens:
            self._tail.append(int(t))
        self._wait -= len(tokens)

    def _lookup(self) -> int | None:
        tail = list(self._tail)
        if len(tail) < KEY:
            return None
        cands = self._index.candidates(tail[-KEY:])
        if not len(cands):
            return None
        ids = self._index.ids
        best, best_run = None, -1
        for pos in reversed(cands.tolist()):
            lo = max(pos - KEY - (len(tail) - KEY), 0)
            before = ids[lo:pos - KEY].tolist()
            run = 0
            while (run < len(before)
                   and before[-1 - run] == tail[-KEY - 1 - run]):
                run += 1
            if run > best_run:
                best, best_run = pos, run
        return best

    def accept_rate(self) -> float:
        """Chance that a copied token survives, from this request's copy
        rounds (sampling)."""
        return ((self._kept + PRIOR_KEPT)
                / (self._kept + self._lost + PRIOR_KEPT + PRIOR_LOST))

    def _sampled_cap(self, r: float) -> int:
        if r >= 1.0:
            return self.longest
        return max(1, min(self.longest, int(math.log(KEEP) / math.log(r))))

    def propose(self, shortest: int, budget: int) -> list[int]:
        """A copy of at least ``shortest`` and at most ``budget`` tokens, or
        an empty list."""
        shortest = max(shortest, 1)
        if not self.sampled:
            shortest = max(shortest, math.ceil(2 * self._draft_yield) - 1)
        if self._wait > 0 or budget < shortest or shortest > self.longest:
            return []
        pos = self._lookup()
        if pos is None:
            return []
        n = min(max(self.length, shortest), budget,
                len(self._index) - pos)
        if self.sampled:
            r = self.accept_rate()
            n = min(n, self._sampled_cap(r))
            expected = 1 + (n if r >= 1.0 else r * (1 - r ** n) / (1 - r))
            if (n >= shortest and self._draft_yield
                    and expected < SAMPLED_MARGIN * self._draft_yield):
                return []
        if n < shortest:
            return []
        return self._index.ids[pos:pos + n].tolist()

    def drafted(self, tokens: int, seconds: float) -> None:
        """Take a drafter round's committed tokens and wall time."""
        if seconds <= 0:
            return
        rate = tokens / seconds
        if self._draft_rate is None:
            self._draft_rate, self._draft_yield = rate, float(tokens)
            return
        self._draft_rate += RATE_WEIGHT * (rate - self._draft_rate)
        self._draft_yield += RATE_WEIGHT * (tokens - self._draft_yield)

    def record(self, copied: int, accepted: int,
               seconds: float | None = None) -> None:
        """Take a copy round's outcome: drafts copied, drafts accepted and,
        when known, the round's wall time."""
        self.rounds += 1
        self.copied += copied
        self.accepted += accepted
        self._kept = self._kept * DECAY + accepted
        self._lost = self._lost * DECAY + (1 if accepted < copied else 0)
        slow = (seconds is not None and seconds > 0
                and self._draft_rate is not None
                and (accepted + 1) / seconds < self._draft_rate)
        if accepted * 4 < copied or slow:
            self._wait = self._pause
            self._pause = min(self._pause * 2, PAUSE_MAX)
            self.length = min(START, self.longest)
        elif accepted >= copied:
            grown = (max(self.length, copied) + START // 2 if self.sampled
                     else max(self.length, copied) * 2)
            self.length = min(grown, self.longest)
            self._pause = PAUSE
