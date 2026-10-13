"""Context copy: speculative drafts taken from the prompt.

When the newest tokens of a generation also occur in the prompt, the prompt
tokens that follow that occurrence are a cheap draft. A round that copies
them verifies the copied run in place of the drafter's block, with the same
acceptance rule: an exact match at temperature 0, and under the stochastic
walk a copied token is a proposal with all its mass on one token. The
drafter still ingests the committed positions, so it drafts the next round
from full context.

The prompt index maps every ``KEY``-token run of the prompt to the
positions that follow it. A lookup takes the generation's last ``KEY``
tokens and prefers the candidate whose agreement with the generation runs
furthest back, then the latest one.

Copy length adapts per request. A copy round that is accepted whole doubles
the next copy, up to the longest. A round that keeps under a quarter of its
copy, or that yields fewer tokens per second than the request's drafter
rounds, is a misfire: copying pauses for a number of tokens that doubles
with each misfire in a row, and the length starts over. A copy is also at
least twice as long as the tokens a drafter round commits on average, so a
drafter that already drafts the text well keeps its rounds. Both tests
matter for block drafters, whose rounds verify a nearly accepted block for
one drafter pass: a wide copy verify that keeps a few tokens costs more
than it saves.
"""

from __future__ import annotations

import math
import os
from collections import deque
from collections.abc import Sequence

KEY = 6             # tokens that must match to propose a copy
REACH = 16          # tokens compared when ranking candidates
START = 8           # first copy length
LONGEST = 24        # default longest copy
CEILING = 31        # verify rows a buffered rotating target cache can trim
CANDIDATES = 32     # latest candidate positions ranked per lookup
PAUSE = 16          # tokens without copies after a first misfire
PAUSE_MAX = 1024
RATE_WEIGHT = 0.25  # weight of the newest drafter round in its averages


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
    """GMLX_CONTEXT_COPY_MAX, clamped to the verify width a target cache
    can roll back."""
    try:
        n = int(os.environ.get("GMLX_CONTEXT_COPY_MAX", LONGEST))
    except ValueError:
        n = LONGEST
    return max(2, min(n, CEILING))


class PromptCopy:
    """Per-request prompt lookup state. ``extend`` takes every emitted
    token, ``propose`` returns a copy or an empty list, ``record`` takes the
    outcome of each copy round."""

    def __init__(self, prompt: Sequence[int], *, longest: int | None = None):
        self._prompt = [int(t) for t in prompt]
        self._grams: dict[tuple[int, ...], list[int]] | None = None
        self._tail: deque[int] = deque(self._prompt[-REACH:], maxlen=REACH)
        self.longest = longest if longest is not None else longest_copy()
        self.length = min(START, self.longest)
        self._wait = 0
        self._pause = PAUSE
        self._draft_rate: float | None = None
        self._draft_yield = 0.0
        self.rounds = 0
        self.copied = 0
        self.accepted = 0

    def _index(self) -> dict[tuple[int, ...], list[int]]:
        # Built on the first lookup, off the prefill's critical path.
        if self._grams is None:
            grams: dict[tuple[int, ...], list[int]] = {}
            p = self._prompt
            for end in range(KEY, len(p)):
                grams.setdefault(tuple(p[end - KEY:end]), []).append(end)
            self._grams = grams
        return self._grams

    def extend(self, tokens: Sequence[int]) -> None:
        for t in tokens:
            self._tail.append(int(t))
        self._wait -= len(tokens)

    def _lookup(self) -> int | None:
        tail = list(self._tail)
        if len(tail) < KEY:
            return None
        cands = self._index().get(tuple(tail[-KEY:]))
        if not cands:
            return None
        p = self._prompt
        best, best_run = None, -1
        for pos in reversed(cands[-CANDIDATES:]):
            run = 0
            while (KEY + run < len(tail) and pos - KEY - 1 - run >= 0
                   and p[pos - KEY - 1 - run] == tail[-KEY - 1 - run]):
                run += 1
            if run > best_run:
                best, best_run = pos, run
        return best

    def propose(self, shortest: int, budget: int) -> list[int]:
        """A copy of at least ``shortest`` and at most ``budget`` tokens, or
        an empty list."""
        shortest = max(shortest, 1, math.ceil(2 * self._draft_yield) - 1)
        if self._wait > 0 or budget < shortest or shortest > self.longest:
            return []
        pos = self._lookup()
        if pos is None:
            return []
        n = min(max(self.length, shortest), budget, len(self._prompt) - pos)
        if n < shortest:
            return []
        return self._prompt[pos:pos + n]

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
        slow = (seconds is not None and seconds > 0
                and self._draft_rate is not None
                and (accepted + 1) / seconds < self._draft_rate)
        if accepted * 4 < copied or slow:
            self._wait = self._pause
            self._pause = min(self._pause * 2, PAUSE_MAX)
            self.length = min(START, self.longest)
        elif accepted >= copied:
            self.length = min(max(self.length, copied) * 2, self.longest)
            self._pause = PAUSE
