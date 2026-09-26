# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 TypeSafe AI
# Ported from typesafe-ai/system-one-adapter-python
# src/system_one_adapter/_utils/confidence_metrics.py at e1d4cc9 (MIT; see
# licenses/system-one-adapter-LICENSE).
"""The official Jev confidence of a choice or score answer."""

from __future__ import annotations


def score_confidence(probs: list[float]) -> float:
    """How closely a score's probabilities gather at the most likely level:
    one minus the expected distance from that level over the same distance
    for a uniform answer, at least 0."""
    if len(probs) == 1:
        return 1.0
    p = _normalize(probs)
    mode = max(range(len(p)), key=p.__getitem__)
    distance = sum(pi * abs(i - mode) for i, pi in enumerate(p))
    center = (len(p) - 1) / 2
    uniform = sum(abs(i - center) for i in range(len(p))) / len(p)
    return max(0.0, 1.0 - distance / uniform)


def choice_confidence(probs: list[float]) -> float:
    """The top probability scaled from uniform (0) to certain (1)."""
    if len(probs) == 1:
        return 1.0
    p = _normalize(probs)
    uniform = 1.0 / len(p)
    return (max(p) - uniform) / (1.0 - uniform)


def _normalize(probs: list[float]) -> list[float]:
    total = sum(probs)
    if total == 0:
        return [1.0 / len(probs)] * len(probs)
    return [p / total for p in probs]
