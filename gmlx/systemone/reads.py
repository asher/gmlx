# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Ported from vLLM examples/features/structured_diffusion/structured_server.py
# at ab3de6edf2 and modified for gmlx (Apache-2.0; see
# licenses/vllm-LICENSE).
"""The read request and result types shared by the engine and the decision
logic, and the canvas seeding rule. No MLX imports."""

from __future__ import annotations

import random
from dataclasses import dataclass

PAD = 0
TURN_CLOSE = 106
# The DiffusionGemma vocabulary size, a constant in the vLLM example.
SEED_VOCAB = 262144
LABEL_ID_CAP = 128


class Cancelled(Exception):
    """The caller asked the decision to stop, for example because the client
    disconnected."""


@dataclass(frozen=True)
class Slot:
    """One question's answer position in the canvas and its label ids."""

    pos: int
    label_ids: tuple[int, ...]


@dataclass(frozen=True)
class ReadRequest:
    """One read of a prompt: the template, its slots, the canvas width and
    one seed per sample."""

    template: tuple[int, ...]
    slots: tuple[Slot, ...]
    width: int
    seeds: tuple[int, ...]
    steps: int = 1
    constrained: bool = True
    pinned: bool = True
    rows_per_pass: int | None = None
    trace: bool = False


@dataclass(frozen=True)
class SlotRead:
    """The argmax id at a slot and the log-probabilities the read reports
    there: the argmax plus the label union."""

    argmax_id: int
    top: dict[int, float]


@dataclass(frozen=True)
class SampleRead:
    seed: int
    canvas_in: tuple[int, ...]
    slots: tuple[SlotRead, ...]
    trace: tuple[tuple[int, ...], ...] | None = None


@dataclass(frozen=True)
class ReadResult:
    samples: tuple[SampleRead, ...]
    prompt_len: int
    timing_ms: dict[str, float]


def label_id_union(slots) -> list[int]:
    """The sorted union of every slot's label ids, capped as vLLM caps a
    request's logprob_token_ids."""
    ids = sorted({i for s in slots for i in _label_ids(s)})
    return ids[:LABEL_ID_CAP]


def _label_ids(slot):
    return slot.label_ids if isinstance(slot, Slot) else slot["label_ids"]


def _pos(slot):
    return slot.pos if isinstance(slot, Slot) else slot["pos"]


def build_canvas(template, slots, width: int, seed: int, vocab: int) -> list[int]:
    """The seed canvas: the template, the turn close, pad to ``width``, and
    one random id per slot drawn in slot order from ``seed``. The draw is
    below ``SEED_VOCAB`` whatever the model reports, as in the vLLM example,
    so a seed gives the same canvas on both. It wraps into a smaller
    ``vocab``."""
    rng = random.Random(seed)
    canvas = list(template) + [TURN_CLOSE]
    canvas += [PAD] * (width - len(canvas))
    for s in slots:
        canvas[_pos(s)] = rng.randrange(SEED_VOCAB) % vocab
    return canvas


def pinned_positions(slots, width: int, steps: int) -> list[int]:
    """Past one step every canvas position that is not a slot holds its
    seed value. One step pins nothing."""
    if steps <= 1:
        return []
    free = {_pos(s) for s in slots}
    return [p for p in range(width) if p not in free]
