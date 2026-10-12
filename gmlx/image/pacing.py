# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Asher Feldman
"""Pause points of a long image evaluation.

A DiT forward builds one graph per step, and the GPU runs it as one
evaluation that can last seconds. ``point(*arrays)`` marks a place
between two blocks where the thread's pacer may evaluate what the forward
has built so far and let other GPU work run before the next block.

With no pacer in force on the thread, a point does nothing, and the
forward evaluates as one graph, as it would without the point. A pacer
that evaluates at a point changes where the GPU work splits, never its
result: the ops and their inputs stay the same.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager

_local = threading.local()


def point(*arrays) -> None:
    """A point between two blocks of an image forward. ``arrays`` are what
    the forward carries to the next block."""
    fn = getattr(_local, "pacer", None)
    if fn is not None:
        fn(arrays)


@contextmanager
def paced(pacer: Callable[[tuple], None] | None) -> Iterator[None]:
    """Run the block with ``pacer`` called at each point on this thread.
    None runs the block with no pacer."""
    prev = getattr(_local, "pacer", None)
    _local.pacer = pacer
    try:
        yield
    finally:
        _local.pacer = prev
