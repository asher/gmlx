"""The keys and values of rows that a DiT runs once and every step reads.

Rows that no step changes, such as the tokens of reference images that
attend only to each other, need not run through the blocks on every step.
A family runs them through the blocks once, keeps each block's K and V
after RoPE, and each step joins them to the keys and values of its own
rows. The rows' queries, feed-forward and outputs then leave the step.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx


@dataclass(frozen=True)
class RefCache:
    """Each block's K and V of the cached rows after RoPE, (1, heads, R,
    head dim), in the order of the blocks."""
    k: tuple
    v: tuple

    def arrays(self) -> list:
        return [*self.k, *self.v]

    @property
    def tokens(self) -> int:
        return int(self.k[0].shape[2]) if self.k else 0

    @property
    def nbytes(self) -> int:
        return sum(int(a.nbytes) for a in self.arrays())


def cache_bytes(blocks: int, tokens: int, dim: int, itemsize: int) -> int:
    """Bytes of the K and V of ``tokens`` rows in ``blocks`` blocks of
    ``dim`` elements a row."""
    return 2 * blocks * tokens * dim * itemsize


def join(keys: list, values: list, cached_k: mx.array,
         cached_v: mx.array) -> tuple[mx.array, mx.array]:
    """Keys and values of a step: its own, in order, then the cached
    ones, broadcast to the step's batch. Every array is (B, heads, L,
    head dim)."""
    b = keys[0].shape[0]
    if cached_k.shape[0] != b:
        cached_k = mx.broadcast_to(cached_k, (b, *cached_k.shape[1:]))
        cached_v = mx.broadcast_to(cached_v, (b, *cached_v.shape[1:]))
    return (mx.concatenate([*keys, cached_k], axis=2),
            mx.concatenate([*values, cached_v], axis=2))
