"""RoPE tables for the kq q and k kernel, which rotates adjacent pairs.

A video family rotates the first ``rot`` dims of each head as split
halves, pair (i, i + rot / 2), and leaves the rest of the head as it is.
``pair_halves`` moves each such pair side by side in q, k and their norm
weights; the dims past ``rot`` keep their order and get cos 1 and sin 0.
The q and k dot product is the same under one permutation of both, so v
and the attention output are untouched.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np


def pair_halves(x: mx.array, rot: int) -> mx.array:
    """The last dim of ``x`` with dims i and i + rot / 2 side by side for
    i < rot / 2, and dims past ``rot`` unchanged."""
    *lead, d = x.shape
    half = rot // 2
    head = x[..., :rot].reshape(*lead, 2, half).swapaxes(-1, -2)
    head = head.reshape(*lead, rot)
    return head if rot == d else mx.concatenate([head, x[..., rot:]], axis=-1)


def angles(positions: np.ndarray, inv_freq: mx.array) -> mx.array:
    """(rows, axes * freqs) float32 angles: each axis's position times
    every frequency, axis by axis."""
    pos = mx.array(np.asarray(positions, np.float64).astype(np.float32))
    inv = inv_freq.astype(mx.float32)
    a = pos[:, :, None] * inv[None, None, :]
    return a.reshape(pos.shape[0], -1)


def tables(positions: np.ndarray, inv_freq: mx.array,
           head_dim: int) -> tuple[mx.array, mx.array]:
    """cos and sin, each (rows, head_dim // 2) float32, in the pair order
    of ``pair_halves``."""
    a = angles(positions, inv_freq)
    pad = head_dim // 2 - a.shape[1]
    if pad < 0:
        raise ValueError(f"{a.shape[1]} angle pairs do not fit a head of "
                         f"{head_dim}")
    cos, sin = mx.cos(a), mx.sin(a)
    if pad:
        n = a.shape[0]
        cos = mx.concatenate([cos, mx.ones((n, pad), mx.float32)], axis=1)
        sin = mx.concatenate([sin, mx.zeros((n, pad), mx.float32)], axis=1)
    return cos, sin
