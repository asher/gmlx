"""Owned DFlash drafters: the block-diffusion base and the DFlash 2 extras.

A DFlash drafter denoises a block ``[anchor, MASK * (block - 1)]`` in one
non-causal pass against a ring of context K/V projected from the target's
fused hidden states (the inject path). DFlash 2 adds a grouped dynamic causal
convolution around every attention and MLP sublayer of the draft path and a
candidate selector that walks a path through the top-k candidates of each
block position.

Leaf modules first (``GroupedDynamicConv``, ``CandidateSelector``); the
drafter classes follow.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


class GroupedDynamicConv(nn.Module):
    """Two-sided grouped dynamic causal convolution over block positions.

    ``base_kernel[side][tap]`` is a per-channel kernel; ``kernel_projection``
    adds a per-position, per-group dynamic term. Taps run over the block
    positions only, zero padded at the block start, so the convolution is
    block-local. ``prepare`` convolves a sublayer input with side 0 and hands
    back side 1's dynamic kernel; ``finish`` convolves the sublayer output
    with it.
    """

    def __init__(self, hidden_size: int, kernel_size: int, group_size: int):
        super().__init__()
        if hidden_size % group_size:
            raise ValueError(
                f"hidden_size {hidden_size} is not a multiple of "
                f"conv_group_size {group_size}")
        self.kernel_size = int(kernel_size)
        self.group_size = int(group_size)
        groups = hidden_size // self.group_size
        self.base_kernel = mx.zeros((2, self.kernel_size, hidden_size))
        self.kernel_projection = nn.Linear(
            hidden_size, 2 * self.kernel_size * groups, bias=False)

    def _convolve(self, x: mx.array, dynamic: mx.array, base: mx.array) -> mx.array:
        """x [B, L, H]; dynamic [B, L, taps, groups]; base [taps, H]."""
        B, L, H = x.shape
        groups = H // self.group_size
        xb = x.reshape(B, L, groups, self.group_size)
        # Same accumulation order as the reference (static term, then the
        # dynamic term, per tap) so bf16 rounding matches it bit for bit.
        out = mx.zeros_like(xb)
        for tap in range(self.kernel_size):
            shifted = xb if tap == 0 else mx.pad(
                xb, [(0, 0), (tap, 0), (0, 0), (0, 0)])[:, :L]
            kernel = base[tap].reshape(1, 1, groups, self.group_size).astype(x.dtype)
            out = out + kernel * shifted
            out = out + dynamic[:, :, tap][..., None] * shifted
        return out.reshape(B, L, H)

    def prepare(self, x: mx.array) -> tuple[mx.array, mx.array]:
        B, L, H = x.shape
        groups = H // self.group_size
        dynamic = self.kernel_projection(x).reshape(
            B, L, 2, self.kernel_size, groups)
        return (self._convolve(x, dynamic[:, :, 0], self.base_kernel[0]),
                dynamic[:, :, 1])

    def finish(self, y: mx.array, dynamic: mx.array) -> mx.array:
        return self._convolve(y, dynamic, self.base_kernel[1])


class CandidateSelector(nn.Module):
    """DFlash 2 path selector over the top-k candidates of each block position.

    An edge from predecessor token ``a`` to candidate ``b`` at position ``p``
    scores ``logits[p][b] + <pred[a] * project(h[p]), succ[b]>``. The drafter
    walks the block sequentially from the anchor, one candidate per position.
    """

    def __init__(self, hidden_size: int, vocab_size: int, rank: int, top_k: int):
        super().__init__()
        self.top_k = int(top_k)
        self.hidden_projection = nn.Linear(hidden_size, rank, bias=False)
        self.predecessor_codebook = nn.Embedding(vocab_size, rank)
        self.successor_codebook = nn.Embedding(vocab_size, rank)

    def lattice(self, hidden: mx.array, logits: mx.array, anchor: mx.array):
        """Edge scores for one block.

        ``hidden`` [L, H] and ``logits`` [L, V] are the rows that draft tokens
        (the anchor row excluded); ``anchor`` is the token the block starts
        from. Returns ``(cands [L, k], first [k], edges [L-1, k, k])``:
        ``first`` scores position 0 from the anchor; ``edges[p-1][i][j]``
        scores candidate ``j`` of position ``p`` after candidate ``i`` of
        position ``p-1``. All lazy; no host sync.
        """
        k = self.top_k
        cands = mx.argpartition(logits, -k, axis=-1)[..., -k:]
        unary = mx.take_along_axis(logits, cands, axis=-1)
        hp = self.hidden_projection(hidden).astype(unary.dtype)
        succ = self.successor_codebook(cands).astype(unary.dtype)          # [L, k, r]
        pred0 = self.predecessor_codebook(anchor.reshape(1)).astype(unary.dtype)
        first = unary[0] + succ[0] @ (pred0[0] * hp[0])                     # [k]
        if cands.shape[0] > 1:
            preds = self.predecessor_codebook(cands[:-1]).astype(unary.dtype)
            edges = (preds * hp[1:, None, :]) @ succ[1:].transpose(0, 2, 1)
            edges = unary[1:, None, :] + edges                              # [L-1, k, k]
        else:
            edges = mx.zeros((0, k, k), dtype=first.dtype)
        return cands, first, edges


def greedy_walk(cands: mx.array, first: mx.array, edges: mx.array) -> mx.array:
    """Argmax path through a lattice, as a lazy scalar chain. Returns [L]."""
    sel = mx.argmax(first)
    toks = [cands[0][sel]]
    for p in range(edges.shape[0]):
        sel = mx.argmax(edges[p][sel])
        toks.append(cands[p + 1][sel])
    return mx.stack(toks)
