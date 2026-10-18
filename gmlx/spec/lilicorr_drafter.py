# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Asher Feldman
"""LiLiCorr drafter: a DFlash backbone with a lattice head that rescores
the top-k candidates of every draft slot (NVIDIA Nemotron-3.5 drafters).

The backbone is DFlash 2's conv-wrapped layers with per-head attention
sinks and no candidate selector. The head follows llama.cpp
``src/models/dflash.cpp`` ``build_lilicorr_scores`` and the walk in
``common/speculative.cpp``:

- Each of the ``block - 1`` draft slots keeps its top-k candidates in
  descending logit order. A node per (slot, candidate) sums the projected
  token embedding, the projected draft hidden of its slot, an MLP over five
  features (log-prob, prob, log-prob minus the slot's best, rank / (k - 1),
  first-rank flag) and slot and rank embeddings.
- A small pre-norm transformer runs over all nodes with a learned per-head
  attention bias, then factor heads give unit ``out`` and ``in`` vectors per
  node and an ``anchor_out`` vector from the encoder row of the last
  committed position.
- Slot 0 scores ``scale * <in, anchor_out>``; slot ``s`` scores candidate
  ``n`` after predecessor ``p`` as ``scale * <in_s[n], out_{s-1}[p]>``.
  Drafts are the greedy walk through that lattice, at any temperature.

The head only runs on a full block, so a round always denoises the trained
block and walks as many slots as the engine asked for.
"""

from __future__ import annotations

from typing import Any, List, Optional

import mlx.core as mx
import mlx.nn as nn

from .dflash_drafter import (
    DFlash2DecoderLayer,
    DFlashConfig,
    DFlashDrafter,
    greedy_path,
    path_estimate,
)
from .drafter_protocol import DraftStash

_FEATURES = 5
_FEATURE_NORM_EPS = 1e-5


class _Ring(list):
    """A drafter's per-layer KV ring plus the encoder row of its last
    committed position, which anchors the lattice."""

    anchor: Optional[mx.array] = None


def _l2_normalize(x: mx.array, eps: float) -> mx.array:
    # ggml_l2_norm: x / max(||x||, eps).
    norm = mx.sqrt((x * x).sum(axis=-1, keepdims=True))
    return x / mx.maximum(norm, eps)


class LiLiCorrLayer(nn.Module):
    def __init__(self, hidden: int, heads: int, mlp: int, eps: float):
        super().__init__()
        self.heads = int(heads)
        self.eps = float(eps)
        self.attn_norm = mx.ones((hidden,))
        self.in_proj = nn.Linear(hidden, 3 * hidden, bias=True)
        self.out_proj = nn.Linear(hidden, hidden, bias=True)
        self.mlp_norm = mx.ones((hidden,))
        self.mlp_up = nn.Linear(hidden, mlp, bias=True)
        self.mlp_down = nn.Linear(mlp, hidden, bias=True)

    def __call__(self, x: mx.array, bias: mx.array) -> mx.array:
        """x [N, H] nodes; bias [heads, N, N] added to the scores."""
        n, hidden = x.shape
        hd = hidden // self.heads
        a = mx.fast.rms_norm(x, self.attn_norm, self.eps)
        q, k, v = mx.split(self.in_proj(a).astype(mx.float32), 3, axis=-1)
        q, k, v = (t.reshape(n, self.heads, hd).transpose(1, 0, 2)
                   for t in (q, k, v))
        scores = (q @ k.transpose(0, 2, 1)) * (hd ** -0.5) + bias
        o = mx.softmax(scores, axis=-1) @ v
        x = x + self.out_proj(o.transpose(1, 0, 2).reshape(n, hidden))
        m = mx.fast.rms_norm(x, self.mlp_norm, self.eps)
        return x + self.mlp_down(nn.silu(self.mlp_up(m)))


class LiLiCorrHead(nn.Module):
    """The lattice scorer over ``slots`` x ``top_k`` candidate nodes."""

    def __init__(self, config: DFlashConfig, embed_dim: int):
        super().__init__()
        hidden = int(config.lilicorr_hidden_size)
        k = int(config.lilicorr_top_k)
        slots = int(config.native_block_size or config.block_size) - 1
        factor = int(config.lilicorr_factor_dim)
        self.top_k = k
        self.eps = float(config.rms_norm_eps)
        self.vector_eps = float(config.lilicorr_vector_eps)
        self.logit_scale = float(config.lilicorr_logit_scale)
        if hidden != embed_dim:
            self.token_proj = nn.Linear(embed_dim, hidden, bias=True)
        self.pass_hidden_proj = nn.Linear(embed_dim, hidden, bias=True)
        self.context_proj = nn.Linear(embed_dim, hidden, bias=True)
        self.feature_norm = nn.LayerNorm(_FEATURES, eps=_FEATURE_NORM_EPS)
        self.feature_up = nn.Linear(_FEATURES, hidden, bias=True)
        self.feature_down = nn.Linear(hidden, hidden, bias=True)
        self.slot_embedding = mx.zeros((slots, hidden))
        self.rank_embedding = mx.zeros((k, hidden))
        self.attention_bias = mx.zeros(
            (config.lilicorr_heads, slots * k, slots * k))
        self.layers = [
            LiLiCorrLayer(hidden, config.lilicorr_heads,
                          config.lilicorr_mlp_size, config.rms_norm_eps)
            for _ in range(int(config.lilicorr_layers))]
        self.output_norm = mx.ones((hidden,))
        self.anchor_norm = mx.ones((hidden,))
        self.factor_input_proj = nn.Linear(3 * hidden, hidden, bias=True)
        self.out_head = nn.Linear(hidden, factor, bias=True)
        self.in_head = nn.Linear(hidden, factor, bias=True)
        self.anchor_out_head = nn.Linear(hidden, factor, bias=True)

    def candidates(self, logits: mx.array):
        """Top-k ids per slot in descending logit order, with their
        log-probabilities under the full-vocab softmax. ``logits`` [S, V]."""
        lg = logits.astype(mx.float32)
        k = self.top_k
        cands = mx.argpartition(-lg, kth=k - 1, axis=-1)[:, :k]
        vals = mx.take_along_axis(lg, cands, axis=-1)
        order = mx.argsort(-vals, axis=-1)
        cands = mx.take_along_axis(cands, order, axis=-1)
        lp = mx.take_along_axis(lg, cands, axis=-1) - mx.logsumexp(
            lg, axis=-1, keepdims=True)
        return cands, lp

    def lattice(self, passes: mx.array, logits: mx.array, embed,
                anchor: mx.array):
        """Lattice of one block.

        ``passes`` [S, D] are the normed draft hiddens of the slots,
        ``logits`` [S, V] their head outputs, ``embed`` the target's token
        embedding and ``anchor`` [D] the encoder row of the last committed
        position. Returns ``(cands [S, k], first [k], edges [S-1, k, k])``:
        ``edges[s-1][p][n]`` scores candidate ``n`` of slot ``s`` after
        candidate ``p`` of slot ``s-1``. All lazy.
        """
        f32 = mx.float32
        slots, k = int(logits.shape[0]), self.top_k
        cands, lp = self.candidates(logits)
        rank = mx.arange(k, dtype=f32)
        rank_feat = rank / (k - 1) if k > 1 else mx.zeros_like(rank)
        first_feat = (rank == 0).astype(f32)
        feat = mx.stack([
            lp, mx.exp(lp), lp - lp[:, :1],
            mx.broadcast_to(rank_feat, lp.shape),
            mx.broadcast_to(first_feat, lp.shape)], axis=-1)        # [S, k, 5]
        feat = self.feature_down(nn.silu(self.feature_up(
            self.feature_norm(feat)))).astype(f32)

        tokens = embed(cands)                                         # [S, k, D]
        if "token_proj" in self:
            tokens = self.token_proj(tokens)
        passes = self.pass_hidden_proj(passes).astype(f32)[:, None, :]
        pos = self.slot_embedding[:, None, :] + self.rank_embedding[None]
        x = (tokens.astype(f32) + passes + feat + pos.astype(f32))
        x = x.reshape(slots * k, -1)
        bias = self.attention_bias.astype(f32)
        for layer in self.layers:
            x = layer(x, bias).astype(f32)
        x = mx.fast.rms_norm(x, self.output_norm, self.eps)

        a = self.context_proj(anchor.reshape(1, -1)).astype(f32)
        a = mx.fast.rms_norm(a, self.anchor_norm, self.eps)
        ab = mx.broadcast_to(a, x.shape)
        factors = nn.silu(self.factor_input_proj(
            mx.concatenate([x, ab, x * ab], axis=-1)).astype(f32))
        eps = self.vector_eps
        out = _l2_normalize(self.out_head(factors).astype(f32), eps)
        inn = _l2_normalize(self.in_head(factors).astype(f32), eps)
        a_out = _l2_normalize(self.anchor_out_head(a).astype(f32), eps)[0]
        out = out.reshape(slots, k, -1)
        inn = inn.reshape(slots, k, -1)
        first = self.logit_scale * (inn[0] @ a_out)
        if slots > 1:
            edges = self.logit_scale * (out[:-1] @ inn[1:].transpose(0, 2, 1))
        else:
            edges = mx.zeros((0, k, k), dtype=f32)
        return cands, first, edges


class LiLiCorrDrafter(DFlashDrafter):
    """DFlash backbone (conv layers, sinks) + the LiLiCorr lattice head."""

    layer_class = DFlash2DecoderLayer
    kind_label = "lilicorr"
    # The walk is greedy at every temperature (llama.cpp's reading), so the
    # verify stays exact-match.
    stochastic_draft = False
    supports_q_stash = False

    def __init__(self, config: DFlashConfig):
        if config.conv_kernel_size <= 0 or config.conv_group_size <= 0:
            raise ValueError("LiLiCorrDrafter needs conv_kernel_size and conv_group_size")
        if not config.lilicorr_layers:
            raise ValueError("LiLiCorrDrafter needs the lilicorr head metadata")
        super().__init__(config)
        self.lilicorr = LiLiCorrHead(config, int(config.hidden_size))
        object.__setattr__(self, "_round_path", None)

    @property
    def round_estimate(self) -> Optional[mx.array]:
        path = self._round_path
        return None if path is None else path_estimate(*path)

    def make_cache(self, left_padding: Optional[List[int]] = None) -> list:
        return _Ring(super().make_cache(left_padding))

    def append_context(self, captures: mx.array, cache=None) -> None:
        ring = self._cache if cache is None else cache
        h_ctx = self.hidden_norm(self.fc(captures))
        for layer, c in zip(self.layers, ring):
            layer.inject(h_ctx, self.rope, c)
        ring.anchor = h_ctx[0, -1]

    def draft_block(
        self,
        last_bonus,
        hidden: mx.array,
        cache,
        block_size: int,
        sampler,
        token_dtype: mx.Dtype = mx.int32,
        greedy: bool = False,
        stash: Optional[DraftStash] = None,
    ) -> mx.array:
        """One round: the full trained block through the backbone, the
        lattice over its slots, and the greedy walk over the first
        ``block_size - 1`` of them."""
        del hidden, cache, sampler, greedy, stash
        native = self._native_block_size
        n = min(int(block_size), native) - 1
        block = self._block_tokens(last_bonus, native, token_dtype)
        h = self._draft_hidden(block)[0, 1:]
        logits = self._logits(h)
        anchor: Any = getattr(self._cache, "anchor", None)
        if anchor is None:
            anchor = mx.zeros((int(self.config.hidden_size),), dtype=h.dtype)
        cands, first, edges = self.lilicorr.lattice(
            h, logits, self.embed_tokens, anchor)
        edges = edges[: n - 1]
        sel = greedy_path(first, edges)
        object.__setattr__(self, "_round_path", (first, edges, sel))
        return mx.take_along_axis(cands[:n], sel[:, None], axis=-1).reshape(1, -1)
