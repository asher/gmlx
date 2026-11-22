# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Asher Feldman
"""The Inkling 2 DSpark drafter (``dflash`` GGUFs with ``dspark_hidden_out``).

Reference: llama.cpp ``src/models/dspark.cpp`` (``graph_inkling2``,
``build_inkling2_head``) and ``common/speculative.cpp`` (the Inkling 2
branch of the DSpark draft), ggml-org PR 32218.

Context pass. The target's taps (layer-visit inputs, the last one the
normed final hidden state) are mixed per draft layer by
``softmax(dspark_fuse[layer])``, normed by the layer's ``dflare_norm`` and
projected to K/V by the layer's own context projections (K normed per head
and rotated). The last row of ``dspark_fuse`` mixes the recurrent head's
context feature, normed by ``dspark_norm``. The draft reads the feature
of the last committed token, the one before the anchor.

Draft pass. Always the trained block: the anchor (the target's normed
token embedding) and ``block - 1`` copies of ``mask_embd``, non-causal over
the ring and the block. A sliding layer's block rows all see the anchor
row's window. The base logits come from the target's head. Rows 1 on are
then corrected by a gated recurrent head over the previous pick's
``markov_w1`` row, the row's hidden state and the context feature; the
correction goes through the target's head too, and the sum is softcapped.
Each pick is the first maximum over the unpadded vocabulary.
"""

from __future__ import annotations

from typing import Any, List, Optional

import mlx.core as mx
import mlx.nn as nn

from gmlx.cache import kv_append
from gmlx.cache.kv_cache import ArraysCache, KVCache
from gmlx.spec.dflash_drafter import (
    DFlashConfig,
    DFlashDrafter,
    Qwen3MLP,
    draw_rows,
)
from gmlx.spec.drafter_protocol import DraftStash


def _rms(x: mx.array, weight: mx.array, eps: float) -> mx.array:
    return mx.fast.rms_norm(x, weight, eps)


def _block_in_place(cache, k: mx.array, v: mx.array):
    """The block rows written into ``cache``'s buffer past its offset, with
    views of the context and block rows, or None where they cannot go in
    place. The next append starts at the offset and writes over them; it
    waits for the draft that reads them (``_wait_for_draft``)."""
    if not kv_append.INPLACE or type(cache) is not KVCache:
        return None
    ks, vs = cache.keys, cache.values
    own = getattr(cache, "_kv_own", None)
    off = cache.offset
    end = off + k.shape[2]
    if (own is None or own[0] != id(cache) or own[1] is not ks
            or own[2] is not vs or getattr(cache, "_kv_hw", None) != off
            or end > ks.shape[2]):
        return None
    kk = kv_append._kq.kv_append(ks, k.astype(ks.dtype), off)
    vv = kv_append._kq.kv_append(vs, v.astype(vs.dtype), off)
    return kk[..., :end, :], vv[..., :end, :]


def _wait_for_draft(cache) -> None:
    """Order the next append after the draft that read its block rows in
    the cache buffer, and keep that append in place."""
    read = cache.__dict__.pop("_draft_read", None)
    if read is None:
        return
    own = getattr(cache, "_kv_own", None)
    held = (own is not None and own[1] is cache.keys
            and own[2] is cache.values)
    cache.keys, cache.values = mx.depends([cache.keys, cache.values], read)
    if held:
        cache._kv_own = (own[0], cache.keys, cache.values)


def first_argmax(row: mx.array) -> mx.array:
    """Index of the first maximum of each row (last axis)."""
    n = row.shape[-1]
    top = mx.max(row, axis=-1, keepdims=True)
    rank = mx.where(row == top, n - mx.arange(n), 0)
    return mx.argmax(rank, axis=-1)


class Inkling2DSparkAttention(nn.Module):
    def __init__(self, config: DFlashConfig, layer_idx: int):
        super().__init__()
        dim = config.hidden_size
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.scale = self.head_dim ** -0.5
        self.is_sliding = config.layer_types[layer_idx] == "sliding_attention"
        # Context rows the anchor row's window holds.
        self.ring_rows = (int(config.sliding_window) - 1 if self.is_sliding
                          else None)
        hd = self.head_dim
        self.q_proj = nn.Linear(dim, self.n_heads * hd, bias=False)
        self.k_proj = nn.Linear(dim, self.n_kv_heads * hd, bias=False)
        self.v_proj = nn.Linear(dim, self.n_kv_heads * hd, bias=False)
        self.o_proj = nn.Linear(self.n_heads * hd, dim, bias=False)
        self.q_norm = nn.RMSNorm(hd, eps=config.rms_norm_eps)
        self.k_norm = nn.RMSNorm(hd, eps=config.rms_norm_eps)
        self.k_ctx_proj = nn.Linear(dim, self.n_kv_heads * hd, bias=False)
        self.v_ctx_proj = nn.Linear(dim, self.n_kv_heads * hd, bias=False)
        self.k_ctx_norm = nn.RMSNorm(hd, eps=config.rms_norm_eps)

    def inject(self, context: mx.array, rope, cache) -> None:
        # Every row goes in, so the cache offset counts the whole context.
        B, S, _ = context.shape
        keys = self.k_ctx_norm(self.k_ctx_proj(context).reshape(
            B, S, self.n_kv_heads, -1)).transpose(0, 2, 1, 3)
        values = self.v_ctx_proj(context).reshape(
            B, S, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        _wait_for_draft(cache)
        cache.update_and_fetch(rope(keys, offset=cache.offset), values)

    def draft(self, x: mx.array, rope, cache) -> mx.array:
        B, L, _ = x.shape
        q = self.q_norm(self.q_proj(x).reshape(
            B, L, self.n_heads, -1)).transpose(0, 2, 1, 3)
        k = self.k_norm(self.k_proj(x).reshape(
            B, L, self.n_kv_heads, -1)).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, L, self.n_kv_heads, -1).transpose(
            0, 2, 1, 3)
        q = rope(q, offset=cache.offset)
        k = rope(k, offset=cache.offset)
        ck, cv = cache.state
        rows = None
        if ck is not None:
            if self.ring_rows is None:
                # The full layer's context is not copied for each draft.
                rows = _block_in_place(cache, k, v)
            elif int(ck.shape[2]) > self.ring_rows:
                ck = ck[:, :, -self.ring_rows:]
                cv = cv[:, :, -self.ring_rows:]
            if rows is None:
                k = mx.concatenate([ck, k], axis=2)
                v = mx.concatenate([cv, v], axis=2)
            else:
                k, v = rows
        o = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)
        if rows is not None:
            cache._draft_read = o
        return self.o_proj(o.transpose(0, 2, 1, 3).reshape(B, L, -1))


class Inkling2DSparkLayer(nn.Module):
    def __init__(self, config: DFlashConfig, layer_idx: int):
        super().__init__()
        dim = config.hidden_size
        self.self_attn = Inkling2DSparkAttention(config, layer_idx)
        self.mlp = Qwen3MLP(dim, config.intermediate_size)
        self.input_layernorm = nn.RMSNorm(dim, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(dim, eps=config.rms_norm_eps)
        self.dflare_norm = nn.RMSNorm(dim, eps=config.rms_norm_eps)

    def inject(self, context: mx.array, rope, cache) -> None:
        self.self_attn.inject(context, rope, cache)

    def draft(self, h: mx.array, rope, cache) -> mx.array:
        h = h + self.self_attn.draft(self.input_layernorm(h), rope, cache)
        return h + self.mlp(self.post_attention_layernorm(h))


class Inkling2DSparkDrafter(DFlashDrafter):
    """The Inkling 2 DSpark drafter on gmlx's DFlash drafter protocol."""

    layer_class = Inkling2DSparkLayer
    kind_label = "dspark"

    def __init__(self, config: DFlashConfig):
        if not config.is_dspark:
            raise ValueError("Inkling2DSparkDrafter needs markov_rank > 0")
        super().__init__(config)
        # No shared encoder: each layer mixes the taps itself.
        for name in ("fc", "hidden_norm"):
            self.pop(name, None)
        hidden = int(config.hidden_size)
        rank = int(config.markov_rank)
        n_targets = len(config.target_layer_ids)
        self.dspark_fuse = mx.zeros((config.num_hidden_layers + 1, n_targets))
        self.dspark_norm = nn.RMSNorm(hidden, eps=config.rms_norm_eps)
        self.mask_embd = mx.zeros((hidden,))
        # The markov table and the target's head cover the padded vocabulary.
        self.markov_w1 = nn.Embedding(
            int(config.padded_vocab_size or config.vocab_size), rank)
        self.joint_proj = nn.Linear(2 * rank + 2 * hidden, 3 * rank, bias=True)
        self.hidden_out = nn.Linear(rank, hidden, bias=True)
        self.embed_norm = None
        # The target's head folded into hidden_out (_folded_head). Kept
        # out of the module tree: it is derived, never loaded or saved.
        object.__setattr__(self, "_fold", None)
        if "full_attention" in config.layer_types:
            # A full layer reads every committed row, so the prompt capture
            # has no row limit; prefill streams it in chunks (seed_chunk).
            self.hidden_capture_limit = None

    # --- lifecycle ----------------------------------------------------------

    def bind(self, target_model):
        lm = getattr(target_model, "language_model", target_model)
        inner = getattr(lm, "model", None)
        if inner is None or not hasattr(inner, "embed_norm"):
            raise AttributeError(
                f"{type(target_model).__name__} is not an Inkling 2 model")
        super().bind(target_model)
        object.__setattr__(self, "_fold", None)
        self.embed_norm = inner.embed_norm
        return self

    def make_cache(self, left_padding: Optional[List[int]] = None) -> list:
        # The layers' rings, then the context feature of the last
        # committed token.
        return super().make_cache(left_padding) + [ArraysCache(size=1)]

    # --- context ------------------------------------------------------------

    def _mix(self, features: mx.array, row: int, weight: mx.array) -> mx.array:
        """features [B, S, T * H] -> rms(sum_t softmax(fuse[row])_t f_t)."""
        B, S, _ = features.shape
        f = features.reshape(B, S, len(self.config.target_layer_ids), -1)
        w = mx.softmax(self.dspark_fuse[row].astype(mx.float32))
        x = (f.astype(mx.float32) * w[:, None]).sum(axis=2)
        return _rms(x, weight.astype(mx.float32),
                    self.config.rms_norm_eps).astype(features.dtype)

    def append_context(self, captures: mx.array, cache=None) -> None:
        caches = self._cache if cache is None else cache
        for i, (layer, c) in enumerate(zip(self.layers, caches)):
            layer.inject(self._mix(captures, i, layer.dflare_norm.weight),
                         self.rope, c)
        last = captures[:, -1:]
        caches[-1][0] = self._mix(last, len(self.layers),
                                  self.dspark_norm.weight)

    def seed_chunk(self, shifted_tokens, hidden: mx.array, cache) -> None:
        """Inject one prefill chunk's taps into a request's cache."""
        del shifted_tokens
        if int(hidden.shape[1]):
            self.append_context(self._captures(hidden), cache)

    def restore_kv(self, caches: list) -> None:
        """Adopt a cache that prefill seeded."""
        self._cache = list(caches)

    # --- draft --------------------------------------------------------------

    def _block_tokens(self, last_bonus, block_size: int, token_dtype) -> mx.array:
        if block_size > self._native_block_size:
            raise RuntimeError(
                f"{type(self).__name__} drafts at most "
                f"{self._native_block_size - 1} token(s)/round; got "
                f"block_size={block_size}")
        if not self._cache:
            raise RuntimeError(
                "reset(target_model) must run before draft_block()")
        bonus = (int(last_bonus) if isinstance(last_bonus, int)
                 else int(last_bonus.reshape(-1)[0].item()))
        return mx.array([[bonus]], dtype=token_dtype)

    def _draft_hidden(self, anchor: mx.array) -> mx.array:
        """The trained block's normed rows [1, block, H] for ``anchor``."""
        h0 = self.embed_norm(self.embed_tokens(anchor))
        rows = self._native_block_size
        masks = mx.broadcast_to(self.mask_embd.astype(h0.dtype),
                                (1, rows - 1, h0.shape[-1]))
        h = mx.concatenate([h0, masks], axis=1)
        for layer, c in zip(self.layers, self._cache):
            h = layer.draft(h, self.rope, c)
        return self.norm(h)

    def _head(self, hidden: mx.array) -> mx.array:
        return self.lm_head(hidden).astype(mx.float32)

    def _folded_head(self) -> tuple[mx.array, mx.array]:
        """The head of ``hidden_out``'s output, as one rank-wide map:
        ``lm_head(hidden_out(z)) = z @ W + b`` with W = lm_head of each
        column of hidden_out [rank, vocab] and b = lm_head of its bias.
        Built once per bind, so a draft row reads rank columns of the
        head instead of all hidden_size."""
        if self._fold is None:
            w = mx.contiguous(self.hidden_out.weight.T)
            fold_w = self.lm_head(w)
            fold_b = self.lm_head(self.hidden_out.bias[None])[0]
            mx.eval(fold_w, fold_b)
            object.__setattr__(self, "_fold",
                               (fold_w, fold_b.astype(mx.float32)))
        return self._fold

    def chain(self, hidden: mx.array, anchor: mx.array, n: int,
              pick=first_argmax) -> tuple[mx.array, mx.array]:
        """Softcapped logits [n, vocab] of rows 1..n of one block and the
        tokens [n] that ``pick`` draws from them. Each row is corrected by
        the recurrent head on the token drawn from the row before."""
        vocab = int(self.config.vocab_size)
        cap = float(self.config.final_logit_softcapping or 0.0)
        context = self._cache[-1][0]
        if context is None:
            raise RuntimeError("the drafter has no committed context")
        context = context.reshape(1, -1).astype(mx.float32)
        base = self._head(hidden[0, 1:n + 1])
        fold_w, fold_b = self._folded_head()
        state = mx.zeros((1, int(self.config.markov_rank)), dtype=mx.float32)
        prev = anchor.reshape(1)
        rows, toks = [], []
        for i in range(n):
            joint = mx.concatenate(
                [state, self.markov_w1(prev).astype(mx.float32),
                 hidden[0, i + 1:i + 2].astype(mx.float32), context], axis=-1)
            gate, cand, latent = mx.split(
                self.joint_proj(joint).astype(mx.float32), 3, axis=-1)
            gate, cand = mx.sigmoid(gate), mx.tanh(cand)
            state = cand + gate * (state - cand)
            logits = (base[i:i + 1]
                      + (mx.tanh(latent).astype(fold_w.dtype) @ fold_w
                         ).astype(mx.float32) + fold_b)
            if cap:
                logits = mx.tanh(logits / cap) * cap
            logits = logits[:, :vocab]
            rows.append(logits[0])
            prev = pick(logits).reshape(1)
            toks.append(prev)
        return mx.stack(rows), mx.concatenate(toks)

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
        """One round: the trained block through the layers, then the
        recurrent head over the first ``block_size - 1`` draft rows. Each
        row's draft is the token the next row is conditioned on."""
        del hidden, cache
        anchor = self._block_tokens(last_bonus, block_size, token_dtype)
        h = self._draft_hidden(anchor)
        # The layers run while the head and the verify graph build.
        mx.async_eval(h)
        vocab = int(self.config.vocab_size)
        n = int(block_size) - 1
        if greedy:
            rows, toks = self.chain(h, anchor[0, 0], n)
            if stash is not None:
                # Records the instrument rows; the picks stay first-max.
                draw_rows(rows, None, vocab=vocab, greedy=True,
                          sampler=sampler, stash=stash)
        else:
            def draw(row):
                return draw_rows(row, None, vocab=vocab, greedy=False,
                                 sampler=sampler, stash=stash)[0]

            _, toks = self.chain(h, anchor[0, 0], n, pick=draw)
        return toks[None].astype(token_dtype)

    # --- parked row contexts ------------------------------------------------

    def row_context_state(self, ctx: list) -> List[Any]:
        return [c.state for c in ctx]
