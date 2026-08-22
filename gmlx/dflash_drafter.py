# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Asher Feldman
# Portions copyright (c) 2026 Prince Canuma (module tree after mlx-vlm's
# DFlashDraftModel, MIT)
"""Owned DFlash drafters: the block-diffusion base and the DFlash 2 extras.

DFlash denoises a whole block at once. Its decoder runs in two modes
(llama.cpp ``src/models/dflash.cpp``):

- inject: the target's captured residuals go through the encoder
  ``hidden_norm(fc(captures))``, and each layer projects that straight to
  K/V (k_norm + RoPE, no ``input_layernorm``, no query, no FFN) and writes it
  into the draft KV ring at the committed tokens' positions.
- draft: ``[last_bonus, MASK x (block-1)]`` is embedded with the target's
  ``tok_embd``, positioned at ``n_past + i``, and run through the layers with
  non-causal attention over the ring plus the block's own transient K/V.
  Drafts are read from rows 1..block-1, and the target's ``lm_head``
  produces the logits.

gmlx's engine splits the two modes across ``prefill_from_target_hidden`` /
``accept_verified_tokens`` (inject) and ``draft_block`` (draft).

DFlash 2 adds a grouped dynamic causal convolution around every attention
and MLP sublayer of the draft path and a candidate selector that walks a path
through the top-k candidates of each block position.

Sliding layers keep a temporal ring of ``sliding_window - 1`` context rows
(plus rollback slack) and mask the block the way the reference does: a block
row at ring offset ``q`` sees context key ``k`` only while ``q - k <
sliding_window``. A missing ``attention.causal`` key means a non-causal block,
which is llama.cpp's behavior.

Positions are relative: seeding only the trailing window of the prompt
shifts context and block queries by the same amount, which RoPE is invariant
to. Correctness never rests on the drafter regardless: the verify walk emits
the target's own tokens, so the drafter moves acceptance, never output.

Stochastic acceptance invariant: the q row a drafter stashes must be built
from the same probability array its token was drawn from. Both drafters are
lossless under that rule. The v1 drafter draws block positions independently
given the forward (which sees only the anchor and MASK tokens), so position
j's row is its proposal conditional on any accepted prefix. DFlash 2's row j
is a genuine conditional on the realized predecessor (the lattice row of the
token drawn at j-1), which is the proposal the verifier divides by.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx_vlm.models.activations import swiglu
from mlx_vlm.models.cache import BufferedRotatingKVCache, KVCache
from mlx_vlm.models.rope_utils import initialize_rope

from .drafter_protocol import DraftStash, native_block_size

_LAYER_TYPES = ("full_attention", "sliding_attention")
# Rollback slack rows the temporal ring keeps beyond its window.
_RING_SLACK = 64


@dataclass
class DFlashConfig:
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rms_norm_eps: float
    vocab_size: int
    max_position_embeddings: int
    rope_theta: float
    block_size: int
    mask_token_id: int
    target_layer_ids: List[int]
    num_target_layers: int
    rope_scaling: Optional[dict] = None
    attention_bias: bool = False
    tie_word_embeddings: bool = False
    # The checkpoint's trained diffusion block: the deepest block the drafter
    # can produce. None leaves block_size as the ceiling.
    native_block_size: Optional[int] = None
    layer_types: List[str] = field(default_factory=list)
    sliding_window: Optional[int] = None
    # None = non-causal block (llama.cpp); the converter's attention.causal
    # key sets it explicitly.
    is_causal: Optional[bool] = None
    final_logit_softcapping: Optional[float] = None
    output_multiplier: float = 1.0
    input_embedding_scale: float = 1.0
    # DFlash 2 only; selector_top_k == 0 means a v1 drafter.
    conv_kernel_size: int = 0
    conv_group_size: int = 0
    selector_rank: int = 0
    selector_top_k: int = 0

    def __post_init__(self):
        n = int(self.num_hidden_layers)
        if not self.layer_types:
            self.layer_types = ["full_attention"] * n
        self.layer_types = list(self.layer_types)
        if len(self.layer_types) != n:
            raise ValueError(
                f"layer_types has {len(self.layer_types)} entries for "
                f"{n} layers")
        unknown = set(self.layer_types) - set(_LAYER_TYPES)
        if unknown:
            raise ValueError(f"unsupported draft layer_types {sorted(unknown)}")
        if "sliding_attention" in self.layer_types and not self.sliding_window:
            raise ValueError("sliding_attention layers need sliding_window")

    @property
    def is_dflash2(self) -> bool:
        return int(self.selector_top_k) > 0


def block_attention_mask(ctx_len: int, block: int, window: Optional[int],
                         causal: bool):
    """Bool mask ``[block, ctx_len + block]`` for a draft block over a ring.

    Ring rows are indexed as time (the ring is temporal), so a block row at
    ``q = ctx_len + i`` keeps context key ``k`` while ``q - k < window``; the
    block term is all-true, or lower-triangular when causal. None when no key
    would be masked. Hand-rolled: ``BufferedRotatingKVCache.make_mask`` is
    causal and sizes its window from a planned insert of the query rows,
    which the draft block never is.
    """
    if window is None and not causal:
        return None
    if window is not None and not causal and ctx_len + block - 1 < window:
        return None
    q = ctx_len + mx.arange(block)[:, None]
    k = mx.arange(ctx_len + block)[None, :]
    in_block = k >= ctx_len
    if causal:
        in_block = in_block & (k <= q)
    in_ctx = k < ctx_len
    if window is not None:
        in_ctx = in_ctx & (q - k < window)
    return in_block | in_ctx


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


# --- draw / stash -------------------------------------------------------------

def _pick(support: Optional[mx.array], idx: mx.array) -> mx.array:
    if support is None:
        return idx
    return mx.take_along_axis(support, idx[:, None], axis=-1).reshape(-1)


def _scatter(vals: mx.array, support: Optional[mx.array], fill: float,
             vocab: int) -> mx.array:
    """Rows [n, W] on support [n, W] -> full-width rows [n, vocab]. A
    ``put_along_axis`` set: the top-k ids are distinct, and an add onto a
    ``-inf`` base would stay ``-inf``."""
    if support is None:
        return vals
    base = mx.full((vals.shape[0], vocab), fill, dtype=vals.dtype)
    return mx.put_along_axis(base, support, vals, axis=-1)


def _second_choice(rows: mx.array) -> mx.array:
    """Index of each row's second-highest entry, [n]."""
    kth = rows.shape[-1] - 2
    k2 = mx.argpartition(rows, kth=kth, axis=-1)[..., -2:]
    v2 = mx.take_along_axis(rows, k2, axis=-1)
    return mx.take_along_axis(
        k2, mx.argmin(v2, axis=-1, keepdims=True), axis=-1).reshape(-1)


def _sample_rows(sampler, rows: mx.array, support: Optional[mx.array], vocab: int):
    """Draw one token per row with the engine's sampler. Full-width rows go
    to the sampler as-is. Compact rows (a support) cannot: mlx_lm's
    make_sampler bounds-checks top_k against the row width, so the draw is
    rebuilt from the sampler's annotated params; a serve sampler guards its
    own width and is called directly; an opaque sampler gets the scattered
    row. Same precedence as speculative._stoch_supported_sampler. The
    compiled sampler must never see a compact row: its ValueError is raised
    mid-trace and leaves ``mx.random.state`` holding a tracer."""
    if support is None:
        toks = sampler(rows).reshape(-1)
        return toks, toks
    if hasattr(sampler, "_filtered") and hasattr(sampler, "temperature"):
        idx = sampler(rows).reshape(-1)
    elif getattr(sampler, "gmlx_sampling_params", None) is not None:
        from .speculative import _pq_probs

        p = sampler.gmlx_sampling_params
        q = _pq_probs(rows, p["temp"], min(int(p["top_k"]), int(rows.shape[-1])),
                      p["top_p"], min_p=p.get("min_p", 0.0))
        idx = mx.random.categorical(mx.log(q), axis=-1)
    else:
        toks = sampler(_scatter(rows, support, float("-inf"), vocab)).reshape(-1)
        return toks, mx.argmax(support == toks[:, None], axis=-1)
    return _pick(support, idx), idx


def draw_rows(rows: mx.array, support: Optional[mx.array], *, vocab: int,
              greedy: bool, sampler, stash: Optional[DraftStash]):
    """Draw one draft token per row and record the round's instrument rows.

    ``rows`` [n, W] are log-domain draft scores; ``support`` [n, W] maps row
    columns to token ids (None when W is the vocab). Greedy -> argmax;
    stochastic (``stash.q`` present) -> a draw from the sharpened proposal,
    which is also the q row stashed; otherwise the engine's sampler. Logging
    never changes the draw. Returns ``(tokens [n], column index [n])``.
    """
    stoch = stash is not None and stash.q is not None
    if greedy:
        idx = mx.argmax(rows, axis=-1)
        toks = _pick(support, idx)
    elif stoch:
        from .speculative import _STOCH_DRAFT, _pq_probs

        q = _pq_probs(rows, *_STOCH_DRAFT)
        idx = mx.random.categorical(mx.log(q), axis=-1)
        toks = _pick(support, idx)
        stash.q.extend(_scatter(q, support, 0.0, vocab))
    else:
        toks, idx = _sample_rows(sampler, rows, support, vocab)
    if stash is not None:
        if stash.pq is not None:
            stash.pq.extend(_scatter(rows, support, float("-inf"), vocab))
        if stash.top2 is not None:
            stash.top2.extend(_pick(support, _second_choice(rows)))
    return toks, idx


# --- module tree --------------------------------------------------------------

class DFlashAttention(nn.Module):
    def __init__(self, config: DFlashConfig, layer_idx: int):
        super().__init__()
        dim = config.hidden_size
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.scale = self.head_dim ** -0.5
        self.is_sliding = config.layer_types[layer_idx] == "sliding_attention"
        self.sliding_window = config.sliding_window if self.is_sliding else None
        self.is_causal = bool(config.is_causal) if config.is_causal is not None else False
        self.q_proj = nn.Linear(dim, self.n_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, dim, bias=False)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def inject(self, h_ctx: mx.array, rope, cache) -> None:
        """Write the context rows' K/V into the ring at the ring's positions."""
        B, S, _ = h_ctx.shape
        keys = self.k_norm(
            self.k_proj(h_ctx).reshape(B, S, self.n_kv_heads, -1)).transpose(0, 2, 1, 3)
        values = self.v_proj(h_ctx).reshape(B, S, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        cache.update_and_fetch(rope(keys, offset=cache.offset), values)

    def draft(self, x: mx.array, rope, cache) -> mx.array:
        """Block rows attend the ring plus their siblings. The block's own
        K/V never enters the ring."""
        B, L, _ = x.shape
        q = self.q_norm(self.q_proj(x).reshape(B, L, self.n_heads, -1)).transpose(0, 2, 1, 3)
        k = self.k_norm(self.k_proj(x).reshape(B, L, self.n_kv_heads, -1)).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        q = rope(q, offset=cache.offset)
        k = rope(k, offset=cache.offset)
        ck, cv = cache.state
        ctx_len = 0
        if ck is not None:
            ctx_len = int(ck.shape[2])
            k = mx.concatenate([ck, k], axis=2)
            v = mx.concatenate([cv, v], axis=2)
        mask = block_attention_mask(ctx_len, L, self.sliding_window, self.is_causal)
        o = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)
        return self.o_proj(o.transpose(0, 2, 1, 3).reshape(B, L, -1))


class Qwen3MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


class DFlashDecoderLayer(nn.Module):
    def __init__(self, config: DFlashConfig, layer_idx: int):
        super().__init__()
        self.self_attn = DFlashAttention(config, layer_idx)
        self.mlp = Qwen3MLP(config.hidden_size, config.intermediate_size)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def inject(self, h_ctx: mx.array, rope, cache) -> None:
        self.self_attn.inject(h_ctx, rope, cache)

    def draft(self, h: mx.array, rope, cache) -> mx.array:
        h = h + self.self_attn.draft(self.input_layernorm(h), rope, cache)
        return h + self.mlp(self.post_attention_layernorm(h))


class DFlash2DecoderLayer(DFlashDecoderLayer):
    def __init__(self, config: DFlashConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.attention_conv = GroupedDynamicConv(
            config.hidden_size, config.conv_kernel_size, config.conv_group_size)
        self.mlp_conv = GroupedDynamicConv(
            config.hidden_size, config.conv_kernel_size, config.conv_group_size)

    def draft(self, h: mx.array, rope, cache) -> mx.array:
        x, kernel = self.attention_conv.prepare(self.input_layernorm(h))
        h = h + self.attention_conv.finish(self.self_attn.draft(x, rope, cache), kernel)
        x, kernel = self.mlp_conv.prepare(self.post_attention_layernorm(h))
        return h + self.mlp_conv.finish(self.mlp(x), kernel)


# --- drafters -----------------------------------------------------------------

class DFlashDrafter(nn.Module):
    """DFlash drafter following gmlx's BatchDrafterProtocol (owned engine)."""

    layer_class = DFlashDecoderLayer
    kind_label = "dflash"
    supports_greedy_draft_argmax = True
    prefer_requested_block_size = False
    cap_at_configured_depth = True
    uses_shared_kv = False
    supports_kv_sidecar = False
    # The stock MTP round doesn't know the packed-hidden target hooks.
    requires_owned_engine = True
    supports_q_stash = True

    def __init__(self, config: DFlashConfig):
        super().__init__()
        self.config = config
        hidden = int(config.hidden_size)
        concat_dim = len(config.target_layer_ids) * hidden
        self.fc = nn.Linear(concat_dim, hidden, bias=False)
        self.hidden_norm = nn.RMSNorm(hidden, eps=config.rms_norm_eps)
        self.layers = [self.layer_class(config, i)
                       for i in range(config.num_hidden_layers)]
        self.norm = nn.RMSNorm(hidden, eps=config.rms_norm_eps)
        self.rope = initialize_rope(
            dims=config.head_dim, base=config.rope_theta, traditional=False,
            scaling_config=config.rope_scaling,
            max_position_embeddings=config.max_position_embeddings)
        self.embed_tokens = None
        self.embed_scale = 1.0
        self.lm_head = None
        self.accept_lens: List[int] = []
        self.draft_lens: List[int] = []
        self._native_block_size = (
            native_block_size(config) or int(config.block_size))
        self._hidden = hidden
        self._n_targets = len(config.target_layer_ids)
        # Only the trailing ring-capacity rows of the prompt capture are
        # usable; the engine trims the retained prefill hidden to this.
        window = int(config.sliding_window or 0)
        self.hidden_capture_limit = (window - 1) if window else None
        self._cache: List[Any] = []

    # --- lifecycle ----------------------------------------------------------

    def bind(self, target_model):
        if hasattr(target_model, "embed_tokens"):
            inner = target_model
        elif hasattr(getattr(target_model, "model", None), "embed_tokens"):
            inner = target_model.model
        elif hasattr(getattr(getattr(target_model, "language_model", None),
                             "model", None), "embed_tokens"):
            inner = target_model.language_model.model
        else:
            raise AttributeError(
                f"cannot find embed_tokens in {type(target_model).__name__}")
        self.embed_tokens = inner.embed_tokens
        self.embed_scale = (
            getattr(self.embed_tokens, "embed_scale",
                    getattr(inner, "embed_scale", 1.0))
            * float(self.config.input_embedding_scale))
        lm = getattr(target_model, "language_model", target_model)
        self.lm_head = (getattr(target_model, "lm_head", None)
                        or getattr(lm, "lm_head", None)
                        or self.embed_tokens.as_linear)
        return self

    def make_cache(self, left_padding: Optional[List[int]] = None) -> list:
        if left_padding is not None:
            raise NotImplementedError(f"{type(self).__name__} is B=1 only")
        caches = []
        for layer_type in self.config.layer_types:
            if layer_type == "sliding_attention":
                # Temporal and slack-backed: the draft path reads cache.state
                # directly and indexes its rows as time.
                caches.append(BufferedRotatingKVCache(
                    max_size=int(self.config.sliding_window) - 1,
                    buffer_size=_RING_SLACK))
            else:
                caches.append(KVCache())
        return caches

    def reset(self, target_model, left_padding: Optional[List[int]] = None) -> list:
        if left_padding is not None:
            raise NotImplementedError(
                f"{type(self).__name__} is B=1 only: batched rounds need "
                "per-row cache offsets in the inject path")
        self.bind(target_model)
        self.accept_lens = []
        self.draft_lens = []
        self._cache = self.make_cache()
        return self._cache

    def draft_eval_state(self) -> List[Any]:
        return [c.state for c in self._cache]

    def set_shared_kv(self, *args, **kwargs) -> None:
        return None

    # --- heads ---------------------------------------------------------------

    def _embed_input_tokens(self, tokens: mx.array) -> mx.array:
        return self.embed_tokens(tokens) * self.embed_scale

    def _logits(self, hidden: mx.array) -> mx.array:
        logits = self.lm_head(hidden)
        if self.config.output_multiplier != 1.0:
            logits = logits * self.config.output_multiplier
        cap = self.config.final_logit_softcapping
        if cap:
            logits = mx.tanh(logits / cap) * cap
        return logits

    # --- inject mode --------------------------------------------------------

    def _captures(self, packed: mx.array) -> mx.array:
        """Trailing ``n_targets*hidden`` of the packed target hidden."""
        want = self._hidden * (1 + self._n_targets)
        if int(packed.shape[-1]) != want:
            raise ValueError(
                f"packed hidden width {packed.shape[-1]} != {want}; the "
                "target's _dflash_capture wiring is missing")
        # Materialize: this feeds the quantized fc, whose kernel reads the
        # buffer directly and would otherwise see the packed strides.
        return mx.contiguous(packed[..., self._hidden:])

    def append_context(self, captures: mx.array) -> None:
        """Encode committed positions and inject their K/V into the ring."""
        h_ctx = self.hidden_norm(self.fc(captures))
        for layer, c in zip(self.layers, self._cache):
            layer.inject(h_ctx, self.rope, c)

    # --- draft mode ---------------------------------------------------------

    def _draft_hidden(self, tokens: mx.array) -> mx.array:
        h = self._embed_input_tokens(tokens)
        for layer, c in zip(self.layers, self._cache):
            h = layer.draft(h, self.rope, c)
        return self.norm(h)

    def _block_tokens(self, last_bonus, block_size: int, token_dtype) -> mx.array:
        if block_size > self._native_block_size:
            raise RuntimeError(
                f"{type(self).__name__} drafts at most "
                f"{self._native_block_size - 1} token(s)/round; got "
                f"block_size={block_size} - cap_at_configured_depth should "
                "have clamped it")
        if not self._cache:
            raise RuntimeError("reset(target_model) must run before draft_block()")
        mask_id = int(self.config.mask_token_id)
        bonus = (int(last_bonus) if isinstance(last_bonus, int)
                 else int(last_bonus.reshape(-1)[0].item()))
        return mx.array([[bonus] + [mask_id] * (block_size - 1)], dtype=token_dtype)

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
        """One round: ``[bonus, MASK x (block_size-1)]`` denoised in a single
        forward; drafts are rows 1..block_size-1."""
        del hidden, cache
        block = self._block_tokens(last_bonus, block_size, token_dtype)
        logits = self._logits(self._draft_hidden(block)[:, 1:])[0]
        toks, _ = draw_rows(logits, None, vocab=int(self.config.vocab_size),
                            greedy=greedy, sampler=sampler, stash=stash)
        return toks[None]

    # --- commit -------------------------------------------------------------

    def prefill_from_target_hidden(
        self,
        input_ids: mx.array,
        hidden: mx.array,
        bonus_token,
        sampler,
        token_dtype: mx.Dtype = mx.int32,
        greedy: bool = False,
        stash: Optional[DraftStash] = None,
    ) -> None:
        """Seed the ring from the trailing prompt hiddens. DFlash needs no
        draft seed: rounds start from the engine-passed bonus token."""
        del input_ids, bonus_token, sampler, token_dtype, greedy, stash
        if int(hidden.shape[1]) == 0:
            return
        limit = self.hidden_capture_limit
        self.append_context(self._captures(hidden[:, -limit:] if limit else hidden))

    def accept_verified_tokens(
        self,
        verify_hidden: mx.array,
        draft_tokens: mx.array,
        accepted: int,
        new_tokens: List[int],
        sampler,
        token_dtype: mx.Dtype = mx.int32,
        greedy: bool = False,
        stash: Optional[DraftStash] = None,
    ) -> None:
        """Inject the committed positions' captures. ``verify_hidden[:, p]`` is
        the target hidden at verify position ``p``; 0..accepted were committed
        (the accepted drafts plus the row the new bonus was sampled from),
        matching the rolled-back target 1:1."""
        del draft_tokens, new_tokens, sampler, token_dtype, greedy, stash
        self.append_context(self._captures(verify_hidden[:, : int(accepted) + 1]))


class DFlash2Drafter(DFlashDrafter):
    """DFlash 2: conv-wrapped draft layers and a selector-walked block."""

    layer_class = DFlash2DecoderLayer
    kind_label = "dflash2"

    def __init__(self, config: DFlashConfig):
        if not config.is_dflash2:
            raise ValueError("DFlash2Drafter needs selector_top_k > 0")
        if config.conv_kernel_size <= 0 or config.conv_group_size <= 0:
            raise ValueError("DFlash2Drafter needs conv_kernel_size and conv_group_size")
        super().__init__(config)
        self.candidate_selector = CandidateSelector(
            config.hidden_size, config.vocab_size, config.selector_rank,
            config.selector_top_k)

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
        del hidden, cache
        block = self._block_tokens(last_bonus, block_size, token_dtype)
        h = self._draft_hidden(block)[:, 1:]
        logits = self._logits(h)[0]
        cands, first, edges = self.candidate_selector.lattice(h[0], logits, block[0, 0])
        if greedy and stash is None:
            return greedy_walk(cands, first, edges)[None]
        # Sequential: each row is the lattice row of the realized predecessor.
        # Draws happen on the compact k-wide row; only stash rows are widened.
        vocab = int(self.config.vocab_size)
        toks = []
        row, support = first[None], cands[0][None]
        for p in range(cands.shape[0]):
            tok, idx = draw_rows(row, support, vocab=vocab, greedy=greedy,
                                 sampler=sampler, stash=stash)
            toks.append(tok[0])
            if p + 1 < cands.shape[0]:
                row, support = edges[p][idx[0]][None], cands[p + 1][None]
        return mx.stack(toks)[None]


# --- target side --------------------------------------------------------------

class DFlashCaptureHooks:
    """Packed-hidden capture for owned qwen3.5 LanguageModels.

    While armed, every hidden the engine sees is ``[trunk | cap ...]``: the
    final normed hidden followed by the raw outputs of the captured layers,
    in layer order. The stock ``__call__`` (mrope position resolution, head)
    stays inherited; this wrapper only installs the owned forward's capture
    sink around it and repacks ``hidden_states[-1]``. The logits hooks slice
    the trunk back out, materialized, before the quantized head.

    ``speculative_verify_hidden`` and ``speculative_verify_logits`` both go
    through ``self(...)`` with ``return_hidden`` set, so both return the
    packed hidden (with correct logits): the contract the drafter wants, and
    independent of which hook ``_mtp_verify_target`` tries first.
    """

    _dflash_capture: Optional[tuple] = None

    def set_dflash_capture(self, layer_ids) -> None:
        # Around nn.Module.__setattr__, which would route a tuple into the
        # parameter tree.
        object.__setattr__(self, "_dflash_capture",
                           tuple(int(i) for i in layer_ids) or None)

    def _dflash_trunk(self, hidden: mx.array) -> mx.array:
        if self._dflash_capture is None:
            return hidden
        # The trunk lead is a strided view of the packed hidden, and the head
        # is a quantized kernel that reads the buffer directly; slicing
        # lazily hands it the packed strides. Materialize first.
        return mx.contiguous(hidden[..., : int(self.args.hidden_size)])

    def __call__(self, inputs, inputs_embeds=None, mask=None, cache=None, **kwargs):
        ids = self._dflash_capture
        if ids is None or not kwargs.get("return_hidden"):
            return super().__call__(inputs, inputs_embeds=inputs_embeds,
                                    mask=mask, cache=cache, **kwargs)
        sink: list = []
        object.__setattr__(self.model, "_dflash_sink", sink)
        object.__setattr__(self.model, "_dflash_layers", ids)
        try:
            out = super().__call__(inputs, inputs_embeds=inputs_embeds,
                                   mask=mask, cache=cache, **kwargs)
        finally:
            object.__setattr__(self.model, "_dflash_sink", None)
        final = out.hidden_states[-1]
        if len(sink) != len(ids):
            if int(final.shape[1]) == 0:
                return out
            raise RuntimeError(
                f"dflash capture collected {len(sink)} of {len(ids)} layers")
        out.hidden_states = [*out.hidden_states[:-1],
                             mx.concatenate([final, *sink], axis=-1)]
        return out

    def speculative_logits_from_hidden(self, hidden: mx.array) -> mx.array:
        return super().speculative_logits_from_hidden(self._dflash_trunk(hidden))

    def speculative_argmax_from_hidden(self, hidden: mx.array):
        return super().speculative_argmax_from_hidden(self._dflash_trunk(hidden))
