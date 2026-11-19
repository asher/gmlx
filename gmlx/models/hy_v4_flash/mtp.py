# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Asher Feldman
"""Hy4-flash MTP: the speculative target and the in-file draft block.

The GGUF carries its draft block as block ``num_hidden_layers`` (42): the
four ``nextn.*`` tensors and one sparse layer (its own pooled indexer, the
latent MoE) with a plain residual where the trunk has hyper-connections.
``wire.remap_mtp_arrays`` maps the extras onto the Qwen drafter tree and
the layer onto ``layers.0``. The forward follows the hy_v4_flash llama.cpp
patch (``graph_mtp``):

    x = eh_proj(concat[enorm(embed(ids)), hnorm(h)])
    x = x + attn(attn_norm(x))
    x = x + ffn_post_norm(moe(ffn_norm(x)))
    h_out = shared_head_norm(x); logits = lm_head(h_out)

``h`` is the trunk's output after the stream collapse and the final norm,
the row its own head reads, and a second draft step is fed ``h_out``.

A rejected draft is rewound in three parts: the sparse layers' keys and
pools trim, a sliding-window layer undoes its last update from the armed
stash (``kv_cache.set_undo_armed``), and the engram token history is
rebuilt from the one the verify forward started with.

The draft block changes speed, never output: the verify forward emits the
target's own tokens.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, List, Optional

import mlx.core as mx
import mlx.nn as nn

from . import model as hy
from . import positions as rope_positions
from gmlx.cache.kv_cache import (
    MTP_UNDO_WIDTH,
    ArraysCache,
    CacheList,
    set_undo_armed,
)
from gmlx.models.deepseek_v4.cache import DS4PoolingCache
from gmlx.spec.drafter_protocol import native_block_size
from gmlx.spec.mtp_drafter import QwenMTPDrafter


@dataclass
class HyV4FlashMTPConfig:
    """``text_config`` is the target's ModelArgs. ``block_size`` is the
    block total (drafts and the bonus token)."""

    text_config: Any
    block_size: int = 2


@dataclass
class _SpecOutput:
    """What the owned engine reads from a ``return_hidden`` call."""

    logits: mx.array
    hidden_states: List[mx.array]
    shared_kv_states: dict = field(default_factory=dict)
    gdn_states: Optional[list] = None


def _trim_leaves(prompt_cache: List[Any]) -> List[Any]:
    """The cache members a rollback trims: every layer's keys and pools.
    The engram token history is rebuilt, not trimmed."""
    leaves: List[Any] = []
    for entry in prompt_cache:
        for leaf in getattr(entry, "caches", None) or (entry,):
            if leaf is not None and not isinstance(leaf, ArraysCache):
                leaves.append(leaf)
    return leaves


def draft_layer_args(args):
    """``args`` with the draft block as one more layer: a sparse layer
    with an indexer, at the head count the file states for it."""
    n = int(args.num_hidden_layers)
    heads = int(args.mtp_num_attention_heads or 0)
    if not heads:
        sparse = [h for h, swa in zip(args.num_attention_heads,
                                      args.sliding_window_pattern)
                  if not swa]
        heads = int(sparse[-1]) if sparse else int(
            args.num_attention_heads[-1])
    return replace(
        args,
        num_hidden_layers=n + 1,
        num_attention_heads=list(args.num_attention_heads[:n]) + [heads],
        sliding_window_pattern=(
            list(args.sliding_window_pattern[:n]) + [False]),
        index_types=list(args.index_types[:n]) + [bool(args.index_topk)],
    )


class HyV4FlashSpecHooks:
    """The ``speculative_*`` hooks the owned MTP engine probes on its
    target. ``hidden_states`` is the trunk's output after its final norm,
    which the draft block's ``hnorm`` reads."""

    # Widest verify the rollback can undo: the pooling caches and the
    # rotating stash keep an undo log only up to this width.
    spec_rollback_width = MTP_UNDO_WIDTH

    def _head(self, out: mx.array) -> mx.array:
        if self.lm_head is None:
            return self.model.embed_tokens.as_linear(out)
        return self.lm_head(out)

    def chunked_prefill_policy(self, **kwargs):
        # The draft block is teacher-forced from each chunk's hiddens.
        return True

    def speculative_logits_from_hidden(self, hidden: mx.array) -> mx.array:
        return self._head(hidden)

    def speculative_argmax_from_hidden(self, hidden: mx.array) -> mx.array:
        return mx.argmax(self._head(hidden), axis=-1)

    def speculative_verify_hidden(self, verify_input: mx.array, prompt_cache):
        """The verify forward, with what a rollback needs: the rotating
        undo stash armed, and the engram history and the image position
        walk it started from."""
        slot = self.model._history_slot(prompt_cache)
        sink = [(slot[0] if slot is not None else None, verify_input,
                 rope_positions.snapshot(slot))]
        set_undo_armed(True)
        try:
            out = self.model(verify_input, prompt_cache)
        finally:
            set_undo_armed(False)
        return out, {}, sink

    def rollback_speculative_cache(
        self, prompt_cache, gdn_states, accepted, block_size: int
    ) -> None:
        """Rewind every layer cache to the accepted prefix of the verify
        block. Every leaf is checked before any is trimmed: the layers
        read their positions from their own offsets, so a partial rewind
        would leave them apart."""
        if isinstance(accepted, mx.array):
            accepted = int(accepted.reshape(-1)[0].item())
        accepted = int(accepted)
        rejected = int(block_size) - accepted - 1
        if rejected <= 0:
            return
        leaves = _trim_leaves(prompt_cache)
        refused = [
            type(leaf).__name__
            for leaf in leaves
            if not (
                leaf._can_trim(rejected)
                if hasattr(leaf, "_can_trim")
                else leaf.is_trimmable()
            )
        ]
        if refused:
            raise RuntimeError(
                f"hy_v4_flash MTP rollback: untrimmable cache leaves "
                f"{refused} (rejected={rejected}); undo log missing or "
                "consumed")
        for leaf in leaves:
            if leaf.trim(rejected) != rejected:
                raise RuntimeError(
                    f"hy_v4_flash MTP rollback: {type(leaf).__name__}.trim"
                    f"({rejected}) refused after is_trimmable(); the cache "
                    "state is now inconsistent")
        slot = self.model._history_slot(prompt_cache)
        if slot is not None and gdn_states:
            history, tokens, walk = gdn_states[0]
            slot[0] = self.model.engram_hash.history_after(
                history, tokens[:, :accepted + 1])
            if walk is not None:
                # The walk again from where the verify started, over the
                # rows that stay.
                rope_positions.restore(slot, walk)
                keys, _ = hy.split_cache(prompt_cache[0])
                rope_positions.plan(
                    slot, None, keys.offset - (accepted + 1), accepted + 1,
                    max(int(self.args.index_kpool) - 1, 0))


class HyV4FlashSpecLM(HyV4FlashSpecHooks, hy.Model):
    """The Hy4-flash ``Model`` with the speculative hooks, for the
    text-only MTP target."""

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
        inputs_embeds: Optional[mx.array] = None,
        n_to_process: Optional[int] = None,
        return_hidden: bool = False,
        return_shared_kv: bool = False,
        **kwargs,
    ):
        # The engine calls language_model(inputs=ids, ...) by keyword. The
        # token ids are what the model reads (the engram hash needs them),
        # and skip_logits leaves out the head.
        skip_logits = kwargs.pop("skip_logits", False)
        del inputs_embeds, n_to_process, kwargs
        out = self.model(inputs, cache)
        logits = None if skip_logits else self._head(out)
        if not (return_hidden or return_shared_kv):
            from gmlx._vendor.mlx_vlm.models.base import LanguageModelOutput

            return LanguageModelOutput(logits=logits)
        return _SpecOutput(logits=logits, hidden_states=[out])


class HyV4FlashMTPLayer(nn.Module):
    """The draft block's layer: the trunk's sparse attention and latent
    MoE, each behind its own norms, on a plain residual."""

    def __init__(self, args, layer_idx: int):
        super().__init__()
        self.self_attn = hy.HyV4FlashAttention(args, layer_idx)
        self.mlp = hy.HyV4FlashMoE(args)
        self.input_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps)
        self.ffn_post_norm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps)

    def __call__(self, x: mx.array, cache: Optional[Any] = None) -> mx.array:
        x = x + self.self_attn(self.input_layernorm(x), cache=cache)
        return x + self.ffn_post_norm(
            self.mlp(self.post_attention_layernorm(x)))


class HyV4FlashMTPDrafter(QwenMTPDrafter):
    """The Qwen drafter algorithm over the Hy4-flash draft block: its own
    keys and pools over the whole prompt, teacher-forced at prefill."""

    prefer_requested_block_size = False
    cap_at_configured_depth = True
    # A pool cache has no clone path.
    supports_kv_sidecar = False
    # The stock mlx-vlm round does not know this target's hooks.
    requires_owned_engine = True

    def __init__(self, config: HyV4FlashMTPConfig):
        nn.Module.__init__(self)
        if int(config.block_size) != 2:
            # The accept trim rewinds the block's own pool cache through
            # its one-update undo log. A deeper rollout appends one update
            # a draft and cannot rewind past the last.
            raise ValueError(
                "HyV4FlashMTPDrafter supports block_size=2 only")
        self.config = config
        self._native_block_size = (
            native_block_size(config) or int(config.block_size))
        args = config.text_config
        layer_args = draft_layer_args(args)

        hidden_size = args.hidden_size
        eps = args.rms_norm_eps
        self.fc = nn.Linear(2 * hidden_size, hidden_size, bias=False)
        self.pre_fc_norm_embedding = nn.RMSNorm(hidden_size, eps=eps)
        self.pre_fc_norm_hidden = nn.RMSNorm(hidden_size, eps=eps)
        self.layers = [
            HyV4FlashMTPLayer(layer_args, int(args.num_hidden_layers))]
        self.norm = nn.RMSNorm(hidden_size, eps=eps)
        self._kpool = int(args.index_kpool)
        self._indexed = bool(args.index_topk)
        # A second draft step reads the block's own output after its head
        # norm, as the reference feeds it.
        self._postnorm_feed = True

        # Bound to the target at reset(): the block shares its embeddings
        # and head.
        self._input_embed = None
        self._input_embed_scale: float = 1.0
        self._lm_head_fn = None

        self._cache: List[Any] = []
        self._seed_token: Optional[mx.array] = None
        self._seed_hidden: Optional[mx.array] = None
        self._round_appended = 0
        # The image pad token's id where the target is the vision
        # container, which spells an image slot as an id past the vocab.
        self._slot_pad: Optional[int] = None

        self.accept_lens: List[float] = []
        self.draft_lens: List[int] = []

    def bind(self, target_model):
        super().bind(target_model)
        lm = getattr(target_model, "language_model", target_model)
        self._slot_pad = getattr(lm, "image_token_id", None)
        return self

    def _prompt_ids(self, ids: mx.array) -> mx.array:
        """Prompt ids as the block embeds them: an image slot reads the
        pad token's row. The block rotates its keys at their slots, so
        after an image its drafts are the 1-D rule's."""
        if self._slot_pad is None:
            return ids
        vocab = int(self.config.text_config.vocab_size)
        return mx.where(ids >= vocab, self._slot_pad, ids)

    def seed_chunk(self, shifted_tokens, hidden, cache) -> None:
        super().seed_chunk(self._prompt_ids(shifted_tokens), hidden, cache)

    def prefill_from_target_hidden(self, input_ids, hidden, *args,
                                   **kwargs) -> None:
        super().prefill_from_target_hidden(
            self._prompt_ids(input_ids), hidden, *args, **kwargs)

    def make_cache(self, left_padding: Optional[List[int]] = None) -> List[Any]:
        if left_padding is not None:
            raise NotImplementedError(
                "HyV4FlashMTPDrafter runs one sequence: the pooled indexer "
                "selects for a single row")
        keys = hy.HyV4FlashKVCache()
        if not self._indexed:
            return [keys]
        pool = DS4PoolingCache(self._kpool, lookback=False)
        pool.quantizable = False
        return [CacheList(keys, pool)]

    def inject_rows(self, *args, **kwargs) -> None:
        raise NotImplementedError(
            "HyV4FlashMTPDrafter runs one sequence: no batched rows")

    def _seed_profile(self, cache: Optional[List[Any]] = None):
        # The block scores its queries as a trunk sparse layer does.
        return hy._prefill_score_profile(
            self.config.text_config, self._cache if cache is None else cache)

    def _forward(self, tokens: mx.array, hidden: mx.array,
                 cache: Optional[List[Any]] = None) -> mx.array:
        """The block over (tokens, the target's normed hidden), before its
        head norm. Positions are the block's own key count. ``cache``
        stands in for the block's own list and never replaces it."""
        embed = self._input_embed(tokens.astype(mx.int32))
        h = mx.concatenate(
            [self.pre_fc_norm_embedding(embed),
             self.pre_fc_norm_hidden(hidden)],
            axis=-1,
        )
        h = self.fc(h)
        caches = self._cache if cache is None else cache
        return self.layers[0](h, caches[0] if caches else None)
