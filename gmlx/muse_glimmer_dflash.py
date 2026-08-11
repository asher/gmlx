# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Asher Feldman
# Portions copyright (c) 2026 Prince Canuma (mlx-vlm DFlashDraftModel, MIT)
"""Muse Glimmer DFlash drafter: mlx-vlm's ``DFlashDraftModel`` weights and
module tree, driven by gmlx's owned speculative engine.

DFlash denoises a whole block at once. Its decoder runs in two modes
(llama.cpp ``src/models/dflash.cpp``):

- **inject** (embd batch): the target's captured residuals go through the
  encoder ``hidden_norm(fc(captures))``, and each layer projects that straight
  to K/V (k_norm + RoPE, no ``input_layernorm``, no query, no FFN) and writes
  it into the draft KV ring at the committed tokens' positions.
- **draft** (token batch): ``[last_bonus, MASK x (block-1)]`` is embedded with
  the target's ``tok_embd``, positioned at ``n_past + i``, and run through the
  layers with **non-causal** attention over the ring plus the block's own
  transient K/V. Drafts are read from rows 1..block-1, and the target's
  ``lm_head`` produces the logits.

mlx-vlm folds both into one ``draft_block(last_bonus, hidden, ...)`` call
because its engine hands the drafter the newly committed hidden each round.
gmlx's engine splits the same information across ``prefill_from_target_hidden``
and ``accept_verified_tokens`` (and passes only the last hidden row to
``draft_block``), so the two modes are split here to match - same math, same
weights, different call boundary.

Positions are relative: seeding only the last ``sliding_window`` prompt rows
shifts context and block queries by the same amount, which RoPE is invariant
to. Correctness never rests on the drafter regardless: the verify walk emits
the target's own tokens, so the drafter moves acceptance, never output.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional

import mlx.core as mx

from mlx_vlm.speculative.drafters.qwen3_dflash.config import DFlashConfig
from mlx_vlm.speculative.drafters.qwen3_dflash.dflash import DFlashDraftModel

from . import muse_glimmer_model as mg


@dataclass
class MuseGlimmerDFlashConfig(DFlashConfig):
    """``DFlashConfig`` plus the Glimmer logit tail. The drafter borrows the
    target's LM head, so it must reproduce the target's scale and softcap."""

    output_multiplier: float = 1.0


class MuseGlimmerDFlashDrafter(DFlashDraftModel):
    """DFlash drafter following gmlx's BatchDrafterProtocol (owned engine)."""

    supports_greedy_draft_argmax = True
    prefer_requested_block_size = False
    cap_at_configured_depth = True
    uses_shared_kv = False
    supports_kv_sidecar = False
    # CLI entry points must route to the owned engine: mlx-vlm's stock MTP
    # round doesn't know the muse_glimmer target hooks (packed hidden).
    requires_owned_engine = True

    def __init__(self, config: MuseGlimmerDFlashConfig):
        super().__init__(config)
        self._native_block_size = int(config.block_size)
        self._hidden = int(config.hidden_size)
        self._n_targets = len(config.target_layer_ids)
        # Only the trailing window of the prompt capture is usable; the engine
        # trims the retained prefill hidden to this many positions.
        self.hidden_capture_limit = int(config.sliding_window or 0) or None
        self._cache: List[Any] = []

    # --- lifecycle ----------------------------------------------------------

    def reset(self, target_model, left_padding: Optional[List[int]] = None) -> list:
        if left_padding is not None:
            raise NotImplementedError(
                "MuseGlimmerDFlashDrafter is B=1 only (v1): batched rounds "
                "need per-row cache offsets in the inject path"
            )
        self.bind(target_model)
        self.accept_lens = []
        self.draft_lens = []
        self._cache = self.make_cache()
        return self._cache

    def draft_eval_state(self) -> List[Any]:
        return [c.state for c in self._cache]

    def set_shared_kv(self, *args, **kwargs) -> None:
        return None

    def make_cache(self, left_padding: Optional[List[int]] = None) -> list:
        if left_padding is not None:
            raise NotImplementedError("MuseGlimmerDFlashDrafter is B=1 only (v1)")
        return super().make_cache()

    def _logits(self, hidden: mx.array) -> mx.array:
        # The borrowed LM head is the target's bare Linear; the scale and
        # softcap live in the target's own head, so reapply both here.
        return mg.scale_and_softcap(
            self.lm_head(hidden), self.config.output_multiplier,
            self.config.final_logit_softcapping or 0.0)

    # --- inject mode --------------------------------------------------------

    def _captures(self, packed: mx.array) -> mx.array:
        """Trailing ``n_targets*hidden`` of the packed target hidden."""
        want = self._hidden * (1 + self._n_targets)
        if int(packed.shape[-1]) != want:
            raise ValueError(
                f"packed hidden width {packed.shape[-1]} != {want}; the "
                "target's _dflash_capture wiring is missing"
            )
        # Materialize: this feeds the quantized fc, whose kernel reads the
        # buffer directly and would otherwise see the packed strides.
        return mx.contiguous(packed[..., self._hidden:])

    def append_context(self, captures: mx.array) -> None:
        """Encode committed positions and inject their K/V into the ring."""
        h_ctx = self.hidden_norm(self.fc(captures))
        B, S, _ = h_ctx.shape
        for layer, c in zip(self.layers, self._cache):
            attn = layer.self_attn
            keys = attn.k_norm(
                attn.k_proj(h_ctx).reshape(B, S, attn.n_kv_heads, -1)
            ).transpose(0, 2, 1, 3)
            values = attn.v_proj(h_ctx).reshape(
                B, S, attn.n_kv_heads, -1).transpose(0, 2, 1, 3)
            c.update_and_fetch(self.rope(keys, offset=c.offset), values)

    # --- draft mode ---------------------------------------------------------

    def _draft_hidden(self, tokens: mx.array) -> mx.array:
        h = self._embed_input_tokens(tokens)
        B, L, _ = h.shape
        for layer, c in zip(self.layers, self._cache):
            attn = layer.self_attn
            x = layer.input_layernorm(h)
            q = attn.q_norm(
                attn.q_proj(x).reshape(B, L, attn.n_heads, -1)
            ).transpose(0, 2, 1, 3)
            k = attn.k_norm(
                attn.k_proj(x).reshape(B, L, attn.n_kv_heads, -1)
            ).transpose(0, 2, 1, 3)
            v = attn.v_proj(x).reshape(
                B, L, attn.n_kv_heads, -1).transpose(0, 2, 1, 3)
            q = self.rope(q, offset=c.offset)
            k = self.rope(k, offset=c.offset)
            ck, cv = c.state
            if ck is not None:
                k = mx.concatenate([ck, k], axis=2)
                v = mx.concatenate([cv, v], axis=2)
            # The block denoises as a whole: every row sees the ring and all
            # its siblings. The block's own K/V never enters the ring.
            o = mx.fast.scaled_dot_product_attention(
                q, k, v, scale=attn.scale, mask=None)
            h = h + attn.o_proj(o.transpose(0, 2, 1, 3).reshape(B, L, -1))
            h = h + layer.mlp(layer.post_attention_layernorm(h))
        return self.norm(h)

    def draft_block(
        self,
        last_bonus,
        hidden: mx.array,
        cache,
        block_size: int,
        sampler,
        token_dtype: mx.Dtype = mx.int32,
        greedy: bool = False,
    ) -> mx.array:
        """One DFlash round: ``[bonus, MASK x (block_size-1)]`` denoised in a
        single forward; drafts are rows 1..block_size-1."""
        del hidden, cache
        if block_size > self._native_block_size:
            raise RuntimeError(
                f"MuseGlimmerDFlashDrafter drafts at most "
                f"{self._native_block_size - 1} token(s)/round; got "
                f"block_size={block_size} - cap_at_configured_depth should "
                f"have clamped it"
            )
        if not self._cache:
            raise RuntimeError("reset(target_model) must run before draft_block()")
        mask_id = int(self.config.mask_token_id)
        bonus = (int(last_bonus) if isinstance(last_bonus, int)
                 else int(last_bonus.reshape(-1)[0].item()))
        block = mx.array([[bonus] + [mask_id] * (block_size - 1)], dtype=token_dtype)
        logits = self._logits(self._draft_hidden(block)[:, 1:])
        return mx.argmax(logits, axis=-1) if greedy else sampler(logits)

    # --- commit -------------------------------------------------------------

    def prefill_from_target_hidden(
        self,
        input_ids: mx.array,
        hidden: mx.array,
        bonus_token,
        sampler,
        token_dtype: mx.Dtype = mx.int32,
        greedy: bool = False,
    ) -> None:
        """Seed the ring from the trailing prompt hiddens. DFlash needs no
        draft seed: rounds start from the engine-passed bonus token."""
        del input_ids, bonus_token, sampler, token_dtype, greedy
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
    ) -> None:
        """Inject the committed positions' captures. ``verify_hidden[:, p]`` is
        the target hidden at verify position ``p``; 0..accepted were committed
        (the accepted drafts plus the row the new bonus was sampled from),
        matching the rolled-back target 1:1."""
        del draft_tokens, new_tokens, sampler, token_dtype, greedy
        self.append_context(self._captures(verify_hidden[:, : int(accepted) + 1]))
