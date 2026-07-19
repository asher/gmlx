# SPDX-License-Identifier: Apache-2.0
# Ported from omlx patches/mlx_lm_mtp (Apache-2.0; see licenses/omlx-LICENSE).
"""DeepSeek-V4-Flash MTP: speculative target wrapper + native-head drafter.

The MTP head is a separate GGUF (arch ``deepseek4_mtp_support``, tensors
``mtp.0.*``): one full V4 transformer block (ratio-0 sliding-window
attention + MoE) plus ``e_proj``/``h_proj`` fusion projections, three
RMSNorms, and its own hyper-connection head. Reference forward (omlx
``patches/mlx_lm_mtp/deepseek_v4_model.py`` MTPBlock):

    x = e_proj(enorm(embed(ids)))[:, :, None, :] + h_proj(hnorm(h_4d))
    x = block(x); logits = lm_head(norm(hc_head(x)))

i.e. the head consumes the target's RAW pre-``hc_head`` 4D hidden
(B, S, hc_mult, hidden) and predicts one token ahead.

Design (block_total = 2 -> 1 draft + bonus, zero speculative.py changes):

- ``DeepseekV4SpecLM`` subclasses the vendored ``Model`` so the base remap's
  ``model.*`` / ``lm_head.*`` weight tree loads onto it unchanged, and adds
  the ``speculative_*`` hooks the owned engine probes. Its verify forward
  arms the rotating-cache one-update undo log (deepseek_v4_cache) so a
  rejected draft can be rolled back even after the sliding-window cache has
  rotated -- ``_buffer_mtp_target_cache`` isinstance-checks mlx-vlm cache
  classes and is a silent no-op for these mlx-lm caches.
- ``DeepseekV4MTPDrafter`` follows QwenMTPDrafter's contract but degenerates
  cleanly at depth 1: ``draft_block`` returns the precomputed seed with zero
  forwards (the drafter's own KV is never trimmed), and
  ``accept_verified_tokens`` does the single teacher-forced MTP forward per
  round that sets the next seed. The head's attention window is
  ``sliding_window`` (128) and RoPE is relative, so prompt seeding is capped
  at the window (``hidden_capture_limit``) -- context beyond it is
  mathematically invisible to the head.
- v1 is B=1 only: ``reset(left_padding=...)`` raises; the target has no
  batch caches for PoolingCache layers yet.

Correctness is drafter-independent: the verify walk emits the target's own
greedy/sampled tokens, so the drafter affects speed (acceptance), never
output. The losslessness gate is the greedy A/B vs plain decode.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, List, Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.base import create_attention_mask

from . import deepseek_v4_model as v4
from .deepseek_v4_cache import ensure_rollback_attached, set_undo_armed
from .deepseek_v4_hyper_connection import HyperHead


@dataclass
class DeepseekV4MTPConfig:
    """Drafter config. ``text`` is the target's ModelArgs with
    ``compress_ratios`` re-extended by the MTP layer's ratio (0) POST-init
    (the dataclass ``__post_init__`` truncates to num_hidden_layers).

    ``block_size`` is the block TOTAL (drafts + bonus). 2 = one draft token
    per round (the seed precomputed by the previous accept, zero draft-time
    forwards); 3 adds one recursive rollout forward that self-conditions the
    head on its own raw output (DeepSeek-V3 MTP recursion), reverted from
    the drafter KV before the accept re-writes the position teacher-forced.
    A literal 1 would break the engine (``bs <= 1: break``).
    """

    text: Any
    block_size: int = 4


@dataclass
class _SpecOutput:
    """Duck-typed output for the owned engine's ``return_hidden`` calls."""

    logits: mx.array
    hidden_states: List[mx.array]
    shared_kv_states: dict = field(default_factory=dict)
    gdn_states: Optional[list] = None


def _collect_cache_leaves(prompt_cache: List[Any]) -> List[Any]:
    """Flatten CacheList entries (local + pool caches) into leaves."""
    leaves: List[Any] = []

    def collect(entry):
        children = getattr(entry, "caches", None)
        if children is not None:
            for child in children:
                collect(child)
        elif entry is not None:
            leaves.append(entry)

    for entry in prompt_cache:
        collect(entry)
    return leaves


class DeepseekV4SpecLM(v4.Model):
    """Vendored V4 ``Model`` + the ``speculative_*`` hooks the owned MTP
    engine probes on the target's ``language_model``."""

    def __init__(self, config):
        super().__init__(config)
        # The verify rollback path needs the rotating undo log on the class.
        ensure_rollback_attached()
        # Hard-disable the L1 shared-APC tier for V4 v1 (spec_engine reads it).
        self._kq_apc_mode = None

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
        # `inputs` is mlx-vlm's LanguageModel parameter name -- its chunked
        # prefill calls language_model(inputs=ids, inputs_embeds=..., ...) by
        # keyword. V4 needs the token ids regardless of inputs_embeds (hash
        # MoE routing + embedding happen inside the backbone), so the embeds
        # are ignored. shared_kv is never used (the drafter owns its KV).
        del inputs_embeds, n_to_process, kwargs
        out, h_raw = self.model(inputs, cache, return_raw_hidden=True)
        logits = self.lm_head(out)
        if not (return_hidden or return_shared_kv):
            # mlx-vlm's AR engine calls model.language_model(...) directly
            # and reads ``outputs.logits`` (the plain-path A/B baseline in
            # bench_tg_depth drives it), so raw logits are not an option.
            from mlx_vlm.models.base import LanguageModelOutput

            return LanguageModelOutput(logits=logits)
        return _SpecOutput(logits=logits, hidden_states=[h_raw])

    # --- speculative hooks (owned engine contract) ---------------------------

    def speculative_logits_from_hidden(self, hidden: mx.array) -> mx.array:
        """Collapse the raw 4D hidden and project: (B,S,4,H) -> (B,S,V)."""
        return self.lm_head(self.model.norm(self.model.hc_head(hidden)))

    def speculative_argmax_from_hidden(self, hidden: mx.array) -> mx.array:
        return mx.argmax(self.speculative_logits_from_hidden(hidden), axis=-1)

    def speculative_verify_hidden(self, verify_input: mx.array, prompt_cache):
        """The single verify forward (qL = drafts + 1, S<=3), with the
        rotating undo log armed so ``rollback_speculative_cache`` can undo
        it on rejection (the PoolingCaches stash their own undo
        unconditionally for L<=3)."""
        set_undo_armed(True)
        try:
            _, h_raw = self.model(verify_input, prompt_cache, return_raw_hidden=True)
        finally:
            set_undo_armed(False)
        return h_raw, {}

    def rollback_speculative_cache(
        self, prompt_cache, gdn_states, accepted: int, block_size: int
    ) -> None:
        """Trim the rejected verify tail from every cache leaf, two-phase:
        verify ALL leaves are trimmable before mutating ANY (the shared
        attention mask is built from layer 0's cache offset, so a partial
        rollback would desync layers and corrupt decode)."""
        del gdn_states
        rejected = int(block_size) - int(accepted) - 1
        if rejected <= 0:
            return
        leaves = _collect_cache_leaves(prompt_cache)
        # PoolingCache exposes an n-aware probe (trim(2) can be refusable
        # while trim(1) is fine); other leaves keep the n-blind contract.
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
                f"DeepSeek-V4 MTP rollback: untrimmable cache leaves {refused} "
                f"(rejected={rejected}); undo log missing or consumed"
            )
        for leaf in leaves:
            if leaf.trim(rejected) != rejected:
                raise RuntimeError(
                    f"DeepSeek-V4 MTP rollback: {type(leaf).__name__}.trim"
                    f"({rejected}) refused after is_trimmable() -- cache "
                    f"state is now inconsistent"
                )


class DeepseekV4MTPDrafter(nn.Module):
    """Single native MTP head; window-capped prompt seeding (see module doc)."""

    supports_greedy_draft_argmax = True
    prefer_requested_block_size = False
    cap_at_configured_depth = True
    uses_shared_kv = False
    supports_kv_sidecar = False
    # CLI entry points must route to the owned engine: mlx-vlm's stock MTP
    # round doesn't know these hooks (4D hidden, rotating undo rollback).
    requires_owned_engine = True

    def __init__(self, config: DeepseekV4MTPConfig):
        super().__init__()
        self.config = config
        args = config.text
        n = int(args.num_hidden_layers)
        if len(args.compress_ratios) <= n or args.compress_ratios[n] != 0:
            raise ValueError(
                "DeepseekV4MTPDrafter: config.text.compress_ratios must be "
                "post-init extended with the MTP layer's ratio 0 "
                f"(got {len(args.compress_ratios)} entries for layer index {n})"
            )
        self._native_block_size = int(config.block_size)
        self._sliding_window = int(args.sliding_window)
        # Engine capture seams keep only this many trailing prompt hiddens:
        # the head attends through a sliding window with relative RoPE, so
        # seeding beyond it is compute + memory with zero effect (raw 4D
        # hidden is hc_mult * hidden * 2 bytes per token).
        self.hidden_capture_limit = self._sliding_window

        dim = args.hidden_size
        eps = args.rms_norm_eps
        self.block = v4.DeepseekV4Block(args, layer_idx=n)  # ratio 0 -> Local
        self.e_proj = nn.Linear(dim, dim, bias=False)
        self.h_proj = nn.Linear(dim, dim, bias=False)
        self.enorm = nn.RMSNorm(dim, eps=eps)
        self.hnorm = nn.RMSNorm(dim, eps=eps)
        self.norm = nn.RMSNorm(dim, eps=eps)
        self.hc_head = HyperHead(args)

        # Bound to the target at reset(): the head shares embeddings + LM head.
        self._input_embed = None
        self._lm_head_fn = None

        # Decode-time state: own KV + the precomputed next-round seed.
        self._cache: List[Any] = []
        self._seed_token: Optional[mx.array] = None
        self._seed_hidden: Optional[mx.array] = None  # raw 4D [B,1,hc,H]
        self._seed_conf: Optional[mx.array] = None  # seed softmax max, [B]
        self._round_appended = 0  # always 0 at depth 1 (no rollout forwards)

        # Confidence gate for the recursive rollouts: each draft's
        # conditional accept tracks the previous step's own probability, so
        # rounds below tau skip the rollout forward and the wider verify.
        # 0 disables the gate (always roll out). The second rollout degrades
        # faster (self-conditioned twice; ~45% conditional accept at the
        # first-step tau), so it takes a stricter default.
        self._rollout_tau = float(os.environ.get("GMLX_MTP_S3_TAU", "0.75"))
        self._rollout_tau2 = float(os.environ.get("GMLX_MTP_S4_TAU", "0.9"))

        self.accept_lens: List[float] = []
        self.draft_lens: List[int] = []

    # --- binding + lifecycle ------------------------------------------------

    def bind(self, target_model) -> "DeepseekV4MTPDrafter":
        """Adopt the target's input embeddings + LM head (shared with MTP)."""
        inner = None
        if hasattr(target_model, "embed_tokens"):
            inner = target_model
        elif hasattr(target_model, "model") and hasattr(
            target_model.model, "embed_tokens"
        ):
            inner = target_model.model
        elif (
            hasattr(target_model, "language_model")
            and hasattr(target_model.language_model, "model")
            and hasattr(target_model.language_model.model, "embed_tokens")
        ):
            inner = target_model.language_model.model
        if inner is None:
            raise AttributeError(
                f"Cannot find embed_tokens in {type(target_model).__name__}"
            )
        self._input_embed = inner.embed_tokens
        lm = getattr(target_model, "language_model", target_model)
        self._lm_head_fn = getattr(target_model, "lm_head", None) or getattr(
            lm, "lm_head", None
        )
        if self._lm_head_fn is None:
            raise AttributeError(
                f"Cannot find lm_head in {type(target_model).__name__}"
            )
        return self

    def make_cache(self) -> List[Any]:
        # Runtime-matching origin: the drafter KV sidecar hands these to the
        # mlx-vlm apc clone path, which isinstance-gates on its own classes.
        from .cache_compat import construction_cache_module

        return [construction_cache_module().RotatingKVCache(
            max_size=self._sliding_window)]

    def reset(self, target_model, left_padding: Optional[List[int]] = None) -> list:
        if left_padding is not None:
            raise NotImplementedError(
                "DeepseekV4MTPDrafter is B=1 only (v1): batched rounds need "
                "BatchRotatingKVCache/BatchPoolingCache wiring on the target"
            )
        self.bind(target_model)
        self.accept_lens = []
        self.draft_lens = []
        self._cache = self.make_cache()
        self._seed_token = None
        self._seed_hidden = None
        self._seed_conf = None
        self._round_appended = 0
        return self._cache

    def draft_eval_state(self) -> List[Any]:
        """Arrays the engine async-evals to keep the drafter state coupled."""
        state: List[Any] = [self._seed_token, self._seed_hidden, self._seed_conf]
        for cache in self._cache:
            state.append(cache.state)
        return state

    def set_shared_kv(self, *args, **kwargs) -> None:
        """No-op: the head attends only to its OWN KV (teacher-forced)."""
        return None

    # --- forward primitives -------------------------------------------------

    def _forward(self, tokens: mx.array, hidden_4d: mx.array) -> mx.array:
        """One head forward over (token, target-raw-hidden) pairs.

        ``hidden_4d`` is the target's RAW pre-hc_head hidden [B,S,hc,H];
        returns the block's RAW 4D output. ``tokens`` are also forwarded to
        the block for the MoE gate signature (layer index >= num_hash_layers,
        so hash routing never fires)."""
        tokens = tokens.astype(mx.int32)
        e = self.e_proj(self.enorm(self._input_embed(tokens)))
        x = e[:, :, None, :] + self.h_proj(self.hnorm(hidden_4d))
        x = mx.contiguous(x)
        mask = create_attention_mask(
            x[:, :, 0, :],
            self._cache[0],
            window_size=self._sliding_window,
            return_array=True,
        )
        return self.block(x, mask, self._cache[0], tokens)

    def _logits(self, h_raw: mx.array) -> mx.array:
        return self._lm_head_fn(self.norm(self.hc_head(h_raw)))

    def _pick(self, logits: mx.array, sampler, greedy: bool) -> mx.array:
        return mx.argmax(logits, axis=-1) if greedy else sampler(logits)

    def _set_seed(self, h_raw: mx.array, sampler, greedy: bool) -> None:
        logits = self._logits(h_raw)
        self._seed_token = self._pick(logits, sampler, greedy)
        self._seed_hidden = h_raw
        if self._rollout_tau > 0.0 and self._native_block_size >= 3:
            lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
            self._seed_conf = mx.exp(lp.max(axis=-1)).reshape(-1)
        else:
            self._seed_conf = None

    # --- seeding (decode-time-only) ----------------------------------------

    def prefill_from_target_hidden(
        self,
        input_ids: mx.array,
        hidden: mx.array,
        bonus_token,
        sampler,
        token_dtype: mx.Dtype = mx.int32,
        greedy: bool = False,
    ) -> None:
        """Teacher-force the last ``<= sliding_window`` prompt positions into
        the head's KV and set the first seed.

        At MTP position p the head takes (token_{p+1}, target_hidden_p) and
        predicts token_{p+2}, so the shifted tokens for the last h_len
        positions are [input_ids[n-h_len+1:], bonus] paired with the last
        h_len hidden rows. h_len <= window, so one forward suffices (no
        chunking; the engine's capture cap already bounds ``hidden``)."""
        n = int(input_ids.shape[1])
        if n == 0 or int(hidden.shape[1]) == 0:
            return
        if isinstance(bonus_token, int):
            bonus = mx.array([[bonus_token]], dtype=token_dtype)
        else:
            bonus = bonus_token.reshape(-1)[:, None].astype(token_dtype)
        n_hidden = int(hidden.shape[1])
        h_len = min(n_hidden, n, self._sliding_window)
        shifted = mx.concatenate(
            [input_ids[:, n - h_len + 1 :].astype(token_dtype), bonus], axis=1
        )
        hid = hidden[:, n_hidden - h_len :]
        h = self._forward(shifted, hid)
        self._set_seed(h[:, -1:], sampler, greedy)

    # --- draft + accept (B==1) ---------------------------------------------

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
        """Draft up to ``block_size - 1`` tokens: the seed plus
        confidence-gated recursive rollouts.

        Draft 0 is the seed, precomputed by the previous round's accept (or
        the prefill) -- zero forwards. Each further draft is a recursive
        rollout: one head forward pairing the previous draft with the head's
        own raw output at that position (the target hidden for the slot does
        not exist yet -- DeepSeek-V3 MTP recursion). Every rollout is gated
        on the previous step's argmax confidence. The rollouts' KV writes
        are reverted before returning so the accept hook's teacher-forced
        write, with target hidden, stays the canonical entry per position."""
        del last_bonus, hidden, cache
        if self._seed_token is None:
            raise RuntimeError(
                "draft_block() without a seed: prefill_from_target_hidden / "
                "accept_verified_tokens must run first"
            )
        if block_size > self._native_block_size:
            raise RuntimeError(
                f"DeepseekV4MTPDrafter drafts at most "
                f"{self._native_block_size - 1} token(s)/round (block_total "
                f"{self._native_block_size}); got block_size={block_size} -- "
                f"cap_at_configured_depth should have clamped this"
            )
        tok = self._seed_token.astype(token_dtype)
        if tok.ndim == 1:
            tok = tok[:, None]
        drafts = tok
        n_rollouts = block_size - 2
        if n_rollouts > 0 and self._seed_hidden is not None:
            cache0 = self._cache[0]
            snap = {}
            for f in ("keys", "values", "offset", "_idx"):
                v = getattr(cache0, f, None)
                snap[f] = (v + 0) if isinstance(v, mx.array) else v
            cur, h_prev, conf = tok, self._seed_hidden, self._seed_conf
            for i in range(n_rollouts):
                tau = self._rollout_tau if i == 0 else self._rollout_tau2
                # Syncs on the previous accept's (or rollout's) tiny graph
                # tail; the walk's main sync already retired everything
                # before it.
                if conf is not None and float(conf[0].item()) < tau:
                    break
                # Steps beyond the first are always tau2-gated.
                want_conf = i + 1 < n_rollouts and self._rollout_tau2 > 0.0
                cur, h_prev, conf = self._rollout_next(
                    cur, h_prev, sampler, greedy, token_dtype, want_conf
                )
                drafts = mx.concatenate([drafts, cur], axis=1)
            for f, v in snap.items():
                setattr(cache0, f, v)
        self._seed_token = None
        self._seed_hidden = None
        self._seed_conf = None
        self._round_appended = 0
        return drafts

    def _rollout_next(
        self,
        tok: mx.array,
        h_prev: mx.array,
        sampler,
        greedy: bool,
        token_dtype: mx.Dtype,
        want_conf: bool = False,
    ) -> tuple[mx.array, mx.array, Optional[mx.array]]:
        """One self-conditioned rollout forward. The caller snapshots and
        restores the drafter KV around the rollout chain (detached copies --
        the L=1 ring write mutates the same array wrapper); later rollouts
        attend to earlier rollouts' provisional entries."""
        h2 = self._forward(tok, h_prev)
        logits = self._logits(h2[:, -1:])
        tok2 = self._pick(logits, sampler, greedy).astype(token_dtype)
        if tok2.ndim == 1:
            tok2 = tok2[:, None]
        conf2 = None
        if want_conf:
            lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
            conf2 = mx.exp(lp.max(axis=-1)).reshape(-1)
        return tok2, h2[:, -1:], conf2

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
        """Teacher-force the accepted draft(s) + bonus into the head with
        TARGET hidden, and precompute the next round's seed (the one MTP
        forward per round). No drafter-KV trim: draft_block never writes it.

        Pairing: verify input was [b, d0, ...]; verify_hidden[:, p] is the
        target hidden AT the p-th of those positions, so (draft_p, hidden_p)
        and finally (bonus, hidden_accepted) each predict one ahead. The
        head KV gains accepted+1 entries -- 1:1 with the rolled-back target."""
        token_chunks: List[mx.array] = []
        hidden_chunks: List[mx.array] = []
        for draft_idx in range(int(accepted)):
            token_chunks.append(draft_tokens[:, draft_idx : draft_idx + 1])
            hidden_chunks.append(verify_hidden[:, draft_idx : draft_idx + 1])
        if new_tokens:
            token_chunks.append(mx.array([[int(new_tokens[-1])]], dtype=token_dtype))
            hidden_chunks.append(
                verify_hidden[:, int(accepted) : int(accepted) + 1]
            )
        if token_chunks:
            tokens = mx.concatenate(token_chunks, axis=1).astype(token_dtype)
            hiddens = mx.concatenate(hidden_chunks, axis=1)
            h = self._forward(tokens, hiddens)
            self._set_seed(h[:, -1:], sampler, greedy)
        self._round_appended = 0
