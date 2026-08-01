# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4-Flash DSpark: the GA (0731) auxiliary draft model.

DSpark replaces the single-block nextn MTP head (deepseek_v4_mtp): a 3-block
drafter (sidecar arch ``deepseek4-dspark``, tensors ``mtp.{0,1,2}.*``) that
reads the target's intermediate hidden states and proposes up to
``dspark.block_size`` (5) tokens per round. Reference forward (upstream
``inference/model.py`` DSparkBlock + ds4's runtime):

- Capture: for each ``dspark.target_layer_ids`` layer (40/41/42), the mean
  over the hc_mult hyper-connection streams of the layer OUTPUT, concatenated
  in ascending layer order -> (B, S, 3*hidden).
- Stage 0 input: ``main_x = main_norm(main_proj(concat))``. Each stage keeps
  its own sliding-window KV ring over per-position ``main_x`` rows (per-stage
  wkv/kv_norm), appended as positions commit.
- A round embeds ``[bonus, noise*4]`` (noise_token_id placeholders) at the
  next 5 positions, runs them through the 3 chained V4 blocks with
  NON-CAUSAL attention (every draft row sees the whole support window and
  all block rows; the block-row KV writes are transient), then the final
  stage's hc_head -> norm -> TARGET lm_head produce 5 logit rows.
- A rank-``markov_rank`` bigram head sequentially biases each row's logits
  with ``markov_w2 @ markov_w1[prev_token]`` before the pick, and a
  confidence head ``sigmoid(proj([hidden_row; markov_w1[prev]]))`` prunes the
  proposal to its confident prefix (ds4 default threshold 0.9).

Where ds4 and the upstream reference disagree, ds4's variant is the default
(it is the parity oracle and its acceptance fixture proves multi-token
accepts on real weights); ``GMLX_DSPARK_REF_SEMANTICS=1`` switches both known
divergences to the reference form:
- main-row KV: ds4 applies the stage's ``attn_norm`` to ``main_x`` before
  ``wkv`` (its hc-mean prelude collapses to a positive scalar on the uniform
  hc broadcast, which the RMSNorm erases); the reference feeds raw main_x.
- confidence features: ds4 uses the post-``mtp.2.norm`` hidden row; the
  reference uses the pre-norm hc_head output.

Correctness is drafter-independent either way: the verify walk emits the
target's own greedy/sampled tokens, so the drafter affects speed (acceptance),
never output. v1 is B=1 only, same as the nextn drafter.

The engine-facing hidden is PACKED: the SpecLM's verify/prefill hidden is
``concat([raw_4d.flatten(hc), cap_40, cap_41, cap_42], -1)`` so every
existing engine seam (slicing, capture trim) works unchanged; the drafter
unpacks the trailing ``3*hidden`` and the SpecLM's logits hooks unflatten
the leading ``hc_mult*hidden``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, List, Optional

import mlx.core as mx
import mlx.nn as nn

from . import deepseek_v4_model as v4
from .deepseek_v4_hyper_connection import HyperHead


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


@dataclass
class DeepseekV4DSparkConfig:
    """``text`` is the target's ModelArgs with ``compress_ratios`` POST-init
    extended by one 0 per stage. ``block_size`` is the engine block TOTAL
    (drafts + bonus), at most ``draft_len + 1``."""

    text: Any
    n_stages: int = 3
    draft_len: int = 5
    noise_token_id: int = 128799
    target_layer_ids: tuple = (40, 41, 42)
    markov_rank: int = 256
    block_size: int = 6


class DSparkLocalAttention(v4.LocalAttention):
    """LocalAttention minus fast-path kernels and causality: DSpark draft
    rows attend bidirectionally over the support window plus all block rows
    (ds4 runs the block through a no-mask attention kernel; the reference's
    topk idx set is equivalently the full window plus the whole block)."""

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        del mask
        B, L, _ = x.shape
        offset = cache.offset if cache is not None else 0

        q = self.wq_b(self.q_norm(self.wq_a(x)))
        q = q.reshape(B, L, self.n_heads, self.head_dim)
        q = mx.fast.rms_norm(q, None, self.config.rms_norm_eps)
        q = q.transpose(0, 2, 1, 3)
        q = self.rope(q, offset)

        kv = self.kv_norm(self.wkv(x)).reshape(B, 1, L, self.head_dim)
        kv = self.rope(kv, offset)
        kv = v4._kv_qat_roundtrip(kv, self.config.qk_rope_head_dim)
        if cache is not None:
            kv, _ = cache.update_and_fetch(kv, mx.zeros((B, 1, L, 0)))

        out = v4.scaled_dot_product_attention(
            q,
            kv,
            kv,
            cache=cache,
            scale=self.scale,
            mask=None,
            sinks=self.attn_sink.astype(q.dtype),
        )
        out = self.rope(out, offset, inverse=True)

        out = out.reshape(B, self.o_groups, -1, L, self.head_dim)
        out = out.transpose(0, 1, 3, 2, 4).flatten(-2)
        out = self.wo_a(out)
        out = out.transpose(0, 2, 1, 3).flatten(-2)
        return self.wo_b(out)


class _DSparkStage(nn.Module):
    """One mtp.k block: a stock V4 block (hc + MLA + MoE) whose attention is
    swapped for the non-causal DSpark variant, plus the main-row KV append."""

    def __init__(self, args, layer_idx: int):
        super().__init__()
        self.block = v4.DeepseekV4Block(args, layer_idx)
        self.block.attn = DSparkLocalAttention(args, layer_idx)

    def append_main_rows(
        self, main_x: mx.array, cache, ref_semantics: bool
    ) -> None:
        """Project ``main_x`` rows into this stage's KV ring (persistent).

        ds4 routes the (hc-broadcast) main rows through the stage's hc-pre +
        attn_norm before wkv; on a uniform broadcast the hc-pre collapses to
        a positive per-row scalar that the RMSNorm cancels, so attn_norm
        alone is the exact equivalent. The reference feeds main_x raw."""
        attn = self.block.attn
        x = main_x if ref_semantics else self.block.attn_norm(main_x)
        B, S, _ = x.shape
        offset = cache.offset
        kv = attn.kv_norm(attn.wkv(x)).reshape(B, 1, S, attn.head_dim)
        kv = attn.rope(kv, offset)
        kv = v4._kv_qat_roundtrip(kv, attn.config.qk_rope_head_dim)
        cache.update_and_fetch(kv, mx.zeros((B, 1, S, 0)))


class DeepseekV4DSparkDrafter(nn.Module):
    """DSpark drafter following the BatchDrafterProtocol (owned engine)."""

    supports_greedy_draft_argmax = True
    prefer_requested_block_size = False
    cap_at_configured_depth = True
    uses_shared_kv = False
    supports_kv_sidecar = False
    requires_owned_engine = True

    def __init__(self, config: DeepseekV4DSparkConfig):
        super().__init__()
        self.config = config
        args = config.text
        n = int(args.num_hidden_layers)
        n_stages = int(config.n_stages)
        if len(args.compress_ratios) < n + n_stages or any(
            args.compress_ratios[n + k] != 0 for k in range(n_stages)
        ):
            raise ValueError(
                "DeepseekV4DSparkDrafter: config.text.compress_ratios must be "
                f"post-init extended with {n_stages} zeros (got "
                f"{len(args.compress_ratios)} entries, {n} trunk layers)"
            )
        if config.block_size > config.draft_len + 1:
            raise ValueError(
                f"block_size {config.block_size} exceeds draft_len + 1 "
                f"({config.draft_len + 1})"
            )
        self._native_block_size = int(config.block_size)
        self._draft_len = int(config.draft_len)
        self._noise_token_id = int(config.noise_token_id)
        self._sliding_window = int(args.sliding_window)
        self._hc_mult = int(args.hc_mult)
        self._hidden = int(args.hidden_size)
        self._n_targets = len(config.target_layer_ids)
        self.hidden_capture_limit = self._sliding_window

        self.stages = [
            _DSparkStage(args, layer_idx=n + k) for k in range(n_stages)
        ]
        self.main_proj = nn.Linear(
            self._hidden * self._n_targets, self._hidden, bias=False
        )
        self.main_norm = nn.RMSNorm(self._hidden, eps=args.rms_norm_eps)
        self.hc_head = HyperHead(args)
        self.norm = nn.RMSNorm(self._hidden, eps=args.rms_norm_eps)
        self.markov_w1 = nn.Embedding(args.vocab_size, config.markov_rank)
        self.markov_w2 = nn.Linear(config.markov_rank, args.vocab_size, bias=False)
        self.confidence_proj = nn.Linear(
            self._hidden + config.markov_rank, 1, bias=False
        )

        # sigmoid threshold on the confidence head; <=0 disables pruning
        # (fixed 5-token blocks, ds4's --dspark-confidence 0 diagnostic mode).
        self._confidence_tau = _env_float("GMLX_DSPARK_CONF", 0.9)
        self._ref_semantics = os.environ.get("GMLX_DSPARK_REF_SEMANTICS") == "1"

        self._input_embed = None
        self._lm_head_fn = None
        self._cache: List[Any] = []

        self.accept_lens: List[float] = []
        self.draft_lens: List[int] = []

    # --- binding + lifecycle ------------------------------------------------

    def bind(self, target_model) -> "DeepseekV4DSparkDrafter":
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
        from .cache_compat import construction_cache_module

        rot = construction_cache_module().RotatingKVCache
        return [rot(max_size=self._sliding_window) for _ in self.stages]

    def reset(self, target_model, left_padding: Optional[List[int]] = None) -> list:
        if left_padding is not None:
            raise NotImplementedError(
                "DeepseekV4DSparkDrafter is B=1 only (v1): batched rounds "
                "need BatchRotatingKVCache/BatchPoolingCache wiring on the "
                "target"
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

    # --- packed-hidden plumbing ---------------------------------------------

    def _captures(self, packed: mx.array) -> mx.array:
        """Trailing ``n_targets*hidden`` of the packed hidden -> (B,S,3H)."""
        lead = self._hc_mult * self._hidden
        want = lead + self._n_targets * self._hidden
        if int(packed.shape[-1]) != want:
            raise ValueError(
                f"packed hidden width {packed.shape[-1]} != {want}; the "
                "target's _dspark_capture wiring is missing"
            )
        return packed[..., lead:]

    def _append_rows(self, captures: mx.array) -> None:
        main_x = self.main_norm(self.main_proj(captures))
        for stage, cache in zip(self.stages, self._cache):
            stage.append_main_rows(main_x, cache, self._ref_semantics)

    # --- seeding ------------------------------------------------------------

    def prefill_from_target_hidden(
        self,
        input_ids: mx.array,
        hidden: mx.array,
        bonus_token,
        sampler,
        token_dtype: mx.Dtype = mx.int32,
        greedy: bool = False,
    ) -> None:
        """Seed each stage's support window from the last ``<= window``
        prompt hiddens. DSpark needs no draft seed: rounds start from the
        engine-passed bonus token."""
        del input_ids, bonus_token, sampler, token_dtype, greedy
        if int(hidden.shape[1]) == 0:
            return
        hid = hidden[:, -self._sliding_window:]
        self._append_rows(self._captures(hid))

    # --- draft + accept (B==1) ----------------------------------------------

    def _pick(self, logits: mx.array, sampler, greedy: bool) -> mx.array:
        return mx.argmax(logits, axis=-1) if greedy else sampler(logits)

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
        """One DSpark round: 5 rows [bonus, noise*4] through the 3 stages,
        markov-biased picks, confidence-pruned to at most ``block_size - 1``
        drafts (min 1: the engine has no skip-round seam, and draft 0 is the
        plain argmax prediction).

        The stage KV writes for the block rows are transient: snapshots are
        restored before returning so ``accept_verified_tokens`` remains the
        only persistent writer (target-hidden rows, 1:1 with the target)."""
        del hidden, cache
        if block_size > self._native_block_size:
            raise RuntimeError(
                f"DeepseekV4DSparkDrafter drafts at most "
                f"{self._native_block_size - 1} token(s)/round; got "
                f"block_size={block_size} -- cap_at_configured_depth should "
                "have clamped this"
            )
        want = min(int(block_size) - 1, self._draft_len)
        if want <= 0:
            raise RuntimeError(f"draft_block: block_size={block_size} < 2")
        if isinstance(last_bonus, mx.array):
            bonus = last_bonus.reshape(1, -1)[:, :1].astype(mx.int32)
        else:
            bonus = mx.array([[int(last_bonus)]], dtype=mx.int32)
        ids = mx.concatenate(
            [
                bonus,
                mx.full((1, self._draft_len - 1), self._noise_token_id, mx.int32),
            ],
            axis=1,
        )

        x = self._input_embed(ids)
        x = mx.contiguous(
            mx.broadcast_to(
                x[:, :, None, :],
                (x.shape[0], x.shape[1], self._hc_mult, x.shape[2]),
            )
        )
        snaps = []
        for c in self._cache:
            snap = {}
            for f in ("keys", "values", "offset", "_idx"):
                v = getattr(c, f, None)
                snap[f] = (v + 0) if isinstance(v, mx.array) else v
            snaps.append(snap)
        for stage, c in zip(self.stages, self._cache):
            x = stage.block(x, None, c, ids)
        for c, snap in zip(self._cache, snaps):
            for f, v in snap.items():
                setattr(c, f, v)

        hc = self.hc_head(x)
        hid_rows = self.norm(hc)
        logits = self._lm_head_fn(hid_rows)
        conf_rows = hc if self._ref_semantics else hid_rows

        tau = self._confidence_tau
        prev = ids[:, :1]
        drafts: List[mx.array] = []
        for i in range(want):
            m = self.markov_w1(prev)[:, 0]
            if tau > 0.0 and i > 0:
                feat = mx.concatenate(
                    [conf_rows[:, i].astype(mx.float32), m.astype(mx.float32)],
                    axis=-1,
                )
                conf = mx.sigmoid(self.confidence_proj(feat))
                # Host sync per gated row (the first retires the stage graph;
                # the rest are tiny). Draft 0 is never gated -- see docstring.
                if float(conf[0, 0].item()) < tau:
                    break
            li = logits[:, i].astype(mx.float32) + self.markov_w2(m).astype(
                mx.float32
            )
            tok = self._pick(li, sampler, greedy).astype(token_dtype)
            drafts.append(tok[:, None] if tok.ndim == 1 else tok)
            prev = drafts[-1].astype(mx.int32)
        return mx.concatenate(drafts, axis=1)

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
        """Append the committed positions' main rows to every stage's window.

        verify_hidden[:, p] is the target (packed) hidden at the p-th verify
        position; positions 0..accepted were committed (accepted drafts plus
        the row the new bonus was sampled from), matching the rolled-back
        target 1:1. No drafter forward is needed -- DSpark drafts from
        scratch each round."""
        del draft_tokens, new_tokens, sampler, token_dtype, greedy
        rows = int(accepted) + 1
        self._append_rows(self._captures(verify_hidden[:, :rows]))
