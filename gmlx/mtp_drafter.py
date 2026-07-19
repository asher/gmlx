"""Owned native-head MTP drafter (Qwen3.5 / 3.6, dense + MoE).

Drop-in replacement for mlx-vlm's ``Qwen3_5MTPDraftModel`` in the owned
speculative engine. Same weight tree (``fc`` / ``pre_fc_norm_embedding`` /
``pre_fc_norm_hidden`` / ``layers.{i}`` / ``norm``) so the GGUF MTP-block remap
(loader ``remap_mtp_arrays`` + ``_install_and_load(..., sanitize=False)``) loads
onto it unchanged, and the transformer block itself is reused from mlx-vlm -- we
own the *draft algorithm*, not the numerics.

Why own it: a place to own the draft algorithm (the rollout feed, the block
schedule, the batched round for M2) without forking mlx-vlm's transformer
numerics. The KV construction matches llama.cpp's draft-mtp: the head teacher-
forces the whole shifted prompt into its own KV during prefill
(``prefill_from_target_hidden``), so the single MTP layer attends over full
prompt context at draft time. Source-verified: llama's process() hook fires on
every prefill ubatch and teacher-forces the full (token, target-hidden) prompt
into the draft KV (common/speculative.cpp); begin() does nothing. Full prompt
context is the depth lever: a 1-token seed erodes acceptance (0.634 vs 0.699
at d4096, 0.585 vs 0.649 at d16384 on Qwen3.6-27B).

The one place this drafter diverges from mlx-vlm (a perf knob, not numerics): the
rollout feeds the head's pre-final-norm output as the next "hidden" (mlx-vlm
feeds the post-norm output, which double-normalizes since pre_fc_norm_hidden
re-normalizes). Toggle with GMLX_MTP_POSTNORM_FEED=1.

Correctness is independent of all of this: the engine's verify step emits the
target's own greedy/sampled tokens, so the drafter only sets how many draft
tokens are accepted (speed), never which tokens (output). The KV construction is
a pure perf knob -- validated token-identical across KV-window choices.

Contract consumed by ``gmlx.speculative`` (and, for B>1, mlx-vlm's
stock ``_mtp_rounds_batch`` until M2 owns the batched round): ``reset``,
``prefill_from_target_hidden``, ``set_shared_kv``, ``draft_block``,
``accept_verified_tokens`` (+ ``_batch`` / ``filter_batch`` for B>1),
``draft_eval_state``, ``accept_lens`` / ``draft_lens``, ``config.block_size``,
and the class capability flags.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from mlx_vlm.models.base import create_attention_mask
from mlx_vlm.models.cache import BatchKVCache, KVCache

from . import prefill_decay
from .envflags import env_bool, env_int
from mlx_vlm.models.qwen3_5.language import Qwen3_5DecoderLayer
from mlx_vlm.models.qwen3_5_moe.language import Qwen3_5MoeDecoderLayer

# The rollout has no target hidden for a drafted token, so it feeds the MTP head's
# own previous-step output as the next "hidden". The head's pre_fc_norm_hidden
# expects a raw (pre-final-norm) hidden -- that is what the seed gets (the target's
# pre-final-norm hidden) -- so the rollout feeds the layer output before self.norm.
# Feeding the post-norm output (mlx-vlm's behavior) double-normalizes it. Flip to
# the old post-norm feed for A/B with GMLX_MTP_POSTNORM_FEED=1.
_POSTNORM_FEED = env_bool("GMLX_MTP_POSTNORM_FEED", False)

# Teacher-force chunk width. Explicit GMLX_HEAD_SEED_CHUNK wins; unset,
# it follows the serve prefill chunk (PREFILL_STEP_SIZE, default 2048) so one
# knob caps both chunk-x-depth attention transients: the TF seed pays the same
# score transient as the target's own prefill, at full depth, in the single
# step between prefill and the first decode round.
_SEED_CHUNK = env_int("GMLX_HEAD_SEED_CHUNK", 0) or None


def _seed_chunk() -> int:
    if _SEED_CHUNK:
        return max(1, _SEED_CHUNK)
    return max(1, env_int("PREFILL_STEP_SIZE", 2048))


class QwenMTPDrafter(nn.Module):
    """Single native MTP head; full-prompt KV (teacher-forced at prefill)."""

    supports_greedy_draft_argmax = True
    prefer_requested_block_size = True
    cap_at_configured_depth = False
    uses_shared_kv = False
    supports_kv_sidecar = True

    def __init__(self, config):
        super().__init__()
        self.config = config
        self._native_block_size = int(config.block_size)
        text_config = config.text_config
        if text_config is None:
            raise ValueError("MTP drafter config.text_config must be set")

        hidden_size = text_config.hidden_size
        mtp_layers = int(getattr(text_config, "mtp_num_hidden_layers", 1))
        layer_config = replace(
            text_config, num_hidden_layers=mtp_layers, full_attention_interval=1
        )
        layer_cls = (
            Qwen3_5MoeDecoderLayer
            if "moe" in getattr(text_config, "model_type", "")
            else Qwen3_5DecoderLayer
        )

        self.fc = nn.Linear(2 * hidden_size, hidden_size, bias=False)
        self.pre_fc_norm_embedding = nn.RMSNorm(hidden_size, eps=text_config.rms_norm_eps)
        self.pre_fc_norm_hidden = nn.RMSNorm(hidden_size, eps=text_config.rms_norm_eps)
        self.layers = [layer_cls(args=layer_config, layer_idx=0) for _ in range(mtp_layers)]
        self.norm = nn.RMSNorm(hidden_size, eps=text_config.rms_norm_eps)

        # Feed the head's pre-final-norm output as the next rollout hidden (see
        # module note). Per-instance so it is A/B-testable without a reload.
        self._postnorm_feed = _POSTNORM_FEED

        # Bound to the target at reset(): the head shares its embeddings + LM head.
        self._input_embed = None
        self._input_embed_scale: float = 1.0
        self._lm_head_fn = None

        # Decode-time-only state: own KV + the precomputed next-round seed.
        self._cache: list[Any] = []
        self._seed_token: mx.array | None = None
        self._seed_hidden: mx.array | None = None
        self._round_appended = 0

        self.accept_lens: list[float] = []
        self.draft_lens: list[int] = []

    # --- binding + lifecycle ------------------------------------------------

    def bind(self, target_model) -> "QwenMTPDrafter":
        """Adopt the target's input embeddings + LM head (Qwen MTP shares both)."""
        inner = None
        if hasattr(target_model, "embed_tokens"):
            inner = target_model
        elif hasattr(target_model, "model") and hasattr(target_model.model, "embed_tokens"):
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
        self._input_embed_scale = float(getattr(inner, "embed_scale", 1.0))
        lm = getattr(target_model, "language_model", target_model)
        self._lm_head_fn = (
            getattr(target_model, "lm_head", None)
            or getattr(lm, "lm_head", None)
            or self._input_embed.as_linear
        )
        return self

    def make_cache(self, left_padding: list[int] | None = None) -> list[Any]:
        if left_padding is not None:
            return [BatchKVCache(left_padding) for _ in self.layers]
        return [KVCache() for _ in self.layers]

    def reset(self, target_model, left_padding: list[int] | None = None) -> list[Any]:
        self.bind(target_model)
        self.accept_lens = []
        self.draft_lens = []
        self._cache = self.make_cache(left_padding)
        self._seed_token = None
        self._seed_hidden = None
        self._round_appended = 0
        return self._cache

    def draft_eval_state(self) -> list[Any]:
        """Arrays the engine async-evals to keep the drafter RNG/cache coupled."""
        state: list[Any] = [self._seed_token, self._seed_hidden]
        for cache in self._cache:
            state.append(cache.state)
        return state

    def export_kv(self) -> list[Any]:
        """The head's live KV caches (for sidecar snapshotting)."""
        return self._cache

    def restore_kv(self, caches: list[Any]) -> None:
        """Adopt restored KV caches (B=1 sidecar warm start; call after reset)."""
        self._cache = list(caches)

    def set_shared_kv(self, *args, **kwargs) -> None:
        """No-op: a Qwen MTP head attends only to its own KV (the full prompt it
        teacher-forced at prefill, then the accepted tokens), never the target's
        shared KV. Positions come from this head's cache offset, so none of the
        engine's shared-KV bookkeeping (offset/position) is needed here."""
        return None

    # --- forward primitives -------------------------------------------------

    def _forward(self, tokens: mx.array, hidden: mx.array) -> mx.array:
        """Run the head over (tokens, hidden); return the PRE-final-norm output.

        Positions are derived inside the attention from each layer's own cache
        offset (position_ids=None), so the head's RoPE frame is its decode-time KV
        length, not the target sequence position.
        """
        embed = self._input_embed(tokens.astype(mx.int32)) * self._input_embed_scale
        h = mx.concatenate(
            [self.pre_fc_norm_embedding(embed), self.pre_fc_norm_hidden(hidden)], axis=-1
        )
        h = self.fc(h)
        for layer, layer_cache in zip(self.layers, self._cache):
            mask = (
                create_attention_mask(h, layer_cache)
                if layer_cache is not None
                else ("causal" if h.shape[1] > 1 else None)
            )
            h = layer(h, mask=mask, cache=layer_cache, position_ids=None)
        return h

    def _logits(self, h_prenorm: mx.array) -> mx.array:
        return self._lm_head_fn(self.norm(h_prenorm))

    def _next_hidden(self, h_prenorm: mx.array) -> mx.array:
        return self.norm(h_prenorm) if self._postnorm_feed else h_prenorm

    def _pick(self, logits: mx.array, sampler, greedy: bool) -> mx.array:
        return mx.argmax(logits, axis=-1) if greedy else sampler(logits)

    def _set_seed(self, h_prenorm: mx.array, sampler, greedy: bool) -> None:
        self._seed_token = self._pick(self._logits(h_prenorm), sampler, greedy)
        self._seed_hidden = self._next_hidden(h_prenorm)

    def _seed_profile(self):
        return prefill_decay.resolve_score_profile(self, self._cache)

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
        """Teacher-force the captured prompt context into the head's KV.

        Teacher-forces the last ``h_len`` prompt positions, where ``h_len`` is
        however much target hidden the engine captured: the full prompt (CLI
        ``stream_speculative`` and the full-prompt serve prefill) seeds the whole
        context -- llama.cpp parity, whose draft-mtp process() hook teacher-forces
        the full prompt into the draft KV during prefill -- and a truncated
        1-token capture (the stock serve prefill, unpatched) degrades to a
        decode-time-only seed. No crash either way; full prompt context lifts
        acceptance at depth over a 1-token seed (measured rate 0.634->0.699 at
        d4096, 0.585->0.649 at d16384 on Qwen3.6-27B).

        At MTP position p the head takes (token_{p+1}, target_hidden_p) and
        predicts token_{p+2}, so the shifted tokens for the last h_len positions
        are [input_ids[n-h_len+1:], bonus], paired with the last h_len hidden rows.
        """
        n = int(input_ids.shape[1])
        if n == 0 or int(hidden.shape[1]) == 0:
            return
        if isinstance(bonus_token, int):
            bonus = mx.array([[bonus_token]], dtype=token_dtype)
        else:
            bonus = bonus_token.reshape(-1)[:, None].astype(token_dtype)
        n_hidden = int(hidden.shape[1])
        h_len = min(n_hidden, n)
        shifted = mx.concatenate(
            [input_ids[:, n - h_len + 1:].astype(token_dtype), bonus], axis=1)
        hid = hidden[:, n_hidden - h_len:, :]
        # Chunk the teacher-force like the target's own prefill. A single call at
        # qL == h_len puts the head's attention on the wide-query SDPA path, whose
        # cost grows super-linearly with prompt length (tens of seconds past ~32k,
        # a ~30 GB attention transient, all charged to the first decode round);
        # chunking keeps every call on the same prefill-shaped path the target
        # uses, with identical offset-causal semantics via the head's KV offset.
        h = None
        base = _seed_chunk()
        heads = prefill_decay.score_heads(getattr(self, "config", None))
        # Resolve against the DRAFTER's own config and cache (the seed
        # transient lives in the head's dense layers), never the target's.
        # No shipping drafter registers a profile, so this is a no-op today.
        profile = self._seed_profile()
        i = 0
        while i < h_len:
            # Depth-decay against the head's own KV offset: the seed's score
            # transient grows with i exactly like the target's prefill. The
            # seed variant honors the kill switch and its own (larger) cap.
            step = prefill_decay.decayed_seed_step(base, i, heads,
                                                   profile=profile)
            j = min(i + step, h_len)
            h = self._forward(shifted[:, i:j], hid[:, i:j, :])
            if j < h_len:
                mx.eval([c.state for c in self._cache])
            i = j
        self._set_seed(h[:, -1:, :], sampler, greedy)

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
        del cache
        if self._input_embed is None or self._lm_head_fn is None:
            raise RuntimeError("bind(target_model) must run before draft_block()")

        if isinstance(last_bonus, int):
            tok = mx.array([[last_bonus]], dtype=token_dtype)
        else:
            tok = last_bonus.reshape(-1)[:, None].astype(token_dtype)
        h_prev = hidden
        tokens: list[mx.array] = []
        self._round_appended = 0

        # Native head: a seed is always set (prefill / accept). draft[0] is the
        # seed token; its hidden drives the first autoregressive step.
        if self._seed_token is not None and self._seed_hidden is not None:
            tok = self._seed_token.astype(token_dtype)
            h_prev = self._seed_hidden
            tokens.append(tok)
            self._seed_token = None
            self._seed_hidden = None

        while len(tokens) < block_size - 1:
            h = self._forward(tok, h_prev)
            self._round_appended += 1
            tok = self._pick(self._logits(h), sampler, greedy)
            h_prev = self._next_hidden(h)
            tokens.append(tok)

        return mx.concatenate(tokens, axis=1)

    def accept_verified_tokens(
        self,
        verify_hidden: mx.array,
        draft_tokens: mx.array,
        accepted: int,
        new_tokens: list[int],
        sampler,
        token_dtype: mx.Dtype = mx.int32,
        greedy: bool = False,
    ) -> None:
        """Trim draft KV, re-process accepted tokens with target hidden, set seed.

        Always re-processes all accepted tokens (not just the delta) so every MTP
        KV entry uses target hidden -- draft_block writes KV with MTP's own hidden
        (autoregressive), which drifts from the target representation the head was
        trained with.  Matches llama.cpp's process() which rebuilds MTP KV from
        target hidden at every round."""
        if self._round_appended > 0:
            for cache in self._cache:
                cache.trim(self._round_appended)

        token_chunks: list[mx.array] = []
        hidden_chunks: list[mx.array] = []
        for draft_idx in range(int(accepted)):
            token_chunks.append(draft_tokens[:, draft_idx : draft_idx + 1])
            hidden_chunks.append(verify_hidden[:, draft_idx : draft_idx + 1, :])
        if new_tokens:
            token_chunks.append(mx.array([[int(new_tokens[-1])]], dtype=token_dtype))
            hidden_chunks.append(verify_hidden[:, int(accepted) : int(accepted) + 1, :])

        if token_chunks:
            tokens = mx.concatenate(token_chunks, axis=1).astype(token_dtype)
            hiddens = mx.concatenate(hidden_chunks, axis=1)
            h = self._forward(tokens, hiddens)
            self._set_seed(h[:, -1:, :], sampler, greedy)
        self._round_appended = 0

    # --- batched accept + filter (B>1 stopgap; M2 owns the batched round) ---

    def accept_verified_tokens_batch(
        self,
        verify_hidden: mx.array,
        draft_tokens: mx.array,
        accepted: list[int],
        new_tokens: list[list[int]],
        sampler,
        token_dtype: mx.Dtype = mx.int32,
        greedy: bool = False,
    ) -> None:
        """Ragged per-row trim + append for B>1 (BatchKVCache prepare/finalize).

        Stopgap so the stock B>1 serve round keeps working with this drafter;
        the batched round is owned in M2. Decode-time-only KV applies per row.
        """
        if len(accepted) <= 1:
            self.accept_verified_tokens(
                verify_hidden, draft_tokens, int(accepted[0]),
                new_tokens[0], sampler, token_dtype, greedy,
            )
            return

        accepted = [int(a) for a in accepted]
        if self._round_appended > 0:
            for cache in self._cache:
                cache.trim(self._round_appended)

        draft_rows = draft_tokens.tolist()
        row_tokens: list[list[int]] = []
        row_hiddens: list[list[mx.array]] = []
        for row, accepted_i in enumerate(accepted):
            tokens_i: list[int] = []
            hiddens_i: list[mx.array] = []
            for draft_idx in range(accepted_i):
                tokens_i.append(int(draft_rows[row][draft_idx]))
                hiddens_i.append(verify_hidden[row : row + 1, draft_idx : draft_idx + 1, :])
            if new_tokens[row]:
                tokens_i.append(int(new_tokens[row][-1]))
                hiddens_i.append(verify_hidden[row : row + 1, accepted_i : accepted_i + 1, :])
            row_tokens.append(tokens_i)
            row_hiddens.append(hiddens_i)

        lengths = [len(t) for t in row_tokens]
        max_len = max(lengths) if lengths else 0
        if max_len == 0:
            self._round_appended = 0
            return

        token_data: list[int] = []
        hidden_rows: list[mx.array] = []
        for tokens_i, hiddens_i in zip(row_tokens, row_hiddens):
            pad = max_len - len(tokens_i)
            token_data.extend(tokens_i + [0] * pad)
            if hiddens_i:
                hidden_row = mx.concatenate(hiddens_i, axis=1)
            else:
                hidden_row = mx.zeros((1, 0, verify_hidden.shape[-1]), dtype=verify_hidden.dtype)
            if pad:
                hidden_row = mx.concatenate(
                    [hidden_row, mx.zeros((1, pad, verify_hidden.shape[-1]), dtype=verify_hidden.dtype)],
                    axis=1,
                )
            hidden_rows.append(hidden_row)

        tokens = mx.array(token_data, dtype=token_dtype).reshape(len(row_tokens), max_len)
        hiddens = mx.concatenate(hidden_rows, axis=0)
        right_padding = [max_len - length for length in lengths]
        if any(right_padding):
            for cache in self._cache:
                prepare = getattr(cache, "prepare", None)
                if callable(prepare):
                    prepare(right_padding=right_padding, lengths=lengths)

        h = self._forward(tokens, hiddens)

        if any(right_padding):
            for cache in self._cache:
                finalize = getattr(cache, "finalize", None)
                if callable(finalize):
                    finalize()

        last_idx = mx.array([length - 1 for length in lengths], dtype=mx.int32)
        last_hidden = mx.take_along_axis(h, last_idx[:, None, None], axis=1)
        self._set_seed(last_hidden, sampler, greedy)
        self._round_appended = 0

    def filter_batch(self, keep) -> None:
        """Keep only the active rows (continuous-batch eviction)."""
        if not isinstance(keep, mx.array):
            keep = mx.array(keep, dtype=mx.int32)
        for cache in self._cache:
            cache_filter = getattr(cache, "filter", None)
            if callable(cache_filter):
                cache_filter(keep)
            elif cache.keys is not None:
                cache.keys = cache.keys[keep]
                cache.values = cache.values[keep]
        if self._seed_token is not None:
            self._seed_token = self._seed_token[keep]
        if self._seed_hidden is not None:
            self._seed_hidden = self._seed_hidden[keep]

    def inject_rows(
        self,
        prompt_tokens: mx.array,
        hidden: mx.array,
        first_tokens: mx.array,
        sampler,
        token_dtype: mx.Dtype = mx.int32,
        greedy: bool = True,
    ) -> None:
        """Prefill the drafter for new rows and extend into the live batch.

        Creates temporary caches, teacher-forces the new rows in isolation,
        then extends the live batch caches and seed state.
        """
        B_new = int(prompt_tokens.shape[0])
        old_cache = self._cache
        old_seed_token = self._seed_token
        old_seed_hidden = self._seed_hidden

        temp_cache = [BatchKVCache([0] * B_new) for _ in self.layers]
        self._cache = temp_cache
        self.prefill_from_target_hidden(
            prompt_tokens, hidden, first_tokens, sampler, token_dtype, greedy)
        new_seed_token = self._seed_token
        new_seed_hidden = self._seed_hidden

        self._cache = old_cache
        for old, temp in zip(old_cache, temp_cache):
            extend_fn = getattr(old, "extend", None)
            if callable(extend_fn):
                extend_fn(temp)
        if old_seed_token is not None and new_seed_token is not None:
            self._seed_token = mx.concatenate([old_seed_token, new_seed_token])
        else:
            self._seed_token = new_seed_token
        if old_seed_hidden is not None and new_seed_hidden is not None:
            self._seed_hidden = mx.concatenate([old_seed_hidden, new_seed_hidden])
        else:
            self._seed_hidden = new_seed_hidden

    # --- weight loading -----------------------------------------------------

    def sanitize(self, weights: dict) -> dict:
        """Strip an ``mtp.`` prefix; expand fused MoE experts. Note: the GGUF
        loader installs with sanitize=False (GGUF norms are already raw), so this
        runs only for non-GGUF load_weights callers."""
        out = {}
        weights = dict(weights)
        expert_prefixes = [
            key[: -len(".experts.gate_up_proj")]
            for key in weights
            if key.endswith(".experts.gate_up_proj")
        ]
        for prefix in expert_prefixes:
            gate_up_weight = weights.pop(f"{prefix}.experts.gate_up_proj")
            gate_weight, up_weights = mx.split(gate_up_weight, 2, axis=-2)
            weights[f"{prefix}.switch_mlp.gate_proj.weight"] = gate_weight
            weights[f"{prefix}.switch_mlp.up_proj.weight"] = up_weights
            weights[f"{prefix}.switch_mlp.down_proj.weight"] = weights.pop(
                f"{prefix}.experts.down_proj"
            )
        for key, value in weights.items():
            if key.startswith("mtp."):
                key = key[len("mtp.") :]
            out[key] = value
        return out
