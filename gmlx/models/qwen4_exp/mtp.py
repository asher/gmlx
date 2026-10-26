# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Asher Feldman
"""Qwen3.8-Flash-Next (qwen4exp) MTP: speculative target wrapper + drafter.

The MTP head reaches gmlx as a companion GGUF that the loader autodetects
beside the target, in one of two layouts. Arch ``qwen4exp-mtp`` (built by
the lab's ``extract-qwen4exp-mtp-gguf.py``) keeps the HF ``mtp.*`` tree.
The llama.cpp ``mtp-*.gguf`` sidecar keeps arch ``qwen4exp`` and stores
the head as block ``num_hidden_layers`` with ``nextn.*`` extras, its two
input projections joined into one ``eh_proj`` (see
:data:`QWEN4EXP_NEXTN_MAP`). Reference forward (ollama
``x/models/qwen4_exp`` mtpDraft.Forward):

    e = fc_embedding(pre_fc_norm_embedding(embed(next_token)))      [B,S,D]
    x = fc_hidden(pre_fc_norm_hidden(h_4d)) + e[:, :, None, :]       [B,S,4,D]

The sidecar computes the same x as eh_proj over the per-stream concat
[pre_fc_norm_embedding(e), pre_fc_norm_hidden(h_4d)], embedding half first
(llama.cpp ``src/models/qwen4exp.cpp`` graph_mtp).
    x = layer(x)             # HC + QSA attention + HC + MoE, one layer
    logits = lm_head(hyper_connection_mixer(x))

``h_4d`` is the target's PRE-mixer four-stream hidden ``[B,S,4,D]``; the
rollout feeds the head's own pre-mixer output as the next hidden.
``pre_fc_norm_hidden`` carries one ``[4 * D]`` gamma; per-stream statistics
(grouped) beat the flat reading 6/6 prompts on acceptance (62.8->69.8 pct
class deltas), so grouped is the default; ``GMLX_Q4_MTP_GROUPED_HNORM=0``
restores the flat reading for A/B.

Design mirrors ``deepseek_v4_mtp``: ``Qwen4ExpSpecLM`` subclasses the
vendored ``Model`` so the text remap loads onto it unchanged and adds the
``speculative_*`` hooks; its verify forward records a rollback sink
(``qwen4_exp_model.rollback_verify_sink``) so the GDN scan / conv states and
the PLE history / conv states rewind to the accepted prefix on rejection.
``Qwen4ExpMTPDrafter`` reuses ``QwenMTPDrafter``'s draft algorithm (full
prompt teacher-forced into the head's own KV, seed precomputed at accept)
on the four-stream hidden. v1 is B=1 (the QSA cache has no batch form).

Correctness is drafter-independent: the verify walk emits the target's own
greedy/sampled tokens, so the drafter sets acceptance (speed), never output.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from typing import Any, List, Optional

import mlx.core as mx
import mlx.nn as nn

from . import model as q4
from gmlx.cache.attention import create_attention_mask
from gmlx.envflags import env_bool, env_int
from gmlx.spec.mtp_drafter import QwenMTPDrafter

MTP_ARCH = "qwen4exp-mtp"

# Draft length policy. The round's draft count follows the seed's argmax
# probability, which the engine evaluates with the seed itself, so the
# read waits on nothing and the rollouts stay lazy like a fixed block's.
# GMLX_Q4_MTP_LADDER lists "conf:drafts" steps, highest first; the first
# step whose conf the seed reaches gives the draft count. Measured on
# Flash-Next UD-IQ4_XS, a seed picked at 0.95 or above is accepted 95 to
# 98 percent of the time and one picked under 0.5 about 25 to 55 percent.
# The long steps pay only when the head is on a run: unless the previous
# round accepted every draft, the count is capped at GMLX_Q4_MTP_STREAK_CAP
# (0 lifts the cap). An empty ladder drafts the fixed block.
DRAFT_LADDER_DEFAULT = "0.9:5,0.5:3,0:1"
STREAK_CAP_DEFAULT = 3
BLOCK_LADDER = 6
BLOCK_FIXED = 4


def draft_ladder() -> tuple:
    """((conf, drafts), ...) highest conf first; () for the fixed block."""
    spec = os.environ.get("GMLX_Q4_MTP_LADDER", DRAFT_LADDER_DEFAULT).strip()
    if not spec or spec == "0":
        return ()
    try:
        return _parse_ladder(spec)
    except ValueError:  # a malformed value keeps the default, like env_int
        return _parse_ladder(DRAFT_LADDER_DEFAULT)


def _parse_ladder(spec: str) -> tuple:
    steps = []
    for part in spec.split(","):
        conf, drafts = part.split(":")
        steps.append((float(conf), max(1, int(drafts))))
    return tuple(sorted(steps, reverse=True))


def draft_streak_cap() -> int:
    return max(0, env_int("GMLX_Q4_MTP_STREAK_CAP", STREAK_CAP_DEFAULT))


def default_block_size() -> int:
    """The block total (drafts + bonus) the drafter requests by default."""
    ladder = draft_ladder()
    if not ladder:
        return BLOCK_FIXED
    return max(BLOCK_FIXED, 1 + max(d for _, d in ladder))

# The llama.cpp sidecar's nextn.* extras on the drafter tree. Its block
# tensors ride the text remap (parse_gguf_name) like a native head's.
QWEN4EXP_NEXTN_MAP = {
    "eh_proj": "eh_proj.weight",
    "enorm": "pre_fc_norm_embedding.weight",
    "hnorm": "pre_fc_norm_hidden.weight",
    "hc_head_norm": "hyper_connection_mixer.norm.weight",
    "hc_head_down": "hyper_connection_mixer.down.weight",
    "hc_head_up": "hyper_connection_mixer.up.weight",
}


@dataclass
class Qwen4ExpMTPConfig:
    """``text`` is the target's ModelArgs; ``block_size`` the block total
    (drafts + bonus) the engine requests by default. ``fused_eh_proj``
    builds the llama.cpp sidecar's single ``eh_proj`` in place of
    ``fc_embedding`` and ``fc_hidden``."""

    text: Any
    block_size: int = 4
    compress_ratio: int = 4
    fused_eh_proj: bool = False


@dataclass
class _SpecOutput:
    """Duck-typed output for the owned engine's ``return_hidden`` calls."""

    logits: mx.array
    hidden_states: List[mx.array]
    shared_kv_states: dict = field(default_factory=dict)
    gdn_states: Optional[list] = None


class Qwen4ExpSpecHooks:
    """The ``speculative_*`` hooks the owned MTP engine probes on the
    target's ``language_model``, shared by the text ``Qwen4ExpSpecLM`` and
    the VLM wrapper's ``LanguageModel`` (vlm_model.py) so ``--mmproj`` and
    MTP compose on one model. Contract: the carrier exposes ``self.model``
    (a ``Qwen4ExpModel``) and ``self._head``."""

    def chunked_prefill_policy(self, **kwargs):
        # The drafter teacher-forces from the retained per-chunk hiddens, so
        # chunked prefill is always safe for this target.
        return True

    def _spec_verify_positions(self, verify_input: mx.array, prompt_cache,
                               rope_deltas=None):
        # Verify-forward position seam. None keeps the scalar-offset fast
        # rope (text targets); the VLM wrapper overrides with its resolved
        # mrope block so verify after an image turn matches plain decode.
        return None

    def speculative_logits_from_hidden(self, hidden: mx.array) -> mx.array:
        """(B,S,4,D) pre-mixer streams -> (B,S,V)."""
        return self._head(self.model.hc_head(hidden))

    def speculative_argmax_from_hidden(self, hidden: mx.array) -> mx.array:
        return mx.argmax(self.speculative_logits_from_hidden(hidden), axis=-1)

    def speculative_verify_hidden(self, verify_input: mx.array, prompt_cache,
                                  rope_deltas=None):
        """The single verify forward (qL = drafts + 1). Returns the streams
        plus the rollback sink in the ``gdn_states`` slot. ``rope_deltas`` is
        the rounds loop's per-row mrope delta."""
        sink: list = []
        _, streams = self.model(
            verify_input, prompt_cache, return_streams=True, gdn_sink=sink,
            position_ids=self._spec_verify_positions(
                verify_input, prompt_cache, rope_deltas))
        return streams, {}, sink

    def rollback_speculative_cache(
        self, prompt_cache, gdn_states, accepted, block_size: int
    ) -> None:
        """Rewind every layer cache to the accepted prefix of the verify
        block: trim the KV leaves, rewind the recurrent leaves from the
        verify sink. Two-phase on the KV leaves (all trimmable before any
        mutation) since the attention mask derives from one layer's offset."""
        if isinstance(accepted, mx.array):
            accepted = int(accepted.reshape(-1)[0].item())
        accepted = int(accepted)
        rejected = int(block_size) - accepted - 1
        if rejected <= 0:
            return
        kv = [c for c in prompt_cache if c is not None and c.is_trimmable()]
        for c in kv:
            if c.trim(rejected) != rejected:
                raise RuntimeError(
                    f"qwen4exp MTP rollback: {type(c).__name__}.trim({rejected}) "
                    "refused; cache state is now inconsistent"
                )
        if gdn_states:
            q4.rollback_verify_sink(gdn_states, accepted + 1)


class Qwen4ExpSpecLM(Qwen4ExpSpecHooks, q4.Model):
    """Vendored qwen4exp ``Model`` + the shared spec hooks."""

    def _head(self, out: mx.array) -> mx.array:
        if self.args.tie_word_embeddings:
            return self.model.embed_tokens.as_linear(out)
        return self.lm_head(out)

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
        # mlx-vlm's chunked prefill calls language_model(inputs=ids, ...) by
        # keyword; shared_kv is never used (the drafter owns its KV).
        # skip_logits leaves out the head; speculative_logits_from_hidden
        # projects the returned streams.
        skip_logits = kwargs.pop("skip_logits", False)
        del n_to_process, kwargs
        want_hidden = return_hidden or return_shared_kv
        out, streams = self.model(
            inputs, cache, input_embeddings=inputs_embeds, return_streams=True)
        logits = None if skip_logits else self._head(out)
        if not want_hidden:
            from gmlx._vendor.mlx_vlm.models.base import LanguageModelOutput

            return LanguageModelOutput(logits=logits)
        return _SpecOutput(logits=logits, hidden_states=[streams])


class Qwen4ExpMTPDrafter(QwenMTPDrafter):
    """Single MTP head on the four-stream hidden; full-prompt KV
    (teacher-forced at prefill), seed precomputed at accept."""

    supports_greedy_draft_argmax = True
    prefer_requested_block_size = True
    cap_at_configured_depth = False
    uses_shared_kv = False
    supports_kv_sidecar = False
    # The 4D hidden and the sink rollback are owned-engine contracts.
    requires_owned_engine = True

    def __init__(self, config: Qwen4ExpMTPConfig):
        nn.Module.__init__(self)
        self.config = config
        args = config.text
        self._native_block_size = int(config.block_size)
        eps = args.rms_norm_eps
        D, hc = args.hidden_size, args.hc_count
        self._hc = hc
        self._eps = eps
        # One full-attention layer at index 0 of its own args: QSA at the
        # configured ratio, no PLE, same rope base as the trunk.
        head_args = replace(
            args,
            num_hidden_layers=1,
            layer_types=["full_attention"],
            compress_ratios=[int(config.compress_ratio)],
            ple_layer_ids=[],
        )
        if config.fused_eh_proj:
            self.eh_proj = nn.Linear(2 * D, D, bias=False)
        else:
            self.fc_embedding = nn.Linear(D, D, bias=False)
            self.fc_hidden = nn.Linear(D, D, bias=False)
        self.pre_fc_norm_embedding = nn.RMSNorm(D, eps=eps)
        self.pre_fc_norm_hidden = nn.RMSNorm(hc * D, eps=eps)
        self.layers = [q4.DecoderLayer(head_args, 0)]
        self.hyper_connection_mixer = q4.HyperConnection(
            D, hc, args.hc_lowrank, eps, inject=False)
        self._grouped_hnorm = env_bool("GMLX_Q4_MTP_GROUPED_HNORM", True)
        # Full-prompt seeding by default (the head attends through QSA over
        # its whole KV); GMLX_Q4_MTP_SEED_WINDOW=n keeps only the trailing
        # n prompt hiddens (20 KB/token at D=2560).
        window = env_int("GMLX_Q4_MTP_SEED_WINDOW", 0)
        self.hidden_capture_limit = window if window > 0 else None
        self._postnorm_feed = False

        self._input_embed = None
        self._input_embed_scale = 1.0
        self._lm_head_fn = None
        self._rope_delta_source = None
        self._cache: list = []
        self._seed_token = None
        self._seed_hidden = None
        self._seed_conf = None
        self._ladder = draft_ladder()
        self._streak_cap = draft_streak_cap()
        self._last_round: tuple = ()
        self._round_appended = 0
        self.accept_lens: list = []
        self.draft_lens: list = []

    # --- lifecycle ---------------------------------------------------------

    def bind(self, target_model) -> "Qwen4ExpMTPDrafter":
        super().bind(target_model)
        # A/B seam (GMLX_Q4_MTP_MROPE_DRAFT=1): draft at the target's
        # resolved mrope offset instead of the head's flat cache offset on
        # image conversations. The head is a separate model; which positions
        # it was trained against is an upstream question, so both arms stay
        # runnable and acceptance decides. Default flat (today's behavior).
        self._rope_delta_source = None
        if env_bool("GMLX_Q4_MTP_MROPE_DRAFT", False):
            lm = getattr(target_model, "language_model", target_model)
            if hasattr(lm, "_rope_deltas"):
                self._rope_delta_source = lambda: lm._rope_deltas
        return self

    def make_cache(self, left_padding: Optional[List[int]] = None) -> List[Any]:
        if left_padding is not None and (
            len(left_padding) != 1 or int(left_padding[0]) != 0
        ):
            raise NotImplementedError(
                "Qwen4ExpMTPDrafter is B=1 only (v1): the QSA key-stream "
                "cache has no batched form")
        return [q4.QSAKVCache(int(self.config.compress_ratio))]

    # --- forward primitives -------------------------------------------------

    def _hidden_norm(self, hidden: mx.array) -> mx.array:
        B, S, hc, D = hidden.shape
        if self._grouped_hnorm:
            return q4._grouped_rms_norm(
                hidden, self.pre_fc_norm_hidden.weight, self._eps)
        return self.pre_fc_norm_hidden(
            hidden.reshape(B, S, hc * D)).reshape(B, S, hc, D)

    def _forward(self, tokens: mx.array, hidden: mx.array,
                 cache: Optional[List[Any]] = None) -> mx.array:
        """One head forward over (next_token, target pre-mixer streams)
        pairs; returns the head's PRE-mixer streams ``[B,S,4,D]``. cache
        overrides the head's own list (seed streaming), never swaps it."""
        tokens = tokens.astype(mx.int32)
        if hidden.ndim != 4:
            raise ValueError(
                f"Qwen4ExpMTPDrafter expects the 4-stream hidden [B,S,hc,D], "
                f"got shape {tuple(hidden.shape)}")
        e = self.pre_fc_norm_embedding(
            self._input_embed(tokens) * self._input_embed_scale)
        h = self._hidden_norm(hidden)
        if "eh_proj" in self:
            e = mx.broadcast_to(e[:, :, None, :], h.shape)
            x = self.eh_proj(mx.concatenate([e, h], axis=-1))
        else:
            x = self.fc_hidden(h) + self.fc_embedding(e)[:, :, None, :]
        c = (self._cache if cache is None else cache)[0]
        mask = create_attention_mask(x[:, :, 0, :], c)
        return self.layers[0](x, tokens, mask=mask, cache=c,
                              positions=self._draft_positions(tokens, c))

    # --- draft length policy ----------------------------------------------

    @staticmethod
    def _conf(logits: mx.array) -> mx.array:
        """The argmax probability of each row, [B]."""
        lp = logits.astype(mx.float32)
        lp = lp - mx.logsumexp(lp, axis=-1, keepdims=True)
        return mx.exp(lp.max(axis=-1)).reshape(-1)

    def reset(self, target_model, left_padding=None):
        self._seed_conf = None
        self._last_round = ()
        return super().reset(target_model, left_padding)

    def _set_seed(self, h_prenorm: mx.array, sampler, greedy: bool) -> None:
        logits = self._logits(h_prenorm)
        self._seed_token = self._pick(logits, sampler, greedy)
        self._seed_hidden = self._next_hidden(h_prenorm)
        self._seed_conf = self._conf(logits) if self._ladder else None

    def draft_eval_state(self) -> list:
        # The seed's confidence shares the seed's graph, so the engine's
        # async eval of the seed covers it and the ladder read waits on
        # nothing more.
        state = super().draft_eval_state()
        if self._seed_conf is not None:
            state.append(self._seed_conf)
        return state

    def accept_verified_tokens(self, verify_hidden, draft_tokens, accepted,
                               new_tokens, sampler, token_dtype=mx.int32,
                               greedy=False):
        self._last_round = (int(draft_tokens.shape[1]), int(accepted))
        return super().accept_verified_tokens(
            verify_hidden, draft_tokens, accepted, new_tokens, sampler,
            token_dtype, greedy)

    def _draft_count(self, conf: Optional[mx.array], block_size: int) -> int:
        """Drafts this round, seed included, at most block_size - 1."""
        n = block_size - 1
        if self._ladder and conf is not None:
            c = float(conf[0].item())
            for step_conf, drafts in self._ladder:
                if c >= step_conf:
                    n = min(n, drafts)
                    break
        if self._streak_cap and self._last_round:
            drafted, accepted = self._last_round
            if accepted < drafted:
                n = min(n, self._streak_cap)
        return max(1, n)

    def draft_block(self, last_bonus, hidden: mx.array, cache, block_size: int,
                    sampler, token_dtype: mx.Dtype = mx.int32,
                    greedy: bool = False) -> mx.array:
        """The seed, then the rollouts the policy allows, all left lazy for
        the engine's async eval."""
        if not self._ladder or self._seed_token is None \
                or self._seed_hidden is None:
            return super().draft_block(last_bonus, hidden, cache, block_size,
                                       sampler, token_dtype, greedy)
        del cache
        tok = self._seed_token.astype(token_dtype)
        h_prev = self._seed_hidden
        conf = self._seed_conf
        self._seed_token = None
        self._seed_hidden = None
        self._seed_conf = None
        self._round_appended = 0
        # Wait for the seed (see QwenMTPDrafter.draft_block); the seed's
        # confidence was evaluated with it.
        mx.eval(tok, h_prev)
        n = self._draft_count(conf, block_size)
        tokens = [tok]
        while len(tokens) < n:
            h = self._forward(tok, h_prev)
            self._round_appended += 1
            tok = self._pick(self._logits(h), sampler, greedy)
            h_prev = self._next_hidden(h)
            tokens.append(tok)
        return mx.concatenate(tokens, axis=1)

    def _draft_positions(self, tokens: mx.array, cache):
        """None (flat cache-offset rope, the default arm) unless the mrope
        A/B arm is bound and the target resolved a rope delta this request."""
        if self._rope_delta_source is None:
            return None
        delta = self._rope_delta_source()
        if delta is None:
            return None
        B, S = tokens.shape
        pos = cache.offset + delta.astype(mx.int32).reshape(B, 1) \
            + mx.arange(S, dtype=mx.int32)[None]
        return mx.broadcast_to(pos[None], (3, B, S))

    def _logits(self, h: mx.array) -> mx.array:
        return self._lm_head_fn(self.hyper_connection_mixer(h))

    def _next_hidden(self, h: mx.array) -> mx.array:
        # The mixer replaces the final norm; the rollout feeds the
        # pre-mixer streams exactly like the target seed.
        return h

    def sanitize(self, weights: dict) -> dict:
        return {
            (k[len("mtp."):] if k.startswith("mtp.") else k): v
            for k, v in weights.items()
        }


def remap_qwen4exp_mtp_arrays(arrays: dict, kquant_meta: dict):
    """Companion-GGUF names are the drafter tree under ``mtp.``; strip the
    prefix and thread the kquant codecs (``.scales`` placeholders included)
    the way the text remap does. Returns ``(weights, kquant_meta, stats)``."""
    weights: dict = {}
    codecs: dict = {}
    stats = {"mapped": 0, "kquant": 0, "skipped": 0}
    for name, arr in arrays.items():
        if name.endswith(".scales") or name.endswith(".biases"):
            continue
        if not name.startswith("mtp."):
            stats["skipped"] += 1
            continue
        hf = name[len("mtp."):]
        if hf.endswith(".input_scale"):
            stats["skipped"] += 1
            continue
        if hf.endswith(".scale"):
            # A weight scale goes to the projection it scales, quantized or
            # float; a norm has none.
            w = arrays.get(name[: -len("scale")] + "weight")
            if w is None or w.ndim < 2:
                stats["skipped"] += 1
            else:
                weights[hf[: -len("scale")] + "out_scale"] = arr.reshape(-1)
                stats["mapped"] += 1
            continue
        weights[hf] = arr
        stats["mapped"] += 1
        codec = kquant_meta.get(name)
        if codec is not None:
            base = hf[:-len(".weight")] if hf.endswith(".weight") else hf
            src = name[:-len(".weight")] if name.endswith(".weight") else name
            weights[base + ".scales"] = arrays.get(
                src + ".scales", mx.zeros((1,), dtype=mx.uint8))
            codecs[hf] = codec
            stats["kquant"] += 1
    return weights, codecs, stats
