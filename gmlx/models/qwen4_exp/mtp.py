# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Asher Feldman
"""Qwen3.8-Flash-Next (qwen4exp) MTP: speculative target wrapper + drafter.

The MTP head ships only in the HF safetensors (``mtp.*``; the llama.cpp
converter drops it), so it reaches gmlx as a companion GGUF (arch
``qwen4exp-mtp``, built by the lab's ``extract-qwen4exp-mtp-gguf.py``) that
the loader autodetects next to the target. Reference forward (ollama
``x/models/qwen4_exp`` mtpDraft.Forward):

    e = fc_embedding(pre_fc_norm_embedding(embed(next_token)))      [B,S,D]
    x = fc_hidden(pre_fc_norm_hidden(h_4d)) + e[:, :, None, :]       [B,S,4,D]
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

from dataclasses import dataclass, field, replace
from typing import Any, List, Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.base import create_attention_mask

from . import model as q4
from gmlx.envflags import env_bool, env_int
from gmlx.spec.mtp_drafter import QwenMTPDrafter

MTP_ARCH = "qwen4exp-mtp"


@dataclass
class Qwen4ExpMTPConfig:
    """``text`` is the target's ModelArgs; ``block_size`` the block total
    (drafts + bonus) the engine requests by default."""

    text: Any
    block_size: int = 4
    compress_ratio: int = 4


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

    def _spec_verify_positions(self, verify_input: mx.array, prompt_cache):
        # Verify-forward position seam. None keeps the scalar-offset fast
        # rope (text targets); the VLM wrapper overrides with its resolved
        # mrope block so verify after an image turn matches plain decode.
        return None

    def speculative_logits_from_hidden(self, hidden: mx.array) -> mx.array:
        """(B,S,4,D) pre-mixer streams -> (B,S,V)."""
        return self._head(self.model.hc_head(hidden))

    def speculative_argmax_from_hidden(self, hidden: mx.array) -> mx.array:
        return mx.argmax(self.speculative_logits_from_hidden(hidden), axis=-1)

    def speculative_verify_hidden(self, verify_input: mx.array, prompt_cache):
        """The single verify forward (qL = drafts + 1). Returns the streams
        plus the rollback sink in the ``gdn_states`` slot."""
        sink: list = []
        _, streams = self.model(
            verify_input, prompt_cache, return_streams=True, gdn_sink=sink,
            position_ids=self._spec_verify_positions(verify_input, prompt_cache))
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
        del n_to_process, kwargs
        want_hidden = return_hidden or return_shared_kv
        out, streams = self.model(
            inputs, cache, input_embeddings=inputs_embeds, return_streams=True)
        logits = self._head(out)
        if not want_hidden:
            from mlx_vlm.models.base import LanguageModelOutput

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

    def _forward(self, tokens: mx.array, hidden: mx.array) -> mx.array:
        """One head forward over (next_token, target pre-mixer streams)
        pairs; returns the head's PRE-mixer streams ``[B,S,4,D]``."""
        tokens = tokens.astype(mx.int32)
        if hidden.ndim != 4:
            raise ValueError(
                f"Qwen4ExpMTPDrafter expects the 4-stream hidden [B,S,hc,D], "
                f"got shape {tuple(hidden.shape)}")
        e = self.fc_embedding(self.pre_fc_norm_embedding(
            self._input_embed(tokens) * self._input_embed_scale))
        x = self.fc_hidden(self._hidden_norm(hidden)) + e[:, :, None, :]
        cache = self._cache[0]
        mask = create_attention_mask(x[:, :, 0, :], cache)
        return self.layers[0](x, tokens, mask=mask, cache=cache,
                              positions=self._draft_positions(tokens, cache))

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
