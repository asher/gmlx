# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Asher Feldman
"""GLM-5.3-Flash MTP: speculative target wrapper + native-head drafter.

The GGUF carries its MTP head in-file as block ``num_hidden_layers`` (45):
the four ``nextn.*`` extras (``eh_proj``/``enorm``/``hnorm``/
``shared_head_norm``) plus one full DSA decoder layer (nope MLA + pooled
indexer + sigmoid MoE) with a PLAIN residual - block 45 ships no
hyper-connection tensors. ``loader.remap_mtp_arrays`` maps the extras onto
the Qwen drafter tree (``fc`` / ``pre_fc_norm_embedding`` /
``pre_fc_norm_hidden`` / ``norm``) and the layer onto ``layers.0``, so the
drafter subclasses ``QwenMTPDrafter`` and swaps the transformer block for a
plain-residual GLM DSA layer. Head forward (upstream safetensors layout;
neither reference runtime implements MTP decode):

    x = eh_proj(concat[enorm(embed(ids)), hnorm(h)])
    h_out = layer(x); logits = lm_head(shared_head_norm(h_out))

``h`` is the trunk's COLLAPSED pre-final-norm hidden [B, S, D] (the stream
mean - the MTP layer is single-stream). The rollout chains the head's own
pre-norm output by default; ``GMLX_MTP_POSTNORM_FEED=1`` A/Bs the post-norm
feed (the DeepSeek-V3 pre-norm convention is unverified here - acceptance
rate is the arbiter).

Verify rollback is two-part (the qwen4_exp recurrent precedent): the MLA
layers' KV + pool leaves trim (PoolingCache stashes a one-update undo for
L <= 6), and the KDA layers replay the accepted prefix from the recorded
pre-verify conv tails + recurrent state (``model.rollback_verify_sink``).

Correctness is drafter-independent: the verify walk emits the target's own
greedy/sampled tokens, so the drafter affects speed (acceptance), never
output. The losslessness gate is the greedy A/B vs plain decode.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.base import create_attention_mask

from . import model as glm5
from gmlx.spec.drafter_protocol import native_block_size
from gmlx.spec.mtp_drafter import _POSTNORM_FEED, QwenMTPDrafter


@dataclass
class Glm5NextMTPConfig:
    """Drafter config. ``text_config`` is the target's ModelArgs; the MTP
    layer is a full-attention MoE layer (index 45 >= the dense prefix).
    ``block_size`` is the block TOTAL (drafts + bonus)."""

    text_config: Any
    block_size: int = 2


@dataclass
class _SpecOutput:
    """Duck-typed output for the owned engine's ``return_hidden`` calls."""

    logits: mx.array
    hidden_states: List[mx.array]
    shared_kv_states: dict = field(default_factory=dict)
    gdn_states: Optional[list] = None


def _collect_kv_leaves(prompt_cache: List[Any]) -> List[Any]:
    """Flatten the trimmable attention-side leaves: CacheList children
    (latent KV + pool caches). KDA ``ArraysCache`` entries are skipped -
    the verify sink rewinds those."""
    leaves: List[Any] = []
    for entry in prompt_cache:
        children = getattr(entry, "caches", None)
        if children is None:
            continue
        for child in children:
            if child is not None:
                leaves.append(child)
    return leaves


class Glm5NextSpecHooks:
    """The ``speculative_*`` hooks the owned MTP engine probes on its
    target. Mixed into both spec targets - the text-only ``Glm5NextSpecLM``
    and the VLM container's ``LanguageModel`` - which share the attribute
    layout the hooks read (``self.model`` backbone, ``self.lm_head``).

    ``hidden_states`` is the collapsed PRE-final-norm trunk hidden - what
    the drafter's ``hnorm`` consumes; the from-hidden hooks apply the final
    norm before the head."""

    def _head(self, out: mx.array) -> mx.array:
        if self.lm_head is None:
            return self.model.embed_tokens.as_linear(out)
        return self.lm_head(out)

    def chunked_prefill_policy(self, **kwargs):
        # The drafter teacher-forces from the retained per-chunk hiddens,
        # so chunked prefill is always safe for this target.
        return True

    # --- speculative hooks (owned engine contract) -------------------------

    def speculative_logits_from_hidden(self, hidden: mx.array) -> mx.array:
        return self._head(self.model.norm(hidden))

    def speculative_argmax_from_hidden(self, hidden: mx.array) -> mx.array:
        return mx.argmax(self.speculative_logits_from_hidden(hidden), axis=-1)

    def speculative_verify_hidden(self, verify_input: mx.array, prompt_cache):
        """The single verify forward (qL = drafts + 1): trunk only, with the
        KDA rollback sink recorded (the PoolingCaches stash their own undo
        unconditionally for L <= 6)."""
        sink: list = []
        _, raw = self.model(
            verify_input, prompt_cache, return_raw_hidden=True, gdn_sink=sink)
        return raw, {}, sink

    def rollback_speculative_cache(
        self, prompt_cache, gdn_states, accepted, block_size: int
    ) -> None:
        """Rewind every layer cache to the accepted prefix of the verify
        block: trim the MLA KV/pool leaves (two-phase - all trimmable before
        any mutation, since the shared attention mask derives from one
        layer's offset), then replay the KDA layers from the sink."""
        if isinstance(accepted, mx.array):
            accepted = int(accepted.reshape(-1)[0].item())
        accepted = int(accepted)
        rejected = int(block_size) - accepted - 1
        if rejected <= 0:
            return
        leaves = _collect_kv_leaves(prompt_cache)
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
                f"glm5_next MTP rollback: untrimmable cache leaves {refused} "
                f"(rejected={rejected}); undo log missing or consumed")
        for leaf in leaves:
            if leaf.trim(rejected) != rejected:
                raise RuntimeError(
                    f"glm5_next MTP rollback: {type(leaf).__name__}.trim"
                    f"({rejected}) refused after is_trimmable() - cache "
                    f"state is now inconsistent")
        if gdn_states:
            glm5.rollback_verify_sink(gdn_states, accepted + 1)


class Glm5NextSpecLM(Glm5NextSpecHooks, glm5.Model):
    """Vendored glm5_next ``Model`` + the speculative hooks, for the
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
        # mlx-vlm's chunked prefill calls language_model(inputs=ids, ...) by
        # keyword; the token ids are authoritative (a GGUF text target has
        # no vision features) and shared_kv is never used.
        del inputs_embeds, n_to_process, kwargs
        normed, raw = self.model(inputs, cache, return_raw_hidden=True)
        logits = self._head(normed)
        if not (return_hidden or return_shared_kv):
            from mlx_vlm.models.base import LanguageModelOutput

            return LanguageModelOutput(logits=logits)
        return _SpecOutput(logits=logits, hidden_states=[raw])


class Glm5NextMTPLayer(nn.Module):
    """The MTP decoder layer: the trunk's DSA attention + MoE with a PLAIN
    residual (GGUF block 45 has no hyper-connection tensors)."""

    def __init__(self, args):
        super().__init__()
        self.self_attn = glm5.Glm5NextMLAAttention(args)
        self.mlp = glm5.Glm5NextMoE(args)
        self.input_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps)

    def __call__(self, x: mx.array, mask, cache) -> mx.array:
        x = x + self.self_attn(self.input_layernorm(x), mask=mask, cache=cache)
        return x + self.mlp(self.post_attention_layernorm(x))


class Glm5NextMTPDrafter(QwenMTPDrafter):
    """Qwen drafter algorithm over the GLM MTP block (full-prompt KV,
    teacher-forced at prefill; decode-time-only trim/accept). Nope MLA
    attends its whole KV, so prompt seeding is uncapped."""

    prefer_requested_block_size = False
    cap_at_configured_depth = True
    # The pool-cache clone path is untested; no sidecar snapshots.
    supports_kv_sidecar = False
    # CLI entry points must route to the owned engine: mlx-vlm's stock MTP
    # round doesn't know the glm5_next target hooks or the KDA sink.
    requires_owned_engine = True

    def __init__(self, config: Glm5NextMTPConfig):
        nn.Module.__init__(self)
        if int(config.block_size) != 2:
            # The accept-path trim rewinds the drafter's own PoolingCache
            # through its one-update undo log; a deeper rollout appends one
            # update per draft and cannot rewind past the last one.
            raise ValueError(
                "Glm5NextMTPDrafter supports block_size=2 only (v1)")
        self.config = config
        self._native_block_size = (
            native_block_size(config) or int(config.block_size))
        args = config.text_config

        hidden_size = args.hidden_size
        eps = args.rms_norm_eps
        self.fc = nn.Linear(2 * hidden_size, hidden_size, bias=False)
        self.pre_fc_norm_embedding = nn.RMSNorm(hidden_size, eps=eps)
        self.pre_fc_norm_hidden = nn.RMSNorm(hidden_size, eps=eps)
        self.layers = [Glm5NextMTPLayer(args)]
        self.norm = nn.RMSNorm(hidden_size, eps=eps)
        self._kpool = args.index_kpool
        self._postnorm_feed = _POSTNORM_FEED

        # Bound to the target at reset(): the head shares embeddings + head.
        self._input_embed = None
        self._input_embed_scale: float = 1.0
        self._lm_head_fn = None

        # Decode-time-only state: own KV + the precomputed next-round seed.
        self._cache: List[Any] = []
        self._seed_token: Optional[mx.array] = None
        self._seed_hidden: Optional[mx.array] = None
        self._round_appended = 0

        self.accept_lens: List[float] = []
        self.draft_lens: List[int] = []

    def make_cache(self, left_padding: Optional[List[int]] = None) -> List[Any]:
        if left_padding is not None:
            raise NotImplementedError(
                "Glm5NextMTPDrafter is B=1 only (v1): the pooled indexer "
                "selection is single-sequence")
        from mlx_lm.models.cache import CacheList as _CacheList
        from mlx_lm.models.cache import KVCache as _KVCache

        from gmlx.cache.compat import construction_cache_module
        from gmlx.models.deepseek_v4.cache import PoolingCache

        cmod = construction_cache_module()
        kv_cls = getattr(cmod, "KVCache", _KVCache)
        list_cls = getattr(cmod, "CacheList", _CacheList)
        pool = PoolingCache(self._kpool, lookback=False)
        pool.quantizable = False
        return [list_cls(kv_cls(), pool)]

    def inject_rows(self, *args, **kwargs) -> None:
        raise NotImplementedError(
            "Glm5NextMTPDrafter is B=1 only (v1): no batched-row injection")

    def _forward(self, tokens: mx.array, hidden: mx.array) -> mx.array:
        """Run the head over (tokens, target pre-norm collapsed hidden);
        return the PRE-shared-head-norm output. The nope MLA has no rope
        frame; the mask offset comes from the head's own KV length."""
        embed = self._input_embed(tokens.astype(mx.int32))
        h = mx.concatenate(
            [self.pre_fc_norm_embedding(embed),
             self.pre_fc_norm_hidden(hidden)],
            axis=-1,
        )
        h = self.fc(h)
        cache = self._cache[0] if self._cache else None
        kv = cache[0] if cache is not None else None
        mask = create_attention_mask(h, kv, return_array=True)
        return self.layers[0](h, mask, cache)
