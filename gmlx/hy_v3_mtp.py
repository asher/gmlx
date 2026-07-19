"""Tencent Hy3 MTP: speculative target wrapper + native-head drafter.

The Hy3 GGUF carries its MTP head in-file as block ``num_hidden_layers`` (80):
the four ``nextn.*`` extras (``eh_proj``/``enorm``/``hnorm``/
``shared_head_norm``) plus one full decoder layer (GQA attention + MoE).
``loader.remap_mtp_arrays`` maps them onto the Qwen drafter tree
(``fc`` / ``pre_fc_norm_embedding`` / ``pre_fc_norm_hidden`` / ``layers.0`` /
``norm``), so the drafter here subclasses ``QwenMTPDrafter`` and only swaps
the transformer block for the vendored Hy3 ``DecoderLayer`` (the GGUF has no
``nextn.embed_tokens`` / ``shared_head_head``; both bind from the target).

Reference forward (vLLM hy_v3_mtp.py, verified by llama.cpp PR #25395):

    x = eh_proj(concat[enorm(embed(ids)), hnorm(h_postnorm)])
    h = final_layernorm(layer(x)); logits = lm_head(h)

Two Hy3-specific facts the overrides encode:

- The head consumes the target's POST-final-norm hidden, and its own output
  is post-``final_layernorm`` -- so the chained state for a deeper draft step
  is likewise post-norm. ``_next_hidden`` is therefore ``norm(h)`` always
  (no ``GMLX_MTP_POSTNORM_FEED`` A/B: pre-norm feed would be wrong, not
  slower).
- The head is trained single-depth (measured per-position acceptance
  0.878 / 0.224 / 0.010), so ``block_size`` defaults to 2 (one draft + bonus,
  zero draft-time forwards: the seed is precomputed at accept) and the
  drafter caps at the configured depth. ``GMLX_HY3_MTP_BLOCK`` raises it
  for measurement.

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

from . import hy_v3_model as hy
from .mtp_drafter import QwenMTPDrafter


@dataclass
class HyV3MTPConfig:
    """Drafter config. ``text_config`` is the target's ModelArgs; the MTP
    decoder layer is built at ``layer_idx = num_hidden_layers`` (>= the dense
    prefix, so it is a MoE layer like GGUF block 80). ``block_size`` is the
    block TOTAL (drafts + bonus)."""

    text_config: Any
    block_size: int = 2


@dataclass
class _SpecOutput:
    """Duck-typed output for the owned engine's ``return_hidden`` calls."""

    logits: mx.array
    hidden_states: List[mx.array]
    shared_kv_states: dict = field(default_factory=dict)
    gdn_states: Optional[list] = None


class HyV3SpecLM(hy.Model):
    """Vendored Hy3 ``Model`` + the ``speculative_*`` hooks the owned MTP
    engine probes on the target's ``language_model``.

    ``hidden_states`` is the POST-final-norm trunk output -- exactly what the
    drafter's ``hnorm`` consumes and what ``_logits`` projects, so the
    from-hidden hooks need no extra norm."""

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
        # mlx-vlm's chunked prefill calls language_model(inputs=ids,
        # inputs_embeds=..., ...) by keyword; the token ids are authoritative
        # (a GGUF text target has no vision features) and shared_kv is never
        # used (the drafter owns its KV).
        del inputs_embeds, n_to_process, kwargs
        h = self.model(inputs, cache)
        logits = self._logits(h)
        if not (return_hidden or return_shared_kv):
            from mlx_vlm.models.base import LanguageModelOutput

            return LanguageModelOutput(logits=logits)
        return _SpecOutput(logits=logits, hidden_states=[h])

    def make_cache(self):
        from mlx_lm.models.cache import KVCache

        return [KVCache() for _ in self.model.layers]

    # --- speculative hooks (owned engine contract) ---------------------------

    def speculative_logits_from_hidden(self, hidden: mx.array) -> mx.array:
        return self._logits(hidden)

    def speculative_argmax_from_hidden(self, hidden: mx.array) -> mx.array:
        return mx.argmax(self._logits(hidden), axis=-1)

    def speculative_verify_hidden(self, verify_input: mx.array, prompt_cache):
        """The single verify forward (qL = drafts + 1): trunk only, no
        lm_head -- the walk computes logits/argmax from the hidden."""
        return self.model(verify_input, prompt_cache), {}

    def rollback_speculative_cache(
        self, prompt_cache, gdn_states, accepted: int, block_size: int
    ) -> None:
        """Trim the rejected verify tail from every layer cache, two-phase:
        verify ALL are trimmable before mutating ANY (the shared attention
        mask is built from layer 0's offset, so a partial rollback would
        desync layers and corrupt decode)."""
        del gdn_states
        rejected = int(block_size) - int(accepted) - 1
        if rejected <= 0:
            return
        refused = [
            type(c).__name__ for c in prompt_cache if not c.is_trimmable()
        ]
        if refused:
            raise RuntimeError(
                f"Hy3 MTP rollback: untrimmable cache leaves {refused} "
                f"(rejected={rejected})"
            )
        for c in prompt_cache:
            if c.trim(rejected) != rejected:
                raise RuntimeError(
                    f"Hy3 MTP rollback: {type(c).__name__}.trim({rejected}) "
                    f"refused after is_trimmable() -- cache state is now "
                    f"inconsistent"
                )


class HyV3MTPDrafter(QwenMTPDrafter):
    """Qwen drafter algorithm over the Hy3 MTP block (full-prompt KV,
    teacher-forced at prefill; decode-time-only trim/accept)."""

    prefer_requested_block_size = False
    cap_at_configured_depth = True
    # CLI entry points must route to the owned engine: mlx-vlm's stock MTP
    # round doesn't know the hy_v3 target hooks.
    requires_owned_engine = True

    def __init__(self, config: HyV3MTPConfig):
        nn.Module.__init__(self)
        self.config = config
        self._native_block_size = int(config.block_size)
        args = config.text_config

        hidden_size = args.hidden_size
        eps = args.rms_norm_eps
        self.fc = nn.Linear(2 * hidden_size, hidden_size, bias=False)
        self.pre_fc_norm_embedding = nn.RMSNorm(hidden_size, eps=eps)
        self.pre_fc_norm_hidden = nn.RMSNorm(hidden_size, eps=eps)
        self.layers = [hy.DecoderLayer(args, layer_idx=args.num_hidden_layers)]
        self.norm = nn.RMSNorm(hidden_size, eps=eps)

        # Bound to the target at reset(): the head shares embeddings + LM head.
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
                "HyV3MTPDrafter is B=1 only (v1): the Hy3 attention takes a "
                "scalar rope offset, not per-row BatchKVCache offsets"
            )
        return super().make_cache()

    def inject_rows(self, *args, **kwargs) -> None:
        raise NotImplementedError(
            "HyV3MTPDrafter is B=1 only (v1): no batched-row injection"
        )

    def _forward(self, tokens: mx.array, hidden: mx.array) -> mx.array:
        """Run the head over (tokens, target post-norm hidden); return the
        PRE-final-norm output. Positions come from the head's own cache
        offset, so its RoPE frame is its decode-time KV length."""
        embed = self._input_embed(tokens.astype(mx.int32))
        h = mx.concatenate(
            [self.pre_fc_norm_embedding(embed), self.pre_fc_norm_hidden(hidden)],
            axis=-1,
        )
        h = self.fc(h)
        for layer, layer_cache in zip(self.layers, self._cache):
            mask = (
                create_attention_mask(h, layer_cache)
                if layer_cache is not None
                else ("causal" if h.shape[1] > 1 else None)
            )
            h = layer(h, mask, cache=layer_cache)
        return h

    def _next_hidden(self, h_prenorm: mx.array) -> mx.array:
        # Hy3 chains the POST-final_layernorm state (vLLM reference; module
        # note): the seed hidden from the target is post-norm too, so every
        # rollout step feeds norm(h).
        return self.norm(h_prenorm)
