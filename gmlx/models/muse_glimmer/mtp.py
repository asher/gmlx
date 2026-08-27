# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Asher Feldman
"""Muse Glimmer speculative target: the ``speculative_*`` hooks plus the
packed-hidden capture the DFlash drafter reads.

The drafter (``muse_glimmer_dflash``) consumes the target's residual stream at
five layers. llama.cpp exposes those as ``t_layer_inp[il]`` - the residual
*entering* layer ``il`` - for ``il`` in the GGUF's ``dflash.target_layers``
([2, 14, 26, 38, 50]); the converter writes those one higher than the HF
``target_layer_ids`` because HF records layer *outputs*. Entering layer 14 is
leaving layer 13, so the capture set here is the GGUF list minus one, taken as
layer outputs (:func:`MuseGlimmerModel.__call__`'s ``capture_layers``).

Capture rides the DeepSeek-V4 packed-hidden seam rather than mlx-vlm's
``capture_layer_ids``/``hidden_states`` route: every engine-facing hidden is
widened to ``[trunk | cap_1 | cap_13 | ... ]`` so the existing slicing and
capture-trim seams work untouched and no engine change is needed. The drafter
unpacks the trailing ``n_targets*hidden``; the logits hooks slice the lead.

Rollback needs no undo log here. The sliding layers hold ``keep=0``
``RotatingKVCache`` leaves, which ``_buffer_mtp_target_cache`` swaps for
``BufferedRotatingKVCache`` before the decode loop - that cache keeps
rollback slack past the window edge, so ``is_trimmable()`` holds however deep
the context runs (a rotated stock ring would refuse, since the evicted slot is
gone).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional

import mlx.core as mx

from . import model as mg


@dataclass
class _SpecOutput:
    """Duck-typed output for the owned engine's ``return_hidden`` calls."""

    logits: mx.array
    hidden_states: List[mx.array]
    shared_kv_states: dict = field(default_factory=dict)
    gdn_states: Optional[list] = None


class SpecHooks:
    """The ``speculative_*`` hook set, shared by the text MTP target and the
    VLM ``language_model`` so ``--mmproj`` and ``--speculative`` compose.

    Requires ``self.model`` (a :class:`MuseGlimmerModel`) and a logits tail;
    both hosts carry the model's ``args`` on ``self.model.args``.
    """

    # Set by the drafter loader: the trunk layer ids whose outputs the drafter
    # consumes. While set, every engine-facing hidden is packed.
    _dflash_capture: Optional[tuple] = None

    def set_dflash_capture(self, layer_ids) -> None:
        """Arm packed-hidden capture on the given trunk layer outputs.

        Assigned around ``nn.Module.__setattr__``, which routes tuples into
        the parameter dict - where this class default would shadow the entry
        on read, and tree walks would meet non-array leaves."""
        object.__setattr__(self, "_dflash_capture", tuple(layer_ids))

    def _spec_logits(self, h: mx.array) -> mx.array:
        args = self.model.args
        out = (self.model.embed_tokens.as_linear(h) if args.tie_word_embeddings
               else self.lm_head(h))
        return mg.scale_and_softcap(
            out, args.output_multiplier, args.final_logit_softcapping)

    def _dflash_pack(self, h: mx.array, captures) -> mx.array:
        return mx.concatenate([h, *captures], axis=-1)

    def _dflash_trunk(self, hidden: mx.array) -> mx.array:
        if self._dflash_capture is None:
            return hidden
        # The trunk lead is a strided view of the packed hidden, and the logit
        # tail is a quantized matmul, whose kernel reads the buffer directly.
        # Slicing lazily hands it the packed strides and it reads the wrong
        # rows, so materialize before the head sees it.
        return mx.contiguous(hidden[..., : self.model.args.hidden_size])

    def chunked_prefill_policy(self, **kwargs):
        # Stock mlx-vlm disables chunked prefill whenever a drafter is
        # attached. The DFlash drafter is window-limited
        # (hidden_capture_limit trailing positions), so last-chunk capture
        # suffices and chunking stays safe.
        return True

    def speculative_logits_from_hidden(self, hidden: mx.array) -> mx.array:
        return self._spec_logits(self._dflash_trunk(hidden))

    def speculative_argmax_from_hidden(self, hidden: mx.array) -> mx.array:
        return mx.argmax(self.speculative_logits_from_hidden(hidden), axis=-1)

    def speculative_verify_hidden(self, verify_input: mx.array, prompt_cache):
        """The single verify forward (qL = drafts + 1): trunk only, no head -
        the walk computes logits/argmax from the returned hidden."""
        if self._dflash_capture is not None:
            h, caps = self.model(
                verify_input, prompt_cache, capture_layers=self._dflash_capture)
            return self._dflash_pack(h, caps), {}
        return self.model(verify_input, prompt_cache), {}

    def rollback_speculative_cache(
        self, prompt_cache, gdn_states, accepted: int, block_size: int
    ) -> None:
        """Trim the rejected verify tail from every layer cache, two-phase:
        verify ALL are trimmable before mutating ANY (the shared attention
        mask is built from one layer's offset, so a partial rollback would
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
                f"Muse Glimmer MTP rollback: untrimmable cache leaves "
                f"{refused} (rejected={rejected}); the sliding leaves must be "
                f"BufferedRotatingKVCache by now"
            )
        for c in prompt_cache:
            if c.trim(rejected) != rejected:
                raise RuntimeError(
                    f"Muse Glimmer MTP rollback: {type(c).__name__}.trim"
                    f"({rejected}) refused after is_trimmable() - cache state "
                    f"is now inconsistent"
                )


class MuseGlimmerSpecLM(SpecHooks, mg.Model):
    """Vendored Muse Glimmer ``Model`` + the speculative hooks, in the shape
    the owned MTP engine drives (``model.language_model``)."""

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
        # keyword. shared_kv is never used (the drafter owns its KV).
        del n_to_process, kwargs
        want_hidden = return_hidden or return_shared_kv
        if want_hidden and self._dflash_capture is not None:
            h, caps = self.model(
                inputs, cache, capture_layers=self._dflash_capture,
                inputs_embeds=inputs_embeds)
            return _SpecOutput(logits=self._spec_logits(h),
                               hidden_states=[self._dflash_pack(h, caps)])
        h = self.model(inputs, cache, inputs_embeds=inputs_embeds)
        logits = self._spec_logits(h)
        if not want_hidden:
            from mlx_vlm.models.base import LanguageModelOutput

            return LanguageModelOutput(logits=logits)
        return _SpecOutput(logits=logits, hidden_states=[h])
