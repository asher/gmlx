# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Asher Feldman
"""Ripple speculative target: the ``speculative_*`` hooks plus the
packed-hidden capture that the DFlash2 drafter reads.

The drafter reads the residual stream at five layers. The GGUF's
``dflash.target_layers`` names the layer inputs, one higher than the HF
``target_layer_ids``. The loader subtracts one, so the capture set here is
a set of layer outputs (``capture_layers`` in :class:`RicursiveModel`).

Each engine-facing hidden is packed as ``[trunk | cap_1 | ... ]``, as in
Muse Glimmer. The drafter reads the trailing taps and the logit hooks read
the lead.

The value residual needs nothing here: x0 is the scaled embedding of the
verify input itself, so each verify forward rebuilds it.

Rollback needs no undo log. ``_buffer_mtp_target_cache`` swaps the sliding
``RotatingKVCache`` leaves for ``BufferedRotatingKVCache`` before decode,
and that cache keeps rollback slack past the window edge.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional

import mlx.core as mx

from . import model as rm


@dataclass
class _SpecOutput:
    """Output for the owned engine's ``return_hidden`` calls."""

    logits: mx.array
    hidden_states: List[mx.array]
    shared_kv_states: dict = field(default_factory=dict)
    gdn_states: Optional[list] = None


class SpecHooks:
    """The ``speculative_*`` hook set. Needs ``self.model`` (a
    :class:`RicursiveModel`) and ``self.head``."""

    # Set by the drafter loader. While set, every engine-facing hidden is
    # packed.
    _dflash_capture: Optional[tuple] = None

    def set_dflash_capture(self, layer_ids) -> None:
        # nn.Module.__setattr__ would put the tuple in the parameter dict.
        object.__setattr__(self, "_dflash_capture", tuple(layer_ids))

    def _dflash_pack(self, h: mx.array, captures) -> mx.array:
        return mx.concatenate([h, *captures], axis=-1)

    def _dflash_trunk(self, hidden: mx.array) -> mx.array:
        if self._dflash_capture is None:
            return hidden
        # The quantized head reads its input buffer directly, so a strided
        # view of the packed hidden must be made contiguous first.
        return mx.contiguous(hidden[..., : self.model.args.hidden_size])

    def chunked_prefill_policy(self, **kwargs):
        # The drafter reads only hidden_capture_limit trailing rows, so
        # capture from the last chunk is enough.
        return True

    def speculative_logits_from_hidden(self, hidden: mx.array) -> mx.array:
        return self.head(self._dflash_trunk(hidden))

    def speculative_argmax_from_hidden(self, hidden: mx.array) -> mx.array:
        return mx.argmax(self.speculative_logits_from_hidden(hidden), axis=-1)

    def speculative_verify_hidden(self, verify_input: mx.array, prompt_cache):
        """One verify forward (drafts + 1 rows), trunk only."""
        if self._dflash_capture is not None:
            h, caps = self.model(
                verify_input, prompt_cache, capture_layers=self._dflash_capture)
            return self._dflash_pack(h, caps), {}
        return self.model(verify_input, prompt_cache), {}

    def rollback_speculative_cache(
        self, prompt_cache, gdn_states, accepted: int, block_size: int
    ) -> None:
        """Trim the rejected verify tail from every layer cache. Every leaf
        is checked before any is changed, because the shared masks read one
        layer's offset."""
        del gdn_states
        rejected = int(block_size) - int(accepted) - 1
        if rejected <= 0:
            return
        refused = [
            type(c).__name__ for c in prompt_cache if not c.is_trimmable()
        ]
        if refused:
            raise RuntimeError(
                f"Ripple MTP rollback: untrimmable cache leaves {refused} "
                f"(rejected={rejected}); the sliding leaves must be "
                f"BufferedRotatingKVCache by now"
            )
        for c in prompt_cache:
            if c.trim(rejected) != rejected:
                raise RuntimeError(
                    f"Ripple MTP rollback: {type(c).__name__}.trim"
                    f"({rejected}) refused after is_trimmable(); cache state "
                    f"is now inconsistent"
                )


class RicursiveSpecLM(SpecHooks, rm.Model):
    """The Ripple ``Model`` plus the speculative hooks, in the shape the
    owned MTP engine drives."""

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
        # mlx-vlm's chunked prefill passes inputs by keyword. The drafter
        # owns its KV, so shared_kv is never used.
        del n_to_process, kwargs
        want_hidden = return_hidden or return_shared_kv
        if want_hidden and self._dflash_capture is not None:
            h, caps = self.model(
                inputs, cache, capture_layers=self._dflash_capture,
                inputs_embeds=inputs_embeds)
            return _SpecOutput(logits=self.head(h),
                               hidden_states=[self._dflash_pack(h, caps)])
        h = self.model(inputs, cache, inputs_embeds=inputs_embeds)
        logits = self.head(h)
        if not want_hidden:
            from gmlx._vendor.mlx_vlm.models.base import LanguageModelOutput

            return LanguageModelOutput(logits=logits)
        return _SpecOutput(logits=logits, hidden_states=[h])
