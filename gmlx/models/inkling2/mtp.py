# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Asher Feldman
"""Inkling 2 speculative target: the ``speculative_*`` hooks and the
hidden-state capture the DSpark drafter reads.

The drafter names its taps as layer-visit inputs (``dflash.target_layers``
[5, 23, 41, 48] for Inkling-2-Mini), where index ``len(layer_visits)`` is
the normed final hidden state. The loader passes them through unchanged.

Each engine-facing hidden is packed as ``[trunk | cap_1 | ...]``. The
trunk is the normed final hidden state, so the logit hooks apply only the
head.

Rollback: the attention leaves are trimmed (the sliding ones are
``BufferedRotatingKVCache`` by decode time, which keeps slack past the
window). Each convolution state is the last ``width - 1`` inputs of its
window, and the verify forward records every window it ran, so the state
after the accepted prefix is a slice of the recorded window.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional

import mlx.core as mx

from . import model as im


@dataclass
class _SpecOutput:
    """Output for the owned engine's ``return_hidden`` calls."""

    logits: mx.array
    hidden_states: List[mx.array]
    shared_kv_states: dict = field(default_factory=dict)
    gdn_states: Optional[list] = None


def rollback_conv_sink(sink: list, n: int) -> None:
    """Set each recorded convolution state to the one after the first
    ``n`` inputs of its verify window."""
    for cache, slot, window in sink:
        width = cache[slot].shape[1]
        cache[slot] = mx.contiguous(window[:, n:n + width])


class SpecHooks:
    """The ``speculative_*`` hook set. Needs ``self.model`` (an
    :class:`Inkling2Model`) and ``self.head``."""

    _dflash_capture: Optional[tuple] = None

    def set_dflash_capture(self, layer_ids) -> None:
        # nn.Module.__setattr__ would put the tuple in the parameter dict.
        ids = tuple(sorted(int(i) for i in layer_ids))
        n = len(self.model.visits)
        if any(i < 0 or i > n for i in ids):
            raise ValueError(
                f"inkling2: drafter taps {ids} outside the {n} layer visits")
        object.__setattr__(self, "_dflash_capture", ids or None)

    def _forward(self, inputs, cache, inputs_embeds=None, conv_sink=None,
                 embed_mask=None):
        """(normed hidden, packed hidden)."""
        caps: list = []
        h = self.model(inputs, cache, inputs_embeds,
                       capture_layer_ids=self._dflash_capture,
                       hidden_sink=caps if self._dflash_capture else None,
                       embed_mask=embed_mask, conv_sink=conv_sink)
        if not self._dflash_capture:
            return h, h
        return h, mx.concatenate([h, *caps], axis=-1)

    def _dflash_trunk(self, hidden: mx.array) -> mx.array:
        if self._dflash_capture is None:
            return hidden
        # The quantized head reads its input buffer directly, so a strided
        # view of the packed hidden must be made contiguous first.
        return mx.contiguous(hidden[..., : self.model.args.hidden_size])

    def chunked_prefill_policy(self, **kwargs):
        return True

    def speculative_logits_from_hidden(self, hidden: mx.array) -> mx.array:
        return self.head(self._dflash_trunk(hidden))

    def speculative_argmax_from_hidden(self, hidden: mx.array) -> mx.array:
        return mx.argmax(self.speculative_logits_from_hidden(hidden), axis=-1)

    def speculative_verify_hidden(self, verify_input: mx.array, prompt_cache):
        """One verify forward; the third value is the convolution sink the
        rollback reads."""
        sink: list = []
        _, packed = self._forward(verify_input, prompt_cache, conv_sink=sink)
        return packed, {}, sink

    def rollback_speculative_cache(
        self, prompt_cache, gdn_states, accepted, block_size: int
    ) -> None:
        """Rewind every cache to the accepted prefix of the verify block.
        Every attention leaf is checked before any is changed."""
        if isinstance(accepted, mx.array):
            accepted = int(accepted.reshape(-1)[0].item())
        rejected = int(block_size) - int(accepted) - 1
        if rejected <= 0:
            return
        n_visits = len(self.model.visits)
        leaves = [c for c in prompt_cache[n_visits:] if c is not None]
        refused = [type(c).__name__ for c in leaves if not c.is_trimmable()]
        if refused:
            raise RuntimeError(
                f"inkling2 MTP rollback: untrimmable cache leaves {refused} "
                f"(rejected={rejected}); the sliding leaves must be "
                f"BufferedRotatingKVCache by now")
        for c in leaves:
            if c.trim(rejected) != rejected:
                raise RuntimeError(
                    f"inkling2 MTP rollback: {type(c).__name__}.trim"
                    f"({rejected}) refused after is_trimmable(); cache state "
                    f"is now inconsistent")
        rollback_conv_sink(gdn_states or [], int(accepted) + 1)


class Inkling2SpecLM(SpecHooks, im.Model):
    """The Inkling 2 ``Model`` plus the speculative hooks, in the shape the
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
        del n_to_process
        embed_mask = kwargs.get("embed_mask")
        if not (return_hidden or return_shared_kv):
            from gmlx._vendor.mlx_vlm.models.base import LanguageModelOutput

            h = self.model(inputs, cache, inputs_embeds, embed_mask=embed_mask)
            return LanguageModelOutput(logits=self.head(h))
        h, packed = self._forward(inputs, cache, inputs_embeds,
                                  embed_mask=embed_mask)
        return _SpecOutput(logits=self.head(h), hidden_states=[packed])
