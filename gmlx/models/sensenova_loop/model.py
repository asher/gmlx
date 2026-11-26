# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Asher Feldman
"""The text-only build of SenseNova Loop, in the mlx-lm model interface.

``Model`` is the owned language model (``language.SenseNovaLoopLanguageModel``)
returning logits, so a plain load and an MTP target share one class and one
parameter tree (``model.*``, ``lm_head``).
"""

from __future__ import annotations

import mlx.core as mx

from .language import SenseNovaLoopLanguageModel, TextConfig, model_config

ModelArgs = TextConfig


class Model(SenseNovaLoopLanguageModel):
    def __init__(self, args: ModelArgs):
        super().__init__(args, model_config(args))

    def __call__(self, inputs: mx.array, cache=None, input_embeddings=None,
                 **kwargs):
        out = super().__call__(inputs, inputs_embeds=input_embeddings,
                               cache=cache, **kwargs)
        return out.logits

    def sanitize(self, weights):
        # GGUF names are final: no mtp.* keys, norms as stored.
        return weights
