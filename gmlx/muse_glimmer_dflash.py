# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Asher Feldman
"""Muse Glimmer DFlash drafter: the owned DFlash base plus the Glimmer logit
tail. The drafter borrows the target's LM head, so it must reproduce the
target's output multiplier and softcap (``muse_glimmer_model.scale_and_softcap``,
fp32 like the llama.cpp oracle)."""

from __future__ import annotations

import mlx.core as mx

from . import muse_glimmer_model as mg
from .dflash_drafter import DFlashConfig, DFlashDrafter

__all__ = ["DFlashConfig", "MuseGlimmerDFlashDrafter"]


class MuseGlimmerDFlashDrafter(DFlashDrafter):
    """DFlash v1 drafter for Muse Glimmer targets."""

    def _logits(self, hidden: mx.array) -> mx.array:
        return mg.scale_and_softcap(
            self.lm_head(hidden), self.config.output_multiplier,
            self.config.final_logit_softcapping or 0.0)
