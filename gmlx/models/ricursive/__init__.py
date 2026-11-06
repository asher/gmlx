# SPDX-License-Identifier: MIT
"""Model package for Ricursive Ripple (GGUF arch ``ricursive``)."""

import gmlx.gen.prefill_decay as _prefill_decay

# Prefill attention is stock SDPA at head dim 128 with a causal or array
# mask, which MLX runs fused, so depth decay prices no score tensor.
_prefill_decay.register_score_profile(
    "ricursive", _prefill_decay.fused_attention_profile)
