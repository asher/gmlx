# SPDX-License-Identifier: MIT
"""Model package for RecursiveSI Flint (GGUF arch ``flint``)."""

import gmlx.gen.prefill_decay as _prefill_decay

# Prefill attention is stock SDPA at head dim 128 with a causal or array
# mask and sinks, which MLX runs fused, so depth decay prices no score
# tensor.
_prefill_decay.register_score_profile(
    "flint", _prefill_decay.fused_attention_profile)
