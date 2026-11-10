# SPDX-License-Identifier: MIT
"""Model package for Liquid AI LFM3 (GGUF arch ``lfm3``)."""

import gmlx.gen.prefill_decay as _prefill_decay

# Prefill attention is stock SDPA at head dim 128 with a causal or array
# mask, which MLX runs fused, so depth decay prices no score tensor.
_prefill_decay.register_score_profile(
    "lfm3", _prefill_decay.fused_attention_profile)
# Every attention layer has the config's head dim, so a reader with no
# cache prices the same fused chunk from the header.
_prefill_decay.register_transient_declaration(
    "lfm3", _prefill_decay.head_dim_declaration)
