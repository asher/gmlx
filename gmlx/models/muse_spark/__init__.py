"""Muse Spark family: the text model."""

import gmlx.gen.prefill_decay as _prefill_decay

# Prefill attention is stock SDPA at head dim 64 with a causal or array
# mask, which MLX runs fused, so depth decay prices no score tensor.
_prefill_decay.register_score_profile(
    "muse_spark", _prefill_decay.fused_attention_profile)
