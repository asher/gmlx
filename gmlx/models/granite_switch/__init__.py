# SPDX-License-Identifier: MIT
"""Model package for IBM Granite Switch (GGUF arch ``graniteswitch``)."""

import gmlx.gen.prefill_decay as _prefill_decay


def _score_profile(model, prompt_cache):
    # Prefill attention is stock SDPA at head dim 128 with a causal or array
    # mask, which MLX runs fused. The router cache holds no attention keys.
    trunk = [c for c in prompt_cache or ()
             if not getattr(c, "kv_policy_exempt", False)]
    return _prefill_decay.fused_attention_profile(model, trunk)


_prefill_decay.register_score_profile("granite_switch", _score_profile)
