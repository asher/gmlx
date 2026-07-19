"""gpt-oss prefill score-transient profile.

gpt-oss runs attention through the stock fused flash kernel
(`mx.fast.scaled_dot_product_attention(..., sinks=...)`, head_dim 64), which
never materializes the dense `[heads, step, S]` score tensor that the default
prefill-decay model assumes -- each head's scores stay threadgroup-local and
the only depth-carried output is the `[step, head_dim]` V-aggregate. The
attention sinks also bar quantized SDPA, so gpt-oss always runs unquantized KV
on this path; there is no materializing branch to fall back to.

Under the dense default the phantom `[64, step, depth]` transient crests
multi-GB at deep prefill and halves the chunk step (2048 -> 512 at ~90k),
throttling MoE-gather prefill ~26% for no memory benefit. This profile models
the real transient as a single fp16 score strip (`heads=1, bytes_per_elem=2`,
full depth) -- a generous upper bound on a transient that is really tiny and
depth-independent -- which holds the full 2048 chunk through ~1.5M-token depth
and only decays past that as a backstop. Stock base step is kept (the fix is
decay-prevention, not a chunk-size change).

Disarms (dense decay stays authoritative) on batched sequences or quantized
KV, where the transient model and kernel path differ: require_cache on the
mlx-lm/vlm KVCache demands a present single-sequence unquantized full-attn
cache, and any batched cache (BatchKVCache / BatchRotatingKVCache, not KVCache
subclasses) drops the match. The sliding-layer RotatingKVCache carries no
`bits` attribute, so it neither arms nor disarms.
"""
from __future__ import annotations

from . import cache_compat
from . import prefill_decay as _prefill_decay

_prefill_score_profile = _prefill_decay.build_score_profile(
    profile=_prefill_decay.ScoreTransientProfile(
        heads=1, bytes_per_elem=2, depth_divisor=1),
    require_cache=cache_compat.cache_types("KVCache"),
    disarm_cache=(cache_compat.cache_types("BatchKVCache")
                  + cache_compat.cache_types("BatchRotatingKVCache")),
)


_prefill_decay.register_score_profile("gpt_oss", _prefill_score_profile)
