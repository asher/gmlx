"""The stock names gmlx owns, written out by hand.

Each vendored module below must hand out the owned object under each of
its names. The tests read their expectations from here, so a name the
copy stops importing from the owned module fails a test.
"""

KV = "gmlx.cache.kv_cache"
ATTN = "gmlx.cache.attention"

# Every class mlx_vlm.models.cache defines.
OWNED_CLASSES = (
    "_BaseCache", "CacheMemory", "ConcatenateKVCache", "QuantizedKVCache",
    "KVCache", "RotatingKVCache", "ArraysCache", "ChunkedKVCache",
    "CacheList", "BatchKVCache", "BatchRotatingKVCache",
    "BufferedRotatingKVCache", "BatchQuantizedKVCache", "PoolingCache",
    "BatchPoolingCache", "SimpleKVCache", "StaticPrefixKVCache",
)

VLM_CACHE = OWNED_CLASSES + (
    "should_quantize_kv_layer", "create_causal_mask", "make_prompt_cache",
    "create_attention_mask", "cache_nbytes", "_kv_memory_profile",
    "_windowed_memory_profile", "_pooling_memory_profile",
    "_dequantize_uniform", "dynamic_roll", "_select_pooling_remainder",
    "_speculative_lengths", "_record_pooling_speculative_inputs",
    "_pooling_speculative_inputs", "_pooling_speculative_nbytes",
)

LM_CACHE = (
    "_BaseCache", "ConcatenateKVCache", "QuantizedKVCache", "KVCache",
    "RotatingKVCache", "ArraysCache", "ChunkedKVCache", "CacheList",
    "BatchKVCache", "BatchRotatingKVCache", "dynamic_roll",
    "make_prompt_cache", "create_attention_mask", "create_causal_mask",
)

VLM_BASE = (
    "create_attention_mask", "create_ssm_mask",
    "align_attention_mask_to_scores",
    "quantized_scaled_dot_product_attention",
    "_turboquant_attention_applies", "scaled_dot_product_attention",
)

LM_BASE = (
    "create_attention_mask", "create_ssm_mask",
    "quantized_scaled_dot_product_attention", "scaled_dot_product_attention",
)

# vendored module -> {name: owned module}. The owned object has the same
# name.
EXPECTED = {
    "gmlx._vendor.mlx_vlm.models.cache": {n: KV for n in VLM_CACHE},
    "gmlx._vendor.mlx_lm.models.cache": {n: KV for n in LM_CACHE},
    "gmlx._vendor.mlx_vlm.models.base": {"create_causal_mask": KV,
                            **{n: ATTN for n in VLM_BASE}},
    "gmlx._vendor.mlx_lm.models.base": {"create_causal_mask": KV,
                           **{n: ATTN for n in LM_BASE}},
}
