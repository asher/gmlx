"""The stock names gmlx binds to owned objects, written out by hand.

The binding tests read their expectations from here and not from
gmlx.cache.bind, so a name dropped from the binding map still fails a test.
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

# stock module -> {name: owned module}. The owned object has the same name.
EXPECTED = {
    "mlx_vlm.models.cache": {n: KV for n in VLM_CACHE},
    "mlx_lm.models.cache": {n: KV for n in LM_CACHE},
    "mlx_vlm.models.base": {"create_causal_mask": KV,
                            **{n: ATTN for n in VLM_BASE}},
    "mlx_lm.models.base": {"create_causal_mask": KV,
                           **{n: ATTN for n in LM_BASE}},
}
