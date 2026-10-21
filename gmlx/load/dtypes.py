"""Activation dtype for the model graph.

Non-quantized parameters and any dequantized embedding table are cast to one
dtype at load, and that dtype is what flows through the graph: a quantized
matmul returns its activation dtype, and the fused decode paths accept
float16 and bfloat16 alike.

The default is "auto". It selects float16 on Apple GPUs before Apple9, and
bfloat16 on all other devices. Apple GPUs before Apple9 are the M1 and the
M2. They have no native bfloat16 arithmetic. On these GPUs the Metal
compiler expands a bfloat16 multiply-add into a software sequence. An
unrolled FMA chain compiles to 5.1 times the machine code of its float16
form on applegpu_g13 and g14. The same chain compiles to 1.85 times on g15
and later.

The k-quant matvecs do not change with the dtype. They accumulate in
float32 and convert only on load. Therefore the cost falls on the
elementwise, norm, rope and softmax kernels between the matvecs.

A Mac Studio M1 Max measured the result on Qwen3.6-27B. float16 gave 7 to
18 percent more decode tokens per second and 24 to 53 percent more prefill
tokens per second than bfloat16. gmlx moved from a loss to a win against
llama.cpp on every K-quant cell.

float16 carries three more mantissa bits than bfloat16 and a much smaller
exponent range. Dot products stay safe because the k-quant kernels
accumulate in float32; the exposed surface is stored activations and the KV
cache. loader._FP32_KEEP_BY_MODEL_TYPE pins any tensor that needs the range
back to float32.
"""
from __future__ import annotations

import re

import mlx.core as mx

from gmlx.device import metal_info
from gmlx.envflags import env_choice

ENV_VAR = "GMLX_ACTIVATION_DTYPE"
# "auto" reads the GPU generation. Name a dtype to override the rule.
DEFAULT = "auto"
# float32 is a reference arm for certification, not a shipping mode: it doubles
# activation memory and drops the 16-bit fused decode paths (their guards admit
# float16 and bfloat16 only), which is exactly what makes it a clean baseline to
# score the two 16-bit widths against. Env-only; the CLI offers the other three.
CHOICES = ("auto", "bfloat16", "bf16", "float16", "fp16", "float32", "fp32")

_BY_NAME = {
    "bfloat16": mx.bfloat16,
    "bf16": mx.bfloat16,
    "float16": mx.float16,
    "fp16": mx.float16,
    "float32": mx.float32,
    "fp32": mx.float32,
}

_SHORT_NAME = {mx.bfloat16: "bf16", mx.float16: "fp16", mx.float32: "fp32"}

# Model types that never run float16. The EmbeddingGemma model card rules
# out float16 activations, so where the rule picks float16 these run float32.
# gemma_embedding is the EmbeddingGemma GGUF. The safetensors build has the
# model type gemma3_text, which every Gemma 3 chat GGUF also gets, so the
# encoder loader applies the rule to it (models/embedders/checkpoint.py).
NO_FLOAT16_MODEL_TYPES = frozenset({"gemma_embedding"})

# Norm weights. GGUF files store them F32. "auto" keeps them float32 for
# the model types below and casts them to the activation dtype for the rest;
# "fp32" keeps them float32 for every model, and "activation" casts them for
# every model. A norm with a float32 weight returns its input's dtype, so
# only the weight is wider (loader.cast_fp32_norm_outputs).
NORM_ENV_VAR = "GMLX_NORM_DTYPE"
NORM_CHOICES = ("auto", "fp32", "float32", "activation")
NORM_WEIGHT_KEY = "norm.weight"
# beam: four sandwich norms per layer plus q/k norms, and the post-norms
# scale each branch just before the residual. bf16 weights put the synthetic
# cert at 1.3e-3 median KLD from an fp64 reference, float32 at 1e-5, and
# greedy text on the real model splits after about 170 tokens. The streamed
# model shows no decode cost.
FP32_NORM_MODEL_TYPES = frozenset({"beam"})


def wider_float(a: mx.Dtype, b: mx.Dtype) -> mx.Dtype:
    """The dtype MLX promotes two float dtypes to: either one when they
    match, float32 for any mix of float16, bfloat16 and float32."""
    return a if a == b else mx.float32


def fp32_norms(model_type: str | None) -> bool:
    """Whether ``model_type`` keeps its F32 norm weights float32."""
    choice = env_choice(NORM_ENV_VAR, "auto", NORM_CHOICES)
    if choice == "activation":
        return False
    if choice in ("fp32", "float32"):
        return True
    return model_type in FP32_NORM_MODEL_TYPES


_ARCH_RE = re.compile(r"applegpu_g(\d+)")
_NATIVE_BF16_GEN = 15  # Apple9 (M3) and later

_arch_gen: int | None = None


def gpu_arch_gen() -> int:
    """Apple GPU generation from the Metal architecture string, 0 if unknown.

    ``applegpu_g13s`` yields 13. The numbering steps once per silicon family,
    so 13 is M1, 14 is M2 and 17 is M5. Probed once and cached.
    """
    global _arch_gen
    if _arch_gen is None:
        try:
            match = _ARCH_RE.search(str((metal_info() or {}).get("architecture", "")))
        except Exception:
            match = None
        _arch_gen = int(match.group(1)) if match else 0
    return _arch_gen


def has_native_bfloat16() -> bool:
    """Whether this GPU runs bfloat16 arithmetic without compiler expansion.

    An unknown architecture reports True, so an unrecognized device keeps the
    default dtype rather than silently switching numerics.
    """
    gen = gpu_arch_gen()
    return gen == 0 or gen >= _NATIVE_BF16_GEN


def activation_dtype(model_type: str | None = None) -> mx.Dtype:
    """The graph's activation dtype, from ``GMLX_ACTIVATION_DTYPE``.

    ``auto`` is the default. It gives float16 on Apple GPUs that have no
    native bfloat16, and bfloat16 on all other devices. An unrecognized
    value also gives ``auto``. For a model type in
    :data:`NO_FLOAT16_MODEL_TYPES`, float16 becomes float32.
    """
    choice = env_choice(ENV_VAR, DEFAULT, CHOICES)
    if choice == "auto":
        dtype = mx.bfloat16 if has_native_bfloat16() else mx.float16
    else:
        dtype = _BY_NAME[choice]
    if dtype == mx.float16 and model_type in NO_FLOAT16_MODEL_TYPES:
        return mx.float32
    return dtype


def dtype_name(dtype: mx.Dtype) -> str:
    """Short label of a float dtype for load logs."""
    return _SHORT_NAME.get(dtype, str(dtype))


def activation_dtype_name(model_type: str | None = None) -> str:
    """Short label for load logs."""
    return dtype_name(activation_dtype(model_type))
