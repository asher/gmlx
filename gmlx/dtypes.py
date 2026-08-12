"""Activation dtype for the model graph.

TEST BRANCH. The default here is float16, not the bfloat16 that gmlx ships,
so that an M1 or M2 box runs the float16 graph with nothing set: no flag, no
config key, no environment variable. Everything is still overridable the
usual way, and GMLX_ACTIVATION_DTYPE=bfloat16 gives back the shipping
behavior for an A/B. Do not merge this branch.

Non-quantized parameters and any dequantized embedding table are cast to one
dtype at load, and that dtype is what flows through the graph: a quantized
matmul returns its activation dtype, and the fused decode paths accept
float16 and bfloat16 alike. The KV cache follows too, since cache blocks are
allocated from the dtype of the keys and values written into them.

float16 exists for Apple GPUs before Apple9 (M1 and M2), which have no
native bfloat16 arithmetic. There the Metal compiler expands a bfloat16
multiply-add into a software sequence: an unrolled FMA chain compiles to
5.1x the machine code of its float16 form on applegpu_g13 and g14, against
1.85x on g15 and later. The k-quant matvecs are unaffected either way, since
they accumulate in float32 and only convert on load, so the cost falls on
the elementwise, norm, rope and softmax kernels between them.

float16 carries three more mantissa bits than bfloat16 and a much smaller
exponent range. Dot products stay safe because the k-quant kernels
accumulate in float32; the exposed surface is stored activations and the KV
cache. loader._FP32_KEEP_BY_MODEL_TYPE pins any tensor that needs the range
back to float32.
"""
from __future__ import annotations

import os
import re
import sys

import mlx.core as mx

from .envflags import env_choice

ENV_VAR = "GMLX_ACTIVATION_DTYPE"
# Shipping gmlx defaults to "bfloat16"; this test branch defaults to float16 so
# an M1 or M2 box needs no flag, key or variable to run the arm under test.
DEFAULT = "float16"
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

_ARCH_RE = re.compile(r"applegpu_g(\d+)")
_NATIVE_BF16_GEN = 15  # Apple9 (M3) and later

_arch_gen: int | None = None
_announced = False


def gpu_arch_gen() -> int:
    """Apple GPU generation from the Metal architecture string, 0 if unknown.

    ``applegpu_g13s`` yields 13. The numbering steps once per silicon family,
    so 13 is M1, 14 is M2 and 17 is M5. Probed once and cached.
    """
    global _arch_gen
    if _arch_gen is None:
        try:
            match = _ARCH_RE.search(str(mx.device_info().get("architecture", "")))
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


def activation_dtype() -> mx.Dtype:
    """The graph's activation dtype, from ``GMLX_ACTIVATION_DTYPE``.

    ``auto`` picks float16 on Apple GPUs without native bfloat16 and bfloat16
    everywhere else. Unset or unrecognized values give float16 on this test
    branch, where shipping gmlx gives bfloat16.
    """
    choice = env_choice(ENV_VAR, DEFAULT, CHOICES)
    dtype = (mx.bfloat16 if has_native_bfloat16() else mx.float16) \
        if choice == "auto" else _BY_NAME[choice]
    _announce(dtype)
    return dtype


def _announce(dtype: mx.Dtype) -> None:
    """Say which width the graph runs at, once per process, on stderr.

    The loader's own ``[dtype]`` line is verbosity-gated, and someone testing
    this branch should not have to pass -v to find out which arm they just
    measured. Test-branch only.
    """
    global _announced
    if _announced:
        return
    _announced = True
    source = "branch default" if os.environ.get(ENV_VAR) is None else ENV_VAR
    print(f"[gmlx] activation dtype: {_SHORT_NAME[dtype]} ({source})",
          file=sys.stderr)


def activation_dtype_name() -> str:
    """Short label for load logs."""
    return _SHORT_NAME[activation_dtype()]
