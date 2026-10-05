"""The activation dtype of an image model, from the chip generation.

M3 and later chips run the DiT and the VAE in bfloat16. M1 and M2 chips
have no native bfloat16, so a DiT would run in float16 there, which has
not been checked against the reference on that hardware. It is refused
unless GMLX_IMAGE_ALLOW_FP16=1, and only families that tolerate float16
run under the override. The VAE then runs in float32, never float16.
float32 is not an option for the DiT, because a K-quant matmul promotes a
float32 input to bfloat16.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx

from .fingerprint import ImageModelError

ALLOW_FP16_ENV = "GMLX_IMAGE_ALLOW_FP16"

# Families whose blocks clamp their attention and feed-forward outputs to
# the float16 range and replace NaN when they run in float16
# (z_image: model.fp16_safe), and whose loader keeps float16 copies of its
# bfloat16 matrices.
FP16_FAMILIES = frozenset({"z_image"})

_NATIVE_BF16_GEN = 15
_CHIPS = {13: "M1", 14: "M2"}


class ImageDtypeError(ImageModelError):
    """An image model that this chip does not run, with the reason."""


@dataclass(frozen=True)
class Activation:
    dit: mx.Dtype
    vae: mx.Dtype
    chip_gen: int


def allow_fp16() -> bool:
    from gmlx.envflags import env_bool

    return env_bool(ALLOW_FP16_ENV, False)


def _gen(gen: int | None) -> int:
    if gen is not None:
        return gen
    from gmlx.load.dtypes import gpu_arch_gen

    return gpu_arch_gen()


def _native_bf16(gen: int) -> bool:
    return gen == 0 or gen >= _NATIVE_BF16_GEN


def vae_dtype(*, gen: int | None = None) -> mx.Dtype:
    """bfloat16 on chips with native bfloat16, else float32."""
    return mx.bfloat16 if _native_bf16(_gen(gen)) else mx.float32


def image_activation(family: str, *, gen: int | None = None,
                     allow: bool | None = None) -> Activation:
    """The DiT and VAE dtypes for ``family`` on this chip. ``gen`` and
    ``allow`` default to the GPU generation and GMLX_IMAGE_ALLOW_FP16."""
    gen = _gen(gen)
    if _native_bf16(gen):
        return Activation(mx.bfloat16, mx.bfloat16, gen)
    chip = _CHIPS.get(gen, f"Apple GPU generation {gen}")
    if family not in FP16_FAMILIES:
        raise ImageDtypeError(
            f"{family} needs bfloat16, which the {chip} chip lacks, and it "
            f"never runs in float16, with or without {ALLOW_FP16_ENV}")
    if not (allow_fp16() if allow is None else allow):
        raise ImageDtypeError(
            f"{family} would run in float16 on the {chip} chip, which is not "
            f"yet checked against the reference there. Set "
            f"{ALLOW_FP16_ENV}=1 to run it anyway")
    return Activation(mx.float16, mx.float32, gen)
