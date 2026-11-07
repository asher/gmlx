"""Float32 products at float32 precision on a device that multiplies
float32 at TF32.

MLX runs float32 matrix products and transposed convolutions at TF32
precision on M5-class GPUs, and convolutions and fused attention at some
channel counts and head sizes, unless ``MLX_ENABLE_TF32=0`` is set before
the first product of the process. It reads the switch once. A model that
needs float32 products in such a process splits each operand into parts
TF32 holds exactly and sums their products, and runs a transposed
convolution as a convolution.
"""

from __future__ import annotations

import mlx.core as mx

_EXACT: dict[str, bool] = {}    # by default device
_CONV_EXACT: dict = {}          # by default device, channels and groups


def f32_products_exact() -> bool:
    """Whether the default device multiplies float32 matrices at float32
    precision. MLX runs float32 matrix products at TF32 precision on
    M5-class GPUs unless ``MLX_ENABLE_TF32=0`` is set before the first
    product of the process, and it reads the switch once. Each device is
    measured once against a product on the CPU."""
    key = str(mx.default_device())
    hit = _EXACT.get(key)
    if hit is None:
        a = mx.random.normal((64, 64), key=mx.random.key(0))
        b = mx.random.normal((64, 64), key=mx.random.key(1))
        gap = mx.abs(mx.matmul(a, b) - mx.matmul(a, b, stream=mx.cpu)).max()
        hit = _EXACT[key] = float(gap) < 1e-3
    return hit


# A TF32 operand keeps 10 of the 23 mantissa bits of a float32.
_TF32_HIGH = 0xFFFFE000


def split(a: mx.array) -> tuple[mx.array, mx.array]:
    """``a`` as the part a TF32 product reads exactly, the low 13 mantissa
    bits cleared, and the rest, which is exact in float32."""
    high = mx.view(mx.bitwise_and(mx.view(a, mx.uint32),
                                  mx.array(_TF32_HIGH, mx.uint32)),
                   mx.float32)
    return high, a - high


def tf32_exact(a: mx.array) -> mx.array:
    """Whether every value of float32 ``a`` is its own high part, as the
    values of a bfloat16 tensor read as float32 are."""
    low = mx.bitwise_and(mx.view(a, mx.uint32),
                         mx.array(~_TF32_HIGH & 0xFFFFFFFF, mx.uint32))
    return mx.all(low == 0)


def exact_matmul(a: mx.array, b: mx.array, *, b_t: bool = False,
                 b_exact: bool = False) -> mx.array:
    """``a @ b`` in float32, or ``a @ b.T`` with ``b_t``, near float32
    precision on a device that multiplies float32 at TF32: the product of
    the high parts and the two cross products, with the low parts read at
    TF32. Over a sum of hundreds of products it holds float32 precision;
    over a few it can read several float32 roundings off. With
    ``b_exact``, ``b`` is its own high part and its cross product is zero,
    so ``b`` is not split."""
    ah, al = split(a)
    if b_exact:
        bh = b.T if b_t else b
        return ah @ bh + al @ bh
    bh, bl = split(b)
    if b_t:
        bh, bl = bh.T, bl.T
    return ah @ bh + (ah @ bl + al @ bh)


def conv1d_exact(channels: int, out_channels: int, groups: int = 1) -> bool:
    """Whether the default device convolves float32 input of ``channels``
    channels to ``out_channels`` in ``groups`` at float32 precision. MLX
    picks the kernel by the channels per group on both sides. Each device
    and shape is measured once against the CPU."""
    key = (str(mx.default_device()), int(channels), int(out_channels),
           int(groups))
    hit = _CONV_EXACT.get(key)
    if hit is None:
        x = mx.random.normal((1, 64, channels), key=mx.random.key(0))
        w = mx.random.normal((out_channels, 3, channels // groups),
                             key=mx.random.key(1))
        want = mx.conv1d(x, w, groups=groups, stream=mx.cpu)
        gap = mx.linalg.norm(mx.conv1d(x, w, groups=groups) - want) \
            / mx.linalg.norm(want)
        hit = _CONV_EXACT[key] = float(gap) < 1e-5
    return hit


def conv1d(x: mx.array, w: mx.array, stride: int = 1, padding: int = 0,
           dilation: int = 1, groups: int = 1) -> mx.array:
    """``mx.conv1d`` at float32 precision: where the device convolves the
    shape at TF32, each operand in three parts a TF32 product reads
    exactly, and the six convolutions of parts whose product is over the
    rounding of a float32, summed from the smallest."""
    def conv(a, b):
        return mx.conv1d(a, b, stride, padding, dilation, groups)

    if conv1d_exact(x.shape[-1], w.shape[0], groups):
        return conv(x, w)
    xs, ws = _split3(x), _split3(w)
    small = (conv(xs[2], ws[0]) + conv(xs[1], ws[1]) + conv(xs[0], ws[2]))
    return small + (conv(xs[1], ws[0]) + conv(xs[0], ws[1])) \
        + conv(xs[0], ws[0])


def _split3(a: mx.array) -> tuple[mx.array, mx.array, mx.array]:
    high, rest = split(a)
    return (high, *split(rest))


def conv_transpose1d(x: mx.array, w: mx.array, stride: int = 1,
                     padding: int = 0) -> mx.array:
    """``mx.conv_transpose1d`` at float32 precision, as the convolution
    of ``x`` with ``stride - 1`` zeros after each sample but the last and
    the kernel reversed. The stock op runs at TF32 on M5-class GPUs, and
    is slower there either way. ``padding`` is at most the kernel size
    less one."""
    b, n, c = x.shape
    if stride > 1:
        x = mx.concatenate([x[:, :, None],
                            mx.zeros((b, n, stride - 1, c), x.dtype)], axis=2)
        x = x.reshape(b, n * stride, c)[:, :(n - 1) * stride + 1]
    return conv1d(x, w[:, ::-1], padding=w.shape[1] - 1 - padding)
