"""Float projections whose result does not depend on the row count.

Stock mlx picks a matmul kernel by the shape of the call, and kernels
sum over K in different orders, so one row's output can move by a bf16
ulp when more rows share the batch. On MoE and gated-delta models two
small float projections turn that ulp into a different decision: the
expert router (top-k over its logits) and the gated-delta decay gate.
With ``GMLX_BATCH_INVARIANT=1`` the loader routes every plain float
``nn.Linear`` with at most ``GMLX_BATCH_INVARIANT_MAX_OUT`` outputs
(default 512, wide enough for a 512-expert router) through a Metal kernel with one thread per (row, output)
and a sequential fp32 walk over K, so a row's result is the same at any
batch size. bf16 inputs take 16-byte loads, other float dtypes an
element loop, both in the same order. The cost is about one percent of
a 512-token prefill on a 35B MoE; decode widths under 64 routed rows
keep the fused router and are outside the guarantee, and so are the
gates gmlx implements as raw arrays rather than ``nn.Linear``. Training
forwards take the stock matmul, since the kernel has no gradient.
"""
from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from gmlx.envflags import env_bool, env_int

__all__ = ["BatchInvariantLinear", "install_batch_invariant_linears"]

_FLOAT = (mx.bfloat16, mx.float16, mx.float32)

_ELEMENT_SRC = """
    uint m = thread_position_in_grid.x;
    uint n = thread_position_in_grid.y;
    uint M = x_shape[0];
    uint N = w_shape[0];
    uint K = w_shape[1];
    if (m >= M || n >= N) return;
    float acc = 0.0f;
    const device T* xr = x + (size_t)m * K;
    const device T* wr = w + (size_t)n * K;
    for (uint k = 0; k < K; ++k) {
        acc += float(xr[k]) * float(wr[k]);
    }
    out[(size_t)m * N + n] = acc;
"""

# bf16 rows viewed as uint32 pairs: element 2j is the low half of uint j,
# a bf16 to f32 conversion is a 16-bit shift, and every 16-byte load
# carries 8 elements summed in index order.
_BF16_SRC = """
    uint n = thread_position_in_grid.x;
    uint m = thread_position_in_grid.y;
    uint M = x_shape[0];
    uint N = w_shape[0];
    uint K2 = w_shape[1];
    if (m >= M || n >= N) return;
    float acc = 0.0f;
    const device uint4* xr = (const device uint4*)(x + (size_t)m * K2);
    const device uint4* wr = (const device uint4*)(w + (size_t)n * K2);
    for (uint j = 0; j < K2 / 4; ++j) {
        uint4 xv = xr[j];
        uint4 wv = wr[j];
        acc += as_type<float>(xv.x << 16) * as_type<float>(wv.x << 16);
        acc += as_type<float>(xv.x & 0xffff0000u) * as_type<float>(wv.x & 0xffff0000u);
        acc += as_type<float>(xv.y << 16) * as_type<float>(wv.y << 16);
        acc += as_type<float>(xv.y & 0xffff0000u) * as_type<float>(wv.y & 0xffff0000u);
        acc += as_type<float>(xv.z << 16) * as_type<float>(wv.z << 16);
        acc += as_type<float>(xv.z & 0xffff0000u) * as_type<float>(wv.z & 0xffff0000u);
        acc += as_type<float>(xv.w << 16) * as_type<float>(wv.w << 16);
        acc += as_type<float>(xv.w & 0xffff0000u) * as_type<float>(wv.w & 0xffff0000u);
    }
    out[(size_t)m * N + n] = acc;
"""

_element_kernel = mx.fast.metal_kernel(
    name="gmlx_invariant_linear", input_names=["x", "w"],
    output_names=["out"], source=_ELEMENT_SRC)
_bf16_kernel = mx.fast.metal_kernel(
    name="gmlx_invariant_linear_bf16", input_names=["x", "w"],
    output_names=["out"], source=_BF16_SRC)


def invariant_linear(x, w, bias=None):
    """``x @ w.T + bias`` with the row-invariant kernels; fp32 accumulate
    and bias, output in ``x``'s dtype. ``w`` is [N, K] float."""
    K = x.shape[-1]
    N = w.shape[0]
    xf = mx.contiguous(x.reshape(-1, K))
    M = xf.shape[0]
    if x.dtype == mx.bfloat16 and w.dtype == mx.bfloat16 and K % 8 == 0:
        out = _bf16_kernel(
            inputs=[xf.view(mx.uint32), mx.contiguous(w).view(mx.uint32)],
            grid=(N, M, 1), threadgroup=(min(N, 32), min(M, 32), 1),
            output_shapes=[(M, N)], output_dtypes=[mx.float32])[0]
    else:
        out = _element_kernel(
            inputs=[xf, mx.contiguous(w)], template=[("T", x.dtype)],
            grid=(M, N, 1), threadgroup=(min(M, 32), min(N, 8), 1),
            output_shapes=[(M, N)], output_dtypes=[mx.float32])[0]
    if bias is not None:
        out = out + bias.astype(mx.float32)
    return out.astype(x.dtype).reshape(x.shape[:-1] + (N,))


class BatchInvariantLinear(nn.Linear):
    """``nn.Linear`` whose float forward runs the row-invariant kernel.
    A float32 weight takes a bf16 or f16 input promoted to float32, the
    stock promotion, so the routers the loader keeps in float32 run the
    kernel too. A non-float weight, a float input narrower than a bf16
    or f16 weight, or a module in training mode (the kernel carries no
    gradient) falls through to the stock forward."""

    def __call__(self, x):
        w = self.weight
        if self.training or w.ndim != 2 or w.dtype not in _FLOAT or x.dtype not in _FLOAT:
            return super().__call__(x)
        if w.dtype == mx.float32 and x.dtype != mx.float32:
            x = x.astype(mx.float32)
        if x.dtype == w.dtype:
            return invariant_linear(x, w, self["bias"] if "bias" in self else None)
        return super().__call__(x)


def install_batch_invariant_linears(model, max_out=None) -> int:
    """Swap every plain float ``nn.Linear`` with at most ``max_out``
    outputs (``GMLX_BATCH_INVARIANT_MAX_OUT``, default 512) onto
    BatchInvariantLinear. Off unless ``GMLX_BATCH_INVARIANT`` is set.
    Returns the number of layers swapped."""
    if not env_bool("GMLX_BATCH_INVARIANT", False):
        return 0
    if max_out is None:
        max_out = env_int("GMLX_BATCH_INVARIANT_MAX_OUT", 512)
    n = 0
    for _, m in model.named_modules():
        if type(m) is not nn.Linear:
            continue
        w = m.weight
        if w.ndim != 2 or w.shape[0] > max_out:
            continue
        m.__class__ = BatchInvariantLinear
        n += 1
    return n
