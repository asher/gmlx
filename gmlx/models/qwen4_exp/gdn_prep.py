"""The gated-delta layer's prefill steps around the scan, one kernel each.

prepare() replaces concat(conv state, rows), the depthwise conv, silu, the
split to q, k, v, the rms norm and scale of q and k and their tile to the
value heads. It reads the conv state and the rows in place. gate() replaces
rms_norm(y, weight) * sigmoid(z).

A thread owns 4 channels of one token and a simdgroup of 32 threads owns
one 128-wide head, which is the layout of MLX's rms norm kernel, and every
step runs in the type and order of the MLX op it replaces. The results have
the bits of the op chain.
"""
import mlx.core as mx

from gmlx.envflags import env_bool

_HEADER = """
#include <metal_simdgroup>
using namespace metal;
template <typename T> inline T gdnp_sigmoid(T x) {
  auto y = 1 / (1 + metal::precise::exp(metal::abs(x)));
  return (x < 0) ? y : 1 - y;
}
"""

_PREPARE = """
    const uint tid = thread_position_in_grid.x;
    const uint t = thread_position_in_grid.y;
    const uint C = uint(Cn);
    const uint keep = uint(Kn) - 1u;
    const uint c0 = tid * 4u;
    const uint head = c0 / 128u;
    InT act[4];
    float acc = 0.0f;
    for (uint i = 0; i < 4u; ++i) {
      const uint c = c0 + i;
      float conv = 0.0f;
      for (uint j = 0; j < uint(Kn); ++j) {
        const uint r = t + j;
        const InT xin = r < keep ? conv_state[size_t(r) * C + c] : rows[size_t(r - keep) * C + c];
        conv += static_cast<float>(xin) * w[c * uint(Kn) + j];
      }
      const InT x = static_cast<InT>(conv);
      act[i] = x * gdnp_sigmoid(x);
      const float xf = act[i];
      acc += xf * xf;
    }
    if (head >= 2u * uint(Hk)) {
      const size_t base = size_t(t) * (C - 2u * uint(Hk) * 128u) + (c0 - 2u * uint(Hk) * 128u);
      for (uint i = 0; i < 4u; ++i) v[base + i] = act[i];
      return;
    }
    acc = simd_sum(acc);
    const float inv = metal::precise::rsqrt(acc / 128u + eps);
    const bool is_q = head < uint(Hk);
    const InT scale = is_q ? sq : sk;
    const uint hk = is_q ? head : head - uint(Hk);
    InT n[4];
    for (uint i = 0; i < 4u; ++i) n[i] = scale * static_cast<InT>(act[i] * inv);
    // Value head hv reads key head hv % Hk: Rn copies, Hk heads apart.
    for (uint rep = 0; rep < uint(Rn); ++rep) {
      const size_t base = ((size_t(t) * uint(Rn) + rep) * uint(Hk) + hk) * 128u + c0 % 128u;
      for (uint i = 0; i < 4u; ++i) {
        if (is_q) q[base + i] = n[i]; else k[base + i] = n[i];
      }
    }
"""

_GATE = """
    const uint tid = thread_position_in_grid.x;
    const uint t = thread_position_in_grid.y;
    const size_t base = (size_t(t) * uint(Hv) * 32u + tid) * 4u;
    const uint d0 = (tid % 32u) * 4u;
    float acc = 0.0f;
    for (uint i = 0; i < 4u; ++i) {
      const float xi = y[base + i];
      acc += xi * xi;
    }
    acc = simd_sum(acc);
    const float inv = metal::precise::rsqrt(acc / 128u + eps);
    for (uint i = 0; i < 4u; ++i) {
      const InT n = wt[d0 + i] * static_cast<InT>(y[base + i] * inv);
      out[base + i] = n * gdnp_sigmoid(z[base + i]);
    }
"""

_DTYPES = (mx.bfloat16, mx.float16)
_kernels = {}


def _kernel(name, inputs, outputs, source):
    if name not in _kernels:
        _kernels[name] = mx.fast.metal_kernel(
            name=name, input_names=inputs, output_names=outputs,
            source=source, header=_HEADER)
    return _kernels[name]


def enabled() -> bool:
    """GMLX_Q4_GDN_PREFILL_FUSE=0 runs the op chain."""
    return env_bool("GMLX_Q4_GDN_PREFILL_FUSE", True)


def prepare_supported(conv_state, rows, weight, head_k_dim) -> bool:
    """One sequence of at least the conv state's rows, 128-wide heads, one
    half-precision dtype throughout."""
    return (rows.ndim == 3 and rows.shape[0] == 1
            and rows.shape[1] >= conv_state.shape[1]
            and head_k_dim == 128 and rows.shape[2] % 128 == 0
            and weight.ndim == 3 and weight.shape[2] == 1
            and weight.shape[1] == conv_state.shape[1] + 1
            and rows.dtype in _DTYPES
            and conv_state.dtype == rows.dtype and weight.dtype == rows.dtype)


def prepare(conv_state, rows, weight, hk, eps, q_scale, k_scale, tile=1):
    """conv_state [1, K - 1, C], rows [1, T, C], weight [C, K, 1].
    Returns q, k [1, T, hk * tile, 128] and v [1, T, C / 128 - 2 hk, 128]."""
    _, T, C = rows.shape
    dt = rows.dtype
    kern = _kernel("gdn_prefill_prepare",
                   ["conv_state", "rows", "w", "eps", "sq", "sk"],
                   ["q", "k", "v"], _PREPARE)
    return kern(
        inputs=[conv_state, rows, weight, mx.array(eps, dtype=mx.float32),
                mx.array(q_scale, dtype=dt), mx.array(k_scale, dtype=dt)],
        template=[("InT", dt), ("Cn", C), ("Kn", weight.shape[1]),
                  ("Hk", hk), ("Rn", tile)],
        grid=(C // 4, T, 1), threadgroup=(32, 1, 1),
        output_shapes=[(1, T, hk * tile, 128), (1, T, hk * tile, 128),
                       (1, T, C // 128 - 2 * hk, 128)],
        output_dtypes=[dt, dt, dt],
    )


def gate_supported(y, z, weight) -> bool:
    """One sequence, 128-wide heads, one half-precision dtype throughout."""
    return (y.ndim == 4 and y.shape[0] == 1 and y.shape[-1] == 128
            and y.shape == z.shape and weight.shape == (128,)
            and y.dtype in _DTYPES
            and z.dtype == y.dtype and weight.dtype == y.dtype)


def gate(y, z, weight, eps):
    """rms_norm(y, weight) * sigmoid(z) on y, z [1, T, Hv, 128]."""
    _, T, hv, _ = y.shape
    kern = _kernel("gdn_prefill_gate", ["y", "z", "wt", "eps"], ["out"], _GATE)
    return kern(
        inputs=[y, z, weight, mx.array(eps, dtype=mx.float32)],
        template=[("InT", y.dtype), ("Hv", hv)],
        grid=(hv * 32, T, 1), threadgroup=(32, 1, 1),
        output_shapes=[y.shape], output_dtypes=[y.dtype],
    )[0]
