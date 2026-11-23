"""The PLE layer's prefill conv in one kernel.

conv_silu() replaces concat(conv state, rows), the dilated depthwise conv
and silu. It reads the conv state and the rows in place. A thread owns 4
channels of one token, sums the taps in float in tap order and rounds once,
as MLX's grouped conv does, and the silu runs in the input type as MLX's
ops take it. The results have the bits of the op chain.
"""
import mlx.core as mx

from gmlx.envflags import env_bool

_HEADER = """
using namespace metal;
template <typename T> inline T plep_sigmoid(T x) {
  auto y = 1 / (1 + metal::precise::exp(metal::abs(x)));
  return (x < 0) ? y : 1 - y;
}
"""

_CONV_SILU = """
    const uint c0 = thread_position_in_grid.x * 4u;
    const uint t = thread_position_in_grid.y;
    const uint C = uint(Cn);
    const uint keep = uint(Ln);
    for (uint i = 0; i < 4u; ++i) {
      const uint c = c0 + i;
      float conv = 0.0f;
      for (uint j = 0; j < uint(Kn); ++j) {
        const uint r = t + j * uint(Dn);
        const InT xin = r < keep ? conv_state[size_t(r) * C + c] : rows[size_t(r - keep) * C + c];
        conv += static_cast<float>(xin) * w[c * uint(Kn) + j];
      }
      const InT x = static_cast<InT>(conv);
      out[size_t(t) * C + c] = x * plep_sigmoid(x);
    }
"""

_DTYPES = (mx.bfloat16, mx.float16)
_kernel = None


def enabled() -> bool:
    """GMLX_Q4_PLE_PREFILL_FUSE=0 runs the op chain."""
    return env_bool("GMLX_Q4_PLE_PREFILL_FUSE", True)


def conv_supported(conv_state, rows, weight, dilation) -> bool:
    """One sequence of a half-precision dtype throughout, a state of
    (K - 1) x dilation rows and channels in fours."""
    return (rows.ndim == 3 and rows.shape[0] == 1 and conv_state.ndim == 3
            and conv_state.shape[0] == 1 and rows.shape[2] % 4 == 0
            and conv_state.shape[2] == rows.shape[2]
            and weight.ndim == 3 and weight.shape[0] == rows.shape[2]
            and weight.shape[2] == 1
            and conv_state.shape[1] == (weight.shape[1] - 1) * dilation
            and rows.dtype in _DTYPES
            and conv_state.dtype == rows.dtype and weight.dtype == rows.dtype)


def conv_silu(conv_state, rows, weight, dilation):
    """silu(conv1d(concat(conv_state, rows))) of a depthwise conv with this
    dilation: conv_state [1, L, C], rows [1, T, C], weight [C, K, 1]."""
    global _kernel
    if _kernel is None:
        _kernel = mx.fast.metal_kernel(
            name="ple_prefill_conv_silu",
            input_names=["conv_state", "rows", "w"], output_names=["out"],
            source=_CONV_SILU, header=_HEADER)
    _, T, C = rows.shape
    return _kernel(
        inputs=[conv_state, rows, weight],
        template=[("InT", rows.dtype), ("Cn", C), ("Kn", weight.shape[1]),
                  ("Dn", dilation), ("Ln", conv_state.shape[1])],
        grid=(C // 4, T, 1), threadgroup=(64, 1, 1),
        output_shapes=[rows.shape], output_dtypes=[rows.dtype],
    )[0]
