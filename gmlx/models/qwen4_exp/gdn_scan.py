"""Prefill scan of the gated-delta recurrence with staged token blocks.

The recurrence per token and value head, on a Dv x Dk state S in fp32:
    S = g S;  m = S k;  S += (v - m) beta k^T;  y = S q

A thread holds 16 key columns of one state row in registers, eight lanes
share a row, a threadgroup of 128 threads carries 16 rows, and blocks of 16
tokens of q, k, v and the gates are staged in threadgroup memory while the
next block is fetched into registers. q and k come at their own head count:
the kernel pairs a value head with its key head, so no tile to the value
heads is built.

The results have the bits of the stock scan. That scan gives a thread 4 key
columns, adds them in sequence, and adds the 32 threads of a state row with
simd_sum, which adds adjacent pairs, then pairs of pairs. Here a thread's
16 columns are 4 of those lanes: each is added in sequence, the 4 lanes as
(p0 + p1) + (p2 + p3), and the 8 threads of the row by xor shuffles 1, 2, 4,
which is the same tree.

scan_staged_gates takes b and a as one joined [b | a] row per token and
computes the decay g = compute_g(A_log, a, dt_bias) and beta = sigmoid(b) as
it fetches them, in the types and order of those ops, so with their bits.

The kernel layout follows prefill_gdn_scan of Splash
(runtime/metal/kernels/prefill/gdn.metal, Apache License 2.0).
"""
import mlx.core as mx

from gmlx.envflags import env_bool

_HEADER = """
#define GDNS_FETCH(START) \\
  for (uint n_ = 0; n_ < 2; ++n_) { \\
    const uint element_ = (n_ * 128u + thread_index) * 8u; \\
    const uint token_ = (START) + element_ / Dk; \\
    if (token_ < tokens) { \\
      const size_t base_ = (size_t(token_) * Hk + hk) * Dk + element_ % Dk; \\
      for (uint h_ = 0; h_ < 2; ++h_) { \\
        key_fetch[n_][h_] = reinterpret_cast<const device vec<InT, 4> *>(k + base_)[h_]; \\
        query_fetch[n_][h_] = reinterpret_cast<const device vec<InT, 4> *>(q + base_)[h_]; \\
      } \\
    } \\
  } \\
  if (thread_index < 32u) { \\
    const uint element_ = thread_index * 8u; \\
    const uint token_ = (START) + element_ / 16u; \\
    if (token_ < tokens) { \\
      const size_t base_ = (size_t(token_) * Hv + hv) * Dv + row_base + element_ % 16u; \\
      for (uint h_ = 0; h_ < 2; ++h_) \\
        value_fetch[h_] = reinterpret_cast<const device vec<InT, 4> *>(v + base_)[h_]; \\
    } \\
  } \\
  if (thread_index < 16u && (START) + thread_index < tokens) { \\
    GDNS_GATE((START) + thread_index) \\
  }

#define GDNS_STAGE() \\
  for (uint n_ = 0; n_ < 2; ++n_) { \\
    const uint element_ = (n_ * 128u + thread_index) * 8u; \\
    for (uint h_ = 0; h_ < 2; ++h_) { \\
      reinterpret_cast<threadgroup float4 *>(keys + element_)[h_] = float4(key_fetch[n_][h_]); \\
      reinterpret_cast<threadgroup float4 *>(queries + element_)[h_] = float4(query_fetch[n_][h_]); \\
    } \\
  } \\
  if (thread_index < 32u) { \\
    for (uint h_ = 0; h_ < 2; ++h_) \\
      reinterpret_cast<threadgroup float4 *>(values + thread_index * 8u)[h_] = float4(value_fetch[h_]); \\
  } \\
  if (thread_index < 16u) { \\
    gates[thread_index] = decay_fetch; \\
    gates[16u + thread_index] = beta_fetch; \\
  }
"""

# The decay and beta of token TOK: read (scan_staged), or computed from the
# joined b | a row as compute_g and mx.sigmoid compute them (scan_staged_gates).
_GATE_READ = """
#define GDNS_GATE(TOK) { \\
    const size_t i_ = size_t(TOK) * Hv + hv; \\
    decay_fetch = float(g[i_]); \\
    beta_fetch = float(beta[i_]); }
"""
_GATE_RAW = """
template <typename T> inline T gdns_sigmoid(T x) {
  auto y = 1 / (1 + metal::precise::exp(metal::abs(x)));
  return (x < 0) ? y : 1 - y;
}
// MLX's log1p and LogAddExp; softplus is logaddexp(x, 0).
inline float gdns_log1p(float x) {
  float xp1 = 1.0f + x;
  if (xp1 == metal::numeric_limits<float>::infinity()) return metal::numeric_limits<float>::infinity();
  if (xp1 == 1.0f) return x;
  return x * (metal::log(xp1) / (xp1 - 1.0f));
}
inline float gdns_logaddexp(float x, float y) {
  if (metal::isnan(x) || metal::isnan(y)) return metal::numeric_limits<float>::quiet_NaN();
  constexpr float inf = metal::numeric_limits<float>::infinity();
  float maxval = metal::max(x, y);
  float minval = metal::min(x, y);
  return (minval == -inf || maxval == inf) ? maxval : (maxval + gdns_log1p(metal::exp(minval - maxval)));
}
#define GDNS_GATE(TOK) { \\
    const size_t i_ = size_t(TOK) * (2u * Hv); \\
    decay_fetch = metal::precise::exp(a_neg * gdns_logaddexp(float(ba[i_ + Hv + hv]) + dt_h, 0.0f)); \\
    beta_fetch = float(gdns_sigmoid<InT>(ba[i_ + hv])); }
"""
_SETUP_RAW = """const float a_neg = -metal::precise::exp(float(A_log[hv]));
    const float dt_h = dt_bias[hv];
    """

_SOURCE = """
    // 128 threads: lane owns 16 key columns of one state row, 8 lanes a row, 16 rows a group.
    threadgroup float keys[16 * 128];
    threadgroup float queries[16 * 128];
    threadgroup float values[16 * 16];
    threadgroup float gates[2 * 16];
    const uint tokens = uint(T);
    const uint group = threadgroup_position_in_grid.x;
    const uint thread_index = thread_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const uint simd_group = simdgroup_index_in_threadgroup;
    const uint groups_per_head = Dv / 16u;
    const uint hv = group / groups_per_head;
    const uint hk = HK_SOURCE;
    const uint row_base = (group % groups_per_head) * 16u;
    const uint row = simd_group * 4u + lane / 8u;
    const uint column = (lane % 8u) * 16u;
    GATE_SETUP
    const size_t state_base = (size_t(hv) * Dv + row_base + row) * Dk + column;

    float4 state[4];
    for (uint c = 0; c < 4; ++c)
      state[c] = float4(reinterpret_cast<const device vec<StT, 4> *>(state_in + state_base)[c]);

    // The prefetch holds the block in its stored type; staging widens it.
    vec<InT, 4> key_fetch[2][2], query_fetch[2][2], value_fetch[2];
    float decay_fetch = 0.0f, beta_fetch = 0.0f;
    for (uint n_ = 0; n_ < 2; ++n_)
      for (uint h_ = 0; h_ < 2; ++h_) { key_fetch[n_][h_] = vec<InT, 4>(0); query_fetch[n_][h_] = vec<InT, 4>(0); }
    value_fetch[0] = vec<InT, 4>(0); value_fetch[1] = vec<InT, 4>(0);

    GDNS_FETCH(0u)
    GDNS_STAGE()
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint start = 0; start < tokens; start += 16u) {
      const bool more = start + 16u < tokens;
      if (more) {
        GDNS_FETCH(start + 16u)
      }
      const uint count = min(16u, tokens - start);
      for (uint t = 0; t < count; ++t) {
        const float d = gates[t], b = gates[16u + t];
        const threadgroup float4 *key = reinterpret_cast<const threadgroup float4 *>(keys + t * 128u + column);
        const threadgroup float4 *query = reinterpret_cast<const threadgroup float4 *>(queries + t * 128u + column);
        // A float4 is one lane of the stock scan: its 4 columns in sequence.
        float4 partial;
        for (uint c = 0; c < 4; ++c) {
          state[c] = state[c] * d;
          float p = 0.0f;
          for (uint e = 0; e < 4; ++e) p += state[c][e] * key[c][e];
          partial[c] = p;
        }
        float memory = (partial.x + partial.y) + (partial.z + partial.w);
        memory += simd_shuffle_xor(memory, 1);
        memory += simd_shuffle_xor(memory, 2);
        memory += simd_shuffle_xor(memory, 4);
        const float delta = (values[t * 16u + row] - memory) * b;
        for (uint c = 0; c < 4; ++c) {
          state[c] = state[c] + key[c] * delta;
          float p = 0.0f;
          for (uint e = 0; e < 4; ++e) p += state[c][e] * query[c][e];
          partial[c] = p;
        }
        float result = (partial.x + partial.y) + (partial.z + partial.w);
        result += simd_shuffle_xor(result, 1);
        result += simd_shuffle_xor(result, 2);
        result += simd_shuffle_xor(result, 4);
        if (lane % 8u == 0u)
          y[(size_t(start + t) * Hv + hv) * Dv + row_base + row] = InT(result);
      }
      if (more) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        GDNS_STAGE()
        threadgroup_barrier(mem_flags::mem_threadgroup);
      }
    }
    for (uint c = 0; c < 4; ++c)
      reinterpret_cast<device vec<StT, 4> *>(state_out + state_base)[c] = vec<StT, 4>(state[c]);
"""

_kernels = {}


def _kernel(tiled: bool, raw: bool = False):
    key = (tiled, raw)
    if key not in _kernels:
        src = (_SOURCE.replace("HK_SOURCE", "hv % Hk" if tiled else "hv / (Hv / Hk)")
               .replace("GATE_SETUP", _SETUP_RAW if raw else ""))
        gates = ["ba", "A_log", "dt_bias"] if raw else ["g", "beta"]
        _kernels[key] = mx.fast.metal_kernel(
            name="gdn_scan_staged" + ("_tiled" if tiled else "") + ("_gates" if raw else ""),
            input_names=["q", "k", "v", *gates, "state_in", "T"],
            output_names=["y", "state_out"],
            source=src,
            header="#include <metal_simdgroup>\nusing namespace metal;\n"
                   + (_GATE_RAW if raw else _GATE_READ) + _HEADER,
        )
    return _kernels[key]


def supported(q, v, g, state) -> bool:
    """One sequence, scalar gate a head, 128 key columns, value rows in sixteens."""
    return (q.shape[0] == 1 and g.ndim == 3 and q.shape[-1] == 128
            and v.shape[-1] % 16 == 0 and v.shape[2] % q.shape[2] == 0
            and q.dtype == v.dtype)


def _launch(kern, q, k, v, gates, state):
    _, T, Hk, Dk = k.shape
    Hv, Dv = v.shape[2:]
    return kern(
        inputs=[q, k, v, *gates, state, T],
        template=[("InT", q.dtype), ("StT", state.dtype), ("Dk", Dk), ("Dv", Dv), ("Hk", Hk), ("Hv", Hv)],
        grid=(128 * Hv * (Dv // 16), 1, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(1, T, Hv, Dv), state.shape],
        output_dtypes=[q.dtype, state.dtype],
    )


def scan_staged(q, k, v, g, beta, state, tiled=True):
    """q, k [1, T, Hk, 128]; v [1, T, Hv, Dv]; g, beta [1, T, Hv]; state [1, Hv, Dv, 128]."""
    return _launch(_kernel(tiled), q, k, v, [g, beta], state)


def gates_supported(q, v, ba, A_log, dt_bias) -> bool:
    """b | a rows of the input dtype, a float32 dt_bias (compute_g adds it to
    a in float32) and a float A_log per value head."""
    hv = v.shape[2]
    return (ba.ndim == 3 and ba.shape[:2] == v.shape[:2]
            and ba.shape[-1] == 2 * hv and ba.dtype == q.dtype
            and dt_bias.dtype == mx.float32 and dt_bias.shape == (hv,)
            and A_log.shape == (hv,)
            and A_log.dtype in (mx.float32, mx.float16, mx.bfloat16))


def scan_staged_gates(q, k, v, ba, A_log, dt_bias, state, tiled=True):
    """scan_staged with g = compute_g(A_log, a, dt_bias) and beta = sigmoid(b)
    computed in the kernel from ba [1, T, 2 Hv], b's columns then a's."""
    return _launch(_kernel(tiled, raw=True), q, k, v, [ba, A_log, dt_bias], state)


def enabled() -> bool:
    """GMLX_GDN_STAGED_SCAN=0 runs the stock scan."""
    return env_bool("GMLX_GDN_STAGED_SCAN", True)


def gates_enabled() -> bool:
    """GMLX_GDN_SCAN_GATES=0 computes the decay and beta with ops."""
    return env_bool("GMLX_GDN_SCAN_GATES", True)
