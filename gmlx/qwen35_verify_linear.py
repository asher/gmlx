"""Owned target-verify projection family for qwen3.5/3.6 MTP targets.

In-tree copy of mlx-vlm's verify-width projection family (the dense
GEMV kernel, the quantized qmv/argmax kernels, the fused decode
concat, and the ``_target_verify_linear(s)`` dispatchers), plus the
gmlx wrappers that fold the bf16 GEMV-ext lever into the default
path. Owned qwen forwards call ``verify_linear`` / ``verify_linears``
here instead of the patched module globals; the upstream family keeps
serving upstream-internal call sites (the stock wrapper head path)
untouched.

The upstream bodies are verbatim copies, source-equality-tested
against the pinned mlx-vlm release, so a release upgrade that changes
a body fails the suite.
"""

from functools import lru_cache
from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from .gdn_patches import _F16_HEAD_GEMV, _f16_head_gemv

__all__ = ["verify_linear", "verify_linears"]


# --- verbatim upstream copies (qwen3_5/language.py) ---

_TARGET_VERIFY_GEMV = (
    mx.fast.metal_kernel(
        name="qwen3_5_target_verify_gemv",
        input_names=["x", "weight"],
        output_names=["out"],
        header="#include <metal_simdgroup>\nusing namespace metal;\n",
        source=r"""
        uint lane = thread_position_in_grid.x;
        uint out_block = thread_position_in_grid.y;
        uint row = thread_position_in_grid.z;

        constexpr int TM = 4;
        constexpr int TN = 4;
        constexpr int SN = 32;
        constexpr int blockN = SN * TN;

        if (row >= R) {
            return;
        }

        int out_row = int(out_block * TM);
        if (out_row >= O) {
            return;
        }

        const device T* in_vec = x + row * K;
        const device T* mat = weight + out_row * K;

        float result[TM] = {0.0f, 0.0f, 0.0f, 0.0f};
        int col = int(lane * TN);
        int n_iter = K / blockN;
        int leftover = K - blockN * n_iter;

        for (int iter = 0; iter < n_iter; ++iter) {
            float v[TN];
            for (int tn = 0; tn < TN; ++tn) {
                v[tn] = static_cast<float>(in_vec[col + tn]);
            }

            for (int tm = 0; tm < TM; ++tm) {
                for (int tn = 0; tn < TN; ++tn) {
                    result[tm] += static_cast<float>(mat[tm * K + col + tn]) * v[tn];
                }
            }

            col += blockN;
        }

        if (leftover > 0) {
            float v[TN];
            for (int tn = 0; tn < TN; ++tn) {
                v[tn] = (col + tn < K) ? static_cast<float>(in_vec[col + tn]) : 0.0f;
            }

            for (int tm = 0; tm < TM; ++tm) {
                for (int tn = 0; tn < TN; ++tn) {
                    T m = (col + tn < K) ? mat[tm * K + col + tn] : T(0);
                    result[tm] += static_cast<float>(m) * v[tn];
                }
            }
        }

        for (int tm = 0; tm < TM; ++tm) {
            for (ushort sn = (SN / 2); sn >= 1; sn >>= 1) {
                result[tm] += simd_shuffle_down(result[tm], sn);
            }
        }

        if (lane == 0) {
            for (int tm = 0; tm < TM; ++tm) {
                out[row * O + out_row + tm] = static_cast<T>(result[tm]);
            }
        }
    """,
    )
    if mx.metal.is_available()
    else None
)


def _use_target_verify_dense(linear, x: mx.array, target_verify: bool) -> bool:
    return (
        _TARGET_VERIFY_GEMV is not None
        and target_verify
        and x.ndim == 3
        and x.shape[1] > 1
        and isinstance(linear, (nn.Linear, nn.QuantizedLinear))
    )


def _target_verify_weight(weight: mx.array, x: mx.array) -> Optional[mx.array]:
    B, L, D = x.shape
    O = weight.shape[0]
    if O < 4 or O % 4 != 0 or D >= 16 * O or weight.dtype != x.dtype:
        return None

    rows = B * L
    rows8 = ((rows + 7) // 8) * 8
    out = _TARGET_VERIFY_GEMV(
        inputs=[x.reshape(rows, D), weight],
        template=[("T", x.dtype), ("K", D), ("O", O), ("R", rows)],
        grid=(32, O // 4, rows8),
        threadgroup=(32, 1, 8),
        output_shapes=[(rows, O)],
        output_dtypes=[x.dtype],
    )[0]
    return out.reshape(B, L, O)


def _target_verify_qlinear_header(bits: int, group_size: int) -> str:
    return r"""
    using namespace metal;

    constant constexpr int SIMD_SIZE = 32;
    constant constexpr int BITS = __BITS__;
    constant constexpr int GS = __GS__;
    constant constexpr int PACK_FACTOR = (BITS == 5 ? 8 : 32 / BITS);
    constant constexpr int BYTES_PER_PACK = (BITS == 5 ? 5 : 32 / 8);
    constant constexpr int PACKS_PER_THREAD = 2;
    constant constexpr int VALUES_PER_THREAD = PACK_FACTOR * PACKS_PER_THREAD;
    constant constexpr int BLOCK_SIZE = VALUES_PER_THREAD * SIMD_SIZE;
    constant constexpr int SCALE_STEP_PER_THREAD = GS / VALUES_PER_THREAD;
    constant constexpr int RESULTS_PER_SIMDGROUP = 4;
    constant constexpr int NUM_SIMDGROUPS = 2;
    constant constexpr int BN = RESULTS_PER_SIMDGROUP * NUM_SIMDGROUPS;

    template <typename T>
    inline float load_vector_exact(const device T* x, thread float* x_thread) {
      float sum = 0.0f;
      if (BITS == 4) {
        for (int i = 0; i < VALUES_PER_THREAD; i += 4) {
          sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
          x_thread[i] = x[i];
          x_thread[i + 1] = x[i + 1] / 16.0f;
          x_thread[i + 2] = x[i + 2] / 256.0f;
          x_thread[i + 3] = x[i + 3] / 4096.0f;
        }
      } else if (BITS == 5) {
        for (int i = 0; i < VALUES_PER_THREAD; i += 8) {
          sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3] + x[i + 4] + x[i + 5] +
              x[i + 6] + x[i + 7];
          x_thread[i] = x[i];
          x_thread[i + 1] = x[i + 1] / 32.0f;
          x_thread[i + 2] = x[i + 2] / 4.0f;
          x_thread[i + 3] = x[i + 3] / 128.0f;
          x_thread[i + 4] = x[i + 4] / 16.0f;
          x_thread[i + 5] = x[i + 5] / 2.0f;
          x_thread[i + 6] = x[i + 6] / 64.0f;
          x_thread[i + 7] = x[i + 7] / 8.0f;
        }
      }
      return sum;
    }

    inline float qdot_exact(
        const device uint8_t* w,
        const thread float* x_thread,
        float scale,
        float bias,
        float sum) {
      float accum = 0.0f;
      if (BITS == 4) {
        const device uint16_t* ws = (const device uint16_t*)w;
        for (int i = 0; i < (VALUES_PER_THREAD / 4); i++) {
          accum +=
              (x_thread[4 * i] * (ws[i] & 0x000f) +
               x_thread[4 * i + 1] * (ws[i] & 0x00f0) +
               x_thread[4 * i + 2] * (ws[i] & 0x0f00) +
               x_thread[4 * i + 3] * (ws[i] & 0xf000));
        }
      } else if (BITS == 5) {
        for (int i = 0; i < (VALUES_PER_THREAD / 8); i++) {
          const thread float* xt = x_thread + 8 * i;
          const device uint8_t* wb = w + 5 * i;

          accum += (wb[0] & 0x1f) * xt[0];
          accum += (wb[0] & 0xe0) * xt[1];
          accum += (wb[1] & 0x3) * (xt[1] * 256.0f);
          accum += (wb[1] & 0x7c) * xt[2];
          accum += (wb[1] & 0x80) * xt[3];
          accum += (wb[2] & 0xf) * (xt[3] * 256.0f);
          accum += (wb[2] & 0xf0) * xt[4];
          accum += (wb[3] & 0x1) * (xt[4] * 256.0f);
          accum += (wb[3] & 0x3e) * xt[5];
          accum += (wb[3] & 0xc0) * xt[6];
          accum += (wb[4] & 0x7) * (xt[6] * 256.0f);
          accum += (wb[4] & 0xf8) * xt[7];
        }
      }
      return scale * accum + sum * bias;
    }
""".replace(
        "__BITS__", str(bits)
    ).replace(
        "__GS__", str(group_size)
    )


_TARGET_VERIFY_QMV_SOURCE = r"""
    uint n_tile = threadgroup_position_in_grid.y;
    uint b_idx = threadgroup_position_in_grid.z;
    uint simd_gid = simdgroup_index_in_threadgroup;
    uint simd_lid = thread_index_in_simdgroup;

    int out_row = int(n_tile) * BN + int(simd_gid) * RESULTS_PER_SIMDGROUP;
    int in_vec_size_w = K_SIZE * BYTES_PER_PACK / PACK_FACTOR;
    int in_vec_size_g = K_SIZE / GS;

    const device uint8_t* ws_base =
        (const device uint8_t*)w + out_row * in_vec_size_w +
        int(simd_lid) * PACKS_PER_THREAD * BYTES_PER_PACK;
    const device T* scales_base =
        scales + out_row * in_vec_size_g + int(simd_lid) / SCALE_STEP_PER_THREAD;
    const device T* biases_base =
        biases + out_row * in_vec_size_g + int(simd_lid) / SCALE_STEP_PER_THREAD;
    const device T* x_base =
        x + int(b_idx) * VERIFY_T * K_SIZE + int(simd_lid) * VALUES_PER_THREAD;

    float result[VERIFY_T][RESULTS_PER_SIMDGROUP];
    float x_thread[VERIFY_T][VALUES_PER_THREAD];
    for (int t = 0; t < VERIFY_T; ++t) {
      for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {
        result[t][row] = 0.0f;
      }
    }

    const device uint8_t* ws = ws_base;
    const device T* sc = scales_base;
    const device T* bs = biases_base;
    const device T* xk = x_base;

    for (int k = 0; k < K_SIZE; k += BLOCK_SIZE) {
      float sums[VERIFY_T];
      for (int t = 0; t < VERIFY_T; ++t) {
        sums[t] = load_vector_exact<T>(xk + t * K_SIZE, x_thread[t]);
      }

      for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {
        const device uint8_t* wl = ws + row * in_vec_size_w;
        const device T* sl = sc + row * in_vec_size_g;
        const device T* bl = bs + row * in_vec_size_g;
        float s = float(sl[0]);
        float b = float(bl[0]);
        for (int t = 0; t < VERIFY_T; ++t) {
          result[t][row] += qdot_exact(wl, x_thread[t], s, b, sums[t]);
        }
      }

      ws += BLOCK_SIZE * BYTES_PER_PACK / PACK_FACTOR;
      sc += BLOCK_SIZE / GS;
      bs += BLOCK_SIZE / GS;
      xk += BLOCK_SIZE;
    }

    for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {
      int n = out_row + row;
      for (int t = 0; t < VERIFY_T; ++t) {
        float r = simd_sum(result[t][row]);
        if (simd_lid == 0) {
          y[(int(b_idx) * VERIFY_T + t) * N_SIZE + n] = T(r);
        }
      }
    }
"""


_TARGET_VERIFY_QARGMAX_SOURCE = r"""
    uint n_tile = threadgroup_position_in_grid.y;
    uint b_idx = threadgroup_position_in_grid.z;
    uint simd_gid = simdgroup_index_in_threadgroup;
    uint simd_lid = thread_index_in_simdgroup;

    int out_row = int(n_tile) * BN + int(simd_gid) * RESULTS_PER_SIMDGROUP;
    int in_vec_size_w = K_SIZE * BYTES_PER_PACK / PACK_FACTOR;
    int in_vec_size_g = K_SIZE / GS;

    threadgroup float tile_best_values[VERIFY_T][NUM_SIMDGROUPS];
    threadgroup int tile_best_indices[VERIFY_T][NUM_SIMDGROUPS];

    const device uint8_t* ws_base =
        (const device uint8_t*)w + out_row * in_vec_size_w +
        int(simd_lid) * PACKS_PER_THREAD * BYTES_PER_PACK;
    const device T* scales_base =
        scales + out_row * in_vec_size_g + int(simd_lid) / SCALE_STEP_PER_THREAD;
    const device T* biases_base =
        biases + out_row * in_vec_size_g + int(simd_lid) / SCALE_STEP_PER_THREAD;
    const device T* x_base =
        x + int(b_idx) * VERIFY_T * K_SIZE + int(simd_lid) * VALUES_PER_THREAD;

    float result[VERIFY_T][RESULTS_PER_SIMDGROUP];
    float x_thread[VERIFY_T][VALUES_PER_THREAD];
    for (int t = 0; t < VERIFY_T; ++t) {
      for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {
        result[t][row] = 0.0f;
      }
    }

    const device uint8_t* ws = ws_base;
    const device T* sc = scales_base;
    const device T* bs = biases_base;
    const device T* xk = x_base;

    for (int k = 0; k < K_SIZE; k += BLOCK_SIZE) {
      float sums[VERIFY_T];
      for (int t = 0; t < VERIFY_T; ++t) {
        sums[t] = load_vector_exact<T>(xk + t * K_SIZE, x_thread[t]);
      }

      for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {
        const device uint8_t* wl = ws + row * in_vec_size_w;
        const device T* sl = sc + row * in_vec_size_g;
        const device T* bl = bs + row * in_vec_size_g;
        float s = float(sl[0]);
        float b = float(bl[0]);
        for (int t = 0; t < VERIFY_T; ++t) {
          result[t][row] += qdot_exact(wl, x_thread[t], s, b, sums[t]);
        }
      }

      ws += BLOCK_SIZE * BYTES_PER_PACK / PACK_FACTOR;
      sc += BLOCK_SIZE / GS;
      bs += BLOCK_SIZE / GS;
      xk += BLOCK_SIZE;
    }

    for (int t = 0; t < VERIFY_T; ++t) {
      float best_value = -3.4028234663852886e38f;
      int best_index = 0;
      for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {
        int n = out_row + row;
        if (n < N_SIZE) {
          float rounded = float(T(simd_sum(result[t][row])));
          if (rounded > best_value) {
            best_value = rounded;
            best_index = n;
          }
        }
      }

      if (simd_lid == 0) {
        tile_best_values[t][simd_gid] = best_value;
        tile_best_indices[t][simd_gid] = best_index;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_gid == 0 && simd_lid == 0) {
      for (int t = 0; t < VERIFY_T; ++t) {
        float best = tile_best_values[t][0];
        int best_idx = tile_best_indices[t][0];
        for (int i = 1; i < NUM_SIMDGROUPS; ++i) {
          float candidate = tile_best_values[t][i];
          int candidate_idx = tile_best_indices[t][i];
          if (candidate > best) {
            best = candidate;
            best_idx = candidate_idx;
          }
        }
        int offset = (int(b_idx) * VERIFY_T + t) * NUM_TILES + int(n_tile);
        tile_values[offset] = T(best);
        tile_indices[offset] = best_idx;
      }
    }
"""


@lru_cache(maxsize=None)
def _target_verify_qmv_kernel(bits, group_size, dtype, verify_t, k_size, n_size):
    dtype_name = {mx.bfloat16: "bf16", mx.float16: "fp16"}.get(dtype, "unk")
    return mx.fast.metal_kernel(
        name=(
            "qwen3_5_target_verify_qmv_"
            f"b{bits}_gs{group_size}_t{verify_t}_k{k_size}_n{n_size}_{dtype_name}"
        ),
        input_names=["x", "w", "scales", "biases"],
        output_names=["y"],
        header=_target_verify_qlinear_header(bits, group_size),
        source=_TARGET_VERIFY_QMV_SOURCE,
    )


@lru_cache(maxsize=None)
def _target_verify_qargmax_kernel(bits, group_size, dtype, verify_t, k_size, n_size):
    dtype_name = {mx.bfloat16: "bf16", mx.float16: "fp16"}.get(dtype, "unk")
    return mx.fast.metal_kernel(
        name=(
            "qwen3_5_target_verify_qargmax_"
            f"b{bits}_gs{group_size}_t{verify_t}_k{k_size}_n{n_size}_{dtype_name}"
        ),
        input_names=["x", "w", "scales", "biases"],
        output_names=["tile_values", "tile_indices"],
        header=_target_verify_qlinear_header(bits, group_size),
        source=_TARGET_VERIFY_QARGMAX_SOURCE,
    )


def _can_target_verify_quantized(linear, x: mx.array) -> bool:
    if (
        not isinstance(linear, nn.QuantizedLinear)
        or x.ndim != 3
        or x.shape[1] < 1
        or linear.bits not in (4, 5)
        or linear.mode != "affine"
        or linear.biases is None
        or x.dtype not in (mx.bfloat16, mx.float16)
        or linear.scales.dtype != x.dtype
        or linear.biases.dtype != x.dtype
    ):
        return False

    _, _, K = x.shape
    N = linear.weight.shape[0]
    return (
        K == linear.weight.shape[1] * 32 // linear.bits and K % 512 == 0 and N % 8 == 0
    )


def _target_verify_quantized_linear(linear, x: mx.array) -> Optional[mx.array]:
    if not _can_target_verify_quantized(linear, x):
        return None

    B, T, K = x.shape
    N = linear.weight.shape[0]

    x = mx.contiguous(x)
    kernel = _target_verify_qmv_kernel(linear.bits, linear.group_size, x.dtype, T, K, N)
    out = kernel(
        inputs=[x, linear.weight, linear.scales, linear.biases],
        template=[
            ("T", x.dtype),
            ("VERIFY_T", int(T)),
            ("K_SIZE", int(K)),
            ("N_SIZE", int(N)),
        ],
        grid=(32, 2 * (N // 8), B),
        threadgroup=(32, 2, 1),
        output_shapes=[(B, T, N)],
        output_dtypes=[x.dtype],
    )[0]
    if "bias" in linear:
        out = out + linear["bias"]
    return out


def _decode_quantized_linears_fused(linears, x: mx.array):
    if (
        x.ndim != 3
        or x.shape[1] != 1
        or len(linears) != 4
        or not all(isinstance(linear, nn.QuantizedLinear) for linear in linears)
    ):
        return None

    first = linears[0]
    if not all(
        linear.bits == first.bits
        and linear.group_size == first.group_size
        and linear.mode == first.mode
        and linear.biases is not None
        and linear.scales.dtype == x.dtype
        and linear.biases.dtype == x.dtype
        and "bias" not in linear
        for linear in linears
    ):
        return None

    cache_key = tuple(
        (id(linear.weight), id(linear.scales), id(linear.biases)) for linear in linears
    )
    cached = getattr(first, "_qwen3_5_fused_decode_linears", None)
    if cached is None or cached[0] != cache_key:
        weights = mx.concatenate([linear.weight for linear in linears], axis=0)
        scales = mx.concatenate([linear.scales for linear in linears], axis=0)
        biases = mx.concatenate([linear.biases for linear in linears], axis=0)
        split_indices = []
        offset = 0
        for linear in linears[:-1]:
            offset += linear.weight.shape[0]
            split_indices.append(offset)
        mx.eval(weights, scales, biases)
        cached = (cache_key, weights, scales, biases, split_indices)
        first._qwen3_5_fused_decode_linears = cached

    _, weights, scales, biases, split_indices = cached
    output = mx.quantized_matmul(
        x,
        weights,
        scales=scales,
        biases=biases,
        transpose=True,
        group_size=first.group_size,
        bits=first.bits,
        mode=first.mode,
    )
    return tuple(mx.split(output, split_indices, axis=-1))


def _target_verify_quantized_argmax(linear, x: mx.array) -> Optional[mx.array]:
    if not _can_target_verify_quantized(linear, x) or "bias" in linear:
        return None

    B, T, K = x.shape
    if T == 1 and 1 < B <= 4:
        out = _target_verify_quantized_argmax(linear, x.transpose(1, 0, 2))
        if out is not None:
            return out.transpose(1, 0)

    N = linear.weight.shape[0]
    num_tiles = N // 8

    x = mx.contiguous(x)
    kernel = _target_verify_qargmax_kernel(
        linear.bits, linear.group_size, x.dtype, T, K, N
    )
    tile_values, tile_indices = kernel(
        inputs=[x, linear.weight, linear.scales, linear.biases],
        template=[
            ("T", x.dtype),
            ("VERIFY_T", int(T)),
            ("K_SIZE", int(K)),
            ("N_SIZE", int(N)),
            ("NUM_TILES", int(num_tiles)),
        ],
        grid=(32, 2 * num_tiles, B),
        threadgroup=(32, 2, 1),
        output_shapes=[(B, T, num_tiles), (B, T, num_tiles)],
        output_dtypes=[x.dtype, mx.int32],
    )
    best_tile = mx.argmax(tile_values, axis=-1)
    return mx.take_along_axis(tile_indices, best_tile[..., None], axis=-1).squeeze(-1)


def _target_verify_timewise(fn, x: mx.array) -> mx.array:
    return mx.concatenate([fn(x[:, i : i + 1]) for i in range(x.shape[1])], axis=1)


def _target_verify_singletons(fn, x: mx.array) -> mx.array:
    rows = []
    for row in range(x.shape[0]):
        rows.append(
            mx.concatenate(
                [fn(x[row : row + 1, i : i + 1]) for i in range(x.shape[1])],
                axis=1,
            )
        )
    return mx.concatenate(rows, axis=0)


def _target_verify_linear(linear, x: mx.array, target_verify: bool) -> mx.array:
    if not _use_target_verify_dense(linear, x, target_verify):
        return linear(x)

    if isinstance(linear, nn.QuantizedLinear):
        if x.shape[0] == 1:
            return linear(x)
        out = _target_verify_quantized_linear(linear, x)
        if out is not None:
            return out
        return _target_verify_timewise(linear, x)

    if isinstance(linear, nn.Linear) and "bias" not in linear:
        out = _target_verify_weight(linear.weight, x)
        if out is not None:
            return out

    return _target_verify_singletons(linear, x)


def _target_verify_linears(linears, x: mx.array, target_verify: bool):
    if not (
        target_verify
        and x.ndim == 3
        and x.shape[1] > 1
        and all(
            isinstance(linear, (nn.Linear, nn.QuantizedLinear)) for linear in linears
        )
    ):
        out = _decode_quantized_linears_fused(linears, x)
        if out is not None:
            return out
        return tuple(linear(x) for linear in linears)

    return tuple(_target_verify_linear(linear, x, target_verify) for linear in linears)


def _target_verify_embedding_as_linear(embedding, x: mx.array, target_verify: bool):
    if not (target_verify and x.ndim == 3 and x.shape[1] > 1):
        return embedding.as_linear(x)

    out = _target_verify_weight(embedding.weight, x)
    if out is not None:
        return out

    return _target_verify_timewise(embedding.as_linear, x)


# --- gmlx wrappers: the composed default path ---


def verify_linear(linear, x: mx.array, target_verify: bool):
    """``_target_verify_linear`` with the bf16 GEMV-ext lever folded in.

    Mirrors the patched-global chain the stock fallback composes: the
    M-stationary GEMV-ext claims verify-shaped non-quantized linears
    first (same condition as gdn_patches._patch_bf16_verify_linear),
    everything else lands on the verbatim upstream dispatcher above.
    """
    if (
        _F16_HEAD_GEMV is not None
        and target_verify
        and x.shape[1] > 1
        and not hasattr(linear, "scales")
    ):
        out = _f16_head_gemv(x, linear.weight)
        b = getattr(linear, "bias", None)
        if b is not None:
            out = out + b
        return out
    return _target_verify_linear(linear, x, target_verify)


def verify_linears(linears, x: mx.array, target_verify: bool):
    """``_target_verify_linears`` routing per-linear calls through the
    wrapper, so the bf16 lever engages exactly where the patched global
    did. Body mirrors the upstream dispatcher."""
    if not (
        target_verify
        and x.ndim == 3
        and x.shape[1] > 1
        and all(
            isinstance(linear, (nn.Linear, nn.QuantizedLinear)) for linear in linears
        )
    ):
        out = _decode_quantized_linears_fused(linears, x)
        if out is not None:
            return out
        return tuple(linear(x) for linear in linears)

    return tuple(verify_linear(linear, x, target_verify) for linear in linears)
