# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Asher Feldman
"""Two-stream mHC routing for SenseNova Loop.

Follows llama.cpp's ``llama_model_sensenova_loop::graph`` (build_route,
build_merge): the router reads both residual streams through an RMS norm
without gain and one float32 linear with bias, and returns

- the branch input ``(sigmoid(l0) a + sigmoid(l1) b) * pre_scale``,
- the write gains ``2 sigmoid(l2)`` and ``2 sigmoid(l3)``,
- a 2x2 doubly stochastic mix from ``l4..l7`` after ``iterations`` Sinkhorn
  rounds in log space (rows, then columns).

The merge is ``a' = m00 a + m01 b + post0 y`` and ``b' = m10 a + m11 b +
post1 y`` for the branch output ``y``. The routing runs in float32. The two
streams sit side by side in one ``[..., 2 * hidden]`` array, float32 as in
llama.cpp unless ``GMLX_SENSENOVA_STREAMS_F32=0`` keeps them in the
activation dtype; the branch input and the layer outputs leave in the
activation dtype.

On the GPU each router runs as one Metal kernel (``route``): the merge of
the branch before it, the RMS norm and the router linear, the coefficients
and the branch input. Its norm and linear sum in their own order, closer to
a float64 reference than MLX's matmul. ``GMLX_SENSENOVA_MHC_FUSED=0`` runs
the router as MLX ops plus two kernels (coefficients with branch input, and
the merge) whose bits equal the ops path (``GMLX_SENSENOVA_MHC_KERNEL=0``).
"""

from __future__ import annotations

import os
from functools import lru_cache

import mlx.core as mx
import mlx.nn as nn

_EPS = 1e-6
# Up to this many rows the router linear runs one matvec a row.
_ROWWISE_MAX_ROWS = 16


def _normalize(x, y):
    # Two-element log-softmax as llama.cpp writes it (softplus of the gap).
    d = x - y
    return -mx.logaddexp(-d, 0.0), -mx.logaddexp(d, 0.0)


@lru_cache(maxsize=None)
def _coef_fn(iterations: int, fixed_identity: bool):
    """logits [..., n] float32 -> coefficients [..., 8] float32:
    pre0, pre1, post0, post1, m00, m01, m10, m11."""

    def fn(logits):
        pre0 = mx.sigmoid(logits[..., 0:1])
        pre1 = mx.sigmoid(logits[..., 1:2])
        post0 = 2.0 * mx.sigmoid(logits[..., 2:3])
        post1 = 2.0 * mx.sigmoid(logits[..., 3:4])
        if fixed_identity:
            one = mx.ones_like(pre0)
            zero = mx.zeros_like(pre0)
            mix = [one, zero, zero, one]
        else:
            m = [logits[..., 4 + i:5 + i] for i in range(4)]
            for _ in range(iterations):
                m[0], m[1] = _normalize(m[0], m[1])
                m[2], m[3] = _normalize(m[2], m[3])
                m[0], m[2] = _normalize(m[0], m[2])
                m[1], m[3] = _normalize(m[1], m[3])
            mix = [mx.exp(v) for v in m]
        return mx.concatenate([pre0, pre1, post0, post1, *mix], axis=-1)

    return mx.compile(fn)


@lru_cache(maxsize=None)
def _branch_fn(pre_scale: float, out_dtype):
    def fn(a, b, coef):
        x = a.astype(mx.float32) * coef[..., 0:1] + b.astype(mx.float32) * coef[..., 1:2]
        if pre_scale != 1.0:
            x = x * pre_scale
        return x.astype(out_dtype)

    return mx.compile(fn)


def _merge(a, b, y, coef):
    af = a.astype(mx.float32)
    bf = b.astype(mx.float32)
    yf = y.astype(mx.float32)
    na = af * coef[..., 4:5] + bf * coef[..., 5:6] + yf * coef[..., 2:3]
    nb = af * coef[..., 6:7] + bf * coef[..., 7:8] + yf * coef[..., 3:4]
    return na.astype(a.dtype), nb.astype(b.dtype)


_merge_ops = mx.compile(_merge)


@lru_cache(maxsize=None)
def _leave_fn(out_dtype):
    def fn(a, b):
        return ((a.astype(mx.float32) + b.astype(mx.float32)) * 0.5).astype(out_dtype)

    return mx.compile(fn)


# Metal kernels. The helpers copy MLX's Metal ``log1p``, ``LogAddExp`` and
# ``Sigmoid``; each product and sum is its own statement, as in a compiled
# MLX graph, so the compiler contracts nothing the ops path does not.
_HEADER = r"""
inline float snl_log1p(float x) {
  float xp1 = 1.0f + x;
  if (xp1 == metal::numeric_limits<float>::infinity()) {
    return metal::numeric_limits<float>::infinity();
  }
  if (xp1 == 1.0f) {
    return x;
  }
  return x * (metal::log(xp1) / (xp1 - 1.0f));
}
inline float snl_logaddexp(float x, float y) {
  if (metal::isnan(x) || metal::isnan(y)) {
    return metal::numeric_limits<float>::quiet_NaN();
  }
  constexpr float inf = metal::numeric_limits<float>::infinity();
  float maxval = metal::max(x, y);
  float minval = metal::min(x, y);
  return (minval == -inf || maxval == inf)
      ? maxval
      : (maxval + snl_log1p(metal::exp(minval - maxval)));
}
inline float snl_sigmoid(float x) {
  auto y = 1 / (1 + metal::precise::exp(metal::abs(x)));
  return (x < 0) ? y : 1 - y;
}
inline void snl_norm(thread float& x, thread float& y) {
  float d = x - y;
  float nd = -d;
  float u = -snl_logaddexp(nd, 0.0f);
  float v = -snl_logaddexp(d, 0.0f);
  x = u;
  y = v;
}
"""

_COEF_BRANCH = r"""
  uint r = threadgroup_position_in_grid.y;
  uint t = thread_position_in_threadgroup.x;
  threadgroup float pre[2];
  if (t == 0) {
    auto l = logits + r * NR;
    device float* o = coef + r * 8;
    float p0 = snl_sigmoid(l[0]);
    float p1 = snl_sigmoid(l[1]);
    o[0] = p0;
    o[1] = p1;
    o[2] = 2.0f * snl_sigmoid(l[2]);
    o[3] = 2.0f * snl_sigmoid(l[3]);
    if (IDENT) {
      o[4] = 1.0f; o[5] = 0.0f; o[6] = 0.0f; o[7] = 1.0f;
    } else {
      float m0 = l[4], m1 = l[5], m2 = l[6], m3 = l[7];
      for (int i = 0; i < ITERS; i++) {
        snl_norm(m0, m1);
        snl_norm(m2, m3);
        snl_norm(m0, m2);
        snl_norm(m1, m3);
      }
      o[4] = metal::precise::exp(m0);
      o[5] = metal::precise::exp(m1);
      o[6] = metal::precise::exp(m2);
      o[7] = metal::precise::exp(m3);
    }
    pre[0] = p0;
    pre[1] = p1;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float c0 = pre[0];
  float c1 = pre[1];
  auto sr = s + (size_t)r * 2 * H;
  device O* xr = x + (size_t)r * H;
  for (uint i = t; i < (uint)H; i += TG) {
    float u = float(sr[i]) * c0;
    float v = float(sr[H + i]) * c1;
    float w = u + v;
    if (SCALED) {
      w = w * pre_scale[0];
    }
    xr[i] = O(w);
  }
"""

_MERGE = r"""
  uint i = thread_position_in_grid.x;
  uint r = thread_position_in_grid.y;
  if (i >= (uint)H) return;
  auto c = coef + r * 8;
  size_t base = (size_t)r * 2 * H;
  float a = float(s[base + i]);
  float b = float(s[base + H + i]);
  float yv = float(y[(size_t)r * H + i]);
  float t1 = a * c[4];
  float t2 = b * c[5];
  float t3 = t1 + t2;
  float t4 = yv * c[2];
  float na = t3 + t4;
  float u1 = a * c[6];
  float u2 = b * c[7];
  float u3 = u1 + u2;
  float u4 = yv * c[3];
  float nb = u3 + u4;
  out[base + i] = T(na);
  out[base + H + i] = T(nb);
"""

_TG = 256


@lru_cache(maxsize=None)
def _coef_branch_kernel():
    return mx.fast.metal_kernel(
        name="sensenova_mhc_coef_branch",
        input_names=["logits", "s", "pre_scale"],
        output_names=["coef", "x"],
        header=_HEADER, source=_COEF_BRANCH)


@lru_cache(maxsize=None)
def _merge_kernel():
    return mx.fast.metal_kernel(
        name="sensenova_mhc_merge",
        input_names=["s", "y", "coef"],
        output_names=["out"],
        source=_MERGE)


def _use_kernel() -> bool:
    return (os.environ.get("GMLX_SENSENOVA_MHC_KERNEL", "1") != "0"
            and mx.default_device() == mx.gpu and mx.metal.is_available())


def enter(h: mx.array, streams_f32: bool) -> mx.array:
    """Both streams start from the layer input."""
    s = mx.concatenate([h, h], axis=-1)
    return s.astype(mx.float32) if streams_f32 else s


def leave(s: mx.array, out_dtype) -> mx.array:
    """The mean of the two streams, in ``out_dtype``."""
    hidden = s.shape[-1] // 2
    return _leave_fn(out_dtype)(s[..., :hidden], s[..., hidden:])


def merge(s: mx.array, y: mx.array, coef: mx.array) -> mx.array:
    """The streams after a branch output ``y``."""
    hidden = y.shape[-1]
    if _use_kernel():
        rows = y.size // hidden
        return _merge_kernel()(
            inputs=[s, y, coef],
            template=[("T", s.dtype), ("H", hidden)],
            grid=(hidden, rows, 1), threadgroup=(min(hidden, _TG), 1, 1),
            output_shapes=[s.shape], output_dtypes=[s.dtype])[0]
    na, nb = _merge_ops(s[..., :hidden], s[..., hidden:], y, coef)
    return mx.concatenate([na, nb], axis=-1)


class MHCRouter(nn.Module):
    """One router: ``weight`` [n_route, 2 * hidden] and ``bias`` [n_route],
    both float32."""

    def __init__(self, hidden_size: int, settings: dict):
        super().__init__()
        self.iterations = int(settings.get("iterations", 20))
        self.fixed_identity = bool(settings.get("fixed_identity", False))
        self.pre_scale = float(settings.get("pre_scale", 1.0))
        n_route = 4 if self.fixed_identity else 8
        self.weight = mx.zeros((n_route, 2 * hidden_size), dtype=mx.float32)
        self.bias = mx.zeros((n_route,), dtype=mx.float32)

    def _linear(self, flat: mx.array) -> mx.array:
        # A float32 matmul with a few rows and 8 outputs runs as one long
        # GEMM tile, about 10x a one-row matvec on M5 and in TF32. Verify
        # widths run one matvec a row instead, the bits of a decode step.
        rows = flat.size // flat.shape[-1]
        if 1 < rows <= _ROWWISE_MAX_ROWS:
            lead = flat.shape[:-1]
            flat = flat.reshape(rows, 1, flat.shape[-1])
            out = [mx.addmm(self.bias, flat[i], self.weight.T) for i in range(rows)]
            return mx.concatenate(out, axis=0).reshape(*lead, -1)
        return mx.addmm(self.bias, flat, self.weight.T)

    def __call__(self, s: mx.array, out_dtype):
        """Returns (branch input in ``out_dtype``, coefficients)."""
        flat = mx.fast.rms_norm(s.astype(mx.float32), None, _EPS)
        logits = self._linear(flat)
        hidden = s.shape[-1] // 2
        if _use_kernel():
            rows = s.size // s.shape[-1]
            coef, x = _coef_branch_kernel()(
                inputs=[logits, s, mx.array([self.pre_scale], mx.float32)],
                template=[("T", s.dtype), ("O", out_dtype), ("H", hidden),
                          ("NR", int(self.weight.shape[0])),
                          ("ITERS", self.iterations),
                          ("IDENT", self.fixed_identity),
                          ("SCALED", self.pre_scale != 1.0), ("TG", _TG)],
                grid=(_TG, rows, 1), threadgroup=(_TG, 1, 1),
                output_shapes=[s.shape[:-1] + (8,), s.shape[:-1] + (hidden,)],
                output_dtypes=[mx.float32, out_dtype])
            return x, coef
        coef = _coef_fn(self.iterations, self.fixed_identity)(logits)
        x = _branch_fn(self.pre_scale, out_dtype)(
            s[..., :hidden], s[..., hidden:], coef)
        return x, coef


# One dispatch a router: the merge of the branch before it (when one is
# pending), the RMS norm and the router linear, the coefficients and the
# branch input. The merge bits equal ``merge``; the norm and the linear sum
# in another order and scale the dots by the norm factor after the sums, so
# the logits differ from the ops path by rounding.
_ROUTE = r"""
  uint r = threadgroup_position_in_grid.y;
  uint t = thread_position_in_threadgroup.x;
  uint lane = thread_index_in_simdgroup;
  uint sg = simdgroup_index_in_threadgroup;
  size_t base = (size_t)r * 2 * H;
  auto c = cprev + r * 8;
  float sq = 0.0f;
  float dot[NR];
  for (int j = 0; j < NR; j++) dot[j] = 0.0f;
  for (uint i = t; i < (uint)H; i += TG) {
    float a = float(s[base + i]);
    float b = float(s[base + H + i]);
    if (MERGE) {
      float yv = float(y[(size_t)r * H + i]);
      float t1 = a * c[4];
      float t2 = b * c[5];
      float t3 = t1 + t2;
      float t4 = yv * c[2];
      float na = t3 + t4;
      float u1 = a * c[6];
      float u2 = b * c[7];
      float u3 = u1 + u2;
      float u4 = yv * c[3];
      float nb = u3 + u4;
      T ta = T(na);
      T tb = T(nb);
      s_out[base + i] = ta;
      s_out[base + H + i] = tb;
      a = float(ta);
      b = float(tb);
    }
    sq = metal::fma(a, a, sq);
    sq = metal::fma(b, b, sq);
    for (int j = 0; j < NR; j++) {
      dot[j] = metal::fma(w[(size_t)j * 2 * H + i], a, dot[j]);
      dot[j] = metal::fma(w[(size_t)j * 2 * H + H + i], b, dot[j]);
    }
  }
  threadgroup float part[TG / 32][NR + 1];
  float v = simd_sum(sq);
  if (lane == 0) part[sg][NR] = v;
  for (int j = 0; j < NR; j++) {
    float d = simd_sum(dot[j]);
    if (lane == 0) part[sg][j] = d;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
  threadgroup float pre[2];
  if (t == 0) {
    float tot = 0.0f;
    float l[8];
    for (int j = 0; j <= NR; j++) {
      float acc = 0.0f;
      for (uint g = 0; g < TG / 32; g++) acc += part[g][j];
      if (j == NR) tot = acc; else l[j] = acc;
    }
    float inv = metal::precise::rsqrt(tot / float(2 * H) + EPS);
    for (int j = 0; j < NR; j++) l[j] = l[j] * inv + bias[j];
    device float* o = coef + r * 8;
    float p0 = snl_sigmoid(l[0]);
    float p1 = snl_sigmoid(l[1]);
    o[0] = p0;
    o[1] = p1;
    o[2] = 2.0f * snl_sigmoid(l[2]);
    o[3] = 2.0f * snl_sigmoid(l[3]);
    if (IDENT) {
      o[4] = 1.0f; o[5] = 0.0f; o[6] = 0.0f; o[7] = 1.0f;
    } else {
      float m0 = l[4], m1 = l[5], m2 = l[6], m3 = l[7];
      for (int i = 0; i < ITERS; i++) {
        snl_norm(m0, m1);
        snl_norm(m2, m3);
        snl_norm(m0, m2);
        snl_norm(m1, m3);
      }
      o[4] = metal::precise::exp(m0);
      o[5] = metal::precise::exp(m1);
      o[6] = metal::precise::exp(m2);
      o[7] = metal::precise::exp(m3);
    }
    pre[0] = p0;
    pre[1] = p1;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float c0 = pre[0];
  float c1 = pre[1];
  device O* xr = x + (size_t)r * H;
  for (uint i = t; i < (uint)H; i += TG) {
    float a = MERGE ? float(s_out[base + i]) : float(s[base + i]);
    float b = MERGE ? float(s_out[base + H + i]) : float(s[base + H + i]);
    float u = a * c0;
    float v2 = b * c1;
    float w2 = u + v2;
    if (SCALED) {
      w2 = w2 * pre_scale[0];
    }
    xr[i] = O(w2);
  }
"""

_ROUTE_TG = 512


@lru_cache(maxsize=None)
def _route_kernel():
    return mx.fast.metal_kernel(
        name="sensenova_mhc_route",
        input_names=["s", "y", "cprev", "w", "bias", "pre_scale"],
        output_names=["s_out", "coef", "x"],
        header=_HEADER, source=_ROUTE.replace("EPS", f"{_EPS!r}f"))


def _use_fused() -> bool:
    return os.environ.get("GMLX_SENSENOVA_MHC_FUSED", "1") != "0" and _use_kernel()


def route(router: MHCRouter, s: mx.array, y, coef_prev, out_dtype):
    """Merge the pending branch output ``y`` (if any) into the streams, then
    run ``router``. Returns (streams, branch input, coefficients)."""
    if _use_fused():
        hidden = s.shape[-1] // 2
        rows = s.size // s.shape[-1]
        merging = y is not None
        if not merging:
            y = mx.zeros((1,), out_dtype)
            coef_prev = mx.zeros((1,), mx.float32)
        s_out, coef, x = _route_kernel()(
            inputs=[s, y, coef_prev, router.weight, router.bias,
                    mx.array([router.pre_scale], mx.float32)],
            template=[("T", s.dtype), ("O", out_dtype), ("H", hidden),
                      ("NR", int(router.weight.shape[0])),
                      ("ITERS", router.iterations),
                      ("IDENT", router.fixed_identity),
                      ("SCALED", router.pre_scale != 1.0),
                      ("MERGE", merging), ("TG", _ROUTE_TG)],
            grid=(_ROUTE_TG, rows, 1), threadgroup=(_ROUTE_TG, 1, 1),
            output_shapes=[s.shape if merging else (1,),
                           s.shape[:-1] + (8,), s.shape[:-1] + (hidden,)],
            output_dtypes=[s.dtype, mx.float32, out_dtype])
        return (s_out if merging else s), x, coef
    if y is not None:
        s = merge(s, y, coef_prev)
    x, coef = router(s, out_dtype)
    return s, x, coef
