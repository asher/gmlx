"""Chunked gated delta rule for training, a differentiable replacement for
mlx-lm's per-token scan (``mlx_lm.models.gated_delta.gated_delta_ops``).

The per-token recurrence, in mlx-lm's layout (state [B, H, Dv, Dk], g the
decay factor in (0, 1]):

    S = g_t * S
    u = beta_t * (v_t - S k_t)
    S = S + u k_t^T
    y = S q_t

is the gated delta rule S_t = g_t (I - beta_t k_t k_t^T) S_{t-1} +
beta_t k_t v_t^T written for S^T. The chunked form (Yang et al., Gated
Delta Networks, the WY representation) processes C tokens at a time:
within a chunk every quantity is a batched matmul over [B, H, N, C, .]
arrays, and only the chunk-to-chunk state carry is sequential, so a
1024-token row costs 16 sequential steps instead of 1024 and the gradient
tape holds per-chunk intermediates instead of per-token states. Everything
runs in float32 and matches the per-token loop to float32 rounding; the
backward comes from MLX autograd through ordinary ops.

Scalar gating only (g of shape [B, T, Hv]); vectorized gating (a decay per
Dk element) stays on the per-token loop. A padding mask (False = padded)
holds the state across padded positions and zeroes their outputs, as the
kernel does. ``gated_delta_chunk`` takes the arguments of
``gated_delta_ops``; ``training_gated_delta_update`` takes those of
``gated_delta_update`` and is what the owned forwards call in training
mode. ``GMLX_TRAIN_GDN_CHUNK=0`` routes them to the loop instead.
"""
from __future__ import annotations

import math
import sys

import mlx.core as mx

from gmlx.envflags import env_bool

# log decay clamp: exp(-80) is below any float32 product that matters, and
# the clamp keeps -inf out of the cumulative sums
LOG_FLOOR = -80.0


def _tri_inverse_from_strict_lower(L: mx.array, C: int, base: int = 16) -> mx.array:
    """(I - L)^-1 for a strictly lower triangular L of size C on the last
    two axes. Blocks of ``base`` or fewer use the nilpotent expansion by
    squaring, (I + L)(I + L^2)(I + L^4)... with log2(C) factors; larger
    sizes split in half and combine the two diagonal inverses through the
    off-diagonal block, M21 = M22 L21 M11. Squaring alone is exact in
    float64 but not in float32 at C = 64: on real chunks the powers of L
    reach 1e9 and cancel to an inverse whose entries never exceed 1, and
    float32 keeps the cancellation error instead of the inverse."""
    if C <= base:
        eye = mx.eye(C, dtype=L.dtype)
        M = eye + L
        P = L
        steps = int(math.ceil(math.log2(C))) if C > 1 else 0
        for _ in range(1, steps):
            P = P @ P
            M = M @ (eye + P)
        return M
    h = C // 2
    M11 = _tri_inverse_from_strict_lower(L[..., :h, :h], h, base)
    M22 = _tri_inverse_from_strict_lower(L[..., h:, h:], C - h, base)
    M21 = M22 @ L[..., h:, :h] @ M11
    top = mx.concatenate([M11, mx.zeros(M11.shape[:-1] + (C - h,), dtype=L.dtype)], axis=-1)
    bot = mx.concatenate([M21, M22], axis=-1)
    return mx.concatenate([top, bot], axis=-2)


def tiled_heads() -> bool:
    """True once gmlx has switched mlx-lm's gated delta module to the GGUF
    tiled K->V head mapping, where value head ``hv`` reads key head
    ``hv % Hk`` instead of ``hv // (Hv / Hk)``. Set by the loader for
    qwen35-family GGUFs with asymmetric linear-attention heads, and read
    here so the chunked scan pairs heads the way the inference path does."""
    import sys
    gd = sys.modules.get("mlx_lm.models.gated_delta")
    return bool(getattr(gd, "_gmlx_tiled_v_patched", False))


def gated_delta_chunk(q: mx.array, k: mx.array, v: mx.array, g: mx.array,
                      beta: mx.array, state: mx.array | None = None,
                      mask: mx.array | None = None,
                      chunk: int = 64, tiled: bool | None = None) -> tuple[mx.array, mx.array]:
    """Chunked gated delta rule with ``gated_delta_ops``' shapes: q, k
    [B, T, Hk, Dk]; v [B, T, Hv, Dv]; g, beta [B, T, Hv] (g the decay
    factor); state [B, Hv, Dv, Dk] or None; mask [B, T] bool or None.
    Returns y [B, T, Hv, Dv] in q's dtype and the final state in float32.
    ``tiled`` picks the K->V head mapping when Hv > Hk: grouped (value
    head hv reads key head hv // r) or tiled (hv % Hk, the GGUF layout);
    None follows ``tiled_heads``."""
    if g.ndim != 3:
        raise ValueError("gated_delta_chunk takes scalar gating g[B, T, Hv]")
    B, T, Hk, Dk = q.shape
    Hv, Dv = v.shape[-2:]
    out_dtype = q.dtype
    if (rep := Hv // Hk) > 1:
        if tiled is None:
            tiled = tiled_heads()
        if tiled:
            q = mx.tile(q, [1, 1, rep, 1])
            k = mx.tile(k, [1, 1, rep, 1])
        else:
            q = mx.repeat(q, rep, -2)
            k = mx.repeat(k, rep, -2)
    H = Hv
    q = q.astype(mx.float32)
    k = k.astype(mx.float32)
    v = v.astype(mx.float32)
    beta = beta.astype(mx.float32)
    lg = mx.maximum(mx.log(g.astype(mx.float32)), LOG_FLOOR)
    if mask is not None:
        m = mask.astype(mx.bool_)[..., None]                    # [B, T, 1]
        beta = mx.where(m, beta, 0.0)       # no update at padded positions
        lg = mx.where(m, lg, 0.0)           # no decay either: the state holds
    # pad T to a multiple of the chunk with held positions
    pad = (-T) % chunk
    if pad:
        q = mx.pad(q, [(0, 0), (0, pad), (0, 0), (0, 0)])
        k = mx.pad(k, [(0, 0), (0, pad), (0, 0), (0, 0)])
        v = mx.pad(v, [(0, 0), (0, pad), (0, 0), (0, 0)])
        beta = mx.pad(beta, [(0, 0), (0, pad), (0, 0)])
        lg = mx.pad(lg, [(0, 0), (0, pad), (0, 0)])
    Tp = T + pad
    N = Tp // chunk
    C = chunk
    # [B, H, N, C, .]
    q = q.transpose(0, 2, 1, 3).reshape(B, H, N, C, Dk)
    k = k.transpose(0, 2, 1, 3).reshape(B, H, N, C, Dk)
    v = v.transpose(0, 2, 1, 3).reshape(B, H, N, C, Dv)
    beta = beta.transpose(0, 2, 1).reshape(B, H, N, C, 1)
    lg = lg.transpose(0, 2, 1).reshape(B, H, N, C)
    lg_cum = mx.cumsum(lg, axis=-1)                             # [B, H, N, C]
    # decay from position j to i inside the chunk, i >= j
    tril = mx.tril(mx.ones((C, C), dtype=mx.bool_), k=0)
    strict = mx.tril(mx.ones((C, C), dtype=mx.bool_), k=-1)
    diff = lg_cum[..., :, None] - lg_cum[..., None, :]          # [B, H, N, C, C]
    decay = mx.where(tril, mx.exp(mx.where(tril, diff, 0.0)), 0.0)
    k_beta = k * beta
    # WY representation: M = (I + strict_lower(diag(beta) K K^T . decay))^-1
    L = -mx.where(strict, (k_beta @ k.transpose(0, 1, 2, 4, 3)) * decay, 0.0)
    M = _tri_inverse_from_strict_lower(L, C)
    u = M @ (v * beta)                                          # [B, H, N, C, Dv]
    w = M @ (k_beta * mx.exp(lg_cum)[..., None])                # [B, H, N, C, Dk]
    qk = mx.where(tril, (q @ k.transpose(0, 1, 2, 4, 3)) * decay, 0.0)
    q_dec = q * mx.exp(lg_cum)[..., None]
    # decay from position i to the chunk end
    k_dec = k * mx.exp(lg_cum[..., -1:] - lg_cum)[..., None]
    g_chunk = mx.exp(lg_cum[..., -1])                           # [B, H, N]
    # inter-chunk carry with S in [Dk, Dv] layout
    if state is None:
        S = mx.zeros((B, H, Dk, Dv), dtype=mx.float32)
    else:
        S = state.astype(mx.float32).transpose(0, 1, 3, 2)
    ys = []
    for n in range(N):
        v_new = u[:, :, n] - w[:, :, n] @ S                     # [B, H, C, Dv]
        y_n = q_dec[:, :, n] @ S + qk[:, :, n] @ v_new
        S = S * g_chunk[:, :, n, None, None] + k_dec[:, :, n].transpose(0, 1, 3, 2) @ v_new
        ys.append(y_n)
    y = mx.stack(ys, axis=2).reshape(B, H, Tp, Dv)[:, :, :T].transpose(0, 2, 1, 3)
    if mask is not None:
        y = mx.where(mask.astype(mx.bool_)[..., None, None], y, 0.0)
    return y.astype(out_dtype), S.transpose(0, 1, 3, 2)


def gated_delta_update_chunked(q, k, v, a, b, A_log, dt_bias, state=None,
                               mask=None, chunk: int = 64):
    """``gated_delta_update``'s signature (gates from a, b, A_log, dt_bias)
    on the chunked path; scalar gating only."""
    from mlx_lm.models.gated_delta import compute_g
    beta = mx.sigmoid(b)
    g = compute_g(A_log, a, dt_bias)
    return gated_delta_chunk(q, k, v, g, beta, state, mask, chunk=chunk)


_F32_GEMM_EXACT: dict[str, bool] = {}   # per default device: a suite moves between devices
_TF32_WARNED = False


def f32_gemm_exact() -> bool:
    """True when the default device multiplies float32 matrices at float32
    precision. MLX runs float32 GEMM through the tensor cores at TF32
    precision on M5-class GPUs unless ``MLX_ENABLE_TF32=0`` is set before
    the first matmul, and the chunked rule feeds that rounding back
    through its state carry until the state diverges, so the training scan
    measures each device once against a CPU product."""
    key = str(mx.default_device())
    hit = _F32_GEMM_EXACT.get(key)
    if hit is None:
        a = mx.random.normal((64, 64), key=mx.random.key(0))
        b = mx.random.normal((64, 64), key=mx.random.key(1))
        gpu = mx.matmul(a, b)
        cpu = mx.matmul(a, b, stream=mx.cpu)
        hit = _F32_GEMM_EXACT[key] = float(mx.abs(gpu - cpu).max()) < 1e-3
    return hit


def chunked_gdn_active(a: mx.array) -> bool:
    """True when the training scan takes the chunked path: scalar gating,
    the switch left on, and exact float32 matmul on the device. Under TF32
    the scan falls back to the loop and says so once."""
    global _TF32_WARNED
    if a.ndim != 3 or not env_bool("GMLX_TRAIN_GDN_CHUNK", True):
        return False
    if f32_gemm_exact():
        return True
    if not _TF32_WARNED:
        _TF32_WARNED = True
        print("[train] warn: float32 matmul runs at TF32 precision, so the gated delta training scan "
              "takes mlx-lm's per-token loop. Set MLX_ENABLE_TF32=0 before the process starts "
              "for the chunked rule.", file=sys.stderr, flush=True)
    return False


def training_gated_delta_update(q, k, v, a, b, A_log, dt_bias, state=None,
                                mask=None, *, chunk: int = 64):
    """The gated delta scan for a forward under gradient tracing, with the
    arguments of ``gated_delta_update``: the chunked rule when
    ``chunked_gdn_active(a)``, else mlx-lm's per-token loop, either one
    inside ``mx.checkpoint`` so the tape keeps the scan's inputs and
    recomputes the rest in the backward."""
    from mlx_lm.models.gated_delta import gated_delta_update

    def scan(q, k, v, a, b, A, d, s, m):
        if chunked_gdn_active(a):
            return gated_delta_update_chunked(q, k, v, a, b, A, d, s, m, chunk=chunk)
        return gated_delta_update(q, k, v, a, b, A, d, s, m, use_kernel=False)

    if state is None:
        state = mx.zeros((q.shape[0], v.shape[-2], v.shape[-1], q.shape[-1]),
                         dtype=mx.float32)
    if mask is None:
        fn = mx.checkpoint(lambda q, k, v, a, b, A, d, s: scan(q, k, v, a, b, A, d, s, None))
        return fn(q, k, v, a, b, A_log, dt_bias, state)
    fn = mx.checkpoint(scan)
    return fn(q, k, v, a, b, A_log, dt_bias, state, mask)


def training_gated_delta_ops(q, k, v, g, beta, state=None, mask=None):
    """The gated delta scan on gate values for a forward under gradient
    tracing, with the arguments of mlx-lm's ``gated_delta_ops``: the
    per-token loop inside ``mx.checkpoint``, so the tape keeps the scan's
    inputs and recomputes the states in the backward. This is the route
    for per-key-channel decay (``g`` of shape [B, T, H, Dk]), which the
    chunked rule does not cover; ``training_gated_delta_update`` is the
    route for scalar gating."""
    from mlx_lm.models.gated_delta import gated_delta_ops

    if state is None:
        state = mx.zeros((q.shape[0], v.shape[-2], v.shape[-1], q.shape[-1]),
                         dtype=mx.float32)
    if mask is None:
        fn = mx.checkpoint(lambda q, k, v, g, b, s: gated_delta_ops(q, k, v, g, b, s, None))
        return fn(q, k, v, g, beta, state)
    fn = mx.checkpoint(gated_delta_ops)
    return fn(q, k, v, g, beta, state, mask)


# ---------------------------------------------------------------------------
# the text-only Qwen3.5 layer: mlx-lm's GatedDeltaNet has no training route
# ---------------------------------------------------------------------------

_TEXT_GDN_PATCH = None


class TrainingGdnInstall:
    """What ``install_training_gdn`` returns: the number of text-only
    gated delta layers in the model the dispatch now covers."""

    def __init__(self, count: int):
        self.count = count


def _text_gdn_training_call(self, inputs, mask=None):
    """mlx-lm's text-only ``GatedDeltaNet.__call__`` for a training forward
    with no cache, the scan sent through ``training_gated_delta_update``.
    The stock forward passes ``use_kernel=not self.training`` and so runs
    the per-token loop with every recurrent state on the gradient tape."""
    import mlx.nn as nn
    B, S, _ = inputs.shape
    qkv = self.in_proj_qkv(inputs)
    z = self.in_proj_z(inputs).reshape(B, S, self.num_v_heads, self.head_v_dim)
    b = self.in_proj_b(inputs)
    a = self.in_proj_a(inputs)
    conv_state = mx.zeros((B, self.conv_kernel_size - 1, self.conv_dim), dtype=inputs.dtype)
    if mask is not None:
        qkv = mx.where(mask[..., None], qkv, 0)
    conv_out = nn.silu(self.conv1d(mx.concatenate([conv_state, qkv], axis=1)))
    q, k, v = [
        t.reshape(B, S, h, d)
        for t, h, d in zip(
            mx.split(conv_out, [self.key_dim, 2 * self.key_dim], -1),
            [self.num_k_heads, self.num_k_heads, self.num_v_heads],
            [self.head_k_dim, self.head_k_dim, self.head_v_dim],
        )
    ]
    inv_scale = k.shape[-1] ** -0.5
    q = (inv_scale ** 2) * mx.fast.rms_norm(q, None, 1e-6)
    k = inv_scale * mx.fast.rms_norm(k, None, 1e-6)
    out, _state = training_gated_delta_update(q, k, v, a, b, self.A_log, self.dt_bias, None, mask)
    out = self.norm(out, z)
    return self.out_proj(out.reshape(B, S, -1))


def _text_gdn_call(self, inputs, mask=None, cache=None):
    if self.training and cache is None and getattr(self, "sharding_group", None) is None:
        return _text_gdn_training_call(self, inputs, mask)
    assert _TEXT_GDN_PATCH is not None
    return _TEXT_GDN_PATCH.stock(self, inputs, mask, cache)


def _qwen3next_gdn_training_call(self, inputs, mask=None):
    """mlx-lm's ``Qwen3NextGatedDeltaNet.__call__`` for a training forward
    with no cache, the scan sent through ``training_gated_delta_update``.
    Covers both projection layouts: the stock fused ``in_proj_qkvz`` and
    the split ``in_proj_qkv`` / ``in_proj_z`` pair gmlx installs for the
    split GGUF wire layout."""
    import mlx.nn as nn
    B, S, _ = inputs.shape
    if hasattr(self, "in_proj_qkv"):
        mixed_qkv = self.in_proj_qkv(inputs)
        z = self.in_proj_z(inputs).reshape(B, S, -1, self.head_v_dim)
        mixed_ba = self.in_proj_ba(inputs).reshape(B, S, self.num_k_heads, -1)
        b, a = mx.split(mixed_ba, [self.num_v_heads // self.num_k_heads], axis=-1)
        b = b.reshape(B, S, self.num_v_heads)
        a = a.reshape(B, S, self.num_v_heads)
    else:
        q, k, v, z, b, a = self.fix_query_key_value_ordering(
            self.in_proj_qkvz(inputs), self.in_proj_ba(inputs))
        mixed_qkv = mx.concatenate(
            [q.reshape(B, S, -1), k.reshape(B, S, -1), v.reshape(B, S, -1)], axis=-1)
    conv_state = mx.zeros((B, self.conv_kernel_size - 1, self.conv_dim), dtype=inputs.dtype)
    if mask is not None:
        mixed_qkv = mx.where(mask[..., None], mixed_qkv, 0)
    conv_out = nn.silu(self.conv1d(mx.concatenate([conv_state, mixed_qkv], axis=1)))
    q, k, v = [
        t.reshape(B, S, h, d)
        for t, h, d in zip(
            mx.split(conv_out, [self.key_dim, 2 * self.key_dim], -1),
            [self.num_k_heads, self.num_k_heads, self.num_v_heads],
            [self.head_k_dim, self.head_k_dim, self.head_v_dim],
        )
    ]
    inv_scale = k.shape[-1] ** -0.5
    q = (inv_scale ** 2) * mx.fast.rms_norm(q, None, 1e-6)
    k = inv_scale * mx.fast.rms_norm(k, None, 1e-6)
    out, _state = training_gated_delta_update(q, k, v, a, b, self.A_log, self.dt_bias, None, mask)
    out = self.norm(out, z)
    return self.out_proj(out.reshape(B, S, -1))


# one ClassPatch per patched class: mlx-lm's Qwen3NextGatedDeltaNet and every
# split-layout subclass the loader swaps in, which carries its own __call__
_QWEN3NEXT_PATCHES: dict = {}


def _qwen3next_gdn_call(self, inputs, mask=None, cache=None):
    if self.training and cache is None:
        return _qwen3next_gdn_training_call(self, inputs, mask)
    for cls in type(self).__mro__:
        patch = _QWEN3NEXT_PATCHES.get(cls)
        if patch is not None:
            return patch.stock(self, inputs, mask, cache)
    raise AssertionError("qwen3next training dispatch installed on an unpatched class")


def install_training_gdn(model) -> TrainingGdnInstall:
    """Route a training forward with no cache of mlx-lm's gated delta
    layers to the checkpointed scan the owned forwards already take:
    ``GatedDeltaNet`` (the class a Qwen3.5 or Qwen3.6 text GGUF loads) and
    ``Qwen3NextGatedDeltaNet`` (Qwen3-Next), whose stock forwards run the
    per-token loop with every recurrent state on the gradient tape.
    Installed once per process at the class, behind the fused decode
    dispatch when that is already in place, and inert outside training
    mode, so inference loads in the same process are untouched. Returns
    the count of such layers in ``model``."""
    global _TEXT_GDN_PATCH
    from gmlx.upstream.patching import ClassPatch
    count = 0
    try:
        from mlx_lm.models.qwen3_5 import GatedDeltaNet
    except ImportError:
        GatedDeltaNet = None
    if GatedDeltaNet is not None:
        if _TEXT_GDN_PATCH is None:
            _TEXT_GDN_PATCH = ClassPatch()
        _TEXT_GDN_PATCH.install(GatedDeltaNet, "__call__", _text_gdn_call)
        count += sum(1 for m in model.modules() if isinstance(m, GatedDeltaNet))
    try:
        from mlx_lm.models.qwen3_next import Qwen3NextGatedDeltaNet
    except ImportError:
        return TrainingGdnInstall(count)
    layers = [m for m in model.modules() if isinstance(m, Qwen3NextGatedDeltaNet)]
    classes = [Qwen3NextGatedDeltaNet]
    for m in layers:
        cls = type(m)
        if cls not in classes and "__call__" in vars(cls):
            classes.append(cls)
    for cls in classes:
        _QWEN3NEXT_PATCHES.setdefault(cls, ClassPatch()).install(cls, "__call__", _qwen3next_gdn_call)
    return TrainingGdnInstall(count + len(layers))
