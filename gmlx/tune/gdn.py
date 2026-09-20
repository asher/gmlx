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

import mlx.core as mx

from gmlx.envflags import env_bool

# log decay clamp: exp(-80) is below any float32 product that matters, and
# the clamp keeps -inf out of the cumulative sums
LOG_FLOOR = -80.0


def _tri_inverse_from_strict_lower(L: mx.array, C: int) -> mx.array:
    """(I - L)^-1 for a strictly lower triangular L of size C on the last
    two axes, by squaring: L is nilpotent, so the inverse is the finite
    product (I + L)(I + L^2)(I + L^4)... with log2(C) factors."""
    eye = mx.eye(C, dtype=L.dtype)
    M = eye + L
    P = L
    steps = int(math.ceil(math.log2(C))) if C > 1 else 0
    for _ in range(1, steps):
        P = P @ P
        M = M @ (eye + P)
    return M


def gated_delta_chunk(q: mx.array, k: mx.array, v: mx.array, g: mx.array,
                      beta: mx.array, state: mx.array | None = None,
                      mask: mx.array | None = None,
                      chunk: int = 64) -> tuple[mx.array, mx.array]:
    """Chunked gated delta rule with ``gated_delta_ops``' shapes: q, k
    [B, T, Hk, Dk]; v [B, T, Hv, Dv]; g, beta [B, T, Hv] (g the decay
    factor); state [B, Hv, Dv, Dk] or None; mask [B, T] bool or None.
    Returns y [B, T, Hv, Dv] in q's dtype and the final state in float32."""
    if g.ndim != 3:
        raise ValueError("gated_delta_chunk takes scalar gating g[B, T, Hv]")
    B, T, Hk, Dk = q.shape
    Hv, Dv = v.shape[-2:]
    out_dtype = q.dtype
    if (rep := Hv // Hk) > 1:
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
    tril = mx.tril(mx.ones((C, C), dtype=mx.bool_))
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


def chunked_gdn_active(a: mx.array) -> bool:
    """True when the training scan takes the chunked path: scalar gating
    and the switch left on."""
    return a.ndim == 3 and env_bool("GMLX_TRAIN_GDN_CHUNK", True)


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
