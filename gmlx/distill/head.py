"""The fused chunked head: logits per chunk with a closed-form backward,
the slot masses of the student over the projected groups, the on-path
log-prob and the boundary mass. The head runs outside the trunk's gradient
transform, and no full logits array outlives a chunk."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from .constants import LOG_FLOOR, NEG_INF

# ---------------------------------------------------------------------------
# fused chunked head
# ---------------------------------------------------------------------------

@dataclass
class HeadSpec:
    """The student's output projection as a pure function of its parameters.

    fn(params, h [C, d]) -> z [C, V]. params is the pytree the function
    reads, kept explicit so the head pass can run outside the trunk's
    gradient transform and detach the head at will. softcap is gemma's
    final_logit_softcapping or None."""
    fn: Callable
    params: Any
    softcap: float | None
    V: int
    live: Callable | None = None   # () -> the module's current parameter tree
    weight: Callable | None = None  # () -> W [V, d] in a float dtype (dequantized once)
    scale: float = 1.0             # the factor fn applies after the projection (Granite, Cohere, MiniCPM)

    def dense_weight(self, params=None):
        """W [V, d] for the closed-form backward (dz @ W). A float weight in
        params wins (trainable heads read the live tree); quantized heads
        dequantize once and cache."""
        if params is not None and isinstance(params, dict) and "weight" in params:
            w = params["weight"]
            if w.dtype in (_MX().float32, _MX().float16, _MX().bfloat16):
                return w
        if self.weight is not None:
            return self.weight()
        return self.params["weight"]

    def current(self):
        """The parameter tree in force now. Under nn.value_and_grad the
        module holds traced arrays, so the loss must read them at call
        time rather than the copy taken when the spec was built."""
        return self.live() if self.live is not None else self.params


def _MX():
    import mlx.core as mx
    return mx


def head_weight_fn(mod) -> Callable:
    """() -> the module's [V, d] weight as bf16, cached; dequantizes affine
    (mlx) and kquant heads row-block by row-block."""
    import mlx.core as mx
    cache: dict[str, Any] = {}

    def get():
        if "w" in cache:
            return cache["w"]
        w = mod.weight
        if w.dtype in (mx.float32, mx.float16, mx.bfloat16):
            cache["w"] = w
            return w
        mode = getattr(mod, "mode", "affine")
        rows = int(w.shape[0])
        parts = []
        step = 8192
        for r in range(0, rows, step):
            if mode == "kquant":
                import mlx_kquant as kq
                block = kq.dequantize(w[r:r + step], mod["scales"], mod.kquant_type)
            else:
                block = mx.dequantize(w[r:r + step], mod.scales[r:r + step],
                                      mod.biases[r:r + step] if getattr(mod, "biases", None) is not None else None,
                                      mod.group_size, mod.bits)
            parts.append(block.astype(mx.bfloat16))
            mx.eval(parts[-1])
        cache["w"] = mx.concatenate(parts, axis=0)
        mx.eval(cache["w"])
        return cache["w"]
    return get


def head_scale(args) -> float:
    """The factor an mlx-lm model applies to its logits after the
    projection: Granite divides by logits_scaling, Cohere multiplies by
    logit_scale, MiniCPM divides the hidden states by hidden_size over
    dim_model_base. 1.0 for every other model."""
    s = 1.0
    ls = getattr(args, "logits_scaling", None)
    if ls:
        s /= float(ls)
    lg = getattr(args, "logit_scale", None)
    if lg is not None:
        s *= float(lg)
    dmb = getattr(args, "dim_model_base", None)
    hs = getattr(args, "hidden_size", None)
    if dmb and hs:
        s /= float(hs) / float(dmb)
    return s


def head_spec_from_model(model) -> HeadSpec:
    """mlx-lm style models: lm_head when present, else the tied embedding
    as_linear; softcap from args.final_logit_softcapping through the
    text_config fallback; the post-projection scale from head_scale."""
    args = getattr(model, "args", None)
    cfg = getattr(args, "__dict__", {}) if args is not None else {}
    cfg = cfg.get("text_config", cfg) if isinstance(cfg, dict) else {}
    softcap = cfg.get("final_logit_softcapping") if isinstance(cfg, dict) else None
    if softcap is not None:
        softcap = float(softcap) or None
    scale = head_scale(args)
    inner = getattr(model, "model", model)
    head = getattr(model, "lm_head", None)
    if head is not None and not (isinstance(getattr(args, "tie_word_embeddings", None), bool)
                                 and args.tie_word_embeddings):
        mod = head

        def fn(params, h):
            mod.update(params)
            return mod(h) * scale if scale != 1.0 else mod(h)
        V = int(mod.weight.shape[0]) if hasattr(mod, "weight") else int(mod(mx_zeros(1, h_dim(inner))).shape[-1])
        return HeadSpec(fn=fn, params=mod.parameters(), softcap=softcap, V=V, live=mod.parameters,
                        weight=head_weight_fn(mod), scale=scale)
    emb = inner.embed_tokens

    def fn2(params, h):
        emb.update(params)
        return emb.as_linear(h) * scale if scale != 1.0 else emb.as_linear(h)
    return HeadSpec(fn=fn2, params=emb.parameters(), softcap=softcap,
                    V=int(emb.weight.shape[0]) if hasattr(emb, "weight") else int(emb.as_linear(mx_zeros(1, h_dim(inner))).shape[-1]),
                    live=emb.parameters, weight=head_weight_fn(emb), scale=scale)


HEAD_PARITY_TOL = 0.05


def head_parity_gap(model, head: HeadSpec, ids) -> float:
    """The largest difference between the model's own logits on ``ids``
    [1, T] and the head applied to the trunk's hidden states, relative to
    the largest logit. A model whose forward changes its logits after the
    projection in a way the head does not carry shows up here."""
    import mlx.core as mx
    from .student import trunk_hidden
    inner = getattr(model, "language_model", model)
    ref = inner(ids)
    ref = getattr(ref, "logits", ref)[0].astype(mx.float32)
    z = _head_logits_f32(head, head.current(), trunk_hidden(inner, ids)[0])
    gap = mx.abs(z - ref).max() / mx.maximum(mx.abs(ref).max(), 1.0)
    mx.eval(gap)
    return float(gap)


def mx_zeros(*shape):
    import mlx.core as mx
    return mx.zeros(shape)


def h_dim(inner) -> int:
    emb = inner.embed_tokens
    w = getattr(emb, "weight", None)
    if w is not None:
        return int(w.shape[1])
    return int(getattr(emb, "dims", 0) or getattr(inner.args, "hidden_size"))


def linear_head(weight, softcap: float | None = None, scale: float = 1.0) -> HeadSpec:
    """A plain [V, d] weight as a head (tests, synthetic sweeps)."""
    def fn(params, h):
        z = h @ params["weight"].T
        return z * scale if scale != 1.0 else z
    return HeadSpec(fn=fn, params={"weight": weight}, softcap=softcap, V=int(weight.shape[0]), scale=scale)


def _head_logits_f32(head: HeadSpec, params, h_c):
    import mlx.core as mx
    z = head.fn(params, h_c)
    if head.softcap:
        z = head.softcap * mx.tanh(z / head.softcap)
    return z.astype(mx.float32)


def _bnd_chunk_fn(head: HeadSpec, next_c, gid_c, group_of, G: int, Kp: int, log_bmask):
    """inner(params, h_c) -> (onpath [c], Q_slot [c, Kp + 1], log_bm [c])
    for a boundary chunk. Index arrays are closed over: gather has no VJP
    with respect to indices."""
    import mlx.core as mx
    V = head.V
    arange_kp = mx.arange(Kp, dtype=mx.int32)

    def inner(params, h_c):
        z = _head_logits_f32(head, params, h_c)
        logZ = mx.logsumexp(z, axis=-1, keepdims=True)
        onpath = mx.take_along_axis(z, next_c[:, None], axis=-1)[:, 0] - logZ[:, 0]
        q = mx.exp(z - logZ)
        c = int(h_c.shape[0])
        slot_map = mx.full((c, G + 1), Kp, dtype=mx.int32)
        slot_map = mx.put_along_axis(
            slot_map, gid_c, mx.broadcast_to(arange_kp[None, :], gid_c.shape), axis=1)
        if group_of is None:
            slot_of_u = slot_map[:, :V]
        else:
            slot_of_u = mx.take(slot_map, group_of, axis=1)
        rows = mx.broadcast_to(mx.arange(c)[:, None], (c, V))
        Q_slot = mx.zeros((c, Kp + 1), dtype=mx.float32).at[rows, slot_of_u].add(q)
        log_bm = mx.logsumexp(z + log_bmask[None, :], axis=-1) - logZ[:, 0]
        return onpath, Q_slot, log_bm
    return inner


def _plain_chunk_fn(head: HeadSpec, next_c):
    import mlx.core as mx

    def inner(params, h_c):
        z = _head_logits_f32(head, params, h_c)
        logZ = mx.logsumexp(z, axis=-1, keepdims=True)
        return (mx.take_along_axis(z, next_c[:, None], axis=-1)[:, 0] - logZ[:, 0],)
    return inner


def _head_chunks(N: int, n_bnd: int, C: int) -> list[tuple[int, int, bool]]:
    """(start, end, is_boundary) head chunks: boundary positions first."""
    out = []
    s = 0
    while s < n_bnd:
        e = min(s + C, n_bnd)
        out.append((s, e, True))
        s = e
    while s < N:
        e = min(s + C, N)
        out.append((s, e, False))
        s = e
    return out


def chunked_head(hidden, head: HeadSpec, next_ids, *, n_bnd: int, target_gid,
                 group_of, G: int, Kp: int, log_bmask, C: int = 512, params=None,
                 eval_each: bool = True):
    """Per-chunk head forward, each chunk evaluated before the next is
    built, so no [C, V] array outlives its chunk (one lazy graph over all
    chunks would hold every chunk's logits at once).

    hidden [N, d] in the loader's order (the first n_bnd rows are boundary
    positions), next_ids [N] the token each position predicts, target_gid
    [n_bnd, Kp] (sentinel G at pads), group_of [V] int32 or None on the
    identity path (G == V), log_bmask [V] f32 (0 for whitespace-initial and
    EOS ids, -inf elsewhere).

    Returns onpath_log_q [N], Q_slot [n_bnd, Kp + 1] (support slots then
    the exact tail slot, linear domain), log_bm [n_bnd]."""
    import mlx.core as mx
    N = int(hidden.shape[0])
    if params is None:
        params = head.current()
    onpath_parts, q_parts, bm_parts = [], [], []
    for s, e, is_bnd in _head_chunks(N, n_bnd, C):
        if is_bnd:
            o, qs, bm = _bnd_chunk_fn(head, next_ids[s:e], target_gid[s:e], group_of, G, Kp,
                                      log_bmask)(params, hidden[s:e])
            q_parts.append(qs)
            bm_parts.append(bm)
        else:
            (o,) = _plain_chunk_fn(head, next_ids[s:e])(params, hidden[s:e])
        onpath_parts.append(o)
        if eval_each:
            mx.eval(o, *((qs, bm) if is_bnd else ()))
    onpath = mx.concatenate(onpath_parts) if onpath_parts else mx.zeros((0,), dtype=mx.float32)
    Q_slot = mx.concatenate(q_parts) if q_parts else mx.zeros((0, Kp + 1), dtype=mx.float32)
    log_bm = mx.concatenate(bm_parts) if bm_parts else mx.zeros((0,), dtype=mx.float32)
    return onpath, Q_slot, log_bm


def chunked_head_vjp(hidden, head: HeadSpec, next_ids, *, n_bnd: int, target_gid, group_of, G: int,
                     Kp: int, log_bmask, C: int, params, d_onpath, d_Qslot, d_logbm,
                     want_params: bool):
    """Closed-form cotangent of the chunked head: dL/dhidden [N, d] (and
    dL/dW summed over chunks when want_params) from the cotangents of its
    three outputs. Per chunk: recompute z, q and the slot map, form
    dL/dz in one expression (softmax and slot-sum Jacobians applied by
    hand, softcap by its derivative), then dh = dz @ W and dW += dz^T h.
    Forward ops only, each chunk evaluated before the next, so no
    transform is nested and no [C, V] array outlives its chunk."""
    import mlx.core as mx
    N = int(hidden.shape[0])
    V = head.V
    W = head.dense_weight(params)
    arange_kp = mx.arange(Kp, dtype=mx.int32)
    in_B = mx.exp(log_bmask)[None, :]            # 0/1 [1, V]
    lo = mx.array(LOG_FLOOR, dtype=mx.float32)
    dh_parts = []
    dW = None
    for s, e, is_bnd in _head_chunks(N, n_bnd, C):
        h_c = hidden[s:e]
        c = int(h_c.shape[0])
        z_pre = head.fn(params, h_c).astype(mx.float32)
        z = head.softcap * mx.tanh(z_pre / head.softcap) if head.softcap else z_pre
        logZ = mx.logsumexp(z, axis=-1, keepdims=True)
        q = mx.exp(z - logZ)
        a = d_onpath[s:e]
        coef = -a[:, None]                        # onpath: a (onehot(next) - q)
        if is_bnd:
            gid_c = target_gid[s:e]
            slot_map = mx.full((c, G + 1), Kp, dtype=mx.int32)
            slot_map = mx.put_along_axis(
                slot_map, gid_c, mx.broadcast_to(arange_kp[None, :], gid_c.shape), axis=1)
            slot_of_u = slot_map[:, :V] if group_of is None else mx.take(slot_map, group_of, axis=1)
            rows = mx.broadcast_to(mx.arange(c)[:, None], (c, V))
            Q_slot = mx.zeros((c, Kp + 1), dtype=mx.float32).at[rows, slot_of_u].add(q)
            cs = d_Qslot[s:e]                     # [c, Kp + 1]
            cu = mx.take_along_axis(cs, slot_of_u, axis=1)   # [c, V]
            cQ = mx.sum(cs * Q_slot, axis=1, keepdims=True)   # [c, 1]
            coef = coef + cu - cQ                  # slot sums: q (c_slot(u) - <c, Q>)
            b = d_logbm[s:e]
            log_bm = mx.logsumexp(z + log_bmask[None, :], axis=-1, keepdims=True) - logZ
            inv_bm = mx.exp(-mx.maximum(log_bm, lo))
            bterm = mx.where(b[:, None] != 0, b[:, None] * (in_B * inv_bm - 1.0), mx.zeros((c, 1)))
            coef = coef + bterm                    # boundary mass: b q (1_B / bm - 1)
        dz = q * coef
        dz = mx.put_along_axis(dz, next_ids[s:e][:, None],
                               mx.take_along_axis(dz, next_ids[s:e][:, None], axis=1) + a[:, None], axis=1)
        if head.softcap:
            dz = dz * (1.0 - (z / head.softcap) ** 2)
        if head.scale != 1.0:
            dz = dz * head.scale                    # fn scales the projection, so dh and dW scale too
        dh_c = (dz.astype(W.dtype) @ W).astype(mx.float32)
        if want_params:
            dW_c = dz.T @ h_c.astype(mx.float32)
            dW = dW_c if dW is None else dW + dW_c
            mx.eval(dh_c, dW)
        else:
            mx.eval(dh_c)
        dh_parts.append(dh_c)
    dh = mx.concatenate(dh_parts) if dh_parts else mx.zeros_like(hidden)
    dparams = {"weight": dW} if (want_params and dW is not None) else None
    return dh, dparams


def log_bmask_from(bmask: np.ndarray):
    import mlx.core as mx
    return mx.array(np.where(bmask, 0.0, NEG_INF).astype(np.float32))


