"""The training objective: bucketed sparse KL over the projected support and
its tail, ALM chunk-likelihood matching, optional CE, and the log-domain
rules that keep every case finite."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import mlx.core as mx

from .cache import log1mexp
from .constants import LOG_FLOOR, NEG_INF
from .head import HeadSpec, chunked_head, chunked_head_vjp

# ---------------------------------------------------------------------------
# loss
# ---------------------------------------------------------------------------

def kl_bern(log_a, log_b):
    """KL(Bern(a) || Bern(b)) from log a, log b in f32. Both logs are
    clamped to [LOG_FLOOR, -1e-7] so a -inf boundary mass on either side
    stays finite (N5 floor) with zero gradient below the floor."""
    import mlx.core as mx
    lo = mx.array(LOG_FLOOR, dtype=mx.float32)
    hi = mx.array(-1e-7, dtype=mx.float32)
    la = mx.maximum(mx.minimum(log_a, hi), lo)
    lb = mx.maximum(mx.minimum(log_b, hi), lo)
    l1a, l1b = log1mexp(la), log1mexp(lb)
    return mx.exp(la) * (la - lb) + mx.exp(l1a) * (l1a - l1b)


def bucketed_kl(target_log_p, log_M, Q_slot, weight, *, mode: str = "bucketed",
                T_dk: float = 1.0):
    """Weighted mean over boundaries of the bucketed sparse KL.

    target_log_p [Nb, Kp] (-inf pads), log_M [Nb], Q_slot [Nb, Kp + 1]
    linear (slot Kp the exact tail), weight [Nb]. Underflow rule N5:
    log Q = log(max(Q, 2^-126)); a floored slot is finite with zero
    gradient and is counted. Pads contribute zero by mx.where. Modes:
    bucketed (KL_bern(M||Q_S) + M KL(p~||q~)), paper (head term only),
    renorm (support-only softmax on both sides). T_dk tempers the
    conditional factor over group masses (both sides) when != 1.
    Returns (loss, aux) with aux["floored"] the floored-slot count."""
    import mlx.core as mx
    Nb, Kp = target_log_p.shape
    if Nb == 0:
        z = mx.zeros((), dtype=mx.float32)
        return z, {"floored": z, "weight_sum": z}
    pad = target_log_p == NEG_INF
    lp = mx.where(pad, mx.zeros_like(target_log_p), target_log_p)
    P = mx.where(pad, mx.zeros_like(lp), mx.exp(lp))
    floor = mx.array(2.0 ** -126, dtype=mx.float32)
    floored = mx.sum((Q_slot[:, :Kp] < floor) & ~pad) + mx.sum(Q_slot[:, Kp] < floor)
    logQ = mx.log(mx.maximum(Q_slot, floor))
    logQ_g = logQ[:, :Kp]
    logQ_tail = logQ[:, Kp]
    logM = mx.minimum(log_M, mx.array(-1e-7, dtype=mx.float32))
    M = mx.exp(log_M)
    if mode == "renorm":
        lq_valid = mx.where(pad, mx.full(lp.shape, NEG_INF, dtype=mx.float32), logQ_g)
        log_q_t = lq_valid - mx.logsumexp(lq_valid, axis=-1, keepdims=True)
        log_p_t = lp - log_M[:, None]
        per = mx.sum(mx.where(pad, mx.zeros_like(lp), mx.exp(log_p_t) * (log_p_t - log_q_t)), axis=-1)
    elif T_dk != 1.0:
        # conditional factor over groups, tempered on both sides
        lq_valid = mx.where(pad, mx.full(lp.shape, NEG_INF, dtype=mx.float32), logQ_g)
        log_q_t = lq_valid / T_dk
        log_q_t = log_q_t - mx.logsumexp(log_q_t, axis=-1, keepdims=True)
        lp_valid = mx.where(pad, mx.full(lp.shape, NEG_INF, dtype=mx.float32), lp)
        log_p_t = lp_valid / T_dk
        log_p_t = log_p_t - mx.logsumexp(log_p_t, axis=-1, keepdims=True)
        cond = mx.sum(mx.where(pad, mx.zeros_like(lp), mx.exp(log_p_t) * (log_p_t - log_q_t)), axis=-1)
        log_QS = mx.logsumexp(lq_valid, axis=-1)
        per = M * cond + kl_bern(logM, log_QS) if mode == "bucketed" else M * cond + M * (logM - log_QS)
    else:
        head_term = mx.sum(mx.where(pad, mx.zeros_like(lp), P * (lp - logQ_g)), axis=-1)
        if mode == "paper":
            per = head_term
        else:
            log1mM = log1mexp(logM)
            bern = mx.where(log_M < -1e-7, mx.exp(log1mM) * (log1mM - logQ_tail), mx.zeros_like(log_M))
            per = head_term + bern
    wsum = mx.sum(weight)
    loss = mx.sum(weight * per) / mx.maximum(wsum, mx.array(1.0, dtype=mx.float32))
    return loss, {"floored": floored, "weight_sum": wsum, "per_boundary": per}


def scatter_back(values, positions, B: int, Tm1: int):
    """[N] gathered values -> [B, T-1] with zeros elsewhere."""
    import mlx.core as mx
    flat = mx.zeros((B * Tm1,), dtype=mx.float32).at[positions].add(values)
    return flat.reshape(B, Tm1)


def alm_term(onpath_full, log_bm_full, chunk_start, chunk_end, chunk_teacher_ll,
             chunk_teacher_log_bm, chunk_mask, *, tau: float = 1.0):
    """Mean over kept chunks of KL_bern(teacher || student) on the chunk
    likelihood times the chunk-end boundary mass. Pads carry start = end =
    0 and are zeroed by chunk_mask; no negative index reaches the gather."""
    import mlx.core as mx
    B, Tm1 = onpath_full.shape
    cs = mx.cumsum(mx.concatenate([mx.zeros((B, 1), dtype=mx.float32), onpath_full], axis=1), axis=1)
    ll = mx.take_along_axis(cs, chunk_end + 1, axis=1) - mx.take_along_axis(cs, chunk_start, axis=1)
    bm_s = mx.take_along_axis(log_bm_full, chunk_end + 1, axis=1)
    log_a = (chunk_teacher_ll + chunk_teacher_log_bm) / tau
    log_b = (ll + bm_s) / tau
    kl = kl_bern(log_a, log_b)
    kl = mx.where(chunk_mask, kl, mx.zeros_like(kl))
    n = mx.sum(chunk_mask)
    return mx.sum(kl) / mx.maximum(n, mx.array(1, dtype=mx.int32)).astype(mx.float32), n


def loss_from_head_outputs(onpath, Q_slot, log_bm, batch: dict, *, knobs: dict, B: int, Tm1: int):
    """The composite objective from the head's per-position outputs and
    the batch tensors: bucketed KL over boundaries, ALM over chunks, CE
    over compute positions. Differentiable in (onpath, Q_slot, log_bm).
    Returns (loss, aux)."""
    import mlx.core as mx
    positions = batch["positions"]
    n_bnd = int(batch["n_bnd"])
    N = int(positions.shape[0])
    zero = mx.zeros((), dtype=mx.float32)
    aux: dict[str, Any] = {"ntoks": mx.array(N), "n_bnd": mx.array(n_bnd)}
    loss = zero
    if knobs["lambda_dk"] and n_bnd > 0:
        dk, dk_aux = bucketed_kl(batch["target_log_p"], batch["log_M"], Q_slot,
                                 batch["bnd_weight"], mode=knobs["loss_mode"], T_dk=knobs["T_dk"])
        aux["dk"] = dk
        aux["floored"] = dk_aux["floored"]
        loss = loss + knobs["lambda_dk"] * dk
    else:
        aux["dk"] = zero
        aux["floored"] = mx.array(0)
    need_full = bool(knobs["lambda_alm"]) or bool(knobs["lambda_ce"])
    if need_full and N > 0:
        onpath_full = scatter_back(onpath, positions, B, Tm1)
    if knobs["lambda_alm"] and N > 0 and n_bnd > 0:
        log_bm_full = scatter_back(log_bm, positions[:n_bnd], B, Tm1)
        alm, n_chunks = alm_term(onpath_full, log_bm_full, batch["chunk_start"], batch["chunk_end"],
                                 batch["chunk_teacher_ll"], batch["chunk_teacher_log_bm"],
                                 batch["chunk_mask"], tau=knobs["tau_alm"])
        aux["alm"] = alm
        aux["n_chunks"] = n_chunks
        loss = loss + knobs["lambda_alm"] * alm
    else:
        aux["alm"] = zero
        aux["n_chunks"] = mx.array(0)
    if knobs["lambda_ce"] and N > 0:
        ce = -mx.sum(onpath) / float(N)
        aux["ce"] = ce
        loss = loss + knobs["lambda_ce"] * ce
    else:
        aux["ce"] = -mx.sum(onpath) / float(max(N, 1)) if N > 0 else zero
    aux["loss"] = loss
    return loss, aux


def detached(x) -> "mx.array":
    """A copy of x outside any function transformation. Arrays made inside
    a grad trace stay referenced by the tape until the outer eval, so a
    head pass that must free each chunk's logits works on a detached copy
    (host round trip, tens of MB per batch)."""
    import mlx.core as mx
    if x.dtype == mx.bfloat16:
        return mx.array(np.asarray(x.astype(mx.float32))).astype(mx.bfloat16)
    return mx.array(np.asarray(x))


def surrogate(loss_value, pairs):
    """loss_value + sum_i <x_i, dx_i> with the inner products' values
    removed: the result equals loss_value and its gradient with respect to
    each x_i is the detached dx_i. This is how a head pass computed outside
    the tape hands its cotangents to the trunk's autodiff."""
    import mlx.core as mx
    out = mx.stop_gradient(loss_value)
    for x, dx in pairs:
        if dx is None:
            continue
        ip = mx.sum(x.astype(mx.float32) * mx.stop_gradient(dx).astype(mx.float32))
        out = out + ip - mx.stop_gradient(ip)
    return out


def gather_positions(hidden_btd, positions):
    """Hidden states at the batch's compute positions: [N, d] from [B, T, d]
    (position t of row b predicts token t+1; the last position has none)."""
    import mlx.core as mx
    B, T, d = hidden_btd.shape
    h = hidden_btd[:, :-1, :].reshape(B * (T - 1), d)
    return mx.take(h, positions, axis=0)


def head_pass(hg, batch: dict, head: HeadSpec, *, group_of, G: int, Kp: int, log_bmask, knobs: dict,
              B: int, Tm1: int, C: int = 512, head_trainable: bool = True, params_static=None):
    """The head half of a training step, run outside any function
    transformation: forward chunk by chunk over the gathered hidden states
    hg [N, d], the small loss in the head's outputs and its gradient, then
    the closed-form chunk backward. Returns (loss, aux, dh, dparams): loss
    and aux evaluated scalars, dh [N, d] float32, dparams the head's
    parameter gradients (None when the head is frozen). Every chunk is
    evaluated before the next, so the live set is one chunk's logits,
    softmax and cotangent plus the slot map.

    Call it outside mx.value_and_grad: inside a transform every array a
    chunk creates stays referenced until the outer eval, and the pass
    grows by one chunk's temporaries per chunk (a 0.6B student at C=512
    measured 86 GB that way against 4 GB outside). The trunk's own
    backward then runs in its own transform through `trunk_surrogate`.
    The inputs are detached anyway so the pass is also correct, if not
    bounded, when a caller such as `distill_loss` runs it inside one.
    params_static is the head's parameter tree read outside the transform
    (the same arrays); the default reads head.current()."""
    import mlx.core as mx
    from mlx.utils import tree_map
    positions = batch["positions"]
    n_bnd = int(batch["n_bnd"])
    N = int(positions.shape[0])
    next_ids = mx.take(batch["student_ids"][:, 1:].reshape(-1), positions)
    kw = dict(n_bnd=n_bnd, target_gid=batch["target_gid"], group_of=group_of, G=G, Kp=Kp,
              log_bmask=log_bmask, C=C)
    if N == 0:
        loss, aux = loss_from_head_outputs(mx.zeros((0,), dtype=mx.float32),
                                           mx.zeros((0, Kp + 1), dtype=mx.float32),
                                           mx.zeros((0,), dtype=mx.float32), batch, knobs=knobs, B=B, Tm1=Tm1)
        mx.eval(loss, *aux.values())
        return loss, aux, mx.zeros(hg.shape, dtype=mx.float32), None
    hg_d = detached(hg)
    if params_static is not None:
        params_d = params_static
    else:
        params = head.current()
        params_d = tree_map(detached, params) if head_trainable else params
    onpath, Q_slot, log_bm = chunked_head(hg_d, head, next_ids, params=params_d, **kw)
    loss, aux = loss_from_head_outputs(onpath, Q_slot, log_bm, batch, knobs=knobs, B=B, Tm1=Tm1)
    _, d_outs = mx.vjp(
        lambda o, q, b: loss_from_head_outputs(o, q, b, batch, knobs=knobs, B=B, Tm1=Tm1)[0],
        [onpath, Q_slot, log_bm], [mx.array(1.0, dtype=mx.float32)])
    mx.eval(loss, *aux.values(), *d_outs)
    # the nested vjp's outputs are tracers of whatever transform the caller
    # is in; detach them or every chunk of the closed-form pass is retained
    d_outs = [detached(x) for x in d_outs]
    loss = detached(loss)
    aux = {k: detached(v) for k, v in aux.items()}
    dh, dparams = chunked_head_vjp(hg_d, head, next_ids, params=params_d, d_onpath=d_outs[0],
                                   d_Qslot=d_outs[1], d_logbm=d_outs[2], want_params=head_trainable, **kw)
    if not head_trainable:
        dparams = None
    mx.eval(dh, *([] if dparams is None else [dparams]))
    return loss, aux, dh, dparams


def trunk_surrogate(hidden_btd, positions, head: HeadSpec, loss_value, dh, dparams=None):
    """Inside the trunk's transform: a scalar equal to loss_value whose
    gradient is dh at the gathered hidden states and dparams at the head's
    live parameters (read through head.current(), so under
    nn.value_and_grad they are the traced arrays). positions None means
    every compute position in flat order, as full_vocab_head_pass uses."""
    from mlx.utils import tree_flatten
    if positions is None:
        B, T, d = hidden_btd.shape
        hg = hidden_btd[:, :-1, :].reshape(B * (T - 1), d)
    else:
        hg = gather_positions(hidden_btd, positions)
    pairs = [(hg, dh)]
    if dparams is not None:
        lp = dict(tree_flatten(head.current()))
        for k, dpar in tree_flatten(dparams):
            pairs.append((lp[k], dpar))
    return surrogate(loss_value, pairs)


def distill_loss(hidden_btd, batch: dict, head: HeadSpec, *, group_of, G: int, Kp: int,
                 log_bmask, knobs: dict, C: int = 512, head_trainable: bool = True,
                 params_static=None):
    """The composite objective from trunk hidden states [B, T, d] and the
    batch tensors, as one differentiable scalar: `head_pass` on the
    gathered hidden states, then `trunk_surrogate` so an outer
    mx.value_and_grad or nn.value_and_grad sees the true value and the
    closed-form gradients. Returns (loss, aux); aux holds the term values,
    ntoks, floored-slot count and kept chunks as evaluated mx scalars.

    Convenient for tests, probes and the reachability control, where the
    hidden states are the argument. The trainer does not call it: with a
    trunk inside the transform the head pass would be retained chunk by
    chunk (see head_pass), so the trainer runs head_pass outside and
    trunk_surrogate inside."""
    B, T, d = hidden_btd.shape
    hg = gather_positions(hidden_btd, batch["positions"])
    loss, aux, dh, dparams = head_pass(hg, batch, head, group_of=group_of, G=G, Kp=Kp, log_bmask=log_bmask,
                                       knobs=knobs, B=B, Tm1=T - 1, C=C, head_trainable=head_trainable,
                                       params_static=params_static)
    return trunk_surrogate(hidden_btd, batch["positions"], head, loss, dh, dparams), aux


def full_vocab_head_pass(h, head: HeadSpec, teacher_lp, compute_mask, C: int, *, head_trainable: bool = True,
                         params_static=None):
    """Full-vocabulary KL(teacher || student) head pass over flat hidden
    states h [B*(T-1), d] with the teacher log-probs materialized
    [B, T-1, V]: the mean over compute positions and the closed-form
    dL/dz = w (q - p) per position. Returns (loss, dh, dparams) like
    head_pass; run it outside any transform for the same reason."""
    import mlx.core as mx
    from mlx.utils import tree_map
    tlp = teacher_lp.reshape(h.shape[0], -1)
    cm = compute_mask.reshape(-1).astype(mx.float32)
    N = int(h.shape[0])
    denom = mx.maximum(mx.sum(cm), mx.array(1.0))
    h_d = detached(h)
    if params_static is not None:
        params_d = params_static
    else:
        params = head.current()
        params_d = tree_map(detached, params) if head_trainable else params
    W = head.dense_weight(params_d)
    tot = mx.zeros((), dtype=mx.float32)
    dh_parts = []
    dW = None
    for s in range(0, N, C):
        e = min(s + C, N)
        h_c = h_d[s:e]
        z_pre = head.fn(params_d, h_c).astype(mx.float32)
        z = head.softcap * mx.tanh(z_pre / head.softcap) if head.softcap else z_pre
        lq = z - mx.logsumexp(z, axis=-1, keepdims=True)
        tl = tlp[s:e]
        p = mx.exp(tl)
        w = (cm[s:e] / denom)[:, None]
        kl = mx.sum(p * (tl - lq), axis=-1, keepdims=True)
        tot = tot + mx.sum(kl * w)
        dz = w * (mx.exp(lq) - p)
        if head.softcap:
            dz = dz * (1.0 - (z / head.softcap) ** 2)
        dh_c = (dz.astype(W.dtype) @ W).astype(mx.float32)
        if head_trainable:
            dW_c = dz.T @ h_c.astype(mx.float32)
            dW = dW_c if dW is None else dW + dW_c
            mx.eval(tot, dh_c, dW)
        else:
            mx.eval(tot, dh_c)
        dh_parts.append(dh_c)
    dh = mx.concatenate(dh_parts) if dh_parts else mx.zeros(h_d.shape, dtype=mx.float32)
    mx.eval(tot, dh)
    return tot, dh, ({"weight": dW} if (head_trainable and dW is not None) else None)


def full_vocab_kl(hidden, head: HeadSpec, teacher_lp, compute_mask, C: int, *, head_trainable: bool = True,
                  params_static=None):
    """Full-vocabulary KL as one differentiable scalar from hidden states
    [B, T, d]: `full_vocab_head_pass` then `trunk_surrogate`. Same caveat
    as distill_loss about calling it inside a transform with a trunk."""
    B, T, d = hidden.shape
    h = hidden[:, :-1, :].reshape(B * (T - 1), d)
    loss, dh, dparams = full_vocab_head_pass(h, head, teacher_lp, compute_mask, C, head_trainable=head_trainable,
                                             params_static=params_static)
    return trunk_surrogate(hidden, None, head, loss, dh, dparams)


