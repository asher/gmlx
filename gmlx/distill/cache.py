"""The teacher pass: trunk forward in chunks, the reduction head that keeps the
top-K log-probs with the on-path, tail and boundary masses, and the memory
arithmetic that sizes the head step."""
from __future__ import annotations

import math

import numpy as np

from gmlx.gen.prefill_plan import head_step

from .constants import NEG_INF

# ---------------------------------------------------------------------------
# teacher reduction and prefill plan
# ---------------------------------------------------------------------------

def reduce_logits(logits, next_ids, *, K: int, log_bmask, onpath_valid,
                  floor: bool = False):
    """Teacher reduction over one head sub-chunk of [n, V] logits.

    Sequenced as separate evals so at most one full-width temporary lives
    beyond the log-softmax (B2). Returns numpy arrays: top_k_log_softmax
    [n, K] f16 descending, top_k_indices [n, K] i32, onpath_log_p [n] f32,
    tail_log_mass [n] f32, log_boundary_mass [n] f32, floor_kld [n] f32
    when floor. next_ids[i] = -1 where there is no next token (on-path 0).
    Pure function of tensors; the shard writer is its caller."""
    import mlx.core as mx
    n, V = logits.shape
    K = min(K, V - 1)
    lf = logits.astype(mx.float32)
    log_softmax = lf - mx.logsumexp(lf, axis=-1, keepdims=True)
    del lf
    idx_part = mx.argpartition(log_softmax, kth=-K, axis=-1)[..., -K:]
    vals_part = mx.take_along_axis(log_softmax, idx_part, axis=-1)
    order = mx.argsort(-vals_part, axis=-1)
    top_idx = mx.take_along_axis(idx_part, order, axis=-1).astype(mx.int32)
    top_val = mx.take_along_axis(vals_part, order, axis=-1)
    safe_next = mx.maximum(next_ids, 0).astype(mx.int32)
    onpath = mx.take_along_axis(log_softmax, safe_next[:, None], axis=-1)[:, 0]
    onpath = mx.where(mx.array(onpath_valid), onpath, mx.zeros_like(onpath))
    mx.eval(top_idx, top_val, onpath)
    # (2) tail: masked copy and logsumexp
    masked = mx.put_along_axis(log_softmax, top_idx, mx.full(top_idx.shape, NEG_INF, dtype=mx.float32), axis=-1)
    tail = mx.logsumexp(masked, axis=-1)
    mx.eval(tail)
    del masked
    # (3) boundary: log_softmax + log_bmask and logsumexp
    bm = mx.logsumexp(log_softmax + log_bmask[None, :], axis=-1)
    mx.eval(bm)
    out = {
        "top_k_log_softmax": np.asarray(top_val.astype(mx.float16)),
        "top_k_indices": np.asarray(top_idx),
        "onpath_log_p": np.asarray(onpath),
        "tail_log_mass": np.asarray(tail),
        "log_boundary_mass": np.asarray(bm),
    }
    if floor:
        # KL between the f32 teacher and its f16-rounded top-K reconstruction
        # with a uniform tail (mlx-kld's kld_from_topk form, log1mexp tail).
        top_f16 = top_val.astype(mx.float16).astype(mx.float32)
        head_mass = mx.logsumexp(top_f16, axis=-1)
        head_safe = mx.minimum(head_mass, mx.array(-1e-7, dtype=mx.float32))
        log_p_tail = log1mexp(head_safe) - math.log(float(V - K))
        p_full = mx.exp(log_softmax)
        cross_head = mx.sum(mx.take_along_axis(p_full, top_idx, axis=-1) * top_f16, axis=-1)
        p_masked = mx.put_along_axis(p_full, top_idx, mx.zeros(top_idx.shape, dtype=mx.float32), axis=-1)
        cross_tail = mx.sum(p_masked, axis=-1) * log_p_tail
        neg_ent = mx.sum(p_full * log_softmax, axis=-1)
        fk = neg_ent - cross_head - cross_tail
        mx.eval(fk)
        out["floor_kld"] = np.asarray(fk)
    return out


def log1mexp(x):
    """log(1 - exp(x)) for x <= 0 in f32, Machler's two regimes."""
    import mlx.core as mx
    cutoff = -math.log(2.0)
    safe_close = mx.minimum(x, mx.array(-1e-30, dtype=mx.float32))
    safe_far = mx.minimum(x, mx.array(cutoff, dtype=mx.float32))
    a = mx.log(-mx.expm1(safe_close))
    b = mx.log1p(-mx.exp(safe_far))
    return mx.where(x > cutoff, a, b)


def measure_bytes_per_v(peak_bytes: float, baseline_bytes: float, step: int, V: int) -> float:
    """(peak - baseline) / (step * V); no rows factor."""
    return (peak_bytes - baseline_bytes) / float(step * V)


def probe_step(measured: float, budgeted: float, V: int, cap_gb: float) -> tuple[int, float]:
    """Assertion with an action: if measured > budgeted, the measured
    constant replaces the budgeted one and the step is re-derived against
    the unchanged cap. Returns (step, constant in force). The constants
    already include the floor field's bytes, so the cap is the only other
    input."""
    if measured <= budgeted:
        return head_step(V, cap_gb, budgeted), budgeted
    return head_step(V, cap_gb, measured), measured


